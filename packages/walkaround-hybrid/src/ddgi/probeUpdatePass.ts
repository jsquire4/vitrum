/**
 * ProbeUpdatePass — DDGI probe update via raw WebGPU compute.
 *
 * Uses the renderer's raw GPUDevice to run two compute passes per frame:
 *  Pass 1 (probeUpdateRays): for each active probe, fire 96 rays via
 *          inline BVH traversal, collect radiance at hit points.
 *  Pass 2 (probeUpdateBlend): blend ray results into octahedral atlas
 *          textures with EWMA temporal hysteresis.
 *
 * This uses raw WebGPU rather than TSL/wgslFn because the compute shader
 * has custom @group/@binding layouts that don't compose naturally with
 * three.js's binding system. The WebGPU backend's `device` property is
 * used directly.
 *
 * three/webgpu coupling: `renderer.backend.device` is accessed directly,
 * but ProbeGrid atlas slots are now backend-agnostic `AtlasTextureSlot`
 * records (see probeGrid.ts). The GPU texture per slot is allocated lazily
 * in `_getOrCreateAtlasTexture` and cached in `_textureCache` keyed on the
 * slot identity. TSL binding (if applyDDGIShading is in use) is the only
 * site still importing from three/webgpu.
 */

import * as THREE from 'three';
import { extractThreePbrScalars } from '@vitrum/three-bindings';
import type { SceneBvh, SceneBvhBuffers } from '@vitrum/shared-bvh';
import type { ProbeGrid, AtlasTextureSlot } from './probeGrid.js';
import type { DDGILight } from './types.js';
import { PROBE_UPDATE_RAYS_WGSL } from './wgsl/probeUpdateRays.wgsl.js';
import { PROBE_UPDATE_BLEND_IRR_WGSL, PROBE_UPDATE_BLEND_VIS_WGSL } from './wgsl/probeUpdateBlend.wgsl.js';
import { packDDGIGridParams } from '../pipeline/resourceManager.js';
import { detectGpu } from '@vitrum/core';
import { RAYS_PER_PROBE } from './ddgiConstants.js';

// Re-export so existing consumers (`import { RAYS_PER_PROBE } from
// 'walkaround-hybrid/ddgi/probeUpdatePass'`) keep working. The constant
// lives in `./ddgiConstants.ts` to break the ESM import cycle between
// this module and its WGSL template files (host imports WGSL, WGSL
// imports the constant; before extraction the WGSL would read TDZ for
// `RAYS_PER_PROBE` and throw at evaluation time).
export { RAYS_PER_PROBE };
// ProbeRay struct: 12 floats / 2 u32 → 16 × 4 bytes = 64 bytes each
const PROBE_RAY_STRIDE_BYTES = 64;

// DDGI material buffer sizing constants.
// `materialsBuf` holds one DDGIMaterial struct per material slot.
// DDGIMaterial WGSL layout: 64 bytes = 16 × f32 (std140, see _uploadMaterials).
/** Maximum number of distinct materials the DDGI probe pass supports. */
export const DDGI_MAX_MATERIALS = 64;
/** Byte stride of one DDGIMaterial struct (must match the WGSL layout). */
export const DDGI_MATERIAL_STRIDE_BYTES = 64;
/** Float stride of one DDGIMaterial entry (64 bytes = 16 × f32). */
export const DDGI_MATERIAL_ENTRY_FLOATS = 16;

/**
 * Pack a list of THREE materials into the GPU-bound DDGIMaterial std140 layout
 * (64 bytes per material, 16 floats each). Pure function — does no GPU calls.
 * Used by both `ProbeUpdatePass._uploadMaterials` (at runtime) and the
 * byte-equivalence test fixture.
 *
 * WGSL layout (offsets in bytes per entry):
 *   offset  0: baseColor: vec3f  (12) + _pad0:  f32 (4)
 *   offset 16: emissive:  vec3f  (12) + roughness: f32 (4)
 *   offset 32: metalness, ior, transmission, _pad1 (4 × f32)
 *   offset 48: attenuationColor: vec3f (12) + flags: u32 (4)
 *     flags bit 0: isGlass (transmission > 0)
 *
 * Defaults (when a THREE field is absent): baseColor [1,1,1], emissive
 * [0,0,0], roughness 0.5, metallic 0, transmission 0, ior 1.5,
 * attenuationColor [1,1,1]. Matches the pre-P2-6.1 inline packer.
 */
export function packDDGIMaterials(mats: readonly THREE.Material[]): ArrayBuffer {
  const ENTRY = DDGI_MATERIAL_ENTRY_FLOATS;
  const buf = new ArrayBuffer(DDGI_MAX_MATERIALS * DDGI_MATERIAL_STRIDE_BYTES);
  const data = new Float32Array(buf);
  // u32 view onto the same backing buffer so the `flags` slot can be written
  // as a real u32 (the WGSL struct declares it as u32; writing 1.0 as f32
  // would land 0x3F800000 ≠ 1u in the GPU read).
  const u32view = new Uint32Array(buf);
  const matsToUse = mats.slice(0, DDGI_MAX_MATERIALS);
  matsToUse.forEach((mat, i) => {
    const base = i * ENTRY;
    const pbr = extractThreePbrScalars(mat);
    data[base + 0] = pbr.baseColor[0];
    data[base + 1] = pbr.baseColor[1];
    data[base + 2] = pbr.baseColor[2];
    data[base + 3] = 0; // _pad0
    data[base + 4] = pbr.emissive[0];
    data[base + 5] = pbr.emissive[1];
    data[base + 6] = pbr.emissive[2];
    data[base + 7] = pbr.roughness;
    data[base + 8] = pbr.metallic;
    data[base + 9] = pbr.ior;
    data[base + 10] = pbr.transmission;
    data[base + 11] = 0; // _pad1
    data[base + 12] = pbr.attenuationColor[0];
    data[base + 13] = pbr.attenuationColor[1];
    data[base + 14] = pbr.attenuationColor[2];
    u32view[base + 15] = pbr.transmission > 0 ? 1 : 0;
  });
  return buf;
}


interface GPUResources {
  device: GPUDevice;
  raysPipeline:      GPUComputePipeline;
  blendIrrPipeline:  GPUComputePipeline;
  blendVisPipeline:  GPUComputePipeline;
  // BVH buffers (replaced on scene rebuild)
  bvhBuf:        GPUBuffer;
  posBuf:        GPUBuffer;
  idxBuf:        GPUBuffer;
  normBuf:       GPUBuffer;
  matIdBuf:      GPUBuffer;
  // Uniform buffers
  materialsBuf:  GPUBuffer;
  lightsBuf:     GPUBuffer;
  gridParamsBuf: GPUBuffer;
  frameParamsBuf:GPUBuffer;
  blendParamsBuf:GPUBuffer;
  // Dynamic per-frame buffers
  rayResultsBuf: GPUBuffer;
  activeProbesBuf: GPUBuffer;
  // Samplers
  linearSampler: GPUSampler;
}

/** Options accepted by ProbeUpdatePass constructor. */
export interface ProbeUpdatePassOptions {
  /**
   * When true, exposes DDGI internal state to `window.__DDGI__` for
   * debugging and e2e inspection. Defaults to false.
   *
   * Gate rationale (Q-WA-3): the original source used an unconditional
   * window global assignment. A constructor option is the simplest
   * library-safe pattern — consumers opt in explicitly rather than
   * having a debug global appear unconditionally in production builds.
   */
  debug?: boolean;
}

export class ProbeUpdatePass {
  private _bvh:  SceneBvh;
  private _grid: ProbeGrid;
  private _gpu:  GPUResources | null = null;
  private _lastBvhVersion = -1;
  private _frameIndex = 0;
  private _maxProbes = 0;
  private _lights: DDGILight[] = [];
  private _debug: boolean;
  // Multiplier applied to every light's intensity when packing the
  // probe-update light UBO. Defaults to 1 so unrelated callers keep the
  // original behaviour. The hybrid pipeline calls setSunIntensityMultiplier(5.0)
  // so DDGI's per-probe Le matches the same primaryLightIntensity used by
  // shade.wgsl — without this, DDGI runs at 1/5 the magnitude of Lo_emit
  // and walls render dark.
  private _sunIntensityMul = 1;

  constructor(bvh: SceneBvh, grid: ProbeGrid, opts: ProbeUpdatePassOptions = {}) {
    this._bvh  = bvh;
    this._grid = grid;
    this._debug = opts.debug ?? false;
  }

  setLights(lights: DDGILight[]): void {
    this._lights = lights;
  }

  /** Multiplier on the sun's stored intensity. Hybrid pipeline
   *  calls this with primaryLightIntensity (5.0) so DDGI's bake of
   *  the sun's Le matches shade.wgsl's Lo_emit. */
  setSunIntensityMultiplier(mul: number): void {
    this._sunIntensityMul = mul;
  }

  /**
   * Initialize GPU resources. Returns false if WebGPU is unavailable.
   * Tries the renderer's WebGPU backend first; falls back to navigator.gpu.
   */
  async init(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }): Promise<boolean> {
    // Hardware-GPU gate. detectGpu() publishes window.__WG__ BEFORE we
    // touch the device, so e2e validation can read the flag even if we
    // refuse to proceed. SwiftShader (Chromium's software rasterizer)
    // compiles WGSL but produces low-chroma "almost passes" — fail-fast
    // here so validation rounds never silently mistake software output
    // for hardware-GPU output.
    const gpu = await detectGpu();
    if (gpu.isWebGPU && gpu.adapterKind === 'swiftshader') {
      console.error(
        `[DDGI] SwiftShader detected (vendor='${gpu.adapterVendor}', architecture='${gpu.adapterArchitecture}'). ` +
        `Refusing to initialize DDGI on software rasterizer. Launch Chrome with hardware GPU enabled to validate DDGI output.`,
      );
      return false;
    }

    // Try renderer's backend first (three.js WebGPURenderer).
    let device: GPUDevice | null = null;
    const backend = renderer.backend;
    if (backend?.isWebGPUBackend && backend.device) {
      device = backend.device;
    } else if (typeof navigator !== 'undefined' && navigator.gpu) {
      // Direct WebGPU path — works when three.js uses WebGLRenderer but
      // the browser exposes WebGPU (Chromium with --enable-unsafe-webgpu).
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          device = await adapter.requestDevice();
        }
      } catch (e) {
        console.warn('[DDGI] navigator.gpu.requestAdapter failed:', e);
      }
    }
    if (!device) {
      console.warn('[DDGI] WebGPU not available (no renderer backend and no navigator.gpu).');
      return false;
    }

    // Compile shaders.
    let raysPipeline: GPUComputePipeline;
    let blendIrrPipeline: GPUComputePipeline;
    let blendVisPipeline: GPUComputePipeline;
    try {
      const raysModule = device.createShaderModule({ code: PROBE_UPDATE_RAYS_WGSL });
      raysPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: raysModule, entryPoint: 'probeUpdateRays' },
      });

      const blendIrrModule = device.createShaderModule({ code: PROBE_UPDATE_BLEND_IRR_WGSL });
      blendIrrPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: blendIrrModule, entryPoint: 'probeUpdateBlendIrradiance' },
      });
      const blendVisModule = device.createShaderModule({ code: PROBE_UPDATE_BLEND_VIS_WGSL });
      blendVisPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: blendVisModule, entryPoint: 'probeUpdateBlendVisibility' },
      });
    } catch (e) {
      console.error('[DDGI] Shader compilation failed:', e);
      return false;
    }

    const linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Allocate placeholder buffers (1 element each, replaced on first update).
    const makeBuffer = (size: number, usage: number) =>
      device.createBuffer({ size: Math.max(size, 16), usage });

    const RO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const RW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const UB = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

    this._gpu = {
      device,
      raysPipeline,
      blendIrrPipeline,
      blendVisPipeline,
      bvhBuf:         makeBuffer(16, RO),
      posBuf:         makeBuffer(12, RO),
      idxBuf:         makeBuffer(12, RO),
      normBuf:        makeBuffer(12, RO),
      matIdBuf:       makeBuffer(4,  RO),
      materialsBuf:   makeBuffer(DDGI_MAX_MATERIALS * DDGI_MATERIAL_STRIDE_BYTES, UB),
      lightsBuf:      makeBuffer(16 * 80 + 16, UB),
      gridParamsBuf:  makeBuffer(64, UB),
      frameParamsBuf: makeBuffer(32, UB),
      blendParamsBuf: makeBuffer(16, UB),
      rayResultsBuf:  makeBuffer(PROBE_RAY_STRIDE_BYTES, RW),
      activeProbesBuf:makeBuffer(4, RO),
      linearSampler,
    };
    return true;
  }

  /**
   * Run one frame of probe updates.
   * @param renderer  The WebGPU renderer
   * @param offset    Which 1/4 of probes to update (0-3)
   * @param stride    Number of update strata (usually 4)
   */
  async runFrame(renderer: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } }, offset: number, stride: number): Promise<void> {
    if (!this._gpu) {
      await this.init(renderer);
      if (!this._gpu) return;
    }
    const { device } = this._gpu;
    const buffers = this._bvh.buffers;
    if (!buffers) return;
    if (this._grid.dirty) this._grid.allocateAtlases();

    const probeCount = this._grid.probeCount;
    if (probeCount === 0) return;

    // Rebuild BVH GPU buffers if scene changed.
    const bvhVersion = buffers.bvhNodes.length + buffers.positions.length;
    if (bvhVersion !== this._lastBvhVersion) {
      this._rebuildBvhBuffers(device, buffers);
      this._lastBvhVersion = bvhVersion;
    }

    // Reallocate ray results buffer if probe count changed.
    const maxProbes = probeCount;
    if (maxProbes !== this._maxProbes) {
      this._gpu.rayResultsBuf.destroy();
      this._gpu.rayResultsBuf = device.createBuffer({
        size: maxProbes * RAYS_PER_PROBE * PROBE_RAY_STRIDE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._maxProbes = maxProbes;
    }

    // Determine which probes are active this frame.
    const activeProbes: number[] = [];
    for (let i = offset; i < probeCount; i += stride) {
      activeProbes.push(i);
    }
    if (activeProbes.length === 0) return;

    // Upload active probe indices.
    const activeArr = new Uint32Array(activeProbes);
    const activeSize = activeArr.byteLength;
    if (this._gpu.activeProbesBuf.size < activeSize) {
      this._gpu.activeProbesBuf.destroy();
      this._gpu.activeProbesBuf = device.createBuffer({
        size: activeSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(this._gpu.activeProbesBuf, 0, activeArr);

    // Update uniforms.
    this._uploadMaterials(device, buffers.materials);
    this._uploadLights(device);
    this._uploadGridParams(device);
    this._uploadFrameParams(device);
    this._uploadBlendParams(device);

    // Ensure grid atlases are allocated.
    if (!this._grid.irradianceA) this._grid.allocateAtlases();

    // Get/create GPU textures for the atlases.
    const irrReadTex  = this._getOrCreateAtlasTexture(device, this._grid.irradianceReadTex!, 'rgba16float');
    const irrWriteTex = this._getOrCreateAtlasTexture(device, this._grid.irradianceWriteTex!, 'rgba16float');
    // Visibility atlas: allocated as RGBAFormat (rgba16float) because WebGPU does not
    // support rg16float as a storage texture. The WGSL shader declares rgba16float too.
    const visReadTex  = this._getOrCreateAtlasTexture(device, this._grid.visibilityReadTex!, 'rgba16float');
    const visWriteTex = this._getOrCreateAtlasTexture(device, this._grid.visibilityWriteTex!, 'rgba16float');

    // Run compute passes.
    const encoder = device.createCommandEncoder();
    this._runRaysPass(encoder, activeProbes.length, irrReadTex);
    this._runBlendIrrPass(encoder, activeProbes.length, irrReadTex, irrWriteTex);
    this._runBlendVisPass(encoder, activeProbes.length, visReadTex, visWriteTex);
    device.queue.submit([encoder.finish()]);

    // Swap ping-pong atlases.
    this._grid.swap();
    this._frameIndex++;

    // Expose internal state for debug/e2e inspection when opted in.
    if (this._debug && typeof window !== 'undefined') {
      (window as unknown as Record<string, unknown>)['__DDGI__'] = {
        frameIndex: this._frameIndex,
        probeCount,
        activeCount: activeProbes.length,
      };
    }
  }

  private _rebuildBvhBuffers(
    device: GPUDevice,
    buffers: SceneBvhBuffers,
  ): void {
    const RO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const upload = (oldBuf: GPUBuffer, data: ArrayBufferLike): GPUBuffer => {
      oldBuf.destroy();
      const arr = data instanceof ArrayBuffer ? data : new Uint8Array(data).buffer;
      const buf = device.createBuffer({
        size: Math.max(arr.byteLength, 16),
        usage: RO,
      });
      device.queue.writeBuffer(buf, 0, arr);
      return buf;
    };
    this._gpu!.bvhBuf  = upload(this._gpu!.bvhBuf,  buffers.bvhNodes.buffer);
    this._gpu!.posBuf  = upload(this._gpu!.posBuf,   buffers.positions.buffer);
    this._gpu!.idxBuf  = upload(this._gpu!.idxBuf,   buffers.indices.buffer);
    this._gpu!.normBuf = upload(this._gpu!.normBuf,  buffers.normals.buffer);
    this._gpu!.matIdBuf= upload(this._gpu!.matIdBuf, buffers.triMaterialId.buffer);
  }

  private _uploadMaterials(device: GPUDevice, mats: THREE.Material[]): void {
    const buf = packDDGIMaterials(mats);
    device.queue.writeBuffer(this._gpu!.materialsBuf, 0, buf);
  }

  private _uploadLights(device: GPUDevice): void {
    // DDGILightUniforms:
    // u32 count, 3 pad, then up to 16 × DDGILight (80 bytes each)
    // DDGILight: kind(u32), pad0,pad1,pad2(3×f32), pos(vec3f), intensity(f32),
    //            dir(vec3f), innerCone(f32), color(vec3f), outerCone(f32)
    // = 4 + 12 + 12 + 4 + 12 + 4 + 12 + 4 = 64 bytes per light
    const MAX = 16;
    const LIGHT_STRIDE = 16; // floats per light (64 bytes)
    const headerSize = 4; // floats for count + 3 pad
    const data = new Float32Array(headerSize + MAX * LIGHT_STRIDE);
    const udata = new Uint32Array(data.buffer);

    const lights = this._lights.filter(l => l.on);
    udata[0] = Math.min(lights.length, MAX);

    lights.slice(0, MAX).forEach((l, i) => {
      const base = (headerSize + i * LIGHT_STRIDE);
      const ubase = base;
      if (l.kind === 'sun') {
        udata[ubase] = 0; // LIGHT_SUN
        // Apply the hybrid pipeline's primaryLightIntensity multiplier
        // so DDGI's per-probe Le bake matches shade.wgsl's Lo_emit.
        // Without this, the stored l.intensity (typically 1.0) makes
        // DDGI 1/5 the magnitude of the rest of the renderer.
        data[base + 4] = 0;    // pos.x
        data[base + 5] = 0;    // pos.y
        data[base + 6] = 0;    // pos.z
        data[base + 7] = l.intensity * this._sunIntensityMul; // intensity
        data[base + 8] = 0;    // dir.x
        data[base + 9] = -1;   // dir.y (sun is from above)
        data[base + 10] = 0;   // dir.z
        data[base + 11] = 0;   // innerCone (unused for sun)
        data[base + 12] = 1;   // color.r
        data[base + 13] = 0.95;// color.g
        data[base + 14] = 0.85;// color.b
        data[base + 15] = 0;   // outerCone (unused for sun)
      } else if (l.kind === 'fixture' || l.kind === 'teaLight') {
        udata[ubase] = 1; // LIGHT_POINT
        const pos = l.position;
        data[base + 4]  = pos?.x ?? 0;
        data[base + 5]  = pos?.y ?? 0;
        data[base + 6]  = pos?.z ?? 0;
        data[base + 7]  = l.intensity;
        data[base + 8]  = 0;
        data[base + 9]  = 0;
        data[base + 10] = 0;
        data[base + 11] = 0;
        data[base + 12] = 1;
        data[base + 13] = 1;
        data[base + 14] = 1;
        data[base + 15] = 0;
      }
    });
    device.queue.writeBuffer(this._gpu!.lightsBuf, 0, data.buffer);
  }

  private _uploadGridParams(device: GPUDevice): void {
    // Use the canonical packer shared with HybridEngine — single source
    // for the 64-byte DDGI grid-params UBO layout.
    const buf = packDDGIGridParams(this._grid.params);
    device.queue.writeBuffer(this._gpu!.gridParamsBuf, 0, buf);
  }

  private _uploadFrameParams(device: GPUDevice): void {
    // 8 floats / 32 bytes; aliased u32 view shares the storage:
    //   data[0..2]  → randomRotation: vec3f  (random per-frame Y rotation)
    //   u32[3]      → frameIndex: u32
    //   u32[4]      → probeCount: u32
    //   u32[5]      → probesPerFrame: u32 (ceil(probeCount / 4))
    //   data[6..7]  → pad to 32 bytes (std140 vec4 alignment)
    const data = new Float32Array(8);
    const u32 = new Uint32Array(data.buffer);
    data[0] = Math.random() * Math.PI * 2;
    data[1] = Math.random() * Math.PI * 2;
    data[2] = Math.random() * Math.PI * 2;
    u32[3] = this._frameIndex;
    u32[4] = this._grid.probeCount;
    u32[5] = Math.ceil(this._grid.probeCount / 4);
    device.queue.writeBuffer(this._gpu!.frameParamsBuf, 0, data.buffer);
  }

  private _uploadBlendParams(device: GPUDevice): void {
    const data = new Float32Array(4);
    const u32 = new Uint32Array(data.buffer);
    u32[0] = Math.ceil(this._grid.probeCount / 4);
    data[1] = 0.97; // HYSTERESIS
    device.queue.writeBuffer(this._gpu!.blendParamsBuf, 0, data.buffer);
  }

  // Cache: AtlasTextureSlot identity → GPUTexture. The slot is a plain
  // record (just width/height); we use it as a WeakMap key so the
  // GPUTexture is released when ProbeGrid drops the slot.
  private _textureCache = new WeakMap<AtlasTextureSlot, GPUTexture>();

  private _getOrCreateAtlasTexture(
    device: GPUDevice,
    slot: AtlasTextureSlot,
    format: GPUTextureFormat,
  ): GPUTexture {
    const cached = this._textureCache.get(slot);
    if (cached) return cached;

    const gpuTex = device.createTexture({
      size: [slot.width, slot.height, 1],
      format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this._textureCache.set(slot, gpuTex);
    return gpuTex;
  }

  private _runRaysPass(
    encoder: GPUCommandEncoder,
    activeCount: number,
    irrReadTex: GPUTexture,
  ): void {
    const g = this._gpu!;

    const bg0 = g.device.createBindGroup({
      layout: g.raysPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: g.bvhBuf } },
        { binding: 1, resource: { buffer: g.posBuf } },
        { binding: 2, resource: { buffer: g.idxBuf } },
        { binding: 3, resource: { buffer: g.normBuf } },
        { binding: 4, resource: { buffer: g.matIdBuf } },
      ],
    });
    const bg1 = g.device.createBindGroup({
      layout: g.raysPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: g.materialsBuf } },
        { binding: 1, resource: { buffer: g.lightsBuf } },
      ],
    });
    const bg2 = g.device.createBindGroup({
      layout: g.raysPipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: g.rayResultsBuf } },
        { binding: 1, resource: { buffer: g.activeProbesBuf } },
        { binding: 2, resource: irrReadTex.createView() },
        { binding: 3, resource: g.linearSampler },
        { binding: 4, resource: { buffer: g.gridParamsBuf } },
        { binding: 5, resource: { buffer: g.frameParamsBuf } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'ddgi-probe-rays' });
    pass.setPipeline(g.raysPipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.setBindGroup(2, bg2);
    // Dispatch one workgroup per active probe.
    pass.dispatchWorkgroups(activeCount);
    pass.end();
  }

  private _runBlendIrrPass(
    encoder: GPUCommandEncoder,
    activeCount: number,
    irrReadTex: GPUTexture,
    irrWriteTex: GPUTexture,
  ): void {
    const g = this._gpu!;
    const bg0 = g.device.createBindGroup({
      layout: g.blendIrrPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: g.rayResultsBuf } },
        { binding: 1, resource: { buffer: g.activeProbesBuf } },
        { binding: 2, resource: { buffer: g.gridParamsBuf } },
        { binding: 3, resource: { buffer: g.blendParamsBuf } },
      ],
    });
    const bg1 = g.device.createBindGroup({
      layout: g.blendIrrPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: irrReadTex.createView() },
        { binding: 1, resource: g.linearSampler },
        { binding: 2, resource: irrWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
      ],
    });

    // Dispatch: global x = activeCount * IRR_CELL, global y = IRR_CELL.
    // workgroup (8,8,1) so dispatchWorkgroups(activeCount, 1, 1) covers one probe.
    const pass = encoder.beginComputePass({ label: 'ddgi-blend-irr' });
    pass.setPipeline(g.blendIrrPipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.dispatchWorkgroups(activeCount, 1, 1);
    pass.end();
  }

  private _runBlendVisPass(
    encoder: GPUCommandEncoder,
    activeCount: number,
    visReadTex: GPUTexture,
    visWriteTex: GPUTexture,
  ): void {
    const g = this._gpu!;
    const bg0 = g.device.createBindGroup({
      layout: g.blendVisPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: g.rayResultsBuf } },
        { binding: 1, resource: { buffer: g.activeProbesBuf } },
        { binding: 2, resource: { buffer: g.gridParamsBuf } },
        { binding: 3, resource: { buffer: g.blendParamsBuf } },
      ],
    });
    const bg1 = g.device.createBindGroup({
      layout: g.blendVisPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: visReadTex.createView() },
        { binding: 1, resource: g.linearSampler },
        { binding: 2, resource: visWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'ddgi-blend-vis' });
    pass.setPipeline(g.blendVisPipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.dispatchWorkgroups(activeCount, 1, 1);
    pass.end();
  }

  /**
   * Expose the cached read-side GPUTextures so external consumers
   * (e.g. the hybrid pipeline) can bind them into the ReSTIR shade pass
   * via WalkaroundGPUPipeline.setDDGIInputs(...). Returns null if compute
   * hasn't yet allocated the textures (called before first runFrame).
   *
   * The textures are flagged with TEXTURE_BINDING + STORAGE_BINDING usage
   * (see _getOrCreateAtlasTexture above), so they're directly bindable as
   * `texture_2d<f32>` in the shade pass.
   */
  getReadAtlasGPUTextures(): { irradiance: GPUTexture; visibility: GPUTexture } | null {
    const irrTex = this._grid.irradianceReadTex;
    const visTex = this._grid.visibilityReadTex;
    if (!irrTex || !visTex) return null;
    const irrGpu = this._textureCache.get(irrTex);
    const visGpu = this._textureCache.get(visTex);
    if (!irrGpu || !visGpu) return null;
    return { irradiance: irrGpu, visibility: visGpu };
  }

  dispose(): void {
    if (!this._gpu) return;
    const g = this._gpu;
    g.bvhBuf.destroy();
    g.posBuf.destroy();
    g.idxBuf.destroy();
    g.normBuf.destroy();
    g.matIdBuf.destroy();
    g.materialsBuf.destroy();
    g.lightsBuf.destroy();
    g.gridParamsBuf.destroy();
    g.frameParamsBuf.destroy();
    g.blendParamsBuf.destroy();
    g.rayResultsBuf.destroy();
    g.activeProbesBuf.destroy();
    this._gpu = null;
    this._textureCache = new WeakMap();
  }
}
