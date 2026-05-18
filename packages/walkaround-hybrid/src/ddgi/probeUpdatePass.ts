/**
 * ProbeUpdatePass — DDGI probe update orchestrator (raw WebGPU compute).
 *
 * Drives the 5-stage DDGI probe update chain per frame:
 *   1. {@link ProbeRaysPass}              — ray-cast 192 rays per active probe via inline BVH traversal.
 *   2. {@link ProbeBlendIrradiancePass}   — EWMA blend ray results into octahedral irradiance atlas.
 *   3. {@link ProbeBlendVisibilityPass}   — EWMA blend hit distances into octahedral visibility atlas.
 *   4. {@link ProbeBorderIrradiancePass}  — Majercik 2019 §3.2 border-pixel replication (irradiance).
 *   5. {@link ProbeBorderVisibilityPass}  — Majercik 2019 §3.2 border-pixel replication (visibility).
 *
 * Each per-phase Pass owns its compiled pipeline + per-dispatch bind-group
 * construction. The orchestrator owns the shared GPU state (BVH buffers,
 * UBOs, ray-results SSBO, scratch textures, atlas ping-pong via ProbeGrid)
 * and constructs a single {@link ProbePassContext} per frame that every
 * pass reads from.
 *
 * Why raw WebGPU rather than TSL/wgslFn: the compute shaders use custom
 * `@group/@binding` layouts that don't compose naturally with three.js's
 * binding system. The WebGPU backend's `device` property is used directly.
 *
 * three/webgpu coupling: `renderer.backend.device` is accessed directly,
 * but ProbeGrid atlas slots are now backend-agnostic `AtlasTextureSlot`
 * records (see probeGrid.ts). The GPU texture per slot is allocated lazily
 * in `_getOrCreateAtlasTexture` and cached in `_textureCache` keyed on the
 * slot identity. TSL binding (if applyDDGIShading is in use) is the only
 * site still importing from three/webgpu.
 */

import * as THREE from 'three';
import type { SceneBvh, SceneBvhBuffers } from '@vitrum/shared-bvh';
import type { ProbeGrid, AtlasTextureSlot } from './probeGrid.js';
import type { DDGILight } from './types.js';
import { packDDGIGridParams } from '../pipeline/resourceManager.js';
import { detectGpu } from '@vitrum/core';
import { RAYS_PER_PROBE } from './ddgiConstants.js';
import {
  packDDGIMaterials,
  packDDGIMaterialsN,
  DDGI_MAX_MATERIALS,
  DDGI_MATERIAL_STRIDE_BYTES,
  DDGI_MATERIAL_ENTRY_FLOATS,
} from './ddgiMaterialPacker.js';
import { ProbeRaysPass } from './passes/probeRaysPass.js';
import { ProbeBlendIrradiancePass } from './passes/probeBlendIrradiancePass.js';
import { ProbeBlendVisibilityPass } from './passes/probeBlendVisibilityPass.js';
import { ProbeBorderIrradiancePass } from './passes/probeBorderIrradiancePass.js';
import { ProbeBorderVisibilityPass } from './passes/probeBorderVisibilityPass.js';
import type { ProbePassContext } from './passes/probePassTypes.js';

// Re-exports so existing consumers (`import { RAYS_PER_PROBE,
// packDDGIMaterials, DDGI_MAX_MATERIALS, … } from
// 'walkaround-hybrid/ddgi/probeUpdatePass'`) keep working. The constants
// and pure-function packers live in dedicated modules to break the ESM
// import cycle between this module and its WGSL template files, and to
// keep the orchestrator focused on the 5-stage dispatch chain.
export {
  RAYS_PER_PROBE,
  packDDGIMaterials,
  DDGI_MAX_MATERIALS,
  DDGI_MATERIAL_STRIDE_BYTES,
  DDGI_MATERIAL_ENTRY_FLOATS,
};

// ProbeRay struct: 12 floats / 2 u32 → 16 × 4 bytes = 64 bytes each
const PROBE_RAY_STRIDE_BYTES = 64;

/**
 * Shared GPU resources owned by the orchestrator and threaded through to
 * every per-phase Pass via {@link ProbePassContext}.
 *
 * The pipelines themselves live inside the per-phase Pass instances
 * (`_raysPass`, `_blendIrrPass`, `_blendVisPass`, `_borderIrrPass`,
 * `_borderVisPass`); the orchestrator only holds shared buffers, scratch
 * textures, and the sampler.
 */
interface GPUResources {
  device: GPUDevice;
  /**
   * Scratch atlas textures for the border fill pass ping-pong.
   *
   * WebGPU forbids binding the same texture as both `texture_2d` (read) and
   * `texture_storage_2d` (write) in a single pipeline. The border pass reads
   * from a scratch copy of the blend output and writes the border pixels back
   * into the blend-output atlas. The host copies write→scratch with
   * `copyTextureToTexture` between the blend and border dispatches.
   *
   * The scratch textures are reallocated lazily when the atlas size changes
   * (keyed on a `width×height` size tag stored in `_irrScratchSize` and
   * `_visScratchSize`). They are destroyed in `dispose()`.
   */
  irrScratchTex: GPUTexture | null;
  visScratchTex: GPUTexture | null;
  // BVH buffers (replaced on scene rebuild)
  bvhBuf:        GPUBuffer;
  posBuf:        GPUBuffer;
  idxBuf:        GPUBuffer;
  normBuf:       GPUBuffer;
  matIdBuf:      GPUBuffer;
  // Uniform buffers
  materialsBuf:    GPUBuffer;
  lightsBuf:       GPUBuffer;
  gridParamsBuf:   GPUBuffer;
  frameParamsBuf:  GPUBuffer;
  blendParamsBuf:  GPUBuffer;
  /** BorderUBO for the irradiance border pass (8 u32 fields = 32 bytes). */
  borderIrrUboBuf: GPUBuffer;
  /** BorderUBO for the visibility border pass (8 u32 fields = 32 bytes). */
  borderVisUboBuf: GPUBuffer;
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
  /**
   * Maximum number of distinct materials supported by the DDGI probe pass.
   * Injected as a compile-time constant into the `probeUpdateRays.wgsl`
   * shader (`array<DDGIMaterial, N>`). Must match the `materialsBuf` size
   * allocated at init (M × DDGI_MATERIAL_STRIDE_BYTES bytes).
   *
   * Defaults to 64. Raise for scenes with more unique materials.
   * Do NOT raise above WebGPU's uniform-buffer array limit (~4096).
   *
   * @since Sprint 16 (M9 audit remediation)
   */
  maxMaterials?: number;
}

export class ProbeUpdatePass {
  private _bvh:  SceneBvh;
  private _grid: ProbeGrid;
  private _gpu:  GPUResources | null = null;
  private _lastBvhVersion = -1;
  private _frameIndex = 0;
  private _maxProbes = 0;
  private _lights: DDGILight[] = [];
  /** Cached size tag for the irradiance scratch texture (`width|height` string). */
  private _irrScratchSize = '';
  /** Cached size tag for the visibility scratch texture (`width|height` string). */
  private _visScratchSize = '';
  private _debug: boolean;
  // Multiplier applied to every light's intensity when packing the
  // probe-update light UBO. Defaults to 1 so unrelated callers keep the
  // original behaviour. The hybrid pipeline calls setSunIntensityMultiplier(5.0)
  // so DDGI's per-probe Le matches the same primaryLightIntensity used by
  // shade.wgsl — without this, DDGI runs at 1/5 the magnitude of Lo_emit
  // and walls render dark.
  private _sunIntensityMul = 1;

  // Sky tint and irradiance scale used by sampleSkyColor() in
  // probeUpdateRays.wgsl for miss rays. Defaults replicate the legacy
  // hardcoded gradient midpoint so Cornell results are unchanged.
  // (B2 audit remediation — previously hardcoded in WGSL).
  private _skyTint: [number, number, number] = [0.4, 0.6, 1.0];
  private _skyIrradiance = 2.0;

  // Max materials for the WGSL compile-time array size (M9 audit remediation).
  private _ddgiMaxMaterials: number;

  // Per-phase Pass instances — each owns its compiled pipeline + bind-group
  // construction. Constructed eagerly so we can await all five compiles in
  // parallel during init().
  private readonly _raysPass:      ProbeRaysPass;
  private readonly _blendIrrPass:  ProbeBlendIrradiancePass;
  private readonly _blendVisPass:  ProbeBlendVisibilityPass;
  private readonly _borderIrrPass: ProbeBorderIrradiancePass;
  private readonly _borderVisPass: ProbeBorderVisibilityPass;

  constructor(bvh: SceneBvh, grid: ProbeGrid, opts: ProbeUpdatePassOptions = {}) {
    this._bvh  = bvh;
    this._grid = grid;
    this._debug = opts.debug ?? false;
    this._ddgiMaxMaterials = opts.maxMaterials ?? DDGI_MAX_MATERIALS;
    if (this._ddgiMaxMaterials < 1) {
      console.warn(`[DDGI] ProbeUpdatePass: maxMaterials=${this._ddgiMaxMaterials} is invalid; clamping to 1.`);
      this._ddgiMaxMaterials = 1;
    }
    this._raysPass      = new ProbeRaysPass(this._ddgiMaxMaterials);
    this._blendIrrPass  = new ProbeBlendIrradiancePass();
    this._blendVisPass  = new ProbeBlendVisibilityPass();
    this._borderIrrPass = new ProbeBorderIrradiancePass();
    this._borderVisPass = new ProbeBorderVisibilityPass();
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
   * Override the sky tint and irradiance scale used by probe miss-rays.
   * Written into {@link FrameParams} at offsets 32–44 so `sampleSkyColor`
   * in WGSL reads scene-specific values rather than a Cornell-tuned gradient.
   *
   * Defaults: tint=(0.4,0.6,1.0) and irradiance=2.0 (matches the former
   * hardcoded WGSL values — no regression for Cornell builds).
   *
   * @param tint      Linear-sRGB sky colour (HDR; values above 1.0 are valid).
   * @param irradiance Scalar multiplier on top of the tint. At 1.0 the tint
   *                   is used as-is. At 2.0 the sky is twice as bright as a
   *                   unit-white sky, which suits open-outdoor scenes.
   */
  setSkyParams(tint: [number, number, number], irradiance: number): void {
    this._skyTint = tint;
    this._skyIrradiance = irradiance;
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

    // Compile all five per-phase pipelines in parallel.
    try {
      await Promise.all([
        this._raysPass.compile(device),
        this._blendIrrPass.compile(device),
        this._blendVisPass.compile(device),
        this._borderIrrPass.compile(device),
        this._borderVisPass.compile(device),
      ]);
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

    // BorderUBO layout: 8 u32 fields = 32 bytes (std140, vec4-aligned pairs).
    // Fields: numProbes, atlasWidth, atlasHeight, _pad0, gridDimX, gridDimY,
    //         gridDimZ, _pad1. Matches struct in probeUpdateBorder.wgsl.ts.
    const BORDER_UBO_BYTES = 32;

    this._gpu = {
      device,
      irrScratchTex:  null,
      visScratchTex:  null,
      bvhBuf:          makeBuffer(16, RO),
      posBuf:          makeBuffer(12, RO),
      idxBuf:          makeBuffer(12, RO),
      normBuf:         makeBuffer(12, RO),
      matIdBuf:        makeBuffer(4,  RO),
      materialsBuf:    makeBuffer(this._ddgiMaxMaterials * DDGI_MATERIAL_STRIDE_BYTES, UB),
      lightsBuf:       makeBuffer(16 * 80 + 16, UB),
      gridParamsBuf:   makeBuffer(64, UB),
      frameParamsBuf:  makeBuffer(48, UB),
      blendParamsBuf:  makeBuffer(16, UB),
      borderIrrUboBuf: makeBuffer(BORDER_UBO_BYTES, UB),
      borderVisUboBuf: makeBuffer(BORDER_UBO_BYTES, UB),
      rayResultsBuf:   makeBuffer(PROBE_RAY_STRIDE_BYTES, RW),
      activeProbesBuf: makeBuffer(4, RO),
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

    // Allocate/refresh border-fill scratch textures (lazy on size change).
    const irrScratch = this._getOrCreateScratchTexture(device, irrWriteTex, 'irr');
    const visScratch = this._getOrCreateScratchTexture(device, visWriteTex, 'vis');

    // Upload the BorderUBOs for this frame (atlas size + grid dims).
    this._uploadBorderUbo(device, irrWriteTex, 'irr');
    this._uploadBorderUbo(device, visWriteTex, 'vis');

    // Build the per-frame dispatch context handed to every Pass.
    const ctx: ProbePassContext = {
      device,
      activeCount: activeProbes.length,
      probeCount,
      irrReadTex,
      irrWriteTex,
      visReadTex,
      visWriteTex,
      irrScratchTex: irrScratch,
      visScratchTex: visScratch,
      bvhBuf:          this._gpu.bvhBuf,
      posBuf:          this._gpu.posBuf,
      idxBuf:          this._gpu.idxBuf,
      normBuf:         this._gpu.normBuf,
      matIdBuf:        this._gpu.matIdBuf,
      materialsBuf:    this._gpu.materialsBuf,
      lightsBuf:       this._gpu.lightsBuf,
      gridParamsBuf:   this._gpu.gridParamsBuf,
      frameParamsBuf:  this._gpu.frameParamsBuf,
      blendParamsBuf:  this._gpu.blendParamsBuf,
      borderIrrUboBuf: this._gpu.borderIrrUboBuf,
      borderVisUboBuf: this._gpu.borderVisUboBuf,
      rayResultsBuf:   this._gpu.rayResultsBuf,
      activeProbesBuf: this._gpu.activeProbesBuf,
      linearSampler:   this._gpu.linearSampler,
    };

    // Run the 5-stage probe-update dispatch chain.
    const encoder = device.createCommandEncoder();
    this._raysPass.dispatch(encoder, ctx);

    // CRITICAL: copy read→write so inactive probes' cells stay in sync.
    // Without this, the per-frame ping-pong (`swap()` below) combined with
    // round-robin updates (STRIDE=8, only 1/8 of probes active per frame)
    // produces a structural bug: every probe always writes to the SAME
    // atlas (parity-matched by probeIdx % STRIDE × frame % 2), and reads
    // its `prev` from the OTHER atlas which it never writes to → prev=0.
    // Hysteresis 0.97 with prev=0 collapses the EMA to 0.03 × newColor,
    // a 33× attenuation that explains why DDGI atlas peaked at ~0.05
    // despite per-ray red-wall radiance of ~6.0. Copying read→write
    // before blend means inactive probes' cells already hold the latest
    // value (from when they were last active, one full STRIDE cycle ago).
    encoder.copyTextureToTexture(
      { texture: irrReadTex },
      { texture: irrWriteTex },
      { width: irrReadTex.width, height: irrReadTex.height, depthOrArrayLayers: 1 },
    );
    encoder.copyTextureToTexture(
      { texture: visReadTex },
      { texture: visWriteTex },
      { width: visReadTex.width, height: visReadTex.height, depthOrArrayLayers: 1 },
    );

    this._blendIrrPass.dispatch(encoder, ctx);
    this._blendVisPass.dispatch(encoder, ctx);

    // Border fill pass (Item 3 — Majercik 2019 §3.2).
    //
    // After blend, `irrWriteTex` and `visWriteTex` have correct interior pixels
    // but zeroed border pixels. We can't bind the same texture as both
    // `texture_2d` (read) and `texture_storage_2d` (write) in a single pipeline
    // pass, so we use a scratch ping-pong:
    //   1. copy write → scratch (so border pass reads complete interior from scratch)
    //   2. border pass reads from scratch, writes border pixels into write
    //
    // The scratch textures are allocated lazily and cached in `_gpu.irrScratchTex`
    // / `_gpu.visScratchTex`, reused every frame as long as atlas size is stable.
    encoder.copyTextureToTexture(
      { texture: irrWriteTex },
      { texture: irrScratch },
      { width: irrWriteTex.width, height: irrWriteTex.height, depthOrArrayLayers: 1 },
    );
    encoder.copyTextureToTexture(
      { texture: visWriteTex },
      { texture: visScratch },
      { width: visWriteTex.width, height: visWriteTex.height, depthOrArrayLayers: 1 },
    );
    this._borderIrrPass.dispatch(encoder, ctx);
    this._borderVisPass.dispatch(encoder, ctx);

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
    // M9: runtime warning when scene exceeds the compiled-in cap.
    if (mats.length > this._ddgiMaxMaterials) {
      console.warn(
        `[DDGI] Scene has ${mats.length} materials but ddgiMaxMaterials=${this._ddgiMaxMaterials}. ` +
        `Materials beyond the cap are ignored. Raise ddgiMaxMaterials in HybridEngineOptions to fix.`,
      );
    }
    const buf = packDDGIMaterialsN(mats, this._ddgiMaxMaterials);
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
    // 12 floats / 48 bytes; aliased u32 view shares the storage:
    //   data[0..2]  → randomRotation: vec3f  (per-frame ray-direction jitter)
    //   u32[3]      → frameIndex: u32
    //   u32[4]      → probeCount: u32
    //   u32[5]      → probesPerFrame: u32 (ceil(probeCount / 4))
    //   data[6..7]  → _pad0, _pad1 (std140 vec4 alignment)
    //   data[8..10] → skyTint: vec3f  (B2 audit: was hardcoded in WGSL)
    //   data[11]    → skyIrradiance: f32
    //
    // Per-frame deterministic SO(3) rotation (Shoemake 1992 "Uniform Random
    // Rotations"). Uses three Halton-base-{2,3,5} quasi-random values seeded
    // by the frame index to produce a uniform-on-SO(3) quaternion, then
    // converts to an axis-angle vec3 consumed by the WGSL probeUpdateRays
    // shader. Replacing the previous all-zeros fixed rotation:
    //   - Decorrelates probe ray samples across frames so the EMA hysteresis
    //     accumulates an effectively larger ray budget over time.
    //   - Eliminates the 192-fixed-direction aliasing that the EMA could not
    //     suppress at 0.97 hysteresis.
    //   - Halton rather than Math.random() avoids correlation clumps.
    // Reference: Majercik et al. 2019 §3.1; Shoemake 1992.
    //
    // (Previous comment noted that the (0,0,0) freeze was a band-aid for
    // EMA instability — that root cause is fixed by the M7 energy-model
    // correction; per-frame rotation is now safe to restore.)
    //
    // NOTE: this Halton-Shoemake sampler will move to @vitrum/shared-samplers
    // in W7-H5 (separate branch). Leaving it inline here per the W4-A8
    // scope boundary.
    const haltonBase = (i: number, base: number): number => {
      let result = 0;
      let f = 1;
      let n = i;
      while (n > 0) {
        f /= base;
        result += f * (n % base);
        n = Math.floor(n / base);
      }
      return result;
    };
    const fi = this._frameIndex + 1;
    const u1 = haltonBase(fi, 2);
    const u2 = haltonBase(fi, 3);
    const u3 = haltonBase(fi, 5);
    // Shoemake quaternion form (uniform distribution on SO(3)).
    const sigma1 = Math.sqrt(1 - u1);
    const sigma2 = Math.sqrt(u1);
    const theta1 = 2 * Math.PI * u2;
    const theta2 = 2 * Math.PI * u3;
    const qw = sigma2 * Math.cos(theta2);
    const qx = sigma1 * Math.sin(theta1);
    const qy = sigma1 * Math.cos(theta1);
    const qz = sigma2 * Math.sin(theta2);
    // Convert quaternion → axis-angle vec3 (WGSL consumer applies this as
    // a Rodrigues rotation to each probe ray direction).
    const angle = 2 * Math.acos(Math.min(1, Math.abs(qw)));
    const sinHalf = Math.sqrt(Math.max(0, 1 - qw * qw));
    let ax: number, ay: number, az: number;
    if (sinHalf < 1e-6) {
      ax = 1; ay = 0; az = 0; // identity — no rotation
    } else {
      ax = qx / sinHalf;
      ay = qy / sinHalf;
      az = qz / sinHalf;
    }
    const data = new Float32Array(12);
    const u32 = new Uint32Array(data.buffer);
    data[0] = ax * angle;
    data[1] = ay * angle;
    data[2] = az * angle;
    u32[3] = this._frameIndex;
    u32[4] = this._grid.probeCount;
    u32[5] = Math.ceil(this._grid.probeCount / 4);
    // data[6..7] = 0 (pad — already zeroed by Float32Array constructor)
    data[8]  = this._skyTint[0];
    data[9]  = this._skyTint[1];
    data[10] = this._skyTint[2];
    data[11] = this._skyIrradiance;
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
      // COPY_SRC | COPY_DST so the host can keep the two ping-pong atlases
      // synchronised via `copyTextureToTexture` at the start of each blend
      // dispatch (inactive probes' cells must mirror the read texture or
      // the round-robin EMA collapses to 3% of newColor per cycle).
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });

    this._textureCache.set(slot, gpuTex);
    return gpuTex;
  }

  /**
   * Return (or lazily create) the scratch texture used by the border fill
   * pass for the given atlas. The scratch texture is an atlas-sized
   * rgba16float texture that serves as a read-only source for the border
   * pass so we don't need to bind the same texture for both read and write.
   *
   * @param device - WebGPU device.
   * @param atlas  - The write-side atlas texture whose dimensions to match.
   * @param which  - 'irr' for irradiance, 'vis' for visibility.
   */
  private _getOrCreateScratchTexture(
    device: GPUDevice,
    atlas: GPUTexture,
    which: 'irr' | 'vis',
  ): GPUTexture {
    const g = this._gpu!;
    const sizeTag = `${atlas.width}|${atlas.height}`;
    if (which === 'irr') {
      if (g.irrScratchTex && this._irrScratchSize === sizeTag) return g.irrScratchTex;
      g.irrScratchTex?.destroy();
      g.irrScratchTex = device.createTexture({
        size: [atlas.width, atlas.height, 1],
        format: 'rgba16float',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      this._irrScratchSize = sizeTag;
      return g.irrScratchTex;
    } else {
      if (g.visScratchTex && this._visScratchSize === sizeTag) return g.visScratchTex;
      g.visScratchTex?.destroy();
      g.visScratchTex = device.createTexture({
        size: [atlas.width, atlas.height, 1],
        format: 'rgba16float',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      this._visScratchSize = sizeTag;
      return g.visScratchTex;
    }
  }

  /**
   * Upload the BorderUBO for the irradiance or visibility border pass.
   * The UBO layout (8 × u32 = 32 bytes):
   *   [0] numProbes   [1] atlasWidth  [2] atlasHeight  [3] _pad0
   *   [4] gridDimX    [5] gridDimY    [6] gridDimZ      [7] _pad1
   */
  private _uploadBorderUbo(
    device: GPUDevice,
    atlas: GPUTexture,
    which: 'irr' | 'vis',
  ): void {
    const g = this._gpu!;
    const data = new Uint32Array(8);
    data[0] = this._grid.probeCount;
    data[1] = atlas.width;
    data[2] = atlas.height;
    data[3] = 0; // _pad0
    data[4] = this._grid.params.dims.x;
    data[5] = this._grid.params.dims.y;
    data[6] = this._grid.params.dims.z;
    data[7] = 0; // _pad1
    const buf = which === 'irr' ? g.borderIrrUboBuf : g.borderVisUboBuf;
    device.queue.writeBuffer(buf, 0, data.buffer);
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
    // Pass-owned pipeline refs always safe to drop.
    this._raysPass.dispose();
    this._blendIrrPass.dispose();
    this._blendVisPass.dispose();
    this._borderIrrPass.dispose();
    this._borderVisPass.dispose();

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
    g.borderIrrUboBuf.destroy();
    g.borderVisUboBuf.destroy();
    g.irrScratchTex?.destroy();
    g.visScratchTex?.destroy();
    g.rayResultsBuf.destroy();
    g.activeProbesBuf.destroy();
    this._gpu = null;
    this._textureCache = new WeakMap();
    this._irrScratchSize = '';
    this._visScratchSize = '';
  }
}
