/**
 * RCDispatcher — per-frame cascade compute scheduler (raw WebGPU).
 *
 * Extracted from `_staging/legacy-source/src/rendering/scene/walkaround/cascadeDispatch.ts`.
 *
 * TSL → raw WebGPU conversion (RD-12).  See TSL_TO_RAW_MAPPING.md for the
 * full mapping rationale.  Summary:
 *   - TSL `storage()` → explicit `GPUBindGroupLayoutEntry` + `GPUBindGroup`
 *   - TSL `compute(fn, [64])` → `GPUComputePipeline` + `passEncoder.dispatchWorkgroups(ceil(n/64))`
 *   - TSL `instanceIndex` → `@builtin(global_invocation_id)` in WGSL entry point
 *   - `wgslFn()` bodies → pre-assembled WGSL module strings (see rc/wgsl/)
 *
 * The converted class:
 *   1. Allocates one `GPURenderPipeline` per cascade pass (cast: 5, merge: 4 = 9 total).
 *   2. Creates one bind group per pass from the caller-supplied GPU buffer handles.
 *   3. Dispatches passes in order: cast C0→C4, then merge C3→C0.
 *
 * Verification status: structural (TypeScript compile + binding-shape unit tests).
 * Behavioral (GPU render correctness) has NOT been verified.
 * See README.md Known Issues for residual risk.
 *
 * Dispatch counts — preserved from source TSL compute() arguments:
 *   Cast pass k:  totalRays = probes[0]*probes[1]*probes[2]*rays
 *                 workgroups = ceil(totalRays / 64)
 *   Merge pass k: totalLower = lowerDim.probes[0]*lowerDim.probes[1]*lowerDim.probes[2]*lowerDim.rays
 *                 workgroups = ceil(totalLower / 64)
 *
 * Workgroup size: 64 — matches `.compute(totalRays, [64])` in original.
 */

import * as THREE from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import type { StorageBufferAttribute } from 'three/webgpu';
import { CASCADE_DIMS, CASCADE_COUNT, fillCascadeDebug, type CascadeBuffers } from './cascadePyramid.js';
import type { SceneBVH } from './bvhCompute.js';
import { PROBE_RAY_CAST_WGSL } from './wgsl/probeRayCast.wgsl.js';
import { CASCADE_MERGE_WGSL } from './wgsl/cascadeMerge.wgsl.js';

// ─── Internal types ───────────────────────────────────────────────────────────

interface CastPassHandles {
  pipeline:  GPUComputePipeline;
  /** Uniform buffer (GPUBuffer wrapping a Float32Array aligned to CascadeUniforms). */
  uniformBuf: GPUBuffer;
  /** CPU-side backing for the uniform buffer — updated each frame. */
  uniformRaw: Float32Array;
  /** Workgroup dispatch count = ceil(totalRays / 64). */
  dispatchX:  number;
}

interface MergePassHandles {
  pipeline:  GPUComputePipeline;
  /** Uniform buffer for MergeUniforms. */
  uniformBuf: GPUBuffer;
  mergeRaw:   Float32Array;
  /** Workgroup dispatch count = ceil(totalLower / 64). */
  dispatchX:  number;
}

interface DispatchHandles {
  castPasses:   CastPassHandles[];
  mergePasses:  MergePassHandles[];
  /** The env texture view bound in every cast pass at binding 6. */
  envTextureView: GPUTextureView;
  /** The env sampler bound in every cast pass at binding 7. */
  envSampler:     GPUSampler;
  /** Bind groups: one array<GPUBindGroup> per cast pass (index k), one per merge pass. */
  castBindGroups:  GPUBindGroup[];
  mergeBindGroups: GPUBindGroup[];
}

// ─── Public interface ─────────────────────────────────────────────────────────

export interface RCDispatchOpts {
  /** Three.js WebGPU renderer — used to access the raw GPUDevice and backend. */
  gl:             WebGPURenderer;
  sceneBVH:       SceneBVH;
  cascadeBuffers: CascadeBuffers;
  sunDirection:   THREE.Vector3;
  sunColor:       THREE.Color;
  envEquirect:    THREE.Texture | null;
  frameSeed:      number;
  /** Smoke-test / fallback mode: fill cascades with debug colours, skip ray-cast. */
  debugFill?:     boolean;
}

// ─── Uniform data builders ────────────────────────────────────────────────────
// These are the exact same packing functions as the original cascadeDispatch.ts.

/** Write CascadeUniforms into an existing Float32Array (avoids realloc per frame). */
function buildCascadeUniformDataInto(
  d: Float32Array,
  k: number,
  cb: CascadeBuffers,
  sunDir: THREE.Vector3,
  sunColor: THREE.Color,
  envIntensity: number,
  frameSeed: number,
): void {
  const dim = CASCADE_DIMS[k]!;
  const rayGridSize = Math.round(Math.sqrt(dim.rays));
  const { probeOriginWorld: o, roomSize: s } = cb;
  // CascadeUniforms layout (matches WGSL struct in probeRayCast.wgsl.ts):
  // probeOriginWorld(3f), _pad0(f)
  // roomSize(3f), _pad1(f)
  // probeCount(3u), raysPerProbe(u)
  // rayGridSize(u), intervalNear(f), intervalFar(f), cascadeIndex(u)
  // sunDirection(3f), _pad2(f)
  // sunColor(3f), envIntensity(f)
  // frameSeed(u), lastCascade(u), _pad4(2u)
  // Total: 40 float/uint values = 160 bytes
  const ui = new Uint32Array(d.buffer);
  d[0]  = o.x; d[1]  = o.y; d[2]  = o.z; d[3]  = 0;
  d[4]  = s.x; d[5]  = s.y; d[6]  = s.z; d[7]  = 0;
  ui[8] = dim.probes[0]; ui[9] = dim.probes[1]; ui[10] = dim.probes[2];
  ui[11] = dim.rays;
  ui[12] = rayGridSize;
  d[13] = dim.intervalNear; d[14] = dim.intervalFar;
  ui[15] = k;
  d[16] = sunDir.x; d[17] = sunDir.y; d[18] = sunDir.z; d[19] = 0;
  d[20] = sunColor.r; d[21] = sunColor.g; d[22] = sunColor.b;
  d[23] = envIntensity;
  ui[24] = frameSeed;
  ui[25] = CASCADE_COUNT - 1;
  ui[26] = 0; ui[27] = 0;
}

function buildMergeUniformData(
  lowerDim: (typeof CASCADE_DIMS)[number],
  upperDim: (typeof CASCADE_DIMS)[number],
  cb: CascadeBuffers,
): Float32Array {
  // MergeUniforms layout (matches WGSL struct in cascadeMerge.wgsl.ts):
  // lowerProbeCount(3u), lowerRayCount(u)
  // upperProbeCount(3u), upperRayCount(u)
  // lowerRayGridSize(u), upperRayGridSize(u), _pad0(2u)
  // probeOriginWorld(3f), _pad1(f)
  // roomSize(3f), _pad2(f)
  // Total: 20 float/uint values = 80 bytes
  const d = new Float32Array(20);
  const ui = new Uint32Array(d.buffer);
  const { probeOriginWorld: o, roomSize: s } = cb;
  ui[0]  = lowerDim.probes[0]; ui[1]  = lowerDim.probes[1]; ui[2]  = lowerDim.probes[2];
  ui[3]  = lowerDim.rays;
  ui[4]  = upperDim.probes[0]; ui[5]  = upperDim.probes[1]; ui[6]  = upperDim.probes[2];
  ui[7]  = upperDim.rays;
  ui[8]  = Math.round(Math.sqrt(lowerDim.rays));
  ui[9]  = Math.round(Math.sqrt(upperDim.rays));
  ui[10] = 0; ui[11] = 0;
  d[12]  = o.x; d[13] = o.y; d[14] = o.z; d[15] = 0;
  d[16]  = s.x; d[17] = s.y; d[18] = s.z; d[19] = 0;
  return d;
}

// ─── GPU buffer helpers ───────────────────────────────────────────────────────

/**
 * Retrieve the raw `GPUBuffer` backing a `StorageBufferAttribute`.
 *
 * Three.js WebGPU renderer allocates the GPU buffer on first use and stores
 * it on the attribute's `__gpuBuffer` property (renderer-internal).  Callers
 * must ensure the renderer has already processed the attribute (i.e. the
 * scene has been rendered at least once) before calling `initialize()`.
 *
 * This is the same access pattern the Three.js backend itself uses internally.
 */
function gpuBufferOf(attr: StorageBufferAttribute): GPUBuffer {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const buf = (attr as unknown as Record<string, unknown>)['__gpuBuffer'] as GPUBuffer | undefined;
  if (!buf) {
    throw new Error(
      '[RCDispatcher] StorageBufferAttribute GPU buffer not yet allocated. ' +
      'Ensure the Three.js WebGPU renderer has processed the scene before calling initialize().',
    );
  }
  return buf;
}

// ─── RCDispatcher class ───────────────────────────────────────────────────────

/**
 * Manages the RC cascade compute pipeline (raw WebGPU).
 *
 * Lifecycle:
 *   1. Construct with `new RCDispatcher()`.
 *   2. Call `dispatchFrame(opts)` each frame.  Handles lazy init internally.
 *   3. Call `dispose()` to release GPU resources.
 */
export class RCDispatcher {
  private _handles: DispatchHandles | null = null;
  private _castShaderModule:  GPUShaderModule | null = null;
  private _mergeShaderModule: GPUShaderModule | null = null;

  /**
   * Dispatch the cascade compute pipeline for one frame.
   *
   * On the first call (or after `dispose()`), pipelines and bind groups are
   * compiled/created lazily.  If WebGPU is not the active backend, falls back
   * to filling all cascades with debug colours.
   */
  async dispatchFrame(opts: RCDispatchOpts): Promise<void> {
    const { gl, cascadeBuffers } = opts;

    if (opts.debugFill) {
      this._debugFill(cascadeBuffers);
      return;
    }

    // Guard: compute dispatch requires a real WebGPU backend.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyGl = gl as any;
    if (anyGl.backend?.isWebGPUBackend !== true) {
      this._debugFill(cascadeBuffers);
      return;
    }

    const device: GPUDevice = anyGl.backend.device as GPUDevice;

    // Lazy init.
    if (!this._handles) {
      try {
        this._handles = this._buildHandles(device, opts);
      } catch (err: unknown) {
        console.error('[RCDispatcher] buildHandles failed:', err);
        return;
      }
    }
    const handles = this._handles;

    // Update per-frame uniforms for each cast pass.
    for (let k = 0; k < CASCADE_COUNT; k++) {
      const pass = handles.castPasses[k]!;
      buildCascadeUniformDataInto(
        pass.uniformRaw, k, cascadeBuffers, opts.sunDirection, opts.sunColor, 1.0, opts.frameSeed,
      );
      device.queue.writeBuffer(pass.uniformBuf, 0, pass.uniformRaw.buffer as ArrayBuffer);
    }

    // Encode compute commands.
    const commandEncoder = device.createCommandEncoder({ label: 'RCDispatcher' });
    const passEncoder = commandEncoder.beginComputePass({ label: 'rc-cascade' });

    // Cast passes C0 → C4.
    for (let k = 0; k < CASCADE_COUNT; k++) {
      const pass = handles.castPasses[k]!;
      passEncoder.setPipeline(pass.pipeline);
      passEncoder.setBindGroup(0, handles.castBindGroups[k]!);
      passEncoder.dispatchWorkgroups(pass.dispatchX);
    }

    // Merge passes C3 → C0 (bottom-up).
    for (let m = 0; m < handles.mergePasses.length; m++) {
      const pass = handles.mergePasses[m]!;
      passEncoder.setPipeline(pass.pipeline);
      passEncoder.setBindGroup(0, handles.mergeBindGroups[m]!);
      passEncoder.dispatchWorkgroups(pass.dispatchX);
    }

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
  }

  /** Release all GPU resources. Next `dispatchFrame()` will re-initialize. */
  dispose(): void {
    if (this._handles) {
      for (const pass of this._handles.castPasses) {
        pass.uniformBuf.destroy();
      }
      for (const pass of this._handles.mergePasses) {
        pass.uniformBuf.destroy();
      }
      this._handles = null;
    }
    this._castShaderModule  = null;
    this._mergeShaderModule = null;
  }

  // ─── Private helpers ────────────────────────────────────────────────────────

  private _debugFill(cascadeBuffers: CascadeBuffers): void {
    fillCascadeDebug(cascadeBuffers);
    for (const attr of cascadeBuffers.gpuCascades) attr.needsUpdate = true;
  }

  /** Build bind group layout for a cast pass (9 entries: 5 BVH+mat SSBOs + cascade +
   *  env texture + sampler + uniform). */
  private _castBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: 'rc-cast-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bvh
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // geom_index
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // geom_position
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // materials
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // triMatId
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // cascadeOut (rw)
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // CascadeUniforms
      ],
    });
  }

  /** Build bind group layout for a merge pass (3 entries: upper + lower cascades + uniforms). */
  private _mergeBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: 'rc-merge-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // upperCascade
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // lowerCascade (rw)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // MergeUniforms
      ],
    });
  }

  /**
   * Build all pipelines and bind groups (one-time setup).
   * Called lazily on first `dispatchFrame()`.
   */
  private _buildHandles(device: GPUDevice, opts: RCDispatchOpts): DispatchHandles {
    const { sceneBVH, cascadeBuffers } = opts;

    // Compile shader modules (shared across all cast passes / merge passes).
    if (!this._castShaderModule) {
      this._castShaderModule = device.createShaderModule({
        label:  'rc-probe-ray-cast',
        code:   PROBE_RAY_CAST_WGSL,
      });
    }
    if (!this._mergeShaderModule) {
      this._mergeShaderModule = device.createShaderModule({
        label:  'rc-cascade-merge',
        code:   CASCADE_MERGE_WGSL,
      });
    }

    const castBGL  = this._castBindGroupLayout(device);
    const mergeBGL = this._mergeBindGroupLayout(device);

    const castPipelineLayout  = device.createPipelineLayout({ bindGroupLayouts: [castBGL] });
    const mergePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [mergeBGL] });

    // BVH GPU buffers (shared across all cast passes).
    const bvhBuf      = gpuBufferOf(sceneBVH.bvhNodes);
    const idxBuf      = gpuBufferOf(sceneBVH.indices);
    const posBuf      = gpuBufferOf(sceneBVH.positions);
    const matBuf      = gpuBufferOf(sceneBVH.materials);
    const triMatBuf   = gpuBufferOf(sceneBVH.triMaterialId);

    // Env texture + sampler.
    const fallbackEnv = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    fallbackEnv.needsUpdate = true;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const envThree = opts.envEquirect ?? fallbackEnv;
    // Access the Three.js WebGPU renderer's internal texture handle.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyGl = opts.gl as any;
    const backend = anyGl.backend;
    const envGpuData = backend.get(envThree) as { texture?: GPUTexture } | undefined;
    const envGpuTex  = envGpuData?.texture;

    let envTextureView: GPUTextureView;
    let envSampler: GPUSampler;

    if (envGpuTex) {
      envTextureView = envGpuTex.createView({ label: 'rc-env-view' });
      envSampler = device.createSampler({
        label:        'rc-env-sampler',
        magFilter:    'linear',
        minFilter:    'linear',
        addressModeU: 'repeat',
        addressModeV: 'clamp-to-edge',
      });
    } else {
      // Fallback: create a 1×1 placeholder until the renderer uploads the env texture.
      const placeholderTex = device.createTexture({
        label:  'rc-env-placeholder',
        size:   [1, 1],
        format: 'rgba8unorm',
        usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: placeholderTex },
        new Uint8Array([0, 0, 0, 255]),
        { bytesPerRow: 4 },
        [1, 1],
      );
      envTextureView = placeholderTex.createView({ label: 'rc-env-placeholder-view' });
      envSampler = device.createSampler({ label: 'rc-env-placeholder-sampler' });
    }

    // ── Cast passes (one per cascade) ──────────────────────────────────────
    const castPasses: CastPassHandles[] = [];
    const castBindGroups: GPUBindGroup[] = [];

    for (let k = 0; k < CASCADE_COUNT; k++) {
      const dim = CASCADE_DIMS[k]!;
      const totalRays = dim.probes[0] * dim.probes[1] * dim.probes[2] * dim.rays;

      // Create per-pass pipeline.
      const pipeline = device.createComputePipeline({
        label:  `rc-cast-C${k}`,
        layout: castPipelineLayout,
        compute: {
          module:     this._castShaderModule!,
          entryPoint: 'probeRayCastKernel',
        },
      });

      // Per-pass uniform buffer for CascadeUniforms (40 floats = 160 bytes).
      const uniformRaw = new Float32Array(40);
      buildCascadeUniformDataInto(uniformRaw, k, cascadeBuffers, opts.sunDirection, opts.sunColor, 1.0, opts.frameSeed);
      const uniformBuf = device.createBuffer({
        label:  `rc-cast-C${k}-uniforms`,
        size:   uniformRaw.byteLength,
        usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(uniformBuf.getMappedRange()).set(uniformRaw);
      uniformBuf.unmap();

      // Cascade output buffer.
      const cascadeAttr = cascadeBuffers.gpuCascades[k]!;
      const cascadeBuf  = gpuBufferOf(cascadeAttr);

      // Build bind group.
      const bindGroup = device.createBindGroup({
        label:  `rc-cast-C${k}-bg`,
        layout: castBGL,
        entries: [
          { binding: 0, resource: { buffer: bvhBuf } },
          { binding: 1, resource: { buffer: idxBuf } },
          { binding: 2, resource: { buffer: posBuf } },
          { binding: 3, resource: { buffer: matBuf } },
          { binding: 4, resource: { buffer: triMatBuf } },
          { binding: 5, resource: { buffer: cascadeBuf } },
          { binding: 6, resource: envTextureView },
          { binding: 7, resource: envSampler },
          { binding: 8, resource: { buffer: uniformBuf } },
        ],
      });

      castPasses.push({ pipeline, uniformBuf, uniformRaw, dispatchX: Math.ceil(totalRays / 64) });
      castBindGroups.push(bindGroup);
    }

    // ── Merge passes (bottom-up: C(N-2) → C0) ────────────────────────────
    const mergePasses: MergePassHandles[] = [];
    const mergeBindGroups: GPUBindGroup[] = [];

    for (let lower = CASCADE_COUNT - 2; lower >= 0; lower--) {
      const lowerDim = CASCADE_DIMS[lower]!;
      const upperDim = CASCADE_DIMS[lower + 1]!;
      const totalLower = lowerDim.probes[0] * lowerDim.probes[1] * lowerDim.probes[2] * lowerDim.rays;

      const pipeline = device.createComputePipeline({
        label:  `rc-merge-${lower}→${lower + 1}`,
        layout: mergePipelineLayout,
        compute: {
          module:     this._mergeShaderModule!,
          entryPoint: 'cascadeMergeKernel',
        },
      });

      // MergeUniforms buffer (20 floats = 80 bytes).
      const mergeRaw = buildMergeUniformData(lowerDim, upperDim, cascadeBuffers);
      const uniformBuf = device.createBuffer({
        label:  `rc-merge-${lower}-uniforms`,
        size:   mergeRaw.byteLength,
        usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(uniformBuf.getMappedRange()).set(mergeRaw);
      uniformBuf.unmap();

      const lowerBuf = gpuBufferOf(cascadeBuffers.gpuCascades[lower]!);
      const upperBuf = gpuBufferOf(cascadeBuffers.gpuCascades[lower + 1]!);

      const bindGroup = device.createBindGroup({
        label:  `rc-merge-${lower}-bg`,
        layout: mergeBGL,
        entries: [
          { binding: 0, resource: { buffer: upperBuf } },
          { binding: 1, resource: { buffer: lowerBuf } },
          { binding: 2, resource: { buffer: uniformBuf } },
        ],
      });

      mergePasses.push({ pipeline, uniformBuf, mergeRaw, dispatchX: Math.ceil(totalLower / 64) });
      mergeBindGroups.push(bindGroup);
    }

    return { castPasses, mergePasses, envTextureView, envSampler, castBindGroups, mergeBindGroups };
  }
}

// ─── Backward-compatible functional API ──────────────────────────────────────
// Mirrors the original `dispatchCascadePasses(opts)` signature for drop-in use.

const _sharedDispatcher = new RCDispatcher();

/**
 * Convenience wrapper: dispatch cascade passes using a shared singleton `RCDispatcher`.
 * This matches the original functional `dispatchCascadePasses()` API.
 */
export async function dispatchCascadePasses(opts: RCDispatchOpts): Promise<void> {
  return _sharedDispatcher.dispatchFrame(opts);
}
