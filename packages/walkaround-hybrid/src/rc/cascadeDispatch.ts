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

/** Narrow view of three.js WebGPU backend for raw buffer + texture binding. */
interface WebGPUBackendView {
  readonly isWebGPUBackend?: boolean;
  readonly device?: GPUDevice;
  get?: (resource: unknown) => unknown;
}

// ─── Internal types ───────────────────────────────────────────────────────────

interface CastPassHandles {
  pipeline:  GPUComputePipeline;
  /** Uniform buffer (GPUBuffer wrapping a Float32Array aligned to CascadeUniforms). */
  cascadeParamsBuf: GPUBuffer;
  /** CPU-side backing for the uniform buffer — updated each frame. */
  cascadeParamsRaw: Float32Array;
  /** Workgroup dispatch count = ceil(totalRays / 64). */
  dispatchX:  number;
}

interface MergePassHandles {
  pipeline:  GPUComputePipeline;
  /** Uniform buffer for MergeUniforms. */
  cascadeParamsBuf: GPUBuffer;
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
  /**
   * E2 — Möller–Trumbore coplanarity threshold (default 1e-5 for metre-scale).
   * Plumbed from HybridEngine.triIntersectEpsilon into CascadeUniforms so the
   * RC shader uses the same epsilon as the WalkaroundUBO shaders.
   */
  triIntersectEpsilon?: number;
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
  triIntersectEpsilon: number,  // E2: UBO-plumbed (was local WGSL const)
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
  // frameSeed(u), lastCascade(u), triIntersectEpsilon(f), _pad4a(u)
  // Total: 40 float/uint values = 160 bytes
  const ui = new Uint32Array(d.buffer);
  // W8 Phase 1A — `probeOriginWorld` / `roomSize` are now plain `[x,y,z]`
  // tuples (was `THREE.Vector3`). Index access matches the WGSL packing.
  d[0]  = o[0]; d[1]  = o[1]; d[2]  = o[2]; d[3]  = 0;
  d[4]  = s[0]; d[5]  = s[1]; d[6]  = s[2]; d[7]  = 0;
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
  d[26] = triIntersectEpsilon;  // E2: was _pad4[0]
  ui[27] = 0;                   // _pad4a
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
  // W8 Phase 1A — plain `[x,y,z]` tuples (was `THREE.Vector3`).
  d[12]  = o[0]; d[13] = o[1]; d[14] = o[2]; d[15] = 0;
  d[16]  = s[0]; d[17] = s[1]; d[18] = s[2]; d[19] = 0;
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
    const backend = (gl as WebGPURenderer).backend as WebGPUBackendView | undefined;
    if (backend?.isWebGPUBackend !== true || backend.device == null) {
      this._debugFill(cascadeBuffers);
      return;
    }

    const device: GPUDevice = backend.device;

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
        pass.cascadeParamsRaw, k, cascadeBuffers, opts.sunDirection, opts.sunColor, 1.0, opts.frameSeed,
        opts.triIntersectEpsilon ?? 1e-5,
      );
      device.queue.writeBuffer(pass.cascadeParamsBuf, 0, pass.cascadeParamsRaw.buffer as ArrayBuffer);
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
        pass.cascadeParamsBuf.destroy();
      }
      for (const pass of this._handles.mergePasses) {
        pass.cascadeParamsBuf.destroy();
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
   * Resolve the environment texture GPU binding. Accesses the Three.js WebGPU
   * renderer's internal `backend.get(texture)` handle to find the uploaded
   * `GPUTexture`. Falls back to a 1×1 black placeholder when the renderer has
   * not yet uploaded the env texture (common on first frame).
   *
   * Extracted from `_buildHandles` to keep the 173-line setup method readable.
   * (WARM-3 fix: was embedded mid-function between BVH extraction and cast loop.)
   */
  private _buildEnvBinding(
    device: GPUDevice,
    opts: RCDispatchOpts,
  ): { envTextureView: GPUTextureView; envSampler: GPUSampler } {
    const fallbackEnv = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    fallbackEnv.needsUpdate = true;
    const envThree = opts.envEquirect ?? fallbackEnv;
    const backend = opts.gl.backend as WebGPUBackendView | undefined;
    const envGpuData = backend?.get?.(envThree) as { texture?: GPUTexture } | undefined;
    const envGpuTex  = envGpuData?.texture;

    if (envGpuTex) {
      return {
        envTextureView: envGpuTex.createView({ label: 'rc-env-view' }),
        envSampler: device.createSampler({
          label:        'rc-env-sampler',
          magFilter:    'linear',
          minFilter:    'linear',
          addressModeU: 'repeat',
          addressModeV: 'clamp-to-edge',
        }),
      };
    }

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
    return {
      envTextureView: placeholderTex.createView({ label: 'rc-env-placeholder-view' }),
      envSampler: device.createSampler({ label: 'rc-env-placeholder-sampler' }),
    };
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
    const { envTextureView, envSampler } = this._buildEnvBinding(device, opts);

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

      // Per-pass STORAGE buffer for CascadeUniforms (40 floats = 160 bytes).
      // Backed by storage (not UNIFORM) because the 160-byte struct exceeds
      // the default maxUniformBufferBindingSize on low-end adapters; the
      // shader binds it as `read-only-storage` which has no such limit.
      // envIntensity is fixed at 1.0 by design: tone mapping is applied per-
      // material downstream, and environment-level scaling is intentionally
      // not exposed at the RC dispatch level. If a future requirement needs
      // it, add `envIntensity?: number` to `RCDispatchOpts` and thread it
      // through.
      const cascadeParamsRaw = new Float32Array(40);
      buildCascadeUniformDataInto(cascadeParamsRaw, k, cascadeBuffers, opts.sunDirection, opts.sunColor, 1.0, opts.frameSeed, opts.triIntersectEpsilon ?? 1e-5);
      const cascadeParamsBuf = device.createBuffer({
        label:  `rc-cast-C${k}-uniforms`,
        size:   cascadeParamsRaw.byteLength,
        usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(cascadeParamsBuf.getMappedRange()).set(cascadeParamsRaw);
      cascadeParamsBuf.unmap();

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
          { binding: 8, resource: { buffer: cascadeParamsBuf } },
        ],
      });

      castPasses.push({ pipeline, cascadeParamsBuf, cascadeParamsRaw, dispatchX: Math.ceil(totalRays / 64) });
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
      const cascadeParamsBuf = device.createBuffer({
        label:  `rc-merge-${lower}-uniforms`,
        size:   mergeRaw.byteLength,
        usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(cascadeParamsBuf.getMappedRange()).set(mergeRaw);
      cascadeParamsBuf.unmap();

      const lowerBuf = gpuBufferOf(cascadeBuffers.gpuCascades[lower]!);
      const upperBuf = gpuBufferOf(cascadeBuffers.gpuCascades[lower + 1]!);

      const bindGroup = device.createBindGroup({
        label:  `rc-merge-${lower}-bg`,
        layout: mergeBGL,
        entries: [
          { binding: 0, resource: { buffer: upperBuf } },
          { binding: 1, resource: { buffer: lowerBuf } },
          { binding: 2, resource: { buffer: cascadeParamsBuf } },
        ],
      });

      mergePasses.push({ pipeline, cascadeParamsBuf, mergeRaw, dispatchX: Math.ceil(totalLower / 64) });
      mergeBindGroups.push(bindGroup);
    }

    return { castPasses, mergePasses, envTextureView, envSampler, castBindGroups, mergeBindGroups };
  }
}

// ─── Backward-compatible functional API ──────────────────────────────────────
// Mirrors the original `dispatchCascadePasses(opts)` signature for drop-in use.

// Single-canvas-scoped by design: this functional API serves the legacy /
// single-host call sites where there is exactly one RC canvas per page.
// Multi-canvas / multi-instance hosts should instantiate `RCDispatcher`
// directly (the class is exported) — a shared dispatcher would otherwise
// share state across canvases that should be independent.
const _sharedDispatcher = new RCDispatcher();

/**
 * Convenience wrapper: dispatch cascade passes using a shared singleton `RCDispatcher`.
 * This matches the original functional `dispatchCascadePasses()` API.
 *
 * @deprecated Module-level singletons violate CLAUDE.md Design Principle 2
 * ("the host owns lifecycle"). Use `new RCDispatcher()` per host and call
 * `dispatcher.dispatchFrame(opts)` directly. Single-canvas hosts can keep
 * one instance for the page; multi-canvas hosts MUST instantiate per
 * dispatcher (the shared singleton corrupts state across independent
 * canvases). Retained for backward compatibility with legacy callers;
 * no production consumer is left in this monorepo (only one test exercises
 * the surface to keep it un-broken).
 */
export async function dispatchCascadePasses(opts: RCDispatchOpts): Promise<void> {
  return _sharedDispatcher.dispatchFrame(opts);
}

/**
 * Tear down GPU resources held by the module-level shared dispatcher used
 * by `dispatchCascadePasses`. Hosts should call this on canvas unmount or
 * page teardown to avoid leaking pipelines / bind groups on hot reload.
 *
 * No-op if `dispatchCascadePasses` was never called for the current page.
 * After calling, subsequent `dispatchCascadePasses` calls reinitialize.
 *
 * @deprecated See `dispatchCascadePasses` deprecation note — instantiate
 * `RCDispatcher` per host and call `.dispose()` on your own instance.
 */
export function disposeSharedDispatcher(): void {
  _sharedDispatcher.dispose();
}
