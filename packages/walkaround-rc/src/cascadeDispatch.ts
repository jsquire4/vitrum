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
 *   1. Allocates one `GPUComputePipeline` per cascade pass (cast: 5, merge: 4 = 9 total).
 *   2. Creates one bind group per pass from the caller-supplied GPU buffer handles.
 *   3. Dispatches passes in order: cast C0→C4, then merge C3→C0.
 *
 * Verification status: structural (TypeScript compile + binding-shape unit tests)
 * PLUS behavioral — the RC merged-BVH path is CPU-brute-force-oracle-validated
 * (tree-shape-invariant, 100% vs ground truth) and exercised by the rcEnabled
 * GPU smoke added with the F-RC1 stride fix (781f66f); the cascade-zero light-
 * model gaps were resolved 2026-06-07 (596c341 RC-has-energy gate + 1e893fa
 * probe-cast emitter NEE, converged A/B 999 dB). See README.md for residual risk.
 *
 * Dispatch counts — preserved from source TSL compute() arguments:
 *   Cast pass k:  totalRays = probes[0]*probes[1]*probes[2]*rays
 *                 workgroups = ceil(totalRays / 64)
 *   Merge pass k: totalLower = lowerDim.probes[0]*lowerDim.probes[1]*lowerDim.probes[2]*lowerDim.rays
 *                 workgroups = ceil(totalLower / 64)
 *
 * Workgroup size: 64 — matches `.compute(totalRays, [64])` in original.
 */

import { CASCADE_DIMS, type CascadeDim } from './cascadePyramid.js';
import { PROBE_RAY_CAST_WGSL } from './wgsl/probeRayCast.wgsl.js';
import { CASCADE_MERGE_WGSL } from './wgsl/cascadeMerge.wgsl.js';

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
  /** Owned placeholder env texture when caller provided none. */
  placeholderEnvTexture?: GPUTexture;
}

// ─── Public interface ─────────────────────────────────────────────────────────

/**
 * Raw-GPU dispatch options. Hosts that own a `GPUDevice` + raw `GPUBuffer`
 * handles (e.g. `HybridEngine`, which runs the WGSL shade pipeline directly
 * and does not use THREE's WebGPU renderer backend) pass this directly to
 * `RCDispatcher.dispatchFrameRaw`.
 *
 * History: the W8 Phase 1B refactor (2026-05-18) split this out from a
 * THREE-tied `RCDispatchOpts` that extracted GPU handles via
 * `StorageBufferAttribute.__gpuBuffer` reach-through. The legacy THREE-tied
 * path was dropped on 2026-05-18 once `RCSubsystem` (the in-engine consumer)
 * was confirmed to be the only call site and was already using this raw
 * variant. The `__gpuBuffer` accessor is gone from the dispatcher.
 */
export interface RCDispatchOptsRaw {
  /** Raw WebGPU device — caller-owned. */
  device:             GPUDevice;

  /** BVH GPU buffers (5 separate SSBOs; layout matches the raw RC SceneBVH contract). */
  bvhNodesBuf:        GPUBuffer;
  bvhIndicesBuf:      GPUBuffer;
  bvhPositionsBuf:    GPUBuffer;
  materialsBuf:       GPUBuffer;
  triMaterialIdBuf:   GPUBuffer;

  /** One cascade-output `GPUBuffer` per cascade, same order as
   *  {@link CASCADE_DIMS}. The dispatcher writes into these (cast) and
   *  reads/writes parents (merge). */
  cascadeBufs:        readonly GPUBuffer[];

  /** Cascade geometry — plain tuples matching `CascadeBuffers.probeOriginWorld`/`roomSize`. */
  probeOriginWorld:   readonly [number, number, number];
  roomSize:           readonly [number, number, number];

  /** Sun direction (world space, normalised) and RGB tint. */
  sunDirection:       readonly [number, number, number];
  sunColor:           readonly [number, number, number];

  /** Environment equirectangular texture — caller supplies a pre-created
   *  view + sampler. Pass `null` to use the dispatcher's 1×1 black placeholder. */
  envTextureView?:    GPUTextureView | null;
  envSampler?:        GPUSampler | null;

  frameSeed:          number;
  /** Möller–Trumbore coplanarity threshold. Default 1e-5. */
  triIntersectEpsilon?: number;

  /** C2 — TLAS traversal (ReSTIR-shared buffers). Omit for merged-only RC BVH. */
  bvhMode?: 'merged' | 'tlas';
  tlasNodeCount?: number;
  tlasNodesBuf?: GPUBuffer;
  tlasInstanceIndicesBuf?: GPUBuffer;
  tlasBlasRootsBuf?: GPUBuffer;
  tlasInstanceWorldToLocalBuf?: GPUBuffer;
  tlasInstanceLocalToWorldBuf?: GPUBuffer;

  /** Rect-area emitter NEE (2026-06-07). The packed `array<EmitterTri>` buffer
   *  (80 bytes/tri — share the main pipeline's `BvhBufferHost._emitterBuffer`)
   *  + its triangle count. Omit (or count 0) to keep RC's prior light model
   *  (sun + emissive geometry + env); the dispatcher binds an 80-byte zero
   *  placeholder so the bind group stays valid. */
  emittersBuf?: GPUBuffer;
  emitterCount?: number;

  /** A7 (2026-06-10): packed `RCLightBuffer` (point/spot analytic lights).
   *  16-byte header (count + 3 pad) + up to 16 × 64-byte RCLight entries =
   *  1040 bytes total. The host packs this via `packRCLights()` in
   *  HybridEngineRC.ts. Omit to bind a 1040-byte zero placeholder (lightCount
   *  stays 0 → loop is a no-op, byte-identical with prior). */
  lightsBuf?: GPUBuffer;
  lightCount?: number;
}

// ─── Uniform data builders ────────────────────────────────────────────────────
// These are the exact same packing functions as the original cascadeDispatch.ts.

/** Write CascadeUniforms into an existing Float32Array (avoids realloc per frame).
 *
 * sunDir / sunColor / cascade geometry are plain `readonly [number, number, number]`
 * tuples — no `THREE.Vector3` / `THREE.Color` coupling.
 *
 * B3b (2026-05-19) — `dims` parameter replaces the module-level
 * `CASCADE_DIMS` lookup. Per-instance dims flow from
 * `HybridEngineOptions.cascadeDims` through `RCDispatcher.constructor`.
 */
function buildCascadeUniformDataInto(
  d: Float32Array,
  k: number,
  probeOriginWorld: readonly [number, number, number],
  roomSize:         readonly [number, number, number],
  sunDir:           readonly [number, number, number],
  sunColor:         readonly [number, number, number],
  envIntensity: number,
  frameSeed: number,
  triIntersectEpsilon: number,  // E2: UBO-plumbed (was local WGSL const)
  bvhMode: number,
  tlasNodeCount: number,
  emitterCount: number,
  lightCount: number,           // A7: point/spot analytic light count
  dims: readonly CascadeDim[] = CASCADE_DIMS,
): void {
  const dim = dims[k]!;
  const rayGridSize = Math.round(Math.sqrt(dim.rays));
  const o = probeOriginWorld;
  const s = roomSize;
  // CascadeUniforms layout (matches WGSL struct in probeRayCast.wgsl.ts):
  // probeOriginWorld(3f), _pad0(f)
  // roomSize(3f), _pad1(f)
  // probeCount(3u), raysPerProbe(u)
  // rayGridSize(u), intervalNear(f), intervalFar(f), cascadeIndex(u)
  // sunDirection(3f), _pad2(f)
  // sunColor(3f), envIntensity(f)
  // frameSeed(u), lastCascade(u), triIntersectEpsilon(f), bvhMode(u)
  // tlasNodeCount(u), emitterCount(u) [slot 29 — RC emitter NEE]
  // lightCount(u) [slot 30 — A7 point/spot lights], _pad3/4/5(u)
  // Total allocation 40 float/uint = 160 bytes.
  const ui = new Uint32Array(d.buffer);
  d[0]  = o[0]; d[1]  = o[1]; d[2]  = o[2]; d[3]  = 0;
  d[4]  = s[0]; d[5]  = s[1]; d[6]  = s[2]; d[7]  = 0;
  ui[8] = dim.probes[0]; ui[9] = dim.probes[1]; ui[10] = dim.probes[2];
  ui[11] = dim.rays;
  ui[12] = rayGridSize;
  d[13] = dim.intervalNear; d[14] = dim.intervalFar;
  ui[15] = k;
  d[16] = sunDir[0]; d[17] = sunDir[1]; d[18] = sunDir[2]; d[19] = 0;
  d[20] = sunColor[0]; d[21] = sunColor[1]; d[22] = sunColor[2];
  d[23] = envIntensity;
  ui[24] = frameSeed;
  ui[25] = dims.length - 1;
  d[26] = triIntersectEpsilon;  // E2: was _pad4[0]
  ui[27] = bvhMode >>> 0;
  ui[28] = tlasNodeCount >>> 0;
  ui[29] = emitterCount >>> 0;
  ui[30] = lightCount >>> 0;    // A7: point/spot analytic light count
  // ui[31..33] = _pad3/4/5 (zero from Float32Array init)
}

function buildMergeUniformData(
  lowerDim: CascadeDim,
  upperDim: CascadeDim,
  probeOriginWorld: readonly [number, number, number],
  roomSize:         readonly [number, number, number],
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
  const o = probeOriginWorld;
  const s = roomSize;
  ui[0]  = lowerDim.probes[0]; ui[1]  = lowerDim.probes[1]; ui[2]  = lowerDim.probes[2];
  ui[3]  = lowerDim.rays;
  ui[4]  = upperDim.probes[0]; ui[5]  = upperDim.probes[1]; ui[6]  = upperDim.probes[2];
  ui[7]  = upperDim.rays;
  ui[8]  = Math.round(Math.sqrt(lowerDim.rays));
  ui[9]  = Math.round(Math.sqrt(upperDim.rays));
  ui[10] = 0; ui[11] = 0;
  d[12]  = o[0]; d[13] = o[1]; d[14] = o[2]; d[15] = 0;
  d[16]  = s[0]; d[17] = s[1]; d[18] = s[2]; d[19] = 0;
  return d;
}

// ─── RCDispatcher class ───────────────────────────────────────────────────────

/**
 * Manages the RC cascade compute pipeline (raw WebGPU).
 *
 * Lifecycle:
 *   1. Construct with `new RCDispatcher()`.
 *   2. Call `dispatchFrameRaw(opts)` each frame.  Handles lazy init internally.
 *   3. Call `dispose()` to release GPU resources.
 *
 * History: this used to expose a THREE-tied `dispatchFrame(opts: RCDispatchOpts)`
 * that reached into `StorageBufferAttribute.__gpuBuffer` (renderer-internal) to
 * extract raw `GPUBuffer` handles. That path was dropped 2026-05-18 once
 * `RCSubsystem` was confirmed to be the only consumer and was already calling
 * the raw entry directly.
 */
export class RCDispatcher {
  private _handles: DispatchHandles | null = null;
  private _castShaderModule:  GPUShaderModule | null = null;
  private _mergeShaderModule: GPUShaderModule | null = null;
  private _lastError: Error | null = null;
  /** Dummy 32-byte TLAS placeholder buffers created in merged mode (see
   *  `_dummyStorageBuffer`). Tracked so `dispose()` can destroy them — they
   *  are not retained on `DispatchHandles`. */
  private _dummyTlasBuffers: GPUBuffer[] = [];
  /** B3b (2026-05-19) — per-instance cascade dimensions. Defaults to the
   *  Cornell-tuned `CASCADE_DIMS`; hosts override via constructor for
   *  non-Cornell aspect ratios / scene scales. */
  private readonly _cascadeDims: readonly CascadeDim[];

  constructor(cascadeDims: readonly CascadeDim[] = CASCADE_DIMS) {
    this._cascadeDims = cascadeDims;
  }

  get lastError(): Error | null {
    return this._lastError;
  }

  /**
   * Dispatch the cascade compute pipeline for one frame.
   * Pipelines and bind groups are compiled/created lazily on the first call
   * (or after `dispose()`).
   *
   * **Important — bind-group invalidation:** if `opts.bvhMode` changes between
   * calls (merged ↔ tlas), or if the AABB / probe-origin bounds change in a way
   * that requires different buffer bindings, the caller MUST call
   * `invalidateBindings()` BEFORE the next `dispatchFrameRaw()` call. Failing to
   * do so leaves the old bind groups bound against the new (different) pipeline
   * layout, which produces silent GPU validation errors or undefined rendering.
   *
   * `RCSubsystem.refitCascadeBounds()` already calls `invalidateBindings()` when
   * the scene AABB changes; callers that construct `RCDispatcher` directly are
   * responsible for calling it on `bvhMode` or buffer-set transitions.
   */
  dispatchFrameRaw(opts: RCDispatchOptsRaw): void {
    const device = opts.device;
    if (!this._handles) {
      try {
        this._handles = this._buildHandlesRaw(device, opts);
        this._lastError = null;
      } catch (err: unknown) {
        const error = err instanceof Error ? err : new Error(String(err));
        this._lastError = error;
        throw new Error(`[RCDispatcher] buildHandlesRaw failed: ${error.message}`);
      }
    }
    const handles = this._handles;

    // Update per-frame uniforms for each cast pass.
    const dims = this._cascadeDims;
    for (let k = 0; k < dims.length; k++) {
      const pass = handles.castPasses[k]!;
      const bvhMode = opts.bvhMode === 'tlas' ? 1 : 0;
      const tlasNodeCount = opts.tlasNodeCount ?? 0;
      const emitterCount = opts.emitterCount ?? 0;
      const lightCount = opts.lightCount ?? 0;
      buildCascadeUniformDataInto(
        pass.cascadeParamsRaw, k,
        opts.probeOriginWorld, opts.roomSize,
        opts.sunDirection, opts.sunColor,
        1.0, opts.frameSeed,
        opts.triIntersectEpsilon ?? 1e-5,
        bvhMode,
        tlasNodeCount,
        emitterCount,
        lightCount,
        dims,
      );
      device.queue.writeBuffer(pass.cascadeParamsBuf, 0, pass.cascadeParamsRaw.buffer);
    }

    // Encode compute commands.
    const commandEncoder = device.createCommandEncoder({ label: 'RCDispatcher' });
    const passEncoder = commandEncoder.beginComputePass({ label: 'rc-cascade' });

    // Cast passes C0 → C(N-1).
    for (let k = 0; k < dims.length; k++) {
      const pass = handles.castPasses[k]!;
      passEncoder.setPipeline(pass.pipeline);
      passEncoder.setBindGroup(0, handles.castBindGroups[k]);
      passEncoder.dispatchWorkgroups(pass.dispatchX);
    }

    // Merge passes C3 → C0 (bottom-up).
    for (let m = 0; m < handles.mergePasses.length; m++) {
      const pass = handles.mergePasses[m]!;
      passEncoder.setPipeline(pass.pipeline);
      passEncoder.setBindGroup(0, handles.mergeBindGroups[m]);
      passEncoder.dispatchWorkgroups(pass.dispatchX);
    }

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
  }

  /** Drop cached bind groups so the next dispatch captures fresh caller buffers. */
  invalidateBindings(): void {
    this._releaseHandles();
    this._lastError = null;
  }

  /** Release all GPU resources. Next `dispatchFrame()` will re-initialize. */
  dispose(): void {
    this._releaseHandles();
    this._lastError = null;
    this._castShaderModule  = null;
    this._mergeShaderModule = null;
  }

  // ─── Private helpers ────────────────────────────────────────────────────────

  /** Cast pass: BVH+mat SSBOs, cascade out, env, uniforms, optional TLAS (C2), analytic lights (A7). */
  private _castBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: 'rc-cast-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // rc_emitters
        // A7 (2026-06-10): analytic point/spot lights (rc_lights: RCLightBuffer).
        // 16-byte header + up to 16 × 64-byte entries = 1040 bytes; a 1040-byte zero
        // placeholder is bound when the scene has no point/spot fixtures.
        { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // rc_lights
      ],
    });
  }

  private _dummyStorageBuffer(device: GPUDevice, label: string, size = 32): GPUBuffer {
    // Merged mode (bvhMode == 0) never traverses the TLAS, but the probe-ray
    // shader STILL declares the five TLAS bindings (group 0, bindings 9–13). The
    // first, `rc_tlas_nodes: array<BVHNode>`, has a 32-byte struct stride → a
    // 32-byte MINIMUM binding size. A 16-byte placeholder is REJECTED by strict
    // backends ("Binding size 16 … less than minimum 32" on lavapipe AND dzn),
    // invalidating the bind group and silently zeroing the whole cascade. Use a
    // 32-byte empty placeholder so the merged-mode bind group is valid on every
    // backend; the u32 / vec4f TLAS bindings (10–13) accept 32 bytes too. The
    // TLAS-mode path uploads real, larger buffers, so it was never affected. This
    // is the exact analogue of the DDGI fix `ea88803` (same root cause); latent
    // because RC's probe shader had no GPU-compile/bind gate (W8 CPU-only) until
    // the RC core-BVH converged A/B exercised it.
    const buf = device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._dummyTlasBuffers.push(buf);
    return buf;
  }

  private _releaseHandles(): void {
    if (this._handles) {
      for (const pass of this._handles.castPasses) {
        pass.cascadeParamsBuf.destroy();
      }
      for (const pass of this._handles.mergePasses) {
        pass.cascadeParamsBuf.destroy();
      }
      this._handles.placeholderEnvTexture?.destroy();
      this._handles = null;
    }
    for (const buf of this._dummyTlasBuffers) {
      buf.destroy();
    }
    this._dummyTlasBuffers = [];
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
   * Resolve the env binding for the raw (THREE-free) path. When the caller
   * supplies both `envTextureView` and `envSampler`, use them. Otherwise
   * create a 1×1 black placeholder.
   */
  private _resolveEnvBindingRaw(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
  ): {
    envTextureView: GPUTextureView;
    envSampler: GPUSampler;
    placeholderEnvTexture?: GPUTexture;
  } {
    if (opts.envTextureView && opts.envSampler) {
      return {
        envTextureView: opts.envTextureView,
        envSampler: opts.envSampler,
      };
    }
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
      placeholderEnvTexture: placeholderTex,
    };
  }

  /**
   * Build all pipelines and bind groups (one-time setup).
   * Called lazily on first `dispatchFrameRaw()`.
   *
   * Takes {@link RCDispatchOptsRaw} (raw GPU types). The W8 Phase 1B
   * refactor (2026-05-18) extracted this from a THREE-tied builder; the
   * legacy entry was dropped the same day.
   */
  private _buildHandlesRaw(device: GPUDevice, opts: RCDispatchOptsRaw): DispatchHandles {
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

    // BVH GPU buffers (shared across all cast passes; caller-provided).
    const bvhBuf      = opts.bvhNodesBuf;
    const idxBuf      = opts.bvhIndicesBuf;
    const posBuf      = opts.bvhPositionsBuf;
    const matBuf      = opts.materialsBuf;
    const triMatBuf   = opts.triMaterialIdBuf;
    const tlasNodesBuf = opts.tlasNodesBuf ?? this._dummyStorageBuffer(device, 'rc-tlas-nodes-dummy');
    const tlasInstBuf = opts.tlasInstanceIndicesBuf ?? this._dummyStorageBuffer(device, 'rc-tlas-inst-dummy');
    const tlasBlasBuf = opts.tlasBlasRootsBuf ?? this._dummyStorageBuffer(device, 'rc-tlas-blas-dummy');
    const tlasW2lBuf = opts.tlasInstanceWorldToLocalBuf ?? this._dummyStorageBuffer(device, 'rc-tlas-w2l-dummy');
    const tlasL2wBuf = opts.tlasInstanceLocalToWorldBuf ?? this._dummyStorageBuffer(device, 'rc-tlas-l2w-dummy');
    // Rect-area emitter NEE buffer (binding 14, array<EmitterTri>). When absent,
    // bind an 80-byte zero placeholder — one EmitterTri stride, the minimum
    // binding size for the runtime array. emitterCount==0 ⇒ the shader's NEE
    // loop never reads it. 80 (not 32) bytes because EmitterTri is 80 bytes;
    // a sub-stride placeholder would be rejected like the TLAS 16-vs-32 class.
    const emittersBuf = opts.emittersBuf ?? this._dummyStorageBuffer(device, 'rc-emitters-dummy', 80);
    // A7 (2026-06-10): analytic point/spot lights buffer (binding 15, RCLightBuffer).
    // 16-byte header (count + 3 pad u32) + 16 × 64-byte RCLight entries = 1040 bytes.
    // When absent, bind a 1040-byte zero placeholder (count=0 → loop no-op).
    // 1040 bytes ≥ the RCLightBuffer struct stride so strict backends accept it.
    const lightsBuf = opts.lightsBuf ?? this._dummyStorageBuffer(device, 'rc-lights-dummy', 1040);

    // Env texture + sampler. If the caller supplied both, use them; otherwise
    // create a 1×1 black placeholder.
    const {
      envTextureView,
      envSampler,
      placeholderEnvTexture,
    } = this._resolveEnvBindingRaw(device, opts);

    // ── Cast passes (one per cascade) ──────────────────────────────────────
    const castPasses: CastPassHandles[] = [];
    const castBindGroups: GPUBindGroup[] = [];

    const cascadeDims = this._cascadeDims;
    for (let k = 0; k < cascadeDims.length; k++) {
      const dim = cascadeDims[k]!;
      const totalRays = dim.probes[0] * dim.probes[1] * dim.probes[2] * dim.rays;

      // Create per-pass pipeline.
      const pipeline = device.createComputePipeline({
        label:  `rc-cast-C${k}`,
        layout: castPipelineLayout,
        compute: {
          module:     this._castShaderModule,
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
      const bvhMode = opts.bvhMode === 'tlas' ? 1 : 0;
      const tlasNodeCount = opts.tlasNodeCount ?? 0;
      const emitterCount = opts.emitterCount ?? 0;
      const lightCount = opts.lightCount ?? 0;
      buildCascadeUniformDataInto(
        cascadeParamsRaw, k,
        opts.probeOriginWorld, opts.roomSize,
        opts.sunDirection, opts.sunColor,
        1.0, opts.frameSeed,
        opts.triIntersectEpsilon ?? 1e-5,
        bvhMode,
        tlasNodeCount,
        emitterCount,
        lightCount,
        cascadeDims,
      );
      const cascadeParamsBuf = device.createBuffer({
        label:  `rc-cast-C${k}-uniforms`,
        size:   cascadeParamsRaw.byteLength,
        usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(cascadeParamsBuf.getMappedRange()).set(cascadeParamsRaw);
      cascadeParamsBuf.unmap();

      // Cascade output buffer (caller-provided raw GPUBuffer).
      const cascadeBuf  = opts.cascadeBufs[k]!;

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
          { binding: 9, resource: { buffer: tlasNodesBuf } },
          { binding: 10, resource: { buffer: tlasInstBuf } },
          { binding: 11, resource: { buffer: tlasBlasBuf } },
          { binding: 12, resource: { buffer: tlasW2lBuf } },
          { binding: 13, resource: { buffer: tlasL2wBuf } },
          { binding: 14, resource: { buffer: emittersBuf } },
          { binding: 15, resource: { buffer: lightsBuf } },  // A7: rc_lights
        ],
      });

      castPasses.push({ pipeline, cascadeParamsBuf, cascadeParamsRaw, dispatchX: Math.ceil(totalRays / 64) });
      castBindGroups.push(bindGroup);
    }

    // ── Merge passes (bottom-up: C(N-2) → C0) ────────────────────────────
    const mergePasses: MergePassHandles[] = [];
    const mergeBindGroups: GPUBindGroup[] = [];

    for (let lower = cascadeDims.length - 2; lower >= 0; lower--) {
      const lowerDim = cascadeDims[lower]!;
      const upperDim = cascadeDims[lower + 1]!;
      const totalLower = lowerDim.probes[0] * lowerDim.probes[1] * lowerDim.probes[2] * lowerDim.rays;

      const pipeline = device.createComputePipeline({
        label:  `rc-merge-${lower}→${lower + 1}`,
        layout: mergePipelineLayout,
        compute: {
          module:     this._mergeShaderModule,
          entryPoint: 'cascadeMergeKernel',
        },
      });

      // MergeUniforms buffer (20 floats = 80 bytes).
      const mergeRaw = buildMergeUniformData(lowerDim, upperDim, opts.probeOriginWorld, opts.roomSize);
      const cascadeParamsBuf = device.createBuffer({
        label:  `rc-merge-${lower}-uniforms`,
        size:   mergeRaw.byteLength,
        usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(cascadeParamsBuf.getMappedRange()).set(mergeRaw);
      cascadeParamsBuf.unmap();

      // Caller-provided raw cascade GPUBuffers (lower + upper).
      const lowerBuf = opts.cascadeBufs[lower]!;
      const upperBuf = opts.cascadeBufs[lower + 1]!;

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

    return {
      castPasses,
      mergePasses,
      envTextureView,
      envSampler,
      castBindGroups,
      mergeBindGroups,
      ...(placeholderEnvTexture ? { placeholderEnvTexture } : {}),
    };
  }
}

// W8 follow-up cleanup (2026-05-18) — the `dispatchCascadePasses` /
// `disposeSharedDispatcher` module-level singleton wrappers were removed
// after grep verified zero production consumers (host code instantiates
// `RCDispatcher` directly now via HybridEngineRC.ts). The singletons
// violated CLAUDE.md Design Principle 2 ("the host owns lifecycle") and
// only existed as a backward-compat surface for legacy callers that no
// longer exist in this monorepo.
