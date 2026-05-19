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

import { CASCADE_DIMS, CASCADE_COUNT } from './cascadePyramid.js';
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

  /** BVH GPU buffers (5 separate SSBOs; layout matches `bvhCompute.SceneBVH`). */
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
  /** Debug-fill mode bypasses GPU compute and writes test colours to CPU
   *  Float32Arrays. Only useful for the legacy `RCDispatchOpts` path which
   *  also owns CPU `Float32Array` mirrors of the cascades. The raw path
   *  has no CPU mirror, so when `debugFill: true` here the dispatcher
   *  simply returns without dispatching. */
  debugFill?:         boolean;
}

// ─── Uniform data builders ────────────────────────────────────────────────────
// These are the exact same packing functions as the original cascadeDispatch.ts.

/** Write CascadeUniforms into an existing Float32Array (avoids realloc per frame).
 *
 * W8 Phase 1B (2026-05-18) — sunDir / sunColor / cascade geometry all take
 * plain `readonly [number, number, number]` tuples; no `THREE.Vector3` /
 * `THREE.Color` dependency. The legacy `RCDispatchOpts` (THREE-tied) path
 * converts `THREE.Vector3 → [x,y,z]` and `THREE.Color → [r,g,b]` at the
 * call site before invoking this helper.
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
): void {
  const dim = CASCADE_DIMS[k]!;
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
  // frameSeed(u), lastCascade(u), triIntersectEpsilon(f), _pad4a(u)
  // Total: 40 float/uint values = 160 bytes
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
  ui[25] = CASCADE_COUNT - 1;
  d[26] = triIntersectEpsilon;  // E2: was _pad4[0]
  ui[27] = 0;                   // _pad4a
}

function buildMergeUniformData(
  lowerDim: (typeof CASCADE_DIMS)[number],
  upperDim: (typeof CASCADE_DIMS)[number],
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

  /**
   * Dispatch the cascade compute pipeline for one frame.
   * Pipelines and bind groups are compiled/created lazily on the first call
   * (or after `dispose()`).
   */
  async dispatchFrameRaw(opts: RCDispatchOptsRaw): Promise<void> {
    if (opts.debugFill) {
      // Raw path has no CPU-side mirror to fill; just no-op for the
      // smoke-test case. (Hosts wanting CPU debug fill should use the
      // legacy `dispatchFrame` path which owns CPU Float32Arrays.)
      return;
    }

    const device = opts.device;
    if (!this._handles) {
      try {
        this._handles = this._buildHandlesRaw(device, opts);
      } catch (err: unknown) {
        console.error('[RCDispatcher] buildHandlesRaw failed:', err);
        return;
      }
    }
    const handles = this._handles;

    // Update per-frame uniforms for each cast pass.
    for (let k = 0; k < CASCADE_COUNT; k++) {
      const pass = handles.castPasses[k]!;
      buildCascadeUniformDataInto(
        pass.cascadeParamsRaw, k,
        opts.probeOriginWorld, opts.roomSize,
        opts.sunDirection, opts.sunColor,
        1.0, opts.frameSeed,
        opts.triIntersectEpsilon ?? 1e-5,
      );
      device.queue.writeBuffer(pass.cascadeParamsBuf, 0, pass.cascadeParamsRaw.buffer);
    }

    // Encode compute commands.
    const commandEncoder = device.createCommandEncoder({ label: 'RCDispatcher' });
    const passEncoder = commandEncoder.beginComputePass({ label: 'rc-cascade' });

    // Cast passes C0 → C4.
    for (let k = 0; k < CASCADE_COUNT; k++) {
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
   * Resolve the env binding for the raw (THREE-free) path. When the caller
   * supplies both `envTextureView` and `envSampler`, use them. Otherwise
   * create a 1×1 black placeholder.
   */
  private _resolveEnvBindingRaw(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
  ): { envTextureView: GPUTextureView; envSampler: GPUSampler } {
    if (opts.envTextureView && opts.envSampler) {
      return { envTextureView: opts.envTextureView, envSampler: opts.envSampler };
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
    };
  }

  /**
   * Build all pipelines and bind groups (one-time setup).
   * Called lazily on first `dispatchFrameRaw()`.
   *
   * W8 Phase 1B (2026-05-18) — refactored to take {@link RCDispatchOptsRaw}
   * (raw GPU types). The legacy THREE-tied {@link dispatchFrame} entry adapts
   * its inputs and calls {@link dispatchFrameRaw}, which calls this method.
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

    // Env texture + sampler. If the caller supplied both, use them; otherwise
    // create a 1×1 black placeholder.
    const { envTextureView, envSampler } = this._resolveEnvBindingRaw(device, opts);

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
      buildCascadeUniformDataInto(
        cascadeParamsRaw, k,
        opts.probeOriginWorld, opts.roomSize,
        opts.sunDirection, opts.sunColor,
        1.0, opts.frameSeed,
        opts.triIntersectEpsilon ?? 1e-5,
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

    return { castPasses, mergePasses, envTextureView, envSampler, castBindGroups, mergeBindGroups };
  }
}

// W8 follow-up cleanup (2026-05-18) — the `dispatchCascadePasses` /
// `disposeSharedDispatcher` module-level singleton wrappers were removed
// after grep verified zero production consumers (host code instantiates
// `RCDispatcher` directly now via HybridEngineRC.ts). The singletons
// violated CLAUDE.md Design Principle 2 ("the host owns lifecycle") and
// only existed as a backward-compat surface for legacy callers that no
// longer exist in this monorepo.
