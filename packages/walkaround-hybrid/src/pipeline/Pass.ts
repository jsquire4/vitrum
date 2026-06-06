/**
 * Pass — a single compute or render stage of the walkaround-hybrid pipeline.
 *
 * The walkaround pipeline is composed of independently-compiled passes
 * scheduled by {@link PassRegistry}. Each pass owns its own bind-group
 * construction and dispatch. The orchestrator iterates a topologically-sorted
 * list of registered passes per frame.
 *
 * Premium-library rationale: before this abstraction, adding a single
 * pass required ~25-30 edits across 6-9 files (complexity sweep
 * 2026-05-17 Theme B). With a Pass, adding one is a single registry
 * entry + (optionally) one shader-string module.
 *
 * Lifecycle:
 *   - Construction is cheap (no GPU work). Concrete passes accept any
 *     pipelines / UBOs / refs they need to share with the rest of the
 *     orchestrator in their constructor so they can stay stateless across
 *     frames.
 *   - {@link Pass.initialize} is awaited once at engine boot per
 *     pass. The walkaround passes share `compilePipelines()` output (so
 *     no per-pass shader recompile here); initialize() can still be used
 *     for pass-local resource allocation if needed.
 *   - {@link Pass.dispatch} runs per frame.
 *   - {@link Pass.dispose} runs once at engine teardown.
 *
 * Compute vs. render passes: dispatch receives a {@link GPUCommandEncoder};
 * the pass internally begins a compute or render pass as needed. The
 * abstraction is encoder-agnostic — the CompositePass begins a render pass
 * and uses the same interface as every other pass.
 */

import type { BGLCache } from './bindGroupLayouts.js';
import type { FrameResources } from './resourceManager.js';
import type { PassLabel } from './timestampQueries.js';
import type { PipelineFrameInputs } from './WalkaroundGPUPipeline.js';

/** Engine-state surface a pass may inspect when deciding whether to run.
 *  Kept structural so this module does not import HybridEngineOptions. */
export interface PassGateOptions {
  /** Active denoiser id (see {@link DenoiserId} in denoisers/index.ts). */
  readonly denoiserMode: string;
  /** Whether PPG (path-guiding) is enabled. */
  readonly ppgEnabled: boolean;
  /** Phase-0 productization — whether the PPG update TRAIN pass dispatches
   *  THIS frame. The orchestrator sets this to
   *  `frameCount % ppgDispatchInterval === 0` so a low quality preset can
   *  amortise the path-guiding training cost across frames. The learned
   *  sTree/dTree GPU buffers PERSIST between train cycles and the gi-ris guided
   *  SAMPLING reads them every frame regardless, so this only gates training
   *  flux accumulation — never guided sampling. Absent ⇒ treated as `true`
   *  (every frame, no behaviour change). Only `PPGUpdatePass`
   *  read it; it has no effect when `ppgEnabled` is false. */
  readonly ppgTrainThisFrame?: boolean;
  /** Phase-0 productization — whether GTAO runs this config. `false` (low
   *  quality preset / `gtaoMode:'off'`) gates off BOTH the GTAO compute pass
   *  and its bilateral upsample. Absent ⇒ treated as `true` (on). */
  readonly gtaoEnabled?: boolean;
  /** Extension bag for forward-compatible flags. */
  readonly [extension: string]: unknown;
}

/** Context provided to {@link Pass.initialize}. */
export interface PassInitContext {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  /** Cache of universal BGLs (frame/scene/ubo/composite/accum/atrous/…). */
  readonly bglCache: BGLCache;
  /** Persistent frame resources — passes may inspect or pre-bind from here. */
  readonly frameResources: FrameResources;
}

/**
 * Per-frame mutable state shared between passes. The orchestrator owns this
 * struct; passes mutate it as they run so downstream passes see the latest
 * texture handles and ping-pong indices.
 *
 * Why this exists: many passes are chained — the denoiser produces a texture
 * the indirect-combine pass reads, which produces a texture the temporal
 * accumulator reads, and so on. Threading those handoffs through pass
 * arguments would re-create the position-encoded coupling we are deleting.
 * The shared state bag keeps the dataflow data-driven without sacrificing
 * the topological declared ordering.
 */
export interface PassFrameState {
  /** Direct-channel denoised radiance texture (set by the denoiser dispatch
   *  the orchestrator runs between `gtao-upsample` and
   *  `indirect-temporal-accum`; defaults to `common.hdrColorTexture` when
   *  the active denoiser is `NoneDenoiser` and returns null). */
  denoisedDirect: GPUTexture;
  /** Texture written by the indirect temporal accumulator and read by the
   *  indirect atrous chain. */
  indirectAccumOut: GPUTexture;
  /** Final denoised indirect texture after the 4-iter atrous-indirect chain. */
  denoisedIndirect: GPUTexture;
  /** Output of indirect-combine; input to temporalAccum. */
  combinedDenoised: GPUTexture;
  /** Current-frame radiance texture written by temporalAccum and read by resolve. */
  writeAccum: GPUTexture;
  /** Previous-frame radiance texture (the inactive ping-pong slot at the
   *  start of the frame); resolve reads this as `prevRadianceView`. */
  readAccum: GPUTexture;
  /** EMA blend weight passed to the temporal accumulator pass. 1.0 on the
   *  first frame (history discarded); `_temporalAccumAlpha` thereafter. */
  alpha: number;
  /** Whether the camera moved this frame above the reset threshold;
   *  consumed by the denoiser dispatch + temporal accumulator. */
  isMoving: boolean;
}

/** Context provided to {@link Pass.dispatch}. */
export interface PassDispatchContext {
  readonly device: GPUDevice;
  readonly encoder: GPUCommandEncoder;
  readonly width: number;
  readonly height: number;
  /** Monotonic accumulator frame index (resets on camera motion). */
  readonly frameIndex: number;
  /** Strictly-monotonic frame counter (does NOT reset). */
  readonly frameCount: number;
  /** Cache of universal BGLs (frame/scene/ubo/composite/accum/atrous/…). */
  readonly bglCache: BGLCache;
  /** Persistent frame resources. */
  readonly resources: FrameResources;
  /** Per-frame inputs (camera matrices, lighting params, swap chain view, …). */
  readonly inputs: PipelineFrameInputs;
  /** Pre-built frame/scene/ubo bind groups — shared by RIS/temporal/spatial/shade. */
  readonly frameBindGroup: GPUBindGroup;
  readonly sceneBindGroup: GPUBindGroup;
  readonly uboBindGroup: GPUBindGroup;
  /** Pre-built DDGI hybrid-layers bind group (slot 3) — used by gi-ris + shade. */
  readonly hybridLayersBindGroup: GPUBindGroup;
  /** Pre-built light-tree bind group (slot 3) — RIS-only DI light selection. */
  readonly lightTreeBindGroup: GPUBindGroup;
  /** Workgroup counts — 8×8 (wgX/wgY), 16×16 (wgX16/wgY16),
   *  half-res 8×8 (halfWgX/halfWgY). */
  readonly wgX: number;
  readonly wgY: number;
  readonly wgX16: number;
  readonly wgY16: number;
  readonly halfWgX: number;
  readonly halfWgY: number;
  /** GTAO AO-compute downscale factor: 2 (`gtaoMode:'on'`, half-res) or 4
   *  (`gtaoMode:'quarter'`, quarter-res). The GTAOPass dispatches at
   *  W/ds × H/ds and packs this into the GTAO UBO so both gtao + gtaoUpsample
   *  shaders map between the AO grid and full-res coords. */
  readonly gtaoDownscale: number;
  /** Pre-resolved gNormalDepth view used as edge-stop input by GTAO + atrous chains. */
  readonly gNormalDepthView: GPUTextureView;
  /** Build a `GPUComputePassDescriptor` with optional timestampWrites. */
  readonly computeDesc: (label: PassLabel) => GPUComputePassDescriptor;
  /** Build the optional render-pass `timestampWrites` struct, or undefined
   *  when timestamp queries aren't active. Used by CompositePass. */
  readonly renderTimestampWrites: (label: PassLabel) => GPURenderPassTimestampWrites | undefined;
  /** Shared mutable frame state used to thread textures between chained
   *  passes (denoiser→indirect-combine→temporalAccum→resolve→composite). */
  readonly frameState: PassFrameState;
}

/**
 * Shared compute-dispatch body for the five passes that reuse the
 * orchestrator-built frame/scene/ubo bind groups (and, optionally, the
 * hybrid-layers group at slot 3): RIS, temporal, spatial, gi-ris, shade.
 *
 * Before this helper each of those passes inlined the identical sequence
 * `beginComputePass(computeDesc(label)) → setPipeline → setBindGroup 0/1/2
 * [+ 3] → dispatchWorkgroups → end`. The only per-pass variation is:
 *   - the timestamp label,
 *   - whether slot 3 (hybrid-layers) is bound, and
 *   - whether the dispatch is full-res (`wgX/wgY`) or half-res
 *     (`halfWgX/halfWgY`).
 *
 * Behavior is byte-identical to the prior hand-written bodies.
 */
export function dispatchSharedBindGroupPass(
  ctx: PassDispatchContext,
  pipeline: GPUComputePipeline,
  opts: {
    /** Timestamp label + compute-pass descriptor key. */
    readonly label: PassLabel;
    /** Bind the DDGI hybrid-layers group at slot 3 (gi-ris + shade). */
    readonly useHybridLayers?: boolean;
    /** Dispatch at half resolution (`halfWgX/halfWgY`) instead of full
     *  (`wgX/wgY`). Used by the Sprint-16 gi-ris pass. */
    readonly halfRes?: boolean;
    /** Additional bind groups bound at explicit slots AFTER the shared
     *  0/1/2 (+ optional 3) groups — e.g. the NRC `@group(4)` group for the
     *  gi-ris compile-time NRC variant. Bound in array order; each entry's
     *  `slot` must match the compiled pipeline layout. Empty/absent ⇒ the
     *  byte-identical pre-extraGroups dispatch. */
    readonly extraGroups?: ReadonlyArray<{ readonly slot: number; readonly group: GPUBindGroup }>;
  },
): void {
  const {
    encoder, computeDesc,
    frameBindGroup, sceneBindGroup, uboBindGroup, hybridLayersBindGroup,
    wgX, wgY, halfWgX, halfWgY,
  } = ctx;
  const pass = encoder.beginComputePass(computeDesc(opts.label));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, frameBindGroup);
  pass.setBindGroup(1, sceneBindGroup);
  pass.setBindGroup(2, uboBindGroup);
  if (opts.useHybridLayers) pass.setBindGroup(3, hybridLayersBindGroup);
  if (opts.extraGroups) {
    for (const { slot, group } of opts.extraGroups) pass.setBindGroup(slot, group);
  }
  const dx = opts.halfRes ? halfWgX : wgX;
  const dy = opts.halfRes ? halfWgY : wgY;
  pass.dispatchWorkgroups(dx, dy, 1);
  pass.end();
}

/**
 * Base class for the five compute passes that share the orchestrator-built
 * frame/scene/ubo bind groups. Concrete subclasses declare only their
 * identity (`id`/`dependencies`/`passLabels`) and the dispatch knobs
 * (`useHybridLayers`/`halfRes`); the dispatch body is provided here.
 *
 * The default `dispatch` runs one {@link dispatchSharedBindGroupPass} per
 * entry in `passLabels`, which covers both single-dispatch passes (one
 * label) and the spatial-reuse pass (two ping-pong labels sharing one
 * pipeline + bind group). `gates()` returns true (these passes always run);
 * `initialize`/`dispose` are no-ops since they reuse the shared compile
 * output and own no pass-private GPU resources.
 */
export abstract class SharedBindGroupPass implements Pass {
  abstract readonly id: string;
  abstract readonly dependencies: readonly string[];
  abstract readonly passLabels: readonly PassLabel[];

  /** Bind the hybrid-layers group at slot 3 (gi-ris + shade override). */
  protected readonly useHybridLayers: boolean = false;
  /** Dispatch at half resolution (gi-ris override). */
  protected readonly halfRes: boolean = false;

  protected readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    for (const label of this.passLabels) {
      dispatchSharedBindGroupPass(ctx, this._pipeline, {
        label,
        useHybridLayers: this.useHybridLayers,
        halfRes: this.halfRes,
      });
    }
  }

  dispose(): void {}
}

export interface Pass {
  /** Stable identifier; must be unique within a {@link PassRegistry}. */
  readonly id: string;

  /** Pass IDs that must run before this pass each frame. The registry
   *  topologically sorts by these. Unknown IDs raise at registration. */
  readonly dependencies: readonly string[];

  /** Timestamp-query labels emitted by this pass. For single-dispatch
   *  passes this is `[this.id]`. Multi-dispatch passes (SpatialReservoirPass,
   *  AtrousIndirectPass) enumerate every label they emit so
   *  `buildPassLayout` can size the GPU querySet correctly.
   *
   *  The relative order MUST match the dispatch order inside
   *  {@link Pass.dispatch}, since `tsWrites` resolves slot indices via
   *  these labels. */
  readonly passLabels: readonly PassLabel[];

  /** Decide whether this pass runs for the given options. Skipped passes
   *  still hold their compiled pipeline; gating is a per-frame decision. */
  gates(opts: PassGateOptions): boolean;

  /** One-time async initialization. Allocate any pass-local resources;
   *  compile any pass-private pipelines (most walkaround passes use the
   *  shared compile output and have a no-op initialize). */
  initialize(ctx: PassInitContext): Promise<void>;

  /** Per-frame dispatch. The pass builds its bind group(s) and submits
   *  its compute or render work via the supplied encoder. */
  dispatch(ctx: PassDispatchContext): void;

  /** Release all GPU resources owned by this pass. */
  dispose(): void;
}
