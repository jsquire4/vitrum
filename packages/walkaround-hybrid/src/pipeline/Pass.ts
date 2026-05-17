/**
 * Pass — a single compute or render stage of the walkaround-hybrid pipeline.
 *
 * The walkaround pipeline is composed of independently-compiled passes
 * scheduled by {@link PassRegistry}. Each pass owns its own bind-group
 * layout, pipeline compilation, per-frame bind-group construction, and
 * dispatch. The orchestrator iterates a topologically-sorted list of
 * registered passes per frame.
 *
 * Premium-library rationale: before this abstraction, adding a single
 * pass required ~25-30 edits across 6-9 files (complexity sweep
 * 2026-05-17 Theme B). With a Pass, adding one is a single registry
 * entry + one shader-string module.
 *
 * Lifecycle:
 *   - Construction is cheap (no GPU work).
 *   - {@link Pass.initialize} is awaited once at engine boot per
 *     pass. Pipelines, BGLs, and pass-local resources are created
 *     here.
 *   - {@link Pass.dispatch} runs per frame.
 *   - {@link Pass.dispose} runs once at engine teardown.
 *
 * Compute vs. render passes: dispatch receives a {@link GPUCommandEncoder};
 * the pass internally begins a compute or render pass as needed. The
 * abstraction is encoder-agnostic.
 */

/** Engine-state surface a pass may inspect when deciding whether to run.
 *  Kept structural so this module does not import HybridEngineOptions. */
export interface PassGateOptions {
  /** Active denoiser id (see {@link DenoiserId} in denoisers/index.ts). */
  readonly denoiserMode: string;
  /** Whether PPG (path-guiding) is enabled. */
  readonly ppgEnabled: boolean;
  /** Extension bag for forward-compatible flags. */
  readonly [extension: string]: unknown;
}

/** Context provided to {@link Pass.initialize}. */
export interface PassInitContext {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
}

/** Context provided to {@link Pass.dispatch}. Resources are typed loosely
 *  to keep this module dependency-free; concrete passes narrow to the
 *  {@link FrameResources} sub-struct they need. */
export interface PassDispatchContext {
  readonly device: GPUDevice;
  readonly encoder: GPUCommandEncoder;
  readonly width: number;
  readonly height: number;
  readonly frameIndex: number;
  readonly resources: unknown;
  /** Optional timestamp-query slot for this pass; null when disabled. */
  readonly timestampWrites: GPUComputePassTimestampWrites | null;
}

export interface Pass {
  /** Stable identifier; must be unique within a {@link PassRegistry}. */
  readonly id: string;

  /** Pass IDs that must run before this pass each frame. The registry
   *  topologically sorts by these. Unknown IDs raise at registration. */
  readonly dependencies: readonly string[];

  /** Decide whether this pass runs for the given options. Skipped passes
   *  still hold their compiled pipeline; gating is a per-frame decision. */
  gates(opts: PassGateOptions): boolean;

  /** One-time async initialization. Compile pipeline, create BGL,
   *  allocate any pass-local resources. */
  initialize(ctx: PassInitContext): Promise<void>;

  /** Per-frame dispatch. The pass builds its bind group(s) and submits
   *  its compute or render work via the supplied encoder. */
  dispatch(ctx: PassDispatchContext): void;

  /** Release all GPU resources owned by this pass. */
  dispose(): void;
}
