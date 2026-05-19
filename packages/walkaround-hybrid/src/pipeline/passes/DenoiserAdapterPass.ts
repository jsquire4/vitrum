/**
 * DenoiserAdapterPass — virtual {@link Pass} wrapper around the active
 * {@link Denoiser}, so denoiser dispatch participates in the same
 * dependency-sorted pass loop as every other stage.
 *
 * Pre-W1-R5b the orchestrator split its per-frame pass loop into TWO halves
 * (up to and including `gtao-upsample`, then `indirect-temporal-accum` …
 * `composite`) bracketing a manual `_activeDenoiser.dispatch()` call. The
 * split index was a separate field on the pipeline. By promoting the
 * denoiser dispatch to a virtual Pass with the right dependency declaration,
 * the orchestrator collapses back to a single uniform for-loop and
 * `frameState.denoisedDirect` becomes a regular ledger entry instead of a
 * special-cased post-dispatch write.
 *
 * Layering note (verbatim from the prior comment at the manual-dispatch
 * site): denoising IS a separate concept from pass scheduling — denoisers
 * return a texture handle (whereas Passes mutate the encoder + frame
 * ledger) and the `Denoiser` lifecycle has the additional
 * `cleanupAfterSubmit` hook. The adapter pass preserves that distinction
 * by delegating to the active Denoiser; it does NOT subsume the Denoiser
 * abstraction.
 *
 * Dependencies:
 *   - `gtao-upsample` — was the last pass run before the denoiser dispatch
 *     in the pre-refactor split.
 *
 * Labels:
 *   - The active Denoiser's `passLabels` are forwarded so the GPU
 *     timestamp-query layout still differs by denoiser mode, just via the
 *     natural Pass mechanism. Pass-through `NoneDenoiser` returns `[]`.
 *
 * Gating:
 *   - The pass dispatches whenever the active denoiser id is not `'none'`.
 *     This mirrors the legacy behaviour where `NoneDenoiser.dispatch`
 *     returned null and the orchestrator fell back to
 *     `frameState.denoisedDirect = common.hdrColorTexture` (which is also
 *     the default value seeded in the frame ledger).
 */

import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import type { Denoiser } from '../denoisers/index.js';

export class DenoiserAdapterPass implements Pass {
  readonly id = 'denoiser-adapter' as const;
  readonly dependencies: readonly string[] = ['gtao-upsample'];

  /** The active Denoiser instance (set once at engine boot; survives until
   *  pipeline `dispose()`). Stored as a getter callback so this class
   *  remains agnostic to where the active denoiser is kept. */
  private readonly _activeDenoiser: () => Denoiser;
  /** Compiled à-trous pipeline shared between the legacy AtrousDenoiser
   *  and the always-on indirect-channel atrous chain. Threaded through
   *  here so the adapter can construct the {@link DenoiserDispatchContext}
   *  without the pipeline keeping a back-reference. */
  private readonly _sharedAtrousPipeline: () => GPUComputePipeline;

  constructor(
    activeDenoiser: () => Denoiser,
    sharedAtrousPipeline: () => GPUComputePipeline,
  ) {
    this._activeDenoiser = activeDenoiser;
    this._sharedAtrousPipeline = sharedAtrousPipeline;
  }

  /** Forward the active denoiser's labels so `buildPassLayout`'s slot
   *  table matches the active dispatch order. Pass-through denoisers
   *  return `[]`. */
  get passLabels(): readonly PassLabel[] {
    return this._activeDenoiser().passLabels;
  }

  gates(): boolean {
    return this._activeDenoiser().id !== 'none';
  }

  async initialize(_ctx: PassInitContext): Promise<void> {
    // The active denoiser's own `initialize` is awaited from the
    // pipeline's `initialize` — this adapter does not duplicate that.
    // A dedicated initialize on the adapter is reserved for future
    // adapter-owned resources (none today).
  }

  dispatch(ctx: PassDispatchContext): void {
    const denoiser = this._activeDenoiser();
    const result = denoiser.dispatch({
      device: ctx.device,
      encoder: ctx.encoder,
      width: ctx.width,
      height: ctx.height,
      frameIndex: ctx.frameIndex,
      resources: ctx.resources,
      sharedAtrousPipeline: this._sharedAtrousPipeline(),
      bglCache: ctx.bglCache,
      gNormalDepthView: ctx.gNormalDepthView,
      atrousDirectSigmas: ctx.inputs.atrousDirectSigmas,
      readAccum: ctx.frameState.readAccum,
      isMoving: ctx.frameState.isMoving,
      wgX16: ctx.wgX16,
      wgY16: ctx.wgY16,
      computeDesc: ctx.computeDesc,
    });
    // Legacy contract: NoneDenoiser returns null → fall back to the raw
    // HDR target. The frame ledger is already seeded with that handle by
    // the orchestrator, so this branch only fires when a real denoiser
    // produced a resolved-radiance texture.
    if (result !== null) {
      ctx.frameState.denoisedDirect = result;
    }
  }

  dispose(): void {
    // The active Denoiser owns its GPU resources and is disposed by the
    // pipeline (which created it). The adapter holds no resources of its
    // own.
  }
}
