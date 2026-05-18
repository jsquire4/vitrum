/**
 * NeuralDenoiser — placeholder for the U-Net neural denoiser (T2.H2 /
 * Chaitanya et al. 2017 / Ronneberger 2015).
 *
 * Reserved for W10 — see `plan/premium-grade-refactor-20260517.md §W10`.
 * Registered as `disabled: true` so `DenoiserRegistry.lookup('neural')`
 * throws a clear error with workstream context, instead of silently
 * falling back to atrous-variance (the legacy behaviour, which the W1-R3
 * refactor removed because no consumer currently selects `'neural'` — see
 * complexity-sweep dead-code analysis).
 *
 * The walkaround-hybrid InferenceGraph scaffold already exists (see
 * `src/neural/InferenceGraph.ts` + `src/neural/unetArchitecture.ts`);
 * W10 will wire it through this denoiser entry once the
 * texture→buffer→inference→texture bridging in HybridEngine is finished.
 */

import {
  DENOISER_PASS_LABELS,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
} from './index.js';

export class NeuralDenoiser implements Denoiser {
  readonly id = 'neural' as const;
  readonly disabled = true;
  /** W10 placeholder: mirrors the atrous-variance layout so a host that
   *  switches to 'neural' before W10 lands does not trip the
   *  buildPassLayout slot-count invariant. */
  readonly passLabels = DENOISER_PASS_LABELS['neural'];

  /** W10 will read `ctx.config.weights` (when `ctx.config?.kind === 'neural'`)
   *  here once the real path lands. Stashing the field doc on the placeholder
   *  so the canonical source of weights is unambiguously the DU, not the
   *  legacy `HybridEngineOptions.neuralWeights` side-channel. */
  async initialize(_ctx: DenoiserInitContext): Promise<void> {
    // No-op — the registry guards against ever reaching this path while
    // `disabled === true`. W10 will replace this stub and pull
    // `_ctx.config?.kind === 'neural' && _ctx.config.weights` from the
    // W3-D4 DU.
  }

  dispatch(_ctx: DenoiserDispatchContext): GPUTexture | null {
    return null;
  }

  resize(_w: number, _h: number): void {
    // No-op.
  }

  dispose(): void {
    // No-op.
  }
}
