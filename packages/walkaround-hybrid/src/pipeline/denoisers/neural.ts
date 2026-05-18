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

import type {
  Denoiser,
  DenoiserDispatchContext,
  DenoiserInitContext,
} from './index.js';

export class NeuralDenoiser implements Denoiser {
  readonly id = 'neural' as const;
  readonly disabled = true;

  async initialize(_ctx: DenoiserInitContext): Promise<void> {
    // No-op — the registry guards against ever reaching this path while
    // `disabled === true`. W10 will replace this stub.
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
