/**
 * NeuralDenoiser — diagnostic stub for the registry-side `'neural'` entry.
 *
 * The actual U-Net neural denoiser (T2.H2 / Chaitanya et al. 2017 /
 * Ronneberger 2015) IS wired in `HybridEngineLifecycle.ts:322` against
 * `InferenceGraph` from `../../neural/InferenceGraph.ts` — that path is
 * the W10 finish. It bypasses the denoiser-registry dispatch because the
 * neural pipeline owns its own texture→buffer→inference→texture bridging
 * separately from the registry's per-pass slot allocation.
 *
 * This stub remains so `DenoiserRegistry.ids()` enumerates `'neural'`
 * (preserving stable diagnostic ordering across denoiser sets) and so
 * `lookup('neural')` throws the canonical "registered but disabled"
 * error if a future caller mistakenly routes through the registry
 * instead of the dedicated HybridEngineLifecycle path.
 *
 * If neural is ever migrated to the registry surface (so the per-pass
 * allocator can manage its buffers uniformly with the other denoisers),
 * delete `disabled = true` and implement `initialize/dispatch` here.
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
  /** Mirrors the atrous-variance layout — the slot allocator inspects this
   *  even for `disabled` entries when buildPassLayout enumerates the union
   *  of all registered denoisers. */
  readonly passLabels = DENOISER_PASS_LABELS['neural'];

  async initialize(_ctx: DenoiserInitContext): Promise<void> {
    // No-op — the registry guards against ever reaching this path while
    // `disabled === true`. The W10 neural pipeline runs out-of-band
    // through HybridEngineLifecycle; see file-level JSDoc.
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
