/**
 * NoneDenoiser — pass-through, no-op denoiser entry.
 *
 * Returns `null` from {@link dispatch}; the orchestrator interprets that
 * as "no denoising work happened; sample the raw HDR target". Registered
 * for symmetry with the `'none'` value in the public `EngineOptions`
 * discriminated union (see `@vitrum/core`); the host can pick it to
 * disable temporal+spatial filtering for debugging or for very-clean
 * scenes where path tracing already converges fast.
 *
 * No GPU resources are allocated; lifecycle hooks are no-ops.
 */

import {
  DENOISER_PASS_LABELS,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
} from './index.js';

export class NoneDenoiser implements Denoiser {
  readonly id = 'none' as const;
  readonly passLabels = DENOISER_PASS_LABELS['none'];

  async initialize(_ctx: DenoiserInitContext): Promise<void> {
    // No resources to allocate; the pipeline samples the raw HDR target.
  }

  dispatch(_ctx: DenoiserDispatchContext): GPUTexture | null {
    return null;
  }

  resize(_w: number, _h: number): void {
    // No resources to resize.
  }

  dispose(): void {
    // No resources to release.
  }
}
