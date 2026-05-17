/**
 * OIDNFinalDenoiser — placeholder for the Intel Open Image Denoise
 * "final" pass.
 *
 * Reserved for W11 — see `plan/premium-grade-refactor-20260517.md §W11`.
 * Registered as `disabled: true` so `DenoiserRegistry.lookup('oidn-final')`
 * throws a clear error with workstream context. The `shared-denoisers`
 * package already ships an `oidnBridge.ts` implementation (see the
 * walkaround-hybrid CLAUDE.md "Where things actually stand" note); W11
 * will wire it through this denoiser entry.
 */

import type {
  Denoiser,
  DenoiserDispatchContext,
  DenoiserInitContext,
} from './index.js';

export class OIDNFinalDenoiser implements Denoiser {
  readonly id = 'oidn-final' as const;
  readonly disabled = true;

  async initialize(_ctx: DenoiserInitContext): Promise<void> {
    // No-op — the registry guards against ever reaching this path while
    // `disabled === true`. W11 will replace this stub.
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
