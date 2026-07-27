/**
 * optimizer.ts — pt-webgpu adapter over the shared inverse scaffolding (WS5).
 *
 * The backend-agnostic optimization scaffolding (image-space L2/L1 loss, the
 * small-vector Adam optimizer, and the parameter-path parser) now lives in
 * `@vitrum/core` (`inverse-scaffolding.ts`) as the single source of truth shared
 * with pt-webgl2. This module re-exports those symbols and adapts the two whose
 * signatures carry a per-backend detail (`paramLength`'s backend attribution),
 * so the rest of pt-webgpu's inverse code imports from here
 * unchanged. pt-webgpu supplies ONLY its gradient source (the path-replay
 * analytic adjoint); that FD-vs-adjoint split is documented and stays
 * per-backend.
 *
 * Ref: Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
 */

import type { InverseParam } from '@vitrum/core';
import { paramLength as sharedParamLength } from '@vitrum/core/inverse-scaffolding';

export {
  Adam,
  DEFAULT_ADAM,
  l2Loss,
  l1Loss,
  lossValue,
  parseParamPath,
  clampParams,
  assertFiniteArray,
  assertFiniteNumber,
  invokeInverseHook,
  normalizeInverseError,
  validateInverseReadback,
  validateInverseSessionOptions,
} from '@vitrum/core/inverse-scaffolding';
export type {
  AdamSnapshot,
  AdamConfig,
  ResolvedParamTarget,
  ParamLayoutEntry,
} from '@vitrum/core/inverse-scaffolding';

/** Components for an {@link InverseParam} kind. scalar → 1, vec2 → 2, rgb → 3. */
export function paramLength(p: InverseParam): number {
  return sharedParamLength(p, 'pt-webgpu');
}
