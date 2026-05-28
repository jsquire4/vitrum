/**
 * WelfordVariance struct + helpers — canonical single source of truth from
 * `@vitrum/shared-denoisers` (welfordVariance.wgsl.ts), used by every
 * variance-aware pass.
 *
 * Split out of common.wgsl.ts (T9-stepA) as the trailing fragment of the
 * shared header. Kept as the last entry in `common`'s aggregate so the
 * composed output is byte-identical to the pre-split `COMMON_WGSL`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { WELFORD_VARIANCE_WGSL } from '@vitrum/shared-denoisers';

export const WELFORD_TAIL_WGSL = /* wgsl */ `// ============================================================
// WelfordVariance — canonical struct + helpers imported from
// @vitrum/shared-denoisers (see welfordVariance.wgsl.ts).
// Single source of truth across all variance-aware passes.
// ============================================================
${WELFORD_VARIANCE_WGSL}

// Surface-texture pattern functions live in surfaceTextures.wgsl.ts (shade-only).

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const WELFORD_TAIL_MODULE: WgslModule = {
  name: "welfordTail",
  source: WELFORD_TAIL_WGSL,
  requires: [],
};
