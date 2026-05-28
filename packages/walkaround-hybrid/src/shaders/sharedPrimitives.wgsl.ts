/**
 * Shared low-level WGSL primitives injected from `@vitrum/shared-samplers`:
 * the PCG random number generator (`rand_f32` etc.), the BSDF primitives
 * (`fresnelSchlick`, cosine-hemisphere sampling, …), and the canonical
 * Rec.709 `luminance` helper — plus the local `safe_normalize` guard.
 *
 * Split out of common.wgsl.ts (T9-stepA).
 *
 * Note: `common` itself pulls `luminance` via this module, so a module that
 * already requires `common` must NOT also require the standalone
 * `luminance` module — that would emit two `fn luminance` definitions.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { BSDF_PRIMITIVES_WGSL, LUMINANCE_WGSL, PCG_WGSL } from '@vitrum/shared-samplers';

export const SHARED_PRIMITIVES_WGSL = /* wgsl */ `// ============================================================
// Shared WGSL primitives
// ============================================================
${PCG_WGSL}
${BSDF_PRIMITIVES_WGSL}

// ============================================================
// Utility
// ============================================================
${LUMINANCE_WGSL}

fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  if (len < 1e-8) { return vec3f(0.0, 1.0, 0.0); }
  return v / len;
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const SHARED_PRIMITIVES_MODULE: WgslModule = {
  name: "sharedPrimitives",
  source: SHARED_PRIMITIVES_WGSL,
  requires: [],
};
