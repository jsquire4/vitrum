/**
 * Shared low-level WGSL primitives injected from `@vitrum/shared-samplers`:
 * the PCG random number generator (`rand_f32` etc.), stateless PCG hash helpers
 * for deterministic jitter/noise, the BSDF primitives (`fresnelSchlick`,
 * cosine-hemisphere sampling, …), and the canonical Rec.709 `luminance` helper
 * — plus the local `safe_normalize` guard.
 *
 * Split out of common.wgsl.ts (T9-stepA).
 *
 * Note: `common` itself pulls `luminance` via this module, so a module that
 * already requires `common` must NOT also require the standalone
 * `luminance` module — that would emit two `fn luminance` definitions.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  BSDF_PRIMITIVES_WGSL,
  LUMINANCE_WGSL,
  PCG_HASH_TO_F32_WGSL,
  PCG_WGSL,
  REPRESENTED_WRS_WGSL,
} from '@vitrum/shared-samplers';

export const SHARED_PRIMITIVES_WGSL = /* wgsl */ `// ============================================================
// Shared WGSL primitives
// ============================================================
${PCG_WGSL}
${REPRESENTED_WRS_WGSL}
${PCG_HASH_TO_F32_WGSL}
${BSDF_PRIMITIVES_WGSL}

// ============================================================
// Utility
// ============================================================
${LUMINANCE_WGSL}

fn safe_normalize(v: vec3f) -> vec3f {
  let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (!(maxComponent > 0.0) || maxComponent > 3.402823e38) {
    return vec3f(0.0, 1.0, 0.0);
  }
  let scaled = v / maxComponent;
  let scaledLength = length(scaled);
  if (!(scaledLength > 0.0) || scaledLength > 3.402823e38) {
    return vec3f(0.0, 1.0, 0.0);
  }
  return scaled / scaledLength;
}

fn safe_length(v: vec3f) -> f32 {
  let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (!(maxComponent > 0.0) || maxComponent > INFINITY) {
    return 0.0;
  }
  let scaledLength = length(v / maxComponent);
  let result = maxComponent * scaledLength;
  return select(0.0, result, result > 0.0 && result <= INFINITY);
}

fn vitrumPcgSeed2(a: u32, b: u32, salt: u32) -> u32 {
  return (a * 1664525u) ^ (b * 1013904223u) ^ (salt * 22695477u);
}

fn vitrumPcgSeed3(a: u32, b: u32, c: u32, salt: u32) -> u32 {
  return vitrumPcgSeed2(a ^ (c * 747796405u), b ^ (c * 277803737u), salt);
}

fn pcgHash2FromSeed(seed: u32) -> vec2f {
  return vec2f(pcgHashToF32(seed), pcgHashToF32(seed ^ 0x9E3779B9u));
}

fn pixelHash2(px: vec2u, salt: u32) -> vec2f {
  return pcgHash2FromSeed(vitrumPcgSeed2(px.x, px.y, salt));
}

fn floatCellHash(p: vec2f, salt: u32) -> f32 {
  return pcgHashToF32(vitrumPcgSeed2(bitcast<u32>(p.x), bitcast<u32>(p.y), salt));
}

fn worldHash2(p: vec3f, salt: u32) -> vec2f {
  return pcgHash2FromSeed(vitrumPcgSeed3(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z), salt));
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const SHARED_PRIMITIVES_MODULE: WgslModule = {
  name: "sharedPrimitives",
  source: SHARED_PRIMITIVES_WGSL,
  requires: [],
};
