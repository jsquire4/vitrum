/**
 * Emitter sampling helpers.
 *
 * Split out of common.wgsl.ts (T9-stepA): the `EmitterSample` struct,
 * `sampleEmitterPoint` (uniform-area triangle sampling), and
 * `sampleEmitterIdx` (binary search over the emitter CDF for importance
 * sampling). Consumes the `EmitterTri` struct from the reservoirDi module.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const EMITTER_SAMPLING_WGSL = /* wgsl */ `// ============================================================
// Emitter sampling helpers
// ============================================================

// Sample a point on an emitter triangle; returns {pos, normal, area, Le, pdfArea}.
struct EmitterSample {
  pos:     vec3f,
  normal:  vec3f,
  Le:      vec3f,
  area:    f32,
  pdfArea: f32,   // uniform-area pdf = 1/area
};

fn sampleEmitterPoint(e: EmitterTri, xi: vec2f) -> EmitterSample {
  // Uniform sampling of a triangle: (1-sqrt(xi.x))*vA + sqrt(xi.x)*(1-xi.y)*vB + sqrt(xi.x)*xi.y*vC
  let s = sqrt(xi.x);
  let u = 1.0 - s;
  let v = s * xi.y;
  let w = s * (1.0 - xi.y);
  let pos = u * e.vA + v * e.vB + w * e.vC;
  var result: EmitterSample;
  result.pos     = pos;
  result.normal  = e.normal;
  result.Le      = e.Le;
  result.area    = 0.0;
  result.pdfArea = 0.0;
  if (e.area > 0.0 && e.area <= 3.402823e38) {
    let inverseArea = 1.0 / e.area;
    if (inverseArea > 0.0 && inverseArea <= 3.402823e38) {
      result.area = e.area;
      result.pdfArea = inverseArea;
    }
  }
  return result;
}

// Binary search over emitter CDF for importance sampling.
fn sampleEmitterIdx(
  emitterCount: u32,
  xi: f32,
) -> u32 {
  var lo = 0u;
  var hi = emitterCount;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    // xi is an exact 24-bit bucket coordinate and every CDF endpoint is on
    // that same grid. Intervals are [previous, endpoint), so an exact endpoint
    // belongs to the following emitter rather than the preceding one.
    if (sceneLoadEmitterCdf(mid) <= xi) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  return min(lo, emitterCount - 1u);
}

fn emitterCdfPmf(
  emitterCount: u32,
  lid: u32,
) -> f32 {
  if (emitterCount == 0u || lid >= emitterCount) {
    return 0.0;
  }
  let here = clamp(sceneLoadEmitterCdf(lid), 0.0, 1.0);
  var prev = 0.0;
  if (lid > 0u) {
    prev = clamp(sceneLoadEmitterCdf(lid - 1u), 0.0, 1.0);
  }
  return max(0.0, here - prev);
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const EMITTER_SAMPLING_MODULE: WgslModule = {
  name: "emitterSampling",
  source: EMITTER_SAMPLING_WGSL,
  requires: [],
};
