/**
 * Stained-glass-specific lighting physics — opt-in WGSL module (T5).
 *
 * Extracted out of the general hybrid `shade.wgsl` pass so that generic
 * scenes stop paying for (and stop visually receiving) the stained-glass-
 * specific 5-tap sky-aperture probe that delivers diffuse sky
 *     illumination through a window cutout (was `lo_sky_aperture`).
 *
 * The aperture is gated by a UBO flag bit (mirroring the proven Radiance-Cascades
 * `sampleCascadeC0` precedent, which early-returns `vec3f(0)` when
 * `rcParams.enabled == 0u`). When its `ubo.stainedGlassFlags`
 * bit is unset the helper early-returns `vec3f(0)` — so flag-off is
 * bit-identical to "no stained-glass term" without a separate shader compile.
 *
 *   bit 1 (SG_FLAG_SKY_APERTURE) → lo_sg_aperture active
 *
 * Bit 0 is consumed by `refractiveCaustics.wgsl.ts` only as the optional
 * stained-glass boost/clamp calibration for the explicitly selected
 * `causticStrategy:'refractive-trace'`; the retired shadow-ray caustic helper
 * no longer ships in this module.
 *
 * The aperture math is identical to the original inline body in shade.wgsl
 * when its flag is on. The helper reads module-scope state
 * (ubo + bvh_index/bvh_position/bvh + bvh_beer) directly, exactly as the
 * inline versions did; only the per-pixel locals are passed as parameters.
 *
 * Bindings: this module declares no bindings of its own. It consumes the
 * group(1) BVH storage buffers + the group(1) `bvh_beer` Beer-Lambert
 * visible-colour TEXTURE (WS1 2026-05-29 — moved off the storage group to free
 * a slot for `bvh_normal`; passed to bvhTraceTintedVisibility by handle, not
 * ptr) + the group(2) `ubo` uniform, all declared by SHADE_WGSL (the only
 * consumer). The shared traversal helper
 * `bvhTraceTintedVisibility`, the `luminance`/`safe_normalize` helpers, and
 * the `INV_PI` constant all come from `common`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const STAINED_GLASS_SHADE_WGSL = /* wgsl */ `

// ── Multi-tap sky aperture probe (T5) ─────────────────────────────────
//
// For non-glass surfaces, ambient-only DDGI doesn't deliver
// perceptible diffuse-sky illumination. Without an explicit aperture
// probe, the back-wall + side walls + floor outside the small caustic
// patch render pitch black, which is un-physical for a room with a
// daylit window.
//
// Probe approach: trace 5 deterministic rays — one along the
// receiver normal + four more rotated 45° toward the sun direction
// (a square-pyramid pattern around the surface "up axis").
//
// T5 — opt-in: early-returns vec3f(0) unless ubo.stainedGlassFlags bit 1
// (SG_FLAG_SKY_APERTURE) is set. The math below is byte-identical to the
// original inline lo_sky_aperture body once the flag is on.
fn lo_sg_aperture(
  pos:     vec3f,
  normal:  vec3f,
  albedo:  vec3f,
  isGlass: bool,
  isMetal: bool,
) -> vec3f {
  if ((ubo.stainedGlassFlags & SG_FLAG_SKY_APERTURE) == 0u) { return vec3f(0.0); }
  if (isGlass || isMetal) { return vec3f(0.0); }
  let skyTint = ubo.skyTint;
  let skyIrradiance = ubo.skyIrradiance;
  let originSky = pos + normal * 1e-3;
  let upAxis = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.y) < 0.99);
  let tangent = safe_normalize(cross(upAxis, normal));
  let bitangent = cross(normal, tangent);
  // 5 taps: centre (0°), four 45° diagonals. Each tap accumulates a
  // SCALAR luminance — opaque hit → 0, clear sky → 1, glass-tinted
  // → ~0.3 (luminance of the tint vector). Going scalar instead of
  // vec3f kills the panel-edge banding.
  let cos45 = 0.7071068;
  let sin45 = 0.7071068;
  var skyAccum = 0.0;
  var weightAccum = 0.0;
  // Centre tap (along normal, weight 1.0). luminance(c) is the canonical
  // Rec.709 helper from COMMON_WGSL (shade requires common).
  {
    let v = bvhTraceTintedVisibility(bvh_beer, originSky, normal, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * 1.0;
    weightAccum = weightAccum + 1.0;
  }
  // Four diagonal taps at 45° off-normal.
  let diag0 = safe_normalize(normal * cos45 + tangent * sin45);
  let diag1 = safe_normalize(normal * cos45 - tangent * sin45);
  let diag2 = safe_normalize(normal * cos45 + bitangent * sin45);
  let diag3 = safe_normalize(normal * cos45 - bitangent * sin45);
  {
    let v = bvhTraceTintedVisibility(bvh_beer, originSky, diag0, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * cos45;
    weightAccum = weightAccum + cos45;
  }
  {
    let v = bvhTraceTintedVisibility(bvh_beer, originSky, diag1, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * cos45;
    weightAccum = weightAccum + cos45;
  }
  {
    let v = bvhTraceTintedVisibility(bvh_beer, originSky, diag2, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * cos45;
    weightAccum = weightAccum + cos45;
  }
  {
    let v = bvhTraceTintedVisibility(bvh_beer, originSky, diag3, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * cos45;
    weightAccum = weightAccum + cos45;
  }
  let skyVisScalar = skyAccum / max(weightAccum, 1e-6);
  let skyVisAvg = vec3f(skyVisScalar);
  return skyVisAvg * skyTint * skyIrradiance * albedo * INV_PI;
}

`;

/**
 * T5 — declarative include-graph entry. Requires `common` for the
 * `bvhTraceTintedVisibility` / `luminance` / `safe_normalize` helpers and the
 * `INV_PI` constant. The `SG_FLAG_*` bit masks + `ubo`/`bvh_*`/`bvh_beer`
 * declarations come from `walkaroundUbo` (via `common`) and SHADE_WGSL
 * respectively; SHADE_MODULE lists this module in its `requires`.
 */
export const STAINED_GLASS_SHADE_MODULE: WgslModule = {
  name: 'stainedGlassShade',
  source: STAINED_GLASS_SHADE_WGSL,
  requires: ['common'],
};
