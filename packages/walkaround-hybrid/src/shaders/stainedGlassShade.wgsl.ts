/**
 * Stained-glass-specific lighting physics — opt-in WGSL module (T5).
 *
 * Extracted out of the general hybrid `shade.wgsl` pass so that generic
 * scenes stop paying for (and stop visually receiving) the two
 * stained-glass-specific direct-lighting terms:
 *
 *   - `lo_sg_caustic`  — sun directional light reaching a receiver through
 *     tinted glass, with the UBO-driven `causticBoost` / `causticVisClamp`
 *     calibration (was `lo_sun_caustic` in shade.wgsl).
 *   - `lo_sg_aperture` — 5-tap sky-aperture probe that delivers diffuse sky
 *     illumination through a window cutout (was `lo_sky_aperture`).
 *
 * Both are gated by a UBO flag bit (mirroring the proven Radiance-Cascades
 * `sampleCascadeC0` precedent, which early-returns `vec3f(0)` when
 * `rcParams.enabled == 0u`). When the corresponding `ubo.stainedGlassFlags`
 * bit is unset the helper early-returns `vec3f(0)` — so flag-off is
 * bit-identical to "no stained-glass term" without a separate shader compile.
 *
 *   bit 0 (SG_FLAG_SUN_CAUSTIC)  → lo_sg_caustic active
 *   bit 1 (SG_FLAG_SKY_APERTURE) → lo_sg_aperture active
 *
 * Default both bits 0 (host opts in via HybridEngineOptions.stainedGlass).
 *
 * The math inside each helper is identical to the original inline bodies in
 * shade.wgsl when its flag is on. The helpers read module-scope state
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

// ── Direct sun lighting with glass-aware tinted shadow ray (T5) ────────
//
// Bullet 4 (caustics on receivers): the sun is treated as a directional
// light reaching the floor/walls.  The shadow ray from the receiver
// toward the sun walks every triangle along the path:
//   - opaque hit  → fully shadowed (visibility = vec3f(0))
//   - glass hit   → multiply visibility by the cell's tint factor
//   - clear hit   → unchanged
// Same skip-on-metal rule: through-glass shadow rays from a came
// bead's irregular surface produce variable visibility per pixel → speckle.
// The ReSTIR-GI Lo_indirect term covers came illumination via the
// half-res reservoir read further below.
//
// T5 — opt-in: early-returns vec3f(0) unless ubo.stainedGlassFlags bit 0
// (SG_FLAG_SUN_CAUSTIC) is set. The math below is byte-identical to the
// original inline lo_sun_caustic body once the flag is on.
fn lo_sg_caustic(
  gid:     vec2u,
  pos:     vec3f,
  normal:  vec3f,
  albedo:  vec3f,
  isGlass: bool,
  isMetal: bool,
) -> vec3f {
  if ((ubo.stainedGlassFlags & SG_FLAG_SUN_CAUSTIC) == 0u) { return vec3f(0.0); }
  if (isGlass || isMetal) { return vec3f(0.0); }
  // Direction TOWARD the sun.  ubo.sunDirection is the unit vector from
  // the world origin toward the sun.
  // Sun-cone sampling for physically-correct caustic penumbra.
  // Real sun has 0.5° angular diameter → 0.25° radius → tan ≈ 0.00436.
  //
  // Sampling strategy: PER-PIXEL DETERMINISTIC, no per-frame variance.
  // Each pixel always samples the SAME point on the sun cone (a
  // function of its (x, y) position only).
  let sunBase = ubo.sunDirection;
  let SUN_ANGULAR_RADIUS = 0.00436;
  let hx = fract(sin(f32(gid.x) * 12.9898 + f32(gid.y) * 78.233) * 43758.5453);
  let hy = fract(sin(f32(gid.x) * 93.989  + f32(gid.y) * 67.345) * 24634.6345);
  let xi = vec2f(hx, hy);
  let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let tan = safe_normalize(cross(upRef, sunBase));
  let bit = cross(sunBase, tan);
  let r2 = SUN_ANGULAR_RADIUS * sqrt(xi.x);
  let phi = 6.2831853 * xi.y;
  let toSun = safe_normalize(sunBase + tan * (r2 * cos(phi)) + bit * (r2 * sin(phi)));
  let nDotSun = max(0.0, dot(normal, toSun));
  if (nDotSun <= 1e-6) { return vec3f(0.0); }
  let vis = bvhTraceTintedVisibility(
    &bvh_index, &bvh_position, &bvh, bvh_beer,
    pos + normal * 1e-3, toSun, 1e6,
  );
  // Sun irradiance × tinted visibility × Lambert(receiver) × CAUSTIC_BOOST.
  // CAUSTIC_BOOST 10 → 22: less-saturated cells (e.g., brown) Beer-Lambert
  // to pow(0.55, 6) ≈ 0.028 — caustics from those cells were below ambient
  // floor brightness, invisible against the soft DDGI cell-tint blob.
  // Audit B1: CAUSTIC_BOOST and the visibility clamp are now UBO-driven.
  // Cornell stained-glass uses 22.0 / 0.6 (the historical calibration);
  // generic scenes pass 1.0 / 1.0 (no boost, no clamp).
  let visClamped = min(vis, vec3f(ubo.causticVisClamp));
  return visClamped * ubo.sunIntensity * nDotSun * albedo * INV_PI * ubo.causticBoost;
}

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
    let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, bvh_beer, originSky, normal, 1e6);
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
    let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, bvh_beer, originSky, diag0, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * cos45;
    weightAccum = weightAccum + cos45;
  }
  {
    let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, bvh_beer, originSky, diag1, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * cos45;
    weightAccum = weightAccum + cos45;
  }
  {
    let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, bvh_beer, originSky, diag2, 1e6);
    let lum = luminance(v);
    skyAccum = skyAccum + lum * cos45;
    weightAccum = weightAccum + cos45;
  }
  {
    let v = bvhTraceTintedVisibility(&bvh_index, &bvh_position, &bvh, bvh_beer, originSky, diag3, 1e6);
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
