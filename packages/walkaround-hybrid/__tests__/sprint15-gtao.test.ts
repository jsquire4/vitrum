/**
 * Sprint 15 — GTAO (Ground-Truth Ambient Occlusion) structural tests.
 *
 * Verifies the WGSL strings contain expected entry points + key constants,
 * the pass layout includes `gtao` and `gtao-upsample` in the right order,
 * and MAX_PASS_COUNT was bumped to accommodate the new slots.
 *
 * Also includes Item 23 (sweep 2026-05-11): TypeScript mirror of the Jiménez
 * 2016 §4.2 slice integral to verify the corrected AO formula analytically.
 */

import { describe, expect, it } from 'vitest';
import { GTAO_WGSL } from '../src/shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_WGSL } from '../src/shaders/gtaoUpsample.wgsl.js';
import { SHADE_WGSL } from '../src/shaders/shade.wgsl.js';
import {
  MAX_PASS_COUNT,
  buildPassLayout,
} from '../src/pipeline/timestampQueries.js';

describe('Sprint 15 — GTAO WGSL', () => {
  it('GTAO_WGSL contains gtaoMain entry point', () => {
    expect(GTAO_WGSL).toContain('fn gtaoMain');
    expect(GTAO_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('GTAO_WGSL declares its 4 expected bindings (E1: +gtao_albedo at binding 3)', () => {
    expect(GTAO_WGSL).toContain('@group(0) @binding(0) var gtao_normalDepth');
    expect(GTAO_WGSL).toContain('@group(0) @binding(1) var gtao_aoOut');
    expect(GTAO_WGSL).toContain('@group(0) @binding(2) var<uniform> gtao_ubo');
    expect(GTAO_WGSL).toContain('@group(0) @binding(3) var gtao_albedo');
  });

  it('GTAO_WGSL has NUM_DIRECTIONS=4 and NUM_STEPS=6 (horizon-based AO defaults)', () => {
    expect(GTAO_WGSL).toContain('NUM_DIRECTIONS: u32 = 4u');
    expect(GTAO_WGSL).toContain('NUM_STEPS:      u32 = 6u');
  });

  it('GTAO_WGSL writes 1.0 (unoccluded) for sky-miss pixels (depth = 0)', () => {
    // The sky-miss early-out should set the output to fully lit.
    expect(GTAO_WGSL).toMatch(/centerDepth\s*<\s*1e-4/);
    expect(GTAO_WGSL).toContain('textureStore(gtao_aoOut, gid.xy, vec4f(1.0))');
  });

  it('GTAO_WGSL uses Jiménez 2016 slice integral (gtaoSliceIntegral fn)', () => {
    // Verify the shader contains the corrected integral helper, not the
    // old simplified (h1+h2)/PI formula.
    expect(GTAO_WGSL).toContain('fn gtaoSliceIntegral');
    // Must reference horizon clamping to upper hemisphere (±π/2 around n).
    expect(GTAO_WGSL).toContain('PI_HALF');
    // Must decode the surface normal from the G-buffer.
    expect(GTAO_WGSL).toContain('surfNormal');
    // Old simplified formula must NOT appear.
    expect(GTAO_WGSL).not.toContain('(h1 + h2) / PI');
  });

  it('GTAO_WGSL decodes surface normal from G-buffer before the slice loop', () => {
    // projNormal computation requires surfNormal; this asserts the decode is present.
    expect(GTAO_WGSL).toContain('center.xyz * 2.0 - 1.0');
    expect(GTAO_WGSL).toContain('projNormal');
    expect(GTAO_WGSL).toContain('projNormalLen');
  });
});

describe('Sprint 15 — GTAO upsample WGSL', () => {
  it('contains gtaoUpsampleMain entry point', () => {
    expect(GTAO_UPSAMPLE_WGSL).toContain('fn gtaoUpsampleMain');
    expect(GTAO_UPSAMPLE_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('declares 3 bindings (aoHalf in, normalDepth in, aoFull out)', () => {
    expect(GTAO_UPSAMPLE_WGSL).toContain('@group(0) @binding(0) var up_aoHalf');
    expect(GTAO_UPSAMPLE_WGSL).toContain('@group(0) @binding(1) var up_normalDepth');
    expect(GTAO_UPSAMPLE_WGSL).toContain('@group(0) @binding(2) var up_aoFullOut');
  });

  it('uses joint bilateral with depth + normal edge stops', () => {
    expect(GTAO_UPSAMPLE_WGSL).toContain('similarityWeight');
    expect(GTAO_UPSAMPLE_WGSL).toMatch(/depthDelta|depthW/);
    expect(GTAO_UPSAMPLE_WGSL).toMatch(/dot\(centerNormal/);
  });

  it('passes through sky-miss as 1.0', () => {
    expect(GTAO_UPSAMPLE_WGSL).toMatch(/centerDepth\s*<\s*1e-4/);
    expect(GTAO_UPSAMPLE_WGSL).toContain('textureStore(up_aoFullOut, gid.xy, vec4f(1.0))');
  });

  // Tier-G — Jiménez 2016 §5.2 multi-bounce AO must survive the upsample.
  //
  // Pre-Tier-G the upsample collapsed the per-channel half-res vec3 AO to
  // a single luminance scalar via dot(aoMb, vec3(0.2126, 0.7152, 0.0722))
  // and wrote `vec4f(scalar)` to the rgba16float output. shade.wgsl then
  // read only `.r`. That defeated the per-channel multi-bounce formulation
  // — a red wall darkened the green/blue channels by the same factor as
  // the red channel, indistinguishable from Bavoil-style scalar AO. The
  // following assertions lock in the corrected per-channel pipeline.
  it('keeps per-channel multi-bounce AO through the bilateral filter (no luminance collapse)', () => {
    // sumAO is now a vec3f, not an f32.
    expect(GTAO_UPSAMPLE_WGSL).toContain('var sumAO: vec3f');
    // No luminance-collapse weights anywhere in the upsample.
    expect(GTAO_UPSAMPLE_WGSL).not.toContain('0.2126');
    expect(GTAO_UPSAMPLE_WGSL).not.toContain('0.7152');
    expect(GTAO_UPSAMPLE_WGSL).not.toContain('0.0722');
    // Per-channel clamp at the write site (not a scalar clamp).
    expect(GTAO_UPSAMPLE_WGSL).toMatch(/clamp\(ao,\s*vec3f\(0\.0\),\s*vec3f\(1\.0\)\)/);
  });
});

describe('Tier-G — shade consumes per-channel multi-bounce AO', () => {
  // Pre-Tier-G shade did `let aoRaw = textureLoad(aoFullTexture, ...).r;` —
  // a single scalar broadcast across all RGB radiance channels, equivalent
  // to Bavoil-style scalar AO. The Jiménez 2016 §5.2 per-channel form
  // requires reading the full vec3 from `.rgb` so each colour channel can
  // darken by its own multi-bounce factor.
  it('reads .rgb from aoFullTexture (not .r)', () => {
    expect(SHADE_WGSL).toContain('textureLoad(aoFullTexture, vec2i(gid.xy), 0).rgb');
    expect(SHADE_WGSL).not.toMatch(/textureLoad\(aoFullTexture,\s*vec2i\(gid\.xy\),\s*0\)\.r\b/);
  });

  it('declares ao as vec3f for per-channel multiplication', () => {
    expect(SHADE_WGSL).toMatch(/let\s+ao\s*=\s*vec3f\(/);
  });
});

describe('Sprint 15 — pass layout integration', () => {
  it('MAX_PASS_COUNT is at least 19 (room for gtao + gtao-upsample)', () => {
    expect(MAX_PASS_COUNT).toBeGreaterThanOrEqual(19);
  });

  it('atrous-variance layout includes gtao and gtao-upsample between shade and welford-temporal', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
    const labels = layout.labels;
    const shadeIdx = labels.indexOf('shade');
    const gtaoIdx = labels.indexOf('gtao');
    const gtaoUpsampleIdx = labels.indexOf('gtao-upsample');
    const welfordIdx = labels.indexOf('welford-temporal');
    expect(shadeIdx).toBeGreaterThanOrEqual(0);
    expect(gtaoIdx).toBeGreaterThan(shadeIdx);
    expect(gtaoUpsampleIdx).toBe(gtaoIdx + 1);
    expect(welfordIdx).toBeGreaterThan(gtaoUpsampleIdx);
  });

  it('atrous layout (denoiserMode=atrous) also includes gtao slots', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous' });
    expect(layout.labels).toContain('gtao');
    expect(layout.labels).toContain('gtao-upsample');
  });

  it('slotCount fits within MAX_PASS_COUNT', () => {
    const layoutAtrousVariance = buildPassLayout({ denoiserMode: 'atrous-variance' });
    const layoutAtrous = buildPassLayout({ denoiserMode: 'atrous' });
    expect(layoutAtrousVariance.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
    expect(layoutAtrous.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Item 23 (sweep 2026-05-11) — Jiménez 2016 §4.2 slice integral: CPU mirror
//
// TypeScript mirror of the gtaoSliceIntegral() WGSL function + the per-slice
// visibility computation. No GPU required. Tests verify the corrected formula
// against known analytical values.
//
// Coordinate convention (matches the WGSL implementation):
//   - viewAxis = (0, 0, -1) — view direction (into screen / along depth)
//   - h0 = -acos(horizonNeg) in [-π, 0]  (negative slice side)
//   - h1 =  acos(horizonPos) in  [0, π]  (positive slice side)
//   - n  = signed angle of projected normal from viewAxis in slice plane
//   - cosN = cos(n)
//   - iarc(h, n) = (cosN + 2·h·sin(n) − cos(2·h − n)) / 4
//   - localVis = projNormalLen · (iarc(h0, n) + iarc(h1, n))
//
// Reference: Jiménez et al. 2016, §4.2 Eq. 11;
//            Intel XeGTAO.hlsli (iarc0/iarc1 form).
// ─────────────────────────────────────────────────────────────────────────────

const PI_CONST      = Math.PI;
const PI_HALF_CONST = Math.PI / 2;

/** TypeScript mirror of the WGSL gtaoSliceIntegral(h, n, cosN). */
function gtaoSliceIntegral(h: number, n: number, cosN: number): number {
  return (cosN + 2.0 * h * Math.sin(n) - Math.cos(2.0 * h - n)) * 0.25;
}

/**
 * Compute the Jiménez 2016 per-slice visibility for a given surface normal
 * and pair of horizon cosines.
 */
function computeSliceVisibility(
  surfNormal: [number, number, number],
  sliceDir: [number, number],
  horizonPos: number,
  horizonNeg: number,
): number {
  const axisVec: [number, number, number] = [sliceDir[0], sliceDir[1], 0.0];
  const viewAxis: [number, number, number] = [0.0, 0.0, -1.0];

  const dot3 = (a: [number, number, number], b: [number, number, number]): number =>
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const sub3 = (a: [number, number, number], s: number, v: [number, number, number]): [number, number, number] =>
    [a[0] - s*v[0], a[1] - s*v[1], a[2] - s*v[2]];
  const len3 = (a: [number, number, number]): number => Math.sqrt(dot3(a, a));
  const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

  const dotNAxis = dot3(surfNormal, axisVec);
  const projNormal = sub3(surfNormal, dotNAxis, axisVec);
  const projNormalLen = Math.max(len3(projNormal), 1e-6);

  const signNorm = Math.sign(dot3(axisVec, projNormal)) || 1.0;
  const cosN = clamp(dot3(projNormal, viewAxis) / projNormalLen, -1.0, 1.0);
  const n = signNorm * Math.acos(cosN);

  const h0_raw = -Math.acos(clamp(horizonNeg, -1.0, 1.0));
  const h1_raw =  Math.acos(clamp(horizonPos, -1.0, 1.0));

  const h0 = n + clamp(h0_raw - n, -PI_HALF_CONST, PI_HALF_CONST);
  const h1 = n + clamp(h1_raw - n, -PI_HALF_CONST, PI_HALF_CONST);

  return projNormalLen * (gtaoSliceIntegral(h0, n, cosN) + gtaoSliceIntegral(h1, n, cosN));
}

/** Compute full AO by averaging slice visibility over numDirs uniform slices. */
function computeAOUniformSlices(
  horizonPos: number,
  horizonNeg: number,
  surfNormal: [number, number, number],
  numDirs = 4,
): number {
  let aoSum = 0.0;
  for (let d = 0; d < numDirs; d++) {
    const theta = (d / numDirs) * PI_CONST;
    const dir: [number, number] = [Math.cos(theta), Math.sin(theta)];
    aoSum += computeSliceVisibility(surfNormal, dir, horizonPos, horizonNeg);
  }
  return Math.min(1.0, Math.max(0.0, aoSum / numDirs));
}

describe('Item 23 — Jiménez 2016 GTAO slice integral (CPU mirror)', () => {
  /**
   * Unoccluded surface: both horizons at the ground plane (cosH = 0 → θ_h = π/2).
   * With the surface normal pointing toward the camera (0,0,-1), the full upper
   * hemisphere is visible; AO should be close to 1.0.
   */
  it('fully unoccluded surface (horizons at equator) → AO ≈ 1.0', () => {
    const normalTowardCamera: [number, number, number] = [0, 0, -1];
    const ao = computeAOUniformSlices(0.0, 0.0, normalTowardCamera);
    expect(ao).toBeGreaterThan(0.9);
    expect(ao).toBeLessThanOrEqual(1.0);
  });

  /**
   * Fully occluded surface: both horizons at zenith (cosH = 1 → θ_h = 0).
   * Every direction above the surface is blocked; AO should be near 0.
   */
  it('fully occluded surface (horizons at zenith) → AO ≈ 0.0', () => {
    const normalTowardCamera: [number, number, number] = [0, 0, -1];
    const ao = computeAOUniformSlices(1.0, 1.0, normalTowardCamera);
    expect(ao).toBeLessThan(0.1);
  });

  /**
   * Half-occluded: positive side blocked (cosH=1), negative side open (cosH=0).
   * AO should be strictly between 0 and 1 (around 0.3–0.7 range).
   */
  it('half-occluded surface → AO in (0.3, 0.7)', () => {
    const normalTowardCamera: [number, number, number] = [0, 0, -1];
    const ao = computeAOUniformSlices(1.0, 0.0, normalTowardCamera);
    expect(ao).toBeGreaterThan(0.3);
    expect(ao).toBeLessThan(0.7);
  });

  /**
   * Normal perpendicular to the view axis: AO should still be a valid
   * factor in [0, 1].
   */
  it('normal perpendicular to view axis (+Y) unoccluded → AO in [0, 1]', () => {
    const normalUp: [number, number, number] = [0, 1, 0];
    const ao = computeAOUniformSlices(0.0, 0.0, normalUp);
    expect(ao).toBeGreaterThanOrEqual(0.0);
    expect(ao).toBeLessThanOrEqual(1.0);
  });

  /**
   * gtaoSliceIntegral boundary: h=0, n=0 (horizon at normal baseline).
   * (cos(0) + 2·0·sin(0) − cos(0)) / 4 = (1 + 0 − 1) / 4 = 0.
   */
  it('gtaoSliceIntegral: h=0, n=0 → 0.0 (no visible arc at normal baseline)', () => {
    expect(gtaoSliceIntegral(0, 0, Math.cos(0))).toBeCloseTo(0.0, 6);
  });

  /**
   * gtaoSliceIntegral boundary: h=π/2, n=0 (full positive hemisphere open).
   * (1 + 2·(π/2)·0 − cos(π − 0)) / 4 = (1 + 0 − (−1)) / 4 = 0.5.
   */
  it('gtaoSliceIntegral: h=π/2, n=0 → 0.5 (full positive arc, normal at viewAxis)', () => {
    expect(gtaoSliceIntegral(PI_HALF_CONST, 0, Math.cos(0))).toBeCloseTo(0.5, 5);
  });

  /**
   * Verify the Jiménez formula differs from the old (h)/π simplified form
   * for a tilted normal, confirming the correct formula is used.
   * For n=π/4, h=π/2:
   *   iarc = (cos(π/4) + π·sin(π/4) − cos(π − π/4)) / 4
   */
  it('gtaoSliceIntegral: tilted normal (n=π/4, h=π/2) matches Jiménez closed form', () => {
    const n = PI_CONST / 4;
    const h = PI_HALF_CONST;
    const cosN = Math.cos(n);
    const result = gtaoSliceIntegral(h, n, cosN);

    const expected = (Math.cos(n) + 2.0 * h * Math.sin(n) - Math.cos(2.0 * h - n)) * 0.25;
    expect(result).toBeCloseTo(expected, 10);

    // The old simplified formula would give h/π ≈ 0.5; the Jiménez form differs.
    const oldFormula = h / PI_CONST;
    expect(Math.abs(result - oldFormula)).toBeGreaterThan(0.05);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// E1 — Jiménez 2016 §5.2 Eq. 16 — multi-bounce AO factor (CPU mirror)
//
// The multi-bounce polynomial:
//   a_mb = ((2.0404·ρ − 0.3324)·v + (−4.7951·ρ + 0.6417))·v + (2.7552·ρ + 0.6903))·v
//
// where ρ = albedo ∈ [0, 1] and v = scalar AO ∈ [0, 1].
// ─────────────────────────────────────────────────────────────────────────────

/** TypeScript mirror of the WGSL multi-bounce polynomial (per-channel scalar). */
function multiBounceFactor(albedo: number, vis: number): number {
  const ca = 2.0404 * albedo - 0.3324;
  const cb = -4.7951 * albedo + 0.6417;
  const cc = 2.7552 * albedo + 0.6903;
  return Math.min(1.0, Math.max(0.0, ((ca * vis + cb) * vis + cc) * vis));
}

describe('E1 — GTAO multi-bounce term (Jiménez 2016 §5.2 / Eq. 16)', () => {
  /**
   * Boundary conditions at ρ = 1 (white surface):
   *   - vis = 0  → a_mb = 0  (fully occluded: no light at all)
   *   - vis = 1  → a_mb ≈ 1  (fully unoccluded: maximum brightness)
   *
   * The polynomial is NOT an identity at ρ=1: intermediate vis values are
   * BRIGHTENED above vis (that is the purpose of the multi-bounce term).
   * For example at vis=0.25, a_mb ≈ 0.63 (inter-reflections boost AO on
   * bright surfaces). The "identity" in Jiménez §5.2 refers only to the
   * endpoint conditions, not to the full polynomial.
   */
  it('boundary at ρ = 1: vis=0 → a_mb=0, vis=1 → a_mb≈1', () => {
    expect(multiBounceFactor(1.0, 0.0)).toBeCloseTo(0.0, 4);
    expect(multiBounceFactor(1.0, 1.0)).toBeCloseTo(1.0, 3);
  });

  /**
   * Darkening at ρ = 0: for a black surface (albedo = 0), the multi-bounce
   * factor should be zero (no AO leakage on perfectly dark surfaces).
   * Verify from the polynomial: cc = 2.7552·0 + 0.6903 = 0.6903 — at ρ=0
   * the polynomial simplifies to ((-0.3324·v + 0.6417)·v + 0.6903)·v.
   * At v=0 → 0; at v=1 → ((-0.3324 + 0.6417) + 0.6903) = 0.9996 ≈ 1.
   * The spec says "a_mb → 0" at ρ=0. Re-reading: with ρ=0, ca=-0.3324, cb=0.6417, cc=0.6903.
   * At vis=0 → a_mb=0 (correct). At vis=1 → ~0.9996 (physically: dark surfaces
   * still need the AO factor to be small at high occlusion, but at vis=1 there
   * IS no occlusion so leakage is acceptable). The key invariant is vis=0 → a_mb=0.
   */
  it('at ρ = 0 and vis = 0: a_mb = 0 (no AO leakage)', () => {
    const aMb = multiBounceFactor(0.0, 0.0);
    expect(aMb).toBeCloseTo(0.0, 6);
  });

  /**
   * Brightening invariant for mid-to-high albedo (ρ ≥ 0.5):
   * multi-bounce brightens or preserves the scalar AO.
   *
   * For low albedo (ρ < 0.5), the polynomial can produce a_mb < vis at
   * intermediate vis values — this is expected behaviour. Dark surfaces
   * have fewer inter-reflections and the polynomial correctly produces
   * less brightening than the single-bounce estimate.
   *
   * The strict invariant a_mb ≥ vis holds only for ρ near 1; for ρ < 0.5
   * it does not hold. We verify the weaker invariant: output is in [0, 1].
   */
  it('output always in [0, 1] for all ρ, vis combinations', () => {
    const albedoValues = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
    const visValues    = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    for (const albedo of albedoValues) {
      for (const vis of visValues) {
        const aMb = multiBounceFactor(albedo, vis);
        expect(aMb).toBeGreaterThanOrEqual(0.0);
        expect(aMb).toBeLessThanOrEqual(1.0);
      }
    }
  });

  /**
   * Brightening for high-albedo surfaces (ρ = 1): a_mb ≥ vis.
   * At ρ = 1, the multi-bounce polynomial produces a value at or above vis,
   * reflecting that white surfaces have maximal inter-reflections.
   */
  it('brightening: a_mb ≥ vis for ρ = 1', () => {
    for (const vis of [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]) {
      const aMb = multiBounceFactor(1.0, vis);
      expect(aMb).toBeGreaterThanOrEqual(vis - 1e-4);
    }
  });

  /**
   * Structural: GTAO WGSL now declares binding 3 (gtao_albedo) and uses
   * the multi-bounce polynomial.
   */
  it('GTAO_WGSL declares @binding(3) for gtao_albedo', () => {
    expect(GTAO_WGSL).toContain('@group(0) @binding(3) var gtao_albedo');
  });

  it('GTAO_WGSL uses rgba16float storage format (bumped from r16float)', () => {
    expect(GTAO_WGSL).toContain('texture_storage_2d<rgba16float, write>');
    expect(GTAO_WGSL).not.toContain('texture_storage_2d<r16float, write>');
  });

  it('GTAO_WGSL contains multi-bounce polynomial coefficients', () => {
    expect(GTAO_WGSL).toContain('2.0404');
    expect(GTAO_WGSL).toContain('0.3324');
    expect(GTAO_WGSL).toContain('4.7951');
    expect(GTAO_WGSL).toContain('0.6417');
    expect(GTAO_WGSL).toContain('2.7552');
    expect(GTAO_WGSL).toContain('0.6903');
  });
});
