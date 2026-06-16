/**
 * Unit pin for the DDGI probe-ray glass Beer-Lambert transmittance (B5,
 * 2026-06-10), mirroring probeUpdateRays.wgsl.ts traceSunVisibility's per-slab
 * factor:
 *   visibility *= transmission · exp(-attenuationColor · (pathLen / attenuationDistance))
 *   pathLen = clamp(distToExit, 0, max(thickness, 1e-4))
 *
 * Analytic limit checks (the prior linear-tint form failed ALL of these — it
 * had no exponential, no thickness, so it did not reduce to Beer-Lambert in
 * any limit). These are closed-form, not Monte-Carlo, so they pin the math
 * exactly.
 */

import { describe, it, expect } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';

type V3 = [number, number, number];

/** Pure-TS mirror of the WGSL per-slab Beer-Lambert factor. */
function beerSlab(
  transmission: number, attenColor: V3, attenDist: number,
  distToExit: number, thickness: number,
): V3 {
  const pathLen = Math.min(Math.max(distToExit, 0), Math.max(thickness, 1e-4));
  const k = pathLen / Math.max(1e-4, attenDist);
  return [
    transmission * Math.exp(-attenColor[0] * k),
    transmission * Math.exp(-attenColor[1] * k),
    transmission * Math.exp(-attenColor[2] * k),
  ];
}

describe('DDGI probe-ray Beer-Lambert glass transmittance (B5)', () => {
  it('σ→0 (clear glass) ⇒ exp(0)=1, passes transmission only', () => {
    const v = beerSlab(0.9, [0, 0, 0], 0.5, 0.2, 0.3);
    expect(v[0]).toBeCloseTo(0.9, 10);
    expect(v[1]).toBeCloseTo(0.9, 10);
    expect(v[2]).toBeCloseTo(0.9, 10);
  });

  it('pathLen→0 ⇒ exp(0)=1 regardless of σ', () => {
    const v = beerSlab(1.0, [5, 10, 20], 0.1, 0.0, 0.0);
    expect(v[0]).toBeCloseTo(1.0, 10);
  });

  it('σ→∞ (or thick slab) ⇒ exp(-∞)=0, opaque', () => {
    const v = beerSlab(1.0, [1e4, 1e4, 1e4], 0.5, 1.0, 1.0);
    expect(v[0]).toBeLessThan(1e-9);
  });

  it('matches closed-form exp(-σ·t/d) for a typical coloured slab', () => {
    // Red-absorbing glass: high σ in G/B, low in R → transmits red.
    const transmission = 1.0;
    const attenColor: V3 = [0.2, 2.0, 2.5];
    const attenDist = 0.4;
    const thickness = 0.5;
    const distToExit = 0.3; // ray exits before reaching `thickness`
    const v = beerSlab(transmission, attenColor, attenDist, distToExit, thickness);
    const k = distToExit / attenDist; // pathLen = 0.3 (< thickness)
    expect(v[0]).toBeCloseTo(Math.exp(-0.2 * k), 10);
    expect(v[1]).toBeCloseTo(Math.exp(-2.0 * k), 10);
    expect(v[2]).toBeCloseTo(Math.exp(-2.5 * k), 10);
    // Physical sanity: red survives more than green/blue → tinted red.
    expect(v[0]).toBeGreaterThan(v[1]);
    expect(v[1]).toBeGreaterThan(v[2]);
  });

  it('clamps path length to thickness (open/non-watertight glass guard)', () => {
    const attenColor: V3 = [3, 3, 3];
    const attenDist = 0.5;
    const thickness = 0.2;
    // Continuation ray exits far away (distToExit huge) — must clamp to thickness.
    const v = beerSlab(1.0, attenColor, attenDist, 1000.0, thickness);
    const kClamped = thickness / attenDist; // 0.4
    expect(v[0]).toBeCloseTo(Math.exp(-3 * kClamped), 10);
  });

  it('multiplicative over 2 slabs equals exp of summed optical depth', () => {
    const a = beerSlab(1.0, [1, 1, 1], 1.0, 0.5, 1.0); // k=0.5
    const b = beerSlab(1.0, [1, 1, 1], 1.0, 0.3, 1.0); // k=0.3
    const combined = a[0] * b[0];
    expect(combined).toBeCloseTo(Math.exp(-(0.5 + 0.3)), 10);
  });

  it('samples material-atlas alpha for DDGI direct-light shadow transmittance', () => {
    const wgsl = makeProbeUpdateRaysWGSL(4);

    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_SLOT_BASE_COLOR: u32 = 0u;');
    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;');
    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET: u32 = 10u;');
    expect(wgsl).toContain('fn ddgiMaterialAlphaCoverageForHit(hit: IntersectionResult) -> DdgiAlphaCoverage');
    expect(wgsl).toContain('let baseColorTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);');
    expect(wgsl).toContain('let alphaTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);');
    expect(wgsl).toContain('out.coverage = clamp(opacity * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);');
    expect(wgsl).toContain('fn ddgiTraceShadowTransmittance(origin: vec3f, dir: vec3f, tMax: f32, skipGlass: bool) -> f32');
    expect(wgsl).toContain('tau = tau * ddgiAlphaShadowTransmittanceForHit(hit);');
    expect(wgsl).toContain('let alphaT = ddgiAlphaShadowTransmittanceForHit(sHit);');
    expect(wgsl).toContain('visibility = visibility * alphaT;');
    expect(wgsl).toContain('let shadowT = ddgiTraceShadowTransmittance(shadowOrig, lightDir, dist - normalBias_p, false);');
    expect(wgsl).toContain('shadowT = ddgiTraceShadowTransmittance(hitPos + n * normalBias, wi, dist - normalBias, false);');
    expect(wgsl).not.toContain('let sHit = bvhTraceFirstHit(sRay);\\n\\t      if (sHit.didHit && sHit.dist < dist - normalBias) { continue; }');
  });
});
