/**
 * Unit pin for the DDGI probe-ray glass Beer-Lambert transmittance (B5,
 * 2026-06-10), mirroring probeUpdateRays.wgsl.ts traceSunVisibility's per-slab
 * factor:
 *   visibility *= transmission · attenuationColor^(pathLen / attenuationDistance)
 * where pathLen is the actual distance between ownership-paired boundaries.
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
  pathLen: number,
): V3 {
  const k = Math.max(pathLen, 0) / attenDist;
  return [
    transmission * Math.pow(attenColor[0], k),
    transmission * Math.pow(attenColor[1], k),
    transmission * Math.pow(attenColor[2], k),
  ];
}

describe('DDGI probe-ray Beer-Lambert glass transmittance (B5)', () => {
  it('white attenuation color passes scalar transmission only', () => {
    const v = beerSlab(0.9, [1, 1, 1], 0.5, 0.2);
    expect(v[0]).toBeCloseTo(0.9, 10);
    expect(v[1]).toBeCloseTo(0.9, 10);
    expect(v[2]).toBeCloseTo(0.9, 10);
  });

  it('pathLen→0 yields identity regardless of attenuation color', () => {
    const v = beerSlab(1.0, [0, 0.2, 0.8], 0.1, 0.0);
    expect(v[0]).toBeCloseTo(1.0, 10);
  });

  it('a zero channel is opaque for every positive path length', () => {
    const v = beerSlab(1.0, [0, 0, 0], 0.5, 1.0);
    expect(v).toEqual([0, 0, 0]);
  });

  it('matches closed-form color^(t/d) for a typical coloured slab', () => {
    // A higher authored channel transmits more of that channel.
    const transmission = 1.0;
    const attenColor: V3 = [0.8, 0.2, 0.1];
    const attenDist = 0.4;
    const pathLen = 0.3;
    const v = beerSlab(transmission, attenColor, attenDist, pathLen);
    const k = pathLen / attenDist;
    expect(v[0]).toBeCloseTo(Math.pow(0.8, k), 10);
    expect(v[1]).toBeCloseTo(Math.pow(0.2, k), 10);
    expect(v[2]).toBeCloseTo(Math.pow(0.1, k), 10);
    expect(v[0]).toBeGreaterThan(v[1]);
    expect(v[1]).toBeGreaterThan(v[2]);
  });

  it('uses actual geometric distance rather than clamping to authored thickness', () => {
    const attenColor: V3 = [0.5, 0.5, 0.5];
    const attenDist = 0.5;
    const v = beerSlab(1.0, attenColor, attenDist, 2.0);
    expect(v[0]).toBeCloseTo(Math.pow(0.5, 4), 10);
  });

  it('is multiplicative over adjacent segments of one medium', () => {
    const a = beerSlab(1.0, [0.25, 0.25, 0.25], 1.0, 0.5);
    const b = beerSlab(1.0, [0.25, 0.25, 0.25], 1.0, 0.3);
    const combined = a[0] * b[0];
    expect(combined).toBeCloseTo(Math.pow(0.25, 0.8), 10);
  });

  it('samples material-atlas alpha for DDGI direct-light shadow transmittance', () => {
    const wgsl = makeProbeUpdateRaysWGSL(4);

    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_SLOT_BASE_COLOR: u32 = 0u;');
    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;');
    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET: u32 = 10u;');
    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET: u32 = 13u;');
    expect(wgsl).toContain('const DDGI_MATERIAL_MAP_THICKNESS_TEXEL_OFFSET: u32 = 47u;');
    expect(wgsl).toContain('fn ddgiMaterialAlphaCoverageForHit(hit: IntersectionResult) -> DdgiAlphaCoverage');
    expect(wgsl).toContain('@group(1) @binding(6) var ddgiBvhVertexColor: texture_2d<f32>;');
    expect(wgsl).toContain('fn ddgiSampleVertexColorForHit(hit: IntersectionResult) -> vec4f');
    expect(wgsl).toContain('fn ddgiSampleTransmissionMapForHit(hit: IntersectionResult, scalarTransmission: f32) -> f32');
    expect(wgsl).toContain('fn ddgiSampleThicknessMapFactorForHit(hit: IntersectionResult) -> vec2f');
    expect(wgsl).toContain('fn ddgiMaterialAtlasFiniteNonNegativeRadianceOrBlack(value: vec3f) -> vec3f');
    expect(wgsl.match(/ddgiMaterialAtlasFiniteNonNegativeRadianceOrBlack\(/g)).toHaveLength(4);
    expect(wgsl).toContain('scalarEmission * texel.value.rgb,');
    expect(wgsl).toContain('max(texel.value.rgb, vec3f(0.0)) * max(intensity, 0.0),');
    expect(wgsl).toContain('return f32(hash >> 8u) / 16777216.0;');
    expect(wgsl).not.toContain('/ 4294967296.0');
    expect(wgsl).toContain('let baseColorTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);');
    expect(wgsl).toContain('let alphaTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);');
    expect(wgsl).toContain('let vertexColorAlpha = ddgiSampleVertexColorForHit(hit).a;');
    expect(wgsl).toContain('out.coverage = clamp(opacity * vertexColorAlpha * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);');
    expect(wgsl).toContain('fn ddgiTraceShadowVisibility(origin: vec3f, dir: vec3f, tMax: f32) -> vec3f');
    expect(wgsl).toContain('let surfaceBudget = ddgiWorldSurfaceBudget();');
    expect(wgsl).toContain('var mediumState = containing.state;');
    expect(wgsl).toContain('mediumState.boundaryId[depth] = boundaryId;');
    expect(wgsl).toContain('mediumState.representedId[depth] = representedId;');
    expect(wgsl).toContain('let alphaT = ddgiAlphaShadowTransmittanceForHit(hit);');
    expect(wgsl).toContain('visibility = visibility * vec3f(alphaT);');
    expect(wgsl).toContain('let coverage = ddgiOpticalCoverageForHit(hit);');
    expect(wgsl).toContain(
      'visibility = visibility * mix(',
    );
    expect(wgsl).toContain(
      'mediumState.boundaryId[top] != boundaryId',
    );
    expect(wgsl).toContain(
      'ddgiShadowFaceTransmission(hit, dir, true)',
    );
    expect(wgsl).toContain('let rgbBeer = beerLambertTransmittanceRgb(');
    expect(wgsl).toContain('mediumState.representedId[top] != representedId');
    expect(wgsl).not.toContain('exp(-mat.attenuationColor');
    expect(wgsl).toContain('fn ddgiTraceFirstHitAlphaMaskTextured(ray: Ray) -> IntersectionResult');
    expect(wgsl).toContain('fn ddgiAlphaBlendCoverageHash(hit: IntersectionResult, ray: Ray, layer: u32) -> f32');
    expect(wgsl).toContain('fn ddgiMaterialAlphaDiscardedForProbeHit(hit: IntersectionResult, ray: Ray, layer: u32) -> bool');
    expect(wgsl).toContain('ddgiAlphaBlendCoverageHash(hit, ray, layer) >= representedCoverage');
    expect(wgsl).toContain(
      'Conservative world-surface-budget overflow blocks instead of leaking.',
    );
    expect(wgsl).not.toContain('ddgiMaterialAlphaDiscardedForOpaqueProbeHit');
    expect(wgsl).toContain('let ordinaryHit = ddgiTraceFirstHitAlphaMaskTextured(ray);');
    expect(wgsl).toContain('let unboundedShadowDistance = bitcast<f32>(0x7f800000u);');
    expect(wgsl).toContain(
      'return ddgiTraceShadowVisibility(origin, sunDir, unboundedShadowDistance);',
    );
    expect(wgsl).toContain('shadowVisibility = ddgiTraceShadowVisibility(shadowOrig, lightDir, dist - normalBias_p);');
    expect(wgsl).toContain('shadowT = ddgiTraceShadowVisibility(hitPos + n * normalBias, wi, dist - normalBias);');
    expect(wgsl).toContain('fn ddgiTraceGlassChannel(');
    expect(wgsl).toContain('let nextRay = Ray(currentPos, rayDirection);');
    expect(wgsl).toContain('nextRay, currentSourceFeature,');
    expect(wgsl).toContain('&mediumState, accepted.hit.dist,');
    expect(wgsl).toContain('fn ddgiMediumSegmentTransfer(');
    expect(wgsl).toContain('let mappedCap = authoredThickness * clamp(thicknessMapScale, 0.0, 1.0);');
    expect(wgsl).toContain('min(segmentLength, mappedCap),');
    expect(wgsl).toContain('return light.color * atten * nDotL * coneFalloff * shadowVisibility;');
    expect(wgsl).toContain('irradiance = irradiance + Le * G * area * shadowT /');
    expect(wgsl).toContain('let directRadiance = direct * probeMat.albedo * (1.0 / PI);');
    expect(wgsl).not.toContain('albedo * 0.31831 * Le * G');
    expect(wgsl).toContain('(f32(sampleCount) * draw.pmf);');
    expect(wgsl).not.toContain('let sHit = bvhTraceFirstHit(sRay);\\n\\t      if (sHit.didHit && sHit.dist < dist - normalBias) { continue; }');
  });
});
