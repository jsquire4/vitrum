/**
 * B1 (road-to-100) — glossy/metal GI structural gate.
 * B1-ior-per-tri (2026-06-10) — per-tri IOR lane structural gate.
 *
 * Pins the shader-level contract of the B1 change so a future refactor cannot
 * silently re-introduce the metal/glass early-outs or the hardcoded
 * roughness/metalness, and so the p̂/consumption-consistency boundary stays
 * documented:
 *   1. ris / restirCastPrimary / shade decode REAL roughness/metal from the
 *      bvh_material texture (no `select(0.85, 0.05, isGlass)` / `metal = 0.0`).
 *   2. shade direct terms retain conductor and glass reflection lobes; glass
 *      does not acquire an opaque diffuse lobe.
 *   3. risGi / risGiNrc no longer punt metals to an empty GI reservoir.
 *      Primary glass pixels use the bounded refracted-GI walk instead of the old
 *      immediate empty-reservoir branch.
 *   4. shade adds the glossy/metal SPECULAR-indirect term (evalGGXSpecularOnly
 *      against the ReSTIR-GI reservoir sample), routed to the un-demodulated
 *      direct channel.
 *   5. B1-ior-per-tri: materialDecode exposes `decodeIor`; risGi glass walk
 *      uses per-tri IOR instead of a fixed 1.5 constant. Camera-prefix
 *      Fresnel/Beer is producer-owned and authored rough transmission uses the
 *      weighted GGX dielectric proposal instead of a delta-only approximation.
 */
import { describe, it, expect } from 'vitest';
import { RIS_WGSL } from '../ris.wgsl.js';
import { SHADE_WGSL } from '../shade.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { RESTIR_CAST_PRIMARY_WGSL } from '../restirCastPrimary.wgsl.js';
import { MATERIAL_DECODE_WGSL } from '../materialDecode.wgsl.js';
import { GGX_BRDF_WGSL } from '../ggxBrdf.wgsl.js';
import { RESTIR_PHAT_WGSL } from '../restirPHat.wgsl.js';
import { RESERVOIR_GI_WGSL } from '../reservoirGi.wgsl.js';

describe('B1 — real per-tri roughness/metalness decode', () => {
  it('materialDecode provides decodeRoughMetal + the bvh_material texel width', () => {
    expect(MATERIAL_DECODE_WGSL).toContain('fn decodeRoughMetal(packed: u32) -> vec2f');
    expect(MATERIAL_DECODE_WGSL).toContain('BVH_MATERIAL_TEX_WIDTH');
  });

  it('ris / cast / shade declare the bvh_material binding (group 1, binding 14)', () => {
    for (const src of [RIS_WGSL, RESTIR_CAST_PRIMARY_WGSL, SHADE_WGSL]) {
      expect(src).toContain('@group(1) @binding(14) var bvh_material: texture_2d<u32>;');
    }
  });

  it('ris / cast / shade no longer hardcode roughness/metalness', () => {
    for (const src of [RIS_WGSL, RESTIR_CAST_PRIMARY_WGSL]) {
      expect(src).toContain('textureLoad(bvh_material');
      expect(src).toContain('sampleRestirDIMaterialPayloadForHit(');
      // The old hardcoded forms must be gone.
      expect(src).not.toContain('select(0.85, 0.05, isGlass)');
    }
    expect(SHADE_WGSL).toContain('textureLoad(bvh_material');
    expect(SHADE_WGSL).toContain('decodeRoughMetal(');
    expect(SHADE_WGSL).not.toContain('select(0.85, 0.05, isGlass)');
  });
});

/** Slice a single `fn name(...) { ... }` body out of a WGSL source so a
 *  negative-match stays scoped to that function. Returns the text from `fn name` to the next
 *  top-level `fn ` declaration. */
function fnBody(src: string, name: string): string {
  const start = src.indexOf(`fn ${name}(`);
  expect(start).toBeGreaterThanOrEqual(0);
  const next = src.indexOf('\nfn ', start + 1);
  return src.slice(start, next < 0 ? undefined : next);
}

describe('B1 — metals and glass receive direct reflection', () => {
  it('generic glass does not self-emit unless stained-glass sun-caustic is opted in', () => {
    const body = fnBody(SHADE_WGSL, 'lo_emit');
    expect(body).toContain('if (!isGlass) { return vec3f(0.0); }');
    expect(body).toContain('if ((ubo.stainedGlassFlags & SG_FLAG_SUN_CAUSTIC) == 0u) { return vec3f(0.0); }');
    expect(body).toContain('return beerAlbedo * trans * ubo.sunIntensity * sunDot * texMod;');
  });

  it('direct, analytic, and sun estimators route glass through the reflection-only BRDF', () => {
    const helper = fnBody(SHADE_WGSL, 'evalDirectSurfaceBrdf');
    expect(helper).toContain('if (isGlass)');
    expect(helper).toContain('return evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(');
    expect(helper).toContain('return evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(');

    for (const name of ['lo_direct', 'lo_analyticNEE', 'lo_sunNEE']) {
      const body = fnBody(SHADE_WGSL, name);
      expect(body).not.toContain('if (isGlass) { return vec3f(0.0); }');
      expect(body).toContain('evalDirectSurfaceBrdf(');
    }
  });

  it('ReSTIR target evaluation uses the same glass reflection-only domain', () => {
    expect(RESERVOIR_GI_WGSL).toContain('isGlass: bool');
    const body = fnBody(RESTIR_PHAT_WGSL, 'restir_di_eval_surface_brdf');
    expect(body).toContain('if (surf.isGlass)');
    expect(body).toContain('evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(');
  });

  it('a dielectric interface has a non-zero normal-incidence reflection term', () => {
    const ior = 1.5;
    const f0 = ((ior - 1) / (ior + 1)) ** 2;
    expect(f0).toBeCloseTo(0.04, 12);
    expect(f0).toBeGreaterThan(0);
  });
});

describe('B1 — glossy/metal GI reservoir (no empty-reservoir punt for metal)', () => {
  it('risGi / risGiNrc handle primary glass with a bounded refracted-GI walk', () => {
    for (const src of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(src).not.toContain('if (isGlass || isMetal)');
      expect(src).not.toContain(`if (isGlass) {
    storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
    return;
  }`);
      expect(src).not.toContain('if (grisOn && isGlass)');
      expect(src).toContain('const GLASS_WALK_MAX_INTERFACES: u32 = 4u;');
      expect(src).toContain('var rGlass: ReservoirGI = emptyReservoirGI();');
      expect(src).toContain('rGlass.historyEpoch = currentGrisEpoch;');
      expect(src).toContain('updateReservoirGIWithMetadata(');
      expect(src).toContain('rGlass.xv = walkHitPos;');
      expect(src).toContain('storeReservoirGI_rw(pixelIdxGi, rGlass);');
    }
  });

  it('the GI producer targets the declared one-bounce DDGI proxy', () => {
    // The live generalized-reuse domain is receiver-independent: source
    // emission and one-bounce DDGI material response are stored at xs, while
    // the canonical target re-evaluates the visible receiver's authored lobes.
    expect(RIS_GI_WGSL).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    expect(RIS_GI_WGSL).toContain('Lo = xsPayload.Lo;');
    expect(RIS_GI_WGSL).toContain('pHat = restir_gi_receiver_phat_from_payload(');
    expect(RIS_GI_WGSL).toContain(') * candidateVisibility;');
    expect(RIS_GI_WGSL).toContain('let receiverPayload = sampleRestirDIMaterialPayloadForHit(');
    expect(RIS_GI_WGSL).toContain('pSrc = alpha * pGuide + (1.0 - alpha) * pCos;');
    expect(RIS_GI_WGSL).toContain('pSrc = cosTheta * INV_PI;');
    expect(RIS_GI_WGSL).toContain('let w = pHat / pSrc;');
  });
});

describe('B1 — glossy/metal specular indirect term', () => {
  it('preserves the (1-metallic) diffuse GI share for fractional metals', () => {
    const body = fnBody(SHADE_WGSL, 'lo_indirect');
    expect(body).toContain('metal:   f32');
    expect(body).toContain('let diffuseWeight = 1.0 - clamp(metal, 0.0, 1.0);');
    expect(body).toContain(
      'return diffuseWeight * (wRestirGi * Lo_indirect + wRc * Lo_rc);',
    );
    expect(body).not.toContain('if (isGlass || isMetal)');
    expect(SHADE_WGSL).toContain(
      'lo_indirect(pix, dims, pos, normal, isGlass, metal)',
    );
  });

  it('ggxBrdf exposes only the rich evaluators consumed by production passes', () => {
    expect(GGX_BRDF_WGSL).toContain('fn evalGGXSpecularOnlyWithSpecular(');
    expect(GGX_BRDF_WGSL).toContain('fn evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(');
    expect(GGX_BRDF_WGSL).toContain('fn evalGGXWithSpecular(');
    expect(GGX_BRDF_WGSL).toContain('fn evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(');
    expect(GGX_BRDF_WGSL).not.toContain('fn evalGGX(');
    expect(GGX_BRDF_WGSL).not.toContain('fn evalGGXSpecularOnly(');
    expect(GGX_BRDF_WGSL).not.toContain('fn evalGGXWithSpecularAnisotropy(');
    expect(GGX_BRDF_WGSL).not.toContain('fn evalGGXSpecularOnlyWithSpecularAnisotropy(');
    expect(GGX_BRDF_WGSL).toContain('fn evalClearcoatLobe(');
    expect(GGX_BRDF_WGSL).toContain('fn evalSheenLobe(');
    expect(GGX_BRDF_WGSL).toContain('fn charlieD(');
    expect(GGX_BRDF_WGSL).toContain('fn materialF0(');
  });

  it('shade computes lo_indirectSpecular and folds it into the un-demodulated direct channel', () => {
    expect(SHADE_WGSL).toContain('fn lo_indirectSpecular(');
    expect(SHADE_WGSL).toContain('evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(albedo, rough, metal, specular.rgb, specular.a, anisotropy.x, anisotropy.y, iridescence, clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb, anisotropyTangent, anisotropyBitangent, normal, clearcoatNormal, wo, wi)');
    expect(SHADE_WGSL).toContain('let Lo_indirectSpec = lo_indirectSpecular(');
    // It joins directRadiance (NOT the demodulated indirect channel).
    expect(SHADE_WGSL).toMatch(/directRadiance\s*=[\s\S]*?Lo_indirectSpec/);
  });

  it('specular indirect is gated off for default-diffuse surfaces (invariant)', () => {
    // metal <= 0 && rough >= SPEC_GI_ROUGH_MAX && default rich controls -> zero.
    const body = fnBody(SHADE_WGSL, 'lo_indirectSpecular');
    expect(body).toContain('let specularDelta = max(');
    expect(body).toContain('specularDelta <= 0.0');
    expect(body).toContain('abs(anisotropy.x) <= 0.0');
    expect(body).toContain('if (metal <= 0.0 && rough >= SPEC_GI_ROUGH_MAX && specularDelta <= 0.0 && abs(anisotropy.x) <= 0.0 && clearcoat.x <= 0.0 && sheen.a <= 0.0 && iridescence.x <= 0.0)');
  });

  it('consumes valid GRIS samples for glossy and metal receivers', () => {
    const body = fnBody(SHADE_WGSL, 'lo_indirectSpecular');
    expect(body).not.toContain('if (ubo.grisReuse == 1u)');
    expect(body).toContain('let grisVisibility = giReservoirVisibility(g);');
    expect(body).toContain('let toS = giReservoirDirectionVector(g, pos);');
    expect(body).toContain('return g.Lo * specBrdf * g.W * grisVisibility;');
  });

  it('does not reinterpret the post-glass transmission reservoir as primary reflection', () => {
    const consumer = fnBody(SHADE_WGSL, 'lo_indirectSpecular');
    expect(consumer).toContain('if (isGlass) { return vec3f(0.0); }');
    for (const producer of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(producer).toContain('rGlass.xv = walkHitPos;');
      expect(producer).toContain('rGlass.nv = walkHitNormal;');
      expect(producer).toContain('Lo_g = Lo_g * glassPathThroughput;');
    }
  });
});

describe('B1-ior-per-tri — per-triangle IOR lane structural pins', () => {
  it('materialDecode preserves IOR=0 and decodes finite bytes over [1,3]', () => {
    expect(MATERIAL_DECODE_WGSL).toContain('fn decodeIor(packed: u32) -> f32');
    expect(MATERIAL_DECODE_WGSL).toContain('(packed >> 8u) & 0xFFu');
    expect(MATERIAL_DECODE_WGSL).toContain('if (byte == 0u) { return 1e6; }');
    expect(MATERIAL_DECODE_WGSL).toContain('1.0 + f32(byte - 1u) / 254.0 * 2.0');
  });

  it('risGi / risGiNrc glass walks no longer use a hardcoded IOR_GLASS=1.5 constant', () => {
    // The fixed `const IOR_GLASS: f32 = 1.5;` must be gone (replaced by decodeIor()).
    for (const src of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(src).not.toContain('const IOR_GLASS: f32 = 1.5;');
      // Per-tri decode must be present.
      expect(src).toContain('decodeIor(glassPrimaryPacked)');
    }
  });

  it('risGi / risGiNrc glass walks bind bvh_material (group 1, binding 14)', () => {
    for (const src of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(src).toContain('@group(1) @binding(14) var bvh_material: texture_2d<u32>;');
    }
  });

  it('shade lo_transmittedGI no longer uses hardcoded GLASS_F0 = 0.04', () => {
    // The hardcoded `const GLASS_F0: f32 = 0.04;` must be gone.
    expect(SHADE_WGSL).not.toContain('const GLASS_F0: f32 = 0.04;');
    expect(SHADE_WGSL).not.toContain('GLASS_F0 +');
  });

  it('shade lo_transmittedGI does not reapply producer-owned interface Fresnel', () => {
    const body = fnBody(SHADE_WGSL, 'lo_transmittedGI');
    expect(body).not.toContain('decodeIor(');
    expect(body).not.toContain('iorMinus1 / iorPlus1');
    expect(body).toContain('camera-side dielectric throughput is already present in g.Lo');
  });

  it('shade lo_transmittedGI blends half-res GI reservoirs and uses the indirect clamp', () => {
    const body = fnBody(SHADE_WGSL, 'lo_transmittedGI');
    expect(body).toContain('let halfPxF = vec2f(gid) * 0.5;');
    expect(body).toContain('for (var k: u32 = 0u; k < 4u; k = k + 1u)');
    expect(body).toContain('Lo_transmitted = Lo_transmitted + g.Lo * INV_PI * cosTheta * g.W * grisVisibility * bw;');
    expect(body).toContain('Lo_transmitted = Lo_transmitted / totalW;');
    expect(body).toContain('let scaledTransmitted = Lo_transmitted * ubo.glassMixScale;');
    expect(body).toContain('return min(scaledTransmitted, ubo.indirectFireflyClamp * ubo.glassMixScale);');
  });

  it('risGi / risGiNrc support authored rough dielectric transmission', () => {
    for (const src of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(src).toContain('ggxSampleDielectricTransmission(');
      expect(src).toContain('faceLayerRoughness(');
      expect(src).not.toContain('GLASS_GI_MAX_ROUGHNESS');
      expect(src).not.toContain('ROUGH_GLASS_THRESHOLD');
    }
  });
});
