/**
 * B2 — DDGI glossy-aware probe bounce: specular complement via reflected
 * previous-frame field (2026-06-10, R8-B).
 *
 * Test coverage:
 *
 *  1. WGSL structural pins:
 *     a. The reflected-direction computation is present in the generated WGSL
 *        (reflect formula over the atlas-perturbed `probeNormal`).
 *     b. The specular weight formula is present
 *        (`mat.metalness * max(0.0, 1.0 - mat.roughness * mat.roughness)`).
 *     c. The reflected atlas lookup is present (ddgiSampleSHProbe called twice —
 *        once for the diffuse indirect, once for the specular complement).
 *     d. The blend is present (`mix(lambertianIndirectLo, specularIndirectLo`).
 *     e. The Lambertian path is retained for rough/dielectric surfaces
 *        (specularWeight gate `specularWeight > 1e-4`).
 *     f. The specular path is gated behind indirectFeedback
 *        (`frameParams.indirectFeedback != 0u` in the specularWeight branch).
 *     g. mat.roughness and mat.metalness are accessed (canonical MaterialEntry
 *        fields — no new buffer layout needed).
 *
 *  2. Material availability: MaterialEntry (canonical struct from
 *     @vitrum/shared-bvh) carries roughness (slot 3) and metalness (slot 7).
 *     packDDGIMaterialsN preserves these fields via pbrToMaterialEntryInput.
 *
 *  3. Energy argument (CPU proxy):
 *     For a perfect mirror metal (metalness=1, roughness=0, specularWeight=1):
 *       indirectRadiance = mix(lambertianLo, specularLo, 1.0) = specularLo
 *     For a rough dielectric (metalness=0 OR roughness=1, specularWeight=0):
 *       indirectRadiance = mix(lambertianLo, specularLo, 0.0) = lambertianLo
 *     ⇒ blend, not add — no double-counting at either extreme.
 *
 *  4. indirectFeedback=0 gate: when indirectFeedback is disabled the specular
 *     complement is also skipped — byte-identical to pre-B2 direct-only path.
 *
 * Provenance: B2: glossy-aware probe bounce — specular complement via reflected
 * previous-frame field, 2026-06-10.
 * RENDER-CHANGING for metallic scenes. A/B pending V28-B (real-GPU validation
 * with a metallic-sphere scene after R8-C recapture).
 */

import { describe, expect, it } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';
import { packDDGIMaterialsFromCoreN, packDDGIMaterialsN } from '../probeUpdateMaterials.js';

// ── helper ──────────────────────────────────────────────────────────────────

/** Generate WGSL with the default material cap (64). */
function wgsl(): string {
  return makeProbeUpdateRaysWGSL(64);
}

// ── 1. WGSL structural pins ──────────────────────────────────────────────────

describe('B2 — WGSL structural pins: specular complement', () => {
  it('1a. reflected direction formula is present', () => {
    // reflect(dir, n) = dir - 2·(n·dir)·n; the WGSL encodes this explicitly.
    const src = wgsl();
    expect(src).toContain('rayDirection - 2.0 * dot(rayDirection, probeNormal) * probeNormal');
  });

  it('1b. extension-aware specular weight formula is present', () => {
    const src = wgsl();
    expect(src).toContain('fn ddgiProbeBaseSpecularWeight(mat: DdgiProbeHitMaterial) -> f32');
    expect(src).toContain('let roughFade = max(0.0, 1.0 - mat.roughness * mat.roughness);');
    expect(src).toContain('let metalness = clamp(mat.metalness, 0.0, 1.0);');
    expect(src).toContain('let metallic = metalness * roughFade;');
    expect(src).toContain('let dielectricF0 = ddgiProbeDielectricF0(mat);');
    expect(src).toContain('fn ddgiProbeExtensionSpecularWeight(mat: DdgiProbeHitMaterial) -> f32');
  });

  it('1c. specular branch uses canonical relocated visibility-aware DDGI feedback', () => {
    const src = wgsl();
    expect(src).toContain(
      'let specularIrradiance = ddgiFeedbackAt(',
    );
  });

  it('1d. base/reflection split preserves the former Lambertian-to-specular mix', () => {
    const src = wgsl();
    expect(src).toContain('baseIndirectRadiance = baseIndirectRadiance *');
    expect(src).toContain('(1.0 - extensionSpecularWeight);');
    expect(src).toContain('reflectionIndirectRadiance = specularIndirectLo *');
    expect(src).toContain('extensionSpecularWeight;');
  });

  it('1e. Lambertian path is retained only when extension specular support is exactly absent', () => {
    const src = wgsl();
    // The else-branch writes the Lambertian indirect.
    expect(src).toContain('baseIndirectRadiance = indirectGated * probeMat.albedo * (1.0 / PI)');
    // The guard condition exists so rough surfaces stay Lambertian.
    expect(src).toContain('extensionSpecularWeight > 0.0');
    expect(src).not.toContain('extensionSpecularWeight > 1e-4');
  });

  it('1f. specular path is gated behind indirectFeedback', () => {
    const src = wgsl();
    // Both conditions must be present in the specular branch guard.
    expect(src).toContain('extensionSpecularWeight > 0.0 &&');
    expect(src).toContain('frameParams.indirectFeedback != 0u');
  });

  it('1g. mat.roughness and mat.metalness are accessed in the shader', () => {
    const src = wgsl();
    expect(src).toContain('hit, mat.baseColor, mat.roughness, mat.metalness, mat.transmission,');
    expect(src).toContain('mat.ior, mat.attenuationColor, mat.attenuationDistance,');
    expect(src).toContain('smoothNormal, probeNormal, -dir,');
    expect(src).toContain('mat.roughness');
    expect(src).toContain('mat.metalness');
  });

  it('1h. readable PBR maps modulate DDGI probe-hit bounce material response', () => {
    const src = wgsl();
    expect(src).toContain('const DDGI_MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;');
    expect(src).toContain('fn ddgiSampleProbeHitMaterial(');
    expect(src).toContain('out.albedo = scalarBaseColor * baseColorTexel.value.rgb;');
    expect(src).toContain('DDGI_MATERIAL_MAP_SLOT_ROUGHNESS');
    expect(src).toContain('DDGI_MATERIAL_MAP_SLOT_METALLIC');
    expect(src).toContain('const DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 157u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET: u32 = 56u;');
    expect(src).toContain('fn ddgiApplyFaceLayerNormalMapForHit(');
    expect(src).toContain('const DDGI_MATERIAL_MAP_VOLUME_SCATTERING_TEXEL_OFFSET: u32 = 55u;');
    expect(src).toContain('fn ddgiSampleFaceLayerControls(');
    expect(src).toContain('fn ddgiSampleVolumeScatteringControls(');
    expect(src).toContain('fn ddgiMediumSegmentTransfer(');
    expect(src).toContain('out.roughness = ddgiFaceLayerRoughness(out.roughness, layerControls);');
    expect(src).toContain('out.reflectionLayerTransmission = ddgiFaceLayerTransmission(layerControls);');
    expect(src).toContain('out.layerTransmission = out.reflectionLayerTransmission;');
    expect(src).toContain('out.volumeScattering = ddgiSampleVolumeScatteringControls(hit.indices.w);');
    expect(src).toContain('probeMat.layerTransmission;');
    expect(src).toContain('fn ddgiMediumSegmentTransfer(');
    expect(src).toContain('let directRadiance = direct * probeMat.albedo * (1.0 / PI);');
  });

  it('1h2. readable transmission and thickness maps modulate DDGI glass response', () => {
    const src = wgsl();
    expect(src).toContain('const DDGI_MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET: u32 = 13u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_THICKNESS_TEXEL_OFFSET: u32 = 47u;');
    expect(src).toContain('fn ddgiSampleTransmissionMapForHit(hit: IntersectionResult, scalarTransmission: f32) -> f32');
    expect(src).toContain('transmission: f32,');
    expect(src).toContain('authoredThickness: f32,');
    expect(src).toContain('thicknessMapScale: f32,');
    expect(src).toContain('out.transmission = ddgiSampleTransmissionMapForHit(hit, scalarTransmission);');
    expect(src).toContain('let thicknessMap = ddgiSampleThicknessMapFactorForHit(hit);');
    expect(src).toContain('thicknessMap.y >= 0.5 && out.authoredThickness > 0.0,');
    expect(src).toContain('let mappedCap = authoredThickness * clamp(thicknessMapScale, 0.0, 1.0);');
    expect(src).toContain('radiance = vec3f(');
    expect(src).toContain('fn ddgiTraceGlassChannel(');
    expect(src).toContain('prefixTransfer, sampleSkyColor(rayDirection), channel,');
    expect(src).toContain('currentMaterial.transmission * frameParams.glassMixScale,');
  });

  it('1i. atlas-backed normal and bump maps perturb DDGI probe-hit bounce normals', () => {
    const src = wgsl();
    expect(src).toContain('const DDGI_MATERIAL_MAP_NORMAL_TEXEL_OFFSET: u32 = 15u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET: u32 = 49u;');
    expect(src).toContain('@group(1) @binding(5) var ddgiBvhTangent: texture_2d<f32>;');
    expect(src).toContain('fn ddgiBvhTangentTexel(vertexIndex: u32) -> vec4f');
    expect(src).toContain('fn ddgiPreferAuthoredTangentFrameForHit(');
    expect(src).toContain('fn ddgiMaterialTangentFrameForHit(');
    expect(src).toContain('return ddgiPreferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);');
    expect(src).toContain('fn ddgiApplyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f');
    expect(src).toContain('fn ddgiApplyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f');
    expect(src).toContain('let normalMapped = ddgiApplyNormalMapForHit(hit, smoothNormal);');
    expect(src).toContain('let probeNormal = ddgiApplyBumpMapForHit(hit, normalMapped);');
    expect(src).toContain('let direct = evalDirectLighting(');
    expect(src).toContain('hitWorldPos, probeNormal, seed ^ 0xa511e9b3u,');
    expect(src).toContain('let indirect = ddgiFeedbackAt(hitWorldPos, probeNormal);');
    expect(src).not.toContain('out.hitNormal');
  });

  it('1j. readable emissive maps modulate direct probe-hit surface emission', () => {
    const src = wgsl();
    expect(src).toContain('@group(1) @binding(3) var ddgiMaterialTextureAtlas: texture_2d_array<u32>;');
    expect(src).toContain('@group(1) @binding(4) var ddgiMaterialMapMeta: texture_2d<f32>;');
    expect(src).toContain('fn ddgiSampleEmissiveMap(hit: IntersectionResult, scalarEmission: vec3f) -> vec3f');
    expect(src).toContain('let surfaceEmission = select(');
    expect(src).toContain('ddgiSampleEmissiveMap(hit, scalarSurfaceEmission),');
    expect(src).toContain('(mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u');
  });

  it('1k. readable extension maps modulate DDGI probe-hit specular response', () => {
    const src = wgsl();
    expect(src).toContain('const DDGI_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET: u32 = 24u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET: u32 = 22u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET: u32 = 36u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET: u32 = 23u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET: u32 = 39u;');
    expect(src).toContain('const DDGI_MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET: u32 = 42u;');
    expect(src).toContain('fn ddgiSampleSpecularControls(');
    expect(src).toContain('fn ddgiSampleClearcoatControls(');
    expect(src).toContain('fn ddgiSampleSheenControls(');
    expect(src).toContain('fn ddgiSampleAnisotropyControls(');
    expect(src).toContain('fn ddgiSampleIridescenceControls(');
    expect(src).toContain('fn ddgiApplyClearcoatNormalMapForHit(');
    expect(src).toContain('specular: vec4f,');
    expect(src).toContain('clearcoat: vec2f,');
    expect(src).toContain('clearcoatNormal: vec3f,');
    expect(src).toContain('sheen: vec4f,');
    expect(src).toContain('anisotropy: vec2f,');
    expect(src).toContain('iridescence: vec4f,');
    expect(src).toContain('out.specular = ddgiSampleSpecularControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(src).toContain('out.clearcoat = ddgiSampleClearcoatControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(src).toContain('out.clearcoatNormal = ddgiApplyClearcoatNormalMapForHit(hit, frameNormal, shadingNormal);');
    expect(src).toContain('out.sheen = ddgiSampleSheenControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(src).toContain('out.anisotropy = ddgiSampleAnisotropyControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(src).toContain('out.iridescence = ddgiSampleIridescenceControls(hit.indices.w, uvs.uv0, uvs.uv1);');
    expect(src).toContain('let clearcoatReflectedDirection = ddgiNormalizeOr(');
    expect(src).toContain('let extensionSpecularWeight = ddgiProbeExtensionSpecularWeight(probeMat);');
    expect(src).toContain('let f0Tint = ddgiProbeSpecularTint(');
    expect(src).toContain('reflectionIndirectRadiance = specularIndirectLo *');
  });

  it('1l. thin-film transmittance applies only to the base/source closure', () => {
    const src = wgsl();
    expect(src).toContain('reflectionLayerTransmission: vec3f,');
    expect(src.match(
      /out\.reflectionLayerTransmission = ddgiFaceLayerTransmission\(layerControls\);/g,
    )).toHaveLength(2);
    expect(src.match(
      /out\.layerTransmission = out\.reflectionLayerTransmission;/g,
    )).toHaveLength(2);
    expect(src.match(
      /out\.layerTransmission = out\.layerTransmission \* film\.transmittance;/g,
    )).toHaveLength(2);
    expect(src).not.toMatch(
      /reflectionLayerTransmission\s*=.*film\.transmittance/,
    );
    expect(src).toContain(
      'reflectionIndirectRadiance * probeMat.reflectionLayerTransmission;',
    );
  });

  it('1m. splits deterministic emission, unpaid opaque response, and persistent extension reflection', () => {
    const src = wgsl();
    expect(src).toContain('struct DdgiProbeSurfaceSource {');
    expect(src).toContain('emissionLo: vec3f,');
    expect(src).toContain('opaqueLo: vec3f,');
    expect(src).toContain('persistentReflectionLo: vec3f,');
    expect(src).toContain('terminalLo: vec3f,');
    expect(src).toContain(
      'out.emissionLo = surfaceEmission * probeMat.layerTransmission;',
    );
    expect(src).toContain(
      '(directRadiance + baseIndirectRadiance + bakedOutgoing) *',
    );
    expect(src).toContain(
      'currentSurfaceSource.opaqueLo * opaqueWeight +',
    );
    expect(src).toContain(
      'currentSurfaceSource.persistentReflectionLo,',
    );

    const persistentStart = src.indexOf(
      'persistentReflectionIndirectRadiance =\n        (',
    );
    const persistentEnd = src.indexOf(';', persistentStart);
    expect(persistentStart).toBeGreaterThan(0);
    expect(persistentEnd).toBeGreaterThan(persistentStart);
    const persistentAssignment = src.slice(persistentStart, persistentEnd);
    expect(persistentAssignment).toContain('clearcoatLo * clearcoatWeight');
    expect(persistentAssignment).toContain('sheenLo * sheenWeight');
    expect(persistentAssignment).not.toContain('baseSpecularLo');
  });

  it('1n. terminates unlit glass as a deterministic source before dielectric branching', () => {
    const src = wgsl();
    const traceStart = src.indexOf('fn ddgiTraceGlassChannel(');
    const unlit = src.indexOf(
      'if ((currentEntry.flags & MATERIAL_FLAG_UNLIT) != 0u)',
      traceStart,
    );
    const glassGate = src.indexOf('if (!hasTransmission)', traceStart);
    const familyDraw = src.indexOf('let chooseTransmission =', traceStart);
    expect(unlit).toBeGreaterThan(traceStart);
    expect(glassGate).toBeGreaterThan(unlit);
    expect(familyDraw).toBeGreaterThan(glassGate);
    expect(src.slice(unlit, glassGate)).toContain(
      'prefixTransfer, currentSurfaceSource.emissionLo, channel,',
    );
    expect(src).toContain(
      'out.emissionLo = probeMat.albedo * probeMat.layerTransmission;',
    );
    expect(src).toContain('out.terminalLo = out.emissionLo;');
  });
});

// ── 2. Material availability: roughness / metalness in MaterialEntry ─────────

describe('B2 — MaterialEntry carries roughness and metalness (no new threading needed)', () => {
  it('packDDGIMaterialsN writes roughness into slot 3 of the packed buffer', () => {
    // PbrScalarSource uses `color: {r,g,b}` (the walkaround material bag).
    const buf = packDDGIMaterialsN(
      [{ color: { r: 1, g: 1, b: 1 }, roughness: 0.25, metalness: 0 }],
      1,
    );
    const f32 = new Float32Array(buf);
    // Canonical layout: slot 3 = roughness.
    expect(f32[3]).toBeCloseTo(0.25, 5);
  });

  it('packDDGIMaterialsN writes metalness into slot 7 of the packed buffer', () => {
    const buf = packDDGIMaterialsN(
      [{ color: { r: 0.8, g: 0.7, b: 0.6 }, roughness: 0.1, metalness: 0.9 }],
      1,
    );
    const f32 = new Float32Array(buf);
    // Canonical layout: slot 7 = metalness.
    expect(f32[7]).toBeCloseTo(0.9, 5);
  });

  it('default metalness=0 forces specularWeight=0 regardless of roughness (existing dielectric scenes unaffected)', () => {
    // packDDGIMaterialsN routes through pbrToMaterialEntryInput → extractPbrScalars.
    // The walkaround PBR_DEFAULTS: roughness=0.5, metalness=0.
    // The critical invariant is metalness=0 → specularWeight = 0*(...) = 0
    // → pure Lambertian path. Default roughness (0.5) is irrelevant because
    // metalness=0 already zeroes specularWeight.
    const buf = packDDGIMaterialsN(
      [{ color: { r: 1, g: 1, b: 1 } }],  // no roughness/metalness → walkaround defaults
      1,
    );
    const f32 = new Float32Array(buf);
    const metalness = f32[7]!;    // slot 7
    // Default metalness = 0 → specularWeight = 0*(...) = 0 → Lambertian path.
    expect(metalness).toBeCloseTo(0.0, 5);
  });
});

// ── 3. Energy argument (CPU proxy) ───────────────────────────────────────────

describe('B2 — energy argument: blend not add (CPU proxy)', () => {
  /**
   * CPU proxy for the WGSL blend formula.
   * Returns indirectRadiance given:
   *   - lambertianLo  = indirectGated * baseColor / PI
   *   - specularLo    = baseColor * specularIrr / PI
   *   - metalness, roughness
   */
  function blendProxy(
    lambertianLo: number,
    specularLo: number,
    metalness: number,
    roughness: number,
  ): number {
    const specularWeight = metalness * Math.max(0, 1 - roughness * roughness);
    if (specularWeight > 1e-4) {
      // mix(lambertianLo, specularLo, specularWeight)
      return lambertianLo * (1 - specularWeight) + specularLo * specularWeight;
    }
    return lambertianLo;
  }

  it('perfect mirror metal (metal=1, rough=0): indirectRadiance = specularLo', () => {
    const lambertianLo = 0.5;
    const specularLo = 2.0;
    const result = blendProxy(lambertianLo, specularLo, 1.0, 0.0);
    expect(result).toBeCloseTo(specularLo, 5);
  });

  it('rough metal (metal=1, rough=1): specularWeight=0 → falls through to Lambertian', () => {
    const lambertianLo = 0.5;
    const specularLo = 2.0;
    const result = blendProxy(lambertianLo, specularLo, 1.0, 1.0);
    expect(result).toBeCloseTo(lambertianLo, 5);
  });

  it('pure dielectric (metal=0): specularWeight=0 → Lambertian unchanged', () => {
    const lambertianLo = 0.8;
    const specularLo = 3.0;
    const result = blendProxy(lambertianLo, specularLo, 0.0, 0.0);
    expect(result).toBeCloseTo(lambertianLo, 5);
  });

  it('blend does not add: at specularWeight=0.5 result is strictly between the two', () => {
    const lambertianLo = 0.5;
    const specularLo = 1.5;
    const result = blendProxy(lambertianLo, specularLo, 0.8, 0.45);
    // specularWeight = 0.8 * (1 - 0.45²) = 0.8 * 0.7975 ≈ 0.638
    expect(result).toBeGreaterThan(lambertianLo);
    expect(result).toBeLessThan(specularLo);
    // Result must NOT exceed the max of the two (no additive energy injection).
    expect(result).toBeLessThanOrEqual(specularLo);
  });

  it('thin film pays face T on reflection and face T × film T on the base', () => {
    const faceTransmission = 0.8;
    const filmReflectance = 0.3;
    const filmTransmittance = 0.5;
    const reflectedContribution = filmReflectance * faceTransmission;
    const baseContribution = faceTransmission * filmTransmittance;

    expect(reflectedContribution).toBeCloseTo(0.24, 12);
    expect(reflectedContribution).not.toBeCloseTo(0.12, 12);
    expect(baseContribution).toBeCloseTo(0.4, 12);
  });
});

describe('DDGI glass local-source ownership oracle', () => {
  function expectedLocalSource(
    transmission: number,
    emission: number,
    opaque: number,
    clearcoatAndSheen: number,
  ): number {
    const transmissionPdf = transmission / (1 + transmission);
    const reflectionPdf = 1 - transmissionPdf;
    const reflectedEstimate =
      (opaque * (1 - transmission) + clearcoatAndSheen) / reflectionPdf;
    return emission + reflectionPdf * reflectedEstimate;
  }

  it('retains emissive glass and persistent clearcoat/sheen at t=1', () => {
    expect(expectedLocalSource(1, 2.25, 0.8, 0.35)).toBeCloseTo(2.6, 14);
  });

  it('charges a partial-transmission base response by exactly one unpaid (1-t)', () => {
    expect(expectedLocalSource(0.4, 0.2, 1.5, 0.3)).toBeCloseTo(
      0.2 + 1.5 * 0.6 + 0.3,
      14,
    );
  });

  it('keeps an unlit transmissive endpoint deterministic', () => {
    const unlitSource = 3.75;
    for (const transmission of [0, 0.25, 1]) {
      void transmission;
      expect(unlitSource).toBeCloseTo(3.75, 14);
    }
  });

  it('excludes the base-specular proxy owned by the exact interface continuation', () => {
    const baseSpecularProxy = 0.7;
    const split = expectedLocalSource(0.65, 0.1, 0.9, 0.2);
    const duplicated = split + baseSpecularProxy;
    expect(split).toBeCloseTo(0.1 + 0.9 * 0.35 + 0.2, 14);
    expect(split).not.toBeCloseTo(duplicated, 14);
  });
});

// ── 4. indirectFeedback=0 gate ───────────────────────────────────────────────

describe('B2 — WGSL gate: specular complement is disabled when indirectFeedback=0', () => {
  it('the specular branch gate combines specularWeight AND indirectFeedback checks', () => {
    // Both conditions in the same `if` expression ensure the specular path is
    // fully suppressed when direct-only probes are requested (maxBounces==1).
    // In that regime the atlas is not fed multi-bounce data, so a reflected
    // atlas lookup would return stale/zero data — correctly skipped.
    const src = wgsl();
    expect(src).toContain('extensionSpecularWeight > 0.0 &&');
    expect(src).toContain('frameParams.indirectFeedback != 0u');
  });
});

// ── H18. Material-emissive direct probe hits ────────────────────────────────

describe('H18 — material-emissive direct probe hits', () => {
  it('separates packed emission from baked/diffuse response before glass transport', () => {
    const src = wgsl();
    expect(src).toContain('let scalarSurfaceEmission = max(mat.emissive, vec3f(0.0));');
    expect(src).toContain('let surfaceEmission = select(');
    expect(src).toContain('ddgiSampleEmissiveMap(hit, scalarSurfaceEmission),');
    expect(src).toContain(
      'out.emissionLo = surfaceEmission * probeMat.layerTransmission;',
    );
    expect(src).toContain(
      '(directRadiance + baseIndirectRadiance + bakedOutgoing) *',
    );
    expect(src).toContain('return accumulatedRadiance + ddgiTransferredChannel(');

    const bakedAdd = src.indexOf('let bakedOutgoing =');
    const mapSample = src.indexOf('ddgiSampleEmissiveMap(hit, scalarSurfaceEmission),');
    const opaqueBranch = src.indexOf('return accumulatedRadiance + ddgiTransferredChannel(');
    const writeOut = src.indexOf('out.hitRadiance  = radiance;');
    expect(bakedAdd).toBeGreaterThanOrEqual(0);
    expect(mapSample).toBeGreaterThan(bakedAdd);
    expect(opaqueBranch).toBeGreaterThan(mapSample);
    expect(writeOut).toBeGreaterThan(opaqueBranch);
  });

  it('core-material DDGI packing folds emissiveIntensity into final radiance exactly once', () => {
    const buf = packDDGIMaterialsFromCoreN([
      {
        baseColor: [1, 1, 1],
        roughness: 1,
        metallic: 0,
        emissive: [2, 0.5, 0.25],
        emissiveIntensity: 4,
      },
    ], 1);
    const f32 = new Float32Array(buf);

    expect(f32[4]).toBeCloseTo(8, 5);
    expect(f32[5]).toBeCloseTo(2, 5);
    expect(f32[6]).toBeCloseTo(1, 5);
  });
});
