/**
 * Shared shade-prologue fragment — the byte-identical material-decode →
 * dispersion-IOR → emissive-on-hit → front-face / layer-tint → thin-film TMM
 * block that opens the per-bounce body in BOTH the full kernel
 * (`kernel.wgsl.ts`) and the lite kernel (`kernelLite.wgsl.ts`).
 *
 * The only difference between the two tiers in this region is the wording of
 * the explanatory comment above the gated emissive-on-hit add (full carries the
 * full 5-line rationale; lite carries a 1-line pointer back to the full file).
 * That comment is injected as a parameter so each tier's composed string stays
 * byte-identical to the pre-dedup inline copy.
 *
 * @param emissiveComment  the comment lines (already prefixed with `    // …`,
 *   no trailing newline) printed immediately above the
 *   `if (!prevSampleAllowsAreaMis)` emissive gate.
 * @param baseColorTexApply  WGSL injected immediately after `var baseColor`
 *   that modulates it by the sampled baseColor texture (full tier only; empty
 *   for lite, which composes no group-3 texture bindings → byte-identical). The
 *   sample is a no-op multiply (vec4(1)) for materials with no baseColor map, so
 *   even on the full tier a textureless scene stays byte-identical. (P2)
 * @param emissiveOwnershipGuard additional full-tier estimator-ownership terms
 *   appended to the emissive-hit gate; lite leaves this empty.
 *
 * The returned fragment begins with `    let matId = …` and ends with the
 * closing `}` of the thin-film block; callers interpolate it where the inline
 * prologue used to live (between the trace-miss `break;` block and
 * `let throughputAtVertex = throughput;`).
 */
export function composeShadePrologueWgsl(
  emissiveComment: string,
  baseColorTexApply = '',
  emissiveTexApply = '',
  ormTexApply = '',
  normalMapApply = '',
  aoApply = '',
  lightMapApply = '',
  bumpMapApply = '',
  transmissionMapApply = '',
  volumeThicknessMapApply = '',
  extensionLobeTexApply = '',
  clearcoatNormalMapApply = '',
  emissiveOwnershipGuard = '',
): string {
  const materialDecl = 'var';
  return /* wgsl */ `    let matId = hitMaterialId(hit);
    ${materialDecl} mat = decodeMaterial(matId);
    var baseColor = mat.baseColor;${baseColorTexApply}${aoApply}
    var roughness = mat.roughness;
    var emissive = mat.emissive;${emissiveTexApply}${lightMapApply}
    var metallic = mat.metallic;${ormTexApply}
    var transmission = mat.transmission;${transmissionMapApply}${volumeThicknessMapApply}${extensionLobeTexApply}
    if (params.spectralEnabled != 0u) {
      mat.sheenColor = vec3f(spectralRgbFactorAtHero(mat.sheenColor, heroLambda));
      mat.specularColor = vec3f(spectralRgbFactorAtHero(mat.specularColor, heroLambda));
    }
    var ior = mat.ior;
    if (params.spectralEnabled != 0u && mat.dispersionAbbe > 0.0) {
      ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);
    }
    let scatteringCoeff = mat.scatteringCoeff;
    let scatteringAnisotropy = mat.scatteringAnisotropy;
    let scatteringRgb = mat.scatteringRgb;
    let hasSpectralAttenuation = mat.hasSpectralAttenuation;
    let frontLayerTx = mat.frontLayerTx;
    let frontLayerRoughness = mat.frontLayerRoughness;
    let backLayerTx = mat.backLayerTx;
    let backLayerRoughness = mat.backLayerRoughness;
    let thinFilmEnabled = mat.thinFilmEnabled;
    let thinFilmLayerCountU = mat.thinFilmLayerCountU;
    let thinFilmIncidentIor = mat.thinFilmIncidentIor;
    let thinFilmAngleDependent = mat.thinFilmAngleDependent;
    let spectralAvgMu = mat.spectralAvgMu;
    let spectralSampleCount = mat.spectralSampleCount;
    let isTranslucent = mat.isTranslucent;

    let hitPos = ray.origin + ray.direction * hit.dist;
    let isFrontFace = hit.frontFace;
    var normal = select(-hit.normal, hit.normal, isFrontFace);${normalMapApply}${bumpMapApply}
    var clearcoatNormal = normal;${clearcoatNormalMapApply}

${emissiveComment}
    if (!prevSampleAllowsAreaMis && !sppmOwnsCurrentEmission${emissiveOwnershipGuard}) {
      // A3 — emitters in spectral mode: upsample the RGB emission to a spectrum
      // via Jakob-Hanika and evaluate at the hero λ, scaled to preserve the
      // emitter's luminance (flat-spectrum × luminance approximation in the
      // limit, true reflectance-shaped emission otherwise). The throughput is
      // already scalar-spectral (see the baseColor swap below) so this product
      // is a genuine single-wavelength radiance contribution. Radiometric note:
      // we treat the authored RGB emission as a D65-relative tristimulus and
      // reconstruct its hero-λ SPD via the same upsampling used for reflectance;
      // for a neutral (white) emitter this reduces to the luminance scale, the
      // documented flat-spectrum approximation. RGB mode: emissive unchanged.
      //
      // Emission is authored on the material below the clearcoat layer. Apply
      // the same ratified KHR_materials_clearcoat view-normal attenuation used
      // by the lower BSDF lobes after the clearcoat normal has been resolved.
      let emitSpectral = spectralEmissionAtHero(emissive, heroLambda);
      let emitContribution = select(emissive, emitSpectral, params.spectralEnabled != 0u);
      let clearcoatEmissionAttenuation =
        1.0 - clearcoatLayerWeight(mat.clearcoat, clearcoatNormal, -ray.direction);
      radiance = radiance + throughput * emitContribution * clearcoatEmissionAttenuation;
    }

    if (mat.isUnlit) {
      if (!firstHitValid) {
        firstHitValid = true;
        firstHitPos = hitPos;
        firstHitNormal = normal;
        firstHitAlbedo = baseColor;
        firstHitDepth = distance(hitPos, primaryRayOrigin);
      }
      var unlitColor = baseColor;
      if (params.spectralEnabled != 0u) {
        let unlitScalar = spectralCombinedReflectanceAtHero(
          baseColor,
          mat.baseColor,
          mat.spectralReflCoeffs,
          mat.hasSpectralReflectance,
          heroLambda,
        );
        unlitColor = vec3f(unlitScalar);
      }
      radiance = radiance + throughput * unlitColor;
      break;
    }
    let layerTx = clamp(select(backLayerTx, frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));
    let layerRoughness = select(backLayerRoughness, frontLayerRoughness, isFrontFace);
    if (layerRoughness >= 0.0) {
      roughness = clamp(layerRoughness, 0.0, 1.0);
    }
    let layerW = select(
      layerTx,
      activeLayerWeightRgb(layerTx, heroLambda, true),
      params.spectralEnabled != 0u && luminance(layerTx) < 0.999,
    );
    baseColor = baseColor * layerW;
    if (!firstHitValid) {
      firstHitValid = true;
      firstHitPos = hitPos;
      firstHitNormal = normal;
      firstHitAlbedo = baseColor;
      firstHitDepth = distance(hitPos, primaryRayOrigin);
    }
    let wo = -ray.direction;
    let thinFilm = ThinFilmInterface(
      thinFilmEnabled, matId, thinFilmLayerCountU,
      thinFilmIncidentIor, ior, thinFilmAngleDependent,
      isFrontFace, params.spectralEnabled != 0u, heroLambda, transmission,
    );
    // A3 — TRUE spectral transport. In spectral mode replace the RGB albedo with
    // a SCALAR spectral reflectance S(λ) at the hero wavelength (Jakob & Hanika
    // 2019 upsampling, solved per-material at pack time). Broadcasting the scalar
    // to all three channels makes the entire downstream RGB BSDF / NEE / MIS
    // machinery carry a genuine single-wavelength quantity: every product
    // throughput·brdf·… stays scalar (r==g==b), so luminance(radiance) at the
    // end is exactly the spectral radiance the hero-λ CMF reconstruction expects.
    // The per-bounce reflectance is now λ-resolved (a red glass attenuates the
    // 460 nm hero path far more than the 630 nm one) — the dispersion / thin-film
    // / Beer μ(λ) terms (already λ-aware) now modulate a truly spectral
    // throughput. RGB mode (spectralEnabled==0) leaves baseColor untouched →
    // byte-identical. Materials lacking packed coeffs (hasSpectralReflectance==0)
    // fall back to the luminance of the RGB albedo (flat-spectrum approximation).
    if (params.spectralEnabled != 0u) {
      let reflScalar = spectralCombinedReflectanceAtHero(
        baseColor,
        mat.baseColor,
        mat.spectralReflCoeffs,
        mat.hasSpectralReflectance,
        heroLambda,
      );
      baseColor = vec3f(reflScalar);
    }`;
}

/** Full-tier emissive-on-hit rationale (5 lines). */
export const SHADE_PROLOGUE_EMISSIVE_COMMENT_FULL =
  `    // Emissive-on-hit: add the surface's own emission, but ONLY on a path the
    // analytic BSDF↔light connection did NOT already account for (camera ray +
    // post-refraction; see prevSampleAllowsAreaMis). On diffuse/glossy bounces
    // the connection at the PREVIOUS vertex already added this light's
    // contribution with its MIS weight, so adding it again here would double-count.`;

/** Lite-tier emissive-on-hit pointer (1 line). */
export const SHADE_PROLOGUE_EMISSIVE_COMMENT_LITE =
  `    // Gated emissive-on-hit (camera + refraction paths only — see kernel.wgsl.ts).`;

/** Full-tier baseColor texture modulation (P2). Injected after `var baseColor`;
 *  no-op multiply (vec4(1)) for materials without a baseColor map. Leads with a
 *  newline + 4-space indent so it sits on its own line after the declaration. */
export const SHADE_PROLOGUE_BASE_COLOR_TEX_APPLY_FULL =
  `\n    baseColor = baseColor * sampleVertexColor(hit.triIndex, hit.baryVW).rgb * sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex).rgb;`;

/** Full-tier emissive texture modulation (P2). Injected after `var emissive`;
 *  no-op (vec4(1)) for materials without an emissive map → byte-identical. */
export const SHADE_PROLOGUE_EMISSIVE_TEX_APPLY_FULL =
  `\n    emissive = emissive * sampleEmissiveTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex).rgb;`;

/** Full-tier ORM (metallicRoughness) texture modulation (P2). Injected after
 *  `var metallic`; glTF packing G=roughness, B=metallic. vec4(1) when absent →
 *  roughness·1 / metallic·1 (both already clamped) → byte-identical. */
export const SHADE_PROLOGUE_ORM_TEX_APPLY_FULL =
  `\n    let ormSample = sampleOrmTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);` +
  `\n    roughness = clamp(roughness * ormSample.g, 0.0, 1.0);` +
  `\n    metallic = clamp(metallic * ormSample.b, 0.0, 1.0);`;

/** Full-tier normal-map perturbation (P2). Injected after `var normal` (the
 *  front-face shading normal); returns the geometric normal unchanged when there
 *  is no normal map → byte-identical. Perturbs before firstHitNormal is captured
 *  so the G-buffer normal is the mapped one. */
export const SHADE_PROLOGUE_NORMAL_MAP_APPLY_FULL =
  `\n    normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex, isFrontFace);`;

/** D3 — AO map: multiply baseColor by the baked occlusion factor (glTF
 *  occlusionTexture, R channel), lerped by aoMapIntensity. sampleAoFactor returns
 *  1 when no aoMap → byte-identical. Applied right after the baseColor texture
 *  modulation so the G-buffer albedo and all downstream BSDF terms see the
 *  occluded albedo (documented biased semantics — see material.wgsl.ts). */
export const SHADE_PROLOGUE_AO_APPLY_FULL =
  `\n    baseColor = baseColor * sampleAoFactor(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);`;

/** D3 — light map: add the baked OUTGOING radiance to emissive at camera-visible
 *  (primary hit, bounce == 0) ONLY — matching pt-webgl2's pathDepth==0 semantic.
 *  Adding it on post-refraction bounces would double-count the baked light that the
 *  refracted-through surface already contributes at the primary hit; camera-only is
 *  the conservative, double-count-proof choice. 0 when no lightMap → byte-identical.
 *  Adds to the emission var so it never enters NEE. */
export const SHADE_PROLOGUE_LIGHT_MAP_APPLY_FULL =
  `\n    emissive = emissive + select(vec3f(0.0), sampleLightMapRadiance(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), bounce == 0u);`;

/** D3 — bump map: perturb the shading normal by the height-field gradient
 *  (applied AFTER the normal map so the two compose). Returns the normal unchanged
 *  when no bumpMap → byte-identical. */
export const SHADE_PROLOGUE_BUMP_MAP_APPLY_FULL =
  `\n    normal = applyBumpMap(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex);`;

export const SHADE_PROLOGUE_CLEARCOAT_NORMAL_MAP_APPLY_FULL =
  `\n    clearcoatNormal = applyClearcoatNormalMap(matId, hit.triIndex, hit.baryVW, clearcoatNormal, hit.instanceIndex);`;

/** KHR_materials_transmission map: multiply the scalar transmission factor by
 *  the texture's R channel. sampleTransmissionTexture returns 1 when absent,
 *  so scalar-only transmission stays byte-identical. */
export const SHADE_PROLOGUE_TRANSMISSION_MAP_APPLY_FULL =
  `\n    transmission = clamp(transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);`;

/** KHR_materials_volume thicknessTexture: multiply the decoded scalar
 *  thicknessFactor by the texture's G channel. The result is consumed as an
 *  approximate closed-surface Beer-Lambert distance clamp. */
export const SHADE_PROLOGUE_VOLUME_THICKNESS_MAP_APPLY_FULL =
  `\n    let volumeThicknessSample = sampleVolumeThicknessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);` +
  `\n    if (volumeThicknessSample >= 0.0) {` +
  `\n      mat.volumeThickness = max(mat.volumeThickness * volumeThicknessSample, 0.0);` +
  `\n      mat.hasVolumeThickness = true;` +
  `\n    }`;

/** Extension-lobe texture maps: full-tier only. These mutate the decoded material
 *  local so every existing downstream BSDF/PDF/NEE call observes the same lobe
 *  parameters without duplicating call-site arguments. */
export const SHADE_PROLOGUE_EXTENSION_LOBE_TEX_APPLY_FULL =
  `\n    mat.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);` +
  `\n    mat.clearcoatRoughness = clamp(mat.clearcoatRoughness * sampleClearcoatRoughnessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);` +
  `\n    mat.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), vec3f(0.0), vec3f(1.0));` +
  `\n    mat.sheenRoughness = clamp(mat.sheenRoughness * sampleSheenRoughnessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);` +
  `\n    mat.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);` +
  `\n    let iridescenceThicknessSample = sampleIridescenceThicknessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);` +
  `\n    if (iridescenceThicknessSample >= 0.0) {` +
  `\n      let iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, iridescenceThicknessSample);` +
  `\n      mat.iridescenceThicknessMin = iridescenceThickness;` +
  `\n      mat.iridescenceThicknessMax = iridescenceThickness;` +
  `\n      if (iridescenceThickness <= 0.0) { mat.iridescence = 0.0; }` +
  `\n    }` +
  `\n    mat.specularColor = max(mat.specularColor * sampleSpecularColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), vec3f(0.0));` +
  `\n    mat.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);`;
