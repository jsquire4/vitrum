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
 *
 * The returned fragment begins with `    let matId = …` and ends with the
 * closing `}` of the thin-film block; callers interpolate it where the inline
 * prologue used to live (between the trace-miss `break;` block and
 * `let throughputAtVertex = throughput;`).
 */
export function composeShadePrologueWgsl(emissiveComment: string, baseColorTexApply = ''): string {
  return /* wgsl */ `    let matId = hitMaterialId(hit);
    let mat = decodeMaterial(matId);
    var baseColor = mat.baseColor;${baseColorTexApply}
    var roughness = mat.roughness;
    let emissive = mat.emissive;
    let metallic = mat.metallic;
    let transmission = mat.transmission;
    var ior = mat.ior;
    if (params.spectralEnabled != 0u && mat.dispersionAbbe >= 1.0) {
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

${emissiveComment}
    if (!prevSampleAllowsAreaMis) {
      radiance = radiance + throughput * emissive;
    }

    let hitPos = ray.origin + ray.direction * hit.dist;
    let isFrontFace = dot(hit.normal, ray.direction) < 0.0;
    let normal = select(-hit.normal, hit.normal, isFrontFace);
    let layerTx = clamp(select(backLayerTx, frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));
    let layerRoughness = select(backLayerRoughness, frontLayerRoughness, isFrontFace);
    if (layerRoughness >= 0.0) {
      roughness = clamp(layerRoughness, 0.02, 1.0);
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
      firstHitDepth = hit.dist;
    }
    let wo = -ray.direction;
    var thinFilmReflectTint = vec3f(1.0);
    var thinFilmTransmitTint = vec3f(1.0);
    if (thinFilmEnabled) {
      let viewCos = clamp(dot(normal, wo), 0.0, 1.0);
      if (params.spectralEnabled != 0u) {
        let rt = thinFilmTmmRt(
          matId,
          thinFilmLayerCountU,
          heroLambda,
          ior,
          thinFilmIncidentIor,
          thinFilmAngleDependent,
          viewCos,
        );
        thinFilmReflectTint = vec3f(clamp(rt.x, 0.0, 1.0));
        thinFilmTransmitTint = vec3f(clamp(rt.y, 0.0, 1.0));
      } else {
        let rtR = thinFilmTmmRt(matId, thinFilmLayerCountU, 630.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
        let rtG = thinFilmTmmRt(matId, thinFilmLayerCountU, 540.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
        let rtB = thinFilmTmmRt(matId, thinFilmLayerCountU, 460.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
        thinFilmReflectTint = clamp(vec3f(rtR.x, rtG.x, rtB.x), vec3f(0.0), vec3f(1.0));
        thinFilmTransmitTint = clamp(vec3f(rtR.y, rtG.y, rtB.y), vec3f(0.0), vec3f(1.0));
      }
      let layerStrength = clamp(0.12 + 0.06 * f32(thinFilmLayerCountU), 0.0, 0.55);
      let filmStrength = clamp(layerStrength * (1.0 - roughness), 0.0, 0.6);
      baseColor = mix(baseColor, baseColor * thinFilmReflectTint, filmStrength);
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
  `\n    baseColor = baseColor * sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW).rgb;`;
