// materialsTexture — THREE-free port of three-gpu-pathtracer's `MaterialsTexture.js`,
// driven from a `@vitrum/core` `MaterialSpec` (NOT a THREE material).
//
// Reproduces the EXACT 85-pixel (340-float) RGBA32F-per-material byte layout the
// fork's GLSL `material_struct` decoder reads texel-for-texel (verified against
// `uniforms/MaterialsTexture.js` + `shader/structs/material_struct.glsl.js`, and
// the §4 layout table in `plan/three-removal/03-scene-bvh-packers.md`).
//
// The GLSL decoder is KEPT UNCHANGED, so this packer is the byte-exact authority:
// any divergence here is a render bug. The packer is intentionally GPU-free and
// pure (Float32Array in → Float32Array out) so it can be golden-tested CPU-side
// like pt-webgpu's `materialPackingCoreEquivalence`.
//
// Provenance: gkjohnson/three-gpu-pathtracer (MIT) — `MaterialsTexture.js`.
// CREDITS.md attributes the absorbed fork.

import type { MaterialSpec, Vec3 } from '@vitrum/core';
import type { MaterialsTextureData } from './sceneTextures.js';

import { MATERIAL_PIXELS } from '../glsl/shader/structs/materialStride.js';

/** Pixels (RGBA32F texels) per material — single-sourced with every GLSL fetch
 *  site via `materialStride.js` (fork base layout 85 + D3 ao/light/bump/env
 *  texels 85..92 = 93). Re-exported for tests and parity guards. */
export { MATERIAL_PIXELS };
/** Floats per material (93 px × 4 channels). */
const MATERIAL_STRIDE = MATERIAL_PIXELS * 4;

/** TRANSLUCENT_BIT — flag (s14.a) bit set for intrinsically scattering media. */
const TRANSLUCENT_BIT = 1 << 4;

// Dispersion: Abbe → strength, evaluated at the Fraunhofer C/F lines.
const FRAUNHOFER_C_NM = 656.3;
const FRAUNHOFER_F_NM = 486.1;

// Uniform-grid spectral attenuation μ(λ): 32 samples, 380→780 nm inclusive.
const SPECTRAL_GRID_SAMPLE_COUNT = 32;
const SPECTRAL_GRID_START_NM = 380.0;
const SPECTRAL_GRID_END_NM = 780.0;

// Thin-film stack: 35 layers × [ior, thicknessNm, extinction].
const THIN_FILM_LAYER_LIMIT = 35;

/** ceil(sqrt(n)) — the square dimension that holds `n` texels row-major (mirrors fork). */
function squareDim(texelCount: number): number {
  return Math.max(1, Math.ceil(Math.sqrt(Math.max(1, texelCount))));
}

/**
 * Dispersion strength from the Abbe number V_d and IOR — exact port of the fork's
 * `dispersionStrengthFromAbbe`. 0 when abbe<=0 || ior<=1 (no dispersion).
 */
function dispersionStrengthFromAbbe(ior: number, abbe: number): number {
  if (abbe <= 0 || ior <= 1) return 0;
  const denom =
    1 / (FRAUNHOFER_F_NM * FRAUNHOFER_F_NM) - 1 / (FRAUNHOFER_C_NM * FRAUNHOFER_C_NM);
  if (Math.abs(denom) < 1e-12) return 0;
  return Math.max(0, (ior - 1) / (abbe * denom));
}

/**
 * Sample a `@vitrum/core` `SpectralCurve` at `lambdaNm` with linear interpolation —
 * port of the fork's `sampleSpectralCurve`, reading the core curve shape
 * (`wavelengthStart` / `wavelengthEnd` / `values`).
 */
function sampleSpectralCurve(curve: MaterialSpec['spectralAttenuation'], lambdaNm: number): number {
  if (!curve) return 0.0;
  const values = curve.values;
  if (!values || values.length < 2) return 0.0;
  const lambdaStart = Number.isFinite(curve.wavelengthStart) ? curve.wavelengthStart : 380.0;
  const lambdaEnd = Number.isFinite(curve.wavelengthEnd) ? curve.wavelengthEnd : 780.0;
  const denom = Math.max(lambdaEnd - lambdaStart, 1e-6);
  const t = Math.min(1.0, Math.max(0.0, (lambdaNm - lambdaStart) / denom));
  const f = t * (values.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(i0 + 1, values.length - 1);
  const a = Number(values[i0] ?? 0.0);
  const b = Number(values[i1] ?? a);
  return a + (b - a) * (f - i0);
}

/**
 * Texture id as a PLAIN FLOAT (-1 = none). The fork stores texture indices as
 * plain floats — `floatBitsToInt` is broken on some devices (Pixel 6); the GLSL
 * decodes via `int(round(...))`. The core scene-binding layer would resolve a
 * `TextureRef` to an atlas index; until that atlas exists in pt-webgl2 we have no
 * id to assign, so every texture id is -1 (none). When the atlas lands this is
 * the one hook that changes (and the GLSL stays byte-compatible).
 */
const NO_TEXTURE = -1;

/** Default attenuation color when `attenuationColor` is absent — fork default (1,1,1). */
const DEFAULT_ATTENUATION_COLOR: Vec3 = [1.0, 1.0, 1.0];
/** Default specular color — fork default (1,1,1). */
const DEFAULT_SPECULAR_COLOR: Vec3 = [1.0, 1.0, 1.0];

/**
 * Pack a list of core `MaterialSpec`s into the 85-px/material RGBA32F square the
 * fork's GLSL `readMaterialInfo` reads. Returns a CPU `MaterialsTextureData` grid
 * (`{ data, dim, kind:'rgba32f', materialCount }`) ready for `gl.texImage2D`.
 *
 * Square sizing: `dim = ceil(sqrt(materials.length * 85))` (matches the fork).
 */
/** Resolve a TextureRef → its atlas layer index (-1 = none / unmapped). */
function mapLayer(
  ref: { handle?: unknown } | undefined,
  layerOf: Map<unknown, number> | undefined,
): number {
  if (ref?.handle == null || layerOf == null) return -1;
  return layerOf.get(ref.handle) ?? -1;
}

/**
 * Write the 2-texel UV-transform encoding the GLSL `readTextureTransform` reads
 * (row1 = (m00,m01,m02) at material texel `texelIdx`; row2 = (m10,m11,m12) at
 * `texelIdx+1`), reproducing THREE's `Matrix3.setUvTransform` (center 0):
 *   row1 = (sx·cos, sx·sin, offsetX),  row2 = (−sy·sin, sy·cos, offsetY).
 * Identity (no transform) → row1=(1,0,0), row2=(0,1,0).
 */
function writeTransform(
  data: Float32Array,
  base: number,
  texelIdx: number,
  ref: { transform?: { offset?: readonly number[]; scale?: readonly number[]; rotation?: number } } | undefined,
): void {
  const t = ref?.transform;
  const sx = t?.scale?.[0] ?? 1;
  const sy = t?.scale?.[1] ?? 1;
  const ox = t?.offset?.[0] ?? 0;
  const oy = t?.offset?.[1] ?? 0;
  const r = t?.rotation ?? 0;
  const c = Math.cos(r);
  const s = Math.sin(r);
  const o = base + texelIdx * 4;
  data[o] = sx * c; data[o + 1] = sx * s; data[o + 2] = ox; data[o + 3] = 0;
  data[o + 4] = -sy * s; data[o + 5] = sy * c; data[o + 6] = oy; data[o + 7] = 0;
}

export function packMaterialsTexture(
  materials: readonly MaterialSpec[],
  layerOf?: Map<unknown, number>,
): MaterialsTextureData {
  const materialCount = materials.length;
  const pixelCount = materialCount * MATERIAL_PIXELS;
  const dim = squareDim(pixelCount);
  const data = new Float32Array(dim * dim * 4);

  let index = 0;
  for (let i = 0; i < materialCount; i += 1) {
    const m = materials[i]!;
    const base = index; // first float of this material's block

    // ── Scalar field resolution (core → fork semantics) ──────────────────────
    const color = m.baseColor;
    const metalness = m.metallic ?? 0.0;
    const roughness = m.roughness ?? 0.0;
    const ior = m.ior ?? 1.5; // THREE's default f0=0.04 ⇒ ior 1.5
    const transmission = m.transmission ?? 0.0;
    // Contract default: pt-webgpu (materialTextures.ts) and walkaround-hybrid both
    // default emissiveIntensity to 1.0 when the field is absent; 0.0 would silently
    // black-out any emissive material whose host did not explicitly set the field.
    const emissiveIntensity = m.emissiveIntensity ?? 1.0;
    const emissive: Vec3 = m.emissive ?? [0.0, 0.0, 0.0];
    const normalScale = m.normalScale ?? 1.0; // core carries a scalar; fork stores (x,y)
    const clearcoat = m.clearcoat ?? 0.0;
    const clearcoatRoughness = m.clearcoatRoughness ?? 0.0;
    const clearcoatNormalScale = m.clearcoatNormalScale ?? 1.0;
    const sheen = m.sheen ?? 0.0;
    const sheenColor: Vec3 = m.sheenColor ?? [0.0, 0.0, 0.0];
    const sheenRoughness = m.sheenRoughness ?? 0.0;
    const iridescence = m.iridescence ?? 0.0;
    const iridescenceIor = m.iridescenceIor ?? 1.3;
    const iridThicknessRange = m.iridescenceThicknessRange ?? [100, 400];
    const specularColor: Vec3 = m.specularColor ?? DEFAULT_SPECULAR_COLOR;
    const specularIntensity = m.specularIntensity ?? 1.0;
    const attenuationColor: Vec3 = m.attenuationColor ?? DEFAULT_ATTENUATION_COLOR;
    const attenuationDistance = m.attenuationDistance ?? Infinity;
    const thickness = m.thickness ?? 0.0;
    const opacity = m.opacity ?? 1.0;
    const alphaTest = m.alphaMode === 'mask' ? (m.alphaCutoff ?? 0.5) : 0.0;
    const transparent = m.alphaMode === 'blend';

    // isThinFilm — fork: thickness===0 && attenuationDistance===Infinity.
    const isThinFilm = thickness === 0.0 && attenuationDistance === Infinity;

    // Atlas layer indices for the maps the GLSL samples (-1 = none).
    const baseColorLayer = mapLayer(m.baseColorMap, layerOf);
    const metalLayer = mapLayer(m.metallicMap, layerOf);
    const roughLayer = mapLayer(m.roughnessMap, layerOf);
    const transmissionLayer = mapLayer(m.transmissionMap, layerOf);
    const emissiveLayer = mapLayer(m.emissiveMap, layerOf);
    const normalLayer = mapLayer(m.normalMap, layerOf);
    const alphaLayer = mapLayer(m.alphaMap, layerOf);
    // D3 — clearcoat / sheen / iridescence / specular maps (GLSL already samples).
    const clearcoatLayer = mapLayer(m.clearcoatMap, layerOf);
    const clearcoatRoughnessLayer = mapLayer(m.clearcoatRoughnessMap, layerOf);
    const clearcoatNormalLayer = mapLayer(m.clearcoatNormalMap, layerOf);
    const sheenColorLayer = mapLayer(m.sheenColorMap, layerOf);
    const sheenRoughnessLayer = mapLayer(m.sheenRoughnessMap, layerOf);
    const iridescenceLayer = mapLayer(m.iridescenceMap, layerOf);
    const iridescenceThicknessLayer = mapLayer(m.iridescenceThicknessMap, layerOf);
    const specularColorLayer = mapLayer(m.specularColorMap, layerOf);
    const specularIntensityLayer = mapLayer(m.specularIntensityMap, layerOf);
    // D3 — aoMap / lightMap / bumpMap (new GLSL consumption sites).
    const aoLayer = mapLayer(m.aoMap, layerOf);
    const lightMapLayer = mapLayer(m.lightMap, layerOf);
    const bumpLayer = mapLayer(m.bumpMap, layerOf);

    // sample 0 — color.rgb / map
    data[index++] = color[0];
    data[index++] = color[1];
    data[index++] = color[2];
    data[index++] = baseColorLayer;

    // sample 1 — metalness / metalnessMap / roughness / roughnessMap
    data[index++] = metalness;
    data[index++] = metalLayer;
    data[index++] = roughness;
    data[index++] = roughLayer;

    // sample 2 — ior / transmission / transmissionMap / emissiveIntensity
    data[index++] = ior;
    data[index++] = transmission;
    data[index++] = transmissionLayer;
    data[index++] = emissiveIntensity;

    // sample 3 — emissive.rgb / emissiveMap
    data[index++] = emissive[0];
    data[index++] = emissive[1];
    data[index++] = emissive[2];
    data[index++] = emissiveLayer;

    // sample 4 — normalMap / normalScale.xy / clearcoat
    data[index++] = normalLayer;
    data[index++] = normalScale;
    data[index++] = normalScale;
    data[index++] = clearcoat;

    // sample 5 — clearcoatMap / clearcoatRoughness / clearcoatRoughnessMap / clearcoatNormalMap
    data[index++] = clearcoatLayer;
    data[index++] = clearcoatRoughness;
    data[index++] = clearcoatRoughnessLayer;
    data[index++] = clearcoatNormalLayer;

    // sample 6 — clearcoatNormalScale.xy / pad / sheen
    data[index++] = clearcoatNormalScale;
    data[index++] = clearcoatNormalScale;
    index++; // pad (fork does `index ++` here)
    data[index++] = sheen;

    // sample 7 — sheenColor.rgb / sheenColorMap
    data[index++] = sheenColor[0];
    data[index++] = sheenColor[1];
    data[index++] = sheenColor[2];
    data[index++] = sheenColorLayer;

    // sample 8 — sheenRoughness / sheenRoughnessMap / iridescenceMap / iridescenceThicknessMap
    data[index++] = sheenRoughness;
    data[index++] = sheenRoughnessLayer;
    data[index++] = iridescenceLayer;
    data[index++] = iridescenceThicknessLayer;

    // sample 9 — iridescence / iridescenceIOR / iridThicknessRange.xy
    data[index++] = iridescence;
    data[index++] = iridescenceIor;
    data[index++] = iridThicknessRange[0];
    data[index++] = iridThicknessRange[1];

    // sample 10 — specularColor.rgb / specularColorMap
    data[index++] = specularColor[0];
    data[index++] = specularColor[1];
    data[index++] = specularColor[2];
    data[index++] = specularColorLayer;

    // sample 11 — specularIntensity / specularIntensityMap / isThinFilm / —
    data[index++] = specularIntensity;
    data[index++] = specularIntensityLayer;
    data[index++] = Number(isThinFilm);
    index++; // pad (fork does `index ++`)

    // sample 12 — attenuationColor.rgb / attenuationDistance
    data[index++] = attenuationColor[0];
    data[index++] = attenuationColor[1];
    data[index++] = attenuationColor[2];
    data[index++] = attenuationDistance;

    // sample 13 — alphaMap / opacity / alphaTest / side
    data[index++] = alphaLayer;
    data[index++] = opacity;
    data[index++] = alphaTest;
    // side: 0 when (!isThinFilm && transmission>0); else FrontSide=1 (core has no
    // BackSide/DoubleSide concept — meshes are single-sided front by convention,
    // and the transmission rule overrides to 0/double-sided for glass).
    if (!isThinFilm && transmission > 0.0) {
      data[index++] = 0;
    } else {
      data[index++] = 1; // FrontSide
    }

    // sample 14 — matte / castShadow / vertexColors|(flat<<1) / flags
    data[index++] = 0; // matte (core has no matte field)
    data[index++] = 1; // castShadow (fork default true)
    data[index++] = 0; // vertexColors | (flatShading<<1) — core meshes carry neither here
    {
      let flags = Number(transparent);
      const scatteringCoeff = m.scatteringCoefficient ?? 0.0;
      if (scatteringCoeff > 0.0) flags |= TRANSLUCENT_BIT;
      data[index++] = flags;
    }

    // sample 15 — sssSigmaT / sssAnisotropyG / dispersionStrength / thinFilmEnabled
    const scatteringCoeff = m.scatteringCoefficient ?? 0.0;
    const scatteringAnisotropy = m.scatteringAnisotropy ?? 0.0;
    const dispersionAbbe = m.dispersionAbbeNumber ?? 0.0;
    const dispersionStrength = dispersionStrengthFromAbbe(ior, dispersionAbbe);
    const thinFilmLayers = m.thinFilmStack?.layers ?? [];
    const thinFilmLayerCount = Math.min(thinFilmLayers.length, THIN_FILM_LAYER_LIMIT);
    const thinFilmEnabled = thinFilmLayerCount > 0 ? 1.0 : 0.0;
    data[index++] = scatteringCoeff;
    data[index++] = scatteringAnisotropy;
    data[index++] = dispersionStrength;
    data[index++] = thinFilmEnabled;

    // sample 16 — sssAlbedo.rgb / thinFilmLayerCount
    const scatterAlbedo = m.scatteringCoefficientRGB;
    if (scatterAlbedo) {
      data[index++] = scatterAlbedo[0];
      data[index++] = scatterAlbedo[1];
      data[index++] = scatterAlbedo[2];
    } else {
      data[index++] = 0.9;
      data[index++] = 0.9;
      data[index++] = 0.9;
    }
    data[index++] = thinFilmLayerCount;

    // sample 17 — thinFilmIncidentIor / angleDependent / 0 / packedFeatureFlags
    const spectralCurve = m.spectralAttenuation ?? null;
    const frontLayer = m.frontLayer ?? null;
    const backLayer = m.backLayer ?? null;
    const hasSpectral = spectralCurve != null;
    const hasFrontLayer = frontLayer != null;
    const hasBackLayer = backLayer != null;
    const thinFilmIncidentIor = m.thinFilmStack?.incidentIor ?? 1.0;
    const thinFilmAngleDependent = m.thinFilmStack?.angleDependent ?? false;
    const packedFeatureFlags =
      (hasSpectral ? 1 : 0) | (hasFrontLayer ? 2 : 0) | (hasBackLayer ? 4 : 0);
    data[index++] = thinFilmIncidentIor;
    data[index++] = thinFilmAngleDependent ? 1.0 : 0.0;
    data[index++] = 0.0;
    data[index++] = packedFeatureFlags;

    // sample 18 — frontLayerTransmission.rgb / frontLayerRoughness
    const frontTx: Vec3 = frontLayer?.transmission ?? [1.0, 1.0, 1.0];
    const frontRoughness =
      frontLayer && Number.isFinite(frontLayer.roughness) ? Number(frontLayer.roughness) : -1.0;
    data[index++] = frontTx[0];
    data[index++] = frontTx[1];
    data[index++] = frontTx[2];
    data[index++] = frontRoughness;

    // sample 19 — backLayerTransmission.rgb / backLayerRoughness
    const backTx: Vec3 = backLayer?.transmission ?? [1.0, 1.0, 1.0];
    const backRoughness =
      backLayer && Number.isFinite(backLayer.roughness) ? Number(backLayer.roughness) : -1.0;
    data[index++] = backTx[0];
    data[index++] = backTx[1];
    data[index++] = backTx[2];
    data[index++] = backRoughness;

    // samples 20..27 (32 floats): uniform-grid spectral attenuation μ(λ), 380..780nm.
    const spectralDenom = Math.max(SPECTRAL_GRID_SAMPLE_COUNT - 1, 1);
    for (let s = 0; s < SPECTRAL_GRID_SAMPLE_COUNT; s += 1) {
      if (hasSpectral) {
        const t = s / spectralDenom;
        const lambdaNm = SPECTRAL_GRID_START_NM + t * (SPECTRAL_GRID_END_NM - SPECTRAL_GRID_START_NM);
        data[index++] = sampleSpectralCurve(spectralCurve, lambdaNm);
      } else {
        data[index++] = 0.0;
      }
    }

    // samples 28..54 (108 floats): thin-film layer payload, [ior, thicknessNm, extinction]×35.
    for (let layerIdx = 0; layerIdx < THIN_FILM_LAYER_LIMIT; layerIdx += 1) {
      if (layerIdx < thinFilmLayerCount) {
        const layer = thinFilmLayers[layerIdx]!;
        data[index++] = layer.ior ?? 1.0;
        data[index++] = layer.thicknessNm ?? 0.0;
        data[index++] = layer.extinctionCoefficient ?? 0.0;
      } else {
        data[index++] = 0.0;
        data[index++] = 0.0;
        data[index++] = 0.0;
      }
    }
    // pad the final 3 floats of sample 54
    data[index++] = 0.0;
    data[index++] = 0.0;
    data[index++] = 0.0;

    // samples 55..84 (30 texels): 15 texture-transform mat3s, 2 texels each, at
    // `texel 55 + 2k` (k per the GLSL `readTextureTransform` order in material_struct).
    // The GLSL only READS a transform when the map id != -1, so write one per mapped
    // texture (others stay zero / unread → identity). Maps without a transform slot
    // (alphaMap) sample with raw uv. The fork's `writeTextureMatrixToArray` is the
    // analogue.
    // Transform-slot order matches the GLSL `readTextureTransform` calls in
    // material_struct.glsl.js (firstTextureTransformIdx + 2k): map(0), metalness(2),
    // roughness(4), transmission(6), emissive(8), normal(10), clearcoat(12),
    // clearcoatNormal(14), clearcoatRoughness(16), sheenColor(18), sheenRoughness(20),
    // iridescence(22), iridescenceThickness(24), specularColor(26), specularIntensity(28).
    // Each slot is 2 texels (mat3 rows), starting at texel 55.
    if (baseColorLayer >= 0) writeTransform(data, base, 55, m.baseColorMap);
    if (metalLayer >= 0) writeTransform(data, base, 57, m.metallicMap);
    if (roughLayer >= 0) writeTransform(data, base, 59, m.roughnessMap);
    if (transmissionLayer >= 0) writeTransform(data, base, 61, m.transmissionMap);
    if (emissiveLayer >= 0) writeTransform(data, base, 63, m.emissiveMap);
    if (normalLayer >= 0) writeTransform(data, base, 65, m.normalMap);
    // D3 — clearcoat / sheen / iridescence / specular transforms (GLSL slots 12..28).
    if (clearcoatLayer >= 0) writeTransform(data, base, 67, m.clearcoatMap);
    if (clearcoatNormalLayer >= 0) writeTransform(data, base, 69, m.clearcoatNormalMap);
    if (clearcoatRoughnessLayer >= 0) writeTransform(data, base, 71, m.clearcoatRoughnessMap);
    if (sheenColorLayer >= 0) writeTransform(data, base, 73, m.sheenColorMap);
    if (sheenRoughnessLayer >= 0) writeTransform(data, base, 75, m.sheenRoughnessMap);
    if (iridescenceLayer >= 0) writeTransform(data, base, 77, m.iridescenceMap);
    if (iridescenceThicknessLayer >= 0) writeTransform(data, base, 79, m.iridescenceThicknessMap);
    if (specularColorLayer >= 0) writeTransform(data, base, 81, m.specularColorMap);
    if (specularIntensityLayer >= 0) writeTransform(data, base, 83, m.specularIntensityMap);

    // D3 — texels 85/86: ao/light/bump map ids + scalars + envMapIntensity
    // (mirrors readMaterialInfo s20/s21 in material_struct.glsl.js).
    let d3 = base + 85 * 4;
    data[d3++] = aoLayer;
    data[d3++] = lightMapLayer;
    data[d3++] = bumpLayer;
    data[d3++] = m.envMapIntensity ?? 1.0;
    data[d3++] = m.aoMapIntensity ?? 1.0;
    data[d3++] = m.lightMapIntensity ?? 1.0;
    data[d3++] = m.bumpScale ?? 1.0;
    data[d3++] = 0; // pad
    // D3 — ao/light/bump transforms at texels 87/89/91 (2 texels per mat3).
    if (aoLayer >= 0) writeTransform(data, base, 87, m.aoMap);
    if (lightMapLayer >= 0) writeTransform(data, base, 89, m.lightMap);
    if (bumpLayer >= 0) writeTransform(data, base, 91, m.bumpMap);

    index = base + MATERIAL_STRIDE;
  }

  return { data, dim, kind: 'rgba32f', materialCount };
}
