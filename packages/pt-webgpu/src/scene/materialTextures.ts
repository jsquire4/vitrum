// materialTextures.ts — P2 host-side texture collection for pt-webgpu.
//
// Given the scene's MaterialSpec[], dedup the texture-map source handles into an
// upload-ordered list and pack a per-material descriptor buffer (texture indices
// + alpha-mode + per-map KHR_texture_transform UV transform / texCoord). The GPU
// upload step (follow-on) turns `sources` into a texture_2d_array; the WGSL
// sampler reads the descriptor buffer (this layout) to sample with the right
// index + UVs. Materials with no maps get index -1 → the sampler skips them, so
// a textureless scene stays byte-identical to the pre-P2 parametric path.

import type { MaterialSpec, TextureRef, TextureWrapMode } from '@vitrum/core';
import type { MaterialTextureLayerUvScale } from './materialTextureArray.js';

/**
 * vec4s per material in the descriptor buffer (MUST match the WGSL
 * `MATERIAL_TEX_VEC4_STRIDE`):
 *   0: {baseColorIdx, normalIdx, roughnessMapIdx, emissiveIdx}   (-1 = no map)
 *   1: {alphaMode (0 opaque/1 mask/2 blend), alphaCutoff, opacity, texCoord}
 *   2: {offsetX, offsetY, scaleX, scaleY}                (baseColor UV transform)
 *   3: {rotation, aoMapIdx, lightMapIdx, bumpMapIdx}     ← D3 (-1 = no map)
 *   4: {aoMapIntensity, lightMapIntensity, bumpScale, envMapIntensity}  ← D3
 *   5: {anisotropy, anisotropyRotation, anisotropyMapIdx, normalScale}  ← D3/PTWG-MAT
 *   6: {alphaMapIdx, transmissionMapIdx, metallicMapIdx, _}      (-1 = no map)
 *   7: {baseColorUvScale.xy, emissiveUvScale.xy}           (per-layer UV-fit)
 *   8: {normalUvScale.xy, roughnessUvScale.xy}
 *   9: {metallicUvScale.xy, aoUvScale.xy}
 *  10: {lightMapUvScale.xy, bumpUvScale.xy}
 *  11: {anisotropyUvScale.xy, alphaUvScale.xy}
 *  12: {transmissionUvScale.xy, _, _}
 *  13: {baseColorWrap.xy, emissiveWrap.xy}                (0 repeat / 1 clamp / 2 mirror)
 *  14: {normalWrap.xy, roughnessWrap.xy}
 *  15: {metallicWrap.xy, aoWrap.xy}
 *  16: {lightMapWrap.xy, bumpWrap.xy}
 *  17: {anisotropyWrap.xy, alphaWrap.xy}
 *  18: {transmissionWrap.xy, _, _}
 *  19-40: per-map UV metadata, 2 vec4s per consumed base map:
 *     vec4 A: {texCoord, offsetX, offsetY, rotation}
 *     vec4 B: {scaleX, scaleY, 0, 0}
 *     map order: baseColor, emissive, normal, roughnessMap, metallicMap, AO,
 *                lightMap, bumpMap, anisotropyMap, alphaMap, transmissionMap
 *  41: {clearcoatMapIdx, clearcoatRoughnessMapIdx, sheenColorMapIdx,
 *       sheenRoughnessMapIdx}
 *  42: {iridescenceMapIdx, iridescenceThicknessMapIdx, specularColorMapIdx,
 *       specularIntensityMapIdx}
 *  43-46: extension-map UV-fit scale pairs
 *  47-50: extension-map wrap mode pairs
 *  51-66: extension-map UV metadata, 2 vec4s per consumed extension map:
 *     map order: clearcoat, clearcoatRoughness, sheenColor, sheenRoughness,
 *                iridescence, iridescenceThickness, specularColor,
 *                specularIntensity
 *  67: {clearcoatNormalMapIdx, clearcoatNormalScale, clearcoatNormalUvFit.xy}
 *  68: {clearcoatNormalWrap.xy, 0, 0}
 *  69-70: clearcoat normal UV metadata (A/B, same shape as the map metadata above)
 *  71: {thicknessMapIdx, thicknessUvFit.xy, _}
 *  72: {thicknessWrap.xy, 0, 0}
 *  73-74: thicknessMap UV metadata (A/B, same shape as the map metadata above)
 *  75: {frontLayerNormalMapIdx, frontLayerNormalScale,
 *       backLayerNormalMapIdx, backLayerNormalScale}
 *  76: {frontLayerNormalUvFit.xy, backLayerNormalUvFit.xy}
 *  77: {frontLayerNormalWrap.xy, backLayerNormalWrap.xy}
 *  78-81: front/back layer normal UV metadata, 2 vec4s per map.
 *
 * D3 (reserved-field consumption) bumped the stride 4 → 6:
 *   - vec4 #3.yzw + vec4 #4.xyz: aoMap / lightMap / bumpMap layer indices and
 *     their intensity / scale scalars. All three maps are LINEAR-space (occlusion,
 *     baked-radiance-as-data, height field) so they share the LINEAR texture array
 *     index space (materialTexturesLinear). A material lacking a given map carries
 *     index -1 (the WGSL sampler returns a no-op), so absent-field scenes stay
 *     byte-identical to the pre-D3 path.
 *   - vec4 #4.w: per-material envMapIntensity (default 1).
 *   - vec4 #5: anisotropy / anisotropyRotation scalars + the optional
 *     anisotropyMap layer index (KHR_materials_anisotropy: RG = tangent rotation
 *     direction, B = strength), also in the LINEAR array. anisotropy == 0 (default)
 *     means the anisotropic GGX path is never taken → byte-identical.
 *   - vec4 #5.w: glTF normalTexture.scale / MaterialSpec.normalScale. Default 1,
 *     so legacy normal-mapped scenes remain byte-identical unless authored scale
 *     asks to dampen or amplify the tangent-space xy perturbation.
 *   - vec4 #6.x: standalone alphaMap layer in the LINEAR array (coverage data,
 *     not color). It multiplies baseColor alpha and opacity in alphaMode mask/blend.
 *   - vec4 #6.y: transmissionMap layer in the LINEAR array. It multiplies the
 *     scalar `MaterialSpec.transmission` (glTF KHR_materials_transmission R channel).
 *   - vec4 #7–#12: per-map UV-fit scales. Heterogeneous texture arrays copy each
 *     source into a max-sized layer; these scales remap repeat-wrapped UVs into
 *     the copied source rectangle instead of sampling padded black texels.
 *   - vec4 #13–#18: per-map wrap modes from TextureRef.wrapS/wrapT. Defaults are
 *     repeat/repeat, matching glTF. Encoded as 0 repeat, 1 clamp, 2 mirrored.
 *   - vec4 #19–#40: per-map `TextureRef.texCoord` and KHR_texture_transform
 *     metadata. Older rows only carried the baseColor transform and every other
 *     map inherited it; these lanes make the full-tier sampler honor each map's
 *     own UV channel/transform without changing the original baseColor lanes.
 *   - vec4 #41–#66: extension-lobe texture maps. Scalar maps are LINEAR-space
 *     data and follow the glTF extension channel conventions (clearcoat R,
 *     clearcoatRoughness G, sheenRoughness A, iridescence R,
 *     iridescenceThickness G, specularIntensity A). Color tint maps
 *     (sheenColor/specularColor) live in the sRGB array.
 *   - vec4 #67–#70: KHR_materials_clearcoat clearcoatNormalTexture. This is a
 *     LINEAR tangent-space normal map with its own scale/UV/wrap metadata, kept
 *     separate from the scalar/color extension lobe map block so existing lanes
 *     do not shift.
 *   - vec4 #71–#74: KHR_materials_volume thicknessTexture. This is a LINEAR
 *     scalar map sampled from G and multiplied by MaterialSpec.thickness to clamp
 *     the approximate Beer-Lambert slab distance.
 *   - vec4 #75–#81: RFE-03 front/back layer normal maps. These are LINEAR
 *     tangent-space normal maps with independent scales, UV metadata, wrap modes,
 *     and heterogeneous-layer UV-fit scales. Shaders face-select them ahead of
 *     the top-level normal map when the selected face has a layer normal.
 */
export const MATERIAL_TEX_UV_META_VEC4_OFFSET = 19;
export const MATERIAL_TEX_UV_META_VEC4S_PER_MAP = 2;
export const MATERIAL_TEX_UV_MAP_COUNT = 11;
export const MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET = 41;
export const MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET = 43;
export const MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET = 47;
export const MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET = 51;
export const MATERIAL_TEX_EXTENSION_MAP_COUNT = 8;
export const MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET = 67;
export const MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET = 68;
export const MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET = 69;
export const MATERIAL_TEX_THICKNESS_VEC4_OFFSET = 71;
export const MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET = 72;
export const MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET = 73;
export const MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET = 75;
export const MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET = 76;
export const MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET = 77;
export const MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET = 78;
export const MATERIAL_TEX_LAYER_NORMAL_MAP_COUNT = 2;
export const MATERIAL_TEX_VEC4_STRIDE =
  MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET +
  MATERIAL_TEX_LAYER_NORMAL_MAP_COUNT * MATERIAL_TEX_UV_META_VEC4S_PER_MAP;
export const MATERIAL_TEX_FLOAT_STRIDE = MATERIAL_TEX_VEC4_STRIDE * 4;

const ALPHA_MODE_INDEX: Readonly<Record<'opaque' | 'mask' | 'blend', number>> = {
  opaque: 0,
  mask: 1,
  blend: 2,
};

const WRAP_MODE_INDEX: Readonly<Record<TextureWrapMode, number>> = {
  repeat: 0,
  'clamp-to-edge': 1,
  'mirrored-repeat': 2,
};

export interface CollectedTextures {
  /** Unique sRGB-decoded texture sources (baseColor + emissive), upload order. */
  readonly sources: unknown[];
  /** Unique LINEAR texture sources (normal + scalar/data maps — must NOT be sRGB-decoded),
   *  a separate index space → its own texture_2d_array. */
  readonly linearSources: unknown[];
  /** Source-layer provenance for host-facing upload diagnostics. */
  readonly sourceInfos: readonly MaterialTextureLayerInfo[];
  /** Linear source-layer provenance for host-facing upload diagnostics. */
  readonly linearSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Per-material descriptor floats (MATERIAL_TEX_FLOAT_STRIDE per material). */
  readonly descriptors: Float32Array;
}

export type MaterialTextureColorSpace = 'srgb' | 'linear';

export interface MaterialTextureLayerUse {
  readonly materialIndex: number;
  readonly field: string;
  readonly colorSpace: MaterialTextureColorSpace;
  readonly texCoord: number;
}

export interface MaterialTextureLayerInfo {
  readonly layer: number;
  readonly uses: readonly MaterialTextureLayerUse[];
}

function uvFitScaleFor(
  scales: readonly MaterialTextureLayerUvScale[],
  layerIdx: number,
): MaterialTextureLayerUvScale {
  if (layerIdx < 0 || layerIdx >= scales.length) return [1, 1];
  return scales[layerIdx] ?? [1, 1];
}

function writeUvFitPair(
  descriptors: Float32Array,
  offset: number,
  scale: MaterialTextureLayerUvScale,
): void {
  descriptors[offset] = scale[0];
  descriptors[offset + 1] = scale[1];
}

function writeDefaultUvFitPairs(descriptors: Float32Array, b: number): void {
  for (let offset = b + 28; offset < b + 52; offset += 2) {
    descriptors[offset] = 1;
    descriptors[offset + 1] = 1;
  }
}

function writeDefaultExtensionUvFitPairs(descriptors: Float32Array, b: number): void {
  const start = b + MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET * 4;
  const end = start + MATERIAL_TEX_EXTENSION_MAP_COUNT * 2;
  for (let offset = start; offset < end; offset += 2) {
    descriptors[offset] = 1;
    descriptors[offset + 1] = 1;
  }
}

function writeWrapPair(
  descriptors: Float32Array,
  offset: number,
  ref: TextureRef | undefined,
): void {
  descriptors[offset] = WRAP_MODE_INDEX[ref?.wrapS ?? 'repeat'];
  descriptors[offset + 1] = WRAP_MODE_INDEX[ref?.wrapT ?? 'repeat'];
}

function writeUvMeta(
  descriptors: Float32Array,
  b: number,
  mapSlot: number,
  ref: TextureRef | undefined,
  metaVec4Offset = MATERIAL_TEX_UV_META_VEC4_OFFSET,
): void {
  const vecBase = b + (metaVec4Offset + mapSlot * MATERIAL_TEX_UV_META_VEC4S_PER_MAP) * 4;
  const t = ref?.transform;
  descriptors[vecBase] = ref?.texCoord ?? 0;
  descriptors[vecBase + 1] = t?.offset?.[0] ?? 0;
  descriptors[vecBase + 2] = t?.offset?.[1] ?? 0;
  descriptors[vecBase + 3] = t?.rotation ?? 0;
  descriptors[vecBase + 4] = t?.scale?.[0] ?? 1;
  descriptors[vecBase + 5] = t?.scale?.[1] ?? 1;
  descriptors[vecBase + 6] = 0;
  descriptors[vecBase + 7] = 0;
}

/** Fill per-map UV-fit descriptor lanes after the texture arrays reveal their
 *  actual per-layer source rects. Same-size layers remain [1,1]. */
export function applyMaterialTextureUvFitScales(
  descriptors: Float32Array,
  sRgbLayerScales: readonly MaterialTextureLayerUvScale[],
  linearLayerScales: readonly MaterialTextureLayerUvScale[],
): void {
  const materialCount = Math.floor(descriptors.length / MATERIAL_TEX_FLOAT_STRIDE);
  for (let mi = 0; mi < materialCount; mi += 1) {
    const b = mi * MATERIAL_TEX_FLOAT_STRIDE;
    // sRGB array maps: baseColor and emissive.
    writeUvFitPair(descriptors, b + 28, uvFitScaleFor(sRgbLayerScales, descriptors[b + 0] ?? -1));
    writeUvFitPair(descriptors, b + 30, uvFitScaleFor(sRgbLayerScales, descriptors[b + 3] ?? -1));
    // Linear array maps: normal, roughness, metallic, AO, light, bump, anisotropy, alpha, transmission.
    writeUvFitPair(descriptors, b + 32, uvFitScaleFor(linearLayerScales, descriptors[b + 1] ?? -1));
    writeUvFitPair(descriptors, b + 34, uvFitScaleFor(linearLayerScales, descriptors[b + 2] ?? -1));
    writeUvFitPair(descriptors, b + 36, uvFitScaleFor(linearLayerScales, descriptors[b + 26] ?? -1));
    writeUvFitPair(descriptors, b + 38, uvFitScaleFor(linearLayerScales, descriptors[b + 13] ?? -1));
    writeUvFitPair(descriptors, b + 40, uvFitScaleFor(linearLayerScales, descriptors[b + 14] ?? -1));
    writeUvFitPair(descriptors, b + 42, uvFitScaleFor(linearLayerScales, descriptors[b + 15] ?? -1));
    writeUvFitPair(descriptors, b + 44, uvFitScaleFor(linearLayerScales, descriptors[b + 22] ?? -1));
    writeUvFitPair(descriptors, b + 46, uvFitScaleFor(linearLayerScales, descriptors[b + 24] ?? -1));
    writeUvFitPair(descriptors, b + 48, uvFitScaleFor(linearLayerScales, descriptors[b + 25] ?? -1));
    const ext = b + MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET * 4;
    const extFit = b + MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET * 4;
    writeUvFitPair(descriptors, extFit, uvFitScaleFor(linearLayerScales, descriptors[ext] ?? -1));
    writeUvFitPair(descriptors, extFit + 2, uvFitScaleFor(linearLayerScales, descriptors[ext + 1] ?? -1));
    writeUvFitPair(descriptors, extFit + 4, uvFitScaleFor(sRgbLayerScales, descriptors[ext + 2] ?? -1));
    writeUvFitPair(descriptors, extFit + 6, uvFitScaleFor(linearLayerScales, descriptors[ext + 3] ?? -1));
    writeUvFitPair(descriptors, extFit + 8, uvFitScaleFor(linearLayerScales, descriptors[ext + 4] ?? -1));
    writeUvFitPair(descriptors, extFit + 10, uvFitScaleFor(linearLayerScales, descriptors[ext + 5] ?? -1));
    writeUvFitPair(descriptors, extFit + 12, uvFitScaleFor(sRgbLayerScales, descriptors[ext + 6] ?? -1));
    writeUvFitPair(descriptors, extFit + 14, uvFitScaleFor(linearLayerScales, descriptors[ext + 7] ?? -1));
    writeUvFitPair(
      descriptors,
      b + MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET * 4 + 2,
      uvFitScaleFor(
        linearLayerScales,
        descriptors[b + MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET * 4] ?? -1,
      ),
    );
    writeUvFitPair(
      descriptors,
      b + MATERIAL_TEX_THICKNESS_VEC4_OFFSET * 4 + 1,
      uvFitScaleFor(
        linearLayerScales,
        descriptors[b + MATERIAL_TEX_THICKNESS_VEC4_OFFSET * 4] ?? -1,
      ),
    );
    const layerNormalBase = b + MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET * 4;
    const layerNormalFitBase = b + MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET * 4;
    writeUvFitPair(
      descriptors,
      layerNormalFitBase,
      uvFitScaleFor(linearLayerScales, descriptors[layerNormalBase] ?? -1),
    );
    writeUvFitPair(
      descriptors,
      layerNormalFitBase + 2,
      uvFitScaleFor(linearLayerScales, descriptors[layerNormalBase + 2] ?? -1),
    );
  }
}

/** Collect + dedup material texture sources and pack the per-material descriptors.
 *  Two index spaces: sRGB (baseColor/emissive) and linear (normal/scalar data) — they
 *  upload to separate arrays so each is sampled in the correct colour space. */
export function collectMaterialTextures(materials: ReadonlyArray<MaterialSpec>): CollectedTextures {
  const sources: unknown[] = [];
  const linearSources: unknown[] = [];
  const makeIndexer = (list: unknown[], colorSpace: MaterialTextureColorSpace) => {
    const handleToIdx = new Map<unknown, number>();
    const usesByLayer: MaterialTextureLayerUse[][] = [];
    const index = (ref: TextureRef | undefined, materialIndex: number, field: string): number => {
      const handle = ref?.handle;
      if (handle == null) return -1;
      let i = handleToIdx.get(handle);
      if (i === undefined) {
        i = list.length;
        list.push(handle);
        handleToIdx.set(handle, i);
      }
      (usesByLayer[i] ??= []).push({
        materialIndex,
        field,
        colorSpace,
        texCoord: ref?.texCoord ?? 0,
      });
      return i;
    };
    const infos = (): readonly MaterialTextureLayerInfo[] =>
      list.map((_, layer) => ({ layer, uses: usesByLayer[layer] ?? [] }));
    return { index, infos };
  };
  const sRgbIndexer = makeIndexer(sources, 'srgb');
  const linearIndexer = makeIndexer(linearSources, 'linear');
  const indexOf = sRgbIndexer.index;        // sRGB array
  const indexOfLinear = linearIndexer.index; // linear array

  const descriptors = new Float32Array(materials.length * MATERIAL_TEX_FLOAT_STRIDE);
  materials.forEach((m, mi) => {
    const b = mi * MATERIAL_TEX_FLOAT_STRIDE;
    const bc = m.baseColorMap;
    // glTF's canonical metallicRoughness texture is one image (G=roughness,
    // B=metallic). Preserve that combined-map behavior when only one side is
    // supplied, but allow distinct authored maps to carry independent UV/wrap.
    const roughnessMap = m.roughnessMap ?? m.metallicMap;
    const metallicMap = m.metallicMap ?? m.roughnessMap;
    descriptors[b + 0] = indexOf(bc, mi, 'baseColorMap');            // baseColorIdx (sRGB array)
    descriptors[b + 1] = indexOfLinear(m.normalMap, mi, 'normalMap');                 // normalIdx (linear array)
    descriptors[b + 2] = indexOfLinear(roughnessMap, mi, 'roughnessMap'); // roughness map (linear; glTF G channel)
    descriptors[b + 3] = indexOf(m.emissiveMap, mi, 'emissiveMap'); // emissiveIdx (sRGB array — same layers as baseColor)
    descriptors[b + 4] = ALPHA_MODE_INDEX[m.alphaMode ?? 'opaque'];
    descriptors[b + 5] = m.alphaCutoff ?? 0.5;
    descriptors[b + 6] = m.opacity ?? 1;
    descriptors[b + 7] = bc?.texCoord ?? 0;
    const t = bc?.transform;
    descriptors[b + 8] = t?.offset?.[0] ?? 0;
    descriptors[b + 9] = t?.offset?.[1] ?? 0;
    descriptors[b + 10] = t?.scale?.[0] ?? 1;
    descriptors[b + 11] = t?.scale?.[1] ?? 1;
    descriptors[b + 12] = t?.rotation ?? 0;
    // D3 — vec4 #3.yzw + #4.xyz: aoMap / lightMap / bumpMap (all LINEAR-space data:
    // occlusion factor, baked outgoing radiance, height field) routed through the
    // linear texture array. Index -1 when absent → the WGSL sampler is a no-op.
    descriptors[b + 13] = indexOfLinear(m.aoMap, mi, 'aoMap');
    descriptors[b + 14] = indexOfLinear(m.lightMap, mi, 'lightMap');
    descriptors[b + 15] = indexOfLinear(m.bumpMap, mi, 'bumpMap');
    descriptors[b + 16] = m.aoMapIntensity ?? 1;
    descriptors[b + 17] = m.lightMapIntensity ?? 1;
    descriptors[b + 18] = m.bumpScale ?? 1;
    descriptors[b + 19] = m.envMapIntensity ?? 1;
    // D3 — vec4 #5: anisotropy scalars + optional KHR_materials_anisotropy map.
    descriptors[b + 20] = m.anisotropy ?? 0;
    descriptors[b + 21] = m.anisotropyRotation ?? 0;
    descriptors[b + 22] = indexOfLinear(m.anisotropyMap, mi, 'anisotropyMap');
    descriptors[b + 23] = m.normalScale ?? 1;
    // Standalone alphaMap is coverage data (linear). BaseColor alpha still
    // participates too; each map carries its own UV metadata below.
    descriptors[b + 24] = indexOfLinear(m.alphaMap, mi, 'alphaMap');
    descriptors[b + 25] = indexOfLinear(m.transmissionMap, mi, 'transmissionMap');
    descriptors[b + 26] = indexOfLinear(metallicMap, mi, 'metallicMap'); // metallic map (linear; glTF B channel)
    descriptors[b + 27] = 0;
    // Extension-lobe texture maps. Color tint maps use the sRGB array; scalar
    // factor/roughness/thickness maps use the LINEAR array.
    const extIndexBase = b + MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET * 4;
    descriptors[extIndexBase] = indexOfLinear(m.clearcoatMap, mi, 'clearcoatMap');
    descriptors[extIndexBase + 1] = indexOfLinear(m.clearcoatRoughnessMap, mi, 'clearcoatRoughnessMap');
    descriptors[extIndexBase + 2] = indexOf(m.sheenColorMap, mi, 'sheenColorMap');
    descriptors[extIndexBase + 3] = indexOfLinear(m.sheenRoughnessMap, mi, 'sheenRoughnessMap');
    descriptors[extIndexBase + 4] = indexOfLinear(m.iridescenceMap, mi, 'iridescenceMap');
    descriptors[extIndexBase + 5] = indexOfLinear(m.iridescenceThicknessMap, mi, 'iridescenceThicknessMap');
    descriptors[extIndexBase + 6] = indexOf(m.specularColorMap, mi, 'specularColorMap');
    descriptors[extIndexBase + 7] = indexOfLinear(m.specularIntensityMap, mi, 'specularIntensityMap');
    const clearcoatNormalBase = b + MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET * 4;
    descriptors[clearcoatNormalBase] = indexOfLinear(m.clearcoatNormalMap, mi, 'clearcoatNormalMap');
    descriptors[clearcoatNormalBase + 1] = m.clearcoatNormalScale ?? 1;
    descriptors[clearcoatNormalBase + 2] = 1;
    descriptors[clearcoatNormalBase + 3] = 1;
    const thicknessBase = b + MATERIAL_TEX_THICKNESS_VEC4_OFFSET * 4;
    descriptors[thicknessBase] = indexOfLinear(m.thicknessMap, mi, 'thicknessMap');
    descriptors[thicknessBase + 1] = 1;
    descriptors[thicknessBase + 2] = 1;
    descriptors[thicknessBase + 3] = 0;
    const layerNormalBase = b + MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET * 4;
    descriptors[layerNormalBase] = indexOfLinear(m.frontLayer?.normalMap, mi, 'frontLayer.normalMap');
    descriptors[layerNormalBase + 1] = m.frontLayer?.normalScale ?? 1;
    descriptors[layerNormalBase + 2] = indexOfLinear(m.backLayer?.normalMap, mi, 'backLayer.normalMap');
    descriptors[layerNormalBase + 3] = m.backLayer?.normalScale ?? 1;
    const layerNormalFitBase = b + MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET * 4;
    descriptors[layerNormalFitBase] = 1;
    descriptors[layerNormalFitBase + 1] = 1;
    descriptors[layerNormalFitBase + 2] = 1;
    descriptors[layerNormalFitBase + 3] = 1;
    writeDefaultUvFitPairs(descriptors, b);
    writeDefaultExtensionUvFitPairs(descriptors, b);
    writeWrapPair(descriptors, b + 52, bc);
    writeWrapPair(descriptors, b + 54, m.emissiveMap);
    writeWrapPair(descriptors, b + 56, m.normalMap);
    writeWrapPair(descriptors, b + 58, roughnessMap);
    writeWrapPair(descriptors, b + 60, metallicMap);
    writeWrapPair(descriptors, b + 62, m.aoMap);
    writeWrapPair(descriptors, b + 64, m.lightMap);
    writeWrapPair(descriptors, b + 66, m.bumpMap);
    writeWrapPair(descriptors, b + 68, m.anisotropyMap);
    writeWrapPair(descriptors, b + 70, m.alphaMap);
    writeWrapPair(descriptors, b + 72, m.transmissionMap);
    const extWrapBase = b + MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET * 4;
    writeWrapPair(descriptors, extWrapBase, m.clearcoatMap);
    writeWrapPair(descriptors, extWrapBase + 2, m.clearcoatRoughnessMap);
    writeWrapPair(descriptors, extWrapBase + 4, m.sheenColorMap);
    writeWrapPair(descriptors, extWrapBase + 6, m.sheenRoughnessMap);
    writeWrapPair(descriptors, extWrapBase + 8, m.iridescenceMap);
    writeWrapPair(descriptors, extWrapBase + 10, m.iridescenceThicknessMap);
    writeWrapPair(descriptors, extWrapBase + 12, m.specularColorMap);
    writeWrapPair(descriptors, extWrapBase + 14, m.specularIntensityMap);
    writeWrapPair(descriptors, b + MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET * 4, m.clearcoatNormalMap);
    writeWrapPair(descriptors, b + MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET * 4, m.thicknessMap);
    writeWrapPair(descriptors, b + MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET * 4, m.frontLayer?.normalMap);
    writeWrapPair(descriptors, b + MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET * 4 + 2, m.backLayer?.normalMap);
    writeUvMeta(descriptors, b, 0, bc);
    writeUvMeta(descriptors, b, 1, m.emissiveMap);
    writeUvMeta(descriptors, b, 2, m.normalMap);
    writeUvMeta(descriptors, b, 3, roughnessMap);
    writeUvMeta(descriptors, b, 4, metallicMap);
    writeUvMeta(descriptors, b, 5, m.aoMap);
    writeUvMeta(descriptors, b, 6, m.lightMap);
    writeUvMeta(descriptors, b, 7, m.bumpMap);
    writeUvMeta(descriptors, b, 8, m.anisotropyMap);
    writeUvMeta(descriptors, b, 9, m.alphaMap);
    writeUvMeta(descriptors, b, 10, m.transmissionMap);
    writeUvMeta(descriptors, b, 0, m.clearcoatMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 1, m.clearcoatRoughnessMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 2, m.sheenColorMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 3, m.sheenRoughnessMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 4, m.iridescenceMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 5, m.iridescenceThicknessMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 6, m.specularColorMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 7, m.specularIntensityMap, MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 0, m.clearcoatNormalMap, MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 0, m.thicknessMap, MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 0, m.frontLayer?.normalMap, MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET);
    writeUvMeta(descriptors, b, 1, m.backLayer?.normalMap, MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET);
  });

  return {
    sources,
    linearSources,
    sourceInfos: sRgbIndexer.infos(),
    linearSourceInfos: linearIndexer.infos(),
    descriptors,
  };
}
