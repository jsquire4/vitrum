// materialTextures.ts — P2 host-side texture collection for pt-webgpu.
//
// Given the scene's MaterialSpec[], dedup the texture-map source handles into an
// upload-ordered list and pack a per-material descriptor buffer (texture indices
// + alpha-mode + per-map KHR_texture_transform UV transform / texCoord). The GPU
// upload step (follow-on) turns `sources` into a texture_2d_array; the WGSL
// sampler reads the descriptor buffer (this layout) to sample with the right
// index + UVs. Materials with no maps get index -1 → the sampler skips them, so
// a textureless scene stays byte-identical to the pre-P2 parametric path.

import type {
  EngineWarning,
  MaterialSpec,
  TextureFilterMode,
  TextureMipFilterMode,
  TextureRef,
  TextureWrapMode,
} from '@vitrum/core';
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
 *  82-87: per-map mip policy, packed as one scalar per consumed map:
 *     0 none / 1 nearest / 2 linear. Map order follows the UV metadata blocks:
 *     baseColor, emissive, normal, roughnessMap, metallicMap, AO, lightMap,
 *     bumpMap, anisotropyMap, alphaMap, transmissionMap, clearcoat,
 *     clearcoatRoughness, sheenColor, sheenRoughness, iridescence,
 *     iridescenceThickness, specularColor, specularIntensity, clearcoatNormal,
 *     thickness, frontLayer.normalMap, backLayer.normalMap.
 *  88-99: per-map filter policy, packed as two scalars per consumed map:
 *     {magFilter, minFilter}, 0 nearest / 1 linear, in the same map order.
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
 *   - vec4 #82–#87: authored mip-filter policy for every map above. The
 *     full-tier shader already uses explicit LOD in compute, so these lanes
 *     let it honor `mipFilter:"none"` and nearest-vs-linear mip selection.
 *   - vec4 #88–#99: authored mag/min filter policy for every map above. Regular
 *     map sampling switches nearest requests to explicit `textureLoad` while
 *     retaining filtered `textureSampleLevel` for linear requests.
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
export const MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET =
  MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET +
  MATERIAL_TEX_LAYER_NORMAL_MAP_COUNT * MATERIAL_TEX_UV_META_VEC4S_PER_MAP;
export const MATERIAL_TEX_MIP_POLICY_MAP_COUNT =
  MATERIAL_TEX_UV_MAP_COUNT + MATERIAL_TEX_EXTENSION_MAP_COUNT + 1 + 1 + MATERIAL_TEX_LAYER_NORMAL_MAP_COUNT;
export const MATERIAL_TEX_MIP_POLICY_VEC4_COUNT = Math.ceil(MATERIAL_TEX_MIP_POLICY_MAP_COUNT / 4);
export const MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET =
  MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET + MATERIAL_TEX_MIP_POLICY_VEC4_COUNT;
export const MATERIAL_TEX_FILTER_POLICY_FLOATS_PER_MAP = 2;
export const MATERIAL_TEX_FILTER_POLICY_VEC4_COUNT = Math.ceil(
  (MATERIAL_TEX_MIP_POLICY_MAP_COUNT * MATERIAL_TEX_FILTER_POLICY_FLOATS_PER_MAP) / 4,
);
export const MATERIAL_TEX_VEC4_STRIDE =
  MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET + MATERIAL_TEX_FILTER_POLICY_VEC4_COUNT;
export const MATERIAL_TEX_FLOAT_STRIDE = MATERIAL_TEX_VEC4_STRIDE * 4;

/** Combined-map resolution context threaded to each slot's `resolve` accessor.
 *  glTF's metallicRoughness is one image (G=roughness, B=metallic); when only one
 *  side is authored, both slots reference it (see `collectMaterialTextures`). */
interface MaterialTextureResolveContext {
  readonly roughnessMap: TextureRef | undefined;
  readonly metallicMap: TextureRef | undefined;
}

/**
 * T2-A — the single ordered texture-map table. Index == the map's slot in the
 * mip/filter policy blocks (0..22). Each entry drives, from ONE loop in
 * `collectMaterialTextures`, all four per-map descriptor writes that previously
 * enumerated this same 23-map list separately:
 *   - `wrap`     → `wrapFloatOffset` (float offset within the material block)
 *   - `mip`      → the slot index (this array position)
 *   - `filter`   → the slot index
 *   - `uvMeta`   → `uvMetaVec4Offset` + `uvMetaSlot`
 *
 * The ORDER here IS the wire format the WGSL sampler reads (mip/filter policy
 * blocks are indexed by slot), and the wrap/uvMeta offsets reproduce the exact
 * pre-refactor layout — pinned byte-identical by
 * `materialTexDescriptorGolden.test.ts`. The block layout doc (top of file) is
 * asserted against this table in `materialTextures.test.ts`.
 *
 * NOTE: the per-map INDEX lanes and interspersed scalar lanes (alphaMode,
 * normalScale, intensities, layer scales) are NOT uniform per-slot — they scatter
 * across bespoke lanes and stay hand-written in `collectMaterialTextures`.
 */
interface TextureMapSlot {
  /** Diagnostic name (matches the layout-doc map order). */
  readonly name: string;
  /** Resolve this slot's TextureRef off the material (+ combined-map context). */
  readonly resolve: (m: MaterialSpec, ctx: MaterialTextureResolveContext) => TextureRef | undefined;
  /** Float offset (relative to the material block base `b`) for the wrap pair. */
  readonly wrapFloatOffset: number;
  /** vec4 offset of this slot's UV-metadata block (A/B pair). */
  readonly uvMetaVec4Offset: number;
  /** Slot index WITHIN that UV-metadata block. */
  readonly uvMetaSlot: number;
}

const EXT_WRAP_BASE = MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET * 4;

export const TEXTURE_MAP_SLOTS: readonly TextureMapSlot[] = [
  // ── Main-block maps (wrap lanes 52..73; UV meta at offset 19) ───────────────
  { name: 'baseColor',    resolve: (m) => m.baseColorMap,         wrapFloatOffset: 52, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 0 },
  { name: 'emissive',     resolve: (m) => m.emissiveMap,          wrapFloatOffset: 54, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 1 },
  { name: 'normal',       resolve: (m) => m.normalMap,            wrapFloatOffset: 56, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 2 },
  { name: 'roughness',    resolve: (_m, c) => c.roughnessMap,     wrapFloatOffset: 58, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 3 },
  { name: 'metallic',     resolve: (_m, c) => c.metallicMap,      wrapFloatOffset: 60, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 4 },
  { name: 'ao',           resolve: (m) => m.aoMap,                wrapFloatOffset: 62, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 5 },
  { name: 'lightMap',     resolve: (m) => m.lightMap,             wrapFloatOffset: 64, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 6 },
  { name: 'bump',         resolve: (m) => m.bumpMap,              wrapFloatOffset: 66, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 7 },
  { name: 'anisotropy',   resolve: (m) => m.anisotropyMap,        wrapFloatOffset: 68, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 8 },
  { name: 'alpha',        resolve: (m) => m.alphaMap,             wrapFloatOffset: 70, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 9 },
  { name: 'transmission', resolve: (m) => m.transmissionMap,      wrapFloatOffset: 72, uvMetaVec4Offset: MATERIAL_TEX_UV_META_VEC4_OFFSET, uvMetaSlot: 10 },
  // ── Extension-lobe maps (wrap at EXT_WRAP_BASE; UV meta at offset 51) ───────
  { name: 'clearcoat',            resolve: (m) => m.clearcoatMap,             wrapFloatOffset: EXT_WRAP_BASE + 0,  uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 0 },
  { name: 'clearcoatRoughness',   resolve: (m) => m.clearcoatRoughnessMap,    wrapFloatOffset: EXT_WRAP_BASE + 2,  uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 1 },
  { name: 'sheenColor',           resolve: (m) => m.sheenColorMap,            wrapFloatOffset: EXT_WRAP_BASE + 4,  uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 2 },
  { name: 'sheenRoughness',       resolve: (m) => m.sheenRoughnessMap,        wrapFloatOffset: EXT_WRAP_BASE + 6,  uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 3 },
  { name: 'iridescence',          resolve: (m) => m.iridescenceMap,           wrapFloatOffset: EXT_WRAP_BASE + 8,  uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 4 },
  { name: 'iridescenceThickness', resolve: (m) => m.iridescenceThicknessMap,  wrapFloatOffset: EXT_WRAP_BASE + 10, uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 5 },
  { name: 'specularColor',        resolve: (m) => m.specularColorMap,         wrapFloatOffset: EXT_WRAP_BASE + 12, uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 6 },
  { name: 'specularIntensity',    resolve: (m) => m.specularIntensityMap,     wrapFloatOffset: EXT_WRAP_BASE + 14, uvMetaVec4Offset: MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET, uvMetaSlot: 7 },
  // ── Bespoke single-map lanes (own wrap + UV-meta offsets) ──────────────────
  { name: 'clearcoatNormal', resolve: (m) => m.clearcoatNormalMap,   wrapFloatOffset: MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET * 4, uvMetaVec4Offset: MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET, uvMetaSlot: 0 },
  { name: 'thickness',       resolve: (m) => m.thicknessMap,         wrapFloatOffset: MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET * 4,        uvMetaVec4Offset: MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET,        uvMetaSlot: 0 },
  { name: 'frontLayerNormal', resolve: (m) => m.frontLayer?.normalMap, wrapFloatOffset: MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET * 4 + 0, uvMetaVec4Offset: MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET, uvMetaSlot: 0 },
  { name: 'backLayerNormal',  resolve: (m) => m.backLayer?.normalMap,  wrapFloatOffset: MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET * 4 + 2, uvMetaVec4Offset: MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET, uvMetaSlot: 1 },
];

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

const MIP_FILTER_INDEX: Readonly<Record<TextureMipFilterMode, number>> = {
  none: 0,
  nearest: 1,
  linear: 2,
};

function isUnsupportedTexCoord(ref: TextureRef | undefined): boolean {
  if (ref?.handle == null) return false;
  const texCoord = ref.texCoord ?? 0;
  return texCoord !== 0 && texCoord !== 1;
}

function safeTexCoord(ref: TextureRef | undefined): number {
  return isUnsupportedTexCoord(ref) ? 0 : (ref?.texCoord ?? 0);
}

function unsupportedTexCoordWarning(
  materialIndex: number,
  field: string,
  texCoord: number,
): EngineWarning {
  const message =
    `[vitrum/pt-webgpu] ignoring material ${materialIndex} ${field}: texCoord ${texCoord} ` +
    `is unsupported; only texCoord 0 and 1 are renderable by this backend.`;
  return {
    code: 'pt-webgpu.material-texture-unsupported-texcoord',
    backend: 'pt-webgpu',
    phase: 'setScene',
    method: 'setScene',
    message,
    details: {
      materialIndex,
      field,
      texCoord,
      supportedTexCoords: [0, 1],
      fallback: 'map-ignored',
    },
  };
}

function createUnsupportedTexCoordWarner(
  structuredWarnings: EngineWarning[],
): (
  materialIndex: number,
  field: string,
  ref: TextureRef | undefined,
) => boolean {
  const warned = new Set<string>();
  return (materialIndex, field, ref): boolean => {
    if (!isUnsupportedTexCoord(ref)) return false;
    const texCoord = ref?.texCoord ?? 0;
    const key = `${materialIndex}:${field}:${texCoord}`;
    if (!warned.has(key)) {
      warned.add(key);
      const warning = unsupportedTexCoordWarning(materialIndex, field, texCoord);
      structuredWarnings.push(warning);
      console.warn(warning.message.replace('[vitrum/pt-webgpu] ', '[pt-webgpu] '));
    }
    return true;
  };
}

export interface CollectedTextures {
  /** Unique sRGB-decoded texture sources (baseColor + extension color-tint maps),
   *  upload order. Emissive is NO LONGER here — it has its own rgba16float array
   *  (see {@link emissiveSources}) so HDR emissive values survive packing. */
  readonly sources: unknown[];
  /** Unique LINEAR texture sources (normal + scalar/data maps — must NOT be sRGB-decoded),
   *  a separate index space → its own texture_2d_array. */
  readonly linearSources: unknown[];
  /** Unique EMISSIVE texture sources → a dedicated rgba16float texture_2d_array.
   *  Own index space (emissiveIdx points here, not the sRGB array). Uploaded to a
   *  linear-float target with a CPU sRGB→linear decode for LDR sources, so HDR
   *  emissive texture values > 1.0 are not clamped to [0,1]. */
  readonly emissiveSources: unknown[];
  /** Source-layer provenance for host-facing upload diagnostics. */
  readonly sourceInfos: readonly MaterialTextureLayerInfo[];
  /** Linear source-layer provenance for host-facing upload diagnostics. */
  readonly linearSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Emissive source-layer provenance for host-facing upload diagnostics. */
  readonly emissiveSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Per-material descriptor floats (MATERIAL_TEX_FLOAT_STRIDE per material). */
  readonly descriptors: Float32Array;
  /** Structured diagnostics for maps dropped because they target texCoord > 1. */
  readonly unsupportedTexCoordWarnings: readonly EngineWarning[];
}

export type MaterialTextureColorSpace = 'srgb' | 'linear';

export interface MaterialTextureLayerUse {
  readonly materialIndex: number;
  readonly field: string;
  readonly colorSpace: MaterialTextureColorSpace;
  readonly texCoord: number;
  readonly magFilter?: TextureFilterMode;
  readonly minFilter?: TextureFilterMode;
  readonly mipFilter?: TextureMipFilterMode;
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

function writeDefaultMipPolicies(descriptors: Float32Array, b: number): void {
  const start = b + MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET * 4;
  for (let i = 0; i < MATERIAL_TEX_MIP_POLICY_MAP_COUNT; i += 1) {
    descriptors[start + i] = MIP_FILTER_INDEX.linear;
  }
}

function writeMipPolicy(
  descriptors: Float32Array,
  b: number,
  mapSlot: number,
  ref: TextureRef | undefined,
): void {
  if (mapSlot < 0 || mapSlot >= MATERIAL_TEX_MIP_POLICY_MAP_COUNT) return;
  descriptors[b + MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET * 4 + mapSlot] =
    MIP_FILTER_INDEX[ref?.mipFilter ?? 'linear'];
}

function writeDefaultFilterPolicies(descriptors: Float32Array, b: number): void {
  const start = b + MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET * 4;
  const count = MATERIAL_TEX_MIP_POLICY_MAP_COUNT * MATERIAL_TEX_FILTER_POLICY_FLOATS_PER_MAP;
  for (let i = 0; i < count; i += 1) {
    descriptors[start + i] = 1;
  }
}

function writeFilterPolicy(
  descriptors: Float32Array,
  b: number,
  mapSlot: number,
  ref: TextureRef | undefined,
): void {
  if (mapSlot < 0 || mapSlot >= MATERIAL_TEX_MIP_POLICY_MAP_COUNT) return;
  const offset =
    b +
    MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET * 4 +
    mapSlot * MATERIAL_TEX_FILTER_POLICY_FLOATS_PER_MAP;
  descriptors[offset] = ref?.magFilter === 'nearest' ? 0 : 1;
  descriptors[offset + 1] = ref?.minFilter === 'nearest' ? 0 : 1;
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
  descriptors[vecBase] = safeTexCoord(ref);
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
  emissiveLayerScales: readonly MaterialTextureLayerUvScale[],
): void {
  const materialCount = Math.floor(descriptors.length / MATERIAL_TEX_FLOAT_STRIDE);
  for (let mi = 0; mi < materialCount; mi += 1) {
    const b = mi * MATERIAL_TEX_FLOAT_STRIDE;
    // sRGB array maps: baseColor. Emissive now lives in the dedicated rgba16float
    // emissive array, so its UV-fit scale reads from that array's layer scales.
    writeUvFitPair(descriptors, b + 28, uvFitScaleFor(sRgbLayerScales, descriptors[b + 0] ?? -1));
    writeUvFitPair(descriptors, b + 30, uvFitScaleFor(emissiveLayerScales, descriptors[b + 3] ?? -1));
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
  const emissiveSources: unknown[] = [];
  const unsupportedTexCoordWarnings: EngineWarning[] = [];
  const warnUnsupportedTexCoord = createUnsupportedTexCoordWarner(unsupportedTexCoordWarnings);
  const makeIndexer = (list: unknown[], colorSpace: MaterialTextureColorSpace) => {
    const handleToIdx = new Map<unknown, number>();
    const usesByLayer: MaterialTextureLayerUse[][] = [];
    const index = (ref: TextureRef | undefined, materialIndex: number, field: string): number => {
      const handle = ref?.handle;
      if (handle == null) return -1;
      if (warnUnsupportedTexCoord(materialIndex, field, ref)) return -1;
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
        texCoord: safeTexCoord(ref),
        ...(ref?.magFilter != null ? { magFilter: ref.magFilter } : {}),
        ...(ref?.minFilter != null ? { minFilter: ref.minFilter } : {}),
        ...(ref?.mipFilter != null ? { mipFilter: ref.mipFilter } : {}),
      });
      return i;
    };
    const infos = (): readonly MaterialTextureLayerInfo[] =>
      list.map((_, layer) => ({ layer, uses: usesByLayer[layer] ?? [] }));
    return { index, infos };
  };
  const sRgbIndexer = makeIndexer(sources, 'srgb');
  const linearIndexer = makeIndexer(linearSources, 'linear');
  // Emissive is authored sRGB but uploaded to a LINEAR rgba16float array (the CPU
  // upload path applies the sRGB decode). Its provenance colorSpace stays 'srgb'
  // (that is the authored encoding), while the layers live in a separate index
  // space from the sRGB baseColor array so HDR emissive survives packing.
  const emissiveIndexer = makeIndexer(emissiveSources, 'srgb');
  const indexOf = sRgbIndexer.index;          // sRGB array
  const indexOfLinear = linearIndexer.index;  // linear array
  const indexOfEmissive = emissiveIndexer.index; // emissive rgba16float array

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
    descriptors[b + 3] = indexOfEmissive(m.emissiveMap, mi, 'emissiveMap'); // emissiveIdx (dedicated rgba16float emissive array)
    descriptors[b + 4] = ALPHA_MODE_INDEX[m.alphaMode ?? 'opaque'];
    descriptors[b + 5] = m.alphaCutoff ?? 0.5;
    descriptors[b + 6] = m.opacity ?? 1;
    descriptors[b + 7] = safeTexCoord(bc);
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
    writeDefaultMipPolicies(descriptors, b);
    writeDefaultFilterPolicies(descriptors, b);
    // T2-A — drive the wrap / mip / filter / UV-metadata writes for all 23 maps
    // from the single ordered TEXTURE_MAP_SLOTS table. The slot index IS the
    // mip/filter policy slot; each entry carries its wrap + UV-metadata offsets.
    // This replaces four separate hand-written 23-map enumerations. The per-map
    // INDEX + scalar lanes above stay hand-written (they are not uniform per slot).
    const resolveContext: MaterialTextureResolveContext = { roughnessMap, metallicMap };
    TEXTURE_MAP_SLOTS.forEach((slot, slotIdx) => {
      const ref = slot.resolve(m, resolveContext);
      writeWrapPair(descriptors, b + slot.wrapFloatOffset, ref);
      writeMipPolicy(descriptors, b, slotIdx, ref);
      writeFilterPolicy(descriptors, b, slotIdx, ref);
      writeUvMeta(descriptors, b, slot.uvMetaSlot, ref, slot.uvMetaVec4Offset);
    });
  });

  return {
    sources,
    linearSources,
    emissiveSources,
    sourceInfos: sRgbIndexer.infos(),
    linearSourceInfos: linearIndexer.infos(),
    emissiveSourceInfos: emissiveIndexer.infos(),
    descriptors,
    unsupportedTexCoordWarnings,
  };
}
