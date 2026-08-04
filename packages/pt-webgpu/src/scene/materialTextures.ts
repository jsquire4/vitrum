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
import { packNonNegativeRadianceScalarF32 } from '@vitrum/shared-bvh';
import type { MaterialTextureLayerUvScale } from './materialTextureArray.js';
import { assertPtWebgpuEnvironmentScaleF32 } from '../environmentRadianceScale.js';

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
 *     their intensity / scale scalars. AO and bump are bounded LINEAR data in
 *     `materialTexturesLinear`; lightMap is outgoing radiance and therefore uses
 *     the dedicated linear-float radiance array shared with emissiveMap. A material
 *     lacking a given map carries index -1 (the WGSL sampler returns a no-op), so
 *     absent-field scenes stay byte-identical to the pre-D3 path.
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

const TEXTURE_MAP_SLOT_INDEX_BY_NAME: ReadonlyMap<string, number> = new Map(
  TEXTURE_MAP_SLOTS.map((slot, index) => [slot.name, index]),
);

export function stagedMapRef(
  material: StagedMaterialTextureInputs,
  name: string,
): StagedTextureRef | undefined {
  const index = TEXTURE_MAP_SLOT_INDEX_BY_NAME.get(name);
  return index == null ? undefined : material.refs[index];
}

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

export interface StagedTextureRef {
  readonly handle: unknown;
  readonly texCoord: number;
  readonly transform: PackedTextureTransform;
  readonly wrapS: TextureWrapMode;
  readonly wrapT: TextureWrapMode;
  readonly magFilter: TextureFilterMode;
  readonly minFilter: TextureFilterMode;
  readonly mipFilter: TextureMipFilterMode;
}

export interface StagedMaterialTextureInputs {
  readonly refs: readonly (StagedTextureRef | undefined)[];
  readonly alphaMode: MaterialSpec['alphaMode'];
  readonly alphaCutoff: MaterialSpec['alphaCutoff'];
  readonly opacity: MaterialSpec['opacity'];
  readonly aoMapIntensity: MaterialSpec['aoMapIntensity'];
  readonly lightMapIntensity: MaterialSpec['lightMapIntensity'];
  readonly bumpScale: MaterialSpec['bumpScale'];
  readonly envMapIntensity: MaterialSpec['envMapIntensity'];
  readonly anisotropy: MaterialSpec['anisotropy'];
  readonly anisotropyRotation: MaterialSpec['anisotropyRotation'];
  readonly normalScale: MaterialSpec['normalScale'];
  readonly clearcoatNormalScale: MaterialSpec['clearcoatNormalScale'];
  readonly transmission: MaterialSpec['transmission'];
  readonly metallic: MaterialSpec['metallic'];
  readonly roughness: MaterialSpec['roughness'];
  readonly thinFilmStackPresent: boolean;
  readonly clearcoat: MaterialSpec['clearcoat'];
  readonly sheen: MaterialSpec['sheen'];
  readonly frontLayerRoughness: number | undefined;
  readonly backLayerRoughness: number | undefined;
  readonly frontLayerNormalScale: number | undefined;
  readonly backLayerNormalScale: number | undefined;
}

function authoredTexCoord(ref: StagedTextureRef | undefined): number {
  return ref?.texCoord ?? 0;
}

interface MaterialUvSetLayout {
  /** GPU UV slot -> authored TextureRef.texCoord. Slots 0/1 are ABI-stable. */
  readonly texCoords: readonly number[];
  readonly slotByTexCoord: ReadonlyMap<number, number>;
}

function materialUvSetLayout(
  materials: readonly StagedMaterialTextureInputs[],
): MaterialUvSetLayout {
  const used = new Set<number>([0, 1]);
  materials.forEach((material, materialIndex) => {
    for (let slotIndex = 0; slotIndex < TEXTURE_MAP_SLOTS.length; slotIndex += 1) {
      const slot = TEXTURE_MAP_SLOTS[slotIndex]!;
      const ref = material.refs[slotIndex];
      if (ref?.handle == null) continue;
      const texCoord = authoredTexCoord(ref);
      if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
        throw new RangeError(
          `collectMaterialTextures: material ${materialIndex} ${slot.name}.texCoord ` +
          `must be a non-negative safe integer (got ${String(texCoord)}).`,
        );
      }
      used.add(texCoord);
    }
  });
  const texCoords = [0, 1, ...[...used].filter((value) => value > 1).sort((a, b) => a - b)];
  return {
    texCoords,
    slotByTexCoord: new Map(texCoords.map((texCoord, slot) => [texCoord, slot])),
  };
}

function packedUvSlot(
  ref: StagedTextureRef | undefined,
  layout: MaterialUvSetLayout,
): number {
  if (ref?.handle == null) return 0;
  return layout.slotByTexCoord.get(authoredTexCoord(ref)) ?? 0;
}

interface PackedTextureTransform {
  readonly offsetX: number;
  readonly offsetY: number;
  readonly scaleX: number;
  readonly scaleY: number;
  readonly rotation: number;
}

function requireTextureTransformFloat32(
  value: number,
  context: string,
): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${context} must be finite.`);
  }
  const packed = Math.fround(value);
  if (!Number.isFinite(packed)) {
    throw new RangeError(`${context} overflows WebGPU float32 storage.`);
  }
  if (value !== 0 && packed === 0) {
    throw new RangeError(`${context} underflows WebGPU float32 storage.`);
  }
  return packed;
}

function snapshotPackedTextureTransform(
  ref: TextureRef | undefined,
  context: string,
): PackedTextureTransform {
  const transform = ref?.transform;
  const offset = transform?.offset;
  const scale = transform?.scale;
  const offsetX = offset?.[0];
  const offsetY = offset?.[1];
  const scaleX = scale?.[0];
  const scaleY = scale?.[1];
  const rotation = transform?.rotation;
  return {
    offsetX: requireTextureTransformFloat32(
      offsetX ?? 0,
      `${context} offset.x`,
    ),
    offsetY: requireTextureTransformFloat32(
      offsetY ?? 0,
      `${context} offset.y`,
    ),
    scaleX: requireTextureTransformFloat32(
      scaleX ?? 1,
      `${context} scale.x`,
    ),
    scaleY: requireTextureTransformFloat32(
      scaleY ?? 1,
      `${context} scale.y`,
    ),
    rotation: requireTextureTransformFloat32(
      rotation ?? 0,
      `${context} rotation`,
    ),
  };
}

function packedTextureTransform(
  ref: StagedTextureRef | undefined,
  _context: string,
): PackedTextureTransform {
  return ref?.transform ?? {
    offsetX: 0,
    offsetY: 0,
    scaleX: 1,
    scaleY: 1,
    rotation: 0,
  };
}

function snapshotTextureRef(
  input: TextureRef | undefined,
  context: string,
  cache: Map<unknown, StagedTextureRef>,
): StagedTextureRef | undefined {
  if (input == null) return undefined;
  if (typeof input !== 'object') {
    throw new TypeError(`${context} must be a TextureRef object.`);
  }
  const cached = cache.get(input);
  if (cached != null) return cached;
  const handle = input.handle;
  const texCoordInput = input.texCoord;
  const transform = snapshotPackedTextureTransform(input, `${context} transform`);
  const wrapSInput = input.wrapS;
  const wrapTInput = input.wrapT;
  const magFilterInput = input.magFilter;
  const minFilterInput = input.minFilter;
  const mipFilterInput = input.mipFilter;
  const texCoord = texCoordInput ?? 0;
  const wrapS = wrapSInput ?? 'repeat';
  const wrapT = wrapTInput ?? 'repeat';
  const magFilter = magFilterInput ?? 'linear';
  const minFilter = minFilterInput ?? 'linear';
  const mipFilter = mipFilterInput ?? 'linear';
  if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
    throw new RangeError(`${context}.texCoord must be a non-negative safe integer.`);
  }
  if (!Object.hasOwn(WRAP_MODE_INDEX, wrapS) || !Object.hasOwn(WRAP_MODE_INDEX, wrapT)) {
    throw new RangeError(`${context} has an unsupported wrap mode.`);
  }
  if (
    (magFilter !== 'nearest' && magFilter !== 'linear') ||
    (minFilter !== 'nearest' && minFilter !== 'linear')
  ) {
    throw new RangeError(`${context} has an unsupported texture filter.`);
  }
  if (mipFilter !== 'none' && mipFilter !== 'nearest' && mipFilter !== 'linear') {
    throw new RangeError(`${context} has an unsupported mip filter.`);
  }
  const snapshot = Object.freeze({
    handle,
    texCoord,
    transform: Object.freeze(transform),
    wrapS,
    wrapT,
    magFilter,
    minFilter,
    mipFilter,
  });
  cache.set(input, snapshot);
  return snapshot;
}

interface MaterialInputSnapshotState {
  readonly materialCache: Map<MaterialSpec, StagedMaterialTextureInputs>;
  readonly refCache: Map<unknown, StagedTextureRef>;
}

/** Opaque operation-scoped token that makes every material/TextureRef accessor
 * observation reusable by MNEE admission, descriptor collection, and atlas
 * staging. The mutable identity caches remain module-private. */
export interface MaterialInputSnapshotContext {
  readonly kind: 'pt-webgpu-material-input-snapshot';
}

const MATERIAL_INPUT_SNAPSHOT_STATES = new WeakMap<
  MaterialInputSnapshotContext,
  MaterialInputSnapshotState
>();

export function createMaterialInputSnapshotContext(): MaterialInputSnapshotContext {
  const context = Object.freeze({
    kind: 'pt-webgpu-material-input-snapshot' as const,
  });
  MATERIAL_INPUT_SNAPSHOT_STATES.set(context, {
    materialCache: new Map<MaterialSpec, StagedMaterialTextureInputs>(),
    refCache: new Map<unknown, StagedTextureRef>(),
  });
  return context;
}

function materialInputSnapshotState(
  context: MaterialInputSnapshotContext,
): MaterialInputSnapshotState {
  const state = MATERIAL_INPUT_SNAPSHOT_STATES.get(context);
  if (state == null) {
    throw new TypeError('collectMaterialTextures: invalid material snapshot context.');
  }
  return state;
}

export function snapshotMaterialTextureInputs(
  materials: ReadonlyArray<MaterialSpec>,
  context: MaterialInputSnapshotContext = createMaterialInputSnapshotContext(),
): readonly StagedMaterialTextureInputs[] {
  if (materials == null || typeof materials !== 'object') {
    throw new TypeError('collectMaterialTextures: materials must be array-like.');
  }
  const materialCount = materials.length;
  if (!Number.isSafeInteger(materialCount) || materialCount < 0) {
    throw new RangeError('collectMaterialTextures: materials.length must be a non-negative safe integer.');
  }
  const descriptorLength = materialCount * MATERIAL_TEX_FLOAT_STRIDE;
  if (!Number.isSafeInteger(descriptorLength)) {
    throw new RangeError('collectMaterialTextures: descriptor length exceeds safe integer range.');
  }
  const state = materialInputSnapshotState(context);
  const { refCache, materialCache } = state;
  const snapshots = new Array<StagedMaterialTextureInputs>(materialCount);
  for (let materialIndex = 0; materialIndex < materialCount; materialIndex += 1) {
    const material = materials[materialIndex];
    if (material == null || typeof material !== 'object') {
      throw new TypeError(`collectMaterialTextures: material ${materialIndex} must be an object.`);
    }
    const cachedMaterial = materialCache.get(material);
    if (cachedMaterial != null) {
      snapshots[materialIndex] = cachedMaterial;
      continue;
    }
    const ref = (value: TextureRef | undefined, field: string): StagedTextureRef | undefined =>
      snapshotTextureRef(
        value,
        `collectMaterialTextures: material ${materialIndex} ${field}`,
        refCache,
      );
    const baseColorMap = ref(material.baseColorMap, 'baseColorMap');
    const emissiveMap = ref(material.emissiveMap, 'emissiveMap');
    const normalMap = ref(material.normalMap, 'normalMap');
    const roughnessMapAuthored = ref(material.roughnessMap, 'roughnessMap');
    const metallicMapAuthored = ref(material.metallicMap, 'metallicMap');
    const roughnessMap = roughnessMapAuthored ?? metallicMapAuthored;
    const metallicMap = metallicMapAuthored ?? roughnessMapAuthored;
    const aoMap = ref(material.aoMap, 'aoMap');
    const lightMap = ref(material.lightMap, 'lightMap');
    const bumpMap = ref(material.bumpMap, 'bumpMap');
    const anisotropyMap = ref(material.anisotropyMap, 'anisotropyMap');
    const alphaMap = ref(material.alphaMap, 'alphaMap');
    const transmissionMap = ref(material.transmissionMap, 'transmissionMap');
    const clearcoatMap = ref(material.clearcoatMap, 'clearcoatMap');
    const clearcoatRoughnessMap = ref(
      material.clearcoatRoughnessMap,
      'clearcoatRoughnessMap',
    );
    const sheenColorMap = ref(material.sheenColorMap, 'sheenColorMap');
    const sheenRoughnessMap = ref(material.sheenRoughnessMap, 'sheenRoughnessMap');
    const iridescenceMap = ref(material.iridescenceMap, 'iridescenceMap');
    const iridescenceThicknessMap = ref(
      material.iridescenceThicknessMap,
      'iridescenceThicknessMap',
    );
    const specularColorMap = ref(material.specularColorMap, 'specularColorMap');
    const specularIntensityMap = ref(
      material.specularIntensityMap,
      'specularIntensityMap',
    );
    const clearcoatNormalMap = ref(material.clearcoatNormalMap, 'clearcoatNormalMap');
    const thicknessMap = ref(material.thicknessMap, 'thicknessMap');
    const frontLayer = material.frontLayer;
    if (frontLayer != null && typeof frontLayer !== 'object') {
      throw new TypeError(`collectMaterialTextures: material ${materialIndex} frontLayer must be an object.`);
    }
    const frontLayerNormalMap = ref(frontLayer?.normalMap, 'frontLayer.normalMap');
    const frontLayerRoughness = frontLayer?.roughness;
    const frontLayerNormalScale = frontLayer?.normalScale;
    const backLayer = material.backLayer;
    if (backLayer != null && typeof backLayer !== 'object') {
      throw new TypeError(`collectMaterialTextures: material ${materialIndex} backLayer must be an object.`);
    }
    const backLayerNormalMap = ref(backLayer?.normalMap, 'backLayer.normalMap');
    const backLayerRoughness = backLayer?.roughness;
    const backLayerNormalScale = backLayer?.normalScale;
    const refs = Object.freeze([
      baseColorMap,
      emissiveMap,
      normalMap,
      roughnessMap,
      metallicMap,
      aoMap,
      lightMap,
      bumpMap,
      anisotropyMap,
      alphaMap,
      transmissionMap,
      clearcoatMap,
      clearcoatRoughnessMap,
      sheenColorMap,
      sheenRoughnessMap,
      iridescenceMap,
      iridescenceThicknessMap,
      specularColorMap,
      specularIntensityMap,
      clearcoatNormalMap,
      thicknessMap,
      frontLayerNormalMap,
      backLayerNormalMap,
    ]);
    const snapshot = Object.freeze({
      refs,
      alphaMode: material.alphaMode,
      alphaCutoff: material.alphaCutoff,
      opacity: material.opacity,
      aoMapIntensity: material.aoMapIntensity,
      lightMapIntensity: material.lightMapIntensity,
      bumpScale: material.bumpScale,
      envMapIntensity: material.envMapIntensity,
      anisotropy: material.anisotropy,
      anisotropyRotation: material.anisotropyRotation,
      normalScale: material.normalScale,
      clearcoatNormalScale: material.clearcoatNormalScale,
      transmission: material.transmission,
      metallic: material.metallic,
      roughness: material.roughness,
      thinFilmStackPresent: material.thinFilmStack != null,
      clearcoat: material.clearcoat,
      sheen: material.sheen,
      frontLayerRoughness,
      backLayerRoughness,
      frontLayerNormalScale,
      backLayerNormalScale,
    });
    materialCache.set(material, snapshot);
    snapshots[materialIndex] = snapshot;
  }
  return Object.freeze(snapshots);
}

export function snapshotMaterialTextureInput(
  material: MaterialSpec,
  context: MaterialInputSnapshotContext,
): StagedMaterialTextureInputs {
  return snapshotMaterialTextureInputs([material], context)[0]!;
}

export interface CollectedTextures {
  /** Unique sRGB-decoded texture sources (baseColor + extension color-tint maps),
   *  upload order. Emissive is NO LONGER here — it has its own rgba16float array
   *  (see {@link emissiveSources}) so HDR emissive values survive packing. */
  readonly sources: unknown[];
  /** Unique LINEAR texture sources (normal + scalar/data maps — must NOT be sRGB-decoded),
   *  a separate index space → its own texture_2d_array. */
  readonly linearSources: unknown[];
  /** Unique outgoing-RADIANCE texture sources (emissiveMap + lightMap) → a
   *  dedicated rgba16float texture_2d_array. The historical property name is
   *  retained for ABI stability. Deduplication keys include the source transfer
   *  domain: emissive LDR bytes are sRGB-decoded, while light-map bytes are linear.
   *  Float32 payloads are always treated as already-linear HDR radiance. */
  readonly emissiveSources: unknown[];
  /** Source-layer provenance for host-facing upload diagnostics. */
  readonly sourceInfos: readonly MaterialTextureLayerInfo[];
  /** Linear source-layer provenance for host-facing upload diagnostics. */
  readonly linearSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Outgoing-radiance source-layer provenance for host-facing diagnostics. */
  readonly emissiveSourceInfos: readonly MaterialTextureLayerInfo[];
  /** Per-material descriptor floats (MATERIAL_TEX_FLOAT_STRIDE per material). */
  readonly descriptors: Float32Array;
  /** Compatibility lane; arbitrary texCoord maps are no longer dropped. */
  readonly unsupportedTexCoordWarnings: readonly EngineWarning[];
  /** Compact GPU UV-slot layout. Entry value is the authored texCoord index. */
  readonly uvSetTexCoords: readonly number[];
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

/** Upload/decode profiles that may safely share one physical array layer. */
export type MaterialTextureRoleProfile =
  | 'color'
  | 'scalar'
  | 'alpha-scalar'
  | 'normal'
  | 'anisotropy'
  | 'radiance';

/**
 * Return the source interpretation required by a material-map role. Handles
 * are deduplicated only within one profile: sharing a host object must never
 * make a tangent-space normal silently double as scalar coverage, or make an
 * authored-alpha map inherit an opaque alpha synthesized for another role.
 */
export function materialTextureRoleProfile(field: string): MaterialTextureRoleProfile {
  if (
    field === 'normalMap' ||
    field === 'clearcoatNormalMap' ||
    field === 'frontLayer.normalMap' ||
    field === 'backLayer.normalMap'
  ) {
    return 'normal';
  }
  if (field === 'anisotropyMap') return 'anisotropy';
  if (field === 'sheenRoughnessMap' || field === 'specularIntensityMap') {
    return 'alpha-scalar';
  }
  if (field === 'emissiveMap' || field === 'lightMap') return 'radiance';
  if (
    field === 'baseColorMap' ||
    field === 'sheenColorMap' ||
    field === 'specularColorMap'
  ) {
    return 'color';
  }
  return 'scalar';
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
  ref: StagedTextureRef | undefined,
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
  ref: StagedTextureRef | undefined,
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
  ref: StagedTextureRef | undefined,
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
  ref: StagedTextureRef | undefined,
  uvLayout: MaterialUvSetLayout,
  context: string,
  metaVec4Offset = MATERIAL_TEX_UV_META_VEC4_OFFSET,
): void {
  const vecBase = b + (metaVec4Offset + mapSlot * MATERIAL_TEX_UV_META_VEC4S_PER_MAP) * 4;
  const transform = packedTextureTransform(ref, context);
  descriptors[vecBase] = packedUvSlot(ref, uvLayout);
  descriptors[vecBase + 1] = transform.offsetX;
  descriptors[vecBase + 2] = transform.offsetY;
  descriptors[vecBase + 3] = transform.rotation;
  descriptors[vecBase + 4] = transform.scaleX;
  descriptors[vecBase + 5] = transform.scaleY;
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
    // sRGB array maps: baseColor. Outgoing-radiance maps live in the dedicated
    // rgba16float array, so both emissive and light-map UV fits read from it.
    writeUvFitPair(descriptors, b + 28, uvFitScaleFor(sRgbLayerScales, descriptors[b + 0] ?? -1));
    writeUvFitPair(descriptors, b + 30, uvFitScaleFor(emissiveLayerScales, descriptors[b + 3] ?? -1));
    // Linear array maps: normal, roughness, metallic, AO, bump, anisotropy, alpha, transmission.
    writeUvFitPair(descriptors, b + 32, uvFitScaleFor(linearLayerScales, descriptors[b + 1] ?? -1));
    writeUvFitPair(descriptors, b + 34, uvFitScaleFor(linearLayerScales, descriptors[b + 2] ?? -1));
    writeUvFitPair(descriptors, b + 36, uvFitScaleFor(linearLayerScales, descriptors[b + 26] ?? -1));
    writeUvFitPair(descriptors, b + 38, uvFitScaleFor(linearLayerScales, descriptors[b + 13] ?? -1));
    writeUvFitPair(descriptors, b + 40, uvFitScaleFor(emissiveLayerScales, descriptors[b + 14] ?? -1));
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
 *  Three GPU arrays preserve the authored sample domains: sRGB color maps,
 *  linear data maps, and linear-float outgoing-radiance maps. */
export function collectMaterialTextures(
  materials: ReadonlyArray<MaterialSpec>,
  snapshotContext: MaterialInputSnapshotContext = createMaterialInputSnapshotContext(),
): CollectedTextures {
  const stagedMaterials = snapshotMaterialTextureInputs(materials, snapshotContext);
  const sources: unknown[] = [];
  const linearSources: unknown[] = [];
  const emissiveSources: unknown[] = [];
  const uvLayout = materialUvSetLayout(stagedMaterials);
  const makeIndexer = (list: unknown[], colorSpace: MaterialTextureColorSpace) => {
    const handleToIdxByProfile = new Map<MaterialTextureRoleProfile, Map<unknown, number>>();
    const usesByLayer: MaterialTextureLayerUse[][] = [];
    const index = (ref: StagedTextureRef | undefined, materialIndex: number, field: string): number => {
      const handle = ref?.handle;
      if (handle == null) return -1;
      const profile = materialTextureRoleProfile(field);
      let handleToIdx = handleToIdxByProfile.get(profile);
      if (handleToIdx == null) {
        handleToIdx = new Map<unknown, number>();
        handleToIdxByProfile.set(profile, handleToIdx);
      }
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
        texCoord: authoredTexCoord(ref),
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
  // Emissive bytes conventionally carry sRGB-encoded colour, whereas light-map
  // bytes are defined by the core contract as linear outgoing radiance. The same
  // object used in both roles must therefore occupy two independently converted
  // layers. Float32 sources remain linear in either role at upload time.
  const radianceHandleToIdx: Readonly<Record<MaterialTextureColorSpace, Map<unknown, number>>> = {
    srgb: new Map<unknown, number>(),
    linear: new Map<unknown, number>(),
  };
  const radianceUsesByLayer: MaterialTextureLayerUse[][] = [];
  const indexOfRadiance = (
    ref: StagedTextureRef | undefined,
    materialIndex: number,
    field: string,
    colorSpace: MaterialTextureColorSpace,
  ): number => {
    const handle = ref?.handle;
    if (handle == null) return -1;
    const handleToIdx = radianceHandleToIdx[colorSpace];
    let i = handleToIdx.get(handle);
    if (i === undefined) {
      i = emissiveSources.length;
      emissiveSources.push(handle);
      handleToIdx.set(handle, i);
    }
    (radianceUsesByLayer[i] ??= []).push({
      materialIndex,
      field,
      colorSpace,
      texCoord: authoredTexCoord(ref),
      ...(ref?.magFilter != null ? { magFilter: ref.magFilter } : {}),
      ...(ref?.minFilter != null ? { minFilter: ref.minFilter } : {}),
      ...(ref?.mipFilter != null ? { mipFilter: ref.mipFilter } : {}),
    });
    return i;
  };
  const indexOf = sRgbIndexer.index;          // sRGB color array
  const indexOfLinear = linearIndexer.index;  // linear array
  const indexOfEmissive = (ref: StagedTextureRef | undefined, materialIndex: number, field: string): number =>
    indexOfRadiance(ref, materialIndex, field, 'srgb');
  const indexOfLightMap = (ref: StagedTextureRef | undefined, materialIndex: number, field: string): number =>
    indexOfRadiance(ref, materialIndex, field, 'linear');

  const descriptors = new Float32Array(stagedMaterials.length * MATERIAL_TEX_FLOAT_STRIDE);
  stagedMaterials.forEach((m, mi) => {
    const b = mi * MATERIAL_TEX_FLOAT_STRIDE;
    const bc = stagedMapRef(m, 'baseColor');
    // glTF's canonical metallicRoughness texture is one image (G=roughness,
    // B=metallic). Preserve that combined-map behavior when only one side is
    // supplied, but allow distinct authored maps to carry independent UV/wrap.
    const roughnessMap = stagedMapRef(m, 'roughness');
    const metallicMap = stagedMapRef(m, 'metallic');
    descriptors[b + 0] = indexOf(bc, mi, 'baseColorMap');            // baseColorIdx (sRGB array)
    descriptors[b + 1] = indexOfLinear(stagedMapRef(m, 'normal'), mi, 'normalMap');  // normalIdx (linear array)
    descriptors[b + 2] = indexOfLinear(roughnessMap, mi, 'roughnessMap'); // roughness map (linear; glTF G channel)
    descriptors[b + 3] = indexOfEmissive(stagedMapRef(m, 'emissive'), mi, 'emissiveMap'); // emissiveIdx
    descriptors[b + 4] = ALPHA_MODE_INDEX[m.alphaMode ?? 'opaque'];
    descriptors[b + 5] = m.alphaCutoff ?? 0.5;
    descriptors[b + 6] = m.opacity ?? 1;
    descriptors[b + 7] = packedUvSlot(bc, uvLayout);
    const baseTransform = packedTextureTransform(
      bc,
      `collectMaterialTextures: material ${mi} baseColor transform`,
    );
    descriptors[b + 8] = baseTransform.offsetX;
    descriptors[b + 9] = baseTransform.offsetY;
    descriptors[b + 10] = baseTransform.scaleX;
    descriptors[b + 11] = baseTransform.scaleY;
    descriptors[b + 12] = baseTransform.rotation;
    // D3 — AO/bump are bounded linear data. lightMap is unbounded outgoing
    // radiance and shares the dedicated rgba16float array with emissiveMap.
    // Index -1 when absent → the WGSL sampler is a no-op.
    descriptors[b + 13] = indexOfLinear(stagedMapRef(m, 'ao'), mi, 'aoMap');
    descriptors[b + 14] = indexOfLightMap(stagedMapRef(m, 'lightMap'), mi, 'lightMap');
    descriptors[b + 15] = indexOfLinear(stagedMapRef(m, 'bump'), mi, 'bumpMap');
    descriptors[b + 16] = m.aoMapIntensity ?? 1;
    descriptors[b + 17] = packNonNegativeRadianceScalarF32(
      m.lightMapIntensity ?? 1,
      `@vitrum/pt-webgpu material ${mi} lightMapIntensity`,
    );
    descriptors[b + 18] = m.bumpScale ?? 1;
    descriptors[b + 19] = assertPtWebgpuEnvironmentScaleF32(
      m.envMapIntensity ?? 1,
      `material ${mi} envMapIntensity`,
    );
    // D3 — vec4 #5: anisotropy scalars + optional KHR_materials_anisotropy map.
    descriptors[b + 20] = m.anisotropy ?? 0;
    descriptors[b + 21] = m.anisotropyRotation ?? 0;
    descriptors[b + 22] = indexOfLinear(stagedMapRef(m, 'anisotropy'), mi, 'anisotropyMap');
    descriptors[b + 23] = m.normalScale ?? 1;
    // Standalone alphaMap is coverage data (linear). BaseColor alpha still
    // participates too; each map carries its own UV metadata below.
    descriptors[b + 24] = indexOfLinear(stagedMapRef(m, 'alpha'), mi, 'alphaMap');
    descriptors[b + 25] = indexOfLinear(stagedMapRef(m, 'transmission'), mi, 'transmissionMap');
    descriptors[b + 26] = indexOfLinear(metallicMap, mi, 'metallicMap'); // metallic map (linear; glTF B channel)
    descriptors[b + 27] = 0;
    // Extension-lobe texture maps. Color tint maps use the sRGB array; scalar
    // factor/roughness/thickness maps use the LINEAR array.
    const extIndexBase = b + MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET * 4;
    descriptors[extIndexBase] = indexOfLinear(stagedMapRef(m, 'clearcoat'), mi, 'clearcoatMap');
    descriptors[extIndexBase + 1] = indexOfLinear(
      stagedMapRef(m, 'clearcoatRoughness'),
      mi,
      'clearcoatRoughnessMap',
    );
    descriptors[extIndexBase + 2] = indexOf(stagedMapRef(m, 'sheenColor'), mi, 'sheenColorMap');
    descriptors[extIndexBase + 3] = indexOfLinear(
      stagedMapRef(m, 'sheenRoughness'),
      mi,
      'sheenRoughnessMap',
    );
    descriptors[extIndexBase + 4] = indexOfLinear(
      stagedMapRef(m, 'iridescence'),
      mi,
      'iridescenceMap',
    );
    descriptors[extIndexBase + 5] = indexOfLinear(
      stagedMapRef(m, 'iridescenceThickness'),
      mi,
      'iridescenceThicknessMap',
    );
    descriptors[extIndexBase + 6] = indexOf(
      stagedMapRef(m, 'specularColor'),
      mi,
      'specularColorMap',
    );
    descriptors[extIndexBase + 7] = indexOfLinear(
      stagedMapRef(m, 'specularIntensity'),
      mi,
      'specularIntensityMap',
    );
    const clearcoatNormalBase = b + MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET * 4;
    descriptors[clearcoatNormalBase] = indexOfLinear(
      stagedMapRef(m, 'clearcoatNormal'),
      mi,
      'clearcoatNormalMap',
    );
    descriptors[clearcoatNormalBase + 1] = m.clearcoatNormalScale ?? 1;
    descriptors[clearcoatNormalBase + 2] = 1;
    descriptors[clearcoatNormalBase + 3] = 1;
    const thicknessBase = b + MATERIAL_TEX_THICKNESS_VEC4_OFFSET * 4;
    descriptors[thicknessBase] = indexOfLinear(stagedMapRef(m, 'thickness'), mi, 'thicknessMap');
    descriptors[thicknessBase + 1] = 1;
    descriptors[thicknessBase + 2] = 1;
    descriptors[thicknessBase + 3] = 0;
    const layerNormalBase = b + MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET * 4;
    descriptors[layerNormalBase] = indexOfLinear(
      stagedMapRef(m, 'frontLayerNormal'),
      mi,
      'frontLayer.normalMap',
    );
    descriptors[layerNormalBase + 1] = m.frontLayerNormalScale ?? 1;
    descriptors[layerNormalBase + 2] = indexOfLinear(
      stagedMapRef(m, 'backLayerNormal'),
      mi,
      'backLayer.normalMap',
    );
    descriptors[layerNormalBase + 3] = m.backLayerNormalScale ?? 1;
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
    TEXTURE_MAP_SLOTS.forEach((slot, slotIdx) => {
      const ref = m.refs[slotIdx];
      writeWrapPair(descriptors, b + slot.wrapFloatOffset, ref);
      writeMipPolicy(descriptors, b, slotIdx, ref);
      writeFilterPolicy(descriptors, b, slotIdx, ref);
      writeUvMeta(
        descriptors,
        b,
        slot.uvMetaSlot,
        ref,
        uvLayout,
        `collectMaterialTextures: material ${mi} ${slot.name} transform`,
        slot.uvMetaVec4Offset,
      );
    });
  });

  return {
    sources,
    linearSources,
    emissiveSources,
    sourceInfos: sRgbIndexer.infos(),
    linearSourceInfos: linearIndexer.infos(),
    emissiveSourceInfos: emissiveSources.map((_, layer) => ({
      layer,
      uses: radianceUsesByLayer[layer] ?? [],
    })),
    descriptors,
    unsupportedTexCoordWarnings: [],
    uvSetTexCoords: uvLayout.texCoords,
  };
}
