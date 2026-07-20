// materialsTexture — THREE-free port of three-gpu-pathtracer's `MaterialsTexture.js`,
// driven from a `@vitrum/core` `MaterialSpec` (NOT a THREE material).
//
// Reproduces the absorbed fork's material texture layout plus Vitrum's extended
// material-map slots as an RGBA32F-per-material record that the GLSL
// `material_struct` decoder reads texel-for-texel.
//
// The GLSL decoder and this packer must move together: any stride or offset
// divergence is a render bug. The packer is intentionally GPU-free and pure
// (Float32Array in → Float32Array out) so it can be golden-tested CPU-side like
// pt-webgpu's `materialPackingCoreEquivalence`.
//
// Provenance: gkjohnson/three-gpu-pathtracer (MIT) — `MaterialsTexture.js`.
// CREDITS.md attributes the absorbed fork.

import type {
  EngineWarning,
  MaterialSpec,
  TextureFilterMode,
  TextureMipFilterMode,
  TextureWrapMode,
  Vec3,
} from '@vitrum/core';
import { rgbToSpectralCoefficients } from '@vitrum/shared-samplers';
import type { MaterialsTextureData } from './sceneTextures.js';
import type { TextureAtlasLayerMap, TextureSampleColorSpace } from './texturesArray.js';

import {
  MATERIAL_MAP_FIELD_ORDER,
  MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
  MATERIAL_PIXELS,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
  MATERIAL_WRAP_TEXEL_OFFSET,
  UV_SET_BIT,
  MATERIAL_TRANSFORM_TEXEL,
  MATERIAL_D3_AUX_TEXEL,
  MATERIAL_AO_TRANSFORM_TEXEL,
  MATERIAL_LIGHTMAP_TRANSFORM_TEXEL,
  MATERIAL_BUMP_TRANSFORM_TEXEL,
  MATERIAL_ALPHA_TRANSFORM_TEXEL,
  MATERIAL_ANISOTROPY_TRANSFORM_TEXEL,
  MATERIAL_VOLUME_THICKNESS_TEXEL,
  MATERIAL_THICKNESS_TRANSFORM_TEXEL,
} from '../glsl/shader/structs/materialStride.js';

/** Pixels (RGBA32F texels) per material — single-sourced with every GLSL fetch
 *  site via `materialStride.js` (fork base layout 85 + D3 ao/light/bump/env
 *  texels 85..92 + alphaMap transform texels 93..94 + anisotropyMap transform
 *  texels 95..96 + thickness payload/transform texels 97..99 + wrap texels
 *  sampler-policy texels 100..120 + spectral reflectance texel 121). Re-exported
 *  for tests and parity guards. */
export {
  MATERIAL_MAP_FIELD_ORDER,
  MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
  MATERIAL_PIXELS,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
  MATERIAL_WRAP_TEXEL_OFFSET,
};
/** Floats per material (MATERIAL_PIXELS px × 4 channels). */
const MATERIAL_STRIDE = MATERIAL_PIXELS * 4;

type PackedMaterialSpec = MaterialSpec & {
  readonly castShadow?: boolean;
  readonly vertexColors?: boolean;
  readonly meshEmitterCastShadowDisabled?: boolean;
};

/** TRANSLUCENT_BIT — flag (s14.a) bit set for intrinsically scattering media. */
const TRANSLUCENT_BIT = 1 << 4;
/** UNLIT_BIT — flag (s14.a) bit set for terminal base-color unlit shading. */
const UNLIT_BIT = 1 << 5;
/** MESH_EMITTER_CAST_SHADOW_DISABLED_BIT — folded mesh-area emitter skips forward BSDF emission. */
const MESH_EMITTER_CAST_SHADOW_DISABLED_BIT = 1 << 6;

// Dispersion: Abbe → strength, evaluated at the Fraunhofer C/F lines.
const FRAUNHOFER_C_NM = 656.3;
const FRAUNHOFER_F_NM = 486.1;

// Uniform-grid spectral attenuation μ(λ): 32 samples, 380→780 nm inclusive.
const SPECTRAL_GRID_SAMPLE_COUNT = 32;
const SPECTRAL_GRID_START_NM = 380.0;
const SPECTRAL_GRID_END_NM = 780.0;

// Thin-film stack: 35 layers × [ior, thicknessNm, extinction].
const THIN_FILM_LAYER_LIMIT = 35;
const ATTENUATION_TRANSMITTANCE_EPSILON = 1e-4;
const MEDIUM_EPSILON = 1e-6;

/** ceil(sqrt(n)) — the square dimension that holds `n` texels row-major (mirrors fork). */
function squareDim(texelCount: number): number {
  return Math.max(1, Math.ceil(Math.sqrt(Math.max(1, texelCount))));
}

function finiteOr(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function nonNegativeFinite(value: number | undefined, fallback = 0.0): number {
  return Math.max(0.0, finiteOr(value, fallback));
}

function sigmaAFromAttenuation(attenuationColor: Vec3, attenuationDistance: number): Vec3 {
  if (!Number.isFinite(attenuationDistance) || attenuationDistance <= 0.0) {
    return [0.0, 0.0, 0.0];
  }
  const sigmaAChannel = (channel: number): number => {
    const transmittance = Math.min(
      Math.max(finiteOr(channel, 1.0), ATTENUATION_TRANSMITTANCE_EPSILON),
      1.0,
    );
    return Math.max(-Math.log(transmittance) / attenuationDistance, 0.0);
  };
  return [
    sigmaAChannel(attenuationColor[0]),
    sigmaAChannel(attenuationColor[1]),
    sigmaAChannel(attenuationColor[2]),
  ];
}

function resolveSssMedium(
  m: PackedMaterialSpec,
  attenuationColor: Vec3,
  attenuationDistance: number,
): { readonly active: boolean; readonly sigmaTMax: number; readonly sigmaS: Vec3 } {
  const scalarSigmaS = nonNegativeFinite(m.scatteringCoefficient, 0.0);
  const sigmaS: Vec3 = m.scatteringCoefficientRGB
    ? [
        nonNegativeFinite(m.scatteringCoefficientRGB[0], 0.0),
        nonNegativeFinite(m.scatteringCoefficientRGB[1], 0.0),
        nonNegativeFinite(m.scatteringCoefficientRGB[2], 0.0),
      ]
    : [scalarSigmaS, scalarSigmaS, scalarSigmaS];
  const sigmaA = sigmaAFromAttenuation(attenuationColor, attenuationDistance);
  const sigmaTMax = Math.max(
    sigmaA[0] + sigmaS[0],
    sigmaA[1] + sigmaS[1],
    sigmaA[2] + sigmaS[2],
  );
  const sigmaSMax = Math.max(sigmaS[0], sigmaS[1], sigmaS[2]);
  return {
    active: sigmaSMax > MEDIUM_EPSILON,
    sigmaTMax: Math.max(sigmaTMax, 0.0),
    sigmaS,
  };
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
 * decodes via `int(round(...))`. Atlas-backed maps pass a layer lookup into this
 * packer; absent, unreadable, or unmapped handles keep the sentinel.
 */
const _NO_TEXTURE = -1;

/** Default attenuation color when `attenuationColor` is absent — fork default (1,1,1). */
const DEFAULT_ATTENUATION_COLOR: Vec3 = [1.0, 1.0, 1.0];
/** Default specular color — fork default (1,1,1). */
const DEFAULT_SPECULAR_COLOR: Vec3 = [1.0, 1.0, 1.0];

const WRAP_MODE_INDEX: Readonly<Record<TextureWrapMode, number>> = {
  repeat: 0,
  'clamp-to-edge': 1,
  'mirrored-repeat': 2,
};

const FILTER_MODE_INDEX: Readonly<Record<TextureFilterMode, number>> = {
  nearest: 0,
  linear: 1,
};

const MIP_FILTER_INDEX: Readonly<Record<TextureMipFilterMode, number>> = {
  none: 0,
  nearest: 1,
  linear: 2,
};

/**
 * Pack a list of core `MaterialSpec`s into the RGBA32F material square the
 * GLSL `readMaterialInfo` reads. Returns a CPU `MaterialsTextureData` grid
 * (`{ data, dim, kind:'rgba32f', materialCount }`) ready for `gl.texImage2D`.
 *
 * Square sizing: `dim = ceil(sqrt(materials.length * MATERIAL_PIXELS))`.
 */
/** Resolve a TextureRef → its atlas layer index (-1 = none / unmapped). */
type TextureLayerLookup = Map<unknown, number> | TextureAtlasLayerMap;

function layerMapFor(
  layerOf: TextureLayerLookup | undefined,
  colorSpace: TextureSampleColorSpace,
): ReadonlyMap<unknown, number> | undefined {
  if (layerOf == null) return undefined;
  if (typeof (layerOf as Map<unknown, number>).get === 'function') return layerOf as Map<unknown, number>;
  return (layerOf as TextureAtlasLayerMap)[colorSpace];
}

type TextureRefLike = { handle?: unknown; texCoord?: number };

interface PackMaterialsTextureOptions {
  readonly vertexColorMaterialIds?: ReadonlySet<number>;
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: string;
  readonly warningMethod?: string;
}

type UnsupportedTexCoordWarner = (
  materialIndex: number,
  field: string,
  ref: TextureRefLike | undefined,
) => boolean;

function isUnsupportedTexCoord(ref: TextureRefLike | undefined): boolean {
  if (ref?.handle == null) return false;
  const texCoord = ref.texCoord ?? 0;
  return texCoord !== 0 && texCoord !== 1;
}

function safeTexCoord(ref: TextureRefLike | undefined): number {
  return isUnsupportedTexCoord(ref) ? 0 : (ref?.texCoord ?? 0);
}

function unsupportedTexCoordWarning(
  materialIndex: number,
  field: string,
  texCoord: number,
  options: PackMaterialsTextureOptions,
): EngineWarning {
  const message =
    `[vitrum/pt-webgl2] ignoring material ${materialIndex} ${field}: texCoord ${texCoord} ` +
    `is unsupported; only texCoord 0 and 1 are renderable by this backend.`;
  return {
    code: 'pt-webgl2.material-texture-unsupported-texcoord',
    backend: 'pt-webgl2',
    phase: options.warningPhase ?? 'setScene',
    method: options.warningMethod ?? 'setScene',
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

function createUnsupportedTexCoordWarner(options: PackMaterialsTextureOptions): UnsupportedTexCoordWarner {
  const warned = new Set<string>();
  return (materialIndex, field, ref): boolean => {
    if (!isUnsupportedTexCoord(ref)) return false;
    const texCoord = ref?.texCoord ?? 0;
    const key = `${materialIndex}:${field}:${texCoord}`;
    if (!warned.has(key)) {
      warned.add(key);
      const warning = unsupportedTexCoordWarning(materialIndex, field, texCoord, options);
      options.onWarning?.(warning);
      console.warn(warning.message.replace('[vitrum/pt-webgl2] ', '[pt-webgl2] '));
    }
    return true;
  };
}

function mapLayer(
  ref: TextureRefLike | undefined,
  layerOf: TextureLayerLookup | undefined,
  colorSpace: TextureSampleColorSpace,
  materialIndex: number,
  field: string,
  warnUnsupportedTexCoord: UnsupportedTexCoordWarner,
): number {
  if (ref?.handle == null || layerOf == null) return -1;
  if (warnUnsupportedTexCoord(materialIndex, field, ref)) return -1;
  return layerMapFor(layerOf, colorSpace)?.get(ref.handle) ?? -1;
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

function writeSamplerPolicy(
  data: Float32Array,
  offset: number,
  ref: {
    wrapS?: TextureWrapMode;
    wrapT?: TextureWrapMode;
    magFilter?: TextureFilterMode;
    minFilter?: TextureFilterMode;
    mipFilter?: TextureMipFilterMode;
  } | undefined,
): void {
  data[offset] = WRAP_MODE_INDEX[ref?.wrapS ?? 'repeat'];
  data[offset + 1] = WRAP_MODE_INDEX[ref?.wrapT ?? 'repeat'];
  data[offset + 2] = MIP_FILTER_INDEX[ref?.mipFilter ?? 'none'];
  data[offset + 3] =
    FILTER_MODE_INDEX[ref?.magFilter ?? 'nearest'] +
    FILTER_MODE_INDEX[ref?.minFilter ?? 'nearest'] * 2;
}

// ── D10.8: per-section packer helpers ────────────────────────────────────
// Each helper writes sequential floats into `data` starting at `index` and returns
// the updated index. All are pure (no side-effects beyond writing to `data`), so the
// split is byte-identical to the original monolithic loop body.

/** D10.8: Resolved atlas layer ids for a single material. */
interface LayerIds {
  baseColor: number;
  metal: number;
  rough: number;
  transmission: number;
  emissive: number;
  normal: number;
  alpha: number;
  clearcoat: number;
  clearcoatRoughness: number;
  clearcoatNormal: number;
  sheenColor: number;
  sheenRoughness: number;
  iridescence: number;
  iridescenceThickness: number;
  specularColor: number;
  specularIntensity: number;
  ao: number;
  lightMap: number;
  bump: number;
  anisotropy: number;
  thickness: number;
  frontLayerNormal: number;
  backLayerNormal: number;
}

/**
 * D10.8: Map table driving `packLayerIds`. Each entry declares the `LayerIds`
 * key, the `MaterialSpec` texture-ref accessor, its sample color space, and the
 * warning `field` label. One row per resolved layer id — iterated once so the
 * per-map arguments (color space, field name) live in data, not 23 positional
 * call sites. `field` doubles as the atlas-warning label the fork's GLSL expects.
 */
interface LayerIdMapEntry {
  readonly key: keyof LayerIds;
  readonly ref: (m: MaterialSpec) => TextureRefLike | undefined;
  readonly colorSpace: TextureSampleColorSpace;
  readonly field: string;
}

const LAYER_ID_MAP: readonly LayerIdMapEntry[] = [
  { key: 'baseColor', ref: (m) => m.baseColorMap, colorSpace: 'srgb', field: 'baseColorMap' },
  { key: 'metal', ref: (m) => m.metallicMap, colorSpace: 'linear', field: 'metallicMap' },
  { key: 'rough', ref: (m) => m.roughnessMap, colorSpace: 'linear', field: 'roughnessMap' },
  { key: 'transmission', ref: (m) => m.transmissionMap, colorSpace: 'linear', field: 'transmissionMap' },
  { key: 'emissive', ref: (m) => m.emissiveMap, colorSpace: 'srgb', field: 'emissiveMap' },
  { key: 'normal', ref: (m) => m.normalMap, colorSpace: 'linear', field: 'normalMap' },
  { key: 'alpha', ref: (m) => m.alphaMap, colorSpace: 'linear', field: 'alphaMap' },
  { key: 'clearcoat', ref: (m) => m.clearcoatMap, colorSpace: 'linear', field: 'clearcoatMap' },
  { key: 'clearcoatRoughness', ref: (m) => m.clearcoatRoughnessMap, colorSpace: 'linear', field: 'clearcoatRoughnessMap' },
  { key: 'clearcoatNormal', ref: (m) => m.clearcoatNormalMap, colorSpace: 'linear', field: 'clearcoatNormalMap' },
  { key: 'sheenColor', ref: (m) => m.sheenColorMap, colorSpace: 'srgb', field: 'sheenColorMap' },
  { key: 'sheenRoughness', ref: (m) => m.sheenRoughnessMap, colorSpace: 'linear', field: 'sheenRoughnessMap' },
  { key: 'iridescence', ref: (m) => m.iridescenceMap, colorSpace: 'linear', field: 'iridescenceMap' },
  { key: 'iridescenceThickness', ref: (m) => m.iridescenceThicknessMap, colorSpace: 'linear', field: 'iridescenceThicknessMap' },
  { key: 'specularColor', ref: (m) => m.specularColorMap, colorSpace: 'srgb', field: 'specularColorMap' },
  { key: 'specularIntensity', ref: (m) => m.specularIntensityMap, colorSpace: 'linear', field: 'specularIntensityMap' },
  { key: 'ao', ref: (m) => m.aoMap, colorSpace: 'linear', field: 'aoMap' },
  { key: 'lightMap', ref: (m) => m.lightMap, colorSpace: 'linear', field: 'lightMap' },
  { key: 'bump', ref: (m) => m.bumpMap, colorSpace: 'linear', field: 'bumpMap' },
  { key: 'anisotropy', ref: (m) => m.anisotropyMap, colorSpace: 'linear', field: 'anisotropyMap' },
  { key: 'thickness', ref: (m) => m.thicknessMap, colorSpace: 'linear', field: 'thicknessMap' },
  { key: 'frontLayerNormal', ref: (m) => m.frontLayer?.normalMap, colorSpace: 'linear', field: 'frontLayer.normalMap' },
  { key: 'backLayerNormal', ref: (m) => m.backLayer?.normalMap, colorSpace: 'linear', field: 'backLayer.normalMap' },
];

/** D10.8: Resolve all atlas layer ids for a material in one pass (avoids re-calling mapLayer). */
function packLayerIds(
  m: MaterialSpec,
  layerOf: TextureLayerLookup | undefined,
  materialIndex: number,
  warnUnsupportedTexCoord: UnsupportedTexCoordWarner,
): LayerIds {
  const ids = {} as LayerIds;
  for (const entry of LAYER_ID_MAP) {
    ids[entry.key] = mapLayer(
      entry.ref(m),
      layerOf,
      entry.colorSpace,
      materialIndex,
      entry.field,
      warnUnsupportedTexCoord,
    );
  }
  return ids;
}

/**
 * D10.8: Write samples 0..19 (the scalar/flag/SSS/spectral-metadata block).
 * Returns the updated index (always base + 20*4 = base + 80).
 */
function packScalarSlots(
  data: Float32Array,
  index: number,
  m: PackedMaterialSpec,
  ids: LayerIds,
): number {
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
  const anisotropy = Math.max(0.0, Math.min(1.0, m.anisotropy ?? 0.0));
  const anisotropyRotation = m.anisotropyRotation ?? 0.0;
  const attenuationColor: Vec3 = m.attenuationColor ?? DEFAULT_ATTENUATION_COLOR;
  const attenuationDistance = m.attenuationDistance ?? Infinity;
  const sssMedium = resolveSssMedium(m, attenuationColor, attenuationDistance);
  const thickness = m.thickness ?? 0.0;
  const opacity = m.opacity ?? 1.0;
  const alphaTest = m.alphaMode === 'mask' ? (m.alphaCutoff ?? 0.5) : 0.0;
  const transparent = m.alphaMode === 'blend';

  // isThinFilm — fork: thickness===0 && attenuationDistance===Infinity.
  const isThinFilm = thickness === 0.0 && attenuationDistance === Infinity;

  // sample 0 — color.rgb / map
  data[index++] = color[0];
  data[index++] = color[1];
  data[index++] = color[2];
  data[index++] = ids.baseColor;

  // sample 1 — metalness / metalnessMap / roughness / roughnessMap
  data[index++] = metalness;
  data[index++] = ids.metal;
  data[index++] = roughness;
  data[index++] = ids.rough;

  // sample 2 — ior / transmission / transmissionMap / emissiveIntensity
  data[index++] = ior;
  data[index++] = transmission;
  data[index++] = ids.transmission;
  data[index++] = emissiveIntensity;

  // sample 3 — emissive.rgb / emissiveMap
  data[index++] = emissive[0];
  data[index++] = emissive[1];
  data[index++] = emissive[2];
  data[index++] = ids.emissive;

  // sample 4 — normalMap / normalScale.xy / clearcoat
  data[index++] = ids.normal;
  data[index++] = normalScale;
  data[index++] = normalScale;
  data[index++] = clearcoat;

  // sample 5 — clearcoatMap / clearcoatRoughness / clearcoatRoughnessMap / clearcoatNormalMap
  data[index++] = ids.clearcoat;
  data[index++] = clearcoatRoughness;
  data[index++] = ids.clearcoatRoughness;
  data[index++] = ids.clearcoatNormal;

  // sample 6 — clearcoatNormalScale.xy / anisotropyMap / sheen
  data[index++] = clearcoatNormalScale;
  data[index++] = clearcoatNormalScale;
  data[index++] = ids.anisotropy;
  data[index++] = sheen;

  // sample 7 — sheenColor.rgb / sheenColorMap
  data[index++] = sheenColor[0];
  data[index++] = sheenColor[1];
  data[index++] = sheenColor[2];
  data[index++] = ids.sheenColor;

  // sample 8 — sheenRoughness / sheenRoughnessMap / iridescenceMap / iridescenceThicknessMap
  data[index++] = sheenRoughness;
  data[index++] = ids.sheenRoughness;
  data[index++] = ids.iridescence;
  data[index++] = ids.iridescenceThickness;

  // sample 9 — iridescence / iridescenceIOR / iridThicknessRange.xy
  data[index++] = iridescence;
  data[index++] = iridescenceIor;
  data[index++] = iridThicknessRange[0];
  data[index++] = iridThicknessRange[1];

  // sample 10 — specularColor.rgb / specularColorMap
  data[index++] = specularColor[0];
  data[index++] = specularColor[1];
  data[index++] = specularColor[2];
  data[index++] = ids.specularColor;

  // sample 11 — specularIntensity / specularIntensityMap / isThinFilm / anisotropy
  data[index++] = specularIntensity;
  data[index++] = ids.specularIntensity;
  data[index++] = Number(isThinFilm);
  data[index++] = anisotropy;

  // sample 12 — attenuationColor.rgb / attenuationDistance
  data[index++] = attenuationColor[0];
  data[index++] = attenuationColor[1];
  data[index++] = attenuationColor[2];
  data[index++] = attenuationDistance;

  // sample 13 — alphaMap / opacity / alphaTest / side
  data[index++] = ids.alpha;
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
  data[index++] = m.castShadow === false ? 0 : 1;
  data[index++] = m.vertexColors === true ? 1 : 0;
  {
    let flags = Number(transparent);
    if (sssMedium.active) flags |= TRANSLUCENT_BIT;
    if (m.shadingModel === 'unlit') flags |= UNLIT_BIT;
    if (m.meshEmitterCastShadowDisabled === true) flags |= MESH_EMITTER_CAST_SHADOW_DISABLED_BIT;
    data[index++] = flags;
  }

  // sample 15 — sssSigmaT / sssAnisotropyG / dispersionStrength / thinFilmEnabled
  const scatteringAnisotropy = m.scatteringAnisotropy ?? 0.0;
  const dispersionAbbe = m.dispersionAbbeNumber ?? 0.0;
  const dispersionStrength = dispersionStrengthFromAbbe(ior, dispersionAbbe);
  const thinFilmLayers = m.thinFilmStack?.layers ?? [];
  const thinFilmLayerCount = Math.min(thinFilmLayers.length, THIN_FILM_LAYER_LIMIT);
  const thinFilmEnabled = thinFilmLayerCount > 0 ? 1.0 : 0.0;
  data[index++] = sssMedium.sigmaTMax;
  data[index++] = scatteringAnisotropy;
  data[index++] = dispersionStrength;
  data[index++] = thinFilmEnabled;

  // sample 16 — sssSigmaS.rgb / thinFilmLayerCount.
  data[index++] = sssMedium.sigmaS[0];
  data[index++] = sssMedium.sigmaS[1];
  data[index++] = sssMedium.sigmaS[2];
  data[index++] = thinFilmLayerCount;

  // sample 17 — thinFilmIncidentIor / angleDependent / anisotropyRotation / packedFeatureFlags
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
  data[index++] = anisotropyRotation;
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

  return index;
}

/**
 * D10.8: Write samples 20..27 (32 floats): uniform-grid spectral attenuation μ(λ), 380..780nm.
 * Returns the updated index (always base + 28*4 = base + 112).
 */
function packSpectralGrid(data: Float32Array, index: number, m: MaterialSpec): number {
  const spectralCurve = m.spectralAttenuation ?? null;
  const hasSpectral = spectralCurve != null;
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
  return index;
}

/**
 * D10.8: Write samples 28..54 (108 floats + 3 pad = 111 floats):
 * thin-film layer payload [ior, thicknessNm, extinction]×35 + pad.
 * Returns the updated index (always base + 55*4 = base + 220).
 */
function packThinFilm(data: Float32Array, index: number, m: MaterialSpec): number {
  const thinFilmLayers = m.thinFilmStack?.layers ?? [];
  const thinFilmLayerCount = Math.min(thinFilmLayers.length, THIN_FILM_LAYER_LIMIT);
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
  return index;
}

/**
 * D10.8: Write texture-transform mat3s at samples 55..84, the D3 auxiliary block
 * at texels 85..92, alphaMapTransform at 93..94, anisotropyMapTransform at
 * 95..96, thickness payload at 97, thicknessMapTransform at 98..99, and per-map
 * sampler policy at
 * MATERIAL_WRAP_TEXEL_OFFSET.. Uses absolute texel offsets from `base` (not
 * `index`) — these writes are non-sequential (the transform slots are at fixed
 * positions).
 */
function packTextureTransforms(
  data: Float32Array,
  base: number,
  m: MaterialSpec,
  ids: LayerIds,
): void {
  // samples 55..84 (30 texels): 15 texture-transform mat3s, 2 texels each, at
  // `texel 55 + 2k` (k per the GLSL `readTextureTransform` order in material_struct).
  // The GLSL only READS a transform when the map id != -1, so write one per mapped
  // texture (others stay zero / unread → identity). The fork's
  // `writeTextureMatrixToArray` is the analogue.
  // Transform-slot order matches the GLSL `readTextureTransform` calls in
  // material_struct.glsl.js (firstTextureTransformIdx + 2k): map(0), metalness(2),
  // roughness(4), transmission(6), emissive(8), normal(10), clearcoat(12),
  // clearcoatNormal(14), clearcoatRoughness(16), sheenColor(18), sheenRoughness(20),
  // iridescence(22), iridescenceThickness(24), specularColor(26), specularIntensity(28).
  // Each slot is 2 texels (mat3 rows), starting at texel 55.
  if (ids.baseColor >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['baseColorMap']!, m.baseColorMap);
  if (ids.metal >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['metallicMap']!, m.metallicMap);
  if (ids.rough >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['roughnessMap']!, m.roughnessMap);
  if (ids.transmission >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['transmissionMap']!, m.transmissionMap);
  if (ids.emissive >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['emissiveMap']!, m.emissiveMap);
  if (ids.normal >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['normalMap']!, m.normalMap);
  // D3 — clearcoat / sheen / iridescence / specular transforms (GLSL slots 12..28).
  if (ids.clearcoat >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['clearcoatMap']!, m.clearcoatMap);
  if (ids.clearcoatNormal >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['clearcoatNormalMap']!, m.clearcoatNormalMap);
  if (ids.clearcoatRoughness >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['clearcoatRoughnessMap']!, m.clearcoatRoughnessMap);
  if (ids.sheenColor >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['sheenColorMap']!, m.sheenColorMap);
  if (ids.sheenRoughness >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['sheenRoughnessMap']!, m.sheenRoughnessMap);
  if (ids.iridescence >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['iridescenceMap']!, m.iridescenceMap);
  if (ids.iridescenceThickness >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['iridescenceThicknessMap']!, m.iridescenceThicknessMap);
  if (ids.specularColor >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['specularColorMap']!, m.specularColorMap);
  if (ids.specularIntensity >= 0) writeTransform(data, base, MATERIAL_TRANSFORM_TEXEL['specularIntensityMap']!, m.specularIntensityMap);
  if (ids.alpha >= 0) writeTransform(data, base, MATERIAL_ALPHA_TRANSFORM_TEXEL, m.alphaMap);
  if (ids.anisotropy >= 0) writeTransform(data, base, MATERIAL_ANISOTROPY_TRANSFORM_TEXEL, m.anisotropyMap);
  if (ids.thickness >= 0) writeTransform(data, base, MATERIAL_THICKNESS_TRANSFORM_TEXEL, m.thicknessMap);

  // D3 — texels 85/86: ao/light/bump map ids + scalars + envMapIntensity
  // (mirrors readMaterialInfo s20/s21 in material_struct.glsl.js).
  // texel 86.a: UV-set bitmask — bit k set means map k samples uv1 (ATTR_UV1)
  // instead of uv0 (ATTR_UV). Bit assignments are single-sourced in materialStride.js.
  let uvSetMask = 0;
  for (const [key, bit] of Object.entries(UV_SET_BIT)) {
    const ref = m[key as keyof MaterialSpec] as { texCoord?: number } | undefined;
    if ((ref?.texCoord ?? 0) === 1) uvSetMask |= bit;
  }

  let d3 = base + MATERIAL_D3_AUX_TEXEL * 4;
  data[d3++] = ids.ao;
  data[d3++] = ids.lightMap;
  data[d3++] = ids.bump;
  data[d3++] = m.envMapIntensity ?? 1.0;
  data[d3++] = m.aoMapIntensity ?? 1.0;
  data[d3++] = m.lightMapIntensity ?? 1.0;
  data[d3++] = m.bumpScale ?? 1.0;
  data[d3++] = uvSetMask; // uv-set bitmask (was pad)
  // D3 — ao/light/bump transforms at texels 87/89/91 (2 texels per mat3).
  if (ids.ao >= 0) writeTransform(data, base, MATERIAL_AO_TRANSFORM_TEXEL, m.aoMap);
  if (ids.lightMap >= 0) writeTransform(data, base, MATERIAL_LIGHTMAP_TRANSFORM_TEXEL, m.lightMap);
  if (ids.bump >= 0) writeTransform(data, base, MATERIAL_BUMP_TRANSFORM_TEXEL, m.bumpMap);

  const volume = base + MATERIAL_VOLUME_THICKNESS_TEXEL * 4;
  data[volume] = m.thickness ?? 0.0;
  data[volume + 1] = ids.thickness;
  data[volume + 2] = 0.0;
  data[volume + 3] = 0.0;

  for (let mapIdx = 0; mapIdx < MATERIAL_MAP_FIELD_ORDER.length; mapIdx += 1) {
    const texel = MATERIAL_WRAP_TEXEL_OFFSET + mapIdx;
    const samplerOffset = base + texel * 4;
    const field = MATERIAL_MAP_FIELD_ORDER[mapIdx] as keyof MaterialSpec;
    writeSamplerPolicy(
      data,
      samplerOffset,
      m[field] as {
        wrapS?: TextureWrapMode;
        wrapT?: TextureWrapMode;
        magFilter?: TextureFilterMode;
        minFilter?: TextureFilterMode;
        mipFilter?: TextureMipFilterMode;
      } | undefined,
    );
  }

  // RFE-03 layer normals: appended payload so the historical base-map wrap and
  // spectral-reflectance lanes stay stable. Layer maps are face-selected shader
  // overrides; when absent the shader falls back to the ordinary normalMap path.
  const layerNormal = base + MATERIAL_LAYER_NORMAL_TEXEL_OFFSET * 4;
  data[layerNormal] = ids.frontLayerNormal;
  data[layerNormal + 1] = m.frontLayer?.normalScale ?? 1.0;
  data[layerNormal + 2] = ids.backLayerNormal;
  data[layerNormal + 3] = m.backLayer?.normalScale ?? 1.0;
  if (ids.frontLayerNormal >= 0) {
    writeTransform(data, base, MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 1, m.frontLayer?.normalMap);
  }
  if (ids.backLayerNormal >= 0) {
    writeTransform(data, base, MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 3, m.backLayer?.normalMap);
  }
  const frontLayerSampler = base + (MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 5) * 4;
  writeSamplerPolicy(data, frontLayerSampler, m.frontLayer?.normalMap);
  const backLayerSampler = base + (MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 6) * 4;
  writeSamplerPolicy(data, backLayerSampler, m.backLayer?.normalMap);
  const layerUv = base + (MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 7) * 4;
  data[layerUv] = ids.frontLayerNormal >= 0 ? safeTexCoord(m.frontLayer?.normalMap) : 0;
  data[layerUv + 1] = ids.backLayerNormal >= 0 ? safeTexCoord(m.backLayer?.normalMap) : 0;
  data[layerUv + 2] = 0;
  data[layerUv + 3] = 0;
}

function packSpectralReflectance(
  data: Float32Array,
  base: number,
  m: MaterialSpec,
): void {
  const baseColor = m.baseColor ?? [1, 1, 1];
  const [c0, c1, c2] = rgbToSpectralCoefficients(
    finiteOr(baseColor[0], 1),
    finiteOr(baseColor[1], 1),
    finiteOr(baseColor[2], 1),
  );
  const offset = base + MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET * 4;
  data[offset] = finiteOr(c0, 0);
  data[offset + 1] = finiteOr(c1, 0);
  data[offset + 2] = finiteOr(c2, 0);
  data[offset + 3] = 1;
}

export function packMaterialsTexture(
  materials: readonly MaterialSpec[],
  layerOf?: TextureLayerLookup,
  options: PackMaterialsTextureOptions = {},
): MaterialsTextureData {
  const materialCount = materials.length;
  const pixelCount = materialCount * MATERIAL_PIXELS;
  const dim = squareDim(pixelCount);
  const data = new Float32Array(dim * dim * 4);
  const warnUnsupportedTexCoord = createUnsupportedTexCoordWarner(options);

  let index = 0;
  for (let i = 0; i < materialCount; i += 1) {
    const source = materials[i]!;
    const m = (options.vertexColorMaterialIds?.has(i) === true
      ? { ...source, vertexColors: true }
      : source) as PackedMaterialSpec;
    const base = index; // first float of this material's block

    const ids = packLayerIds(m, layerOf, i, warnUnsupportedTexCoord);

    // samples 0..19: scalar fields, layer ids, SSS, thin-film metadata
    index = packScalarSlots(data, index, m, ids);
    // samples 20..27: spectral attenuation grid
    index = packSpectralGrid(data, index, m);
    // samples 28..54: thin-film layer payload
    index = packThinFilm(data, index, m);
    // samples 55..92: texture-transform mat3s + D3 ao/light/bump auxiliary block
    packTextureTransforms(data, base, m, ids);
    // sample 111: per-material Jakob-Hanika spectral reflectance coefficients.
    packSpectralReflectance(data, base, m);

    index = base + MATERIAL_STRIDE;
  }

  return { data, dim, kind: 'rgba32f', materialCount };
}
