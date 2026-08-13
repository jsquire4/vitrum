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

import {
  effectiveMaterialIor,
  validateMaterialSpec,
  type MaterialSpec,
  type TextureFilterMode,
  type TextureMipFilterMode,
  type TextureWrapMode,
  type Vec3,
} from '@vitrum/core';
import { packRadianceRgbScaleF32 } from '@vitrum/shared-bvh';
import {
  rgbToSpectralCoefficients,
  dispersionStrengthFromAbbe,
  resolveEmissiveIntensity,
  sampleSpectralCurve as sharedSampleSpectralCurve,
  sigmaAFromAttenuation as sharedSigmaAFromAttenuation,
  type SpectralCurveSampleOptions,
} from '@vitrum/shared-samplers';
import type { MaterialsTextureData } from './sceneTextures.js';
import type {
  MaterialTextureAtlasLayerMaps,
  TextureAtlasLayerMap,
  TextureAtlasStorageClass,
  TextureSampleColorSpace,
} from './texturesArray.js';
import {
  snapshotMaterialTextureInputs,
  textureColorSpaceForMapKey,
  textureStorageClassForMapKey,
} from './texturesArray.js';
import {
  requireFiniteFloat32,
  requireNonNegativeFloat32,
} from './float32Policy.js';
import { materialEmissionExcludedFromMeshNee } from './meshEmitterPolicy.js';

import {
  MATERIAL_MAP_FIELD_ORDER,
  MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
  MATERIAL_UV_SELECTOR_TEXEL_OFFSET,
  MATERIAL_PIXELS,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
  MATERIAL_WRAP_TEXEL_OFFSET,
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
 *  sampler-policy texels 100..120 + spectral reflectance texel 121 + layer
 *  normal texels 122..129 + UV selector texels 130..135). Re-exported
 *  for tests and parity guards. */
export {
  MATERIAL_MAP_FIELD_ORDER,
  MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
  MATERIAL_UV_SELECTOR_TEXEL_OFFSET,
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

/** UNLIT_BIT — flag (s14.a) bit set for terminal base-color unlit shading. */
const UNLIT_BIT = 1 << 5;
/** DOUBLE_SIDED_BIT — authored surface emission is visible from both orientations. */
const DOUBLE_SIDED_BIT = 1 << 1;
/** MESH_EMITTER_CAST_SHADOW_DISABLED_BIT — folded mesh-area emitter skips forward BSDF emission. */
const MESH_EMITTER_CAST_SHADOW_DISABLED_BIT = 1 << 6;
/** MESH_EMITTER_NEE_DISABLED_BIT — surface emission has no mesh-light NEE competitor. */
const MESH_EMITTER_NEE_DISABLED_BIT = 1 << 7;

// Uniform-grid spectral attenuation μ(λ): 32 samples, 380→780 nm inclusive.
const SPECTRAL_GRID_SAMPLE_COUNT = 32;
const SPECTRAL_GRID_START_NM = 380.0;
const SPECTRAL_GRID_END_NM = 780.0;

// Core SpectralCurve requires at least three finite, non-negative values and
// strictly ordered finite bounds. The shared sampler enforces that contract.
const PT_WEBGL2_SPECTRAL_CURVE_OPTIONS: SpectralCurveSampleOptions = {
  minValueCount: 3,
};

// Thin-film stack: 35 layers × [ior, thicknessNm, extinction].
// Declared per-backend in `@vitrum/core` `BackendSupportDetails.thinFilmLayerLimit`
// (D1 = Option B); pt-webgl2 keeps its 35-layer GLSL stride.
export const THIN_FILM_LAYER_LIMIT = 35;
/** ceil(sqrt(n)) — the square dimension that holds `n` texels row-major (mirrors fork). */
function squareDim(texelCount: number): number {
  return Math.max(1, Math.ceil(Math.sqrt(Math.max(1, texelCount))));
}

function resolveSssMedium(
  m: PackedMaterialSpec,
  attenuationColor: Vec3,
  attenuationDistance: number,
  context: string,
): {
  readonly active: boolean;
  readonly scatters: boolean;
  readonly sigmaTMax: number;
  readonly sigmaS: Vec3;
} {
  const scalarSigmaS = requireNonNegativeFloat32(
    m.scatteringCoefficient ?? 0.0,
    `${context}.scatteringCoefficient`,
  );
  const sigmaS: Vec3 = m.scatteringCoefficientRGB
    ? [
        requireNonNegativeFloat32(m.scatteringCoefficientRGB[0], `${context}.scatteringCoefficientRGB[0]`),
        requireNonNegativeFloat32(m.scatteringCoefficientRGB[1], `${context}.scatteringCoefficientRGB[1]`),
        requireNonNegativeFloat32(m.scatteringCoefficientRGB[2], `${context}.scatteringCoefficientRGB[2]`),
      ]
    : [scalarSigmaS, scalarSigmaS, scalarSigmaS];
  const sigmaA = sharedSigmaAFromAttenuation(attenuationColor, attenuationDistance);
  const sigmaAMax = Math.max(sigmaA[0], sigmaA[1], sigmaA[2]);
  // A zero attenuation-color channel has the exact Beer coefficient +Infinity.
  // It must remain exactly black for every positive distance, but +Infinity is
  // not a usable exponential free-flight proposal: decoding sigmaS / sigmaTMax
  // would collapse every finite scattering lane and the survival ratio would
  // form Inf - Inf. Sample from the largest *finite* represented extinction
  // instead. Infinite channels are deterministic zero-survival lanes and are
  // still evaluated by the authored Beer law in the shader.
  let finiteSigmaTMax = 0.0;
  for (let channel = 0; channel < 3; channel += 1) {
    const sigmaT = sigmaA[channel]! + sigmaS[channel]!;
    if (Number.isFinite(sigmaT)) {
      finiteSigmaTMax = Math.max(finiteSigmaTMax, sigmaT);
    }
  }
  const sigmaSMax = Math.max(sigmaS[0], sigmaS[1], sigmaS[2]);
  // Free-flight sampling uses one scalar majorant for both RGB and hero-
  // wavelength paths.  The shader interpolates the packed 32-sample spectral
  // grid linearly, so the largest packed endpoint is an exact upper bound for
  // every wavelength the shader can evaluate.  Omitting this term made an
  // authored spectral absorption peak larger than the proposal density.
  let spectralSigmaAMax = 0.0;
  if (m.spectralAttenuation != null) {
    const spectralDenom = Math.max(SPECTRAL_GRID_SAMPLE_COUNT - 1, 1);
    for (let s = 0; s < SPECTRAL_GRID_SAMPLE_COUNT; s += 1) {
      const t = s / spectralDenom;
      const lambdaNm = SPECTRAL_GRID_START_NM +
        t * (SPECTRAL_GRID_END_NM - SPECTRAL_GRID_START_NM);
      const spectralSigmaA = requireNonNegativeFloat32(
        sharedSampleSpectralCurve(
          m.spectralAttenuation,
          lambdaNm,
          PT_WEBGL2_SPECTRAL_CURVE_OPTIONS,
        ),
        `${context}.spectralAttenuation sample ${s}`,
      );
      if (Number.isFinite(spectralSigmaA)) {
        spectralSigmaAMax = Math.max(spectralSigmaAMax, spectralSigmaA);
      }
    }
  }
  return {
    // Closed transmissive regions own geometric Beer transport whenever any
    // absorption/scattering channel is authored, even at thickness=0. The
    // latter means "derive distance from the closed boundary", not "thin
    // sheet". Preserve an explicitly authored spectral curve as a bulk signal
    // even if its current samples are all zero so mutation cannot silently
    // change the topology class.
    active:
      sigmaSMax > 0 ||
      sigmaAMax > 0 ||
      m.spectralAttenuation != null,
    // Free-flight collision sampling is only meaningful when the medium
    // actually scatters; Beer-Lambert is exact for pure absorption. This must
    // stay in lockstep with `isParticipatingMedium` in sceneTraceFeatures.ts,
    // which drives the FEATURE_FOG compile gate. Packing the free-flight bit on
    // absorption-only media routed them to the survival-ratio arm (which returns
    // ~1.0) while FEATURE_FOG was off, cancelling their absorption entirely.
    scatters: sigmaSMax > 0,
    sigmaTMax: requireNonNegativeFloat32(
      Math.max(finiteSigmaTMax, spectralSigmaAMax + sigmaSMax, 0.0),
      `${context} derived sigmaT majorant`,
    ),
    sigmaS,
  };
}

// `dispersionStrengthFromAbbe` and `sampleSpectralCurve` are single-sourced with
// pt-webgpu via shared-samplers. pt-webgl2's spectral edge handling (≥2 values,
// 380/780 fallback) is preserved via `PT_WEBGL2_SPECTRAL_CURVE_OPTIONS`.

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

// Largest positive binary32 integer strictly inside GLSL ES's signed-int
// conversion range. Every selector/policy lane is round-tripped float -> int.
const GLSL_INT_MAX_EXACT_F32 = 2_147_483_520;

function requireExactGlslInt(
  value: unknown,
  context: string,
  minimum = 0,
): number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > GLSL_INT_MAX_EXACT_F32 ||
    Math.fround(value) !== value
  ) {
    throw new RangeError(
      `${context} must be an exact float32 integer in ` +
        `[${minimum}, ${GLSL_INT_MAX_EXACT_F32}] (got ${String(value)}).`,
    );
  }
  return value;
}

function packedSamplerPolicyLane(
  payload: unknown,
  mode: number,
  context: string,
): number {
  const exactPayload = requireExactGlslInt(payload, `${context} payload`);
  const packed = BigInt(exactPayload) * 4n + BigInt(mode);
  if (packed > BigInt(GLSL_INT_MAX_EXACT_F32)) {
    throw new RangeError(`${context} exceeds the GLSL signed-int descriptor range.`);
  }
  return requireExactGlslInt(Number(packed), context);
}

function requireScaleFloat32(value: number, context: string): number {
  const stored = requireFiniteFloat32(value, context);
  if (value !== 0 && stored === 0) {
    throw new RangeError(`${context} underflows WebGL float32 storage.`);
  }
  return stored;
}

/**
 * Pack a list of core `MaterialSpec`s into the RGBA32F material square the
 * GLSL `readMaterialInfo` reads. Returns a CPU `MaterialsTextureData` grid
 * (`{ data, dim, kind:'rgba32f', materialCount }`) ready for `gl.texImage2D`.
 *
 * Square sizing: `dim = ceil(sqrt(materials.length * MATERIAL_PIXELS))`.
 */
/** Resolve a TextureRef → its atlas layer index (-1 = none / unmapped). */
type TextureLayerLookup =
  | Map<unknown, number>
  | TextureAtlasLayerMap
  | MaterialTextureAtlasLayerMaps;

function atlasMapFor(
  layerOf: TextureLayerLookup | undefined,
  storageClass: TextureAtlasStorageClass,
): TextureAtlasLayerMap | undefined {
  if (layerOf == null || typeof (layerOf as Map<unknown, number>).get === 'function') {
    return undefined;
  }
  if ('ldr' in layerOf && 'hdr' in layerOf) {
    return layerOf[storageClass] ?? undefined;
  }
  return layerOf as TextureAtlasLayerMap;
}

function layerMapFor(
  layerOf: TextureLayerLookup | undefined,
  storageClass: TextureAtlasStorageClass,
  colorSpace: TextureSampleColorSpace,
): ReadonlyMap<unknown, number> | undefined {
  if (layerOf == null) return undefined;
  if (typeof (layerOf as Map<unknown, number>).get === 'function') return layerOf as Map<unknown, number>;
  return atlasMapFor(layerOf, storageClass)?.[colorSpace];
}

type TextureRefLike = { handle?: unknown; texCoord?: number };

interface PackMaterialsTextureOptions {
  readonly vertexColorMaterialIds?: ReadonlySet<number>;
  /** Scene-local authored texCoord -> dense attributesArray layer mapping. */
  readonly uvLayerByTexCoord?: ReadonlyMap<number, number>;
}

const PRIVATE_MATERIAL_FLAGS = new Set([
  'castShadow',
  'vertexColors',
  'meshEmitterCastShadowDisabled',
]);

function validatedPackedMaterial(
  source: MaterialSpec,
  materialIndex: number,
): PackedMaterialSpec {
  if (source == null || typeof source !== 'object' || Array.isArray(source)) {
    throw new TypeError(`pt-webgl2 material ${materialIndex} must be a material record.`);
  }
  const sourceRecord = source as unknown as Record<PropertyKey, unknown>;
  const publicRecord: Record<PropertyKey, unknown> = {};
  const privateFlags: Record<string, boolean> = {};
  for (const key of Reflect.ownKeys(source)) {
    const value = sourceRecord[key];
    if (typeof key === 'string' && PRIVATE_MATERIAL_FLAGS.has(key)) {
      if (value !== undefined && typeof value !== 'boolean') {
        throw new TypeError(
          `pt-webgl2 material ${materialIndex}.${key} must be boolean when supplied.`,
        );
      }
      if (value !== undefined) privateFlags[key] = value;
    } else {
      Object.defineProperty(publicRecord, key, {
        value,
        enumerable: Object.prototype.propertyIsEnumerable.call(source, key),
        configurable: true,
        writable: true,
      });
    }
  }
  const publicMaterial = publicRecord as unknown as MaterialSpec;
  validateMaterialSpec(publicMaterial, `pt-webgl2 material ${materialIndex}`);
  return { ...publicMaterial, ...privateFlags } as PackedMaterialSpec;
}

export function assertThinFilmLayerLimit(
  m: MaterialSpec,
  context = 'material',
): number {
  const requested = m.thinFilmStack?.layers?.length ?? 0;
  if (requested > THIN_FILM_LAYER_LIMIT) {
    throw new RangeError(
      `[vitrum/pt-webgl2] ${context} declares ${requested} thin-film layers; ` +
      `the exact backend limit is ${THIN_FILM_LAYER_LIMIT}.`,
    );
  }
  return requested;
}

function mapLayer(
  ref: TextureRefLike | undefined,
  layerOf: TextureLayerLookup | undefined,
  storageClass: TextureAtlasStorageClass,
  colorSpace: TextureSampleColorSpace,
): number {
  if (ref?.handle == null || layerOf == null) return -1;
  const mapped = layerMapFor(layerOf, storageClass, colorSpace)?.get(ref.handle);
  if (mapped == null) return -1;
  return requireExactGlslInt(
    mapped,
    `pt-webgl2 ${storageClass}/${colorSpace} material texture layer`,
    -1,
  );
}

function defaultUvAttributeLayer(texCoord: number): number {
  if (texCoord === 0) return 2;
  if (texCoord === 1) return 4;
  return requireExactGlslInt(
    5 + (texCoord - 2),
    `pt-webgl2 default UV attribute layer for texCoord ${texCoord}`,
  );
}

function resolveUvAttributeLayer(
  ref: TextureRefLike | undefined,
  options: PackMaterialsTextureOptions,
): number {
  const texCoord = ref?.texCoord ?? 0;
  if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
    throw new RangeError(
      `pt-webgl2: TextureRef.texCoord must be a non-negative safe integer (got ${String(texCoord)})`,
    );
  }
  if (options.uvLayerByTexCoord == null) return defaultUvAttributeLayer(texCoord);
  const layer = options.uvLayerByTexCoord.get(texCoord);
  if (layer == null) {
    throw new Error(
      `pt-webgl2: scene UV layout has no attribute layer for TextureRef.texCoord ${texCoord}`,
    );
  }
  return requireExactGlslInt(
    layer,
    `pt-webgl2 UV attribute layer for TextureRef.texCoord ${texCoord}`,
  );
}

/**
 * Write the 2-texel UV-transform encoding the GLSL `readTextureTransform` reads
 * (row1 = (m00,m01,m02) at material texel `texelIdx`; row2 = (m10,m11,m12) at
 * `texelIdx+1`) for the core `rotate(uv * scale, rotation) + offset`
 * contract:
 *   row1 = (sx·cos, −sy·sin, offsetX), row2 = (sx·sin, sy·cos, offsetY).
 * Identity (no transform) → row1=(1,0,0), row2=(0,1,0).
 */
function writeTransform(
  data: Float32Array,
  base: number,
  texelIdx: number,
  ref: { transform?: { offset?: readonly number[]; scale?: readonly number[]; rotation?: number } } | undefined,
): void {
  const t = ref?.transform;
  const context = `pt-webgl2 material texture transform texel ${texelIdx}`;
  const component = (value: number, name: string): number => {
    const stored = requireFiniteFloat32(value, `${context} ${name}`);
    if (value !== 0 && stored === 0) {
      throw new RangeError(`${context} ${name} underflows WebGL float32 storage.`);
    }
    return stored;
  };
  const sx = component(t?.scale?.[0] ?? 1, 'scale.x');
  const sy = component(t?.scale?.[1] ?? 1, 'scale.y');
  const ox = component(t?.offset?.[0] ?? 0, 'offset.x');
  const oy = component(t?.offset?.[1] ?? 0, 'offset.y');
  const r = component(t?.rotation ?? 0, 'rotation');
  const c = requireFiniteFloat32(Math.cos(r), `${context} cos(rotation)`);
  const s = requireFiniteFloat32(Math.sin(r), `${context} sin(rotation)`);
  const row = (value: number, name: string): number =>
    requireFiniteFloat32(value, `${context} ${name}`);
  const o = base + texelIdx * 4;
  data[o] = row(sx * c, 'row0.x');
  data[o + 1] = row(-sy * s, 'row0.y');
  data[o + 2] = ox;
  data[o + 3] = 0;
  data[o + 4] = row(sx * s, 'row1.x');
  data[o + 5] = row(sy * c, 'row1.y');
  data[o + 6] = oy;
  data[o + 7] = 0;
}

function writeSamplerPolicy(
  data: Float32Array,
  offset: number,
  ref: {
    handle?: unknown;
    wrapS?: TextureWrapMode;
    wrapT?: TextureWrapMode;
    magFilter?: TextureFilterMode;
    minFilter?: TextureFilterMode;
    mipFilter?: TextureMipFilterMode;
  } | undefined,
  layerOf: TextureLayerLookup | undefined,
  storageClass: TextureAtlasStorageClass,
  colorSpace: TextureSampleColorSpace,
): void {
  const atlasMap = atlasMapFor(layerOf, storageClass);
  const placement =
    ref?.handle == null ? undefined : atlasMap?.placements?.[colorSpace].get(ref.handle);
  const dimensions =
    ref?.handle == null ? undefined : atlasMap?.dimensions?.get(ref.handle);
  const width = placement?.width ?? dimensions?.[0] ?? 0;
  const height = placement?.height ?? dimensions?.[1] ?? 0;
  const offsetX = placement?.x ?? 0;
  const offsetY = placement?.y ?? 0;
  // Pack the native source extent into the high integer bits while retaining
  // wrap mode modulo four. The other two lanes likewise retain mip/filter
  // policy modulo four while carrying the source-rectangle offset. A zero
  // extent is the low-level helper's legacy common-size mode; production atlas
  // builds always supply exact placement metadata.
  data[offset] = packedSamplerPolicyLane(
    width,
    WRAP_MODE_INDEX[ref?.wrapS ?? 'repeat'],
    'pt-webgl2 material sampler policy width/wrapS',
  );
  data[offset + 1] = packedSamplerPolicyLane(
    height,
    WRAP_MODE_INDEX[ref?.wrapT ?? 'repeat'],
    'pt-webgl2 material sampler policy height/wrapT',
  );
  data[offset + 2] = packedSamplerPolicyLane(
    offsetX,
    MIP_FILTER_INDEX[ref?.mipFilter ?? 'linear'],
    'pt-webgl2 material sampler policy offsetX/mipFilter',
  );
  const filterPair =
    FILTER_MODE_INDEX[ref?.magFilter ?? 'nearest'] +
    FILTER_MODE_INDEX[ref?.minFilter ?? 'nearest'] * 2;
  data[offset + 3] = packedSamplerPolicyLane(
    offsetY,
    filterPair,
    'pt-webgl2 material sampler policy offsetY/minMagFilter',
  );
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
 * key, the `MaterialSpec` texture-ref accessor, and its sample color space.
 * One row per resolved layer id keeps those per-map arguments in data rather
 * than 23 positional call sites.
 */
interface LayerIdMapEntry {
  readonly key: keyof LayerIds;
  readonly ref: (m: MaterialSpec) => TextureRefLike | undefined;
  readonly colorSpace: TextureSampleColorSpace;
  readonly storageClass: TextureAtlasStorageClass;
}

const LAYER_ID_MAP: readonly LayerIdMapEntry[] = [
  { key: 'baseColor', ref: (m) => m.baseColorMap, colorSpace: 'srgb', storageClass: 'ldr' },
  { key: 'metal', ref: (m) => m.metallicMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'rough', ref: (m) => m.roughnessMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'transmission', ref: (m) => m.transmissionMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'emissive', ref: (m) => m.emissiveMap, colorSpace: 'srgb', storageClass: 'hdr' },
  { key: 'normal', ref: (m) => m.normalMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'alpha', ref: (m) => m.alphaMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'clearcoat', ref: (m) => m.clearcoatMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'clearcoatRoughness', ref: (m) => m.clearcoatRoughnessMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'clearcoatNormal', ref: (m) => m.clearcoatNormalMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'sheenColor', ref: (m) => m.sheenColorMap, colorSpace: 'srgb', storageClass: 'ldr' },
  { key: 'sheenRoughness', ref: (m) => m.sheenRoughnessMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'iridescence', ref: (m) => m.iridescenceMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'iridescenceThickness', ref: (m) => m.iridescenceThicknessMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'specularColor', ref: (m) => m.specularColorMap, colorSpace: 'srgb', storageClass: 'ldr' },
  { key: 'specularIntensity', ref: (m) => m.specularIntensityMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'ao', ref: (m) => m.aoMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'lightMap', ref: (m) => m.lightMap, colorSpace: 'linear', storageClass: 'hdr' },
  { key: 'bump', ref: (m) => m.bumpMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'anisotropy', ref: (m) => m.anisotropyMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'thickness', ref: (m) => m.thicknessMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'frontLayerNormal', ref: (m) => m.frontLayer?.normalMap, colorSpace: 'linear', storageClass: 'ldr' },
  { key: 'backLayerNormal', ref: (m) => m.backLayer?.normalMap, colorSpace: 'linear', storageClass: 'ldr' },
];

/** D10.8: Resolve all atlas layer ids for a material in one pass (avoids re-calling mapLayer). */
function packLayerIds(
  m: MaterialSpec,
  layerOf: TextureLayerLookup | undefined,
): LayerIds {
  const ids = {} as LayerIds;
  for (const entry of LAYER_ID_MAP) {
    ids[entry.key] = mapLayer(
      entry.ref(m),
      layerOf,
      entry.storageClass,
      entry.colorSpace,
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
  materialIndex: number,
): number {
  // ── Scalar field resolution (core → fork semantics) ──────────────────────
  const color = m.baseColor;
  const metalness = m.metallic ?? 0.0;
  const roughness = m.roughness ?? 0.0;
  const ior = effectiveMaterialIor(m.ior); // default 1.5; authored 0 is the infinite-IOR endpoint
  const transmission = m.transmission ?? 0.0;
  // Contract default: pt-webgpu (materialTextures.ts) and walkaround-hybrid both
  // default emissiveIntensity to 1.0 when the field is absent; 0.0 would silently
  // black-out any emissive material whose host did not explicitly set the field.
  const packedEmission = packRadianceRgbScaleF32(
    m.emissive ?? [0.0, 0.0, 0.0],
    resolveEmissiveIntensity(m.emissiveIntensity),
    `pt-webgl2 material ${materialIndex} emissive`,
  );
  const emissiveIntensity = packedEmission.scale;
  const emissive: Vec3 = packedEmission.value;
  const normalScale = requireScaleFloat32(
    m.normalScale ?? 1.0,
    `pt-webgl2 material ${materialIndex}.normalScale`,
  ); // core carries a scalar; fork stores (x,y)
  const clearcoat = m.clearcoat ?? 0.0;
  const clearcoatRoughness = m.clearcoatRoughness ?? 0.0;
  const clearcoatNormalScale = requireScaleFloat32(
    m.clearcoatNormalScale ?? 1.0,
    `pt-webgl2 material ${materialIndex}.clearcoatNormalScale`,
  );
  const sheen = m.sheen ?? 0.0;
  const sheenColor: Vec3 = m.sheenColor ?? [0.0, 0.0, 0.0];
  const sheenRoughness = m.sheenRoughness ?? 0.0;
  const iridescence = m.iridescence ?? 0.0;
  const iridescenceIor = m.iridescenceIor ?? 1.3;
  const iridThicknessRange = m.iridescenceThicknessRange ?? [100, 400];
  const specularColor: Vec3 = m.specularColor ?? DEFAULT_SPECULAR_COLOR;
  const specularIntensity = m.specularIntensity ?? 1.0;
  const anisotropy = Math.max(0.0, Math.min(1.0, m.anisotropy ?? 0.0));
  const anisotropyRotation = requireScaleFloat32(
    m.anisotropyRotation ?? 0.0,
    `pt-webgl2 material ${materialIndex}.anisotropyRotation`,
  );
  const attenuationColor: Vec3 = m.attenuationColor ?? DEFAULT_ATTENUATION_COLOR;
  const attenuationDistance = m.attenuationDistance ?? Infinity;
  const sssMedium = resolveSssMedium(
    m,
    attenuationColor,
    attenuationDistance,
    `pt-webgl2 material ${materialIndex}`,
  );
  const thickness = m.thickness ?? 0.0;
  const opacity = m.opacity ?? 1.0;
  const alphaTest = m.alphaMode === 'mask' ? (m.alphaCutoff ?? 0.5) : 0.0;
  const transparent = m.alphaMode === 'blend';


  // Only transmissive material participates in the sheet-vs-solid topology
  // decision. A positive thickness or any authored bulk optical coefficient
  // then requires a closed geometric region. Clear zero-thickness
  // transmission is the sole virtual-sheet case; attenuation-only and
  // spectral-only glass retain geometric Beer distance even when no
  // representative thickness cap was authored.
  const hasBulkTransmission = transmission > 0.0 &&
    (thickness > 0.0 || sssMedium.active);
  // A sheet is a transmissive zero-thickness interface. Opaque materials are
  // neither sheets nor bulk media and must not enter the compound-interface
  // sampling path merely because they have no authored volume thickness.
  const isThinFilm = transmission > 0.0 && !hasBulkTransmission;

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
  // side: 0 = both orientations, 1 = front only. Authored double-sided
  // surfaces are two-sided directly. Closed transmissive volumes also keep
  // both interfaces traversable even when their exterior visibility policy is
  // one-sided, otherwise a refracted path could never hit its exit boundary.
  if (m.doubleSided === true || (!isThinFilm && transmission > 0.0)) {
    data[index++] = 0;
  } else {
    data[index++] = 1; // FrontSide
  }

  // sample 14 — matte / castShadow / vertexColors|fogVolume / flags
  data[index++] = 0; // matte (core has no matte field)
  data[index++] = m.castShadow === false ? 0 : 1;
  data[index++] =
    (m.vertexColors === true ? 1 : 0) |
    (sssMedium.scatters && transmission > 0 ? 4 : 0);
  {
    let flags = Number(transparent);
    if (m.doubleSided === true) flags |= DOUBLE_SIDED_BIT;
    if (m.shadingModel === 'unlit') flags |= UNLIT_BIT;
    if (m.meshEmitterCastShadowDisabled === true) flags |= MESH_EMITTER_CAST_SHADOW_DISABLED_BIT;
    if (materialEmissionExcludedFromMeshNee(m)) flags |= MESH_EMITTER_NEE_DISABLED_BIT;
    data[index++] = flags;
  }

  // sample 15 — sssSigmaT / sssAnisotropyG / dispersionStrength / thinFilmEnabled
  const scatteringAnisotropy = requireFiniteFloat32(
    m.scatteringAnisotropy ?? 0.0,
    `pt-webgl2 material ${materialIndex}.scatteringAnisotropy`,
  );
  const dispersionAbbe = m.dispersionAbbeNumber ?? 0.0;
  const dispersionStrength = requireNonNegativeFloat32(
    dispersionStrengthFromAbbe(ior, dispersionAbbe),
    `pt-webgl2 material ${materialIndex} derived dispersion strength`,
  );
  const thinFilmLayerCount = assertThinFilmLayerLimit(m);
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
  const thinFilmAngleDependent =
    m.thinFilmStack != null && m.thinFilmStack.angleDependent !== false;
  const packedFeatureFlags =
    (hasSpectral ? 1 : 0) | (hasFrontLayer ? 2 : 0) | (hasBackLayer ? 4 : 0);
  data[index++] = thinFilmIncidentIor;
  data[index++] = thinFilmAngleDependent ? 1.0 : 0.0;
  data[index++] = anisotropyRotation;
  data[index++] = packedFeatureFlags;

  // sample 18 — frontLayerTransmission.rgb / frontLayerRoughness
  const frontTx: Vec3 = frontLayer?.transmission ?? [1.0, 1.0, 1.0];
  const frontRoughness = requireFiniteFloat32(
    frontLayer?.roughness ?? -1.0,
    `pt-webgl2 material ${materialIndex}.frontLayer.roughness`,
  );
  data[index++] = frontTx[0];
  data[index++] = frontTx[1];
  data[index++] = frontTx[2];
  data[index++] = frontRoughness;

  // sample 19 — backLayerTransmission.rgb / backLayerRoughness
  const backTx: Vec3 = backLayer?.transmission ?? [1.0, 1.0, 1.0];
  const backRoughness = requireFiniteFloat32(
    backLayer?.roughness ?? -1.0,
    `pt-webgl2 material ${materialIndex}.backLayer.roughness`,
  );
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
function packSpectralGrid(
  data: Float32Array,
  index: number,
  m: MaterialSpec,
  materialIndex: number,
): number {
  const spectralCurve = m.spectralAttenuation ?? null;
  const hasSpectral = spectralCurve != null;
  const spectralDenom = Math.max(SPECTRAL_GRID_SAMPLE_COUNT - 1, 1);
  for (let s = 0; s < SPECTRAL_GRID_SAMPLE_COUNT; s += 1) {
    if (hasSpectral) {
      const t = s / spectralDenom;
      const lambdaNm = SPECTRAL_GRID_START_NM + t * (SPECTRAL_GRID_END_NM - SPECTRAL_GRID_START_NM);
      data[index++] = requireNonNegativeFloat32(
        sharedSampleSpectralCurve(
          spectralCurve,
          lambdaNm,
          PT_WEBGL2_SPECTRAL_CURVE_OPTIONS,
        ),
        `pt-webgl2 material ${materialIndex} spectral attenuation sample ${s}`,
      );
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
function packThinFilm(
  data: Float32Array,
  index: number,
  m: MaterialSpec,
  materialIndex: number,
): number {
  const thinFilmLayers = m.thinFilmStack?.layers ?? [];
  const thinFilmLayerCount = assertThinFilmLayerLimit(m);
  for (let layerIdx = 0; layerIdx < THIN_FILM_LAYER_LIMIT; layerIdx += 1) {
    if (layerIdx < thinFilmLayerCount) {
      const layer = thinFilmLayers[layerIdx]!;
      data[index++] = requireNonNegativeFloat32(
        layer.ior ?? 1.0,
        `pt-webgl2 material ${materialIndex}.thinFilmStack.layers[${layerIdx}].ior`,
      );
      data[index++] = requireNonNegativeFloat32(
        layer.thicknessNm ?? 0.0,
        `pt-webgl2 material ${materialIndex}.thinFilmStack.layers[${layerIdx}].thicknessNm`,
      );
      data[index++] = requireNonNegativeFloat32(
        layer.extinctionCoefficient ?? 0.0,
        `pt-webgl2 material ${materialIndex}.thinFilmStack.layers[${layerIdx}].extinctionCoefficient`,
      );
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
  materialIndex: number,
  envMapIntensity: number,
  lightMapIntensity: number,
  ids: LayerIds,
  options: PackMaterialsTextureOptions,
  layerOf: TextureLayerLookup | undefined,
): void {
  // samples 55..84 (30 texels): 15 texture-transform mat3s, 2 texels each, at
  // `texel 55 + 2k` (k per the GLSL `readTextureTransform` order in material_struct).
  // The GLSL only READS a transform when the map id != -1, so write one per mapped
  // texture (others stay zero / unread → identity). The fork's
  // `writeTextureMatrixToArray` is the analogue.
  // Transform-slot order matches the GLSL `readTextureTransform` calls in
  // material_mapped_rich.glsl.ts (firstTextureTransformIdx + 2k): map(0), metalness(2),
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
  // (mirrors readMaterialInfo s20/s21 in material_mapped_rich.glsl.ts).
  // Texel 86.a remains reserved; the arbitrary UV-layer selector table below is
  // the single runtime representation of per-map texCoord selection.
  let d3 = base + MATERIAL_D3_AUX_TEXEL * 4;
  data[d3++] = ids.ao;
  data[d3++] = ids.lightMap;
  data[d3++] = ids.bump;
  data[d3++] = envMapIntensity;
  data[d3++] = m.aoMapIntensity ?? 1.0;
  data[d3++] = lightMapIntensity;
  data[d3++] = requireScaleFloat32(
    m.bumpScale ?? 1.0,
    `pt-webgl2 material ${materialIndex}.bumpScale`,
  );
  data[d3++] = 0.0;
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
      layerOf,
      textureStorageClassForMapKey(field),
      textureColorSpaceForMapKey(field),
    );
  }

  // RFE-03 layer normals: appended payload so the historical base-map wrap and
  // spectral-reflectance lanes stay stable. Layer maps are face-selected shader
  // overrides; when absent the shader falls back to the ordinary normalMap path.
  const layerNormal = base + MATERIAL_LAYER_NORMAL_TEXEL_OFFSET * 4;
  data[layerNormal] = ids.frontLayerNormal;
  data[layerNormal + 1] = requireScaleFloat32(
    m.frontLayer?.normalScale ?? 1.0,
    `pt-webgl2 material ${materialIndex}.frontLayer.normalScale`,
  );
  data[layerNormal + 2] = ids.backLayerNormal;
  data[layerNormal + 3] = requireScaleFloat32(
    m.backLayer?.normalScale ?? 1.0,
    `pt-webgl2 material ${materialIndex}.backLayer.normalScale`,
  );
  if (ids.frontLayerNormal >= 0) {
    writeTransform(data, base, MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 1, m.frontLayer?.normalMap);
  }
  if (ids.backLayerNormal >= 0) {
    writeTransform(data, base, MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 3, m.backLayer?.normalMap);
  }
  const frontLayerSampler = base + (MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 5) * 4;
  writeSamplerPolicy(data, frontLayerSampler, m.frontLayer?.normalMap, layerOf, 'ldr', 'linear');
  const backLayerSampler = base + (MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 6) * 4;
  writeSamplerPolicy(data, backLayerSampler, m.backLayer?.normalMap, layerOf, 'ldr', 'linear');
  const layerUv = base + (MATERIAL_LAYER_NORMAL_TEXEL_OFFSET + 7) * 4;
  data[layerUv] = resolveUvAttributeLayer(m.frontLayer?.normalMap, options);
  data[layerUv + 1] = resolveUvAttributeLayer(m.backLayer?.normalMap, options);
  data[layerUv + 2] = 0;
  data[layerUv + 3] = 0;

  // Scalable selector table: one dense attributesArray layer per mapped-rich
  // slot. Authored texCoord ids never enter GLSL and therefore need not be dense.
  for (let mapIndex = 0; mapIndex < MATERIAL_MAP_FIELD_ORDER.length; mapIndex += 1) {
    const field = MATERIAL_MAP_FIELD_ORDER[mapIndex]!;
    const ref = m[field as keyof MaterialSpec] as TextureRefLike | undefined;
    data[base + MATERIAL_UV_SELECTOR_TEXEL_OFFSET * 4 + mapIndex] =
      resolveUvAttributeLayer(ref, options);
  }
}

function packSpectralReflectance(
  data: Float32Array,
  base: number,
  m: MaterialSpec,
  materialIndex: number,
): void {
  const baseColor = m.baseColor;
  const [c0, c1, c2] = rgbToSpectralCoefficients(
    baseColor[0],
    baseColor[1],
    baseColor[2],
  );
  const offset = base + MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET * 4;
  data[offset] = requireFiniteFloat32(
    c0,
    `pt-webgl2 material ${materialIndex} spectral reflectance coefficient 0`,
  );
  data[offset + 1] = requireFiniteFloat32(
    c1,
    `pt-webgl2 material ${materialIndex} spectral reflectance coefficient 1`,
  );
  data[offset + 2] = requireFiniteFloat32(
    c2,
    `pt-webgl2 material ${materialIndex} spectral reflectance coefficient 2`,
  );
  data[offset + 3] = 1;
}

function validatePackedMaterialStorage(
  data: Float32Array,
  materialCount: number,
): void {
  const attenuationDistanceLane = 12 * 4 + 3;
  for (let materialIndex = 0; materialIndex < materialCount; materialIndex += 1) {
    const base = materialIndex * MATERIAL_STRIDE;
    for (let lane = 0; lane < MATERIAL_STRIDE; lane += 1) {
      const value = data[base + lane]!;
      if (Number.isFinite(value)) continue;
      if (lane === attenuationDistanceLane && value === Number.POSITIVE_INFINITY) continue;
      throw new RangeError(
        `pt-webgl2 material ${materialIndex} packed lane ${lane} must be finite float32` +
          (lane === attenuationDistanceLane ? ' or the +Infinity attenuation-distance ABI' : '') +
          ` (got ${String(value)}).`,
      );
    }
  }
}

export function packMaterialsTexture(
  materials: readonly MaterialSpec[],
  layerOf?: TextureLayerLookup,
  options: PackMaterialsTextureOptions = {},
): MaterialsTextureData {
  const materialSnapshot = snapshotMaterialTextureInputs(materials);
  const materialCount = materialSnapshot.length;
  const packedMaterials = new Array<PackedMaterialSpec>(materialCount);
  const envMapIntensities = new Float32Array(materialCount);
  const lightMapIntensities = new Float32Array(materialCount);
  for (let materialIndex = 0; materialIndex < materialCount; materialIndex += 1) {
    const material = validatedPackedMaterial(
      materialSnapshot[materialIndex]!,
      materialIndex,
    );
    packedMaterials[materialIndex] = material;
    envMapIntensities[materialIndex] = requireNonNegativeFloat32(
      material.envMapIntensity ?? 1,
      `pt-webgl2 material ${materialIndex}.envMapIntensity`,
    );
    lightMapIntensities[materialIndex] = requireNonNegativeFloat32(
      material.lightMapIntensity ?? 1,
      `pt-webgl2 material ${materialIndex}.lightMapIntensity`,
    );
    assertThinFilmLayerLimit(material, `material ${materialIndex}`);
  }
  const pixelCount = materialCount * MATERIAL_PIXELS;
  if (!Number.isSafeInteger(pixelCount)) {
    throw new RangeError('pt-webgl2: material texture texel count exceeds JavaScript safe integers.');
  }
  const dim = squareDim(pixelCount);
  if (!Number.isSafeInteger(dim * dim * 4)) {
    throw new RangeError('pt-webgl2: material texture allocation exceeds JavaScript safe integers.');
  }
  const data = new Float32Array(dim * dim * 4);

  let index = 0;
  for (let i = 0; i < materialCount; i += 1) {
    const source = packedMaterials[i]!;
    const m = (options.vertexColorMaterialIds?.has(i) === true
      ? { ...source, vertexColors: true }
      : source) as PackedMaterialSpec;
    const base = index; // first float of this material's block

    const ids = packLayerIds(m, layerOf);

    // samples 0..19: scalar fields, layer ids, SSS, thin-film metadata
    index = packScalarSlots(data, index, m, ids, i);
    // samples 20..27: spectral attenuation grid
    index = packSpectralGrid(data, index, m, i);
    // samples 28..54: thin-film layer payload
    index = packThinFilm(data, index, m, i);
    // samples 55..92: texture-transform mat3s + D3 ao/light/bump auxiliary block
    packTextureTransforms(
      data,
      base,
      m,
      i,
      envMapIntensities[i]!,
      lightMapIntensities[i]!,
      ids,
      options,
      layerOf,
    );
    // sample 111: per-material Jakob-Hanika spectral reflectance coefficients.
    packSpectralReflectance(data, base, m, i);

    index = base + MATERIAL_STRIDE;
  }

  validatePackedMaterialStorage(data, materialCount);
  return { data, dim, kind: 'rgba32f', materialCount };
}
