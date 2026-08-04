// texturesArray — the two material-map arrays consumed by the trace shader:
// normalized RGBA8 parameters/colors and linear RGBA16F outgoing radiance.
// Per-map material fields remain plain layer ids because the field statically
// selects the appropriate sampler.
//
// THREE-free: a `TextureRef.handle` is opaque (`EnvironmentMapRef = unknown`). For
// the THREE-free path tracer the handle must expose CPU pixels — either a raw
// `{ width, height, data }` payload (the on-ramp form, like the env G2 bridge) or a
// THREE `DataTexture`-shaped `{ image: { data, width, height } }`. Image/ImageBitmap
// sources (canvas readback) are a documented host-side follow-up; this packer reads
// the DataTexture/raw forms that cover procedural + baked textures.
//
// All layers share one storage dimension (sampler2DArray requirement), but
// smaller sources share layers in power-of-two-aligned tiles. Each source keeps
// its authored width/height and independent mip pyramid. GLSL samples with the
// native extent plus the packed tile offset, so mixed resolutions do not change
// filtering footprints or bleed between neighboring sources.

import type {
  MaterialSpec,
  SurfaceAbsorptionLayer,
  TextureRef,
  UvTransform,
} from '@vitrum/core';
import {
  packNonNegativeRadianceScalarF32,
  packRadianceRgbProductF32,
  packRadianceRgbScaleF32,
  type RadianceRgb,
} from '@vitrum/shared-bvh';
import { resolveEmissiveIntensity } from '@vitrum/shared-samplers';
import {
  finiteFloat16Bits,
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from './halfFloat.js';

/** The material-map fields the fork GLSL samples (others are inert until wired).
 *  D3 (Wave C) added the clearcoat/sheen/iridescence/specular maps — the fork
 *  `get_surface_record` ALREADY samples each (see clearcoatMap…specularIntensityMap
 *  in get_surface_record_function.glsl.js), only the packer wired them as NO_TEXTURE.
 *  aoMap/lightMap/bumpMap are NEW GLSL (added to material_struct + get_surface_record)
 *  so they are gathered here too. */
const SAMPLED_MAP_KEYS = [
  'baseColorMap',
  'metallicMap',
  'roughnessMap',
  'transmissionMap',
  'emissiveMap',
  'normalMap',
  'alphaMap',
  // D3 — clearcoat / sheen / iridescence / specular maps (GLSL already samples).
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'specularColorMap',
  'specularIntensityMap',
  // D3 — aoMap / lightMap / bumpMap (new GLSL consumption sites).
  'aoMap',
  'lightMap',
  'bumpMap',
  // KHR_materials_anisotropy: RG = tangent direction, B = strength.
  'anisotropyMap',
  // KHR_materials_volume: G = scalar multiplier for thicknessFactor.
  'thicknessMap',
] as const satisfies ReadonlyArray<keyof MaterialSpec>;

const MATERIAL_TEXTURE_SNAPSHOT_KEYS = [
  ...SAMPLED_MAP_KEYS,
  // Displacement is consumed while the shared merge builds geometry rather
  // than by this atlas, but it is still a TextureRef validated by the material
  // packer and therefore belongs to the same immutable input boundary.
  'displacementMap',
] as const satisfies ReadonlyArray<keyof MaterialSpec>;

const TEXTURE_REF_SNAPSHOT_KEYS = [
  'handle',
  'texCoord',
  'transform',
  'wrapS',
  'wrapT',
  'magFilter',
  'minFilter',
  'mipFilter',
] as const satisfies ReadonlyArray<keyof TextureRef>;

const UV_TRANSFORM_SNAPSHOT_KEYS = [
  'offset',
  'scale',
  'rotation',
] as const satisfies ReadonlyArray<keyof UvTransform>;

const SURFACE_LAYER_SNAPSHOT_KEYS = [
  'transmission',
  'roughness',
  'normalMap',
  'normalScale',
] as const satisfies ReadonlyArray<keyof SurfaceAbsorptionLayer>;

const MATERIAL_TEXTURE_INPUT_SNAPSHOT = Symbol('vitrum.pt-webgl2.material-texture-input-snapshot');

type MaterialTextureInputSnapshot = readonly MaterialSpec[] & {
  readonly [MATERIAL_TEXTURE_INPUT_SNAPSHOT]: true;
};

interface MaterialTextureSnapshotContext {
  readonly materials: WeakMap<object, MaterialSpec>;
  readonly textureRefs: WeakMap<object, TextureRef>;
  readonly transforms: WeakMap<object, UvTransform>;
  readonly vectorPairs: WeakMap<object, unknown>;
  readonly surfaceLayers: WeakMap<object, SurfaceAbsorptionLayer>;
}

function isRecord(value: unknown): value is Record<PropertyKey, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function defineSnapshotValue(
  target: Record<PropertyKey, unknown>,
  key: PropertyKey,
  value: unknown,
  enumerable = true,
): void {
  Object.defineProperty(target, key, {
    value,
    enumerable,
    configurable: true,
    writable: true,
  });
}

function snapshotVectorPair(
  value: unknown,
  context: MaterialTextureSnapshotContext,
): unknown {
  if (
    value == null ||
    typeof value !== 'object' ||
    (!Array.isArray(value) && !ArrayBuffer.isView(value))
  ) {
    return value;
  }
  const object = value as object;
  const cached = context.vectorPairs.get(object);
  if (cached !== undefined) return cached;

  const length = (value as { readonly length?: unknown }).length;
  // A UV transform pair has exactly two entries. Preserve valid input by
  // reading each authored element once. Malformed lengths become an empty
  // ordinary array so the canonical core validator rejects them without ever
  // revisiting a hostile source container.
  const snapshot: unknown[] = [];
  context.vectorPairs.set(object, snapshot);
  if (length === 2) {
    snapshot.push(
      (value as ArrayLike<unknown>)[0],
      (value as ArrayLike<unknown>)[1],
    );
  }
  return Object.freeze(snapshot);
}

function snapshotUvTransform(
  value: unknown,
  context: MaterialTextureSnapshotContext,
): unknown {
  if (!isRecord(value)) return value;
  const cached = context.transforms.get(value);
  if (cached !== undefined) return cached;

  const snapshot: Record<PropertyKey, unknown> = {};
  context.transforms.set(value, snapshot as unknown as UvTransform);
  const ownKeys = Reflect.ownKeys(value);
  const ownKeySet = new Set<PropertyKey>(ownKeys);
  for (const key of ownKeys) {
    defineSnapshotValue(
      snapshot,
      key,
      value[key],
      Object.prototype.propertyIsEnumerable.call(value, key),
    );
  }
  // Preserve inherited known fields while still reading each getter once.
  for (const key of UV_TRANSFORM_SNAPSHOT_KEYS) {
    if (ownKeySet.has(key)) continue;
    const field = value[key];
    if (field !== undefined) defineSnapshotValue(snapshot, key, field);
  }
  if (Object.prototype.hasOwnProperty.call(snapshot, 'offset')) {
    snapshot.offset = snapshotVectorPair(snapshot.offset, context);
  }
  if (Object.prototype.hasOwnProperty.call(snapshot, 'scale')) {
    snapshot.scale = snapshotVectorPair(snapshot.scale, context);
  }
  return Object.freeze(snapshot);
}

function snapshotTextureRef(
  value: unknown,
  context: MaterialTextureSnapshotContext,
): unknown {
  if (!isRecord(value)) return value;
  const cached = context.textureRefs.get(value);
  if (cached !== undefined) return cached;

  const snapshot: Record<PropertyKey, unknown> = {};
  context.textureRefs.set(value, snapshot as unknown as TextureRef);
  const ownKeys = Reflect.ownKeys(value);
  const ownKeySet = new Set<PropertyKey>(ownKeys);
  for (const key of ownKeys) {
    defineSnapshotValue(
      snapshot,
      key,
      value[key],
      Object.prototype.propertyIsEnumerable.call(value, key),
    );
  }
  for (const key of TEXTURE_REF_SNAPSHOT_KEYS) {
    if (ownKeySet.has(key)) continue;
    const field = value[key];
    if (field !== undefined || key === 'handle') {
      defineSnapshotValue(snapshot, key, field);
    }
  }
  if (Object.prototype.hasOwnProperty.call(snapshot, 'transform')) {
    snapshot.transform = snapshotUvTransform(snapshot.transform, context);
  }
  return Object.freeze(snapshot);
}

function snapshotSurfaceLayer(
  value: unknown,
  context: MaterialTextureSnapshotContext,
): unknown {
  if (!isRecord(value)) return value;
  const cached = context.surfaceLayers.get(value);
  if (cached !== undefined) return cached;

  const snapshot: Record<PropertyKey, unknown> = {};
  context.surfaceLayers.set(
    value,
    snapshot as unknown as SurfaceAbsorptionLayer,
  );
  const ownKeys = Reflect.ownKeys(value);
  const ownKeySet = new Set<PropertyKey>(ownKeys);
  for (const key of ownKeys) {
    defineSnapshotValue(
      snapshot,
      key,
      value[key],
      Object.prototype.propertyIsEnumerable.call(value, key),
    );
  }
  for (const key of SURFACE_LAYER_SNAPSHOT_KEYS) {
    if (ownKeySet.has(key)) continue;
    const field = value[key];
    if (field !== undefined) defineSnapshotValue(snapshot, key, field);
  }
  if (Object.prototype.hasOwnProperty.call(snapshot, 'normalMap')) {
    snapshot.normalMap = snapshotTextureRef(snapshot.normalMap, context);
  }
  return Object.freeze(snapshot);
}

function snapshotMaterial(
  source: MaterialSpec,
  materialIndex: number,
  context: MaterialTextureSnapshotContext,
): MaterialSpec {
  if (!isRecord(source)) {
    throw new TypeError(`pt-webgl2 material ${materialIndex} must be a material record.`);
  }
  const cached = context.materials.get(source);
  if (cached !== undefined) return cached;

  const snapshot: Record<PropertyKey, unknown> = {};
  context.materials.set(source, snapshot as unknown as MaterialSpec);
  // This is the sole read of every authored top-level material property. The
  // rest of the WebGL2 layout/atlas/material transaction consumes plain data.
  const ownKeys = Reflect.ownKeys(source);
  const ownKeySet = new Set<PropertyKey>(ownKeys);
  for (const key of ownKeys) {
    defineSnapshotValue(
      snapshot,
      key,
      source[key as keyof MaterialSpec],
      Object.prototype.propertyIsEnumerable.call(source, key),
    );
  }
  for (const key of MATERIAL_TEXTURE_SNAPSHOT_KEYS) {
    if (!ownKeySet.has(key)) {
      const field = source[key];
      if (field !== undefined) defineSnapshotValue(snapshot, key, field);
    }
    if (Object.prototype.hasOwnProperty.call(snapshot, key)) {
      snapshot[key] = snapshotTextureRef(snapshot[key], context);
    }
  }
  for (const key of ['frontLayer', 'backLayer'] as const) {
    if (ownKeySet.has(key)) continue;
    const field = source[key];
    if (field !== undefined) defineSnapshotValue(snapshot, key, field);
  }
  if (Object.prototype.hasOwnProperty.call(snapshot, 'frontLayer')) {
    snapshot.frontLayer = snapshotSurfaceLayer(snapshot.frontLayer, context);
  }
  if (Object.prototype.hasOwnProperty.call(snapshot, 'backLayer')) {
    snapshot.backLayer = snapshotSurfaceLayer(snapshot.backLayer, context);
  }
  return Object.freeze(snapshot) as unknown as MaterialSpec;
}

/**
 * Capture one immutable-by-value WebGL2 material-input transaction.
 *
 * Authored list indices, material properties, TextureRef properties, UV
 * transforms, and transform vector elements are each evaluated once. Reused
 * material/TextureRef/transform identities are memoized within the transaction,
 * so alternating getters cannot make UV layout, atlas membership, descriptor
 * packing, and shader material records disagree.
 *
 * @internal Exported for the scene-build orchestrator and hostile-input tests.
 */
export function snapshotMaterialTextureInputs(
  materials: readonly MaterialSpec[],
): readonly MaterialSpec[] {
  if (
    Object.getOwnPropertyDescriptor(materials, MATERIAL_TEXTURE_INPUT_SNAPSHOT)
      ?.value === true
  ) {
    return materials;
  }
  const materialCount = materials.length;
  if (!Number.isSafeInteger(materialCount) || materialCount < 0) {
    throw new RangeError('pt-webgl2: material count must be a non-negative safe integer.');
  }
  const context: MaterialTextureSnapshotContext = {
    materials: new WeakMap(),
    textureRefs: new WeakMap(),
    transforms: new WeakMap(),
    vectorPairs: new WeakMap(),
    surfaceLayers: new WeakMap(),
  };
  const snapshot = new Array<MaterialSpec>(materialCount);
  for (let materialIndex = 0; materialIndex < materialCount; materialIndex += 1) {
    // Read each authored list slot exactly once before any packing phase.
    snapshot[materialIndex] = snapshotMaterial(
      materials[materialIndex]!,
      materialIndex,
      context,
    );
  }
  Object.defineProperty(snapshot, MATERIAL_TEXTURE_INPUT_SNAPSHOT, {
    value: true,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  return Object.freeze(snapshot) as MaterialTextureInputSnapshot;
}

// ── D10.12: TextureHandleHint ─────────────────────────────────────────────────
// Optional hints that a TextureRef handle can expose to make readHandlePixels
// unambiguous without relying on the stride heuristic (which can mis-classify
// 3-channel RGB data, for example). A host that provides these hints avoids the
// ambiguous-stride console.warn and gets deterministic decoding.
//
// Usage: attach a `__vitrum_hint__` property to the texture handle object, OR
// pass a wrapper that implements this interface as the handle.
//
// channels: 1 | 2 | 3 | 4 — number of channels per pixel in `data`.
//   If omitted, only exact one- or four-channel payloads can be inferred. RG/RGB
//   payloads require the hint so authored channel layouts are never guessed.
// dataType: 'uint8' | 'uint16' | 'float16' | 'half-float' | 'float32' —
//   encoding of each channel value. Uint16Array without a dataType hint is
//   treated as normalized uint16, matching glTF decode handles.
//   If omitted, inferred from the ArrayLike type (Uint8Array→uint8 etc.).

export interface TextureHandleHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float16' | 'half-float' | 'float32';
  /**
   * Source encoding hint. By default, color/tint map roles are treated as sRGB
   * sources. RGBA8 color-role tiles retain/produce sRGB-coded RGB for precision
   * and are decoded per shader fetch; RGBA16F radiance tiles are CPU-linear.
   * Scalar/data map roles stay linear. Set `colorSpace:'linear'` for a color
   * map handle that is already linear-light.
   */
  readonly colorSpace?: TextureSampleColorSpace;
}

export type TextureSampleColorSpace = 'srgb' | 'linear';
export type TextureAtlasStorageClass = 'ldr' | 'hdr';
export type TextureAtlasFormat = 'rgba8unorm' | 'rgba16f';
export type TextureAtlasData = Uint8Array | Uint16Array;

export interface TextureAtlasPlacement {
  readonly layer: number;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface TextureAtlasLayerMap {
  readonly srgb: ReadonlyMap<unknown, number>;
  readonly linear: ReadonlyMap<unknown, number>;
  /** Native level-0 width/height for each handle. */
  readonly dimensions?: ReadonlyMap<unknown, readonly [number, number]>;
  /** Role-aware source rectangles. A handle sampled in both roles has two placements. */
  readonly placements?: {
    readonly srgb: ReadonlyMap<unknown, TextureAtlasPlacement>;
    readonly linear: ReadonlyMap<unknown, TextureAtlasPlacement>;
  };
}

export interface TextureAtlas {
  /**
   * RGBA8 for normalized material parameters or RGBA16F for outgoing-radiance
   * maps. `layerCount` layers each have `dim × dim` texels.
   */
  readonly data: TextureAtlasData;
  readonly storageClass: TextureAtlasStorageClass;
  readonly format: TextureAtlasFormat;
  readonly dim: number;
  /** Complete format-native mip chain; level 0 aliases `data`. */
  readonly mipLevels: readonly TextureAtlasMipLevel[];
  readonly layerCount: number;
  /**
   * Production material lookup. Color/tint and data roles may share a handle
   * safely; packed placements carry both the array layer and tile offset.
   */
  readonly layerOfByColorSpace: TextureAtlasLayerMap;
  /** Native width/height by source entry, in first-seen handle/role order. */
  readonly sourceDimensions: readonly (readonly [number, number])[];
  /** Packed source rectangle by source entry, parallel to `sourceDimensions`. */
  readonly sourcePlacements: readonly TextureAtlasPlacement[];
}

export interface TextureAtlasMipLevel {
  readonly data: TextureAtlasData;
  readonly dim: number;
}

/** Material records choose the atlas from their statically-known map slot. */
export interface MaterialTextureAtlasLayerMaps {
  readonly ldr: TextureAtlasLayerMap | null;
  readonly hdr: TextureAtlasLayerMap | null;
}

export function textureAtlasLayerCapacity(layerCount: number, maxLayers: number): number {
  const count = Math.max(0, Math.floor(layerCount));
  const limit = Math.max(0, Math.floor(maxLayers));
  if (count === 0 || limit === 0) return 0;
  // Atlas-membership changes are rebuilt transactionally by setScene; there is
  // no resident texSubImage3D layer-growth path. Reserve exactly the live layers
  // instead of doubling immutable storage for headroom that cannot be consumed.
  return Math.min(limit, count);
}

/** Four normalized bytes preserve ordinary map storage at the device's full extent. */
export const MATERIAL_LDR_TEXTURE_ATLAS_BYTES_PER_TEXEL = 4;
/** Radiance maps retain values above one in IEEE-754 binary16. */
export const MATERIAL_HDR_TEXTURE_ATLAS_BYTES_PER_TEXEL = 8;
/**
 * The LDR/HDR atlas pair shares this explicit ceiling for both retained CPU mip
 * chains and immutable GPU storage, including spare layers. The single-atlas
 * packing seam is independently bounded by the same ceiling.
 */
export const MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES = 512 * 1024 * 1024;

function checkedTextureAtlasShape(dim: number, layerCount: number): void {
  if (
    !Number.isSafeInteger(dim) ||
    !Number.isSafeInteger(layerCount) ||
    dim <= 0 ||
    layerCount <= 0
  ) {
    throw new Error(
      `[pt-webgl2] material texture atlas dimensions and layer count must be positive safe integers ` +
        `(received dim=${String(dim)}, layers=${String(layerCount)})`,
    );
  }
}

function textureAtlasBytesPerTexel(storageClass: TextureAtlasStorageClass): number {
  return storageClass === 'hdr'
    ? MATERIAL_HDR_TEXTURE_ATLAS_BYTES_PER_TEXEL
    : MATERIAL_LDR_TEXTURE_ATLAS_BYTES_PER_TEXEL;
}

function textureAtlasStorageByteLengthBigInt(
  dim: number,
  layerCount: number,
  storageClass: TextureAtlasStorageClass,
): bigint {
  checkedTextureAtlasShape(dim, layerCount);
  let totalTexels = 0n;
  let levelDim = dim;
  while (true) {
    totalTexels += BigInt(levelDim) * BigInt(levelDim) * BigInt(layerCount);
    if (levelDim === 1) break;
    levelDim = Math.max(1, Math.floor(levelDim / 2));
  }
  return totalTexels * BigInt(textureAtlasBytesPerTexel(storageClass));
}

/** Exact bytes reserved by a complete format-native mip chain. */
export function textureAtlasStorageByteLength(
  dim: number,
  layerCount: number,
  storageClass: TextureAtlasStorageClass = 'ldr',
): number {
  const bytes = textureAtlasStorageByteLengthBigInt(dim, layerCount, storageClass);
  if (bytes > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(
      `[pt-webgl2] material texture atlas byte length exceeds JavaScript's safe integer range ` +
        `(dim=${dim}, layers=${layerCount}, bytes=${bytes.toString()})`,
    );
  }
  return Number(bytes);
}

/**
 * Validate live atlas storage against the device and byte ceilings. Atlas
 * membership changes rebuild the array, so capacity is exactly `layerCount`.
 */
export function textureAtlasLayerCapacityForStorage(
  dim: number,
  layerCount: number,
  maxLayers: number,
  storageClass: TextureAtlasStorageClass = 'ldr',
): number {
  checkedTextureAtlasShape(dim, layerCount);
  if (!Number.isSafeInteger(maxLayers) || maxLayers <= 0) {
    throw new Error(
      `[pt-webgl2] MAX_ARRAY_TEXTURE_LAYERS must be a positive safe integer ` +
        `(received ${String(maxLayers)})`,
    );
  }
  const bytesPerLayer = textureAtlasStorageByteLengthBigInt(dim, 1, storageClass);
  const budgetLayers = Number(
    BigInt(MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES) / bytesPerLayer,
  );
  if (layerCount > budgetLayers) {
    const liveBytes = textureAtlasStorageByteLengthBigInt(dim, layerCount, storageClass);
    throw new Error(
      `[pt-webgl2] material texture atlas ${dim}²×${layerCount} mip chain requires ` +
        `${liveBytes.toString()} bytes, exceeding the ` +
        `${MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES}-byte storage budget.`,
    );
  }
  if (layerCount > maxLayers) {
    throw new Error(
      `[pt-webgl2] material texture atlas needs ${layerCount} layers but the device ` +
        `supports ${maxLayers}.`,
    );
  }
  return textureAtlasLayerCapacity(layerCount, Math.min(maxLayers, budgetLayers));
}

export interface MaterialTextureAtlasLayerCapacities {
  readonly ldr: number;
  readonly hdr: number;
  readonly storageBytes: number;
}

/**
 * Validate the paired live arrays under one 512 MiB ceiling. Atlas membership
 * mutations rebuild the full scene transaction, so neither array reserves
 * immutable spare layers.
 */
export function materialTextureAtlasLayerCapacities(
  ldr: Pick<TextureAtlas, 'dim' | 'layerCount'> | null,
  hdr: Pick<TextureAtlas, 'dim' | 'layerCount'> | null,
  maxLayers: number,
): MaterialTextureAtlasLayerCapacities {
  if (!Number.isSafeInteger(maxLayers) || maxLayers <= 0) {
    throw new Error(
      `[pt-webgl2] MAX_ARRAY_TEXTURE_LAYERS must be a positive safe integer ` +
        `(received ${String(maxLayers)})`,
    );
  }
  const ldrCapacity =
    ldr == null
      ? 0
      : textureAtlasLayerCapacityForStorage(
          ldr.dim,
          ldr.layerCount,
          maxLayers,
          'ldr',
        );
  const hdrCapacity =
    hdr == null
      ? 0
      : textureAtlasLayerCapacityForStorage(
          hdr.dim,
          hdr.layerCount,
          maxLayers,
          'hdr',
        );
  const ldrBytesPerLayer =
    ldr == null ? 0 : textureAtlasStorageByteLength(ldr.dim, 1, 'ldr');
  const hdrBytesPerLayer =
    hdr == null ? 0 : textureAtlasStorageByteLength(hdr.dim, 1, 'hdr');
  const liveBytes =
    (ldr?.layerCount ?? 0) * ldrBytesPerLayer +
    (hdr?.layerCount ?? 0) * hdrBytesPerLayer;
  if (liveBytes > MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES) {
    throw new Error(
      `[pt-webgl2] combined material texture atlases require ${liveBytes} bytes ` +
        `(${ldr?.layerCount ?? 0} RGBA8 layers and ${hdr?.layerCount ?? 0} RGBA16F layers), ` +
        `exceeding the ${MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES}-byte storage budget.`,
    );
  }

  return {
    ldr: ldrCapacity,
    hdr: hdrCapacity,
    storageBytes: liveBytes,
  };
}

interface TextureAtlasBuildOptions {
  readonly warningMethod?: string;
  /** Device array-layer limit used to reject an impossible atlas before source validation. */
  readonly maxArrayTextureLayers?: number;
  /** `ldr` is the ordinary-map RGBA8 atlas; `hdr` contains emissive/light maps. */
  readonly storageClass?: TextureAtlasStorageClass;
}

function handleType(handle: unknown): string {
  return handle == null ? 'null' : Object.prototype.toString.call(handle);
}

const SRGB_MAP_KEYS = new Set<keyof MaterialSpec>([
  'baseColorMap',
  'emissiveMap',
  'sheenColorMap',
  'specularColorMap',
]);

const HDR_MAP_KEYS = new Set<keyof MaterialSpec>(['emissiveMap', 'lightMap']);

export function textureColorSpaceForMapKey(key: keyof MaterialSpec): TextureSampleColorSpace {
  return SRGB_MAP_KEYS.has(key) ? 'srgb' : 'linear';
}

export function textureStorageClassForMapKey(
  key: keyof MaterialSpec,
): TextureAtlasStorageClass {
  return HDR_MAP_KEYS.has(key) ? 'hdr' : 'ldr';
}

interface CollectedTextureHandle {
  readonly handle: unknown;
  readonly colorSpace: TextureSampleColorSpace;
  /** Bit n accepts an authored n-channel payload. Shared handles intersect
   *  every consuming role so one permissive use cannot hide a stricter one. */
  allowedChannelsMask: number;
  readonly roleLabels: string[];
  usesMipFiltering: boolean;
}

const CHANNEL_1 = 1 << 1;
const CHANNEL_2 = 1 << 2;
const CHANNEL_3 = 1 << 3;
const CHANNEL_4 = 1 << 4;
const ANY_CHANNELS = CHANNEL_1 | CHANNEL_2 | CHANNEL_3 | CHANNEL_4;

interface TextureChannelRole {
  readonly label: string;
  readonly allowedChannelsMask: number;
}

function textureChannelRoleForMapKey(key: keyof MaterialSpec): TextureChannelRole {
  if (key === 'normalMap' || key === 'clearcoatNormalMap') {
    return { label: key, allowedChannelsMask: CHANNEL_3 | CHANNEL_4 };
  }
  if (key === 'anisotropyMap') {
    return { label: key, allowedChannelsMask: CHANNEL_3 | CHANNEL_4 };
  }
  if (key === 'sheenRoughnessMap' || key === 'specularIntensityMap') {
    return { label: key, allowedChannelsMask: CHANNEL_4 };
  }
  if (key === 'metallicMap') {
    // Public raw expansion is R -> RRR1 and RG -> RG01. Metallic consumes B:
    // a luminance source is meaningful, while an exact RG source would silently
    // force every metallic sample to zero. RGB/RGBA author B directly.
    return {
      label: key,
      allowedChannelsMask: CHANNEL_1 | CHANNEL_3 | CHANNEL_4,
    };
  }
  return { label: key, allowedChannelsMask: ANY_CHANNELS };
}

const LAYER_NORMAL_CHANNEL_ROLE: TextureChannelRole = {
  label: 'surface-layer normalMap',
  allowedChannelsMask: CHANNEL_3 | CHANNEL_4,
};

function collectHandle(
  handles: CollectedTextureHandle[],
  seen: Map<unknown, Map<TextureSampleColorSpace, number>>,
  ref: { readonly handle?: unknown; readonly mipFilter?: string } | undefined,
  colorSpace: TextureSampleColorSpace,
  role: TextureChannelRole,
): void {
  const handle = ref?.handle;
  if (handle == null) return;
  let seenByColorSpace = seen.get(handle);
  if (seenByColorSpace == null) {
    seenByColorSpace = new Map<TextureSampleColorSpace, number>();
    seen.set(handle, seenByColorSpace);
  }
  const existingIndex = seenByColorSpace.get(colorSpace);
  // Omitted mipFilter follows the public TextureRef default (`linear`), so the
  // generated mip chain is reachable unless the author explicitly selects
  // `none`.
  const usesMipFiltering = ref?.mipFilter !== 'none';
  if (existingIndex == null) {
    seenByColorSpace.set(colorSpace, handles.length);
    handles.push({
      handle,
      colorSpace,
      allowedChannelsMask: role.allowedChannelsMask,
      roleLabels: [role.label],
      usesMipFiltering,
    });
  } else {
    const existing = handles[existingIndex];
    if (existing != null) {
      existing.allowedChannelsMask &= role.allowedChannelsMask;
      if (!existing.roleLabels.includes(role.label)) existing.roleLabels.push(role.label);
      if (existing.allowedChannelsMask === 0) {
        throw new Error(
          `[pt-webgl2] one material texture handle has incompatible channel roles: ` +
            existing.roleLabels.join(', '),
        );
      }
      if (usesMipFiltering) existing.usesMipFiltering = true;
    }
  }
}

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

function texturePayloadError(
  handle: unknown,
  options: TextureAtlasBuildOptions | undefined,
  reason: string,
): Error {
  const operation = options?.warningMethod != null ? ` during ${options.warningMethod}` : '';
  return new Error(
    `[pt-webgl2] authored material texture${operation} is not CPU-readable: ${reason}. ` +
      'Provide one coherent raw { width, height, data } payload, a DataTexture-shaped image, ' +
      'or an immutable cpuMirror with an explicit channels hint for RG/RGB data. ' +
      `Handle type: ${handleType(handle)}`,
  );
}

function checkedHdrFloat16Bits(
  value: number,
  handle: unknown,
  options: TextureAtlasBuildOptions | undefined,
  context: string,
): number {
  try {
    return finiteFloat16Bits(value, context);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw texturePayloadError(handle, options, reason);
  }
}

const VALID_CHANNEL_COUNTS = new Set<number>([1, 2, 3, 4]);
const VALID_DATA_TYPES = new Set<string>(['uint8', 'uint16', 'float16', 'half-float', 'float32']);

interface TexturePixelPayload {
  readonly width?: number;
  readonly height?: number;
  readonly data?: ArrayLike<number>;
  readonly channels?: number;
  readonly dataType?: string;
  readonly colorSpace?: string;
}

function hasSharedArrayBufferBacking(data: ArrayLike<number>): boolean {
  if (!ArrayBuffer.isView(data)) return false;
  const inputBuffer = data.buffer;
  return typeof SharedArrayBuffer !== 'undefined' &&
    inputBuffer instanceof SharedArrayBuffer;
}

type ResolvedTextureDataType = NonNullable<TextureHandleHint['dataType']>;

interface InspectedTexturePixels {
  readonly width: number;
  readonly height: number;
  readonly pixelCount: number;
  readonly data: ArrayLike<number>;
  readonly stride: number;
  readonly dataType: ResolvedTextureDataType;
  readonly sourceColorSpace?: TextureSampleColorSpace;
  /** Bytes held by the immutable raw snapshot; absent before staging. */
  readonly stagingByteLength?: number;
}

interface TextureAtlasCpuBudget {
  usedBytes: bigint;
  readonly limitBytes: bigint;
}

/**
 * One packing transaction owns one inspection token and one immutable raw
 * snapshot per opaque handle. Atlas entries remain role/color-space specific,
 * but they never re-observe user-controlled metadata or texels.
 */
interface TextureAtlasSourceTransaction {
  readonly inspectedByHandle: Map<unknown, InspectedTexturePixels>;
  readonly stagedByHandle: Map<unknown, InspectedTexturePixels>;
}

function createTextureAtlasSourceTransaction(): TextureAtlasSourceTransaction {
  return {
    inspectedByHandle: new Map<unknown, InspectedTexturePixels>(),
    stagedByHandle: new Map<unknown, InspectedTexturePixels>(),
  };
}

function textureSourceElementBytes(dataType: ResolvedTextureDataType): number {
  if (dataType === 'uint8') return 1;
  return dataType === 'float32' ? 4 : 2;
}

function reserveTextureAtlasCpuBytes(
  budget: TextureAtlasCpuBudget,
  byteCount: bigint,
  label: string,
): void {
  if (byteCount < 0n || byteCount > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError(`[pt-webgl2] ${label} byte length exceeds JavaScript's safe range.`);
  }
  const aggregate = budget.usedBytes + byteCount;
  if (aggregate > budget.limitBytes) {
    throw new RangeError(
      `[pt-webgl2] ${label} requires ${byteCount.toString()} CPU bytes ` +
        `(aggregate ${aggregate.toString()}), exceeding the ` +
        `${budget.limitBytes.toString()}-byte material texture staging budget.`,
    );
  }
  budget.usedBytes = aggregate;
}

function releaseTextureAtlasCpuBytes(
  budget: TextureAtlasCpuBudget,
  byteCount: bigint,
): void {
  if (byteCount < 0n || byteCount > budget.usedBytes) {
    throw new Error('[pt-webgl2] internal material texture staging-budget underflow.');
  }
  budget.usedBytes -= byteCount;
}

function decodeTextureChannel(dataType: ResolvedTextureDataType, value: number): number {
  switch (dataType) {
    case 'uint8':
      return value / 255;
    case 'uint16':
      return value / 65535;
    case 'float16':
    case 'half-float':
      return float16BitsToFloat32(value);
    case 'float32':
      return value;
  }
}

/** Read RGBA float pixels from a TextureRef handle (raw payload or DataTexture-shaped).
 *  D10.12: optional TextureHandleHint (`__vitrum_hint__` property on the handle, or the
 *  handle itself implementing TextureHandleHint) provides explicit channels/dataType.
 *  Every authored payload is validated exactly and failure is synchronous: no texture
 *  may enter the retained scene after a guessed layout or partial CPU decode. */
function inspectHandlePixels(
  handle: unknown,
  options?: TextureAtlasBuildOptions,
): InspectedTexturePixels {
  const h = handle as {
    width?: number;
    height?: number;
    data?: ArrayLike<number>;
    image?: TexturePixelPayload;
    cpuMirror?: TexturePixelPayload;
    // D10.12: optional hint as a direct property on the handle
    __vitrum_hint__?: TextureHandleHint;
    channels?: number;
    dataType?: string;
    colorSpace?: string;
  } | null;
  if (h == null) {
    throw texturePayloadError(handle, options, 'the handle is null');
  }
  // Select one coherent payload. In particular, a partial cpuMirror must not be
  // completed with dimensions or data from a separate mutable image object.
  const cpuMirror = h.cpuMirror;
  let src: ArrayLike<number> | undefined;
  let rawWidth: number | undefined;
  let rawHeight: number | undefined;
  let sourceChannels: number | undefined;
  let sourceDataType: string | undefined;
  let selectedColorSpace: string | undefined;
  let selectedHandlePayload = false;
  if (cpuMirror != null) {
    src = cpuMirror.data;
    rawWidth = cpuMirror.width;
    rawHeight = cpuMirror.height;
    sourceChannels = cpuMirror.channels;
    sourceDataType = cpuMirror.dataType;
    selectedColorSpace = cpuMirror.colorSpace;
  } else {
    const handleData = h.data;
    if (handleData != null) {
      selectedHandlePayload = true;
      src = handleData;
      rawWidth = h.width;
      rawHeight = h.height;
      sourceChannels = h.channels;
      sourceDataType = h.dataType;
      selectedColorSpace = h.colorSpace;
    } else {
      const image = h.image;
      if (image != null) {
        src = image.data;
        rawWidth = image.width;
        rawHeight = image.height;
        sourceChannels = image.channels;
        sourceDataType = image.dataType;
        selectedColorSpace = image.colorSpace;
      }
    }
  }
  if (src == null) {
    throw texturePayloadError(
      handle,
      options,
      'no cpuMirror, raw data, or DataTexture-shaped image data was supplied',
    );
  }
  if (hasSharedArrayBufferBacking(src)) {
    throw texturePayloadError(
      handle,
      options,
      'SharedArrayBuffer-backed material texels are not accepted because a concurrently ' +
        'mutable buffer cannot provide an exact scene-upload snapshot',
    );
  }
  const rawValueCount = src.length;
  if (typeof rawValueCount !== 'number') {
    throw texturePayloadError(
      handle,
      options,
      'no cpuMirror, raw data, or DataTexture-shaped image data was supplied',
    );
  }
  if (
    typeof rawWidth !== 'number' ||
    typeof rawHeight !== 'number' ||
    !Number.isSafeInteger(rawWidth) ||
    !Number.isSafeInteger(rawHeight) ||
    rawWidth <= 0 ||
    rawHeight <= 0
  ) {
    throw texturePayloadError(
      handle,
      options,
      `width and height must be positive safe integers (received ${String(rawWidth)}×${String(rawHeight)})`,
    );
  }
  const width = rawWidth;
  const height = rawHeight;
  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount) || !Number.isSafeInteger(pixelCount * 4)) {
    throw texturePayloadError(
      handle,
      options,
      `the ${width}×${height} decoded RGBA element count exceeds JavaScript's safe integer range`,
    );
  }
  const valueCount = rawValueCount;
  if (!Number.isSafeInteger(valueCount) || valueCount < 0) {
    throw texturePayloadError(
      handle,
      options,
      `data.length must be a finite non-negative safe integer (received ${String(valueCount)})`,
    );
  }

  // D10.12: resolve hint from __vitrum_hint__ property, or direct channels/dataType on handle.
  // Use type assertions at the object-literal level to satisfy exactOptionalPropertyTypes:
  // only include a property in the literal when the source value is non-null.
  const fallbackChannels = sourceChannels ?? (selectedHandlePayload ? undefined : h.channels);
  const fallbackDataType = sourceDataType ?? (selectedHandlePayload ? undefined : h.dataType);
  const fallbackColorSpace = selectedColorSpace ?? (selectedHandlePayload ? undefined : h.colorSpace);
  const outerHint = h.__vitrum_hint__;
  const outerHintChannels = outerHint?.channels;
  const outerHintDataType = outerHint?.dataType;
  const outerHintColorSpace = outerHint?.colorSpace;
  const hintedChannels = outerHintChannels ?? fallbackChannels;
  const hintedDataType = outerHintDataType ?? fallbackDataType;
  const hintedColorSpace = outerHintColorSpace ?? fallbackColorSpace;
  if (hintedChannels != null && !VALID_CHANNEL_COUNTS.has(hintedChannels)) {
    throw texturePayloadError(
      handle,
      options,
      `channels must be one of 1, 2, 3, or 4 (received ${String(hintedChannels)})`,
    );
  }
  if (hintedDataType != null && !VALID_DATA_TYPES.has(hintedDataType)) {
    throw texturePayloadError(
      handle,
      options,
      `dataType "${String(hintedDataType)}" is unsupported`,
    );
  }
  if (hintedColorSpace != null && hintedColorSpace !== 'srgb' && hintedColorSpace !== 'linear') {
    throw texturePayloadError(
      handle,
      options,
      `colorSpace "${String(hintedColorSpace)}" is unsupported`,
    );
  }

  let stride: number;
  if (hintedChannels != null) {
    stride = hintedChannels;
  } else if (valueCount === pixelCount) {
    stride = 1;
  } else if (valueCount === pixelCount * 4) {
    stride = 4;
  } else if (valueCount === pixelCount * 2 || valueCount === pixelCount * 3) {
    const inferred = valueCount / pixelCount;
    throw texturePayloadError(
      handle,
      options,
      `the ${inferred}-channel layout is ambiguous without __vitrum_hint__.channels`,
    );
  } else {
    throw texturePayloadError(
      handle,
      options,
      `data length ${valueCount} does not describe exact 1- or 4-channel pixels for ${width}×${height}`,
    );
  }
  const expectedValueCount = pixelCount * stride;
  if (valueCount !== expectedValueCount) {
    throw texturePayloadError(
      handle,
      options,
      `data length ${valueCount} does not equal width×height×channels (${expectedValueCount})`,
    );
  }

  const backingType = Object.prototype.toString.call(src);
  let dataType = hintedDataType as TextureHandleHint['dataType'] | undefined;
  if (dataType == null) {
    if (backingType === '[object Uint8Array]' || backingType === '[object Uint8ClampedArray]') {
      dataType = 'uint8';
    } else if (backingType === '[object Uint16Array]') {
      dataType = 'uint16';
    } else if (backingType === '[object Float32Array]') {
      dataType = 'float32';
    } else {
      throw texturePayloadError(
        handle,
        options,
        `pixel backing ${backingType} cannot be inferred; use Uint8Array, Uint16Array, or Float32Array`,
      );
    }
  }
  const compatibleBacking =
    (dataType === 'uint8' &&
      (backingType === '[object Uint8Array]' || backingType === '[object Uint8ClampedArray]')) ||
    ((dataType === 'uint16' || dataType === 'float16' || dataType === 'half-float') &&
      backingType === '[object Uint16Array]') ||
    (dataType === 'float32' && backingType === '[object Float32Array]') ||
    // Immutable public texture-source mirrors intentionally expose a frozen
    // getter-safe ArrayLike view instead of the mutable typed-array object.
    // An explicit dataType makes any generic ArrayLike unambiguous; every lane
    // is still range-checked and copied exactly once before decode.
    (hintedDataType != null && backingType === '[object Object]');
  if (!compatibleBacking) {
    const expected =
      dataType === 'uint8'
        ? 'Uint8Array or Uint8ClampedArray'
        : dataType === 'float32'
          ? 'Float32Array'
          : 'Uint16Array';
    throw texturePayloadError(
      handle,
      options,
      `dataType "${dataType}" requires ${expected}, received ${backingType}`,
    );
  }
  // A Float32 payload is already in a linear numeric domain unless the host
  // explicitly labels it sRGB. This matches the shared decoder and prevents an
  // unhinted HDR emissive payload from being clamped to [0, 1] and decoded a
  // second time merely because emissiveMap is a color-role map.
  const sourceColorSpace: TextureSampleColorSpace | undefined =
    hintedColorSpace === 'srgb'
      ? 'srgb'
      : hintedColorSpace === 'linear'
        ? 'linear'
        : dataType === 'float32'
          ? 'linear'
          : undefined;
  return Object.freeze({
    width,
    height,
    pixelCount,
    data: src,
    stride,
    dataType,
    ...(sourceColorSpace ? { sourceColorSpace } : {}),
  });
}

/** Validate one collected role entry against cached per-handle metadata. */
function validateCollectedTextureChannels(
  collected: CollectedTextureHandle,
  inspected: InspectedTexturePixels,
  options?: TextureAtlasBuildOptions,
): void {
  // Role/channel compatibility is metadata-only and must fail before staging
  // allocation or the first indexed read from a hostile ArrayLike payload.
  if ((collected.allowedChannelsMask & (1 << inspected.stride)) !== 0) return;

  const allowed: number[] = [];
  for (let channels = 1; channels <= 4; channels += 1) {
    if ((collected.allowedChannelsMask & (1 << channels)) !== 0) allowed.push(channels);
  }
  throw texturePayloadError(
    collected.handle,
    options,
    `${collected.roleLabels.join(' + ')} requires source channels ` +
      `${allowed.join(', ')} (received ${inspected.stride})`,
  );
}

function inspectCollectedHandlePixels(
  collected: CollectedTextureHandle,
  transaction: TextureAtlasSourceTransaction,
  options?: TextureAtlasBuildOptions,
): InspectedTexturePixels {
  let inspected = transaction.inspectedByHandle.get(collected.handle);
  if (inspected == null) {
    inspected = inspectHandlePixels(collected.handle, options);
    transaction.inspectedByHandle.set(collected.handle, inspected);
  }
  validateCollectedTextureChannels(collected, inspected, options);
  return inspected;
}

/**
 * Snapshot one user-controlled payload into the exact raw staging array later
 * decoded into level zero. `data.length` was captured by inspection; each
 * indexed element is read once here, validated, canonicalized, and never read
 * from the host object again.
 */
function stageInspectedPixels(
  inspected: InspectedTexturePixels,
  handle: unknown,
  budget: TextureAtlasCpuBudget,
  options?: TextureAtlasBuildOptions,
): InspectedTexturePixels {
  const valueCount = inspected.pixelCount * inspected.stride;
  const byteLength = BigInt(valueCount) * BigInt(textureSourceElementBytes(inspected.dataType));
  reserveTextureAtlasCpuBytes(budget, byteLength, 'material texture raw snapshot');
  let snapshot: Uint8Array | Uint16Array | Float32Array;
  try {
    snapshot = inspected.dataType === 'uint8'
      ? new Uint8Array(valueCount)
      : inspected.dataType === 'float32'
        ? new Float32Array(valueCount)
        : new Uint16Array(valueCount);
  } catch (error) {
    releaseTextureAtlasCpuBytes(budget, byteLength);
    throw error;
  }

  try {
    const integerMax = inspected.dataType === 'uint8'
      ? 255
      : inspected.dataType === 'float32'
        ? null
        : 65535;
    for (let index = 0; index < valueCount; index += 1) {
      const value = Number(inspected.data[index]);
      if (!Number.isFinite(value)) {
        throw texturePayloadError(
          handle,
          options,
          `decoded pixel data must be finite at the raw-input boundary ` +
            `(value ${index} is ${String(value)})`,
        );
      }
      if (
        integerMax != null &&
        (!Number.isInteger(value) || value < 0 || value > integerMax)
      ) {
        throw texturePayloadError(
          handle,
          options,
          `raw pixel value ${index} must be an integer in [0, ${integerMax}] ` +
            `for ${inspected.dataType}`,
        );
      }
      if (inspected.dataType === 'float32') {
        const stored = Math.fround(value);
        if (!Number.isFinite(stored)) {
          throw texturePayloadError(
            handle,
            options,
            `raw pixel value ${index} overflows float32 storage`,
          );
        }
        if (value !== 0 && stored === 0) {
          throw texturePayloadError(
            handle,
            options,
            `raw pixel value ${index} is nonzero but underflows to zero in float32`,
          );
        }
        snapshot[index] = stored;
      } else {
        snapshot[index] = value;
      }
    }
  } catch (error) {
    releaseTextureAtlasCpuBytes(budget, byteLength);
    throw error;
  }

  return Object.freeze({
    ...inspected,
    data: snapshot,
    stagingByteLength: Number(byteLength),
  });
}

function decodedChannel(
  inspected: InspectedTexturePixels,
  index: number,
  handle: unknown,
  options: TextureAtlasBuildOptions | undefined,
): number {
  const value = decodeTextureChannel(inspected.dataType, Number(inspected.data[index]));
  if (!Number.isFinite(value)) {
    throw texturePayloadError(
      handle,
      options,
      `decoded pixel data must be finite (value ${index} decoded to ${String(value)})`,
    );
  }
  return value;
}

/** Validate mutable source values before any decoded-layer allocation. */
function validateInspectedPixels(
  inspected: InspectedTexturePixels,
  handle: unknown,
  colorSpace: TextureSampleColorSpace,
  storageClass: TextureAtlasStorageClass,
  options?: TextureAtlasBuildOptions,
): void {
  const valueCount = inspected.pixelCount * inspected.stride;
  for (let index = 0; index < valueCount; index += 1) {
    const value = decodedChannel(inspected, index, handle, options);
    const channel = index % inspected.stride;
    const isRgbChannel = channel < Math.min(3, inspected.stride);
    const sourceIsSrgb =
      colorSpace === 'srgb' &&
      isRgbChannel &&
      inspected.sourceColorSpace !== 'linear';
    if ((storageClass === 'ldr' || sourceIsSrgb) && (value < 0 || value > 1)) {
      throw texturePayloadError(
        handle,
        options,
        `decoded ${storageClass === 'ldr' ? 'LDR' : 'sRGB'} value ${String(value)} ` +
          'is outside the normalized [0, 1] range',
      );
    }
    const linearValue = sourceIsSrgb ? srgbToLinear(value) : value;
    if (storageClass === 'hdr') {
      if (isRgbChannel && linearValue < 0) {
        throw texturePayloadError(
          handle,
          options,
          `decoded HDR outgoing-radiance RGB value ${String(linearValue)} ` +
            'must be non-negative',
        );
      }
      checkedHdrFloat16Bits(
        linearValue,
        handle,
        options,
        `decoded HDR value ${index}`,
      );
    }
  }
}

/** Decode directly into one atlas layer without retaining an RGBA source copy. */
function blitInspectedLayer(
  inspected: InspectedTexturePixels,
  handle: unknown,
  dim: number,
  data: TextureAtlasData,
  placement: TextureAtlasPlacement,
  colorSpace: TextureSampleColorSpace,
  storageClass: TextureAtlasStorageClass,
  options?: TextureAtlasBuildOptions,
): void {
  const layerBase = placement.layer * dim * dim * 4;
  const sourceIsSrgb = colorSpace === 'srgb' && inspected.sourceColorSpace !== 'linear';
  const writeChannel = (
    offset: number,
    value: number,
    rgbChannel: boolean,
  ): void => {
    if (storageClass === 'hdr') {
      const linearValue = rgbChannel && sourceIsSrgb ? srgbToLinear(value) : value;
      if (rgbChannel && linearValue < 0) {
        throw texturePayloadError(
          handle,
          options,
          `decoded HDR outgoing-radiance RGB value ${String(linearValue)} ` +
            'must be non-negative',
        );
      }
      (data as Uint16Array)[offset] = checkedHdrFloat16Bits(
        linearValue,
        handle,
        options,
        `decoded HDR atlas channel ${offset}`,
      );
      return;
    }
    if (value < 0 || value > 1) {
      throw texturePayloadError(
        handle,
        options,
        `decoded LDR value ${String(value)} is outside the normalized [0, 1] range`,
      );
    }
    const encoded =
      rgbChannel && colorSpace === 'srgb' && !sourceIsSrgb
        ? linearToSrgb(value)
        : value;
    (data as Uint8Array)[offset] = Math.round(encoded * 255);
  };
  for (let y = 0; y < inspected.height; y += 1) {
    for (let x = 0; x < inspected.width; x += 1) {
      const sourcePixel = y * inspected.width + x;
      const s = sourcePixel * inspected.stride;
      const d =
        layerBase +
        ((placement.y + y) * dim + placement.x + x) * 4;
      const r = decodedChannel(inspected, s, handle, options);
      const g = inspected.stride > 1 ? decodedChannel(inspected, s + 1, handle, options) : r;
      // Match native RG texture expansion: R stays R, G stays G, absent B is 0,
      // and absent A is 1. A one-channel source remains luminance and expands to
      // RGB; a two-channel source is data-bearing RG rather than luminance-alpha.
      const b =
        inspected.stride > 2
          ? decodedChannel(inspected, s + 2, handle, options)
          : inspected.stride === 1
            ? r
            : 0;
      const a = inspected.stride >= 4 ? decodedChannel(inspected, s + 3, handle, options) : 1;
      writeChannel(d, r, true);
      writeChannel(d + 1, g, true);
      writeChannel(d + 2, b, true);
      writeChannel(d + 3, a, false);
    }
  }
}

/**
 * Preflight every allocation in a format-native atlas mip chain. The returned
 * counts are the exact typed-array lengths used while packing; hostile dimensions or
 * layer counts fail deterministically instead of reaching RangeError/OOM.
 */
export function textureAtlasMipElementCounts(
  dim: number,
  layerCount: number,
  storageClass: TextureAtlasStorageClass = 'ldr',
): readonly number[] {
  checkedTextureAtlasShape(dim, layerCount);
  const bytesPerElement = storageClass === 'hdr' ? 2n : 1n;
  const maxBytes = BigInt(MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES);
  const counts: number[] = [];
  let totalElements = 0n;
  let levelDim = dim;
  let lod = 0;
  while (true) {
    const levelElements = BigInt(levelDim) * BigInt(levelDim) * 4n * BigInt(layerCount);
    const nextTotal = totalElements + levelElements;
    const nextBytes = nextTotal * bytesPerElement;
    if (levelElements * bytesPerElement > maxBytes || nextBytes > maxBytes) {
      throw new Error(
        `[pt-webgl2] material texture atlas ${dim}²×${layerCount} mip chain exceeds the ` +
          `512 MiB CPU staging budget at mip ${lod} (${nextBytes.toString()} bytes of ` +
          `${storageClass === 'hdr' ? 'RGBA16F' : 'RGBA8'} data). ` +
          'Reduce material texture resolution or unique texture count.',
      );
    }
    if (levelElements > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(
        '[pt-webgl2] material texture atlas mip element count exceeds JavaScript safe integer range',
      );
    }
    counts.push(Number(levelElements));
    totalElements = nextTotal;
    if (levelDim === 1) break;
    levelDim = Math.max(1, Math.floor(levelDim / 2));
    lod += 1;
  }
  return counts;
}

/**
 * Bound the complete JS-side atlas construction before its first retained array.
 * Source values decode directly into level zero and immutable GPU storage
 * reserves spare layers, so the retained mip chain is the only atlas staging
 * allocation.
 */
function preflightTextureAtlasCpuAllocations(
  dim: number,
  layerCount: number,
  storageClass: TextureAtlasStorageClass,
  options?: TextureAtlasBuildOptions,
): readonly number[] {
  const mipElementCounts = textureAtlasMipElementCounts(dim, layerCount, storageClass);
  const configuredMaxLayers = options?.maxArrayTextureLayers;
  if (
    configuredMaxLayers != null &&
    (!Number.isSafeInteger(configuredMaxLayers) || configuredMaxLayers <= 0)
  ) {
    throw new Error(
      `[pt-webgl2] MAX_ARRAY_TEXTURE_LAYERS must be a positive safe integer ` +
        `(received ${String(configuredMaxLayers)})`,
    );
  }
  if (configuredMaxLayers != null && layerCount > configuredMaxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas needs ${layerCount} layers but this device only supports ` +
        `${configuredMaxLayers} — reduce the number of unique material textures in the scene.`,
    );
  }
  return mipElementCounts;
}

interface AtlasPlacementSource {
  readonly index: number;
  readonly width: number;
  readonly height: number;
  readonly tileSize: number;
}

interface AtlasPackingLayer {
  readonly layer: number;
  readonly freeBySize: Map<number, { readonly x: number; readonly y: number }[]>;
}

function nextPowerOfTwo(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `[pt-webgl2] material texture tile extent must be a positive safe integer ` +
        `(received ${String(value)}).`,
    );
  }
  let result = 1;
  while (result < value) {
    if (result > Number.MAX_SAFE_INTEGER / 2) {
      throw new RangeError(
        `[pt-webgl2] material texture tile extent ${value} has no safe power-of-two container.`,
      );
    }
    result *= 2;
  }
  return result;
}

function previousPowerOfTwo(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `[pt-webgl2] material texture atlas extent must be a positive safe integer ` +
        `(received ${String(value)}).`,
    );
  }
  const result = 2 ** Math.floor(Math.log2(value));
  if (!Number.isSafeInteger(result) || result <= 0) {
    throw new RangeError(
      `[pt-webgl2] material texture atlas extent ${value} has no safe power-of-two floor.`,
    );
  }
  return result;
}

function allocateAtlasTile(
  packingLayer: AtlasPackingLayer,
  tileSize: number,
  packingDim: number,
): TextureAtlasPlacement | null {
  let availableSize = tileSize;
  while (
    availableSize <= packingDim &&
    (packingLayer.freeBySize.get(availableSize)?.length ?? 0) === 0
  ) {
    availableSize *= 2;
  }
  if (availableSize > packingDim) return null;

  const freeAtSize = packingLayer.freeBySize.get(availableSize);
  const block = freeAtSize?.shift();
  if (block == null) return null;
  const x = block.x;
  const y = block.y;
  while (availableSize > tileSize) {
    availableSize /= 2;
    let free = packingLayer.freeBySize.get(availableSize);
    if (free == null) {
      free = [];
      packingLayer.freeBySize.set(availableSize, free);
    }
    // Retain the other three quadrants in deterministic row-major order and
    // continue splitting the top-left quadrant.
    free.push(
      { x: x + availableSize, y },
      { x, y: y + availableSize },
      { x: x + availableSize, y: y + availableSize },
    );
  }
  return { layer: packingLayer.layer, x, y, width: 0, height: 0 };
}

/**
 * Pack source rectangles into power-of-two-aligned square tiles. Alignment
 * guarantees that each source keeps a disjoint rectangle at every mip level.
 * A largest NPOT source gets one layer, while unrelated smaller sources are
 * buddy-packed into separate shared layers instead of inheriting its extent
 * once per map.
 */
function packAtlasSourcePlacements(
  sourceDimensions: readonly (readonly [number, number])[],
  dim: number,
): readonly TextureAtlasPlacement[] {
  const packingDim = previousPowerOfTwo(dim);
  const sources: AtlasPlacementSource[] = sourceDimensions.map(([width, height], index) => ({
    index,
    width,
    height,
    tileSize: nextPowerOfTwo(Math.max(width, height)),
  }));
  sources.sort((a, b) => b.tileSize - a.tileSize || a.index - b.index);

  const placements = new Array<TextureAtlasPlacement>(sources.length);
  const sharedLayers: AtlasPackingLayer[] = [];
  let layerCount = 0;
  for (const source of sources) {
    let placement: TextureAtlasPlacement | null = null;
    if (source.tileSize <= packingDim) {
      for (const packingLayer of sharedLayers) {
        placement = allocateAtlasTile(packingLayer, source.tileSize, packingDim);
        if (placement != null) break;
      }
      if (placement == null) {
        const packingLayer: AtlasPackingLayer = {
          layer: layerCount++,
          freeBySize: new Map([
            [packingDim, [{ x: 0, y: 0 }]],
          ]),
        };
        sharedLayers.push(packingLayer);
        placement = allocateAtlasTile(packingLayer, source.tileSize, packingDim);
      }
    } else {
      placement = {
        layer: layerCount++,
        x: 0,
        y: 0,
        width: 0,
        height: 0,
      };
    }
    if (placement == null) {
      throw new Error('[pt-webgl2] internal material texture atlas placement failure');
    }
    placements[source.index] = {
      ...placement,
      width: source.width,
      height: source.height,
    };
  }
  return placements;
}

function readAtlasMipChannel(
  data: TextureAtlasData,
  index: number,
  storageClass: TextureAtlasStorageClass,
  colorSpace: TextureSampleColorSpace,
  channel: number,
): number {
  if (storageClass === 'hdr') {
    return float16BitsToFloat32((data as Uint16Array)[index] ?? 0);
  }
  const value = ((data as Uint8Array)[index] ?? 0) / 255;
  return colorSpace === 'srgb' && channel < 3 ? srgbToLinear(value) : value;
}

function writeAtlasMipChannel(
  data: TextureAtlasData,
  index: number,
  value: number,
  storageClass: TextureAtlasStorageClass,
  colorSpace: TextureSampleColorSpace,
  channel: number,
  checkedHdrContext?: string,
): void {
  if (storageClass === 'hdr') {
    (data as Uint16Array)[index] =
      checkedHdrContext == null
        ? float32ToFloat16Bits(value)
        : finiteFloat16Bits(value, checkedHdrContext);
    return;
  }
  const encoded =
    colorSpace === 'srgb' && channel < 3 ? linearToSrgb(value) : value;
  (data as Uint8Array)[index] = Math.round(Math.max(0, Math.min(1, encoded)) * 255);
}

function buildAtlasMipLevels(
  data: TextureAtlasData,
  dim: number,
  layerCount: number,
  mipElementCounts: readonly number[],
  sourceDimensions: readonly (readonly [number, number])[],
  sourcePlacements: readonly TextureAtlasPlacement[],
  sourceColorSpaces: readonly TextureSampleColorSpace[],
  sourceUsesMipFiltering: readonly boolean[],
  storageClass: TextureAtlasStorageClass,
): readonly TextureAtlasMipLevel[] {
  const levels: TextureAtlasMipLevel[] = [{ data, dim }];
  let src = data;
  let srcDim = dim;
  let lod = 1;
  while (srcDim > 1) {
    const dstDim = Math.max(1, Math.floor(srcDim / 2));
    const dstElementCount = mipElementCounts[lod];
    if (dstElementCount == null) {
      throw new Error('[pt-webgl2] internal material texture atlas mip allocation plan mismatch');
    }
    const dst: TextureAtlasData =
      storageClass === 'hdr'
        ? new Uint16Array(dstElementCount)
        : new Uint8Array(dstElementCount);
    for (let sourceIndex = 0; sourceIndex < sourceDimensions.length; sourceIndex += 1) {
      const sourceBase = sourceDimensions[sourceIndex];
      const placement = sourcePlacements[sourceIndex];
      if (sourceBase == null) {
        throw new Error('[pt-webgl2] internal material texture source-dimension mismatch');
      }
      if (placement == null) {
        throw new Error('[pt-webgl2] internal material texture source-placement mismatch');
      }
      const colorSpace = sourceColorSpaces[sourceIndex];
      if (colorSpace == null) {
        throw new Error('[pt-webgl2] internal material texture source-color-space mismatch');
      }
      const usesMipFiltering = sourceUsesMipFiltering[sourceIndex] === true;
      const maxSourceLevel = Math.floor(
        Math.log2(Math.max(sourceBase[0], sourceBase[1])),
      );
      if (lod > maxSourceLevel) continue;
      const srcLayerBase = placement.layer * srcDim * srcDim * 4;
      const dstLayerBase = placement.layer * dstDim * dstDim * 4;
      const srcOriginX = Math.floor(placement.x / 2 ** (lod - 1));
      const srcOriginY = Math.floor(placement.y / 2 ** (lod - 1));
      const dstOriginX = Math.floor(placement.x / 2 ** lod);
      const dstOriginY = Math.floor(placement.y / 2 ** lod);
      const sourceWidth = Math.max(1, Math.floor(sourceBase[0] / 2 ** (lod - 1)));
      const sourceHeight = Math.max(1, Math.floor(sourceBase[1] / 2 ** (lod - 1)));
      const targetWidth = Math.max(1, Math.floor(sourceBase[0] / 2 ** lod));
      const targetHeight = Math.max(1, Math.floor(sourceBase[1] / 2 ** lod));
      for (let y = 0; y < targetHeight; y += 1) {
        for (let x = 0; x < targetWidth; x += 1) {
          const dstOffset =
            dstLayerBase +
            ((dstOriginY + y) * dstDim + dstOriginX + x) * 4;
          const sx0 = Math.floor((x * sourceWidth) / targetWidth);
          const sx1 = Math.max(sx0 + 1, Math.floor(((x + 1) * sourceWidth) / targetWidth));
          const sy0 = Math.floor((y * sourceHeight) / targetHeight);
          const sy1 = Math.max(sy0 + 1, Math.floor(((y + 1) * sourceHeight) / targetHeight));
          const sums: [number, number, number, number] = [0, 0, 0, 0];
          let count = 0;
          for (let sy = sy0; sy < sy1; sy += 1) {
            for (let sx = sx0; sx < sx1; sx += 1) {
              const srcOffset =
                srcLayerBase +
                ((srcOriginY + Math.min(sourceHeight - 1, sy)) * srcDim +
                  srcOriginX +
                  Math.min(sourceWidth - 1, sx)) *
                  4;
              sums[0] += readAtlasMipChannel(src, srcOffset, storageClass, colorSpace, 0);
              sums[1] += readAtlasMipChannel(src, srcOffset + 1, storageClass, colorSpace, 1);
              sums[2] += readAtlasMipChannel(src, srcOffset + 2, storageClass, colorSpace, 2);
              sums[3] += readAtlasMipChannel(src, srcOffset + 3, storageClass, colorSpace, 3);
              count += 1;
            }
          }
          for (let c = 0; c < 4; c += 1) {
            writeAtlasMipChannel(
              dst,
              dstOffset + c,
              sums[c]! / Math.max(1, count),
              storageClass,
              colorSpace,
              c,
              storageClass === 'hdr' && usesMipFiltering
                ? `generated HDR mip ${lod} source ${sourceIndex} ` +
                  `texel (${x}, ${y}) channel ${c}`
                : undefined,
            );
          }
        }
      }
    }
    levels.push({ data: dst, dim: dstDim });
    src = dst;
    srcDim = dstDim;
    lod += 1;
  }
  return levels;
}

interface TextureAtlasPlan {
  readonly storageClass: TextureAtlasStorageClass;
  readonly inspected: readonly {
    readonly handle: unknown;
    readonly colorSpace: TextureSampleColorSpace;
    readonly pixels: InspectedTexturePixels;
    readonly usesMipFiltering: boolean;
  }[];
  readonly dim: number;
  readonly sourceDimensions: readonly (readonly [number, number])[];
  readonly sourcePlacements: readonly TextureAtlasPlacement[];
  readonly layerCount: number;
  readonly mipElementCounts: readonly number[];
}

/**
 * Inspect handles and compute exact atlas geometry without scanning pixel values
 * or allocating any retained atlas array. The paired production path plans both
 * storage classes before applying their shared CPU byte ceiling.
 */
function planTextureAtlas(
  materials: readonly MaterialSpec[],
  transaction: TextureAtlasSourceTransaction,
  options?: TextureAtlasBuildOptions,
): TextureAtlasPlan | null {
  const storageClass = options?.storageClass ?? 'ldr';
  // unique (handle, color-space role) pairs in first-seen order
  const handles: CollectedTextureHandle[] = [];
  const seen = new Map<unknown, Map<TextureSampleColorSpace, number>>();
  for (const m of materials) {
    for (const key of SAMPLED_MAP_KEYS) {
      if (textureStorageClassForMapKey(key) !== storageClass) continue;
      const ref = m[key] as { readonly handle?: unknown } | undefined;
      collectHandle(
        handles,
        seen,
        ref,
        textureColorSpaceForMapKey(key),
        textureChannelRoleForMapKey(key),
      );
    }
    if (storageClass === 'ldr') {
      collectHandle(
        handles,
        seen,
        m.frontLayer?.normalMap,
        'linear',
        LAYER_NORMAL_CHANNEL_ROLE,
      );
      collectHandle(
        handles,
        seen,
        m.backLayer?.normalMap,
        'linear',
        LAYER_NORMAL_CHANNEL_ROLE,
      );
    }
  }
  if (handles.length === 0) return null;

  // Inspect every layer without allocating decoded RGBA. Only the retained mip
  // chain is staged on the CPU; source decode writes directly into level zero.
  const inspected: {
    handle: unknown;
    colorSpace: TextureSampleColorSpace;
    pixels: InspectedTexturePixels;
    usesMipFiltering: boolean;
  }[] = [];
  let dim = 0;
  for (const collected of handles) {
    const { handle, colorSpace, usesMipFiltering } = collected;
    const source = inspectCollectedHandlePixels(collected, transaction, options);
    inspected.push({ handle, colorSpace, pixels: source, usesMipFiltering });
    dim = Math.max(dim, source.width, source.height);
  }

  const sourceDimensions = inspected.map(
    ({ pixels: source }) => [source.width, source.height] as const,
  );
  const sourcePlacements = packAtlasSourcePlacements(sourceDimensions, dim);
  const layerCount =
    sourcePlacements.reduce((max, placement) => Math.max(max, placement.layer), -1) + 1;
  const mipElementCounts = preflightTextureAtlasCpuAllocations(
    dim,
    layerCount,
    storageClass,
    options,
  );

  return {
    storageClass,
    inspected,
    dim,
    sourceDimensions,
    sourcePlacements,
    layerCount,
    mipElementCounts,
  };
}

function stageTextureAtlasPlan(
  plan: TextureAtlasPlan,
  budget: TextureAtlasCpuBudget,
  transaction: TextureAtlasSourceTransaction,
  options?: TextureAtlasBuildOptions,
): TextureAtlasPlan {
  const startingBytes = budget.usedBytes;
  const addedHandles: unknown[] = [];
  try {
    return {
      ...plan,
      inspected: plan.inspected.map((entry) => {
        let pixels = transaction.stagedByHandle.get(entry.handle);
        if (pixels == null) {
          pixels = stageInspectedPixels(entry.pixels, entry.handle, budget, options);
          transaction.stagedByHandle.set(entry.handle, pixels);
          addedHandles.push(entry.handle);
        }
        return { ...entry, pixels };
      }),
    };
  } catch (error) {
    // Every successfully-created snapshot in this plan becomes unreachable on
    // failure. Roll the aggregate reservation back as one transaction so a
    // caller that catches and continues never inherits phantom staging bytes.
    for (const handle of addedHandles) transaction.stagedByHandle.delete(handle);
    budget.usedBytes = startingBytes;
    throw error;
  }
}

function validateTextureAtlasPlan(
  plan: TextureAtlasPlan,
  options?: TextureAtlasBuildOptions,
): void {
  // Validate the immutable raw snapshots before allocating level zero.
  const { inspected, storageClass } = plan;
  for (const { handle, colorSpace, pixels: source } of inspected) {
    if (source.stagingByteLength == null) {
      throw new Error('[pt-webgl2] internal material texture plan was not staged.');
    }
    validateInspectedPixels(source, handle, colorSpace, storageClass, options);
  }
}

function materializeTextureAtlasPlan(
  plan: TextureAtlasPlan,
  materials: readonly MaterialSpec[],
  options?: TextureAtlasBuildOptions,
): TextureAtlas {
  const {
    storageClass,
    inspected,
    dim,
    sourceDimensions,
    sourcePlacements,
    layerCount,
    mipElementCounts,
  } = plan;
  const data: TextureAtlasData =
    storageClass === 'hdr'
      ? new Uint16Array(mipElementCounts[0]!)
      : new Uint8Array(mipElementCounts[0]!);
  const layerOfByColorSpace: TextureAtlasLayerMap = {
    srgb: new Map<unknown, number>(),
    linear: new Map<unknown, number>(),
    dimensions: new Map<unknown, readonly [number, number]>(),
    placements: {
      srgb: new Map<unknown, TextureAtlasPlacement>(),
      linear: new Map<unknown, TextureAtlasPlacement>(),
    },
  };
  inspected.forEach(({ handle, colorSpace, pixels: source }, sourceIndex) => {
    const placement = sourcePlacements[sourceIndex];
    if (placement == null) {
      throw new Error('[pt-webgl2] internal material texture source-placement mismatch');
    }
    (layerOfByColorSpace[colorSpace] as Map<unknown, number>).set(handle, placement.layer);
    (layerOfByColorSpace.dimensions as Map<unknown, readonly [number, number]>).set(handle, [
      source.width,
      source.height,
    ]);
    (
      layerOfByColorSpace.placements?.[colorSpace] as
        | Map<unknown, TextureAtlasPlacement>
        | undefined
    )?.set(handle, placement);
    blitInspectedLayer(
      source,
      handle,
      dim,
      data,
      placement,
      colorSpace,
      storageClass,
      options,
    );
  });
  const atlas: TextureAtlas = {
    data,
    storageClass,
    format: storageClass === 'hdr' ? 'rgba16f' : 'rgba8unorm',
    dim,
    mipLevels: buildAtlasMipLevels(
      data,
      dim,
      layerCount,
      mipElementCounts,
      sourceDimensions,
      sourcePlacements,
      inspected.map(({ colorSpace }) => colorSpace),
      inspected.map(({ usesMipFiltering }) => usesMipFiltering),
      storageClass,
    ),
    layerCount,
    layerOfByColorSpace,
    sourceDimensions,
    sourcePlacements,
  };
  validateMaterialRadianceEnvelope(atlas, materials);
  return atlas;
}

type RadianceTextureRef = NonNullable<MaterialSpec['emissiveMap']>;

/**
 * Validate the exact binary16 texels that WebGL samples against the exact
 * binary32 material factor that the materials texture publishes. This runs
 * after CPU atlas materialization but before any WebGL allocation or upload.
 */
function validateRadianceMapUse(
  atlas: TextureAtlas,
  ref: RadianceTextureRef,
  colorSpace: TextureSampleColorSpace,
  factor: RadianceRgb,
  label: string,
): void {
  const handle = ref.handle;
  if (handle == null) return;
  const placement = atlas.layerOfByColorSpace.placements?.[colorSpace].get(handle);
  if (placement == null) {
    throw new Error(
      `[pt-webgl2] internal HDR material texture placement missing for ${label}.`,
    );
  }
  const maxReachableLod =
    ref.mipFilter !== 'none'
      ? Math.floor(Math.log2(Math.max(placement.width, placement.height)))
      : 0;
  for (let lod = 0; lod <= maxReachableLod; lod += 1) {
    const level = atlas.mipLevels[lod];
    if (level == null || !(level.data instanceof Uint16Array)) {
      throw new Error(
        `[pt-webgl2] internal HDR material texture mip ${lod} missing for ${label}.`,
      );
    }
    const originX = Math.floor(placement.x / 2 ** lod);
    const originY = Math.floor(placement.y / 2 ** lod);
    const width = Math.max(1, Math.floor(placement.width / 2 ** lod));
    const height = Math.max(1, Math.floor(placement.height / 2 ** lod));
    const layerBase = placement.layer * level.dim * level.dim * 4;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const offset =
          layerBase + ((originY + y) * level.dim + originX + x) * 4;
        const texel: RadianceRgb = [
          float16BitsToFloat32(level.data[offset] ?? 0),
          float16BitsToFloat32(level.data[offset + 1] ?? 0),
          float16BitsToFloat32(level.data[offset + 2] ?? 0),
        ];
        packRadianceRgbProductF32(
          factor,
          texel,
          `${label} mip ${lod} texel (${x}, ${y}) radiance`,
        );
      }
    }
  }
}

function validateMaterialRadianceEnvelope(
  atlas: TextureAtlas,
  materials: readonly MaterialSpec[],
): void {
  if (atlas.storageClass !== 'hdr') return;
  materials.forEach((material, materialIndex) => {
    if (material.emissiveMap != null) {
      const emissiveFactor = packRadianceRgbScaleF32(
        material.emissive ?? [0, 0, 0],
        resolveEmissiveIntensity(material.emissiveIntensity),
        `pt-webgl2 material ${materialIndex} emissiveMap factor`,
      ).scaled;
      validateRadianceMapUse(
        atlas,
        material.emissiveMap,
        'srgb',
        emissiveFactor,
        `pt-webgl2 material ${materialIndex}.emissiveMap`,
      );
    }
    if (material.lightMap != null) {
      const lightMapIntensity = packNonNegativeRadianceScalarF32(
        material.lightMapIntensity ?? 1,
        `pt-webgl2 material ${materialIndex}.lightMapIntensity`,
      );
      validateRadianceMapUse(
        atlas,
        material.lightMap,
        'linear',
        [lightMapIntensity, lightMapIntensity, lightMapIntensity],
        `pt-webgl2 material ${materialIndex}.lightMap`,
      );
    }
  });
}

/**
 * Build one material-map atlas. This remains the focused single-atlas seam for
 * tests and tooling. Production scene packing uses `packMaterialTextureAtlases`
 * so the LDR/HDR pair is shape-preflighted against one shared byte ceiling before
 * either atlas is materialized.
 */
export function packTextureAtlas(
  materials: readonly MaterialSpec[],
  options?: TextureAtlasBuildOptions,
): TextureAtlas | null {
  const materialSnapshot = snapshotMaterialTextureInputs(materials);
  const sourceTransaction = createTextureAtlasSourceTransaction();
  const planned = planTextureAtlas(materialSnapshot, sourceTransaction, options);
  if (planned == null) return null;
  const atlasBytes = textureAtlasStorageByteLengthBigInt(
    planned.dim,
    planned.layerCount,
    planned.storageClass,
  );
  const budget: TextureAtlasCpuBudget = {
    usedBytes: atlasBytes,
    limitBytes: BigInt(MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES),
  };
  const plan = stageTextureAtlasPlan(planned, budget, sourceTransaction, options);
  validateTextureAtlasPlan(plan, options);
  return materializeTextureAtlasPlan(plan, materialSnapshot, options);
}

export interface MaterialTextureAtlases {
  readonly ldr: TextureAtlas | null;
  readonly hdr: TextureAtlas | null;
}

export interface MaterialTextureAtlasPlanSummary {
  readonly ldr: Pick<TextureAtlas, 'dim' | 'layerCount'> | null;
  readonly hdr: Pick<TextureAtlas, 'dim' | 'layerCount'> | null;
}

interface PreparedMaterialTextureAtlases {
  readonly materials: readonly MaterialSpec[];
  readonly ldrPlan: TextureAtlasPlan | null;
  readonly hdrPlan: TextureAtlasPlan | null;
  readonly ldrOptions: TextureAtlasBuildOptions;
  readonly hdrOptions: TextureAtlasBuildOptions;
}

/**
 * Plan both material arrays first, then enforce the aggregate retained CPU mip
 * budget before scanning values or allocating typed arrays. This prevents two
 * individually legal near-512 MiB chains from creating a near-1 GiB transient.
 */
function prepareMaterialTextureAtlases(
  materials: readonly MaterialSpec[],
  options?: Omit<TextureAtlasBuildOptions, 'storageClass'>,
): PreparedMaterialTextureAtlases {
  const materialSnapshot = snapshotMaterialTextureInputs(materials);
  const ldrOptions: TextureAtlasBuildOptions = {
    ...options,
    storageClass: 'ldr',
  };
  const hdrOptions: TextureAtlasBuildOptions = {
    ...options,
    storageClass: 'hdr',
  };
  const sourceTransaction = createTextureAtlasSourceTransaction();
  const plannedLdr = planTextureAtlas(materialSnapshot, sourceTransaction, ldrOptions);
  const plannedHdr = planTextureAtlas(materialSnapshot, sourceTransaction, hdrOptions);
  const ldrBytes =
    plannedLdr == null
      ? 0n
      : textureAtlasStorageByteLengthBigInt(
          plannedLdr.dim,
          plannedLdr.layerCount,
          'ldr',
        );
  const hdrBytes =
    plannedHdr == null
      ? 0n
      : textureAtlasStorageByteLengthBigInt(
          plannedHdr.dim,
          plannedHdr.layerCount,
          'hdr',
        );
  const combinedBytes = ldrBytes + hdrBytes;
  if (combinedBytes > BigInt(MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES)) {
    throw new Error(
      `[pt-webgl2] combined material texture atlas CPU mip chains require ` +
        `${combinedBytes.toString()} bytes (${ldrBytes.toString()} RGBA8 + ` +
        `${hdrBytes.toString()} RGBA16F), exceeding the shared ` +
      `${MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES}-byte staging budget before allocation.`,
    );
  }

  // Reserve the complete retained mip pair before adding immutable raw
  // snapshots. The resulting aggregate is the actual peak transaction memory:
  // source snapshots remain live while level zero and its mip chain are built.
  const budget: TextureAtlasCpuBudget = {
    usedBytes: combinedBytes,
    limitBytes: BigInt(MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES),
  };
  const ldrPlan = plannedLdr == null
    ? null
    : stageTextureAtlasPlan(plannedLdr, budget, sourceTransaction, ldrOptions);
  const hdrPlan = plannedHdr == null
    ? null
    : stageTextureAtlasPlan(plannedHdr, budget, sourceTransaction, hdrOptions);
  if (ldrPlan != null) validateTextureAtlasPlan(ldrPlan, ldrOptions);
  if (hdrPlan != null) validateTextureAtlasPlan(hdrPlan, hdrOptions);
  return {
    materials: materialSnapshot,
    ldrPlan,
    hdrPlan,
    ldrOptions,
    hdrOptions,
  };
}

/**
 * Validate a paired atlas candidate without allocating its retained mip arrays.
 * Mutation fallbacks use this before returning to the full setScene transaction,
 * avoiding a throwaway materialization immediately before the real one.
 */
export function preflightMaterialTextureAtlases(
  materials: readonly MaterialSpec[],
  options?: Omit<TextureAtlasBuildOptions, 'storageClass'>,
): MaterialTextureAtlasPlanSummary {
  const prepared = prepareMaterialTextureAtlases(materials, options);
  const summarize = (
    plan: TextureAtlasPlan | null,
  ): Pick<TextureAtlas, 'dim' | 'layerCount'> | null =>
    plan == null ? null : { dim: plan.dim, layerCount: plan.layerCount };
  return {
    ldr: summarize(prepared.ldrPlan),
    hdr: summarize(prepared.hdrPlan),
  };
}

/** Validate and materialize a production LDR/HDR atlas pair under one budget. */
export function packMaterialTextureAtlases(
  materials: readonly MaterialSpec[],
  options?: Omit<TextureAtlasBuildOptions, 'storageClass'>,
): MaterialTextureAtlases {
  const prepared = prepareMaterialTextureAtlases(materials, options);
  return {
    ldr:
      prepared.ldrPlan == null
        ? null
        : materializeTextureAtlasPlan(
            prepared.ldrPlan,
            prepared.materials,
            prepared.ldrOptions,
          ),
    hdr:
      prepared.hdrPlan == null
        ? null
        : materializeTextureAtlasPlan(
            prepared.hdrPlan,
            prepared.materials,
            prepared.hdrOptions,
          ),
  };
}

function checkedTextureAtlasLayerCapacity(
  gl: WebGL2RenderingContext,
  atlas: TextureAtlas,
  opts?: { readonly layerCapacity?: number },
): number {
  // Size guards: exceed MAX_TEXTURE_SIZE or MAX_ARRAY_TEXTURE_LAYERS and the
  // GPU allocation silently fails on many drivers — throw an actionable error first.
  if (gl.isContextLost()) {
    throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');
  }
  const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (atlas.dim > maxSize) {
    throw new Error(
      `pt-webgl2: material texture atlas needs a ${atlas.dim}² layer but this device only supports ` +
        `${maxSize}² — reduce the resolution of material textures in the scene.`,
    );
  }
  const maxLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  if (atlas.layerCount > maxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas needs ${atlas.layerCount} layers but this device only supports ` +
        `${maxLayers} — reduce the number of unique material textures in the scene.`,
    );
  }
  const layerCapacity =
    opts?.layerCapacity ??
    textureAtlasLayerCapacityForStorage(
      atlas.dim,
      atlas.layerCount,
      maxLayers,
      atlas.storageClass,
    );
  if (layerCapacity < atlas.layerCount || layerCapacity > maxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas allocation requested ${layerCapacity} layers for ` +
        `${atlas.layerCount} live layers on a device with ${maxLayers} maximum layers.`,
    );
  }
  const allocationBytes = textureAtlasStorageByteLengthBigInt(
    atlas.dim,
    layerCapacity,
    atlas.storageClass,
  );
  if (allocationBytes > BigInt(MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES)) {
    throw new Error(
      `pt-webgl2: material texture atlas allocation requests ${allocationBytes.toString()} bytes ` +
        `for ${layerCapacity} ${atlas.format} layers, exceeding the ` +
        `${MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES}-byte storage budget.`,
    );
  }
  return layerCapacity;
}

function configureTextureAtlasParameters(gl: WebGL2RenderingContext): void {
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function uploadTextureAtlasStorage(
  gl: WebGL2RenderingContext,
  atlas: TextureAtlas,
  layerCapacity: number,
): void {
  const internalFormat = atlas.format === 'rgba16f' ? gl.RGBA16F : gl.RGBA8;
  const uploadType = atlas.format === 'rgba16f' ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
  // Allocate the exact live layer count. Membership mutations are handled by a
  // candidate-first full scene transaction, never by in-place array growth.
  gl.texStorage3D(
    gl.TEXTURE_2D_ARRAY,
    atlas.mipLevels.length,
    internalFormat,
    atlas.dim,
    atlas.dim,
    layerCapacity,
  );
  atlas.mipLevels.forEach((level, lod) => {
    gl.texSubImage3D(
      gl.TEXTURE_2D_ARRAY,
      lod,
      0,
      0,
      0,
      level.dim,
      level.dim,
      atlas.layerCount,
      gl.RGBA,
      uploadType,
      level.data,
    );
  });
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_BASE_LEVEL, 0);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAX_LEVEL, atlas.mipLevels.length - 1);
}

/** Upload an RGBA8 or RGBA16F TEXTURE_2D_ARRAY (manual shader filtering). */
export function uploadTextureAtlas(
  gl: WebGL2RenderingContext,
  atlas: TextureAtlas,
  opts?: { readonly layerCapacity?: number },
): WebGLTexture {
  const layerCapacity = checkedTextureAtlasLayerCapacity(gl, atlas, opts);
  const tex = gl.createTexture();
  if (tex == null)
    throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');
  try {
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    configureTextureAtlasParameters(gl);
    uploadTextureAtlasStorage(gl, atlas, layerCapacity);
    if (typeof gl.getError === 'function' && typeof gl.NO_ERROR === 'number') {
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        throw new Error(
          `pt-webgl2: failed to upload material texture atlas (WebGL error 0x${error.toString(16)})`,
        );
      }
    }
    return tex;
  } catch (error) {
    gl.deleteTexture(tex);
    throw error;
  } finally {
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  }
}
