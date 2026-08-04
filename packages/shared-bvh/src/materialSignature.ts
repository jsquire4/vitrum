/**
 * materialSignature.ts — structural material-dedup signature extracted from
 * worldSpaceMerge.ts (D12-9, pure move). Mirrors `snapshotPreBuildMaterials`'s
 * value-dedup signature. Re-exported from worldSpaceMerge.ts for
 * backward-compatibility (the package entrypoint continues to export
 * `materialSig` + `HandleIdRegistry` from worldSpaceMerge.js).
 */

import type { MaterialSpec, TextureRef } from '@vitrum/core';

// ──────────────────────────────────────────────────────────────────────────
// Material dedup — mirrors snapshotPreBuildMaterials' value-dedup signature
// ──────────────────────────────────────────────────────────────────────────

type TextureMapField = Extract<{
  [K in keyof MaterialSpec]: MaterialSpec[K] extends TextureRef | undefined ? K : never;
}[keyof MaterialSpec], string>;

const TEXTURE_MAP_FIELDS: readonly TextureMapField[] = [
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'displacementMap',
  'lightMap',
];

const FLOAT32_TOKEN_VIEW = new DataView(new ArrayBuffer(4));

/**
 * Canonical token for the exact float32 value a GPU payload would receive.
 *
 * Finite values are rounded once with `Math.fround` and encoded as their
 * big-endian IEEE-754 bits. This preserves the otherwise-string-colliding
 * `+0`/`-0` pair and every adjacent representable float32 value. All NaN
 * payloads share one deterministic token because JavaScript arithmetic does
 * not preserve a portable NaN payload; signed infinities remain distinct.
 * A finite number that overflows float32 intentionally receives the matching
 * signed-infinity token.
 */
function float32Token(value: number): string {
  const rounded = Math.fround(value);
  if (Number.isNaN(rounded)) return 'f32:nan';
  if (rounded === Number.POSITIVE_INFINITY) return 'f32:+inf';
  if (rounded === Number.NEGATIVE_INFINITY) return 'f32:-inf';
  FLOAT32_TOKEN_VIEW.setFloat32(0, rounded, false);
  return `f32:${FLOAT32_TOKEN_VIEW.getUint32(0, false).toString(16).padStart(8, '0')}`;
}

/**
 * Apply a field default only when the value is omitted. Explicit NaN and
 * infinities are signed rather than silently collapsing to the default.
 */
function numberSig(value: number | undefined, fallback: number): string {
  return float32Token(value === undefined ? fallback : value);
}

function vecSig(
  value: readonly number[] | undefined,
  fallback: readonly number[],
  count: 2 | 3,
): string {
  const parts: string[] = [];
  for (let i = 0; i < count; i += 1) {
    parts.push(numberSig(value?.[i], fallback[i] ?? 0));
  }
  return parts.join(',');
}

function textureRefLike(value: unknown): TextureRef | undefined {
  if (value == null || typeof value !== 'object') return undefined;
  if ('handle' in value) return value;
  return { handle: value };
}

function textureTexCoordSig(texCoord: number | undefined): string {
  const uv = texCoord ?? 0;
  if (uv === 0 || uv === 1) return `uv${uv}`;
  return `uvUnsupported=${numberSig(uv, 0)}`;
}

function textureRefSig(value: unknown): string {
  const ref = textureRefLike(value);
  if (ref?.handle == null) return '';
  const transform = ref.transform;
  return [
    handleId(ref.handle),
    textureTexCoordSig(ref.texCoord),
    `off=${vecSig(transform?.offset, [0, 0], 2)}`,
    `scale=${vecSig(transform?.scale, [1, 1], 2)}`,
    `rot=${numberSig(transform?.rotation, 0)}`,
    `wrap=${ref.wrapS ?? 'repeat'},${ref.wrapT ?? 'repeat'}`,
    // TextureRef omission has the library-wide linear mip default. Keep the
    // structural signature semantic: omission and an explicit `linear` request
    // must deduplicate to the same material record.
    `filter=${ref.magFilter ?? ''},${ref.minFilter ?? ''},${ref.mipFilter ?? 'linear'}`,
  ].join(';');
}

function textureMapSig(m: MaterialSpec): string {
  return TEXTURE_MAP_FIELDS
    .map((field) => `${field}=${textureRefSig(m[field])}`)
    .join('|');
}

function surfaceLayerSig(
  layer: MaterialSpec['frontLayer'],
): string {
  if (layer == null) return '';
  return [
    `tx=${vecSig(layer.transmission, [1, 1, 1], 3)}`,
    `rough=${layer.roughness === undefined ? 'absent' : numberSig(layer.roughness, 0)}`,
    `normal=${textureRefSig(layer.normalMap)}`,
    `normalScale=${numberSig(layer.normalScale, 1)}`,
  ].join(';');
}

function materialExtensionSig(m: MaterialSpec): string {
  const skipEmitter = m.extensions?.['skipEmitter'] === true ? '1' : '0';
  const rawSurfaceTextureId = m.extensions?.['surfaceTextureId'];
  const surfaceTextureId =
    Number.isSafeInteger(rawSurfaceTextureId) &&
    (rawSurfaceTextureId as number) >= 0 &&
    (rawSurfaceTextureId as number) <= 7
      ? rawSurfaceTextureId as number
      : 0;
  return `skipEmitter=${skipEmitter};surfaceTextureId=${surfaceTextureId}`;
}

function stableJsonSig(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'number') return float32Token(value);
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJsonSig).join(',')}]`;
  if (typeof value === 'object') {
    return `{${Object.keys(value as Record<string, unknown>)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJsonSig((value as Record<string, unknown>)[key])}`)
      .join(',')}}`;
  }
  if (typeof value === 'bigint') return `${value.toString()}n`;
  if (typeof value === 'symbol') return `symbol:${value.description ?? ''}`;
  if (typeof value === 'function') return `function:${value.name}`;
  return '';
}

/**
 * Structural material signature — the core `MaterialSpec` counterpart to
 * `legacy/bvhCommon.ts:snapshotPreBuildMaterials`'s `matSig`. Hashes the fields
 * the merged-BVH GI/PT consumers read: base PBR/alpha/transmission scalars,
 * lobe-extension scalars, Beer-Lambert fields, all packed texture-map refs
 * including handle identity + UV transform/sampler metadata, and pt-webgl2's
 * folded mesh-emitter shadow flag. Numeric tokens use the same Float32 precision
 * as the GPU payloads and serialize canonical IEEE-754 bits, so adjacent values
 * and signed zero cannot be rounded or stringified together. Omitted values use
 * the documented field default; explicit NaN/+Infinity/-Infinity remain
 * deterministic, distinct tokens. Map identity uses the opaque `TextureRef.handle`
 * (the core analogue of THREE's `texture.uuid`).
 *
 * Beer-Lambert fields (`attenuationColor`, `attenuationDistance`, `thickness`) are
 * `attenuationDistance` is the one infinite-default field: omission and explicit
 * +Infinity therefore share `f32:+inf`; NaN and -Infinity remain distinct.
 */
export function materialSig(m: MaterialSpec): string {
  const colS = vecSig(m.baseColor, [0, 0, 0], 3);
  const emS = vecSig(m.emissive, [0, 0, 0], 3);
  const acS = vecSig(m.attenuationColor, [1, 1, 1], 3);
  const adS = numberSig(m.attenuationDistance, Number.POSITIVE_INFINITY);
  const meshEmitterShadow = (m as MaterialSpec & {
    meshEmitterCastShadowDisabled?: boolean;
  }).meshEmitterCastShadowDisabled === true ? '1' : '0';
  return [
    `base=${colS}`,
    `em=${emS}`,
    `emI=${numberSig(m.emissiveIntensity, 1)}`,
    `rough=${numberSig(m.roughness, 0.5)}`,
    `metal=${numberSig(m.metallic, 0)}`,
    `shade=${m.shadingModel ?? 'pbr'}`,
    `alpha=${m.alphaMode ?? 'opaque'},${numberSig(m.alphaCutoff, 0.5)},${numberSig(m.opacity, 1)}`,
    `side=${m.doubleSided === true ? '1' : '0'}`,
    `trans=${numberSig(m.transmission, 0)}`,
    `ior=${numberSig(m.ior, 1.5)}`,
    `beer=${acS},${adS},${numberSig(m.thickness, 0)}`,
    `mapScalar=${numberSig(m.normalScale, 1)},${numberSig(m.clearcoatNormalScale, 1)},${numberSig(m.aoMapIntensity, 1)},${numberSig(m.bumpScale, 1)},${numberSig(m.lightMapIntensity, 1)},${numberSig(m.envMapIntensity, 1)}`,
    `spec=${vecSig(m.specularColor, [1, 1, 1], 3)},${numberSig(m.specularIntensity, 1)}`,
    `coatSheen=${numberSig(m.clearcoat, 0)},${numberSig(m.clearcoatRoughness, 0)},${numberSig(m.sheen, 0)},${vecSig(m.sheenColor, [0, 0, 0], 3)},${numberSig(m.sheenRoughness, 0)}`,
    `aniso=${numberSig(m.anisotropy, 0)},${numberSig(m.anisotropyRotation, 0)}`,
    `iridescence=${numberSig(m.iridescence, 0)},${numberSig(m.iridescenceIor, 1.3)},${vecSig(m.iridescenceThicknessRange, [100, 400], 2)}`,
    `reservedDisp=${textureRefSig(m.displacementMap)},${numberSig(m.displacementScale, 1)},${numberSig(m.displacementBias, 0)},${numberSig(m.displacementSubdivisions, 0)}`,
    `volume=${numberSig(m.scatteringCoefficient, 0)},${numberSig(m.scatteringAnisotropy, 0)},${vecSig(m.scatteringCoefficientRGB, [0, 0, 0], 3)}`,
    `spectral=${stableJsonSig(m.spectralAttenuation)},${numberSig(m.dispersionAbbeNumber, 0)}`,
    `layers=${surfaceLayerSig(m.frontLayer)},${surfaceLayerSig(m.backLayer)},${stableJsonSig(m.thinFilmStack)}`,
    `maps=${textureMapSig(m)}`,
    `meshEmitterShadow=${meshEmitterShadow}`,
    `extensions=${materialExtensionSig(m)}`,
  ].join('|');
}

/**
 * Stable per-object identity registry for the material dedup signature.
 *
 * The module-level WeakMap is LOAD-BEARING: handle identity must persist
 * ACROSS merge calls so the same object (e.g. the same decoded ImageBitmap)
 * always maps to the same signature token. Wrapping it in an exported object
 * enables test-time reset without exposing the raw module globals.
 *
 * `reset()` is intentionally not called in production — it would invalidate
 * cached signatures and break dedup continuity. Call it only in test teardown
 * to prevent cross-test object-identity bleed.
 */
export const HandleIdRegistry = {
  _ids: new WeakMap<object, string>(),
  _symbolIds: new Map<symbol, string>(),
  _seq: 0,
  /** Return the stable id for `handle`. Assigns a new one on first encounter. */
  get(handle: object): string {
    let id = this._ids.get(handle);
    if (id === undefined) {
      id = `h${this._seq++}`;
      this._ids.set(handle, id);
    }
    return id;
  },
  /** Return a stable identity token for a symbol-valued opaque handle. */
  getSymbol(handle: symbol): string {
    let id = this._symbolIds.get(handle);
    if (id === undefined) {
      id = `s${this._seq++}`;
      this._symbolIds.set(handle, id);
    }
    return id;
  },
  /**
   * Reset the registry — for TEST USE ONLY.
   * Clears all assigned ids and resets the sequence counter.
   * Do NOT call in production: existing handle ids become stale.
   */
  reset(): void {
    this._ids = new WeakMap();
    this._symbolIds = new Map();
    this._seq = 0;
  },
};

/** A stable per-handle identity string for the dedup signature. Objects use
 *  {@link HandleIdRegistry}; primitives stringify directly; absent handles
 *  contribute the empty string. */
function handleId(handle: unknown): string {
  if (handle == null) return '';
  if (typeof handle === 'object' || typeof handle === 'function') {
    return `object:${HandleIdRegistry.get(handle)}`;
  }
  switch (typeof handle) {
    case 'string':
      // JSON quoting makes the token injective and delimiter-safe inside the
      // larger structural material signature.
      return `string:${JSON.stringify(handle)}`;
    case 'number':
      if (Number.isNaN(handle)) return 'number:nan';
      if (Object.is(handle, -0)) return 'number:-0';
      if (handle === Number.POSITIVE_INFINITY) return 'number:+inf';
      if (handle === Number.NEGATIVE_INFINITY) return 'number:-inf';
      return `number:${handle.toString()}`;
    case 'boolean':
      return `boolean:${handle ? '1' : '0'}`;
    case 'bigint':
      return `bigint:${handle.toString()}`;
    case 'symbol':
      return `symbol:${HandleIdRegistry.getSymbol(handle)}`;
    case 'undefined':
      return '';
  }
  return '';
}
