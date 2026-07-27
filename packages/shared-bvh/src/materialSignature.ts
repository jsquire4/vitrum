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

function finiteSig(value: number | undefined, fallback: number): string {
  const v = Number.isFinite(value) ? (value as number) : fallback;
  return String(Math.fround(v));
}

function rawNumberSig(value: number | undefined, fallback: number): string {
  if (value === undefined) return finiteSig(undefined, fallback);
  return Number.isFinite(value) ? String(Math.fround(value)) : String(value);
}

function vecSig(
  value: readonly number[] | undefined,
  fallback: readonly number[],
  count: 2 | 3,
): string {
  const parts: string[] = [];
  for (let i = 0; i < count; i += 1) {
    parts.push(finiteSig(value?.[i], fallback[i] ?? 0));
  }
  return parts.join(',');
}

function rawVecSig(
  value: readonly number[] | undefined,
  fallback: readonly number[],
  count: 2,
): string {
  const parts: string[] = [];
  for (let i = 0; i < count; i += 1) {
    parts.push(rawNumberSig(value?.[i], fallback[i] ?? 0));
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
  return `uvUnsupported=${rawNumberSig(uv, 0)}`;
}

function textureRefSig(value: unknown): string {
  const ref = textureRefLike(value);
  if (ref?.handle == null) return '';
  const transform = ref.transform;
  return [
    handleId(ref.handle),
    textureTexCoordSig(ref.texCoord),
    `off=${rawVecSig(transform?.offset, [0, 0], 2)}`,
    `scale=${rawVecSig(transform?.scale, [1, 1], 2)}`,
    `rot=${rawNumberSig(transform?.rotation, 0)}`,
    `wrap=${ref.wrapS ?? 'repeat'},${ref.wrapT ?? 'repeat'}`,
    `filter=${ref.magFilter ?? ''},${ref.minFilter ?? ''},${ref.mipFilter ?? ''}`,
  ].join(';');
}

function textureMapSig(m: MaterialSpec): string {
  return TEXTURE_MAP_FIELDS
    .map((field) => `${field}=${textureRefSig(m[field])}`)
    .join('|');
}

function stableJsonSig(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(4) : String(value);
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
 * as the GPU payloads, so atlas/material metadata differences that survive upload
 * cannot be rounded away by the dedup key. Map identity uses the opaque
 * `TextureRef.handle` (the core analogue of THREE's `texture.uuid`).
 *
 * Beer-Lambert fields (`attenuationColor`, `attenuationDistance`, `thickness`) are
 * included to match `materialSetHashFloats` in `sceneBvh.ts`. Infinity
 * `attenuationDistance` is normalised to the token `'Inf'` so the signature remains
 * a stable string (JSON.stringify of Infinity produces `null`).
 */
export function materialSig(m: MaterialSpec): string {
  const colS = vecSig(m.baseColor, [0, 0, 0], 3);
  const emS = vecSig(m.emissive, [0, 0, 0], 3);
  // Beer-Lambert fields — must match materialSetHashFloats in sceneBvh.ts.
  const acS = vecSig(m.attenuationColor, [1, 1, 1], 3);
  const adRaw = m.attenuationDistance;
  const adS = adRaw == null
    ? 'Inf'
    : !isFinite(adRaw)
      ? 'Inf'
      : adRaw.toFixed(4);
  const meshEmitterShadow = (m as MaterialSpec & {
    meshEmitterCastShadowDisabled?: boolean;
  }).meshEmitterCastShadowDisabled === true ? '1' : '0';
  return [
    `base=${colS}`,
    `em=${emS}`,
    `emI=${finiteSig(m.emissiveIntensity, 1)}`,
    `rough=${finiteSig(m.roughness, 0.5)}`,
    `metal=${finiteSig(m.metallic, 0)}`,
    `shade=${m.shadingModel ?? 'pbr'}`,
    `alpha=${m.alphaMode ?? 'opaque'},${finiteSig(m.alphaCutoff, 0.5)},${finiteSig(m.opacity, 1)}`,
    `trans=${finiteSig(m.transmission, 0)}`,
    `ior=${finiteSig(m.ior, 1.5)}`,
    `beer=${acS},${adS},${finiteSig(m.thickness, 0)}`,
    `mapScalar=${finiteSig(m.normalScale, 1)},${finiteSig(m.clearcoatNormalScale, 1)},${finiteSig(m.aoMapIntensity, 1)},${finiteSig(m.bumpScale, 1)},${finiteSig(m.lightMapIntensity, 1)},${finiteSig(m.envMapIntensity, 1)}`,
    `spec=${vecSig(m.specularColor, [1, 1, 1], 3)},${finiteSig(m.specularIntensity, 1)}`,
    `coatSheen=${finiteSig(m.clearcoat, 0)},${finiteSig(m.clearcoatRoughness, 0)},${finiteSig(m.sheen, 0)},${vecSig(m.sheenColor, [0, 0, 0], 3)},${finiteSig(m.sheenRoughness, 0)}`,
    `aniso=${finiteSig(m.anisotropy, 0)},${finiteSig(m.anisotropyRotation, 0)}`,
    `iridescence=${finiteSig(m.iridescence, 0)},${finiteSig(m.iridescenceIor, 1.3)},${vecSig(m.iridescenceThicknessRange, [100, 400], 2)}`,
    `reservedDisp=${textureRefSig(m.displacementMap)},${finiteSig(m.displacementScale, 1)},${finiteSig(m.displacementBias, 0)},${finiteSig(m.displacementSubdivisions, 0)}`,
    `volume=${finiteSig(m.scatteringCoefficient, 0)},${finiteSig(m.scatteringAnisotropy, 0)},${vecSig(m.scatteringCoefficientRGB, [0, 0, 0], 3)}`,
    `spectral=${stableJsonSig(m.spectralAttenuation)},${finiteSig(m.dispersionAbbeNumber, 0)}`,
    `layers=${stableJsonSig(m.frontLayer)},${stableJsonSig(m.backLayer)},${stableJsonSig(m.thinFilmStack)}`,
    `maps=${textureMapSig(m)}`,
    `meshEmitterShadow=${meshEmitterShadow}`,
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
  /**
   * Reset the registry — for TEST USE ONLY.
   * Clears all assigned ids and resets the sequence counter.
   * Do NOT call in production: existing handle ids become stale.
   */
  reset(): void {
    this._ids = new WeakMap();
    this._seq = 0;
  },
};

/** A stable per-handle identity string for the dedup signature. Objects use
 *  {@link HandleIdRegistry}; primitives stringify directly; absent handles
 *  contribute the empty string. */
function handleId(handle: unknown): string {
  if (handle == null) return '';
  if (typeof handle === 'object' || typeof handle === 'function') {
    return HandleIdRegistry.get(handle);
  }
  // eslint-disable-next-line @typescript-eslint/no-base-to-string -- at this point handle is a primitive (guarded: not object/function), String() is safe
  return String(handle);
}
