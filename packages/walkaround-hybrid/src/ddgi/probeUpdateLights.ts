/**
 * DDGI probe-update runtime storage-buffer packing.
 *
 * The 16-byte header is followed by one 64-byte light record and one 16-byte
 * alias-table record for every active light.
 * There is no renderer-authored light cap; the only upper bound is the WebGPU
 * device's storage-buffer limit, which ProbeUpdatePass checks before upload.
 */
import { buildAliasTable, luminance } from '@vitrum/shared-samplers';
import type { DDGILight } from './types.js';
import {
  assertNonNegativeDdgiNumber,
  assertValidDdgiLights,
} from './inputValidation.js';

const LIGHT_STRIDE_FLOATS = 16;
const HEADER_FLOATS = 4;
const ALIAS_STRIDE_FLOATS = 4;
export const DDGI_PROBE_LIGHTS_ABI_MAGIC = 0x444c4131;
export const DDGI_LIGHT_KIND_SUN = 0;
export const DDGI_LIGHT_KIND_POINT = 1;
export const DDGI_LIGHT_KIND_SPOT = 2;
export const DDGI_LIGHT_KIND_MASK = 0x7fffffff;
export const DDGI_LIGHT_CAST_SHADOW_DISABLED = 0x80000000;
/** Header-only buffer size used before the first non-empty light upload. */
export const DDGI_PROBE_LIGHTS_BUFFER_BYTES =
  HEADER_FLOATS * Float32Array.BYTES_PER_ELEMENT;

/** Exact byte length of a packed DDGI light buffer. */
export function ddgiProbeLightsBufferByteLength(count: number): number {
  if (!Number.isSafeInteger(count) || count < 0 || count > 0xffff_ffff) {
    throw new RangeError('[DDGI] active probe-light count must fit in u32.');
  }
  const bytes = (
    HEADER_FLOATS + count * (LIGHT_STRIDE_FLOATS + ALIAS_STRIDE_FLOATS)
  )
    * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(bytes)) {
    throw new RangeError('[DDGI] probe-light buffer byte length is not safe.');
  }
  return bytes;
}

/** Solid-angle support used by the probe-light alias proposal. */
export function ddgiProbeLightSolidAngle(kindWord: number, outerCone: number): number {
  const kind = kindWord & DDGI_LIGHT_KIND_MASK;
  return kind === DDGI_LIGHT_KIND_SPOT
    ? 2 * Math.PI * Math.max(0, 1 - outerCone)
    : 4 * Math.PI;
}

export function packDDGIProbeLights(
  lights: readonly DDGILight[],
  sunIntensityMul: number,
): ArrayBuffer {
  assertValidDdgiLights(lights, 'packDDGIProbeLights lights');
  assertNonNegativeDdgiNumber(
    sunIntensityMul,
    'packDDGIProbeLights sunIntensityMul',
  );
  const active = lights.filter((l) => l.on);
  const data = new Float32Array(ddgiProbeLightsBufferByteLength(active.length) / 4);
  const udata = new Uint32Array(data.buffer);
  udata[0] = active.length;
  udata[1] = HEADER_FLOATS;
  udata[2] = HEADER_FLOATS + active.length * LIGHT_STRIDE_FLOATS;
  udata[3] = DDGI_PROBE_LIGHTS_ABI_MAGIC;

  active.forEach((l, i) => {
    const base = HEADER_FLOATS + i * LIGHT_STRIDE_FLOATS;
    const ubase = base;
    const shadowFlag = l.castShadow === false ? DDGI_LIGHT_CAST_SHADOW_DISABLED : 0;
    if (l.kind === 'sun') {
      // Sun travel direction. When a `@vitrum/core` `directional` emitter
      // drives the sun, `coreEmitterToDDGILight` carries its real direction
      // here (already negated to a travel direction). The WGSL shader negates
      // again (`normalize(-light.direction)`) to recover the toward-light dir.
      // Absent direction → legacy hardcoded straight-down sun (0,-1,0) so a
      // host-supplied sun light with no direction is unchanged.
      const dir = l.direction;
      const col = l.color;
      udata[ubase] = (DDGI_LIGHT_KIND_SUN | shadowFlag) >>> 0;
      data[base + 4] = 0;
      data[base + 5] = 0;
      data[base + 6] = 0;
      data[base + 7] = l.intensity * sunIntensityMul;
      data[base + 8] = dir?.x ?? 0;
      data[base + 9] = dir?.y ?? -1;
      data[base + 10] = dir?.z ?? 0;
      data[base + 11] = typeof l.angularRadius === 'number' && Number.isFinite(l.angularRadius)
        ? Math.max(0, l.angularRadius)
        : 0;
      // Sun chroma: from the emitter when present; else the legacy warm-white
      // (1,0.95,0.85) the packer hardcoded before scene-directional wiring.
      data[base + 12] = col?.r ?? 1;
      data[base + 13] = col?.g ?? 0.95;
      data[base + 14] = col?.b ?? 0.85;
      data[base + 15] = 0;
    } else if (l.kind === 'fixture' || l.kind === 'teaLight') {
      const pos = l.position;
      const col = l.color;
      // [1,2] = distance/decay, [8,9,10] = spot cone axis
      // (forward beam/travel; 0 for a point → no cone in the shader),
      // [11] = innerCone (cosInner), [15] = outerCone (cosOuter). These map
      // to the DDGILight WGSL struct's distance / decay / direction /
      // innerCone / outerCone.
      const spot = l.spotAxis;
      const isSpot = spot != null &&
        (l.spotCosInner != null || l.spotCosOuter != null);
      udata[ubase] = (
        (isSpot ? DDGI_LIGHT_KIND_SPOT : DDGI_LIGHT_KIND_POINT) | shadowFlag
      ) >>> 0;
      data[base + 1] = typeof l.distance === 'number' && l.distance > 0 ? l.distance : 0;
      data[base + 2] = typeof l.decay === 'number' ? l.decay : 2;
      data[base + 4] = pos?.x ?? 0;
      data[base + 5] = pos?.y ?? 0;
      data[base + 6] = pos?.z ?? 0;
      data[base + 7] = l.intensity;
      data[base + 8] = spot?.x ?? 0;
      data[base + 9] = spot?.y ?? 0;
      data[base + 10] = spot?.z ?? 0;
      data[base + 11] = l.spotCosInner ?? 0;
      data[base + 12] = col?.r ?? 1;
      data[base + 13] = col?.g ?? 1;
      data[base + 14] = col?.b ?? 1;
      data[base + 15] = l.spotCosOuter ?? 0;
    }
  });

  const weights = active.map((_light, index) => {
    const base = HEADER_FLOATS + index * LIGHT_STRIDE_FLOATS;
    const emittedLuminance = luminance(
      data[base + 12]!,
      data[base + 13]!,
      data[base + 14]!,
    ) * data[base + 7]!;
    if ((udata[base]! & DDGI_LIGHT_KIND_MASK) === DDGI_LIGHT_KIND_SUN) {
      return emittedLuminance;
    }
    const solidAngle = ddgiProbeLightSolidAngle(
      udata[base]!,
      data[base + 15]!,
    );
    return emittedLuminance * solidAngle;
  });
  const alias = buildAliasTable(weights);
  new Uint8Array(data.buffer).set(
    new Uint8Array(alias.data),
    udata[2] * Uint32Array.BYTES_PER_ELEMENT,
  );
  return data.buffer;
}
