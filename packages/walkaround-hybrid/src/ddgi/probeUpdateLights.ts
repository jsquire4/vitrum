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
  assertDdgiU32,
  assertNonNegativeDdgiNumber,
  assertValidDdgiLights,
} from './inputValidation.js';
import {
  canonicalizeLightingDirectionF32,
  multiplyNonNegativeLightingFloat32,
  packFiniteLightingFloat32,
  packLightingRgbScaleEnvelopeF32,
  packNonNegativeLightingFloat32,
  packNonNegativeLightingRgbF32,
} from '../lightingFloat32.js';

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

type PackedRgb = readonly [number, number, number];

function packLightColor(
  color: { readonly r: number; readonly g: number; readonly b: number } | undefined,
  fallback: PackedRgb,
  label: string,
): [number, number, number] {
  return packNonNegativeLightingRgbF32(
    color == null ? fallback : [color.r, color.g, color.b],
    label,
  );
}

function packLightDirection(
  direction: { readonly x: number; readonly y: number; readonly z: number },
  label: string,
): [number, number, number] {
  return canonicalizeLightingDirectionF32(
    [direction.x, direction.y, direction.z],
    label,
  );
}

/**
 * Alias proposal weights never cross a Float32 wire. Keep their product in
 * binary64 so a valid finite-f32 radiance near the top of its range is not
 * rejected merely because the heuristic multiplies it by a 4π support angle.
 * `buildAliasTable` max-normalizes these finite host values before publishing
 * represented f32 thresholds and PMFs.
 */
function ddgiAliasProposalWeight(
  emittedLuminance: number,
  solidAngle: number,
  label: string,
): number {
  const weight = emittedLuminance * solidAngle;
  if (!Number.isFinite(weight) || weight < 0) {
    throw new RangeError(`${label} must be finite and non-negative.`);
  }
  return weight;
}

/** Exact byte length of a packed DDGI light buffer. */
export function ddgiProbeLightsBufferByteLength(count: number): number {
  assertDdgiU32(count, '[DDGI] active probe-light count');
  const wordCount =
    HEADER_FLOATS + count * (LIGHT_STRIDE_FLOATS + ALIAS_STRIDE_FLOATS);
  assertDdgiU32(wordCount, '[DDGI] probe-light record/alias word count');
  const bytes = wordCount * Float32Array.BYTES_PER_ELEMENT;
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
  const packedSunIntensityMul = packNonNegativeLightingFloat32(
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
      const packedDirection = dir == null
        ? [0, -1, 0] as [number, number, number]
        : packLightDirection(dir, `packDDGIProbeLights lights[${i}].direction`);
      const packedColor = packLightColor(
        l.color,
        [1, 0.95, 0.85],
        `packDDGIProbeLights lights[${i}].color`,
      );
      const packedIntensity = multiplyNonNegativeLightingFloat32(
        packNonNegativeLightingFloat32(
          l.intensity,
          `packDDGIProbeLights lights[${i}].intensity`,
        ),
        packedSunIntensityMul,
        `packDDGIProbeLights lights[${i}] sun intensity×multiplier`,
      );
      udata[ubase] = (DDGI_LIGHT_KIND_SUN | shadowFlag) >>> 0;
      data[base + 4] = 0;
      data[base + 5] = 0;
      data[base + 6] = 0;
      data[base + 7] = packedIntensity;
      data[base + 8] = packedDirection[0];
      data[base + 9] = packedDirection[1];
      data[base + 10] = packedDirection[2];
      data[base + 11] = typeof l.angularRadius === 'number'
        ? packNonNegativeLightingFloat32(
            l.angularRadius,
            `packDDGIProbeLights lights[${i}].angularRadius`,
          )
        : 0;
      // Sun chroma: from the emitter when present; else the legacy warm-white
      // (1,0.95,0.85) the packer hardcoded before scene-directional wiring.
      data[base + 12] = packedColor[0];
      data[base + 13] = packedColor[1];
      data[base + 14] = packedColor[2];
      data[base + 15] = 0;
    } else if (l.kind === 'fixture' || l.kind === 'teaLight') {
      const pos = l.position;
      const packedColor = packLightColor(
        l.color,
        [1, 1, 1],
        `packDDGIProbeLights lights[${i}].color`,
      );
      // [1,2] = distance/decay, [8,9,10] = spot cone axis
      // (forward beam/travel; 0 for a point → no cone in the shader),
      // [11] = innerCone (cosInner), [15] = outerCone (cosOuter). These map
      // to the DDGILight WGSL struct's distance / decay / direction /
      // innerCone / outerCone.
      const spot = l.spotAxis;
      const isSpot = spot != null &&
        (l.spotCosInner != null || l.spotCosOuter != null);
      const packedSpot = isSpot
        ? packLightDirection(
            spot,
            `packDDGIProbeLights lights[${i}].spotAxis`,
          )
        : [0, 0, 0] as [number, number, number];
      udata[ubase] = (
        (isSpot ? DDGI_LIGHT_KIND_SPOT : DDGI_LIGHT_KIND_POINT) | shadowFlag
      ) >>> 0;
      data[base + 1] = typeof l.distance === 'number' && l.distance > 0
        ? packNonNegativeLightingFloat32(
            l.distance,
            `packDDGIProbeLights lights[${i}].distance`,
          )
        : 0;
      data[base + 2] = packNonNegativeLightingFloat32(
        typeof l.decay === 'number' ? l.decay : 2,
        `packDDGIProbeLights lights[${i}].decay`,
      );
      data[base + 4] = packFiniteLightingFloat32(
        pos?.x ?? 0,
        `packDDGIProbeLights lights[${i}].position.x`,
      );
      data[base + 5] = packFiniteLightingFloat32(
        pos?.y ?? 0,
        `packDDGIProbeLights lights[${i}].position.y`,
      );
      data[base + 6] = packFiniteLightingFloat32(
        pos?.z ?? 0,
        `packDDGIProbeLights lights[${i}].position.z`,
      );
      data[base + 7] = packNonNegativeLightingFloat32(
        l.intensity,
        `packDDGIProbeLights lights[${i}].intensity`,
      );
      data[base + 8] = packedSpot[0];
      data[base + 9] = packedSpot[1];
      data[base + 10] = packedSpot[2];
      data[base + 11] = packFiniteLightingFloat32(
        l.spotCosInner ?? 0,
        `packDDGIProbeLights lights[${i}].spotCosInner`,
      );
      data[base + 12] = packedColor[0];
      data[base + 13] = packedColor[1];
      data[base + 14] = packedColor[2];
      data[base + 15] = packFiniteLightingFloat32(
        l.spotCosOuter ?? 0,
        `packDDGIProbeLights lights[${i}].spotCosOuter`,
      );
    }
  });

  const weights = active.map((_light, index) => {
    const base = HEADER_FLOATS + index * LIGHT_STRIDE_FLOATS;
    const emitted = packLightingRgbScaleEnvelopeF32(
      [data[base + 12]!, data[base + 13]!, data[base + 14]!],
      data[base + 7]!,
      `packDDGIProbeLights lights[${index}]`,
    ).scaled;
    const emittedLuminance = ddgiAliasProposalWeight(
      luminance(emitted[0], emitted[1], emitted[2]),
      1,
      `packDDGIProbeLights lights[${index}] emitted luminance`,
    );
    if ((udata[base]! & DDGI_LIGHT_KIND_MASK) === DDGI_LIGHT_KIND_SUN) {
      return emittedLuminance;
    }
    const solidAngle = ddgiProbeLightSolidAngle(
      udata[base]!,
      data[base + 15]!,
    );
    return ddgiAliasProposalWeight(
      emittedLuminance,
      solidAngle,
      `packDDGIProbeLights lights[${index}] alias weight`,
    );
  });
  const alias = buildAliasTable(weights);
  new Uint8Array(data.buffer).set(
    new Uint8Array(alias.data),
    udata[2] * Uint32Array.BYTES_PER_ELEMENT,
  );
  return data.buffer;
}
