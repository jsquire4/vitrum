import type { DDGILight } from './types.js';
import {
  DDGI_NORMAL_BIAS_FACTOR_F32,
  DDGI_F32_MAX,
  DDGI_PROBE_SPACING_MAX,
  DDGI_PROBE_SPACING_MIN,
  DDGI_U32_MAX,
} from './ddgiNumericLimits.js';
export {
  DDGI_F32_MAX,
  DDGI_F32_MIN_POSITIVE,
  DDGI_U32_MAX,
} from './ddgiNumericLimits.js';

const COMMON_DDGI_LIGHT_KEYS = [
  'kind',
  'id',
  'intensity',
  'on',
  'castShadow',
  'color',
] as const;
const DDGI_LIGHT_KEYS_BY_KIND = {
  sun: new Set<string>([
    ...COMMON_DDGI_LIGHT_KEYS,
    'direction',
    'angularRadius',
  ]),
  fixture: new Set<string>([
    ...COMMON_DDGI_LIGHT_KEYS,
    'position',
    'spotAxis',
    'spotCosInner',
    'spotCosOuter',
    'distance',
    'decay',
  ]),
  teaLight: new Set<string>([
    ...COMMON_DDGI_LIGHT_KEYS,
    'position',
    'distance',
    'decay',
  ]),
} as const;

function assertExactEnumerableKeys(
  value: object,
  allowed: ReadonlySet<string>,
  label: string,
): void {
  for (const key of Reflect.ownKeys(value)) {
    if (!Object.prototype.propertyIsEnumerable.call(value, key)) continue;
    if (typeof key !== 'string' || !allowed.has(key)) {
      const rendered = typeof key === 'symbol' ? key.toString() : JSON.stringify(key);
      throw new TypeError(`${label} contains unsupported enumerable field ${rendered}.`);
    }
  }
}

/** Fail before a non-finite host value can enter a DDGI CPU/GPU mirror. */
export function assertFiniteDdgiNumber(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite.`);
  }
}

/** Canonicalize a JS number to a finite value that the DDGI f32 ABI can publish. */
export function packFiniteDdgiFloat32(value: number, label: string): number {
  assertFiniteDdgiNumber(value, label);
  if (Math.abs(value) > DDGI_F32_MAX) {
    throw new RangeError(`${label} must be within the finite float32 range.`);
  }
  const packed = Math.fround(value);
  if (!Number.isFinite(packed)) {
    throw new RangeError(`${label} must remain finite when published as float32.`);
  }
  return packed;
}

/** Positive values must remain strictly positive after f32 publication. */
export function packPositiveDdgiFloat32(value: number, label: string): number {
  if (!(value > 0)) throw new RangeError(`${label} must be > 0.`);
  const packed = packFiniteDdgiFloat32(value, label);
  if (!(packed > 0)) {
    throw new RangeError(`${label} must remain > 0 when published as float32.`);
  }
  return packed;
}

/** Canonical spacing envelope for every positive product used by DDGI WGSL. */
export function packDdgiProbeSpacingFloat32(value: number, label: string): number {
  const packed = packPositiveDdgiFloat32(value, label);
  if (packed < DDGI_PROBE_SPACING_MIN || packed > DDGI_PROBE_SPACING_MAX) {
    throw new RangeError(
      `${label} must be within [${DDGI_PROBE_SPACING_MIN}, ${DDGI_PROBE_SPACING_MAX}] for finite derived float32 arithmetic.`,
    );
  }
  const normalBias = Math.fround(packed * DDGI_NORMAL_BIAS_FACTOR_F32);
  if (!(normalBias > 0) || !(Math.fround(normalBias * normalBias) > 0)) {
    throw new RangeError(`${label} must preserve a positive float32 normal-bias square.`);
  }
  for (const [factor, derivedLabel] of [
    [1e-6, 'minimum ray offset'],
    [1e-5, 'minimum hit distance'],
    [0.05, 'classification distance'],
    [0.2, 'relocation distance'],
    [0.25, 'receiver bias'],
    [0.45, 'maximum relocation'],
    [16, 'visibility open distance'],
  ] as const) {
    const derived = Math.fround(packed * Math.fround(factor));
    if (!(derived > 0) || !Number.isFinite(derived)) {
      throw new RangeError(`${label} must preserve a positive finite float32 ${derivedLabel}.`);
    }
  }
  return packed;
}

/** Validate an integer carried by a DDGI u32 lane. */
export function assertDdgiU32(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > DDGI_U32_MAX) {
    throw new RangeError(`${label} must be an unsigned 32-bit integer.`);
  }
}

/** A probe cadence is a count, not a continuously-valued tuning parameter. */
export function assertPositiveDdgiInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer.`);
  }
}

/**
 * The full-blend retry state currently stores one Set entry per cadence
 * stratum. Keep that explicitly bounded so a hostile-but-integer tuning value
 * cannot request an OOM-sized allocation before it reaches the u32 shader ABI.
 */
export const DDGI_MAX_PROBE_UPDATE_DIVISOR = 65_536;

export function validateDdgiProbeUpdateDivisor(
  value: number,
  label = 'DDGI probe update divisor',
): number {
  assertPositiveDdgiInteger(value, label);
  assertDdgiU32(value, label);
  if (value > DDGI_MAX_PROBE_UPDATE_DIVISOR) {
    throw new RangeError(
      `${label} must be <= ${DDGI_MAX_PROBE_UPDATE_DIVISOR}.`,
    );
  }
  return value;
}

/** Offsets may be outside the stride (they are reduced modulo it), but integer. */
export function assertDdgiInteger(value: number, label: string): void {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new RangeError(`${label} must be a finite integer.`);
  }
}

export function assertFiniteDdgiVec3(
  value: readonly [number, number, number],
  label: string,
): void {
  for (let i = 0; i < 3; i++) {
    assertFiniteDdgiNumber(value[i]!, `${label}[${i}]`);
  }
}

export function assertNonNegativeDdgiNumber(value: number, label: string): void {
  assertFiniteDdgiNumber(value, label);
  if (value < 0) throw new RangeError(`${label} must be non-negative.`);
}

export function assertDdgiUnitInterval(value: number, label: string): void {
  assertFiniteDdgiNumber(value, label);
  if (value < 0 || value > 1) {
    throw new RangeError(`${label} must be within [0, 1].`);
  }
}

export function assertDdgiBoolean(
  value: unknown,
  label: string,
): asserts value is boolean {
  if (typeof value !== 'boolean') throw new TypeError(`${label} must be boolean.`);
}

function assertFiniteObjectVec3(
  value: { readonly x: number; readonly y: number; readonly z: number },
  label: string,
): void {
  assertExactEnumerableKeys(value, new Set(['x', 'y', 'z']), label);
  assertFiniteDdgiNumber(value.x, `${label}.x`);
  assertFiniteDdgiNumber(value.y, `${label}.y`);
  assertFiniteDdgiNumber(value.z, `${label}.z`);
}

function assertNonZeroDirection(
  value: { readonly x: number; readonly y: number; readonly z: number },
  label: string,
): void {
  assertFiniteObjectVec3(value, label);
  if (value.x === 0 && value.y === 0 && value.z === 0) {
    throw new RangeError(`${label} must be non-zero.`);
  }
}

function assertNonNegativeColor(
  value: { readonly r: number; readonly g: number; readonly b: number },
  label: string,
): void {
  assertExactEnumerableKeys(value, new Set(['r', 'g', 'b']), label);
  assertNonNegativeDdgiNumber(value.r, `${label}.r`);
  assertNonNegativeDdgiNumber(value.g, `${label}.g`);
  assertNonNegativeDdgiNumber(value.b, `${label}.b`);
}

/** Validate every field consumed by the DDGI light packer before snapshotting. */
export function assertValidDdgiLights(
  lights: readonly DDGILight[],
  label = 'DDGI lights',
): void {
  // Check the runtime surface through an unknown alias so Array.isArray's
  // `any[]` predicate cannot erase the declared DDGILight element type below.
  const runtimeLights: unknown = lights;
  if (!Array.isArray(runtimeLights)) {
    throw new TypeError(`${label} must be an array.`);
  }
  lights.forEach((light, index) => {
    const lightLabel = `${label}[${index}]`;
    if (light == null || typeof light !== 'object') {
      throw new TypeError(`${lightLabel} must be an object.`);
    }
    if (
      light.kind !== 'sun' &&
      light.kind !== 'fixture' &&
      light.kind !== 'teaLight'
    ) {
      throw new TypeError(
        `${lightLabel}.kind must be 'sun', 'fixture', or 'teaLight'; received ${JSON.stringify(light.kind)}.`,
      );
    }
    assertExactEnumerableKeys(
      light,
      DDGI_LIGHT_KEYS_BY_KIND[light.kind],
      lightLabel,
    );
    if (
      light.id !== undefined &&
      (typeof light.id !== 'string' || light.id.trim().length === 0)
    ) {
      throw new TypeError(`${lightLabel}.id must be a non-empty string when supplied.`);
    }
    assertNonNegativeDdgiNumber(light.intensity, `${lightLabel}.intensity`);
    assertDdgiBoolean(light.on, `${lightLabel}.on`);
    if (light.castShadow !== undefined) {
      assertDdgiBoolean(light.castShadow, `${lightLabel}.castShadow`);
    }
    if (light.position !== undefined) {
      assertFiniteObjectVec3(light.position, `${lightLabel}.position`);
    }
    if (light.direction !== undefined) {
      assertNonZeroDirection(light.direction, `${lightLabel}.direction`);
    }
    if (light.color !== undefined) {
      assertNonNegativeColor(light.color, `${lightLabel}.color`);
    }
    if (light.angularRadius !== undefined) {
      assertNonNegativeDdgiNumber(
        light.angularRadius,
        `${lightLabel}.angularRadius`,
      );
      if (light.angularRadius > Math.PI) {
        throw new RangeError(`${lightLabel}.angularRadius must be <= PI.`);
      }
    }
    if (light.distance !== undefined) {
      assertNonNegativeDdgiNumber(light.distance, `${lightLabel}.distance`);
    }
    if (light.decay !== undefined) {
      assertNonNegativeDdgiNumber(light.decay, `${lightLabel}.decay`);
    }

    const hasSpotField =
      light.spotAxis !== undefined ||
      light.spotCosInner !== undefined ||
      light.spotCosOuter !== undefined;
    if (hasSpotField) {
      if (
        light.spotAxis === undefined ||
        light.spotCosInner === undefined ||
        light.spotCosOuter === undefined
      ) {
        throw new TypeError(
          `${lightLabel} spotAxis, spotCosInner, and spotCosOuter must be supplied together.`,
        );
      }
      assertNonZeroDirection(light.spotAxis, `${lightLabel}.spotAxis`);
      assertFiniteDdgiNumber(light.spotCosInner, `${lightLabel}.spotCosInner`);
      assertFiniteDdgiNumber(light.spotCosOuter, `${lightLabel}.spotCosOuter`);
      if (light.spotCosInner < -1 || light.spotCosInner > 1) {
        throw new RangeError(`${lightLabel}.spotCosInner must be within [-1, 1].`);
      }
      if (light.spotCosOuter < -1 || light.spotCosOuter > 1) {
        throw new RangeError(`${lightLabel}.spotCosOuter must be within [-1, 1].`);
      }
      if (light.spotCosInner < light.spotCosOuter) {
        throw new RangeError(
          `${lightLabel}.spotCosInner must be >= spotCosOuter.`,
        );
      }
    }
  });
}
