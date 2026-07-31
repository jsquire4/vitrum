/**
 * Canonical binary32 publication policy for runtime lighting controls.
 *
 * Zero is an authored value for non-negative radiometric scalars. A positive
 * value that rounds to zero is not equivalent—it silently disables a light—so
 * positive underflow is rejected alongside overflow.
 */

import {
  multiplyNonNegativeRadianceScalarsF32,
  packNonNegativeRadianceRgbF32,
  packNonNegativeRadianceScalarF32,
  packRadianceRgbScaleF32,
  type RadianceRgb,
} from '@vitrum/shared-bvh';

export type LightingVec3 = RadianceRgb;

function lightingRangeError(label: string, detail: string, value: unknown): RangeError {
  return new RangeError(
    `[vitrum/walkaround-hybrid] ${label} ${detail}; received ${String(value)}.`,
  );
}

/** Pack one finite signed scalar. Per-lane underflow to signed zero is allowed. */
export function packFiniteLightingFloat32(
  value: number,
  label: string,
): number {
  if (!Number.isFinite(value)) {
    throw lightingRangeError(label, 'must be finite', value);
  }
  const packed = Math.fround(value);
  if (!Number.isFinite(packed)) {
    throw lightingRangeError(label, 'must remain finite after Float32 packing', value);
  }
  return packed === 0 ? 0 : packed;
}

/**
 * Pack one non-negative radiometric/control scalar.
 *
 * Authored zero remains valid. Positive values must remain positive after
 * binary32 packing because zero changes light/cutoff/falloff semantics.
 */
export function packNonNegativeLightingFloat32(
  value: number,
  label: string,
): number {
  return packNonNegativeRadianceScalarF32(value, label);
}

/**
 * Snapshot a non-negative RGB tuple into the exact values a GPU f32 tuple sees.
 * Individual positive lanes may underflow, but an authored non-black tuple may
 * not collapse completely to black.
 */
export function packNonNegativeLightingRgbF32(
  value: LightingVec3,
  label: string,
): [number, number, number] {
  return packNonNegativeRadianceRgbF32(value, label);
}

/**
 * Validate and snapshot an RGB×scalar radiance product in shader evaluation
 * order. Individual output lanes may underflow; a positive non-black source and
 * positive scale may not disappear completely.
 */
export function packLightingRgbScaleEnvelopeF32(
  value: LightingVec3,
  scale: number,
  label: string,
): {
  readonly value: [number, number, number];
  readonly scale: number;
  readonly scaled: [number, number, number];
} {
  return packRadianceRgbScaleF32(value, scale, label);
}

/**
 * Stable normalization + binary32 snapshot for a required direction.
 *
 * Scaling by the largest component before `hypot` avoids overflow for finite
 * Number-range inputs and avoids underflow for subnormal inputs. The returned
 * tuple is fresh, finite, non-zero, approximately unit length, and cannot be
 * changed later by mutating the host's input array.
 */
export function canonicalizeLightingDirectionF32(
  value: LightingVec3,
  label: string,
): [number, number, number] {
  for (let index = 0; index < 3; index += 1) {
    const component = value[index]!;
    if (!Number.isFinite(component)) {
      throw lightingRangeError(`${label}[${index}]`, 'must be finite', component);
    }
  }
  const scale = Math.max(
    Math.abs(value[0]),
    Math.abs(value[1]),
    Math.abs(value[2]),
  );
  if (!(scale > 0)) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] ${label} must be non-zero.`,
    );
  }
  const scaled: [number, number, number] = [
    value[0] / scale,
    value[1] / scale,
    value[2] / scale,
  ];
  const length = Math.hypot(scaled[0], scaled[1], scaled[2]);
  if (!(length > 0) || !Number.isFinite(length)) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] ${label} could not be normalized safely.`,
    );
  }
  const packed = scaled.map((component) =>
    Math.fround(component / length)
  ) as [number, number, number];
  if (
    packed.some((component) => !Number.isFinite(component)) ||
    packed.every((component) => component === 0)
  ) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] ${label} must remain a finite non-zero ` +
        'direction after Float32 packing.',
    );
  }
  return packed.map((component) => component === 0 ? 0 : component) as [
    number,
    number,
    number,
  ];
}

/**
 * Multiply two already-validated non-negative lighting scalars in binary32.
 * Zero from either authored operand is valid; two positive operands may not
 * overflow or collapse to zero.
 */
export function multiplyNonNegativeLightingFloat32(
  left: number,
  right: number,
  label: string,
): number {
  return multiplyNonNegativeRadianceScalarsF32(left, right, label);
}
