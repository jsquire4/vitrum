/**
 * Finite-binary32 policy for standalone Radiance Cascades environment input.
 *
 * Individual RGB channels may underflow to zero. Invalid inputs and any
 * NaN/Infinity product fail the complete RGB stage closed.
 */

export type RcEnvironmentRgb = readonly [number, number, number];

export const RC_ENVIRONMENT_RADIANCE_SCALE_WGSL = /* wgsl */ `
fn rcScaleEnvironmentRadiance(value: vec3f, scale: f32) -> vec3f {
  let maxFinite = bitcast<f32>(0x7f7fffffu);
  if (
    any(value < vec3f(0.0)) ||
    !all(value == value) ||
    any(abs(value) > vec3f(maxFinite)) ||
    !(scale >= 0.0) ||
    scale != scale ||
    abs(scale) > maxFinite
  ) {
    return vec3f(0.0);
  }
  let scaled = value * scale;
  if (!all(scaled == scaled) || any(abs(scaled) > vec3f(maxFinite))) {
    return vec3f(0.0);
  }
  return scaled;
}
`;

/** CPU mirror used by exact numeric tests. */
export function scaleRcEnvironmentRadianceF32(
  value: RcEnvironmentRgb,
  scale: number,
): [number, number, number] {
  const packedValue = value.map(Math.fround) as [number, number, number];
  const packedScale = Math.fround(scale);
  if (
    packedValue.some((component) => component < 0 || !Number.isFinite(component)) ||
    packedScale < 0 ||
    !Number.isFinite(packedScale)
  ) {
    return [0, 0, 0];
  }
  const scaled = packedValue.map((component) =>
    Math.fround(component * packedScale)
  ) as [number, number, number];
  return scaled.some((component) => !Number.isFinite(component))
    ? [0, 0, 0]
    : scaled;
}

/** Validate one non-negative scalar immediately before uniform publication. */
export function assertRcEnvironmentScaleF32(
  value: number,
  label: string,
): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(
      `[vitrum/walkaround-rc] ${label} must be finite and non-negative; ` +
        `received ${String(value)}.`,
    );
  }
  const packed = Math.fround(value);
  if (!Number.isFinite(packed) || (value > 0 && packed === 0)) {
    throw new RangeError(
      `[vitrum/walkaround-rc] ${label} must remain finite and positive after ` +
        `Float32 packing when non-zero; received ${String(value)}.`,
    );
  }
  return packed === 0 ? 0 : packed;
}

/**
 * Validate already-scaled scalar-sky radiance before uniform publication.
 * Per-channel underflow is valid while complete positive-RGB collapse is not.
 */
export function assertRcEnvironmentRadianceF32(
  value: RcEnvironmentRgb,
  label: string,
): [number, number, number] {
  const packed = value.map((channel, index) => {
    if (!Number.isFinite(channel) || channel < 0) {
      throw new RangeError(
        `[vitrum/walkaround-rc] ${label}[${index}] must be finite and ` +
          `non-negative; received ${String(channel)}.`,
      );
    }
    const rounded = Math.fround(channel);
    if (!Number.isFinite(rounded)) {
      throw new RangeError(
        `[vitrum/walkaround-rc] ${label}[${index}] must remain finite after ` +
          `Float32 packing; received ${String(channel)}.`,
      );
    }
    return rounded === 0 ? 0 : rounded;
  }) as [number, number, number];
  if (
    value.some((channel) => channel > 0) &&
    packed.every((channel) => channel === 0)
  ) {
    throw new RangeError(
      `[vitrum/walkaround-rc] positive ${label} underflows entirely to zero ` +
        'in Float32.',
    );
  }
  return packed;
}
