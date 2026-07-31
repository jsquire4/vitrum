const TWO_PI = 2 * Math.PI;

/**
 * Canonical environment-dome angle publication shared by every renderer.
 *
 * Reduce the finite JavaScript angle before Float32 packing so a huge but
 * periodic angle cannot become Infinity in a shader uniform. Signed zero is
 * normalized because +0 and -0 describe the same rotation.
 */
export function canonicalizeEnvironmentRotationF32(
  value: number,
  context = 'environment rotationY',
): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(
      `${context} must be finite; received ${String(value)}.`,
    );
  }
  const packed = Math.fround(value % TWO_PI);
  if (!Number.isFinite(packed)) {
    throw new RangeError(
      `${context} must remain finite after Float32 packing.`,
    );
  }
  return Object.is(packed, -0) ? 0 : packed;
}
