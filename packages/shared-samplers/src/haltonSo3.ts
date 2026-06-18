/**
 * Deterministic Halton-Shoemake SO(3) sampler.
 *
 * The returned vec3 is an axis-angle rotation: direction is the rotation axis
 * and length is the angle in radians. This is convenient for Rodrigues-style
 * shader rotation uniforms.
 */

function haltonBase(i: number, base: number): number {
  let result = 0;
  let f = 1;
  let n = i;
  while (n > 0) {
    f /= base;
    result += f * (n % base);
    n = Math.floor(n / base);
  }
  return result;
}

/** Axis-angle vec3 for a frame-indexed, deterministic uniform SO(3) sequence. */
export function haltonSO3AxisAngleFromFrameIndex(frameIndex: number): [number, number, number] {
  const fi = frameIndex + 1;
  const u1 = haltonBase(fi, 2);
  const u2 = haltonBase(fi, 3);
  const u3 = haltonBase(fi, 5);
  const sigma1 = Math.sqrt(1 - u1);
  const sigma2 = Math.sqrt(u1);
  const theta1 = 2 * Math.PI * u2;
  const theta2 = 2 * Math.PI * u3;
  const qw = sigma2 * Math.cos(theta2);
  const qx = sigma1 * Math.sin(theta1);
  const qy = sigma1 * Math.cos(theta1);
  const qz = sigma2 * Math.sin(theta2);
  const angle = 2 * Math.acos(Math.min(1, Math.abs(qw)));
  const sinHalf = Math.sqrt(Math.max(0, 1 - qw * qw));
  let ax: number;
  let ay: number;
  let az: number;
  if (sinHalf < 1e-6) {
    ax = 1;
    ay = 0;
    az = 0;
  } else {
    ax = qx / sinHalf;
    ay = qy / sinHalf;
    az = qz / sinHalf;
  }
  return [ax * angle, ay * angle, az * angle];
}
