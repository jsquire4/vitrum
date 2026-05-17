/**
 * quaternionSampling — uniform SO(3) rotation sampling via the
 * Halton-base-{2,3,5} → Shoemake-quaternion → axis-angle pipeline.
 *
 * Use case: per-frame deterministic random rotations for QMC ray
 * decorrelation in probe-update / cascade-update passes. Walkaround's
 * DDGI probe-update pass calls this once per frame, seeded by the
 * frame index, to rotate the otherwise-fixed 192 probe ray directions
 * so the EMA hysteresis accumulates an effectively larger ray budget
 * over time.
 *
 * Mathematical pipeline:
 *   1. Halton sequence (bases 2, 3, 5) → three quasi-random uniforms in [0,1).
 *   2. Shoemake (Graphics Gems III §III.6, 1992) maps those three uniforms
 *      to a unit quaternion uniformly distributed on SO(3).
 *   3. Quaternion → axis-angle (axis × angle, the form WGSL Rodrigues
 *      rotation routines consume).
 *
 * References:
 *   Shoemake 1992 — "Uniform Random Rotations", Graphics Gems III §III.6.
 *   Halton 1964 — "Algorithm 247: Radical-inverse quasi-random point sequence".
 *   Majercik et al. 2019 §3.1 — per-frame SO(3) rotation for DDGI probes.
 *
 * Generic enough for any compute pass needing a deterministic uniform
 * rotation, hence its home in @vitrum/shared-samplers. Walkaround's
 * probeUpdatePass was the original (and presently only) consumer.
 */

/**
 * Van der Corput / Halton radical-inverse of `i` in the given prime `base`.
 *
 * Returns a value in `[0, 1)`. Quasi-random: low discrepancy across
 * sequential `i`, no correlation clumps the way Math.random() can produce.
 */
export function haltonBase(i: number, base: number): number {
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

/**
 * Sample a uniformly-distributed rotation in SO(3), returned in axis-angle
 * vec3 form (the axis scaled by the rotation angle in radians).
 *
 * Determinism: same `frameIndex` always returns the same rotation; this is
 * a QMC sequence, not a stochastic sampler.
 *
 * The returned `[ax·θ, ay·θ, az·θ]` is the conventional axis-angle form
 * consumed by Rodrigues rotation formulas (WGSL `rotateAngleAxis`, etc.).
 * Magnitude of the returned vector equals the rotation angle θ in radians.
 *
 * @param frameIndex Sample index (non-negative). Internally offset by +1
 *                   so frameIndex=0 still produces a meaningful rotation
 *                   (Halton at i=0 is identically zero).
 * @returns `[ax*angle, ay*angle, az*angle]` — axis-angle vec3.
 */
export function sampleUniformRotationAxisAngle(
  frameIndex: number,
): [number, number, number] {
  const fi = frameIndex + 1;
  const u1 = haltonBase(fi, 2);
  const u2 = haltonBase(fi, 3);
  const u3 = haltonBase(fi, 5);

  // Shoemake quaternion form (uniform distribution on SO(3)).
  const sigma1 = Math.sqrt(1 - u1);
  const sigma2 = Math.sqrt(u1);
  const theta1 = 2 * Math.PI * u2;
  const theta2 = 2 * Math.PI * u3;
  const qw = sigma2 * Math.cos(theta2);
  const qx = sigma1 * Math.sin(theta1);
  const qy = sigma1 * Math.cos(theta1);
  const qz = sigma2 * Math.sin(theta2);

  // Convert quaternion → axis-angle vec3 (WGSL consumer applies this as
  // a Rodrigues rotation to each ray direction).
  const angle = 2 * Math.acos(Math.min(1, Math.abs(qw)));
  const sinHalf = Math.sqrt(Math.max(0, 1 - qw * qw));
  let ax: number, ay: number, az: number;
  if (sinHalf < 1e-6) {
    ax = 1;
    ay = 0;
    az = 0; // identity — no rotation
  } else {
    ax = qx / sinHalf;
    ay = qy / sinHalf;
    az = qz / sinHalf;
  }
  return [ax * angle, ay * angle, az * angle];
}
