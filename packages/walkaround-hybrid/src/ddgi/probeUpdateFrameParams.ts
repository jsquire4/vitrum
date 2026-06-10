/**
 * Per-frame probe-update FrameParams packing (Halton–Shoemake SO(3)).
 */
import { DDGI_BLEND_PARAMS_UBO, DDGI_FRAME_PARAMS_UBO } from './probeUpdateUbos.js';

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

/** Axis-angle vec3 for Rodrigues rotation in probeUpdateRays.wgsl. */
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

export interface ProbeUpdateFrameParamsInput {
  frameIndex: number;
  totalProbes: number;
  skyTint: readonly [number, number, number];
  skyIrradiance: number;
  glassMixScale: number;
  /** Phase-0 productization — round-robin probe-update divisor
   *  (`probesPerFrame = ceil(totalProbes / divisor)`). Default 4 reproduces
   *  the historical hardcoded `/4` 4-frame full-grid cycle. */
  updateDivisor?: number;
  /** H46-A — DDGI indirect-feedback gate. `true` (default) folds the
   *  previous-frame irradiance atlas into the bounce surface (the
   *  infinite-bounce diffuse EMA, maxBounces >= 2). `false` drops it →
   *  direct-only probes (maxBounces == 1). NOT a PT bounce cap. */
  indirectFeedback?: boolean;
  /**
   * Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
   * `true` activates the equirect env-map sample path in the WGSL
   * `sampleSkyColor` function; `false` (default) keeps the existing
   * procedural sky gradient so scenes without an HDRI are byte-identical.
   */
  hasEnv?: boolean;
  /**
   * Y-axis rotation (radians) for the equirect env lookup. Matches the H6
   * `envRotateYNeg` convention in `environmentSample.wgsl`:
   *   map-lookup-dir = RY(-envRotationY) · worldDir
   * Pass 0 when `hasEnv` is false (no-op).
   */
  envRotationY?: number;
  /**
   * Radiance intensity multiplier applied to the env-map texel after lookup.
   * Matches `envParams.intensity` in `environmentSample.wgsl`.
   * Pass 0 or omit when `hasEnv` is false (no-op).
   */
  envIntensity?: number;
}

/** Clamp the divisor to ≥ 1 so `probesPerFrame` never exceeds `totalProbes`. */
function safeDivisor(divisor: number | undefined): number {
  return Math.max(1, Math.floor(divisor ?? 4));
}

export function packProbeUpdateFrameParams(input: ProbeUpdateFrameParamsInput): ArrayBuffer {
  const data = new ArrayBuffer(DDGI_FRAME_PARAMS_UBO.sizeBytes);
  DDGI_FRAME_PARAMS_UBO.pack(new DataView(data), 0, {
    randomRotation: haltonSO3AxisAngleFromFrameIndex(input.frameIndex),
    frameIndex: input.frameIndex,
    totalProbes: input.totalProbes,
    probesPerFrame: Math.ceil(input.totalProbes / safeDivisor(input.updateDivisor)),
    _pad0: 0,
    _pad1: 0,
    skyTint: [input.skyTint[0], input.skyTint[1], input.skyTint[2]] as const,
    skyIrradiance: input.skyIrradiance,
    glassMixScale: input.glassMixScale,
    // H46-A — default true (multi-bounce EMA) preserves the historical
    // behaviour byte-for-byte (the old _pad2 wrote 0; here `true` writes 1,
    // which is the value the gated `select` treats as "fold indirect", i.e.
    // identical radiance to the pre-gate `direct + indirect`). Host sets false
    // for maxBounces == 1.
    indirectFeedback: (input.indirectFeedback ?? true) ? 1 : 0,
    // Wave 4 — HDRI into DDGI probe misses (2026-06-10).
    hasEnv: (input.hasEnv ?? false) ? 1 : 0,
    envRotationY: input.envRotationY ?? 0,
    envIntensity: input.envIntensity ?? 0,
  });
  return data;
}

export const DDGI_PROBE_BLEND_HYSTERESIS = 0.97;

/**
 * Pack the blend-params UBO.
 *
 * @param totalProbes - Total probe count for the current grid.
 * @param updateDivisor - Round-robin stride (≥ 1); default 4.
 * @param hysteresisOverride - When provided, overrides the steady-state 0.97.
 *   Pass `0.0` for a full-replace blend (H16 invalidate path): EMA weight = 0
 *   means `newValue = (1 − 0) × freshSample + 0 × history = freshSample`,
 *   clearing stale atlas data in one probe-update cycle.
 */
export function packProbeUpdateBlendParams(
  totalProbes: number,
  updateDivisor?: number,
  hysteresisOverride?: number,
): ArrayBuffer {
  const data = new ArrayBuffer(DDGI_BLEND_PARAMS_UBO.sizeBytes);
  DDGI_BLEND_PARAMS_UBO.pack(new DataView(data), 0, {
    // MUST match the ray pass's coverage (same divisor) so the blend kernel
    // only blends probes that received fresh rays this frame.
    probesPerFrame: Math.ceil(totalProbes / safeDivisor(updateDivisor)),
    hysteresis: hysteresisOverride ?? DDGI_PROBE_BLEND_HYSTERESIS,
  });
  return data;
}
