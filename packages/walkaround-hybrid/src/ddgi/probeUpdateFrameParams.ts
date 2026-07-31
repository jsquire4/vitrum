/**
 * Per-frame probe-update FrameParams packing (Halton-Shoemake SO(3)).
 */
import { haltonSO3AxisAngleFromFrameIndex } from '@vitrum/shared-samplers';
import { DDGI_BLEND_PARAMS_UBO, DDGI_FRAME_PARAMS_UBO } from './probeUpdateUbos.js';
import {
  assertDdgiBoolean,
  assertDdgiUnitInterval,
  assertFiniteDdgiVec3,
  assertNonNegativeDdgiNumber,
} from './inputValidation.js';
import {
  assertWalkaroundEnvironmentRgbScaleEnvelopeF32,
  assertWalkaroundEnvironmentScaleF32,
  packWalkaroundEnvironmentRotationF32,
} from '../environment/environmentRadianceScale.js';
export { haltonSO3AxisAngleFromFrameIndex };

export interface ProbeUpdateFrameParamsInput {
  frameIndex: number;
  skyTint: readonly [number, number, number];
  skyIrradiance: number;
  glassMixScale: number;
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

function assertDdgiUint32(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be a uint32 integer.`);
  }
}

export function packProbeUpdateFrameParams(input: ProbeUpdateFrameParamsInput): ArrayBuffer {
  assertDdgiUint32(input.frameIndex, 'DDGI frame index');
  assertFiniteDdgiVec3(input.skyTint, 'DDGI sky tint');
  input.skyTint.forEach((channel, index) => {
    assertNonNegativeDdgiNumber(channel, `DDGI sky tint[${index}]`);
  });
  assertNonNegativeDdgiNumber(input.skyIrradiance, 'DDGI sky irradiance');
  assertDdgiUnitInterval(input.glassMixScale, 'DDGI glass mix scale');
  if (input.indirectFeedback !== undefined) {
    assertDdgiBoolean(input.indirectFeedback, 'DDGI indirect feedback');
  }
  if (input.hasEnv !== undefined) {
    assertDdgiBoolean(input.hasEnv, 'DDGI environment hasEnv');
  }
  const envRotationY = packWalkaroundEnvironmentRotationF32(
    input.envRotationY ?? 0,
    'DDGI environment rotation',
  );
  const envIntensity = assertWalkaroundEnvironmentScaleF32(
    input.envIntensity ?? 0,
    'DDGI environment intensity',
  );
  const packedSky = assertWalkaroundEnvironmentRgbScaleEnvelopeF32(
    input.skyTint,
    input.skyIrradiance,
    'DDGI scalar-sky radiance',
  );
  const data = new ArrayBuffer(DDGI_FRAME_PARAMS_UBO.sizeBytes);
  DDGI_FRAME_PARAMS_UBO.pack(new DataView(data), 0, {
    randomRotation: haltonSO3AxisAngleFromFrameIndex(input.frameIndex),
    frameIndex: input.frameIndex,
    skyTint: packedSky.value,
    skyIrradiance: packedSky.scale,
    glassMixScale: input.glassMixScale,
    // H46-A — default true (multi-bounce EMA) preserves the historical
    // behaviour byte-for-byte (the old _pad2 wrote 0; here `true` writes 1,
    // which is the value the gated `select` treats as "fold indirect", i.e.
    // identical radiance to the pre-gate `direct + indirect`). Host sets false
    // for maxBounces == 1.
    indirectFeedback: (input.indirectFeedback ?? true) ? 1 : 0,
    // Wave 4 — HDRI into DDGI probe misses (2026-06-10).
    hasEnv: (input.hasEnv ?? false) ? 1 : 0,
    envRotationY,
    envIntensity,
  });
  return data;
}

export const DDGI_PROBE_BLEND_HYSTERESIS = 0.97;

/**
 * Pack the blend-params UBO.
 *
 * @param hysteresisOverride - When provided, overrides the steady-state 0.97.
 *   Pass `0.0` for a full-replace blend (H16 invalidate path): EMA weight = 0
 *   means `newValue = (1 − 0) × freshSample + 0 × history = freshSample`,
 *   clearing stale atlas data in one probe-update cycle.
 */
export function packProbeUpdateBlendParams(
  hysteresisOverride?: number,
): ArrayBuffer {
  const hysteresis = hysteresisOverride ?? DDGI_PROBE_BLEND_HYSTERESIS;
  assertDdgiUnitInterval(hysteresis, 'DDGI blend hysteresis');
  const data = new ArrayBuffer(DDGI_BLEND_PARAMS_UBO.sizeBytes);
  DDGI_BLEND_PARAMS_UBO.pack(new DataView(data), 0, {
    hysteresis,
  });
  return data;
}
