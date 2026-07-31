/**
 * Canonical finite-binary32 policy for walkaround environment radiance.
 *
 * Per-channel underflow to zero is legitimate. Invalid inputs and any
 * NaN/Infinity product fail the complete RGB stage closed.
 */
import { canonicalizeEnvironmentRotationF32 } from '@vitrum/shared-samplers';

export type WalkaroundEnvironmentRgb = readonly [number, number, number];

export const WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL = /* wgsl */ `
fn walkaroundScaleEnvironmentRadiance(value: vec3f, scale: f32) -> vec3f {
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

/** CPU mirror used by exact numeric tests and host product-envelope checks. */
export function scaleWalkaroundEnvironmentRadianceF32(
  value: WalkaroundEnvironmentRgb,
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

/** CPU mirror of the two-stage global-then-material shader product. */
export function stagedWalkaroundEnvironmentRadianceF32(
  texel: WalkaroundEnvironmentRgb,
  globalIntensity: number,
  materialIntensity: number,
): [number, number, number] {
  return scaleWalkaroundEnvironmentRadianceF32(
    scaleWalkaroundEnvironmentRadianceF32(texel, globalIntensity),
    materialIntensity,
  );
}

/**
 * Validate a non-negative scalar immediately before binary32 publication.
 * Authored zero stays zero; a positive scalar may neither overflow nor collapse.
 */
export function assertWalkaroundEnvironmentScaleF32(
  value: number,
  label: string,
): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] ${label} must be finite and non-negative; ` +
        `received ${String(value)}.`,
    );
  }
  const packed = Math.fround(value);
  if (!Number.isFinite(packed) || (value > 0 && packed === 0)) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] ${label} must remain finite and positive after ` +
        `Float32 packing when non-zero; received ${String(value)}.`,
    );
  }
  return packed === 0 ? 0 : packed;
}

/**
 * Publish a scalar radiance product with the same operand-by-operand binary32
 * staging used by WGSL. Positive inputs may not overflow or collapse to zero.
 */
export function stageWalkaroundEnvironmentScaleProductF32(
  value: number,
  scale: number,
  label: string,
): number {
  const packedValue = assertWalkaroundEnvironmentScaleF32(
    value,
    `${label} value`,
  );
  const packedScale = assertWalkaroundEnvironmentScaleF32(
    scale,
    `${label} scale`,
  );
  const product = Math.fround(packedValue * packedScale);
  if (!Number.isFinite(product)) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] ${label} product must remain finite in Float32.`,
    );
  }
  if (packedValue > 0 && packedScale > 0 && product === 0) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] positive ${label} product underflows to zero ` +
        'in Float32.',
    );
  }
  return product === 0 ? 0 : product;
}

/** Canonical finite angle written to every walkaround/DDGI environment f32 lane. */
export function packWalkaroundEnvironmentRotationF32(
  value: number,
  label = 'HDRI rotationY',
): number {
  return canonicalizeEnvironmentRotationF32(
    value,
    `[vitrum/walkaround-hybrid] ${label}`,
  );
}

/** Validate and return one RGB-radiance × scalar publication envelope. */
export function assertWalkaroundEnvironmentRgbScaleEnvelopeF32(
  value: WalkaroundEnvironmentRgb,
  scale: number,
  label: string,
): {
  readonly value: [number, number, number];
  readonly scale: number;
  readonly scaled: [number, number, number];
} {
  const packedValue = value.map((channel, index) =>
    assertWalkaroundEnvironmentScaleF32(
      channel,
      `${label}[${index}]`,
    )
  ) as [number, number, number];
  const packedScale = assertWalkaroundEnvironmentScaleF32(scale, label);
  const scaled = packedValue.map((channel) =>
    Math.fround(channel * packedScale)
  ) as [number, number, number];
  if (scaled.some((channel) => !Number.isFinite(channel))) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] ${label} product must remain finite in Float32.`,
    );
  }
  if (
    packedScale > 0 &&
    packedValue.some((channel) => channel > 0) &&
    scaled.every((channel) => channel === 0)
  ) {
    throw new RangeError(
      `[vitrum/walkaround-hybrid] positive ${label} product underflows entirely ` +
        'to zero in Float32.',
    );
  }
  return { value: packedValue, scale: packedScale, scaled };
}

/**
 * Validate the exact RGB map × global-intensity publication envelope.
 *
 * Individual positive channels may underflow. A positive source map with a
 * positive intensity may not collapse in its entirety, because that would make
 * a visibly non-black environment unreachable after upload.
 */
export function assertWalkaroundEnvironmentMapScaleEnvelopeF32(
  rgba: Float32Array,
  intensity: number,
): number {
  const packedIntensity = assertWalkaroundEnvironmentScaleF32(
    intensity,
    'environment intensity',
  );
  let hasPositiveSource = false;
  let hasPositiveScaledChannel = false;
  for (let offset = 0; offset < rgba.length; offset += 4) {
    const rgb: [number, number, number] = [
      rgba[offset] ?? 0,
      rgba[offset + 1] ?? 0,
      rgba[offset + 2] ?? 0,
    ];
    if (rgb.some((channel) => channel < 0 || !Number.isFinite(channel))) {
      throw new RangeError(
        `[vitrum/walkaround-hybrid] environment texel ${offset / 4} ` +
          'must contain finite, non-negative Float32 radiance.',
      );
    }
    for (let channel = 0; channel < 3; channel += 1) {
      const source = rgb[channel]!;
      const product = Math.fround(source * packedIntensity);
      if (!Number.isFinite(product)) {
        throw new RangeError(
          `[vitrum/walkaround-hybrid] environment texel ${offset / 4} ` +
            'multiplied by its intensity must remain finite in Float32.',
        );
      }
      hasPositiveSource ||= source > 0;
      hasPositiveScaledChannel ||= product > 0;
    }
  }
  if (
    packedIntensity > 0 &&
    hasPositiveSource &&
    !hasPositiveScaledChannel
  ) {
    throw new RangeError(
      '[vitrum/walkaround-hybrid] positive environment radiance multiplied by its ' +
        'positive intensity underflows entirely to zero in Float32.',
    );
  }
  return packedIntensity;
}

/**
 * Validate the final globally-scaled environment × receiver-material stage.
 * Every overflow is rejected. Individual channel underflow is accepted, while
 * complete positive-map collapse for a positive material scale is not.
 */
export function assertWalkaroundEnvironmentMaterialEnvelopeF32(
  rgba: Float32Array,
  globalIntensity: number,
  materialIntensities: readonly number[],
): void {
  const packedGlobal =
    assertWalkaroundEnvironmentMapScaleEnvelopeF32(rgba, globalIntensity);
  let maxGlobalRadiance = 0;
  for (let offset = 0; offset + 2 < rgba.length; offset += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const product = Math.fround(
        (rgba[offset + channel] ?? 0) * packedGlobal,
      );
      maxGlobalRadiance = Math.max(maxGlobalRadiance, product);
    }
  }
  for (let index = 0; index < materialIntensities.length; index += 1) {
    const materialIntensity = assertWalkaroundEnvironmentScaleF32(
      materialIntensities[index] ?? 1,
      `material ${index} envMapIntensity`,
    );
    const maxFinalRadiance = Math.fround(
      maxGlobalRadiance * materialIntensity,
    );
    if (!Number.isFinite(maxFinalRadiance)) {
      throw new RangeError(
        `[vitrum/walkaround-hybrid] material ${index} envMapIntensity makes ` +
          'the globally-scaled environment exceed Float32 range.',
      );
    }
    if (
      materialIntensity > 0 &&
      maxGlobalRadiance > 0 &&
      maxFinalRadiance === 0
    ) {
      throw new RangeError(
        `[vitrum/walkaround-hybrid] material ${index} envMapIntensity makes the ` +
          'positive globally-scaled environment underflow entirely to zero in Float32.',
      );
    }
  }
}
