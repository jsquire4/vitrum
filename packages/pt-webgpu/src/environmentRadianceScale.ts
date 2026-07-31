/**
 * Canonical finite-binary32 policy for pt-webgpu environment radiance.
 *
 * Every shader path stages the authored product in this order:
 *   texel radiance × global HDRI intensity × optional D65 spectral expansion
 *   × receiver material intensity.
 *
 * A single channel underflowing to zero is a valid IEEE-754 result. Invalid
 * inputs and any NaN/Infinity product fail the complete RGB stage closed.
 */
import {
  canonicalizeEnvironmentRotationF32,
  HERO_D65_MAX_NORMALISED_F32,
} from '@vitrum/shared-samplers';

export type PtWebgpuEnvironmentRgb = readonly [number, number, number];

/** Maximum binary32 D65 lane expansion used by spectralEmissionAtHero. */
export const PT_WEBGPU_MAX_D65_SPECTRAL_EXPANSION_F32 =
  HERO_D65_MAX_NORMALISED_F32;

export const PT_WEBGPU_ENVIRONMENT_RADIANCE_SCALE_WGSL = /* wgsl */ `
fn ptScaleEnvironmentRadiance(value: vec3f, scale: f32) -> vec3f {
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

/** CPU mirror used by exact numeric tests and CPU transport oracles. */
export function scalePtWebgpuEnvironmentRadianceF32(
  value: PtWebgpuEnvironmentRgb,
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
export function stagedPtWebgpuEnvironmentRadianceF32(
  texel: PtWebgpuEnvironmentRgb,
  globalIntensity: number,
  materialIntensity: number,
): [number, number, number] {
  return scalePtWebgpuEnvironmentRadianceF32(
    scalePtWebgpuEnvironmentRadianceF32(texel, globalIntensity),
    materialIntensity,
  );
}

/**
 * Validate a non-negative scalar immediately before binary32 publication.
 * Authored zero stays zero; a positive scalar may neither overflow nor collapse.
 */
export function assertPtWebgpuEnvironmentScaleF32(
  value: number,
  label: string,
): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(
      `[vitrum/pt-webgpu] ${label} must be finite and non-negative; received ${String(value)}.`,
    );
  }
  const packed = Math.fround(value);
  if (!Number.isFinite(packed) || (value > 0 && packed === 0)) {
    throw new RangeError(
      `[vitrum/pt-webgpu] ${label} must remain finite and positive after Float32 packing ` +
        `when non-zero; received ${String(value)}.`,
    );
  }
  return packed === 0 ? 0 : packed;
}

/** Canonical finite angle written to every pt-webgpu environment f32 lane. */
export function packPtWebgpuEnvironmentRotationF32(
  value: number,
  label = 'HDRI rotationY',
): number {
  return canonicalizeEnvironmentRotationF32(
    value,
    `[vitrum/pt-webgpu] ${label}`,
  );
}

/**
 * Reject a scene-level final receiver stage that cannot be represented in
 * binary32. Individual channel underflow remains valid; only complete positive
 * environment collapse for a positive material scale is rejected.
 */
export function assertPtWebgpuEnvironmentMaterialEnvelopeF32(
  rgba: Float32Array,
  globalIntensity: number,
  materialIntensities: readonly number[],
  spectralEnabled = false,
): void {
  const packedGlobal = assertPtWebgpuEnvironmentScaleF32(
    globalIntensity,
    'HDRI intensity',
  );
  let maxGlobalRadiance = 0;
  for (let offset = 0; offset + 2 < rgba.length; offset += 4) {
    const rgb: [number, number, number] = [
      rgba[offset] ?? 0,
      rgba[offset + 1] ?? 0,
      rgba[offset + 2] ?? 0,
    ];
    for (let channel = 0; channel < 3; channel += 1) {
      const product = Math.fround(rgb[channel]! * packedGlobal);
      if (!Number.isFinite(product)) {
        throw new RangeError(
          `[vitrum/pt-webgpu] HDRI texel ${offset / 4} multiplied by its ` +
            'global and material intensity envelope must remain finite in Float32.',
        );
      }
      maxGlobalRadiance = Math.max(maxGlobalRadiance, product);
    }
  }
  const maxReceiverInput = spectralEnabled
    ? Math.fround(
        maxGlobalRadiance *
          PT_WEBGPU_MAX_D65_SPECTRAL_EXPANSION_F32,
      )
    : maxGlobalRadiance;
  if (!Number.isFinite(maxReceiverInput)) {
    throw new RangeError(
      '[vitrum/pt-webgpu] globally-scaled HDRI radiance exceeds Float32 ' +
        'range in the spectral D65 emission stage.',
    );
  }

  for (let index = 0; index < materialIntensities.length; index += 1) {
    const materialIntensity = assertPtWebgpuEnvironmentScaleF32(
      materialIntensities[index] ?? 1,
      `material ${index} envMapIntensity`,
    );
    const maxFinalRadiance = Math.fround(
      maxReceiverInput * materialIntensity,
    );
    if (!Number.isFinite(maxFinalRadiance)) {
      throw new RangeError(
        `[vitrum/pt-webgpu] material ${index} envMapIntensity makes the ` +
          'globally-scaled environment exceed Float32 range.',
      );
    }
    if (
      materialIntensity > 0 &&
      maxReceiverInput > 0 &&
      maxFinalRadiance === 0
    ) {
      throw new RangeError(
        `[vitrum/pt-webgpu] material ${index} envMapIntensity makes the positive ` +
          'globally-scaled environment underflow entirely to zero in Float32.',
      );
    }
  }
}
