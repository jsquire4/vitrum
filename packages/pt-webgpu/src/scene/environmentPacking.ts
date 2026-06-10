import type { Scene } from '@vitrum/core';
import { luminance } from '@vitrum/shared-samplers';

interface EnvironmentParams {
  readonly tint: readonly [number, number, number];
  readonly sunDirection: readonly [number, number, number];
  readonly sunStrength: number;
  /**
   * H14-E: HDRI radiance intensity multiplier, separate from `sunStrength`.
   * `scene.environment.intensity ?? 1` when a valid HDRI is present; 0 otherwise.
   * Uploaded to `params.environmentHdriIntensity` so the equirect lookup is NOT
   * gated by the procedural-sky sun strength.
   */
  readonly hdriIntensity: number;
  /**
   * H6: HDRI environment rotation around the world +Y axis, in radians (default 0).
   * Uploaded to `params.environmentTint.w` (the previously-zero .w lane of
   * environmentTint — no layout change).  The WGSL lookups apply rotateYNeg by this
   * value before UV indexing, and rotateYPos after CDF sampling.
   * Non-HDRI environments keep this at 0 (rotationY does not apply to procedural sky).
   */
  readonly hdriRotationY: number;
  readonly hdriWidth: number;
  readonly hdriHeight: number;
  readonly hasHdri: boolean;
  readonly hdriTexels: Float32Array;
  readonly hdriCdf: Float32Array;
  readonly warnings: readonly string[];
}

/** Empty / no-environment slot — neutral tint, no HDRI sampling.
 *  File-local — only consumed by `environmentParams()` below; 2026-05-18
 *  dead-code sweep verified zero external consumers. */
function emptyEnvironmentParams(): EnvironmentParams {
  return {
    tint: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunStrength: 0,
    hdriIntensity: 0,
    hdriRotationY: 0,
    hdriWidth: 0,
    hdriHeight: 0,
    hasHdri: false,
    hdriTexels: new Float32Array(0),
    hdriCdf: new Float32Array(0),
    warnings: [],
  };
}

/**
 * Build environment params from a procedural-sky scene environment.
 *
 * The RGB tint is heuristic, not physically derived:
 *  - Base tint is `[0.9, 0.95, 1.0]` (Rayleigh-skewed blue, slightly green-
 *    weighted to mimic noon sky chromaticity).
 *  - `mieCoefficient * 10` is subtracted from the red channel (mie scattering
 *    biases the dome toward warmer hues by reducing blue dominance, so we
 *    pull red DOWN to keep the dome blue-ish).
 *  - Floor of 0.2 prevents the tint from going to black for huge mie values.
 *  - All channels multiply by `intensity`, the scene-level dome brightness.
 */
// File-local — only consumed by `environmentParams()` below; 2026-05-18
// dead-code sweep verified zero external consumers.
function buildProceduralSkyEnvironmentParams(
  env: Extract<Scene['environment'], { kind: 'procedural-sky' }>,
): EnvironmentParams {
  const d = env.sunDirection;
  const len = Math.hypot(d[0], d[1], d[2]);
  const sunDir: readonly [number, number, number] =
    len < 1e-8 ? [0, 1, 0] : [d[0] / len, d[1] / len, d[2] / len];
  const intensity = env.intensity ?? 1;
  const tintBoost = Math.max(0.2, 1 - env.mieCoefficient * 10);
  return {
    tint: [0.9 * tintBoost * intensity, 0.95 * intensity, 1.0 * intensity],
    sunDirection: sunDir,
    sunStrength: Math.max(0, intensity),
    hdriIntensity: 0,     // No HDRI present for procedural-sky
    hdriRotationY: 0,     // rotationY applies to HDRI only; no-op for procedural-sky
    hdriWidth: 0,
    hdriHeight: 0,
    hasHdri: false,
    hdriTexels: new Float32Array(0),
    hdriCdf: new Float32Array(0),
    warnings: [],
  };
}

export function environmentParams(scene: Scene): EnvironmentParams {
  if (scene.environment.kind === 'none') {
    return emptyEnvironmentParams();
  }
  if (scene.environment.kind === 'procedural-sky') {
    return buildProceduralSkyEnvironmentParams(scene.environment);
  }
  type HdriLike = { width?: number; height?: number; data?: ArrayLike<number> };
  const hdri = scene.environment.hdri as HdriLike;
  const width = Number(hdri?.width ?? 0);
  const height = Number(hdri?.height ?? 0);
  const data = hdri?.data;
  const pixelCount = width * height;
  const hasRgb =
    pixelCount > 0 &&
    data != null &&
    typeof data.length === 'number' &&
    data.length >= pixelCount * 3;
  if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0 && hasRgb) {
    // Detect whether the caller provided RGBA (stride 4) or RGB (stride 3) data.
    // A w·h·4-length buffer passes the >= w·h·3 gate but must be decoded at
    // stride 4 or every pixel after the first will read from the wrong offset.
    const isRgba = data!.length >= pixelCount * 4;
    const stride = isRgba ? 4 : 3;
    // Warn once on ambiguous / unexpected RGBA input so the host is aware of the
    // implicit stride detection (there is no authoritative flag in the scene API).
    const warnings: string[] = [];
    if (isRgba) {
      warnings.push(
        '[vitrum/pt-webgpu] HDRI data.length matches a w×h×4 RGBA layout; ' +
          'decoding at stride 4. Pass a w×h×3 RGB array to suppress this warning.',
      );
    }
    const texels = new Float32Array(pixelCount * 4);
    const cdf = new Float32Array(pixelCount + 1);
    let totalWeight = 0;
    for (let i = 0; i < pixelCount; i += 1) {
      const r = Number(data![i * stride] ?? 0);
      const g = Number(data![i * stride + 1] ?? 0);
      const b = Number(data![i * stride + 2] ?? 0);
      texels[i * 4] = r;
      texels[i * 4 + 1] = g;
      texels[i * 4 + 2] = b;
      const y = (i / width) | 0;
      const theta = ((y + 0.5) / height) * Math.PI;
      const weight = Math.max(0, luminance(r, g, b) * Math.sin(theta));
      totalWeight += weight;
      cdf[i + 1] = totalWeight;
    }
    if (totalWeight > 1e-12) {
      const dOmegaBase = ((2 * Math.PI) / width) * (Math.PI / height);
      for (let i = 0; i < pixelCount; i += 1) {
        cdf[i + 1] = (cdf[i + 1] ?? 0) / totalWeight;
        const y = (i / width) | 0;
        const theta = ((y + 0.5) / height) * Math.PI;
        const sinTheta = Math.max(Math.sin(theta), 1e-5);
        const pmf = Math.max((cdf[i + 1] ?? 0) - (cdf[i] ?? 0), 0);
        texels[i * 4 + 3] = pmf / (dOmegaBase * sinTheta);
      }
      cdf[0] = 0;
      cdf[pixelCount] = 1;
      const hdriIntensity = scene.environment.intensity ?? 1;
      // H6: pass rotationY through; default 0 (identity rotation).
      const hdriRotationY = scene.environment.rotationY ?? 0;
      return {
        tint: [1, 1, 1],
        sunDirection: [0, 1, 0],
        // H14-E: sunStrength drives the procedural-sky sun gate (environmentSun.w).
        // For a pure-HDRI scene, set sun.w to 0 so the sky NEE branch doesn't fire.
        // The HDRI radiance uses its own hdriIntensity lane (→ params.environmentHdriIntensity).
        sunStrength: 0,
        hdriIntensity,
        hdriRotationY,
        hdriWidth: width,
        hdriHeight: height,
        hasHdri: true,
        hdriTexels: texels,
        hdriCdf: cdf,
        warnings,
      };
    }
    // All pixels are black (totalWeight ≤ 1e-12) — the HDRI data was valid but
    // has zero luminance. Report accurately rather than misattributing this to
    // missing pixel data.
    return {
      tint: [1, 1, 1],
      sunDirection: [0, 1, 0],
      sunStrength: 0,
      hdriIntensity: 0,
      hdriRotationY: 0,
      hdriWidth: 0,
      hdriHeight: 0,
      hasHdri: false,
      hdriTexels: new Float32Array(0),
      hdriCdf: new Float32Array(0),
      warnings: [
        ...warnings,
        'HDRI environment has zero total luminance (all-black or transparent pixels); falling back to procedural sky model.',
      ],
    };
  }
  return {
    tint: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunStrength: 0,
    hdriIntensity: 0,
    hdriRotationY: 0,
    hdriWidth: 0,
    hdriHeight: 0,
    hasHdri: false,
    hdriTexels: new Float32Array(0),
    hdriCdf: new Float32Array(0),
    warnings: [
      'HDRI environment lacks CPU pixel data (width/height/data); falling back to procedural sky model.',
    ],
  };
}
