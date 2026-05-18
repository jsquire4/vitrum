import type { Scene } from '@vitrum/core';

interface EnvironmentParams {
  readonly tint: readonly [number, number, number];
  readonly sunDirection: readonly [number, number, number];
  readonly sunStrength: number;
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
  if (
    Number.isFinite(width) &&
    Number.isFinite(height) &&
    width > 0 &&
    height > 0 &&
    data != null &&
    typeof data.length === 'number' &&
    data.length >= width * height * 3
  ) {
    const pixelCount = width * height;
    const texels = new Float32Array(pixelCount * 4);
    const cdf = new Float32Array(pixelCount + 1);
    let totalWeight = 0;
    for (let i = 0; i < pixelCount; i += 1) {
      const r = Number(data[i * 3] ?? 0);
      const g = Number(data[i * 3 + 1] ?? 0);
      const b = Number(data[i * 3 + 2] ?? 0);
      texels[i * 4] = r;
      texels[i * 4 + 1] = g;
      texels[i * 4 + 2] = b;
      const y = (i / width) | 0;
      const theta = ((y + 0.5) / height) * Math.PI;
      const weight = Math.max(0, (0.2126 * r + 0.7152 * g + 0.0722 * b) * Math.sin(theta));
      totalWeight += weight;
      cdf[i + 1] = totalWeight;
    }
    if (totalWeight > 1e-12) {
      const dOmegaBase = (2 * Math.PI / width) * (Math.PI / height);
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
      return {
        tint: [1, 1, 1],
        sunDirection: [0, 1, 0],
        sunStrength: scene.environment.intensity ?? 1,
        hdriWidth: width,
        hdriHeight: height,
        hasHdri: true,
        hdriTexels: texels,
        hdriCdf: cdf,
        warnings: [],
      };
    }
  }
  return {
    tint: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunStrength: 0,
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
