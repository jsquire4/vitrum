// Scale-derived defaults and backend selection for createEngine().

import type { Vec3 } from '@vitrum/core';

export type EnginePreference = 'realtime' | 'quality' | 'quality-webgpu' | 'auto';

/** Threshold above which 'auto' falls back from walkaround-hybrid to a PT backend. */
export const AUTO_REALTIME_TRIANGLE_BUDGET = 500_000;

export const DEFAULT_PRIMARY_LIGHT_DIR: Vec3 = Object.freeze([0.3, -0.7, 0.6]);
export const DEFAULT_PRIMARY_LIGHT_INTENSITY = 1.0;
export const DEFAULT_SKY_TINT: Vec3 = Object.freeze([0.5, 0.7, 1.0]);
export const DEFAULT_SKY_IRRADIANCE = 0.3;

export interface ScaleDefaults {
  readonly cameraMoveResetThresholdSq: number;
  readonly temporalAccumAlpha: number;
  readonly emitterDist2Floor: number;
  readonly triIntersectEpsilon: number;
}

export function deriveScaleDefaults(D: number): ScaleDefaults {
  return {
    cameraMoveResetThresholdSq: (D * 1e-3) ** 2,
    temporalAccumAlpha: 0.01,
    emitterDist2Floor: (D * 1e-4) ** 2,
    triIntersectEpsilon: D * 1e-6,
  };
}

export function pickBackend(
  prefer: EnginePreference,
  hasWebGPU: boolean,
  triangleCount: number,
  needsTlas = false,
): 'walkaround-hybrid' | 'pt-webgl2' | 'pt-webgpu' {
  if (prefer === 'quality-webgpu') return hasWebGPU ? 'pt-webgpu' : 'pt-webgl2';
  if (prefer === 'quality') {
    if (needsTlas && hasWebGPU) return 'pt-webgpu';
    return 'pt-webgl2';
  }
  if (prefer === 'realtime') {
    if (!hasWebGPU) return 'pt-webgl2';
    return 'walkaround-hybrid';
  }
  if (hasWebGPU && triangleCount < AUTO_REALTIME_TRIANGLE_BUDGET) {
    return 'walkaround-hybrid';
  }
  if (hasWebGPU) {
    return 'pt-webgpu';
  }
  if (needsTlas) {
    // WebGL-only host: merged BVH is the only pt-webgl2 path; caller should warn.
    return 'pt-webgl2';
  }
  return 'pt-webgl2';
}
