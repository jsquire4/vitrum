// frameUniformsPacker — pure builder of the per-frame `FrameUniforms` payload for
// the pt-webgl2 accumulation draw (extracted from index.ts #frameUniforms, T3-D /
// D11-1). The caller supplies its config + scene state as a plain record so this
// module stays free of the engine class.

import type { FrameInput, Scene } from '@vitrum/core';
import type { FrameUniforms } from './glResources.js';
import type { PTEngineWebGL2Options } from '../options.js';
import { TONEMAP_MODE_INDEX } from '@vitrum/shared-samplers';
import { invertMat4, makeRotationYMat4 } from '../mat4.js';

import { sharedBdptWavelengthForSeed } from './sharedBdptWavelength.js';
const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** Engine-config + scene-state slice consumed by `packFrameUniforms`. */
export interface FrameUniformsConfig {
  readonly scene: Scene | null;
  readonly hasEnvMap: boolean;
  readonly materialLodDepth: number;
  readonly backgroundBlur: number;
  readonly spectralEnabled: boolean;
  readonly backgroundAlpha: number;
  readonly bdpt: boolean;
  readonly bdptMaxLightBounces: number;
  readonly bdptSceneCenter: readonly [number, number, number];
  readonly bdptSceneRadius: number;
  readonly dof: PTEngineWebGL2Options['dof'];
}

/**
 * Pack the per-frame `FrameUniforms` for the accumulation draw. Throws on a
 * singular view/projection matrix.
 */
export function packFrameUniforms(
  input: FrameInput,
  bounces: number,
  w: number,
  h: number,
  cfg: FrameUniformsConfig,
  accumulatedSample = 0,
): FrameUniforms {
  const cameraWorldMatrix = invertMat4(input.viewMatrix);
  const invProjectionMatrix = invertMat4(input.projMatrix);
  if (cameraWorldMatrix == null || invProjectionMatrix == null) {
    throw new Error('renderFrame: singular view/projection matrix');
  }
  // H6 FIX (2026-06-09): honour the HDRI environment's `intensity` contract field
  // (was hardcoded to 1, so `environment.intensity` was silently ignored).
  // Mirrors pt-webgpu (environmentPacking.ts:54: `env.intensity ?? 1`).
  const env = cfg.scene?.environment;
  const envIntensity = env != null && env.kind === 'hdri' ? env.intensity ?? 1 : 1;
  const sharedBdptWavelength = sharedBdptWavelengthForSeed(
    input.frameSeed,
    accumulatedSample,
  );
  return {
    resolution: [w, h],
    bounces,
    transmissiveBounces: bounces,
    filterGlossyFactor: input.quality?.filteredGlossyFactor ?? 0,
    materialLodDepth: cfg.materialLodDepth,
    cameraWorldMatrix,
    invProjectionMatrix,
    environmentIntensity: cfg.hasEnvMap ? envIntensity : 0,
    // H6 FIX (2026-06-09): honour HdriEnvironment.rotationY (CCW env dome rotation
    // around +Y, radians).  Convention: a world-space direction `d` looks up the
    // UNROTATED map at `RY(−rotationY) * d`, so the uniform matrix is
    // makeRotationYMat4(−rotationY).  The GLSL then evaluates:
    //   envRotation3x3 = mat3(environmentRotation)   → RY(−rotationY)
    //   lookupDir      = envRotation3x3 * worldDir   → RY(−rotationY) * d ✓
    // rotationY = 0 → identity → byte-identical to the pre-H6 IDENTITY_MAT4 path.
    environmentRotation: (env?.kind === 'hdri' && env.rotationY != null && env.rotationY !== 0)
      ? makeRotationYMat4(-(env.rotationY))
      : IDENTITY_MAT4,
    backgroundBlur: cfg.backgroundBlur,
    spectralEnabled: cfg.spectralEnabled,
    backgroundAlpha: cfg.backgroundAlpha,
    // A5 — BDPT host-driver inputs (no-op when bdpt:false).
    bdpt: cfg.bdpt,
    bdptMaxLightBounces: cfg.bdptMaxLightBounces,
    bdptSceneCenter: cfg.bdptSceneCenter,
    bdptSceneRadius: cfg.bdptSceneRadius,
    bdptSharedWavelengthNm: sharedBdptWavelength.wavelengthNm,
    bdptSharedWavelengthPdf: sharedBdptWavelength.pdf,
    dof:
      cfg.dof != null
        ? {
            focusDistance: cfg.dof.focusDistance,
            bokehSize: cfg.dof.bokehSize,
            apertureBlades: cfg.dof.apertureBlades ?? 0,
            apertureRotation: cfg.dof.apertureRotation ?? 0,
            anamorphicRatio: cfg.dof.anamorphicRatio ?? 1,
          }
        : null,
    // ── Tonemap / present-pass dials (2026-06-10) ─────────────────────────
    // Matches the contract (FrameQualitySettings) and the walkaround-hybrid
    // orchestrator wiring (HybridEngineFrameOrchestrator.ts:764).
    // Default: aces(0) @ 1.0 @ srgb(0) — same as walkaround and the contract.
    //
    // CONTRACT-DEFAULT TENSION: pt-webgl2 previously returned raw linear HDR
    // (no present pass).  Adding the present pass with default aces+srgb
    // changes the default visual output.  Hosts that relied on the raw HDR
    // should pass quality.tonemap='none' + quality.outputColorSpace='linear'.
    tonemapMode:      TONEMAP_MODE_INDEX[input.quality?.tonemap ?? 'aces'],
    exposure:         input.quality?.exposure ?? 1.0,
    outputColorSpace: input.quality?.outputColorSpace === 'linear' ? 1 : 0,
  };
}
