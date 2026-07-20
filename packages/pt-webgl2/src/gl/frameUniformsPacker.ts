// frameUniformsPacker — pure builder of the per-frame `FrameUniforms` payload for
// the pt-webgl2 accumulation draw (extracted from index.ts #frameUniforms, T3-D /
// D11-1). BEHAVIOR-PRESERVING: the returned FrameUniforms object is byte-identical
// to the pre-extraction inline construction; the caller supplies its config +
// scene state as a plain record so this module stays free of the engine class.

import type { FrameInput, Scene } from '@vitrum/core';
import type { FrameUniforms } from './glResources.js';
import type { PTEngineWebGL2Options } from '../options.js';
import { CAUCHY_CROWN_GLASS, TONEMAP_MODE_INDEX } from '@vitrum/shared-samplers';
import { invertMat4, makeRotationYMat4 } from '../mat4.js';

const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

// H2 follow-on — scene-global spectral coefficients (the GLSL declares u_jakobCoeffs +
// iorCauchyA/B/C as global uniforms, not per-material).
//   • iorCauchy: Crown Glass three-term Cauchy IOR (n(λ)). Uploaded only when
//     spectral is on so any material carrying `dispersionAbbeNumber` (→ per-material
//     dispersionStrength in the materials texture) actually disperses. All-zero =
//     the GLSL `cauchyEnabled` fast-path (no dispersion), which is the non-spectral
//     and the no-dispersion default → byte-identical when spectral:false.
//   • jakobCoeffs: stays the flat (0,0,0) ⇒ S≡½ no-op. u_jakobCoeffs is a SINGLE
//     global reflectance the GLSL uses only for representative MEDIUM albedo
//     (volume single-scatter / SSS); baseColor reflectance is per-material via
//     the materials texture, solved once at scene build.
const SPECTRAL_IOR_CAUCHY: readonly [number, number, number] = [
  CAUCHY_CROWN_GLASS.A,
  CAUCHY_CROWN_GLASS.B,
  CAUCHY_CROWN_GLASS.C,
];
const FLAT_JAKOB_COEFFS: readonly [number, number, number] = [0, 0, 0];
const NO_IOR_CAUCHY: readonly [number, number, number] = [0, 0, 0];

/** Engine-config + scene-state slice consumed by `packFrameUniforms`. */
export interface FrameUniformsConfig {
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map' | undefined;
  readonly scene: Scene | null;
  readonly hasEnvMap: boolean;
  readonly materialLodDepth: number;
  readonly backgroundBlur: number;
  readonly spectralEnabled: boolean;
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
  readonly backgroundAlpha: number;
  readonly bdpt: boolean;
  readonly bdptMaxLightBounces: number;
  readonly dof: PTEngineWebGL2Options['dof'];
}

/**
 * Pack the per-frame `FrameUniforms` for the accumulation draw. Byte-identical to
 * the pre-extraction `#frameUniforms` method (index.ts). Throws on a singular
 * view/projection matrix (same as before).
 */
export function packFrameUniforms(
  input: FrameInput,
  bounces: number,
  w: number,
  h: number,
  cfg: FrameUniformsConfig,
): FrameUniforms {
  const cameraWorldMatrix = invertMat4(input.viewMatrix);
  const invProjectionMatrix = invertMat4(input.projMatrix);
  if (cameraWorldMatrix == null || invProjectionMatrix == null) {
    throw new Error('renderFrame: singular view/projection matrix');
  }
  const caustic =
    cfg.causticStrategy === 'manifold-nee' ? 1 : cfg.causticStrategy === 'photon-map' ? 2 : 0;
  // H6 FIX (2026-06-09): honour the HDRI environment's `intensity` contract field
  // (was hardcoded to 1, so `environment.intensity` was silently ignored).
  // Mirrors pt-webgpu (environmentPacking.ts:54: `env.intensity ?? 1`).
  const env = cfg.scene?.environment;
  const envIntensity = env != null && env.kind === 'hdri' ? env.intensity ?? 1 : 1;
  return {
    resolution: [w, h],
    bounces,
    transmissiveBounces: bounces,
    filterGlossyFactor: input.quality?.filteredGlossyFactor ?? 0,
    materialLodDepth: cfg.materialLodDepth,
    radianceClamp: 0,
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
    causticStrategy: caustic,
    mneeMaxIterations: cfg.mneeMaxIterations,
    mneeMaxChainLength: cfg.mneeMaxChainLength,
    backgroundAlpha: cfg.backgroundAlpha,
    // A5 — BDPT host-driver inputs (no-op when bdpt:false).
    bdpt: cfg.bdpt,
    bdptMaxLightBounces: cfg.bdptMaxLightBounces,
    // H2 follow-on — global spectral coefficients. Cauchy IOR only when spectral is
    // on (else the no-dispersion fast path → byte-identical); Jakob stays flat.
    iorCauchy: cfg.spectralEnabled ? SPECTRAL_IOR_CAUCHY : NO_IOR_CAUCHY,
    jakobCoeffs: FLAT_JAKOB_COEFFS,
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
