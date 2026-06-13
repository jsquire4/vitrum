import type { EngineOptions } from '@vitrum/core';
import type { WebGl2TraceTier } from './traceTier.js';

// `WebGl2TraceTier` is owned by ./traceTier.ts (the tier-selection module);
// re-exported here so consumers of the options surface keep importing it from
// one place. (Previously this module re-declared an identical union — D9 dedup.)
/** @public — public option type for pt-webgl2 consumers; controls quality tier selection. */
export type { WebGl2TraceTier };

/**
 * Options for {@link createPTEngine_WebGL2}. Narrows the core `EngineOptions.device`
 * (typed `unknown`) to a host-owned `WebGL2RenderingContext`. The engine allocates
 * GL resources against it but NEVER loses or destroys the context — the host owns
 * the lifecycle (design principle #2).
 */
export interface PTEngineWebGL2Options extends EngineOptions {
  readonly device: WebGL2RenderingContext;
  /** Force a tier; otherwise auto-selected from WebGL2 caps (see traceTier.ts). */
  readonly traceTier?: WebGl2TraceTier;
  /** Spectral hero-wavelength rendering (S3). Default false. */
  readonly spectral?: boolean;
  /** Bidirectional path tracing (S3). Default false. */
  readonly bdpt?: boolean;
  /**
   * Optional texture-fetch LOD by bounce depth. `0` (default) disables the optimization
   * and samples material textures at every bounce, preserving the historical highest-
   * fidelity behavior. A positive value samples textures only while
   * `pathDepth <= materialLodDepth`; deeper bounces use flat material constants.
   */
  readonly materialLodDepth?: number;
  /**
   * Naming note for `causticStrategy: 'manifold-nee'`:
   *
   * pt-webgl2's `'manifold-nee'` is a **deterministic refraction-walk heuristic**,
   * NOT the Newton-solve Manifold Next-Event Estimation (MNEE) of pt-webgpu.
   * It walks the refracted ray chain step-by-step, checks if it escapes to the
   * environment (`reachedLight = true`), then adds `throughput * color * pow10Focus`
   * as a caustic weight. There is no Newton solver, no constraint manifold, and no
   * unbiased connection. The option name is kept for API stability; the real MNEE
   * port from pt-webgpu is tracked as a road-to-100 fidelity item.
   *
   * Similarly, `'photon-map'` is a deterministic cone-traced estimate (8 sample cone
   * rays, inverse-distance kernel, escaped rays add `+1.0` energy) — a known
   * approximation (~21% energy bias at typical cone sizes), NOT a full photon-map or
   * bidirectional density estimation. Documented here so callers set expectations
   * correctly.
   */
  readonly _causticStrategyDoc?: never; // JSDoc-only; not a real field.
  /**
   * Opacity of directly-visible background (sky/HDRI) pixels, in [0, 1].
   * Default 1 (opaque — the environment is visible behind the scene). Values < 1
   * make the background partially/fully transparent (alpha coverage), which forces
   * the alpha-composite accumulation regime so the transparency composites
   * correctly. Before this existed the uniform defaulted to 0 and directly-visible
   * env never accumulated (rendered black) — see items_to_fix §H3.
   */
  readonly backgroundAlpha?: number;
  /**
   * Camera projection model (flag-plumbing audit 2026-06-10). The GLSL
   * `getCameraRay` fully implements all three; the host declares which one its
   * `projMatrix` represents (the matrix alone can't disambiguate equirect, which
   * is not a linear projection). Default `'perspective'` — byte-identical to the
   * prior fixed CAMERA_TYPE=0 path.
   *   • 'perspective'     — standard pinhole (CAMERA_TYPE 0)
   *   • 'orthographic'    — parallel rays along camera −Z (CAMERA_TYPE 1)
   *   • 'equirectangular' — 360° panoramic capture (CAMERA_TYPE 2)
   */
  readonly cameraType?: 'perspective' | 'orthographic' | 'equirectangular';
  /**
   * Thin-lens depth of field (flag-plumbing audit 2026-06-10). When set, enables
   * the GLSL FEATURE_DOF aperture sampler and uploads these PhysicalCamera
   * uniforms. Omitted (default) → pinhole camera, byte-identical to the prior
   * fixed FEATURE_DOF=0 path. All distances are in scene units; `bokehSize` is the
   * aperture diameter in millimetres (matches the fork's `bokehSize·0.5·1e-3` scale).
   */
  readonly dof?: {
    readonly focusDistance: number;
    /** Aperture diameter (mm). Larger = shallower depth of field. */
    readonly bokehSize: number;
    /** Aperture polygon sides; 0 = perfect circle (default 0). */
    readonly apertureBlades?: number;
    /** Aperture rotation (radians, default 0). */
    readonly apertureRotation?: number;
    /** Anamorphic squeeze ratio (1 = spherical, default 1). */
    readonly anamorphicRatio?: number;
  };
}
