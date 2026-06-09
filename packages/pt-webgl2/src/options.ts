import type { EngineOptions } from '@vitrum/core';
import type { WebGl2TraceTier } from './traceTier.js';

// `WebGl2TraceTier` is owned by ./traceTier.ts (the tier-selection module);
// re-exported here so consumers of the options surface keep importing it from
// one place. (Previously this module re-declared an identical union — D9 dedup.)
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
   * Opacity of directly-visible background (sky/HDRI) pixels, in [0, 1].
   * Default 1 (opaque — the environment is visible behind the scene). Values < 1
   * make the background partially/fully transparent (alpha coverage), which forces
   * the alpha-composite accumulation regime so the transparency composites
   * correctly. Before this existed the uniform defaulted to 0 and directly-visible
   * env never accumulated (rendered black) — see items_to_fix §H3.
   */
  readonly backgroundAlpha?: number;
}
