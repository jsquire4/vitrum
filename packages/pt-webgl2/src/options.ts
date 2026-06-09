import type { EngineOptions } from '@vitrum/core';

export type WebGl2TraceTier = 'full' | 'lite';

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
}
