// Scene description — backend-agnostic.
//
// Environment — hemispheric / global light source.

import type { Vec3 } from './math.js';

/**
 * Opaque environment-map handle. Unlike a material `TextureRef`, an env map is
 * NOT UV-transformed / texCoord-indexed — its representation is backend-specific
 * (for example, a raw `{ width, height, data }` equirect payload for pt-webgl2
 * or pt-webgpu). The binding layer / host supplies it; the
 * backend interprets it. Core never inspects it.
 */
export type EnvironmentMapRef = unknown;

export type SceneEnvironment =
  | HdriEnvironment
  | ProceduralSkyEnvironment
  | NoneEnvironment;

export interface HdriEnvironment {
  readonly kind: 'hdri';
  readonly hdri: EnvironmentMapRef;
  readonly intensity?: number;            // default 1
  /**
   * Counter-clockwise rotation of the environment dome around the world +Y axis,
   * in radians (default 0).
   *
   * **Convention:** a world-space direction `d` samples the *unrotated* map at
   * the direction `rotateY(d, -rotationY)`.  Equivalently, the CDF-sampled
   * importance direction (generated from the unrotated CDF) is rotated by
   * `+rotationY` to produce the corresponding world-space light direction.
   *
   * **Per-backend support:**
   * - `pt-webgl2` — IMPLEMENTED: builds a column-major 4×4 rotation matrix
   *   (`makeRotationYMat4(-rotationY)`) and uploads it as the GLSL
   *   `environmentRotation` uniform.  The shader applies `mat3(environmentRotation)
   *   * worldDir` before the equirect UV lookup, i.e. it rotates the lookup
   *   direction by `−rotationY` — exactly the convention above.
   * - `pt-webgpu` — IMPLEMENTED: packs the angle into
   *   `params.environmentTint.w`; the WGSL `environmentLookup` helper rotates
   *   the lookup direction by `−rotationY` before computing UV, and
   *   `sampleEnvironmentImportance` rotates the CDF-sampled direction by
   *   `+rotationY` to yield the world-space sample direction.
   * - `walkaround-hybrid` — IMPLEMENTED: the scene-load path builds equirect
   *   importance-sampling inverse-CDFs (sinθ-weighted, rotationY-aware) and
   *   rotates directional IBL samples by `+rotationY`. The realtime shaders use
   *   the shared `envRadiance` path for sky misses, ReSTIR-DI environment
   *   candidates, GI/NRC escape rays, transparent OIT, and material
   *   `envMapIntensity` lighting. DDGI probe updates receive the same
   *   environment texture and rotation through their probe environment bindings.
   *   Promotion evidence still lives in GPU validation gates because these are
   *   finite realtime/probe samples, but `rotationY` is an active transport
   *   parameter rather than a tint-only hook.
   */
  readonly rotationY?: number;            // radians, default 0
}

export interface ProceduralSkyEnvironment {
  readonly kind: 'procedural-sky';
  readonly sunDirection: Vec3;
  readonly turbidity: number;
  readonly rayleigh: number;
  readonly mieCoefficient: number;
  /**
   * Henyey–Greenstein asymmetry `g`, authored strictly inside `(-1, 1)`.
   * Sky evaluation uses the shared ±0.999999 numerical stability cap.
   */
  readonly mieDirectionalG: number;
  readonly intensity?: number;
}

export interface NoneEnvironment {
  readonly kind: 'none';
}
