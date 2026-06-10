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
   * - `walkaround-hybrid` — DOCUMENTED NO-OP: the backend reduces the HDRI to a
   *   non-directional scalar tint (solid-angle-weighted average colour).
   *   Directional structure — and therefore any rotation of it — is not
   *   representable in the walkaround GI model.  `rotationY` is silently ignored
   *   and a warning is emitted via the environment-resolve path.
   */
  readonly rotationY?: number;            // radians, default 0
}

export interface ProceduralSkyEnvironment {
  readonly kind: 'procedural-sky';
  readonly sunDirection: Vec3;
  readonly turbidity: number;
  readonly rayleigh: number;
  readonly mieCoefficient: number;
  readonly mieDirectionalG: number;
  readonly intensity?: number;
}

export interface NoneEnvironment {
  readonly kind: 'none';
}
