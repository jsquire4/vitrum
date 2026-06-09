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
