// Scene description — backend-agnostic.
//
// Environment — hemispheric / global light source.

import type { Vec3 } from './math.js';
import type { TextureRef } from './material.js';

export type SceneEnvironment =
  | HdriEnvironment
  | ProceduralSkyEnvironment
  | NoneEnvironment;

export interface HdriEnvironment {
  readonly kind: 'hdri';
  readonly hdri: TextureRef;
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
