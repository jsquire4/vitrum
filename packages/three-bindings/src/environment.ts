/**
 * environment.ts — THREE.Scene environment → @vitrum/core SceneEnvironment.
 *
 * Resolves the scene's equirect/HDRI environment texture. Solid-color
 * backgrounds are not IBL sources and are treated as no environment.
 */

import type * as THREE from 'three';
import type { SceneEnvironment } from '@vitrum/core';

/**
 * Resolve the @vitrum/core SceneEnvironment from a THREE.Scene.
 *
 * - `scene.environment` set → `{ kind: 'hdri', hdri: texture }`.
 * - `scene.background` as a solid Color (no environment map) → `{ kind: 'none' }`.
 * - Nothing set → `{ kind: 'none' }`.
 *
 * Asymmetry note: this direction is the THREE → vitrum mapping. The reverse
 * direction (`vitrumSceneToThree`) DOES handle `ProceduralSkyEnvironment` by
 * substituting a procedural sky shader and warning. Mapping the reverse here
 * — turning a `THREE.Sky` mesh / shader into a `ProceduralSkyEnvironment` —
 * is unimplemented because no host currently constructs three.js scenes from
 * a `THREE.Sky` source. If a host needs this, read uniforms `{ turbidity,
 * mieCoefficient, mieDirectionalG, rayleigh }` off the sky material and emit
 * `{ kind: 'procedural-sky', ... }`. Until then a solid-color background is
 * not an IBL source — treat as no environment.
 */
export function resolveEnvironment(threeScene: THREE.Scene): SceneEnvironment {
  if (threeScene.environment != null) {
    return { kind: 'hdri', hdri: threeScene.environment };
  }
  return { kind: 'none' };
}
