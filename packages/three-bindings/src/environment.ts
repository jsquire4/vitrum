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
 */
export function resolveEnvironment(threeScene: THREE.Scene): SceneEnvironment {
  if (threeScene.environment != null) {
    return { kind: 'hdri', hdri: threeScene.environment };
  }
  // TODO: ProceduralSkyEnvironment is not handled here. A THREE.Sky object
  // with uniforms { turbidity, mieCoefficient, mieDirectionalG, rayleigh }
  // would feed this branch. See core/scene.ts ProceduralSkyEnvironment for
  // the expected fields and how the resolved environment is consumed
  // downstream.
  // A solid-color background is not an IBL source — treat as no environment.
  return { kind: 'none' };
}
