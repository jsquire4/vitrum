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
 * - `scene.environment` set → `{ kind: 'hdri', hdri: texture, intensity, rotationY }`.
 *   `intensity` mirrors `scene.environmentIntensity` (default 1); `rotationY`
 *   mirrors `scene.environmentRotation.y` (default 0). Capturing both fields
 *   prevents env intensity / rotation from being silently dropped on the
 *   THREE → vitrum → THREE round trip used by the PT engine env-update path.
 * - `scene.background` as a solid Color (no environment map) → `{ kind: 'none' }`.
 * - Nothing set → `{ kind: 'none' }`.
 *
 * Asymmetry note: this direction is the THREE → vitrum mapping. The reverse
 * direction (`vitrumSceneToThree`) does NOT substitute a procedural sky shader
 * — it warns and falls through to a dark background. Mapping the reverse here
 * — turning a `THREE.Sky` mesh / shader into a `ProceduralSkyEnvironment` —
 * is unimplemented because no host currently constructs three.js scenes from
 * a `THREE.Sky` source. If a host needs this, read uniforms `{ turbidity,
 * mieCoefficient, mieDirectionalG, rayleigh }` off the sky material and emit
 * `{ kind: 'procedural-sky', ... }`. Until then a solid-color background is
 * not an IBL source — treat as no environment.
 */
export function resolveEnvironment(threeScene: THREE.Scene): SceneEnvironment {
  if (threeScene.environment != null) {
    // THREE.Scene.environmentIntensity defaults to 1; environmentRotation is
    // a THREE.Euler. We capture only the Y rotation because HdriEnvironment
    // models a yaw around world up (matches WebGLPathTracer's equirect map
    // rotation behavior — full Euler isn't needed for hemispheric IBL).
    const intensity = threeScene.environmentIntensity ?? 1;
    const rotationY = threeScene.environmentRotation?.y ?? 0;
    return { kind: 'hdri', hdri: threeScene.environment, intensity, rotationY };
  }
  // Some hosts use background-only HDRI setups (environment unset).
  if ((threeScene.background as THREE.Texture | null | undefined)?.isTexture === true) {
    const bg = threeScene.background as THREE.Texture;
    const intensity = threeScene.backgroundIntensity ?? 1;
    const rotationY = threeScene.backgroundRotation?.y ?? 0;
    return { kind: 'hdri', hdri: bg, intensity, rotationY };
  }
  return { kind: 'none' };
}
