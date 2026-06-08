import type { Scene } from '@vitrum/core';

interface ThreeBindingsModule {
  readonly sceneFromThreeJS: (scene: ThreeSceneLike) => Scene;
}

// Structural scene contract for hosts that hand the facade a THREE.Scene. The
// engine package intentionally avoids a `three` value import; conversion is
// delegated to @vitrum/three-bindings only after this guard succeeds.
export interface ThreeSceneLike {
  readonly isScene: true;
  readonly [key: string]: unknown;
}

export function isThreeScene(s: Scene | ThreeSceneLike): s is ThreeSceneLike {
  return typeof s === 'object'
    && s != null
    && (s as { isScene?: unknown }).isScene === true;
}

export async function sceneFromThreeSceneLike(scene: ThreeSceneLike): Promise<Scene> {
  try {
    const { sceneFromThreeJS } = await import('@vitrum/three-bindings') as unknown as ThreeBindingsModule;
    return sceneFromThreeJS(scene);
  } catch (err) {
    throw new Error(
      'createEngine: failed to load @vitrum/three-bindings while converting a THREE.Scene. ' +
      'Pass a @vitrum/core Scene for the THREE-free path, or install @vitrum/three-bindings ' +
      'and three before passing a THREE scene. Original error: ' + String(err),
    );
  }
}
