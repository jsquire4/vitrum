import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { patchEmitterInScene, patchPrimitiveInScene } from '../scene/patchScene.js';

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [0.1, 0.2, 0.3],
          roughness: 0.5,
          metallic: 0.2,
        },
      },
    ],
    emitters: [
      {
        kind: 'directional',
        id: 'sun',
        direction: [0, -1, 0],
        color: [1, 1, 1],
        intensity: 2,
      },
    ],
    environment: { kind: 'none' },
  };
}

describe('patchScene helpers', () => {
  it('patches primitive by id', () => {
    const scene = makeScene();
    const primitive = scene.primitives[0];
    if (primitive == null) {
      throw new Error('missing primitive in test scene');
    }
    const next = patchPrimitiveInScene(scene, 'mesh-a', {
      material: {
        ...primitive.material,
        roughness: 0.1,
      },
    });
    const nextPrimitive = next.primitives[0];
    if (nextPrimitive == null) {
      throw new Error('missing patched primitive in test scene');
    }
    expect(nextPrimitive.material.roughness).toBe(0.1);
    expect(primitive.material.roughness).toBe(0.5);
  });

  it('patches emitter by id', () => {
    const scene = makeScene();
    const emitter = scene.emitters[0];
    if (emitter == null) {
      throw new Error('missing emitter in test scene');
    }
    const next = patchEmitterInScene(scene, 'sun', { intensity: 5 });
    const nextEmitter = next.emitters[0];
    if (nextEmitter == null) {
      throw new Error('missing patched emitter in test scene');
    }
    expect(nextEmitter.intensity).toBe(5);
    expect(emitter.intensity).toBe(2);
  });

  it('throws on missing ids', () => {
    const scene = makeScene();
    expect(() => patchPrimitiveInScene(scene, 'missing', {})).toThrow();
    expect(() => patchEmitterInScene(scene, 'missing', {})).toThrow();
  });
});
