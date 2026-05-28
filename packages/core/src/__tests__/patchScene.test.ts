import { describe, expect, it } from 'vitest';
import type { Scene } from '../scene/index.js';
import { patchEmitterInScene, patchPrimitiveInScene } from '../scene/patchScene.js';

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
        ]),
        normals: new Float32Array([
          0, 0, 1,
          0, 0, 1,
          0, 0, 1,
        ]),
        material: {
          baseColor: [0.1, 0.2, 0.3],
          roughness: 0.5,
          metallic: 0.2,
        },
      },
      {
        kind: 'analytic',
        id: 'sphere-a',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: {
          baseColor: [1, 0, 0],
          roughness: 0.3,
          metallic: 0,
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
  it('patches a primitive by id and leaves the original scene unmutated', () => {
    const scene = makeScene();
    const primitive = scene.primitives[0];
    if (primitive == null) throw new Error('missing primitive in test scene');

    const next = patchPrimitiveInScene(scene, 'mesh-a', {
      material: { ...primitive.material, roughness: 0.1 },
    });

    const nextPrimitive = next.primitives[0];
    if (nextPrimitive == null) throw new Error('missing patched primitive');
    expect(nextPrimitive.material.roughness).toBe(0.1);

    // Original scene is untouched (pure helper).
    expect(primitive.material.roughness).toBe(0.5);
    expect(next).not.toBe(scene);
    expect(next.primitives).not.toBe(scene.primitives);
    expect(next.emitters).toBe(scene.emitters);
  });

  it('patches an emitter by id and leaves the original scene unmutated', () => {
    const scene = makeScene();
    const emitter = scene.emitters[0];
    if (emitter == null) throw new Error('missing emitter in test scene');

    const next = patchEmitterInScene(scene, 'sun', { intensity: 5 });
    const nextEmitter = next.emitters[0];
    if (nextEmitter == null) throw new Error('missing patched emitter');

    expect(nextEmitter.intensity).toBe(5);
    expect(emitter.intensity).toBe(2);
    expect(next.primitives).toBe(scene.primitives);
  });

  it('throws when the primitive id is missing (with the unified updatePrimitive: prefix)', () => {
    const scene = makeScene();
    expect(() => patchPrimitiveInScene(scene, 'missing', {})).toThrow(
      /^updatePrimitive: .*not found/,
    );
  });

  it('throws when the emitter id is missing (with the unified updateEmitter: prefix)', () => {
    const scene = makeScene();
    expect(() => patchEmitterInScene(scene, 'missing', {})).toThrow(
      /^updateEmitter: .*not found/,
    );
  });

  it('throws when a primitive patch tries to change the id', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', { id: 'mesh-b' } as never),
    ).toThrow(/id cannot be changed/);
  });

  it('throws when an emitter patch tries to change the id', () => {
    const scene = makeScene();
    expect(() =>
      patchEmitterInScene(scene, 'sun', { id: 'moon' } as never),
    ).toThrow(/id cannot be changed/);
  });

  it('throws when a primitive patch tries to change the kind', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', { kind: 'analytic' } as never),
    ).toThrow(/kind cannot change/);
  });

  it('throws when an emitter patch tries to change the kind', () => {
    const scene = makeScene();
    expect(() =>
      patchEmitterInScene(scene, 'sun', { kind: 'point' } as never),
    ).toThrow(/kind cannot change/);
  });

  it('validates analytic params and rejects a wrong-length shape param', () => {
    const scene = makeScene();
    // sphere expects 4 values; supply 3 → must throw via validateAnalyticParams.
    expect(() =>
      patchPrimitiveInScene(scene, 'sphere-a', {
        params: new Float32Array([0, 0, 0]),
      } as never),
    ).toThrow(/expects 4/);
  });

  it('throws when an analytic primitive resolves to missing shape/params', () => {
    // Defensive guard (patchScene.ts): a malformed analytic primitive lacking
    // `params`, patched on an unrelated field, must surface as an explicit
    // "requires shape and params" error rather than producing a broken Scene.
    const base = makeScene();
    const malformed = {
      kind: 'analytic',
      id: 'broken-analytic',
      shape: 'sphere',
      // params intentionally omitted
      material: { baseColor: [0, 1, 0], roughness: 0.4, metallic: 0 },
    };
    const scene: Scene = { ...base, primitives: [...base.primitives, malformed as never] };
    expect(() =>
      patchPrimitiveInScene(scene, 'broken-analytic', {
        material: { baseColor: [0, 0, 1], roughness: 0.2, metallic: 0 },
      } as never),
    ).toThrow(/requires shape and params/);
  });

  it('accepts a valid analytic param update', () => {
    const scene = makeScene();
    const next = patchPrimitiveInScene(scene, 'sphere-a', {
      params: new Float32Array([1, 2, 3, 4]),
    } as never);
    const patched = next.primitives[1];
    if (patched == null || patched.kind !== 'analytic') {
      throw new Error('missing patched analytic primitive');
    }
    expect(Array.from(patched.params)).toEqual([1, 2, 3, 4]);
  });

  it('rejects analytic "params" on a mesh-like primitive', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', {
        params: new Float32Array([0, 0, 0, 1]),
      } as never),
    ).toThrow(/cannot accept analytic "params"/);
  });
});
