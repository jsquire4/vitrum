import { describe, expect, it } from 'vitest';
import type { Scene } from '../scene/index.js';
import type { AnalyticShape } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import { partitionSceneBySupport, type SupportSets } from '../scene/partitionSceneBySupport.js';

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.1, 0.2, 0.3], roughness: 0.5, metallic: 0.2 },
      },
      {
        kind: 'analytic',
        id: 'sphere-a',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: { baseColor: [1, 0, 0], roughness: 0.3, metallic: 0 },
      },
      {
        kind: 'analytic',
        id: 'capsule-a',
        shape: 'capsule',
        params: new Float32Array([0, 0, 0, 0, 1, 0, 0.5]),
        material: { baseColor: [0, 1, 0], roughness: 0.4, metallic: 0 },
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
      {
        kind: 'point',
        id: 'lamp',
        position: [1, 2, 3],
        color: [1, 1, 1],
        intensity: 5,
      },
    ],
    environment: { kind: 'none' },
  };
}

/** Sets that accept everything in `makeScene`. */
const ALL_SUPPORTED: SupportSets = {
  supportedPrimitiveKinds: new Set<Scene['primitives'][number]['kind']>([
    'mesh',
    'instanced-mesh',
    'analytic',
    'skinned-mesh',
  ]),
  supportedEmitterKinds: new Set<SceneEmitter['kind']>([
    'directional',
    'point',
    'spot',
    'rect-area',
    'disc-area',
    'mesh-area',
  ]),
  supportedAnalyticShapes: new Set<AnalyticShape>(['sphere', 'box', 'capsule', 'cylinder']),
  supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri', 'procedural-sky']),
};

describe('partitionSceneBySupport', () => {
  it('returns every node and no warnings when the scene is fully supported', () => {
    const scene = makeScene();
    const { supported, warnings } = partitionSceneBySupport(scene, ALL_SUPPORTED);

    expect(warnings).toEqual([]);
    expect(supported.primitives).toHaveLength(3);
    expect(supported.emitters).toHaveLength(2);
    expect(supported.environment).toBe(scene.environment);
  });

  it('drops an unsupported primitive KIND with a warning', () => {
    const scene = makeScene();
    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      // analytic primitives no longer supported
      supportedPrimitiveKinds: new Set<Scene['primitives'][number]['kind']>([
        'mesh',
        'instanced-mesh',
        'skinned-mesh',
      ]),
    });

    expect(supported.primitives.map((p) => p.id)).toEqual(['mesh-a']);
    // Both analytic primitives are dropped by the kind filter (one warning each).
    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toMatch(/sphere-a/);
    expect(warnings[0]).toMatch(/analytic/);
    expect(warnings[0]).toMatch(/not supported/);
  });

  it('drops an analytic primitive whose SHAPE is unsupported with a warning', () => {
    const scene = makeScene();
    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      // sphere is fine, capsule is not
      supportedAnalyticShapes: new Set<AnalyticShape>(['sphere', 'box']),
    });

    expect(supported.primitives.map((p) => p.id)).toEqual(['mesh-a', 'sphere-a']);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/capsule-a/);
    expect(warnings[0]).toMatch(/capsule/);
    expect(warnings[0]).toMatch(/not supported/);
  });

  it('drops an unsupported emitter kind with a warning', () => {
    const scene = makeScene();
    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      supportedEmitterKinds: new Set<SceneEmitter['kind']>(['directional']),
    });

    expect(supported.emitters.map((e) => e.id)).toEqual(['sun']);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/lamp/);
    expect(warnings[0]).toMatch(/point/);
  });

  it('warns about an unsupported environment but never drops the scene', () => {
    const scene: Scene = { ...makeScene(), environment: { kind: 'procedural-sky',
      sunDirection: [0, 1, 0], turbidity: 2, rayleigh: 1, mieCoefficient: 0.005, mieDirectionalG: 0.8 } };
    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri']),
    });

    // Environment is carried through unchanged (warn, don't drop).
    expect(supported.environment).toBe(scene.environment);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/procedural-sky/);
  });

  it('places no restriction on a facet whose Set is omitted', () => {
    const scene = makeScene();
    // Only restrict emitters; primitives / analytic shapes / environment are
    // unconstrained and must all pass through.
    const { supported, warnings } = partitionSceneBySupport(scene, {
      supportedEmitterKinds: new Set<SceneEmitter['kind']>(['directional', 'point']),
    });

    expect(supported.primitives).toHaveLength(3);
    expect(supported.emitters).toHaveLength(2);
    expect(warnings).toEqual([]);
  });

  it('returns no warnings for an empty scene', () => {
    const empty: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };
    const { supported, warnings } = partitionSceneBySupport(empty, ALL_SUPPORTED);

    expect(supported.primitives).toEqual([]);
    expect(supported.emitters).toEqual([]);
    expect(warnings).toEqual([]);
  });

  it('does not mutate the input scene (pure helper)', () => {
    const scene = makeScene();
    const primCountBefore = scene.primitives.length;
    const emitCountBefore = scene.emitters.length;
    const firstPrim = scene.primitives[0];
    const firstEmit = scene.emitters[0];

    const { supported } = partitionSceneBySupport(scene, {
      supportedPrimitiveKinds: new Set<Scene['primitives'][number]['kind']>(['mesh']),
      supportedEmitterKinds: new Set<SceneEmitter['kind']>(['directional']),
    });

    // Original scene + its arrays are untouched.
    expect(scene.primitives).toHaveLength(primCountBefore);
    expect(scene.emitters).toHaveLength(emitCountBefore);
    expect(scene.primitives[0]).toBe(firstPrim);
    expect(scene.emitters[0]).toBe(firstEmit);

    // A new Scene + new arrays are returned (no aliasing of the filtered arrays).
    expect(supported).not.toBe(scene);
    expect(supported.primitives).not.toBe(scene.primitives);
    expect(supported.emitters).not.toBe(scene.emitters);
    // Surviving node identities are preserved (filter, not deep clone).
    expect(supported.primitives[0]).toBe(firstPrim);
    expect(supported.emitters[0]).toBe(firstEmit);
  });
});
