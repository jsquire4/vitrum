import { describe, expect, it } from 'vitest';
import type { Scene } from '../scene/index.js';
import type { AnalyticShape } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import { partitionSceneBySupport, type SupportSets } from '../scene/partitionSceneBySupport.js';
import { validateScene } from '../scene/validation.js';

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

  it('converts unsupported analytic primitive kinds to canonical meshes with warnings', () => {
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

    expect(supported.primitives.map((p) => p.id)).toEqual(['mesh-a', 'sphere-a', 'capsule-a']);
    expect(supported.primitives.every((p) => p.kind === 'mesh')).toBe(true);
    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toMatch(/sphere-a/);
    expect(warnings[0]).toMatch(/analytic/);
    expect(warnings[0]).toMatch(/not supported/);
  });

  it('converts an unsupported analytic shape to its canonical mesh with a warning', () => {
    const scene = makeScene();
    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      // sphere is fine, capsule is not
      supportedAnalyticShapes: new Set<AnalyticShape>(['sphere', 'box']),
    });

    expect(supported.primitives.map((p) => p.id)).toEqual(['mesh-a', 'sphere-a', 'capsule-a']);
    expect(supported.primitives[2]!.kind).toBe('mesh');
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

  it('fallbackMesh: converts analytic to MeshPrimitive when analytic kind is unsupported', () => {
    // Scene has one analytic with a fallbackMesh; the backend does not accept 'analytic'.
    const fallback = {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
    };
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'sphere-with-fallback',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),
          material: { baseColor: [1, 0, 0], roughness: 0.3, metallic: 0 },
          fallbackMesh: fallback,
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      supportedPrimitiveKinds: new Set<Scene['primitives'][number]['kind']>(['mesh']),
    });

    // Converted to mesh, NOT dropped.
    expect(supported.primitives).toHaveLength(1);
    expect(supported.primitives[0]!.kind).toBe('mesh');
    expect(supported.primitives[0]!.id).toBe('sphere-with-fallback');
    // Warning still emitted.
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/sphere-with-fallback/);
    expect(warnings[0]).toMatch(/fallbackMesh/);
  });

  it('fallbackMesh: converts analytic to MeshPrimitive when analytic SHAPE is unsupported', () => {
    // The kind 'analytic' is accepted, but the specific shape 'capsule' is not.
    const fallback = {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    };
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'capsule-with-fallback',
          shape: 'capsule',
          params: new Float32Array([0, 0, 0, 0, 1, 0, 0.5]),
          material: { baseColor: [0, 1, 0], roughness: 0.4, metallic: 0 },
          fallbackMesh: fallback,
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      // sphere is fine, capsule is not
      supportedAnalyticShapes: new Set<AnalyticShape>(['sphere', 'box']),
    });

    expect(supported.primitives).toHaveLength(1);
    expect(supported.primitives[0]!.kind).toBe('mesh');
    expect(supported.primitives[0]!.id).toBe('capsule-with-fallback');
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/capsule-with-fallback/);
    expect(warnings[0]).toMatch(/fallbackMesh/);
  });

  it('fallbackMesh absent: unsupported analytic uses the canonical shape tessellator', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'capsule-no-fallback',
          shape: 'capsule',
          params: new Float32Array([0, 0, 0, 0, 1, 0, 0.5]),
          material: { baseColor: [0, 1, 0], roughness: 0.4, metallic: 0 },
          // fallbackMesh intentionally absent
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      supportedAnalyticShapes: new Set<AnalyticShape>(['sphere', 'box']),
    });

    expect(supported.primitives).toHaveLength(1);
    expect(supported.primitives[0]!.kind).toBe('mesh');
    expect(supported.primitives[0]!.id).toBe('capsule-no-fallback');
    if (supported.primitives[0]!.kind !== 'mesh') throw new Error('expected generated mesh');
    expect(supported.primitives[0]!.positions.length).toBeGreaterThan(0);
    expect(supported.primitives[0]!.indices?.length).toBeGreaterThan(0);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/capsule-no-fallback/);
    expect(warnings[0]).toMatch(/canonical generated mesh/);
  });

  it('drops a mesh-area emitter whose target primitive was dropped', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'floor',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.8, metallic: 0 },
        },
        {
          kind: 'skinned-mesh',
          id: 'banner',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [4, 4, 4] },
          skinIndices: new Uint32Array(12),
          skinWeights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
          bones: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
          boneInverses: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        },
      ],
      emitters: [
        { kind: 'directional', id: 'sun', direction: [0, -1, 0], color: [1, 1, 1], intensity: 2 },
        { kind: 'mesh-area', id: 'banner-light', meshId: 'banner', color: [1, 1, 1], intensity: 3 },
      ],
      environment: { kind: 'none' },
    };

    // Backend accepts mesh-area emitters but NOT skinned-mesh primitives, so the
    // emitter's target is dropped by the primitive filter above it.
    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      supportedPrimitiveKinds: new Set<Scene['primitives'][number]['kind']>(['mesh']),
    });

    expect(supported.primitives.map((p) => p.id)).toEqual(['floor']);
    // The dangling mesh-area emitter must NOT survive into the supported scene.
    expect(supported.emitters.map((e) => e.id)).toEqual(['sun']);
    expect(warnings).toHaveLength(2);
    expect(warnings[1]).toMatch(/banner-light/);
    expect(warnings[1]).toMatch(/mesh-area/);
    expect(warnings[1]).toMatch(/banner/);
    expect(warnings[1]).toMatch(/not supported/);

    // The supported partition must itself be an ingestible Scene.
    expect(() => validateScene(supported)).not.toThrow();
  });

  it('keeps a mesh-area emitter whose analytic target was converted to a mesh', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'glow-sphere',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0, emissive: [5, 5, 5] },
        },
      ],
      emitters: [
        { kind: 'mesh-area', id: 'glow', meshId: 'glow-sphere', color: [1, 1, 1], intensity: 2 },
      ],
      environment: { kind: 'none' },
    };

    // 'analytic' is unsupported but 'mesh' is accepted, so the primitive is
    // converted (id preserved) rather than dropped — the emitter must survive.
    const { supported, warnings } = partitionSceneBySupport(scene, {
      ...ALL_SUPPORTED,
      supportedPrimitiveKinds: new Set<Scene['primitives'][number]['kind']>(['mesh']),
    });

    expect(supported.primitives.map((p) => p.id)).toEqual(['glow-sphere']);
    expect(supported.primitives[0]!.kind).toBe('mesh');
    expect(supported.emitters.map((e) => e.id)).toEqual(['glow']);
    // Only the analytic→mesh conversion warning; no emitter drop.
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/glow-sphere/);
    expect(() => validateScene(supported)).not.toThrow();
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
