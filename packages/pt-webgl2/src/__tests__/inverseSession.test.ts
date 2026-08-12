import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene, SceneEmitter } from '@vitrum/core';
import {
  WebGl2FiniteDifferenceInverseSession,
  type WebGl2InverseEngineHooks,
} from '../inverse/finiteDifferenceSession.js';

const MATERIAL: MaterialSpec = {
  baseColor: [0.2, 0.1, 0.1],
  roughness: 1,
  metallic: 0,
};

function makeScene(): Scene {
  const primitive: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(6),
    indices: new Uint32Array([0, 1, 2]),
    material: { ...MATERIAL },
  };
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}

function patchTriMaterial(scene: Scene, patch: Partial<MaterialSpec>): void {
  (scene.primitives[0] as { material: MaterialSpec }).material = {
    ...scene.primitives[0]!.material,
    ...patch,
  };
}

function deferred<T>(): {
  readonly promise: Promise<T>;
  resolve(value: T): void;
  reject(error: unknown): void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function hooksFor(
  scene: Scene,
  renderAndReadback: WebGl2InverseEngineHooks['renderAndReadback'],
  patchMaterial: WebGl2InverseEngineHooks['patchMaterial'] =
    (_id, patch) => patchTriMaterial(scene, patch),
): WebGl2InverseEngineHooks {
  return {
    getScene: () => scene,
    renderAndReadback,
    patchMaterial,
    patchEmitter: () => {
      throw new Error('unexpected emitter patch');
    },
  };
}

describe('pt-webgl2 finite-difference inverse session', () => {
  it('optimizes material RGB parameters through backend patch hooks', async () => {
    const scene = makeScene();
    const patches: Partial<MaterialSpec>[] = [];
    const session = new WebGl2FiniteDifferenceInverseSession({
      getScene: () => scene,
      renderAndReadback: async () => {
        const c = scene.primitives[0]!.material.baseColor;
        return { rgba: new Float32Array([c[0], c[1], c[2], 1]), channels: 4 };
      },
      patchMaterial: (_primitiveId, patch) => {
        patches.push(patch);
        patchTriMaterial(scene, patch);
      },
      patchEmitter: () => {
        throw new Error('unexpected emitter patch');
      },
    }, {
      target: { width: 1, height: 1, channels: 3, data: new Float32Array([0.8, 0.1, 0.1]) },
      parameters: [{ path: 'materials.tri.baseColor', kind: 'rgb', min: 0, max: 1 }],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.05, fdEpsilon: 1e-2 },
    });

    const before = session.currentValues()[0]![0]!;
    const result = await session.step();

    expect(session.method).toBe('finite-difference');
    expect(session.parameterMethods).toEqual(['finite-difference']);
    expect(result.loss).toBeGreaterThan(0);
    expect(result.gradient[0]![0]).toBeLessThan(0);
    expect(session.currentValues()[0]![0]!).toBeGreaterThan(before);
    expect(scene.primitives[0]!.material.baseColor[0]).toBeGreaterThan(before);
    expect(patches.length).toBeGreaterThan(1);
  });

  it('rejects path-replay instead of silently switching to finite-difference', () => {
    const scene = makeScene();
    expect(() => new WebGl2FiniteDifferenceInverseSession({
      getScene: () => scene,
      renderAndReadback: async () => ({ rgba: new Float32Array([0, 0, 0, 1]), channels: 4 }),
      patchMaterial: () => {},
      patchEmitter: () => {},
    }, {
      target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
      parameters: [{ path: 'materials.tri.roughness', kind: 'scalar' }],
      method: 'path-replay',
    })).toThrow(/finite-difference only/);
  });

  it('rejects attenuationDistance fitting when the scene has no finite absorbing medium seed', () => {
    const scene = makeScene();

    expect(() => new WebGl2FiniteDifferenceInverseSession({
      getScene: () => scene,
      renderAndReadback: async () => ({ rgba: new Float32Array([0, 0, 0, 1]), channels: 4 }),
      patchMaterial: (primitiveId, patch) => {
        patchTriMaterial(scene, patch);
        expect(primitiveId).toBe('tri');
      },
      patchEmitter: (_emitterId: string, _patch: Partial<SceneEmitter>) => {},
    }, {
      target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
      parameters: [{ path: 'materials.tri.attenuationDistance', kind: 'scalar' }],
    })).toThrow(/finite positive scene attenuationDistance/);
  });

  it('accepts an explicit finite attenuationDistance seed and patches it into the scene', () => {
    const scene = makeScene();
    const session = new WebGl2FiniteDifferenceInverseSession({
      getScene: () => scene,
      renderAndReadback: async () => ({ rgba: new Float32Array([0, 0, 0, 1]), channels: 4 }),
      patchMaterial: (primitiveId, patch) => {
        patchTriMaterial(scene, patch);
        expect(primitiveId).toBe('tri');
      },
      patchEmitter: (_emitterId: string, _patch: Partial<SceneEmitter>) => {},
    }, {
      target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
      parameters: [{ path: 'materials.tri.attenuationDistance', kind: 'scalar', initial: [2.5] }],
    });

    expect(session.currentValues()[0]).toEqual([expect.closeTo(2.5, 6)]);
    expect(scene.primitives[0]!.material.attenuationDistance).toBeCloseTo(2.5, 6);
    session.dispose();
  });

  it('rejects reentrant steps and invalidates an awaited step when disposed', async () => {
    const scene = makeScene();
    const pending = deferred<{ rgba: Float32Array; channels: 4 }>();
    const session = new WebGl2FiniteDifferenceInverseSession(
      hooksFor(scene, () => pending.promise),
      {
        target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
        parameters: [{ path: 'materials.tri.roughness', kind: 'scalar' }],
      },
    );

    const first = session.step();
    await expect(session.step()).rejects.toThrow(/already in progress/);
    session.dispose();
    pending.resolve({ rgba: new Float32Array([0, 0, 0, 1]), channels: 4 });
    await expect(first).rejects.toThrow(/disposed while the step was in progress/);
    expect(session.currentValues()).toEqual([[1]]);
    expect(scene.primitives[0]!.material.roughness).toBe(1);
  });

  it('restores the finite-difference perturbation when probe rendering rejects', async () => {
    const scene = makeScene();
    let renders = 0;
    const session = new WebGl2FiniteDifferenceInverseSession(
      hooksFor(scene, async () => {
        renders += 1;
        if (renders === 2) throw new Error('probe failed');
        const c = scene.primitives[0]!.material.baseColor;
        return { rgba: new Float32Array([c[0], c[1], c[2], 1]), channels: 4 };
      }),
      {
        target: { width: 1, height: 1, data: new Float32Array([0.8, 0.1, 0.1]) },
        parameters: [{
          path: 'materials.tri.baseColor',
          kind: 'rgb',
          initial: [0.3, 0.1, 0.1],
        }],
      },
    );

    await expect(session.step()).rejects.toThrow(/probe failed/);
    expect(session.currentValues()[0]).toEqual([
      expect.closeTo(0.3, 6),
      expect.closeTo(0.1, 6),
      expect.closeTo(0.1, 6),
    ]);
    expect(scene.primitives[0]!.material.baseColor).toEqual([
      expect.closeTo(0.3, 6),
      expect.closeTo(0.1, 6),
      expect.closeTo(0.1, 6),
    ]);
  });

  it('rolls back a partially-applied constructor override', () => {
    const scene = makeScene();
    let patchCount = 0;
    expect(() => new WebGl2FiniteDifferenceInverseSession(
      hooksFor(
        scene,
        async () => ({ rgba: new Float32Array([0, 0, 0, 1]), channels: 4 }),
        (_id, patch) => {
          patchCount += 1;
          if (patchCount === 2) throw new Error('initial roughness patch failed');
          patchTriMaterial(scene, patch);
        },
      ),
      {
        target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
        parameters: [
          { path: 'materials.tri.baseColor', kind: 'rgb', initial: [0.9, 0.8, 0.7] },
          { path: 'materials.tri.roughness', kind: 'scalar', initial: [0.25] },
        ],
      },
    )).toThrow(/initial roughness patch failed/);
    expect(scene.primitives[0]!.material.baseColor).toEqual(MATERIAL.baseColor);
    expect(scene.primitives[0]!.material.roughness).toBe(MATERIAL.roughness);
  });

  it('aggregates constructor rollback failures', () => {
    const scene = makeScene();
    let patchCount = 0;
    expect(() => new WebGl2FiniteDifferenceInverseSession(
      hooksFor(
        scene,
        async () => ({ rgba: new Float32Array([0, 0, 0, 1]), channels: 4 }),
        (_id, patch) => {
          patchCount += 1;
          if (patchCount >= 2) throw new Error('patch unavailable');
          patchTriMaterial(scene, patch);
        },
      ),
      {
        target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
        parameters: [
          { path: 'materials.tri.baseColor', kind: 'rgb', initial: [0.9, 0.8, 0.7] },
          { path: 'materials.tri.roughness', kind: 'scalar', initial: [0.25] },
        ],
      },
    )).toThrow(AggregateError);
  });

  it('rolls Adam, flat values, and scene back so a retry matches a clean step', async () => {
    const makeDeterministicHooks = (
      scene: Scene,
      failPatch: (count: number) => boolean,
    ): WebGl2InverseEngineHooks => {
      let patchCount = 0;
      return hooksFor(
        scene,
        async () => {
          const c = scene.primitives[0]!.material.baseColor;
          return { rgba: new Float32Array([c[0], c[1], c[2], 1]), channels: 4 };
        },
        (_id, patch) => {
          patchCount += 1;
          if (failPatch(patchCount)) throw new Error('updated scene patch failed');
          patchTriMaterial(scene, patch);
        },
      );
    };
    const options = {
      target: { width: 1, height: 1, data: new Float32Array([0.8, 0.1, 0.1]) },
      parameters: [{
        path: 'materials.tri.baseColor',
        kind: 'rgb' as const,
        initial: [0.3, 0.1, 0.1],
      }],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.05, fdEpsilon: 1e-2 },
    };
    const failedScene = makeScene();
    const failedSession = new WebGl2FiniteDifferenceInverseSession(
      makeDeterministicHooks(failedScene, (count) => count === 5),
      options,
    );
    await expect(failedSession.step()).rejects.toThrow(/updated scene patch failed/);
    expect(failedSession.currentValues()[0]![0]).toBeCloseTo(0.3, 6);

    const cleanScene = makeScene();
    const cleanSession = new WebGl2FiniteDifferenceInverseSession(
      makeDeterministicHooks(cleanScene, () => false),
      options,
    );
    const [retry, clean] = await Promise.all([failedSession.step(), cleanSession.step()]);
    expect(retry).toEqual(clean);
    expect(failedScene.primitives[0]!.material.baseColor)
      .toEqual(cleanScene.primitives[0]!.material.baseColor);
  });

  it('poisons a session when step rollback cannot restore the scene', async () => {
    const scene = makeScene();
    let failPatches = false;
    const session = new WebGl2FiniteDifferenceInverseSession(
      hooksFor(
        scene,
        async () => ({ rgba: new Float32Array([0, 0, 0, 1]), channels: 4 }),
        (_id, patch) => {
          if (failPatches) throw new Error('scene patch unavailable');
          patchTriMaterial(scene, patch);
        },
      ),
      {
        target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
        parameters: [{ path: 'materials.tri.roughness', kind: 'scalar' }],
      },
    );
    failPatches = true;
    await expect(session.step()).rejects.toThrow(AggregateError);
    failPatches = false;
    await expect(session.step()).rejects.toThrow(/session is poisoned/);
  });

  it('rejects malformed readbacks and non-finite losses transactionally', async () => {
    const scene = makeScene();
    const malformed = new WebGl2FiniteDifferenceInverseSession(
      hooksFor(scene, async () => ({ rgba: new Float32Array(3), channels: 4 })),
      {
        target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
        parameters: [{ path: 'materials.tri.roughness', kind: 'scalar' }],
      },
    );
    await expect(malformed.step()).rejects.toThrow(/data length/);
    expect(scene.primitives[0]!.material.roughness).toBe(1);

    const overflow = new WebGl2FiniteDifferenceInverseSession(
      hooksFor(
        scene,
        async () => ({
          rgba: new Float32Array([3e38, 3e38, 3e38, 1]),
          channels: 4,
        }),
      ),
      {
        target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
        parameters: [{ path: 'materials.tri.roughness', kind: 'scalar' }],
      },
    );
    const largeFinite = await overflow.step();
    expect(Number.isFinite(largeFinite.loss)).toBe(true);
  });

  it('normalizes non-Error hook rejections before exposing them', async () => {
    const scene = makeScene();
    const session = new WebGl2FiniteDifferenceInverseSession(
      hooksFor(scene, async () => {
        // eslint-disable-next-line @typescript-eslint/only-throw-error -- verifies rejection normalization
        throw 'raw rejection';
      }),
      {
        target: { width: 1, height: 1, data: new Float32Array([0, 0, 0]) },
        parameters: [{ path: 'materials.tri.roughness', kind: 'scalar' }],
      },
    );
    let caught: unknown;
    try {
      await session.step();
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toContain('non-Error value');
    expect(scene.primitives[0]!.material.roughness).toBe(1);
  });
});
