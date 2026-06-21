import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene, SceneEmitter } from '@vitrum/core';
import { WebGl2FiniteDifferenceInverseSession } from '../inverse/finiteDifferenceSession.js';

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
    expect(result.loss).toBeGreaterThan(0);
    expect(result.gradient[0]![0]).toBeLessThan(0);
    expect(session.currentValues()[0]![0]!).toBeGreaterThan(before);
    expect(scene.primitives[0]!.material.baseColor[0]).toBeGreaterThan(before);
    expect(patches.length).toBeGreaterThan(1);
  });

  it('downgrades requested path replay with a structured diagnostic', () => {
    const scene = makeScene();
    const diagnostics: string[] = [];
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
      parameters: [{ path: 'materials.tri.roughness', kind: 'scalar' }],
      method: 'path-replay',
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic.code),
    });

    expect(session.method).toBe('finite-difference');
    expect(session.diagnostics).toEqual([
      expect.objectContaining({ code: 'path-replay-hook-missing' }),
    ]);
    expect(diagnostics).toEqual(['path-replay-hook-missing']);
  });
});
