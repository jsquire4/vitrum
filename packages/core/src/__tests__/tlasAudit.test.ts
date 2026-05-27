import { describe, expect, it } from 'vitest';
import { asMat4 } from '../scene/math.js';
import { auditSceneNeedsTlas } from '../scene/tlasAudit.js';
import type { Scene } from '../scene/index.js';

function unitTri(id: string): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

describe('auditSceneNeedsTlas', () => {
  it('merged-bvh-ok for one mesh', () => {
    const scene: Scene = {
      primitives: [unitTri('a')],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(auditSceneNeedsTlas(scene).needsTlas).toBe(false);
  });

  it('prefer-tlas-backend for multi-mesh', () => {
    const scene: Scene = {
      primitives: [unitTri('a'), unitTri('b')],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(auditSceneNeedsTlas(scene).needsTlas).toBe(true);
  });

  it('prefer-tlas-backend for instancing', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'instanced-mesh',
        id: 'inst',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        instances: [asMat4(new Float32Array(16)), asMat4(new Float32Array(16))],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(auditSceneNeedsTlas(scene).totalInstanceCount).toBe(2);
  });
});
