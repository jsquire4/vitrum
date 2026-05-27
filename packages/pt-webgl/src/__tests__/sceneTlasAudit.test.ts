import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { auditPtWebglSceneForTlas } from '../sceneTlasAudit.js';

function unitTri(id: string): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

describe('auditPtWebglSceneForTlas', () => {
  it('merged-bvh-ok for a single mesh', () => {
    const scene: Scene = {
      primitives: [unitTri('a')],
      emitters: [],
      environment: { kind: 'none' },
    };
    const audit = auditPtWebglSceneForTlas(scene);
    expect(audit.needsTlas).toBe(false);
    expect(audit.recommendation).toBe('merged-bvh-ok');
  });

  it('prefer-tlas-backend for multi-mesh scenes', () => {
    const scene: Scene = {
      primitives: [unitTri('a'), unitTri('b')],
      emitters: [],
      environment: { kind: 'none' },
    };
    const audit = auditPtWebglSceneForTlas(scene);
    expect(audit.needsTlas).toBe(true);
    expect(audit.recommendation).toBe('prefer-tlas-backend');
  });

  it('prefer-tlas-backend for instanced meshes', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'instanced-mesh',
        id: 'inst',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        instances: [
          asMat4(new Float32Array(16)),
          asMat4(new Float32Array(16)),
        ],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const audit = auditPtWebglSceneForTlas(scene);
    expect(audit.needsTlas).toBe(true);
    expect(audit.totalInstanceCount).toBe(2);
  });
});
