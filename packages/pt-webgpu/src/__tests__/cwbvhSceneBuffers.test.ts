import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

function fiveTriangleMesh(id: string): Scene['primitives'][number] {
  const positions: number[] = [];
  const normals: number[] = [];
  for (let tri = 0; tri < 5; tri += 1) {
    const x = tri * 2;
    positions.push(
      x, 0, 0,
      x + 1, 0, 0,
      x, 1, 0,
    );
    normals.push(
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    );
  }
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    material: { baseColor: [0.6, 0.6, 0.6], roughness: 0.5, metallic: 0 },
  };
}

function oneTriangleMesh(id: string): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([20, 0, 0, 21, 0, 0, 20, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.4, 0.4, 0.4], roughness: 0.5, metallic: 0 },
  };
}

describe('pt-webgpu CWBVH scene-buffer prototype pack', () => {
  it('packs a CWBVH forest and remaps TLAS BLAS roots away from binary node indices', () => {
    const scene: Scene = {
      primitives: [
        fiveTriangleMesh('wide-a'),
        oneTriangleMesh('single-b'),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene);
    expect(packed.tlasBlasRoots.length).toBe(2);
    expect(packed.cwbvhTlasBlasRoots.length).toBe(2);
    expect(packed.cwbvhNodeCount).toBeGreaterThan(0);
    expect(packed.cwbvhNodeBounds.length).toBe(packed.cwbvhNodeCount * 6);
    expect(packed.cwbvhChildBoundsPacked.length).toBe(packed.cwbvhNodeCount * 8 * 3);
    expect(packed.cwbvhChildMeta.length).toBe(packed.cwbvhNodeCount * 8 * 3);
    expect(packed.cwbvhChildCount.length).toBe(packed.cwbvhNodeCount);

    const secondBinaryRoot = packed.tlasBlasRoots[1]!;
    const secondWideRoot = packed.cwbvhTlasBlasRoots[1]!;
    expect(secondBinaryRoot).toBeGreaterThan(1);
    expect(secondWideRoot).toBeLessThan(secondBinaryRoot);
    expect(packed.cwbvhBinaryRootToWideRoot[secondBinaryRoot]).toBe(secondWideRoot);
  });
});
