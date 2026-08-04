import { describe, expect, it } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import {
  applyMatrix4MergedWorldF32,
  applyMatrix4ShaderF32,
} from '../../../shared-bvh/src/worldTransforms.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

const CUBE_INDICES = new Uint32Array([
  0, 2, 1, 0, 3, 2,
  4, 5, 6, 4, 6, 7,
  0, 1, 5, 0, 5, 4,
  3, 7, 6, 3, 6, 2,
  0, 4, 7, 0, 7, 3,
  1, 2, 6, 1, 6, 5,
]);

const CANCELLATION_TRANSFORM = asMat4(new Float32Array([
  1.2345670461654663, 0, 0, 0,
  -0.9876539707183838, 1, 0, 0,
  0.33333298563957214, 0, 1, 0,
  43731892, 0, 0, 1,
]));

function cancellationScene(): Scene {
  const x = 33856616;
  const y = -23691874;
  const z = 19167270;
  const d = 128;
  return {
    primitives: [{
      kind: 'mesh',
      id: 'cancellation-cube',
      positions: new Float32Array([
        x, y, z,
        x + d, y, z,
        x + d, y + d, z,
        x, y + d, z,
        x, y, z + d,
        x + d, y, z + d,
        x + d, y + d, z + d,
        x, y + d, z + d,
      ]),
      normals: new Float32Array(24).fill(1),
      indices: CUBE_INDICES,
      transform: CANCELLATION_TRANSFORM,
      material: {
        baseColor: [1, 1, 1],
        roughness: 0,
        metallic: 0,
        transmission: 1,
        thickness: 1,
      },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pt-webgpu optical-topology representation arithmetic', () => {
  it('validates and packs the exact transform arithmetic each executed tier uses', () => {
    const scene = cancellationScene();
    const full = buildPackedScene(scene, { geometryMode: 'tlas' });
    const lite = buildPackedScene(scene, { geometryMode: 'merged' });

    const local = [
      full.positions[0]!,
      full.positions[1]!,
      full.positions[2]!,
    ] as const;
    const fullTransform = asMat4(new Float32Array(
      Array.from(full.tlasInstanceLocalToWorld.slice(0, 16)),
    ));
    const executedFull = applyMatrix4ShaderF32(
      fullTransform, local[0], local[1], local[2],
    ).point;
    const expectedFull = applyMatrix4ShaderF32(
      CANCELLATION_TRANSFORM, local[0], local[1], local[2],
    ).point;
    const expectedLite = applyMatrix4MergedWorldF32(
      CANCELLATION_TRANSFORM, local[0], local[1], local[2],
    ).point;

    expect(executedFull).toEqual(expectedFull);
    expect(Array.from(lite.positions.slice(0, 3))).toEqual(expectedLite);
    expect(expectedFull).not.toEqual(expectedLite);
    expect(full.indices[3]).toBeGreaterThan(0);
    expect(lite.indices[3]).toBeGreaterThan(0);
  });
});
