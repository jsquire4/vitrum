import { describe, expect, it } from 'vitest';
import type {
  MaterialSpec,
  Scene,
  SkinnedMeshPrimitive,
} from '@vitrum/core';
import { pickPrimitiveCpu, type PickCamera } from '../pickPrimitiveCpu.js';

function perspective(
  fovDegrees: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array {
  const f = 1 / Math.tan((fovDegrees * Math.PI) / 360);
  const range = 1 / (near - far);
  const matrix = new Float32Array(16);
  matrix[0] = f / aspect;
  matrix[5] = f;
  matrix[10] = (far + near) * range;
  matrix[11] = -1;
  matrix[14] = 2 * far * near * range;
  return matrix;
}

const IDENTITY = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

const MATERIAL: MaterialSpec = {
  baseColor: [0.8, 0.8, 0.8],
  roughness: 1,
  metallic: 0,
};

const CAMERA: PickCamera = {
  viewMatrix: new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, -5, 1,
  ]),
  projMatrix: perspective(60, 1, 0.1, 100),
  cameraPosition: [0, 0, 5],
};

function translatedSkinnedQuad(
  id: string,
  restCenterX: number,
  boneTranslationX: number,
): SkinnedMeshPrimitive {
  const bone = new Float32Array(IDENTITY);
  bone[12] = boneTranslationX;
  return {
    kind: 'skinned-mesh',
    id,
    positions: new Float32Array([
      restCenterX - 1, -1, 0,
      restCenterX + 1, -1, 0,
      restCenterX + 1, 1, 0,
      restCenterX - 1, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    skinIndices: new Uint32Array([0, 0, 0, 0]),
    skinWeights: new Float32Array([1, 1, 1, 1]),
    skinInfluencesPerVertex: 1,
    bones: bone,
    boneInverses: new Float32Array(IDENTITY),
    material: MATERIAL,
  };
}

function scene(primitive: SkinnedMeshPrimitive): Scene {
  return {
    primitives: [primitive],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pickPrimitiveCpu skinned geometry', () => {
  it('ray-tests the current bone pose instead of the rest-pose positions', () => {
    const movedOntoRay = translatedSkinnedQuad('onto-ray', 4, -4);
    expect(pickPrimitiveCpu(scene(movedOntoRay), CAMERA, 50, 50, 100, 100))
      .toBe('onto-ray');

    const movedOffRay = translatedSkinnedQuad('off-ray', 0, 4);
    expect(pickPrimitiveCpu(scene(movedOffRay), CAMERA, 50, 50, 100, 100))
      .toBeNull();
  });
});
