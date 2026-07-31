import { describe, expect, it } from 'vitest';
import type {
  AnalyticPrimitive,
  AnalyticShape,
  MaterialSpec,
  Scene,
  SkinnedMeshPrimitive,
} from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { pickPrimitiveCpu, type PickCamera } from '../pickPrimitiveCpu.js';

function perspective(fovDegrees: number, aspect: number, near: number, far: number): Float32Array {
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

function infinitePerspective(
  fovDegrees: number,
  aspect: number,
  near: number,
): Float32Array {
  const f = 1 / Math.tan((fovDegrees * Math.PI) / 360);
  const matrix = new Float32Array(16);
  matrix[0] = f / aspect;
  matrix[5] = f;
  matrix[10] = -1;
  matrix[11] = -1;
  matrix[14] = -2 * near;
  return matrix;
}

function orthographic(
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Float32Array {
  const matrix = new Float32Array(16);
  matrix[0] = 2 / (right - left);
  matrix[5] = 2 / (top - bottom);
  matrix[10] = -2 / (far - near);
  matrix[12] = -(right + left) / (right - left);
  matrix[13] = -(top + bottom) / (top - bottom);
  matrix[14] = -(far + near) / (far - near);
  matrix[15] = 1;
  return matrix;
}

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

const MATERIAL: MaterialSpec = {
  baseColor: [0.8, 0.8, 0.8],
  roughness: 1,
  metallic: 0,
};

const CAMERA: PickCamera = {
  viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]),
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
      restCenterX - 1,
      -1,
      0,
      restCenterX + 1,
      -1,
      0,
      restCenterX + 1,
      1,
      0,
      restCenterX - 1,
      1,
      0,
    ]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
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

function analyticScene(
  shape: AnalyticShape,
  params: readonly number[],
  transform?: Float32Array,
): Scene {
  const primitive: AnalyticPrimitive = {
    kind: 'analytic',
    id: `analytic-${shape}`,
    shape,
    params: Float32Array.from(params),
    material: MATERIAL,
    ...(transform != null
      ? { transform: transform as NonNullable<AnalyticPrimitive['transform']> }
      : {}),
  };
  return {
    primitives: [primitive],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pickPrimitiveCpu skinned geometry', () => {
  it('ray-tests the current bone pose instead of the rest-pose positions', () => {
    const movedOntoRay = translatedSkinnedQuad('onto-ray', 4, -4);
    expect(pickPrimitiveCpu(scene(movedOntoRay), CAMERA, 50, 50, 100, 100)).toBe('onto-ray');

    const movedOffRay = translatedSkinnedQuad('off-ray', 0, 4);
    expect(pickPrimitiveCpu(scene(movedOffRay), CAMERA, 50, 50, 100, 100)).toBeNull();
  });
});

describe('pickPrimitiveCpu orthographic camera', () => {
  it('uses a per-pixel origin with parallel view rays', () => {
    const camera: PickCamera = {
      ...CAMERA,
      projMatrix: orthographic(-2, 2, -2, 2, 0.1, 100),
    };
    const target: Scene['primitives'][number] = {
      kind: 'mesh',
      id: 'right-panel',
      positions: new Float32Array([0.8, -0.2, 0, 1.2, -0.2, 0, 1.2, 0.2, 0, 0.8, 0.2, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
      material: MATERIAL,
    };
    const orthoScene: Scene = {
      primitives: [target],
      emitters: [],
      environment: { kind: 'none' },
    };

    expect(pickPrimitiveCpu(orthoScene, camera, 75, 50, 100, 100)).toBe('right-panel');
    expect(pickPrimitiveCpu(orthoScene, camera, 50, 50, 100, 100)).toBeNull();
  });
});

describe('pickPrimitiveCpu projective ray construction', () => {
  it('supports an infinite far plane', () => {
    const camera: PickCamera = {
      ...CAMERA,
      projMatrix: infinitePerspective(60, 1, 0.1),
    };
    expect(
      pickPrimitiveCpu(
        scene(translatedSkinnedQuad('infinite-far-target', 0, 0)),
        camera,
        50,
        50,
        100,
        100,
      ),
    ).toBe('infinite-far-target');
  });

  it('does not pick geometry between the pinhole and the authored near plane', () => {
    const beforeNear: Scene['primitives'][number] = {
      kind: 'mesh',
      id: 'before-near',
      positions: new Float32Array([
        -0.01, -0.01, 4.95,
        0.01, -0.01, 4.95,
        0, 0.01, 4.95,
      ]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: MATERIAL,
    };
    expect(pickPrimitiveCpu({
      primitives: [beforeNear],
      emitters: [],
      environment: { kind: 'none' },
    }, CAMERA, 50, 50, 100, 100)).toBeNull();
  });
});

describe('pickPrimitiveCpu triangle tolerances', () => {
  it('picks a nanometer-scale perpendicular triangle without an absolute area floor', () => {
    const target: Scene['primitives'][number] = {
      kind: 'mesh',
      id: 'nanometer-triangle',
      positions: new Float32Array([-5e-9, -5e-9, 0, 5e-9, -5e-9, 0, 0, 5e-9, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
      material: MATERIAL,
    };
    const smallTriangleScene: Scene = {
      primitives: [target],
      emitters: [],
      environment: { kind: 'none' },
    };

    expect(pickPrimitiveCpu(smallTriangleScene, CAMERA, 50, 50, 100, 100)).toBe(
      'nanometer-triangle',
    );
  });

  it('picks geometry behind a valid tiny affine transform', () => {
    const scale = 1e-8;
    const target: Scene['primitives'][number] = {
      kind: 'mesh',
      id: 'tiny-transform-triangle',
      positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: MATERIAL,
      transform: asMat4(
        new Float32Array([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1]),
      ),
    };
    const tinyTransformScene: Scene = {
      primitives: [target],
      emitters: [],
      environment: { kind: 'none' },
    };

    expect(pickPrimitiveCpu(tinyTransformScene, CAMERA, 50, 50, 100, 100)).toBe(
      'tiny-transform-triangle',
    );
  });
});

describe('pickPrimitiveCpu exact analytic geometry', () => {
  it.each([
    ['sphere', [0, 0, 0, 1]],
    ['box', [0, 0, 0, 1, 1, 1]],
    ['cylinder', [0, 0, 0, 1, 1]],
    ['capsule', [0, -1, 0, 0, 1, 0, 0.5]],
    ['h-channel-came', [2, 1, 2, 0.25]],
  ] as const)('hits the declared %s surface', (shape, params) => {
    expect(pickPrimitiveCpu(analyticScene(shape, params), CAMERA, 50, 50, 100, 100)).toBe(
      `analytic-${shape}`,
    );
  });

  it.each([
    ['box', [1.25, 0, 0, 0.1, 2, 2]],
    ['cylinder', [1.25, 0, 0, 0.1, 2]],
    ['capsule', [1.25, -2, 0, 1.25, 2, 0, 0.1]],
  ] as const)(
    'rejects a %s bounding-volume false positive outside its silhouette',
    (shape, params) => {
      expect(pickPrimitiveCpu(analyticScene(shape, params), CAMERA, 50, 50, 100, 100)).toBeNull();
    },
  );

  it('intersects in local space under a non-uniform affine transform', () => {
    // A unit sphere scaled to a tall ellipsoid. At this pixel the local ray
    // crosses the ellipsoid, while the old first-column bounding-sphere scale
    // (0.25) missed it.
    const transform = new Float32Array([0.25, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(
      pickPrimitiveCpu(analyticScene('sphere', [0, 0, 0, 1], transform), CAMERA, 50, 40, 100, 100),
    ).toBe('analytic-sphere');
  });

  it('does not report an internal capsule component seam as the nearest boundary', () => {
    const insideCamera: PickCamera = {
      viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -0.5, 1]),
      projMatrix: perspective(60, 1, 0.1, 100),
      cameraPosition: [0, 0, 0.5],
    };
    const capsule: AnalyticPrimitive = {
      kind: 'analytic',
      id: 'capsule-shell',
      shape: 'capsule',
      params: new Float32Array([0, 0, -1, 0, 0, 1, 1]),
      material: MATERIAL,
    };
    const innerPlane: Scene['primitives'][number] = {
      kind: 'mesh',
      id: 'inner-plane',
      positions: new Float32Array([-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0, 0.5, -0.5]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: MATERIAL,
    };
    const nestedScene: Scene = {
      primitives: [capsule, innerPlane],
      emitters: [],
      environment: { kind: 'none' },
    };

    // The plane is one unit ahead. The capsule's real exit is 2.5 units ahead;
    // the old union-of-full-spheres shortcut incorrectly returned the internal
    // cap overlap at 0.5 and selected the capsule.
    expect(pickPrimitiveCpu(nestedScene, insideCamera, 50, 50, 100, 100)).toBe('inner-plane');
  });

  it.each([1e-10, 1e20])(
    'preserves every analytic silhouette at world scale %s',
    (scale) => {
      const scaledCamera: PickCamera = {
        viewMatrix: new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, -5 * scale, 1,
        ]),
        projMatrix: perspective(60, 1, 0.1 * scale, 100 * scale),
        cameraPosition: [0, 0, 5 * scale],
      };
      const scaledShapes = [
        ['sphere', [0, 0, 0, scale]],
        ['box', [0, 0, 0, scale, scale, scale]],
        ['cylinder', [0, 0, 0, scale, scale]],
        ['capsule', [0, -scale, 0, 0, scale, 0, 0.5 * scale]],
        ['h-channel-came', [2 * scale, scale, 2 * scale, 0.25 * scale]],
      ] as const;

      for (const [shape, params] of scaledShapes) {
        expect(
          pickPrimitiveCpu(
            analyticScene(shape, params),
            scaledCamera,
            50,
            50,
            100,
            100,
          ),
          shape,
        ).toBe(`analytic-${shape}`);
      }
    },
  );
});
