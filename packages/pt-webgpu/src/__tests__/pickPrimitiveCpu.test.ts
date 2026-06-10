/**
 * Unit tests for `pickPrimitiveCpu` (T3.G #30) as consumed by
 * PTEngineWebGPU.debug.pickPrimitive.
 *
 * The helper lives in @vitrum/shared-bvh; these tests verify the unproject +
 * ray-cast contract as seen from the pt-webgpu side (same column-major
 * convention and FrameInput camera layout used by renderFrame).
 */
import { describe, it, expect } from 'vitest';
import { pickPrimitiveCpu, type PickCamera } from '@vitrum/shared-bvh';
import type { Scene, MeshPrimitive, AnalyticPrimitive, MaterialSpec, Mat4 } from '@vitrum/core';

// three.js-style symmetric perspective (column-major, NDC z ∈ [-1,1]).
function makePerspective(fovDeg: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan((fovDeg * Math.PI) / 360);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

const DUMMY_MAT = { baseColor: [1, 1, 1], roughness: 1, metallic: 0 } as unknown as MaterialSpec;

/** Axis-aligned quad (half-size 1) centred at (cx,cy,z), facing +Z. */
function quad(id: string, cx: number, cy: number, z: number): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([cx - 1, cy - 1, z, cx + 1, cy - 1, z, cx + 1, cy + 1, z, cx - 1, cy + 1, z]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: DUMMY_MAT,
  };
}

// Camera at (0,0,5) looking down −Z. view matrix: translate -5 in Z (column-major).
const CAMERA: PickCamera = {
  viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]),
  projMatrix: makePerspective(60, 1, 0.1, 100),
  cameraPosition: [0, 0, 5],
};
const W = 100;
const H = 100;

const makeScene = (...prims: ReadonlyArray<unknown>): Scene =>
  ({ primitives: prims, emitters: [], environment: {} }) as unknown as Scene;

describe('pickPrimitiveCpu (pt-webgpu perspective)', () => {
  it('hits a mesh quad centred on the screen', () => {
    expect(pickPrimitiveCpu(makeScene(quad('Q', 0, 0, 0)), CAMERA, W / 2, H / 2, W, H)).toBe('Q');
  });

  it('misses the geometry at a corner pixel', () => {
    // The quad only covers a small central region; the frustum corner is outside it.
    expect(pickPrimitiveCpu(makeScene(quad('Q', 0, 0, 0)), CAMERA, 0, 0, W, H)).toBeNull();
  });

  it('returns the CLOSER of two overlapping quads (depth sort)', () => {
    // A at z=0, B at z=2 (3 units closer to camera at z=5).
    const result = pickPrimitiveCpu(makeScene(quad('A', 0, 0, 0), quad('B', 0, 0, 2)), CAMERA, W / 2, H / 2, W, H);
    expect(result).toBe('B');
  });

  it('picks an analytic sphere at the origin', () => {
    const sphere: AnalyticPrimitive = {
      kind: 'analytic',
      id: 'Sph',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: DUMMY_MAT,
    } as unknown as AnalyticPrimitive;
    expect(pickPrimitiveCpu(makeScene(sphere), CAMERA, W / 2, H / 2, W, H)).toBe('Sph');
  });

  it('honours the primitive transform (translated off the centre ray)', () => {
    const moved: MeshPrimitive = {
      ...quad('A', 0, 0, 0),
      // Translate 10 units right — ray down the centre axis misses it.
      transform: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1]) as unknown as Mat4,
    };
    expect(pickPrimitiveCpu(makeScene(moved), CAMERA, W / 2, H / 2, W, H)).toBeNull();
  });

  it('returns null for an empty scene', () => {
    expect(pickPrimitiveCpu(makeScene(), CAMERA, W / 2, H / 2, W, H)).toBeNull();
  });

  it('returns null when width or height is zero', () => {
    expect(pickPrimitiveCpu(makeScene(quad('Q', 0, 0, 0)), CAMERA, 50, 50, 0, 0)).toBeNull();
  });
});
