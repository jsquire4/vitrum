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

const MAT = { baseColor: [0.8, 0.8, 0.8], roughness: 1, metallic: 0 } as unknown as MaterialSpec;

/** Axis-aligned quad (half-size 1) centred at (cx,cy,z), facing +Z. */
function quad(id: string, cx: number, cy: number, z: number): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([cx - 1, cy - 1, z, cx + 1, cy - 1, z, cx + 1, cy + 1, z, cx - 1, cy + 1, z]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: MAT,
  };
}

// Camera at (0,0,5) looking down −Z, 60° vertical fov, square viewport.
const camera: PickCamera = {
  viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]),
  projMatrix: makePerspective(60, 1, 0.1, 100),
  cameraPosition: [0, 0, 5],
};
const W = 100;
const H = 100;
const scene = (...prims: ReadonlyArray<unknown>): Scene =>
  ({ primitives: prims, emitters: [], environment: {} }) as unknown as Scene;

describe('pickPrimitiveCpu (T3.G)', () => {
  it('picks the mesh under the centre pixel (the acceptance case)', () => {
    expect(pickPrimitiveCpu(scene(quad('A', 0, 0, 0)), camera, W / 2, H / 2, W, H)).toBe('A');
  });

  it('returns null when the ray misses all geometry (corner pixel → frustum edge)', () => {
    expect(pickPrimitiveCpu(scene(quad('A', 0, 0, 0)), camera, 0, 0, W, H)).toBe(null);
  });

  it('returns the NEARER of two overlapping meshes (depth ordering)', () => {
    // A at z=0, B at z=2 (closer to the camera at z=5). Centre ray hits both; B wins.
    expect(pickPrimitiveCpu(scene(quad('A', 0, 0, 0), quad('B', 0, 0, 2)), camera, W / 2, H / 2, W, H)).toBe('B');
  });

  it('picks an analytic sphere at the origin', () => {
    const sphere = {
      kind: 'analytic',
      id: 'S',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: MAT,
    } as unknown as AnalyticPrimitive;
    expect(pickPrimitiveCpu(scene(sphere), camera, W / 2, H / 2, W, H)).toBe('S');
  });

  it('honours a primitive transform (translate the quad off the centre ray → miss)', () => {
    const moved: MeshPrimitive = {
      ...quad('A', 0, 0, 0),
      transform: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1]) as unknown as Mat4,
    };
    expect(pickPrimitiveCpu(scene(moved), camera, W / 2, H / 2, W, H)).toBe(null);
  });

  it('returns null for an empty scene', () => {
    expect(pickPrimitiveCpu(scene(), camera, W / 2, H / 2, W, H)).toBe(null);
  });

  it('returns null for a zero-size viewport (guard)', () => {
    expect(pickPrimitiveCpu(scene(quad('A', 0, 0, 0)), camera, 0, 0, 0, 0)).toBe(null);
  });
});
