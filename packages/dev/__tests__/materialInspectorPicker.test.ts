/**
 * materialInspectorPicker.test.ts — Unit tests for MaterialInspector's
 * host-side picker math.
 *
 * The component re-exports its picker internals under double-underscore names
 * for test reach-through only. We verify the Möller-Trumbore triangle
 * intersector + the closest-mesh selector against a simple unit-cube +
 * displaced-cube scene with deterministic rays.
 */

import { describe, it, expect } from 'vitest';
import {
  __pickerRayTri,
  __pickerPickClosestMesh,
  __pickerMat4Inverse,
} from '../src/react/MaterialInspector.js';
import type { Scene, MeshPrimitive, Vec3 } from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Test fixture: single quad in the XY plane at z=0, plus a second quad at z=2.
// ────────────────────────────────────────────────────────────────────────────

function makeQuad(id: string, z: number): MeshPrimitive {
  // Two triangles forming a unit quad centred at (0, 0, z), facing +Z.
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([
      -0.5, -0.5, z,
       0.5, -0.5, z,
       0.5,  0.5, z,
      -0.5,  0.5, z,
    ]),
    normals: new Float32Array([
      0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1,
    ]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
  };
}

describe('MaterialInspector picker math', () => {
  it('rayTriangleIntersect hits a forward triangle', () => {
    const origin: Vec3 = [0, 0, -5];
    const direction: Vec3 = [0, 0, 1];
    const v0: [number, number, number] = [-1, -1, 0];
    const v1: [number, number, number] = [ 1, -1, 0];
    const v2: [number, number, number] = [ 0,  1, 0];
    const t = __pickerRayTri(origin, direction, v0, v1, v2);
    expect(t).not.toBeNull();
    expect(t).toBeCloseTo(5, 5);
  });

  it('rayTriangleIntersect misses a triangle outside the ray cone', () => {
    const origin: Vec3 = [0, 0, -5];
    const direction: Vec3 = [0, 0, 1];
    const v0: [number, number, number] = [10, 10, 0];
    const v1: [number, number, number] = [11, 10, 0];
    const v2: [number, number, number] = [10, 11, 0];
    const t = __pickerRayTri(origin, direction, v0, v1, v2);
    expect(t).toBeNull();
  });

  it('rayTriangleIntersect rejects backward intersection (t < 0)', () => {
    const origin: Vec3 = [0, 0, 5];
    const direction: Vec3 = [0, 0, 1];      // shooting away from the triangle at z=0
    const v0: [number, number, number] = [-1, -1, 0];
    const v1: [number, number, number] = [ 1, -1, 0];
    const v2: [number, number, number] = [ 0,  1, 0];
    const t = __pickerRayTri(origin, direction, v0, v1, v2);
    expect(t).toBeNull();
  });

  it('pickClosestMesh selects the nearer quad', () => {
    const scene: Scene = {
      primitives: [makeQuad('near', 0), makeQuad('far', 2)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const id = __pickerPickClosestMesh(scene, [0, 0, -5], [0, 0, 1]);
    expect(id).toBe('near');
  });

  it('pickClosestMesh returns null when no triangle is hit', () => {
    const scene: Scene = {
      primitives: [makeQuad('near', 0)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const id = __pickerPickClosestMesh(scene, [100, 100, -5], [0, 0, 1]);
    expect(id).toBeNull();
  });

  it('pickClosestMesh ignores non-mesh primitives', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'sphere',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
        makeQuad('quad', 2),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const id = __pickerPickClosestMesh(scene, [0, 0, -5], [0, 0, 1]);
    expect(id).toBe('quad');
  });

  it('mat4Inverse round-trip on identity is identity', () => {
    const m = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    const inv = __pickerMat4Inverse(m);
    expect(inv).not.toBeNull();
    if (inv == null) return;
    for (let i = 0; i < 16; i++) {
      expect(inv[i]).toBeCloseTo(m[i] ?? 0, 6);
    }
  });

  it('mat4Inverse returns null for a singular matrix', () => {
    const singular = new Float32Array(16); // all zeros
    expect(__pickerMat4Inverse(singular)).toBeNull();
  });
});
