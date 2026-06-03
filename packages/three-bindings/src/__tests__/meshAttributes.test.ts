/**
 * meshAttributes.test.ts — P1 mesh-contract additions: convertMesh now reads
 * vertex colors and the 2nd UV channel (uv1, with uv2 fallback) instead of
 * dropping them at the door.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { convertMesh } from '../mesh.js';

function triGeometry(): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), 3));
  return geo;
}

describe('convertMesh — vertex colors + 2nd UV (P1)', () => {
  it('reads color + uv1 attributes', () => {
    const geo = triGeometry();
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1]), 2));
    geo.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array([0.5, 0.5, 1, 1, 0, 0]), 2));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 3));
    const prim = convertMesh(new THREE.Mesh(geo, new THREE.MeshStandardMaterial()));
    expect(prim.uvs).toBeInstanceOf(Float32Array);
    expect(prim.uv1).toEqual(new Float32Array([0.5, 0.5, 1, 1, 0, 0]));
    expect(prim.colors).toEqual(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
  });

  it('falls back to uv2 for the 2nd UV set (older THREE geometry)', () => {
    const geo = triGeometry();
    geo.setAttribute('uv2', new THREE.BufferAttribute(new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 2));
    const prim = convertMesh(new THREE.Mesh(geo, new THREE.MeshStandardMaterial()));
    expect(prim.uv1).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]));
  });

  it('omits uv1 / colors when absent', () => {
    const prim = convertMesh(new THREE.Mesh(triGeometry(), new THREE.MeshStandardMaterial()));
    expect(prim.uv1).toBeUndefined();
    expect(prim.colors).toBeUndefined();
  });
});
