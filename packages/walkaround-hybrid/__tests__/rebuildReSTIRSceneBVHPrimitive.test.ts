import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { packSceneFromCore, rebuildPrimitiveBlas } from '@vitrum/shared-bvh';
import * as THREE from 'three';
import {
  buildReSTIRSceneBVHFromVitrumScene,
  rebuildReSTIRSceneBVHPrimitive,
} from '../src/restir/sceneBvhFromCore.js';

function twoBoxScene(offsetB = 0): Scene {
  const positionsB = new Float32Array([
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
  ]);
  if (offsetB !== 0) {
    for (let i = 0; i < positionsB.length; i += 3) positionsB[i] = (positionsB[i] ?? 0) + offsetB;
  }
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'box-a',
        positions: new Float32Array([
          0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        ]),
        normals: new Float32Array(24).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
        indices: new Uint32Array([
          0, 1, 2, 4, 1, 2, 1, 5, 6, 5, 4, 6, 0, 2, 3, 2, 6, 7, 0, 1, 3, 1, 5, 3,
          3, 5, 7, 5, 6, 7, 0, 4, 3, 4, 6, 7,
        ]),
        material: { baseColor: [0.6, 0.6, 0.6], roughness: 0.5, metallic: 0 },
      },
      {
        kind: 'mesh',
        id: 'box-b',
        positions: positionsB,
        normals: new Float32Array(24).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
        indices: new Uint32Array([
          0, 1, 2, 4, 1, 2, 1, 5, 6, 5, 4, 6, 0, 2, 3, 2, 6, 7, 0, 1, 3, 1, 5, 3,
          3, 5, 7, 5, 6, 7, 0, 4, 3, 4, 6, 7,
        ]),
        material: { baseColor: [0.4, 0.4, 0.8], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function threeRootsFor(scene: Scene): THREE.Scene {
  const root = new THREE.Scene();
  for (const prim of scene.primitives) {
    if (prim.kind !== 'mesh') continue;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(prim.positions.slice(), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(prim.normals.slice(), 3));
    if (prim.indices) geo.setIndex(Array.from(prim.indices));
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xffffff }));
    mesh.name = prim.id;
    root.add(mesh);
  }
  return root;
}

describe('rebuildReSTIRSceneBVHPrimitive', () => {
  it('preserves buffer sizes when BLAS splice applies', () => {
    const scene = twoBoxScene();
    const roots = threeRootsFor(scene);
    const buffers = buildReSTIRSceneBVHFromVitrumScene(scene, [roots]);
    expect(buffers.scenePack).toBeDefined();

    const moved = twoBoxScene(0.05);
    const rebuilt = rebuildReSTIRSceneBVHPrimitive(moved, 'box-b', [roots], buffers);
    if ('ok' in rebuilt && rebuilt.ok === false) {
      throw new Error(rebuilt.reason);
    }
    expect(rebuilt.bvhPositions.byteLength).toBe(buffers.bvhPositions.byteLength);
    expect(rebuilt.scenePack?.triangleCount).toBe(buffers.scenePack?.triangleCount);

    const blas = rebuildPrimitiveBlas(moved, 'box-b', buffers.scenePack!, {
      tlas: true,
      resolveMaterialId: () => 0,
    });
    expect(blas.ok).toBe(true);
    if (blas.ok) expect(blas.strategy).toBe('splice');
  });
});
