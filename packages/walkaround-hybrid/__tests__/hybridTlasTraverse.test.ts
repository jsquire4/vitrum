/**
 * TLAS production-path pins — CPU pack + hybrid BVH build (no GPU required).
 */
import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { buildTlas10InstThreeScene } from '../../../examples/shared/src/buildBenchmarkScenes.js';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import { packSceneFromCore, refitTlasTransforms } from '@vitrum/shared-bvh';
import * as THREE from 'three';
import {
  buildReSTIRSceneBVHFromVitrumScene,
  resolveReSTIRBvhMode,
} from '../src/restir/sceneBvhFromCore.js';

function twoOffsetMeshes(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'wall-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.2], roughness: 0.5, metallic: 0 },
        transform: asMat4(new Float32Array([
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1,
        ])),
      },
      {
        kind: 'mesh',
        id: 'wall-b',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.2, 0.2, 0.8], roughness: 0.5, metallic: 0 },
        transform: asMat4(new Float32Array([
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
        ])),
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function threeRootsFor(scene: Scene): THREE.Scene {
  const root = new THREE.Scene();
  for (const prim of scene.primitives) {
    if (prim.kind === 'instanced-mesh') continue;
    if (prim.kind !== 'mesh') continue;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(prim.positions.slice(), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(prim.normals.slice(), 3));
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xffffff }));
    mesh.uuid = prim.id;
    mesh.name = prim.id;
    if (prim.transform) mesh.matrix.fromArray(prim.transform);
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrixWorld(true);
    root.add(mesh);
  }
  return root;
}

describe('hybrid TLAS production path', () => {
  it('auto-selects TLAS for multi-mesh and instanced scenes', () => {
    expect(resolveReSTIRBvhMode(twoOffsetMeshes())).toBe('tlas');
    const threeScene = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.2, 0.2, 0.2),
      new THREE.MeshPhysicalMaterial(),
      10,
    );
    const m = new THREE.Matrix4();
    for (let i = 0; i < 10; i += 1) {
      m.makeTranslation(i * 0.3, 0, 0);
      im.setMatrixAt(i, m);
    }
    im.instanceMatrix.needsUpdate = true;
    threeScene.add(im);
    const vitrum = sceneFromThreeJS(threeScene);
    expect(resolveReSTIRBvhMode(vitrum)).toBe('tlas');
    expect(vitrum.primitives.some((p) => p.kind === 'instanced-mesh')).toBe(true);
  });

  it('buildReSTIRSceneBVHFromVitrumScene uploads TLAS buffers for two meshes', () => {
    const scene = twoOffsetMeshes();
    const roots = threeRootsFor(scene);
    const buffers = buildReSTIRSceneBVHFromVitrumScene(scene, [roots]);
    expect(buffers.bvhMode).toBe('tlas');
    expect(buffers.tlas?.nodeCount).toBeGreaterThan(0);
    expect(buffers.primitiveTlasBindings).toHaveLength(2);
  });

  it('TLAS transform refit updates instance matrices (production updatePrimitive path)', () => {
    const scene = twoOffsetMeshes();
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const moved: Scene = {
      ...scene,
      primitives: scene.primitives.map((p) => {
        if (p.kind !== 'mesh' || p.id !== 'wall-b') return p;
        return {
          ...p,
          transform: asMat4(new Float32Array([
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1,
          ])),
        };
      }),
    };
    const refit = refitTlasTransforms(moved, packed.primitiveTlasBindings, {
      tlasNodes: packed.tlasNodes,
      tlasInstanceIndices: packed.tlasInstanceIndices,
      tlasBlasRoots: packed.tlasBlasRoots,
      tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
    });
    expect(refit.ok).toBe(true);
    if (!refit.ok) return;
    expect(refit.tlasInstanceLocalToWorld[12 + 16]).toBeCloseTo(3, 4);
  });

  it('PR-6 tlas10inst benchmark scene packs and builds ReSTIR buffers', () => {
    const threeScene = buildTlas10InstThreeScene();
    const vitrum = sceneFromThreeJS(threeScene);
    expect(vitrum.primitives.length).toBeGreaterThanOrEqual(2);
    const packed = packSceneFromCore(vitrum, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.triangleCount).toBeGreaterThan(50);
    expect(packed.primitiveTlasBindings.some((b) => b.instanceCount === 10)).toBe(true);
    const buffers = buildReSTIRSceneBVHFromVitrumScene(vitrum, [threeScene]);
    expect(buffers.bvhMode).toBe('tlas');
    expect(buffers.tlas?.nodeCount).toBeGreaterThan(0);
  });

  it('InstancedMesh THREE scene converts to 10-instance TLAS pack', () => {
    const threeScene = new THREE.Scene();
    const im = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.2, 0.2, 0.2),
      new THREE.MeshPhysicalMaterial(),
      10,
    );
    const m = new THREE.Matrix4();
    for (let i = 0; i < 10; i += 1) {
      m.makeTranslation(i * 0.3, 0, 0);
      im.setMatrixAt(i, m);
    }
    im.instanceMatrix.needsUpdate = true;
    threeScene.add(im);
    const vitrum = sceneFromThreeJS(threeScene);
    const inst = vitrum.primitives.find((p) => p.kind === 'instanced-mesh');
    expect(inst).toBeDefined();
    if (inst?.kind !== 'instanced-mesh') return;
    expect(inst.instances).toHaveLength(10);
    const packed = packSceneFromCore(vitrum, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.primitiveTlasBindings[0]?.instanceCount).toBe(10);
    expect(packed.tlasNodeCount).toBeGreaterThan(0);
  });
});
