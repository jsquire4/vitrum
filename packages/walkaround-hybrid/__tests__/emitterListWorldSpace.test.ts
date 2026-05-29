/**
 * Pins WHY `buffersFromScenePack` must run a SECOND `buildSceneBVH`
 * (`sharedWorld`) to feed `buildEmitterList`, rather than reusing the
 * already-packed `geo: ScenePackResult`.
 *
 * Investigation (2026-05-29): a dedup was proposed — feed `buildEmitterList`
 * from `geo` instead of rebuilding a parallel BVH. It is NOT safe. The two
 * builds produce emitter inputs in DIFFERENT coordinate spaces and DIFFERENT
 * triangle orderings:
 *
 *   • `packSceneFromCore` stores per-primitive BLAS positions in LOCAL/object
 *     space (it copies `primitive.positions` verbatim, scenePack.ts:750-752),
 *     with the world transform held separately in the TLAS instance matrices.
 *   • `buildSceneBVH` uses `StaticGeometryGenerator` with
 *     `applyWorldTransforms = true` (bvhCommon.ts:484-491), baking each mesh's
 *     `matrixWorld` into every vertex → WORLD-space positions.
 *
 * `buildEmitterList` derives triangle area, geometric/averaged face normal,
 * centroids and AABBs from these positions, applies a WORLD-space sun-direction
 * dot in `classifyTriangleEmitter`, and APPENDS world-space RectAreaLight tris
 * from `collectRectAreaLightEmitterTris`. So it fundamentally requires
 * world-space geometry. Feeding it `geo`'s local-space positions would place
 * every emitter triangle at the wrong world location whenever a mesh has a
 * non-identity transform — changing the emitter list, CDF, light tree, RNG
 * stratification, and the rendered image.
 *
 * These tests assert the emitter geometry is WORLD-space and DIVERGES from
 * `geo`'s local-space pack, so any future "just use geo" reroute breaks here.
 */

import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { packSceneFromCore } from '@vitrum/shared-bvh';
import * as THREE from 'three';
import { buildReSTIRSceneBVHFromVitrumScene } from '../src/restir/sceneBvhFromCore.js';

const EMITTER_FLOATS = 20; // 80-byte EmitterTri stride / 4

/** Single emissive triangle, positioned at the origin in LOCAL space. */
function emissiveTriScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'lamp',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint32Array([0, 1, 2]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/**
 * THREE root mirroring the scene, with the named mesh given an emissive
 * MeshStandardMaterial and an optional world translation applied so LOCAL
 * and WORLD space diverge.
 */
function threeRootsFor(scene: Scene, worldTranslate: THREE.Vector3): THREE.Scene {
  const root = new THREE.Scene();
  for (const prim of scene.primitives) {
    if (prim.kind !== 'mesh') continue;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(prim.positions.slice(), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(prim.normals.slice(), 3));
    if (prim.indices) geo.setIndex(Array.from(prim.indices));
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff });
    mat.emissive = new THREE.Color(1, 1, 1);
    mat.emissiveIntensity = 5;
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = prim.id;
    mesh.position.copy(worldTranslate);
    root.add(mesh);
  }
  root.updateMatrixWorld(true);
  return root;
}

function firstEmitterVertexA(buffers: { emitters: { cpuData: ArrayBuffer } }): [number, number, number] {
  const floats = new Float32Array(buffers.emitters.cpuData);
  return [floats[0]!, floats[1]!, floats[2]!];
}

describe('emitter list is built in world space (dedup is load-bearing)', () => {
  it('emitter triangle vertex follows the mesh world transform, not local space', () => {
    const scene = emissiveTriScene();
    const translate = new THREE.Vector3(10, 20, 30);
    const roots = threeRootsFor(scene, translate);

    const buffers = buildReSTIRSceneBVHFromVitrumScene(scene, [roots]);

    // Exactly one emissive triangle ⇒ exactly one emitter (no synthetic
    // placeholder, no extra rect-area-light tris).
    expect(buffers.emitterCount).toBe(1);

    // vertexA in the emitter list must be the LOCAL (0,0,0) shifted by the
    // world translation → (10,20,30). If the emitter list were sourced from
    // `geo` (local space) it would read (0,0,0) and this assertion fails.
    const [ax, ay, az] = firstEmitterVertexA(buffers);
    expect(ax).toBeCloseTo(10, 5);
    expect(ay).toBeCloseTo(20, 5);
    expect(az).toBeCloseTo(30, 5);
  });

  it('geo (packSceneFromCore) carries LOCAL-space positions — proves the divergence', () => {
    const scene = emissiveTriScene();
    // geo positions are local/object space regardless of any world transform.
    const geo = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    // First vertex is the local (0,0,0) — NOT the world (10,20,30) the
    // emitter list above resolved to. Same primitive, two different spaces.
    expect(geo.positions[0]).toBeCloseTo(0, 6);
    expect(geo.positions[1]).toBeCloseTo(0, 6);
    expect(geo.positions[2]).toBeCloseTo(0, 6);
  });

  it('identity transform: emitter vertex matches local (sanity baseline)', () => {
    const scene = emissiveTriScene();
    const roots = threeRootsFor(scene, new THREE.Vector3(0, 0, 0));
    const buffers = buildReSTIRSceneBVHFromVitrumScene(scene, [roots]);
    expect(buffers.emitterCount).toBe(1);
    const [ax, ay, az] = firstEmitterVertexA(buffers);
    expect(ax).toBeCloseTo(0, 5);
    expect(ay).toBeCloseTo(0, 5);
    expect(az).toBeCloseTo(0, 5);
  });
});

// Reference the stride constant so it documents the EmitterTri layout the
// decode above relies on (vertexA occupies floats 0..2 of the 20-float entry).
void EMITTER_FLOATS;
