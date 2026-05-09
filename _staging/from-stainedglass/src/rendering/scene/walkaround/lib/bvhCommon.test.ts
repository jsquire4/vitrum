/**
 * Regression tests for the Tier 2 shared `buildSceneBVH` core.
 *
 * Promoted from per-branch `buildSceneBvhEmitters.test.ts` files (DDGI's
 * `49f4c5b`, RC's `0e67583`) — both branches independently re-derived the
 * same single-root + reorder-invariant matId invariants that this module
 * canonicalises.
 *
 * Two interlocking concerns guarded:
 *
 * (a) SINGLE-ROOT INVARIANT. The WGSL `bvhIntersectFirstHit` traversal
 *     used by every walkaround branch takes a single root buffer pointer
 *     (`bvh._roots[0]`). By default three-mesh-bvh builds ONE root per
 *     material group (one root per source mesh in our case), so the
 *     traversal would see only the first source mesh's triangles — every
 *     other mesh invisible to ray casting. The fix collapses all groups
 *     into one before the BVH build so MeshBVH produces a single unified
 *     root.
 *
 * (b) PER-TRIANGLE matId REORDER SAFETY. Once groups are collapsed,
 *     MeshBVH's SAH partition freely scatters triangles spatially across
 *     the merged index buffer. A slot-based per-triangle matId lookup
 *     (using stale `merged.groups`) silently corrupts thousands of
 *     triangles. The fix snapshots a per-vertex matId map BEFORE the
 *     BVH build (vertices are NOT reordered — StaticGeometryGenerator
 *     concatenates each source mesh's vertex range contiguously, so each
 *     merged vertex belongs to exactly one source mesh forever) and uses
 *     it post-build to derive `materialIds[t] = vertexMatId[indices[t*3]]`.
 *
 * Pre-fix (multi-root, no collapse): assertion 1 fails (multi-root).
 * If multi-root were "fixed" by collapsing groups WITHOUT also snapshotting
 * per-vertex matIds, assertion 2 fails (~thousands of mismatches because
 * the SAH reorder scatters glass and floor tris across each other's slot
 * ranges). Only the combined fix passes both.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { faceToGeometry } from '@/geometry/triangulation/faceToGeometry';
import {
  buildSceneBVH,
  type SceneBVHCommonResult,
} from '@/rendering/scene/walkaround/lib/bvhCommon';
import type { VertexData } from '@/types/geometry';

// MeshBVH exposes _roots at runtime but it's not in the .d.ts. We need
// the runtime field to assert the root count.
interface BvhWithRoots {
  _roots: ArrayBuffer[];
}

function hexBoundary(cx: number, cy: number, r: number): VertexData[] {
  const verts: VertexData[] = [];
  for (let i = 0; i < 6; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 3;
    verts.push({
      id: `v${i}`,
      x: cx + r * Math.cos(a),
      y: cy + r * Math.sin(a),
      halfEdgeId: null,
    });
  }
  return verts;
}

interface SceneBuildResult {
  scene: THREE.Scene;
  meshOrder: THREE.Mesh[];
  /**
   * Per-source-mesh material kind, in mesh-add order. Each entry is
   * 'glass' (transmission 0.7) or 'floor' (transmission 0). The merged
   * geometry concatenates source meshes in this order, so we can derive
   * the per-vertex expected kind from each mesh's position-attribute count.
   */
  expectedKindPerMesh: ('glass' | 'floor')[];
  /** Vertex count per source mesh, in the same order. */
  vertCountPerMesh: number[];
}

function buildHoneycombScene(opts: { includeFloor: boolean }): SceneBuildResult {
  const scene = new THREE.Scene();
  const meshOrder: THREE.Mesh[] = [];
  const expectedKindPerMesh: ('glass' | 'floor')[] = [];
  const vertCountPerMesh: number[] = [];
  const radius = 1.5;
  const wHex = Math.sqrt(3) * radius;
  const h = 1.5 * radius;
  const depthInches = 3.0 / 25.4;
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 5; col++) {
      const ccx = col * wHex + (row % 2 === 0 ? 0 : wHex / 2);
      const ccy = row * h;
      const boundary = hexBoundary(ccx, ccy, radius);
      const geo = faceToGeometry(boundary, depthInches, false);
      if (!geo) continue;
      const mat = new THREE.MeshPhysicalMaterial({ transmission: 0.7 });
      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      meshOrder.push(mesh);
      expectedKindPerMesh.push('glass');
      vertCountPerMesh.push((geo.attributes.position as THREE.BufferAttribute).count);
    }
  }
  if (opts.includeFloor) {
    const ROOM_W = 192, ROOM_D = 168, FLOOR_Y = -64;
    const floorGeo = new THREE.PlaneGeometry(ROOM_W, ROOM_D, 32, 32);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x808080 });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.position.set(0, FLOOR_Y, 84);
    floorMesh.rotation.set(-Math.PI / 2, 0, 0);
    floorMesh.name = 'surface_floor_living';
    scene.add(floorMesh);
    meshOrder.push(floorMesh);
    expectedKindPerMesh.push('floor');
    vertCountPerMesh.push((floorGeo.attributes.position as THREE.BufferAttribute).count);
  }
  scene.updateMatrixWorld(true);
  return { scene, meshOrder, expectedKindPerMesh, vertCountPerMesh };
}

function classifyMaterial(mat: THREE.Material): 'glass' | 'floor' {
  const phys = mat as THREE.MeshPhysicalMaterial;
  return (phys.transmission ?? 0) > 0.5 ? 'glass' : 'floor';
}

/**
 * For every triangle slot in the post-build merged geometry, count how
 * many have a CORRECT material assignment.
 *
 * "Correct" means: the source mesh whose vertices are physically at
 * `indices[t*3..t*3+3]` (looked up via the pre-build mesh-vertex-range
 * map) has the same kind ('glass' vs 'floor') as the LUT entry at
 * `materials[triMaterialId[t]]`.
 */
function countCorrectlyClassifiedTriangles(
  result: SceneBVHCommonResult,
  expectedKindPerMesh: ('glass' | 'floor')[],
  vertCountPerMesh: number[],
): { correct: number; total: number } {
  // Build a vertex → expected-kind map using the source-mesh vertex
  // ranges. StaticGeometryGenerator concatenates each source mesh's
  // vertex range contiguously in input order, so this is exact.
  const totalVerts = vertCountPerMesh.reduce((a, b) => a + b, 0);
  const expectedKindPerVertex = new Array<'glass' | 'floor'>(totalVerts);
  let cursor = 0;
  for (let m = 0; m < expectedKindPerMesh.length; m++) {
    const kind = expectedKindPerMesh[m]!;
    const vc = vertCountPerMesh[m]!;
    for (let i = 0; i < vc; i++) expectedKindPerVertex[cursor + i] = kind;
    cursor += vc;
  }

  const triCount = result.triMaterialId.length;
  let correct = 0;
  for (let t = 0; t < triCount; t++) {
    const v0 = result.indices[t * 3]!;
    const physicalKind = expectedKindPerVertex[v0]!;
    const matId = result.triMaterialId[t]!;
    const assignedKind = classifyMaterial(result.materials[matId]!);
    if (physicalKind === assignedKind) correct++;
  }
  return { correct, total: triCount };
}

function countGlassTris(result: SceneBVHCommonResult): number {
  let count = 0;
  for (let t = 0; t < result.triMaterialId.length; t++) {
    if (
      classifyMaterial(result.materials[result.triMaterialId[t]!]!) === 'glass'
    ) {
      count++;
    }
  }
  return count;
}

describe('buildSceneBVH — single-root + per-tri matId survives BVH index reorder', () => {
  it('20 hex panels alone — single-root BVH, all glass tris correctly classified', () => {
    const { scene, expectedKindPerMesh, vertCountPerMesh } =
      buildHoneycombScene({ includeFloor: false });
    const result = buildSceneBVH(scene);

    // Concern (a): single root.
    const internal = result.bvh as unknown as BvhWithRoots;
    expect(internal._roots.length).toBe(1);

    // Concern (b): per-tri matId reorder safety.
    const { correct, total } = countCorrectlyClassifiedTriangles(
      result, expectedKindPerMesh, vertCountPerMesh,
    );
    expect(correct).toBe(total);
    expect(total).toBeGreaterThan(150); // ~400 tris for 20 extruded hex cells
  });

  it('20 hex panels + tessellated floor — single-root BVH with 100% correct matId assignment', () => {
    const { scene, expectedKindPerMesh, vertCountPerMesh } =
      buildHoneycombScene({ includeFloor: true });
    const result = buildSceneBVH(scene);

    const internal = result.bvh as unknown as BvhWithRoots;
    expect(internal._roots.length).toBe(1);

    const { correct, total } = countCorrectlyClassifiedTriangles(
      result, expectedKindPerMesh, vertCountPerMesh,
    );
    expect(correct).toBe(total);
    expect(total).toBeGreaterThan(2000); // 400 glass + 2048 floor
  });

  it('glass-triangle count matches floor-less baseline when floor is present', () => {
    const noFloor = buildSceneBVH(buildHoneycombScene({ includeFloor: false }).scene);
    const withFloor = buildSceneBVH(buildHoneycombScene({ includeFloor: true }).scene);
    expect(countGlassTris(withFloor)).toBe(countGlassTris(noFloor));
  });
});

describe('buildSceneBVH — positionStride 3 vs 4 layout', () => {
  it('default stride is 3 (raster / TSL path) and produces 12-byte-per-vertex layout', () => {
    const { scene, vertCountPerMesh } = buildHoneycombScene({ includeFloor: false });
    const result = buildSceneBVH(scene);

    expect(result.positionStrideFloats).toBe(3);
    const expectedVerts = vertCountPerMesh.reduce((a, b) => a + b, 0);
    expect(result.positions.length).toBe(expectedVerts * 3);
    expect(result.normals.length).toBe(expectedVerts * 3);
  });

  it('positionStride: 4 produces 16-byte-per-vertex layout with .w zero-filled', () => {
    const { scene, vertCountPerMesh } = buildHoneycombScene({ includeFloor: false });
    const result = buildSceneBVH(scene, { positionStride: 4 });

    expect(result.positionStrideFloats).toBe(4);
    const expectedVerts = vertCountPerMesh.reduce((a, b) => a + b, 0);
    expect(result.positions.length).toBe(expectedVerts * 4);
    expect(result.normals.length).toBe(expectedVerts * 4);

    // .w slot is zero-filled — caller is expected to pack UV / color
    // into .w as a sibling post-process if needed (ReSTIR's pattern).
    for (let i = 0; i < expectedVerts; i++) {
      expect(result.positions[i * 4 + 3]).toBe(0);
      expect(result.normals[i * 4 + 3]).toBe(0);
    }
  });

  it('positionStride: 4 .xyz values match positionStride: 3 raw values', () => {
    // Critical correctness: stride-4 must contain the same vertex
    // positions in .xyz as stride-3 contains in .xyz. This is what makes
    // the stride-4 buffer a drop-in replacement for ReSTIR's existing
    // WGSL `bvhIntersectFirstHit` (which reads positions[idx*4 + 0..2]
    // via `array<vec3f>`).
    const { scene } = buildHoneycombScene({ includeFloor: false });
    const r3 = buildSceneBVH(scene, { positionStride: 3 });
    // Note: building twice may produce different SAH orderings on the
    // index buffer, but the position attribute is the merged-vertex
    // buffer and is BVH-invariant — same vertex layout regardless of
    // build order. Build a fresh scene for the second build to avoid
    // reusing the BVH-mutated index buffer (it doesn't affect positions
    // but keeps the test hermetic).
    const { scene: scene2 } = buildHoneycombScene({ includeFloor: false });
    const r4 = buildSceneBVH(scene2, { positionStride: 4 });

    const vertCount = r3.positions.length / 3;
    expect(r4.positions.length / 4).toBe(vertCount);

    for (let i = 0; i < vertCount; i++) {
      expect(r4.positions[i * 4 + 0]).toBe(r3.positions[i * 3 + 0]);
      expect(r4.positions[i * 4 + 1]).toBe(r3.positions[i * 3 + 1]);
      expect(r4.positions[i * 4 + 2]).toBe(r3.positions[i * 3 + 2]);
    }
  });

  it('triMaterialId classification is independent of positionStride choice', () => {
    const { scene: s3, expectedKindPerMesh: e3, vertCountPerMesh: v3 } =
      buildHoneycombScene({ includeFloor: true });
    const r3 = buildSceneBVH(s3, { positionStride: 3 });

    const { scene: s4, expectedKindPerMesh: e4, vertCountPerMesh: v4 } =
      buildHoneycombScene({ includeFloor: true });
    const r4 = buildSceneBVH(s4, { positionStride: 4 });

    const c3 = countCorrectlyClassifiedTriangles(r3, e3, v3);
    const c4 = countCorrectlyClassifiedTriangles(r4, e4, v4);
    expect(c3.correct).toBe(c3.total);
    expect(c4.correct).toBe(c4.total);
    expect(c3.total).toBe(c4.total);
  });
});

describe('buildSceneBVH — BVH-node buffer is single-root and 32-byte-per-node', () => {
  it('bvhNodes buffer length is a multiple of 8 floats (= 32 bytes per BVHNode)', () => {
    const { scene } = buildHoneycombScene({ includeFloor: true });
    const result = buildSceneBVH(scene);
    // BVHNode = 6 f32 (bounds) + 2 u32 (rightChildOrTriOffset, splitAxis)
    //        = 8 × 4 bytes = 32 bytes.
    // The Float32Array view exposes that as `length / 8` nodes.
    expect(result.bvhNodes.length % 8).toBe(0);
    expect(result.bvhNodes.length).toBeGreaterThan(0);
  });
});

describe('buildSceneBVH — proxyMeshNames substitution', () => {
  it('substitutes named meshes with 1×1 proxy plane during BVH build but restores afterwards', () => {
    const { scene } = buildHoneycombScene({ includeFloor: true });

    // Capture the floor mesh's geometry identity BEFORE the build.
    let floorMesh: THREE.Mesh | null = null;
    scene.traverse((obj) => {
      if (obj.name === 'surface_floor_living') floorMesh = obj as THREE.Mesh;
    });
    expect(floorMesh).not.toBeNull();
    const originalGeo = (floorMesh as unknown as THREE.Mesh).geometry;
    const originalTriCount =
      ((originalGeo.index?.count ?? 0) / 3) ||
      ((originalGeo.attributes['position'] as THREE.BufferAttribute).count / 3);
    expect(originalTriCount).toBeGreaterThan(1000); // 32×32 = 2048 tris

    const proxiedResult = buildSceneBVH(scene, {
      proxyMeshNames: new Set(['surface_floor_living']),
    });

    // Post-build: visual geometry must be restored to the original
    // (32×32) — the visual mesh must NOT be permanently mutated.
    expect((floorMesh as unknown as THREE.Mesh).geometry).toBe(originalGeo);

    // The merged BVH should walk far fewer floor tris than the visual
    // mesh has (2 proxy tris vs 2048 visual tris) — so total tri count
    // is dominated by the 20 hex panels, not the floor.
    const baselineNoFloor = buildSceneBVH(
      buildHoneycombScene({ includeFloor: false }).scene,
    );
    const proxiedTriCount = proxiedResult.indices.length / 3;
    const baselineTriCount = baselineNoFloor.indices.length / 3;
    // Proxy adds 2 floor tris on top of the panel tris.
    expect(proxiedTriCount).toBe(baselineTriCount + 2);
  });

  it('without proxyMeshNames, dense floor tris are walked verbatim', () => {
    const { scene } = buildHoneycombScene({ includeFloor: true });
    const result = buildSceneBVH(scene); // no proxy
    const triCount = result.indices.length / 3;
    // 20 hex panels (~400 tris) + 32×32 floor (2048 tris) >> 1000.
    expect(triCount).toBeGreaterThan(2000);
  });
});

describe('buildSceneBVH — empty / filtered scene produces valid empty result', () => {
  it('returns a usable result when no meshes match the filter', () => {
    const scene = new THREE.Scene();
    // Add a Light (non-Mesh) so traversal has something to visit.
    scene.add(new THREE.DirectionalLight());
    scene.updateMatrixWorld(true);

    const result = buildSceneBVH(scene);
    expect(result.materials.length).toBe(0);
    expect(result.indices.length).toBe(3); // sentinel triangle
    expect(result.positions.length).toBeGreaterThan(0);
    expect(result.bvhNodes.length).toBeGreaterThan(0);
  });
});
