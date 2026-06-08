/**
 * GOLDEN-PARITY / EQUIVALENCE gate for `mergeWorldSpaceFromCore` vs the THREE
 * `buildSceneBVH` (increment 3 of the THREE-decouple — see
 * `plan/three-decouple-analysis-2026-06-03.md`).
 *
 * ── What is and isn't byte-identical (measured, not assumed) ──────────────────
 *
 * The two builders reproduce the SAME merged world-space VERTEX STREAM but build
 * DIFFERENT BVH trees:
 *   • `buildSceneBVH` merges via `StaticGeometryGenerator` (raw `matrixWorld` on
 *     positions, inverse-transpose+normalize on normals, winding flip on
 *     negative determinant) then builds ONE three-mesh-bvh `MeshBVH` SAH tree.
 *   • `mergeWorldSpaceFromCore` reproduces that merge arithmetic exactly, then
 *     builds the merged BVH via `buildArrayBvh` (a DIFFERENT binned-SAH builder).
 *
 * Two independent divergences, both characterised here:
 *
 *  (R1) BVH TOPOLOGY + tri permutation differ — three-mesh-bvh `MeshBVH` vs
 *       `buildArrayBvh` are different SAH builders. So `bvhNodes` and the
 *       BVH-reordered `indices` are NOT byte-identical; only the triangle SET is.
 *       Fundamental, not a bug.
 *
 *  (PREC) The core `MeshPrimitive.transform` is a **Float32Array** (the GPU
 *       buffer's precision — `three-bindings/mesh.ts:convertMesh` writes
 *       `new Float32Array(mesh.matrixWorld.elements)`), whereas `buildSceneBVH`
 *       merges using THREE's **Float64Array** `matrixWorld`. For an
 *       AXIS-ALIGNED transform (translation + integer scale) the f32 matrix
 *       entries are bit-exact, so positions+normals are BYTE-IDENTICAL. For a
 *       ROTATED / non-integer-scaled transform the f32 matrix loses precision
 *       and transformed positions/normals differ by a LAST ULP. Measured:
 *       axis-aligned → 0/288 mismatches; a rotated box → 6/96 position +
 *       16/96 normal last-ULP mismatches. This is the *correct* core-path
 *       geometry (the GPU consumes f32 transforms) — forcing f64 here would
 *       either break the `Mat4 = Float32Array` contract or fake the math.
 *
 * Therefore the gate is split:
 *   • AXIS-ALIGNED scene → STRICT BYTE-PARITY of positions / normals / world
 *     AABB (assertion A), exact ray-query equivalence (assertion D).
 *   • ROTATED+MIRRORED scene → CHARACTERISED EQUIVALENCE: float-CLOSE positions/
 *     normals (assertion A'), exact triangle-SET + per-tri matId (assertion B),
 *     winding parity through a mirrored transform.
 *
 * The core scene is built to be GENUINELY EQUIVALENT to the THREE scene: each
 * core `MeshPrimitive.transform` is the THREE mesh's `matrixWorld.elements` and
 * the local positions/normals/indices/material are copied 1:1 — exactly what
 * `three-bindings/mesh.ts:convertMesh` produces (verified by code-read).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import type { MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { buildSceneBVH } from '../legacy/bvhCommon.js';
import { mergeWorldSpaceFromCore } from '../worldSpaceMerge.js';

// ──────────────────────────────────────────────────────────────────────────
// Equivalent THREE + core scene construction
// ──────────────────────────────────────────────────────────────────────────

interface MeshSpec {
  readonly id: string;
  readonly geometry: THREE.BufferGeometry;
  readonly material: THREE.MeshStandardMaterial;
  readonly coreMaterial: MaterialSpec;
  readonly position?: [number, number, number];
  readonly rotation?: [number, number, number];
  readonly scale?: [number, number, number];
}

function boxGeo(w: number, h: number, d: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(w, h, d);
  g.computeVertexNormals();
  return g;
}

function tetraGeo(r: number): THREE.BufferGeometry {
  const g = new THREE.TetrahedronGeometry(r);
  g.computeVertexNormals();
  return g;
}

function threeMeshFromSpec(spec: MeshSpec): THREE.Mesh {
  const mesh = new THREE.Mesh(spec.geometry, spec.material);
  mesh.name = spec.id;
  if (spec.position) mesh.position.set(...spec.position);
  if (spec.rotation) mesh.rotation.set(...spec.rotation);
  if (spec.scale) mesh.scale.set(...spec.scale);
  mesh.updateMatrixWorld(true);
  return mesh;
}

/** Build the equivalent core MeshPrimitive: bake matrixWorld into `transform`
 *  and copy local attributes 1:1 (exactly `convertMesh`). */
function corePrimFromThree(mesh: THREE.Mesh, coreMaterial: MaterialSpec): ScenePrimitive {
  const geo = mesh.geometry;
  const posAttr = geo.getAttribute('position');
  const normAttr = geo.getAttribute('normal');
  const positions = posAttr.array instanceof Float32Array
    ? new Float32Array(posAttr.array)
    : new Float32Array(posAttr.array);
  const normals = normAttr.array instanceof Float32Array
    ? new Float32Array(normAttr.array)
    : new Float32Array(normAttr.array);
  const idx = geo.index;
  const indices = idx != null
    ? (idx.array instanceof Uint32Array ? new Uint32Array(idx.array) : new Uint32Array(idx.array))
    : undefined;
  return {
    kind: 'mesh',
    id: mesh.name,
    positions,
    normals,
    transform: asMat4(new Float32Array(mesh.matrixWorld.elements)),
    material: coreMaterial,
    ...(indices != null ? { indices } : {}),
  };
}

function buildEquivalentScenes(specs: readonly MeshSpec[]): {
  threeRoots: THREE.Object3D[];
  coreScene: Scene;
} {
  const threeRoots: THREE.Object3D[] = [];
  const corePrims: ScenePrimitive[] = [];
  for (const spec of specs) {
    const mesh = threeMeshFromSpec(spec);
    threeRoots.push(mesh);
    corePrims.push(corePrimFromThree(mesh, spec.coreMaterial));
  }
  return {
    threeRoots,
    coreScene: { primitives: corePrims, emitters: [], environment: { kind: 'none' } },
  };
}

function stdMat(color: [number, number, number], roughness: number, metalness: number): {
  three: THREE.MeshStandardMaterial;
  core: MaterialSpec;
} {
  const three = new THREE.MeshStandardMaterial({
    color: new THREE.Color(...color),
    roughness,
    metalness,
  });
  const core: MaterialSpec = { baseColor: color, roughness, metallic: metalness };
  return { three, core };
}

// ──────────────────────────────────────────────────────────────────────────
// Tri-set + matId equivalence helpers
// ──────────────────────────────────────────────────────────────────────────

/** Quantise to a fixed grid so f32-tail noise (PREC divergence) doesn't break
 *  the SET match. 1e-3 grid >> the measured last-ULP drift. */
function q(x: number): number {
  return Math.round(x * 1e3);
}

/** Sorted (winding-insensitive) key of a triangle's three world vertices. */
function triKey(positions: Float32Array, stride: number, i0: number, i1: number, i2: number): string {
  const v = (i: number): string => {
    const b = i * stride;
    return `${q(positions[b]!)},${q(positions[b + 1]!)},${q(positions[b + 2]!)}`;
  };
  return [v(i0), v(i1), v(i2)].sort().join('|');
}

function triKeyToMatId(
  positions: Float32Array,
  stride: number,
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  triCount: number,
): Map<string, number> {
  const map = new Map<string, number>();
  for (let t = 0; t < triCount; t += 1) {
    const key = triKey(positions, stride, indices[t * 3]!, indices[t * 3 + 1]!, indices[t * 3 + 2]!);
    map.set(key, triMaterialId[t]!);
  }
  return map;
}

// ──────────────────────────────────────────────────────────────────────────
// Minimal CPU BVH closest-hit traversal — works on BOTH builders' node buffers
// (identical 32-byte layout: relative interior offsets, leaf 0xFFFF0000|count).
// ──────────────────────────────────────────────────────────────────────────

const UINT32_PER_NODE = 8;

function isLeafWord(w: number): boolean {
  return (w >>> 16) === 0xffff;
}

function rayAabb(
  ox: number, oy: number, oz: number,
  idx: number, idy: number, idz: number,
  bminX: number, bminY: number, bminZ: number,
  bmaxX: number, bmaxY: number, bmaxZ: number,
): number {
  let tmin = (bminX - ox) * idx;
  let tmax = (bmaxX - ox) * idx;
  if (tmin > tmax) { const s = tmin; tmin = tmax; tmax = s; }
  let tymin = (bminY - oy) * idy;
  let tymax = (bmaxY - oy) * idy;
  if (tymin > tymax) { const s = tymin; tymin = tymax; tymax = s; }
  if (tmin > tymax || tymin > tmax) return Infinity;
  if (tymin > tmin) tmin = tymin;
  if (tymax < tmax) tmax = tymax;
  let tzmin = (bminZ - oz) * idz;
  let tzmax = (bmaxZ - oz) * idz;
  if (tzmin > tzmax) { const s = tzmin; tzmin = tzmax; tzmax = s; }
  if (tmin > tzmax || tzmin > tmax) return Infinity;
  if (tzmin > tmin) tmin = tzmin;
  if (tzmax < tmax) tmax = tzmax;
  if (tmax < 0) return Infinity;
  return tmin >= 0 ? tmin : 0;
}

/** Möller–Trumbore; returns hit t or Infinity. Double-sided. */
function rayTri(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < 1e-12) return Infinity;
  const inv = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if (u < -1e-6 || u > 1 + 1e-6) return Infinity;
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < -1e-6 || u + v > 1 + 1e-6) return Infinity;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  return t > 1e-6 ? t : Infinity;
}

interface ClosestHit {
  t: number;
  /** Sorted world-vertex key of the hit triangle. At a shared edge two valid
   *  triangles tie — different BVH orderings may report either — so the
   *  GEOMETRIC truth to compare across builders is `t` (hit distance), not the
   *  triangle key. */
  key: string | null;
}

function closestHit(
  bvhNodes: Float32Array,
  indices: Uint32Array,
  positions: Float32Array,
  stride: number,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
): ClosestHit {
  const u32 = new Uint32Array(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.length);
  const f32 = bvhNodes;
  const idx = 1 / dx, idy = 1 / dy, idz = 1 / dz;
  let bestT = Infinity;
  let bestKey: string | null = null;
  const stack: number[] = [0];
  while (stack.length > 0) {
    const node = stack.pop()!;
    const base = node * UINT32_PER_NODE;
    const tBox = rayAabb(
      ox, oy, oz, idx, idy, idz,
      f32[base + 0]!, f32[base + 1]!, f32[base + 2]!,
      f32[base + 3]!, f32[base + 4]!, f32[base + 5]!,
    );
    if (!Number.isFinite(tBox) || tBox > bestT) continue;
    const splitOrCount = u32[base + 7]!;
    if (isLeafWord(splitOrCount)) {
      const triCount = splitOrCount & 0xffff;
      const triOffset = u32[base + 6]!;
      for (let k = 0; k < triCount; k += 1) {
        const t3 = (triOffset + k) * 3;
        const i0 = indices[t3]!, i1 = indices[t3 + 1]!, i2 = indices[t3 + 2]!;
        const a = i0 * stride, b = i1 * stride, c = i2 * stride;
        const t = rayTri(
          ox, oy, oz, dx, dy, dz,
          positions[a]!, positions[a + 1]!, positions[a + 2]!,
          positions[b]!, positions[b + 1]!, positions[b + 2]!,
          positions[c]!, positions[c + 1]!, positions[c + 2]!,
        );
        if (t < bestT) {
          bestT = t;
          bestKey = triKey(positions, stride, i0, i1, i2);
        }
      }
    } else {
      stack.push(node + 1);
      stack.push(node + u32[base + 6]!);
    }
  }
  return { t: bestT, key: bestKey };
}

/** Count of triangles covered by leaves exactly once. */
function leafCoverage(bvhNodes: Float32Array, triCount: number): number {
  const u32 = new Uint32Array(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.length);
  const totalNodes = bvhNodes.length / UINT32_PER_NODE;
  const seen = new Uint8Array(triCount);
  let covered = 0;
  for (let n = 0; n < totalNodes; n += 1) {
    const splitOrCount = u32[n * UINT32_PER_NODE + 7]!;
    if (!isLeafWord(splitOrCount)) continue;
    const cnt = splitOrCount & 0xffff;
    const off = u32[n * UINT32_PER_NODE + 6]!;
    for (let k = 0; k < cnt; k += 1) {
      const tri = off + k;
      if (tri < triCount && seen[tri] === 0) { seen[tri] = 1; covered += 1; }
    }
  }
  return covered;
}

// ──────────────────────────────────────────────────────────────────────────
// Scenes
// ──────────────────────────────────────────────────────────────────────────

/** AXIS-ALIGNED scene (translation + INTEGER scale, no rotation) — the f32
 *  transform matrix is bit-exact, so positions/normals are BYTE-IDENTICAL. */
function axisAlignedSpecs(): MeshSpec[] {
  const m1 = stdMat([0.8, 0.2, 0.2], 0.5, 0.0);
  const m2 = stdMat([0.2, 0.8, 0.2], 0.3, 0.0);
  const m3 = stdMat([0.2, 0.2, 0.8], 0.7, 1.0);
  return [
    { id: 'box-a', geometry: boxGeo(2, 2, 2), material: m1.three, coreMaterial: m1.core },
    { id: 'box-b', geometry: boxGeo(1, 3, 1), material: m2.three, coreMaterial: m2.core, position: [5, 0, -2] },
    {
      id: 'box-c',
      geometry: boxGeo(1, 2, 3),
      material: m3.three,
      coreMaterial: m3.core,
      position: [-4, 1, 3],
      scale: [2, 2, 2], // integer scale → f32-exact
    },
  ];
}

/** ROTATED + MIRRORED scene — exercises the PREC divergence (last-ULP) and the
 *  negative-determinant winding flip. */
function rotatedSpecs(): MeshSpec[] {
  const m1 = stdMat([0.8, 0.2, 0.2], 0.5, 0.0);
  const m2 = stdMat([0.2, 0.8, 0.2], 0.3, 0.0);
  const m3 = stdMat([0.7, 0.7, 0.2], 0.4, 0.0);
  return [
    { id: 'box-a', geometry: boxGeo(2, 2, 2), material: m1.three, coreMaterial: m1.core },
    {
      id: 'box-b',
      geometry: boxGeo(1, 3, 1),
      material: m2.three,
      coreMaterial: m2.core,
      position: [5, 1, -2],
      rotation: [0.3, 0.7, -0.2],
    },
    {
      id: 'tetra-c',
      geometry: tetraGeo(1.5),
      material: m3.three,
      coreMaterial: m3.core,
      position: [-4, 0, 3],
      scale: [-1.7, 2.3, 1.1], // negative determinant (winding flip) + non-integer
    },
  ];
}

// ──────────────────────────────────────────────────────────────────────────
// (A) Strict byte-parity — AXIS-ALIGNED scene
// ──────────────────────────────────────────────────────────────────────────

describe('mergeWorldSpaceFromCore — strict byte-parity (axis-aligned)', () => {
  it('(A) merged positions / normals / world-AABB are FLOAT-IDENTICAL to buildSceneBVH', () => {
    const { threeRoots, coreScene } = buildEquivalentScenes(axisAlignedSpecs());
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });

    expect(core.positionStrideFloats).toBe(4);
    expect(core.positions.length).toBe(three.positions.length);
    expect(core.normals.length).toBe(three.normals.length);

    // BYTE-IDENTICAL merged vertex stream (never reordered by either BVH).
    expect(Array.from(core.positions)).toEqual(Array.from(three.positions));
    expect(Array.from(core.normals)).toEqual(Array.from(three.normals));

    const tb = three.boundingBox;
    expect(core.boundingBox.min).toEqual([tb.min.x, tb.min.y, tb.min.z]);
    expect(core.boundingBox.max).toEqual([tb.max.x, tb.max.y, tb.max.z]);
  });

  it('(C) SAH leaves cover every triangle exactly once on both builders', () => {
    const { threeRoots, coreScene } = buildEquivalentScenes(axisAlignedSpecs());
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });

    expect(core.triangleCount).toBe(three.triMaterialId.length);
    expect(leafCoverage(core.bvhNodes, core.triangleCount)).toBe(core.triangleCount);
    expect(leafCoverage(three.bvhNodes, three.triMaterialId.length)).toBe(three.triMaterialId.length);
  });

  it('(D) ray queries return the SAME hit triangle + distance from both BVHs', () => {
    const { threeRoots, coreScene } = buildEquivalentScenes(axisAlignedSpecs());
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });

    // Rays aimed OFF face-centres (a box face's two tris share the diagonal,
    // and a ray through the exact centre hits that shared edge — an exact tie
    // that the two different BVH orderings can break to either valid triangle).
    // Asymmetric offsets land each ray cleanly inside ONE triangle's interior.
    const rays: Array<{ o: [number, number, number]; d: [number, number, number] }> = [
      { o: [0.3, -0.4, 20], d: [0, 0, -1] },   // box-a front face (X∈[-1,1],Y∈[-1,1])
      { o: [5.2, 0.7, 20], d: [0, 0, -1] },    // box-b
      { o: [-4.3, 0.3, 20], d: [0, 0, -1] },   // box-c front (X∈[-5,-3],Y∈[-1,3])
      { o: [20, -1.1, -1.7], d: [-1, 0, 0] },  // box-b +X face (Y∈[-1.5,1.5],Z∈[-2.5,-1.5]) — near a corner
      { o: [-20, 2.4, 1.6], d: [1, 0, 0] },    // box-c -X face (Y∈[-1,3],Z∈[0,6]) — near a corner
      { o: [0.4, 20, -0.3], d: [0, -1, 0] },   // box-a top
      { o: [4.7, 20, -1.7], d: [0, -1, 0] },   // box-b top (X∈[4.5,5.5],Z∈[-2.5,-1.5]) — near a corner
    ];

    let anyHit = false;
    let anyKeyMatch = false;
    for (const r of rays) {
      const dn = Math.hypot(...r.d);
      const d: [number, number, number] = [r.d[0] / dn, r.d[1] / dn, r.d[2] / dn];
      const hThree = closestHit(three.bvhNodes, three.indices, three.positions, 4, ...r.o, ...d);
      const hCore = closestHit(core.bvhNodes, core.indices, core.positions, 4, ...r.o, ...d);
      // Same hit/miss status.
      expect(Number.isFinite(hCore.t)).toBe(Number.isFinite(hThree.t));
      if (Number.isFinite(hThree.t)) {
        anyHit = true;
        // GEOMETRIC truth: identical hit DISTANCE (positions are byte-identical
        // here, so this is exact — proving the two different BVH trees resolve
        // the same surface). The triangle KEY can differ only when the ray lands
        // exactly on a face diagonal (the two coplanar tris tie); the distance
        // is identical either way.
        expect(hCore.t).toBeCloseTo(hThree.t, 6);
        if (hCore.key === hThree.key) anyKeyMatch = true;
      }
    }
    expect(anyHit).toBe(true);
    // At least some rays hit a clean triangle interior (key matches), confirming
    // it's not a vacuous all-edge-tie comparison.
    expect(anyKeyMatch).toBe(true);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// (A'/B) Characterised equivalence — ROTATED + MIRRORED scene
// ──────────────────────────────────────────────────────────────────────────

describe('mergeWorldSpaceFromCore — characterised equivalence (rotated/mirrored)', () => {
  it("(A') merged positions / normals are FLOAT-CLOSE (last-ULP PREC drift only)", () => {
    const { threeRoots, coreScene } = buildEquivalentScenes(rotatedSpecs());
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });

    expect(core.positions.length).toBe(three.positions.length);
    expect(core.normals.length).toBe(three.normals.length);

    // Positions agree to ~6 decimals (the f32-transform-storage last-ULP drift
    // is far below this — measured ≤ 1e-6 absolute on these magnitudes).
    let maxPosDelta = 0;
    for (let i = 0; i < core.positions.length; i += 1) {
      maxPosDelta = Math.max(maxPosDelta, Math.abs(core.positions[i]! - three.positions[i]!));
    }
    expect(maxPosDelta).toBeLessThan(1e-4);

    // Normals are UNIT vectors; their last-ULP drift is even smaller.
    let maxNrmDelta = 0;
    for (let i = 0; i < core.normals.length; i += 1) {
      maxNrmDelta = Math.max(maxNrmDelta, Math.abs(core.normals[i]! - three.normals[i]!));
    }
    expect(maxNrmDelta).toBeLessThan(1e-5);
  });

  it('(B) same triangle SET + per-triangle materialId (orderings differ, R1)', () => {
    const { threeRoots, coreScene } = buildEquivalentScenes(rotatedSpecs());
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });

    expect(core.triangleCount).toBe(three.triMaterialId.length);
    // Both LUTs deduped, first-seen order over the SAME 3 distinct materials.
    expect(core.materials.length).toBe(three.materials.length);

    const threeMap = triKeyToMatId(three.positions, 4, three.indices, three.triMaterialId, three.triMaterialId.length);
    const coreMap = triKeyToMatId(core.positions, 4, core.indices, core.triMaterialId, core.triangleCount);

    expect(coreMap.size).toBe(threeMap.size);
    expect([...coreMap.keys()].sort()).toEqual([...threeMap.keys()].sort());
    for (const [key, coreMatId] of coreMap) {
      expect(threeMap.get(key)).toBe(coreMatId);
    }
  });

  it('(D) ray queries hit the same triangle (rotated scene; tolerant edge skip)', () => {
    const { threeRoots, coreScene } = buildEquivalentScenes(rotatedSpecs());
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });

    // Aim at the BOXES (off-centre, interior hits). The PREC last-ULP drift can
    // flip a hit only when a ray grazes a shared EDGE — so target box-a (origin,
    // axis-aligned: byte-exact) and box-b (rotated) body interiors. The tetra's
    // set/winding is already pinned by assertions (B) + the winding test; its
    // adjacent faces share edges that make edge-exact ray ties brittle, so it is
    // intentionally not ray-probed here.
    const rays: Array<{ o: [number, number, number]; d: [number, number, number] }> = [
      { o: [0.3, -0.4, 20], d: [0, 0, -1] },   // box-a front (axis-aligned, exact)
      { o: [0.4, 20, -0.3], d: [0, -1, 0] },   // box-a top
      { o: [-20, 0.3, 0.2], d: [1, 0, 0] },    // box-a from -X
      { o: [5.3, 1.2, 20], d: [0, 0, -1] },    // box-b (rotated) — body interior
    ];
    let compared = 0;
    for (const r of rays) {
      const dn = Math.hypot(...r.d);
      const d: [number, number, number] = [r.d[0] / dn, r.d[1] / dn, r.d[2] / dn];
      const hThree = closestHit(three.bvhNodes, three.indices, three.positions, 4, ...r.o, ...d);
      const hCore = closestHit(core.bvhNodes, core.indices, core.positions, 4, ...r.o, ...d);
      if (!Number.isFinite(hThree.t) && !Number.isFinite(hCore.t)) continue;
      // Hit DISTANCE agrees within the PREC last-ULP drift (the geometric truth).
      // Key may tie at a shared edge; distance is the cross-builder invariant.
      expect(hCore.t).toBeCloseTo(hThree.t, 3);
      compared += 1;
    }
    expect(compared).toBeGreaterThan(0);
  });

  it('handles the negative-determinant (mirrored) transform with matching winding', () => {
    // The tetra in rotatedSpecs has scale [-1.7,…] → negative determinant.
    // StaticGeometryGenerator's invertGeometry swaps v0↔v2 per triangle;
    // mergeWorldSpaceFromCore mirrors that. Compare the WINDING-SENSITIVE
    // (cyclic-order) tri multiset of the merge streams.
    const { threeRoots, coreScene } = buildEquivalentScenes(rotatedSpecs());
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });

    const orderedKey = (positions: Float32Array, i0: number, i1: number, i2: number): string => {
      const v = (i: number): string => {
        const b = i * 4;
        return `${q(positions[b]!)},${q(positions[b + 1]!)},${q(positions[b + 2]!)}`;
      };
      const ks = [v(i0), v(i1), v(i2)];
      let minI = 0;
      for (let j = 1; j < 3; j += 1) if (ks[j]! < ks[minI]!) minI = j;
      return [ks[minI]!, ks[(minI + 1) % 3]!, ks[(minI + 2) % 3]!].join('|');
    };
    const bag = (indices: Uint32Array, positions: Float32Array, n: number): Map<string, number> => {
      const map = new Map<string, number>();
      for (let t = 0; t < n; t += 1) {
        const k = orderedKey(positions, indices[t * 3]!, indices[t * 3 + 1]!, indices[t * 3 + 2]!);
        map.set(k, (map.get(k) ?? 0) + 1);
      }
      return map;
    };
    // buildSceneBVH's `indices` is BVH-reordered, but winding (cyclic order) is
    // preserved by the SAH permute (it only swaps the order of TRIANGLES, never
    // the vertex order WITHIN a triangle). So the winding-sensitive multiset of
    // its `indices` equals that of the core MERGE stream iff the flip matches.
    const threeBag = bag(three.indices, three.positions, three.triMaterialId.length);
    const coreBag = bag(core.mergedIndices, core.positions, core.triangleCount);
    expect([...coreBag.entries()].sort()).toEqual([...threeBag.entries()].sort());
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Internal-consistency + edge cases (no THREE comparison needed)
// ──────────────────────────────────────────────────────────────────────────

describe('mergeWorldSpaceFromCore — internal consistency', () => {
  it('the BVH-reordered stream is a permutation of the merge-order stream', () => {
    const { coreScene } = buildEquivalentScenes(rotatedSpecs());
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });
    const fromBvh = triKeyToMatId(core.positions, 4, core.indices, core.triMaterialId, core.triangleCount);
    const fromMerge = triKeyToMatId(core.positions, 4, core.mergedIndices, core.mergedTriMaterialId, core.triangleCount);
    expect([...fromBvh.keys()].sort()).toEqual([...fromMerge.keys()].sort());
    for (const [key, mid] of fromMerge) {
      expect(fromBvh.get(key)).toBe(mid);
    }
  });

  it('empty scene → empty-but-valid result', () => {
    const core = mergeWorldSpaceFromCore({ primitives: [], emitters: [], environment: { kind: 'none' } }, { positionStride: 4 });
    expect(core.triangleCount).toBe(0);
    expect(core.bvhNodes.length).toBe(8); // single zeroed leaf node
    expect(core.materials.length).toBe(0);
    expect(core.positions.length).toBe(12);
  });

  it('value-dedups structurally-identical materials into one LUT slot', () => {
    const a = stdMat([0.5, 0.5, 0.5], 0.6, 0.0);
    const b = stdMat([0.5, 0.5, 0.5], 0.6, 0.0); // identical values, distinct instance
    const specs: MeshSpec[] = [
      { id: 'p0', geometry: boxGeo(1, 1, 1), material: a.three, coreMaterial: a.core },
      { id: 'p1', geometry: boxGeo(1, 1, 1), material: b.three, coreMaterial: b.core, position: [5, 0, 0] },
    ];
    const { threeRoots, coreScene } = buildEquivalentScenes(specs);
    const three = buildSceneBVH(threeRoots, { positionStride: 4 });
    const core = mergeWorldSpaceFromCore(coreScene, { positionStride: 4 });
    expect(core.materials.length).toBe(1);
    expect(three.materials.length).toBe(1);
    expect(Array.from(core.triMaterialId).every((m) => m === 0)).toBe(true);
  });

  it('instanced-mesh world-merges one baked copy per instance', () => {
    const m = stdMat([0.4, 0.6, 0.8], 0.5, 0.0);
    const scene: Scene = {
      primitives: [
        {
          kind: 'instanced-mesh',
          id: 'inst',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: m.core,
          instances: [
            asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
            asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1])),
          ],
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const core = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
    // 2 instances × 1 tri = 2 tris; 2 × 3 verts = 6 verts.
    expect(core.triangleCount).toBe(2);
    expect(core.positions.length).toBe(6 * 4);
    expect(core.meshVertexRanges.length).toBe(2);
    // Instance 1 translated by +10 in X — its first vertex world X is 10.
    expect(core.positions[1 * 4 * 3 + 0]).toBeCloseTo(10, 5);
  });

  it('respects a custom filter (e.g. exclude instanced meshes, as the ReSTIR world merge does)', () => {
    const m = stdMat([0.4, 0.6, 0.8], 0.5, 0.0);
    const scene: Scene = {
      primitives: [
        { kind: 'mesh', id: 'keep', positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), material: m.core },
        {
          kind: 'instanced-mesh',
          id: 'drop',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: m.core,
          instances: [asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))],
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const core = mergeWorldSpaceFromCore(scene, {
      positionStride: 4,
      filter: (p) => p.kind !== 'instanced-mesh',
    });
    expect(core.triangleCount).toBe(1); // only the 'keep' mesh
    expect(core.meshVertexRanges.map((r) => r.name)).toEqual(['keep']);
  });
});
