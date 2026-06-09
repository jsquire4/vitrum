import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '@vitrum/shared-bvh';
import { packBvhTextureData, type BvhTextureData } from './bvhTextureAdapter.js';

// ─────────────────────────────────────────────────────────────────────────────
// GPU-FREE linchpin gate. We CPU-traverse the packed BVH textures using EXACTLY
// the fork GLSL's logic (relative right child, nodeIndex+1 left, 0xFFFF0000|count
// leaf word, index→position triangle fetch) and compare closest-hits against an
// independent brute-force over ALL triangles (Möller-Trumbore). A correct BVH
// returns identical closest-hits on any valid ray — this is the F-TLAS1/F-RC1
// stride-bug guard applied to the shared-bvh → WebGL2 texture path.
// ─────────────────────────────────────────────────────────────────────────────

const GREY: MaterialSpec = { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 };

function quad(id: string, v: [number, number, number][]): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([...v[0]!, ...v[1]!, ...v[2]!, ...v[3]!]),
    normals: new Float32Array(12),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
}

/** A spread of 12 triangles (6 quads at varied positions) → a real multi-node BVH. */
function multiNodeScene(): Scene {
  const n = -0.5;
  const p = 0.5;
  const at = (ox: number, oy: number, oz: number, id: string): MeshPrimitive =>
    quad(id, [[n + ox, n + oy, oz], [p + ox, n + oy, oz], [p + ox, p + oy, oz], [n + ox, p + oy, oz]]);
  return {
    primitives: [
      at(0, 0, 0, 'a'), at(3, 0, -1, 'b'), at(-3, 1, 1, 'c'),
      at(0, 4, 2, 'd'), at(2, -3, 0.5, 'e'), at(-2, -2, -1, 'f'),
    ],
    emitters: [],
    environment: { kind: 'none' },
  } as Scene;
}

// ── ray/triangle/AABB math (independent of the engine GLSL) ──────────────────
type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function rayTri(orig: V3, dir: V3, a: V3, b: V3, c: V3): number {
  const e1 = sub(b, a);
  const e2 = sub(c, a);
  const pv = cross(dir, e2);
  const det = dot(e1, pv);
  if (Math.abs(det) < 1e-12) return Infinity;
  const inv = 1 / det;
  const tv = sub(orig, a);
  const u = dot(tv, pv) * inv;
  if (u < -1e-7 || u > 1 + 1e-7) return Infinity;
  const qv = cross(tv, e1);
  const v = dot(dir, qv) * inv;
  if (v < -1e-7 || u + v > 1 + 1e-7) return Infinity;
  const t = dot(e2, qv) * inv;
  return t > 1e-6 ? t : Infinity;
}

function rayAabb(orig: V3, invDir: V3, lo: V3, hi: V3): boolean {
  let tmin = -Infinity;
  let tmax = Infinity;
  for (let i = 0; i < 3; i += 1) {
    const t1 = (lo[i]! - orig[i]!) * invDir[i]!;
    const t2 = (hi[i]! - orig[i]!) * invDir[i]!;
    tmin = Math.max(tmin, Math.min(t1, t2));
    tmax = Math.min(tmax, Math.max(t1, t2));
  }
  return tmax >= Math.max(tmin, 0);
}

function vertAt(d: BvhTextureData, vi: number): V3 {
  const s = vi * 4;
  return [d.position[s]!, d.position[s + 1]!, d.position[s + 2]!];
}

/** CPU traversal of the packed textures using the GLSL leaf/child decoding. */
function traverseBvh(d: BvhTextureData, orig: V3, dir: V3): { t: number; tri: number } {
  const invDir: V3 = [1 / dir[0], 1 / dir[1], 1 / dir[2]];
  let best = Infinity;
  let bestTri = -1;
  const stack = [0];
  while (stack.length > 0) {
    const i = stack.pop()!;
    const lo: V3 = [d.bounds[2 * i * 4]!, d.bounds[2 * i * 4 + 1]!, d.bounds[2 * i * 4 + 2]!];
    const hi: V3 = [d.bounds[(2 * i + 1) * 4]!, d.bounds[(2 * i + 1) * 4 + 1]!, d.bounds[(2 * i + 1) * 4 + 2]!];
    if (!rayAabb(orig, invDir, lo, hi)) continue;
    const word7 = d.contents[i * 4]!;
    const word6 = d.contents[i * 4 + 1]!;
    // unsigned-safe leaf test (JS `&` is signed-int32; matches shared-bvh buildArrayBvh:50)
    const isLeaf = (word7 >>> 16) === 0xffff;
    if (isLeaf) {
      const count = word7 & 0xffff;
      for (let t = word6; t < word6 + count; t += 1) {
        const ti = t * 4;
        const hit = rayTri(orig, dir, vertAt(d, d.index[ti]!), vertAt(d, d.index[ti + 1]!), vertAt(d, d.index[ti + 2]!));
        if (hit < best) { best = hit; bestTri = t; }
      }
    } else {
      stack.push(i + 1);          // left child
      stack.push(i + word6);      // right child (RELATIVE offset)
    }
  }
  return { t: best, tri: bestTri };
}

/** Brute-force over all triangles (the independent oracle). */
function bruteForce(d: BvhTextureData, orig: V3, dir: V3): { t: number; tri: number } {
  let best = Infinity;
  let bestTri = -1;
  for (let t = 0; t < d.triangleCount; t += 1) {
    const ti = t * 4;
    const hit = rayTri(orig, dir, vertAt(d, d.index[ti]!), vertAt(d, d.index[ti + 1]!), vertAt(d, d.index[ti + 2]!));
    if (hit < best) { best = hit; bestTri = t; }
  }
  return { t: best, tri: bestTri };
}

describe('BVH texture adapter — packed textures traverse correctly (vs brute force)', () => {
  const pack = mergeWorldSpaceFromCore(multiNodeScene(), { positionStride: 4 });
  const d = packBvhTextureData(pack);

  it('produces a multi-node BVH (interior nodes exist, so traversal is exercised)', () => {
    expect(d.nodeCount).toBeGreaterThan(1);
    expect(d.triangleCount).toBe(12);
  });

  it('texel layout is correct: bounds 2/node, contents 1/node, RGBA-strided', () => {
    expect(d.bounds.length).toBe(d.boundsDim * d.boundsDim * 4);
    expect(d.contents.length).toBe(d.contentsDim * d.contentsDim * 4);
    expect(d.position.length).toBe(d.positionDim * d.positionDim * 4);
    // every vertex's .w was forced to 1.0
    for (let v = 0; v < d.vertexCount; v += 1) expect(d.position[v * 4 + 3]).toBe(1.0);
  });

  it('CPU-traversal of the packed textures matches brute-force on a ray sweep', () => {
    // 60 rays from a shell around the geometry aimed at the origin region.
    let tested = 0;
    let hits = 0;
    for (let a = 0; a < 6; a += 1) {
      for (let b = 0; b < 10; b += 1) {
        const theta = (a / 6) * Math.PI * 2;
        const phi = (b / 10) * Math.PI - Math.PI / 2;
        const R = 12;
        const orig: V3 = [R * Math.cos(phi) * Math.cos(theta), R * Math.sin(phi), R * Math.cos(phi) * Math.sin(theta)];
        const target: V3 = [0, 0, 0];
        const dRaw = sub(target, orig);
        const len = Math.hypot(...dRaw) || 1;
        const dir: V3 = [dRaw[0] / len, dRaw[1] / len, dRaw[2] / len];
        const bvh = traverseBvh(d, orig, dir);
        const bf = bruteForce(d, orig, dir);
        expect(bvh.tri).toBe(bf.tri);                 // same closest triangle (or both -1 on miss)
        if (Number.isFinite(bf.t)) { expect(bvh.t).toBeCloseTo(bf.t, 4); hits += 1; }
        tested += 1;
      }
    }
    expect(tested).toBe(60);
    expect(hits).toBeGreaterThan(0); // the sweep actually hits geometry (the test isn't vacuous)
  });
});
