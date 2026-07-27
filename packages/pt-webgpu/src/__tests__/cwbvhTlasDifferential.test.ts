import { describe, expect, it } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import {
  BVH_NODE_FLOATS,
  CWBVH_CHILD_BOUNDS_PACKED_U32,
  CWBVH_CHILD_BOUNDS_U16,
  CWBVH_CHILDREN,
  intersectCompressedWideBvhFirstHit,
  isLeafSplit,
  type CompressedWideBvhBuildResult,
  type CwbvhRay,
} from '@vitrum/shared-bvh';
import {
  CWBVH_ROOT_PAIR_MAGIC,
  buildPackedScene,
  packCwbvhRootPair,
  rebuildTlasForSceneTransforms,
} from '../scene/uploadSceneBuffers.js';

type Packed = ReturnType<typeof buildPackedScene>;
type V3 = readonly [number, number, number];

const material = { baseColor: [0.7, 0.7, 0.7] as const, roughness: 0.5, metallic: 0 };

function transform(xScale: number, yScale: number, zScale: number, tx: number, ty: number, tz: number) {
  return asMat4(new Float32Array([
    xScale, 0, 0, 0,
    0, yScale, 0, 0,
    0, 0, zScale, 0,
    tx, ty, tz, 1,
  ]));
}

function scene(instanceTransforms: readonly ReturnType<typeof asMat4>[]): Scene {
  return {
    primitives: [
      {
        kind: 'instanced-mesh',
        id: 'instances',
        positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material,
        instances: [...instanceTransforms],
      },
      {
        kind: 'mesh',
        id: 'second-blas',
        positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material,
        transform: transform(1, 1, 1, 0, -5, -1),
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function unpackCwbvh(packed: Packed): CompressedWideBvhBuildResult {
  const bounds = new Uint16Array(packed.cwbvhNodeCount * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16);
  for (let i = 0; i < packed.cwbvhChildBoundsPacked.length; i += 1) {
    const word = packed.cwbvhChildBoundsPacked[i]!;
    bounds[i * 2] = word & 0xffff;
    bounds[i * 2 + 1] = word >>> 16;
  }
  expect(packed.cwbvhChildBoundsPacked.length).toBe(
    packed.cwbvhNodeCount * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_PACKED_U32,
  );
  return {
    bvhNodes: packed.bvhNodes,
    reorderedIndices: packed.indices,
    reorderedTriMaterialIds: packed.triMaterialIds,
    reorderedToSourceTriangle: Uint32Array.from({ length: packed.triangleCount }, (_, i) => i),
    cwbvhNodeBounds: packed.cwbvhNodeBounds,
    cwbvhChildBounds: bounds,
    cwbvhChildMeta: packed.cwbvhChildMeta,
    cwbvhChildCount: packed.cwbvhChildCount,
    cwbvhNodeCount: packed.cwbvhNodeCount,
  };
}

function point(matrix: Float32Array, p: V3): [number, number, number] {
  return [
    matrix[0]! * p[0] + matrix[4]! * p[1] + matrix[8]! * p[2] + matrix[12]!,
    matrix[1]! * p[0] + matrix[5]! * p[1] + matrix[9]! * p[2] + matrix[13]!,
    matrix[2]! * p[0] + matrix[6]! * p[1] + matrix[10]! * p[2] + matrix[14]!,
  ];
}

function direction(matrix: Float32Array, d: V3): [number, number, number] {
  const raw: V3 = [
    matrix[0]! * d[0] + matrix[4]! * d[1] + matrix[8]! * d[2],
    matrix[1]! * d[0] + matrix[5]! * d[1] + matrix[9]! * d[2],
    matrix[2]! * d[0] + matrix[6]! * d[1] + matrix[10]! * d[2],
  ];
  const inv = 1 / Math.hypot(...raw);
  return [raw[0] * inv, raw[1] * inv, raw[2] * inv];
}

function sub(a: V3, b: V3): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

interface LocalBinaryHit { readonly didHit: boolean; readonly dist: number; readonly triangleIndex: number }

function binaryLocalHit(packed: Packed, ray: CwbvhRay, root: number, tMin: number, tMax: number): LocalBinaryHit {
  const words = new Uint32Array(packed.bvhNodes.buffer, packed.bvhNodes.byteOffset, packed.bvhNodes.length);
  const stack = [root];
  let closest = tMax;
  let triangleIndex = -1;
  const triangle = (tri: number): number | null => {
    const ib = tri * 4;
    const p = (index: number): V3 => {
      const base = index * 4;
      return [packed.positions[base]!, packed.positions[base + 1]!, packed.positions[base + 2]!];
    };
    const a = p(packed.indices[ib]!);
    const b = p(packed.indices[ib + 1]!);
    const c = p(packed.indices[ib + 2]!);
    const e1 = sub(b, a);
    const e2 = sub(c, a);
    const n: V3 = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ];
    const det = -dot(ray.direction, n);
    if (Math.abs(det) < 1e-5) return null;
    const invDet = 1 / det;
    const ao = sub(ray.origin, a);
    const dao: V3 = [
      ao[1] * ray.direction[2] - ao[2] * ray.direction[1],
      ao[2] * ray.direction[0] - ao[0] * ray.direction[2],
      ao[0] * ray.direction[1] - ao[1] * ray.direction[0],
    ];
    const u = dot(e2, dao) * invDet;
    const v = -dot(e1, dao) * invDet;
    const value = dot(ao, n) * invDet;
    return u < -1e-5 || v < -1e-5 || 1 - u - v < -1e-5 || value < 1e-5 ? null : value;
  };
  const aabb = (base: number): boolean => {
    let lo = tMin;
    let hi = closest;
    for (const axis of [0, 1, 2] as const) {
      const origin = ray.origin[axis];
      const d = ray.direction[axis];
      const min = packed.bvhNodes[base + axis]!;
      const max = packed.bvhNodes[base + 3 + axis]!;
      if (Math.abs(d) < 1e-30) {
        if (origin < min || origin > max) return false;
        continue;
      }
      let a = (min - origin) / d;
      let b = (max - origin) / d;
      if (a > b) [a, b] = [b, a];
      lo = Math.max(lo, a);
      hi = Math.min(hi, b);
      if (hi < lo) return false;
    }
    return true;
  };
  while (stack.length > 0) {
    const node = stack.pop()!;
    const base = node * BVH_NODE_FLOATS;
    if (!aabb(base)) continue;
    const right = words[base + 6]!;
    const split = words[base + 7]!;
    if (!isLeafSplit(split)) {
      stack.push(node + right, node + 1);
      continue;
    }
    for (let i = 0; i < (split & 0xffff); i += 1) {
      const tri = right + i;
      const hit = triangle(tri);
      if (hit != null && hit >= tMin && hit < closest) {
        closest = hit;
        triangleIndex = tri;
      }
    }
  }
  return { didHit: triangleIndex >= 0, dist: closest, triangleIndex };
}

function tracePacked(packed: Packed, worldRay: CwbvhRay, tMin: number, tMax: number) {
  const cwbvh = unpackCwbvh(packed);
  const binaryBest = { instance: -1, triangle: -1, dist: tMax };
  const wideBest = { instance: -1, triangle: -1, dist: tMax };
  for (let instance = 0; instance < packed.tlasBlasRoots.length; instance += 1) {
    const pair = packed.cwbvhTlasBlasRoots.subarray(instance * 4, instance * 4 + 4);
    expect(Array.from(pair)).toEqual(Array.from(packCwbvhRootPair(pair[1]!, pair[2]!)));
    const w2l = packed.tlasInstanceWorldToLocal.subarray(instance * 16, instance * 16 + 16);
    const l2w = packed.tlasInstanceLocalToWorld.subarray(instance * 16, instance * 16 + 16);
    const localRay: CwbvhRay = { origin: point(w2l, worldRay.origin), direction: direction(w2l, worldRay.direction) };
    const localStart = point(w2l, [
      worldRay.origin[0] + worldRay.direction[0] * tMin,
      worldRay.origin[1] + worldRay.direction[1] * tMin,
      worldRay.origin[2] + worldRay.direction[2] * tMin,
    ]);
    const localEnd = point(w2l, [
      worldRay.origin[0] + worldRay.direction[0] * tMax,
      worldRay.origin[1] + worldRay.direction[1] * tMax,
      worldRay.origin[2] + worldRay.direction[2] * tMax,
    ]);
    const localTMin = Math.max(dot(sub(localStart, localRay.origin), localRay.direction), 0);
    const localTMax = Math.max(dot(sub(localEnd, localRay.origin), localRay.direction), localTMin);
    const binary = binaryLocalHit(packed, localRay, pair[1]!, localTMin, localTMax);
    const wide = intersectCompressedWideBvhFirstHit(cwbvh, packed.positions, localRay, {
      root: pair[2]!, tMin: localTMin, tMax: localTMax,
    });
    expect(wide.didHit).toBe(binary.didHit);
    expect(wide.triangleIndex).toBe(binary.triangleIndex);
    if (binary.didHit) expect(wide.dist).toBeCloseTo(binary.dist, 5);
    for (const [kind, hit] of [['binary', binary], ['wide', wide]] as const) {
      if (!hit.didHit) continue;
      const localPosition: V3 = [
        localRay.origin[0] + localRay.direction[0] * hit.dist,
        localRay.origin[1] + localRay.direction[1] * hit.dist,
        localRay.origin[2] + localRay.direction[2] * hit.dist,
      ];
      const worldPosition = point(l2w, localPosition);
      const worldDist = dot(sub(worldPosition, worldRay.origin), worldRay.direction);
      const best = kind === 'binary' ? binaryBest : wideBest;
      if (worldDist > tMin && worldDist < best.dist) {
        best.instance = instance;
        best.triangle = hit.triangleIndex;
        best.dist = worldDist;
      }
    }
  }
  expect(wideBest.instance).toBe(binaryBest.instance);
  expect(wideBest.triangle).toBe(binaryBest.triangle);
  expect(wideBest.dist).toBeCloseTo(binaryBest.dist, 5);
  return wideBest;
}

describe('pt-webgpu CWBVH TLAS/BLAS differential', () => {
  it('matches canonical BLAS traversal through multi-BLAS nonuniform transforms', () => {
    const packed = buildPackedScene(scene([
      transform(1, 1, 1, -4, 0, 0),
      transform(2, 0.5, 1.5, 3, 2, 4),
    ]));
    expect(packed.tlasBlasRoots).toHaveLength(3);
    expect(packed.cwbvhTlasBlasRoots).toHaveLength(12);
    expect(new Set(packed.tlasBlasRoots).size).toBe(2);
    const hits = [
      tracePacked(packed, { origin: [-4, 0, 20], direction: [0, 0, -1] }, 0.25, 40),
      tracePacked(packed, { origin: [3, 2, 20], direction: [0, 0, -1] }, 0.25, 40),
      tracePacked(packed, { origin: [0, -5, 20], direction: [0, 0, -1] }, 0.25, 40),
    ];
    expect(hits.map((hit) => hit.instance)).toEqual([0, 1, 2]);
  });

  it('preserves paired roots across transform refit and rebuilds them on topology change', () => {
    const initialScene = scene([transform(1, 1, 1, -4, 0, 0), transform(2, 0.5, 1.5, 3, 2, 4)]);
    const packed = buildPackedScene(initialScene);
    const movedScene = scene([transform(1, 1, 1, 7, 1, 3), transform(2, 0.5, 1.5, 3, 2, 4)]);
    const refit = rebuildTlasForSceneTransforms(movedScene, packed.primitiveTlasBindings, {
      tlasNodes: packed.tlasNodes,
      tlasInstanceIndices: packed.tlasInstanceIndices,
      tlasBlasRoots: packed.tlasBlasRoots,
      tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
    });
    expect(refit.ok).toBe(true);
    if (!refit.ok) return;
    const refitPacked: Packed = { ...packed, ...refit };
    expect(Array.from(refitPacked.cwbvhTlasBlasRoots)).toEqual(Array.from(packed.cwbvhTlasBlasRoots));
    expect(tracePacked(refitPacked, { origin: [7, 1, 20], direction: [0, 0, -1] }, 0.5, 40).instance).toBe(0);

    const changed = buildPackedScene(scene([
      transform(1, 1, 1, 7, 1, 3),
      transform(2, 0.5, 1.5, 3, 2, 4),
      transform(0.75, 1.25, 2, -8, 3, 5),
    ]));
    expect(changed.tlasBlasRoots).toHaveLength(4);
    expect(changed.cwbvhTlasBlasRoots).toHaveLength(16);
    for (let i = 0; i < changed.tlasBlasRoots.length; i += 1) {
      const record = changed.cwbvhTlasBlasRoots.subarray(i * 4, i * 4 + 4);
      expect(record[0]).toBe(CWBVH_ROOT_PAIR_MAGIC);
      expect(Array.from(record)).toEqual(Array.from(packCwbvhRootPair(record[1]!, record[2]!)));
    }
    expect(tracePacked(changed, { origin: [-8, 3, 20], direction: [0, 0, -1] }, 0.5, 40).instance).toBe(2);
  });
});
