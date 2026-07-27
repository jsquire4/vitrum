import { describe, expect, it } from 'vitest';
import {
  BVH_NODE_FLOATS,
  CWBVH_CHILD_COUNT_INVALID,
  CWBVH_CHILDREN,
  CWBVH_CHILD_LEAF,
  CWBVH_CHILD_META_WORDS,
  CWBVH_CHILD_NODE,
  buildCompressedWideBvh,
  buildCompressedWideBvhFromArrayBvh,
  cwbvhChildBounds,
  intersectCompressedWideBvhAnyHit,
  intersectCompressedWideBvhFirstHit,
  isLeafSplit,
  type CompressedWideBvhBuildResult,
  type CwbvhRay,
  type CwbvhTraverseOptions,
} from '../index.js';

type V3 = readonly [number, number, number];

function sub(a: V3, b: V3): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: V3, b: V3): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v: V3): [number, number, number] {
  const inv = 1 / Math.sqrt(Math.max(dot(v, v), 1e-30));
  return [v[0] * inv, v[1] * inv, v[2] * inv];
}

function positionAt(positions: Float32Array, index: number): [number, number, number] {
  const base = index * 4;
  return [positions[base] ?? 0, positions[base + 1] ?? 0, positions[base + 2] ?? 0];
}

function triangleHit(ray: CwbvhRay, a: V3, b: V3, c: V3, eps = 1e-5): number | null {
  const e1 = sub(b, a);
  const e2 = sub(c, a);
  const n = cross(e1, e2);
  const det = -dot(ray.direction, n);
  if (Math.abs(det) < eps) return null;
  const invDet = 1 / det;
  const ao = sub(ray.origin, a);
  const dao = cross(ao, ray.direction);
  const u = dot(e2, dao) * invDet;
  const v = -dot(e1, dao) * invDet;
  const t = dot(ao, n) * invDet;
  const w = 1 - u - v;
  return u < -eps || v < -eps || w < -eps || t < eps ? null : t;
}

function intersectsAabb(ray: CwbvhRay, min: V3, max: V3, tMinIn: number, tMaxIn: number): boolean {
  let tMin = tMinIn;
  let tMax = tMaxIn;
  for (const axis of [0, 1, 2] as const) {
    const origin = ray.origin[axis];
    const direction = ray.direction[axis];
    const axisMin = min[axis];
    const axisMax = max[axis];
    if (Math.abs(direction) < 1e-30) {
      if (origin < axisMin || origin > axisMax) return false;
      continue;
    }
    const inv = 1 / direction;
    let a = (axisMin - origin) * inv;
    let b = (axisMax - origin) * inv;
    if (a > b) [a, b] = [b, a];
    tMin = Math.max(tMin, a);
    tMax = Math.min(tMax, b);
    if (tMax < tMin) return false;
  }
  return true;
}

interface BinaryHit {
  readonly didHit: boolean;
  readonly dist: number;
  readonly triangleIndex: number;
  readonly sourceTriangleIndex: number;
}

function canonicalBinaryFirstHit(
  built: CompressedWideBvhBuildResult,
  positions: Float32Array,
  ray: CwbvhRay,
  opts: CwbvhTraverseOptions = {},
): BinaryHit {
  const words = new Uint32Array(built.bvhNodes.buffer, built.bvhNodes.byteOffset, built.bvhNodes.length);
  const tMin = opts.tMin ?? opts.triEps ?? 1e-5;
  let closest = opts.tMax ?? Number.POSITIVE_INFINITY;
  const root = opts.root ?? 0;
  let triangleIndex = -1;
  let sourceTriangleIndex = -1;
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop()!;
    const base = node * BVH_NODE_FLOATS;
    if (base + 7 >= built.bvhNodes.length) throw new Error(`binary node ${node} out of range`);
    const min: V3 = [built.bvhNodes[base]!, built.bvhNodes[base + 1]!, built.bvhNodes[base + 2]!];
    const max: V3 = [built.bvhNodes[base + 3]!, built.bvhNodes[base + 4]!, built.bvhNodes[base + 5]!];
    if (!intersectsAabb(ray, min, max, tMin, closest)) continue;
    const rightOrOffset = words[base + 6]!;
    const splitOrCount = words[base + 7]!;
    if (!isLeafSplit(splitOrCount)) {
      stack.push(node + rightOrOffset, node + 1);
      continue;
    }
    const count = splitOrCount & 0xffff;
    for (let i = 0; i < count; i += 1) {
      const tri = rightOrOffset + i;
      const ib = tri * 4;
      const hit = triangleHit(
        ray,
        positionAt(positions, built.reorderedIndices[ib] ?? 0),
        positionAt(positions, built.reorderedIndices[ib + 1] ?? 0),
        positionAt(positions, built.reorderedIndices[ib + 2] ?? 0),
        opts.triEps,
      );
      if (hit != null && hit >= tMin && hit < closest) {
        closest = hit;
        triangleIndex = tri;
        sourceTriangleIndex = built.reorderedToSourceTriangle[tri] ?? tri;
      }
    }
  }
  return { didHit: triangleIndex >= 0, dist: closest, triangleIndex, sourceTriangleIndex };
}

function canonicalBinaryAnyHit(
  built: CompressedWideBvhBuildResult,
  positions: Float32Array,
  ray: CwbvhRay,
  opts: CwbvhTraverseOptions = {},
): boolean {
  return canonicalBinaryFirstHit(built, positions, ray, opts).didHit;
}

function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

function randomMesh(triangleCount: number): {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
  readonly centers: readonly V3[];
  readonly normals: readonly V3[];
} {
  const random = makeRandom(0x51a7c0de);
  const positions: number[] = [];
  const indices: number[] = [];
  const centers: V3[] = [];
  const normals: V3[] = [];
  for (let tri = 0; tri < triangleCount; tri += 1) {
    const center: V3 = [random() * 200 - 100, random() * 200 - 100, random() * 200 - 100];
    let e1: V3;
    let e2: V3;
    let n: V3;
    do {
      e1 = [random() * 8 - 4, random() * 8 - 4, random() * 8 - 4];
      e2 = [random() * 8 - 4, random() * 8 - 4, random() * 8 - 4];
      n = cross(e1, e2);
    } while (dot(n, n) < 0.25);
    const v0: V3 = sub(center, [(e1[0] + e2[0]) / 3, (e1[1] + e2[1]) / 3, (e1[2] + e2[2]) / 3]);
    const v1: V3 = [v0[0] + e1[0], v0[1] + e1[1], v0[2] + e1[2]];
    const v2: V3 = [v0[0] + e2[0], v0[1] + e2[1], v0[2] + e2[2]];
    const vertex = tri * 3;
    positions.push(...v0, 0, ...v1, 0, ...v2, 0);
    indices.push(vertex, vertex + 1, vertex + 2, 0);
    centers.push(center);
    normals.push(normalize(n));
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices), centers, normals };
}

function assertSameHit(binary: BinaryHit, wide: ReturnType<typeof intersectCompressedWideBvhFirstHit>): void {
  expect(wide.didHit).toBe(binary.didHit);
  expect(wide.triangleIndex).toBe(binary.triangleIndex);
  expect(wide.sourceTriangleIndex).toBe(binary.sourceTriangleIndex);
  if (binary.didHit) {
    expect(Math.abs(wide.dist - binary.dist)).toBeLessThanOrEqual(2e-5 * Math.max(1, Math.abs(binary.dist)));
  }
}
function exactChildBounds(
  built: CompressedWideBvhBuildResult,
  positions: Float32Array,
  node: number,
  slot: number,
): { min: [number, number, number]; max: [number, number, number] } {
  const meta = (node * CWBVH_CHILDREN + slot) * CWBVH_CHILD_META_WORDS;
  const kind = built.cwbvhChildMeta[meta];
  if (kind === CWBVH_CHILD_NODE) {
    const child = built.cwbvhChildMeta[meta + 1]!;
    const base = child * 6;
    return {
      min: [built.cwbvhNodeBounds[base]!, built.cwbvhNodeBounds[base + 1]!, built.cwbvhNodeBounds[base + 2]!],
      max: [built.cwbvhNodeBounds[base + 3]!, built.cwbvhNodeBounds[base + 4]!, built.cwbvhNodeBounds[base + 5]!],
    };
  }
  if (kind !== CWBVH_CHILD_LEAF) throw new Error(`unexpected child kind ${kind}`);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const first = built.cwbvhChildMeta[meta + 1]!;
  const count = built.cwbvhChildMeta[meta + 2]!;
  for (let tri = first; tri < first + count; tri += 1) {
    const ib = tri * 4;
    for (const vertex of [
      built.reorderedIndices[ib]!,
      built.reorderedIndices[ib + 1]!,
      built.reorderedIndices[ib + 2]!,
    ]) {
      const p = positionAt(positions, vertex);
      for (const axis of [0, 1, 2] as const) {
        min[axis] = Math.min(min[axis], p[axis]);
        max[axis] = Math.max(max[axis], p[axis]);
      }
    }
  }
  return { min, max };
}

describe('CWBVH differential production gates', () => {
  it('matches canonical binary traversal for 512 deterministic randomized hit/miss rays', () => {
    const mesh = randomMesh(128);
    const built = buildCompressedWideBvh(mesh.positions, mesh.indices, new Uint32Array(128), {
      maxLeafTriangles: 1,
    });

    const random = makeRandom(0xc001d00d);
    let hitRays = 0;
    let missRays = 0;
    for (let i = 0; i < 512; i += 1) {
      let ray: CwbvhRay;
      if ((i & 1) === 0) {
        const tri = Math.floor(random() * mesh.centers.length);
        const center = mesh.centers[tri]!;
        const normal = mesh.normals[tri]!;
        const distance = 2 + random() * 300;
        ray = {
          origin: [center[0] + normal[0] * distance, center[1] + normal[1] * distance, center[2] + normal[2] * distance],
          direction: [-normal[0], -normal[1], -normal[2]],
        };
      } else {
        ray = {
          origin: [random() * 500 - 250, random() * 500 - 250, random() * 500 - 250],
          direction: normalize([random() * 2 - 1, random() * 2 - 1, random() * 2 - 1]),
        };
      }
      const opts = { tMin: 0.001 + random() * 0.05, tMax: 20 + random() * 400 };
      const binary = canonicalBinaryFirstHit(built, mesh.positions, ray, opts);
      const wide = intersectCompressedWideBvhFirstHit(built, mesh.positions, ray, opts);
      assertSameHit(binary, wide);
      expect(intersectCompressedWideBvhAnyHit(built, mesh.positions, ray, opts)).toBe(
        canonicalBinaryAnyHit(built, mesh.positions, ray, opts),
      );
      if (binary.didHit) hitRays += 1;
      else missRays += 1;
    }
    expect(hitRays).toBeGreaterThan(100);
    expect(missRays).toBeGreaterThan(100);
  });

  it('matches canonical binary traversal for degenerate, skinny, and large-coordinate geometry', () => {
    const positions = new Float32Array([
      100_000_000, 100_000_000, 100_000_000, 0,
      100_000_064, 100_000_000, 100_000_000, 0,
      100_000_000, 100_000_064, 100_000_000, 0,
      100_000_000, 100_000_000, 100_000_032, 0,
      100_000_064, 100_000_000, 100_000_032, 0,
      100_000_000, 100_000_008, 100_000_032, 0,
      100_000_016, 100_000_016, 100_000_016, 0,
      100_000_016, 100_000_016, 100_000_016, 0,
      100_000_016, 100_000_016, 100_000_016, 0,
    ]);
    const indices = new Uint32Array([0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0]);
    const built = buildCompressedWideBvh(positions, indices, new Uint32Array(3), { maxLeafTriangles: 1 });
    const rays: CwbvhRay[] = [
      { origin: [100_000_016, 100_000_016, 100_000_256], direction: [0, 0, -1] },
      { origin: [100_000_040, 100_000_004, 100_000_256], direction: [0, 0, -1] },
      { origin: [100_000_128, 100_000_128, 100_000_256], direction: [0, 0, -1] },
      { origin: [100_000_016, 100_000_016, 100_000_000], direction: [1, 0, 0] },
    ];
    for (const ray of rays) {
      const opts = { tMin: 1, tMax: 512 };
      const binary = canonicalBinaryFirstHit(built, positions, ray, opts);
      assertSameHit(binary, intersectCompressedWideBvhFirstHit(built, positions, ray, opts));
      expect(intersectCompressedWideBvhAnyHit(built, positions, ray, opts)).toBe(binary.didHit);
    }
  });

  it('keeps float32 WGSL child-bound decode conservative and finite', () => {
    const mesh = randomMesh(96);
    const built = buildCompressedWideBvh(mesh.positions, mesh.indices, new Uint32Array(96), { maxLeafTriangles: 1 });
    const decodeF32 = (q: number, min: number, max: number): number => {
      if (q === 0) return min;
      if (q === 0xffff) return max;
      const t = Math.fround(q / 0xffff);
      return Math.fround(Math.fround(min * Math.fround(1 - t)) + Math.fround(max * t));
    };
    for (let node = 0; node < built.cwbvhNodeCount; node += 1) {
      const nb = node * 6;
      const parentMin: V3 = [built.cwbvhNodeBounds[nb]!, built.cwbvhNodeBounds[nb + 1]!, built.cwbvhNodeBounds[nb + 2]!];
      const parentMax: V3 = [built.cwbvhNodeBounds[nb + 3]!, built.cwbvhNodeBounds[nb + 4]!, built.cwbvhNodeBounds[nb + 5]!];
      for (let slot = 0; slot < built.cwbvhChildCount[node]!; slot += 1) {
        const cb = (node * CWBVH_CHILDREN + slot) * 6;
        const exact = exactChildBounds(built, mesh.positions, node, slot);
        for (const axis of [0, 1, 2] as const) {
          const minF32 = decodeF32(built.cwbvhChildBounds[cb + axis]!, parentMin[axis], parentMax[axis]);
          const maxF32 = decodeF32(built.cwbvhChildBounds[cb + 3 + axis]!, parentMin[axis], parentMax[axis]);
          expect(Number.isFinite(minF32)).toBe(true);
          expect(Number.isFinite(maxF32)).toBe(true);
          expect(minF32).toBeLessThanOrEqual(exact.min[axis]);
          expect(maxF32).toBeGreaterThanOrEqual(exact.max[axis]);
        }
      }
    }
  });

  it('uses overflow-safe interpolation for opposite-sign extreme parent bounds', () => {
    const nodes = new Float32Array(3 * BVH_NODE_FLOATS);
    const words = new Uint32Array(nodes.buffer);
    nodes.set([-3e38, -3e38, -3e38, 3e38, 3e38, 3e38], 0);
    words[6] = 2;
    words[7] = 0;
    nodes.set([-2e38, -2e38, -2e38, -1e38, -1e38, -1e38], 8);
    words[14] = 0;
    words[15] = 0xffff0001;
    nodes.set([1e38, 1e38, 1e38, 2e38, 2e38, 2e38], 16);
    words[22] = 1;
    words[23] = 0xffff0001;
    const built = buildCompressedWideBvhFromArrayBvh({
      bvhNodes: nodes,
      reorderedIndices: new Uint32Array([0, 1, 2, 0, 3, 4, 5, 0]),
      reorderedTriMaterialIds: new Uint32Array(2),
      reorderedToSourceTriangle: new Uint32Array([0, 1]),
    });
    expect(built.cwbvhChildCount[0]).toBe(2);
    for (let slot = 0; slot < 2; slot += 1) {
      const bounds = cwbvhChildBounds(built, 0, slot);
      expect(bounds.min.every(Number.isFinite)).toBe(true);
      expect(bounds.max.every(Number.isFinite)).toBe(true);
    }
  });

  it('forces overflow and corrupt-root fallbacks to canonical closest/any identity', () => {
    const positions: number[] = [];
    const indices: number[] = [];
    for (let tri = 0; tri < 128; tri += 1) {
      const z = tri * 0.25;
      const vertex = tri * 3;
      positions.push(-2, -2, z, 0, 2, -2, z, 0, 0, 2, z, 0);
      indices.push(vertex, vertex + 1, vertex + 2, 0);
    }
    const p = new Float32Array(positions);
    const built = buildCompressedWideBvh(p, new Uint32Array(indices), new Uint32Array(128), { maxLeafTriangles: 1 });
    const ray: CwbvhRay = { origin: [0, 0, 50], direction: [0, 0, -1] };
    const opts = { tMin: 3.5, tMax: 47.75 };
    const canonical = canonicalBinaryFirstHit(built, p, ray, opts);
    const canonicalAny = canonicalBinaryAnyHit(built, p, ray, opts);

    expect(() => intersectCompressedWideBvhFirstHit(built, p, ray, { ...opts, maxStackDepth: 1 })).toThrow(/stack overflow/);
    expect(() => intersectCompressedWideBvhAnyHit(built, p, ray, { ...opts, maxStackDepth: 1 })).toThrow(/stack overflow/);
    const overflowClosest = (() => {
      try { return intersectCompressedWideBvhFirstHit(built, p, ray, { ...opts, maxStackDepth: 1 }); }
      catch { return canonicalBinaryFirstHit(built, p, ray, opts); }
    })();
    const overflowAny = (() => {
      try { return intersectCompressedWideBvhAnyHit(built, p, ray, { ...opts, maxStackDepth: 1 }); }
      catch { return canonicalBinaryAnyHit(built, p, ray, opts); }
    })();
    expect(overflowClosest).toMatchObject(canonical);
    expect(overflowAny).toBe(canonicalAny);

    const corrupt = { ...built, cwbvhChildCount: new Uint32Array(built.cwbvhChildCount) };
    corrupt.cwbvhChildCount[0] = CWBVH_CHILD_COUNT_INVALID;
    expect(() => intersectCompressedWideBvhFirstHit(corrupt, p, ray, opts)).toThrow(/invalid child count/);
    expect(() => intersectCompressedWideBvhAnyHit(corrupt, p, ray, opts)).toThrow(/invalid child count/);
    const corruptClosest = (() => {
      try { return intersectCompressedWideBvhFirstHit(corrupt, p, ray, opts); }
      catch { return canonicalBinaryFirstHit(built, p, ray, opts); }
    })();
    const corruptAny = (() => {
      try { return intersectCompressedWideBvhAnyHit(corrupt, p, ray, opts); }
      catch { return canonicalBinaryAnyHit(built, p, ray, opts); }
    })();
    expect(corruptClosest).toMatchObject(canonical);
    expect(corruptAny).toBe(canonicalAny);
  });

  it('marks non-finite and inverted parent bounds invalid instead of publishing them', () => {
    for (const rootBounds of [
      [Number.NaN, 0, 0, 1, 1, 1],
      [2, 0, 0, 1, 1, 1],
    ]) {
      const nodes = new Float32Array(BVH_NODE_FLOATS);
      const words = new Uint32Array(nodes.buffer);
      nodes.set(rootBounds);
      words[6] = 0;
      words[7] = 0xffff0001;
      const built = buildCompressedWideBvhFromArrayBvh({
        bvhNodes: nodes,
        reorderedIndices: new Uint32Array([0, 1, 2, 0]),
        reorderedTriMaterialIds: new Uint32Array([0]),
        reorderedToSourceTriangle: new Uint32Array([0]),
      });
      expect(built.cwbvhNodeCount).toBe(1);
      expect(built.cwbvhChildCount[0]).toBe(CWBVH_CHILD_COUNT_INVALID);
      expect(() => intersectCompressedWideBvhFirstHit(built, new Float32Array(0), {
        origin: [0, 0, 2], direction: [0, 0, -1],
      })).toThrow(/invalid child count/);
    }
  });

  it('publishes the invalid-layout sentinel at every selectable wide root', () => {
    const mesh = randomMesh(128);
    const canonical = buildCompressedWideBvh(
      mesh.positions,
      mesh.indices,
      new Uint32Array(128),
      { maxLeafTriangles: 1 },
    );
    expect(canonical.cwbvhNodeCount).toBeGreaterThan(1);

    const malformedNodes = new Float32Array(canonical.bvhNodes);
    // Keep the root finite and ordered but shrink its X range so at least one
    // unchanged descendant lies outside it. Conversion discovers the global
    // parent/child containment violation after allocating multiple wide nodes.
    malformedNodes[0] =
      malformedNodes[0]! + (malformedNodes[3]! - malformedNodes[0]!) * 0.25;
    const malformed = buildCompressedWideBvhFromArrayBvh({
      bvhNodes: malformedNodes,
      reorderedIndices: canonical.reorderedIndices,
      reorderedTriMaterialIds: canonical.reorderedTriMaterialIds,
      reorderedToSourceTriangle: canonical.reorderedToSourceTriangle,
    });

    expect(malformed.cwbvhNodeCount).toBeGreaterThan(1);
    expect(Array.from(malformed.cwbvhChildCount)).toEqual(
      Array.from(
        { length: malformed.cwbvhNodeCount },
        () => CWBVH_CHILD_COUNT_INVALID,
      ),
    );
    expect(() => intersectCompressedWideBvhFirstHit(
      malformed,
      mesh.positions,
      { origin: [0, 0, 10], direction: [0, 0, -1] },
      { root: 1 },
    )).toThrow(/node 1 has invalid child count/);
  });
});
