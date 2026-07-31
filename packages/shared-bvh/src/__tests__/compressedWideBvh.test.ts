import { describe, expect, it } from 'vitest';
import {
  CWBVH_CHILD_BOUNDS_U16,
  CWBVH_CHILD_LEAF,
  CWBVH_CHILD_META_WORDS,
  CWBVH_CHILD_NODE,
  CWBVH_CHILDREN,
  CWBVH_CHILD_BOUNDS_PACKED_U32,
  CWBVH_CHILD_EMPTY,
  CWBVH_TRAVERSAL_STACK_DEPTH,
  buildCompressedWideBvh,
  cwbvhChildBounds,
  intersectCompressedWideBvhAnyHit,
  intersectCompressedWideBvhFirstHit,
  packCwbvhBuildBoundsForWgsl,
  packCwbvhChildBoundsForWgsl,
  reorderCwbvhTrianglePayloads,
  requiredCwbvhTraversalStackEntries,
  type CompressedWideBvhBuildResult,
  type CwbvhRay,
} from '../index.js';

function makeGrid(size: number): {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
  readonly triMaterialIds: Uint32Array;
} {
  const positions: number[] = [];
  for (let y = 0; y <= size; y += 1) {
    for (let x = 0; x <= size; x += 1) {
      positions.push(x, y, 0, 0);
    }
  }

  const indices: number[] = [];
  const triMaterialIds: number[] = [];
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const v0 = y * (size + 1) + x;
      const v1 = v0 + 1;
      const v2 = v0 + size + 2;
      const v3 = v0 + size + 1;
      indices.push(v0, v1, v2, 0, v0, v2, v3, 0);
      triMaterialIds.push((x + y) & 3, (x + y + 1) & 3);
    }
  }

  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
    triMaterialIds: new Uint32Array(triMaterialIds),
  };
}

function positionAt(positions: Float32Array, index: number): readonly [number, number, number] {
  const o = index * 4;
  return [positions[o] ?? 0, positions[o + 1] ?? 0, positions[o + 2] ?? 0];
}

function sub(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function intersectTriangle(
  ray: CwbvhRay,
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  c: readonly [number, number, number],
): number | null {
  const e1 = sub(b, a);
  const e2 = sub(c, a);
  const n = cross(e1, e2);
  const det = -dot(ray.direction, n);
  if (Math.abs(det) < 1e-5) return null;
  const invDet = 1 / det;
  const ao = sub(ray.origin, a);
  const dao = cross(ao, ray.direction);
  const u = dot(e2, dao) * invDet;
  const v = -dot(e1, dao) * invDet;
  const t = dot(ao, n) * invDet;
  const w = 1 - u - v;
  if (u < -1e-5 || v < -1e-5 || w < -1e-5 || t < 1e-5) return null;
  return t;
}

function bruteForceFirstHit(
  positions: Float32Array,
  indices: Uint32Array,
  ray: CwbvhRay,
): { readonly didHit: boolean; readonly dist: number; readonly sourceTriangleIndex: number } {
  let best = Number.POSITIVE_INFINITY;
  let sourceTriangleIndex = -1;
  const triCount = Math.floor(indices.length / 4);
  for (let tri = 0; tri < triCount; tri += 1) {
    const ib = tri * 4;
    const t = intersectTriangle(
      ray,
      positionAt(positions, indices[ib] ?? 0),
      positionAt(positions, indices[ib + 1] ?? 0),
      positionAt(positions, indices[ib + 2] ?? 0),
    );
    if (t != null && t < best) {
      best = t;
      sourceTriangleIndex = tri;
    }
  }
  return { didHit: sourceTriangleIndex >= 0, dist: best, sourceTriangleIndex };
}

function bruteForceAnyHit(
  positions: Float32Array,
  indices: Uint32Array,
  ray: CwbvhRay,
): boolean {
  return bruteForceFirstHit(positions, indices, ray).didHit;
}

function growBounds(
  bounds: { min: [number, number, number]; max: [number, number, number] },
  p: readonly [number, number, number],
): void {
  bounds.min[0] = Math.min(bounds.min[0], p[0]);
  bounds.min[1] = Math.min(bounds.min[1], p[1]);
  bounds.min[2] = Math.min(bounds.min[2], p[2]);
  bounds.max[0] = Math.max(bounds.max[0], p[0]);
  bounds.max[1] = Math.max(bounds.max[1], p[1]);
  bounds.max[2] = Math.max(bounds.max[2], p[2]);
}

function concatFloat32(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function concatUint16(a: Uint16Array, b: Uint16Array): Uint16Array {
  const out = new Uint16Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function concatUint32(a: Uint32Array, b: Uint32Array): Uint32Array {
  const out = new Uint32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function offsetCwbvhMeta(
  meta: Uint32Array,
  nodeOffset: number,
  triangleOffset: number,
): Uint32Array {
  const out = new Uint32Array(meta);
  for (let i = 0; i < out.length; i += CWBVH_CHILD_META_WORDS) {
    const kind = out[i] ?? CWBVH_CHILD_EMPTY;
    if (kind === CWBVH_CHILD_NODE) {
      out[i + 1] = (out[i + 1] ?? 0) + nodeOffset;
    } else if (kind === CWBVH_CHILD_LEAF) {
      out[i + 1] = (out[i + 1] ?? 0) + triangleOffset;
    }
  }
  return out;
}

function offsetSourceTriangles(source: Uint32Array, triangleOffset: number): Uint32Array {
  const out = new Uint32Array(source.length);
  for (let i = 0; i < source.length; i += 1) out[i] = (source[i] ?? 0) + triangleOffset;
  return out;
}

function concatCwbvhRoots(
  a: CompressedWideBvhBuildResult,
  b: CompressedWideBvhBuildResult,
): CompressedWideBvhBuildResult {
  const triangleOffset = a.reorderedToSourceTriangle.length;
  return {
    ...a,
    bvhNodes: concatFloat32(a.bvhNodes, b.bvhNodes),
    reorderedIndices: concatUint32(a.reorderedIndices, b.reorderedIndices),
    reorderedTriMaterialIds: concatUint32(a.reorderedTriMaterialIds, b.reorderedTriMaterialIds),
    reorderedToSourceTriangle: concatUint32(
      a.reorderedToSourceTriangle,
      offsetSourceTriangles(b.reorderedToSourceTriangle, triangleOffset),
    ),
    cwbvhNodeBounds: concatFloat32(a.cwbvhNodeBounds, b.cwbvhNodeBounds),
    cwbvhChildBounds: concatUint16(a.cwbvhChildBounds, b.cwbvhChildBounds),
    cwbvhChildMeta: concatUint32(
      a.cwbvhChildMeta,
      offsetCwbvhMeta(b.cwbvhChildMeta, a.cwbvhNodeCount, triangleOffset),
    ),
    cwbvhChildCount: concatUint32(a.cwbvhChildCount, b.cwbvhChildCount),
    cwbvhNodeCount: a.cwbvhNodeCount + b.cwbvhNodeCount,
  };
}

describe('compressedWideBvh', () => {
  it('proves fixed-stack capacity from the emitted wide-node topology', () => {
    const depth = CWBVH_TRAVERSAL_STACK_DEPTH + 1;
    const nodeCount = depth * 2;
    const childCount = new Uint32Array(nodeCount);
    const childMeta = new Uint32Array(
      nodeCount * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS,
    );

    for (let level = 0; level < depth; level += 1) {
      const deepNode = level * 2;
      const siblingNode = deepNode + 1;
      childCount[siblingNode] = 1;
      let offset =
        siblingNode * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS;
      childMeta[offset] = CWBVH_CHILD_LEAF;

      if (level + 1 < depth) {
        childCount[deepNode] = 2;
        offset = deepNode * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS;
        childMeta[offset] = CWBVH_CHILD_NODE;
        childMeta[offset + 1] = siblingNode;
        offset += CWBVH_CHILD_META_WORDS;
        childMeta[offset] = CWBVH_CHILD_NODE;
        childMeta[offset + 1] = deepNode + 2;
      } else {
        childCount[deepNode] = 1;
        offset = deepNode * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS;
        childMeta[offset] = CWBVH_CHILD_LEAF;
      }
    }

    expect(
      requiredCwbvhTraversalStackEntries(childMeta, childCount),
    ).toBeGreaterThan(CWBVH_TRAVERSAL_STACK_DEPTH);

    const cyclicCounts = new Uint32Array([1]);
    const cyclicMeta = new Uint32Array(
      CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS,
    );
    cyclicMeta[0] = CWBVH_CHILD_NODE;
    cyclicMeta[1] = 0;
    expect(
      requiredCwbvhTraversalStackEntries(cyclicMeta, cyclicCounts),
    ).toBe(Number.POSITIVE_INFINITY);
  });

  it('returns an empty wide root for zero triangles', () => {
    const built = buildCompressedWideBvh(new Float32Array(0), new Uint32Array(0), new Uint32Array(0));

    expect(built.cwbvhNodeCount).toBe(1);
    expect(built.cwbvhChildCount[0]).toBe(0);
    expect(built.cwbvhNodeBounds.length).toBe(6);
    expect(built.cwbvhChildBounds.length).toBe(CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16);
    expect(built.cwbvhChildMeta.length).toBe(CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS);
    expect(built.cwbvhBuildStatus.traversal).toBe('empty');

    const hit = intersectCompressedWideBvhFirstHit(built, new Float32Array(0), {
      origin: [0, 0, 1],
      direction: [0, 0, -1],
    });
    expect(hit.didHit).toBe(false);
  });

  it('packs a single triangle as one wide node with one leaf child', () => {
    const positions = new Float32Array([
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
    ]);
    const indices = new Uint32Array([0, 1, 2, 0]);
    const built = buildCompressedWideBvh(positions, indices, new Uint32Array([3]));

    expect(built.cwbvhNodeCount).toBe(1);
    expect(built.cwbvhChildCount[0]).toBe(1);
    expect(built.cwbvhChildMeta[0]).toBe(CWBVH_CHILD_LEAF);
    expect(built.cwbvhChildMeta[1]).toBe(0);
    expect(built.cwbvhChildMeta[2]).toBe(1);

    const hit = intersectCompressedWideBvhFirstHit(built, positions, {
      origin: [0.25, 0.25, 2],
      direction: [0, 0, -1],
    });
    expect(hit.didHit).toBe(true);
    expect(hit.sourceTriangleIndex).toBe(0);
    expect(hit.dist).toBeCloseTo(2, 6);
  });

  it('collapses a binary tree into bounded 8-wide child slots with deterministic bytes', () => {
    const mesh = makeGrid(6);
    const a = buildCompressedWideBvh(mesh.positions, mesh.indices, mesh.triMaterialIds, { maxLeafTriangles: 1 });
    const b = buildCompressedWideBvh(mesh.positions, mesh.indices, mesh.triMaterialIds, { maxLeafTriangles: 1 });
    const binaryNodeCount = a.bvhNodes.length / 8;

    expect(a.cwbvhNodeCount).toBeGreaterThan(1);
    expect(a.cwbvhNodeCount).toBeLessThan(binaryNodeCount);
    expect(a.cwbvhChildBounds.length).toBe(a.cwbvhNodeCount * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16);
    expect(a.cwbvhChildMeta.length).toBe(a.cwbvhNodeCount * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS);

    for (let node = 0; node < a.cwbvhNodeCount; node += 1) {
      const count = a.cwbvhChildCount[node] ?? 0;
      expect(count).toBeGreaterThanOrEqual(1);
      expect(count).toBeLessThanOrEqual(CWBVH_CHILDREN);
      for (let slot = 0; slot < count; slot += 1) {
        const meta = node * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS + slot * CWBVH_CHILD_META_WORDS;
        const kind = a.cwbvhChildMeta[meta];
        expect(kind === CWBVH_CHILD_NODE || kind === CWBVH_CHILD_LEAF).toBe(true);
      }
    }

    expect(Array.from(a.cwbvhNodeBounds)).toEqual(Array.from(b.cwbvhNodeBounds));
    expect(Array.from(a.cwbvhChildBounds)).toEqual(Array.from(b.cwbvhChildBounds));
    expect(Array.from(a.cwbvhChildMeta)).toEqual(Array.from(b.cwbvhChildMeta));
    expect(Array.from(a.cwbvhChildCount)).toEqual(Array.from(b.cwbvhChildCount));
  });

  it('packs child bounds into explicit u32 words for WGSL storage reads', () => {
    const mesh = makeGrid(3);
    const built = buildCompressedWideBvh(mesh.positions, mesh.indices, mesh.triMaterialIds, { maxLeafTriangles: 1 });
    const packed = packCwbvhBuildBoundsForWgsl(built);

    expect(packed.length).toBe(built.cwbvhNodeCount * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_PACKED_U32);
    for (let i = 0; i < packed.length; i += 1) {
      const lo = built.cwbvhChildBounds[i * 2] ?? 0;
      const hi = built.cwbvhChildBounds[i * 2 + 1] ?? 0;
      expect(packed[i]).toBe((lo | (hi << 16)) >>> 0);
      expect(packed[i]! & 0xffff).toBe(lo);
      expect((packed[i]! >>> 16) & 0xffff).toBe(hi);
    }
  });

  it('rejects malformed odd-length child-bound arrays for WGSL packing', () => {
    expect(() => packCwbvhChildBoundsForWgsl(new Uint16Array([1, 2, 3]))).toThrow(
      /must be even/,
    );
  });

  it('keeps quantized child bounds conservative after dequantization', () => {
    const mesh = makeGrid(4);
    const built = buildCompressedWideBvh(mesh.positions, mesh.indices, mesh.triMaterialIds, { maxLeafTriangles: 1 });

    for (let node = 0; node < built.cwbvhNodeCount; node += 1) {
      const count = built.cwbvhChildCount[node] ?? 0;
      for (let slot = 0; slot < count; slot += 1) {
        const meta = node * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS + slot * CWBVH_CHILD_META_WORDS;
        const kind = built.cwbvhChildMeta[meta] ?? 0;
        const child = cwbvhChildBounds(built, node, slot);
        const expected = {
          min: [Infinity, Infinity, Infinity] as [number, number, number],
          max: [-Infinity, -Infinity, -Infinity] as [number, number, number],
        };

        if (kind === CWBVH_CHILD_NODE) {
          const childNode = built.cwbvhChildMeta[meta + 1] ?? 0;
          const nb = childNode * 6;
          growBounds(expected, [
            built.cwbvhNodeBounds[nb + 0] ?? 0,
            built.cwbvhNodeBounds[nb + 1] ?? 0,
            built.cwbvhNodeBounds[nb + 2] ?? 0,
          ]);
          growBounds(expected, [
            built.cwbvhNodeBounds[nb + 3] ?? 0,
            built.cwbvhNodeBounds[nb + 4] ?? 0,
            built.cwbvhNodeBounds[nb + 5] ?? 0,
          ]);
        } else if (kind === CWBVH_CHILD_LEAF) {
          const triOffset = built.cwbvhChildMeta[meta + 1] ?? 0;
          const triCount = built.cwbvhChildMeta[meta + 2] ?? 0;
          for (let tri = triOffset; tri < triOffset + triCount; tri += 1) {
            const ib = tri * 4;
            growBounds(expected, positionAt(mesh.positions, built.reorderedIndices[ib] ?? 0));
            growBounds(expected, positionAt(mesh.positions, built.reorderedIndices[ib + 1] ?? 0));
            growBounds(expected, positionAt(mesh.positions, built.reorderedIndices[ib + 2] ?? 0));
          }
        }

        for (const axis of [0, 1, 2] as const) {
          expect(child.min[axis]).toBeLessThanOrEqual(expected.min[axis] + 1e-6);
          expect(child.max[axis]).toBeGreaterThanOrEqual(expected.max[axis] - 1e-6);
        }
      }
    }
  });

  it('matches brute-force first-hit results across hit and miss rays', () => {
    const mesh = makeGrid(5);
    const built = buildCompressedWideBvh(mesh.positions, mesh.indices, mesh.triMaterialIds, { maxLeafTriangles: 1 });
    const rays: CwbvhRay[] = [
      { origin: [0.25, 0.75, 3], direction: [0, 0, -1] },
      { origin: [1.25, 2.75, 3], direction: [0, 0, -1] },
      { origin: [4.25, 3.75, 3], direction: [0, 0, -1] },
      { origin: [-0.5, 2, 3], direction: [0, 0, -1] },
      { origin: [2, 5.5, 3], direction: [0, 0, -1] },
    ];

    for (const ray of rays) {
      const brute = bruteForceFirstHit(mesh.positions, mesh.indices, ray);
      const wide = intersectCompressedWideBvhFirstHit(built, mesh.positions, ray);
      expect(wide.didHit).toBe(brute.didHit);
      if (brute.didHit) {
        expect(wide.dist).toBeCloseTo(brute.dist, 6);
        expect(wide.sourceTriangleIndex).toBe(brute.sourceTriangleIndex);
      }
    }
  });

  it('matches brute-force any-hit results across hit and miss rays', () => {
    const mesh = makeGrid(5);
    const built = buildCompressedWideBvh(mesh.positions, mesh.indices, mesh.triMaterialIds, { maxLeafTriangles: 1 });
    const rays: CwbvhRay[] = [
      { origin: [0.25, 0.75, 3], direction: [0, 0, -1] },
      { origin: [2.25, 4.75, 3], direction: [0, 0, -1] },
      { origin: [5.5, 2, 3], direction: [0, 0, -1] },
      { origin: [2, -0.5, 3], direction: [0, 0, -1] },
    ];

    for (const ray of rays) {
      expect(intersectCompressedWideBvhAnyHit(built, mesh.positions, ray)).toBe(
        bruteForceAnyHit(mesh.positions, mesh.indices, ray),
      );
    }
  });

  it('can traverse a nonzero wide root in a concatenated renderer-shaped forest', () => {
    const positions = new Float32Array([
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
      10, 0, 0, 0,
      11, 0, 0, 0,
      10, 1, 0, 0,
    ]);
    const first = buildCompressedWideBvh(
      positions,
      new Uint32Array([0, 1, 2, 0]),
      new Uint32Array([1]),
    );
    const second = buildCompressedWideBvh(
      positions,
      new Uint32Array([3, 4, 5, 0]),
      new Uint32Array([2]),
    );
    const forest = concatCwbvhRoots(first, second);
    const secondRoot = first.cwbvhNodeCount;
    const ray: CwbvhRay = { origin: [10.25, 0.25, 2], direction: [0, 0, -1] };

    expect(intersectCompressedWideBvhFirstHit(forest, positions, ray).didHit).toBe(false);
    expect(intersectCompressedWideBvhAnyHit(forest, positions, ray)).toBe(false);

    const hit = intersectCompressedWideBvhFirstHit(forest, positions, ray, { root: secondRoot });
    expect(hit.didHit).toBe(true);
    expect(hit.dist).toBeCloseTo(2, 6);
    expect(hit.sourceTriangleIndex).toBe(1);
    expect(intersectCompressedWideBvhAnyHit(forest, positions, ray, { root: secondRoot })).toBe(true);

    expect(() => intersectCompressedWideBvhFirstHit(forest, positions, ray, { root: forest.cwbvhNodeCount })).toThrow(
      /CWBVH root/,
    );
  });

  it('mirrors the WGSL skipGlass transmission-nibble filter', () => {
    const positions = new Float32Array([
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
    ]);
    // The lowest positive packed transmission code is already glass; alpha and
    // other low-nibble metadata are intentionally outside this predicate.
    const glassPayload = 1 << 4;
    const built = buildCompressedWideBvh(
      positions,
      new Uint32Array([0, 1, 2, glassPayload]),
      new Uint32Array([0]),
    );
    const withPayload = {
      ...built,
      reorderedIndices: reorderCwbvhTrianglePayloads(built, new Uint32Array([glassPayload])),
    };
    const ray: CwbvhRay = { origin: [0.25, 0.25, 1], direction: [0, 0, -1] };

    expect(built.reorderedIndices[3]).toBe(0);
    expect(withPayload.reorderedIndices[3]).toBe(glassPayload);
    expect(intersectCompressedWideBvhAnyHit(withPayload, positions, ray)).toBe(true);
    expect(intersectCompressedWideBvhAnyHit(withPayload, positions, ray, { skipGlass: true })).toBe(false);
    expect(intersectCompressedWideBvhFirstHit(withPayload, positions, ray).didHit).toBe(true);
    expect(intersectCompressedWideBvhFirstHit(withPayload, positions, ray, { skipGlass: true }).didHit).toBe(false);
    expect(() => reorderCwbvhTrianglePayloads(built, new Uint32Array([glassPayload]), 3)).toThrow(
      /indexStride >= 4/,
    );
  });

  it('intersects a well-conditioned nanometer-scale triangle', () => {
    const positions = new Float32Array([
      -5e-9, -5e-9, 0, 0,
      5e-9, -5e-9, 0, 0,
      0, 5e-9, 0, 0,
    ]);
    const built = buildCompressedWideBvh(
      positions,
      new Uint32Array([0, 1, 2, 0]),
      new Uint32Array([0]),
    );
    const ray: CwbvhRay = {
      origin: [0, 0, 1],
      direction: [0, 0, -1],
    };

    const hit = intersectCompressedWideBvhFirstHit(built, positions, ray);
    expect(hit.didHit).toBe(true);
    expect(hit.dist).toBeCloseTo(1, 6);
    expect(intersectCompressedWideBvhAnyHit(built, positions, ray)).toBe(true);
  });
});
