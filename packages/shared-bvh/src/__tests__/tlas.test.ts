import { describe, it, expect, vi } from 'vitest';
import { buildTlas, refitTlas, tlasIntersect, type TlasInstance } from '../tlas.js';

const IDENT16 = (): Float32Array => new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function makeInstance(
  blasId: number,
  min: [number, number, number],
  max: [number, number, number],
): TlasInstance {
  return { blasId, aabbMin: min, aabbMax: max, worldToLocal: IDENT16() };
}

describe('buildTlas', () => {
  it('builds a single-instance TLAS with the root AABB equal to the instance AABB', () => {
    const inst = makeInstance(0, [0, 0, 0], [1, 1, 1]);
    const tlas = buildTlas([inst]);
    expect(tlas.nodeCount).toBe(1);
    const f32 = new Float32Array(tlas.nodes.buffer);
    expect(f32[0]).toBe(0); expect(f32[1]).toBe(0); expect(f32[2]).toBe(0);
    expect(f32[3]).toBe(1); expect(f32[4]).toBe(1); expect(f32[5]).toBe(1);
    // Leaf flag in u32[7].
    expect(tlas.nodes[7]! >>> 16).toBe(0xffff);
    expect((tlas.nodes[7]! & 0x0000ffff)).toBe(1);
  });

  it('builds a two-instance TLAS — root AABB is the union of both', () => {
    const a = makeInstance(0, [0, 0, 0], [1, 1, 1]);
    const b = makeInstance(1, [10, 10, 10], [11, 11, 11]);
    const tlas = buildTlas([a, b]);
    expect(tlas.nodeCount).toBe(3);     // root + 2 leaves
    const f32 = new Float32Array(tlas.nodes.buffer);
    expect(f32[0]).toBe(0);   expect(f32[1]).toBe(0);   expect(f32[2]).toBe(0);
    expect(f32[3]).toBe(11);  expect(f32[4]).toBe(11);  expect(f32[5]).toBe(11);
    // Root is interior — high bits ≠ TLAS leaf flag.
    expect(tlas.nodes[7]! >>> 16).not.toBe(0xffff);
  });

  it('preserves blasId per-instance', () => {
    const a = makeInstance(7, [0, 0, 0], [1, 1, 1]);
    const b = makeInstance(42, [5, 0, 0], [6, 1, 1]);
    const c = makeInstance(99, [10, 0, 0], [11, 1, 1]);
    const tlas = buildTlas([a, b, c]);
    expect(Array.from(tlas.blasRoots)).toEqual([7, 42, 99]);
  });

  it('preserves worldToLocal transforms in original input order', () => {
    const t1 = new Float32Array(IDENT16());
    t1[12] = 100;
    const t2 = new Float32Array(IDENT16());
    t2[12] = -50;
    const tlas = buildTlas([
      { blasId: 0, aabbMin: [0,0,0], aabbMax: [1,1,1], worldToLocal: t1 },
      { blasId: 1, aabbMin: [50,0,0], aabbMax: [51,1,1], worldToLocal: t2 },
    ]);
    expect(tlas.instanceTransforms[0 * 16 + 12]).toBe(100);
    expect(tlas.instanceTransforms[1 * 16 + 12]).toBe(-50);
  });

  it('throws on empty input', () => {
    expect(() => buildTlas([])).toThrow(/empty/);
  });

  it('throws on inverted AABB', () => {
    expect(() => buildTlas([
      { blasId: 0, aabbMin: [5,0,0], aabbMax: [0,1,1], worldToLocal: IDENT16() },
    ])).toThrow(/inverted/);
  });

  it('throws on malformed worldToLocal length', () => {
    expect(() => buildTlas([
      { blasId: 0, aabbMin: [0,0,0], aabbMax: [1,1,1], worldToLocal: new Float32Array(12) },
    ])).toThrow(/length 12/);
  });

  it('filters non-finite instances instead of poisoning the root AABB', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const tlas = buildTlas([
        makeInstance(0, [0, 0, 0], [1, 1, 1]),
        makeInstance(1, [Number.NaN, 5, 5], [6, 6, 6]),
      ]);

      expect(tlas.nodeCount).toBe(1);
      expect(Array.from(tlas.instanceIndices)).toEqual([0]);
      expect(Array.from(tlas.blasRoots)).toEqual([0, 1]);
      const f32 = new Float32Array(tlas.nodes.buffer);
      for (let i = 0; i < 6; i += 1) {
        expect(Number.isFinite(f32[i]!)).toBe(true);
      }
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Instance 1 has non-finite'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('filters instances with non-finite worldToLocal transforms', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const badTransform = IDENT16();
      badTransform[12] = Infinity;
      const tlas = buildTlas([
        makeInstance(0, [0, 0, 0], [1, 1, 1]),
        { blasId: 1, aabbMin: [5, 5, 5], aabbMax: [6, 6, 6], worldToLocal: badTransform },
      ]);

      expect(Array.from(tlas.instanceIndices)).toEqual([0]);
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Instance 1 has non-finite'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('throws clearly when every TLAS instance is non-finite', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => buildTlas([
        makeInstance(0, [Number.NaN, 0, 0], [1, 1, 1]),
      ])).toThrow(/no valid finite instances/);
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('tlasIntersect', () => {
  it('returns the single instance for a ray that hits it dead-on', () => {
    const inst = makeInstance(0, [0, 0, 0], [1, 1, 1]);
    const tlas = buildTlas([inst]);
    const hits = tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0]);
    expect(hits).toEqual([0]);
  });

  it('returns empty for a ray that misses every instance', () => {
    const inst = makeInstance(0, [0, 0, 0], [1, 1, 1]);
    const tlas = buildTlas([inst]);
    const hits = tlasIntersect(tlas, [-5, 100, 100], [1, 0, 0]);
    expect(hits).toEqual([]);
  });

  it('returns only the instance the ray hits when two are separated along X', () => {
    const a = makeInstance(0, [0, 0, 0], [1, 1, 1]);
    const b = makeInstance(1, [100, 0, 0], [101, 1, 1]);
    const tlas = buildTlas([a, b]);
    const hitsLeft = tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0], 10);
    expect(hitsLeft.sort()).toEqual([0]);
    const hitsThroughBoth = tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0]);
    expect(hitsThroughBoth.sort()).toEqual([0, 1]);
  });

  it('returns all instances when their AABBs overlap', () => {
    const a = makeInstance(0, [0, 0, 0], [2, 2, 2]);
    const b = makeInstance(1, [1, 1, 1], [3, 3, 3]);
    const tlas = buildTlas([a, b]);
    const hits = tlasIntersect(tlas, [-5, 1.5, 1.5], [1, 0, 0]);
    expect(hits.sort()).toEqual([0, 1]);
  });

  it('handles ray with zero direction component (parallel-to-slab edge case)', () => {
    const inst = makeInstance(0, [0, 0, 0], [1, 1, 1]);
    const tlas = buildTlas([inst]);
    // Direction has zero Y → parallel to Y slab. Should still hit.
    const hits = tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0]);
    expect(hits).toEqual([0]);
  });
});

describe('refitTlas', () => {
  it('refits the leaf AABB after an instance moves', () => {
    const a = makeInstance(0, [0, 0, 0], [1, 1, 1]);
    const b = makeInstance(1, [10, 0, 0], [11, 1, 1]);
    const tlas = buildTlas([a, b]);

    // Move instance 1 to a new world position.
    refitTlas(tlas, [
      { min: [0, 0, 0], max: [1, 1, 1] },                   // unchanged
      { min: [100, 0, 0], max: [101, 1, 1] },               // moved
    ]);

    // Root AABB now reflects new bounds.
    const f32 = new Float32Array(tlas.nodes.buffer);
    expect(f32[3]).toBeCloseTo(101);

    // Ray that previously hit (10..11) should now MISS, but a ray at x=100 should HIT.
    expect(tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0], 15)).toEqual([0]);
    expect(tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0]).sort()).toEqual([0, 1]);
  });

  it('throws when AABB count mismatches instance count', () => {
    const tlas = buildTlas([
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [5, 0, 0], [6, 1, 1]),
    ]);
    expect(() => refitTlas(tlas, [
      { min: [0, 0, 0], max: [1, 1, 1] },
    ])).toThrow(/expected 2 AABBs/);
  });

  it('rejects non-finite refit AABBs before mutating node bounds', () => {
    const tlas = buildTlas([
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [5, 0, 0], [6, 1, 1]),
    ]);
    const before = Array.from(new Float32Array(tlas.nodes.buffer).slice(0, 6));

    expect(() => refitTlas(tlas, [
      { min: [0, 0, 0], max: [1, 1, 1] },
      { min: [Number.NaN, 0, 0], max: [6, 1, 1] },
    ])).toThrow(/non-finite AABB/);

    expect(Array.from(new Float32Array(tlas.nodes.buffer).slice(0, 6))).toEqual(before);
  });

  it('rejects inverted refit AABBs before mutating node bounds', () => {
    const tlas = buildTlas([
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [5, 0, 0], [6, 1, 1]),
    ]);
    const before = Array.from(new Float32Array(tlas.nodes.buffer).slice(0, 6));

    expect(() => refitTlas(tlas, [
      { min: [0, 0, 0], max: [1, 1, 1] },
      { min: [7, 0, 0], max: [6, 1, 1] },
    ])).toThrow(/inverted AABB/);

    expect(Array.from(new Float32Array(tlas.nodes.buffer).slice(0, 6))).toEqual(before);
  });

  it('refit preserves traversal correctness with many instances', () => {
    // 10 instances arrayed along X.
    const insts: TlasInstance[] = [];
    for (let i = 0; i < 10; i++) {
      insts.push(makeInstance(i, [i * 10, 0, 0], [i * 10 + 1, 1, 1]));
    }
    const tlas = buildTlas(insts);

    // Refit identity (same AABBs).
    refitTlas(tlas, insts.map((i) => ({ min: i.aabbMin, max: i.aabbMax })));
    // Ray at y=0.5,z=0.5 along +X should hit all 10.
    const hits = tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0]);
    expect(hits.sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('buildTlas — larger spatial distribution', () => {
  it('100 random-grid instances each get correctly classified by traversal', () => {
    const insts: TlasInstance[] = [];
    // 10×10 grid in the XY plane.
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) {
        const id = y * 10 + x;
        insts.push(makeInstance(
          id,
          [x * 2.0, y * 2.0, 0],
          [x * 2.0 + 1, y * 2.0 + 1, 1],
        ));
      }
    }
    // Use maxLeafInstances=1 so the candidate set equals the precise hit set
    // (see tlasIntersect docstring re: leaf-level conservatism).
    const tlas = buildTlas(insts, { maxLeafInstances: 1 });
    // 100 leaves at maxLeafInstances=1; the binary tree built by SAH adds
    // at most N-1 interior nodes for N leaves (a balanced tree hits that
    // ceiling). 100 leaves → up to 199 total nodes; minimum is 100 if
    // every record were a leaf-only fallback (which can't happen at
    // N=100 with distinct centroids).
    expect(tlas.nodeCount).toBeGreaterThanOrEqual(100);
    expect(tlas.nodeCount).toBeLessThanOrEqual(199);

    // Ray along +X at row y=3.5 hits the 10 instances on that row (y ∈ [6, 7]).
    // (3 * 2 = 6, instance AABB is [6..7] on Y; ray y=6.5 misses none on that row.)
    const hits = tlasIntersect(tlas, [-1, 6.5, 0.5], [1, 0, 0]);
    expect(hits.length).toBe(10);
    // All should be from row y=3 → ids 30..39.
    for (const h of hits) {
      expect(h).toBeGreaterThanOrEqual(30);
      expect(h).toBeLessThan(40);
    }
  });

  it('returns candidates for maxLeafInstances>1 — caller responsible for sub-leaf filtering', () => {
    const insts: TlasInstance[] = [];
    for (let i = 0; i < 4; i++) {
      // Cluster 4 instances in a tight bucket so they end up in one leaf.
      insts.push(makeInstance(i, [0, i, 0], [1, i + 0.4, 1]));
    }
    const tlas = buildTlas(insts, { maxLeafInstances: 4 });
    expect(tlas.nodeCount).toBe(1);   // single leaf
    // Ray at y=0.2 only physically hits instance 0 (y range [0, 0.4]).
    // But because the leaf union covers y=[0, 3.4], the candidate set
    // includes all 4 — caller must re-test against per-instance AABBs.
    const hits = tlasIntersect(tlas, [-5, 0.2, 0.5], [1, 0, 0]);
    expect(hits.length).toBe(4);
  });
});
