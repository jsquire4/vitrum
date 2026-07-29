import { describe, it, expect, vi } from 'vitest';
import {
  TLAS_MAX_BUILD_DEPTH,
  buildTlas,
  refitTlas,
  refitTlasInstances,
  tlasRefitNodeIndices,
  tlasIntersect,
  validateTlasBuild,
  type TlasBufferView,
  type TlasInstance,
} from '../tlas.js';
import { TLAS_TRAVERSAL_STACK_DEPTH } from '../strides.js';

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

function makeCombTlas(interiorDepth: number): TlasBufferView {
  const nodeCount = interiorDepth * 2 + 1;
  const instanceCount = interiorDepth + 1;
  const nodes = new Uint32Array(nodeCount * 8);
  const bounds = new Float32Array(nodes.buffer);
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    const base = nodeIndex * 8;
    bounds[base] = 0;
    bounds[base + 1] = 0;
    bounds[base + 2] = 0;
    bounds[base + 3] = 1;
    bounds[base + 4] = 1;
    bounds[base + 5] = 1;
  }
  for (let depth = 0; depth < interiorDepth; depth += 1) {
    const base = depth * 8;
    const rightChild = interiorDepth * 2 - depth;
    nodes[base + 6] = rightChild - depth;
    nodes[base + 7] = 0;
  }
  nodes[interiorDepth * 8 + 6] = 0;
  nodes[interiorDepth * 8 + 7] = 0xffff0001;
  for (let depth = interiorDepth - 1; depth >= 0; depth -= 1) {
    const nodeIndex = interiorDepth * 2 - depth;
    nodes[nodeIndex * 8 + 6] = interiorDepth - depth;
    nodes[nodeIndex * 8 + 7] = 0xffff0001;
  }
  const instanceTransforms = new Float32Array(instanceCount * 16);
  for (let i = 0; i < instanceCount; i += 1) {
    instanceTransforms.set(IDENT16(), i * 16);
  }
  return {
    nodes,
    nodeCount,
    instanceIndices: Uint32Array.from({ length: instanceCount }, (_, index) => index),
    blasRoots: new Uint32Array(instanceCount),
    instanceTransforms,
  };
}

/** Deep-right comb: the validator's fixed left-first walk retains almost no
 * siblings, while a ray that visits right first retains one at every level. */
function makeRightCombTlas(interiorDepth: number): TlasBufferView {
  const nodeCount = interiorDepth * 2 + 1;
  const instanceCount = interiorDepth + 1;
  const nodes = new Uint32Array(nodeCount * 8);
  const bounds = new Float32Array(nodes.buffer);
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    const base = nodeIndex * 8;
    bounds[base] = 0;
    bounds[base + 1] = 0;
    bounds[base + 2] = 0;
    bounds[base + 3] = 1;
    bounds[base + 4] = 1;
    bounds[base + 5] = 1;
  }
  for (let depth = 0; depth < interiorDepth; depth += 1) {
    const interior = depth * 2;
    nodes[interior * 8 + 6] = 2;
    nodes[interior * 8 + 7] = 0;
    const leftLeaf = interior + 1;
    nodes[leftLeaf * 8 + 6] = depth;
    nodes[leftLeaf * 8 + 7] = 0xffff0001;
  }
  const finalLeaf = interiorDepth * 2;
  nodes[finalLeaf * 8 + 6] = interiorDepth;
  nodes[finalLeaf * 8 + 7] = 0xffff0001;
  const instanceTransforms = new Float32Array(instanceCount * 16);
  for (let i = 0; i < instanceCount; i += 1) {
    instanceTransforms.set(IDENT16(), i * 16);
  }
  return {
    nodes,
    nodeCount,
    instanceIndices: Uint32Array.from(
      { length: instanceCount },
      (_, index) => index,
    ),
    blasRoots: new Uint32Array(instanceCount),
    instanceTransforms,
  };
}

function copyIntoNodeSubview(data: TlasBufferView): {
  readonly view: TlasBufferView;
  readonly backing: Uint32Array;
  readonly prefixWords: number;
} {
  const prefixWords = 5;
  const backing = new Uint32Array(prefixWords + data.nodes.length + 7);
  backing.fill(0xdeadbeef);
  backing.set(data.nodes, prefixWords);
  return {
    view: {
      ...data,
      nodes: backing.subarray(prefixWords, prefixWords + data.nodes.length),
    },
    backing,
    prefixWords,
  };
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
    ])).toThrow(/exact 16-element Float32Array/);
  });

  it('rejects malformed numeric build options instead of coercing them', () => {
    const instances = [makeInstance(0, [0, 0, 0], [1, 1, 1])];
    for (const maxLeafInstances of [0, 1.5, Number.NaN, Infinity, 0x10000]) {
      expect(() => buildTlas(instances, { maxLeafInstances })).toThrow(
        /maxLeafInstances must be a finite safe integer/,
      );
    }
    for (const numBins of [1, 2.5, Number.NaN, Infinity, 257]) {
      expect(() => buildTlas(instances, { numBins })).toThrow(
        /numBins must be a finite safe integer/,
      );
    }
    for (const maxDepth of [-1, 1.5, Number.NaN, Infinity, TLAS_MAX_BUILD_DEPTH + 1]) {
      expect(() => buildTlas(instances, { maxDepth })).toThrow(
        /maxDepth must be a finite safe integer/,
      );
    }
  });

  it('deterministically balances coincident centroids and publishes exact stack proof', () => {
    const instances = Array.from(
      { length: 2_048 },
      (_, index) => makeInstance(index, [0, 0, 0], [1, 1, 1]),
    );
    const first = buildTlas(instances, { maxLeafInstances: 1 });
    const second = buildTlas(instances, { maxLeafInstances: 1 });

    expect(first.buildStatus.balancedFallbackReasons['degenerate-centroids']).toBeGreaterThan(0);
    expect(first.buildStatus.validation.maxDepth).toBe(11);
    expect(first.buildStatus.validation.maxLeafInstances).toBe(1);
    expect(first.buildStatus.validation.maxTraversalStackEntries).toBeLessThanOrEqual(
      TLAS_TRAVERSAL_STACK_DEPTH,
    );
    expect(first.buildStatus.validation.traversalStackCapacity).toBe(
      TLAS_TRAVERSAL_STACK_DEPTH,
    );
    expect(first.nodes).toEqual(second.nodes);
    expect(first.instanceIndices).toEqual(second.instanceIndices);
  });

  it('replaces a depth-unsafe skewed SAH split with a deterministic median split', () => {
    const instances = Array.from({ length: 16 }, (_, index) => {
      const x = index === 15 ? 1_000_000 : index;
      return makeInstance(index, [x, 0, 0], [x + 0.5, 1, 1]);
    });
    const tlas = buildTlas(instances, {
      maxLeafInstances: 1,
      maxDepth: 4,
      numBins: 16,
    });
    expect(tlas.buildStatus.balancedFallbackReasons['depth-safety']).toBeGreaterThan(0);
    expect(tlas.buildStatus.validation.maxDepth).toBe(4);
    expect(tlas.buildStatus.validation.maxTraversalStackEntries).toBe(5);
    expect(new Set(tlas.instanceIndices).size).toBe(16);
  });

  it('rejects a non-finite instance instead of warning and filtering it', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => buildTlas([
        makeInstance(0, [0, 0, 0], [1, 1, 1]),
        makeInstance(1, [Number.NaN, 5, 5], [6, 6, 6]),
      ])).toThrow(/aabbMin\[0\].*finite/);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('rejects instances with non-finite worldToLocal transforms', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const badTransform = IDENT16();
      badTransform[12] = Infinity;
      expect(() => buildTlas([
        makeInstance(0, [0, 0, 0], [1, 1, 1]),
        { blasId: 1, aabbMin: [5, 5, 5], aabbMax: [6, 6, 6], worldToLocal: badTransform },
      ])).toThrow(/non-finite AABB or worldToLocal/);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('rejects the first non-finite TLAS instance without warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => buildTlas([
        makeInstance(0, [Number.NaN, 0, 0], [1, 1, 1]),
      ])).toThrow(/aabbMin\[0\].*finite/);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('validateTlasBuild', () => {
  it('reports order-independent stack occupancy for both comb orientations', () => {
    for (const data of [
      makeCombTlas(TLAS_MAX_BUILD_DEPTH),
      makeRightCombTlas(TLAS_MAX_BUILD_DEPTH),
    ]) {
      const report = validateTlasBuild(data, {
        maxDepth: TLAS_MAX_BUILD_DEPTH,
        maxLeafInstances: 1,
      });
      expect(report.maxDepth).toBe(TLAS_MAX_BUILD_DEPTH);
      expect(report.maxTraversalStackEntries).toBe(TLAS_MAX_BUILD_DEPTH + 1);
      expect(report.maxTraversalStackEntries).toBeLessThanOrEqual(
        TLAS_TRAVERSAL_STACK_DEPTH,
      );
    }
  });

  it('rejects a comb deeper than the canonical accepted build depth', () => {
    expect(() => validateTlasBuild(makeCombTlas(TLAS_MAX_BUILD_DEPTH + 1))).toThrow(
      /depth .* exceeds maximum|reaches maximum depth/,
    );
  });

  it('rejects duplicate permutation ranges and unreachable nodes', () => {
    const duplicate = makeCombTlas(2);
    duplicate.nodes[3 * 8 + 6] = 0;
    expect(() => validateTlasBuild(duplicate)).toThrow(
      /permutation slot 0 is referenced more than once/,
    );

    const unreachable = makeCombTlas(2);
    unreachable.nodes[6] = 2;
    expect(() => validateTlasBuild(unreachable)).toThrow(
      /referenced more than once|unreachable/,
    );
  });

  it('rejects duplicate original-instance references within distinct permutation slots', () => {
    const duplicateInstance = buildTlas([
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [2, 0, 0], [3, 1, 1]),
    ]);
    duplicateInstance.instanceIndices[1] = duplicateInstance.instanceIndices[0]!;

    expect(() => validateTlasBuild(duplicateInstance)).toThrow(
      /original instance .* referenced more than once/,
    );
  });

  it('rejects interior bounds that do not enclose either child exactly', () => {
    const instances = [
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [10, 0, 0], [11, 1, 1]),
    ];
    const undersizedMax = buildTlas(instances);
    new Float32Array(undersizedMax.nodes.buffer)[3] = 10.5;
    expect(() => validateTlasBuild(undersizedMax)).toThrow(
      /bounds do not enclose child/,
    );

    const oversizedMin = buildTlas(instances);
    new Float32Array(oversizedMin.nodes.buffer)[0] = 0.5;
    expect(() => validateTlasBuild(oversizedMin)).toThrow(
      /bounds do not enclose child/,
    );
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

  it('treats signed zero and subnormal directions as parallel without boundary NaNs', () => {
    const tlas = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    for (const parallel of [0, -0, Number.MIN_VALUE, -Number.MIN_VALUE]) {
      expect(tlasIntersect(tlas, [-5, 0, 0.5], [1, parallel, 0])).toEqual([0]);
      expect(tlasIntersect(tlas, [-5, 1, 0.5], [1, parallel, -0])).toEqual([0]);
      expect(tlasIntersect(tlas, [-5, -Number.MIN_VALUE, 0.5], [1, parallel, 0])).toEqual([]);
      expect(tlasIntersect(tlas, [-5, 1 + Number.EPSILON, 0.5], [1, parallel, 0])).toEqual([]);
    }
  });

  it('uses an inclusive tMax boundary, matching the WGSL TLAS slab test', () => {
    const tlas = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    expect(tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0], 5)).toEqual([0]);
    expect(tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0], 5 - 1e-6)).toEqual([]);
  });

  it('deterministically rejects non-finite rays and invalid tMax values', () => {
    const tlas = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    expect(tlasIntersect(tlas, [Number.NaN, 0.5, 0.5], [1, 0, 0])).toEqual([]);
    expect(tlasIntersect(tlas, [-5, 0.5, 0.5], [Infinity, 0, 0])).toEqual([]);
    expect(tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0], Number.NaN)).toEqual([]);
    expect(tlasIntersect(tlas, [-5, 0.5, 0.5], [1, 0, 0], -Infinity)).toEqual([]);
  });

  it('rejects a zero-length ray direction', () => {
    const tlas = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    expect(tlasIntersect(tlas, [0.5, 0.5, 0.5], [0, -0, 0])).toEqual([]);
  });

  it('reads packed nodes relative to a Uint32Array subview byte offset', () => {
    const original = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    const { view, backing, prefixWords } = copyIntoNodeSubview(original);

    expect(tlasIntersect(view, [-5, 0.5, 0.5], [1, 0, 0])).toEqual([0]);
    expect(Array.from(backing.slice(0, prefixWords))).toEqual(
      Array.from({ length: prefixWords }, () => 0xdeadbeef),
    );
  });

  it('rejects truncated buffers and corrupt child, leaf, instance, and bound metadata', () => {
    const two = buildTlas([
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [2, 0, 0], [3, 1, 1]),
    ]);
    expect(() => tlasIntersect({ ...two, nodes: two.nodes.slice(0, 8) }, [-5, 0.5, 0.5], [1, 0, 0]))
      .toThrow(/shorter than nodeCount/);

    const badChild = buildTlas([
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [2, 0, 0], [3, 1, 1]),
    ]);
    badChild.nodes[6] = 1;
    expect(() => tlasIntersect(badChild, [-5, 0.5, 0.5], [1, 0, 0]))
      .toThrow(/invalid child references/);

    const badLeaf = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    badLeaf.nodes[6] = 1;
    expect(() => tlasIntersect(badLeaf, [-5, 0.5, 0.5], [1, 0, 0]))
      .toThrow(/invalid leaf range/);

    const badInstance = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    badInstance.instanceIndices[0] = 1;
    expect(() => tlasIntersect(badInstance, [-5, 0.5, 0.5], [1, 0, 0]))
      .toThrow(/instance reference 1/);

    const badBounds = buildTlas([makeInstance(0, [0, 0, 0], [1, 1, 1])]);
    new Float32Array(badBounds.nodes.buffer)[0] = Number.NaN;
    expect(() => tlasIntersect(badBounds, [-5, 0.5, 0.5], [1, 0, 0]))
      .toThrow(/invalid bounds/);
  });
});

describe('refitTlas', () => {
  it('refits only the changed instance leaf and its ancestors', () => {
    const instances = Array.from({ length: 8 }, (_, index) =>
      makeInstance(index, [index * 10, 0, 0], [index * 10 + 1, 1, 1]));
    const tlas = buildTlas(instances, { maxLeafInstances: 1 });
    const before = new Uint8Array(tlas.nodes.buffer.slice(0));
    const aabbs = instances.map((instance) => ({
      min: instance.aabbMin,
      max: instance.aabbMax,
    }));
    aabbs[5] = { min: [500, 0, 0], max: [501, 1, 1] };

    const affected = refitTlasInstances(tlas, aabbs, [5]);
    const affectedSet = new Set(affected);
    expect(affected.length).toBeGreaterThan(1);
    expect(affected.length).toBeLessThan(tlas.nodeCount);
    for (let node = 0; node < tlas.nodeCount; node += 1) {
      const current = new Uint8Array(tlas.nodes.buffer, node * 32, 32);
      const previous = before.slice(node * 32, node * 32 + 32);
      if (!affectedSet.has(node)) {
        expect(Array.from(current)).toEqual(Array.from(previous));
      }
    }
    expect(new Float32Array(tlas.nodes.buffer)[3]).toBeCloseTo(501);
  });

  it('retains unaffected instances when a changed instance shares their leaf', () => {
    const instances = [
      makeInstance(0, [-10, 0, 0], [-9, 1, 1]),
      makeInstance(1, [0, 0, 0], [1, 1, 1]),
      makeInstance(2, [10, 0, 0], [11, 1, 1]),
      makeInstance(3, [20, 0, 0], [21, 1, 1]),
    ];
    const tlas = buildTlas(instances, { maxLeafInstances: 4 });
    const permutationBefore = Array.from(tlas.instanceIndices);
    const transformsBefore = Array.from(tlas.instanceTransforms);
    const aabbs = instances.map((instance) => ({
      min: instance.aabbMin,
      max: instance.aabbMax,
    }));
    aabbs[1] = { min: [2, 0, 0], max: [3, 1, 1] };

    expect(Array.from(refitTlasInstances(tlas, aabbs, [1]))).toEqual([0]);
    const bounds = new Float32Array(
      tlas.nodes.buffer,
      tlas.nodes.byteOffset,
      tlas.nodes.length,
    );
    expect(Array.from(bounds.slice(0, 6))).toEqual([-10, 0, 0, 21, 1, 1]);
    expect(Array.from(tlas.instanceIndices)).toEqual(permutationBefore);
    expect(Array.from(tlas.instanceTransforms)).toEqual(transformsBefore);
  });

  it('deduplicates repeated changed instances and overlapping ancestor paths', () => {
    const instances = Array.from({ length: 8 }, (_, index) =>
      makeInstance(index, [index * 10, 0, 0], [index * 10 + 1, 1, 1]));
    const tlas = buildTlas(instances, { maxLeafInstances: 1 });
    const repeated = Array.from(tlasRefitNodeIndices(tlas, [1, 1, 2, 2]));
    const unique = Array.from(tlasRefitNodeIndices(tlas, [1, 2]));

    expect(repeated).toEqual(unique);
    expect(repeated.length).toBe(new Set(repeated).size);
    expect(repeated.at(-1)).toBe(0);

    const order = new Map(repeated.map((node, index) => [node, index]));
    const parents = new Int32Array(tlas.nodeCount);
    parents.fill(-1);
    for (let node = 0; node < tlas.nodeCount; node += 1) {
      const base = node * 8;
      if ((tlas.nodes[base + 7]! >>> 16) === 0xffff) continue;
      parents[node + 1] = node;
      parents[node + tlas.nodes[base + 6]!] = node;
    }
    for (const node of repeated) {
      const parent = parents[node]!;
      if (parent >= 0 && order.has(parent)) {
        expect(order.get(node)!).toBeLessThan(order.get(parent)!);
      }
    }
  });

  it('refits a Uint32Array node subview without touching backing sentinels', () => {
    const instances = [
      makeInstance(0, [0, 0, 0], [1, 1, 1]),
      makeInstance(1, [10, 0, 0], [11, 1, 1]),
    ];
    const original = buildTlas(instances);
    const { view, backing, prefixWords } = copyIntoNodeSubview(original);
    refitTlas(view, [
      { min: [0, 0, 0], max: [1, 1, 1] },
      { min: [100, 0, 0], max: [101, 1, 1] },
    ]);

    expect(Array.from(backing.slice(0, prefixWords))).toEqual(
      Array.from({ length: prefixWords }, () => 0xdeadbeef),
    );
    expect(Array.from(backing.slice(prefixWords + view.nodes.length))).toEqual(
      Array.from({ length: 7 }, () => 0xdeadbeef),
    );
    const bounds = new Float32Array(
      view.nodes.buffer,
      view.nodes.byteOffset,
      view.nodes.length,
    );
    expect(bounds[3]).toBeCloseTo(101);
  });

  it('cannot construct a topology that silently omits a malformed instance', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => buildTlas([
        makeInstance(0, [0, 0, 0], [1, 1, 1]),
        makeInstance(1, [Number.NaN, 0, 0], [2, 1, 1]),
      ])).toThrow(/aabbMin\[0\].*finite/);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

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
