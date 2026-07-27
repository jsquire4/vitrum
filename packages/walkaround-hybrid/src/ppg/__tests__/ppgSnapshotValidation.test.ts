import { describe, expect, it } from 'vitest';
import { dTreeAccumulateFlux } from '../dTree.js';
import { buildSTree, splitOverflowLeaves } from '../sTree.js';
import {
  deserialiseSTree,
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  serialiseSTree,
  STREE_HEADER_F32,
  STREE_NODE_F32,
  type SerialisedSTree,
} from '../serialise.js';
import { validateSerialisedSTree } from '../validateSerialisedSTree.js';

const BOUNDS = {
  min: [0, 0, 0] as [number, number, number],
  max: [2, 1, 1] as [number, number, number],
};
const OPTIONS = {
  maxSpatialCells: 8,
  maxDTreeNodesPerCell: 341,
  sceneBounds: BOUNDS,
};

function clonePacked(value: SerialisedSTree): SerialisedSTree {
  return {
    sTreeBuf: new Float32Array(value.sTreeBuf),
    dTreeBuf: new Float32Array(value.dTreeBuf),
    dTreeOffsets: new Uint32Array(value.dTreeOffsets),
  };
}

function canonical(): SerialisedSTree {
  const tree = buildSTree(BOUNDS, 2);
  dTreeAccumulateFlux(tree.dTrees[0]!, [0.1, 0.2], 3);
  dTreeAccumulateFlux(tree.dTrees[0]!, [0.9, 0.8], 7);
  return serialiseSTree(tree, OPTIONS.maxDTreeNodesPerCell);
}

function splitSpatialTree(): SerialisedSTree {
  const tree = buildSTree(BOUNDS, 1);
  splitOverflowLeaves(tree, 0, 8, new Uint32Array([1]));
  splitOverflowLeaves(tree, 0, 8, new Uint32Array([1, 0]));
  return serialiseSTree(tree, OPTIONS.maxDTreeNodesPerCell);
}

function expectInvalid(value: SerialisedSTree, pattern: RegExp): void {
  expect(() => validateSerialisedSTree(value, OPTIONS)).toThrow(pattern);
}

describe('validateSerialisedSTree', () => {
  it('accepts the canonical serializer and pins an exact deserialize/reserialize roundtrip', () => {
    const packed = canonical();
    expect(() => validateSerialisedSTree(packed, OPTIONS)).not.toThrow();

    const restored = deserialiseSTree(packed, BOUNDS);
    const repacked = serialiseSTree(restored, OPTIONS.maxDTreeNodesPerCell);
    expect(repacked.sTreeBuf).toEqual(packed.sTreeBuf);
    expect(repacked.dTreeBuf).toEqual(packed.dTreeBuf);
    expect(repacked.dTreeOffsets).toEqual(packed.dTreeOffsets);
  });

  it('rejects NaN, negative flux, inconsistent subtree mass, and oversize node counts', () => {
    const nan = clonePacked(canonical());
    nan.dTreeBuf[DTREE_HEADER_F32 + 4] = Number.NaN;
    expectInvalid(nan, /must be finite/);

    const negative = clonePacked(canonical());
    negative.dTreeBuf[DTREE_HEADER_F32 + 4] = -1;
    expectInvalid(negative, /negative flux/);

    const mass = clonePacked(canonical());
    mass.dTreeBuf[DTREE_HEADER_F32 + 4] = mass.dTreeBuf[DTREE_HEADER_F32 + 4]! + 1;
    expectInvalid(mass, /subtree flux|total flux/);

    const oversize = clonePacked(canonical());
    oversize.dTreeBuf[0] = OPTIONS.maxDTreeNodesPerCell + 1;
    expectInvalid(oversize, /node count exceeds its cap/);
  });

  it('rejects reserved words and one-float-step directional partition drift', () => {
    const sTreeHeaderReserved = clonePacked(canonical());
    sTreeHeaderReserved.sTreeBuf[2] = 1;
    expectInvalid(sTreeHeaderReserved, /sTree reserved header words/);

    const sTreeNodeReserved = clonePacked(canonical());
    sTreeNodeReserved.sTreeBuf[STREE_HEADER_F32 + 11] = 1;
    expectInvalid(sTreeNodeReserved, /sTree node 0 reserved field 11/);

    const dTreeHeaderReserved = clonePacked(canonical());
    dTreeHeaderReserved.dTreeBuf[3] = 1;
    expectInvalid(dTreeHeaderReserved, /dTree 0 reserved header word/);

    const directionalPartition = clonePacked(canonical());
    const firstChild = Number(
      directionalPartition.dTreeBuf[DTREE_HEADER_F32 + 6],
    );
    const firstChildBase =
      DTREE_HEADER_F32 + firstChild * DTREE_NODE_F32;
    directionalPartition.dTreeBuf[firstChildBase] = 2 ** -149;
    expectInvalid(directionalPartition, /dTree 0 child bounds/);
  });

  it('rejects dTree cycles/shared children, unreachable data, bad offsets, and truncation', () => {
    const cycle = clonePacked(canonical());
    cycle.dTreeBuf[DTREE_HEADER_F32 + 6] = 0;
    expectInvalid(cycle, /invalid children|cyclic/);

    const shared = clonePacked(canonical());
    const secondInterior = DTREE_HEADER_F32 + 2 * DTREE_NODE_F32;
    shared.dTreeBuf[secondInterior + 6] = shared.dTreeBuf[DTREE_HEADER_F32 + 6]!;
    expectInvalid(shared, /^Invalid PPG snapshot: dTree /);

    const trailing = clonePacked(canonical());
    const withTrailing = new Float32Array(trailing.dTreeBuf.length + DTREE_NODE_F32);
    withTrailing.set(trailing.dTreeBuf);
    trailing.dTreeBuf = withTrailing;
    expectInvalid(trailing, /trailing or missing data/);

    const badOffset = clonePacked(splitSpatialTree());
    badOffset.dTreeOffsets[1] = badOffset.dTreeOffsets[1]! + 1;
    expectInvalid(badOffset, /offset 1 is not contiguous/);

    const truncated = clonePacked(canonical());
    truncated.dTreeBuf = truncated.dTreeBuf.slice(0, -1);
    expectInvalid(truncated, /truncated|trailing or missing data/);
  });

  it('rejects sTree cycles/shared children, unreachable topology, and AABB partition drift', () => {
    const packed = splitSpatialTree();
    expect(() => validateSerialisedSTree(packed, OPTIONS)).not.toThrow();

    const cycle = clonePacked(packed);
    const interior = STREE_HEADER_F32 + STREE_NODE_F32;
    cycle.sTreeBuf[interior + 8] = 1;
    expectInvalid(cycle, /invalid children|cyclic|child/);

    const shared = clonePacked(packed);
    shared.sTreeBuf[STREE_HEADER_F32 + 9] = shared.sTreeBuf[STREE_HEADER_F32 + 8]!;
    expectInvalid(shared, /invalid children|multiply referenced/);

    const unreachable = clonePacked(packed);
    unreachable.sTreeBuf[interior + 7] = -1;
    unreachable.sTreeBuf[interior + 8] = -1;
    unreachable.sTreeBuf[interior + 9] = -1;
    unreachable.sTreeBuf[interior + 10] = 0;
    expectInvalid(unreachable, /leaf count|unreachable|permutation/);

    const partition = clonePacked(packed);
    const leftChild = Number(partition.sTreeBuf[STREE_HEADER_F32 + 8]);
    partition.sTreeBuf[STREE_HEADER_F32 + leftChild * STREE_NODE_F32 + 4] =
      partition.sTreeBuf[STREE_HEADER_F32 + leftChild * STREE_NODE_F32 + 4]! + 0.25;
    expectInvalid(partition, /child .* max axis/);
  });

  it('rejects a root trained for different bounds before deserialisation', () => {
    const packed = clonePacked(canonical());
    packed.sTreeBuf[STREE_HEADER_F32] = 0.25;
    expectInvalid(packed, /root min axis 0/);
  });

  it('rejects large-world child AABB drift that a relative epsilon would hide', () => {
    const bounds = {
      min: [1_000_000_000, 0, 0] as [number, number, number],
      max: [1_100_000_000, 1, 1] as [number, number, number],
    };
    const tree = buildSTree(bounds, 1);
    splitOverflowLeaves(tree, 0, 8, new Uint32Array([1]));
    const packed = serialiseSTree(tree, OPTIONS.maxDTreeNodesPerCell);
    const root = STREE_HEADER_F32;
    const leftChild = Number(packed.sTreeBuf[root + 8]);
    const leftMaxX =
      STREE_HEADER_F32 + leftChild * STREE_NODE_F32 + 4;
    packed.sTreeBuf[leftMaxX] =
      packed.sTreeBuf[leftMaxX]! + 8192;

    expect(() =>
      validateSerialisedSTree(packed, {
        ...OPTIONS,
        sceneBounds: bounds,
      }),
    ).toThrow(/child .* max axis 0/);
  });
});
