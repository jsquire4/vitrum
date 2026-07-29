import { describe, expect, it } from 'vitest';
import { validateBvhEncoding } from '../validateBvhEncoding.js';

const NODE_WORDS = 8;
const LEAF = 0xffff0000;

function threeNodeTree(): Float32Array {
  const nodes = new Float32Array(3 * NODE_WORDS);
  const words = new Uint32Array(nodes.buffer);

  // Root encloses two one-triangle leaves.
  nodes.set([0, 0, 0, 2, 1, 1], 0);
  words[6] = 2;
  words[7] = 0;

  nodes.set([0, 0, 0, 1, 1, 1], NODE_WORDS);
  words[NODE_WORDS + 6] = 0;
  words[NODE_WORDS + 7] = LEAF | 1;

  nodes.set([1, 0, 0, 2, 1, 1], NODE_WORDS * 2);
  words[NODE_WORDS * 2 + 6] = 1;
  words[NODE_WORDS * 2 + 7] = LEAF | 1;
  return nodes;
}

describe('validateBvhEncoding structural publication proof', () => {
  it('returns the exact structural and fixed-stack proof for a valid tree', () => {
    expect(validateBvhEncoding(threeNodeTree(), 3, {
      triangleCount: 2,
    })).toEqual({
      nodeCount: 3,
      rootCount: 1,
      interiorNodeCount: 1,
      leafNodeCount: 2,
      maxDepth: 1,
      maxTraversalStackEntries: 2,
      traversalStackCapacity: 60,
      triangleCount: 2,
    });
  });

  it('accepts a disjoint concatenated forest only when every root is declared', () => {
    const nodes = new Float32Array(2 * NODE_WORDS);
    const words = new Uint32Array(nodes.buffer);
    nodes.set([0, 0, 0, 1, 1, 1], 0);
    words[6] = 0;
    words[7] = LEAF | 1;
    nodes.set([2, 0, 0, 3, 1, 1], NODE_WORDS);
    words[NODE_WORDS + 6] = 1;
    words[NODE_WORDS + 7] = LEAF | 1;

    expect(validateBvhEncoding(nodes, 2, {
      roots: [0, 1],
      triangleCount: 2,
    }).rootCount).toBe(2);
    expect(() => validateBvhEncoding(nodes, 2, {
      roots: [0],
      triangleCount: 2,
    })).toThrow(/unreachable/);
  });

  it('rejects malformed extent, roots, topology, axes, and bounds', () => {
    expect(() => validateBvhEncoding(new Float32Array(7), 1)).toThrow(
      /expected exactly 8/,
    );

    expect(() => validateBvhEncoding(threeNodeTree(), 3, {
      roots: [0, 0],
    })).toThrow(/declared more than once/);

    const badOffset = threeNodeTree();
    new Uint32Array(badOffset.buffer)[6] = 1;
    expect(() => validateBvhEncoding(badOffset, 3)).toThrow(
      /invalid relative right-child offset/,
    );

    const badAxis = threeNodeTree();
    new Uint32Array(badAxis.buffer)[7] = 3;
    expect(() => validateBvhEncoding(badAxis, 3)).toThrow(/invalid split axis/);

    const nonFinite = threeNodeTree();
    nonFinite[NODE_WORDS] = Number.NaN;
    expect(() => validateBvhEncoding(nonFinite, 3)).toThrow(/non-finite bounds/);

    const inverted = threeNodeTree();
    inverted[NODE_WORDS] = 2;
    inverted[NODE_WORDS + 3] = 1;
    expect(() => validateBvhEncoding(inverted, 3)).toThrow(/inverted bounds/);

    const notEnclosed = threeNodeTree();
    notEnclosed[NODE_WORDS * 2 + 3] = 3;
    expect(() => validateBvhEncoding(notEnclosed, 3)).toThrow(
      /do not enclose child/,
    );
  });

  it('rejects duplicate reachability and insufficient shader stack capacity', () => {
    const sharedChild = new Float32Array(5 * NODE_WORDS);
    const words = new Uint32Array(sharedChild.buffer);
    for (let node = 0; node < 5; node += 1) {
      sharedChild.set([0, 0, 0, 1, 1, 1], node * NODE_WORDS);
    }
    // Root: left=1, right=4. Node 1: left=2, right=4, sharing node 4.
    words[6] = 4;
    words[7] = 0;
    words[NODE_WORDS + 6] = 3;
    words[NODE_WORDS + 7] = 0;
    words[NODE_WORDS * 2 + 7] = LEAF;
    words[NODE_WORDS * 3 + 7] = LEAF;
    words[NODE_WORDS * 4 + 7] = LEAF;
    expect(() => validateBvhEncoding(sharedChild, 5)).toThrow(
      /reachable more than once/,
    );

    expect(() => validateBvhEncoding(threeNodeTree(), 3, {
      traversalStackCapacity: 1,
    })).toThrow(/exceeding shader capacity 1/);
  });

  it('proves arbitrary child order for a deep-right/shallow-left tree', () => {
    const nodes = new Float32Array(9 * NODE_WORDS);
    const words = new Uint32Array(nodes.buffer);
    for (let node = 0; node < 9; node += 1) {
      nodes.set([0, 0, 0, 1, 1, 1], node * NODE_WORDS);
      words[node * NODE_WORDS + 7] = LEAF;
    }
    // Every interior has a shallow left leaf and a deep right subtree:
    // 0 -> (1,2), 2 -> (3,4), 4 -> (5,6), 6 -> (7,8).
    // A left-first DFS never observes more than two array entries, while a ray
    // that chooses the right child first retains four siblings and needs five.
    for (const interior of [0, 2, 4, 6]) {
      words[interior * NODE_WORDS + 6] = 2;
      words[interior * NODE_WORDS + 7] = 0;
    }

    const proof = validateBvhEncoding(nodes, 9, {
      triangleCount: 0,
      traversalStackCapacity: 5,
    });
    expect(proof.maxDepth).toBe(4);
    expect(proof.maxTraversalStackEntries).toBe(5);
    expect(() => validateBvhEncoding(nodes, 9, {
      triangleCount: 0,
      traversalStackCapacity: 4,
    })).toThrow(/requires 5 live stack entries/);
  });

  it('rejects leaf ranges that escape, overlap, or leave gaps', () => {
    const escaped = threeNodeTree();
    new Uint32Array(escaped.buffer)[NODE_WORDS * 2 + 6] = 2;
    expect(() => validateBvhEncoding(escaped, 3, {
      triangleCount: 2,
    })).toThrow(/exceeds triangle capacity/);

    const overlap = threeNodeTree();
    new Uint32Array(overlap.buffer)[NODE_WORDS * 2 + 6] = 0;
    expect(() => validateBvhEncoding(overlap, 3, {
      triangleCount: 2,
    })).toThrow(/overlaps/);

    const gap = threeNodeTree();
    new Uint32Array(gap.buffer)[NODE_WORDS + 6] = 1;
    expect(() => validateBvhEncoding(gap, 3, {
      triangleCount: 3,
    })).toThrow(/leaves a gap/);
  });
});
