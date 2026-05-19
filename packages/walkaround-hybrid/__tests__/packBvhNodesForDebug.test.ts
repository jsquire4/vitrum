import { describe, it, expect } from 'vitest';
import { packBvhNodesForDebug } from '../src/debug/packBvhNodesForDebug.js';

const LEAFNODE_FLAG_HIGH16 = 0xffff;

/**
 * Build a packed 8-u32 / 32-byte BVH node buffer from a list of plain
 * objects. Each entry is either an interior (rightChild + splitAxis) or
 * a leaf (triOffset + triCount, with the leaf flag stamped into the high
 * 16 bits of split).
 */
function buildBvhBuffer(
  nodes: ReadonlyArray<
    | { kind: 'interior'; min: [number, number, number]; max: [number, number, number]; rightChild: number; splitAxis: 0 | 1 | 2 }
    | { kind: 'leaf';     min: [number, number, number]; max: [number, number, number]; triOffset: number;   triCount: number }
  >,
): ArrayBuffer {
  const buf = new ArrayBuffer(nodes.length * 32);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i]!;
    f32[i * 8 + 0] = n.min[0];
    f32[i * 8 + 1] = n.min[1];
    f32[i * 8 + 2] = n.min[2];
    f32[i * 8 + 3] = n.max[0];
    f32[i * 8 + 4] = n.max[1];
    f32[i * 8 + 5] = n.max[2];
    if (n.kind === 'interior') {
      u32[i * 8 + 6] = n.rightChild;
      u32[i * 8 + 7] = n.splitAxis;
    } else {
      u32[i * 8 + 6] = n.triOffset;
      // Leaf split word = (0xFFFF << 16) | triCount
      u32[i * 8 + 7] = (LEAFNODE_FLAG_HIGH16 << 16) | (n.triCount & 0xffff);
    }
  }
  return buf;
}

describe('packBvhNodesForDebug', () => {
  it('returns empty Float32Array for an empty buffer', () => {
    const out = packBvhNodesForDebug(new ArrayBuffer(0));
    expect(out.length).toBe(0);
  });

  it('single-node leaf tree: root depth=0, bounds copied verbatim', () => {
    const buf = buildBvhBuffer([
      { kind: 'leaf', min: [-1, -2, -3], max: [4, 5, 6], triOffset: 0, triCount: 2 },
    ]);
    const out = packBvhNodesForDebug(buf);
    expect(out.length).toBe(8);
    expect(out[0]).toBe(-1); expect(out[1]).toBe(-2); expect(out[2]).toBe(-3);
    expect(out[3]).toBe(4);  expect(out[4]).toBe(5);  expect(out[5]).toBe(6);
    expect(out[6]).toBe(0); // depth
    expect(out[7]).toBe(0); // pad
  });

  it('two-leaf tree: root interior (depth=0), both leaves depth=1, and the leaves do NOT get walked past', () => {
    // Nodes: 0=root interior (right=2, split=0), 1=left leaf, 2=right leaf.
    // CRITICAL: with the prior `& 0xFFFF0000 === 0xFFFF0000` int32 bug,
    // the leaf-flag check always returned false → the depth-pass treated
    // both leaves as interior and walked into their `triOffset` field as
    // if it were a node index, corrupting the depth output.
    const buf = buildBvhBuffer([
      { kind: 'interior', min: [0, 0, 0], max: [10, 1, 1], rightChild: 2, splitAxis: 0 },
      { kind: 'leaf',     min: [0, 0, 0], max: [1,  1, 1], triOffset: 99, triCount: 4 },
      { kind: 'leaf',     min: [9, 0, 0], max: [10, 1, 1], triOffset: 77, triCount: 4 },
    ]);
    const out = packBvhNodesForDebug(buf);
    expect(out.length).toBe(24);
    expect(out[0 * 8 + 6]).toBe(0); // root depth
    expect(out[1 * 8 + 6]).toBe(1); // left leaf depth
    expect(out[2 * 8 + 6]).toBe(1); // right leaf depth
  });

  it('deep tree: left-spine yields monotonically increasing depths', () => {
    // 5-node tree:
    //   0 = interior(right=4, split=0)
    //     1 = interior(right=3, split=1)
    //       2 = leaf
    //       3 = leaf
    //     4 = leaf
    const buf = buildBvhBuffer([
      { kind: 'interior', min: [0, 0, 0], max: [4, 1, 1], rightChild: 4, splitAxis: 0 },
      { kind: 'interior', min: [0, 0, 0], max: [2, 1, 1], rightChild: 3, splitAxis: 1 },
      { kind: 'leaf',     min: [0, 0, 0], max: [1, 1, 1], triOffset: 0, triCount: 2 },
      { kind: 'leaf',     min: [1, 0, 0], max: [2, 1, 1], triOffset: 2, triCount: 2 },
      { kind: 'leaf',     min: [3, 0, 0], max: [4, 1, 1], triOffset: 4, triCount: 2 },
    ]);
    const out = packBvhNodesForDebug(buf);
    const depthsExpect = [0, 1, 2, 2, 1];
    for (let i = 0; i < 5; i++) {
      expect(out[i * 8 + 6]).toBe(depthsExpect[i]);
    }
  });

  it('leaf-flag check tolerates a triCount that collides with the int32 sign bit', () => {
    // Build a leaf whose split word, interpreted as int32, is negative
    // (0xFFFF0001 → -65535). The `>>> 16 === 0xFFFF` check handles this
    // cleanly; the `& 0xFFFF0000 === 0xFFFF0000` check did not.
    const buf = buildBvhBuffer([
      { kind: 'leaf', min: [0, 0, 0], max: [1, 1, 1], triOffset: 0, triCount: 1 },
    ]);
    const u32 = new Uint32Array(buf);
    // Sanity: confirm the split word's int32-projection is negative.
    expect(u32[7]! | 0).toBeLessThan(0);
    const out = packBvhNodesForDebug(buf);
    // Single-leaf root must have depth 0; with the bug, the traversal
    // would have descended into rightChild=triOffset=0 in a loop and
    // either thrashed sp or overwritten depths[0] inconsistently.
    expect(out[0 * 8 + 6]).toBe(0);
  });

  it('does not infinite-loop when a leaf has triOffset pointing back into the tree', () => {
    // Leaf at idx 1 has triOffset=0 (a valid node index). With the
    // leaf-flag bug, the depth-pass would have walked into idx 0 again,
    // re-pushed children, and looped — eventually exhausting the
    // pre-sized stack (nodeCount = 2). With the fix, the leaf is
    // detected immediately and traversal terminates.
    const buf = buildBvhBuffer([
      { kind: 'interior', min: [0, 0, 0], max: [2, 1, 1], rightChild: 1, splitAxis: 0 },
      { kind: 'leaf',     min: [1, 0, 0], max: [2, 1, 1], triOffset: 0, triCount: 2 },
    ]);
    // Just calling this without it throwing or hanging is the test.
    const out = packBvhNodesForDebug(buf);
    expect(out[0 * 8 + 6]).toBe(0);
    expect(out[1 * 8 + 6]).toBe(1);
  });
});
