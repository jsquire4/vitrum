import { describe, expect, it } from 'vitest';
import { computeBvhStats } from '../src/vanilla/bvhStats.js';

const LEAF = 0xffff0001;

function makeNodes(
  entries: ReadonlyArray<{ rightOrTriangleOffset: number; splitOrCount: number }>,
): Float32Array {
  const nodes = new Float32Array(entries.length * 8);
  const words = new Uint32Array(nodes.buffer);
  for (let i = 0; i < entries.length; i++) {
    const base = i * 8;
    words[base + 6] = entries[i]!.rightOrTriangleOffset >>> 0;
    words[base + 7] = entries[i]!.splitOrCount >>> 0;
  }
  return nodes;
}

describe('computeBvhStats', () => {
  it('derives depth by traversing child links instead of reading packed offsets', () => {
    // root: left=1, right=4; node 1: left=2, right=3.
    const nodes = makeNodes([
      { rightOrTriangleOffset: 4, splitOrCount: 0 },
      { rightOrTriangleOffset: 2, splitOrCount: 1 },
      { rightOrTriangleOffset: 100, splitOrCount: LEAF },
      { rightOrTriangleOffset: 101, splitOrCount: LEAF },
      { rightOrTriangleOffset: 102, splitOrCount: LEAF },
    ]);

    expect(computeBvhStats(nodes)).toEqual({
      nodeCount: 5,
      maxDepth: 2,
      avgDepth: 1.2,
      histogram: [1, 2, 2],
    });
  });

  it('reports a single leaf at depth zero regardless of its triangle offset', () => {
    const nodes = makeNodes([
      { rightOrTriangleOffset: 1234, splitOrCount: LEAF },
    ]);

    expect(computeBvhStats(nodes)).toEqual({
      nodeCount: 1,
      maxDepth: 0,
      avgDepth: 0,
      histogram: [1],
    });
  });

  it('does not loop forever on a malformed self-referencing interior node', () => {
    const nodes = makeNodes([
      { rightOrTriangleOffset: 0, splitOrCount: 0 },
    ]);

    expect(computeBvhStats(nodes)).toEqual({
      nodeCount: 1,
      maxDepth: 0,
      avgDepth: 0,
      histogram: [1],
    });
  });
});
