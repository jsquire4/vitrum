import { describe, expect, it } from 'vitest';
import { packBvhNodesForDebug } from '../../walkaround-hybrid/src/debug/packBvhNodesForDebug.js';
import { computeBvhStats } from '../src/vanilla/bvhStats.js';

const LEAF = 0xffff0001;

function makeInternalNodes(
  entries: ReadonlyArray<{
    readonly rightOrTriangleOffset: number;
    readonly splitOrCount: number;
  }>,
): ArrayBuffer {
  const buffer = new ArrayBuffer(entries.length * 32);
  const words = new Uint32Array(buffer);
  for (let i = 0; i < entries.length; i += 1) {
    const base = i * 8;
    words[base + 6] = entries[i]!.rightOrTriangleOffset >>> 0;
    words[base + 7] = entries[i]!.splitOrCount >>> 0;
  }
  return buffer;
}

describe('computeBvhStats', () => {
  it('integrates the internal producer with the public depth-layout consumer', () => {
    // Internal relative layout:
    // root 0: left=1, right=4; node 1: left=2, right=3.
    // The producer turns those links into public depth slots before the dev
    // consumer sees them.
    const publicNodes = packBvhNodesForDebug(makeInternalNodes([
      { rightOrTriangleOffset: 4, splitOrCount: 0 },
      { rightOrTriangleOffset: 2, splitOrCount: 1 },
      { rightOrTriangleOffset: 100, splitOrCount: LEAF },
      { rightOrTriangleOffset: 101, splitOrCount: LEAF },
      { rightOrTriangleOffset: 102, splitOrCount: LEAF },
    ]));

    expect(computeBvhStats(publicNodes)).toEqual({
      nodeCount: 5,
      maxDepth: 2,
      avgDepth: 1.2,
      histogram: [1, 2, 2],
    });
  });

  it('counts every tree in the producer’s concatenated BLAS forest', () => {
    const publicNodes = packBvhNodesForDebug(makeInternalNodes([
      { rightOrTriangleOffset: 77, splitOrCount: LEAF },
      { rightOrTriangleOffset: 2, splitOrCount: 0 },
      { rightOrTriangleOffset: 88, splitOrCount: LEAF },
      { rightOrTriangleOffset: 99, splitOrCount: LEAF },
    ]));

    expect(computeBvhStats(publicNodes)).toEqual({
      nodeCount: 4,
      maxDepth: 1,
      avgDepth: 0.5,
      histogram: [2, 2],
    });
  });

  it('reports an empty public table', () => {
    expect(computeBvhStats(new Float32Array(0))).toEqual({
      nodeCount: 0,
      maxDepth: 0,
      avgDepth: 0,
      histogram: [],
    });
  });

  it('fails closed on ragged tables or impossible public depths', () => {
    expect(() => computeBvhStats(new Float32Array(7))).toThrow(
      /exactly 8 floats per node/,
    );
    const invalidDepth = new Float32Array(8);
    invalidDepth[6] = Number.POSITIVE_INFINITY;
    expect(() => computeBvhStats(invalidDepth)).toThrow(/invalid public debug depth/);
  });
});
