import { describe, expect, it } from 'vitest';
import { buildSTree } from '../src/ppg/sTree.js';
import { resetAccumulators } from '../src/ppg/sTree.js';
import { dTreeSample } from '../src/ppg/dTree.js';

describe('PPG training utilities', () => {
  const aabb = { min: [0, 0, 0] as [number, number, number], max: [1, 1, 1] as [number, number, number] };

  it('dTreeSample returns octUV and pdf for a built tree', () => {
    const sTree = buildSTree(aabb);
    const dTree = sTree.dTrees[0]!;
    const { octUV, pdf } = dTreeSample(dTree, 0.25, 0.75);
    expect(octUV[0]).toBeGreaterThanOrEqual(0);
    expect(octUV[1]).toBeGreaterThanOrEqual(0);
    expect(pdf).toBeGreaterThan(0);
  });

  it('resetAccumulators clears leaf flux', () => {
    const sTree = buildSTree(aabb);
    const dTree = sTree.dTrees[0]!;
    dTree.nodes[0]!.flux = 10;
    dTree.totalFlux = 10;
    resetAccumulators(sTree);
    expect(dTree.totalFlux).toBe(0);
    expect(dTree.nodes[0]!.flux).toBe(0);
  });
});
