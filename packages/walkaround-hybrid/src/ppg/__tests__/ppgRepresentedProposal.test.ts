import { describe, expect, it } from 'vitest';
import { REPRESENTED_PROPOSAL_BUCKET_COUNT } from '@vitrum/shared-samplers';
import {
  buildEmptyDTree,
  dTreePdf,
  dTreeSample,
  recomputeDTreeInteriorFlux,
} from '../dTree.js';
import {
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  serialiseDTree,
} from '../serialise.js';
import { buildPpgRepresentedQueryView } from '../ppgRepresentedProposal.js';

function adversarialTree() {
  const tree = buildEmptyDTree(1);
  const weights = [2 ** -30, 2 ** -30, 1, 0];
  for (let child = 0; child < 4; child += 1) {
    tree.nodes[1 + child]!.flux = weights[child]!;
  }
  recomputeDTreeInteriorFlux(tree);
  return tree;
}

describe('PPG represented query proposal', () => {
  it('overlays exact subtree/leaf buckets without mutating snapshot bytes', () => {
    const raw = serialiseDTree(adversarialTree());
    const source = {
      sTreeBuf: new Float32Array([0, 0, 0, 0]),
      dTreeBuf: raw,
      dTreeOffsets: new Uint32Array([0]),
    };
    const view = buildPpgRepresentedQueryView(source);

    expect(source.dTreeBuf[DTREE_HEADER_F32 + 6]).toBe(1);
    for (let child = 0; child < 4; child += 1) {
      const base = DTREE_HEADER_F32 + (1 + child) * DTREE_NODE_F32;
      expect(source.dTreeBuf[base + 6]).toBe(-1);
    }

    const rootBase = DTREE_HEADER_F32;
    expect(view.dTreeBuf[rootBase + 5]).toBe(REPRESENTED_PROPOSAL_BUCKET_COUNT);
    expect(view.dTreeBuf[rootBase + 6]).toBe(1);
    const buckets = [1, 1, REPRESENTED_PROPOSAL_BUCKET_COUNT - 2, 0];
    for (let child = 0; child < 4; child += 1) {
      const base = DTREE_HEADER_F32 + (1 + child) * DTREE_NODE_F32;
      expect(view.dTreeBuf[base + 5]).toBeCloseTo(Math.PI, 6);
      expect(view.dTreeBuf[base + 6]).toBe(buckets[child]);
      expect(view.dTreeBuf[base + 7]).toBe(1);
    }
  });

  it('makes CPU sampling and PDF evaluation use the same bucket intervals', () => {
    const tree = adversarialTree();
    const invBuckets = 1 / REPRESENTED_PROPOSAL_BUCKET_COUNT;

    const representedPdf = (buckets: number): number => Math.fround(
      Math.fround(buckets * invBuckets) / Math.fround(Math.PI),
    );
    expect(dTreePdf(tree, [0.25, 0.25])).toBe(representedPdf(1));
    expect(dTreePdf(tree, [0.75, 0.25])).toBe(representedPdf(1));
    expect(dTreePdf(tree, [0.25, 0.75])).toBe(
      representedPdf(REPRESENTED_PROPOSAL_BUCKET_COUNT - 2),
    );
    // A zero-mass leaf in a live guide is not a uniform-guide region.
    expect(dTreePdf(tree, [0.75, 0.75])).toBe(0);

    expect(dTreeSample(tree, 0, 0.25).octUV[1]).toBeLessThan(0.5);
    const second = dTreeSample(tree, invBuckets, 0.75).octUV;
    expect(second[0]).toBeGreaterThanOrEqual(0.5);
    expect(second[1]).toBeLessThan(0.5);
    const third = dTreeSample(tree, 2 * invBuckets, 0.25).octUV;
    expect(third[0]).toBeLessThan(0.5);
    expect(third[1]).toBeGreaterThanOrEqual(0.5);
  });

  it('retains the uniform-sphere fallback only for a wholly cold tree', () => {
    const tree = buildEmptyDTree(1);
    expect(dTreePdf(tree, [0.75, 0.75])).toBe(
      Math.fround(1 / Math.fround(4 * Math.PI)),
    );
    expect(dTreeSample(tree, 0.25, 0.75)).toEqual({
      octUV: [0.25, 0.75],
      pdf: Math.fround(1 / Math.fround(4 * Math.PI)),
    });
  });

  it('sums exact represented counts through a deep tree and a clamp-promoted root', () => {
    const tree = buildEmptyDTree(2);
    let positiveLeaves = 0;
    for (let index = 0; index < tree.nodes.length; index += 1) {
      const node = tree.nodes[index]!;
      if (!node.isLeaf) continue;
      node.flux = index % 5 === 0 ? 0 : index % 2 === 0 ? 2 ** -30 : 1;
      if (node.flux > 0) positiveLeaves += 1;
    }
    recomputeDTreeInteriorFlux(tree);
    const raw = serialiseDTree(tree);
    const view = buildPpgRepresentedQueryView({
      sTreeBuf: new Float32Array(4),
      dTreeBuf: raw,
      dTreeOffsets: new Uint32Array([0]),
    });

    let representedPositiveLeaves = 0;
    for (let index = 0; index < tree.nodes.length; index += 1) {
      const base = DTREE_HEADER_F32 + index * DTREE_NODE_F32;
      const node = tree.nodes[index]!;
      if (node.isLeaf) {
        const buckets = view.dTreeBuf[base + 6]!;
        if (node.flux > 0) {
          expect(buckets).toBeGreaterThan(0);
          representedPositiveLeaves += 1;
        } else {
          expect(buckets).toBe(0);
        }
        continue;
      }
      let children = 0;
      for (let child = 0; child < 4; child += 1) {
        const childIndex = node.firstChild + child;
        const childBase = DTREE_HEADER_F32 + childIndex * DTREE_NODE_F32;
        children += view.dTreeBuf[
          childBase + (tree.nodes[childIndex]!.isLeaf ? 6 : 5)
        ]!;
      }
      expect(view.dTreeBuf[base + 5]).toBe(children);
    }
    expect(representedPositiveLeaves).toBe(positiveLeaves);
    expect(view.dTreeBuf[DTREE_HEADER_F32 + 5]).toBe(REPRESENTED_PROPOSAL_BUCKET_COUNT);

    const clamped = serialiseDTree(tree, 1);
    const clampedView = buildPpgRepresentedQueryView({
      sTreeBuf: new Float32Array(4),
      dTreeBuf: clamped,
      dTreeOffsets: new Uint32Array([0]),
    });
    expect(clampedView.dTreeBuf[DTREE_HEADER_F32 + 7]).toBe(1);
    expect(clampedView.dTreeBuf[DTREE_HEADER_F32 + 6]).toBe(
      REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
  });
});
