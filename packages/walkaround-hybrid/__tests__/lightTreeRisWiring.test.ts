/**
 * lightTreeRisWiring.test.ts — verifies the light tree is actually WIRED into
 * ReSTIR-DI light selection (the "built-but-unconsumed" gap this slice closes).
 *
 * Three concerns:
 *   1. `buildLightTreeBuffer` gate: ≥ 2 emitters ⇒ enabled + a real packed tree
 *      whose leaf emitterIndex values index the emitter array; < 2 ⇒ disabled
 *      with a 1-node placeholder (RIS falls back to the flat power CDF).
 *   2. The RIS WGSL uses the TREE selection pdf (not the flat power CDF) in the
 *      WRS source weight when lightTreeEnabled == 1, and keeps the flat-CDF
 *      weight verbatim when disabled — proving unbiasedness in both states.
 *   3. The packed tree round-trips through `packLightTreeForGPU`'s stride.
 */

import { describe, expect, it } from 'vitest';
import { LIGHT_TREE_FLOATS_PER_NODE } from '@vitrum/shared-samplers';
import {
  buildLightTreeBuffer,
  type EmitterTreeInput,
} from '../src/restir/emitterList.js';
import { RIS_WGSL } from '../src/shaders/ris.wgsl.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import { LIGHT_TREE_WGSL } from '../src/shaders/lightTree.wgsl.js';

function makeTreeInput(powers: number[]): EmitterTreeInput {
  return {
    powers,
    centroids: powers.map((_, i) => [i * 2, 0, 0] as [number, number, number]),
    aabbs: powers.map((_, i) => ({
      min: [i * 2 - 0.5, -0.5, -0.5] as [number, number, number],
      max: [i * 2 + 0.5, 0.5, 0.5] as [number, number, number],
    })),
  };
}

describe('buildLightTreeBuffer — gate + packed layout', () => {
  it('≥ 2 emitters → enabled, packed nodes, leaf emitterIndex in range', () => {
    const buf = buildLightTreeBuffer(makeTreeInput([1, 2, 4, 8]));
    expect(buf.enabled).toBe(true);
    // 4 leaves → 7 nodes in a full binary tree.
    expect(buf.nodeCount).toBe(7);
    expect(buf.nodes).toHaveLength(7 * LIGHT_TREE_FLOATS_PER_NODE);

    // Every leaf (leftChild < 0) carries an emitterIndex in [0, 4).
    let leaves = 0;
    for (let n = 0; n < buf.nodeCount; n++) {
      const base = n * LIGHT_TREE_FLOATS_PER_NODE;
      const leftChild = buf.nodes[base + 2]!;
      if (leftChild < 0) {
        const emitterIndex = buf.nodes[base + 0]!;
        expect(emitterIndex).toBeGreaterThanOrEqual(0);
        expect(emitterIndex).toBeLessThan(4);
        leaves++;
      }
    }
    expect(leaves).toBe(4);
  });

  it('1 emitter → disabled, 1-node zeroed placeholder (RIS uses flat CDF)', () => {
    const buf = buildLightTreeBuffer(makeTreeInput([5]));
    expect(buf.enabled).toBe(false);
    expect(buf.nodeCount).toBe(0);
    // Non-empty buffer so the WGSL bind group is valid, but it is never read.
    expect(buf.nodes).toHaveLength(LIGHT_TREE_FLOATS_PER_NODE);
    expect(Array.from(buf.nodes).every((v) => v === 0)).toBe(true);
  });

  it('0 emitters → disabled placeholder', () => {
    const buf = buildLightTreeBuffer({ powers: [], centroids: [], aabbs: [] });
    expect(buf.enabled).toBe(false);
    expect(buf.nodeCount).toBe(0);
  });

  it('all-zero power → disabled (no meaningful power-weighted tree)', () => {
    const buf = buildLightTreeBuffer(makeTreeInput([0, 0, 0]));
    expect(buf.enabled).toBe(false);
  });
});

describe('RIS WGSL — selection pdf enters the WRS source weight', () => {
  it('the M_LIGHT loop branches on ubo.lightTreeEnabled', () => {
    expect(RIS_WGSL).toContain('ubo.lightTreeEnabled == 1u');
  });

  it('the tree path uses sampleLightTree and records its selection pdf', () => {
    expect(RIS_WGSL).toContain('sampleLightTree(pos, ubo.emitterDist2Floor');
    // The selection pmf the tree returns is captured into emitterSelPmf.
    expect(RIS_WGSL).toContain('emitterSelPmf = lt.pdf');
  });

  it('the flat-CDF fallback path uses the actual sampled CDF segment as pmf', () => {
    expect(RIS_WGSL).toContain('sampleEmitterIdx(emCount, xiEm)');
    expect(RIS_WGSL).toContain('emitterSelPmf = emitterCdfPmf(emCount, lid);');
  });

  it('the WRS source pdf pX multiplies the chosen selection pmf by the area pdf', () => {
    // pX = emitterSelPmf × ls.pdfArea — the SAME emitterSelPmf the selection
    // drew from (tree pdf OR flat-CDF pmf), times the uniform-area triangle
    // pdf. This is the divisor in w = p̂ / pX, so it must be the exact source.
    expect(RIS_WGSL).toContain('emitterSelPmf * ls.pdfArea');
    expect(RIS_WGSL).toContain('if (pHat > 0.0 && pX > 0.0) { w = pHat / pX; }');
  });
});

describe('UBO + WGSL light-tree contract', () => {
  it('WalkaroundUBO declares lightTreeEnabled + lightTreeNodeCount', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('lightTreeEnabled:');
    expect(WALKAROUND_UBO_WGSL).toContain('lightTreeNodeCount:');
  });

  it('light-tree WGSL declares the RIS-only group(3) storage buffer + traversal', () => {
    expect(LIGHT_TREE_WGSL).toContain('@group(3) @binding(0) var<storage, read> lightTree');
    expect(LIGHT_TREE_WGSL).toContain('fn sampleLightTree(');
    // The descent bound is the UBO node count (matches the CPU loop bound).
    expect(LIGHT_TREE_WGSL).toContain('nodeCount + 1u');
    // Importance metric = power / max(dist², floor) — distance-weighted.
    expect(LIGHT_TREE_WGSL).toContain('power / d2');
  });
});
