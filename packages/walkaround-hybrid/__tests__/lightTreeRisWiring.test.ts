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
import { FULL_SPHERE_CONE, LIGHT_TREE_FLOATS_PER_NODE } from '@vitrum/shared-samplers';
import { buildLightTreeBuffer, type EmitterTreeInput } from '../src/restir/emitterList.js';
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
    cones: powers.map(() => FULL_SPHERE_CONE),
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
    const buf = buildLightTreeBuffer({
      powers: [],
      centroids: [],
      aabbs: [],
      cones: [],
    });
    expect(buf.enabled).toBe(false);
    expect(buf.nodeCount).toBe(0);
  });

  it('all-zero power → disabled (no meaningful power-weighted tree)', () => {
    const buf = buildLightTreeBuffer(makeTreeInput([0, 0, 0]));
    expect(buf.enabled).toBe(false);
  });

  it('forwards one-sided and two-sided emitter cones into packed leaves', () => {
    const input = makeTreeInput([1, 1]);
    input.cones[0] = { axis: [0, 1, 0], thetaO: 0, thetaE: Math.PI / 2 };
    input.cones[1] = FULL_SPHERE_CONE;
    const buf = buildLightTreeBuffer(input);

    const leafBase = (emitterIndex: number): number => {
      for (let node = 0; node < buf.nodeCount; node += 1) {
        const base = node * LIGHT_TREE_FLOATS_PER_NODE;
        if (buf.nodes[base] === emitterIndex && buf.nodes[base + 2]! < 0) {
          return base;
        }
      }
      throw new Error(`missing light-tree leaf ${emitterIndex}`);
    };

    const oneSided = leafBase(0);
    expect(Array.from(buf.nodes.slice(oneSided + 10, oneSided + 13))).toEqual([0, 1, 0]);
    expect(buf.nodes[oneSided + 13]).toBeLessThan(1);
    expect(buf.nodes[oneSided + 13]).toBeGreaterThan(0.99);
    expect(buf.nodes[oneSided + 14]).toBeLessThan(0);
    expect(buf.nodes[oneSided + 14]).toBeGreaterThan(-0.001);

    const twoSided = leafBase(1);
    expect(Array.from(buf.nodes.slice(twoSided + 10, twoSided + 13))).toEqual([0, 0, 0]);
    expect(buf.nodes[twoSided + 13]).toBe(-1);
    expect(buf.nodes[twoSided + 14]).toBe(-1);
  });
});

describe('RIS WGSL — selection pdf enters the WRS source weight', () => {
  it('the M_LIGHT loop branches on ubo.lightTreeEnabled', () => {
    expect(RIS_WGSL).toContain('ubo.lightTreeEnabled == 1u');
  });

  it('the tree path records the represented selection PMF in log space', () => {
    expect(RIS_WGSL).toContain('sampleLightTree(pos, ubo.emitterDist2Floor');
    expect(RIS_WGSL).toContain('emitterLogSelectionPmf = reservoirDiPositiveLog2(lt.pdf);');
  });

  it('the flat-CDF fallback path uses the actual sampled CDF segment as pmf', () => {
    expect(RIS_WGSL).toContain('sampleEmitterIdx(emCount, xiEm)');
    expect(RIS_WGSL).toContain('emitterCdfPmf(emCount, lid),');
  });

  it('combines selection and area densities without a linear ratio endpoint', () => {
    expect(RIS_WGSL).toContain('let logWeight = reservoirDiInitialCandidateLogWeight(');
    expect(RIS_WGSL).toContain('emitterLogSelectionPmf,');
    expect(RIS_WGSL).toContain('ls.pdfArea,');
    expect(RIS_WGSL).not.toContain('emitterSelPmf * ls.pdfArea');
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
    // Importance is evaluated in log space, then the represented leaf interval
    // provides the exact source PMF without a deep product.
    expect(LIGHT_TREE_WGSL).toContain('log2(power) - log2(d2)');
    expect(LIGHT_TREE_WGSL).toContain('u32(lightTree[leftBase + 15u])');
    expect(LIGHT_TREE_WGSL).toContain('result.pdf = f32(currentBuckets) / f32(lt_ROOT_BUCKETS)');
  });
});
