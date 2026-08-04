import { describe, expect, it } from 'vitest';
import {
  buildEmptyDTree,
  dTreeAccumulateFlux,
  dTreeSample,
  findDTreeLeaf,
  refineDTree,
  recomputeDTreeInteriorFlux,
} from '../dTree.js';
import { serialiseDTree } from '../serialise.js';
import { PPG_UPDATE_WGSL } from '../ppgUpdate.wgsl.js';
import { PPG_PDF_WGSL } from '../ppgPdf.wgsl.js';
import { PPG_MIS_ALPHA } from '../ppgConstants.js';
import { RIS_GI_WGSL } from '../../shaders/risGi.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from '../../shaders/risGiGlassWalk.wgsl.js';
import {
  REPRESENTED_PROPOSAL_BUCKET_COUNT,
  createRepresentedWrsStateF32,
  representedWrsSelectedLogCorrection,
  updateRepresentedWrsF32,
} from '@vitrum/shared-samplers';

describe('PPG estimator hardening', () => {
  it('never selects a zero-mass leading CDF interval at the exact zero RNG endpoint', () => {
    const tree = buildEmptyDTree(1);
    tree.nodes[1]!.flux = 0;
    tree.nodes[2]!.flux = 10;
    tree.nodes[3]!.flux = 0;
    tree.nodes[4]!.flux = 0;
    tree.nodes[0]!.flux = 10;
    tree.totalFlux = 10;

    const sample = dTreeSample(tree, 0, 0.5);
    expect(findDTreeLeaf(tree, sample.octUV)).toBe(2);
    expect(sample.pdf).toBe(
      Math.fround(1 / Math.fround(tree.nodes[2]!.solidAngle)),
    );
  });

  it('recomputes newly-created interior subtree flux before the refined guide is sampled', () => {
    const tree = buildEmptyDTree(1);
    tree.nodes[1]!.flux = 100;
    tree.nodes[2]!.flux = 1;
    tree.nodes[3]!.flux = 1;
    tree.nodes[4]!.flux = 1;
    tree.nodes[0]!.flux = 103;
    tree.totalFlux = 103;

    refineDTree(tree, 0.01, 0, 4);

    const promoted = tree.nodes[1]!;
    expect(promoted.isLeaf).toBe(false);
    const childSum = [0, 1, 2, 3].reduce(
      (sum, offset) => sum + tree.nodes[promoted.firstChild + offset]!.flux,
      0,
    );
    expect(promoted.flux).toBeCloseTo(childSum, 12);
    expect(promoted.flux).toBeCloseTo(100, 12);
  });

  it('respects a per-cell node cap during refinement instead of uploading a truncated topology', () => {
    const tree = buildEmptyDTree(0);
    tree.nodes[0]!.flux = 10;
    tree.totalFlux = 10;

    refineDTree(tree, 0, 0, 4, 1);
    expect(tree.nodes).toHaveLength(1);
    expect(tree.nodes[0]!.isLeaf).toBe(true);
  });

  it('gives a clamp-promoted interior node a valid positive solid angle and PDF mass', () => {
    const tree = buildEmptyDTree(2);
    for (const node of tree.nodes) {
      if (node.isLeaf) node.flux = 1;
    }
    tree.totalFlux = 16;
    recomputeDTreeInteriorFlux(tree);

    const packed = serialiseDTree(tree, 1);
    const rootBase = 4;
    expect(packed[rootBase + 4]).toBe(16);
    expect(packed[rootBase + 5]).toBeCloseTo(4 * Math.PI, 6);
    expect(packed[rootBase + 6]).toBe(-1);
    expect(packed[rootBase + 7]).toBe(1);
  });

  it('rejects non-finite or fractional initial depths', () => {
    expect(() => buildEmptyDTree(Number.NaN)).toThrow(RangeError);
    expect(() => buildEmptyDTree(1.5)).toThrow(RangeError);
    expect(() => buildEmptyDTree(-1)).toThrow(RangeError);
  });

  it('uses represented exp2(H) as an unbiased multi-bin histogram oracle', () => {
    const proposal = [0.65, 0.2, 0.1, 0.05] as const;
    const targetMass = [0, 0.3, 0.8, 1.7] as const;
    const candidatesPerReservoir = 8;
    const trials = 200_000;
    const totals = new Float64Array(targetMass.length);
    let state = 0x91e10da5;
    const random = (): number => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x1_0000_0000;
    };
    const drawBin = (): number => {
      const r = random();
      let cdf = 0;
      for (let bin = 0; bin < proposal.length; bin++) {
        cdf += proposal[bin]!;
        if (r < cdf) return bin;
      }
      return proposal.length - 1;
    };

    for (let trial = 0; trial < trials; trial++) {
      let selected = -1;
      const wrs = createRepresentedWrsStateF32();
      for (let candidate = 0; candidate < candidatesPerReservoir; candidate++) {
        const bin = drawBin();
        const weight = targetMass[bin]! / proposal[bin]!;
        if (
          weight > 0 &&
          updateRepresentedWrsF32(wrs, Math.log2(weight), random)
        ) {
          selected = bin;
        }
      }
      if (selected >= 0) {
        const H =
          representedWrsSelectedLogCorrection(wrs) -
          Math.log2(candidatesPerReservoir);
        totals[selected] = totals[selected]! + 2 ** H;
      }
    }

    for (let bin = 0; bin < targetMass.length; bin++) {
      const estimate = totals[bin]! / trials;
      const tolerance = Math.max(0.01, targetMass[bin]! * 0.02);
      expect(Math.abs(estimate - targetMass[bin]!)).toBeLessThan(tolerance);
    }
  });

  it('proves the fixed 0.5 defensive mixture is normalized, unbiased, and support-robust', () => {
    // Four positive-cosine hemisphere bins plus one below-surface bin. The
    // learned guide deliberately has a blind spot in bin 0 and half its mass
    // outside the receiver hemisphere; the cosine proposal restores target
    // support with the fixed alpha contract.
    const cosinePmf = [0.55, 0.3, 0.1, 0.05, 0] as const;
    const guidePmf = [0, 0.1, 0.15, 0.25, 0.5] as const;
    const targetMass = [1.7, 0.4, 2.1, 0.2, 0] as const;
    const alpha = PPG_MIS_ALPHA;
    const mixture = cosinePmf.map(
      (pCos, bin) => alpha * guidePmf[bin]! + (1 - alpha) * pCos,
    );

    expect(alpha).toBe(0.5);
    expect(cosinePmf.reduce<number>((sum, p) => sum + p, 0)).toBeCloseTo(1, 15);
    expect(guidePmf.reduce<number>((sum, p) => sum + p, 0)).toBeCloseTo(1, 15);
    expect(mixture.reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 15);
    expect(guidePmf[0]).toBe(0);
    expect(mixture[0]).toBeGreaterThan(0);

    const exactImportanceExpectation = mixture.reduce(
      (sum, pSrc, bin) => sum + (pSrc > 0 ? pSrc * targetMass[bin]! / pSrc : 0),
      0,
    );
    expect(exactImportanceExpectation).toBeCloseTo(
      targetMass.reduce<number>((sum, mass) => sum + mass, 0),
      15,
    );

    expect(RIS_GI_WGSL).toContain('if (bern < alpha)');
    expect(RIS_GI_WGSL).toContain(
      'logPSrc = reservoirGiLogProposalMixture(alpha, pGuide, pCos);',
    );
    expect(RIS_GI_WGSL).toContain('if (!reservoirGiValidLog(logPSrc))');
    expect(RIS_GI_WGSL).toContain('let logWeight = logPHat - logPSrc;');
    expect(PPG_PDF_WGSL).toContain('!ppgDTreeRootValidGi(dOff)');
    expect(PPG_PDF_WGSL).toContain(
      'if (distributionBuckets == 0u) { return 0.0; }',
    );
    expect(PPG_PDF_WGSL).toContain(
      'return (f32(distributionBuckets) * PPG_INV_REPRESENTED_BUCKETS) / solidAng;',
    );
    expect(PPG_PDF_WGSL).toContain('var remaining = pcgNext(rng) >> 8u;');
    expect(PPG_PDF_WGSL).not.toContain('max(solidAng, 1e-12)');
  });

  it('uses one local-patch fallback contract for partial dTree corruption', () => {
    expect(PPG_PDF_WGSL).toContain(
      'return childBucketSum == ppgDTreeNodeBucketCount(base);',
    );
    expect(PPG_PDF_WGSL).toContain(
      'if (!ppgDTreeChildrenValidGi(dTreeOffset, nodeCount, idx, base))',
    );
    expect(PPG_PDF_WGSL).toContain(
      'let distributionBase = ppgDTreeFindDistributionBase(dOff, octUV);',
    );
    expect(PPG_PDF_WGSL).toContain(
      'let solidAng = ppgDTreePatchSolidAngle(distributionBase);',
    );
    expect(PPG_PDF_WGSL).toContain('ppgQueryArena_gi[15] != arenaWords');
  });

  it('preserves support and publishes the actual represented PDF of an extremely small positive-flux leaf', () => {
    const tree = buildEmptyDTree(1);
    const tinyFlux = 1e-20;
    tree.nodes[1]!.flux = tinyFlux;
    tree.nodes[2]!.flux = 1;
    tree.nodes[3]!.flux = 1;
    tree.nodes[4]!.flux = 1;
    tree.nodes[0]!.flux = 3 + tinyFlux;
    tree.totalFlux = 3 + tinyFlux;

    const sample = dTreeSample(tree, 0, 0.5);
    const expected = Math.fround(
      Math.fround(1 / REPRESENTED_PROPOSAL_BUCKET_COUNT) /
        Math.fround(tree.nodes[1]!.solidAngle),
    );
    expect(findDTreeLeaf(tree, sample.octUV)).toBe(1);
    expect(sample.pdf).toBe(expected);
    expect(sample.pdf).toBeGreaterThan(0);
    expect(sample.pdf).toBeLessThan(1e-7);
  });

  it('keeps formerly overflowing glass-candidate weights selectable in log WRS', () => {
    const maxF32 = 3.402823466e38;
    const retainedWeight = maxF32 * 0.75;
    const nextWeight = maxF32 * 0.5;
    expect(Number.isFinite(Math.fround(retainedWeight))).toBe(true);
    expect(Number.isFinite(Math.fround(nextWeight))).toBe(true);
    expect(nextWeight > maxF32 - retainedWeight).toBe(true);
    const wrs = createRepresentedWrsStateF32();
    let selected = -1;
    if (updateRepresentedWrsF32(wrs, Math.log2(retainedWeight), () => 0)) {
      selected = 0;
    }
    if (updateRepresentedWrsF32(wrs, Math.log2(nextWeight), () => 0)) {
      selected = 1;
    }
    const H = representedWrsSelectedLogCorrection(wrs) - Math.log2(2);
    expect(selected).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(H)).toBe(true);
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let logWeight = logPHat - logPSrc;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('var wrs = representedWrsInit();');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'finaliseGIReservoirFromNativeWrs(',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('&reservoir,');
    expect(NATIVE_GLASS_GI_WGSL).toContain('&wrs,');
    expect(NATIVE_GLASS_GI_WGSL).not.toContain('w_sum');
  });

  it('pins correctly-rounded exp2(H) endpoint training and lock-free f32 CAS in WGSL', () => {
    expect(PPG_UPDATE_WGSL).toContain('let reservoirH = bitcast<f32>(ppgReservoirGiCurrent[b + 11u]);');
    expect(PPG_UPDATE_WGSL).toContain('PPG_LOG2_ROUND_TO_ZERO: f32 = -150.0');
    expect(PPG_UPDATE_WGSL).toContain('PPG_LOG2_OVERFLOW: f32 = 128.0');
    expect(PPG_UPDATE_WGSL).toContain('reservoirH <= PPG_LOG2_ROUND_TO_ZERO');
    expect(PPG_UPDATE_WGSL).toContain('reservoirH < PPG_LOG2_OVERFLOW');
    expect(PPG_UPDATE_WGSL).toContain('trainingMass = min(exp2(reservoirH), MAX_FINITE_F32);');
    expect(PPG_UPDATE_WGSL).not.toContain('exp2(clamp(');
    expect(PPG_UPDATE_WGSL).toContain('if (!(trainingMass > 0.0) || trainingMass > MAX_FINITE_F32) { return; }');
    expect(PPG_UPDATE_WGSL).toContain('if (!(value > 0.0) || value > MAX_FINITE_F32) { return; }');
    expect(PPG_UPDATE_WGSL).toContain('atomicCompareExchangeWeak');
    expect(PPG_UPDATE_WGSL).toContain('if (oldValue == MAX_FINITE_F32) { return; }');
    expect(PPG_UPDATE_WGSL).toMatch(/loop\s*\{/);
    expect(PPG_UPDATE_WGSL).not.toContain('MAX_FLUX_CAS_ATTEMPTS');
    expect(PPG_UPDATE_WGSL).not.toContain('luminance(');
    expect(PPG_UPDATE_WGSL).not.toContain('reservoirWSum');
    expect(PPG_UPDATE_WGSL).not.toContain('/ f32(reservoirM)');
  });

  it('preserves subnormal and high finite PPG training mass without flooring or truncation', () => {
    const maxF32 = Math.fround(3.402823466e38);
    const endpoint = (H: number): number => {
      if (!Number.isFinite(H) || H <= -150) return 0;
      if (H >= 128) return maxF32;
      return Math.min(Math.fround(2 ** H), maxF32);
    };

    expect(endpoint(-200)).toBe(0);
    expect(endpoint(-150)).toBe(0);
    expect(endpoint(-149.5)).toBe(2 ** -149);
    expect(endpoint(-100)).toBe(Math.fround(2 ** -100));
    expect(endpoint(127.5)).toBe(Math.fround(2 ** 127.5));
    expect(endpoint(127.5)).toBeLessThan(maxF32);
    expect(endpoint(200)).toBe(maxF32);
  });

  it('trains from the persisted robust reconnection direction instead of subtracting endpoints', () => {
    expect(PPG_UPDATE_WGSL).toContain('ppgReservoirGiCurrent[b + 20u]');
    expect(PPG_UPDATE_WGSL).toContain('ppgReservoirGiCurrent[b + 21u]');
    expect(PPG_UPDATE_WGSL).toContain('ppgReservoirGiCurrent[b + 22u]');
    expect(PPG_UPDATE_WGSL).not.toContain('let samplePoint = vec3f(');
    expect(PPG_UPDATE_WGSL).not.toContain('samplePoint - pos');
  });

  it('updates every directional ancestor automatically and rejects invalid deposits transactionally', () => {
    const tree = buildEmptyDTree(2);
    dTreeAccumulateFlux(tree, [0.1, 0.1], 7);
    const firstLeaf = findDTreeLeaf(tree, [0.1, 0.1]);
    expect(tree.nodes[firstLeaf]!.flux).toBe(7);
    expect(tree.nodes[1]!.flux).toBe(7);
    expect(tree.nodes[0]!.flux).toBe(7);
    expect(tree.totalFlux).toBe(7);

    dTreeAccumulateFlux(tree, [0.9, 0.9], 3);
    const secondLeaf = findDTreeLeaf(tree, [0.9, 0.9]);
    expect(tree.nodes[secondLeaf]!.flux).toBe(3);
    expect(tree.nodes[4]!.flux).toBe(3);
    expect(tree.nodes[0]!.flux).toBe(10);
    expect(tree.totalFlux).toBe(10);

    const before = tree.nodes.map((node) => node.flux);
    expect(() => dTreeAccumulateFlux(tree, [Number.NaN, 0.5], 1)).toThrow(/octUV/);
    expect(() => dTreeAccumulateFlux(tree, [0.5, 0.5], Number.POSITIVE_INFINITY)).toThrow(/flux/);
    expect(tree.nodes.map((node) => node.flux)).toEqual(before);
    expect(tree.totalFlux).toBe(10);
  });

  it('stages interior recomputation so malformed topology cannot partially publish mass', () => {
    const tree = buildEmptyDTree(1);
    tree.nodes[0]!.flux = 999;
    tree.nodes[1]!.flux = 1;
    tree.nodes[2]!.flux = 2;
    tree.nodes[3]!.flux = 3;
    tree.nodes[4]!.flux = 4;
    tree.totalFlux = 777;
    tree.nodes.push({ ...tree.nodes[4]!, flux: 5 });
    const before = tree.nodes.map((node) => node.flux);

    expect(() => recomputeDTreeInteriorFlux(tree)).toThrow(/unreachable/);
    expect(tree.nodes.map((node) => node.flux)).toEqual(before);
    expect(tree.totalFlux).toBe(777);
  });
});
