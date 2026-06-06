import { describe, expect, it } from 'vitest';
import { buildSTree } from '../src/ppg/sTree.js';
import { resetAccumulators } from '../src/ppg/sTree.js';
import { dTreeSample, buildEmptyDTree } from '../src/ppg/dTree.js';

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

  // ── Leaf-v decorrelation (Task 0.1 correctness fix) ──────────────────────
  // The GPU production sampler (ppgPdf.wgsl `ppgDTreeSampleLeafBase` ->
  // `ppgSampleGuidedDir`) draws fresh randoms for the leaf u,v jitter after
  // the flux-proportional descent, so the leaf jitter is independent of which
  // leaf was picked. The CPU oracle previously REUSED the descent-consumed
  // `u0` for `vSample`, correlating the v-position with the descent path. The
  // fix carries the rescaled descent residual into vSample instead. These
  // tests pin the NEW decorrelated behavior.
  it('dTreeSample leaf-v is decorrelated from the raw descent input u0', () => {
    // Depth-1 dTree: root (interior) + 4 leaves NW,NE,SW,SE.
    //   NW: u∈[0,0.5] v∈[0,0.5]   NE: u∈[0.5,1] v∈[0,0.5]
    //   SW: u∈[0,0.5] v∈[0.5,1]   SE: u∈[0.5,1] v∈[0.5,1]
    // Put flux on NW (=4) and NE (=6); totalFlux = 10. Descent order is
    // NW(1),NE(2),SW(3),SE(4): u0∈(0.4,1.0] all descend into the SAME leaf NE.
    const dTree = buildEmptyDTree(1);
    expect(dTree.nodes).toHaveLength(5); // root + 4 leaves
    const NW = dTree.nodes[1]!;
    const NE = dTree.nodes[2]!;
    NW.flux = 4;
    NE.flux = 6;
    dTree.totalFlux = 10;

    // Two distinct u0 values that both descend into NE (u0 > 0.4).
    const a = dTreeSample(dTree, 0.5, 0.3);
    const b = dTreeSample(dTree, 0.9, 0.3);

    // Both land in NE's rectangle [0.5,1] × [0,0.5].
    for (const s of [a, b]) {
      expect(s.octUV[0]).toBeGreaterThanOrEqual(NE.u0);
      expect(s.octUV[0]).toBeLessThanOrEqual(NE.u1);
      expect(s.octUV[1]).toBeGreaterThanOrEqual(NE.v0);
      expect(s.octUV[1]).toBeLessThanOrEqual(NE.v1);
    }

    // DECORRELATION: with the OLD bug, vSample = v0 + u0·(v1−v0) — a FIXED
    // linear function of the raw u0 (slope = v1−v0 = 0.5). Assert the v-coords
    // are NOT that linear function of u0 anymore.
    const oldVa = NE.v0 + 0.5 * (NE.v1 - NE.v0); // 0.25 — old correlated value
    const oldVb = NE.v0 + 0.9 * (NE.v1 - NE.v0); // 0.45 — old correlated value
    expect(a.octUV[1]).not.toBeCloseTo(oldVa, 6);
    expect(b.octUV[1]).not.toBeCloseTo(oldVb, 6);

    // The NEW v-coords come from the rescaled descent residual:
    //   residual = u0·total − (cumFluxBeforeNE) = u0·10 − 4;  uLeaf = residual/6
    //   vSample  = v0 + uLeaf·(v1−v0)
    const uLeafA = (0.5 * 10 - 4) / 6; // 1/6
    const uLeafB = (0.9 * 10 - 4) / 6; // 5/6
    expect(a.octUV[1]).toBeCloseTo(NE.v0 + uLeafA * (NE.v1 - NE.v0), 6);
    expect(b.octUV[1]).toBeCloseTo(NE.v0 + uLeafB * (NE.v1 - NE.v0), 6);

    // uSample still uses the independent u1 (unchanged, was already correct).
    expect(a.octUV[0]).toBeCloseTo(NE.u0 + 0.3 * (NE.u1 - NE.u0), 6);
    expect(b.octUV[0]).toBeCloseTo(NE.u0 + 0.3 * (NE.u1 - NE.u0), 6);

    // PDF math is untouched: pdf = (leafFlux/total)/solidAngle.
    const expectedPdf = (NE.flux / dTree.totalFlux) / NE.solidAngle;
    expect(a.pdf).toBeCloseTo(expectedPdf, 9);
    expect(b.pdf).toBeCloseTo(expectedPdf, 9);
  });

  it('dTreeSample is deterministic for a fixed (u0,u1) pair', () => {
    // The oracle must stay deterministic (tests rely on it): same inputs →
    // identical sample. The decorrelation fix preserves this.
    const dTree = buildEmptyDTree(1);
    dTree.nodes[1]!.flux = 4;
    dTree.nodes[2]!.flux = 6;
    dTree.totalFlux = 10;
    const x = dTreeSample(dTree, 0.7, 0.2);
    const y = dTreeSample(dTree, 0.7, 0.2);
    expect(x.octUV[0]).toBe(y.octUV[0]);
    expect(x.octUV[1]).toBe(y.octUV[1]);
    expect(x.pdf).toBe(y.pdf);
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
