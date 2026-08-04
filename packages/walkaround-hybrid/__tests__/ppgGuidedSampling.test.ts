/**
 * W9 guided sampling — gi-ris PPG mixture-pdf + explicit RIS weight tests.
 *
 * These CPU-port the WGSL math added to `risGi.wgsl.ts` (the α-mixture RIS
 * source pdf) and `ppgPdf.wgsl.ts` (the dTree pdf-eval that the gi-ris loop
 * evaluates for the chosen direction). They pin:
 *
 *   (a) ppg-OFF (α = 0) → the declared diffuse/geometric proxy reduces
 *       algebraically to the pre-PPG cosine shortcut `luminance(Lo)`.
 *   (b) ppg-ON (α > 0) → the WEIGHT uses the mixture
 *       p_src = α·p_guide + (1−α)·p_cos (not just sampling).
 *   (c) The GPU-port p_guide (sTree descent → dTree flux-proportional pdf)
 *       matches the CPU `dTreePdf` oracle on a trained tree for every probe
 *       direction.
 *   (d) Unbiasedness sanity — a Monte-Carlo estimator of a flat target
 *       integral has the SAME expected value whether candidates are drawn
 *       from the pure cosine pdf or from the α-mixture, as long as the weight
 *       uses the matching source pdf (the defensive two-pdf evaluation).
 *
 * Reference: Müller et al. 2017 §3.2 (dTree), §3.4 (MIS mixture).
 */

import { describe, it, expect } from 'vitest';
import {
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  serialiseSTree,
  gpuTraverseSTreeLeaf,
  gpuTraverseDTreeLeaf,
} from '../src/ppg/serialise.js';
import { buildSTree, sTreeAccumulate } from '../src/ppg/sTree.js';
import {
  dTreePdf,
  refineDTree,
  findDTreeLeaf,
  recomputeDTreeInteriorFlux,
} from '../src/ppg/dTree.js';
import type { AABB, STree } from '../src/ppg/types.js';
import { RIS_GI_WGSL, RIS_GI_MODULE } from '../src/shaders/risGi.wgsl.js';
import { PPG_PDF_WGSL } from '../src/ppg/ppgPdf.wgsl.js';
import { PPG_MIS_ALPHA } from '../src/ppg/ppgConstants.js';
import { buildPpgRepresentedQueryView } from '../src/ppg/ppgRepresentedProposal.js';
import { WALKAROUND_UBO_SIZE_BYTES } from '../src/pipeline/constants.js';
import { updateUBO } from '../src/pipeline/uboUpdater.js';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';

const FOUR_PI = 4 * Math.PI;
const INV_PI = 1 / Math.PI;
const LUM_W: [number, number, number] = [0.2126, 0.7152, 0.0722];

const SCENE_AABB: AABB = { min: [-10, -10, -10], max: [10, 10, 10] };

// ── Equal-area cylindrical map matching ppgPdf.wgsl's ppgDirToUv ─────────────
// u=(1-z)/2 is uniform in z; v=atan2(y,x)/(2π)+0.5 is azimuth. This is the
// production PPG guide-map convention in both ppgPdf.wgsl and ppgUpdate.wgsl.
function dirToPpgUv(d: [number, number, number]): [number, number] {
  const z = Math.max(-1, Math.min(1, d[2]));
  const u = (1 - z) * 0.5;
  const v = Math.max(0, Math.min(1, Math.atan2(d[1], d[0]) / (2 * Math.PI) + 0.5));
  return [u, v];
}

function ppgUvToDir(uv: [number, number]): [number, number, number] {
  const z = 1 - 2 * uv[0];
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = 2 * Math.PI * (uv[1] - 0.5);
  return [r * Math.cos(phi), r * Math.sin(phi), z];
}

function luminance(c: [number, number, number]): number {
  return c[0] * LUM_W[0] + c[1] * LUM_W[1] + c[2] * LUM_W[2];
}

// ── CPU port of ppgPdf.wgsl's represented dTree query contract ──────────────
// The production query shader validates each local child partition before it
// descends. On a partial count/pointer/patch corruption, both sampling and PDF
// evaluation stop at the same current patch and interpret its published
// subtree buckets as a local-uniform proposal. These helpers mirror that path.
const REPRESENTED_BUCKETS = 2 ** 24;

function queryNodeBase(dOff: number, nodeIndex: number): number {
  return dOff + DTREE_HEADER_F32 + nodeIndex * DTREE_NODE_F32;
}

function queryNodeBucketCount(buf: Float32Array, base: number): number {
  const kind = buf[base + 7]!;
  const encoded = buf[base + (kind === 1 ? 6 : 5)]!;
  if (
    (kind !== 0 && kind !== 1) ||
    !Number.isInteger(encoded) ||
    encoded < 0 ||
    encoded > REPRESENTED_BUCKETS
  ) return -1;
  return encoded;
}

function queryPatchValid(buf: Float32Array, base: number): boolean {
  const u0 = buf[base + 0]!;
  const v0 = buf[base + 1]!;
  const u1 = buf[base + 2]!;
  const v1 = buf[base + 3]!;
  return u0 >= 0 && v0 >= 0 && u1 <= 1 && v1 <= 1 && u1 > u0 && v1 > v0;
}

function queryPatchSolidAngle(buf: Float32Array, base: number): number {
  const du = Math.fround(buf[base + 2]! - buf[base + 0]!);
  const dv = Math.fround(buf[base + 3]! - buf[base + 1]!);
  return Math.fround(Math.fround(Math.fround(FOUR_PI) * du) * dv);
}

function queryChildrenValid(
  buf: Float32Array,
  dOff: number,
  nodeCount: number,
  nodeIndex: number,
  base: number,
): boolean {
  if (!queryPatchValid(buf, base) || queryNodeBucketCount(buf, base) < 0) return false;
  const firstChild = buf[base + 6]!;
  if (
    !Number.isInteger(firstChild) ||
    firstChild <= nodeIndex ||
    firstChild < 0 ||
    firstChild + 3 >= nodeCount
  ) return false;
  const u0 = buf[base + 0]!;
  const v0 = buf[base + 1]!;
  const u1 = buf[base + 2]!;
  const v1 = buf[base + 3]!;
  const uMid = Math.fround(Math.fround(u0 + u1) * 0.5);
  const vMid = Math.fround(Math.fround(v0 + v1) * 0.5);
  let sum = 0;
  for (let child = 0; child < 4; child += 1) {
    const childBase = queryNodeBase(dOff, firstChild + child);
    const buckets = queryNodeBucketCount(buf, childBase);
    if (!queryPatchValid(buf, childBase) || buckets < 0) return false;
    const right = (child & 1) !== 0;
    const bottom = (child & 2) !== 0;
    if (
      buf[childBase + 0] !== (right ? uMid : u0) ||
      buf[childBase + 1] !== (bottom ? vMid : v0) ||
      buf[childBase + 2] !== (right ? u1 : uMid) ||
      buf[childBase + 3] !== (bottom ? v1 : vMid)
    ) return false;
    sum += buckets;
  }
  return sum === queryNodeBucketCount(buf, base);
}

function queryRootValid(buf: Float32Array, dOff: number): boolean {
  const nodeCount = buf[dOff]!;
  const rootBase = queryNodeBase(dOff, 0);
  return Number.isInteger(nodeCount) && nodeCount >= 1 &&
    queryNodeBucketCount(buf, rootBase) === REPRESENTED_BUCKETS &&
    queryPatchValid(buf, rootBase) &&
    buf[rootBase + 0] === 0 && buf[rootBase + 1] === 0 &&
    buf[rootBase + 2] === 1 && buf[rootBase + 3] === 1;
}

function queryDistributionBase(
  buf: Float32Array,
  dOff: number,
  uv: [number, number],
): number {
  const nodeCount = buf[dOff]!;
  let nodeIndex = 0;
  let fallbackBase = queryNodeBase(dOff, 0);
  for (let step = 0; step < 32; step += 1) {
    const base = queryNodeBase(dOff, nodeIndex);
    fallbackBase = base;
    if (buf[base + 7] === 1) return base;
    if (!queryChildrenValid(buf, dOff, nodeCount, nodeIndex, base)) return base;
    const uMid = Math.fround(Math.fround(buf[base + 0]! + buf[base + 2]!) * 0.5);
    const vMid = Math.fround(Math.fround(buf[base + 1]! + buf[base + 3]!) * 0.5);
    const firstChild = buf[base + 6]!;
    nodeIndex = firstChild + (uv[0] >= uMid ? 1 : 0) + (uv[1] >= vMid ? 2 : 0);
  }
  return fallbackBase;
}

function gpuPortSampleGuideUv(
  dTreeBuf: Float32Array,
  dOff: number,
  selectionBucket: number,
  jitter: [number, number],
): [number, number] {
  if (!queryRootValid(dTreeBuf, dOff)) return jitter;
  const nodeCount = dTreeBuf[dOff]!;
  let nodeIndex = 0;
  let remaining = Math.max(0, Math.min(REPRESENTED_BUCKETS - 1, selectionBucket | 0));
  let base = queryNodeBase(dOff, 0);
  for (let step = 0; step < 32; step += 1) {
    base = queryNodeBase(dOff, nodeIndex);
    if (dTreeBuf[base + 7] === 1) break;
    if (!queryChildrenValid(dTreeBuf, dOff, nodeCount, nodeIndex, base)) break;
    const firstChild = dTreeBuf[base + 6]!;
    let selected = false;
    for (let child = 0; child < 4; child += 1) {
      const childBase = queryNodeBase(dOff, firstChild + child);
      const buckets = queryNodeBucketCount(dTreeBuf, childBase);
      if (remaining < buckets) {
        nodeIndex = firstChild + child;
        selected = true;
        break;
      }
      remaining -= buckets;
    }
    if (!selected) break;
  }
  return [
    dTreeBuf[base + 0]! + jitter[0] * (dTreeBuf[base + 2]! - dTreeBuf[base + 0]!),
    dTreeBuf[base + 1]! + jitter[1] * (dTreeBuf[base + 3]! - dTreeBuf[base + 1]!),
  ];
}

function gpuPortEvalPdf(
  sTreeBuf: Float32Array,
  dTreeBuf: Float32Array,
  dTreeOffsets: Uint32Array,
  pos: [number, number, number],
  wi: [number, number, number],
): number {
  const sBase = gpuTraverseSTreeLeaf(sTreeBuf, pos);
  const dTreeIndex = sTreeBuf[sBase + 10]! | 0;
  const dOff = dTreeOffsets[dTreeIndex]!;
  if (!queryRootValid(dTreeBuf, dOff)) return Math.fround(1 / Math.fround(FOUR_PI));
  const guideUV = dirToPpgUv(wi);
  const distributionBase = queryDistributionBase(dTreeBuf, dOff, guideUV);
  const distributionBuckets = queryNodeBucketCount(dTreeBuf, distributionBase);
  const solidAng = queryPatchSolidAngle(dTreeBuf, distributionBase);
  if (!(distributionBuckets > 0) || !(solidAng > 0)) return 0;
  return Math.fround(
    Math.fround(distributionBuckets / REPRESENTED_BUCKETS) / solidAng,
  );
}

// ── The exact gi-ris explicit RIS weight (CPU port of risGi.wgsl) ───────────
//   ppg-OFF (alpha == 0): diffuse-default pHat / pCos == luminance(Lo)
//   ppg-ON  (alpha > 0):  w = pHat / (alpha·pGuide + (1-alpha)·pCos)
//                         pHat = luminance(Lo)·cosθ·INV_PI ; pCos = cosθ·INV_PI
function risGiWeight(
  Lo: [number, number, number],
  cosTheta: number,
  alpha: number,
  pGuide: number,
): number {
  const pHat = luminance(Lo) * cosTheta * INV_PI;
  if (!(pHat > 0) || !Number.isFinite(pHat)) return 0;
  if (alpha > 0) {
    const pCos = cosTheta * INV_PI;
    const pSrc = alpha * pGuide + (1 - alpha) * pCos;
    return pSrc > 0 && Number.isFinite(pSrc) ? pHat / pSrc : 0;
  }
  return luminance(Lo);
}

// Build a trained sTree: one split spatially + a flux-spiked dTree so p_guide
// is non-uniform and exercises the descent.
function buildTrainedSTree(): STree {
  const sTree = buildSTree(SCENE_AABB);
  // Spike directional flux toward +X at one position, then refine.
  const hotUv = dirToPpgUv([1, 0, 0]);
  for (let i = 0; i < 200; i++) {
    sTreeAccumulate(sTree, [0, 0, 0], hotUv, 5.0);
    sTreeAccumulate(sTree, [0, 0, 0], [0.1, 0.1], 0.05);    // tail
    sTreeAccumulate(sTree, [0, 0, 0], [0.9, 0.9], 0.05);
  }
  for (const dTree of sTree.dTrees) refineDTree(dTree);
  return sTree;
}

describe('W9 gi-ris — ppg-OFF explicit RIS weight bit-identity (α = 0)', () => {
  it('reduces to luminance(Lo) exactly for any cosTheta / Lo when α = 0', () => {
    const cases: Array<{ Lo: [number, number, number]; cos: number }> = [
      { Lo: [1, 1, 1], cos: 0.5 },
      { Lo: [0.3, 0.7, 0.2], cos: 0.9 },
      { Lo: [12, 0.1, 4], cos: 0.01 },
      { Lo: [2.5, 2.5, 2.5], cos: 0.9999 },
      { Lo: [0.001, 0.002, 0.0005], cos: 0.3 },
    ];
    for (const { Lo, cos } of cases) {
      // α = 0 diffuse-default path algebraically reduces to luminance(Lo).
      const w = risGiWeight(Lo, cos, 0, /*pGuide unused*/ 1.23);
      expect(w).toBe(luminance(Lo));
    }
  });

  it('pGuide is never consulted on the α = 0 path (gate proven by value-independence)', () => {
    const Lo: [number, number, number] = [0.4, 0.5, 0.6];
    const cos = 0.42;
    const wA = risGiWeight(Lo, cos, 0, 0.0001);
    const wB = risGiWeight(Lo, cos, 0, 9999.0);
    expect(wA).toBe(wB);          // independent of pGuide
    expect(wA).toBe(luminance(Lo));
  });
});

describe('W9 gi-ris — ppg-ON explicit RIS weight uses the mixture pdf (α > 0)', () => {
  it('CPU guide-map oracle matches the production equal-area cylindrical convention', () => {
    expect(dirToPpgUv([0, 0, 1])).toEqual([0, 0.5]);
    expect(dirToPpgUv([1, 0, 0])).toEqual([0.5, 0.5]);
    expect(dirToPpgUv([0, 1, 0])).toEqual([0.5, 0.75]);
    expect(dirToPpgUv([0, -1, 0])).toEqual([0.5, 0.25]);
  });

  it('weight uses p_src = α·p_guide + (1−α)·p_cos, NOT the cosine shortcut', () => {
    const Lo: [number, number, number] = [0.6, 0.6, 0.6];
    const cos = 0.5;
    const alpha = 0.5;
    const pGuide = 0.4; // arbitrary non-cosine guide pdf
    const w = risGiWeight(Lo, cos, alpha, pGuide);

    const pHat = luminance(Lo) * cos * INV_PI;
    const pCos = cos * INV_PI;
    const pSrc = alpha * pGuide + (1 - alpha) * pCos;
    expect(w).toBeCloseTo(pHat / pSrc, 12);

    // It must DIFFER from the cosine shortcut whenever p_guide ≠ p_cos.
    expect(w).not.toBeCloseTo(luminance(Lo), 6);
  });

  it('as α → 0⁺ the mixture weight converges to the cosine shortcut (continuity)', () => {
    const Lo: [number, number, number] = [0.8, 0.4, 0.2];
    const cos = 0.7;
    const pGuide = 1.5;
    const wTiny = risGiWeight(Lo, cos, 1e-6, pGuide);
    expect(wTiny).toBeCloseTo(luminance(Lo), 4);
  });

  it('default α from ppgConstants is in (0,1) so the mixture is well-formed', () => {
    expect(PPG_MIS_ALPHA).toBeGreaterThan(0);
    expect(PPG_MIS_ALPHA).toBeLessThan(1);
  });
});

describe('W9 gi-ris — GPU-port p_guide matches CPU dTreePdf oracle', () => {
  it('every probe direction agrees with dTreePdf on the trained cell', () => {
    const sTree = buildTrainedSTree();
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = buildPpgRepresentedQueryView(
      serialiseSTree(sTree),
    );
    const pos: [number, number, number] = [0, 0, 0];

    // The CPU oracle for this position: descend sTree, get that cell's dTree.
    const cpuLeafIdx = (() => {
      // findSTreeLeaf is internal to sTree; reuse the GPU-port to find the
      // cell, then map back to the CPU dTree via the dTreeIndex field.
      const sBase = gpuTraverseSTreeLeaf(sTreeBuf, pos);
      return sTreeBuf[sBase + 10]! | 0;
    })();
    const cpuDTree = sTree.dTrees[cpuLeafIdx]!;

    // Probe a spread of world directions (upper + lower hemisphere).
    const probes: Array<[number, number, number]> = [
      [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0],
      [0.577, 0.577, 0.577], [-0.577, 0.577, 0.577],
      [0.1, 0.2, 0.97], [0.5, -0.5, 0.707], [-0.3, -0.4, 0.866],
    ];
    for (const wi of probes) {
      const gpu = gpuPortEvalPdf(sTreeBuf, dTreeBuf, dTreeOffsets, pos, wi);
      // CPU oracle: same production guide UV → dTreePdf on the CPU dTree.
      const guideUV = dirToPpgUv(wi);
      const cpu = dTreePdf(cpuDTree, guideUV);
      expect(gpu).toBeCloseTo(cpu, 5);
    }
  });

  it('GPU-port and CPU descend to the SAME dTree leaf for each probe', () => {
    const sTree = buildTrainedSTree();
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(sTree);
    const pos: [number, number, number] = [0, 0, 0];
    const sBase = gpuTraverseSTreeLeaf(sTreeBuf, pos);
    const dTreeIndex = sTreeBuf[sBase + 10]! | 0;
    const cpuDTree = sTree.dTrees[dTreeIndex]!;
    const dOff = dTreeOffsets[dTreeIndex]!;
    const cellBuf = dTreeBuf.subarray(dOff);

    const probes: Array<[number, number, number]> = [
      [0, 0, 1], [0.2, 0.1, 0.97], [-0.4, 0.3, 0.866], [0.6, 0.6, 0.52],
    ];
    for (const wi of probes) {
      const guideUV = dirToPpgUv(wi);
      const gpuLeafBase = gpuTraverseDTreeLeaf(cellBuf, guideUV);
      const cpuLeafIdx = findDTreeLeaf(cpuDTree, guideUV);
      // The leaf's flux + solidAngle must match between the two descents.
      const gpuFlux = cellBuf[gpuLeafBase + 4]!;
      const gpuSA = cellBuf[gpuLeafBase + 5]!;
      expect(gpuFlux).toBeCloseTo(cpuDTree.nodes[cpuLeafIdx]!.flux, 5);
      expect(gpuSA).toBeCloseTo(cpuDTree.nodes[cpuLeafIdx]!.solidAngle, 5);
    }
  });

  it('a fresh (untrained) cell falls back to the uniform 1/(4π) guide pdf', () => {
    const sTree = buildSTree(SCENE_AABB); // no accumulation → totalFlux = 0
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = buildPpgRepresentedQueryView(
      serialiseSTree(sTree),
    );
    const p = gpuPortEvalPdf(sTreeBuf, dTreeBuf, dTreeOffsets, [0, 0, 0], [0, 0, 1]);
    expect(p).toBe(Math.fround(1 / Math.fround(FOUR_PI)));
  });

  it('keeps the defensive partial-root fallback sampler/PDF pair uniform', () => {
    const sTree = buildTrainedSTree();
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = buildPpgRepresentedQueryView(
      serialiseSTree(sTree),
    );
    const dTreeIndex = sTreeBuf[gpuTraverseSTreeLeaf(sTreeBuf, [0, 0, 0]) + 10]! | 0;
    const rootBase = dTreeOffsets[dTreeIndex]! + 4;
    const dOff = dTreeOffsets[dTreeIndex]!;
    expect(dTreeBuf[rootBase + 7]).toBe(0);
    dTreeBuf[rootBase + 5] = 123; // impossible in a valid query view

    const sampledUv = gpuPortSampleGuideUv(dTreeBuf, dOff, 7, [0.2, 0.8]);
    expect(sampledUv).toEqual([0.2, 0.8]);

    const p = gpuPortEvalPdf(
      sTreeBuf,
      dTreeBuf,
      dTreeOffsets,
      [0, 0, 0],
      [0, 0, 1],
    );
    expect(p).toBe(Math.fround(1 / Math.fround(FOUR_PI)));
  });

  it.each([
    ['child bucket sum', (buf: Float32Array, childBase: number, dOff: number) => {
      const grandchild = queryNodeBase(dOff, buf[childBase + 6]!);
      buf[grandchild + 6] = buf[grandchild + 6]! + 1;
    }],
    ['child pointer', (buf: Float32Array, childBase: number, _dOff: number) => {
      buf[childBase + 6] = 1.5;
    }],
    ['child patch', (buf: Float32Array, childBase: number, dOff: number) => {
      const grandchild = queryNodeBase(dOff, buf[childBase + 6]!);
      buf[grandchild + 0] = Math.fround(buf[grandchild + 0]! + 0.01);
    }],
  ])('keeps a partial lower-level %s corruption sampler/PDF symmetric', (_label, corrupt) => {
    const sTree = buildSTree(SCENE_AABB, 2);
    const dTree = sTree.dTrees[0]!;
    for (let index = 0; index < dTree.nodes.length; index += 1) {
      const node = dTree.nodes[index]!;
      if (node.isLeaf) node.flux = index >= 5 && index <= 8 ? 4 : 1;
    }
    recomputeDTreeInteriorFlux(dTree);
    const query = buildPpgRepresentedQueryView(serialiseSTree(sTree));
    const dOff = query.dTreeOffsets[0]!;
    const rootBase = queryNodeBase(dOff, 0);
    const firstChildIndex = query.dTreeBuf[rootBase + 6]!;
    const firstChildBase = queryNodeBase(dOff, firstChildIndex);
    const firstChildBuckets = queryNodeBucketCount(query.dTreeBuf, firstChildBase);
    expect(firstChildBuckets).toBeGreaterThan(0);
    expect(firstChildBuckets).toBeLessThan(REPRESENTED_BUCKETS);

    corrupt(query.dTreeBuf, firstChildBase, dOff);
    const sampledUv = gpuPortSampleGuideUv(query.dTreeBuf, dOff, 0, [0.25, 0.75]);
    expect(sampledUv[0]).toBeGreaterThanOrEqual(query.dTreeBuf[firstChildBase + 0]!);
    expect(sampledUv[0]).toBeLessThan(query.dTreeBuf[firstChildBase + 2]!);
    expect(sampledUv[1]).toBeGreaterThanOrEqual(query.dTreeBuf[firstChildBase + 1]!);
    expect(sampledUv[1]).toBeLessThan(query.dTreeBuf[firstChildBase + 3]!);

    const expected = Math.fround(
      Math.fround(firstChildBuckets / REPRESENTED_BUCKETS) /
        queryPatchSolidAngle(query.dTreeBuf, firstChildBase),
    );
    const evaluated = gpuPortEvalPdf(
      query.sTreeBuf,
      query.dTreeBuf,
      query.dTreeOffsets,
      [0, 0, 0],
      ppgUvToDir(sampledUv),
    );
    expect(evaluated).toBe(expected);
  });
});

describe('W9 gi-ris — unbiasedness sanity (cosine vs mixture source pdf)', () => {
  // On a FLAT target f(ω) = 1 over the hemisphere, the cosine-importance
  // estimator E[ f·cos / p_src ] with p_src = the cosine pdf integrates the
  // cosine-weighted hemisphere ∫cosθ dω = π. The SAME integral estimated with
  // the α-mixture source pdf (and the matching mixture weight) must converge
  // to the SAME value — that equality is the unbiasedness property: switching
  // the proposal distribution (cosine vs guided) while using the correct
  // source pdf in the weight leaves the expectation unchanged.
  function estimate(
    alpha: number,
    pGuideFn: (wi: [number, number, number]) => number,
    sampleGuided: (rng: () => number) => [number, number, number],
    nSamples: number,
    seed: number,
  ): number {
    let state = seed >>> 0;
    const rng = (): number => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0xffffffff;
    };
    let sum = 0;
    for (let i = 0; i < nSamples; i++) {
      let wi: [number, number, number];
      if (alpha > 0 && rng() < alpha) {
        wi = sampleGuided(rng);
      } else {
        // cosine-weighted hemisphere about +Z (Malley): r=√u, θ=2πv.
        const u = rng(), v = rng();
        const r = Math.sqrt(u);
        const phi = 2 * Math.PI * v;
        wi = [r * Math.cos(phi), r * Math.sin(phi), Math.sqrt(Math.max(0, 1 - u))];
      }
      const cos = Math.max(0, wi[2]);
      if (cos < 1e-4) continue;
      // Flat target: f = 1. The estimator integrand is f·cos / p_src.
      const pCos = cos * INV_PI;
      const pSrc = alpha > 0 ? alpha * pGuideFn(wi) + (1 - alpha) * pCos : pCos;
      if (pSrc <= 1e-12) continue;
      sum += (1.0 * cos) / pSrc;
    }
    return sum / nSamples;
  }

  it('cosine-only and α-mixture estimators converge to the same ∫cosθ dω = π', () => {
    // Guided proposal: uniform-sphere sample (so p_guide = 1/4π — the uniform
    // fallback, which is exactly what an untrained dTree returns). Using a
    // uniform-sphere guide keeps the test self-contained while still
    // exercising the mixture weight with a genuinely different proposal.
    const pGuideUniform = (): number => 1 / FOUR_PI;
    const sampleUniformSphere = (rng: () => number): [number, number, number] => {
      const z = rng() * 2 - 1;
      const phi = rng() * 2 * Math.PI;
      const rxy = Math.sqrt(Math.max(0, 1 - z * z));
      return [rxy * Math.cos(phi), rxy * Math.sin(phi), z];
    };

    const N = 400_000;
    const cosineOnly = estimate(0, pGuideUniform, sampleUniformSphere, N, 0xABCDEF);
    const mixture = estimate(0.5, pGuideUniform, sampleUniformSphere, N, 0x123456);

    // Both target ∫_hemisphere cosθ dω = π.
    expect(cosineOnly).toBeCloseTo(Math.PI, 1);
    expect(mixture).toBeCloseTo(Math.PI, 1);
    // And they agree with each other (the unbiasedness equality).
    expect(Math.abs(mixture - cosineOnly)).toBeLessThan(0.05);
  });
});

describe('W9 gi-ris — UBO byte layout (ppg-OFF bit-identity + ppg-ON gate)', () => {
  // Minimal inputs covering only the fields updateUBO reads. Cast through
  // unknown because updateUBO ignores swapChain* / sigma / gtao* / adaptive*.
  const m = new Float32Array(16).fill(0);
  const baseInputs: PipelineFrameInputs = {
    camera: { viewMatrix: m, projMatrix: m, prevViewProjMatrix: m, cameraPos: [1, 2, 3] },
    screen: { screenWidth: 64, screenHeight: 48, frameSeed: 7, swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm' },
    lighting: {
      emitterCount: 2,
      primaryLightDir: [0, 1, 0], primaryLightIntensity: 3,
      skyTint: [0.1, 0.2, 0.3], skyIrradiance: 0.5,
      emitterDist2Floor: 0.01, directFireflyClamp: 4, causticBoost: 1, causticVisClamp: 1,
      lightTreeEnabled: 0, lightTreeNodeCount: 0,
    },
    restirDI: { temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05 },
    restirGI: {
      restirGiWCap: 16, restirGiIrrClamp: 5, restirGiMClamp: 50,
      restirGiSpatialRadiusPx: 12, restirGiSpatialNormalDotMin: 0.906,
      restirGiSpatialCoplanarTol: 0.05,
    },
    gtao: { gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2, gtaoBilateralDepthSigma: 0.25, adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1 },
    filter: {
      triIntersectEpsilon: 1e-5, rayOriginBias: 1e-3, glassMixScale: 0.7,
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
  };

  function captureUBO(ppg?: { enabled: boolean; mixAlpha: number }): ArrayBuffer {
    let captured: ArrayBuffer | null = null;
    const device = {
      queue: {
        writeBuffer: (_buf: unknown, _off: number, data: ArrayBuffer) => {
          captured = data.slice(0);
        },
      },
    } as unknown as GPUDevice;
    if (ppg) updateUBO(device, {} as GPUBuffer, baseInputs, ppg);
    else updateUBO(device, {} as GPUBuffer, baseInputs);
    if (captured === null) throw new Error('writeBuffer not called');
    return captured;
  }

  it('the UBO is 432 bytes and remains 16-byte aligned', () => {
    expect(captureUBO().byteLength).toBe(WALKAROUND_UBO_SIZE_BYTES);
    expect(WALKAROUND_UBO_SIZE_BYTES % 16).toBe(0);
  });

  it('ppg-OFF writes ppgEnabled=0 @348 and ppgMixAlpha=0 @352', () => {
    const buf = captureUBO(); // default ppg = { enabled:false, mixAlpha:0 }
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    expect(u32[87]).toBe(0); // offset 348 — ppgEnabled
    expect(f32[88]).toBe(0); // offset 352 — ppgMixAlpha
  });

  it('ppg-ON writes ppgEnabled=1 @348 and the supplied α @352', () => {
    const buf = captureUBO({ enabled: true, mixAlpha: PPG_MIS_ALPHA });
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    expect(u32[87]).toBe(1);
    expect(f32[88]).toBeCloseTo(PPG_MIS_ALPHA, 6);
  });

  it('bytes 0..347 are BIT-IDENTICAL between ppg-OFF and ppg-ON (only the tail changes)', () => {
    const off = new Uint8Array(captureUBO({ enabled: false, mixAlpha: 0 }));
    const on = new Uint8Array(captureUBO({ enabled: true, mixAlpha: PPG_MIS_ALPHA }));
    for (let i = 0; i < 348; i++) {
      expect(off[i]).toBe(on[i]);
    }
  });

  it('an enabled:false gate forces α=0 even if a non-zero mixAlpha is passed', () => {
    const buf = captureUBO({ enabled: false, mixAlpha: 0.9 });
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    expect(u32[87]).toBe(0);
    expect(f32[88]).toBe(0); // gate forces α to 0 — preserves ppg-OFF identity
  });
});

describe('W9 gi-ris — WGSL structure pins', () => {
  it('risGi reads the PPG gate + α from the UBO', () => {
    expect(RIS_GI_WGSL).toContain('ubo.ppgEnabled == 1u');
    expect(RIS_GI_WGSL).toContain('ubo.ppgMixAlpha');
  });

  it('risGi gates the Bernoulli draw on alpha > 0 (no extra rng when PPG off)', () => {
    expect(RIS_GI_WGSL).toMatch(/if \(alpha > 0\.0\) \{\s*\n\s*let bern = rand_f32/);
  });

  it('risGi computes the proposal-mixture weight in the log domain on the α>0 path', () => {
    expect(RIS_GI_WGSL).toContain('let pGuide = ppgEvalPdf(pos, wi)');
    expect(RIS_GI_WGSL).toContain(
      'logPSrc = reservoirGiLogProposalMixture(alpha, pGuide, pCos);',
    );
    expect(RIS_GI_WGSL).toContain('if (!reservoirGiValidLog(logPSrc))');
    expect(RIS_GI_WGSL).toContain('let logWeight = logPHat - logPSrc;');
  });

  it('risGi keeps the generalized proxy target on the α==0 path', () => {
    expect(RIS_GI_WGSL).toMatch(
      /} else {\s*\n\s*logPSrc = reservoirGiLogPositive\(pCos\);/,
    );
    expect(RIS_GI_WGSL).toContain('let logWeight = logPHat - logPSrc;');
    expect(RIS_GI_WGSL).toContain(
      'let receiverPHat = restir_gi_receiver_phat_from_payload(',
    );
    expect(RIS_GI_WGSL).toContain(
      'let logPHat = reservoirGiLogPositiveProduct(receiverPHat, candidateVisibility);',
    );
  });

  it('risGi requires the ppgPdf module', () => {
    expect(RIS_GI_MODULE.requires).toContain('ppgPdf');
  });

  it('ppgPdf reads all PPG trees through the packed group(3) query arena', () => {
    expect(PPG_PDF_WGSL).toContain('@group(3) @binding(6) var<storage, read> ppgQueryArena_gi');
    expect(PPG_PDF_WGSL).toContain('ppgQueryArena_gi[ppgQueryArena_gi[4] + word]');
    expect(PPG_PDF_WGSL).toContain('ppgQueryArena_gi[ppgQueryArena_gi[7] + word]');
    expect(PPG_PDF_WGSL).toContain('ppgQueryArena_gi[ppgQueryArena_gi[10] + word]');
    expect(PPG_PDF_WGSL).not.toContain('@group(3) @binding(7)');
    expect(PPG_PDF_WGSL).not.toContain('@group(3) @binding(8)');
  });

  it('ppgPdf evaluates the exact represented leaf mass divided by solid angle', () => {
    expect(PPG_PDF_WGSL).toContain(
      'if (distributionBuckets == 0u) { return 0.0; }',
    );
    expect(PPG_PDF_WGSL).toContain(
      '(f32(distributionBuckets) * PPG_INV_REPRESENTED_BUCKETS) / solidAng',
    );
    expect(PPG_PDF_WGSL).not.toContain('max(solidAng, 1e-12)');
    expect(PPG_PDF_WGSL).toContain('!ppgDTreeRootValidGi(dOff)');
  });
});
