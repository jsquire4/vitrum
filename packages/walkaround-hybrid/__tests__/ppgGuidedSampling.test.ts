/**
 * W9 guided sampling — gi-ris PPG mixture-pdf + explicit RIS weight tests.
 *
 * These CPU-port the WGSL math added to `risGi.wgsl.ts` (the α-mixture RIS
 * source pdf) and `ppgPdf.wgsl.ts` (the dTree pdf-eval that the gi-ris loop
 * evaluates for the chosen direction). They pin:
 *
 *   (a) ppg-OFF (α = 0) → the explicit RIS weight reduces EXACTLY (bit-for-bit
 *       in f64; the WGSL takes the literal shortcut to avoid even an f32 ULP)
 *       to the pre-PPG cosine shortcut `luminance(Lo)`.
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
  serialiseSTree,
  gpuTraverseSTreeLeaf,
  gpuTraverseDTreeLeaf,
} from '../src/ppg/serialise.js';
import { buildSTree, sTreeAccumulate } from '../src/ppg/sTree.js';
import { dTreePdf, refineDTree, findDTreeLeaf } from '../src/ppg/dTree.js';
import type { AABB, STree } from '../src/ppg/types.js';
import { RIS_GI_WGSL, RIS_GI_MODULE } from '../src/shaders/risGi.wgsl.js';
import { PPG_PDF_WGSL } from '../src/ppg/ppgPdf.wgsl.js';
import { PPG_MIS_ALPHA } from '../src/ppg/ppgConstants.js';
import { updateUBO } from '../src/pipeline/uboUpdater.js';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';

const FOUR_PI = 4 * Math.PI;
const INV_PI = 1 / Math.PI;
const LUM_W: [number, number, number] = [0.2126, 0.7152, 0.0722];

const SCENE_AABB: AABB = { min: [-10, -10, -10], max: [10, 10, 10] };

// ── Octahedral encode matching ppgPdf.wgsl's ppgDirToOctUv ──────────────────
// = octEncode(dir) [-1,1]² then *0.5+0.5 → [0,1]². Identical to the producer's
// `dirToOct` (ppgUpdate.wgsl). We re-derive it here so the test pins the exact
// convention the gi-ris guide pdf-eval uses.
function dirToOctUv(d: [number, number, number]): [number, number] {
  const l1 = Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]);
  const n: [number, number, number] = l1 > 1e-20
    ? [d[0] / l1, d[1] / l1, d[2] / l1]
    : [0, 0, 0];
  let ox: number, oy: number;
  if (n[2] >= 0) {
    ox = n[0];
    oy = n[1];
  } else {
    const sx = n[0] >= 0 ? 1 : -1;
    const sy = n[1] >= 0 ? 1 : -1;
    ox = (1 - Math.abs(n[1])) * sx;
    oy = (1 - Math.abs(n[0])) * sy;
  }
  return [ox * 0.5 + 0.5, oy * 0.5 + 0.5];
}

function luminance(c: [number, number, number]): number {
  return c[0] * LUM_W[0] + c[1] * LUM_W[1] + c[2] * LUM_W[2];
}

// ── CPU port of ppgPdf.wgsl `ppgEvalPdf` (sTree descent → dTree leaf pdf) ────
// Mirrors the WGSL EXACTLY, reading the serialised flat buffers (same code the
// WGSL kernel executes). p_guide(ωi) = (leafFlux/totalFlux)/solidAngle, with a
// 1/(4π) uniform fallback when the cell has no flux.
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
  const totalFlux = dTreeBuf[dOff + 2]!;
  if (totalFlux <= 0) return 1 / FOUR_PI;
  const octUV = dirToOctUv(wi);
  // gpuTraverseDTreeLeaf takes the per-cell sub-buffer; slice the cell block.
  const cellBuf = dTreeBuf.subarray(dOff);
  const leafBase = gpuTraverseDTreeLeaf(cellBuf, octUV);
  const leafFlux = cellBuf[leafBase + 4]!;
  const solidAng = cellBuf[leafBase + 5]!;
  return (leafFlux / totalFlux) / Math.max(solidAng, 1e-12);
}

// ── The exact gi-ris explicit RIS weight (CPU port of risGi.wgsl) ───────────
//   ppg-OFF (alpha == 0): w = luminance(Lo)  (literal cosine shortcut)
//   ppg-ON  (alpha > 0):  w = pHat / (alpha·pGuide + (1-alpha)·pCos)
//                         pHat = luminance(Lo)·cosθ·INV_PI ; pCos = cosθ·INV_PI
function risGiWeight(
  Lo: [number, number, number],
  cosTheta: number,
  alpha: number,
  pGuide: number,
): number {
  const pHat = luminance(Lo) * cosTheta * INV_PI;
  if (pHat < 1e-9) return 0;
  if (alpha > 0) {
    const pCos = cosTheta * INV_PI;
    const pSrc = alpha * pGuide + (1 - alpha) * pCos;
    return pSrc > 1e-12 ? pHat / pSrc : 0;
  }
  return luminance(Lo);
}

// Build a trained sTree: one split spatially + a flux-spiked dTree so p_guide
// is non-uniform and exercises the descent.
function buildTrainedSTree(): STree {
  const sTree = buildSTree(SCENE_AABB);
  // Spike directional flux toward +Z at one position, then refine.
  for (let i = 0; i < 200; i++) {
    sTreeAccumulate(sTree, [0, 0, 0], [0.5, 0.5], 5.0);     // +Z-ish leaf
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
      // α = 0 path takes the literal shortcut → EXACTLY luminance(Lo).
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
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(sTree);
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
      // CPU oracle: same octUV → dTreePdf on the CPU dTree.
      const octUV = dirToOctUv(wi);
      const cpu = dTreePdf(cpuDTree, octUV);
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
      const octUV = dirToOctUv(wi);
      const gpuLeafBase = gpuTraverseDTreeLeaf(cellBuf, octUV);
      const cpuLeafIdx = findDTreeLeaf(cpuDTree, octUV);
      // The leaf's flux + solidAngle must match between the two descents.
      const gpuFlux = cellBuf[gpuLeafBase + 4]!;
      const gpuSA = cellBuf[gpuLeafBase + 5]!;
      expect(gpuFlux).toBeCloseTo(cpuDTree.nodes[cpuLeafIdx]!.flux, 5);
      expect(gpuSA).toBeCloseTo(cpuDTree.nodes[cpuLeafIdx]!.solidAngle, 5);
    }
  });

  it('a fresh (untrained) cell falls back to the uniform 1/(4π) guide pdf', () => {
    const sTree = buildSTree(SCENE_AABB); // no accumulation → totalFlux = 0
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(sTree);
    const p = gpuPortEvalPdf(sTreeBuf, dTreeBuf, dTreeOffsets, [0, 0, 0], [0, 0, 1]);
    expect(p).toBeCloseTo(1 / FOUR_PI, 9);
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
      emitterCount: 2, totalEmissivePower: 5,
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
      triIntersectEpsilon: 1e-5, glassMixScale: 0.7,
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    nrc: {},
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

  it('the UBO is 416 bytes (16-byte aligned — ReGIR grid block appended)', () => {
    expect(captureUBO().byteLength).toBe(416);
    expect(416 % 16).toBe(0);
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

  it('risGi computes the explicit mixture weight w = p̂ / p_src on the α>0 path', () => {
    expect(RIS_GI_WGSL).toContain('let pGuide = ppgEvalPdf(pos, wi)');
    expect(RIS_GI_WGSL).toContain('let pSrc = alpha * pGuide + (1.0 - alpha) * pCos');
    expect(RIS_GI_WGSL).toContain('w = select(0.0, pHat / pSrc, pSrc > 1e-12)');
  });

  it('risGi keeps the literal luminance(Lo) shortcut on the α==0 path', () => {
    expect(RIS_GI_WGSL).toMatch(/\} else \{\s*\n\s*w = luminance\(Lo\);/);
  });

  it('risGi requires the ppgPdf module', () => {
    expect(RIS_GI_MODULE.requires).toContain('ppgPdf');
  });

  it('ppgPdf declares the three PPG tree buffers on group(3) bindings 6/7/8', () => {
    expect(PPG_PDF_WGSL).toContain('@group(3) @binding(6) var<storage, read> ppgSTreeBuf_gi');
    expect(PPG_PDF_WGSL).toContain('@group(3) @binding(7) var<storage, read> ppgDTreeBuf_gi');
    expect(PPG_PDF_WGSL).toContain('@group(3) @binding(8) var<storage, read> ppgDTreeOffsets_gi');
  });

  it('ppgPdf evaluates p_guide as (leafFlux/totalFlux)/solidAngle (mirrors dTreePdf)', () => {
    expect(PPG_PDF_WGSL).toContain('(leafFlux / totalFlux) / max(solidAng, 1e-12)');
    expect(PPG_PDF_WGSL).toContain('if (totalFlux <= 0.0) { return 1.0 / PPG_FOUR_PI; }');
  });
});
