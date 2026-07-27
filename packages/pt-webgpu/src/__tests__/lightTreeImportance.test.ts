/**
 * WS2 — Many-light importance sampling (pt-webgpu).
 *
 * TDD spec for the power-weighted light-tree selection that replaces pt-webgpu's
 * uniform random emitter pick. Covers:
 *   1. Per-emitter-type power formula  (area: luminance(Le)·area; delta: luminance(Le)).
 *   2. Partition-of-unity of the selection pmf (Σ pdf == 1 over the emitter set).
 *   3. Power-field packing round-trip (CPU build → packLightTreeForGPU → re-read).
 *   4. UNBIASEDNESS Monte-Carlo convergence: the power-weighted single-sample NEE
 *      estimator E[ contrib / p_select ] equals the uniform estimator in the mean.
 *   5. VARIANCE REDUCTION: with a ≥5× power spread the power-weighted estimator's
 *      variance is < 0.8× the uniform estimator's at equal mean.
 *
 * The CPU light-tree traversal (`sampleLightTreeCPU` / `lightTreePdfCPU`) is the
 * byte-for-byte reference the WGSL `sampleLightTree` mirrors, so verifying the
 * estimator here pins the GPU NEE math (modulo the WGSL string, which is grepped
 * structurally in lightTreeConsumption.test.ts).
 *
 * Refs:
 *   - Conty Estévez & Kulla 2018, "Importance Sampling of Many Lights with
 *     Adaptive Tree Splitting" — power × spatial-proximity tree descent.
 *   - Shirley, Smits, Wang, Zimmerman 1996, "Monte Carlo Techniques for Direct
 *     Lighting Calculations" — power-weighted light-list partition.
 */

import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  buildLightTree,
  packLightTreeForGPU,
  lightTreePdfCPU,
  sampleLightTreeCPU,
  LIGHT_TREE_FLOATS_PER_NODE,
  luminance,
} from '@vitrum/shared-samplers';
import {
  emitterPower,
  buildLightTreeInputForScene,
  packEmitterArrays,
  AREA_LIGHT_KINDS,
} from '../scene/emitterPacking.js';

// ---------------------------------------------------------------------------
// Deterministic xorshift RNG so the MC tests are reproducible.
// ---------------------------------------------------------------------------
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 0xffffffff;
  };
}

function emptyScene(): Scene {
  return {
    primitives: [],
    emitters: [],
    environment: { kind: 'none' },
  } as unknown as Scene;
}

// ===========================================================================
// 1. Per-emitter-type power formula
// ===========================================================================
describe('WS2 — per-emitter-type power formula', () => {
  it('keeps a tiny positive procedural sky in light-tree selector parity', () => {
    const input = buildLightTreeInputForScene(emptyScene(), {
      envSummary: { hasHdri: false, sunStrength: 1e-12, tint: [1, 1, 1] },
    });
    expect(input.powers).toHaveLength(1);
    expect(input.powers[0]).toBeGreaterThan(0);
    const source = buildLightTreeInputForScene.toString();
    expect(source).toContain('envSummary.sunStrength > 0');
    expect(source).not.toContain('envSummary.sunStrength > 1e-6');
  });

  it('delta light (point/spot/directional) power == luminance(radiance)', () => {
    const radiance: [number, number, number] = [2, 4, 6];
    const expected = luminance(radiance[0], radiance[1], radiance[2]);
    expect(emitterPower(radiance, { kind: 'delta' })).toBeCloseTo(expected, 12);
  });

  it('area light power == luminance(radiance) · area', () => {
    const radiance: [number, number, number] = [1, 1, 1];
    const area = 3.5;
    const expected = luminance(1, 1, 1) * area;
    expect(emitterPower(radiance, { kind: 'area', area })).toBeCloseTo(expected, 12);
  });

  it('power scales linearly with radiance magnitude', () => {
    const r1: [number, number, number] = [1, 1, 1];
    const r2: [number, number, number] = [2, 2, 2];
    const p1 = emitterPower(r1, { kind: 'delta' });
    const p2 = emitterPower(r2, { kind: 'delta' });
    expect(p2 / p1).toBeCloseTo(2, 12);
  });

  it('power is non-negative and zero for a black emitter', () => {
    expect(emitterPower([0, 0, 0], { kind: 'delta' })).toBe(0);
    expect(emitterPower([0, 0, 0], { kind: 'area', area: 10 })).toBe(0);
  });

  it('AREA_LIGHT_KINDS enumerates exactly the positional area emitters', () => {
    expect([...AREA_LIGHT_KINDS].sort()).toEqual(['disc-area', 'mesh-area', 'rect-area']);
  });
});

// ===========================================================================
// 2. + 3. Build / partition-of-unity / round-trip on a real Scene
// ===========================================================================
describe('WS2 — light-tree build over a multi-light scene', () => {
  // 3 point lights of differing brightness + 1 rect-area light.
  const scene: Scene = {
    ...emptyScene(),
    emitters: [
      { id: 'p0', kind: 'point', position: [0, 5, 0], color: [1, 1, 1], intensity: 1 },
      { id: 'p1', kind: 'point', position: [4, 5, 0], color: [1, 1, 1], intensity: 10 },
      { id: 'p2', kind: 'point', position: [-4, 5, 0], color: [1, 1, 1], intensity: 0.5 },
      {
        id: 'r0',
        kind: 'rect-area',
        position: [0, 8, 4],
        uAxis: [1, 0, 0],
        vAxis: [0, 0, 1],
        color: [1, 1, 1],
        intensity: 2,
      },
    ],
  } as unknown as Scene;

  const input = buildLightTreeInputForScene(scene);

  it('produces one tree leaf per selectable light, in NEE-walk order', () => {
    // No directional / env in this scene → 3 point + 1 area = 4 leaves.
    expect(input.powers.length).toBe(4);
    expect(input.centroids.length).toBe(4);
    expect(input.aabbs.length).toBe(4);
  });

  it('the brightest light carries the most power', () => {
    // p1 (intensity 10) must dominate.
    const maxIdx = input.powers.indexOf(Math.max(...input.powers));
    expect(maxIdx).toBe(1);
  });

  it('selection pmf is a partition of unity (Σ pdf == 1) at any shading point', () => {
    const { nodes } = buildLightTree(input);
    const shadingPoints: [number, number, number][] = [
      [0, 0, 0],
      [3, 1, -2],
      [-5, 2, 5],
      [0, 100, 0],
    ];
    for (const x of shadingPoints) {
      let total = 0;
      for (let e = 0; e < input.powers.length; e++) {
        total += lightTreePdfCPU(nodes, x, 1e-3, e);
      }
      expect(total).toBeCloseTo(1, 6);
    }
  });

  it('every reachable leaf has a strictly-positive selection pdf (no infinite RIS weight)', () => {
    const { nodes } = buildLightTree(input);
    const x: [number, number, number] = [0, 0, 0];
    for (let e = 0; e < input.powers.length; e++) {
      expect(lightTreePdfCPU(nodes, x, 1e-3, e)).toBeGreaterThan(0);
    }
  });

  it('power-field packing round-trips through packLightTreeForGPU', () => {
    const { nodes } = buildLightTree(input);
    const packed = packLightTreeForGPU(nodes);
    expect(packed.length).toBe(nodes.length * LIGHT_TREE_FLOATS_PER_NODE);
    for (let i = 0; i < nodes.length; i++) {
      const base = i * LIGHT_TREE_FLOATS_PER_NODE;
      // [0] emitterIndex, [1] totalPower, [2] leftChild, [3] rightChild
      expect(packed[base + 0]).toBe(nodes[i]!.emitterIndex);
      expect(packed[base + 1]).toBeCloseTo(nodes[i]!.totalPower, 5);
      expect(packed[base + 2]).toBe(nodes[i]!.leftChild);
      expect(packed[base + 3]).toBe(nodes[i]!.rightChild);
    }
  });
});

// ===========================================================================
// V22 — kernel/tree directional-slot ALIGNMENT (GPU-surfaced bias fix, 2026-05-29)
// ===========================================================================
describe('V22 — light-tree directional slot mirrors the kernel NEE gate exactly', () => {
  it('no directional emitter produces no packed directional record', () => {
    const packed = packEmitterArrays(emptyScene());
    expect(packed.directionalLightCount).toBe(0);
    expect(packed.directionalLightsData.length).toBe(0);
  });

  it('a directional-less multi-light scene builds NO directional leaf (leaf count = real lights only)', () => {
    // 3 point lights, no directional, no env → exactly 3 leaves. Pre-fix, the tree
    // omitted a directional leaf the kernel STILL shaded (phantom), shifting every
    // point light's emitterIndex one slot off the kernel walk → a biased pick.
    const scene: Scene = {
      ...emptyScene(),
      emitters: [
        { id: 'p0', kind: 'point', position: [0, 5, 0], color: [1, 1, 1], intensity: 2 },
        { id: 'p1', kind: 'point', position: [4, 5, 0], color: [1, 1, 1], intensity: 5 },
        { id: 'p2', kind: 'point', position: [-4, 5, 0], color: [1, 1, 1], intensity: 1 },
      ],
    } as unknown as Scene;
    const input = buildLightTreeInputForScene(scene);
    expect(input.powers.length).toBe(3);
    // Leaf 0 must be the FIRST point light (kernel slot 0 when no directional),
    // not a phantom directional — its power tracks p0's luminance, not [1,1,1].
    expect(input.powers[0]).toBeCloseTo(luminance(2, 2, 2), 5);
  });

  it('a REAL directional emitter ⇒ irradiance = color·intensity and a directional leaf at index 0', () => {
    const scene: Scene = {
      ...emptyScene(),
      emitters: [
        { id: 'd0', kind: 'directional', direction: [0, -1, 0], color: [1, 0.5, 0.25], intensity: 3 },
        { id: 'p0', kind: 'point', position: [0, 5, 0], color: [1, 1, 1], intensity: 2 },
      ],
    } as unknown as Scene;
    const packed = packEmitterArrays(scene);
    expect(Array.from(packed.directionalLightsData.slice(4, 7))).toEqual([3, 1.5, 0.75]);
    const input = buildLightTreeInputForScene(scene);
    // directional (slot 0) + 1 point = 2 leaves; leaf 0 is the directional.
    expect(input.powers.length).toBe(2);
    expect(input.powers[0]).toBeCloseTo(luminance(3, 1.5, 0.75), 5);
  });
});

// ===========================================================================
// 4. + 5. Unbiasedness + variance reduction (single-sample NEE estimator)
// ===========================================================================
describe('WS2 — power-weighted NEE estimator is unbiased + lower-variance', () => {
  // 2-light scene with a 10:1 power spread (≥5×). Both lights are equally far
  // from the shading point so the proximity term is symmetric; the selection is
  // then dominated by POWER, which is exactly the variance-reduction regime.
  const x: [number, number, number] = [0, 0, 0];

  // Light "contribution" if selected = its (unweighted) NEE term. We use a
  // simple positive scalar per light proportional to its power (the ground-truth
  // single-light contribution); the estimator must average to Σ contrib.
  const contrib = [1.0, 10.0]; // 10:1 spread
  const groundTruthSum = contrib[0]! + contrib[1]!;

  // Tree built so that selection ∝ power (equal distances ⇒ proximity cancels).
  const input = {
    powers: [1.0, 10.0],
    centroids: [
      [-3, 0, 4] as [number, number, number],
      [3, 0, 4] as [number, number, number],
    ],
    aabbs: [
      { min: [-3, 0, 4] as [number, number, number], max: [-3, 0, 4] as [number, number, number] },
      { min: [3, 0, 4] as [number, number, number], max: [3, 0, 4] as [number, number, number] },
    ],
  };
  const { nodes } = buildLightTree(input);
  const N = 200000;
  const dist2Floor = 1e-3;

  function runEstimator(selectIdx: () => { idx: number; pdf: number }): {
    mean: number;
    variance: number;
  } {
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < N; i++) {
      const { idx, pdf } = selectIdx();
      const est = pdf > 0 ? contrib[idx]! / pdf : 0;
      sum += est;
      sumSq += est * est;
    }
    const mean = sum / N;
    const variance = sumSq / N - mean * mean;
    return { mean, variance };
  }

  it('uniform-pick estimator is unbiased (E ≈ Σ contrib)', () => {
    const rng = makeRng(0x1234);
    const uniform = runEstimator(() => {
      const idx = Math.min(1, Math.floor(rng() * 2));
      return { idx, pdf: 0.5 };
    });
    expect(uniform.mean).toBeCloseTo(groundTruthSum, 0.6);
    expect(Math.abs(uniform.mean - groundTruthSum) / groundTruthSum).toBeLessThan(0.02);
  });

  it('power-weighted estimator is unbiased and matches the uniform estimator in the mean', () => {
    const rng = makeRng(0x1234);
    const uniformRng = makeRng(0x99);
    const power = runEstimator(() => {
      const s = sampleLightTreeCPU(nodes, x, dist2Floor, rng);
      return { idx: s.emitterIndex, pdf: s.pdf };
    });
    const uniform = runEstimator(() => {
      const idx = Math.min(1, Math.floor(uniformRng() * 2));
      return { idx, pdf: 0.5 };
    });
    // Both unbiased → equal in the mean (within MC noise).
    expect(Math.abs(power.mean - groundTruthSum) / groundTruthSum).toBeLessThan(0.02);
    expect(Math.abs(power.mean - uniform.mean) / groundTruthSum).toBeLessThan(0.03);
  });

  it('variance reduction: var_power < 0.8 · var_uniform for a 10:1 power spread', () => {
    const rng = makeRng(0x1234);
    const uniformRng = makeRng(0x99);
    const power = runEstimator(() => {
      const s = sampleLightTreeCPU(nodes, x, dist2Floor, rng);
      return { idx: s.emitterIndex, pdf: s.pdf };
    });
    const uniform = runEstimator(() => {
      const idx = Math.min(1, Math.floor(uniformRng() * 2));
      return { idx, pdf: 0.5 };
    });
    expect(power.variance).toBeLessThan(0.8 * uniform.variance);
  });
});
