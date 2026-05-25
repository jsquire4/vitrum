/**
 * neuralAcceptance.test.ts — W10 acceptance: `denoiser: 'neural'` is a real,
 * selectable mode.
 *
 * Two layers of coverage:
 *
 *   1. Wiring tests (always run):
 *      - Constructing `HybridEngine({ denoiser: 'neural', neuralWeights })`
 *        with a non-empty `ModelWeights` does NOT throw at the validation
 *        gate (proves the engine accepts neural as a real selection now,
 *        not a placeholder).
 *      - Constructing `HybridEngine({ denoiser: 'neural' })` WITHOUT
 *        weights throws a clear "missing weights" error with a pointer to
 *        the training docs (proves the contract that neural requires a
 *        model checkpoint is enforced).
 *      - `WALKAROUND_DENOISER_UNET_SPEC` + `buildRandomWeightsForSpec`
 *        produce a `ModelWeights` whose per-layer weight + bias counts match
 *        the architecture spec's documented OIKW/IOKW arithmetic — proving
 *        that callers can construct valid weights deterministically without
 *        a trained checkpoint (the smoke-test path the W10 example uses).
 *
 *   2. GPU acceptance test (skipped unless `VITRUM_NEURAL_ACCEPTANCE=1`):
 *      - Reads harness-produced metrics from `VITRUM_NEURAL_ACCEPTANCE_METRICS`
 *        and asserts neural variance is at least 20% lower per channel than
 *        atrous-variance on the benchmark ROI.
 *
 *      This test is intentionally environment-gated because Node test
 *      environments do not expose a real `GPUDevice`, and the acceptance
 *      metric requires trained weights + a benchmark capture harness.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { readFileSync } from 'node:fs';
import { buildRandomWeightsForSpec } from '../src/neural/weights.js';
import type { ModelWeights } from '../src/neural/weights.js';
import { WALKAROUND_DENOISER_UNET_SPEC } from '../src/neural/unetArchitecture.js';

// ─── Mock GPUDevice for construction-only tests ─────────────────────────────

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(() => ({})),
    createBuffer:         vi.fn(() => ({ destroy: vi.fn() })),
    createShaderModule:   vi.fn(() => ({})),
    createComputePipeline:vi.fn(() => ({})),
    createBindGroupLayout:vi.fn(() => ({})),
    createBindGroup:      vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    queue:                { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeThreeScene(): THREE.Scene {
  const scene = new THREE.Scene();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  scene.add(new THREE.Mesh(geom, new THREE.MeshBasicMaterial()));
  return scene;
}

// ─── Wiring tests (always run) ──────────────────────────────────────────────

describe('W10 — neural denoiser wiring', () => {
  it('HybridEngine constructor accepts denoiser: "neural" when neuralWeights is provided', async () => {
    const { HybridEngine } = await import('../src/HybridEngine.js');
    const weights = buildRandomWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC, 42);

    // Construction must NOT throw — this is the contract that 'neural' is a
    // real, selectable mode after W10 (it would have thrown pre-W10 because
    // the validator only allows the enabled set).
    expect(() => new HybridEngine({
      device:               makeMockDevice(),
      width:                64,
      height:               64,
      primaryLightDir:      [0, -1, 0],
      primaryLightIntensity:1,
      skyTint:              [0.2, 0.4, 0.8],
      skyIrradiance:        0.5,
      threeScene:           makeThreeScene(),
      denoiser:             'neural',
      neuralWeights:        weights,
    })).not.toThrow();
  });

  it('HybridEngine constructor throws when denoiser: "neural" is selected without weights', async () => {
    const { HybridEngine } = await import('../src/HybridEngine.js');

    expect(() => new HybridEngine({
      device:               makeMockDevice(),
      width:                64,
      height:               64,
      primaryLightDir:      [0, -1, 0],
      primaryLightIntensity:1,
      skyTint:              [0.2, 0.4, 0.8],
      skyIrradiance:        0.5,
      threeScene:           makeThreeScene(),
      denoiser:             'neural',
      // neuralWeights deliberately omitted — must throw.
    })).toThrow(/neural.*weights|neuralWeights.*required/i);
  });

  it('HybridEngine error message for missing weights points to the training docs', async () => {
    const { HybridEngine } = await import('../src/HybridEngine.js');
    let err: unknown;
    try {
      new HybridEngine({
        device:               makeMockDevice(),
        width:                64,
        height:               64,
        primaryLightDir:      [0, -1, 0],
        primaryLightIntensity:1,
        skyTint:              [0.2, 0.4, 0.8],
        skyIrradiance:        0.5,
        threeScene:           makeThreeScene(),
        denoiser:             'neural',
      });
    } catch (e) { err = e; }
    expect(err).toBeInstanceOf(Error);
    // The contract says the error should reference loadWeightsFromArrayBuffer
    // OR the training tooling so callers know how to recover.
    expect(String((err as Error).message)).toMatch(/loadWeightsFromArrayBuffer|train\.py|training/i);
  });
});

// ─── Weights helper: shape integrity ─────────────────────────────────────────

describe('W10 — buildRandomWeightsForSpec produces architecture-shaped weights', () => {
  it('produces one LayerWeights entry per architecture layer in order', () => {
    const spec = WALKAROUND_DENOISER_UNET_SPEC;
    const weights = buildRandomWeightsForSpec(spec, 42);
    expect(weights.layers.length).toBe(spec.layers.length);
    for (let i = 0; i < spec.layers.length; i++) {
      expect(weights.layers[i]!.name).toBe(spec.layers[i]!.name);
    }
  });

  it('parameterised layers have weight count == inC × outC × kH × kW (OIKW / IOKW)', () => {
    const spec = WALKAROUND_DENOISER_UNET_SPEC;
    const weights = buildRandomWeightsForSpec(spec, 42);

    let paramSum = 0;
    for (let i = 0; i < spec.layers.length; i++) {
      const layer = spec.layers[i]!;
      const lw = weights.layers[i]!;
      if (layer.weightLayout === 'none') {
        expect(lw.weights.length).toBe(0);
        expect(lw.biases.length).toBe(0);
        continue;
      }
      const expectedW = layer.params.inC * layer.params.outC * (layer.params.kH ?? 1) * (layer.params.kW ?? 1);
      const expectedB = layer.params.outC;
      expect(lw.weights.length).toBe(expectedW);
      expect(lw.biases.length).toBe(expectedB);
      paramSum += expectedW + expectedB;
    }
    // Sanity: total param count should be in the documented 1-3 MB range
    // (the spec.paramCount is a tracked target; rough integer agreement is fine).
    expect(paramSum).toBeGreaterThan(400_000);
    expect(paramSum).toBeLessThan(1_000_000);
  });

  it('weights are deterministic across calls with the same seed', () => {
    const a = buildRandomWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC, 12345);
    const b = buildRandomWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC, 12345);
    // Spot-check a parameterised layer — full bit-equal would be slow.
    const idx = a.layers.findIndex((l) => l.weights.length > 0);
    expect(idx).toBeGreaterThanOrEqual(0);
    const aw = a.layers[idx]!.weights;
    const bw = b.layers[idx]!.weights;
    expect(aw.length).toBe(bw.length);
    for (let i = 0; i < aw.length; i++) {
      if (aw[i] !== bw[i]) {
        throw new Error(`deterministic seed broken at layer '${a.layers[idx]!.name}' index ${i}`);
      }
    }
  });

  it('weights are finite and bounded (He init ⇒ |w| ≤ sqrt(2) for inC=1, kH=kW=1)', () => {
    const weights = buildRandomWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC, 7);
    for (const lw of weights.layers) {
      for (const v of lw.weights) {
        expect(Number.isFinite(v)).toBe(true);
        expect(Math.abs(v)).toBeLessThan(2.0);  // generous bound for the worst-case fan-in
      }
      for (const v of lw.biases) {
        expect(Number.isFinite(v)).toBe(true);
        expect(Math.abs(v)).toBeLessThan(0.011);
      }
    }
  });
});

// ─── GPU acceptance (gated; only runs with VITRUM_NEURAL_ACCEPTANCE=1) ───────

const GPU_ACCEPTANCE_ENABLED =
  typeof process !== 'undefined' &&
  process.env != null &&
  process.env['VITRUM_NEURAL_ACCEPTANCE'] === '1';

describe.skipIf(!GPU_ACCEPTANCE_ENABLED)('W10 — neural denoiser GPU acceptance (numerical noise comparison)', () => {
  function readMetrics(): {
    readonly atrousVariance: readonly [number, number, number];
    readonly neuralVariance: readonly [number, number, number];
  } {
    const path = process.env['VITRUM_NEURAL_ACCEPTANCE_METRICS'];
    if (!path) {
      throw new Error(
        'VITRUM_NEURAL_ACCEPTANCE=1 requires VITRUM_NEURAL_ACCEPTANCE_METRICS=<json file> ' +
        'produced by tools/benchmark-runner.',
      );
    }
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as {
      atrousVariance?: unknown;
      neuralVariance?: unknown;
    };
    const toTriplet = (value: unknown, name: string): [number, number, number] => {
      if (
        !Array.isArray(value) ||
        value.length !== 3 ||
        value.some((v) => typeof v !== 'number' || !Number.isFinite(v))
      ) {
        throw new Error(`Invalid ${name} in ${path}; expected [number, number, number].`);
      }
      return [value[0] as number, value[1] as number, value[2] as number];
    };
    return {
      atrousVariance: toTriplet(parsed.atrousVariance, 'atrousVariance'),
      neuralVariance: toTriplet(parsed.neuralVariance, 'neuralVariance'),
    };
  }

  it('on a noisy Cornell-box render, neural denoiser reduces flat-region variance vs atrous-variance', async () => {
    const metrics = readMetrics();
    for (let i = 0; i < 3; i += 1) {
      expect(metrics.neuralVariance[i]!).toBeLessThan(metrics.atrousVariance[i]! * 0.8);
    }
  });
});

// Type-only re-export to keep the import line non-dead for the editor.
const _ModelWeightsTypeOnly: ModelWeights | undefined = undefined;
void _ModelWeightsTypeOnly;
