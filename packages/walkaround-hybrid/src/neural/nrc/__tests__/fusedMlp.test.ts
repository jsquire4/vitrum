// fusedMlp.test.ts — CPU-only unit tests for the fused NRC MLP module.
//
// These pin the host-side INVARIANTS that the fused kernel relies on: the
// concatenated-param layer plan, the f16 codec round-trip (mixed-precision
// weight downcast), the WGSL codegen shape (f16 enable, binding count, packed
// per-layer offsets), and the shared-memory budget arithmetic.
//
// The GPU NUMERICAL correctness (FD gradient check + learning test) is proven
// separately by the lavapipe Deno harness (fusedMlpHarness.ts), which needs a
// real WebGPU adapter that vitest's node env does not provide. This file is the
// always-on guard against layout/codegen regressions.

import { describe, it, expect } from 'vitest';
import {
  planLayers,
  resolveActiveSampleWindow,
  f32ToF16Bits,
  f16BitsToF32,
} from '../fusedMlpTrainer.ts';
import {
  fusedForwardWgsl, fusedBackwardWgsl, gradFinalizeWgsl, downcastF16Wgsl,
  fusedMlpWorkgroupStorageBytes,
  type FusedMlpWgslOptions,
} from '../wgsl/fusedMlp.wgsl.ts';

const MULLER: FusedMlpWgslOptions = { useF16: true, W: 64, OUT_W: 3, HIDDEN: 6, TILE_B: 64 };

describe('fused NRC MLP — layer plan', () => {
  it('lays out the Müller 6×64 core (input padded to W) as concatenated params', () => {
    const plan = planLayers({ inW: 2, W: 64, outW: 3, hidden: 6 });
    // weight layers = hidden + 1 = 7. node-layers (widths) = [64, 64×6, 3].
    expect(plan.wlayers).toBe(7);
    // first 6 layers are 64->64, last is 64->3.
    expect(plan.inW).toEqual([64, 64, 64, 64, 64, 64, 64]);
    expect(plan.outW).toEqual([64, 64, 64, 64, 64, 64, 3]);
    // total weights: 6 × (64×64) + 64×3 = 24576 + 192 = 24768.
    expect(plan.totalW).toBe(6 * 64 * 64 + 64 * 3);
    expect(plan.totalW).toBe(24768);
    // total biases: 6×64 + 3 = 387.
    expect(plan.totalB).toBe(6 * 64 + 3);
    expect(plan.totalB).toBe(387);
    // offsets are the running cumulative sums.
    expect(plan.wOff[0]).toBe(0);
    expect(plan.wOff[1]).toBe(64 * 64);
    expect(plan.bOff[1]).toBe(64);
  });

  it('handles a tiny net (FD-check sizing) correctly', () => {
    const plan = planLayers({ inW: 4, W: 8, outW: 3, hidden: 2 });
    expect(plan.wlayers).toBe(3);
    expect(plan.inW).toEqual([8, 8, 8]);   // input padded 4->8
    expect(plan.outW).toEqual([8, 8, 3]);
    expect(plan.totalW).toBe(8 * 8 + 8 * 8 + 8 * 3);
    expect(plan.totalB).toBe(8 + 8 + 3); // 19 — an ODD count (alignment edge case)
  });
});

describe('fused NRC MLP — active train window', () => {
  it('dispatches only filled dense records while preserving full-batch default', () => {
    expect(resolveActiveSampleWindow(4096, 32)).toEqual({ samples: 4096, tiles: 128 });
    expect(resolveActiveSampleWindow(4096, 32, 33)).toEqual({ samples: 33, tiles: 2 });
    expect(resolveActiveSampleWindow(4096, 32, 32)).toEqual({ samples: 32, tiles: 1 });
    expect(resolveActiveSampleWindow(4096, 32, 0)).toEqual({ samples: 0, tiles: 0 });
  });

  it('clamps invalid active counts to the allocated record capacity', () => {
    expect(resolveActiveSampleWindow(128, 32, 999)).toEqual({ samples: 128, tiles: 4 });
    expect(resolveActiveSampleWindow(128, 32, -5)).toEqual({ samples: 0, tiles: 0 });
    expect(resolveActiveSampleWindow(128, 32, Number.NaN)).toEqual({ samples: 128, tiles: 4 });
  });
});

describe('fused NRC MLP — f16 codec (mixed precision)', () => {
  it('round-trips representative weight magnitudes within f16 precision', () => {
    const src = new Float32Array([0, 1, -1, 0.5, -0.5, 0.1, -0.123, 2.5, 0.001, -0.0078125]);
    const back = f16BitsToF32(f32ToF16Bits(src));
    for (let i = 0; i < src.length; i++) {
      // f16 has ~3 decimal digits; relative error <= 2^-10 ≈ 1e-3 for normals.
      const tol = Math.max(1e-3, Math.abs(src[i]!) * 1e-3);
      expect(Math.abs(back[i]! - src[i]!)).toBeLessThanOrEqual(tol + 1e-7);
    }
  });

  it('encodes exact powers of two losslessly', () => {
    const src = new Float32Array([1, 2, 4, 0.5, 0.25, 0.125]);
    const back = f16BitsToF32(f32ToF16Bits(src));
    for (let i = 0; i < src.length; i++) expect(back[i]).toBe(src[i]);
  });

  it('preserves sign of negative zero-adjacent values', () => {
    const src = new Float32Array([-0.7, 0.7]);
    const back = f16BitsToF32(f32ToF16Bits(src));
    expect(back[0]! < 0).toBe(true);
    expect(back[1]! > 0).toBe(true);
  });
});

describe('fused NRC MLP — WGSL codegen', () => {
  it('emits enable f16 only on the f16 path', () => {
    expect(fusedForwardWgsl({ ...MULLER, useF16: true })).toContain('enable f16;');
    expect(fusedForwardWgsl({ ...MULLER, useF16: false })).not.toContain('enable f16;');
    expect(fusedBackwardWgsl({ ...MULLER, useF16: true })).toContain('enable f16;');
    expect(fusedBackwardWgsl({ ...MULLER, useF16: false })).not.toContain('enable f16;');
  });

  it('uses f16 scalar types on the f16 path and f32 otherwise', () => {
    const f16 = fusedForwardWgsl({ ...MULLER, useF16: true });
    const f32 = fusedForwardWgsl({ ...MULLER, useF16: false });
    expect(f16).toContain('array<f16>');
    expect(f16).toContain('var<workgroup> actA : array<f16,');
    expect(f32).toContain('array<f32>');
    expect(f32).toContain('var<workgroup> actA : array<f32,');
  });

  it('keeps the storage-buffer binding count <= 8 (portable to default adapters)', () => {
    // Count @binding(...) entries declared as var<storage,...>; the packed
    // per-layer offsets live in the uniform so we stay under the WebGPU default.
    const countStorage = (src: string) =>
      (src.match(/var<storage[^>]*>/g) ?? []).length;
    expect(countStorage(fusedForwardWgsl(MULLER))).toBeLessThanOrEqual(8);
    expect(countStorage(fusedBackwardWgsl(MULLER))).toBeLessThanOrEqual(8);
    // forward has exactly 5 storage buffers (weights, biases, inputs, acts, z).
    expect(countStorage(fusedForwardWgsl(MULLER))).toBe(5);
    // backward has 7 (weights, targets, acts, z, gradWfx, gradBfx, gradInputFx).
    expect(countStorage(fusedBackwardWgsl(MULLER))).toBe(8);
  });

  it('packs per-layer offsets as a fixed-size uniform vec4 array sized to WLAYERS', () => {
    const src = fusedForwardWgsl(MULLER);
    // WLAYERS = HIDDEN+1 = 7.
    expect(src).toContain('array<vec4<u32>, 7>');
  });

  it('sizes the resident workgroup tiles to TILE_B × W', () => {
    const src = fusedForwardWgsl(MULLER); // 64 × 64 = 4096
    expect(src).toContain('var<workgroup> actA : array<f16, 4096>');
    expect(src).toContain('var<workgroup> actB : array<f16, 4096>');
  });

  it('uses workgroup_size = W (one invocation per neuron column)', () => {
    expect(fusedForwardWgsl(MULLER)).toContain('@workgroup_size(64, 1, 1)');
    expect(fusedBackwardWgsl(MULLER)).toContain('@workgroup_size(64, 1, 1)');
  });

  it('backward uses i32 fixed-point grad atomics (no f32 atomics in core WGSL)', () => {
    const src = fusedBackwardWgsl(MULLER);
    expect(src).toContain('array<atomic<i32>>');
    expect(src).toContain('atomicAdd');
    expect(src).toContain('GRAD_FP');
  });

  it('grad finalize clears the fixed-point buffer (atomicExchange to 0)', () => {
    expect(gradFinalizeWgsl()).toContain('atomicExchange');
    expect(downcastF16Wgsl()).toContain('f16(src[idx])');
  });
});

describe('fused NRC MLP — shared-memory budget', () => {
  const sharedBytes = (tileB: number, W: number, scBytes: number) => 2 * tileB * W * scBytes;

  it('the f16 Müller tile has the two workgroup arrays emitted by the shader', () => {
    expect(sharedBytes(64, 64, 2)).toBe(16_384);
    expect(fusedMlpWorkgroupStorageBytes(MULLER)).toBe(16_384);
    expect(sharedBytes(64, 64, 2)).toBeLessThanOrEqual(32768);
  });

  it('the default f32 TILE_B=32 path exactly fits the 16 KB guaranteed floor', () => {
    expect(sharedBytes(64, 64, 4)).toBe(32_768);
    expect(sharedBytes(32, 64, 4)).toBe(16_384);
    expect(fusedMlpWorkgroupStorageBytes({ ...MULLER, useF16: false, TILE_B: 32 })).toBe(16_384);
  });
});
