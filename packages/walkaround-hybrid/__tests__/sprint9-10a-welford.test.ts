/**
 * Sprint 9 (Phase 6) — Welford variance struct, sample-budget shader,
 * resolve shader, and variance buffer allocation tests.
 *
 * Test strategy: structural/string-content validation only — no real WebGPU
 * device required. The same "defensive structural" pattern used in
 * rc-bindings.test.ts and sprint2-cellPower.test.ts.
 *
 * Coverage:
 *   1. COMMON_WGSL — WelfordVariance struct + helpers present, layout comment
 *      references RG32Float (8 bytes/texel).
 *   1a. CPU oracle — Welford update/variance and sample-budget tier semantics
 *      checked against independent two-pass sample-variance vectors.
 *   2. SAMPLE_BUDGET_WGSL — entry point present, required bindings declared,
 *      sampleTierFromVariance function present, threshold params bound.
 *   3. RESOLVE_WGSL — entry point present, required bindings declared,
 *      isShadedPixel and clampCoord helpers present, motion-vector fallback
 *      comment present.
 *   4. createVarianceBuffer — mock GPUDevice, asserts format + usage flags.
 *   5. FrameResources type — varianceBuffer field declared (import check).
 */

import { describe, it, expect, vi } from 'vitest';
import { COMMON_WGSL } from '../src/shaders/common.wgsl.js';
import { SAMPLE_BUDGET_WGSL } from '../src/shaders/sampleBudget.wgsl.js';
import { RESOLVE_WGSL } from '../src/shaders/resolve.wgsl.js';
import { SCREEN_COORD_HELPERS_WGSL } from '../src/shaders/screenCoordHelpers.wgsl.js';
import { WELFORD_TEMPORAL_WGSL } from '../src/shaders/welfordTemporal.wgsl.js';
import { createVarianceBuffer, createFrameResources } from '../src/pipeline/resourceManager.js';

// WebGPU global polyfills for the Node test environment — see helpers file.
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
installWebGPUPolyfills();

type WelfordState = {
  mean: number;
  m2: number;
};

function updateWelfordReference(prev: WelfordState, sample: number, n: number): WelfordState {
  const delta = sample - prev.mean;
  const mean = prev.mean + delta / n;
  const m2 = prev.m2 + delta * (sample - mean);
  return { mean, m2 };
}

function runWelfordReference(samples: readonly number[]): WelfordState {
  let state: WelfordState = { mean: 0, m2: 0 };
  for (const [idx, sample] of samples.entries()) {
    state = updateWelfordReference(state, sample, idx + 1);
  }
  return state;
}

function directSampleVariance(samples: readonly number[]): number {
  if (samples.length < 2) return 0;
  const mean = samples.reduce((sum, sample) => sum + sample, 0) / samples.length;
  const m2 = samples.reduce((sum, sample) => sum + (sample - mean) ** 2, 0);
  return m2 / (samples.length - 1);
}

function welfordVarianceReference(state: WelfordState, n: number): number {
  if (n < 2) return 0;
  return state.m2 / (n - 1);
}

function sampleTierFromVarianceReference(v: number, thresholdLow: number, thresholdHigh: number): 1 | 2 | 4 {
  if (v < thresholdLow) return 1;
  if (v < thresholdHigh) return 2;
  return 4;
}

function sampleBudgetTierReference(samples: readonly number[], thresholdLow: number, thresholdHigh: number): 1 | 2 | 4 {
  const n = samples.length;
  const state = runWelfordReference(samples);
  const variance = welfordVarianceReference(state, n);
  return n < 2 ? 4 : sampleTierFromVarianceReference(variance, thresholdLow, thresholdHigh);
}

// ─── 1. COMMON_WGSL — WelfordVariance struct ─────────────────────────────────

describe('COMMON_WGSL — WelfordVariance struct (Sprint 9 / Decision 13)', () => {
  it('contains the WelfordVariance struct declaration', () => {
    expect(COMMON_WGSL).toContain('struct WelfordVariance');
  });

  it('WelfordVariance has mean field (f32)', () => {
    expect(COMMON_WGSL).toContain('mean: f32');
  });

  it('WelfordVariance has m2 field (f32)', () => {
    expect(COMMON_WGSL).toContain('m2:   f32');
  });

  it('contains welfordUpdate function', () => {
    expect(COMMON_WGSL).toContain('fn welfordUpdate');
  });

  it('welfordUpdate computes delta = sample - prev.mean', () => {
    expect(COMMON_WGSL).toContain('let delta = sample - prev.mean');
  });

  it('welfordUpdate updates mean via delta / f32(n)', () => {
    expect(COMMON_WGSL).toContain('prev.mean + delta / f32(n)');
  });

  it('welfordUpdate accumulates M2 correctly', () => {
    expect(COMMON_WGSL).toContain('prev.m2   + delta * (sample - mean)');
  });

  it('welfordUpdate returns WelfordVariance struct', () => {
    expect(COMMON_WGSL).toContain('return WelfordVariance(mean, m2)');
  });

  it('contains welfordVariance function', () => {
    expect(COMMON_WGSL).toContain('fn welfordVariance');
  });

  it('welfordVariance returns 0 for n < 2 (degenerate guard)', () => {
    expect(COMMON_WGSL).toContain('if (n < 2u) { return 0.0; }');
  });

  it('welfordVariance divides M2 by (n - 1) — unbiased estimator', () => {
    expect(COMMON_WGSL).toContain('state.m2 / f32(n - 1u)');
  });

  it('contains Decision 13 version marker comment', () => {
    expect(COMMON_WGSL).toContain('Decision 13');
  });

  it('contains RG32Float layout documentation', () => {
    // The layout comment must reference RG32Float so future sprints know the texel format.
    expect(COMMON_WGSL).toContain('RG32Float');
  });

  it('layout comment specifies r = mean', () => {
    expect(COMMON_WGSL).toContain('r = mean');
  });

  it('layout comment specifies g = M2', () => {
    expect(COMMON_WGSL).toContain('g = M2');
  });

  it('RG32Float texel is 8 bytes: 2 × f32 × 4 bytes/f32', () => {
    // Structural assertion: WelfordVariance has exactly 2 f32 fields.
    // This is verified by counting f32 fields in the struct — if more are added,
    // the RG32Float layout breaks and Sprint 10a bindings will be wrong.
    // Use \s+ to handle alignment spaces (e.g. "m2:   f32" vs "mean: f32").
    const structMatch = COMMON_WGSL.match(/struct WelfordVariance \{[^}]+\}/);
    expect(structMatch).not.toBeNull();
    const structBody = structMatch![0]!;
    const f32Fields = (structBody.match(/:\s+f32/g) ?? []).length;
    // Exactly 2 f32 fields = exactly 8 bytes = RG32Float.
    expect(f32Fields).toBe(2);
  });
});

// ─── 1a. Welford/sample-budget behavior oracle ───────────────────────────────

describe('Welford/sample-budget behavior oracle', () => {
  it('matches an independent two-pass unbiased variance calculation', () => {
    const samples = [0.25, 1.0, 0.5, 2.0, 1.25] as const;
    const state = runWelfordReference(samples);
    const directMean = samples.reduce((sum, sample) => sum + sample, 0) / samples.length;

    expect(state.mean).toBeCloseTo(directMean, 12);
    expect(welfordVarianceReference(state, samples.length)).toBeCloseTo(directSampleVariance(samples), 12);
  });

  it('keeps the unbiased variance degenerate guard at zero for n < 2', () => {
    expect(welfordVarianceReference({ mean: 0, m2: 999 }, 0)).toBe(0);
    expect(welfordVarianceReference({ mean: 0.75, m2: 999 }, 1)).toBe(0);
  });

  it('preserves sample-tier threshold edges and first-frame low-confidence override', () => {
    const thresholdLow = 0.01;
    const thresholdHigh = 0.10;

    expect(sampleBudgetTierReference([], thresholdLow, thresholdHigh)).toBe(4);
    expect(sampleBudgetTierReference([0.5], thresholdLow, thresholdHigh)).toBe(4);
    expect(sampleBudgetTierReference([1.0, 1.02, 1.01], thresholdLow, thresholdHigh)).toBe(1);
    expect(sampleBudgetTierReference([0.9, 1.1], thresholdLow, thresholdHigh)).toBe(2);
    expect(sampleBudgetTierReference([0.0, 1.0], thresholdLow, thresholdHigh)).toBe(4);

    expect(sampleTierFromVarianceReference(thresholdLow, thresholdLow, thresholdHigh)).toBe(2);
    expect(sampleTierFromVarianceReference(thresholdHigh, thresholdLow, thresholdHigh)).toBe(4);
  });
});

// ─── 1b. WELFORD_TEMPORAL_WGSL (Sprint 10a host integration) ─────────────────

describe('WELFORD_TEMPORAL_WGSL — luminance Welford accumulation for atrous-variance', () => {
  it('contains welfordTemporalMain entry point', () => {
    expect(WELFORD_TEMPORAL_WGSL).toContain('fn welfordTemporalMain');
  });

  it('calls welfordUpdate on non-reset path', () => {
    expect(WELFORD_TEMPORAL_WGSL).toContain('welfordUpdate(');
  });
});

// ─── 2. SAMPLE_BUDGET_WGSL ───────────────────────────────────────────────────

describe('SAMPLE_BUDGET_WGSL — variance-driven dispatch tier shader', () => {
  it('is a non-empty string', () => {
    expect(typeof SAMPLE_BUDGET_WGSL).toBe('string');
    expect(SAMPLE_BUDGET_WGSL.length).toBeGreaterThan(0);
  });

  it('contains sampleBudgetKernel entry point', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('fn sampleBudgetKernel');
  });

  it('entry point is @compute @workgroup_size(8, 8, 1)', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('@compute @workgroup_size(8, 8, 1)');
    expect(SAMPLE_BUDGET_WGSL).toContain('fn sampleBudgetKernel');
  });

  it('uses @builtin(global_invocation_id) for pixel dispatch', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('@builtin(global_invocation_id)');
  });

  it('declares SampleBudgetUniforms struct with threshold_low and threshold_high', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('struct SampleBudgetUniforms');
    expect(SAMPLE_BUDGET_WGSL).toContain('threshold_low');
    expect(SAMPLE_BUDGET_WGSL).toContain('threshold_high');
  });

  it('binds SampleBudgetUniforms at @group(0) @binding(0)', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('@group(0) @binding(0)');
    expect(SAMPLE_BUDGET_WGSL).toContain('u_budget');
  });

  it('binds variance storage texture at @group(0) @binding(1)', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('@group(0) @binding(1)');
    expect(SAMPLE_BUDGET_WGSL).toContain('t_variance');
  });

  it('binds tier output texture at @group(0) @binding(2)', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('@group(0) @binding(2)');
    expect(SAMPLE_BUDGET_WGSL).toContain('t_tier_out');
  });

  it('binds sample count uniform at @group(0) @binding(3)', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('@group(0) @binding(3)');
    expect(SAMPLE_BUDGET_WGSL).toContain('u_sampleCount');
  });

  it('contains sampleTierFromVariance helper function', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('fn sampleTierFromVariance');
  });

  it('sampleTierFromVariance returns 1u for low variance', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('return 1u');
  });

  it('sampleTierFromVariance returns 2u for medium variance', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('return 2u');
  });

  it('sampleTierFromVariance returns 4u for high variance', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('return 4u');
  });

  it('forces first-frame pixels to low-confidence tier 4 before variance is meaningful', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('n < 2u');
    expect(SAMPLE_BUDGET_WGSL).toMatch(/select\(\s*sampleTierFromVariance[\s\S]*?4u[\s\S]*?n\s*<\s*2u/);
  });

  it('writes tier to t_tier_out via textureStore', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('textureStore(t_tier_out');
  });

  it('reads WelfordVariance from t_variance via textureLoad', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('textureLoad(t_variance');
  });

  it('calls welfordVariance to compute variance from Welford state', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('welfordVariance(');
  });

  it('guards against out-of-bounds invocations', () => {
    expect(SAMPLE_BUDGET_WGSL).toContain('u_budget.screenWidth');
    expect(SAMPLE_BUDGET_WGSL).toContain('u_budget.screenHeight');
  });
});

// ─── 3. RESOLVE_WGSL ─────────────────────────────────────────────────────────

describe('RESOLVE_WGSL — checkerboard upsampling + temporal reprojection', () => {
  it('is a non-empty string', () => {
    expect(typeof RESOLVE_WGSL).toBe('string');
    expect(RESOLVE_WGSL.length).toBeGreaterThan(0);
  });

  it('contains resolveKernel entry point', () => {
    expect(RESOLVE_WGSL).toContain('fn resolveKernel');
  });

  it('entry point is @compute @workgroup_size(8, 8, 1)', () => {
    expect(RESOLVE_WGSL).toContain('@compute @workgroup_size(8, 8, 1)');
    expect(RESOLVE_WGSL).toContain('fn resolveKernel');
  });

  it('declares ResolveUniforms struct with frameParity', () => {
    expect(RESOLVE_WGSL).toContain('struct ResolveUniforms');
    expect(RESOLVE_WGSL).toContain('frameParity');
  });

  it('binds ResolveUniforms at @group(0) @binding(0)', () => {
    expect(RESOLVE_WGSL).toContain('@group(0) @binding(0)');
    expect(RESOLVE_WGSL).toContain('u_resolve');
  });

  it('binds current-frame radiance at @group(0) @binding(1)', () => {
    expect(RESOLVE_WGSL).toContain('@group(0) @binding(1)');
    expect(RESOLVE_WGSL).toContain('t_current_radiance');
  });

  it('binds previous-frame radiance at @group(0) @binding(2)', () => {
    expect(RESOLVE_WGSL).toContain('@group(0) @binding(2)');
    expect(RESOLVE_WGSL).toContain('t_prev_radiance');
  });

  it('binds motion vector texture at @group(0) @binding(3)', () => {
    expect(RESOLVE_WGSL).toContain('@group(0) @binding(3)');
    expect(RESOLVE_WGSL).toContain('t_motion_vectors');
  });

  it('binds resolved output at @group(0) @binding(4)', () => {
    expect(RESOLVE_WGSL).toContain('@group(0) @binding(4)');
    expect(RESOLVE_WGSL).toContain('t_resolved_out');
  });

  it('contains isShadedPixel helper function', () => {
    expect(RESOLVE_WGSL).toContain('fn isShadedPixel');
  });

  it('isShadedPixel uses checkerboard (px + py) & 1 == frameParity', () => {
    expect(RESOLVE_WGSL).toContain('(px + py) & 1u') ;
    expect(RESOLVE_WGSL).toContain('frameParity');
  });

  it('contains clampCoord helper for safe texture access (D5.4: now in screenCoordHelpers)', () => {
    // D5.4 dedup: clampCoord moved to the shared screenCoordHelpers module.
    // Verify it exists there and that RESOLVE_WGSL references the shared module
    // via a comment (the function is no longer inline in RESOLVE_WGSL itself).
    expect(SCREEN_COORD_HELPERS_WGSL).toContain('fn clampCoord');
    expect(RESOLVE_WGSL).toContain('screenCoordHelpers');
  });

  it('reads motion vector directly via textureLoad on full-resolution texture', () => {
    // P3-C.1 wire-in changed motion-vector binding to a screen-sized
    // texture_2d<f32>; the old 1×1-fallback `readMotionVector` helper is gone
    // because the host now always allocates motionVectorTexture at full res.
    expect(RESOLVE_WGSL).toContain('textureLoad(t_motion_vectors');
  });

  it('checkerboardOn uniform gates the gap-fill branch (passthrough when 0)', () => {
    // P3-C.1: until shade.wgsl is upgraded to write sparsely, the resolve
    // pass runs in passthrough mode (checkerboardOn=0) — all pixels copy
    // through from current radiance. The flag is part of ResolveUniforms.
    expect(RESOLVE_WGSL).toContain('checkerboardOn');
    expect(RESOLVE_WGSL).toContain('if (checkerboardOn == 0u) { return true; }');
  });

  it('writes resolved radiance via textureStore', () => {
    expect(RESOLVE_WGSL).toContain('textureStore(t_resolved_out');
  });

  it('reads current-frame radiance for shaded pixels', () => {
    expect(RESOLVE_WGSL).toContain('textureLoad(t_current_radiance');
  });

  it('reads previous-frame radiance for gap pixels', () => {
    expect(RESOLVE_WGSL).toContain('textureLoad(t_prev_radiance');
  });

  it('consumes the producer-authored framebuffer-pixel motion without a second y flip', () => {
    // motionVectors.wgsl performs the NDC→y-down framebuffer conversion once.
    expect(RESOLVE_WGSL).toContain('let deltaPx = vec2i(round(mv));');
    expect(RESOLVE_WGSL).not.toContain('-mv.y');
  });

  it('guards against out-of-bounds invocations', () => {
    expect(RESOLVE_WGSL).toContain('u_resolve.screenWidth');
    expect(RESOLVE_WGSL).toContain('u_resolve.screenHeight');
  });
});

// ─── 4. createVarianceBuffer ─────────────────────────────────────────────────

describe('createVarianceBuffer — RG32Float storage texture factory', () => {
  /**
   * Build a minimal mock GPUDevice that records the last createTexture call.
   * We only need to verify the descriptor passed to createTexture — we don't
   * need a real GPU.
   */
  function makeMockDevice(): {
    device: GPUDevice;
    lastDescriptor: GPUTextureDescriptor | null;
  } {
    const state: { lastDescriptor: GPUTextureDescriptor | null } = { lastDescriptor: null };
    const mockTexture = {} as GPUTexture;
    const device = {
      createTexture(desc: GPUTextureDescriptor): GPUTexture {
        state.lastDescriptor = desc;
        return mockTexture;
      },
    } as unknown as GPUDevice;
    return { device, lastDescriptor: state.lastDescriptor! };
  }

  it('returns a GPUTexture-typed value (duck-type check)', () => {
    const { device } = makeMockDevice();
    const result = createVarianceBuffer(device, 1280, 720);
    // The mock device returns a non-null object for every createTexture call.
    expect(result).not.toBeNull();
  });

  it('creates a texture with rg32float format', () => {
    let capturedDesc: GPUTextureDescriptor | null = null;
    const mockDevice = {
      createTexture(desc: GPUTextureDescriptor): GPUTexture {
        capturedDesc = desc;
        return {} as GPUTexture;
      },
    } as unknown as GPUDevice;

    createVarianceBuffer(mockDevice, 1920, 1080);
    expect(capturedDesc).not.toBeNull();
    expect((capturedDesc as unknown as GPUTextureDescriptor).format).toBe('rg32float');
  });

  it('creates a texture with the supplied dimensions', () => {
    let capturedDesc: GPUTextureDescriptor | null = null;
    const mockDevice = {
      createTexture(desc: GPUTextureDescriptor): GPUTexture {
        capturedDesc = desc;
        return {} as GPUTexture;
      },
    } as unknown as GPUDevice;

    createVarianceBuffer(mockDevice, 640, 480);
    expect(capturedDesc).not.toBeNull();
    const size = (capturedDesc as unknown as GPUTextureDescriptor).size as [number, number];
    expect(size[0]).toBe(640);
    expect(size[1]).toBe(480);
  });

  it('includes STORAGE_BINDING in the usage flags', () => {
    let capturedDesc: GPUTextureDescriptor | null = null;
    const mockDevice = {
      createTexture(desc: GPUTextureDescriptor): GPUTexture {
        capturedDesc = desc;
        return {} as GPUTexture;
      },
    } as unknown as GPUDevice;

    createVarianceBuffer(mockDevice, 256, 256);
    const usage = (capturedDesc as unknown as GPUTextureDescriptor).usage;
    expect(usage & GPUTextureUsage.STORAGE_BINDING).toBeTruthy();
  });

  it('includes TEXTURE_BINDING in the usage flags (Sprint 10a atrous-variance samples it)', () => {
    let capturedDesc: GPUTextureDescriptor | null = null;
    const mockDevice = {
      createTexture(desc: GPUTextureDescriptor): GPUTexture {
        capturedDesc = desc;
        return {} as GPUTexture;
      },
    } as unknown as GPUDevice;

    createVarianceBuffer(mockDevice, 256, 256);
    const usage = (capturedDesc as unknown as GPUTextureDescriptor).usage;
    expect(usage & GPUTextureUsage.TEXTURE_BINDING).toBeTruthy();
  });

  it('includes COPY_SRC in the usage flags (allows CPU readback for test harness)', () => {
    let capturedDesc: GPUTextureDescriptor | null = null;
    const mockDevice = {
      createTexture(desc: GPUTextureDescriptor): GPUTexture {
        capturedDesc = desc;
        return {} as GPUTexture;
      },
    } as unknown as GPUDevice;

    createVarianceBuffer(mockDevice, 256, 256);
    const usage = (capturedDesc as unknown as GPUTextureDescriptor).usage;
    expect(usage & GPUTextureUsage.COPY_SRC).toBeTruthy();
  });
});

// ─── 5. FrameResources — varianceBuffer field ─────────────────────────────────

describe('FrameResources — varianceBuffer field (Sprint 9)', () => {
  it('createFrameResources is exported from resourceManager', async () => {
    const mod = await import('../src/pipeline/resourceManager.js');
    expect(typeof mod.createFrameResources).toBe('function');
  });

  it('createVarianceBuffer is exported from resourceManager', async () => {
    const mod = await import('../src/pipeline/resourceManager.js');
    expect(typeof mod.createVarianceBuffer).toBe('function');
  });

  it('destroyFrameResources is exported from resourceManager', async () => {
    const mod = await import('../src/pipeline/resourceManager.js');
    expect(typeof mod.destroyFrameResources).toBe('function');
  });

  it('FrameResources interface includes varianceBuffer (mock device smoke test)', () => {
    // Build a minimal mock that satisfies createFrameResources's GPUDevice calls.
    // This validates that the varianceBuffer field is present in the returned object
    // without running a real GPU pipeline.
    const textureMock = { destroy: vi.fn(), createView: vi.fn(() => ({})) };
    const bufferMock  = { destroy: vi.fn() };
    const samplerMock = {};

    const mockQueue = {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
    };
    const mockDevice = {
      createBuffer:  vi.fn(() => bufferMock),
      createTexture: vi.fn(() => textureMock),
      createSampler: vi.fn(() => samplerMock),
      queue: mockQueue,
    } as unknown as GPUDevice;

    const res = createFrameResources(mockDevice, 64, 64);

    // W1-R2 — FrameResources is now grouped into per-algorithm sub-structs.
    // varianceBuffer + friends moved under `res.common`.
    expect(res.common).toHaveProperty('varianceBuffer');
    expect(res.common.varianceBuffer).not.toBeNull();
    expect(res.common).toHaveProperty('varianceBufferAux');
    expect(res.common).toHaveProperty('atrousVarianceEstimateTexture');
    expect(res.common).toHaveProperty('motionVectorTexture');
  });

  it('destroyFrameResources calls destroy on varianceBuffer', async () => {
    const destroyMock = vi.fn();
    const textureMock = { destroy: destroyMock, createView: vi.fn(() => ({})) };
    const bufferMock  = { destroy: vi.fn() };
    const samplerMock = {};

    const mockDevice = {
      createBuffer:  vi.fn(() => bufferMock),
      createTexture: vi.fn(() => textureMock),
      createSampler: vi.fn(() => samplerMock),
      queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    } as unknown as GPUDevice;

    const { destroyFrameResources } = await import('../src/pipeline/resourceManager.js');
    const res = createFrameResources(mockDevice, 64, 64);
    destroyFrameResources(res);

    // destroy must have been called at least once for the variance buffer
    // (and for all other textures — varianceBuffer is one of the GPUTexture
    // mocks that uses textureMock's destroy).
    expect(destroyMock).toHaveBeenCalled();
  });
});
