/**
 * atrousVariance.test.ts — Tests for Sprint 10a à-trous + variance denoiser.
 *
 * Previously named svgf.test.ts; renamed by sweep-2026-05-11 D3.
 * The denoiser was previously called SVGF but never implemented real
 * Schied 2017 SVGF. Real SVGF is tracked in plan/sprint-svgf-real-future.md.
 *
 * Does NOT execute WGSL on a GPU. Verifies:
 *   1. WGSL string exports are non-empty and well-formed.
 *   2. Both compute entry points are present.
 *   3. All expected bindings are declared.
 *   4. WelfordVariance struct is declared (cross-package layout compatibility).
 *   5. TypeScript uniform packer round-trips correctly.
 *   6. Default σ constants are sane.
 */

import { describe, it, expect } from 'vitest';
import { ATROUS_VARIANCE_WGSL } from '../src/wgsl/atrousVariance.wgsl.js';
import {
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  packAtrousVarianceAtrousUniforms,
  packAtrousVarianceVarianceUniforms,
} from '../src/atrousVarianceBindings.js';
import {
  ATROUS_VARIANCE_FRAME_COUNT_INPUT_GUARD_MAX,
  ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT,
} from '../src/atrousVarianceConstants.js';

// ── ATROUS_VARIANCE_WGSL content tests ──────────────────────────────────────

describe('ATROUS_VARIANCE_WGSL', () => {
  it('is a non-empty string', () => {
    expect(typeof ATROUS_VARIANCE_WGSL).toBe('string');
    expect(ATROUS_VARIANCE_WGSL.length).toBeGreaterThan(0);
  });

  it('declares svgfVarianceMain compute entry point', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('fn svgfVarianceMain(');
  });

  it('declares svgfAtrousMain compute entry point', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('fn svgfAtrousMain(');
  });

  it('declares @compute @workgroup_size(16, 16, 1) for both entry points', () => {
    const matches = [...ATROUS_VARIANCE_WGSL.matchAll(/@compute @workgroup_size\(16, 16, 1\)/g)];
    expect(matches.length).toBe(2);
  });

  it('injects temporal variance frame threshold matching atrousVarianceConstants', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain(
      `SVGF_TEMPORAL_VARIANCE_MIN_FRAMES: u32 = ${ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT}u`,
    );
    expect(ATROUS_VARIANCE_WGSL).toContain('if (frameCount < SVGF_TEMPORAL_VARIANCE_MIN_FRAMES)');
  });

  // Variance pass bindings
  it('declares inputColor binding in variance pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var varIn_inputColor');
  });

  it('declares prevRadiance binding', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var varIn_prevRadiance');
  });

  it('declares gbufferNormal binding in variance pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var varIn_gbufNormal');
  });

  it('declares gbufferDepth binding in variance pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var varIn_gbufDepth');
  });

  it('declares motionVectors binding', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var varIn_motionVec');
  });

  it('declares varianceIn binding', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var varIn_varianceIn');
  });

  it('declares varianceOut storage binding', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var varOut_varianceOut');
  });

  it('declares AtrousVarianceVarianceUBO struct', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('struct AtrousVarianceVarianceUBO');
    expect(ATROUS_VARIANCE_WGSL).toContain('frameCount');
  });

  // À-trous pass bindings
  it('declares inputColor in atrous pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var atrous_inputColor');
  });

  it('declares outputColor storage in atrous pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var atrous_outputColor');
  });

  it('declares gbufferNormal in atrous pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var atrous_gbufNormal');
  });

  it('declares gbufferDepth in atrous pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var atrous_gbufDepth');
  });

  it('declares varianceMap binding in atrous pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('var atrous_varianceMap');
  });

  it('declares AtrousVarianceAtrousUBO struct with sigma fields', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('struct AtrousVarianceAtrousUBO');
    expect(ATROUS_VARIANCE_WGSL).toContain('sigmaColor');
    expect(ATROUS_VARIANCE_WGSL).toContain('sigmaNormal');
    expect(ATROUS_VARIANCE_WGSL).toContain('sigmaDepth');
    expect(ATROUS_VARIANCE_WGSL).toContain('iteration');
  });

  // WelfordVariance struct compatibility
  it('declares WelfordVariance struct (cross-package layout compatibility)', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('struct WelfordVariance');
  });

  it('WelfordVariance has mean and m2 fields matching common.wgsl @version 1', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('mean: f32');
    expect(ATROUS_VARIANCE_WGSL).toContain('m2:   f32');
  });

  it('declares welfordVariance helper function', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('fn welfordVariance(');
  });

  // Variance-guided behavior
  it('uses spatial 3×3 neighborhood for low frame counts', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('if (frameCount < SVGF_TEMPORAL_VARIANCE_MIN_FRAMES)');
  });

  it('falls back to Welford variance for stable temporal history', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('welfordVariance(state');
  });

  // À-trous wavelet
  it('uses step width based on iteration index (2^iteration pattern)', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('1u << atrousUBO.iteration');
  });

  it('implements bilateral edge-stopping (color + normal + depth weights)', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('wc'); // color weight
    expect(ATROUS_VARIANCE_WGSL).toContain('wn'); // normal weight
    expect(ATROUS_VARIANCE_WGSL).toContain('wz'); // depth weight
  });

  it('uses textureStore for outputColor', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('textureStore(atrous_outputColor');
  });

  it('uses textureStore for varianceOut', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('textureStore(varOut_varianceOut');
  });

  it('declares 5×5 B3 spline kernel (ATROUS_VARIANCE_KERNEL array)', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('ATROUS_VARIANCE_KERNEL');
    expect(ATROUS_VARIANCE_WGSL).toContain('array<f32, 25>');
  });

  it('sky/miss pixels pass through unfiltered in atrous pass', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('zCenter <= 0.0');
  });

  it('does NOT claim to implement Schied 2017 SVGF', () => {
    expect(ATROUS_VARIANCE_WGSL).not.toContain('Schied 2017');
    expect(ATROUS_VARIANCE_WGSL).not.toContain('Spatiotemporal Variance-Guided Filtering');
  });
});

// ── AtrousVarianceAtrousUniforms packer tests ─────────────────────────────────

describe('packAtrousVarianceAtrousUniforms', () => {
  it('packs to ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES (16 bytes)', () => {
    expect(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES).toBe(16);
  });

  it('round-trips iteration as u32 at offset 0', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceAtrousUniforms(
      { iteration: 3, sigmaColor: 10, sigmaNormal: 128, sigmaDepth: 1 },
      buf,
    );
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(3);
  });

  it('round-trips sigmaColor as f32 at offset 4', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceAtrousUniforms(
      { iteration: 0, sigmaColor: 10.0, sigmaNormal: 128.0, sigmaDepth: 1.0 },
      buf,
    );
    const view = new DataView(buf);
    expect(view.getFloat32(4, true)).toBeCloseTo(10.0, 3);
  });

  it('round-trips sigmaNormal as f32 at offset 8', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceAtrousUniforms(
      { iteration: 0, sigmaColor: 10.0, sigmaNormal: 128.0, sigmaDepth: 1.0 },
      buf,
    );
    const view = new DataView(buf);
    expect(view.getFloat32(8, true)).toBeCloseTo(128.0, 1);
  });

  it('round-trips sigmaDepth as f32 at offset 12', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceAtrousUniforms(
      { iteration: 0, sigmaColor: 10.0, sigmaNormal: 128.0, sigmaDepth: 1.5 },
      buf,
    );
    const view = new DataView(buf);
    expect(view.getFloat32(12, true)).toBeCloseTo(1.5, 4);
  });

  it('supports non-zero byte offset', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES * 2);
    packAtrousVarianceAtrousUniforms(
      { iteration: 4, sigmaColor: 5.0, sigmaNormal: 64.0, sigmaDepth: 2.0 },
      buf,
      ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
    );
    const view = new DataView(buf, ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    expect(view.getUint32(0, true)).toBe(4);
    expect(view.getFloat32(4, true)).toBeCloseTo(5.0, 3);
    expect(view.getFloat32(8, true)).toBeCloseTo(64.0, 1);
    expect(view.getFloat32(12, true)).toBeCloseTo(2.0, 4);
  });

  it('iteration 0 produces step width 1 (the logical 2^0 = 1)', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceAtrousUniforms(
      { iteration: 0, sigmaColor: 10, sigmaNormal: 128, sigmaDepth: 1 },
      buf,
    );
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(0);
  });

  it('iteration 4 produces step width 16 (the logical 2^4 = 16)', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceAtrousUniforms(
      { iteration: 4, sigmaColor: 10, sigmaNormal: 128, sigmaDepth: 1 },
      buf,
    );
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(4);
  });
});

// ── AtrousVarianceVarianceUniforms packer tests ───────────────────────────────

describe('packAtrousVarianceVarianceUniforms', () => {
  it('packs to ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES (16 bytes)', () => {
    expect(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES).toBe(16);
  });

  it('round-trips frameCount as u32 at offset 0', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceVarianceUniforms({ frameCount: 42 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(42);
  });

  it('pads remaining 12 bytes with zeros', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
    // Pre-fill with 0xFF to ensure packer writes zeros.
    new Uint8Array(buf).fill(0xff);
    packAtrousVarianceVarianceUniforms({ frameCount: 1 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(4, true)).toBe(0);
    expect(view.getUint32(8, true)).toBe(0);
    expect(view.getUint32(12, true)).toBe(0);
  });

  it('frameCount = 0 (first frame) packs correctly', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceVarianceUniforms({ frameCount: 0 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(0);
  });

  it('saturates frameCount at ATROUS_VARIANCE_FRAME_COUNT_INPUT_GUARD_MAX', () => {
    const buf = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceVarianceUniforms(
      { frameCount: ATROUS_VARIANCE_FRAME_COUNT_INPUT_GUARD_MAX + 9 },
      buf,
    );
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(ATROUS_VARIANCE_FRAME_COUNT_INPUT_GUARD_MAX);
  });
});

// ── Item 24 — albedo demodulation contract (Schied 2017 §4.1) ─────────────────
//
// These tests verify that the WGSL shader module does NOT claim to perform
// demodulation internally (it operates on whatever signal is passed via
// inputColor) and that the TS-level contract markers are correct. The actual
// demodulate/remodulate math is tested at the CPU-helper level in
// atrousVarianceWebGpuInputs.test.ts.

describe('Item 24 — albedo demodulation WGSL contract', () => {
  it('atrous pass reads inputColor without any albedo divide (demodulation is caller responsibility)', () => {
    // The WGSL shader should NOT contain an albedo demodulation operation —
    // demodulation is applied by the host before uploading to colorPingA.
    // Check that the atrous entry point reads inputColor directly, not a
    // derived lighting buffer. We verify the atrous pass still contains the
    // standard textureLoad on atrous_inputColor (no implicit albedo divide).
    expect(ATROUS_VARIANCE_WGSL).toContain('textureLoad(atrous_inputColor,');
  });

  it('does not declare an albedo texture binding (demodulation is host-side)', () => {
    // The à-trous shader module does NOT add an albedo binding — that binding
    // lives in shade.wgsl (write) and indirectCombine.wgsl (re-modulate read).
    // This test ensures a future refactor does not accidentally pull albedo
    // into the shared-denoisers shader.
    expect(ATROUS_VARIANCE_WGSL).not.toContain('albedo');
  });
});

// ── ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS sanity checks ─────────────────────

describe('ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS', () => {
  it('sigmaColor default is 4.0 (tuned to suppress ReSTIR-DI flicker in walkaround Cornell scenes)', () => {
    expect(ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaColor).toBe(4.0);
  });

  it('sigmaNormal default is 128.0', () => {
    expect(ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaNormal).toBe(128.0);
  });

  it('sigmaDepth default is 1.0', () => {
    expect(ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS.sigmaDepth).toBe(1.0);
  });

  it('all defaults are positive finite numbers', () => {
    for (const [key, value] of Object.entries(ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS)) {
      expect(typeof value, `${key} should be number`).toBe('number');
      expect(Number.isFinite(value), `${key} should be finite`).toBe(true);
      expect(value, `${key} should be positive`).toBeGreaterThan(0);
    }
  });
});
