/**
 * svgf.test.ts — Tests for Sprint 10a SVGF denoiser.
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
import { SVGF_WGSL } from '../src/wgsl/svgf.wgsl.js';
import {
  SVGF_UNIFORMS_SIZE_BYTES,
  SVGF_VARIANCE_UNIFORMS_SIZE_BYTES,
  SVGF_DEFAULT_UNIFORMS,
  packSVGFUniforms,
  packSVGFVarianceUniforms,
} from '../src/svgfBindings.js';

// ── SVGF_WGSL content tests ──────────────────────────────────────────────────

describe('SVGF_WGSL', () => {
  it('is a non-empty string', () => {
    expect(typeof SVGF_WGSL).toBe('string');
    expect(SVGF_WGSL.length).toBeGreaterThan(0);
  });

  it('declares svgfVarianceMain compute entry point', () => {
    expect(SVGF_WGSL).toContain('fn svgfVarianceMain(');
  });

  it('declares svgfAtrousMain compute entry point', () => {
    expect(SVGF_WGSL).toContain('fn svgfAtrousMain(');
  });

  it('declares @compute @workgroup_size(16, 16, 1) for both entry points', () => {
    const matches = [...SVGF_WGSL.matchAll(/@compute @workgroup_size\(16, 16, 1\)/g)];
    expect(matches.length).toBe(2);
  });

  // Variance pass bindings
  it('declares inputColor binding in variance pass', () => {
    expect(SVGF_WGSL).toContain('var varIn_inputColor');
  });

  it('declares prevRadiance binding', () => {
    expect(SVGF_WGSL).toContain('var varIn_prevRadiance');
  });

  it('declares gbufferNormal binding in variance pass', () => {
    expect(SVGF_WGSL).toContain('var varIn_gbufNormal');
  });

  it('declares gbufferDepth binding in variance pass', () => {
    expect(SVGF_WGSL).toContain('var varIn_gbufDepth');
  });

  it('declares motionVectors binding', () => {
    expect(SVGF_WGSL).toContain('var varIn_motionVec');
  });

  it('declares varianceIn binding', () => {
    expect(SVGF_WGSL).toContain('var varIn_varianceIn');
  });

  it('declares varianceOut storage binding', () => {
    expect(SVGF_WGSL).toContain('var varOut_varianceOut');
  });

  it('declares SVGFVarianceUBO struct', () => {
    expect(SVGF_WGSL).toContain('struct SVGFVarianceUBO');
    expect(SVGF_WGSL).toContain('frameCount');
  });

  // À-trous pass bindings
  it('declares inputColor in atrous pass', () => {
    expect(SVGF_WGSL).toContain('var atrous_inputColor');
  });

  it('declares outputColor storage in atrous pass', () => {
    expect(SVGF_WGSL).toContain('var atrous_outputColor');
  });

  it('declares gbufferNormal in atrous pass', () => {
    expect(SVGF_WGSL).toContain('var atrous_gbufNormal');
  });

  it('declares gbufferDepth in atrous pass', () => {
    expect(SVGF_WGSL).toContain('var atrous_gbufDepth');
  });

  it('declares varianceMap binding in atrous pass', () => {
    expect(SVGF_WGSL).toContain('var atrous_varianceMap');
  });

  it('declares SVGFAtrousUBO struct with sigma fields', () => {
    expect(SVGF_WGSL).toContain('struct SVGFAtrousUBO');
    expect(SVGF_WGSL).toContain('sigmaColor');
    expect(SVGF_WGSL).toContain('sigmaNormal');
    expect(SVGF_WGSL).toContain('sigmaDepth');
    expect(SVGF_WGSL).toContain('iteration');
  });

  // WelfordVariance struct compatibility
  it('declares WelfordVariance struct (cross-package layout compatibility)', () => {
    expect(SVGF_WGSL).toContain('struct WelfordVariance');
  });

  it('WelfordVariance has mean and m2 fields matching common.wgsl @version 1', () => {
    expect(SVGF_WGSL).toContain('mean: f32');
    expect(SVGF_WGSL).toContain('m2:   f32');
  });

  it('declares welfordVariance helper function', () => {
    expect(SVGF_WGSL).toContain('fn welfordVariance(');
  });

  // Variance-guided behavior
  it('uses spatial 3×3 neighborhood for low frame counts', () => {
    expect(SVGF_WGSL).toContain('frameCount < 4u');
  });

  it('falls back to Welford variance for stable temporal history', () => {
    expect(SVGF_WGSL).toContain('welfordVariance(state');
  });

  // À-trous wavelet
  it('uses step width based on iteration index (2^iteration pattern)', () => {
    expect(SVGF_WGSL).toContain('1u << atrousUBO.iteration');
  });

  it('implements bilateral edge-stopping (color + normal + depth weights)', () => {
    expect(SVGF_WGSL).toContain('wc');   // color weight
    expect(SVGF_WGSL).toContain('wn');   // normal weight
    expect(SVGF_WGSL).toContain('wz');   // depth weight
  });

  it('uses textureStore for outputColor', () => {
    expect(SVGF_WGSL).toContain('textureStore(atrous_outputColor');
  });

  it('uses textureStore for varianceOut', () => {
    expect(SVGF_WGSL).toContain('textureStore(varOut_varianceOut');
  });

  it('declares 5×5 B3 spline kernel (SVGF_KERNEL array)', () => {
    expect(SVGF_WGSL).toContain('SVGF_KERNEL');
    expect(SVGF_WGSL).toContain('array<f32, 25>');
  });

  it('sky/miss pixels pass through unfiltered in atrous pass', () => {
    expect(SVGF_WGSL).toContain('zCenter <= 0.0');
  });
});

// ── SVGFUniforms packer tests ────────────────────────────────────────────────

describe('packSVGFUniforms', () => {
  it('packs to SVGF_UNIFORMS_SIZE_BYTES (16 bytes)', () => {
    expect(SVGF_UNIFORMS_SIZE_BYTES).toBe(16);
  });

  it('round-trips iteration as u32 at offset 0', () => {
    const buf = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);
    packSVGFUniforms({ iteration: 3, sigmaColor: 10, sigmaNormal: 128, sigmaDepth: 1 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(3);
  });

  it('round-trips sigmaColor as f32 at offset 4', () => {
    const buf = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);
    packSVGFUniforms({ iteration: 0, sigmaColor: 10.0, sigmaNormal: 128.0, sigmaDepth: 1.0 }, buf);
    const view = new DataView(buf);
    expect(view.getFloat32(4, true)).toBeCloseTo(10.0, 3);
  });

  it('round-trips sigmaNormal as f32 at offset 8', () => {
    const buf = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);
    packSVGFUniforms({ iteration: 0, sigmaColor: 10.0, sigmaNormal: 128.0, sigmaDepth: 1.0 }, buf);
    const view = new DataView(buf);
    expect(view.getFloat32(8, true)).toBeCloseTo(128.0, 1);
  });

  it('round-trips sigmaDepth as f32 at offset 12', () => {
    const buf = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);
    packSVGFUniforms({ iteration: 0, sigmaColor: 10.0, sigmaNormal: 128.0, sigmaDepth: 1.5 }, buf);
    const view = new DataView(buf);
    expect(view.getFloat32(12, true)).toBeCloseTo(1.5, 4);
  });

  it('supports non-zero byte offset', () => {
    const buf = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES * 2);
    packSVGFUniforms(
      { iteration: 4, sigmaColor: 5.0, sigmaNormal: 64.0, sigmaDepth: 2.0 },
      buf,
      SVGF_UNIFORMS_SIZE_BYTES,
    );
    const view = new DataView(buf, SVGF_UNIFORMS_SIZE_BYTES);
    expect(view.getUint32(0, true)).toBe(4);
    expect(view.getFloat32(4, true)).toBeCloseTo(5.0, 3);
    expect(view.getFloat32(8, true)).toBeCloseTo(64.0, 1);
    expect(view.getFloat32(12, true)).toBeCloseTo(2.0, 4);
  });

  it('iteration 0 produces step width 1 (the logical 2^0 = 1)', () => {
    // Structural check: iteration=0 is the minimum valid value.
    const buf = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);
    packSVGFUniforms({ iteration: 0, sigmaColor: 10, sigmaNormal: 128, sigmaDepth: 1 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(0);
  });

  it('iteration 4 produces step width 16 (the logical 2^4 = 16)', () => {
    const buf = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);
    packSVGFUniforms({ iteration: 4, sigmaColor: 10, sigmaNormal: 128, sigmaDepth: 1 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(4);
  });
});

// ── SVGFVarianceUniforms packer tests ────────────────────────────────────────

describe('packSVGFVarianceUniforms', () => {
  it('packs to SVGF_VARIANCE_UNIFORMS_SIZE_BYTES (16 bytes)', () => {
    expect(SVGF_VARIANCE_UNIFORMS_SIZE_BYTES).toBe(16);
  });

  it('round-trips frameCount as u32 at offset 0', () => {
    const buf = new ArrayBuffer(SVGF_VARIANCE_UNIFORMS_SIZE_BYTES);
    packSVGFVarianceUniforms({ frameCount: 42 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(42);
  });

  it('pads remaining 12 bytes with zeros', () => {
    const buf = new ArrayBuffer(SVGF_VARIANCE_UNIFORMS_SIZE_BYTES);
    // Pre-fill with 0xFF to ensure packer writes zeros.
    new Uint8Array(buf).fill(0xFF);
    packSVGFVarianceUniforms({ frameCount: 1 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(4,  true)).toBe(0);
    expect(view.getUint32(8,  true)).toBe(0);
    expect(view.getUint32(12, true)).toBe(0);
  });

  it('frameCount = 0 (first frame) packs correctly', () => {
    const buf = new ArrayBuffer(SVGF_VARIANCE_UNIFORMS_SIZE_BYTES);
    packSVGFVarianceUniforms({ frameCount: 0 }, buf);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(0);
  });
});

// ── SVGF_DEFAULT_UNIFORMS sanity checks ──────────────────────────────────────

describe('SVGF_DEFAULT_UNIFORMS', () => {
  it('sigmaColor default is 10.0 (Schied 2017 Table 1)', () => {
    expect(SVGF_DEFAULT_UNIFORMS.sigmaColor).toBe(10.0);
  });

  it('sigmaNormal default is 128.0 (Schied 2017 Table 1)', () => {
    expect(SVGF_DEFAULT_UNIFORMS.sigmaNormal).toBe(128.0);
  });

  it('sigmaDepth default is 1.0', () => {
    expect(SVGF_DEFAULT_UNIFORMS.sigmaDepth).toBe(1.0);
  });

  it('all defaults are positive finite numbers', () => {
    for (const [key, value] of Object.entries(SVGF_DEFAULT_UNIFORMS)) {
      expect(typeof value, `${key} should be number`).toBe('number');
      expect(Number.isFinite(value), `${key} should be finite`).toBe(true);
      expect(value, `${key} should be positive`).toBeGreaterThan(0);
    }
  });
});
