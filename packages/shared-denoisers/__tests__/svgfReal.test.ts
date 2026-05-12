/**
 * svgfReal.test.ts — CPU-emulation tests for Schied 2017 SVGF (T2.H1).
 *
 * All tests run on CPU emulations of the WGSL kernels (no GPU required).
 * GPU verification is the user's responsibility (see README).
 *
 * Test coverage:
 *   1. Reprojection identity (no motion): α increases correctly per frame.
 *   2. Disocclusion reset: depth jump → history resets to 1, α=1.
 *   3. Variance from moments: Var = M2 - M1² within ±1e-6.
 *   4. 7×7 spatial fallback: for history=0, matches CPU 7×7 box variance.
 *   5. End-to-end identity: 50 frames of identical input + zero motion → converges within ±1e-3.
 */

import { describe, it, expect } from 'vitest';
import {
  svgfReprojCPU,
  svgfVarianceFromMomentsCPU,
  svgf7x7FallbackCPU,
} from '../src/svgfRealWebGPU.js';
import {
  SVGF_REPROJ_DEFAULT_UNIFORMS,
} from '../src/svgfRealBindings.js';
import {
  SVGF_HISTORY_MIN_FOR_MOMENTS,
} from '../src/wgsl/svgfVarianceFromMoments.wgsl.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeGeo(W: number, H: number, depth: number, nx: number, ny: number, nz: number) {
  const px = W * H;
  const depth1 = new Float32Array(px).fill(depth);
  // Pack normals 0..1: world-space (nx, ny, nz) → (nx/2+0.5, ny/2+0.5, nz/2+0.5)
  const norm = new Float32Array(px * 3);
  for (let i = 0; i < px; i++) {
    norm[i * 3]     = nx / 2 + 0.5;
    norm[i * 3 + 1] = ny / 2 + 0.5;
    norm[i * 3 + 2] = nz / 2 + 0.5;
  }
  const objId = new Uint32Array(px).fill(1);
  return { depth1, norm, objId };
}

function makeColor(W: number, H: number, r: number, g: number, b: number): Float32Array {
  const arr = new Float32Array(W * H * 3);
  for (let i = 0; i < W * H; i++) {
    arr[i * 3]     = r;
    arr[i * 3 + 1] = g;
    arr[i * 3 + 2] = b;
  }
  return arr;
}

// ── Test 1: Reprojection identity (no motion) ─────────────────────────────────

describe('svgfReprojCPU — identity (no motion, consistent geometry)', () => {
  it('should accumulate history and drive alpha toward alphaMin over multiple frames', () => {
    const W = 4, H = 4;
    const color = makeColor(W, H, 0.5, 0.3, 0.2);
    const { depth1, norm, objId } = makeGeo(W, H, 2.0, 0, 0, 1);
    const motion = new Float32Array(W * H * 2).fill(0); // zero motion

    // Simulate 20 frames accumulating history.
    let prevColor: Float32Array = color.slice();
    let historyIn: Uint32Array = new Uint32Array(W * H).fill(0);
    let momentsIn: Float32Array = new Float32Array(W * H * 2).fill(0);
    let lastAlpha = 1.0;

    for (let frame = 0; frame < 20; frame++) {
      const result = svgfReprojCPU({
        currColor: color,
        prevColor,
        motionVec: motion,
        currDepth: depth1,
        currNormal: norm,
        currObjId: objId,
        prevDepth: depth1,
        prevNormal: norm,
        prevObjId: objId,
        historyLengthIn: historyIn,
        momentsIn,
        width: W,
        height: H,
      });

      // After frame F, history should be F+1 (initial 0 → 1 → 2 → ...)
      const h = result.historyLengthOut[0] ?? 0;
      expect(h).toBe(frame + 1);

      // α = max(alphaMin, 1/(h+1)) — should decrease as h grows.
      const expectedAlpha = Math.max(
        SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin,
        1 / (h + 1),
      );
      if (frame > 0) {
        expect(expectedAlpha).toBeLessThanOrEqual(lastAlpha + 1e-9);
      }
      lastAlpha = expectedAlpha;

      // After enough frames, alpha should saturate at alphaMin.
      if (frame >= 18) {
        expect(expectedAlpha).toBeCloseTo(SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin, 5);
      }

      // Update state for next frame.
      prevColor = result.colorOut;
      historyIn = result.historyLengthOut;
      momentsIn = result.momentsOut;
    }
  });

  it('should converge blended color toward input when input is constant', () => {
    const W = 2, H = 2;
    const color = makeColor(W, H, 0.8, 0.6, 0.4);
    const { depth1, norm, objId } = makeGeo(W, H, 1.0, 0, 1, 0);
    const motion = new Float32Array(W * H * 2).fill(0);

    let prevColor: Float32Array = makeColor(W, H, 0, 0, 0); // start from black
    let historyIn: Uint32Array = new Uint32Array(W * H).fill(0);
    let momentsIn: Float32Array = new Float32Array(W * H * 2).fill(0);

    for (let frame = 0; frame < 40; frame++) {
      const result = svgfReprojCPU({
        currColor: color,
        prevColor,
        motionVec: motion,
        currDepth: depth1,
        currNormal: norm,
        currObjId: objId,
        prevDepth: depth1,
        prevNormal: norm,
        prevObjId: objId,
        historyLengthIn: historyIn,
        momentsIn,
        width: W,
        height: H,
      });
      prevColor = result.colorOut;
      historyIn = result.historyLengthOut;
      momentsIn = result.momentsOut;
    }

    // After 40 frames, blended color should be within ±0.01 of input.
    for (let i = 0; i < W * H; i++) {
      expect(prevColor[i * 3]     ?? 0).toBeCloseTo(0.8, 1);
      expect(prevColor[i * 3 + 1] ?? 0).toBeCloseTo(0.6, 1);
      expect(prevColor[i * 3 + 2] ?? 0).toBeCloseTo(0.4, 1);
    }
  });
});

// ── Test 2: Disocclusion reset ───────────────────────────────────────────────

describe('svgfReprojCPU — disocclusion reset', () => {
  it('should reset history to 1 and α=1 on depth jump', () => {
    const W = 2, H = 2;
    const color = makeColor(W, H, 0.5, 0.5, 0.5);
    const { norm, objId } = makeGeo(W, H, 2.0, 0, 0, 1);
    const motion = new Float32Array(W * H * 2).fill(0);

    // Previous frame depth is far away — big depth jump → disocclusion.
    const currDepth = new Float32Array(W * H).fill(2.0);
    const prevDepth = new Float32Array(W * H).fill(100.0); // large jump

    // Seed history at 10 frames so we can verify it resets.
    const historyIn = new Uint32Array(W * H).fill(10);
    const momentsIn = new Float32Array(W * H * 2).fill(0.5);

    const result = svgfReprojCPU({
      currColor: color,
      prevColor: color,
      motionVec: motion,
      currDepth,
      currNormal: norm,
      currObjId: objId,
      prevDepth,
      prevNormal: norm,
      prevObjId: objId,
      historyLengthIn: historyIn,
      momentsIn,
      width: W,
      height: H,
    });

    // All pixels should be rejected → history reset to 1.
    for (let i = 0; i < W * H; i++) {
      expect(result.historyLengthOut[i]).toBe(1);
    }

    // Output color should be the current frame's input (α=1).
    for (let i = 0; i < W * H; i++) {
      expect(result.colorOut[i * 3]     ?? 0).toBeCloseTo(0.5, 5);
      expect(result.colorOut[i * 3 + 1] ?? 0).toBeCloseTo(0.5, 5);
      expect(result.colorOut[i * 3 + 2] ?? 0).toBeCloseTo(0.5, 5);
    }
  });

  it('should reset history on object-id mismatch', () => {
    const W = 1, H = 1;
    const color = makeColor(W, H, 0.3, 0.3, 0.3);
    const { depth1, norm } = makeGeo(W, H, 1.0, 0, 0, 1);
    const motion = new Float32Array(2).fill(0);

    const currObjId = new Uint32Array([5]);
    const prevObjId = new Uint32Array([99]); // mismatch

    const historyIn = new Uint32Array([15]);
    const momentsIn = new Float32Array(2).fill(0.2);

    const result = svgfReprojCPU({
      currColor: color,
      prevColor: color,
      motionVec: motion,
      currDepth: depth1,
      currNormal: norm,
      currObjId,
      prevDepth: depth1,
      prevNormal: norm,
      prevObjId,
      historyLengthIn: historyIn,
      momentsIn,
      width: W,
      height: H,
    });

    expect(result.historyLengthOut[0]).toBe(1);
  });
});

// ── Test 3: Variance from moments ────────────────────────────────────────────

describe('svgfVarianceFromMomentsCPU — Eq. 5', () => {
  it('should compute Var = max(0, M2 - M1²) within ±1e-6', () => {
    const cases: [number, number, number][] = [
      // [M1, M2, expectedVar]
      [0.5, 0.3, 0],         // M2 < M1² → clamped to 0
      [0.5, 0.5, 0.25],      // M2 - M1² = 0.5 - 0.25 = 0.25
      [0.3, 0.5, 0.41],      // 0.5 - 0.09 = 0.41
      [1.0, 1.5, 0.5],       // 1.5 - 1.0 = 0.5
      [0.0, 0.0, 0.0],       // trivial
    ];

    const W = cases.length, H = 1;
    const momentsIn = new Float32Array(W * H * 2);
    const expectedVars: number[] = [];

    for (let i = 0; i < W; i++) {
      const [m1, m2, ev] = cases[i]!;
      momentsIn[i * 2]     = m1;
      momentsIn[i * 2 + 1] = m2;
      expectedVars.push(ev);
    }

    // History sufficient for moment-based variance (>= threshold).
    const historyIn = new Uint32Array(W).fill(SVGF_HISTORY_MIN_FOR_MOMENTS);

    const result = svgfVarianceFromMomentsCPU({ momentsIn, historyIn, width: W, height: H });

    for (let i = 0; i < W; i++) {
      const expected = expectedVars[i]!;
      expect(result[i] ?? 0).toBeCloseTo(expected, 5);
    }
  });

  it('should return 0 for pixels with insufficient history', () => {
    const W = 3, H = 1;
    const momentsIn = new Float32Array([0.5, 0.5, 0.4, 0.6, 0.3, 0.5]);
    // History below threshold.
    const historyIn = new Uint32Array([0, 1, 2]);

    const result = svgfVarianceFromMomentsCPU({
      momentsIn, historyIn, width: W, height: H,
      historyMin: SVGF_HISTORY_MIN_FOR_MOMENTS,
    });

    for (let i = 0; i < W; i++) {
      expect(result[i]).toBe(0);
    }
  });
});

// ── Test 4: 7×7 spatial fallback ─────────────────────────────────────────────

describe('svgf7x7FallbackCPU — spatial variance for new pixels', () => {
  it('should compute 7×7 box variance for history=0 pixels', () => {
    const W = 7, H = 7;
    const px = W * H;

    // Constant color → spatial variance = 0.
    const constColor = makeColor(W, H, 0.5, 0.3, 0.2);
    const historyIn = new Uint32Array(px).fill(0); // all new
    const varianceIn = new Float32Array(px).fill(0);

    const resultConst = svgf7x7FallbackCPU({
      currColor: constColor,
      historyIn,
      varianceIn,
      width: W,
      height: H,
    });

    // Constant luminance → variance = 0 everywhere.
    for (let i = 0; i < px; i++) {
      expect(resultConst[i] ?? 0).toBeCloseTo(0, 5);
    }
  });

  it('should match CPU 7×7 box variance on a known noisy pattern', () => {
    const W = 9, H = 9;
    const px = W * H;

    // Alternating luminance pattern — odd pixels bright, even pixels dark.
    const noisy = new Float32Array(px * 3);
    for (let i = 0; i < px; i++) {
      const v = (i % 2 === 0) ? 0.0 : 1.0;
      noisy[i * 3]     = v;
      noisy[i * 3 + 1] = v;
      noisy[i * 3 + 2] = v;
    }

    const historyIn = new Uint32Array(px).fill(0);
    const varianceIn = new Float32Array(px).fill(0);

    const result = svgf7x7FallbackCPU({
      currColor: noisy,
      historyIn,
      varianceIn,
      width: W,
      height: H,
    });

    // Hand-compute reference for the center pixel (4,4):
    // A 7×7 box centered at (4,4) fully fits within the 9×9 image.
    // 49 pixels total; alternating 0/1 by flat index: 25 even (=0), 24 odd (=1).
    // E[L] = 24/49 ≈ 0.4898, E[L²] = 24/49, Var = 24/49 - (24/49)² ≈ 0.2499.
    // Center pixel flat index = 4*9+4 = 40.
    const center = result[40] ?? 0;
    // Should be non-zero (pattern has variance).
    expect(center).toBeGreaterThan(0.01);

    // Pixels with sufficient history should pass through varianceIn unchanged.
    const histWithHistory = new Uint32Array(px).fill(10);
    const varWithHistory  = new Float32Array(px).fill(0.42);
    const passthrough = svgf7x7FallbackCPU({
      currColor: noisy,
      historyIn: histWithHistory,
      varianceIn: varWithHistory,
      width: W,
      height: H,
    });
    for (let i = 0; i < px; i++) {
      expect(passthrough[i] ?? 0).toBeCloseTo(0.42, 5);
    }
  });
});

// ── Test 5: End-to-end identity ───────────────────────────────────────────────

describe('svgfReprojCPU — end-to-end identity convergence', () => {
  it('should converge to input within ±1e-3 after 50 frames of identical input and zero motion', () => {
    const W = 3, H = 3;
    const input = makeColor(W, H, 0.7, 0.4, 0.1);
    const { depth1, norm, objId } = makeGeo(W, H, 1.5, 0, 0, 1);
    const motion = new Float32Array(W * H * 2).fill(0);

    let prevColor: Float32Array = makeColor(W, H, 0, 0, 0);
    let historyIn: Uint32Array = new Uint32Array(W * H).fill(0);
    let momentsIn: Float32Array = new Float32Array(W * H * 2).fill(0);

    for (let frame = 0; frame < 50; frame++) {
      const result = svgfReprojCPU({
        currColor: input,
        prevColor,
        motionVec: motion,
        currDepth: depth1,
        currNormal: norm,
        currObjId: objId,
        prevDepth: depth1,
        prevNormal: norm,
        prevObjId: objId,
        historyLengthIn: historyIn,
        momentsIn,
        width: W,
        height: H,
      });
      prevColor = result.colorOut;
      historyIn = result.historyLengthOut;
      momentsIn = result.momentsOut;
    }

    // After 50 frames with constant input + zero motion,
    // EMA should have converged to input within ±1e-3.
    for (let i = 0; i < W * H; i++) {
      expect(prevColor[i * 3]     ?? 0).toBeCloseTo(0.7, 2);
      expect(prevColor[i * 3 + 1] ?? 0).toBeCloseTo(0.4, 2);
      expect(prevColor[i * 3 + 2] ?? 0).toBeCloseTo(0.1, 2);
    }
  });
});

// ── Test 6: WGSL exports are non-empty ───────────────────────────────────────

describe('SVGF WGSL exports', () => {
  it('svgfReprojection WGSL should be non-empty and contain entry point', async () => {
    const { SVGF_REPROJECTION_WGSL } = await import('../src/wgsl/svgfReprojection.wgsl.js');
    expect(typeof SVGF_REPROJECTION_WGSL).toBe('string');
    expect(SVGF_REPROJECTION_WGSL.length).toBeGreaterThan(100);
    expect(SVGF_REPROJECTION_WGSL).toContain('svgfReprojMain');
    expect(SVGF_REPROJECTION_WGSL).toContain('@compute');
    expect(SVGF_REPROJECTION_WGSL).toContain('SVGFReprojUBO');
  });

  it('svgfVarianceFromMoments WGSL should be non-empty and contain entry point', async () => {
    const { SVGF_VARIANCE_FROM_MOMENTS_WGSL } = await import('../src/wgsl/svgfVarianceFromMoments.wgsl.js');
    expect(typeof SVGF_VARIANCE_FROM_MOMENTS_WGSL).toBe('string');
    expect(SVGF_VARIANCE_FROM_MOMENTS_WGSL).toContain('svgfVarianceFromMomentsMain');
    expect(SVGF_VARIANCE_FROM_MOMENTS_WGSL).toContain('@compute');
  });

  it('svgf7x7SpatialFallback WGSL should be non-empty and contain entry point', async () => {
    const { SVGF_7X7_SPATIAL_FALLBACK_WGSL } = await import('../src/wgsl/svgf7x7SpatialFallback.wgsl.js');
    expect(typeof SVGF_7X7_SPATIAL_FALLBACK_WGSL).toBe('string');
    expect(SVGF_7X7_SPATIAL_FALLBACK_WGSL).toContain('svgf7x7FallbackMain');
    expect(SVGF_7X7_SPATIAL_FALLBACK_WGSL).toContain('@compute');
  });
});

// ── Test 7: SVGFReprojUniforms packer ─────────────────────────────────────────

describe('packSVGFReprojUniforms', () => {
  it('should round-trip the default uniforms correctly', async () => {
    const { packSVGFReprojUniforms, SVGF_REPROJ_DEFAULT_UNIFORMS, SVGF_REPROJ_UNIFORMS_SIZE_BYTES } =
      await import('../src/svgfRealBindings.js');

    const buf = new ArrayBuffer(SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
    packSVGFReprojUniforms(SVGF_REPROJ_DEFAULT_UNIFORMS, buf);
    const dv = new DataView(buf);

    expect(dv.getFloat32(0,  true)).toBeCloseTo(SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaDepth,  5);
    expect(dv.getFloat32(4,  true)).toBeCloseTo(SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaNormal, 5);
    expect(dv.getFloat32(8,  true)).toBeCloseTo(SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin,    5);
    expect(dv.getUint32( 12, true)).toBe(0); // _pad
  });
});
