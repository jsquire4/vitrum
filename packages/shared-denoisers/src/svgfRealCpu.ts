/**
 * svgfRealCpu.ts — CPU emulation of the SVGF compute passes (test oracles).
 *
 * These functions mirror the per-pixel logic of the corresponding WGSL
 * kernels in `wgsl/svgfReprojection.wgsl`, `wgsl/svgfVarianceFromMoments.wgsl`,
 * and `wgsl/svgf7x7SpatialFallback.wgsl`. They exist purely so the test
 * suite can verify the algorithm without spinning a GPU device.
 *
 * Extracted from svgfRealWebGPU.ts by the W4-A7 refactor (sweep H8).
 */

import { SVGF_REPROJ_DEFAULT_UNIFORMS } from './svgfRealBindings.js';
// Rec.709 luminance hoisted to @vitrum/shared-samplers (no THREE dep, no
// peer-dep load). Local alias `lumCPU` kept so the dense call-site usage
// below reads as it always has.
import { luminance as lumCPU } from '@vitrum/shared-samplers';

/**
 * CPU emulation of svgfReprojMain — for unit tests only.
 *
 * Performs the per-pixel bilinear reprojection + disocclusion test + EMA
 * blend identical to the WGSL kernel, operating on flat Float32Array/Uint32Array
 * buffers instead of GPU textures.
 *
 * @returns Updated { color, historyLength, moments } arrays.
 */
export interface SVGFReprojCPUInput {
  /** Current-frame RGB, row-major, length W*H*3. */
  readonly currColor:       Float32Array;
  /** Previous-frame RGB (post-EMA), row-major, length W*H*3. */
  readonly prevColor:       Float32Array;
  /** Screen-space motion vector (pixel delta), RG interleaved, length W*H*2. */
  readonly motionVec:       Float32Array;
  /** Current linear depth per pixel, length W*H. */
  readonly currDepth:       Float32Array;
  /** Current world-space normals, packed 0..1 in XYZ, row-major, length W*H*3. */
  readonly currNormal:      Float32Array;
  /** Current object ID per pixel (u32), length W*H. */
  readonly currObjId:       Uint32Array;
  /** Previous frame depth per pixel, length W*H. */
  readonly prevDepth:       Float32Array;
  /** Previous frame normals, packed 0..1, length W*H*3. */
  readonly prevNormal:      Float32Array;
  /** Previous object ID per pixel, length W*H. */
  readonly prevObjId:       Uint32Array;
  /** Previous history-length per pixel (u32), length W*H. */
  readonly historyLengthIn: Uint32Array;
  /** Previous moments M1, M2 interleaved, length W*H*2. */
  readonly momentsIn:       Float32Array;
  readonly width:  number;
  readonly height: number;
  readonly sigmaDepth?:  number;
  readonly sigmaNormal?: number;
  readonly alphaMin?:    number;
}

export interface SVGFReprojCPUOutput {
  readonly colorOut:        Float32Array;
  readonly historyLengthOut: Uint32Array;
  readonly momentsOut:      Float32Array;
}

/**
 * CPU emulation of svgfReprojMain (used by unit tests).
 * Identical logic to the WGSL kernel; operates on flat arrays.
 */
export function svgfReprojCPU(input: SVGFReprojCPUInput): SVGFReprojCPUOutput {
  const W = input.width;
  const H = input.height;
  const sigmaDepth  = input.sigmaDepth  ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaDepth;
  const sigmaNormal = input.sigmaNormal ?? SVGF_REPROJ_DEFAULT_UNIFORMS.sigmaNormal;
  const alphaMin    = input.alphaMin    ?? SVGF_REPROJ_DEFAULT_UNIFORMS.alphaMin;

  const colorOut   = new Float32Array(W * H * 3);
  const histOut    = new Uint32Array(W * H);
  const momentsOut = new Float32Array(W * H * 2);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const mvX = input.motionVec[pi * 2]     ?? 0;
      const mvY = input.motionVec[pi * 2 + 1] ?? 0;

      const prevFX = x + mvX;
      const prevFY = y + mvY;
      const prevX  = Math.floor(prevFX);
      const prevY  = Math.floor(prevFY);
      const fracX  = prevFX - prevX;
      const fracY  = prevFY - prevY;

      const zCurr = input.currDepth[pi] ?? 0;
      const nCurrX = (input.currNormal[pi * 3]     ?? 0) * 2 - 1;
      const nCurrY = (input.currNormal[pi * 3 + 1] ?? 0) * 2 - 1;
      const nCurrZ = (input.currNormal[pi * 3 + 2] ?? 0) * 2 - 1;
      const objIdCurr = input.currObjId[pi] ?? 0;

      // Bilinear taps
      const tapOffsets: [number, number][] = [[0,0],[1,0],[0,1],[1,1]];
      const tapWeights = [
        (1 - fracX) * (1 - fracY),
        fracX       * (1 - fracY),
        (1 - fracX) * fracY,
        fracX       * fracY,
      ];

      let accR = 0, accG = 0, accB = 0;
      let accM1 = 0, accM2 = 0, accH = 0, accW = 0;

      for (let t = 0; t < 4; t++) {
        const [ox, oy] = tapOffsets[t]!;
        const tx = prevX + ox;
        const ty = prevY + oy;
        if (tx < 0 || ty < 0 || tx >= W || ty >= H) continue;

        const ti = ty * W + tx;
        const zPrev = input.prevDepth[ti] ?? 0;
        const nPX = (input.prevNormal[ti * 3]     ?? 0) * 2 - 1;
        const nPY = (input.prevNormal[ti * 3 + 1] ?? 0) * 2 - 1;
        const nPZ = (input.prevNormal[ti * 3 + 2] ?? 0) * 2 - 1;
        const oPrev = input.prevObjId[ti] ?? 0;

        // Disocclusion test (Eq. 2)
        if (Math.abs(zCurr - zPrev) > sigmaDepth * Math.max(zCurr, zPrev) + 1e-4) continue;
        const nDot = nCurrX * nPX + nCurrY * nPY + nCurrZ * nPZ;
        if (nDot < sigmaNormal) continue;
        if (oPrev !== objIdCurr) continue;

        const w = tapWeights[t]!;
        accR += (input.prevColor[ti * 3]     ?? 0) * w;
        accG += (input.prevColor[ti * 3 + 1] ?? 0) * w;
        accB += (input.prevColor[ti * 3 + 2] ?? 0) * w;
        accM1 += (input.momentsIn[ti * 2]     ?? 0) * w;
        accM2 += (input.momentsIn[ti * 2 + 1] ?? 0) * w;
        accH  += (input.historyLengthIn[ti] ?? 0) * w;
        accW  += w;
      }

      const currR = input.currColor[pi * 3]     ?? 0;
      const currG = input.currColor[pi * 3 + 1] ?? 0;
      const currB = input.currColor[pi * 3 + 2] ?? 0;

      let newH: number;
      let alpha: number;
      let prevR = 0, prevG = 0, prevB = 0;
      let prevM1 = 0, prevM2 = 0;

      if (accW > 1e-6) {
        const invW = 1 / accW;
        prevR = accR * invW;
        prevG = accG * invW;
        prevB = accB * invW;
        prevM1 = accM1 * invW;
        prevM2 = accM2 * invW;
        newH = Math.trunc(accH * invW) + 1;
        alpha = Math.max(alphaMin, 1 / (newH + 1));
      } else {
        newH = 1;
        alpha = 1;
      }

      const blendR = alpha * currR + (1 - alpha) * prevR;
      const blendG = alpha * currG + (1 - alpha) * prevG;
      const blendB = alpha * currB + (1 - alpha) * prevB;
      const lCurr  = lumCPU(currR, currG, currB);
      const newM1  = alpha * lCurr          + (1 - alpha) * prevM1;
      const newM2  = alpha * lCurr * lCurr  + (1 - alpha) * prevM2;

      colorOut[pi * 3]     = blendR;
      colorOut[pi * 3 + 1] = blendG;
      colorOut[pi * 3 + 2] = blendB;
      histOut[pi]          = newH;
      momentsOut[pi * 2]   = newM1;
      momentsOut[pi * 2+1] = newM2;
    }
  }

  return { colorOut, historyLengthOut: histOut, momentsOut };
}

/**
 * CPU emulation of svgfVarianceFromMomentsMain.
 * Returns a Float32Array of scalar variance per pixel (length W*H).
 */
export function svgfVarianceFromMomentsCPU(opts: {
  readonly momentsIn:   Float32Array; // M1,M2 interleaved, length W*H*2
  readonly historyIn:   Uint32Array;  // length W*H
  readonly width:       number;
  readonly height:      number;
  readonly historyMin?: number;       // default 4
}): Float32Array {
  const { width: W, height: H, momentsIn, historyIn } = opts;
  const threshold = opts.historyMin ?? 4;
  const out = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const h = historyIn[i] ?? 0;
    if (h >= threshold) {
      const m1 = momentsIn[i * 2]     ?? 0;
      const m2 = momentsIn[i * 2 + 1] ?? 0;
      out[i] = Math.max(0, m2 - m1 * m1);
    }
  }
  return out;
}

/**
 * CPU emulation of svgf7x7FallbackMain.
 * Returns a Float32Array of scalar variance per pixel (length W*H),
 * merging spatial estimates for pixels with insufficient history.
 */
export function svgf7x7FallbackCPU(opts: {
  readonly currColor:   Float32Array; // RGB row-major, length W*H*3
  readonly historyIn:   Uint32Array;  // length W*H
  readonly varianceIn:  Float32Array; // from svgfVarianceFromMomentsCPU, length W*H
  readonly width:       number;
  readonly height:      number;
  readonly historyMin?: number;       // default 4
}): Float32Array {
  const { width: W, height: H, currColor, historyIn, varianceIn } = opts;
  const threshold = opts.historyMin ?? 4;
  const out = new Float32Array(W * H);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pi = y * W + x;
      const h = historyIn[pi] ?? 0;

      if (h >= threshold) {
        out[pi] = varianceIn[pi] ?? 0;
        continue;
      }

      // 7×7 spatial variance estimate
      let sumL = 0, sumL2 = 0, n = 0;
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const ni = ny * W + nx;
          const lum = lumCPU(
            currColor[ni * 3]     ?? 0,
            currColor[ni * 3 + 1] ?? 0,
            currColor[ni * 3 + 2] ?? 0,
          );
          sumL  += lum;
          sumL2 += lum * lum;
          n++;
        }
      }
      if (n > 1) {
        const mean = sumL / n;
        out[pi] = Math.max(0, sumL2 / n - mean * mean);
      }
    }
  }

  return out;
}
