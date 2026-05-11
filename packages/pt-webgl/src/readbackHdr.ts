/**
 * Float HDR readback helpers for offline denoisers (OIDN) and tooling.
 */

import { ZERO_SAMPLE_COUNT_EPSILON } from './accumulationSampleEpsilon.js';
import type { WebGLRenderer } from 'three';
import type { WebGLRenderTarget } from 'three';

/**
 * Pure stage: row-major RGBA float texels → interleaved RGB (same layout as readback).
 * Use for tests and tooling; `pixelCount` is width × height.
 */
export function accumulationFloatRgbaToRgb(
  rawRgba: Float32Array,
  pixelCount: number,
  divideByAlpha: boolean,
): Float32Array {
  const out = new Float32Array(pixelCount * 3);
  for (let i = 0; i < pixelCount; i += 1) {
    const j = i * 4;
    let r: number = rawRgba[j] ?? 0;
    let g: number = rawRgba[j + 1] ?? 0;
    let b: number = rawRgba[j + 2] ?? 0;
    const a: number = rawRgba[j + 3] ?? 0;
    if (divideByAlpha) {
      if (a <= ZERO_SAMPLE_COUNT_EPSILON) {
        r = 0;
        g = 0;
        b = 0;
      } else {
        const ia = 1 / a;
        r *= ia;
        g *= ia;
        b *= ia;
      }
    }
    const o = i * 3;
    out[o] = r;
    out[o + 1] = g;
    out[o + 2] = b;
  }
  return out;
}

/**
 * Reads RGB(A) half/float accumulation into a Float32 RGB buffer (row-major, interleaved RGB).
 * When `divideByAlpha` is true, each texel is interpreted as sum(rgb)/count(alpha) HDR.
 * Texels with α ≤ ZERO_SAMPLE_COUNT_EPSILON are treated as **no samples** and yield black (avoids denoiser spikes).
 */
export function readAccumulationRgbFloat(
  renderer: WebGLRenderer,
  target: WebGLRenderTarget,
  width: number,
  height: number,
  divideByAlpha: boolean,
): Float32Array {
  const pixelCount = width * height;
  const px = pixelCount * 4;
  const raw = new Float32Array(px);
  renderer.readRenderTargetPixels(target, 0, 0, width, height, raw);
  return accumulationFloatRgbaToRgb(raw, pixelCount, divideByAlpha);
}
