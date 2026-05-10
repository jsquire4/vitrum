/**
 * Float HDR readback helpers for offline denoisers (OIDN) and tooling.
 */

import type { WebGLRenderer } from 'three';
import type { WebGLRenderTarget } from 'three';

/**
 * Reads RGB(A) half/float accumulation into a Float32 RGB buffer (row-major, interleaved RGB).
 * When `divideByAlpha` is true, each texel is interpreted as sum(rgb)/count(alpha) HDR.
 */
export function readAccumulationRgbFloat(
  renderer: WebGLRenderer,
  target: WebGLRenderTarget,
  width: number,
  height: number,
  divideByAlpha: boolean,
): Float32Array {
  const px = width * height * 4;
  const raw = new Float32Array(px);
  renderer.readRenderTargetPixels(target, 0, 0, width, height, raw);

  const out = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i += 1) {
    const j = i * 4;
    let r: number = raw[j] ?? 0;
    let g: number = raw[j + 1] ?? 0;
    let b: number = raw[j + 2] ?? 0;
    const a: number = raw[j + 3] ?? 0;
    if (divideByAlpha) {
      const ia = 1 / Math.max(a, 1e-6);
      r *= ia;
      g *= ia;
      b *= ia;
    }
    const o = i * 3;
    out[o] = r;
    out[o + 1] = g;
    out[o + 2] = b;
  }
  return out;
}
