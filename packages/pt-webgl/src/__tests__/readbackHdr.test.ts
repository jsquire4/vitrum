import { describe, expect, it, vi } from 'vitest';
import type { WebGLRenderer } from 'three';
import type { WebGLRenderTarget } from 'three';
import { accumulationFloatRgbaToRgb, readAccumulationRgbFloat } from '../readbackHdr.js';
import { ZERO_SAMPLE_COUNT_EPSILON } from '../accumulationSampleEpsilon.js';

describe('accumulationFloatRgbaToRgb', () => {
  it('divides sum RGB by sample count when divideByAlpha is true', () => {
    const raw = new Float32Array([
      6, 12, 18, 3,
      1, 2, 4, 1,
    ]);
    const out = accumulationFloatRgbaToRgb(raw, 2, true);
    expect(Array.from(out)).toEqual([2, 4, 6, 1, 2, 4]);
  });

  it('outputs black when count is zero or below epsilon under divideByAlpha', () => {
    const raw = new Float32Array([
      99, 99, 99, 0,
      0, 0, 0, 1e-7,
    ]);
    const out = accumulationFloatRgbaToRgb(raw, 2, true);
    expect(Array.from(out)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('treats alpha at ZERO_SAMPLE_COUNT_EPSILON as no samples', () => {
    const raw = new Float32Array([
      99, 99, 99, ZERO_SAMPLE_COUNT_EPSILON,
      6, 12, 18, 3,
    ]);
    const out = accumulationFloatRgbaToRgb(raw, 2, true);
    expect(Array.from(out)).toEqual([0, 0, 0, 2, 4, 6]);
  });

  it('passes through RGB unchanged when divideByAlpha is false', () => {
    const raw = new Float32Array([1.5, 2.5, 3.5, 0, 9, 8, 7, 100]);
    const out = accumulationFloatRgbaToRgb(raw, 2, false);
    expect(Array.from(out)).toEqual([1.5, 2.5, 3.5, 9, 8, 7]);
  });
});

describe('readAccumulationRgbFloat', () => {
  it('reads render target pixels then applies accumulationFloatRgbaToRgb', () => {
    const fill = new Float32Array([10, 20, 30, 5, 0, 0, 0, 0]);
    const renderer = {
      readRenderTargetPixels: vi.fn((_t: WebGLRenderTarget, _x: number, _y: number, w: number, h: number, buf: Float32Array) => {
        expect(w).toBe(2);
        expect(h).toBe(1);
        buf.set(fill);
      }),
    } as unknown as WebGLRenderer;

    const target = {} as WebGLRenderTarget;
    const out = readAccumulationRgbFloat(renderer, target, 2, 1, true);
    expect(renderer.readRenderTargetPixels).toHaveBeenCalledOnce();
    expect(Array.from(out)).toEqual([2, 4, 6, 0, 0, 0]);
  });
});
