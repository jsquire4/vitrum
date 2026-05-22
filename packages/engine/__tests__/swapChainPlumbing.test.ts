// Verifies that attachVitrum plumbs `swapChainView` + `swapChainFormat`
// into FrameInput when (and only when) the canvas has a WebGPU context.
//
// HybridEngine.renderFrame skips frames outright when input.swapChainView
// is undefined (HybridEngine.ts:979); per A2 the engine facade must wire
// the per-frame view so a WebGPU-backed host using attachVitrum doesn't
// see a black canvas.
//
// We don't drive a real RAF loop here — that requires a DOM + WebGPU device,
// covered by the shader-compile-ci smoke. Instead we exercise the two pure
// helpers attachVitrum delegates to: `detectWebGPUSwapChain` (per-engine
// signal, called once) and `acquireSwapChainView` (per-frame, called inside
// the tick). Together they pin the FrameInput shape promised by A2.

import { describe, it, expect, vi } from 'vitest';
import {
  detectWebGPUSwapChain,
  acquireSwapChainView,
  toPhysicalViewport,
} from '../src/lifecycle/vanilla.js';

// ──────────────────────────────────────────────────────────────────────────
// Test helpers — fake canvas, fake GPUCanvasContext.

function makeFakeWebGPUContext(opts?: {
  format?: GPUTextureFormat;
  getCurrentTextureThrows?: boolean;
  hasGetConfiguration?: boolean;
}): GPUCanvasContext {
  const fakeView = { __fakeView: true } as unknown as GPUTextureView;
  const fakeTexture = {
    createView: vi.fn().mockReturnValue(fakeView),
  } as unknown as GPUTexture;
  const ctx: Partial<GPUCanvasContext> & {
    getConfiguration?: () => { format?: GPUTextureFormat } | null;
  } = {
    getCurrentTexture: vi.fn(() => {
      if (opts?.getCurrentTextureThrows) throw new Error('boom');
      return fakeTexture;
    }),
  };
  if (opts?.hasGetConfiguration !== false) {
    // Cast through unknown — the real GPUCanvasConfigurationOut has many
    // required fields (viewFormats, device, usage, colorSpace, alphaMode);
    // detectWebGPUSwapChain only reads `format`.
    ctx.getConfiguration = (() => ({
      format: opts?.format ?? 'bgra8unorm',
    })) as unknown as GPUCanvasContext['getConfiguration'];
  }
  return ctx as GPUCanvasContext;
}

function makeFakeCanvas(ctx: GPUCanvasContext | null): HTMLCanvasElement {
  const fakeCanvas = {
    getContext: vi.fn((kind: string) => (kind === 'webgpu' ? ctx : null)),
  };
  return fakeCanvas as unknown as HTMLCanvasElement;
}

// ──────────────────────────────────────────────────────────────────────────

describe('detectWebGPUSwapChain', () => {
  it('returns the context + format when the canvas has a configured WebGPU context', () => {
    const ctx = makeFakeWebGPUContext({ format: 'bgra8unorm' });
    const canvas = makeFakeCanvas(ctx);
    const result = detectWebGPUSwapChain(canvas);
    expect(result.context).toBe(ctx);
    expect(result.format).toBe('bgra8unorm');
  });

  it('honours rgba8unorm if the context was configured with it', () => {
    const ctx = makeFakeWebGPUContext({ format: 'rgba8unorm' });
    const canvas = makeFakeCanvas(ctx);
    const result = detectWebGPUSwapChain(canvas);
    expect(result.format).toBe('rgba8unorm');
  });

  it('returns context:null when canvas.getContext("webgpu") returns null (WebGL host)', () => {
    const canvas = makeFakeCanvas(null);
    const result = detectWebGPUSwapChain(canvas);
    expect(result.context).toBeNull();
    expect(result.format).toBeUndefined();
  });

  it('falls back to bgra8unorm when getConfiguration is unavailable', () => {
    const ctx = makeFakeWebGPUContext({ hasGetConfiguration: false });
    const canvas = makeFakeCanvas(ctx);
    const result = detectWebGPUSwapChain(canvas);
    expect(result.context).toBe(ctx);
    // Without getConfiguration() (older browsers) the helper falls back to
    // either getPreferredCanvasFormat() or the bgra8unorm safe default.
    expect(typeof result.format).toBe('string');
    expect(result.format).toMatch(/^(bgra8unorm|rgba8unorm)$/);
  });

  it('returns context:null when getContext throws (e.g. test env)', () => {
    const canvas = {
      getContext: vi.fn(() => { throw new Error('no webgpu'); }),
    } as unknown as HTMLCanvasElement;
    const result = detectWebGPUSwapChain(canvas);
    expect(result.context).toBeNull();
  });
});

describe('acquireSwapChainView', () => {
  it('returns a GPUTextureView when the context produces a texture', () => {
    const ctx = makeFakeWebGPUContext();
    const view = acquireSwapChainView(ctx);
    expect(view).toBeDefined();
    // Verify the call chain — getCurrentTexture invoked exactly once per
    // call (per WebGPU spec, must be re-acquired each frame).
    expect(ctx.getCurrentTexture).toHaveBeenCalledTimes(1);
  });

  it('returns undefined when context is null (WebGL host)', () => {
    const view = acquireSwapChainView(null);
    expect(view).toBeUndefined();
  });

  it('returns undefined when getCurrentTexture throws (e.g. zero-size canvas)', () => {
    const ctx = makeFakeWebGPUContext({ getCurrentTextureThrows: true });
    const view = acquireSwapChainView(ctx);
    expect(view).toBeUndefined();
  });

  it('re-acquires on each call (no caching across frames per WebGPU spec)', () => {
    const ctx = makeFakeWebGPUContext();
    acquireSwapChainView(ctx);
    acquireSwapChainView(ctx);
    acquireSwapChainView(ctx);
    expect(ctx.getCurrentTexture).toHaveBeenCalledTimes(3);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// FrameInput shape pin — the integration the A2 spec asks for. We don't run
// the rAF loop; we just exercise the same composition attachVitrum performs.

describe('FrameInput swap-chain plumbing (A2)', () => {
  it('FrameInput contains swapChainView + swapChainFormat when canvas is WebGPU', () => {
    const ctx = makeFakeWebGPUContext({ format: 'bgra8unorm' });
    const canvas = makeFakeCanvas(ctx);
    const { context, format } = detectWebGPUSwapChain(canvas);
    const swapChainView = acquireSwapChainView(context);
    // Same spread attachVitrum does at the FrameInput construction site.
    const frameInput = {
      ...(swapChainView != null ? { swapChainView, swapChainFormat: format } : {}),
    };
    expect(frameInput.swapChainView).toBeDefined();
    expect(frameInput.swapChainFormat).toBe('bgra8unorm');
  });

  it('FrameInput omits swap-chain fields when canvas is WebGL (no webgpu context)', () => {
    const canvas = makeFakeCanvas(null);
    const { context, format } = detectWebGPUSwapChain(canvas);
    const swapChainView = acquireSwapChainView(context);
    const frameInput = {
      ...(swapChainView != null ? { swapChainView, swapChainFormat: format } : {}),
    } as { swapChainView?: unknown; swapChainFormat?: unknown };
    expect(frameInput.swapChainView).toBeUndefined();
    expect(frameInput.swapChainFormat).toBeUndefined();
  });
});

describe('viewport contract plumbing', () => {
  it('converts CSS dimensions to physical pixels with DPR', () => {
    const viewport = toPhysicalViewport(640.8, 359.4, 2);
    expect(viewport.width).toBe(1281);
    expect(viewport.height).toBe(718);
    expect(viewport.devicePixelRatio).toBe(2);
  });

  it('guards non-finite DPR and enforces minimum dimensions', () => {
    const viewport = toPhysicalViewport(0, 0, Number.NaN);
    expect(viewport.width).toBe(1);
    expect(viewport.height).toBe(1);
    expect(viewport.devicePixelRatio).toBe(1);
  });
});
