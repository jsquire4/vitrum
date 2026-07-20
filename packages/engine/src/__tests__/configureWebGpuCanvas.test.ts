// configureWebGpuCanvas.test.ts — characterizes the shared WebGPU canvas-context
// configure helper extracted from createEngine's two WebGPU constructors
// (walkaround-hybrid + pt-webgpu). Both call sites must configure the canvas
// identically; this pins the exact configure() call args.

import { describe, it, expect, vi, afterEach } from 'vitest';
import { configureWebGpuCanvas } from '../configureWebGpuCanvas.js';

const FAKE_DEVICE = { destroy() {} } as unknown as GPUDevice;

function makeCanvas(ctx: { configure: (...args: unknown[]) => unknown } | null): HTMLCanvasElement {
  return {
    getContext: (kind: string) => (kind === 'webgpu' ? ctx : null),
  } as unknown as HTMLCanvasElement;
}

const origNavigator = globalThis.navigator;

afterEach(() => {
  Object.defineProperty(globalThis, 'navigator', { value: origNavigator, configurable: true });
});

describe('configureWebGpuCanvas', () => {
  it('configures the webgpu context with device, preferred format, opaque alpha', () => {
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { getPreferredCanvasFormat: () => 'rgba8unorm' as GPUTextureFormat } },
      configurable: true,
    });
    const configure = vi.fn();
    const canvas = makeCanvas({ configure });
    configureWebGpuCanvas(canvas, FAKE_DEVICE);
    expect(configure).toHaveBeenCalledTimes(1);
    expect(configure).toHaveBeenCalledWith({
      device: FAKE_DEVICE,
      format: 'rgba8unorm',
      alphaMode: 'opaque',
    });
  });

  it('falls back to bgra8unorm when getPreferredCanvasFormat is absent', () => {
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const configure = vi.fn();
    configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE);
    expect(configure).toHaveBeenCalledWith({
      device: FAKE_DEVICE,
      format: 'bgra8unorm',
      alphaMode: 'opaque',
    });
  });

  it('is a no-op (no throw) when the canvas has no webgpu context', () => {
    expect(() => configureWebGpuCanvas(makeCanvas(null), FAKE_DEVICE)).not.toThrow();
  });

  it('swallows a throwing configure (best-effort)', () => {
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const configure = vi.fn(() => { throw new Error('boom'); });
    expect(() => configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE)).not.toThrow();
  });

  it('reports throwing configure through the optional callback', () => {
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const err = new Error('lost');
    const configure = vi.fn(() => { throw err; });
    const onError = vi.fn();
    configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE, onError);
    expect(onError).toHaveBeenCalledWith(err);
  });

  it('guards throwing optional error callbacks', () => {
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const configure = vi.fn(() => { throw new Error('configure failed'); });
    const onError = vi.fn(() => { throw new Error('host callback failed'); });
    expect(() => configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE, onError)).not.toThrow();
    expect(onError).toHaveBeenCalledTimes(1);
  });

  // V1-7 — swap-chain-required backends must fail fast on configure failure so
  // they do not render a permanently-black canvas; offscreen backends keep the
  // historical swallow behaviour.
  it('offscreen path (required omitted/false) still swallows a configure failure', () => {
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const err = new Error('lost');
    const configure = vi.fn(() => { throw err; });
    const onError = vi.fn();
    expect(() => configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE, onError)).not.toThrow();
    expect(() => configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE, onError, { required: false })).not.toThrow();
    expect(onError).toHaveBeenCalledWith(err);
  });

  it('required:true re-throws a configure failure (after invoking onError)', () => {
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const err = new Error('swapchain configure failed');
    const configure = vi.fn(() => { throw err; });
    const onError = vi.fn();
    expect(() =>
      configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE, onError, { required: true }),
    ).toThrow(err);
    expect(onError).toHaveBeenCalledWith(err);
  });

  it('required:true does not throw on the happy path', () => {
    Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
    const configure = vi.fn();
    expect(() =>
      configureWebGpuCanvas(makeCanvas({ configure }), FAKE_DEVICE, undefined, { required: true }),
    ).not.toThrow();
    expect(configure).toHaveBeenCalledTimes(1);
  });
});
