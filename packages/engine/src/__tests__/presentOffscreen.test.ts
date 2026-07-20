// presentOffscreen.test.ts — R2 / V1-1 presentation-gap regression pins.
//
// Verifies the offscreen-backend blit wiring in attachVitrum without a real GPU:
//   (1) a presenter blits the FrameOutput texture to the canvas WebGPU context
//       for backends that expose getPresentationSource() (offscreen-texture);
//   (2) exactly one render pass is encoded per rendered frame;
//   (3) NO presenter is constructed for a swapchain backend that presents itself
//       (getPresentationSource absent / returns null);
//   (4) after a simulated progressive handoff the presenter blits the CONVERGED
//       (pt-webgpu) source, not the frozen realtime texture;
//   (5) teardown disposes the presenter.
//
// Real-GPU present correctness is validated at the T1 GPU smoke / browser check;
// this pins the host-side wiring on a stub GPUDevice/GPUCanvasContext.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Engine, FrameOutput, Scene } from '@vitrum/core';

const createEngineMock = vi.hoisted(() => vi.fn());

vi.mock('../createEngine.js', () => ({
  createEngine: createEngineMock,
}));

import { attachVitrum } from '../lifecycle/vanilla.js';
import { createOffscreenPresenter } from '../presentOffscreen.js';

const sceneA: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };

const identityElements = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function makeCamera() {
  return {
    updateMatrixWorld: vi.fn(),
    matrixWorldInverse: { elements: identityElements },
    projectionMatrix: { elements: identityElements },
    position: { x: 0, y: 0, z: 0 },
  };
}

/** A stub GPUCanvasContext recording configure/getCurrentTexture calls. */
function makeContext() {
  const currentView = { __view: true };
  const currentTexture = { createView: vi.fn(() => currentView) };
  return {
    configure: vi.fn(),
    getCurrentTexture: vi.fn(() => currentTexture),
    __currentTexture: currentTexture,
  } as unknown as GPUCanvasContext & { configure: ReturnType<typeof vi.fn> };
}

/** A stub GPUDevice recording pipeline/pass/submit activity. */
function makeDevice() {
  const passEncoder = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    draw: vi.fn(),
    end: vi.fn(),
  };
  const commandEncoder = {
    beginRenderPass: vi.fn(() => passEncoder),
    finish: vi.fn(() => ({ __cmd: true })),
  };
  const device = {
    createShaderModule: vi.fn(() => ({ __module: true })),
    createRenderPipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({ __bgl: true })) })),
    createSampler: vi.fn(() => ({ __sampler: true })),
    createBindGroup: vi.fn(() => ({ __bg: true })),
    createCommandEncoder: vi.fn(() => commandEncoder),
    queue: { submit: vi.fn() },
  };
  return { device: device as unknown as GPUDevice, passEncoder, commandEncoder };
}

/** Build an engine stub. `presentSource` when provided is returned by
 *  getPresentationSource(); omit the method entirely to model a swapchain
 *  backend that presents itself. */
function makeEngine(opts: {
  presentSource?: { device: unknown; texture: unknown } | null;
  omitPresent?: boolean;
} = {}) {
  const rendered: FrameOutput = {
    kind: 'rendered',
    primaryRadiance: { __tex: true } as never,
    samplesAccumulated: 1,
    isConverged: false,
  };
  const engine: Record<string, unknown> = {
    backendId: 'pt-webgpu',
    state: 'ready',
    capabilities: { presentationMode: 'offscreen-texture' },
    setScene: vi.fn(),
    renderFrame: vi.fn(() => rendered),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    getScene: vi.fn(() => sceneA),
    onFrame: vi.fn(() => vi.fn()),
    onProgress: vi.fn(() => vi.fn()),
    onError: vi.fn(() => vi.fn()),
  };
  if (!opts.omitPresent) {
    engine.getPresentationSource = vi.fn(() => opts.presentSource ?? null);
  }
  return engine as unknown as Engine;
}

describe('attachVitrum offscreen presentation (R2 / V1-1)', () => {
  const originalRaf = globalThis.requestAnimationFrame;
  const originalCancelRaf = globalThis.cancelAnimationFrame;
  let tickCb: FrameRequestCallback | undefined;
  let ctx: ReturnType<typeof makeContext>;

  function makeCanvas(): HTMLCanvasElement {
    return {
      width: 300,
      height: 150,
      clientWidth: 300,
      clientHeight: 150,
      getContext: vi.fn((kind: string) => (kind === 'webgpu' ? ctx : null)),
    } as unknown as HTMLCanvasElement;
  }

  beforeEach(() => {
    createEngineMock.mockReset();
    ctx = makeContext();
    tickCb = undefined;
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      // Capture the first tick callback but do NOT auto-run it (the test drives frames).
      value: vi.fn((cb: FrameRequestCallback) => { tickCb = cb; return 1; }),
      configurable: true,
    });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', {
      value: vi.fn(),
      configurable: true,
    });
    // navigator.gpu.getPreferredCanvasFormat for the presenter.
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } },
      configurable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'requestAnimationFrame', { value: originalRaf, configurable: true });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', { value: originalCancelRaf, configurable: true });
  });

  it('blits the FrameOutput texture and encodes exactly one render pass per offscreen frame', async () => {
    const { device, commandEncoder } = makeDevice();
    const tex = { __srcTex: true, createView: vi.fn(() => ({ __srcView: true })) };
    const engine = makeEngine({ presentSource: { device, texture: tex } });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      engine: engine as never,
      scene: sceneA,
      camera: makeCamera(),
    });

    expect(tickCb).toBeTypeOf('function');
    tickCb!(0);

    // Presenter constructed (configured the webgpu context) + one pass + submit.
    expect(ctx.configure).toHaveBeenCalledTimes(1);
    expect(commandEncoder.beginRenderPass).toHaveBeenCalledTimes(1);
    expect(device.queue.submit).toHaveBeenCalledTimes(1);
    // Source texture view was bound.
    expect(tex.createView).toHaveBeenCalled();

    handle.dispose();
  });

  it('does NOT construct a presenter for a swapchain backend that presents itself', async () => {
    const engine = makeEngine({ omitPresent: true });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      engine: engine as never,
      scene: sceneA,
      camera: makeCamera(),
    });

    tickCb!(0);

    // No WebGPU context configured → no presenter built.
    expect(ctx.configure).not.toHaveBeenCalled();

    handle.dispose();
  });

  it('after a progressive handoff, blits the converged (offscreen) source not the realtime texture', async () => {
    const { device } = makeDevice();
    const realtimeTex = { __realtime: true, createView: vi.fn(() => ({})) };
    const convergedTex = { __converged: true, createView: vi.fn(() => ({})) };

    // Model the progressive wrapper: getPresentationSource returns null while
    // realtime is driving (swapchain self-present), then the converged source.
    let source: { device: unknown; texture: unknown } | null = null;
    const engine = makeEngine();
    (engine as unknown as { getPresentationSource: () => unknown }).getPresentationSource = () => source;

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      engine: engine as never,
      scene: sceneA,
      camera: makeCamera(),
    });

    // Frame 1: realtime phase — source null, nothing blitted.
    source = null;
    tickCb!(0);
    expect(ctx.configure).not.toHaveBeenCalled();
    expect(realtimeTex.createView).not.toHaveBeenCalled();

    // Frame 2: handoff — converged source.
    source = { device, texture: convergedTex };
    tickCb!(16);
    expect(ctx.configure).toHaveBeenCalledTimes(1);
    expect(convergedTex.createView).toHaveBeenCalled();
    expect(realtimeTex.createView).not.toHaveBeenCalled();

    handle.dispose();
  });

  it('teardown disposes the presenter', async () => {
    const { device } = makeDevice();
    const tex = { createView: vi.fn(() => ({})) };
    const engine = makeEngine({ presentSource: { device, texture: tex } });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      engine: engine as never,
      scene: sceneA,
      camera: makeCamera(),
    });
    tickCb!(0);
    // Presenter now exists; disposing the handle must not throw and must stop
    // presenting on subsequent (defensive) calls.
    handle.dispose();
    // A second render after dispose (stopped loop) does nothing — no extra submit.
    const submitsBefore = (device.queue.submit as ReturnType<typeof vi.fn>).mock.calls.length;
    tickCb!(32);
    expect((device.queue.submit as ReturnType<typeof vi.fn>).mock.calls.length).toBe(submitsBefore);
  });
});

describe('createOffscreenPresenter (R2 / V1-1)', () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } },
      configurable: true,
    });
  });

  it('configures the context, builds one pipeline, and encodes a single fullscreen-triangle pass on present', () => {
    const { device, passEncoder, commandEncoder } = makeDevice();
    const ctx = makeContext();
    const canvas = {} as HTMLCanvasElement;

    const presenter = createOffscreenPresenter({ device, canvas, context: ctx });
    expect(ctx.configure).toHaveBeenCalledTimes(1);
    expect(device.createRenderPipeline).toHaveBeenCalledTimes(1);
    expect(device.createSampler).toHaveBeenCalledTimes(1);

    const source = { createView: vi.fn(() => ({ __v: true })) } as unknown as GPUTexture;
    presenter.present(source);

    expect(commandEncoder.beginRenderPass).toHaveBeenCalledTimes(1);
    expect(passEncoder.draw).toHaveBeenCalledWith(3);
    expect(device.queue.submit).toHaveBeenCalledTimes(1);

    // After dispose, present is a no-op.
    presenter.dispose();
    presenter.present(source);
    expect(device.queue.submit).toHaveBeenCalledTimes(1);
  });
});
