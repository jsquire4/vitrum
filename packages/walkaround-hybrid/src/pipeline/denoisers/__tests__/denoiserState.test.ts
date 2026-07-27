import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import type { InferenceGraph } from '../../../neural/InferenceGraph.js';
import { AtrousDenoiser } from '../atrous.js';
import { AtrousVarianceDenoiser } from '../atrousVariance.js';
import { BmfrDenoiser } from '../bmfr.js';
import { DenoiserRegistry } from '../index.js';
import type { DenoiserDispatchContext, DenoiserState } from '../index.js';
import { NeuralDenoiser } from '../neural.js';
import { NoneDenoiser } from '../none.js';
import { OIDNFinalDenoiser } from '../oidnFinal.js';
import { registerBuiltinDenoisers } from '../registerBuiltinDenoisers.js';
import { SVGFRealDenoiser } from '../svgfReal.js';
import { shouldResetDenoiserHistory } from '../historyReset.js';
import { NEURAL_F32_TENSOR_STORAGE } from '../../../neural/tensorPrecision.js';

const fakeGraph = (): InferenceGraph =>
  ({ run() {}, tensorStorage: NEURAL_F32_TENSOR_STORAGE }) as unknown as InferenceGraph;

describe('Denoiser.state', () => {
  it('reports ready for synchronous denoisers', () => {
    const denoisers = [
      new NoneDenoiser(),
      new AtrousDenoiser(),
      new AtrousVarianceDenoiser(),
      new BmfrDenoiser(),
      new SVGFRealDenoiser(),
    ];

    for (const denoiser of denoisers) {
      expect(denoiser.state()).toEqual({ status: 'ready' } satisfies DenoiserState);
    }
  });

  it('reports a retryable neural failure until graph-backed resources are initialized', () => {
    expect(new NeuralDenoiser().state()).toEqual({
      status: 'fallback',
      reason: 'inference graph not supplied',
    } satisfies DenoiserState);

    const denoiser = new NeuralDenoiser({ inferenceGraph: fakeGraph() });
    expect(denoiser.state()).toEqual({
      status: 'failed',
      reason: 'neural denoiser is not initialized',
      retryable: true,
    } satisfies DenoiserState);
  });

  it('reports neural ready only for a fully published wrapper generation', () => {
    const denoiser = new NeuralDenoiser({ inferenceGraph: fakeGraph() });
    const seam = denoiser as unknown as {
      _device: GPUDevice;
      _graphW: number;
      _graphH: number;
      _packPipeline: GPUComputePipeline;
      _unpackPipeline: GPUComputePipeline;
      _packParamsBuf: GPUBuffer;
      _unpackParamsBuf: GPUBuffer;
      _lifecycleState: 'ready';
      _failureReason: null;
      // The four tensor GPU buffers are now grouped under _tensorBuffers (D4.9).
      _tensorBuffers: {
        noisyBuf: GPUBuffer; albedoBuf: GPUBuffer;
        normalsBuf: GPUBuffer; outputBuf: GPUBuffer;
        outputTex: GPUTexture; width: number; height: number;
      } | null;
      _lastFallbackReason: string | null;
    };
    seam._device = {} as GPUDevice;
    seam._lifecycleState = 'ready';
    seam._failureReason = null;
    seam._graphW = 64;
    seam._graphH = 64;
    seam._packPipeline = {} as GPUComputePipeline;
    seam._unpackPipeline = {} as GPUComputePipeline;
    seam._packParamsBuf = {} as GPUBuffer;
    seam._unpackParamsBuf = {} as GPUBuffer;
    seam._tensorBuffers = {
      noisyBuf: {} as GPUBuffer,
      albedoBuf: {} as GPUBuffer,
      normalsBuf: {} as GPUBuffer,
      outputBuf: {} as GPUBuffer,
      outputTex: {} as GPUTexture,
      width: 64,
      height: 64,
    };

    expect(denoiser.state()).toEqual({ status: 'ready' } satisfies DenoiserState);

    seam._lastFallbackReason = 'size changed from 64x64 to 128x64';
    expect(denoiser.state()).toEqual({ status: 'ready' } satisfies DenoiserState);
  });

  it('routes neural size-mismatch fallback through structured warnings', () => {
    const warnings: EngineWarning[] = [];
    const denoiser = new NeuralDenoiser({
      inferenceGraph: fakeGraph(),
      onWarning: (warning) => warnings.push(warning),
    });
    const hdrColorTexture = {} as GPUTexture;
    const seam = denoiser as unknown as {
      _device: GPUDevice;
      _graphW: number;
      _graphH: number;
      _packPipeline: GPUComputePipeline;
      _unpackPipeline: GPUComputePipeline;
      _packParamsBuf: GPUBuffer;
      _unpackParamsBuf: GPUBuffer;
      _lifecycleState: 'ready';
      _failureReason: null;
      _tensorBuffers: {
        noisyBuf: GPUBuffer; albedoBuf: GPUBuffer;
        normalsBuf: GPUBuffer; outputBuf: GPUBuffer;
        outputTex: GPUTexture; width: number; height: number;
      } | null;
    };
    const device = {} as GPUDevice;
    seam._lifecycleState = 'ready';
    seam._failureReason = null;
    seam._device = device;
    seam._graphW = 64;
    seam._graphH = 64;
    seam._packPipeline = {} as GPUComputePipeline;
    seam._unpackPipeline = {} as GPUComputePipeline;
    seam._packParamsBuf = {} as GPUBuffer;
    seam._unpackParamsBuf = {} as GPUBuffer;
    seam._tensorBuffers = {
      noisyBuf: {} as GPUBuffer,
      albedoBuf: {} as GPUBuffer,
      normalsBuf: {} as GPUBuffer,
      outputBuf: {} as GPUBuffer,
      outputTex: {} as GPUTexture,
      width: 64,
      height: 64,
    };

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => denoiser.dispatch({
        device,
        width: 128,
        height: 64,
        resources: { common: { hdrColorTexture } },
      } as unknown as DenoiserDispatchContext)).toThrow(/size changed/);

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toMatchObject({
        code: 'walkaround-hybrid.neural-size-mismatch-failed',
        backend: 'walkaround-hybrid',
        phase: 'renderFrame',
        method: 'renderFrame',
        details: {
          previousWidth: 64,
          previousHeight: 64,
          width: 128,
          height: 64,
          state: 'failed',
          missing: 'retained model weights',
        },
      });
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('forwards the host warning sink into the registered neural denoiser', () => {
    const onWarning = vi.fn();
    const registry = new DenoiserRegistry();
    registerBuiltinDenoisers(registry, {
      neuralInferenceGraph: fakeGraph(),
      onWarning,
    });

    const neural = registry.lookup('neural') as unknown as {
      _onWarning: ((warning: EngineWarning) => void) | null;
    };
    expect(neural._onWarning).toBe(onWarning);
  });

  it('rejects a zero neural dimension before creating any GPU resource', async () => {
    const createShaderModule = vi.fn();
    const createComputePipelineAsync = vi.fn();
    const createBuffer = vi.fn();
    const denoiser = new NeuralDenoiser({ inferenceGraph: fakeGraph() });

    await expect(denoiser.initialize({
      device: { createShaderModule, createComputePipelineAsync, createBuffer } as unknown as GPUDevice,
      width: 0,
      height: 8,
    } as unknown as Parameters<NeuralDenoiser['initialize']>[0])).rejects.toThrow(
      /unsupported internal render size 0x8/,
    );

    expect(createShaderModule).not.toHaveBeenCalled();
    expect(createComputePipelineAsync).not.toHaveBeenCalled();
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('destroys wrapper buffers created before a later tensor allocation fails', () => {
    vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 });
    const first = { destroy: vi.fn() } as unknown as GPUBuffer;
    const second = { destroy: vi.fn() } as unknown as GPUBuffer;
    const allocationFailure = new Error('forced wrapper allocation failure');
    const createBuffer = vi.fn()
      .mockReturnValueOnce(first)
      .mockReturnValueOnce(second)
      .mockImplementationOnce(() => { throw allocationFailure; });
    const createTexture = vi.fn();
    const denoiser = new NeuralDenoiser({ inferenceGraph: fakeGraph() });
    const seam = denoiser as unknown as {
      _allocTensorBuffers(device: GPUDevice, width: number, height: number): void;
      _tensorBuffers: unknown;
    };

    try {
      expect(() => seam._allocTensorBuffers(
        { createBuffer, createTexture } as unknown as GPUDevice,
        8,
        8,
      )).toThrow(allocationFailure);
      expect(first.destroy).toHaveBeenCalledOnce();
      expect(second.destroy).toHaveBeenCalledOnce();
      expect(createTexture).not.toHaveBeenCalled();
      expect(seam._tensorBuffers).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('fails durably on a zero-size resize without allocating neural tensors', () => {
    const warnings: EngineWarning[] = [];
    const createBuffer = vi.fn();
    const denoiser = new NeuralDenoiser({
      inferenceGraph: fakeGraph(),
      onWarning: warning => warnings.push(warning),
    });
    const seam = denoiser as unknown as {
      _device: GPUDevice;
      _graphW: number;
      _graphH: number;
    };
    seam._device = { createBuffer } as unknown as GPUDevice;
    seam._graphW = 8;
    seam._graphH = 8;

    denoiser.resize(0, 8);

    expect(createBuffer).not.toHaveBeenCalled();
    expect(denoiser.state()).toEqual({
      status: 'failed',
      reason: expect.stringContaining('unsupported neural internal render size 0x8'),
      retryable: false,
    });
    expect(warnings).toEqual([expect.objectContaining({
      code: 'walkaround-hybrid.neural-unsupported-shape-failed',
      method: 'resize',
      details: expect.objectContaining({
        width: 0,
        height: 8,
        neuralAllocationAttempted: false,
        state: 'failed',
      }),
    })]);
  });

  it('reports OIDN disabled, warmup, fallback, in-flight, ready, and failed states', () => {
    expect(new OIDNFinalDenoiser().state()).toEqual({
      status: 'fallback',
      reason: 'OIDN modelUrl not supplied',
    } satisfies DenoiserState);

    const denoiser = new OIDNFinalDenoiser({ modelUrl: '/models/oidn.onnx' });
    expect(denoiser.state()).toEqual({
      status: 'warming-up',
      reason: 'OIDN denoiser is not initialized',
    } satisfies DenoiserState);

    const seam = denoiser as unknown as {
      _device: GPUDevice;
      _denoisedOutputTexture: GPUTexture;
      _warmupInFlight: boolean;
      _inFlight: boolean;
      _haveDenoisedOutput: boolean;
      _lastFailureReason: string | null;
      _lastFailureRetryable: boolean;
    };
    seam._warmupInFlight = true;
    expect(denoiser.state()).toEqual({
      status: 'warming-up',
      reason: 'preloading OIDN model',
    } satisfies DenoiserState);

    seam._warmupInFlight = false;
    seam._device = {} as GPUDevice;
    seam._denoisedOutputTexture = {} as GPUTexture;
    expect(denoiser.state()).toEqual({
      status: 'fallback',
      reason: 'waiting for first OIDN output',
    } satisfies DenoiserState);

    seam._inFlight = true;
    expect(denoiser.state()).toEqual({
      status: 'in-flight',
      reason: 'OIDN inference cycle in flight',
    } satisfies DenoiserState);

    seam._inFlight = false;
    seam._haveDenoisedOutput = true;
    expect(denoiser.state()).toEqual({ status: 'ready' } satisfies DenoiserState);

    seam._lastFailureReason = 'OIDN inference cycle failed: boom';
    seam._lastFailureRetryable = true;
    expect(denoiser.state()).toEqual({
      status: 'failed',
      reason: 'OIDN inference cycle failed: boom',
      retryable: true,
    } satisfies DenoiserState);
  });
  it('defers OIDN copies until post-submit and submits before mapAsync', () => {
    vi.stubGlobal('GPUBufferUsage', { COPY_DST: 1, MAP_READ: 2 });
    vi.stubGlobal('GPUMapMode', { READ: 1 });
    const events: string[] = [];
    const never = new Promise<void>(() => undefined);
    const readback = {
      mapAsync: () => { events.push('map'); return never; },
      destroy: vi.fn(),
    } as unknown as GPUBuffer;
    const copyTextureToBuffer = vi.fn(() => events.push('copy'));
    const device = {
      createBuffer: vi.fn(() => readback),
      createCommandEncoder: vi.fn(() => ({
        copyTextureToBuffer,
        finish: () => { events.push('finish'); return {}; },
      })),
      queue: {
        submit: vi.fn(() => events.push('submit')),
        writeTexture: vi.fn(),
      },
    } as unknown as GPUDevice;
    const denoiser = new OIDNFinalDenoiser({ modelUrl: '/models/oidn.onnx' });
    const output = {} as GPUTexture;
    const hdr = {} as GPUTexture;
    const seam = denoiser as unknown as {
      _device: GPUDevice;
      _denoisedOutputTexture: GPUTexture;
    };
    seam._device = device;
    seam._denoisedOutputTexture = output;
    const ctx = {
      width: 1,
      height: 1,
      resources: { common: {
        hdrColorTexture: hdr,
        albedoTexture: {} as GPUTexture,
        gNormalDepthTexture: {} as GPUTexture,
      } },
      encoder: { copyTextureToBuffer: vi.fn() },
    } as unknown as DenoiserDispatchContext;

    try {
      expect(denoiser.dispatch(ctx)).toBe(hdr);
      expect(events).toEqual([]);
      denoiser.afterFrameSubmit();
      expect(events).toEqual(['copy', 'copy', 'copy', 'finish', 'submit', 'map', 'map', 'map']);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('destroys buffers created before a later readback allocation fails', () => {
    vi.stubGlobal('GPUBufferUsage', { COPY_DST: 1, MAP_READ: 2 });
    const colorReadback = { destroy: vi.fn() } as unknown as GPUBuffer;
    const allocationFailure = new Error('albedo allocation failed');
    const createBuffer = vi.fn()
      .mockReturnValueOnce(colorReadback)
      .mockImplementationOnce(() => { throw allocationFailure; });
    const device = {
      createCommandEncoder: vi.fn(() => ({ copyTextureToBuffer: vi.fn(), finish: vi.fn() })),
      createBuffer,
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;
    const denoiser = new OIDNFinalDenoiser({ modelUrl: '/models/oidn.onnx' });
    const seam = denoiser as unknown as {
      _readbackTextures(
        device: GPUDevice,
        ctx: DenoiserDispatchContext,
        width: number,
        height: number,
      ): unknown;
    };
    const ctx = { resources: { common: {} } } as unknown as DenoiserDispatchContext;

    try {
      expect(() => seam._readbackTextures(device, ctx, 1, 1)).toThrow(allocationFailure);
      expect(colorReadback.destroy).toHaveBeenCalledOnce();
      expect(createBuffer).toHaveBeenCalledTimes(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('returns raw HDR after a failed cycle instead of stale denoised output', () => {
    const denoiser = new OIDNFinalDenoiser({ modelUrl: '/models/oidn.onnx' });
    const hdr = {} as GPUTexture;
    const seam = denoiser as unknown as {
      _device: GPUDevice;
      _denoisedOutputTexture: GPUTexture;
      _haveDenoisedOutput: boolean;
      _lastFailureReason: string;
    };
    seam._device = {} as GPUDevice;
    seam._denoisedOutputTexture = {} as GPUTexture;
    seam._haveDenoisedOutput = true;
    seam._lastFailureReason = 'failed';
    expect(denoiser.dispatch({
      resources: { common: { hdrColorTexture: hdr } },
    } as unknown as DenoiserDispatchContext)).toBe(hdr);
  });

});

describe('shouldResetDenoiserHistory', () => {
  it('treats frame zero as the mutation reset signal from requestAccumReset', () => {
    expect(shouldResetDenoiserHistory(0, false)).toBe(true);
    expect(shouldResetDenoiserHistory(4, false)).toBe(false);
  });

  it('also resets on camera motion regardless of frame index', () => {
    expect(shouldResetDenoiserHistory(12, true)).toBe(true);
  });
});
