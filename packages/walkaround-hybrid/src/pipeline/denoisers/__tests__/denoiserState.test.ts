import { describe, expect, it } from 'vitest';
import type { InferenceGraph } from '../../../neural/InferenceGraph.js';
import { AtrousDenoiser } from '../atrous.js';
import { AtrousVarianceDenoiser } from '../atrousVariance.js';
import { BmfrDenoiser } from '../bmfr.js';
import type { DenoiserState } from '../index.js';
import { NeuralDenoiser } from '../neural.js';
import { NoneDenoiser } from '../none.js';
import { OIDNFinalDenoiser } from '../oidnFinal.js';
import { SVGFRealDenoiser } from '../svgfReal.js';

const fakeGraph = (): InferenceGraph =>
  ({ run() {} }) as unknown as InferenceGraph;

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

  it('reports neural fallback until graph-backed resources are initialized', () => {
    expect(new NeuralDenoiser().state()).toEqual({
      status: 'fallback',
      reason: 'inference graph not supplied',
    } satisfies DenoiserState);

    const denoiser = new NeuralDenoiser({ inferenceGraph: fakeGraph() });
    expect(denoiser.state()).toEqual({
      status: 'fallback',
      reason: 'neural denoiser is not initialized',
    } satisfies DenoiserState);
  });

  it('reports neural ready and observable size-mismatch fallback', () => {
    const denoiser = new NeuralDenoiser({ inferenceGraph: fakeGraph() });
    const seam = denoiser as unknown as {
      _device: GPUDevice;
      _packPipeline: GPUComputePipeline;
      _unpackPipeline: GPUComputePipeline;
      _packParamsBuf: GPUBuffer;
      _unpackParamsBuf: GPUBuffer;
      // The four tensor GPU buffers are now grouped under _tensorBuffers (D4.9).
      _tensorBuffers: {
        noisyBuf: GPUBuffer; albedoBuf: GPUBuffer;
        normalsBuf: GPUBuffer; outputBuf: GPUBuffer;
        outputTex: GPUTexture; width: number; height: number;
      } | null;
      _lastFallbackReason: string | null;
    };
    seam._device = {} as GPUDevice;
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
    expect(denoiser.state()).toEqual({
      status: 'fallback',
      reason: 'size changed from 64x64 to 128x64',
    } satisfies DenoiserState);
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
});
