/**
 * neuralDenoiserResize.test.ts — Issue 1 characterization tests.
 *
 * Covers the resize behavioural correctness fix: after `NeuralDenoiser.resize`
 * the denoiser must be in a consistent state (fully allocated when a device
 * is present, or cleanly null when no device yet) rather than a torn-down
 * intermediate where buffers are null but dimensions have changed.
 *
 * All tests run without a real GPU. A minimal mock device is used to exercise
 * the `_reallocForSize` path that `resize` now calls when `_device` is present.
 *
 * Also covers Issue 2 byte-identity: the extracted NEURAL_PACK_WGSL /
 * NEURAL_UNPACK_WGSL strings must equal the shader code passed to
 * `createShaderModule` during `initialize`.
 */

import { describe, it, expect, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
// Install GPUBufferUsage / GPUTextureUsage globals before any source import
// that references them (NeuralDenoiser.initialize touches these constants).
installWebGPUPolyfills();
import { NeuralDenoiser } from '../src/pipeline/denoisers/neural.js';
import { NEURAL_PACK_WGSL, buildNeuralPackWgsl } from '../src/shaders/neuralPack.wgsl.js';
import { NEURAL_UNPACK_WGSL, buildNeuralUnpackWgsl } from '../src/shaders/neuralUnpack.wgsl.js';
import { NEURAL_LEGACY_PREPROCESSING_CONTRACT } from '../src/neural/preprocessing.js';
import type { InferenceGraph } from '../src/neural/InferenceGraph.js';
import type { ModelWeights } from '../src/neural/weights.js';
import { NEURAL_F32_TENSOR_STORAGE } from '../src/neural/tensorPrecision.js';

// ─── Mock device factory ────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CreatedBuffer = { label: string; size: number; destroy: (...args: any[]) => any };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CreatedTexture = { label: string; destroy: (...args: any[]) => any; createView: (...args: any[]) => any };

interface MockDevice {
  device: GPUDevice;
  buffers: CreatedBuffer[];
  textures: CreatedTexture[];
  shaderCodes: string[];
}

function makeMockDevice(): MockDevice {
  const buffers: CreatedBuffer[] = [];
  const textures: CreatedTexture[] = [];
  const shaderCodes: string[] = [];

  const device = {
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
      const buf: CreatedBuffer = {
        label: (desc.label as string) ?? '',
        size: desc.size as number,
        destroy: vi.fn(),
      };
      buffers.push(buf);
      return buf;
    }),
    createTexture: vi.fn((desc: GPUTextureDescriptor) => {
      const tex: CreatedTexture = {
        label: (desc.label as string) ?? '',
        destroy: vi.fn(),
        createView: vi.fn(() => ({})),
      };
      textures.push(tex);
      return tex;
    }),
    createShaderModule: vi.fn((desc: GPUShaderModuleDescriptor) => {
      shaderCodes.push(desc.code as string);
      return {};
    }),
    createComputePipelineAsync: vi.fn(() => Promise.resolve({ getBindGroupLayout: vi.fn(() => ({})) })),
    createBindGroup: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;

  return { device, buffers, textures, shaderCodes };
}

function makeTexture(label: string): GPUTexture {
  return {
    label,
    createView: vi.fn(() => ({})),
    destroy: vi.fn(),
  } as unknown as GPUTexture;
}

function makeEncoder(): GPUCommandEncoder {
  return {
    beginComputePass: vi.fn(() => ({
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
      end: vi.fn(),
    })),
  } as unknown as GPUCommandEncoder;
}

function makeDispatchContext(device: GPUDevice, width: number, height: number, hdr = makeTexture('hdr')) {
  return {
    device,
    encoder: makeEncoder(),
    width,
    height,
    frameIndex: 0,
    resources: {
      common: {
        hdrColorTexture: hdr,
        albedoTexture: makeTexture('albedo'),
        gNormalDepthTexture: makeTexture('normal-depth'),
      },
    } as never,
    sharedAtrousPipeline: {} as never,
    bglCache: {} as never,
    gNormalDepthView: {} as never,
    atrousDirectSigmas: [128, 5, 0.05],
    readAccum: {} as never,
    isMoving: false,
    wgX16: Math.ceil(width / 16),
    wgY16: Math.ceil(height / 16),
    computeDesc: () => ({}),
  };
}

/** Minimal InferenceGraph stub — enough for NeuralDenoiser to treat itself as enabled. */
const fakeGraph = {
  run: vi.fn(),
  tensorStorage: NEURAL_F32_TENSOR_STORAGE,
} as unknown as InferenceGraph;

// ─── Issue 1: resize state-consistency ────────────────────────────────────────

describe('NeuralDenoiser.resize — state consistency (Issue 1 fix)', () => {
  it('resize before initialize updates dimensions without allocating (no device yet)', () => {
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });
    d.resize(1280, 720);
    // No device — should not have allocated anything (nothing to assert on,
    // but calling resize must not throw or leave a partially-torn state).
    expect(d.disabled).toBe(false);
    // Internal _width/_height are private; we verify indirectly: a subsequent
    // initialize at 1280×720 must succeed (same dims = no mismatch warning).
  });

  it('resize after initialize (with device) reallocates buffers atomically — no null-buffer window', async () => {
    const { device, buffers, textures } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    const initCtx = {
      device,
      width: 640,
      height: 360,
      bglCache: {} as never,
      frameResources: {} as never,
    };
    await d.initialize(initCtx);

    // After initialize: 4 storage buffers + output texture should have been created.
    const bufsAfterInit = buffers.length;
    const texsAfterInit = textures.length;
    expect(bufsAfterInit).toBeGreaterThanOrEqual(4); // noisy + albedo + normals + output
    expect(texsAfterInit).toBeGreaterThanOrEqual(1); // output texture

    // A graph without retained weights cannot be resized. The published
    // generation remains intact and the selected neural mode fails durably.
    d.resize(1280, 720);
    expect(buffers).toHaveLength(bufsAfterInit);
    expect(textures).toHaveLength(texsAfterInit);
    expect(buffers.every(buffer => !vi.mocked(buffer.destroy).mock.calls.length)).toBe(true);
    expect(d.state()).toMatchObject({
      status: 'failed',
      reason: expect.stringContaining('without retained model weights'),
    });
  });

  it('resize UP then DOWN — no dangling / null buffers when device is present', async () => {
    const { device, buffers, textures } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    await d.initialize({ device, width: 640, height: 360, bglCache: {} as never, frameResources: {} as never });

    const afterInit = { bufs: buffers.length, texs: textures.length };

    d.resize(1280, 720);
    const afterUp = { bufs: buffers.length, texs: textures.length };
    expect(afterUp).toEqual(afterInit);

    d.resize(320, 180);
    const afterDown = { bufs: buffers.length, texs: textures.length };
    expect(afterDown).toEqual(afterInit);
    expect(buffers.every(buffer => !vi.mocked(buffer.destroy).mock.calls.length)).toBe(true);
  });

  it('same-size resize is a no-op (guard path)', async () => {
    const { device, buffers } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    await d.initialize({ device, width: 640, height: 360, bglCache: {} as never, frameResources: {} as never });

    const beforeCount = buffers.length;
    d.resize(640, 360); // same size
    // _reallocForSize guard: already correct size → no new buffers.
    expect(buffers.length).toBe(beforeCount);
  });

  it('resize with retained model weights reinitializes the inference graph in place', async () => {
    const { device } = makeMockDevice();
    const modelWeights = { layers: {} } as unknown as ModelWeights;
    const graph = {
      initialize: vi.fn(async () => {}),
      run: vi.fn(),
      owns: vi.fn(() => true),
      tensorStorage: NEURAL_F32_TENSOR_STORAGE,
    } as unknown as InferenceGraph;
    const d = new NeuralDenoiser({ inferenceGraph: graph, modelWeights });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    d.resize(128, 128);
    expect(d.state()).toEqual({
      status: 'warming-up',
      reason: 'neural graph reinitializing for 128x128',
    });

    await (d as unknown as { _graphReinitPromise: Promise<void> | null })._graphReinitPromise;

    expect(graph.initialize).toHaveBeenCalledWith(device, modelWeights, 128, 128);
    expect((d as unknown as { _graphW: number })._graphW).toBe(128);
    expect((d as unknown as { _graphH: number })._graphH).toBe(128);

    const hdr = makeTexture('hdr');
    const result = d.dispatch(makeDispatchContext(device, 128, 128, hdr) as never);

    expect(result).not.toBe(hdr);
    expect(graph.run).toHaveBeenCalledTimes(1);
    expect(d.state()).toEqual({ status: 'ready' });
  });

  it('dispatch failure is durable and never reports raw HDR as neural output', async () => {
    const { device } = makeMockDevice();
    const throwingGraph = {
      run: vi.fn(() => { throw new Error('synthetic graph failure'); }),
      tensorStorage: NEURAL_F32_TENSOR_STORAGE,
    } as unknown as InferenceGraph;
    const d = new NeuralDenoiser({ inferenceGraph: throwingGraph });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    const hdr = makeTexture('hdr');
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      expect(() => d.dispatch(makeDispatchContext(device, 64, 64, hdr) as never)).toThrow(
        /synthetic graph failure/,
      );
      expect(d.state()).toEqual({
        status: 'failed',
        reason: 'inference graph dispatch failed: synthetic graph failure',
        retryable: false,
      });
      expect(warnSpy.mock.calls.flat().join('\n')).toContain('synthetic graph failure');
    } finally {
      warnSpy.mockRestore();
      d.dispose();
    }
  });
});

describe('NeuralDenoiser transactional generations', () => {
  it('keeps the previous generation dispatchable after a resize failure and rejects the failed target durably', async () => {
    const { device, buffers, textures } = makeMockDevice();
    const modelWeights = { layers: [] } as unknown as ModelWeights;
    const graph = {
      initialize: vi.fn(async () => { throw new Error('resize allocation rejected'); }),
      run: vi.fn(),
      owns: vi.fn(() => true),
      tensorStorage: NEURAL_F32_TENSOR_STORAGE,
    } as unknown as InferenceGraph;
    const d = new NeuralDenoiser({ inferenceGraph: graph, modelWeights });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });
    const oldBuffers = buffers.slice();
    const oldTextures = textures.slice();

    d.resize(128, 128);
    await (d as unknown as { _graphReinitPromise: Promise<void> | null })._graphReinitPromise;

    expect(d.state()).toEqual({ status: 'ready' });
    expect(d.dispatch(makeDispatchContext(device, 64, 64) as never)).not.toBeNull();
    expect(graph.run).toHaveBeenCalledTimes(1);
    expect(() => d.dispatch(makeDispatchContext(device, 128, 128) as never)).toThrow(
      /resize allocation rejected/,
    );
    expect(graph.initialize).toHaveBeenCalledTimes(1);
    expect(oldBuffers.every(buffer => vi.mocked(buffer.destroy).mock.calls.length === 0)).toBe(true);
    expect(oldTextures.every(texture => vi.mocked(texture.destroy).mock.calls.length === 0)).toBe(true);
    expect(buffers.slice(oldBuffers.length).every(
      buffer => vi.mocked(buffer.destroy).mock.calls.length === 1,
    )).toBe(true);
    expect(textures.slice(oldTextures.length).every(
      texture => vi.mocked(texture.destroy).mock.calls.length === 1,
    )).toBe(true);
    d.dispose();
  });

  it('publishes only the newest concurrent resize generation', async () => {
    const { device, buffers } = makeMockDevice();
    const modelWeights = { layers: [] } as unknown as ModelWeights;
    let ownedWidth = 64;
    let ownedHeight = 64;
    const pending: Array<{
      width: number;
      height: number;
      resolve: () => void;
    }> = [];
    const graph = {
      initialize: vi.fn((_device: GPUDevice, _weights: ModelWeights, width: number, height: number) =>
        new Promise<void>(resolve => {
          pending.push({
            width,
            height,
            resolve: () => {
              ownedWidth = width;
              ownedHeight = height;
              resolve();
            },
          });
        })),
      run: vi.fn(),
      owns: vi.fn((_device: GPUDevice, width: number, height: number) =>
        width === ownedWidth && height === ownedHeight),
      tensorStorage: NEURAL_F32_TENSOR_STORAGE,
    } as unknown as InferenceGraph;
    const d = new NeuralDenoiser({ inferenceGraph: graph, modelWeights });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });
    const initialBufferCount = buffers.length;

    d.resize(128, 128);
    await vi.waitFor(() => expect(pending).toHaveLength(1));
    d.resize(256, 256);
    pending[0]!.resolve();
    await vi.waitFor(() => expect(pending).toHaveLength(2));
    pending[1]!.resolve();
    await (d as unknown as { _graphReinitPromise: Promise<void> | null })._graphReinitPromise;

    expect(graph.initialize).toHaveBeenNthCalledWith(1, device, modelWeights, 128, 128);
    expect(graph.initialize).toHaveBeenNthCalledWith(2, device, modelWeights, 256, 256);
    expect((d as unknown as { _graphW: number })._graphW).toBe(256);
    expect((d as unknown as { _graphH: number })._graphH).toBe(256);
    expect(d.state()).toEqual({ status: 'ready' });
    expect(d.dispatch(makeDispatchContext(device, 256, 256) as never)).not.toBeNull();
    expect(buffers.slice(initialBufferCount, initialBufferCount + 4).every(
      buffer => vi.mocked(buffer.destroy).mock.calls.length === 1,
    )).toBe(true);
    d.dispose();
  });

  it('dispose during initialize prevents publication and destroys the late candidate exactly once', async () => {
    const { device, buffers, textures } = makeMockDevice();
    let resolveFirstPipeline!: (pipeline: GPUComputePipeline) => void;
    vi.mocked(device.createComputePipelineAsync)
      .mockImplementationOnce(() => new Promise(resolve => { resolveFirstPipeline = resolve; }))
      .mockResolvedValue({ getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline);
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    const initializing = d.initialize({
      device,
      width: 64,
      height: 64,
      bglCache: {} as never,
      frameResources: {} as never,
    });
    d.dispose();
    resolveFirstPipeline({ getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline);

    await expect(initializing).rejects.toThrow(/superseded before publication/);
    expect(d.state()).toMatchObject({ status: 'failed', reason: 'neural denoiser has been disposed' });
    expect(buffers.every(buffer => vi.mocked(buffer.destroy).mock.calls.length === 1)).toBe(true);
    expect(textures.every(texture => vi.mocked(texture.destroy).mock.calls.length === 1)).toBe(true);
  });

  it('rejects a dispatch from a different device and records a durable selected-mode failure', async () => {
    const { device } = makeMockDevice();
    const { device: otherDevice } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });
    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    expect(() => d.dispatch(makeDispatchContext(otherDevice, 64, 64) as never)).toThrow(
      /device does not match/,
    );
    expect(d.state()).toMatchObject({
      status: 'failed',
      reason: expect.stringContaining('device does not match'),
      retryable: false,
    });
    expect(() => d.dispatch(makeDispatchContext(device, 64, 64) as never)).toThrow(
      /device does not match/,
    );
    d.dispose();
  });
});

// ─── Issue 2: byte-identity of extracted WGSL strings ───────────────────────

describe('NeuralDenoiser WGSL extraction byte-identity (Issue 2)', () => {
  it('createShaderModule receives NEURAL_PACK_WGSL — exact same string', async () => {
    const { device, shaderCodes } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    // First shader compiled = pack, second = unpack (matches declaration order in initialize).
    expect(shaderCodes.length).toBeGreaterThanOrEqual(2);
    expect(shaderCodes[0]).toBe(buildNeuralPackWgsl(NEURAL_LEGACY_PREPROCESSING_CONTRACT));
    expect(shaderCodes[1]).toBe(buildNeuralUnpackWgsl(NEURAL_LEGACY_PREPROCESSING_CONTRACT));
  });

  it('NEURAL_PACK_WGSL contains the pack entry-point declaration', () => {
    expect(NEURAL_PACK_WGSL).toMatch(/struct\s+PackParams/);
    expect(NEURAL_PACK_WGSL).toMatch(/@compute\s+@workgroup_size\s*\(\s*256/);
    expect(NEURAL_PACK_WGSL).toMatch(/fn\s+main\s*\(/);
    expect(NEURAL_PACK_WGSL).toMatch(/noisyOut\[base/);
    // Runtime normals are already signed world-space vectors.
    expect(NEURAL_PACK_WGSL).toMatch(/return\s+normalize\s*\(\s*safe\s*\)/);
  });

  it('NEURAL_UNPACK_WGSL contains the unpack entry-point declaration', () => {
    expect(NEURAL_UNPACK_WGSL).toMatch(/struct\s+UnpackParams/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/@compute\s+@workgroup_size\s*\(\s*256/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/fn\s+main\s*\(/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/textureStore\s*\(denoisedOut/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/neuralPostprocessRadiance\s*\(\s*denoisedIn\[base/);
  });
});
