/**
 * inferenceGraphReinit.test.ts — double-initialize leak guard.
 *
 * `InferenceGraph.initialize()` overwrites `_tensors` / `_allocatedBuffers`
 * with a fresh allocation. If it is ever called a SECOND time without an
 * intervening `dispose()`, every GPU buffer from the first allocation would be
 * orphaned (no remaining reference → never `.destroy()`'d → GPU memory leak).
 *
 * The defensive guard at the top of `initialize()` disposes the prior
 * allocation first. This test verifies that on a re-initialize, ALL of the
 * first allocation's buffers are destroyed (no leak), using a stub device that
 * records create/destroy calls — no real GPU required.
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

import { InferenceGraph } from '../InferenceGraph.js';
import { buildUNetSpec } from '../unetArchitecture.js';
import {
  buildRandomWeightsForSpec,
  NEURAL_ARCHITECTURE_ID,
  NEURAL_F16_METRIC_DOMAIN,
  NEURAL_F16_QUANTIZATION,
  type ModelWeights,
} from '../weights.js';
import { neuralCheckpointPayloadSha256 } from '../checkpointDigest.js';
import { NEURAL_PREPROCESSING_CONTRACT } from '../preprocessing.js';

interface StubBuffer {
  label: string;
  id: number;
  destroyed: boolean;
  size: number;
  destroyCount: number;
}

function makeStubDevice(
  failBufferAt?: number,
  features: readonly GPUFeatureName[] = [],
  maxComputeWorkgroupsPerDimension = 65_535,
) {
  let nextBufId = 0;
  const allBuffers: StubBuffer[] = [];
  const shaderSources: string[] = [];
  type FailureStage = 'buffer' | 'shader' | 'pipeline' | 'bindGroup';
  let failNextStage: FailureStage | null = null;
  let successfulCallsBeforeFailure = 0;
  let nextScopeError: string | null = null;
  const maybeFail = (stage: FailureStage): void => {
    if (failNextStage !== stage) return;
    if (successfulCallsBeforeFailure > 0) {
      successfulCallsBeforeFailure--;
      return;
    }
    failNextStage = null;
    throw new Error(`forced ${stage} failure`);
  };

  const device = {
    limits: { maxComputeWorkgroupsPerDimension },
    features: new Set(features),
    createBuffer(desc: { label?: string; size: number; usage: number; mappedAtCreation?: boolean }): StubBuffer {
      if (nextBufId === failBufferAt) {
        throw new Error('forced buffer allocation failure');
      }
      maybeFail('buffer');
      const buf: StubBuffer = {
        label: desc.label ?? '',
        id: nextBufId++,
        size: desc.size,
        destroyed: false,
        destroyCount: 0,
      };
      (buf as unknown as Record<string, unknown>).destroy = () => {
        buf.destroyed = true;
        buf.destroyCount++;
      };
      if (desc.mappedAtCreation) {
        const ab = new ArrayBuffer(desc.size);
        (buf as unknown as Record<string, unknown>).getMappedRange = () => ab;
        (buf as unknown as Record<string, unknown>).unmap = () => {};
      }
      allBuffers.push(buf);
      return buf;
    },
    createShaderModule(desc: { code?: string }) {
      maybeFail('shader');
      if (typeof desc.code === 'string') shaderSources.push(desc.code);
      return {};
    },
    async createComputePipelineAsync(_desc: unknown) {
      maybeFail('pipeline');
      return { getBindGroupLayout(_g: number) { return {}; } };
    },
    createBindGroup(_desc: unknown) {
      maybeFail('bindGroup');
      return {};
    },
    queue: { writeBuffer(_b: unknown, _o: number, _d: unknown) {} },
    pushErrorScope(_filter: GPUErrorFilter) {},
    async popErrorScope() {
      if (nextScopeError == null) return null;
      const message = nextScopeError;
      nextScopeError = null;
      return { message } as GPUError;
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish() { return {}; },
      };
    },
  } as unknown as GPUDevice;

  return {
    device,
    allBuffers,
    shaderSources,
    failNext(stage: FailureStage, successfulCalls = 0) {
      failNextStage = stage;
      successfulCallsBeforeFailure = successfulCalls;
    },
    failNextScope(message: string) { nextScopeError = message; },
  };
}

function makeStubWeights(spec: ReturnType<typeof buildUNetSpec>): ModelWeights {
  return buildRandomWeightsForSpec(spec, 0x1eaf);
}

function makeCertifiedWeights(spec: ReturnType<typeof buildUNetSpec>): ModelWeights {
  const base = makeStubWeights(spec);
  return {
    ...base,
    formatVersion: 2,
    checkpoint: {
      id: 'lifecycle-f16',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'lifecycle-test',
      captureBackend: 'webgpu',
      tonemap: 'linear-hdr',
      hardware: 'stub',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      qualityReport: { status: 'pass', reportPath: 'lifecycle-test.json' },
      tensorStorage: 'f16-compatible',
      mixedPrecision: {
        checkpointSha256: neuralCheckpointPayloadSha256(base.layers),
        architecture: NEURAL_ARCHITECTURE_ID,
        preprocessing: NEURAL_PREPROCESSING_CONTRACT,
        quantization: NEURAL_F16_QUANTIZATION,
        metricDomain: NEURAL_F16_METRIC_DOMAIN,
        validationCorpusSha256: '2'.repeat(64),
        status: 'pass',
        validationScenes: 8,
        maxAbsError: 0.01,
        meanAbsError: 0.001,
        psnrDb: 40,
        finiteOutputs: true,
        outputMin: 0,
        outputMax: 16,
        accumulation: 'f32',
        weights: 'f32',
      },
    },
  };
}

function externalBuffer(size: number, label: string): GPUBuffer {
  return {
    size,
    label,
    destroy() {},
  } as unknown as GPUBuffer;
}

describe('InferenceGraph — double-initialize leak guard', () => {
  it('re-initialize destroys ALL of the first allocation\'s buffers', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const { device, allBuffers } = makeStubDevice();

    const graph = new InferenceGraph(spec);
    await graph.initialize(device, weights, 32, 32);

    const firstGenBuffers = allBuffers.slice();
    expect(firstGenBuffers.length).toBeGreaterThan(0);
    // None destroyed yet.
    expect(firstGenBuffers.every(b => !b.destroyed)).toBe(true);

    // Second initialize WITHOUT a dispose() — guard must release the first gen.
    await graph.initialize(device, weights, 64, 64);

    // Every first-generation buffer must now be destroyed (no leak).
    const leaked = firstGenBuffers.filter(b => !b.destroyed);
    expect(leaked).toHaveLength(0);

    // The graph remains usable after re-initialize.
    expect(graph.ready).toBe(true);

    graph.dispose();
  });

  it('first initialize on a fresh instance destroys nothing', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const { device, allBuffers } = makeStubDevice();

    const graph = new InferenceGraph(spec);
    await graph.initialize(device, weights, 32, 32);

    // The guard is a no-op on a never-initialized instance: nothing destroyed.
    expect(allBuffers.some(b => b.destroyed)).toBe(false);
    graph.dispose();
  });

  it('accepts an odd first shape through a padded graph and exact logical output', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const { device, allBuffers } = makeStubDevice();
    const graph = new InferenceGraph(spec);

    await expect(graph.initialize(device, weights, 9, 8)).resolves.toBeUndefined();
    expect(graph.width).toBe(9);
    expect(graph.height).toBe(8);
    expect(graph.inferenceWidth).toBe(16);
    expect(graph.inferenceHeight).toBe(8);
    expect(allBuffers.length).toBeGreaterThan(0);
    const bytes = 9 * 8 * 3 * 4;
    expect(() => graph.run(
      externalBuffer(bytes, 'noisy'),
      externalBuffer(bytes, 'albedo'),
      externalBuffer(bytes, 'normals'),
      externalBuffer(bytes, 'output'),
      device.createCommandEncoder(),
    )).not.toThrow();
    graph.dispose();
  });

  it('consumes the device workgroup-axis limit and rejects overflow before allocation', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const control = makeStubDevice(undefined, [], 2);
    const graph = new InferenceGraph(spec);

    await expect(graph.initialize(control.device, weights, 8, 8)).rejects.toThrow(
      /workgroup axis 1=3 exceeds maxComputeWorkgroupsPerDimension=2/,
    );
    expect(control.allBuffers).toHaveLength(0);
    expect(graph.ready).toBe(false);
  });

  it('keeps the current graph generation intact after an invalid logical resize preflight', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const { device, allBuffers } = makeStubDevice();
    const graph = new InferenceGraph(spec);
    await graph.initialize(device, weights, 8, 8);
    const currentBuffers = allBuffers.slice();

    await expect(graph.initialize(device, weights, 0, 8)).rejects.toThrow(
      /unsupported internal render size 0x8/,
    );

    expect(allBuffers).toHaveLength(currentBuffers.length);
    expect(currentBuffers.every(buffer => !buffer.destroyed)).toBe(true);
    expect(graph.ready).toBe(true);
    graph.dispose();
  });

  it('destroys every earlier graph buffer when a later allocation throws', async () => {
    const spec = buildUNetSpec();
    const weights = makeStubWeights(spec);
    const { device, allBuffers } = makeStubDevice(5);
    const graph = new InferenceGraph(spec);

    await expect(graph.initialize(device, weights, 8, 8)).rejects.toThrow(
      /forced buffer allocation failure/,
    );

    expect(allBuffers.length).toBeGreaterThan(0);
    expect(allBuffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(graph.ready).toBe(false);

  });
  it.each([
    { stage: 'buffer' as const, successfulCalls: 2 },
    { stage: 'shader' as const, successfulCalls: 0 },
    { stage: 'pipeline' as const, successfulCalls: 0 },
    { stage: 'bindGroup' as const, successfulCalls: 0 },
    { stage: 'errorScope' as const, successfulCalls: 0 },
  ])(
    'preserves the published generation when candidate $stage construction fails',
    async ({ stage, successfulCalls }) => {
      const spec = buildUNetSpec();
      const weights = makeStubWeights(spec);
      const control = makeStubDevice();
      const graph = new InferenceGraph(spec);
      await graph.initialize(control.device, weights, 8, 8);

      const oldBuffers = control.allBuffers.slice();
      const oldTelemetry = graph.memoryTelemetry;
      expect(oldTelemetry).not.toBeNull();
      expect(graph.state).toBe('ready');
      expect(graph.owns(control.device, 8, 8)).toBe(true);

      if (stage === 'errorScope') {
        control.failNextScope('forced validation scope failure');
      } else {
        control.failNext(stage, successfulCalls);
      }

      await expect(graph.initialize(control.device, weights, 9, 8)).rejects.toThrow(
        stage === 'errorScope' ? /forced validation scope failure/ : new RegExp(`forced ${stage} failure`),
      );

      const candidateBuffers = control.allBuffers.slice(oldBuffers.length);
      expect(candidateBuffers.length).toBeGreaterThan(0);
      expect(candidateBuffers.every(buffer => buffer.destroyCount === 1)).toBe(true);
      expect(oldBuffers.every(buffer => buffer.destroyCount === 0)).toBe(true);
      expect(graph.state).toBe('ready');
      expect(graph.ready).toBe(true);
      expect(graph.device).toBe(control.device);
      expect(graph.width).toBe(8);
      expect(graph.height).toBe(8);
      expect(graph.memoryTelemetry).toBe(oldTelemetry);
      expect(graph.lastFailure).toMatch(/forced/);
      expect(graph.owns(control.device, 8, 8)).toBe(true);

      const bytes = 8 * 8 * 3 * 4;
      const encoder = control.device.createCommandEncoder();
      expect(() => graph.run(
        externalBuffer(bytes, 'noisy'),
        externalBuffer(bytes, 'albedo'),
        externalBuffer(bytes, 'normals'),
        externalBuffer(bytes, 'output'),
        encoder,
      )).not.toThrow();
      graph.dispose();
    },
  );

  it('publishes f32 -> f16 -> resized f16 generations without mixed-generation buffers', async () => {
    const spec = buildUNetSpec();
    const weights = makeCertifiedWeights(spec);
    const f32 = makeStubDevice();
    const f16 = makeStubDevice(undefined, ['shader-f16']);
    const graph = new InferenceGraph(spec);

    await graph.initialize(f32.device, weights, 8, 8, 'auto');
    const f32Buffers = f32.allBuffers.slice();
    expect(graph.tensorStorage.precision).toBe('f32');
    expect(f32Buffers.every(buffer => !buffer.destroyed)).toBe(true);

    await graph.initialize(f16.device, weights, 8, 8, 'auto');
    const firstF16Buffers = f16.allBuffers.slice();
    expect(graph.tensorStorage.precision).toBe('f16');
    expect(graph.device).toBe(f16.device);
    expect(f16.shaderSources).toHaveLength(spec.layers.length + 1);
    expect(f16.shaderSources.every(source => source.trimStart().startsWith('enable f16;'))).toBe(true);
    expect(f16.shaderSources.every(source => source.includes('array<f16>'))).toBe(true);
    expect(f16.shaderSources.some(source => source.includes('weights : array<f32>'))).toBe(true);
    expect(f16.shaderSources.some(source => source.includes('biases  : array<f32>'))).toBe(true);
    expect(f32Buffers.every(buffer => buffer.destroyCount === 1)).toBe(true);
    expect(firstF16Buffers.every(buffer => !buffer.destroyed)).toBe(true);

    await graph.initialize(f16.device, weights, 16, 16);
    const resizedF16Buffers = f16.allBuffers.slice(firstF16Buffers.length);
    expect(graph.tensorStorage.precision).toBe('f16');
    expect(graph.width).toBe(16);
    expect(firstF16Buffers.every(buffer => buffer.destroyCount === 1)).toBe(true);
    expect(resizedF16Buffers.every(buffer => !buffer.destroyed)).toBe(true);
    graph.dispose();
  });

  it('retains the published f32 generation when an eligible f16 candidate fails', async () => {
    const spec = buildUNetSpec();
    const weights = makeCertifiedWeights(spec);
    const f32 = makeStubDevice();
    const f16 = makeStubDevice(undefined, ['shader-f16']);
    const graph = new InferenceGraph(spec);

    await graph.initialize(f32.device, weights, 8, 8, 'auto');
    const publishedF32Buffers = f32.allBuffers.slice();
    f16.failNext('pipeline');

    await expect(graph.initialize(f16.device, weights, 16, 16, 'auto')).rejects.toThrow(
      /forced pipeline failure/,
    );

    expect(graph.state).toBe('ready');
    expect(graph.tensorStorage.precision).toBe('f32');
    expect(graph.device).toBe(f32.device);
    expect(graph.width).toBe(8);
    expect(graph.height).toBe(8);
    expect(publishedF32Buffers.every(buffer => buffer.destroyCount === 0)).toBe(true);
    expect(f16.allBuffers.length).toBeGreaterThan(0);
    expect(f16.allBuffers.every(buffer => buffer.destroyCount === 1)).toBe(true);
    graph.dispose();
  });
});
