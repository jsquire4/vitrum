import { describe, expect, it } from 'vitest';

import { InferenceGraph } from '../InferenceGraph.js';
import { executeNeuralInferenceCpu } from '../cpuInference.js';
import {
  NEURAL_PREPROCESSING_CONTRACT,
  postprocessNeuralRadiance,
  preprocessNeuralInputs,
} from '../preprocessing.js';
import {
  deriveParamCount,
  WALKAROUND_DENOISER_UNET_SPEC,
  type UNetSpec,
} from '../unetArchitecture.js';
import { maxSupportedNeuralResolutionForAspect } from '../tensorMemoryPlanner.js';
import {
  NEURAL_ARCHITECTURE_ID,
  NEURAL_F16_METRIC_DOMAIN,
  NEURAL_F16_QUANTIZATION,
  type ModelWeights,
} from '../weights.js';
import { neuralCheckpointPayloadSha256 } from '../checkpointDigest.js';
import { decodeNeuralTensor, encodeNeuralTensor } from '../float16.js';
import { CONV2D_WGSL } from '../wgsl/conv2d.wgsl.js';
import { RELU_WGSL } from '../wgsl/relu.wgsl.js';
import { SKIP_CONNECTION_WGSL } from '../wgsl/skipConnection.wgsl.js';
import { TRANSPOSED_CONV2D_WGSL } from '../wgsl/transposedConv2d.wgsl.js';

const GPU_PARITY_ENABLED =
  typeof navigator !== 'undefined' &&
  navigator.gpu != null;

function projectionSpec(): UNetSpec {
  const layers = [
    {
      name: 'pack',
      kind: 'inputPack',
      inputs: ['noisyColor', 'albedo', 'normals'],
      output: 'enc_input',
      params: { inC: 9, outC: 9 },
      weightLayout: 'none',
    },
    {
      name: 'proj',
      kind: 'conv2d',
      inputs: ['enc_input'],
      output: 'denoised',
      params: { inC: 9, outC: 3, kH: 1, kW: 1, stride: 1, padding: 0 },
      weightLayout: 'OIKW',
    },
  ] as const;
  return {
    name: 'cpu-gpu-projection-parity',
    inputChannels: 9,
    outputChannels: 3,
    layers,
    paramCount: deriveParamCount(layers),
  };
}

function projectionWeights(): ModelWeights {
  const values = new Float32Array(27);
  values[0] = 1;
  values[10] = 1;
  values[20] = 1;
  return {
    formatVersion: 2,
    checkpoint: {
      id: 'gpu-parity',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'gpu-parity',
      captureBackend: 'webgpu',
      tonemap: 'linear-hdr',
      hardware: 'runtime-adapter',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      qualityReport: { status: 'pass', reportPath: 'gpu-parity.json' },
    },
    layers: [{ name: 'proj', weights: values, biases: new Float32Array(3) }],
  };
}
function canonicalWeights(): ModelWeights {
  const spec = WALKAROUND_DENOISER_UNET_SPEC;
  const layers = spec.layers
    .filter(layer => layer.kind === 'conv2d' || layer.kind === 'transposedConv2d')
    .map((layer, layerIndex) => {
      const { inC, outC, kH = 1, kW = 1 } = layer.params;
      const values = new Float32Array(inC * outC * kH * kW);
      for (let i = 0; i < values.length; i++) {
        values[i] = (((i * 13 + layerIndex * 7) % 19) - 9) * 0.0001;
      }
      const biases = Float32Array.from(
        { length: outC },
        (_, channel) => (((channel * 5 + layerIndex * 3) % 11) - 5) * 0.0002,
      );
      return { name: layer.name, weights: values, biases };
    });
  return {
    formatVersion: 2,
    checkpoint: {
      id: 'canonical-gpu-parity',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'canonical-gpu-parity',
      captureBackend: 'webgpu',
      tonemap: 'linear-hdr',
      hardware: 'runtime-adapter',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      qualityReport: { status: 'pass', reportPath: 'canonical-gpu-parity.json' },
    },
    layers,
  };
}
function certifiedCanonicalWeights(): ModelWeights {
  const base = canonicalWeights();
  return {
    ...base,
    checkpoint: {
      ...base.checkpoint!,
      tensorStorage: 'f16-compatible',
      mixedPrecision: {
        checkpointSha256: neuralCheckpointPayloadSha256(base.layers),
        architecture: NEURAL_ARCHITECTURE_ID,
        preprocessing: NEURAL_PREPROCESSING_CONTRACT,
        quantization: NEURAL_F16_QUANTIZATION,
        metricDomain: NEURAL_F16_METRIC_DOMAIN,
        validationCorpusSha256: '3'.repeat(64),
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



describe.skipIf(!GPU_PARITY_ENABLED)('neural CPU/WebGPU parity', () => {
  it('matches the CPU oracle on a real WebGPU dispatch', async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter == null) {
      throw new Error('WebGPU browser run requested but requestAdapter() returned null');
    }
    const device = await adapter.requestDevice();
    const spec = projectionSpec();
    const weights = projectionWeights();
    const width = 8;
    const height = 8;
    const rgbLength = width * height * 3;
    const byteLength = rgbLength * 4;
    const noisyColor = Float32Array.from(
      { length: rgbLength },
      (_, index) => (index % 13) * 4,
    );
    noisyColor[7] = Number.NaN;
    noisyColor[11] = 100;
    const albedo = new Float32Array(rgbLength).fill(0.5);
    const normals = new Float32Array(rgbLength);
    for (let pixel = 0; pixel < width * height; pixel++) normals[pixel * 3 + 1] = 1;
    const cpu = executeNeuralInferenceCpu(
      spec,
      weights,
      width,
      height,
      { noisyColor, albedo, normals },
    );

    const createInput = (label: string, values: Float32Array): GPUBuffer => {
      const buffer = device.createBuffer({
        label,
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(
        buffer,
        0,
        values.buffer,
        values.byteOffset,
        values.byteLength,
      );
      return buffer;
    };
    const noisyBuffer = createInput('parity-noisy', noisyColor);
    const albedoBuffer = createInput('parity-albedo', albedo);
    const normalsBuffer = createInput('parity-normals', normals);
    const outputBuffer = device.createBuffer({
      label: 'parity-output',
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const readback = device.createBuffer({
      label: 'parity-readback',
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const packedByteLength = width * height * 9 * 4;
    const packedReadback = device.createBuffer({
      label: 'parity-packed-readback',
      size: packedByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const graph = new InferenceGraph(spec);

    try {
      await graph.initialize(device, weights, width, height);
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder({ label: 'neural-parity' });
      graph.run(noisyBuffer, albedoBuffer, normalsBuffer, outputBuffer, encoder);
      encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, byteLength);
      const tensors = (
        graph as unknown as {
          readonly _tensors: ReadonlyMap<string, { readonly buf: GPUBuffer }>;
        }
      )._tensors;
      const packedBuffer = tensors.get('enc_input')?.buf;
      if (packedBuffer == null) throw new Error('ready graph did not publish enc_input');
      encoder.copyBufferToBuffer(packedBuffer, 0, packedReadback, 0, packedByteLength);
      device.queue.submit([encoder.finish()]);
      await Promise.all([
        readback.mapAsync(GPUMapMode.READ),
        packedReadback.mapAsync(GPUMapMode.READ),
      ]);
      const gpu = new Float32Array(readback.getMappedRange().slice(0));
      const validationError = await device.popErrorScope();
      expect(validationError?.message).toBeUndefined();
      const gpuPacked = new Float32Array(packedReadback.getMappedRange().slice(0));
      readback.unmap();
      packedReadback.unmap();

      const expectedPacked = preprocessNeuralInputs(
        noisyColor,
        albedo,
        normals,
        NEURAL_PREPROCESSING_CONTRACT,
      );
      expect(Array.from(gpuPacked)).toEqual(Array.from(expectedPacked));
      expect(gpu).toHaveLength(cpu.modelOutput.length);
      for (let i = 0; i < gpu.length; i++) {
        expect(gpu[i], `model output element ${i}`).toBeCloseTo(cpu.modelOutput[i]!, 6);
      }
    } finally {
      graph.dispose();
      noisyBuffer.destroy();
      albedoBuffer.destroy();
      normalsBuffer.destroy();
      outputBuffer.destroy();
      readback.destroy();
      device.destroy();
      packedReadback.destroy();
    }
  });
  it('matches the CPU oracle through the full canonical 8x8 U-Net graph', async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter == null) {
      throw new Error('WebGPU browser run requested but requestAdapter() returned null');
    }
    const device = await adapter.requestDevice();
    const spec = WALKAROUND_DENOISER_UNET_SPEC;
    const weights = canonicalWeights();
    const width = 8;
    const height = 8;
    const rgbLength = width * height * 3;
    const byteLength = rgbLength * 4;
    const noisyColor = Float32Array.from(
      { length: rgbLength },
      (_, index) => ((index * 7) % 29) * 0.5,
    );
    const albedo = Float32Array.from(
      { length: rgbLength },
      (_, index) => (index % 5) / 4,
    );
    const normals = new Float32Array(rgbLength);
    for (let pixel = 0; pixel < width * height; pixel++) {
      normals[pixel * 3] = 0.25;
      normals[pixel * 3 + 1] = 0.5;
      normals[pixel * 3 + 2] = 0.75;
    }
    const cpu = executeNeuralInferenceCpu(
      spec,
      weights,
      width,
      height,
      { noisyColor, albedo, normals },
    );
    const createInput = (label: string, values: Float32Array): GPUBuffer => {
      const buffer = device.createBuffer({
        label,
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(
        buffer,
        0,
        values.buffer,
        values.byteOffset,
        values.byteLength,
      );
      return buffer;
    };
    const noisyBuffer = createInput('canonical-parity-noisy', noisyColor);
    const albedoBuffer = createInput('canonical-parity-albedo', albedo);
    const normalsBuffer = createInput('canonical-parity-normals', normals);
    const outputBuffer = device.createBuffer({
      label: 'canonical-parity-output',
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const readback = device.createBuffer({
      label: 'canonical-parity-readback',
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const graph = new InferenceGraph(spec);

    try {
      const maximum = maxSupportedNeuralResolutionForAspect(device, spec, weights, 16, 9);
      console.info(
        'NEURAL_ADAPTER_LIMIT',
        JSON.stringify({
          width: maximum.width,
          height: maximum.height,
          maxBufferSize: device.limits.maxBufferSize,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxTextureDimension2D: device.limits.maxTextureDimension2D,
        }),
      );
      expect(maximum.width).toBeGreaterThanOrEqual(width);
      expect(maximum.height).toBeGreaterThanOrEqual(height);
      expect(maximum.width % 8).toBe(0);
      expect(maximum.height % 8).toBe(0);

      await graph.initialize(device, weights, width, height);
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder({ label: 'canonical-neural-parity' });
      graph.run(noisyBuffer, albedoBuffer, normalsBuffer, outputBuffer, encoder);
      encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const gpu = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      expect(await device.popErrorScope()).toBeNull();

      expect(gpu).toHaveLength(cpu.modelOutput.length);
      let maximumAbsoluteError = 0;
      for (let i = 0; i < gpu.length; i++) {
        expect(Number.isFinite(gpu[i])).toBe(true);
        maximumAbsoluteError = Math.max(
          maximumAbsoluteError,
          Math.abs(gpu[i]! - cpu.modelOutput[i]!),
        );
      }
      expect(maximumAbsoluteError).toBeLessThanOrEqual(0.0001);
    } finally {
      graph.dispose();
      noisyBuffer.destroy();
      albedoBuffer.destroy();
      normalsBuffer.destroy();
      outputBuffer.destroy();
      readback.destroy();
      device.destroy();
    }
  });

  it('compiles and matches CPU f16 rounding through the certified full canonical graph', async ({ skip }) => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter == null) {
      throw new Error('WebGPU browser run requested but requestAdapter() returned null');
    }
    const adapterFeatures = [...adapter.features].sort();
    console.info(
      'NEURAL_F16_ADAPTER_PROBE',
      JSON.stringify({
        features: adapterFeatures,
        vendor: adapter.info?.vendor ?? '',
        architecture: adapter.info?.architecture ?? '',
        device: adapter.info?.device ?? '',
        description: adapter.info?.description ?? '',
      }),
    );
    if (!adapter.features.has('shader-f16')) {
      console.warn(
        'NEURAL_F16_ADAPTER_UNAVAILABLE',
        "adapter did not expose 'shader-f16'; full f16 compile/parity was not executed",
      );
      return skip();
    }

    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const spec = WALKAROUND_DENOISER_UNET_SPEC;
    const weights = certifiedCanonicalWeights();
    const width = 8;
    const height = 8;
    const rgbLength = width * height * 3;
    const byteLength = rgbLength * 2;
    const noisyColor = Float32Array.from(
      { length: rgbLength },
      (_, index) => ((index * 7) % 29) * 0.5,
    );
    const albedo = Float32Array.from(
      { length: rgbLength },
      (_, index) => (index % 5) / 4,
    );
    const normals = new Float32Array(rgbLength);
    for (let pixel = 0; pixel < width * height; pixel++) {
      normals[pixel * 3] = 0.25;
      normals[pixel * 3 + 1] = 0.5;
      normals[pixel * 3 + 2] = 0.75;
    }
    const cpu = executeNeuralInferenceCpu(
      spec,
      weights,
      width,
      height,
      { noisyColor, albedo, normals },
      'f16',
    );

    const createInput = (label: string, values: Float32Array): GPUBuffer => {
      const encoded = encodeNeuralTensor(values, 'f16');
      const buffer = device.createBuffer({
        label,
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(
        buffer,
        0,
        encoded.buffer,
        encoded.byteOffset,
        encoded.byteLength,
      );
      return buffer;
    };
    const noisyBuffer = createInput('canonical-f16-parity-noisy', noisyColor);
    const albedoBuffer = createInput('canonical-f16-parity-albedo', albedo);
    const normalsBuffer = createInput('canonical-f16-parity-normals', normals);
    const outputBuffer = device.createBuffer({
      label: 'canonical-f16-parity-output',
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const readback = device.createBuffer({
      label: 'canonical-f16-parity-readback',
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const graph = new InferenceGraph(spec);

    try {
      expect(spec.layers).toHaveLength(25);
      await graph.initialize(device, weights, width, height, 'f16');
      expect(graph.tensorStorage.precision).toBe('f16');

      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder({ label: 'canonical-neural-f16-parity' });
      graph.run(noisyBuffer, albedoBuffer, normalsBuffer, outputBuffer, encoder);
      encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const gpuModel = decodeNeuralTensor(
        new Uint16Array(readback.getMappedRange().slice(0)),
        'f16',
      );
      readback.unmap();
      expect(await device.popErrorScope()).toBeNull();

      const gpuDenoised = Float32Array.from(
        gpuModel,
        value => postprocessNeuralRadiance(value, NEURAL_PREPROCESSING_CONTRACT),
      );
      let maxAbsError = 0;
      let totalAbsError = 0;
      for (let i = 0; i < gpuDenoised.length; i++) {
        expect(Number.isFinite(gpuDenoised[i]), `denoised element ${i}`).toBe(true);
        expect(gpuDenoised[i]).toBeGreaterThanOrEqual(0);
        expect(gpuDenoised[i]).toBeLessThanOrEqual(NEURAL_PREPROCESSING_CONTRACT.radianceClamp);
        const error = Math.abs(gpuDenoised[i]! - cpu.denoised[i]!);
        maxAbsError = Math.max(maxAbsError, error);
        totalAbsError += error;
      }
      const meanAbsError = totalAbsError / gpuDenoised.length;
      console.info(
        'NEURAL_F16_FULL_PARITY',
        JSON.stringify({ layers: spec.layers.length, maxAbsError, meanAbsError }),
      );
      expect(maxAbsError).toBeLessThanOrEqual(0.05);
      expect(meanAbsError).toBeLessThanOrEqual(0.005);
    } finally {
      graph.dispose();
      noisyBuffer.destroy();
      albedoBuffer.destroy();
      normalsBuffer.destroy();
      outputBuffer.destroy();
      readback.destroy();
      device.destroy();
    }
  });


  it('creates real WebGPU pipelines for every finite-safe neural kernel', async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter == null) {
      throw new Error('WebGPU browser run requested but requestAdapter() returned null');
    }
    const device = await adapter.requestDevice();
    const kernels = [
      ['conv2d', CONV2D_WGSL, 'conv2dMain'],
      ['transposed-conv2d', TRANSPOSED_CONV2D_WGSL, 'transposedConv2dMain'],
      ['relu', RELU_WGSL, 'reluMain'],
      ['skip-connection', SKIP_CONNECTION_WGSL, 'skipConnectionMain'],
    ] as const;

    try {
      for (const [label, code, entryPoint] of kernels) {
        device.pushErrorScope('validation');
        const module = device.createShaderModule({ label, code });
        const compilation = await module.getCompilationInfo();
        expect(
          compilation.messages.filter((message) => message.type === 'error'),
          label,
        ).toEqual([]);
        await device.createComputePipelineAsync({
          label,
          layout: 'auto',
          compute: { module, entryPoint },
        });
        expect(await device.popErrorScope(), label).toBeNull();
      }
    } finally {
      device.destroy();
    }
  });
});
describe('neural CPU/WebGPU parity gate', () => {
  it('is opt-in and disabled unless explicitly requested', () => {
    expect(GPU_PARITY_ENABLED).toBe(typeof navigator !== 'undefined' && navigator.gpu != null);
  });
});
