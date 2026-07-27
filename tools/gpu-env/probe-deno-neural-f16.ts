/** Reproducible native-wgpu probe for the neural denoiser's generated f16 WGSL. */
import { buildInputPackerWgsl } from '../../packages/walkaround-hybrid/src/neural/inputPacker.ts';
import { neuralLayerWgslForStorage } from '../../packages/walkaround-hybrid/src/neural/mixedPrecisionWgsl.ts';
import { NEURAL_PREPROCESSING_CONTRACT } from '../../packages/walkaround-hybrid/src/neural/preprocessing.ts';
import {
  NEURAL_F16_TENSOR_STORAGE,
  NEURAL_F32_TENSOR_STORAGE,
  type NeuralTensorPrecision,
  type NeuralTensorStorageContract,
} from '../../packages/walkaround-hybrid/src/neural/tensorPrecision.ts';
import { BILINEAR_UPSAMPLE_WGSL } from '../../packages/walkaround-hybrid/src/neural/wgsl/bilinearUpsample.wgsl.ts';
import { CONV2D_WGSL } from '../../packages/walkaround-hybrid/src/neural/wgsl/conv2d.wgsl.ts';
import { RELU_WGSL } from '../../packages/walkaround-hybrid/src/neural/wgsl/relu.wgsl.ts';
import { SKIP_CONNECTION_WGSL } from '../../packages/walkaround-hybrid/src/neural/wgsl/skipConnection.wgsl.ts';
import { TRANSPOSED_CONV2D_WGSL } from '../../packages/walkaround-hybrid/src/neural/wgsl/transposedConv2d.wgsl.ts';
import { InferenceGraph } from '../../packages/walkaround-hybrid/src/neural/InferenceGraph.ts';
import { executeNeuralInferenceCpu } from '../../packages/walkaround-hybrid/src/neural/cpuInference.ts';
import { neuralCheckpointPayloadSha256 } from '../../packages/walkaround-hybrid/src/neural/checkpointDigest.ts';
import { decodeNeuralTensor, encodeNeuralTensor } from '../../packages/walkaround-hybrid/src/neural/float16.ts';
import { postprocessNeuralRadiance } from '../../packages/walkaround-hybrid/src/neural/preprocessing.ts';
import { WALKAROUND_DENOISER_UNET_SPEC } from '../../packages/walkaround-hybrid/src/neural/unetArchitecture.ts';
import {
  NEURAL_ARCHITECTURE_ID,
  NEURAL_F16_METRIC_DOMAIN,
  NEURAL_F16_QUANTIZATION,
  type ModelWeights,
} from '../../packages/walkaround-hybrid/src/neural/weights.ts';

function canonicalWeights(): ModelWeights {
  const layers = WALKAROUND_DENOISER_UNET_SPEC.layers
    .filter(layer => layer.kind === 'conv2d' || layer.kind === 'transposedConv2d')
    .map((layer, layerIndex) => {
      const { inC, outC, kH = 1, kW = 1 } = layer.params;
      return {
        name: layer.name,
        weights: Float32Array.from(
          { length: inC * outC * kH * kW },
          (_, index) => (((index * 13 + layerIndex * 7) % 19) - 9) * 0.0001,
        ),
        biases: Float32Array.from(
          { length: outC },
          (_, channel) => (((channel * 5 + layerIndex * 3) % 11) - 5) * 0.0002,
        ),
      };
    });
  return {
    formatVersion: 2,
    layers,
    checkpoint: {
      id: 'deno-native-f16-parity',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'deno-native-wgpu',
      captureBackend: 'webgpu',
      tonemap: 'linear-hdr',
      hardware: 'runtime-adapter',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      qualityReport: { status: 'pass', reportPath: 'deno-native-f16-parity.json' },
      tensorStorage: 'f16-compatible',
      mixedPrecision: {
        checkpointSha256: neuralCheckpointPayloadSha256(layers),
        architecture: NEURAL_ARCHITECTURE_ID,
        preprocessing: NEURAL_PREPROCESSING_CONTRACT,
        quantization: NEURAL_F16_QUANTIZATION,
        metricDomain: NEURAL_F16_METRIC_DOMAIN,
        validationCorpusSha256: '4'.repeat(64),
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
interface SignalStats {
  readonly min: number;
  readonly max: number;
  readonly nonZero: number;
  readonly checksum: number;
  readonly finite: boolean;
}

function signalStats(values: Float32Array): SignalStats {
  let min = Infinity;
  let max = -Infinity;
  let nonZero = 0;
  let checksum = 0;
  let finite = true;
  for (let index = 0; index < values.length; index++) {
    const value = values[index]!;
    finite &&= Number.isFinite(value);
    min = Math.min(min, value);
    max = Math.max(max, value);
    if (value !== 0) nonZero++;
    checksum += value * (index + 1);
  }
  return { min, max, nonZero, checksum, finite };
}

interface KernelInput {
  readonly binding: number;
  readonly values: Float32Array;
  /** Activations follow the selected tensor precision; parameters remain f32. */
  readonly representation: 'tensor' | 'f32';
}

async function runStorageKernel(
  device: GPUDevice,
  label: string,
  source: string,
  entryPoint: string,
  storage: NeuralTensorStorageContract,
  inputs: readonly KernelInput[],
  uniform: Uint32Array,
  outputElementCount: number,
  dispatch: readonly [number, number, number],
): Promise<Float32Array> {
  const buffers: GPUBuffer[] = [];
  let readback: GPUBuffer | null = null;
  let readbackMapped = false;
  let scopeOpen = true;
  device.pushErrorScope('validation');
  try {
    const module = device.createShaderModule({ label: `${label}-module`, code: source });
    const pipeline = await device.createComputePipelineAsync({
      label: `${label}-pipeline`,
      layout: 'auto',
      compute: { module, entryPoint },
    });
    const entries: GPUBindGroupEntry[] = [];
    for (const input of inputs) {
      const encoded = input.representation === 'tensor'
        ? encodeNeuralTensor(input.values, storage.precision)
        : input.values;
      const buffer = device.createBuffer({
        label: `${label}-binding-${input.binding}`,
        size: Math.max(4, encoded.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      buffers.push(buffer);
      device.queue.writeBuffer(
        buffer,
        0,
        encoded.buffer as ArrayBuffer,
        encoded.byteOffset,
        encoded.byteLength,
      );
      entries.push({ binding: input.binding, resource: { buffer } });
    }

    const outputBytes = outputElementCount * storage.bytesPerScalar;
    const output = device.createBuffer({
      label: `${label}-output`,
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    buffers.push(output);
    entries.push({ binding: 3, resource: { buffer: output } });

    const uniformBuffer = device.createBuffer({
      label: `${label}-uniform`,
      size: Math.max(48, uniform.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    buffers.push(uniformBuffer);
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      uniform.buffer as ArrayBuffer,
      uniform.byteOffset,
      uniform.byteLength,
    );
    entries.push({ binding: 4, resource: { buffer: uniformBuffer } });

    const bindGroup = device.createBindGroup({
      label: `${label}-bind-group`,
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    readback = device.createBuffer({
      label: `${label}-readback`,
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    readbackMapped = true;
    const bytes = readback.getMappedRange().slice(0);
    const decoded = storage.precision === 'f16'
      ? decodeNeuralTensor(new Uint16Array(bytes), 'f16')
      : decodeNeuralTensor(new Float32Array(bytes), 'f32');
    readback.unmap();
    readbackMapped = false;
    const scoped = await device.popErrorScope();
    scopeOpen = false;
    if (scoped != null) throw new Error(scoped.message);
    return decoded;
  } finally {
    if (readbackMapped) readback?.unmap();
    readback?.destroy();
    for (const buffer of buffers) buffer.destroy();
    if (scopeOpen) await device.popErrorScope();
  }
}

function maxAbsoluteError(actual: Float32Array, expected: Float32Array): number {
  if (actual.length !== expected.length) return Infinity;
  let max = 0;
  for (let index = 0; index < actual.length; index++) {
    max = Math.max(max, Math.abs(actual[index]! - expected[index]!));
  }
  return max;
}

async function runBilinearMicrocase(
  device: GPUDevice,
  precision: NeuralTensorPrecision,
): Promise<Record<string, unknown>> {
  const storage = precision === 'f16'
    ? NEURAL_F16_TENSOR_STORAGE
    : NEURAL_F32_TENSOR_STORAGE;
  const expected = Float32Array.from([
    // Independent analytic half-pixel oracle: signed taps are individually
    // clamped to a 2x2 source at every border.
    1, 1.25, 1.75, 2,
    1.5, 1.75, 2.25, 2.5,
    2.5, 2.75, 3.25, 3.5,
    3, 3.25, 3.75, 4,
  ]);
  const actual = await runStorageKernel(
    device,
    `deno-bilinear-${precision}`,
    neuralLayerWgslForStorage('bilinearUpsample', BILINEAR_UPSAMPLE_WGSL, storage),
    'bilinearUpsampleMain',
    storage,
    [{ binding: 0, values: Float32Array.from([1, 2, 3, 4]), representation: 'tensor' }],
    Uint32Array.from([2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    16,
    [1, 1, 1],
  );
  const error = maxAbsoluteError(actual, expected);
  return {
    label: 'bilinear-border',
    precision,
    ok: error === 0,
    maxAbsError: error,
    stats: signalStats(actual),
  };
}

const MICRO_TCONV_WEIGHTS = Float32Array.from([
  1, 2, 3, 4,
  0.5, 1, 1.5, 2,
  -1, 0.25, 0.5, -0.5,
]);
const MICRO_TCONV_BIASES = Float32Array.from([0.5, 1, 1.5]);

function scatterTransposedConvMicroReference(): Float32Array {
  const input = [1, 2, 3, 4];
  const output = new Float32Array(4 * 4 * 3);
  for (let pixel = 0; pixel < 16; pixel++) output.set(MICRO_TCONV_BIASES, pixel * 3);
  for (let iy = 0; iy < 2; iy++) {
    for (let ix = 0; ix < 2; ix++) {
      for (let kh = 0; kh < 2; kh++) {
        const oy = iy * 2 - 1 + kh * 2;
        if (oy < 0 || oy >= 4) continue;
        for (let kw = 0; kw < 2; kw++) {
          const ox = ix * 2 - 1 + kw * 2;
          if (ox < 0 || ox >= 4) continue;
          for (let oc = 0; oc < 3; oc++) {
            const outputIndex = (oy * 4 + ox) * 3 + oc;
            const weightIndex = oc * 4 + kh * 2 + kw;
            output[outputIndex] = Math.fround(
              output[outputIndex]! +
              Math.fround(input[iy * 2 + ix]! * MICRO_TCONV_WEIGHTS[weightIndex]!),
            );
          }
        }
      }
    }
  }
  return output;
}

async function runTransposedConvMicrocase(
  device: GPUDevice,
  precision: NeuralTensorPrecision,
): Promise<Record<string, unknown>> {
  const storage = precision === 'f16'
    ? NEURAL_F16_TENSOR_STORAGE
    : NEURAL_F32_TENSOR_STORAGE;
  const actual = await runStorageKernel(
    device,
    `deno-transposed-conv-${precision}`,
    neuralLayerWgslForStorage('transposedConv2d', TRANSPOSED_CONV2D_WGSL, storage),
    'transposedConv2dMain',
    storage,
    [
      { binding: 0, values: Float32Array.from([1, 2, 3, 4]), representation: 'tensor' },
      { binding: 1, values: MICRO_TCONV_WEIGHTS, representation: 'f32' },
      { binding: 2, values: MICRO_TCONV_BIASES, representation: 'f32' },
    ],
    // 2x2x1 -> 4x4x3: k=2, stride=2, padding=1, dilation=2,
    // outputPadding=1. The independent oracle scatters taps forward.
    Uint32Array.from([2, 2, 1, 3, 2, 2, 2, 1, 2, 1, 0, 0]),
    4 * 4 * 3,
    [1, 1, 3],
  );
  const reference = scatterTransposedConvMicroReference();
  const expected = decodeNeuralTensor(
    encodeNeuralTensor(reference, precision),
    precision,
  );
  const error = maxAbsoluteError(actual, expected);
  return {
    label: 'transposed-conv-custom-shape',
    precision,
    ok: error === 0,
    maxAbsError: error,
    stats: signalStats(actual),
  };
}



const EXPECTED_LAYER_KINDS = [
  'conv2d', 'inputPack', 'relu', 'skipAdd', 'transposedConv2d',
] as const;
const EXPECTED_DISPATCH_FINGERPRINT = [
  'neural-inputPack:1,1,1', 'neural-enc1_conv:1,1,24',
  'neural-enc1_relu:6,1,1', 'neural-enc1_down:1,1,24',
  'neural-enc2_conv:1,1,48', 'neural-enc2_relu:3,1,1',
  'neural-enc2_down:1,1,48', 'neural-enc3_conv:1,1,96',
  'neural-enc3_relu:2,1,1', 'neural-enc3_down:1,1,96',
  'neural-bottleneck:1,1,192', 'neural-bn_relu:1,1,1',
  'neural-dec3_up:1,1,96', 'neural-dec3_skip:2,1,1',
  'neural-dec3_conv:1,1,96', 'neural-dec3_relu:2,1,1',
  'neural-dec2_up:1,1,48', 'neural-dec2_skip:3,1,1',
  'neural-dec2_conv:1,1,48', 'neural-dec2_relu:3,1,1',
  'neural-dec1_up:1,1,24', 'neural-dec1_skip:6,1,1',
  'neural-dec1_conv:1,1,24', 'neural-dec1_relu:6,1,1',
  'neural-proj:1,1,3',
].join('|');


const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (adapter == null) {
  console.log(JSON.stringify({ ok: false, reason: 'requestAdapter returned null' }));
  Deno.exit(2);
}

const features = [...adapter.features].sort();
const supportsShaderF16 = adapter.features.has('shader-f16');
const info = adapter.info;
console.log(JSON.stringify({
  phase: 'adapter',
  vendor: info.vendor,
  architecture: info.architecture,
  device: info.device,
  description: info.description,
  features,
  supportsShaderF16,
  limits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
  },
}));

const device = await adapter.requestDevice({
  requiredFeatures: supportsShaderF16 ? ['shader-f16'] : [],
});
const sources = [
  ['input-pack', buildInputPackerWgsl(
    NEURAL_PREPROCESSING_CONTRACT,
    NEURAL_F16_TENSOR_STORAGE,
  ), 'inputPackMain'],
  ['conv2d', neuralLayerWgslForStorage(
    'conv2d', CONV2D_WGSL, NEURAL_F16_TENSOR_STORAGE,
  ), 'conv2dMain'],
  ['transposed-conv2d', neuralLayerWgslForStorage(
    'transposedConv2d', TRANSPOSED_CONV2D_WGSL, NEURAL_F16_TENSOR_STORAGE,
  ), 'transposedConv2dMain'],
  ['relu', neuralLayerWgslForStorage(
    'relu', RELU_WGSL, NEURAL_F16_TENSOR_STORAGE,
  ), 'reluMain'],
  ['skip-add', neuralLayerWgslForStorage(
    'skipAdd', SKIP_CONNECTION_WGSL, NEURAL_F16_TENSOR_STORAGE,
  ), 'skipConnectionMain'],
  ['bilinear-upsample', neuralLayerWgslForStorage(
    'bilinearUpsample', BILINEAR_UPSAMPLE_WGSL, NEURAL_F16_TENSOR_STORAGE,
  ), 'bilinearUpsampleMain'],
] as const;

let compiled = 0;
let compilationErrorCount = 0;
for (const [label, code, entryPoint] of sources) {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label: `deno-neural-${label}`, code });
  const compilation = await module.getCompilationInfo();
  const sourceErrors = compilation.messages
    .filter(message => message.type === 'error')
    .map(message => message.message);
  let pipelineError: string | null = null;
  compilationErrorCount += sourceErrors.length;
  try {
    await device.createComputePipelineAsync({
      label: `deno-neural-pipeline-${label}`,
      layout: 'auto',
      compute: { module, entryPoint },
    });
  } catch (error) {
    pipelineError = error instanceof Error ? error.message : String(error);
  }
  const scoped = await device.popErrorScope();
  console.log(JSON.stringify({
    phase: 'shader',
    label,
    sourceErrors,
    pipelineError,
    validationError: scoped?.message ?? null,
  }));
  if (sourceErrors.length === 0 && pipelineError == null && scoped == null) compiled++;
}
const microResults: Array<Record<string, unknown>> = [];
let microError: string | null = null;
try {
  microResults.push(await runBilinearMicrocase(device, 'f32'));
  microResults.push(await runTransposedConvMicrocase(device, 'f32'));
  if (supportsShaderF16 && compiled === sources.length) {
    microResults.push(await runBilinearMicrocase(device, 'f16'));
    microResults.push(await runTransposedConvMicrocase(device, 'f16'));
  }
} catch (error) {
  microError = error instanceof Error ? error.message : String(error);
}
const expectedMicroCount = supportsShaderF16 ? 4 : 2;
const microOk = microError == null &&
  microResults.length === expectedMicroCount &&
  microResults.every(result => result.ok === true);
console.log(JSON.stringify({
  phase: 'extension-micro',
  expectedMicroCount,
  microOk,
  microError,
  results: microResults,
}));
let parityExecuted = false;
let parityOk = false;
let parityError: string | null = null;
let maxAbsError: number | null = null;
let meanAbsError: number | null = null;
if (supportsShaderF16 && compiled === sources.length) {
  const spec = WALKAROUND_DENOISER_UNET_SPEC;
  const weights = canonicalWeights();
  const width = 8;
  const height = 8;
  const rgbLength = width * height * 3;
  const byteLength = rgbLength * 2;

  let weightMin = Infinity;
  let weightMax = -Infinity;
  let weightNonZero = 0;
  let biasMin = Infinity;
  let biasMax = -Infinity;
  let biasNonZero = 0;
  for (const layer of weights.layers) {
    for (const value of layer.weights) {
      weightMin = Math.min(weightMin, value);
      weightMax = Math.max(weightMax, value);
      if (value !== 0) weightNonZero++;
    }
    for (const value of layer.biases) {
      biasMin = Math.min(biasMin, value);
      biasMax = Math.max(biasMax, value);
      if (value !== 0) biasNonZero++;
    }
  }
  if (!(weightMin < 0 && weightMax > 0 && weightNonZero > 0 &&
        biasMin < 0 && biasMax > 0 && biasNonZero > 0)) {
    throw new Error('canonical parity weights/biases are not nonzero mixed-sign signals');
  }

  const noisyColor = Float32Array.from(
    { length: rgbLength },
    (_, index) => ((index * 7) % 29) * 0.5,
  );
  const albedo = Float32Array.from(
    { length: rgbLength },
    (_, index) => (index % 5) / 4,
  );
  const normals = Float32Array.from(
    { length: rgbLength },
    (_, index) => index % 3 === 1 ? 0.5 : 0.25,
  );
  const inputStats = {
    noisyColor: signalStats(noisyColor),
    albedo: signalStats(albedo),
    normals: signalStats(normals),
  };
  if (Object.values(inputStats).some(
    stats => !stats.finite || stats.nonZero === 0 || stats.max <= stats.min
  )) {
    throw new Error('canonical parity inputs are zero, constant, or non-finite');
  }

  const cpu = executeNeuralInferenceCpu(
    spec,
    weights,
    width,
    height,
    { noisyColor, albedo, normals },
    'f16',
  );
  const makeInput = (label: string, values: Float32Array): GPUBuffer => {
    const encoded = encodeNeuralTensor(values, 'f16');
    const buffer = device.createBuffer({
      label,
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      buffer, 0, encoded.buffer as ArrayBuffer, encoded.byteOffset, encoded.byteLength,
    );
    return buffer;
  };
  const noisyBuffer = makeInput('deno-native-f16-noisy', noisyColor);
  const albedoBuffer = makeInput('deno-native-f16-albedo', albedo);
  const normalBuffer = makeInput('deno-native-f16-normal', normals);
  const outputBuffer = device.createBuffer({
    label: 'deno-native-f16-output',
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    label: 'deno-native-f16-readback',
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const graph = new InferenceGraph(spec);
  let intermediateReadback: GPUBuffer | null = null;
  let intermediateStats: SignalStats | null = null;
  let outputStats: SignalStats | null = null;
  const passLabels: string[] = [];
  const dispatches: Array<{ label: string; x: number; y: number; z: number }> = [];
  parityExecuted = true;
  try {
    await graph.initialize(device, weights, width, height, 'f16');
    const tensors = (
      graph as unknown as {
        readonly _tensors: ReadonlyMap<
          string,
          { readonly buf: GPUBuffer; readonly dims: { H: number; W: number; C: number } }
        >;
      }
    )._tensors;
    const intermediate = tensors.get('enc1_feat');
    if (intermediate == null) throw new Error("ready graph is missing 'enc1_feat'");
    const intermediateBytes =
      intermediate.dims.H * intermediate.dims.W * intermediate.dims.C * 2;
    intermediateReadback = device.createBuffer({
      label: 'deno-native-f16-intermediate-readback',
      size: intermediateBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    device.pushErrorScope('validation');
    const nativeEncoder = device.createCommandEncoder({ label: 'deno-native-f16-parity' });
    const tracingEncoder = {
      beginComputePass(descriptor?: GPUComputePassDescriptor) {
        const label = descriptor?.label ?? '';
        passLabels.push(label);
        const pass = nativeEncoder.beginComputePass(descriptor);
        return {
          setPipeline(pipeline: GPUComputePipeline) { pass.setPipeline(pipeline); },
          setBindGroup(index: number, bindGroup: GPUBindGroup) {
            pass.setBindGroup(index, bindGroup);
          },
          dispatchWorkgroups(x: number, y = 1, z = 1) {
            dispatches.push({ label, x, y, z });
            pass.dispatchWorkgroups(x, y, z);
          },
          end() {
            pass.end();
            if (label === 'neural-enc1_conv') {
              nativeEncoder.copyBufferToBuffer(
                intermediate.buf, 0, intermediateReadback!, 0, intermediateBytes,
              );
            }
          },
        };
      },
      copyBufferToBuffer(
        source: GPUBuffer,
        sourceOffset: number,
        destination: GPUBuffer,
        destinationOffset: number,
        size: number,
      ) {
        nativeEncoder.copyBufferToBuffer(
          source, sourceOffset, destination, destinationOffset, size,
        );
      },
    } as unknown as GPUCommandEncoder;

    graph.run(noisyBuffer, albedoBuffer, normalBuffer, outputBuffer, tracingEncoder);
    nativeEncoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, byteLength);
    device.queue.submit([nativeEncoder.finish()]);
    await Promise.all([
      readback.mapAsync(GPUMapMode.READ),
      intermediateReadback.mapAsync(GPUMapMode.READ),
    ]);
    const gpuModel = decodeNeuralTensor(
      new Uint16Array(readback.getMappedRange().slice(0)),
      'f16',
    );
    const intermediateValues = decodeNeuralTensor(
      new Uint16Array(intermediateReadback.getMappedRange().slice(0)),
      'f16',
    );
    readback.unmap();
    intermediateReadback.unmap();
    const scoped = await device.popErrorScope();
    if (scoped != null) throw new Error(scoped.message);

    const expectedLabels = spec.layers.map(layer =>
      layer.kind === 'inputPack' ? 'neural-inputPack' : `neural-${layer.name}`
    );
    const layerKinds = [...new Set(spec.layers.map(layer => layer.kind))].sort();
    const dispatchFingerprint = dispatches
      .map(dispatch => `${dispatch.label}:${dispatch.x},${dispatch.y},${dispatch.z}`)
      .join('|');
    if (spec.layers.length !== 25 ||
        dispatches.length !== 25 ||
        JSON.stringify(passLabels) !== JSON.stringify(expectedLabels) ||
        JSON.stringify(layerKinds) !== JSON.stringify(EXPECTED_LAYER_KINDS) ||
        dispatchFingerprint !== EXPECTED_DISPATCH_FINGERPRINT ||
        dispatches.some(dispatch => dispatch.x < 1 || dispatch.y < 1 || dispatch.z < 1)) {
      throw new Error(
        `canonical dispatch trace mismatch: layers=${spec.layers.length}, ` +
        `kinds=${JSON.stringify(layerKinds)}, dispatches=${dispatches.length}, ` +
        `fingerprint=${dispatchFingerprint}, labels=${JSON.stringify(passLabels)}`,
      );
    }

    intermediateStats = signalStats(intermediateValues);
    if (!intermediateStats.finite ||
        intermediateStats.nonZero === 0 ||
        intermediateStats.max <= intermediateStats.min) {
      throw new Error('captured enc1_conv intermediate is zero, constant, or non-finite');
    }
    if (intermediateStats.min !== -0.004550933837890625 ||
        intermediateStats.max !== 0.0053253173828125 ||
        intermediateStats.nonZero !== 1536 ||
        intermediateStats.checksum !== -55.784843027591705) {
      throw new Error(
        `captured enc1_conv intermediate changed: ${JSON.stringify(intermediateStats)}`,
      );
    }

    const gpuDenoised = Float32Array.from(
      gpuModel,
      value => postprocessNeuralRadiance(value, NEURAL_PREPROCESSING_CONTRACT),
    );
    outputStats = signalStats(gpuDenoised);
    let max = 0;
    let total = 0;
    const finiteAndBounded = outputStats.finite &&
      outputStats.nonZero > 0 &&
      outputStats.max > outputStats.min &&
      outputStats.min >= 0 &&
      outputStats.max <= NEURAL_PREPROCESSING_CONTRACT.radianceClamp;
    for (let index = 0; index < gpuDenoised.length; index++) {
      const error = Math.abs(gpuDenoised[index]! - cpu.denoised[index]!);
      max = Math.max(max, error);
      total += error;
    }
    maxAbsError = max;
    meanAbsError = total / gpuDenoised.length;
    if (outputStats.min !== 0 ||
        outputStats.max !== 0.0031948089599609375 ||
        outputStats.nonZero !== 64 ||
        outputStats.checksum !== 19.52667236328125) {
      throw new Error(`canonical f16 output changed: ${JSON.stringify(outputStats)}`);
    }
    parityOk = finiteAndBounded && max <= 0.05 && meanAbsError <= 0.005;
  } catch (error) {
    parityError = error instanceof Error ? error.message : String(error);
  } finally {
    graph.dispose();
    noisyBuffer.destroy();
    albedoBuffer.destroy();
    normalBuffer.destroy();
    outputBuffer.destroy();
    readback.destroy();
    intermediateReadback?.destroy();
  }
  console.log(JSON.stringify({
    phase: 'parity',
    layers: spec.layers.length,
    width,
    height,
    layerKinds: [...new Set(spec.layers.map(layer => layer.kind))].sort(),
    dispatchCount: dispatches.length,
    dispatches,
    inputStats,
    parameterStats: {
      weightMin,
      weightMax,
      weightNonZero,
      biasMin,
      biasMax,
      biasNonZero,
    },
    intermediateStats,
    outputStats,
    parityExecuted,
    parityOk,
    parityError,
    maxAbsError,
    meanAbsError,
  }));
}


console.log(JSON.stringify({
  phase: 'result',
  supportsShaderF16,
  compiled,
  total: sources.length,
  compilationErrorCount,
  microOk,
  ok: supportsShaderF16 && compilationErrorCount === 0 && compiled === sources.length &&
    microOk && parityExecuted && parityOk,
}));
device.destroy();

// No f16 feature is an honest environmental absence, not a probe failure.
Deno.exit(supportsShaderF16 &&
  (compilationErrorCount !== 0 || compiled !== sources.length || !microOk || !parityOk) ? 1 : 0);
