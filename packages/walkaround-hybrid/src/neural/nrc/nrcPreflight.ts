import type { NrcConfig } from './nrcSubsystem.js';
import { NRC_DIAGNOSTIC_BYTES } from './nrcDiagnostics.js';
import {
  createNrcInferenceArenaLayout,
  createNrcRuntimeArenaLayout,
  validateNrcArenaLayouts,
  type NrcInferenceArenaLayout,
  type NrcRuntimeArenaLayout,
} from './nrcArena.js';

const U32_MAX = 0xffff_ffff;
const F32_BYTES = Float32Array.BYTES_PER_ELEMENT;

export interface NrcBufferAllocation {
  readonly count: number;
  readonly bytesEach: number;
  readonly totalBytes: number;
}

export interface NrcResourceFootprint {
  readonly inW: number;
  readonly recordStride: number;
  readonly recordScalars: number;
  readonly recordBytes: number;
  readonly readbackBytes: number;
  readonly tableScalars: number;
  readonly totalWeights: number;
  readonly totalBiases: number;
  readonly levelResolutions: readonly number[];
  readonly inferenceArenaLayout: NrcInferenceArenaLayout;
  readonly runtimeArenaLayout: NrcRuntimeArenaLayout;
  readonly storageBindings: Readonly<Record<string, number>>;
  readonly uniformBindings: Readonly<Record<string, number>>;
  readonly persistentAllocations: Readonly<Record<string, NrcBufferAllocation>>;
  readonly peakBufferBytes: number;
  readonly liveTrainableBytes: number;
  readonly candidateTrainableBytes: number;
  readonly peakTrainableBytes: number;
  readonly persistentBufferCount: number;
  readonly persistentBufferBytes: number;
  readonly peakResidentBufferCount: number;
  readonly peakResidentBufferBytes: number;
  readonly maxDispatchWorkgroups: number;
  readonly workgroupStorageBytes: number;
}

function checked(label: string, value: number, max = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > max) {
    throw new RangeError(`NRC ${label} exceeds its checked integer range; got ${value}`);
  }
  return value;
}

function add(label: string, ...values: number[]): number {
  return checked(label, values.reduce((sum, value) => sum + value, 0));
}

function mul(label: string, ...values: number[]): number {
  let result = 1;
  for (const value of values) {
    checked(label, value);
    result *= value;
    checked(label, result);
  }
  return result;
}

function bytes(label: string, scalars: number, scalarBytes: number): number {
  const raw = mul(`${label} bytes`, scalars, scalarBytes);
  return Math.max(16, checked(`${label} aligned bytes`, Math.ceil(raw / 4) * 4));
}

function allocation(label: string, count: number, bytesEach: number): NrcBufferAllocation {
  checked(`${label} buffer count`, count);
  checked(`${label} bytes each`, bytesEach);
  return {
    count,
    bytesEach,
    totalBytes: mul(`${label} allocation bytes`, count, bytesEach),
  };
}

export function validateNrcAabb(
  aabbMin: readonly [number, number, number],
  aabbMax: readonly [number, number, number],
): void {
  for (let axis = 0; axis < 3; axis++) {
    const min = aabbMin[axis]!;
    const max = aabbMax[axis]!;
    if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min)) {
      throw new RangeError(`NRC AABB axis ${axis} must be finite and strictly ordered; got [${min}, ${max}]`);
    }
  }
}

export function computeNrcResourceFootprint(cfg: NrcConfig): NrcResourceFootprint {
  if (cfg.maxNrcResidentBytes !== undefined
      && (!Number.isSafeInteger(cfg.maxNrcResidentBytes)
          || cfg.maxNrcResidentBytes <= 0)) {
    throw new RangeError(
      `NRC maxNrcResidentBytes must be a positive safe integer; got ${cfg.maxNrcResidentBytes}`,
    );
  }
  const inW = add('encoded input width', mul('encoded hash width', cfg.levels, cfg.featuresPerEntry), mul('one-blob width', 2, cfg.oneBlobBins), 7);
  checked('encoded input width', inW, U32_MAX);
  if (inW > cfg.width) throw new RangeError(`NRC encoded input width ${inW} exceeds MLP width ${cfg.width}`);
  const recordStride = add('record stride', inW, 6);
  checked('record stride', recordStride, U32_MAX);
  const recordScalars = mul('record scalar count', cfg.recordCap, recordStride);
  const recordBytes = bytes('record buffer', recordScalars, F32_BYTES);
  const readbackBytes = add('readback buffer bytes', recordBytes, NRC_DIAGNOSTIC_BYTES);
  const tableScalars = mul('hash table scalar count', cfg.levels, cfg.tableSize, cfg.featuresPerEntry);
  checked('hash table scalar count', tableScalars, U32_MAX);
  const hiddenWeights = mul('hidden weight count', cfg.hidden, cfg.width, cfg.width);
  const outputWeights = mul('output weight count', cfg.width, 3);
  const totalWeights = add('MLP weight count', hiddenWeights, outputWeights);
  const totalBiases = add('MLP bias count', mul('hidden bias count', cfg.hidden, cfg.width), 3);
  checked('MLP weight count', totalWeights, U32_MAX);
  checked('MLP bias count', totalBiases, U32_MAX);
  const nodeLayers = add('node layer count', cfg.hidden, 2);
  const scalarBytes = cfg.useF16 ? 2 : 4;

  const levelResolutions: number[] = [];
  for (let level = 0; level < cfg.levels; level++) {
    const resolution = Math.floor(cfg.nMin * Math.pow(cfg.growth, level));
    if (!Number.isSafeInteger(resolution) || resolution <= 0 || resolution > U32_MAX) {
      throw new RangeError(`NRC level ${level} resolution is outside u32 range; got ${resolution}`);
    }
    levelResolutions.push(resolution);
  }

  const storageBindings: Record<string, number> = {
    weights: bytes('forward weights', totalWeights, scalarBytes),
    biases: bytes('forward biases', totalBiases, scalarBytes),
    weightMaster: bytes('master weights', totalWeights, F32_BYTES),
    biasMaster: bytes('master biases', totalBiases, F32_BYTES),
    inputs: bytes('inputs', mul('input scalars', cfg.recordCap, inW), scalarBytes),
    targets: bytes('targets', mul('target scalars', cfg.recordCap, 3), F32_BYTES),
    activations: bytes('activations', mul('activation scalars', cfg.recordCap, nodeLayers, cfg.width), scalarBytes),
    preActivations: bytes('pre-activations', mul('pre-activation scalars', cfg.recordCap, nodeLayers, cfg.width), scalarBytes),
    weightGradients: bytes('weight gradients', totalWeights, F32_BYTES),
    biasGradients: bytes('bias gradients', totalBiases, F32_BYTES),
    inputGradients: bytes('input gradients', mul('input gradient scalars', cfg.recordCap, inW), F32_BYTES),
    tables: bytes('hash tables', tableScalars, F32_BYTES),
    levels: bytes('level descriptors', cfg.levels, 16),
    records: recordBytes,
    slotClaims: bytes('slot claims', cfg.recordCap, 4),
    diagnostics: NRC_DIAGNOSTIC_BYTES,
    tableGradients: bytes('table gradients', tableScalars, F32_BYTES),
    positions: bytes('query positions', mul('position scalars', cfg.recordCap, 3), F32_BYTES),
    candidateWeights: bytes('candidate forward weights', totalWeights, scalarBytes),
    candidateBiases: bytes('candidate forward biases', totalBiases, scalarBytes),
    candidateWeightMaster: bytes('candidate master weights', totalWeights, F32_BYTES),
    candidateBiasMaster: bytes('candidate master biases', totalBiases, F32_BYTES),
    candidateTables: bytes('candidate hash tables', tableScalars, F32_BYTES),
    weightMoment: bytes('weight moment', totalWeights, F32_BYTES),
    biasMoment: bytes('bias moment', totalBiases, F32_BYTES),
    tableMoment: bytes('table moment', tableScalars, F32_BYTES),
    candidateWeightMoment: bytes('candidate weight moment', totalWeights, F32_BYTES),
    candidateBiasMoment: bytes('candidate bias moment', totalBiases, F32_BYTES),
    candidateTableMoment: bytes('candidate table moment', tableScalars, F32_BYTES),
  };
  const uniformBindings: Record<string, number> = {
    queryConfig: 48,
    layerPlan: add('layer-plan UBO bytes', 16, mul('layer-plan rows', cfg.hidden + 1, 16)),
    adam: 48,
    finalize: 16,
    encodeBackward: 32,
    ...(cfg.useF16 ? { downcast: 16 } : {}),
  };
  const inferenceArena = createNrcInferenceArenaLayout({
    weightsBytes: storageBindings.weightMaster!,
    biasesBytes: storageBindings.biasMaster!,
    tablesBytes: storageBindings.tables!,
    levelsBytes: storageBindings.levels!,
  });
  const runtimeArena = createNrcRuntimeArenaLayout({
    diagnosticsBytes: storageBindings.diagnostics!,
    claimsBytes: storageBindings.slotClaims!,
    recordsBytes: storageBindings.records!,
  });
  storageBindings.inferenceArena = inferenceArena.byteSize;
  storageBindings.runtimeArena = runtimeArena.byteSize;
  const liveTrainableBytes = add(
    'live trainable footprint',
    storageBindings.weights!, storageBindings.biases!,
    storageBindings.weightMaster!, storageBindings.biasMaster!,
    mul('live weight moment footprint', 2, storageBindings.weightMoment!),
    mul('live bias moment footprint', 2, storageBindings.biasMoment!),
    storageBindings.tables!,
    mul('live table moment footprint', 2, storageBindings.tableMoment!),
  );
  const candidateTrainableBytes = add(
    'candidate trainable footprint',
    storageBindings.candidateWeights!, storageBindings.candidateBiases!,
    storageBindings.candidateWeightMaster!, storageBindings.candidateBiasMaster!,
    mul('candidate weight moment footprint', 2, storageBindings.candidateWeightMoment!),
    mul('candidate bias moment footprint', 2, storageBindings.candidateBiasMoment!),
    storageBindings.candidateTables!,
    mul('candidate table moment footprint', 2, storageBindings.candidateTableMoment!),
  );
  const peakTrainableBytes = add(
    'peak doubled trainable footprint', liveTrainableBytes, candidateTrainableBytes,
  );

  // Multiplicities mirror every persistent GPUBuffer allocation in
  // NrcSubsystem + FusedMlpTrainer + HashGridTableTrainer. The default f32
  // configuration totals 49 resident buffers; f16 adds two downcast UBOs.
  const persistentAllocations: Record<string, NrcBufferAllocation> = {
    mlpForwardWeights: allocation('MLP forward weights', 2, storageBindings.weights!),
    mlpForwardBiases: allocation('MLP forward biases', 2, storageBindings.biases!),
    mlpMasterWeights: allocation('MLP master weights', 2, storageBindings.weightMaster!),
    mlpMasterBiases: allocation('MLP master biases', 2, storageBindings.biasMaster!),
    mlpWeightMoments: allocation('MLP weight moments', 4, storageBindings.weightMoment!),
    mlpBiasMoments: allocation('MLP bias moments', 4, storageBindings.biasMoment!),
    hashTables: allocation('hash tables', 2, storageBindings.tables!),
    hashTableMoments: allocation('hash-table moments', 4, storageBindings.tableMoment!),
    inputs: allocation('input', 1, storageBindings.inputs!),
    targets: allocation('target', 1, storageBindings.targets!),
    activations: allocation('activation', 1, storageBindings.activations!),
    preActivations: allocation('pre-activation', 1, storageBindings.preActivations!),
    weightGradientBuffers: allocation('weight-gradient', 2, storageBindings.weightGradients!),
    biasGradientBuffers: allocation('bias-gradient', 2, storageBindings.biasGradients!),
    inputGradientBuffers: allocation('input-gradient', 2, storageBindings.inputGradients!),
    tableGradientBuffers: allocation('table-gradient', 2, storageBindings.tableGradients!),
    positions: allocation('position', 1, storageBindings.positions!),
    levels: allocation('level descriptor', 1, storageBindings.levels!),
    inferenceArenas: allocation('inference arena', 2, storageBindings.inferenceArena),
    runtimeArena: allocation('runtime arena', 1, storageBindings.runtimeArena),
    queryConfigUniform: allocation('query-config uniform', 1, uniformBindings.queryConfig!),
    layerPlanUniform: allocation('layer-plan uniform', 1, uniformBindings.layerPlan!),
    adamUniforms: allocation('Adam uniforms', 3, uniformBindings.adam!),
    finalizeUniforms: allocation('finalize uniforms', 4, uniformBindings.finalize!),
    encodeBackwardUniform: allocation('encode-backward uniform', 1, uniformBindings.encodeBackward!),
    downcastUniforms: allocation('downcast uniforms', cfg.useF16 ? 2 : 0, 16),
  };
  const allocationValues = Object.values(persistentAllocations);
  const persistentBufferCount = add(
    'persistent GPU-buffer count', ...allocationValues.map((entry) => entry.count),
  );
  const persistentBufferBytes = add(
    'persistent GPU-buffer bytes', ...allocationValues.map((entry) => entry.totalBytes),
  );
  // Exactly one generation-tagged MAP_READ buffer may overlap the persistent
  // set; recordCopyForReadback rejects a second concurrent ticket.
  const peakResidentBufferCount = add('peak resident GPU-buffer count', persistentBufferCount, 1);
  const peakResidentBufferBytes = add('peak resident GPU-buffer bytes', persistentBufferBytes, readbackBytes);
  const peakBufferBytes = Math.max(
    readbackBytes,
    ...Object.values(storageBindings),
    ...Object.values(uniformBindings),
  );
  const maxDispatchWorkgroups = Math.max(
    Math.ceil(cfg.recordCap / cfg.tileB),
    Math.ceil(cfg.recordCap / 64),
    // The fused backward path finalises one dL/dX scalar for every active
    // record/input lane. This is the largest default dispatch
    // (4096 records * 39 lanes / 64 = 2496 workgroups), so it must participate
    // in the adapter preflight alongside the per-record, parameter, and table
    // kernels.
    Math.ceil(mul('input-gradient dispatch lanes', cfg.recordCap, inW) / 64),
    Math.ceil(totalWeights / 64),
    Math.ceil(totalBiases / 64),
    Math.ceil(tableScalars / 64),
  );
  const workgroupStorageBytes = mul('workgroup storage bytes', 2, cfg.tileB, cfg.width, scalarBytes);
  return {
    inW, recordStride, recordScalars, recordBytes, readbackBytes,
    tableScalars, totalWeights, totalBiases, levelResolutions,
    inferenceArenaLayout: inferenceArena,
    runtimeArenaLayout: runtimeArena,
    storageBindings, uniformBindings, persistentAllocations, peakBufferBytes,
    liveTrainableBytes, candidateTrainableBytes, peakTrainableBytes,
    persistentBufferCount, persistentBufferBytes,
    peakResidentBufferCount, peakResidentBufferBytes,
    maxDispatchWorkgroups, workgroupStorageBytes,
  };
}

function reportedLimit(device: GPUDevice, name: keyof GPUSupportedLimits): number | undefined {
  const value = Number(device.limits[name]);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function enforceLimit(label: string, required: number, available: number | undefined): void {
  if (available !== undefined && required > available) {
    throw new RangeError(`NRC ${label} requires ${required}, adapter reports ${available}`);
  }
}

export function preflightNrcResources(
  device: GPUDevice,
  cfg: NrcConfig,
  aabbMin: readonly [number, number, number],
  aabbMax: readonly [number, number, number],
): NrcResourceFootprint {
  validateNrcAabb(aabbMin, aabbMax);
  const footprint = computeNrcResourceFootprint(cfg);
  if (cfg.maxNrcResidentBytes !== undefined
      && footprint.peakResidentBufferBytes > cfg.maxNrcResidentBytes) {
    throw new RangeError(
      `NRC peak resident GPU-buffer footprint requires ${footprint.peakResidentBufferBytes}, `
      + `host maxNrcResidentBytes budget is ${cfg.maxNrcResidentBytes}`,
    );
  }
  if (cfg.useF16 && !device.features.has('shader-f16')) {
    throw new Error('NRC useF16 requires the shader-f16 device feature');
  }
  const maxBufferSize = reportedLimit(device, 'maxBufferSize');
  const maxStorageBinding = reportedLimit(device, 'maxStorageBufferBindingSize');
  validateNrcArenaLayouts({
    inference: {
      layout: footprint.inferenceArenaLayout,
      allocationBytes: footprint.storageBindings.inferenceArena!,
      epoch: 1,
      generation: 0,
    },
    runtime: {
      layout: footprint.runtimeArenaLayout,
      allocationBytes: footprint.storageBindings.runtimeArena!,
      epoch: 1,
      generation: 0,
      recordCap: cfg.recordCap,
      recordStride: footprint.recordStride,
    },
    limits: {
      ...(maxBufferSize === undefined ? {} : { maxBufferSize }),
      ...(maxStorageBinding === undefined
        ? {}
        : { maxStorageBufferBindingSize: maxStorageBinding }),
    },
  });
  const maxUniformBinding = reportedLimit(device, 'maxUniformBufferBindingSize');
  for (const [name, size] of Object.entries(footprint.storageBindings)) {
    enforceLimit(`${name} buffer size`, size, maxBufferSize);
    enforceLimit(`${name} storage binding size`, size, maxStorageBinding);
  }
  enforceLimit('readback buffer size', footprint.readbackBytes, maxBufferSize);
  for (const [name, size] of Object.entries(footprint.uniformBindings)) {
    enforceLimit(`${name} uniform binding size`, size, maxUniformBinding);
    enforceLimit(`${name} buffer size`, size, maxBufferSize);
  }
  enforceLimit('MLP workgroup width', cfg.width, reportedLimit(device, 'maxComputeInvocationsPerWorkgroup'));
  enforceLimit('MLP workgroup X size', cfg.width, reportedLimit(device, 'maxComputeWorkgroupSizeX'));
  enforceLimit('fixed 64-lane kernels', 64, reportedLimit(device, 'maxComputeInvocationsPerWorkgroup'));
  enforceLimit('fixed 64-lane kernel X size', 64, reportedLimit(device, 'maxComputeWorkgroupSizeX'));
  enforceLimit('workgroup storage', footprint.workgroupStorageBytes, reportedLimit(device, 'maxComputeWorkgroupStorageSize'));
  enforceLimit('dispatch workgroup count', footprint.maxDispatchWorkgroups, reportedLimit(device, 'maxComputeWorkgroupsPerDimension'));
  // Production gi-ris NRC is deliberately pinned to the portable WebGPU floor:
  // frame/scene/ubo/hybrid-NRC = four groups and exactly eight storage bindings.
  enforceLimit('gi-ris bind groups', 4, reportedLimit(device, 'maxBindGroups'));
  enforceLimit(
    'gi-ris storage bindings', 8,
    reportedLimit(device, 'maxStorageBuffersPerShaderStage'),
  );
  enforceLimit(
    'gi-ris sampled textures', 16,
    reportedLimit(device, 'maxSampledTexturesPerShaderStage'),
  );
  return footprint;
}
