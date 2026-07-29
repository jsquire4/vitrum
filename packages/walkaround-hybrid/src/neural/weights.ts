/**
 * weights.ts — ModelWeights interface + binary weight loader for the vitrum neural denoiser.
 *
 * Binary format (.vitrum-model):
 * ─────────────────────────────
 * Header (v2):
 *   [u32 magic=0xDEAF1984, u32 version=2, u32 layerCount,
 *    u32 metadataLength, u8[metadataLength] canonical UTF-8 JSON metadata]
 * Legacy v1 omits metadataLength and metadata bytes.
 * Per layer:
 *   [u32 nameLen, char[nameLen] name (UTF-8),
 *    u32 weightCount, f32[weightCount] weights,
 *    u32 biasCount,   f32[biasCount]   biases]
 *
 * Weight layout per layer kind:
 *   Conv2D:          OIKW  (outputC × inputC × kH × kW)  — matches PyTorch Conv2d
 *   ConvTranspose2D: IOKW  (inputC × outputC × kH × kW) — matches PyTorch ConvTranspose2d
 *   All others:      No weights (empty arrays).

 *
 * This format is round-tripped by tools/neural-denoiser-training/export_weights.py.
 */
import {
  NEURAL_PREPROCESSING_CONTRACT,
  isNeuralPreprocessingContract,
  neuralPreprocessingContractsEqual,
  type NeuralPreprocessingContract,
} from './preprocessing.js';
import { neuralCheckpointPayloadSha256 } from './checkpointDigest.js';

export const VITRUM_MODEL_MAGIC   = 0xDEAF1984 >>> 0;
export const VITRUM_MODEL_LEGACY_VERSION = 1;
export const VITRUM_MODEL_VERSION = 2;
export const NEURAL_MAX_PARAMETER_MAGNITUDE = 1024;
export const NEURAL_F16_VALIDATION_BOUNDS = Object.freeze({
  maxAbsError: 0.05,
  meanAbsError: 0.005,
  minPsnrDb: 35,
});
export const NEURAL_ARCHITECTURE_ID = 'vitrum-unet-9x3-v1' as const;
export const NEURAL_F16_QUANTIZATION =
  'f16-storage-per-logical-layer-f32-weight-bias-accumulation' as const;
export const NEURAL_F16_METRIC_DOMAIN = 'postprocessed-linear-hdr' as const;

// ── ModelWeights ──────────────────────────────────────────────────────────────

/** Per-layer weight payload. */
export interface LayerWeights {
  /** Layer name matching the LayerSpec.name in unetArchitecture.ts. */
  readonly name: string;
  /** Weight tensor as Float32Array. Length = 0 for non-parameterized layers. */
  readonly weights: Float32Array;
  /** Bias tensor as Float32Array. Length = 0 for layers with no bias. */
  readonly biases: Float32Array;
}

/** Full model weights for one U-Net checkpoint. */
export interface ModelWeights {
  /** Layer weights in execution order (matches UNetSpec.layers order). */
  readonly layers: readonly LayerWeights[];
  /** Binary container version; absent only for programmatically built weights. */
  readonly formatVersion?: 1 | 2;
  /** Optional provenance/quality metadata for production-readiness decisions. */
  readonly checkpoint?: NeuralCheckpointMetadata;
}

export type NeuralCheckpointAuxInput = 'albedo' | 'normal' | 'depth' | 'motion';

export interface NeuralCheckpointQualityReport {
  readonly status: 'pass' | 'fail' | 'unknown';
  readonly reportPath?: string;
  readonly validationScenes?: number;
  readonly psnrDb?: number;
  readonly ssim?: number;
}
export interface NeuralMixedPrecisionValidation {
  readonly checkpointSha256: string;
  readonly architecture: typeof NEURAL_ARCHITECTURE_ID;
  readonly preprocessing: NeuralPreprocessingContract;
  readonly quantization: typeof NEURAL_F16_QUANTIZATION;
  readonly metricDomain: typeof NEURAL_F16_METRIC_DOMAIN;
  readonly validationCorpusSha256: string;
  readonly status: 'pass' | 'fail';
  readonly validationScenes: number;
  readonly maxAbsError: number;
  readonly meanAbsError: number;
  readonly psnrDb: number;
  readonly finiteOutputs: boolean;
  readonly outputMin: number;
  readonly outputMax: number;
  readonly accumulation: 'f32';
  readonly weights: 'f32';
}


export interface NeuralCheckpointMetadata {
  readonly id: string;
  readonly trainingSamples: number;
  readonly noisySpp: number;
  readonly cleanSpp: number;
  readonly auxiliaryInputs: readonly NeuralCheckpointAuxInput[];
  readonly captureSource: string;
  /**
   * Activation/input/output tensor storage validated for this checkpoint.
   * Missing (including older v2 metadata) and `f32` are f32-only. The
   * `f16-compatible` value allows the runtime to select f16 storage when the
   * actual GPUDevice has `shader-f16`; learned weights and biases remain f32.
   */
  readonly tensorStorage?: 'f32' | 'f16-compatible';
  /** Required certification payload when tensorStorage is f16-compatible. */
  readonly mixedPrecision?: NeuralMixedPrecisionValidation;
  readonly captureBackend: string;
  readonly tonemap: string;
  readonly hardware: string;
  /** Exact preprocessing used for both training inputs and training targets. */
  readonly preprocessing?: NeuralPreprocessingContract;
  readonly qualityReport: NeuralCheckpointQualityReport;
}

export interface NeuralCheckpointProductionAssessment {
  readonly productionReady: boolean;
  readonly missing: readonly string[];
  readonly metadata?: NeuralCheckpointMetadata;
}

export const NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS = Object.freeze({
  minTrainingSamples: 500,
  noisySpp: 1,
  minCleanSpp: 4096,
  requiredAuxiliaryInputs: ['albedo', 'normal'] as const,
  requiredQualityStatus: 'pass' as const,
});

export function assessNeuralCheckpointProductionReadiness(
  weights: ModelWeights | undefined,
): NeuralCheckpointProductionAssessment {
  const missing: string[] = [];
  if (weights?.formatVersion !== VITRUM_MODEL_VERSION) {
    missing.push(`formatVersion=${VITRUM_MODEL_VERSION}`);
  }
  const metadata = weights?.checkpoint;
  if (metadata == null) {
    missing.push('checkpoint metadata');
    return { productionReady: false, missing };
  }

  if (typeof metadata.id !== 'string' || metadata.id.length === 0) {
    missing.push('checkpoint.id');
  }
  if (!isFiniteAtLeast(metadata.trainingSamples, NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.minTrainingSamples)) {
    missing.push(`trainingSamples>=${NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.minTrainingSamples}`);
  }
  if (metadata.noisySpp !== NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.noisySpp) {
    missing.push(`noisySpp=${NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.noisySpp}`);
  }
  if (!isFiniteAtLeast(metadata.cleanSpp, NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.minCleanSpp)) {
    missing.push(`cleanSpp>=${NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.minCleanSpp}`);
  }
  const auxiliaryInputs = Array.isArray(metadata.auxiliaryInputs) ? metadata.auxiliaryInputs : [];
  if (auxiliaryInputs.length === 0) {
    missing.push('auxiliaryInputs');
  }
  for (const input of NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.requiredAuxiliaryInputs) {
    if (!auxiliaryInputs.includes(input)) {
      missing.push(`auxiliaryInputs.${input}`);
    }
  }
  for (const field of ['captureSource', 'captureBackend', 'tonemap', 'hardware'] as const) {
    if (typeof metadata[field] !== 'string' || metadata[field].length === 0) {
      missing.push(field);
    }
  }
  if (metadata.qualityReport?.status !== NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS.requiredQualityStatus) {
    missing.push('qualityReport.status=pass');
  }
  if (typeof metadata.qualityReport?.reportPath !== 'string' || metadata.qualityReport.reportPath.length === 0) {
    missing.push('qualityReport.reportPath');
  }
  if (!isNeuralPreprocessingContract(metadata.preprocessing)) {
    missing.push('preprocessing');
  } else if (!neuralPreprocessingContractsEqual(
    metadata.preprocessing,
    NEURAL_PREPROCESSING_CONTRACT,
  )) {
    missing.push('preprocessing=runtime-contract');
  }

  if (metadata.tensorStorage === 'f16-compatible' && !isNeuralCheckpointF16Compatible(weights)) {
    missing.push('mixedPrecision=f16-certified');
  }

  return { productionReady: missing.length === 0, missing, metadata };
}

export function isNeuralCheckpointProductionReady(weights: ModelWeights | undefined): boolean {
  return assessNeuralCheckpointProductionReadiness(weights).productionReady;
}

function isFiniteAtLeast(value: number, min: number): boolean {
  return Number.isFinite(value) && value >= min;
}

export function isNeuralCheckpointF16Compatible(
  metadataOrWeights: NeuralCheckpointMetadata | ModelWeights | undefined,
): boolean {
  const weights: ModelWeights | undefined =
    metadataOrWeights != null && 'layers' in metadataOrWeights
    ? metadataOrWeights
    : undefined;
  const metadata: NeuralCheckpointMetadata | undefined = weights != null
    ? weights.checkpoint
    : metadataOrWeights as NeuralCheckpointMetadata | undefined;
  return metadata?.tensorStorage === 'f16-compatible' &&
    mixedPrecisionCertificationValid(metadata.mixedPrecision, metadata.preprocessing) &&
    (weights == null ||
      metadata.mixedPrecision?.checkpointSha256 === neuralCheckpointPayloadSha256(weights.layers));
}

function mixedPrecisionCertificationValid(
  report: NeuralMixedPrecisionValidation | undefined,
  preprocessing: NeuralPreprocessingContract | undefined,
): boolean {
  return isNeuralPreprocessingContract(preprocessing) &&
    isNeuralPreprocessingContract(report?.preprocessing) &&
    neuralPreprocessingContractsEqual(preprocessing, report.preprocessing) &&
    /^[0-9a-f]{64}$/.test(report.checkpointSha256) &&
    /^[0-9a-f]{64}$/.test(report.validationCorpusSha256) &&
    report.architecture === NEURAL_ARCHITECTURE_ID &&
    report.quantization === NEURAL_F16_QUANTIZATION &&
    report.metricDomain === NEURAL_F16_METRIC_DOMAIN &&
    report?.status === 'pass' && report.finiteOutputs === true &&
    report.accumulation === 'f32' && report.weights === 'f32' &&
    Number.isInteger(report.validationScenes) && report.validationScenes > 0 &&
    [report.maxAbsError, report.meanAbsError, report.psnrDb, report.outputMin, report.outputMax].every(Number.isFinite) &&
    report.maxAbsError >= 0 &&
    report.maxAbsError <= NEURAL_F16_VALIDATION_BOUNDS.maxAbsError &&
    report.meanAbsError >= 0 &&
    report.meanAbsError <= NEURAL_F16_VALIDATION_BOUNDS.meanAbsError &&
    report.psnrDb >= NEURAL_F16_VALIDATION_BOUNDS.minPsnrDb &&
    report.outputMin >= 0 && report.outputMin <= report.outputMax &&
    report.outputMax <= preprocessing.radianceClamp;
}

type UnknownRecord = Record<string, unknown>;

const NEURAL_CHECKPOINT_AUX_INPUTS = new Set<NeuralCheckpointAuxInput>([
  'albedo',
  'normal',
  'depth',
  'motion',
]);
const NEURAL_CHECKPOINT_QUALITY_STATUSES = new Set([
  'pass',
  'fail',
  'unknown',
] as const);

const NEURAL_CHECKPOINT_TENSOR_STORAGE = new Set([
  'f32',
  'f16-compatible',
] as const);
function requireRecord(value: unknown, label: string): UnknownRecord {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`[loadWeightsFromArrayBuffer] ${label} must be a JSON object`);
  }
  return value as UnknownRecord;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== 'string') {
    throw new Error(`[loadWeightsFromArrayBuffer] ${label} must be a string`);
  }
  return value;
}

function requireInteger(
  value: unknown,
  label: string,
  minimum: number,
): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] ${label} must be an integer >= ${minimum}`,
    );
  }
  return value as number;
}

function optionalFiniteNumber(
  value: unknown,
  label: string,
): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`[loadWeightsFromArrayBuffer] ${label} must be finite`);
  }
  return value;
}

function requireFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`[loadWeightsFromArrayBuffer] ${label} must be finite`);
  }
  return value;
}

function parseCheckpointMetadata(value: unknown): NeuralCheckpointMetadata {
  const metadata = requireRecord(value, 'v2 metadata');
  const auxValue = metadata.auxiliaryInputs;
  if (!Array.isArray(auxValue)) {
    throw new Error('[loadWeightsFromArrayBuffer] metadata.auxiliaryInputs must be an array');
  }
  const entries: readonly unknown[] = auxValue;
  const auxiliaryInputs: NeuralCheckpointAuxInput[] = [];
  const seenAux = new Set<NeuralCheckpointAuxInput>();
  for (let i = 0; i < auxValue.length; i++) {
    const entry = entries[i];
    if (typeof entry !== 'string' || !NEURAL_CHECKPOINT_AUX_INPUTS.has(entry as NeuralCheckpointAuxInput)) {
      throw new Error(
        `[loadWeightsFromArrayBuffer] metadata.auxiliaryInputs[${i}] has unknown enum value '${String(entry)}'`,
      );
    }
    const input = entry as NeuralCheckpointAuxInput;
    if (seenAux.has(input)) {
      throw new Error(
        `[loadWeightsFromArrayBuffer] metadata.auxiliaryInputs contains duplicate '${input}'`,
      );
    }
    seenAux.add(input);
    auxiliaryInputs.push(input);
  }

  const quality = requireRecord(metadata.qualityReport, 'metadata.qualityReport');
  const qualityStatus = quality.status;
  if (
    typeof qualityStatus !== 'string' ||
    !NEURAL_CHECKPOINT_QUALITY_STATUSES.has(
      qualityStatus as 'pass' | 'fail' | 'unknown',
    )
  ) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] metadata.qualityReport.status has unknown enum value '${String(qualityStatus)}'`,
    );
  }
  const reportPath = quality.reportPath === undefined
    ? undefined
    : requireString(quality.reportPath, 'metadata.qualityReport.reportPath');
  const validationScenes = quality.validationScenes === undefined
    ? undefined
    : requireInteger(quality.validationScenes, 'metadata.qualityReport.validationScenes', 0);
  const psnrDb = optionalFiniteNumber(quality.psnrDb, 'metadata.qualityReport.psnrDb');
  const ssim = optionalFiniteNumber(quality.ssim, 'metadata.qualityReport.ssim');
  if (ssim !== undefined && (ssim < 0 || ssim > 1)) {
    throw new Error('[loadWeightsFromArrayBuffer] metadata.qualityReport.ssim must be in [0, 1]');
  }

  if (!isNeuralPreprocessingContract(metadata.preprocessing)) {
    throw new Error(
      '[loadWeightsFromArrayBuffer] metadata.preprocessing has an unsupported schema or enum value',
    );
  }

  const tensorStorage = metadata.tensorStorage;
  if (
    tensorStorage !== undefined &&
    (typeof tensorStorage !== 'string' ||
      !NEURAL_CHECKPOINT_TENSOR_STORAGE.has(tensorStorage as 'f32' | 'f16-compatible'))
  ) {
    throw new Error(
      '[loadWeightsFromArrayBuffer] metadata.tensorStorage has unknown enum value ' +
      Object.prototype.toString.call(tensorStorage),
    );
  }

  let mixedPrecision: NeuralMixedPrecisionValidation | undefined;
  if (metadata.mixedPrecision !== undefined) {
    const mixed = requireRecord(metadata.mixedPrecision, 'metadata.mixedPrecision');
    const status = requireString(mixed.status, 'metadata.mixedPrecision.status');
    if (status !== 'pass' && status !== 'fail') {
      throw new Error(
        `[loadWeightsFromArrayBuffer] metadata.mixedPrecision.status has unknown enum value '${status}'`,
      );
    }
    if (typeof mixed.finiteOutputs !== 'boolean') {
      throw new Error('[loadWeightsFromArrayBuffer] metadata.mixedPrecision.finiteOutputs must be boolean');
    }
    if (mixed.accumulation !== 'f32' || mixed.weights !== 'f32') {
      throw new Error(
        '[loadWeightsFromArrayBuffer] metadata.mixedPrecision requires f32 accumulation and weights',
      );
    }
    if (!isNeuralPreprocessingContract(mixed.preprocessing)) {
      throw new Error(
        '[loadWeightsFromArrayBuffer] metadata.mixedPrecision.preprocessing has an unsupported schema or enum value',
      );
    }
    const architecture = requireString(
      mixed.architecture,
      'metadata.mixedPrecision.architecture',
    );
    const quantization = requireString(
      mixed.quantization,
      'metadata.mixedPrecision.quantization',
    );
    const metricDomain = requireString(
      mixed.metricDomain,
      'metadata.mixedPrecision.metricDomain',
    );
    if (
      architecture !== NEURAL_ARCHITECTURE_ID ||
      quantization !== NEURAL_F16_QUANTIZATION ||
      metricDomain !== NEURAL_F16_METRIC_DOMAIN
    ) {
      throw new Error(
        '[loadWeightsFromArrayBuffer] metadata.mixedPrecision architecture/quantization/metric domain is unsupported',
      );
    }
    mixedPrecision = {
      checkpointSha256: requireString(mixed.checkpointSha256, 'metadata.mixedPrecision.checkpointSha256'),
      architecture: NEURAL_ARCHITECTURE_ID,
      preprocessing: mixed.preprocessing,
      quantization: NEURAL_F16_QUANTIZATION,
      metricDomain: NEURAL_F16_METRIC_DOMAIN,
      validationCorpusSha256: requireString(
        mixed.validationCorpusSha256,
        'metadata.mixedPrecision.validationCorpusSha256',
      ),
      status,
      validationScenes: requireInteger(
        mixed.validationScenes,
        'metadata.mixedPrecision.validationScenes',
        1,
      ),
      maxAbsError: requireFiniteNumber(mixed.maxAbsError, 'metadata.mixedPrecision.maxAbsError'),
      meanAbsError: requireFiniteNumber(mixed.meanAbsError, 'metadata.mixedPrecision.meanAbsError'),
      psnrDb: requireFiniteNumber(mixed.psnrDb, 'metadata.mixedPrecision.psnrDb'),
      finiteOutputs: mixed.finiteOutputs,
      outputMin: requireFiniteNumber(mixed.outputMin, 'metadata.mixedPrecision.outputMin'),
      outputMax: requireFiniteNumber(mixed.outputMax, 'metadata.mixedPrecision.outputMax'),
      accumulation: 'f32',
      weights: 'f32',
    };
    if (
      mixedPrecision.maxAbsError < 0 ||
      mixedPrecision.meanAbsError < 0 ||
      mixedPrecision.outputMin < 0 ||
      mixedPrecision.outputMin > mixedPrecision.outputMax
    ) {
      throw new Error('[loadWeightsFromArrayBuffer] metadata.mixedPrecision metrics/bounds are invalid');
    }
  }
  if (
    tensorStorage === 'f16-compatible' &&
    !mixedPrecisionCertificationValid(mixedPrecision, metadata.preprocessing)
  ) {
    throw new Error(
      '[loadWeightsFromArrayBuffer] f16-compatible tensorStorage requires passing mixedPrecision certification within preprocessing output bounds',
    );
  }

  return {
    id: requireString(metadata.id, 'metadata.id'),
    trainingSamples: requireInteger(metadata.trainingSamples, 'metadata.trainingSamples', 0),
    noisySpp: requireInteger(metadata.noisySpp, 'metadata.noisySpp', 1),
    cleanSpp: requireInteger(metadata.cleanSpp, 'metadata.cleanSpp', 1),
    auxiliaryInputs,
    captureSource: requireString(metadata.captureSource, 'metadata.captureSource'),
    captureBackend: requireString(metadata.captureBackend, 'metadata.captureBackend'),
    tonemap: requireString(metadata.tonemap, 'metadata.tonemap'),
    hardware: requireString(metadata.hardware, 'metadata.hardware'),
    preprocessing: {
      version: metadata.preprocessing.version,
      color: metadata.preprocessing.color,
      radianceScale: metadata.preprocessing.radianceScale,
      radianceClamp: metadata.preprocessing.radianceClamp,
      albedoRange: [0, 1],
      normalEncoding: metadata.preprocessing.normalEncoding,
      nonFinite: metadata.preprocessing.nonFinite,
    },
    qualityReport: {
      status: qualityStatus as 'pass' | 'fail' | 'unknown',
      ...(reportPath !== undefined ? { reportPath } : {}),
      ...(validationScenes !== undefined ? { validationScenes } : {}),
      ...(psnrDb !== undefined ? { psnrDb } : {}),
      ...(ssim !== undefined ? { ssim } : {}),
    },
    ...(tensorStorage !== undefined
      ? { tensorStorage: tensorStorage as 'f32' | 'f16-compatible' }
      : {}),
    ...(mixedPrecision !== undefined
      ? { mixedPrecision }
      : {}),
  };
}

/**
 * Return metadata with a stable field order so TypeScript and Python exporters
 * produce deterministic v2 bytes for the same checkpoint.
 */
function canonicalCheckpointMetadata(
  metadata: NeuralCheckpointMetadata,
): NeuralCheckpointMetadata {
  return parseCheckpointMetadata(metadata);
}

function stableJsonStringify(value: unknown): string {
  const canonicalize = (entry: unknown): unknown => {
    if (Array.isArray(entry)) return entry.map(canonicalize);
    if (entry != null && typeof entry === 'object') {
      const record = entry as Record<string, unknown>;
      const result: Record<string, unknown> = {};
      for (const key of Object.keys(record).sort()) {
        result[key] = canonicalize(record[key]);
      }
      return result;
    }
    return entry;
  };
  return JSON.stringify(canonicalize(value));
}

// ── Spec validation ─────────────────────────────────────────────────────────

type WeightLayerSpec = {
  readonly name: string;
  readonly kind: string;
  readonly weightLayout: 'OIKW' | 'IOKW' | 'none';
  readonly params: {
    readonly inC: number;
    readonly outC: number;
    readonly kH?: number;
    readonly kW?: number;
  };
};

export interface WeightSpec {
  readonly layers: readonly WeightLayerSpec[];
}

/**
 * Validate that a checkpoint exactly matches the supplied inference spec before
 * any GPU buffers are allocated.
 *
 * Exported training checkpoints are allowed to omit parameterless layers
 * (`relu`, `skipAdd`, `inputPack`, etc.), but every parameterized conv layer
 * must be present exactly once with the expected weight/bias lengths and finite
 * f32 payloads. Unknown layers, duplicate layers, parameterless layers with
 * payloads, and malformed values throw synchronously so `denoiser:'neural'`
 * cannot silently run with placeholder weights.
 */
export function validateWeightsForSpec(spec: WeightSpec, weights: ModelWeights): void {
  const specByName = new Map<string, WeightLayerSpec>();
  for (const layer of spec.layers) {
    if (specByName.has(layer.name)) {
      throw new Error(`[validateWeightsForSpec] duplicate spec layer '${layer.name}'`);
    }
    specByName.set(layer.name, layer);
  }

  const supplied = new Set<string>();
  for (const layerWeights of weights.layers) {
    if (supplied.has(layerWeights.name)) {
      throw new Error(`[validateWeightsForSpec] duplicate weights for layer '${layerWeights.name}'`);
    }
    supplied.add(layerWeights.name);

    const layer = specByName.get(layerWeights.name);
    if (layer == null) {
      throw new Error(`[validateWeightsForSpec] unknown layer '${layerWeights.name}' in checkpoint`);
    }

    const expected = expectedParamCounts(layer);
    if (layerWeights.weights.length !== expected.weights) {
      throw new Error(
        `[validateWeightsForSpec] layer '${layerWeights.name}' weight length ` +
        `${layerWeights.weights.length} != expected ${expected.weights}`,
      );
    }
    if (layerWeights.biases.length !== expected.biases) {
      throw new Error(
        `[validateWeightsForSpec] layer '${layerWeights.name}' bias length ` +
        `${layerWeights.biases.length} != expected ${expected.biases}`,
      );
    }
    assertFiniteArray(layerWeights.weights, `${layerWeights.name}.weights`);
    assertFiniteArray(layerWeights.biases, `${layerWeights.name}.biases`);
  }

  for (const layer of spec.layers) {
    const expected = expectedParamCounts(layer);
    if ((expected.weights > 0 || expected.biases > 0) && !supplied.has(layer.name)) {
      throw new Error(`[validateWeightsForSpec] missing weights for layer '${layer.name}'`);
    }
  }
}

function expectedParamCounts(layer: WeightLayerSpec): { weights: number; biases: number } {
  if (layer.kind !== 'conv2d' && layer.kind !== 'transposedConv2d') {
    return { weights: 0, biases: 0 };
  }
  const { inC, outC, kH = 1, kW = 1 } = layer.params;
  return {
    weights: inC * outC * kH * kW,
    biases: outC,
  };
}

function assertFiniteArray(values: Float32Array, label: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`[validateWeightsForSpec] ${label}[${i}] is not finite`);
    }
    if (Math.abs(values[i]!) > NEURAL_MAX_PARAMETER_MAGNITUDE) {
      throw new Error(
        `[validateWeightsForSpec] ${label}[${i}] exceeds magnitude bound ` +
        `${NEURAL_MAX_PARAMETER_MAGNITUDE}`,
      );
    }
  }
}

// ── Binary loader ─────────────────────────────────────────────────────────────

/**
 * Load model weights from an ArrayBuffer in the vitrum-model binary format.
 *
 * @throws {Error} on magic mismatch, unsupported version, or truncated data.
 *
 * Byte-level layout:
 *   Offset 0: u32 magic (little-endian)
 *   Offset 4: u32 version
 *   Offset 8: u32 layerCount
 *   Offset 12: layer records (variable-length)
 *
 * Each layer record:
 *   u32 nameLen
 *   u8[nameLen] name (UTF-8, not null-terminated)
 *   u32 weightCount
 *   f32[weightCount] weights
 *   u32 biasCount
 *   f32[biasCount] biases
 */
export function loadWeightsFromArrayBuffer(bytes: ArrayBuffer): ModelWeights {
  const view = new DataView(bytes);
  let offset = 0;

  const readU32 = (): number => {
    if (offset + 4 > bytes.byteLength) {
      throw new Error(`[loadWeightsFromArrayBuffer] truncated at offset ${offset}`);
    }
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readBytes = (count: number, label: string): Uint8Array => {
    if (!Number.isSafeInteger(count) || count < 0 || offset + count > bytes.byteLength) {
      throw new Error(`[loadWeightsFromArrayBuffer] truncated reading ${label} at offset ${offset}`);
    }
    const value = new Uint8Array(bytes.slice(offset, offset + count));
    offset += count;
    return value;
  };
  const readString = (count: number, label: string): string => {
    const encoded = readBytes(count, label);
    try {
      return new TextDecoder('utf-8', { fatal: true }).decode(encoded);
    } catch (error) {
      throw new Error(`[loadWeightsFromArrayBuffer] invalid UTF-8 in ${label}: ${errorMessage(error)}`);
    }
  };
  const readF32Array = (count: number): Float32Array => {
    if (!Number.isSafeInteger(count) || count < 0 || count > Math.floor((bytes.byteLength - offset) / 4)) {
      throw new Error(`[loadWeightsFromArrayBuffer] truncated reading ${count} f32s at offset ${offset}`);
    }
    const value = new Float32Array(bytes.slice(offset, offset + count * 4));
    offset += count * 4;
    return value;
  };

  const magic = readU32();
  if (magic !== VITRUM_MODEL_MAGIC) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] invalid magic: expected 0x${VITRUM_MODEL_MAGIC.toString(16).toUpperCase()}, ` +
      `got 0x${magic.toString(16).toUpperCase()}`,
    );
  }
  const version = readU32();
  if (version !== VITRUM_MODEL_LEGACY_VERSION && version !== VITRUM_MODEL_VERSION) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] unsupported version ${version} ` +
      `(supported: ${VITRUM_MODEL_LEGACY_VERSION}, ${VITRUM_MODEL_VERSION})`,
    );
  }
  const layerCount = readU32();

  let checkpoint: NeuralCheckpointMetadata | undefined;
  if (version === VITRUM_MODEL_VERSION) {
    const metadataLength = readU32();
    const metadataText = readString(metadataLength, 'v2 metadata JSON');
    let parsed: unknown;
    try {
      parsed = JSON.parse(metadataText);
    } catch (error) {
      throw new Error(`[loadWeightsFromArrayBuffer] invalid v2 metadata JSON: ${errorMessage(error)}`);
    }
    const metadataRecord = requireRecord(parsed, 'v2 metadata');
    if (Object.keys(metadataRecord).length > 0) checkpoint = parseCheckpointMetadata(metadataRecord);
  }

  const minimumLayerRecordBytes = 12;
  if (layerCount > Math.floor((bytes.byteLength - offset) / minimumLayerRecordBytes)) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] impossible layerCount ${layerCount} for ${bytes.byteLength - offset} remaining bytes`,
    );
  }

  const layers: LayerWeights[] = [];
  const layerNames = new Set<string>();
  for (let i = 0; i < layerCount; i++) {
    const nameLength = readU32();
    if (nameLength === 0) {
      throw new Error(`[loadWeightsFromArrayBuffer] layer ${i} has an empty name`);
    }
    const name = readString(nameLength, `layer ${i} name`);
    if (layerNames.has(name)) {
      throw new Error(`[loadWeightsFromArrayBuffer] duplicate layer name '${name}'`);
    }
    layerNames.add(name);
    const weights = readF32Array(readU32());
    const biases = readF32Array(readU32());
    assertFiniteArray(weights, `${name}.weights`);
    assertFiniteArray(biases, `${name}.biases`);
    layers.push({ name, weights, biases });
  }
  if (offset !== bytes.byteLength) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] trailing ${bytes.byteLength - offset} byte(s) after ` +
      `${layerCount} layer record(s) at offset ${offset}`,
    );
  }
  const loaded: ModelWeights = {
    layers,
    formatVersion: version,
    ...(checkpoint !== undefined ? { checkpoint } : {}),
  };
  if (
    checkpoint?.tensorStorage === 'f16-compatible' &&
    checkpoint.mixedPrecision?.checkpointSha256 !== neuralCheckpointPayloadSha256(layers)
  ) {
    throw new Error(
      '[loadWeightsFromArrayBuffer] mixedPrecision checkpointSha256 does not match the ordered tensor payload',
    );
  }
  return loaded;
}

// ── Binary serialiser (CPU-side mirror of export_weights.py) ──────────────────

/**
 * Serialize ModelWeights to the vitrum-model binary format.
 *
 * This is a TypeScript mirror of `tools/neural-denoiser-training/export_weights.py`'s
 * output stage, used in tests for round-trip validation.
 */
export interface SerializeWeightsOptions {
  readonly version?: 1 | 2;
}

export function serializeWeightsToArrayBuffer(
  weights: ModelWeights,
  options: SerializeWeightsOptions = {},
): ArrayBuffer {
  const version: number = options.version ?? VITRUM_MODEL_VERSION;
  if (version !== VITRUM_MODEL_LEGACY_VERSION && version !== VITRUM_MODEL_VERSION) {
    throw new RangeError(`unsupported vitrum model version ${version}`);
  }
  const checkpoint = weights.checkpoint;
  if (
    version === VITRUM_MODEL_VERSION &&
    checkpoint?.tensorStorage === 'f16-compatible' &&
    checkpoint.mixedPrecision?.checkpointSha256 !== neuralCheckpointPayloadSha256(weights.layers)
  ) {
    throw new Error(
      '[serializeWeightsToArrayBuffer] mixedPrecision checkpointSha256 does not match the ordered tensor payload',
    );
  }
  const encoder = new TextEncoder();
  const metadataValue = version !== VITRUM_MODEL_VERSION || weights.checkpoint === undefined
    ? {}
    : canonicalCheckpointMetadata(weights.checkpoint);
  const metadataBytes = version === VITRUM_MODEL_VERSION
    ? encoder.encode(stableJsonStringify(metadataValue))
    : new Uint8Array(0);

  if (weights.layers.length > 0xffff_ffff) {
    throw new RangeError('too many layers for vitrum-model u32 header');
  }
  const layerNames = new Set<string>();
  let totalBytes = 12 + (version === VITRUM_MODEL_VERSION ? 4 + metadataBytes.byteLength : 0);
  for (const layer of weights.layers) {
    if (layer.name.length === 0) {
      throw new Error('[serializeWeightsToArrayBuffer] layer names must not be empty');
    }
    if (layerNames.has(layer.name)) {
      throw new Error(`[serializeWeightsToArrayBuffer] duplicate layer name '${layer.name}'`);
    }
    layerNames.add(layer.name);
    assertFiniteArray(layer.weights, `${layer.name}.weights`);
    assertFiniteArray(layer.biases, `${layer.name}.biases`);
    const nameBytes = encoder.encode(layer.name);
    totalBytes += 4 + nameBytes.byteLength + 4 + layer.weights.byteLength + 4 + layer.biases.byteLength;
    if (!Number.isSafeInteger(totalBytes) || totalBytes > 0xffff_ffff) {
      throw new RangeError('vitrum-model binary exceeds the u32-addressable format limit');
    }
  }

  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  let offset = 0;
  const writeU32 = (value: number): void => {
    view.setUint32(offset, value >>> 0, true);
    offset += 4;
  };
  const writeRaw = (value: Uint8Array): void => {
    bytes.set(value, offset);
    offset += value.byteLength;
  };
  const writeF32Array = (value: Float32Array): void =>
    writeRaw(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
  const writeString = (value: string): void => {
    const encoded = encoder.encode(value);
    writeU32(encoded.byteLength);
    writeRaw(encoded);
  };

  writeU32(VITRUM_MODEL_MAGIC);
  writeU32(version);
  writeU32(weights.layers.length);
  if (version === VITRUM_MODEL_VERSION) {
    writeU32(metadataBytes.byteLength);
    writeRaw(metadataBytes);
  }
  for (const layer of weights.layers) {
    writeString(layer.name);
    writeU32(layer.weights.length);
    writeF32Array(layer.weights);
    writeU32(layer.biases.length);
    writeF32Array(layer.biases);
  }
  return buffer;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
