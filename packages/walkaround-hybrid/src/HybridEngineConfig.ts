/**
 * HybridEngineConfig — construction-time option parsing for {@link HybridEngine}.
 *
 * Extracted from `HybridEngine.ts` (R3 B-chain decomposition sweep).
 *
 * Contains:
 *   - {@link ParsedHybridEngineConfig} — the immutable derived-config record type.
 *   - {@link validateHybridEngineOptions} — pure throws (lite/neural/OIDN/denoiser guards).
 *   - {@link deriveHybridEngineConfig} — defaulting into the derived record plus construction diagnostics.
 *   - {@link parseHybridEngineOptions} — thin orchestrator over the two halves.
 *
 * No `this` dependency, no GPU side effects. Construction warnings preserve the
 * historical console output and also route through host `onWarning` when supplied.
 */

import type { EngineWarning } from '@vitrum/core';
import { quantizeOpenUnitProbabilityF32 } from '@vitrum/shared-samplers';
import type { HybridEngineOptions } from './HybridEngineOptions.js';
import { VALID_DENOISERS } from './HybridEngineOptions.js';
import { ATROUS_DIRECT_SIGMAS, ATROUS_INDIRECT_SIGMAS } from './pipeline/constants.js';
import { packStainedGlassFlags } from './pipeline/uboUpdater.js';
import {
  readTunables,
  readInitTunables,
  TUNABLE_DEFINITIONS,
  type Tunables,
  type InitTunables,
} from './HybridEngineTuning.js';
import { resolveQualityPreset } from './HybridEngineQualityPreset.js';
import { fingerprintHybridPipelineRebuildKey } from './HybridEngineFrameOrchestrator.js';
import type { ReSTIRBvhMode } from './restir/bvhCore.js';
import {
  assessNeuralCheckpointProductionReadiness,
  validateWeightsForSpec,
  type ModelWeights,
  type NeuralCheckpointProductionAssessment,
} from './neural/weights.js';
import { WALKAROUND_DENOISER_UNET_SPEC } from './neural/unetArchitecture.js';
import { walkaroundNeuralInferenceExtent } from './neural/shapeContract.js';
import { preflightTensorDims } from './neural/tensorDimSolver.js';
import { neuralDeviceLimitFailures } from './neural/tensorMemoryPlanner.js';
import {
  resolveNeuralTensorStorageDecision,
  type NeuralTensorPrecision,
  type NeuralTensorStoragePreference,
} from './neural/tensorPrecision.js';
import {
  PPG_MIS_ALPHA,
  PPG_MAX_SPATIAL_CELLS,
  PPG_MAX_DTREE_NODES_PER_CELL,
} from './ppg/ppgConstants.js';
import {
  RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
  RC_MAX_TRANSMITTED_INTERFACE_BUDGET,
  RC_MIN_TRANSMITTED_INTERFACE_BUDGET,
  validateCascadeDims,
} from '@vitrum/walkaround-rc';
import { assertValidDdgiLights } from './ddgi/inputValidation.js';
import {
  REGIR_MAX_CANDIDATES_PER_CELL,
  resolveReGIRConfig,
} from './pipeline/ReGIRCoordinator.js';
import {
  resolveNrcConfig,
  type NrcConfig,
} from './neural/nrc/nrcSubsystem.js';
import {
  DEFAULT_FRAME_RESOURCE_BUDGET_BYTES,
  DEFAULT_MAX_RESTIR_RESERVOIR_SCALE,
} from './pipeline/frameResourcePlan.js';
import {
  canonicalizeLightingDirectionF32,
  packNonNegativeLightingFloat32,
  packNonNegativeLightingRgbF32,
} from './lightingFloat32.js';

/** Default per-frame target interval (~60 FPS soft-cap). */
const DEFAULT_TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

/**
 * Closed constructor vocabulary. The `Record<keyof HybridEngineOptions, true>`
 * constraint makes additions to the public interface fail compilation until
 * this runtime boundary is updated as well.
 */
const HYBRID_ENGINE_OPTION_KEYS = {
  device: true,
  width: true,
  height: true,
  frameResourceResolutionPolicy: true,
  maxPersistentFrameResourceBytes: true,
  restirReservoirScale: true,
  gpuSkinning: true,
  isSceneReady: true,
  pipelineRebuildKey: true,
  getPipelineRebuildKey: true,
  primaryLightDir: true,
  primaryLightIntensity: true,
  skyTint: true,
  skyIrradiance: true,
  lights: true,
  ddgiMaxMaterials: true,
  verbose: true,
  debug: true,
  denoiser: true,
  neuralWeights: true,
  extensions: true,
  qualityTier: true,
  tier: true,
  gtaoMode: true,
  diSpatialPasses: true,
  giSpatialPasses: true,
  ddgiUpdateDivisor: true,
  targetFrameIntervalMs: true,
  cameraMoveResetThresholdSq: true,
  temporalAccumAlpha: true,
  tuning: true,
  caustic: true,
  stainedGlass: true,
  adaptiveSamplingThresholds: true,
  gtao: true,
  indirectFireflyClamp: true,
  atrousDirectSigmas: true,
  atrousIndirectSigmas: true,
  grisReuse: true,
  restirPtReuse: true,
  checkerboardRendering: true,
  checkerboardMotionThresholdSq: true,
  ppgEnabled: true,
  ppgMaxSpatialCells: true,
  ppgMaxDTreeNodesPerCell: true,
  ppgMixAlpha: true,
  ppgDispatchInterval: true,
  regir: true,
  rcEnabled: true,
  rcTransmittedInterfaceBudget: true,
  rcWeight: true,
  cascadeDims: true,
  nrcEnabled: true,
  nrcConfig: true,
  nrcWarmupSteps: true,
  nrcSpreadC: true,
  nrcMaxResidentBytes: true,
  maxBounces: true,
  maxSamplesPerPixel: true,
  onWarning: true,
  causticStrategy: true,
  causticOptions: true,
} as const satisfies Readonly<Record<keyof HybridEngineOptions, true>>;

/**
 * Backend-owned options accepted through `@vitrum/engine`'s advanced bag.
 * Device and surface dimensions are owned by the facade and are deliberately
 * absent from this type and from the runtime vocabulary below.
 */
export type HybridEngineAdvancedOptions = Partial<
  Omit<HybridEngineOptions, 'device' | 'width' | 'height'>
>;

const {
  device: _deviceOption,
  width: _widthOption,
  height: _heightOption,
  ...HYBRID_ENGINE_ADVANCED_OPTION_KEYS
} = HYBRID_ENGINE_OPTION_KEYS;

const HYBRID_ENGINE_OPTION_KEY_SET = new Set(
  Object.keys(HYBRID_ENGINE_OPTION_KEYS),
);
const HYBRID_ENGINE_ADVANCED_OPTION_KEY_SET = new Set(
  Object.keys(HYBRID_ENGINE_ADVANCED_OPTION_KEYS),
);

function assertPlainDataObject(
  value: unknown,
  path: string,
): asserts value is Readonly<Record<string, unknown>> {
  if (
    value == null ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    ArrayBuffer.isView(value)
  ) {
    throw new TypeError(`[HybridEngine] ${path} must be a plain object.`);
  }
  const prototype = Object.getPrototypeOf(value) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(
      `[HybridEngine] ${path} must have Object.prototype or null prototype.`,
    );
  }
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string') {
      throw new TypeError(
        `[HybridEngine] ${path}: symbol keys are not supported (${String(key)}).`,
      );
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor) || !descriptor.enumerable) {
      throw new TypeError(
        `[HybridEngine] ${path}.${key} must be an enumerable own data property.`,
      );
    }
  }
}

function assertPlainObjectWithKnownKeys(
  value: unknown,
  path: string,
  allowedKeys: ReadonlySet<string>,
): asserts value is Readonly<Record<string, unknown>> {
  assertPlainDataObject(value, path);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string') continue;
    if (!allowedKeys.has(key)) {
      throw new TypeError(`[HybridEngine] ${path}: unknown key "${key}".`);
    }
  }
}

const TUNING_KEYS = new Set(TUNABLE_DEFINITIONS.map((definition) => definition.key));
const CAUSTIC_KEYS = new Set(['boost', 'visClamp']);
const CAUSTIC_OPTION_KEYS = new Set([
  'mneeMaxIterations',
  'mneeMaxChainLength',
  'mneeMultiplicityTrials',
]);
const STAINED_GLASS_KEYS = new Set(['sunCaustic', 'skyAperture']);
const GTAO_KEYS = new Set([
  'radiusPx',
  'intensity',
  'depthThresholdWorldUnits',
  'bilateralDepthSigma',
]);
const REGIR_KEYS = new Set([
  'enabled',
  'cellsPerAxis',
  'candidatesPerCell',
  'survivorsPerCell',
]);
const CASCADE_DIM_KEYS = new Set(['probes', 'rays', 'intervalNear', 'intervalFar']);
const NRC_CONFIG_KEY_RECORD = {
  levels: true,
  featuresPerEntry: true,
  tableSize: true,
  nMin: true,
  growth: true,
  oneBlobBins: true,
  width: true,
  hidden: true,
  spreadC: true,
  recordCap: true,
  learningRate: true,
  tableLearningRate: true,
  useF16: true,
  tileB: true,
  maxNrcResidentBytes: true,
  warmupSteps: true,
} as const satisfies Readonly<Record<keyof NrcConfig, true>>;
const NRC_CONFIG_KEYS = new Set(Object.keys(NRC_CONFIG_KEY_RECORD));
const DDGI_LIGHT_KEYS = new Set([
  'kind',
  'id',
  'intensity',
  'on',
  'castShadow',
  'position',
  'direction',
  'angularRadius',
  'color',
  'spotAxis',
  'spotCosInner',
  'spotCosOuter',
  'distance',
  'decay',
]);
const XYZ_KEYS = new Set(['x', 'y', 'z']);
const RGB_KEYS = new Set(['r', 'g', 'b']);
const WALKAROUND_HYBRID_EXTENSION_KEYS = new Set([
  'oidnModelUrl',
  'oidnExecutionProviders',
  'neuralTensorStorage',
  'bvhMode',
  'resolveEnvironmentMap',
]);

function assertKnownHybridEngineOptionKeys(opts: HybridEngineOptions): void {
  assertPlainObjectWithKnownKeys(
    opts,
    'options',
    HYBRID_ENGINE_OPTION_KEY_SET,
  );
  if (opts.tuning !== undefined) {
    assertPlainObjectWithKnownKeys(opts.tuning, 'options.tuning', TUNING_KEYS);
  }
  if (opts.caustic !== undefined) {
    assertPlainObjectWithKnownKeys(opts.caustic, 'options.caustic', CAUSTIC_KEYS);
  }
  if (opts.stainedGlass !== undefined) {
    assertPlainObjectWithKnownKeys(
      opts.stainedGlass,
      'options.stainedGlass',
      STAINED_GLASS_KEYS,
    );
  }
  if (opts.gtao !== undefined) {
    assertPlainObjectWithKnownKeys(opts.gtao, 'options.gtao', GTAO_KEYS);
  }
  if (opts.regir !== undefined) {
    assertPlainObjectWithKnownKeys(opts.regir, 'options.regir', REGIR_KEYS);
  }
  if (opts.nrcConfig !== undefined) {
    assertPlainObjectWithKnownKeys(
      opts.nrcConfig,
      'options.nrcConfig',
      NRC_CONFIG_KEYS,
    );
  }
  if (opts.cascadeDims !== undefined) {
    if (!Array.isArray(opts.cascadeDims)) {
      throw new TypeError('[HybridEngine] options.cascadeDims must be an array.');
    }
    opts.cascadeDims.forEach((dim, index) => {
      assertPlainObjectWithKnownKeys(
        dim,
        `options.cascadeDims[${index}]`,
        CASCADE_DIM_KEYS,
      );
    });
  }
  if (opts.lights !== undefined) {
    if (!Array.isArray(opts.lights)) {
      throw new TypeError('[HybridEngine] options.lights must be an array.');
    }
    opts.lights.forEach((light, index) => {
      assertPlainObjectWithKnownKeys(light, `options.lights[${index}]`, DDGI_LIGHT_KEYS);
      if (light.position !== undefined) {
        assertPlainObjectWithKnownKeys(
          light.position,
          `options.lights[${index}].position`,
          XYZ_KEYS,
        );
      }
      if (light.direction !== undefined) {
        assertPlainObjectWithKnownKeys(
          light.direction,
          `options.lights[${index}].direction`,
          XYZ_KEYS,
        );
      }
      if (light.spotAxis !== undefined) {
        assertPlainObjectWithKnownKeys(
          light.spotAxis,
          `options.lights[${index}].spotAxis`,
          XYZ_KEYS,
        );
      }
      if (light.color !== undefined) {
        assertPlainObjectWithKnownKeys(
          light.color,
          `options.lights[${index}].color`,
          RGB_KEYS,
        );
      }
    });
  }
  // Foreign extension namespaces remain open by contract. The namespace this
  // backend owns is closed and data-only, so misspellings cannot survive until
  // asynchronous initialization.
  if (opts.extensions !== undefined) {
    assertPlainDataObject(opts.extensions, 'options.extensions');
    const owned = opts.extensions['walkaround-hybrid'];
    if (owned !== undefined) {
      assertPlainObjectWithKnownKeys(
        owned,
        "options.extensions['walkaround-hybrid']",
        WALKAROUND_HYBRID_EXTENSION_KEYS,
      );
    }
  }
}

function finiteOr(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function finitePositiveIntOr(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value)
    ? Math.max(1, Math.floor(value))
    : fallback;
}

function optionalFinitePositiveInt(value: number | undefined): number | undefined {
  if (value === undefined || !Number.isFinite(value)) return undefined;
  return Math.max(1, Math.floor(value));
}

function finiteTupleOr(
  value: readonly [number, number, number] | undefined,
  fallback: readonly [number, number, number],
): readonly [number, number, number] {
  return value != null && value.every(Number.isFinite) ? value : fallback;
}

function assertBoolean(value: unknown, path: string): void {
  if (value !== undefined && typeof value !== 'boolean') {
    throw new TypeError(`[HybridEngine] ${path} must be boolean.`);
  }
}

function assertFunction(value: unknown, path: string): void {
  if (value !== undefined && typeof value !== 'function') {
    throw new TypeError(`[HybridEngine] ${path} must be a function.`);
  }
}

function assertFiniteNumber(
  value: unknown,
  path: string,
  bounds: { readonly min?: number; readonly max?: number; readonly minExclusive?: boolean } = {},
): void {
  if (value === undefined) return;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`[HybridEngine] ${path} must be a finite number.`);
  }
  if (
    bounds.min !== undefined &&
    (bounds.minExclusive ? value <= bounds.min : value < bounds.min)
  ) {
    throw new RangeError(
      `[HybridEngine] ${path} must be ${bounds.minExclusive ? '>' : '>='} ${bounds.min}.`,
    );
  }
  if (bounds.max !== undefined && value > bounds.max) {
    throw new RangeError(`[HybridEngine] ${path} must be <= ${bounds.max}.`);
  }
}

function assertSafeInteger(
  value: unknown,
  path: string,
  minimum: number,
  maximum = Number.MAX_SAFE_INTEGER,
): void {
  if (value === undefined) return;
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new RangeError(
      `[HybridEngine] ${path} must be a safe integer in [${minimum}, ${maximum}].`,
    );
  }
}

function assertEnum(
  value: unknown,
  path: string,
  values: readonly string[],
): void {
  if (value !== undefined && (typeof value !== 'string' || !values.includes(value))) {
    throw new TypeError(
      `[HybridEngine] ${path} must be one of ${values.map((entry) => JSON.stringify(entry)).join(', ')}.`,
    );
  }
}

function assertDenseDataArray(
  value: unknown,
  path: string,
): asserts value is readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`[HybridEngine] ${path} must be an array.`);
  }
  for (const key of Reflect.ownKeys(value)) {
    if (key === 'length') continue;
    if (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key)) {
      throw new TypeError(
        `[HybridEngine] ${path} contains unsupported own key ${String(key)}.`,
      );
    }
    const index = Number(key);
    if (!Number.isSafeInteger(index) || index >= value.length) {
      throw new TypeError(`[HybridEngine] ${path} contains an invalid array index ${key}.`);
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor) || !descriptor.enumerable) {
      throw new TypeError(
        `[HybridEngine] ${path}[${key}] must be an enumerable own data property.`,
      );
    }
  }
  for (let index = 0; index < value.length; index += 1) {
    if (!Object.prototype.hasOwnProperty.call(value, index)) {
      throw new TypeError(`[HybridEngine] ${path} must not contain holes.`);
    }
  }
}

function assertFiniteTuple(
  value: unknown,
  path: string,
  options: { readonly nonNegative?: boolean; readonly positive?: boolean } = {},
): void {
  if (value === undefined) return;
  assertDenseDataArray(value, path);
  if (value.length !== 3) {
    throw new TypeError(`[HybridEngine] ${path} must be an exact three-number tuple.`);
  }
  value.forEach((component, index) => {
    assertFiniteNumber(component, `${path}[${index}]`, {
      ...(options.positive ? { min: 0, minExclusive: true } : {}),
      ...(options.nonNegative ? { min: 0 } : {}),
    });
  });
}

function validateHybridOptionValueDomains(opts: HybridEngineOptions): void {
  for (const key of [
    'gpuSkinning',
    'verbose',
    'debug',
    'grisReuse',
    'restirPtReuse',
    'checkerboardRendering',
    'ppgEnabled',
    'rcEnabled',
    'nrcEnabled',
  ] as const) {
    assertBoolean(opts[key], `options.${key}`);
  }
  assertFunction(opts.isSceneReady, 'options.isSceneReady');
  assertFunction(opts.getPipelineRebuildKey, 'options.getPipelineRebuildKey');
  assertFunction(opts.onWarning, 'options.onWarning');

  if (
    opts.pipelineRebuildKey !== undefined &&
    opts.pipelineRebuildKey !== null &&
    typeof opts.pipelineRebuildKey !== 'string' &&
    (typeof opts.pipelineRebuildKey !== 'number' || !Number.isFinite(opts.pipelineRebuildKey))
  ) {
    throw new TypeError(
      '[HybridEngine] options.pipelineRebuildKey must be a string, finite number, or null.',
    );
  }

  assertFiniteTuple(opts.primaryLightDir, 'options.primaryLightDir');
  if (
    opts.primaryLightDir !== undefined &&
    opts.primaryLightDir[0] === 0 &&
    opts.primaryLightDir[1] === 0 &&
    opts.primaryLightDir[2] === 0
  ) {
    throw new RangeError('[HybridEngine] options.primaryLightDir must be non-zero.');
  }
  if (opts.primaryLightDir !== undefined) {
    canonicalizeLightingDirectionF32(
      opts.primaryLightDir,
      'options.primaryLightDir',
    );
  }
  assertFiniteTuple(opts.skyTint, 'options.skyTint', { nonNegative: true });
  if (opts.skyTint !== undefined) {
    packNonNegativeLightingRgbF32(opts.skyTint, 'options.skyTint');
  }
  assertFiniteNumber(opts.primaryLightIntensity, 'options.primaryLightIntensity', { min: 0 });
  if (opts.primaryLightIntensity !== undefined) {
    packNonNegativeLightingFloat32(
      opts.primaryLightIntensity,
      'options.primaryLightIntensity',
    );
  }
  assertFiniteNumber(opts.skyIrradiance, 'options.skyIrradiance', { min: 0 });
  if (opts.skyIrradiance !== undefined) {
    packNonNegativeLightingFloat32(
      opts.skyIrradiance,
      'options.skyIrradiance',
    );
  }
  if (opts.lights !== undefined) {
    assertDenseDataArray(opts.lights, 'options.lights');
    assertValidDdgiLights(opts.lights, 'options.lights');
  }

  assertEnum(opts.qualityTier, 'options.qualityTier', ['ultra', 'high', 'medium', 'low']);
  assertEnum(opts.tier, 'options.tier', ['full', 'lite']);
  assertEnum(opts.gtaoMode, 'options.gtaoMode', ['on', 'quarter', 'off']);
  assertEnum(
    opts.frameResourceResolutionPolicy,
    'options.frameResourceResolutionPolicy',
    ['auto', 'native'],
  );
  assertSafeInteger(
    opts.maxPersistentFrameResourceBytes,
    'options.maxPersistentFrameResourceBytes',
    1,
  );
  assertSafeInteger(
    opts.restirReservoirScale,
    'options.restirReservoirScale',
    1,
    DEFAULT_MAX_RESTIR_RESERVOIR_SCALE,
  );
  if (
    opts.restirReservoirScale !== undefined
    && opts.restirReservoirScale !== 1
    && (opts.ppgEnabled === true || opts.nrcEnabled === true)
  ) {
    throw new RangeError(
      '[HybridEngine] restirReservoirScale > 1 is incompatible with PPG/NRC training layouts; ' +
      'use scale 1 or disable ppgEnabled/nrcEnabled.',
    );
  }
  if (opts.diSpatialPasses !== undefined && opts.diSpatialPasses !== 1 && opts.diSpatialPasses !== 2) {
    throw new RangeError('[HybridEngine] options.diSpatialPasses must be 1 or 2.');
  }
  if (opts.giSpatialPasses !== undefined && opts.giSpatialPasses !== 1 && opts.giSpatialPasses !== 2) {
    throw new RangeError('[HybridEngine] options.giSpatialPasses must be 1 or 2.');
  }
  assertSafeInteger(opts.ddgiUpdateDivisor, 'options.ddgiUpdateDivisor', 1);
  assertSafeInteger(opts.ppgDispatchInterval, 'options.ppgDispatchInterval', 1);
  assertSafeInteger(opts.nrcWarmupSteps, 'options.nrcWarmupSteps', 0);
  if (opts.maxSamplesPerPixel !== undefined) {
    throw new TypeError(
      '[HybridEngine] maxSamplesPerPixel is unsupported by walkaround-hybrid: ' +
      'this realtime backend does not accumulate samples (capabilities.accumulates=false). ' +
      'Use per-frame quality controls or a path-tracing backend for SPP caps.',
    );
  }
  if (opts.targetFrameIntervalMs !== null) {
    assertFiniteNumber(opts.targetFrameIntervalMs, 'options.targetFrameIntervalMs', { min: 0 });
  }
  assertFiniteNumber(opts.cameraMoveResetThresholdSq, 'options.cameraMoveResetThresholdSq', { min: 0 });
  assertFiniteNumber(opts.temporalAccumAlpha, 'options.temporalAccumAlpha', { min: 0, max: 1 });
  assertFiniteNumber(opts.checkerboardMotionThresholdSq, 'options.checkerboardMotionThresholdSq', { min: 0 });
  assertFiniteNumber(opts.nrcSpreadC, 'options.nrcSpreadC', { min: 0 });
  assertFiniteNumber(opts.rcWeight, 'options.rcWeight', { min: 0, max: 1 });
  // Validates every nested NRC field and the three compatibility aliases at
  // the synchronous public boundary, before adapter/device acquisition.
  resolveHybridNrcConfig(opts);

  if (opts.tuning !== undefined) {
    for (const definition of TUNABLE_DEFINITIONS) {
      assertFiniteNumber(opts.tuning[definition.key], `options.tuning.${definition.key}`);
    }
  }
  assertFiniteNumber(opts.caustic?.boost, 'options.caustic.boost', { min: 0 });
  assertFiniteNumber(opts.caustic?.visClamp, 'options.caustic.visClamp', { min: 0, max: 1 });
  assertBoolean(opts.stainedGlass?.sunCaustic, 'options.stainedGlass.sunCaustic');
  assertBoolean(opts.stainedGlass?.skyAperture, 'options.stainedGlass.skyAperture');
  assertFiniteNumber(opts.gtao?.radiusPx, 'options.gtao.radiusPx', { min: 0, minExclusive: true });
  assertFiniteNumber(opts.gtao?.intensity, 'options.gtao.intensity', { min: 0 });
  assertFiniteNumber(opts.gtao?.depthThresholdWorldUnits, 'options.gtao.depthThresholdWorldUnits', { min: 0 });
  assertFiniteNumber(opts.gtao?.bilateralDepthSigma, 'options.gtao.bilateralDepthSigma', { min: 0, minExclusive: true });

  if (opts.adaptiveSamplingThresholds !== undefined) {
    assertDenseDataArray(opts.adaptiveSamplingThresholds, 'options.adaptiveSamplingThresholds');
    if (opts.adaptiveSamplingThresholds.length !== 2) {
      throw new TypeError('[HybridEngine] options.adaptiveSamplingThresholds must be an exact [low, high] tuple.');
    }
    const [low, high] = opts.adaptiveSamplingThresholds;
    assertFiniteNumber(low, 'options.adaptiveSamplingThresholds[0]', { min: 0 });
    assertFiniteNumber(high, 'options.adaptiveSamplingThresholds[1]', { min: 0 });
    if (low > high) {
      throw new RangeError('[HybridEngine] options.adaptiveSamplingThresholds low must be <= high.');
    }
  }
  assertFiniteTuple(opts.indirectFireflyClamp, 'options.indirectFireflyClamp', { nonNegative: true });
  assertFiniteTuple(opts.atrousDirectSigmas, 'options.atrousDirectSigmas', { positive: true });
  assertFiniteTuple(opts.atrousIndirectSigmas, 'options.atrousIndirectSigmas', { positive: true });

  if (opts.regir !== undefined) {
    assertBoolean(opts.regir.enabled, 'options.regir.enabled');
    assertSafeInteger(
      opts.regir.cellsPerAxis,
      'options.regir.cellsPerAxis',
      1,
      0xffff_ffff,
    );
    assertSafeInteger(
      opts.regir.candidatesPerCell,
      'options.regir.candidatesPerCell',
      1,
      REGIR_MAX_CANDIDATES_PER_CELL,
    );
    assertSafeInteger(
      opts.regir.survivorsPerCell,
      'options.regir.survivorsPerCell',
      1,
      0xffff_ffff,
    );
    // Cross-field shader-domain validation (cells³ × survivors and storage
    // addressing) belongs at the synchronous public constructor boundary.
    resolveReGIRConfig(opts.regir);
  }
  if (opts.cascadeDims !== undefined) {
    assertDenseDataArray(opts.cascadeDims, 'options.cascadeDims');
    validateCascadeDims(opts.cascadeDims, 'options.cascadeDims');
  }

  const extension = readWalkaroundHybridExt(opts);
  if (extension?.oidnModelUrl !== undefined && !hasOidnModelUrl(extension.oidnModelUrl)) {
    throw new TypeError(
      "[HybridEngine] options.extensions['walkaround-hybrid'].oidnModelUrl must be a non-empty string.",
    );
  }
  if (extension?.oidnExecutionProviders !== undefined) {
    assertDenseDataArray(
      extension.oidnExecutionProviders,
      "options.extensions['walkaround-hybrid'].oidnExecutionProviders",
    );
    if (extension.oidnExecutionProviders.length === 0) {
      throw new TypeError('[HybridEngine] oidnExecutionProviders must not be empty.');
    }
    const seen = new Set<string>();
    for (const provider of extension.oidnExecutionProviders) {
      if (provider !== 'webnn' && provider !== 'webgpu' && provider !== 'wasm') {
        throw new TypeError(`[HybridEngine] unsupported OIDN execution provider ${String(provider)}.`);
      }
      if (seen.has(provider)) {
        throw new TypeError(`[HybridEngine] duplicate OIDN execution provider ${provider}.`);
      }
      seen.add(provider);
    }
  }
  assertEnum(extension?.neuralTensorStorage, 'neuralTensorStorage', ['auto', 'f32', 'f16']);
  assertEnum(extension?.bvhMode, 'bvhMode', ['merged', 'tlas']);
  assertFunction(extension?.resolveEnvironmentMap, 'resolveEnvironmentMap');
}

type HybridDenoiser = (typeof VALID_DENOISERS)[number];
export type ResolvedHybridDenoiser = Exclude<HybridDenoiser, 'auto'>;
export type DenoiserAutoResolutionReason =
  | 'host-neural-weights'
  | 'host-neural-weights-not-production-ready'
  | 'host-neural-weights-device-infeasible'
  | 'host-oidn-model-url'
  | 'lite-neural-unavailable'
  | 'no-host-model-assets';

export interface DenoiserAutoResolution {
  readonly requested: 'auto';
  readonly resolved: ResolvedHybridDenoiser;
  readonly reason: DenoiserAutoResolutionReason;
  readonly packageProvidesProductionWeights: false;
  readonly defaultEnabled: false;
  readonly neuralCheckpointProductionReady: boolean;
  readonly neuralCheckpointMissing: readonly string[];
  readonly neuralDeviceFailure?: string;
  readonly neuralTensorPrecision?: NeuralTensorPrecision;
}

function emitConfigWarning(opts: HybridEngineOptions, warning: EngineWarning): void {
  console.warn(warning.message);
  if (opts.onWarning == null) return;
  try {
    opts.onWarning(warning);
  } catch {
    // Host warning callbacks must not break construction-time option parsing.
  }
}

function warnLiteBvhModeOverride(opts: HybridEngineOptions): void {
  emitConfigWarning(opts, {
    code: 'walkaround-hybrid.lite-bvh-mode-overridden',
    backend: 'walkaround-hybrid',
    phase: 'construction',
    method: 'createWalkaroundEngine_Hybrid',
    message:
      `[HybridEngine] tier:'lite' overrides bvhMode:'tlas' → 'merged' ` +
      `(TLAS scene buffers exceed the lite resource budget). Instanced/` +
      `multi-mesh scene fidelity is reduced. Use tier:'full' for TLAS.`,
    details: {
      tier: 'lite',
      requestedBvhMode: 'tlas',
      effectiveBvhMode: 'merged',
      fallback: 'merged-bvh',
    },
  });
}

/**
 * The construction-time-immutable config the engine derives from its options —
 * no `this` dependency, no GPU side effects. Extracting the ~80 LOC
 * of defaulting + validation that produced these out of the constructor (WD
 * decomposition sweep) keeps the constructor focused on object wiring (DDGI /
 * RC subsystem creation, capabilities, init coordinator, debug surface) that
 * genuinely needs `this`.
 *
 * Behaviour-preserving: `parseHybridEngineOptions` throws the same three
 * `TypeError`s in the same order as the inline constructor did, and applies
 * the same defaults. The constructor assigns each field verbatim from the
 * returned record.
 */
export interface ParsedHybridEngineConfig {
  readonly frameResourceResolutionPolicy: 'auto' | 'native';
  readonly maxPersistentFrameResourceBytes: number;
  readonly restirReservoirScale: 1 | 2 | 3 | 4 | undefined;
  readonly denoiser: ResolvedHybridDenoiser;
  readonly denoiserAutoResolution: DenoiserAutoResolution | undefined;
  readonly neuralWeights: ModelWeights | undefined;
  readonly neuralCheckpointAssessment: NeuralCheckpointProductionAssessment;
  readonly neuralTensorStorage: NeuralTensorStoragePreference;
  readonly oidnModelUrl: string | undefined;
  readonly oidnExecutionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  readonly restirBvhModeOverride: ReSTIRBvhMode | undefined;
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
  readonly mneeMultiplicityTrials: number;
  readonly targetFrameIntervalMs: number | null;
  readonly tunables: Tunables;
  readonly initTunables: InitTunables;
  readonly indirectFireflyClamp: readonly [number, number, number];
  readonly atrousDirectSigmas: readonly [number, number, number];
  readonly atrousIndirectSigmas: readonly [number, number, number];
  readonly stainedGlassFlags: number;
  /** Maximum transmitted dielectric interfaces per RC probe ray. */
  readonly rcTransmittedInterfaceBudget: number;
  /** NRC (Müller et al. 2021) cache flag mirrored into the per-frame UBO
   *  (0 = off / verbatim DDGI suffix, 1 = on). The load-bearing gate is
   *  COMPILE-TIME: `nrcEnabled` selects the `risGiNrc` GI shader variant at
   *  engine creation (a UBO flag alone can't add the @group(4) NRC bindings).
   *  When ON, the suffix cache-query + per-frame training passes are live.
   *  FORBIDDEN on tier:'lite'. */
  readonly nrcEnabled: number;
  /** Fully resolved NRC query/trainer/allocation contract. */
  readonly nrcConfig: NrcConfig;
  /** PPG (Müller 2017) guided-sampling flag (0 = off, 1 = on). COMPILE-TIME
   *  at the pipeline level: `ppgEnabled` builds the ppg-update pipeline and
   *  drives the UBO gate; OFF is bit-identical to the cosine kernel.
   *  FORBIDDEN on tier:'lite'. (G-P1.1 follow-up: opts.ppgEnabled used to be
   *  read only by the lite-tier guard and never forwarded to the pipeline —
   *  PPG was inert through the public API.) */
  readonly ppgEnabled: number;
  /** H47 — maximum PPG sTree spatial cells, threaded to `allocatePPGResources`.
   *  `undefined` ⇒ use allocatePPGResources default (1 024). */
  readonly ppgMaxSpatialCells: number | undefined;
  /** H29 — maximum per-cell PPG dTree nodes, threaded to shader compile and
   *  `allocatePPGResources`. `undefined` ⇒ use the default 341-node stride. */
  readonly ppgMaxDTreeNodesPerCell: number | undefined;
  /** PPG guide/cosine MIS mixture alpha; strictly 0 < alpha < 1. */
  readonly ppgMixAlpha: number;
  /** Checkerboard half-res shading (HybridEngineOptions.checkerboardRendering).
   *  `false` by default (no preset ⇒ ultra ⇒ off); the `medium`/`low` presets
   *  enable it, `ultra`/`high` keep it off. Threaded into
   *  `pipeline.initialize({ checkerboard, checkerboardMotionThresholdSq })`; OFF
   *  is bit-identical to the pre-checkerboard pipeline (shade + both spatial
   *  passes + ris dispatch full-res and ResolvePass passes through).
   *  GPU-validated on dzn — see WalkaroundGPUPipeline `_checkerboard`. */
  readonly checkerboard: boolean;
  readonly staticPipelineRebuildKey: string | number | null;
  readonly getPipelineRebuildKey: (() => string | number | null | undefined) | undefined;
  readonly rebuildKeyFingerprintSeen: string;
  readonly maxBounces: number;
  readonly verbose: boolean;
  readonly debug: boolean;
  // ── Phase-0 productization — quality-preset-resolved knobs ───────────────
  /** Resolved GTAO dispatch mode (preset, overridden by `opts.gtaoMode`). */
  readonly gtaoMode: 'on' | 'quarter' | 'off';
  /** Resolved ReSTIR-DI spatial pass count (preset, overridden by opts). */
  readonly diSpatialPasses: 1 | 2;
  /** Resolved ReSTIR-GI spatial pass count (preset, overridden by opts). */
  readonly giSpatialPasses: 1 | 2;
  /** Resolved DDGI round-robin probe-update divisor (preset, overridden by opts). */
  readonly ddgiUpdateDivisor: number;
  /** Resolved PPG train-pass dispatch cadence (preset, overridden by opts).
   *  Threaded into `pipeline.initialize` so the ppg-update pass gates on
   *  `frameCount % ppgDispatchInterval`. Always >= 1. */
  readonly ppgDispatchInterval: number;
  /** ReGIR (Boksansky 2021) grid-based DI light-selection config (pass-through
   *  from opts; `undefined` ⇒ off). Threaded into `pipeline.initialize`. */
  readonly regirConfig: Partial<import('./pipeline/ReGIRCoordinator.js').ReGIRConfig> | undefined;
  /** Resolved initial internal-resolution factor (preset; per-frame
   *  `quality.resolutionFactor` still overrides at runtime). */
  readonly resolutionFactor: number;
}

/**
 * The extracted `extensions['walkaround-hybrid']` sub-object shape — read by
 * both {@link validateHybridEngineOptions} (oidnModelUrl presence) and
 * {@link deriveHybridEngineConfig} (oidnModelUrl / providers / bvhMode).
 */
type WalkaroundHybridExt = {
  oidnModelUrl?: string;
  oidnExecutionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  neuralTensorStorage?: NeuralTensorStoragePreference;
  bvhMode?: 'merged' | 'tlas';
  resolveEnvironmentMap?: unknown;
};

function resolvePpgMixAlpha(value: number | undefined): number {
  try {
    return quantizeOpenUnitProbabilityF32(
      value ?? PPG_MIS_ALPHA,
      '[HybridEngine] ppgMixAlpha',
    );
  } catch (error) {
    throw new TypeError(
      '[HybridEngine] ppgMixAlpha must be finite and strictly between 0 and 1.',
      { cause: error },
    );
  }
}

/**
 * Resolve the complete NRC contract plus the three legacy top-level aliases.
 * Aliases never silently override the nested contract: duplicate values must
 * agree, otherwise construction fails before any GPU work.
 */
export function resolveHybridNrcConfig(
  opts: Pick<
    HybridEngineOptions,
    'nrcConfig' | 'nrcWarmupSteps' | 'nrcSpreadC' | 'nrcMaxResidentBytes'
  >,
): NrcConfig {
  const nested = opts.nrcConfig ?? {};
  const assertAliasAgreement = (
    aliasName: string,
    nestedName: keyof NrcConfig,
    aliasValue: number | undefined,
  ): void => {
    const nestedValue = nested[nestedName];
    if (
      aliasValue !== undefined &&
      nestedValue !== undefined &&
      aliasValue !== nestedValue
    ) {
      throw new TypeError(
        `[HybridEngine] ${aliasName} (${aliasValue}) disagrees with ` +
        `nrcConfig.${String(nestedName)} (${String(nestedValue)}); supply only ` +
        `nrcConfig.${String(nestedName)}.`,
      );
    }
  };
  assertAliasAgreement('nrcWarmupSteps', 'warmupSteps', opts.nrcWarmupSteps);
  assertAliasAgreement('nrcSpreadC', 'spreadC', opts.nrcSpreadC);
  assertAliasAgreement(
    'nrcMaxResidentBytes',
    'maxNrcResidentBytes',
    opts.nrcMaxResidentBytes,
  );
  return resolveNrcConfig({
    ...nested,
    ...(opts.nrcWarmupSteps !== undefined
      ? { warmupSteps: opts.nrcWarmupSteps }
      : {}),
    ...(opts.nrcSpreadC !== undefined ? { spreadC: opts.nrcSpreadC } : {}),
    ...(opts.nrcMaxResidentBytes !== undefined
      ? { maxNrcResidentBytes: opts.nrcMaxResidentBytes }
      : {}),
  });
}

function resolveMaxBounces(value: number | undefined): number {
  if (value === undefined) return 2;
  return value;
}

function readWalkaroundHybridExt(opts: HybridEngineOptions): WalkaroundHybridExt | undefined {
  return (opts.extensions as undefined | {
    'walkaround-hybrid'?: WalkaroundHybridExt;
  })?.['walkaround-hybrid'];
}

function hasOidnModelUrl(oidnModelUrl: string | undefined): oidnModelUrl is string {
  return typeof oidnModelUrl === 'string' && oidnModelUrl.length > 0;
}

function validateSuppliedNeuralWeights(opts: HybridEngineOptions): void {
  if (opts.tier === 'lite' || opts.neuralWeights == null) return;
  try {
    validateWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC, opts.neuralWeights);
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    throw new TypeError(
      `[HybridEngine] neuralWeights must match the canonical walkaround U-Net ` +
      `checkpoint contract before they can enable denoiser:'neural' or advertise ` +
      `neural support: ${reason}`,
    );
  }
}

function resolvePresetDenoiser(
  preset: ReturnType<typeof resolveQualityPreset>,
): ResolvedHybridDenoiser {
  return preset.denoiser ?? 'atrous-variance';
}

interface NeuralDeviceFeasibility {
  readonly precision?: NeuralTensorPrecision;
  readonly failure?: string;
}

function assessNeuralDeviceFeasibility(
  opts: HybridEngineOptions,
  preset: ReturnType<typeof resolveQualityPreset>,
  preference: NeuralTensorStoragePreference,
): NeuralDeviceFeasibility {
  if (opts.neuralWeights == null) return { failure: 'neural weights are missing' };
  const width = Math.max(1, Math.round(opts.width * preset.resolutionFactor));
  const height = Math.max(1, Math.round(opts.height * preset.resolutionFactor));
  // Resolve the host's exact logical size to the private padded U-Net lattice
  // before adapter feasibility is assessed. Odd host sizes are not a fallback.
  const extent = walkaroundNeuralInferenceExtent(width, height);
  try {
    const decision = resolveNeuralTensorStorageDecision(
      opts.device,
      opts.neuralWeights,
      preference,
    );
    const dims = preflightTensorDims(
      WALKAROUND_DENOISER_UNET_SPEC,
      extent.inferenceWidth,
      extent.inferenceHeight,
    );
    const failures = neuralDeviceLimitFailures(
      opts.device,
      WALKAROUND_DENOISER_UNET_SPEC,
      opts.neuralWeights,
      dims,
      decision.storage,
    );
    return failures.length === 0
      ? { precision: decision.storage.precision }
      : { precision: decision.storage.precision, failure: failures.join('; ') };
  } catch (error) {
    return { failure: error instanceof Error ? error.message : String(error) };
  }
}

function resolveHybridDenoiser(
  opts: HybridEngineOptions,
  preset: ReturnType<typeof resolveQualityPreset>,
  oidnModelUrl: string | undefined,
  neuralTensorStorage: NeuralTensorStoragePreference,
): { denoiser: ResolvedHybridDenoiser; autoResolution: DenoiserAutoResolution | undefined } {
  const fallback = resolvePresetDenoiser(preset);
  const neuralCheckpointAssessment = assessNeuralCheckpointProductionReadiness(opts.neuralWeights);
  if (opts.denoiser !== 'auto') {
    return {
      denoiser: opts.denoiser ?? fallback,
      autoResolution: undefined,
    };
  }

  let resolved: ResolvedHybridDenoiser = fallback;
  let reason: DenoiserAutoResolutionReason = 'no-host-model-assets';
  let neuralDevice: NeuralDeviceFeasibility | undefined;
  if (opts.tier !== 'lite' && opts.neuralWeights != null && neuralCheckpointAssessment.productionReady) {
    neuralDevice = assessNeuralDeviceFeasibility(opts, preset, neuralTensorStorage);
    if (neuralDevice.failure == null) {
      resolved = 'neural';
      reason = 'host-neural-weights';
    } else {
      reason = 'host-neural-weights-device-infeasible';
    }
  } else if (opts.tier !== 'lite' && opts.neuralWeights != null) {
    reason = 'host-neural-weights-not-production-ready';
  } else if (opts.tier === 'lite' && opts.neuralWeights != null) {
    reason = 'lite-neural-unavailable';
  }
  if (reason !== 'host-neural-weights' && hasOidnModelUrl(oidnModelUrl)) {
    resolved = 'oidn-final';
    reason = 'host-oidn-model-url';
  }

  return {
    denoiser: resolved,
    autoResolution: {
      requested: 'auto',
      resolved,
      reason,
      packageProvidesProductionWeights: false,
      defaultEnabled: false,
      neuralCheckpointProductionReady: neuralCheckpointAssessment.productionReady,
      neuralCheckpointMissing: neuralCheckpointAssessment.missing,
      ...(neuralDevice?.failure !== undefined
        ? { neuralDeviceFailure: neuralDevice.failure }
        : {}),
      ...(neuralDevice?.precision !== undefined ? { neuralTensorPrecision: neuralDevice.precision } : {}),
    },
  };
}

/**
 * Pure construction-time validation of `HybridEngineOptions` — throws the
 * three (well, six) `TypeError`s the constructor relies on, in the exact same
 * order as the pre-Theme-H inline path. No defaulting, no derived config, no
 * `this`, no GPU side effects: this is the independently-testable "does this
 * option object describe a buildable engine?" gate.
 *
 * Throw order (load-bearing — tests pin it):
 *   1. tier:'lite' forbids rcEnabled / ppgEnabled / denoiser:'neural' /
 *      nrcEnabled (lite validated FIRST so it is the host's first signal);
 *   2. unsupported denoiser enum;
 *   3. denoiser:'neural' without neuralWeights;
 *   4. denoiser:'oidn-final' without extensions['walkaround-hybrid'].oidnModelUrl.
 *   5. supplied full-tier neuralWeights must match the canonical U-Net spec.
 */

function validateGrisReuseOptions(opts: HybridEngineOptions): void {
  if (opts.grisReuse === false || opts.restirPtReuse === false) {
    throw new TypeError(
      '[HybridEngine] grisReuse=false/restirPtReuse=false requests the retired ' +
      'compact ReSTIR-GI reuse path. Generalized reconnection-shift reuse is ' +
      'always enabled; remove the option.',
    );
  }
}

function warnDeprecatedGrisReuseOptions(opts: HybridEngineOptions): void {
  if (opts.grisReuse === undefined && opts.restirPtReuse === undefined) return;
  emitConfigWarning(opts, {
    code: 'walkaround-hybrid.gi-reuse-option-deprecated',
    backend: 'walkaround-hybrid',
    phase: 'construction',
    method: 'createWalkaroundEngine_Hybrid',
    message:
      '[HybridEngine] grisReuse/restirPtReuse is deprecated because generalized ' +
      'reconnection-shift GI reuse is always enabled. Remove the option; this ' +
      'is a diffuse/geometric one-bounce proxy, not ReSTIR PT.',
    details: {
      replacement: 'omit-option',
      suppliedGrisReuse: opts.grisReuse,
      suppliedRestirPtReuse: opts.restirPtReuse,
      effectiveValue: true,
    },
  });
}

function rejectDisabledSubsystemOptions(
  opts: HybridEngineOptions,
  enabled: boolean,
  gate: 'ppgEnabled' | 'nrcEnabled' | 'rcEnabled',
  fields: readonly (keyof HybridEngineOptions)[],
): void {
  if (enabled) return;
  const supplied = fields.filter((field) => opts[field] !== undefined);
  if (supplied.length === 0) return;
  throw new TypeError(
    `[HybridEngine] ${supplied.join(', ')} require ${gate}:true; ` +
    `the subsystem is construction-time immutable, so these options would otherwise have no effect.`,
  );
}

/**
 * Validate the facade-owned walkaround advanced bag before adapter requests,
 * canvas acquisition, GPU capability reads, or allocation. This function is
 * intentionally synchronous and GPU-independent. Construction-only ownership
 * fields (`device`, `width`, `height`) are rejected instead of ignored.
 */
export function validateHybridEngineAdvancedOptions(
  value: unknown,
): asserts value is HybridEngineAdvancedOptions {
  assertPlainObjectWithKnownKeys(
    value,
    'advanced options',
    HYBRID_ENGINE_ADVANCED_OPTION_KEY_SET,
  );
  validateHybridEngineOptions(value as unknown as HybridEngineOptions);
}

export function validateHybridEngineOptions(opts: HybridEngineOptions): void {
  assertKnownHybridEngineOptionKeys(opts);
  validateHybridOptionValueDomains(opts);
  if (
    opts.ddgiMaxMaterials !== undefined &&
    (!Number.isSafeInteger(opts.ddgiMaxMaterials) || opts.ddgiMaxMaterials < 1)
  ) {
    throw new RangeError(
      '[HybridEngine] ddgiMaxMaterials must be a positive safe integer.',
    );
  }
  const causticStrategy: unknown = opts.causticStrategy;
  if (
    causticStrategy !== undefined &&
    causticStrategy !== 'none' &&
    causticStrategy !== 'refractive-trace' &&
    causticStrategy !== 'manifold-nee' &&
    causticStrategy !== 'photon-map'
  ) {
    throw new TypeError(
      `[HybridEngine] unsupported causticStrategy ${JSON.stringify(causticStrategy)}. ` +
      `walkaround-hybrid implements 'none', 'refractive-trace', and 'manifold-nee'.`,
    );
  }
  if (causticStrategy === 'photon-map') {
    throw new TypeError(
      `[HybridEngine] causticStrategy='${causticStrategy}' is a valid core strategy, but ` +
      `walkaround-hybrid does not support it. Select 'none', 'refractive-trace', or 'manifold-nee'; the engine will not ` +
      `silently substitute a different transport estimator.`,
    );
  }
  if (opts.causticOptions !== undefined) {
    assertPlainObjectWithKnownKeys(
      opts.causticOptions,
      'causticOptions',
      CAUSTIC_OPTION_KEYS,
    );
  }
  if (opts.causticOptions !== undefined &&
      causticStrategy !== 'manifold-nee' &&
      Object.keys(opts.causticOptions).length > 0) {
    throw new TypeError(
      '[HybridEngine] causticOptions require causticStrategy="manifold-nee".',
    );
  }
  const mneeMaxIterations = opts.causticOptions?.mneeMaxIterations;
  if (mneeMaxIterations !== undefined &&
      (!Number.isSafeInteger(mneeMaxIterations) ||
       mneeMaxIterations < 1 || mneeMaxIterations > 32)) {
    throw new RangeError(
      '[HybridEngine] causticOptions.mneeMaxIterations must be a safe integer in [1, 32].',
    );
  }
  const mneeMaxChainLength = opts.causticOptions?.mneeMaxChainLength;
  if (mneeMaxChainLength !== undefined &&
      (!Number.isSafeInteger(mneeMaxChainLength) ||
       mneeMaxChainLength < 1 || mneeMaxChainLength > 8)) {
    throw new RangeError(
      '[HybridEngine] causticOptions.mneeMaxChainLength must be a safe integer in [1, 8].',
    );
  }
  const mneeMultiplicityTrials = opts.causticOptions?.mneeMultiplicityTrials;
  if (mneeMultiplicityTrials !== undefined &&
      (!Number.isSafeInteger(mneeMultiplicityTrials) ||
       mneeMultiplicityTrials < 1 || mneeMultiplicityTrials > 32)) {
    throw new RangeError(
      '[HybridEngine] causticOptions.mneeMultiplicityTrials must be a safe integer in [1, 32].',
    );
  }
  if (
    opts.maxBounces !== undefined &&
    (!Number.isSafeInteger(opts.maxBounces) || opts.maxBounces < 1 || opts.maxBounces > 2)
  ) {
    throw new RangeError(
      `[HybridEngine] maxBounces must be 1 (direct-only DDGI) or 2 ` +
      `(multi-bounce DDGI equilibrium); got ${String(opts.maxBounces)}.`,
    );
  }
  if (
    opts.rcTransmittedInterfaceBudget !== undefined &&
    (!Number.isSafeInteger(opts.rcTransmittedInterfaceBudget) ||
      opts.rcTransmittedInterfaceBudget < RC_MIN_TRANSMITTED_INTERFACE_BUDGET ||
      opts.rcTransmittedInterfaceBudget > RC_MAX_TRANSMITTED_INTERFACE_BUDGET)
  ) {
    throw new RangeError(
      `[HybridEngine] rcTransmittedInterfaceBudget must be a safe integer in ` +
      `[${RC_MIN_TRANSMITTED_INTERFACE_BUDGET}, ${RC_MAX_TRANSMITTED_INTERFACE_BUDGET}].`,
    );
  }

  // Phase-0 productization — hybrid LITE tier (Deliverable 3). Lite runs the
  // same explicit layouts but with reduced work/memory: it forbids the
  // resource-heavy optional subsystems and forces the merged-BVH runtime path.
  // Validated FIRST so the throws are the host's first signal.
  if (opts.tier === 'lite') {
    if (opts.rcEnabled === true) {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids rcEnabled — Radiance Cascades ` +
        `allocate 5 extra cascade GPUBuffers + a separate BVH that the lite ` +
        `resource budget cannot fit. Use tier:'full' for RC.`,
      );
    }
    if (opts.ppgEnabled === true) {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids ppgEnabled — Practical Path ` +
        `Guiding allocates an sTree/dTree GPU buffer set the lite budget ` +
        `cannot fit. Use tier:'full' for PPG.`,
      );
    }
    if (opts.denoiser === 'neural') {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids denoiser:'neural' — the U-Net ` +
        `InferenceGraph + weight buffers exceed the lite budget. Use ` +
        `'atrous-variance' / 'atrous' on lite, or tier:'full' for neural.`,
      );
    }
    if (opts.denoiser === 'bmfr') {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids denoiser:'bmfr' — overlapping ` +
        `32×32 direct-QR fits allocate a per-block coefficient buffer and run ` +
        `a two-pass compute pipeline outside the lite resource budget. Use ` +
        `tier:'full' and select denoiser:'bmfr' explicitly.`,
      );
    }
    if (opts.nrcEnabled === true) {
      throw new TypeError(
        `[HybridEngine] tier:'lite' forbids nrcEnabled — Neural Radiance ` +
        `Caching allocates a multiresolution hash-grid feature-table set + the ` +
        `fused-MLP weight/Adam buffers the lite budget cannot fit. Use ` +
        `tier:'full' for NRC.`,
      );
    }
  }


  validateGrisReuseOptions(opts);
  if (
    opts.ppgMixAlpha !== undefined &&
    (!Number.isFinite(opts.ppgMixAlpha) || opts.ppgMixAlpha <= 0 || opts.ppgMixAlpha >= 1)
  ) {
    throw new TypeError(
      '[HybridEngine] ppgMixAlpha must be finite and strictly between 0 and 1 ' +
      'because both proposal components need positive support.',
    );
  }
  if (
    opts.ppgMaxSpatialCells !== undefined &&
    (!Number.isSafeInteger(opts.ppgMaxSpatialCells) ||
      opts.ppgMaxSpatialCells < 1 ||
      opts.ppgMaxSpatialCells > PPG_MAX_SPATIAL_CELLS)
  ) {
    throw new TypeError(
      `[HybridEngine] ppgMaxSpatialCells must be an integer in [1, ${PPG_MAX_SPATIAL_CELLS}].`,
    );
  }
  if (
    opts.ppgMaxDTreeNodesPerCell !== undefined &&
    (!Number.isSafeInteger(opts.ppgMaxDTreeNodesPerCell) ||
      opts.ppgMaxDTreeNodesPerCell < 1 ||
      opts.ppgMaxDTreeNodesPerCell > PPG_MAX_DTREE_NODES_PER_CELL)
  ) {
    throw new TypeError(
      `[HybridEngine] ppgMaxDTreeNodesPerCell must be an integer in [1, ${PPG_MAX_DTREE_NODES_PER_CELL}].`,
    );
  }

  // Audit B7: validate the denoiser option at construction so an unsupported
  // value does not silently coerce to atrous-variance and produce wrong output.
  // Supported values are enumerated in VALID_DENOISERS (single source of truth
  // in HybridEngineOptions.ts). `'none'` is the pass-through denoiser, and
  // `'bmfr'` is a real denoiser (Koskela 2019 — see denoisers/bmfr.ts).
  if (
    opts.denoiser !== undefined &&
    !(VALID_DENOISERS as ReadonlyArray<string>).includes(opts.denoiser)
  ) {
    throw new TypeError(
      `[HybridEngine] unsupported denoiser '${String(opts.denoiser)}'. ` +
      `walkaround-hybrid supports: 'none' | 'auto' | 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'neural' | 'oidn-final'.`,
    );
  }
  // T2.H2 — 'neural' requires neuralWeights to be provided.
  if (opts.denoiser === 'neural' && !opts.neuralWeights) {
    throw new TypeError(
      `[HybridEngine] denoiser: 'neural' requires neuralWeights to be provided. ` +
      `Load weights via loadWeightsFromArrayBuffer() from a .vitrum-model file, ` +
      `or train one with tools/neural-denoiser-training/train.py. ` +
      `See tools/neural-denoiser-training/README.md for instructions.`,
    );
  }
  // W11 — 'oidn-final' requires extensions['walkaround-hybrid'].oidnModelUrl.
  const extension = readWalkaroundHybridExt(opts);
  const oidnModelUrl = extension?.oidnModelUrl;
  const tensorStorage = extension?.neuralTensorStorage;
  if (
    tensorStorage !== undefined &&
    tensorStorage !== 'auto' &&
    tensorStorage !== 'f32' &&
    tensorStorage !== 'f16'
  ) {
    throw new TypeError("[HybridEngine] neuralTensorStorage must be 'auto', 'f32', or 'f16'.");
  }
  if (opts.denoiser === 'oidn-final' &&
      (typeof oidnModelUrl !== 'string' || oidnModelUrl.length === 0)) {
    throw new TypeError(
      `[HybridEngine] denoiser: 'oidn-final' requires ` +
      `extensions['walkaround-hybrid'].oidnModelUrl (non-empty string) ` +
      `pointing at the bundled OIDN ONNX model file ` +
      `(e.g. '/models/oidn_rt_hdr_alb_nrm.onnx'). ` +
      `See plan/premium-grade-refactor-20260517.md §W11 + ` +
      `packages/shared-denoisers/src/oidnBridge.ts for the model-URL convention.`,
    );
  }
  // Any full-tier `neuralWeights` value is semantically meaningful: denoiser:'auto'
  // selects neural from it, and capabilities report neural support from its
  // presence. Validate the checkpoint contract here so a non-null malformed object
  // cannot leak into construction diagnostics and fail only later in async init.
  validateSuppliedNeuralWeights(opts);
  if (opts.denoiser === 'neural') {
    const assessment = assessNeuralCheckpointProductionReadiness(opts.neuralWeights);
    if (!assessment.productionReady) {
      throw new TypeError(
        `[HybridEngine] denoiser:'neural' requires a v2 production checkpoint ` +
        `whose metadata matches the runtime preprocessing contract; missing: ` +
        assessment.missing.join(', '),
      );
    }
  }
  rejectDisabledSubsystemOptions(
    opts,
    opts.ppgEnabled === true,
    'ppgEnabled',
    [
      'ppgMaxSpatialCells',
      'ppgMaxDTreeNodesPerCell',
      'ppgMixAlpha',
      'ppgDispatchInterval',
    ],
  );
  rejectDisabledSubsystemOptions(
    opts,
    opts.nrcEnabled === true,
    'nrcEnabled',
    ['nrcConfig', 'nrcWarmupSteps', 'nrcSpreadC', 'nrcMaxResidentBytes'],
  );
  rejectDisabledSubsystemOptions(
    opts,
    opts.rcEnabled === true,
    'rcEnabled',
    ['rcTransmittedInterfaceBudget', 'rcWeight', 'cascadeDims'],
  );
}

/**
 * Defaulting of `HybridEngineOptions` into the immutable derived config, given
 * an already-resolved quality `preset`. ASSUMES the options have already
 * passed {@link validateHybridEngineOptions} (it does not re-throw the
 * validation `TypeError`s). Behaviour-preserving: every field is defaulted
 * exactly as the pre-Theme-H inline path produced it. Construction diagnostics
 * are emitted here because lite-tier bvh-mode coercion is resolved here.
 *
 * @param preset resolved {@link resolveQualityPreset} output for the engine's
 *   effective quality tier (the caller resolves the tier so the lite-biased
 *   `'medium'` default + explicit `qualityTier` override live in one place).
 */
export function deriveHybridEngineConfig(
  opts: HybridEngineOptions,
  preset: ReturnType<typeof resolveQualityPreset>,
): ParsedHybridEngineConfig {
  const isLite = opts.tier === 'lite';
  validateGrisReuseOptions(opts);
  const nrcConfig = resolveHybridNrcConfig(opts);
  warnDeprecatedGrisReuseOptions(opts);
  // Effective options overlay: the preset supplies fallbacks for the knobs it
  // governs, so the existing table-driven `readTunables` / denoiser /
  // targetFrameInterval logic picks them up unchanged. Explicit opts win.
  const effectiveOpts: HybridEngineOptions = {
    ...opts,
    ...(opts.adaptiveSamplingThresholds === undefined && preset.adaptiveSamplingThresholds !== undefined
      ? { adaptiveSamplingThresholds: preset.adaptiveSamplingThresholds }
      : {}),
  };

  const whExt = readWalkaroundHybridExt(opts);
  const oidnModelUrl = whExt?.oidnModelUrl;
  const neuralCheckpointAssessment = assessNeuralCheckpointProductionReadiness(opts.neuralWeights);
  const neuralTensorStorage = whExt?.neuralTensorStorage ?? 'auto';
  const denoiser = resolveHybridDenoiser(opts, preset, oidnModelUrl, neuralTensorStorage);

  return {
    frameResourceResolutionPolicy:
      opts.frameResourceResolutionPolicy ?? 'auto',
    maxPersistentFrameResourceBytes:
      opts.maxPersistentFrameResourceBytes ?? DEFAULT_FRAME_RESOURCE_BUDGET_BYTES,
    restirReservoirScale:
      opts.ppgEnabled === true || opts.nrcEnabled === true
        ? 1
        : opts.restirReservoirScale,
    // Preset supplies the denoiser fallback (low ⇒ 'atrous'); explicit
    // opts.denoiser wins, then the engine default 'atrous-variance'.
    denoiser: denoiser.denoiser,
    denoiserAutoResolution: denoiser.autoResolution,
    neuralWeights: opts.neuralWeights,
    neuralCheckpointAssessment,
    neuralTensorStorage,
    oidnModelUrl,
    oidnExecutionProviders: whExt?.oidnExecutionProviders,
    // Lite forces merged BVH (drops the 5 TLAS scene-group buffers — the lite
    // buffer-axis win) regardless of any host bvhMode override; warn so the
    // host knows instanced-scene fidelity is reduced on this weak adapter.
    restirBvhModeOverride: isLite
      ? (whExt?.bvhMode === 'tlas'
          ? (warnLiteBvhModeOverride(opts), 'merged')
          : 'merged')
      : whExt?.bvhMode,
    mneeMaxIterations: opts.causticOptions?.mneeMaxIterations ?? 8,
    mneeMaxChainLength: opts.causticOptions?.mneeMaxChainLength ?? 3,
    mneeMultiplicityTrials: opts.causticOptions?.mneeMultiplicityTrials ?? 8,
    // Precedence: explicit opts → preset → engine default (~60 FPS cap).
    // The preset never carries `null`, so it cannot accidentally disable the
    // cap; only an explicit `opts.targetFrameIntervalMs: null` does that.
    targetFrameIntervalMs: opts.targetFrameIntervalMs === null
      ? null
      : finiteOr(
          opts.targetFrameIntervalMs,
          preset.targetFrameIntervalMs ?? DEFAULT_TARGET_FRAME_INTERVAL_MS,
        ),
    // Library-generality tunables — table-driven; defaults preserve Cornell
    // behaviour, hosts override via HybridEngineOptions. `effectiveOpts`
    // carries the preset's adaptiveSamplingThresholds fallback.
    tunables: readTunables(effectiveOpts),
    initTunables: readInitTunables(opts),
    // 2026-05-18 sweep — `indirectFireflyClamp` is tuple-typed so it lives
    // outside the number-typed Tunables table; default preserves Cornell.
    indirectFireflyClamp: finiteTupleOr(opts.indirectFireflyClamp, [1.0, 1.0, 1.0]),
    // 2026-05-19 B3a — atrous DIRECT/INDIRECT sigmas; tuple-typed same as
    // indirectFireflyClamp. Defaults sourced from the single-source-of-truth
    // constants in bindGroupBuilders.ts (no duplicated literals).
    atrousDirectSigmas: finiteTupleOr(opts.atrousDirectSigmas, [
      ATROUS_DIRECT_SIGMAS.sigmaN, ATROUS_DIRECT_SIGMAS.sigmaZ, ATROUS_DIRECT_SIGMAS.sigmaC,
    ]),
    atrousIndirectSigmas: finiteTupleOr(opts.atrousIndirectSigmas, [
      ATROUS_INDIRECT_SIGMAS.sigmaN, ATROUS_INDIRECT_SIGMAS.sigmaZ, ATROUS_INDIRECT_SIGMAS.sigmaC,
    ]),
    // T5 — stained-glass opt-in flag bits. Default 0 (both terms OFF); hosts
    // opt in via opts.stainedGlass. Packed once here (construction-time
    // config); threaded into pipeline.renderFrame via _denoiserFilterDeps.
    stainedGlassFlags: packStainedGlassFlags({
      sunCaustic: opts.stainedGlass?.sunCaustic,
      skyAperture: opts.stainedGlass?.skyAperture,
    }),
    rcTransmittedInterfaceBudget: opts.rcTransmittedInterfaceBudget
      ?? RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
    // NRC cache flag. Default 0 (OFF) so the gi-ris suffix is bit-identical to
    // the verbatim DDGI-atlas estimate unless a host opts in via opts.nrcEnabled
    // (which tier:'lite' forbids — validated above). The real gate is compile-time
    // (selects the risGiNrc variant); this value is mirrored into the UBO.
    nrcEnabled: opts.nrcEnabled === true ? 1 : 0,
    // One resolved object drives shader constants, allocations, trainer
    // precision/cadence, and device preflight.
    nrcConfig,
    // PPG guided sampling. Default 0 (OFF) — bit-identical cosine kernel.
    // Forwarded to pipeline.initialize so the ppg-update pipeline is actually
    // built when a host opts in (tier:'lite' forbids it — validated above).
    ppgEnabled: opts.ppgEnabled === true ? 1 : 0,
    // PPG guide/cosine mixture weight. The default remains the paper value, but
    // hosts can tune/A-B favorable scenes without patching the coordinator.
    ppgMixAlpha: resolvePpgMixAlpha(opts.ppgMixAlpha),
    // Checkerboard half-res shading. Explicit opt wins, else the preset value
    // (ON for medium/low degradation tiers, OFF for ultra/high). No preset ⇒
    // ultra ⇒ OFF ⇒ shade + both spatial passes + ris shade every pixel +
    // ResolvePass passes through = bit-identical to the pre-checkerboard
    // pipeline. GPU-validated on dzn (whole-frame 1.46× at medium/low).
    checkerboard: opts.checkerboardRendering ?? preset.checkerboard,
    staticPipelineRebuildKey: opts.pipelineRebuildKey ?? null,
    getPipelineRebuildKey: opts.getPipelineRebuildKey,
    rebuildKeyFingerprintSeen: fingerprintHybridPipelineRebuildKey(
      opts.getPipelineRebuildKey?.() ?? opts.pipelineRebuildKey ?? null,
    ),
    maxBounces: resolveMaxBounces(opts.maxBounces),
    verbose: opts.verbose ?? false,
    debug: opts.debug ?? false,
    // Phase-0 productization — quality-preset-resolved structural / gating
    // knobs. Explicit per-knob opts override the preset.
    gtaoMode: opts.gtaoMode ?? preset.gtaoMode,
    diSpatialPasses: opts.diSpatialPasses ?? preset.diSpatialPasses,
    giSpatialPasses: opts.giSpatialPasses ?? preset.giSpatialPasses,
    ddgiUpdateDivisor: finitePositiveIntOr(opts.ddgiUpdateDivisor, preset.ddgiUpdateDivisor),
    // PPG train-pass cadence: explicit opt wins, else the preset value. Clamp
    // to ≥ 1 here too (the pipeline re-clamps, but keep the resolved config
    // honest so a debug surface reading it sees the effective value).
    ppgDispatchInterval: finitePositiveIntOr(opts.ppgDispatchInterval, preset.ppgDispatchInterval),
    // H47 — PPG max spatial cells. Pass-through; undefined = allocatePPGResources
    // default (1 024). Finite host values are normalized to positive integers.
    ppgMaxSpatialCells: optionalFinitePositiveInt(opts.ppgMaxSpatialCells),
    // H29 — PPG max dTree nodes per spatial cell. Pass-through; undefined =
    // allocatePPGResources / buildPpgUpdateWgsl default (341). The pipeline
    // threads this to BOTH shader compile and resource allocation so the GPU
    // update kernel stride matches the buffers.
    ppgMaxDTreeNodesPerCell: optionalFinitePositiveInt(opts.ppgMaxDTreeNodesPerCell),
    // ReGIR (Boksansky 2021) grid-based DI light selection. Pass-through from
    // opts; `undefined` ⇒ off (the pipeline's resolveReGIRConfig default).
    regirConfig: opts.regir,
    resolutionFactor: preset.resolutionFactor,
  };
}

/**
 * Parse + validate `HybridEngineOptions` into the immutable derived config.
 * Thin orchestrator over the two independently-testable halves:
 *   1. {@link validateHybridEngineOptions} — the pure throws (lite-mode
 *      violations, bad denoiser/neural/OIDN combos), in load-bearing order.
 *   2. {@link deriveHybridEngineConfig} — the 45-field defaulting record, given
 *      the already-resolved quality preset.
 *
 * The quality-tier resolution (`opts.qualityTier ?? (isLite ? 'medium' :
 * 'ultra')`) lives HERE — between the throws and the derive — so the
 * lite-biased default + explicit override exist in exactly one place. It has no
 * `this` or GPU dependency; construction diagnostics may emit through the host
 * warning sink. Behaviour-preserving over the pre-Theme-H inline path:
 * same throws in the same order, same derived config. See
 * {@link ParsedHybridEngineConfig}.
 */
export function parseHybridEngineOptions(opts: HybridEngineOptions): ParsedHybridEngineConfig {
  validateHybridEngineOptions(opts);

  // Phase-0 productization — resolve the coarse quality preset, then let
  // explicit per-knob options OVERRIDE it inside `deriveHybridEngineConfig`
  // (preset is a baseline, not a lock). `ultra` (the default) is byte-identical
  // to the pre-Phase-0 defaults. Lite biases the default tier to `'medium'`
  // (still overridable by an explicit `qualityTier`).
  const effectiveQualityTier = opts.qualityTier ?? (opts.tier === 'lite' ? 'medium' : 'ultra');
  const preset = resolveQualityPreset(effectiveQualityTier);

  const config = deriveHybridEngineConfig(opts, preset);
  if (config.denoiser === 'neural') {
    const feasibility = assessNeuralDeviceFeasibility(opts, preset, config.neuralTensorStorage);
    if (feasibility.failure != null) {
      throw new TypeError(
        `[HybridEngine] denoiser:'neural' is infeasible before allocation: ${feasibility.failure}`,
      );
    }
  }
  return config;
}
