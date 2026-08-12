// pt-webgpu factory option validation + warning sequence (T3-B god-file split, 2026-07-20).
//
// Extracted verbatim from `createPTEngine_WebGPU` in `index.ts` (~240 lines of
// throw/warn validation). Behaviour is byte-identical: the factory now calls
// `validatePtWebgpuOptions(opts)` which performs the same ordered sequence of
// structural throws + structured `onWarning` emissions and returns the resolved
// `{ traceTier, effectiveOpts }` the factory then hands to the engine constructor.
//
// Shared numeric caps and the small pure resolvers (`emitPteWarning`,
// `resolveBdptMaxLightBounces`, etc.) also live here now — the engine class in
// `index.ts` re-imports them, so there is a single source of truth.

import {
  resolveFrameCameraPosition,
  type EngineWarning,
  type FrameInput,
} from '@vitrum/core';
import {
  oidnModelUrlIsHostProvided,
  resolveOidnModelUrl,
} from '@vitrum/shared-denoisers';
import type { PTEngineWebGPUOptions } from './index.js';
import { resolvePtWebgpuTraceTier, type PtWebgpuTraceTier } from './traceTier.js';
import {
  PT_WEBGPU_BDPT_SUPPORT,
  PT_WEBGPU_DENOISER_VALUES as PT_WEBGPU_MANIFEST_DENOISER_VALUES,
  PT_WEBGPU_IMPLEMENTED_DENOISER_VALUES as PT_WEBGPU_MANIFEST_IMPLEMENTED_DENOISER_VALUES,
  PT_WEBGPU_SAMPLING_VALUES as PT_WEBGPU_MANIFEST_SAMPLING_VALUES,
  ptWebgpuSupportManifest,
} from './supportManifest.js';
import {
  PT_WEBGPU_BDPT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from './webgpuLimits.js';

const PT_WEBGPU_OPTION_KEY_RECORD = {
  device: true,
  maxBounces: true,
  maxSamplesPerPixel: true,
  denoiser: true,
  onWarning: true,
  causticStrategy: true,
  causticOptions: true,
  extensions: true,
  traceTier: true,
  spectral: true,
  bdpt: true,
  oneEdgeReconnectionReuse: true,
  oneEdgeReconnectionReuseOptions: true,
  restirPtReuse: true,
  restirPtReuseOptions: true,
  bvhTraversal: true,
  bdptOptions: true,
  lightTreeImportanceSampling: true,
  sampling: true,
  cameraVisibleEmitters: true,
  oidn: true,
  oidnBridgeLoader: true,
  oidnReadbackFn: true,
  debug: true,
} as const satisfies Readonly<Record<keyof PTEngineWebGPUOptions, true>>;
const PT_WEBGPU_OPTION_KEYS = new Set(Object.keys(PT_WEBGPU_OPTION_KEY_RECORD));

const PT_WEBGPU_DENOISER_VALUES =
  new Set<string>(PT_WEBGPU_MANIFEST_DENOISER_VALUES);
const PT_WEBGPU_IMPLEMENTED_DENOISER_VALUES =
  new Set<string>(PT_WEBGPU_MANIFEST_IMPLEMENTED_DENOISER_VALUES);
const PT_WEBGPU_TRACE_TIER_VALUES = new Set(['full', 'lite']);
const PT_WEBGPU_CAUSTIC_STRATEGY_VALUES = new Set(['none', 'manifold-nee', 'photon-map']);
const PT_WEBGPU_SAMPLING_VALUES =
  new Set<string>(PT_WEBGPU_MANIFEST_SAMPLING_VALUES);
const PT_WEBGPU_BVH_TRAVERSAL_VALUES = new Set(['binary', 'cwbvh-closest']);
const PT_WEBGPU_OIDN_EXECUTION_PROVIDER_VALUES = new Set(['webnn', 'webgpu', 'wasm']);

function describeUnknown(value: unknown): string {
  if (value === null) return 'null';
  if (typeof value === 'string') return value;
  if (
    typeof value === 'number' ||
    typeof value === 'boolean' ||
    typeof value === 'bigint' ||
    typeof value === 'undefined'
  ) {
    return String(value);
  }
  if (typeof value === 'symbol') return value.toString();
  if (typeof value === 'function') return `[function ${value.name || 'anonymous'}]`;
  return Object.prototype.toString.call(value);
}

function assertOptionsObject(value: unknown, label: string): asserts value is Record<string, unknown> {
  if (
    value === null ||
    typeof value !== 'object' ||
    Array.isArray(value) ||
    ArrayBuffer.isView(value)
  ) {
    throw new TypeError(`createPTEngine_WebGPU: ${label} must be an object`);
  }
  const prototype = Object.getPrototypeOf(value) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(
      `createPTEngine_WebGPU: ${label} must have Object.prototype or null prototype`,
    );
  }
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string') {
      throw new TypeError(
        `createPTEngine_WebGPU: ${label} contains unsupported symbol key ${String(key)}`,
      );
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor) || !descriptor.enumerable) {
      throw new TypeError(
        `createPTEngine_WebGPU: ${label}.${key} must be an enumerable own data property`,
      );
    }
  }
}

function assertKnownKeys(
  value: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  label: string,
): void {
  assertOptionsObject(value, label);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new TypeError(
      `createPTEngine_WebGPU: ${label} contains unknown key(s): ${unknown.join(', ')}`,
    );
  }
}

function assertOptionalBoolean(value: unknown, label: string): void {
  if (value !== undefined && typeof value !== 'boolean') {
    throw new TypeError(`createPTEngine_WebGPU: ${label} must be a boolean when supplied`);
  }
}

function assertOptionalEnum(
  value: unknown,
  allowed: ReadonlySet<string>,
  label: string,
): void {
  if (value !== undefined && (typeof value !== 'string' || !allowed.has(value))) {
    const received = describeUnknown(value);
    throw new RangeError(
      `createPTEngine_WebGPU: ${label} is unsupported (got ${received}); ` +
        `expected one of ${Array.from(allowed).map((entry) => JSON.stringify(entry)).join(', ')}`,
    );
  }
}

function assertPtWebgpuExtensions(value: unknown): void {
  if (value === undefined) return;
  assertOptionsObject(value, 'extensions');
  const keys = Object.keys(value);
  if (keys.length === 0) return;
  const graduatedMigrations: Readonly<Record<string, string>> = {
    'vitrum.ptWebgpu.spectralHeroWavelength': 'spectral:true',
    'vitrum.ptWebgpu.bdpt': 'bdpt:true plus bdptOptions',
    'vitrum.ptWebgpu.oidn': "denoiser:'oidn-final' plus oidn:{modelUrl}",
  };
  const migrations = Object.entries(graduatedMigrations)
    .filter(([prefix]) => keys.some((key) => key.startsWith(prefix)))
    .map(([prefix, replacement]) => `${prefix} → ${replacement}`);
  const migrationSuffix = migrations.length > 0
    ? ` Migrate graduated keys as follows: ${migrations.join('; ')}.`
    : '';
  throw new TypeError(
    `createPTEngine_WebGPU: extensions contains unsupported key(s): ` +
      `${keys.map((key) => JSON.stringify(key)).join(', ')}. ` +
      `No pt-webgpu extension keys are currently accepted.${migrationSuffix}`,
  );
}

export const PT_WEBGPU_MAX_BOUNCES = 8;
/** Default path depth for stills. Aligned with pt-webgl2's professional default of 8. */
export const PT_WEBGPU_DEFAULT_BOUNCES = PT_WEBGPU_MAX_BOUNCES;
export const BDPT_MAX_LIGHT_BOUNCES =
  PT_WEBGPU_BDPT_SUPPORT.maxLightVertices;
// D2 (2026-07-20): raised 1 → 2 unconditionally. With maxLv=2 the kernel
// connection loop `for lvi=1u; lvi<maxLv` executes lvi=1, so BDPT does real
// light-path connections out of the box instead of being silently inert at the
// old default of 1 (which performed zero connections). BDPT remains opt-in
// (`bdpt:true`); this only fixes its default light-bounce count.
export const BDPT_DEFAULT_LIGHT_BOUNCES = 2;

export function emitPteWarning(
  opts: Pick<PTEngineWebGPUOptions, 'onWarning'>,
  warning: EngineWarning,
  ...consoleArgs: readonly unknown[]
): void {
  console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
  try {
    opts.onWarning?.(warning);
  } catch {
    // Host warning callbacks must not break engine construction.
  }
}

function applyDefaultOidnModelUrl(opts: PTEngineWebGPUOptions): PTEngineWebGPUOptions {
  if (opts.denoiser !== 'auto' && opts.denoiser !== 'oidn-final') return opts;
  if (typeof opts.oidn?.modelUrl === 'string') return opts;
  const modelUrl = resolveOidnModelUrl(undefined);
  return {
    ...opts,
    oidn: opts.oidn != null ? { ...opts.oidn, modelUrl } : { modelUrl },
  };
}

function resolvePtWebgpuAutoDenoiser(
  opts: PTEngineWebGPUOptions,
  hostProvidedModelUrl: boolean,
): PTEngineWebGPUOptions {
  if (opts.denoiser !== 'auto') return opts;
  const reason = hostProvidedModelUrl ? 'host-oidn-model-url' : 'default-oidn-model-url';
  emitPteWarning(opts, {
    code: 'pt-webgpu.denoiser-auto-resolved',
    backend: 'pt-webgpu',
    phase: 'construction',
    method: 'createPTEngine_WebGPU',
    message:
      `[vitrum/pt-webgpu] denoiser:'auto' resolved to 'oidn-final' (${reason}). ` +
      (hostProvidedModelUrl
        ? 'Using host oidn.modelUrl for the async final-pass OIDN denoiser.'
        : `Using the default Intel RT HDR alb+nrm ONNX (${resolveOidnModelUrl(undefined)}). ` +
          'Override with oidn.modelUrl to self-host the weights.'),
    details: {
      requested: 'auto',
      resolved: 'oidn-final',
      reason,
      packageProvidesProductionWeights: false,
    },
  });
  return { ...opts, denoiser: 'oidn-final' };
}

export function resolveBdptMaxLightBounces(requested: number | undefined): number {
  return requested ?? BDPT_DEFAULT_LIGHT_BOUNCES;
}

function assertFiniteArray(value: unknown, length: number, label: string): void {
  if (value == null || typeof value !== 'object' || !('length' in value) ||
      (value as { readonly length?: unknown }).length !== length) {
    throw new TypeError(`renderFrame: ${label} must be an array-like value of length ${length}`);
  }
  const array = value as ArrayLike<unknown>;
  for (let index = 0; index < length; index += 1) {
    const component = array[index];
    if (typeof component !== 'number' || !Number.isFinite(component)) {
      throw new RangeError(`renderFrame: ${label}[${index}] must be finite (got ${String(component)})`);
    }
  }
}

function assertFiniteNumber(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number') throw new TypeError(`renderFrame: ${label} must be a number`);
  if (!Number.isFinite(value)) throw new RangeError(`renderFrame: ${label} must be finite (got ${value})`);
}

function assertNonNegativeFloat32(
  value: unknown,
  label: string,
): asserts value is number {
  assertFiniteNumber(value, label);
  if (value < 0) {
    throw new RangeError(`renderFrame: ${label} must be >= 0 (got ${value})`);
  }
  const stored = Math.fround(value);
  if (!Number.isFinite(stored)) {
    throw new RangeError(
      `renderFrame: ${label} must be representable as a finite float32 (got ${value})`,
    );
  }
  if (value > 0 && stored === 0) {
    throw new RangeError(
      `renderFrame: ${label} underflows float32 storage (got ${value})`,
    );
  }
}

function assertU32(value: unknown, label: string): asserts value is number {
  assertFiniteNumber(value, label);
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new RangeError(`renderFrame: ${label} must be an integer in 0..4294967295 (got ${value})`);
  }
}

function assertPositiveSafeInteger(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `renderFrame: ${label} must be a positive safe integer (got ${String(value)})`,
    );
  }
}

/** Validate the eager render-target dimensions accepted by `Engine.setSize`. */
export function validatePtWebgpuPixelSize(
  method: string,
  width: unknown,
  height: unknown,
): asserts width is number {
  for (const [label, value] of [['width', width], ['height', height]] as const) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(
        `${method}: ${label} must be a positive safe integer (got ${String(value)})`,
      );
    }
  }
}

/** Validate every numeric field before renderFrame mutates state or allocates GPU resources. */
export function validatePtWebgpuFrameInput(input: FrameInput): void {
  if (input == null || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('renderFrame: input must be a FrameInput object');
  }
  assertFiniteArray(input.viewMatrix, 16, 'viewMatrix');
  assertFiniteArray(input.projMatrix, 16, 'projMatrix');
  if (input.cameraPosition !== undefined) {
    assertFiniteArray(input.cameraPosition, 3, 'cameraPosition');
  }
  resolveFrameCameraPosition(input, 'PTEngineWebGPU.renderFrame');
  if (input.prevViewMatrix !== undefined) assertFiniteArray(input.prevViewMatrix, 16, 'prevViewMatrix');
  if (input.prevProjMatrix !== undefined) assertFiniteArray(input.prevProjMatrix, 16, 'prevProjMatrix');
  if (input.viewport == null || typeof input.viewport !== 'object' || Array.isArray(input.viewport)) {
    throw new TypeError('renderFrame: viewport must be an object');
  }
  assertPositiveSafeInteger(input.viewport.width, 'viewport.width');
  assertPositiveSafeInteger(input.viewport.height, 'viewport.height');
  assertFiniteNumber(input.viewport.devicePixelRatio, 'viewport.devicePixelRatio');
  if (input.viewport.devicePixelRatio <= 0) {
    throw new RangeError(
      `renderFrame: viewport.devicePixelRatio must be > 0 (got ${input.viewport.devicePixelRatio})`,
    );
  }
  assertU32(input.frameIndex, 'frameIndex');
  assertU32(input.frameSeed, 'frameSeed');
  const quality = input.quality;
  if (quality === undefined) return;
  if (quality === null || typeof quality !== 'object' || Array.isArray(quality)) {
    throw new TypeError('renderFrame: quality must be an object when supplied');
  }
  for (const field of ['samplesTarget', 'bounces'] as const) {
    const value = quality[field];
    if (value === undefined) continue;
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(
        `renderFrame: quality.${field} must be a positive safe integer (got ${String(value)})`,
      );
    }
  }
  if (quality.resolutionFactor !== undefined) {
    assertFiniteNumber(quality.resolutionFactor, 'quality.resolutionFactor');
    if (quality.resolutionFactor <= 0 || quality.resolutionFactor > 1) {
      throw new RangeError(
        `renderFrame: quality.resolutionFactor must be in (0, 1] (got ${quality.resolutionFactor})`,
      );
    }
  }
  if (quality.exposure !== undefined) {
    assertNonNegativeFloat32(quality.exposure, 'quality.exposure');
  }
  if (quality.filteredGlossyFactor !== undefined) {
    assertFiniteNumber(quality.filteredGlossyFactor, 'quality.filteredGlossyFactor');
    if (quality.filteredGlossyFactor !== 0) {
      throw new RangeError(
        'renderFrame: quality.filteredGlossyFactor is unsupported by pt-webgpu; ' +
          'use 0 or omit it.',
      );
    }
  }
  if (quality.tonemap !== undefined &&
      !['aces', 'agx', 'reinhard', 'linear', 'none'].includes(quality.tonemap)) {
    throw new RangeError(`renderFrame: quality.tonemap is unsupported (got ${String(quality.tonemap)})`);
  }
  if (quality.outputColorSpace !== undefined &&
      quality.outputColorSpace !== 'srgb' && quality.outputColorSpace !== 'linear') {
    throw new RangeError(
      `renderFrame: quality.outputColorSpace is unsupported (got ${String(quality.outputColorSpace)})`,
    );
  }
}

/**
 * The native BDPT t=1 endpoint implements the finite-area perspective-camera
 * measure. An affine/orthographic projection has a delta directional measure
 * and cannot silently reuse that estimator.
 */
export function assertPtWebgpuBdptFrameCameraSupported(input: FrameInput): void {
  const projection = input.projMatrix;
  const affineHomogeneousRow =
    projection[3] === 0 &&
    projection[7] === 0 &&
    projection[11] === 0;
  if (affineHomogeneousRow) {
    throw new RangeError(
      'PTEngineWebGPU.renderFrame: bdpt:true supports perspective camera ' +
        'projections only; an orthographic/affine projection was supplied.',
    );
  }
}

function assertRestirPtReuseSupported(device: GPUDevice, traceTier: PtWebgpuTraceTier): void {
  if (traceTier !== 'full') {
    throw new Error(
      'createPTEngine_WebGPU: oneEdgeReconnectionReuse requires traceTier "full"; the selected lite tier cannot bind the reconnection reservoirs.',
    );
  }
  const maxBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (maxBuffers < PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
    throw new Error(
      `createPTEngine_WebGPU: oneEdgeReconnectionReuse requires maxStorageBuffersPerShaderStage >= ` +
        `${PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE}; device exposes ${maxBuffers}. ` +
        'Request the one-edge reconnection limit floor when acquiring the GPUDevice.',
    );
  }
}

function assertCwbvhClosestSupported(
  device: GPUDevice,
  traceTier: PtWebgpuTraceTier,
  restirPtReuse: boolean,
): void {
  if (traceTier !== 'full') {
    throw new Error(
      'createPTEngine_WebGPU: bvhTraversal:\'cwbvh-closest\' requires traceTier "full"; the selected lite tier does not bind full-tier TLAS/material/CWBVH groups.',
    );
  }
  const required = restirPtReuse
    ? PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE
    : PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const maxBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (maxBuffers < required) {
    throw new Error(
      "createPTEngine_WebGPU: bvhTraversal:'cwbvh-closest' requires " +
        `maxStorageBuffersPerShaderStage >= ${required}; device exposes ${maxBuffers}. ` +
        'Request the CWBVH traversal limit floor when acquiring the GPUDevice, or omit bvhTraversal to use the binary BVH.',
    );
  }
}

function assertBdptCameraSplatSupported(
  device: GPUDevice,
  traceTier: PtWebgpuTraceTier,
  restirPtReuse: boolean,
  cwbvhClosest: boolean,
): void {
  const support =
    ptWebgpuSupportManifest(traceTier).bidirectionalPathTracing;
  if (support == null) {
    throw new Error(
      'createPTEngine_WebGPU: bdpt:true is unavailable in the selected support profile.',
    );
  }
  if (support.cameraSplatStrategy !== 'native') {
    throw new Error(
      'createPTEngine_WebGPU: bdpt:true requires a native t=1 camera-splat strategy.',
    );
  }
  const reconnectionStorageBindings =
    PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE -
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const cwbvhStorageBindings =
    PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE -
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const required =
    PT_WEBGPU_BDPT_REQUIRED_STORAGE_BUFFERS_PER_STAGE +
    (restirPtReuse ? reconnectionStorageBindings : 0) +
    (cwbvhClosest ? cwbvhStorageBindings : 0);
  const maxBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (maxBuffers < required) {
    throw new Error(
      'createPTEngine_WebGPU: bdpt:true requires the native t=1 camera-splat ' +
        `layout with maxStorageBuffersPerShaderStage >= ${required}; device exposes ${maxBuffers}. ` +
        'Request the BDPT limit floor when acquiring the GPUDevice.',
    );
  }
}

/** The resolved values `createPTEngine_WebGPU` needs after validation. */
export interface ValidatedPtWebgpuOptions {
  readonly traceTier: PtWebgpuTraceTier;
  readonly effectiveOpts: PTEngineWebGPUOptions;
}

/**
 * Validate + normalize `createPTEngine_WebGPU` options: perform every structural
 * throw and emit every construction-time `onWarning`, then return the resolved
 * trace tier and denoiser-resolved effective options.
 * Byte-identical to the former inline factory body (T3-B extraction).
 */
export function validatePtWebgpuOptions(
  opts: PTEngineWebGPUOptions,
  internal?: { readonly deviceIndependent?: boolean },
): ValidatedPtWebgpuOptions {
  const rawOpts: unknown = opts;
  assertOptionsObject(rawOpts, 'options');
  assertKnownKeys(rawOpts, PT_WEBGPU_OPTION_KEYS, 'options');
  assertOptionalEnum(rawOpts.traceTier, PT_WEBGPU_TRACE_TIER_VALUES, 'traceTier');
  assertOptionalEnum(rawOpts.causticStrategy, PT_WEBGPU_CAUSTIC_STRATEGY_VALUES, 'causticStrategy');
  assertOptionalEnum(rawOpts.sampling, PT_WEBGPU_SAMPLING_VALUES, 'sampling');
  assertOptionalEnum(rawOpts.bvhTraversal, PT_WEBGPU_BVH_TRAVERSAL_VALUES, 'bvhTraversal');
  assertOptionalEnum(rawOpts.denoiser, PT_WEBGPU_DENOISER_VALUES, 'denoiser');
  if (
    rawOpts.denoiser !== undefined &&
    !PT_WEBGPU_IMPLEMENTED_DENOISER_VALUES.has(rawOpts.denoiser as string)
  ) {
    throw new RangeError(
      `createPTEngine_WebGPU: denoiser=${JSON.stringify(rawOpts.denoiser)} is ` +
        `unsupported by this converged backend; implemented modes are ` +
        `'none', 'auto', and 'oidn-final'. The request is not degraded to a ` +
        `different estimator.`,
    );
  }
  for (const key of [
    'spectral',
    'bdpt',
    'oneEdgeReconnectionReuse',
    'restirPtReuse',
    'lightTreeImportanceSampling',
    'cameraVisibleEmitters',
    'debug',
  ] as const) {
    assertOptionalBoolean(rawOpts[key], key);
  }
  if (
    rawOpts.oneEdgeReconnectionReuse !== undefined &&
    rawOpts.restirPtReuse !== undefined &&
    rawOpts.oneEdgeReconnectionReuse !== rawOpts.restirPtReuse
  ) {
    throw new TypeError(
      'createPTEngine_WebGPU: oneEdgeReconnectionReuse and deprecated ' +
      'restirPtReuse disagree; supply one value.',
    );
  }
  if (rawOpts.onWarning !== undefined && typeof rawOpts.onWarning !== 'function') {
    throw new TypeError('createPTEngine_WebGPU: onWarning must be a function when supplied');
  }
  assertPtWebgpuExtensions(rawOpts.extensions);
  if (
    internal?.deviceIndependent !== true &&
    (opts.device == null || typeof opts.device.createCommandEncoder !== 'function')
  ) {
    throw new TypeError('createPTEngine_WebGPU: device must be a GPUDevice instance');
  }
  const maxBounces = opts.maxBounces;
  if (maxBounces !== undefined && typeof maxBounces !== 'number') {
    throw new TypeError('createPTEngine_WebGPU: maxBounces must be a number when supplied');
  }
  if (
    maxBounces !== undefined &&
    (!Number.isFinite(maxBounces) || !Number.isInteger(maxBounces) ||
      maxBounces < 1 || maxBounces > PT_WEBGPU_MAX_BOUNCES)
  ) {
    throw new RangeError(
      'createPTEngine_WebGPU: maxBounces must be an integer in 1..' +
        PT_WEBGPU_MAX_BOUNCES + ' (got ' + maxBounces + ')',
    );
  }
  const maxSpp = opts.maxSamplesPerPixel;
  if (maxSpp !== undefined && typeof maxSpp !== 'number') {
    throw new TypeError('createPTEngine_WebGPU: maxSamplesPerPixel must be a number when supplied');
  }
  if (
    maxSpp !== undefined &&
    (!Number.isFinite(maxSpp) || !Number.isInteger(maxSpp) || maxSpp < 1 || maxSpp > 0xffffffff)
  ) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxSamplesPerPixel must be an integer in 1..4294967295 (got ${maxSpp})`,
    );
  }
  const bdptOptions = opts.bdptOptions;
  if (bdptOptions !== undefined && (bdptOptions === null || typeof bdptOptions !== 'object' || Array.isArray(bdptOptions))) {
    throw new TypeError('createPTEngine_WebGPU: bdptOptions must be an object when supplied');
  }
  if (bdptOptions !== undefined) {
    assertKnownKeys(
      bdptOptions,
      new Set(['maxLightBounces']),
      'bdptOptions',
    );
  }
  const bdptMaxLightBounces = opts.bdptOptions?.maxLightBounces;
  if (bdptMaxLightBounces !== undefined && typeof bdptMaxLightBounces !== 'number') {
    throw new TypeError('createPTEngine_WebGPU: bdptOptions.maxLightBounces must be a number when supplied');
  }
  if (
    bdptMaxLightBounces !== undefined &&
    (!Number.isFinite(bdptMaxLightBounces) ||
      !Number.isInteger(bdptMaxLightBounces) ||
      bdptMaxLightBounces < 1 ||
      bdptMaxLightBounces > BDPT_MAX_LIGHT_BOUNCES)
  ) {
    throw new RangeError(
      'createPTEngine_WebGPU: bdptOptions.maxLightBounces must be an integer in 1..' +
        BDPT_MAX_LIGHT_BOUNCES + ' (got ' + bdptMaxLightBounces + ')',
    );
  }
  if (opts.bdpt !== true && bdptOptions !== undefined && Object.keys(bdptOptions).length > 0) {
    throw new Error(
      'createPTEngine_WebGPU: non-empty bdptOptions requires bdpt:true; ' +
        'the tuning object is not silently ignored when BDPT is disabled.',
    );
  }
  const causticOptions = opts.causticOptions;
  if (causticOptions !== undefined && (causticOptions === null || typeof causticOptions !== 'object' || Array.isArray(causticOptions))) {
    throw new TypeError('createPTEngine_WebGPU: causticOptions must be an object when supplied');
  }
  if (causticOptions !== undefined) {
    assertKnownKeys(
      causticOptions,
      new Set(['mneeMaxIterations', 'mneeMaxChainLength']),
      'causticOptions',
    );
  }
  for (const [field, maximum] of [
    ['mneeMaxIterations', 32],
    ['mneeMaxChainLength', 8],
  ] as const) {
    const value = causticOptions?.[field];
    if (value === undefined) continue;
    if (typeof value !== 'number') {
      throw new TypeError(`createPTEngine_WebGPU: causticOptions.${field} must be a number when supplied`);
    }
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1 || value > maximum) {
      throw new RangeError(
        `createPTEngine_WebGPU: causticOptions.${field} must be an integer in 1..${maximum} (got ${value})`,
      );
    }
  }
  if (
    opts.causticStrategy !== 'manifold-nee' &&
    causticOptions !== undefined &&
    Object.keys(causticOptions).length > 0
  ) {
    throw new Error(
      'createPTEngine_WebGPU: non-empty causticOptions requires ' +
        'causticStrategy="manifold-nee"; MNEE tuning is not silently ignored.',
    );
  }
  if (
    opts.oneEdgeReconnectionReuseOptions !== undefined &&
    opts.restirPtReuseOptions !== undefined
  ) {
    throw new TypeError(
      'createPTEngine_WebGPU: supply oneEdgeReconnectionReuseOptions or the ' +
      'deprecated restirPtReuseOptions alias, not both.',
    );
  }
  const restirPtOptions =
    opts.oneEdgeReconnectionReuseOptions ?? opts.restirPtReuseOptions;
  const restirPtOptionsLabel =
    opts.oneEdgeReconnectionReuseOptions !== undefined
      ? 'oneEdgeReconnectionReuseOptions'
      : 'restirPtReuseOptions';
  if (
    restirPtOptions !== undefined &&
    (restirPtOptions === null || typeof restirPtOptions !== 'object' ||
      Array.isArray(restirPtOptions))
  ) {
    throw new TypeError(
      `createPTEngine_WebGPU: ${restirPtOptionsLabel} must be an object when supplied`,
    );
  }
  if (restirPtOptions !== undefined) {
    assertOptionsObject(restirPtOptions, restirPtOptionsLabel);
  }
  const restirPtOptionKeys = restirPtOptions == null ? [] : Object.keys(restirPtOptions);
  const unknownRestirPtOptionKeys = restirPtOptionKeys.filter(
    (key) => key !== 'mClamp',
  );
  if (unknownRestirPtOptionKeys.length > 0) {
    throw new TypeError(
      `createPTEngine_WebGPU: ${restirPtOptionsLabel} contains unknown key(s): ` +
        unknownRestirPtOptionKeys.join(', '),
    );
  }
  const restirPtMClamp = restirPtOptions?.mClamp;
  if (restirPtMClamp !== undefined && typeof restirPtMClamp !== 'number') {
    throw new TypeError(
      `createPTEngine_WebGPU: ${restirPtOptionsLabel}.mClamp must be a number when supplied`,
    );
  }
  if (
    restirPtMClamp !== undefined &&
    (!Number.isFinite(restirPtMClamp) || !Number.isInteger(restirPtMClamp) ||
      restirPtMClamp < 1 || restirPtMClamp > 4095)
  ) {
    throw new RangeError(
      `createPTEngine_WebGPU: ${restirPtOptionsLabel}.mClamp must be an integer in 1..4095 ` +
        `(got ${restirPtMClamp})`,
    );
  }
  const oneEdgeReconnectionRequested =
    (opts.oneEdgeReconnectionReuse ?? opts.restirPtReuse) === true;
  if (!oneEdgeReconnectionRequested && restirPtOptionKeys.length > 0) {
    throw new Error(
      `createPTEngine_WebGPU: non-empty ${restirPtOptionsLabel} requires ` +
        'oneEdgeReconnectionReuse:true (or deprecated restirPtReuse:true); ' +
        'the tuning object is not silently ignored when reuse is disabled.',
    );
  }
  const hostProvidedOidnUrl = oidnModelUrlIsHostProvided(opts.oidn?.modelUrl);
  const optsWithOidnUrl = applyDefaultOidnModelUrl(opts);
  const oidnOptions = optsWithOidnUrl.oidn;
  if (oidnOptions !== undefined &&
      (oidnOptions === null || typeof oidnOptions !== 'object' || Array.isArray(oidnOptions))) {
    throw new TypeError('createPTEngine_WebGPU: oidn must be an object when supplied');
  }
  if (oidnOptions !== undefined) {
    assertKnownKeys(
      oidnOptions,
      new Set(['modelUrl', 'executionProviders']),
      'oidn',
    );
    if (typeof oidnOptions.modelUrl !== 'string') {
      throw new TypeError('createPTEngine_WebGPU: oidn.modelUrl must be a string');
    }
    if (oidnOptions.modelUrl.trim().length === 0) {
      throw new RangeError('createPTEngine_WebGPU: oidn.modelUrl must not be empty');
    }
    if (oidnOptions.executionProviders !== undefined) {
      if (!Array.isArray(oidnOptions.executionProviders)) {
        throw new TypeError(
          'createPTEngine_WebGPU: oidn.executionProviders must be an array when supplied',
        );
      }
      for (const provider of oidnOptions.executionProviders) {
        assertOptionalEnum(
          provider,
          PT_WEBGPU_OIDN_EXECUTION_PROVIDER_VALUES,
          'oidn.executionProviders[]',
        );
      }
    }
  }
  if (opts.oidnBridgeLoader !== undefined && typeof opts.oidnBridgeLoader !== 'function') {
    throw new TypeError('createPTEngine_WebGPU: oidnBridgeLoader must be a function when supplied');
  }
  if (opts.oidnReadbackFn !== undefined && typeof opts.oidnReadbackFn !== 'function') {
    throw new TypeError('createPTEngine_WebGPU: oidnReadbackFn must be a function when supplied');
  }
  const oidnModeRequested = opts.denoiser === 'oidn-final' || opts.denoiser === 'auto';
  if (
    !oidnModeRequested &&
    (opts.oidn !== undefined || opts.oidnBridgeLoader !== undefined || opts.oidnReadbackFn !== undefined)
  ) {
    throw new Error(
      "createPTEngine_WebGPU: oidn/oidnBridgeLoader/oidnReadbackFn require denoiser:'oidn-final' or 'auto'; " +
        'OIDN configuration is not silently ignored by another denoiser mode.',
    );
  }
  if (internal?.deviceIndependent === true) {
    return {
      traceTier: 'full',
      effectiveOpts: optsWithOidnUrl,
    };
  }
  const traceTier = resolvePtWebgpuTraceTier(opts.device, opts.traceTier);
  const effectiveOpts = resolvePtWebgpuAutoDenoiser(optsWithOidnUrl, hostProvidedOidnUrl);
  if (traceTier === 'lite' && opts.causticStrategy != null && opts.causticStrategy !== 'none') {
    throw new Error(
      `createPTEngine_WebGPU: causticStrategy=${JSON.stringify(opts.causticStrategy)} requires ` +
        'traceTier "full"; lite does not silently redirect a requested caustic estimator to none.',
    );
  }
  if (traceTier === 'lite' && opts.lightTreeImportanceSampling === true) {
    throw new Error(
      'createPTEngine_WebGPU: lightTreeImportanceSampling:true requires traceTier "full"; ' +
        'the lite kernel uses uniform emitter selection.',
    );
  }
  if (traceTier === 'lite' && opts.cameraVisibleEmitters === true) {
    throw new Error(
      'createPTEngine_WebGPU: cameraVisibleEmitters:true requires traceTier "full" because ' +
        'lite does not ingest explicit mesh-area emitters.',
    );
  }
  if (
    opts.bdpt === true &&
    ptWebgpuSupportManifest(traceTier).bidirectionalPathTracing == null
  ) {
    throw new Error(
      'createPTEngine_WebGPU: bdpt:true requires a support profile with ' +
        'bidirectionalPathTracing; the selected lite profile does not compose ' +
        'the BDPT connection or invocation-private path state.',
    );
  }
  const cwbvhClosestRequested = opts.bvhTraversal === 'cwbvh-closest';

  if (oneEdgeReconnectionRequested) {
    assertRestirPtReuseSupported(opts.device, traceTier);
  }
  if (cwbvhClosestRequested) {
    assertCwbvhClosestSupported(
      opts.device,
      traceTier,
      oneEdgeReconnectionRequested,
    );
  }
  if (opts.bdpt === true) {
    assertBdptCameraSplatSupported(
      opts.device,
      traceTier,
      oneEdgeReconnectionRequested,
      cwbvhClosestRequested,
    );
  }
  if (traceTier === 'full') {
    console.info(
      '[vitrum/pt-webgpu] Full trace tier: TLAS, analytic shapes, HDRI, area lights, motion/variance aux, caustics.',
    );
  } else {
    emitPteWarning(opts, {
      code: 'pt-webgpu.lite-tier',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        '[vitrum/pt-webgpu] Lite trace tier (software-adapter fallback): merged mesh-like BVH, directional/point/spot/rect-area/disc-area emitters, HDRI and procedural-sky environments. ' +
        'Disabled on lite: analytic shapes, TLAS, mesh-area emitters, caustics, BDPT, and motion/variance aux buffers. ' +
        `On a discrete GPU host, request a device with maxStorageBuffersPerShaderStage >= ${PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE} and maxStorageTexturesPerShaderStage >= 5, or pass traceTier: "full" after verifying limits.`,
      details: { traceTier },
    });
  }
  return { traceTier, effectiveOpts };
}

/**
 * Validate a createEngine advanced bag without acquiring or inspecting a
 * GPUAdapter/GPUDevice. The concrete factory reuses the same checks.
 */
export function validatePtWebgpuAdvancedOptions(
  opts: Partial<Omit<PTEngineWebGPUOptions, 'device'>>,
): void {
  validatePtWebgpuOptions(
    opts as PTEngineWebGPUOptions,
    { deviceIndependent: true },
  );
}
