// Factory-boundary validation for createPTEngine_WebGL2.
//
// This module deliberately validates the runtime JavaScript payload rather than
// trusting TypeScript. Deserialised options, plain JS hosts, and cast callers
// must fail before capability probing or GL resource construction.

import type { EngineWarning } from '@vitrum/core';
import type { PTEngineWebGL2Options } from './options.js';
import { WEBGL2_MAX_BOUNCES } from './limits.js';
import { DEFAULT_RENDER_TARGET_BUDGET_BYTES } from './gl/renderTargetBudget.js';

// General BDPT uses a fixed-size texture stack: one endpoint plus up to seven
// surface/medium extensions. Four vertices is the balanced production default;
// hosts can explicitly trade work for depth anywhere in the bounded 1..8 range.
export const BDPT_DEFAULT_LIGHT_BOUNCES = 4;
export const BDPT_MAX_LIGHT_BOUNCES = 8;

const GL_INT_MAX = 0x7fff_ffff;

const TOP_LEVEL_KEY_RECORD = {
  device: true,
  maxBounces: true,
  maxSamplesPerPixel: true,
  denoiser: true,
  onWarning: true,
  causticStrategy: true,
  causticOptions: true,
  extensions: true,
  maxRenderTargetBytes: true,
  traceTier: true,
  spectral: true,
  bdpt: true,
  sampling: true,
  bdptOptions: true,
  materialLodDepth: true,
  backgroundAlpha: true,
  backgroundBlur: true,
  cameraType: true,
  dof: true,
  oidn: true,
  oidnBridgeLoader: true,
  oidnReadbackFn: true,
} as const satisfies Readonly<Record<keyof PTEngineWebGL2Options, true>>;
const TOP_LEVEL_KEYS = new Set(Object.keys(TOP_LEVEL_KEY_RECORD));
const BDPT_OPTION_KEYS = new Set(['maxLightBounces']);
const DOF_KEYS = new Set([
  'focusDistance',
  'bokehSize',
  'apertureBlades',
  'apertureRotation',
  'anamorphicRatio',
]);
const OIDN_KEYS = new Set(['modelUrl', 'executionProviders']);
const TRACE_TIERS = new Set(['full', 'lite']);
const SAMPLING_MODES = new Set(['pcg', 'sobol']);
const CAMERA_TYPES = new Set(['perspective', 'orthographic', 'equirectangular']);
const CAUSTIC_STRATEGIES = new Set(['bdpt']);
const WEBGL2_DENOISERS = new Set(['none', 'auto', 'oidn-final']);
const OIDN_EXECUTION_PROVIDERS = new Set(['webnn', 'webgpu', 'wasm']);

function assertKnownObject(
  label: string,
  value: unknown,
  knownKeys: ReadonlySet<string>,
): asserts value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a non-array object`);
  }
  if (ArrayBuffer.isView(value)) {
    throw new TypeError(`${label} must be a plain object`);
  }
    const prototype: unknown = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(`${label} must have Object.prototype or null prototype`);
  }
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string') {
      throw new TypeError(`${label} contains unsupported symbol keys (${String(key)})`);
    }
    if (!knownKeys.has(key)) {
      throw new TypeError(`${label} contains unsupported field "${key}"`);
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor) || !descriptor.enumerable) {
      throw new TypeError(`${label}.${key} must be an enumerable own data property`);
    }
  }
}

function describeValidationValue(value: unknown): string {
  return value !== null && typeof value === 'object'
    ? Object.prototype.toString.call(value)
    : String(value);
}

function isUnknownArray(value: unknown): value is readonly unknown[] {
  return Array.isArray(value);
}

function assertExtensionBag(value: unknown): void {
  if (value === undefined) return;
  const extensionKeys = value != null && typeof value === 'object'
    ? new Set(Object.getOwnPropertyNames(value))
    : new Set<string>();
  assertKnownObject('createPTEngine_WebGL2: extensions', value, extensionKeys);
}

function assertOptionalBoolean(label: string, value: unknown): void {
  if (value !== undefined && typeof value !== 'boolean') {
    throw new TypeError(`${label} must be a boolean (got ${describeValidationValue(value)})`);
  }
}

function assertOptionalEnum(
  label: string,
  value: unknown,
  supported: ReadonlySet<string>,
): void {
  if (value === undefined) return;
  if (typeof value !== 'string' || !supported.has(value)) {
    throw new RangeError(
      `${label} must be one of ${Array.from(supported, (entry) => `"${entry}"`).join(', ')} ` +
          `(got ${describeValidationValue(value)})`,
    );
  }
}

function assertFiniteNumber(label: string, value: unknown): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite (got ${String(value)})`);
  }
}

function assertOptionalFunction(label: string, value: unknown): void {
  if (value !== undefined && typeof value !== 'function') {
    throw new TypeError(`${label} must be a function when supplied`);
  }
}

export function resolveBdptMaxLightBounces(value: number | undefined): number {
  if (value === undefined) return BDPT_DEFAULT_LIGHT_BOUNCES;
  return value;
}

export function emitWebgl2Warning(
  opts: Pick<PTEngineWebGL2Options, 'onWarning'>,
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

function hasOidnModelUrl(opts: Pick<PTEngineWebGL2Options, 'oidn'>): boolean {
  return typeof opts.oidn?.modelUrl === 'string' && opts.oidn.modelUrl.trim().length > 0;
}

export function resolveWebgl2AutoDenoiser(opts: PTEngineWebGL2Options): PTEngineWebGL2Options {
  if (opts.denoiser !== 'auto') return opts;
  const resolved = hasOidnModelUrl(opts) ? 'oidn-final' : 'none';
  const reason = resolved === 'oidn-final' ? 'host-oidn-model-url' : 'no-host-model-assets';
  emitWebgl2Warning(opts, {
    code: 'pt-webgl2.denoiser-auto-resolved',
    backend: 'pt-webgl2',
    phase: 'construction',
    method: 'createPTEngine_WebGL2',
    message:
      `[vitrum/pt-webgl2] denoiser:'auto' resolved to '${resolved}' (${reason}). ` +
      `pt-webgl2 ships no OIDN model; provide oidn.modelUrl to enable the async final-pass OIDN denoiser.`,
    details: {
      requested: 'auto',
      resolved,
      reason,
      packageProvidesProductionWeights: false,
    },
  });
  return { ...opts, denoiser: resolved };
}

function validateDof(value: unknown): void {
  if (value === undefined) return;
  assertKnownObject('createPTEngine_WebGL2: dof', value, DOF_KEYS);

  assertFiniteNumber('createPTEngine_WebGL2: dof.focusDistance', value.focusDistance);
  if (value.focusDistance <= 0) {
    throw new RangeError(
      `createPTEngine_WebGL2: dof.focusDistance must be > 0 ` +
        `(got ${String(value.focusDistance)})`,
    );
  }
  assertFiniteNumber('createPTEngine_WebGL2: dof.bokehSize', value.bokehSize);
  if (value.bokehSize < 0) {
    throw new RangeError(
      `createPTEngine_WebGL2: dof.bokehSize must be >= 0 ` +
        `(got ${String(value.bokehSize)})`,
    );
  }
  if (value.apertureBlades !== undefined) {
    const blades = value.apertureBlades;
    if (
      typeof blades !== 'number' ||
      !Number.isSafeInteger(blades) ||
      (blades !== 0 && (blades < 3 || blades > GL_INT_MAX))
    ) {
      throw new RangeError(
        `createPTEngine_WebGL2: dof.apertureBlades must be 0 or an integer in ` +
            `3..${GL_INT_MAX} (got ${describeValidationValue(blades)})`,
      );
    }
  }
  if (value.apertureRotation !== undefined) {
    assertFiniteNumber(
      'createPTEngine_WebGL2: dof.apertureRotation',
      value.apertureRotation,
    );
  }
  if (value.anamorphicRatio !== undefined) {
    assertFiniteNumber(
      'createPTEngine_WebGL2: dof.anamorphicRatio',
      value.anamorphicRatio,
    );
    if (value.anamorphicRatio <= 0) {
      throw new RangeError(
        `createPTEngine_WebGL2: dof.anamorphicRatio must be > 0 ` +
          `(got ${String(value.anamorphicRatio)})`,
      );
    }
  }
}

function validateOidn(value: unknown): void {
  if (value === undefined) return;
  assertKnownObject('createPTEngine_WebGL2: oidn', value, OIDN_KEYS);
  if (typeof value.modelUrl !== 'string' || value.modelUrl.trim().length === 0) {
    throw new TypeError(
      'createPTEngine_WebGL2: oidn.modelUrl must be a non-empty string',
    );
  }
    if (value.executionProviders !== undefined) {
      const executionProviders: unknown = value.executionProviders;
      if (!isUnknownArray(executionProviders) || executionProviders.length === 0) {
        throw new TypeError(
          'createPTEngine_WebGL2: oidn.executionProviders must be a non-empty array',
        );
      }
      for (let i = 0; i < executionProviders.length; i += 1) {
        const provider = executionProviders[i];
      if (typeof provider !== 'string' || !OIDN_EXECUTION_PROVIDERS.has(provider)) {
        throw new RangeError(
          `createPTEngine_WebGL2: oidn.executionProviders[${i}] is unsupported ` +
            `(got ${String(provider)})`,
        );
      }
    }
  }
}

/**
 * Validate the complete factory payload and resolve denoiser:auto.
 * No capability query, shader work, GL allocation, or engine-state mutation may
 * occur before this function succeeds.
 */
export function validateAndResolveWebgl2Options(
  opts: PTEngineWebGL2Options,
  internal?: { readonly deviceIndependent?: boolean },
): PTEngineWebGL2Options {
  assertKnownObject('createPTEngine_WebGL2: options', opts, TOP_LEVEL_KEYS);

  if (internal?.deviceIndependent !== true) {
    const gl = opts.device;
    if (gl == null || typeof gl !== 'object' || typeof gl.createFramebuffer !== 'function') {
      throw new TypeError('createPTEngine_WebGL2: device must be a WebGL2RenderingContext');
    }
  }
  assertOptionalFunction('createPTEngine_WebGL2: onWarning', opts.onWarning);
  assertExtensionBag(opts.extensions);
  assertOptionalBoolean('createPTEngine_WebGL2: spectral', opts.spectral);
  assertOptionalBoolean('createPTEngine_WebGL2: bdpt', opts.bdpt);
  assertOptionalEnum('createPTEngine_WebGL2: traceTier', opts.traceTier, TRACE_TIERS);
  assertOptionalEnum('createPTEngine_WebGL2: sampling', opts.sampling, SAMPLING_MODES);
  assertOptionalEnum('createPTEngine_WebGL2: cameraType', opts.cameraType, CAMERA_TYPES);
  assertOptionalEnum(
    'createPTEngine_WebGL2: causticStrategy',
    opts.causticStrategy,
    CAUSTIC_STRATEGIES,
  );
  assertOptionalEnum('createPTEngine_WebGL2: denoiser', opts.denoiser, WEBGL2_DENOISERS);
  assertOptionalFunction(
    'createPTEngine_WebGL2: oidnBridgeLoader',
    opts.oidnBridgeLoader,
  );
  assertOptionalFunction(
    'createPTEngine_WebGL2: oidnReadbackFn',
    opts.oidnReadbackFn,
  );

  if (
    opts.maxBounces !== undefined &&
    (!Number.isInteger(opts.maxBounces) ||
      opts.maxBounces < 1 ||
      opts.maxBounces > WEBGL2_MAX_BOUNCES)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxBounces must be an integer in the supported range ` +
        `1..${WEBGL2_MAX_BOUNCES} (got ${String(opts.maxBounces)})`,
    );
  }
  if (
    opts.maxRenderTargetBytes !== undefined &&
    (!Number.isSafeInteger(opts.maxRenderTargetBytes) || opts.maxRenderTargetBytes <= 0)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxRenderTargetBytes must be a positive safe integer ` +
        `(default ${DEFAULT_RENDER_TARGET_BUDGET_BYTES}; got ${String(opts.maxRenderTargetBytes)})`,
    );
  }
  if (
    opts.maxSamplesPerPixel !== undefined &&
    (!Number.isSafeInteger(opts.maxSamplesPerPixel) || opts.maxSamplesPerPixel < 1)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxSamplesPerPixel must be a positive safe integer ` +
        `(got ${String(opts.maxSamplesPerPixel)})`,
    );
  }
  if (
    opts.materialLodDepth !== undefined &&
    (!Number.isSafeInteger(opts.materialLodDepth) ||
      opts.materialLodDepth < 0 ||
      opts.materialLodDepth > GL_INT_MAX)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: materialLodDepth must be an integer in 0..${GL_INT_MAX} ` +
        `(got ${String(opts.materialLodDepth)})`,
    );
  }

  if (opts.causticOptions !== undefined) {
    throw new RangeError(
      'createPTEngine_WebGL2: causticOptions are not accepted for the "bdpt" ' +
        'caustic strategy; use bdptOptions.maxLightBounces',
    );
  }
  if (opts.causticStrategy === 'bdpt' && opts.bdpt === false) {
    throw new RangeError(
      'createPTEngine_WebGL2: causticStrategy:"bdpt" requires BDPT; omit bdpt ' +
        'or set bdpt:true instead of bdpt:false',
    );
  }

  if (opts.bdptOptions !== undefined) {
    assertKnownObject(
      'createPTEngine_WebGL2: bdptOptions',
      opts.bdptOptions,
      BDPT_OPTION_KEYS,
    );
  }
  const bdptMaxLightBounces = opts.bdptOptions?.maxLightBounces;
  if (
    bdptMaxLightBounces !== undefined &&
    (!Number.isInteger(bdptMaxLightBounces) ||
      bdptMaxLightBounces < 1 ||
      bdptMaxLightBounces > BDPT_MAX_LIGHT_BOUNCES)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: bdptOptions.maxLightBounces must be an integer in ` +
        `the supported range 1..${BDPT_MAX_LIGHT_BOUNCES} ` +
        `(got ${String(bdptMaxLightBounces)})`,
    );
  }
  const bdptSelected = opts.bdpt === true || opts.causticStrategy === 'bdpt';
  if (
    opts.bdptOptions !== undefined &&
    (Object.getOwnPropertyNames(opts.bdptOptions).length > 0 ||
      bdptMaxLightBounces !== undefined) &&
    !bdptSelected
  ) {
    throw new RangeError(
      'createPTEngine_WebGL2: bdptOptions requires bdpt:true or ' +
        'causticStrategy:"bdpt"; BDPT-only tuning must not be inert',
    );
  }

  if (opts.backgroundAlpha !== undefined) {
    assertFiniteNumber('createPTEngine_WebGL2: backgroundAlpha', opts.backgroundAlpha);
    if (opts.backgroundAlpha < 0 || opts.backgroundAlpha > 1) {
      throw new RangeError(
        `createPTEngine_WebGL2: backgroundAlpha must be in [0, 1] ` +
          `(got ${String(opts.backgroundAlpha)})`,
      );
    }
  }
  if (opts.backgroundBlur !== undefined) {
    assertFiniteNumber('createPTEngine_WebGL2: backgroundBlur', opts.backgroundBlur);
    if (opts.backgroundBlur < 0) {
      throw new RangeError(
        `createPTEngine_WebGL2: backgroundBlur must be >= 0 ` +
          `(got ${String(opts.backgroundBlur)})`,
      );
    }
  }

  validateDof(opts.dof);
  if (opts.cameraType === 'equirectangular' && opts.dof !== undefined) {
    throw new RangeError(
      'createPTEngine_WebGL2: dof is unsupported when cameraType is "equirectangular"; ' +
        'thin-lens depth of field has no coherent full-sphere focal plane',
    );
  }

  validateOidn(opts.oidn);
  if (
    internal?.deviceIndependent === true &&
    opts.denoiser === 'oidn-final' &&
    !hasOidnModelUrl(opts)
  ) {
    throw new Error(
      "createPTEngine_WebGL2: denoiser: 'oidn-final' requires oidn: { modelUrl }",
    );
  }
  if (internal?.deviceIndependent === true) return opts;
  const effectiveOpts = resolveWebgl2AutoDenoiser(opts);
  if (effectiveOpts.denoiser === 'oidn-final' && !hasOidnModelUrl(effectiveOpts)) {
    throw new Error(
      "createPTEngine_WebGL2: denoiser: 'oidn-final' is not turnkey - it " +
        'requires TWO host-provided assets that vitrum does not ship: ' +
        '(1) oidn: { modelUrl } - a non-empty URL to an OIDN ONNX model ' +
        '(use oidn_rt_hdr_alb_nrm.onnx when supplying albedo + normal aux); ' +
        "and (2) the 'onnxruntime-web' optional peer dependency installed in " +
        'the host. Omit the `denoiser` option to render without a final denoise.',
    );
  }
  return effectiveOpts;
}

/**
 * Validate a createEngine advanced bag without touching a canvas or GL context.
 * The exact same value-domain checks are used by the concrete backend factory.
 */
export function validateWebgl2AdvancedOptions(
  opts: Partial<Omit<PTEngineWebGL2Options, 'device'>>,
): void {
  validateAndResolveWebgl2Options(
    opts as PTEngineWebGL2Options,
    { deviceIndependent: true },
  );
}
