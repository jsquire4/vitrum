/**
 * inverse-scaffolding.ts — backend-agnostic inverse-rendering scaffolding.
 *
 * The two GPU backends (`pt-webgpu` via `inverse/optimizer.ts` +
 * `inverse/paramResolution.ts`, and `pt-webgl2` via
 * `inverse/finiteDifferenceSession.ts`) historically re-implemented the SAME
 * host-agnostic optimization scaffolding: a small-vector Adam optimizer, the
 * image-space L2/L1 loss, the `materials.<id>.<field>` / `emitters.<id>.<field>`
 * parameter-path parser, and the material/emitter field-metadata tables
 * (kind / length / clamp range / scene-read / scene-patch). Core already owns
 * the `createInverseSession` contract (`inverse.ts`), so the scaffolding lives
 * here as the single source of truth. Backends supply ONLY their gradient source
 * — finite-difference re-renders for pt-webgl2, the path-replay analytic adjoint
 * for pt-webgpu — which is the documented, correct FD-vs-adjoint capability
 * split and stays per-backend.
 *
 * The field metadata is a single `MATERIAL_PARAM_DESCRIPTORS` table (plus an
 * `EMITTER_PARAM_DESCRIPTORS` peer). Every consumer (kind classification,
 * clamp range, scene read, scene patch) is driven off these descriptors, so a
 * new optimizable field is added in exactly ONE place.
 *
 * Refs: Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015;
 *       Vicini, Speierer, Jakob, "Path Replay Backpropagation," SIGGRAPH 2021;
 *       Nimier-David, Vicini, Zeltner, Jakob, "Radiative Backpropagation,"
 *       SIGGRAPH 2020.
 */

import type {
  InverseGradientMethod,
  InverseLoss,
  InverseOptimizerConfig,
  InverseParam,
  InverseSessionOptions,
  InverseTargetImage,
} from './inverse.js';
import type { MaterialSpec, SceneEmitter, Vec2, Vec3 } from './scene/index.js';
import type { Scene, ScenePrimitive } from './scene/index.js';
import type { BackendSupportMode } from './engine/capabilities.js';

// ── image-space loss ─────────────────────────────────────────────────────────

/**
 * Read an interleaved image sample, mapping any non-finite value (NaN / ±Inf) to
 * 0. A path tracer legitimately produces firefly pixels that reach ±Inf once
 * encoded into the accumulation target, and `NaN` can appear from a degenerate
 * sample; neither is valid radiance. Without this guard a SINGLE bad pixel
 * poisons the whole mean image loss (`Inf - t = Inf`, `Inf*Inf = Inf`,
 * `Inf/N = Inf`, and any NaN propagates), which in turn NaNs the
 * finite-difference gradient and the Adam step — silently stalling the optimizer
 * at its initial value. Mapping to 0 keeps the loss finite and comparable across
 * probe renders (N is unchanged), so the gradient stays meaningful.
 */
function finiteSample(buf: Float32Array, i: number): number {
  const v = buf[i];
  return v !== undefined && Number.isFinite(v) ? v : 0;
}

/**
 * Mean per-pixel per-channel L2 (squared) loss between a rendered RGB image
 * (interleaved float, `renderChannels` per pixel) and the target. Both are read
 * over their first 3 channels (RGB); alpha is ignored. The two images must
 * share width·height; channel counts may differ (3 vs 4).
 *
 * Returns { loss, dLoss_dRendered } where `dLoss_dRendered` is the per-RGB-pixel
 * gradient of the loss (interleaved RGB, length width·height·3) — the adjoint
 * pass multiplies this into dRendered/dθ. For mean-squared error over N RGB
 * samples, dLoss/dRendered_i = 2·(rendered_i − target_i) / N.
 */
export function l2Loss(
  rendered: Float32Array,
  renderChannels: 3 | 4,
  target: InverseTargetImage,
): { loss: number; dLoss_dRendered: Float32Array } {
  const { width, height } = target;
  const targetChannels = target.channels ?? 3;
  const n = width * height;
  const N = n * 3; // RGB samples
  const dLoss = new Float32Array(N);
  let loss = 0;
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < 3; c++) {
      const r = finiteSample(rendered, p * renderChannels + c);
      const t = finiteSample(target.data, p * targetChannels + c);
      const diff = r - t;
      loss += diff * diff;
      dLoss[p * 3 + c] = (2 * diff) / N;
    }
  }
  return { loss: loss / N, dLoss_dRendered: dLoss };
}

/** Scalar-only loss (no gradient array allocated) — the finite-difference probe
 *  loop only needs the loss VALUE, so this avoids allocating a width·height·3
 *  gradient buffer per probe. `kind` selects squared (l2) vs absolute (l1). */
export function lossValue(
  rendered: Float32Array,
  renderChannels: 3 | 4,
  target: InverseTargetImage,
  kind: 'l2' | 'l1',
): number {
  const { width, height } = target;
  const targetChannels = target.channels ?? 3;
  const n = width * height;
  const N = n * 3;
  let loss = 0;
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < 3; c++) {
      const r = finiteSample(rendered, p * renderChannels + c);
      const t = finiteSample(target.data, p * targetChannels + c);
      const diff = r - t;
      loss += kind === 'l2' ? diff * diff : Math.abs(diff);
    }
  }
  return loss / N;
}

/** Mean per-pixel per-channel L1 (absolute) loss. Same return shape as
 *  {@link l2Loss}; dLoss/dRendered_i = sign(rendered_i − target_i) / N. */
export function l1Loss(
  rendered: Float32Array,
  renderChannels: 3 | 4,
  target: InverseTargetImage,
): { loss: number; dLoss_dRendered: Float32Array } {
  const { width, height } = target;
  const targetChannels = target.channels ?? 3;
  const n = width * height;
  const N = n * 3;
  const dLoss = new Float32Array(N);
  let loss = 0;
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < 3; c++) {
      const r = finiteSample(rendered, p * renderChannels + c);
      const t = finiteSample(target.data, p * targetChannels + c);
      const diff = r - t;
      loss += Math.abs(diff);
      dLoss[p * 3 + c] = Math.sign(diff) / N;
    }
  }
  return { loss: loss / N, dLoss_dRendered: dLoss };
}

// ── Adam optimizer (small flat vector) ───────────────────────────────────────

export interface AdamConfig {
  readonly learningRate: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly epsilon: number;
}

/** Copy of Adam's mutable state, used to roll back a failed scene update. */
export interface AdamSnapshot {
  readonly t: number;
  readonly m: Float32Array;
  readonly v: Float32Array;
}

export const DEFAULT_ADAM: AdamConfig = {
  learningRate: 1e-2,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
};

/** A flat Adam state over a fixed-length parameter vector. One instance per
 *  session; `step()` mutates the supplied `params` in place. */
export class Adam {
  readonly #cfg: AdamConfig;
  readonly #m: Float32Array;
  readonly #v: Float32Array;
  #t = 0;

  constructor(length: number, cfg: AdamConfig = DEFAULT_ADAM) {
    if (!Number.isSafeInteger(length) || length <= 0) {
      throw new Error('Adam: length must be a positive safe integer.');
    }
    validateAdamConfig(cfg, 'Adam');
    this.#cfg = { ...cfg };
    this.#m = new Float32Array(length);
    this.#v = new Float32Array(length);
  }

  /** Apply one Adam update: params ← params − lr·m̂/(√v̂+ε). Mutates `params`. */
  step(params: Float32Array, grad: Float32Array): void {
    if (params.length !== this.#m.length || grad.length !== this.#m.length) {
      throw new Error(
        `Adam.step: params and gradient must both have length ${this.#m.length}.`,
      );
    }
    assertFiniteArray(params, 'Adam.step params');
    assertFiniteArray(grad, 'Adam.step gradient');
    const nextT = this.#t + 1;
    if (!Number.isSafeInteger(nextT)) {
      throw new Error('Adam.step: optimizer step counter overflowed.');
    }
    const { learningRate, beta1, beta2, epsilon } = this.#cfg;
    const bc1 = 1 - Math.pow(beta1, nextT);
    const bc2 = 1 - Math.pow(beta2, nextT);
    const nextM = new Float32Array(this.#m.length);
    const nextV = new Float32Array(this.#v.length);
    const nextParams = new Float32Array(params.length);
    for (let i = 0; i < params.length; i++) {
      const g = grad[i]!;
      nextM[i] = beta1 * this.#m[i]! + (1 - beta1) * g;
      nextV[i] = beta2 * this.#v[i]! + (1 - beta2) * g * g;
      const mHat = nextM[i]! / bc1;
      const vHat = nextV[i]! / bc2;
      nextParams[i] = params[i]! - (learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
    }
    assertFiniteArray(nextM, 'Adam.step first moment');
    assertFiniteArray(nextV, 'Adam.step second moment');
    assertFiniteArray(nextParams, 'Adam.step result');
    this.#t = nextT;
    this.#m.set(nextM);
    this.#v.set(nextV);
    params.set(nextParams);
  }

  snapshot(): AdamSnapshot {
    return { t: this.#t, m: this.#m.slice(), v: this.#v.slice() };
  }

  restore(snapshot: AdamSnapshot): void {
    if (
      !Number.isSafeInteger(snapshot.t) ||
      snapshot.t < 0 ||
      snapshot.m.length !== this.#m.length ||
      snapshot.v.length !== this.#v.length
    ) {
      throw new Error('Adam.restore: incompatible snapshot.');
    }
    assertFiniteArray(snapshot.m, 'Adam.restore first moment');
    assertFiniteArray(snapshot.v, 'Adam.restore second moment');
    this.#t = snapshot.t;
    this.#m.set(snapshot.m);
    this.#v.set(snapshot.v);
  }
}

// ── parameter packing / resolution ────────────────────────────────────────────

/** Components for an {@link InverseParam} kind. scalar → 1, vec2 → 2, rgb → 3. */
export function paramLength(p: InverseParam, backend: string): number {
  switch (p.kind) {
    case 'scalar': return 1;
    case 'vec2': return 2;
    case 'rgb': return 3;
    default: {
      const _exhaustive: never = p.kind;
      throw new Error(
        `inverse: unknown parameter kind ${String(_exhaustive)} in ${backend} ` +
          `for path "${p.path}".`,
      );
    }
  }
}

/** A resolved scene address: which target object + field a parameter touches. */
export interface ResolvedParamTarget {
  readonly domain: 'materials' | 'emitters';
  readonly id: string;
  readonly field: string;
}

/** Parse a dotted parameter path (`materials.<id>.<field>` /
 *  `emitters.<id>.<field>`). The id may itself contain dots (parsed greedily:
 *  the last segment is the field, the first is the domain, the middle is the id).
 *  Throws on an unrecognised domain or a malformed path. */
export function parseParamPath(path: string): ResolvedParamTarget {
  if (typeof path !== 'string' || path.length === 0) {
    throw new Error('inverse: parameter path must be a non-empty string.');
  }
  const segments = path.split('.');
  if (segments.length < 3) {
    throw new Error(
      `inverse: parameter path "${path}" must be ` +
        '"materials.<id>.<field>" or "emitters.<id>.<field>".',
    );
  }
  const domain = segments[0];
  const field = segments[segments.length - 1]!;
  const id = segments.slice(1, segments.length - 1).join('.');
  if (domain !== 'materials' && domain !== 'emitters') {
    throw new Error(
      `inverse: parameter path "${path}" has unknown domain "${domain}" ` +
        '(expected "materials" or "emitters").',
    );
  }
  if (id.length === 0 || field.length === 0) {
    throw new Error(
      `inverse: parameter path "${path}" must contain a non-empty id and field.`,
    );
  }
  return { domain, id, field };
}

export const MAX_INVERSE_SAMPLES_PER_STEP = 4096;

const INVERSE_SESSION_OPTION_KEYS = {
  target: true,
  parameters: true,
  loss: true,
  method: true,
  samplesPerStep: true,
  optimizer: true,
  onDiagnostic: true,
} as const satisfies Readonly<Record<keyof InverseSessionOptions, true>>;

const INVERSE_TARGET_KEYS = {
  data: true,
  width: true,
  height: true,
  channels: true,
} as const satisfies Readonly<Record<keyof InverseTargetImage, true>>;

const INVERSE_PARAM_KEYS = {
  path: true,
  kind: true,
  initial: true,
  min: true,
  max: true,
} as const satisfies Readonly<Record<keyof InverseParam, true>>;

const INVERSE_OPTIMIZER_KEYS = {
  learningRate: true,
  beta1: true,
  beta2: true,
  epsilon: true,
  fdEpsilon: true,
} as const satisfies Readonly<Record<keyof InverseOptimizerConfig, true>>;

function assertPlainDataRecord(
  value: unknown,
  label: string,
  allowedKeys: ReadonlySet<string>,
): asserts value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`createInverseSession: ${label} must be a plain object.`);
  }
  const prototype: unknown = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new Error(
      `createInverseSession: ${label} must have Object.prototype or null prototype.`,
    );
  }
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string') {
      throw new Error(
        `createInverseSession: ${label} contains unsupported symbol key ${String(key)}.`,
      );
    }
    if (!allowedKeys.has(key)) {
      throw new Error(
        `createInverseSession: ${label} contains unknown key ${JSON.stringify(key)}.`,
      );
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor) || !descriptor.enumerable) {
      throw new Error(
        `createInverseSession: ${label}.${key} must be an enumerable own data property.`,
      );
    }
  }
}

function assertDenseDataArray(
  value: unknown,
  label: string,
): asserts value is readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`createInverseSession: ${label} must be an array.`);
  }
  for (const key of Reflect.ownKeys(value)) {
    if (key === 'length') continue;
    if (typeof key !== 'string' || !/^(?:0|[1-9][0-9]*)$/.test(key)) {
      throw new Error(
        `createInverseSession: ${label} contains unsupported property ${String(key)}.`,
      );
    }
    const index = Number(key);
    if (!Number.isSafeInteger(index) || index >= value.length) {
      throw new Error(`createInverseSession: ${label} contains invalid index ${key}.`);
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor) || !descriptor.enumerable) {
      throw new Error(
        `createInverseSession: ${label}[${key}] must be an enumerable own data property.`,
      );
    }
  }
  for (let index = 0; index < value.length; index += 1) {
    if (!Object.prototype.hasOwnProperty.call(value, index)) {
      throw new Error(`createInverseSession: ${label} must be dense (hole at index ${index}).`);
    }
  }
}

export interface ValidatedInverseSessionOptions {
  readonly target: InverseTargetImage;
  readonly parameters: readonly InverseParam[];
  readonly loss: InverseLoss;
  readonly method: InverseGradientMethod;
  readonly samplesPerStep: number;
  readonly optimizer: AdamConfig & { readonly fdEpsilon: number };
  readonly onDiagnostic?: InverseSessionOptions['onDiagnostic'];
}

function assertFinitePositive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`createInverseSession: ${label} must be finite and greater than zero.`);
  }
}

function validateAdamConfig(cfg: AdamConfig, prefix: string): void {
  if (!Number.isFinite(cfg.learningRate) || cfg.learningRate <= 0) {
    throw new Error(`${prefix}: learningRate must be finite and greater than zero.`);
  }
  if (!Number.isFinite(cfg.beta1) || cfg.beta1 < 0 || cfg.beta1 >= 1) {
    throw new Error(`${prefix}: beta1 must be finite and in [0, 1).`);
  }
  if (!Number.isFinite(cfg.beta2) || cfg.beta2 < 0 || cfg.beta2 >= 1) {
    throw new Error(`${prefix}: beta2 must be finite and in [0, 1).`);
  }
  if (!Number.isFinite(cfg.epsilon) || cfg.epsilon <= 0) {
    throw new Error(`${prefix}: epsilon must be finite and greater than zero.`);
  }
}

/** Validate and defensively copy all host-owned session configuration before a
 * backend mutates the scene. */
export function validateInverseSessionOptions(
  opts: InverseSessionOptions,
  backend: string,
): ValidatedInverseSessionOptions {
  assertPlainDataRecord(
    opts,
    'options',
    new Set(Object.keys(INVERSE_SESSION_OPTION_KEYS)),
  );
  assertDenseDataArray(opts.parameters, 'options.parameters');
  if (opts.parameters.length === 0) {
    throw new Error('createInverseSession: at least one parameter is required.');
  }
  const { target } = opts;
  assertPlainDataRecord(
    target,
    'options.target',
    new Set(Object.keys(INVERSE_TARGET_KEYS)),
  );
  if (
    !Number.isSafeInteger(target.width) ||
    target.width <= 0 ||
    !Number.isSafeInteger(target.height) ||
    target.height <= 0
  ) {
    throw new Error(
      'createInverseSession: target width and height must be positive safe integers.',
    );
  }
  const channels = target.channels ?? 3;
  if (channels !== 3 && channels !== 4) {
    throw new Error('createInverseSession: target channels must be exactly 3 or 4.');
  }
  if (!(target.data instanceof Float32Array)) {
    throw new Error('createInverseSession: target data must be a Float32Array.');
  }
  const expectedLength = target.width * target.height * channels;
  if (!Number.isSafeInteger(expectedLength) || target.data.length !== expectedLength) {
    throw new Error(
      `createInverseSession: target data length ${target.data.length} does not match ` +
        `${target.width}×${target.height}×${channels} (${expectedLength}).`,
    );
  }
  assertFiniteArray(target.data, 'createInverseSession target data');

  const loss = opts.loss ?? 'l2';
  if (loss !== 'l1' && loss !== 'l2') {
    throw new Error(`createInverseSession: unsupported loss "${String(loss)}".`);
  }
  const method = opts.method ?? 'finite-difference';
  if (method !== 'finite-difference' && method !== 'path-replay') {
    throw new Error(
      `createInverseSession: unsupported gradient method "${String(method)}".`,
    );
  }
  const samplesPerStep = opts.samplesPerStep ?? 8;
  if (
    !Number.isSafeInteger(samplesPerStep) ||
    samplesPerStep < 1 ||
    samplesPerStep > MAX_INVERSE_SAMPLES_PER_STEP
  ) {
    throw new Error(
      `createInverseSession: samplesPerStep must be an integer in [1, ${MAX_INVERSE_SAMPLES_PER_STEP}].`,
    );
  }
  if (opts.onDiagnostic !== undefined && typeof opts.onDiagnostic !== 'function') {
    throw new Error(
      'createInverseSession: onDiagnostic must be a function when supplied.',
    );
  }
  if (opts.optimizer !== undefined) {
    assertPlainDataRecord(
      opts.optimizer,
      'options.optimizer',
      new Set(Object.keys(INVERSE_OPTIMIZER_KEYS)),
    );
  }
  const optimizerInput: InverseOptimizerConfig = opts.optimizer ?? {};
  const optimizer = {
    learningRate: optimizerInput.learningRate ?? DEFAULT_ADAM.learningRate,
    beta1: optimizerInput.beta1 ?? DEFAULT_ADAM.beta1,
    beta2: optimizerInput.beta2 ?? DEFAULT_ADAM.beta2,
    epsilon: optimizerInput.epsilon ?? DEFAULT_ADAM.epsilon,
    fdEpsilon: optimizerInput.fdEpsilon ?? 1e-3,
  };
  validateAdamConfig(optimizer, 'createInverseSession');
  assertFinitePositive(optimizer.fdEpsilon, 'optimizer.fdEpsilon');

  const seenTargets = new Set<string>();
  const rawParameters: readonly unknown[] = opts.parameters;
  const parameters = rawParameters.map((rawSource, index): InverseParam => {
    assertPlainDataRecord(
      rawSource,
      `options.parameters[${index}]`,
      new Set(Object.keys(INVERSE_PARAM_KEYS)),
    );
    const record = rawSource;
    if (typeof record.path !== 'string') {
      throw new Error(`createInverseSession: parameter ${index} path must be a string.`);
    }
    const path = record.path;
    const rawKind = record.kind;
    if (rawKind !== 'scalar' && rawKind !== 'vec2' && rawKind !== 'rgb') {
      throw new Error(
        `createInverseSession: parameter "${path}" has unsupported kind "${String(rawKind)}" in ${backend}.`,
      );
    }
    const min = record.min;
    if (min !== undefined && (typeof min !== 'number' || !Number.isFinite(min))) {
      throw new Error(
        `createInverseSession: parameter "${path}" min must be finite when provided.`,
      );
    }
    const max = record.max;
    if (max !== undefined && (typeof max !== 'number' || !Number.isFinite(max))) {
      throw new Error(
        `createInverseSession: parameter "${path}" max must be finite when provided.`,
      );
    }
    if (typeof min === 'number' && typeof max === 'number' && min > max) {
      throw new Error(
        `createInverseSession: parameter "${path}" min must not exceed max.`,
      );
    }
    const rawInitial = record.initial;
    let initial: number[] | undefined;
    if (rawInitial !== undefined) {
      assertDenseDataArray(rawInitial, `parameter "${path}" initial`);
      const components: readonly unknown[] = rawInitial;
      initial = components.map((component) => {
        if (typeof component !== 'number' || !Number.isFinite(component)) {
          throw new Error(
            `createInverseSession: parameter "${path}" initial value must be finite.`,
          );
        }
        return component;
      });
    }
    const source: InverseParam = {
      path,
      kind: rawKind,
      ...(initial != null ? { initial } : {}),
      ...(typeof min === 'number' ? { min } : {}),
      ...(typeof max === 'number' ? { max } : {}),
    };
    const address = parseParamPath(path);
    const key = `${address.domain}\u0000${address.id}\u0000${address.field}`;
    if (seenTargets.has(key)) {
      throw new Error(
        `createInverseSession: duplicate or overlapping parameter path "${path}".`,
      );
    }
    seenTargets.add(key);
    paramLength(source, backend);
    if (initial != null) {
      const expected = paramLength(source, backend);
      if (initial.length !== expected) {
        throw new Error(
          `createInverseSession: parameter "${path}" initial value has length ` +
            `${initial.length}, expected ${expected}.`,
        );
      }
    }
    return source;
  });

  return {
    target: {
      data: target.data.slice(),
      width: target.width,
      height: target.height,
      channels,
    },
    parameters,
    loss,
    method,
    samplesPerStep,
    optimizer,
    ...(opts.onDiagnostic != null ? { onDiagnostic: opts.onDiagnostic } : {}),
  };
}

/** Enforce the backend readback ABI before indexing it in a loss function. */
export function validateInverseReadback(
  data: Float32Array,
  channels: number,
  width: number,
  height: number,
  label: string,
): asserts channels is 3 | 4 {
  if (!(data instanceof Float32Array)) {
    throw new Error(`InverseSession.step: ${label} data must be a Float32Array.`);
  }
  if (channels !== 3 && channels !== 4) {
    throw new Error(`InverseSession.step: ${label} channels must be exactly 3 or 4.`);
  }
  if (
    !Number.isSafeInteger(width) ||
    width <= 0 ||
    !Number.isSafeInteger(height) ||
    height <= 0
  ) {
    throw new Error(
      `InverseSession.step: ${label} width and height must be positive safe integers.`,
    );
  }
  const expected = width * height * channels;
  if (!Number.isSafeInteger(expected) || data.length !== expected) {
    throw new Error(
      `InverseSession.step: ${label} data length ${data.length} does not match expected ${expected}.`,
    );
  }
  assertFiniteArray(data, `InverseSession.step ${label}`);
}

export function assertFiniteNumber(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`InverseSession.step: ${label} must be finite.`);
  }
}

export function assertFiniteArray(values: Float32Array, label: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`${label} contains a non-finite value at index ${i}.`);
    }
  }
}

/** Convert arbitrary hook throws/rejections into a stable Error surface. */
export function normalizeInverseError(value: unknown, operation: string): Error {
  if (value instanceof Error) return value;
  let rendered: string;
  try {
    const json = JSON.stringify(value);
    rendered = json === undefined ? String(value) : json;
  } catch {
    rendered = String(value);
  }
  return new Error(
    `${operation}: hook threw or rejected with a non-Error value (${rendered}).`,
  );
}

/** Invoke a synchronous engine hook through the normalized error boundary. */
export function invokeInverseHook<T>(operation: string, hook: () => T): T {
  try {
    return hook();
  } catch (error) {
    throw normalizeInverseError(error, operation);
  }
}

// ── field-metadata descriptor tables (single source of truth) ─────────────────

/** The continuity class a param descriptor maps to. */
export type ParamFieldKind = 'scalar' | 'vec2' | 'rgb';

/**
 * One optimizable material/emitter field, driving ALL four consumer operations
 * (kind classification, default clamp range, scene read, scene patch) off one
 * row. `read` returns the current scene value; `patch` builds the incremental
 * update record. This is the single MATERIAL_PARAM_DESCRIPTORS-style table both
 * backends consume; adding a field is a one-line addition here.
 */
export interface MaterialParamDescriptor {
  readonly kind: ParamFieldKind;
  /** Default [min, max] clamp applied after an optimizer step when the param
   *  supplies no explicit min/max. */
  readonly clamp: readonly [number, number];
  readonly read: (m: MaterialSpec) => number[];
  readonly patch: (value: readonly number[]) => Partial<MaterialSpec>;
}

/** Emitter-field peer of {@link MaterialParamDescriptor}. */
export interface EmitterParamDescriptor {
  readonly kind: ParamFieldKind;
  readonly clamp: readonly [number, number];
  readonly read: (e: SceneEmitter) => number[];
  readonly patch: (value: readonly number[]) => Partial<SceneEmitter>;
}

const INF = Infinity;

function exactPatchComponents(
  value: readonly number[],
  length: number,
  label: string,
): readonly number[] {
  if (value.length !== length) {
    throw new Error(
      `inverse: ${label} patch requires exactly ${length} component${length === 1 ? '' : 's'} ` +
        `(got ${value.length}).`,
    );
  }
  for (let index = 0; index < value.length; index += 1) {
    if (!Number.isFinite(value[index])) {
      throw new Error(`inverse: ${label} patch component ${index} must be finite.`);
    }
  }
  return value;
}

function scalar(value: readonly number[]): number {
  return exactPatchComponents(value, 1, 'scalar')[0]!;
}

function vec2(value: readonly number[]): Vec2 {
  const exact = exactPatchComponents(value, 2, 'vec2');
  return [exact[0]!, exact[1]!] as unknown as Vec2;
}

function vec3(value: readonly number[]): Vec3 {
  const exact = exactPatchComponents(value, 3, 'rgb');
  return [exact[0]!, exact[1]!, exact[2]!] as unknown as Vec3;
}

export const MATERIAL_PARAM_DESCRIPTORS: Readonly<Record<string, MaterialParamDescriptor>> = {
  baseColor: {
    kind: 'rgb', clamp: [0, 1],
    read: (m) => [...m.baseColor],
    patch: (v) => ({ baseColor: vec3(v) }),
  },
  roughness: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.roughness],
    patch: (v) => ({ roughness: scalar(v) }),
  },
  metallic: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.metallic],
    patch: (v) => ({ metallic: scalar(v) }),
  },
  emissive: {
    kind: 'rgb', clamp: [0, INF],
    read: (m) => [...(m.emissive ?? [0, 0, 0])],
    patch: (v) => ({ emissive: vec3(v) }),
  },
  emissiveIntensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.emissiveIntensity ?? 1],
    patch: (v) => ({ emissiveIntensity: scalar(v) }),
  },
  opacity: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.opacity ?? 1],
    patch: (v) => ({ opacity: scalar(v) }),
  },
  alphaCutoff: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.alphaCutoff ?? 0.5],
    patch: (v) => ({ alphaCutoff: scalar(v) }),
  },
  ior: {
    kind: 'scalar', clamp: [1, 2.5],
    read: (m) => [m.ior ?? 1.5],
    patch: (v) => ({ ior: scalar(v) }),
  },
  transmission: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.transmission ?? 0],
    patch: (v) => ({ transmission: scalar(v) }),
  },
  thickness: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.thickness ?? 0],
    patch: (v) => ({ thickness: scalar(v) }),
  },
  attenuationColor: {
    kind: 'rgb', clamp: [1e-4, 1],
    read: (m) => [...(m.attenuationColor ?? [1, 1, 1])],
    patch: (v) => ({ attenuationColor: vec3(v) }),
  },
  attenuationDistance: {
    kind: 'scalar', clamp: [1e-6, INF],
    read: (m) => [m.attenuationDistance ?? Number.POSITIVE_INFINITY],
    patch: (v) => ({ attenuationDistance: scalar(v) }),
  },
  dispersionAbbeNumber: {
    kind: 'scalar', clamp: [1e-6, INF],
    read: (m) => [m.dispersionAbbeNumber ?? 0],
    patch: (v) => ({ dispersionAbbeNumber: scalar(v) }),
  },
  scatteringCoefficient: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.scatteringCoefficient ?? 0],
    patch: (v) => ({ scatteringCoefficient: scalar(v) }),
  },
  scatteringAnisotropy: {
    kind: 'scalar', clamp: [-0.95, 0.95],
    read: (m) => [m.scatteringAnisotropy ?? 0],
    patch: (v) => ({ scatteringAnisotropy: scalar(v) }),
  },
  scatteringCoefficientRGB: {
    kind: 'rgb', clamp: [0, INF],
    read: (m) => [...(m.scatteringCoefficientRGB ?? [0, 0, 0])],
    patch: (v) => ({ scatteringCoefficientRGB: vec3(v) }),
  },
  specularColor: {
    kind: 'rgb', clamp: [0, 1],
    read: (m) => [...(m.specularColor ?? [1, 1, 1])],
    patch: (v) => ({ specularColor: vec3(v) }),
  },
  specularIntensity: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.specularIntensity ?? 1],
    patch: (v) => ({ specularIntensity: scalar(v) }),
  },
  clearcoat: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.clearcoat ?? 0],
    patch: (v) => ({ clearcoat: scalar(v) }),
  },
  clearcoatRoughness: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.clearcoatRoughness ?? 0],
    patch: (v) => ({ clearcoatRoughness: scalar(v) }),
  },
  sheen: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.sheen ?? 0],
    patch: (v) => ({ sheen: scalar(v) }),
  },
  sheenColor: {
    kind: 'rgb', clamp: [0, 1],
    read: (m) => [...(m.sheenColor ?? [1, 1, 1])],
    patch: (v) => ({ sheenColor: vec3(v) }),
  },
  sheenRoughness: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.sheenRoughness ?? 0],
    patch: (v) => ({ sheenRoughness: scalar(v) }),
  },
  iridescence: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.iridescence ?? 0],
    patch: (v) => ({ iridescence: scalar(v) }),
  },
  iridescenceIor: {
    kind: 'scalar', clamp: [1, 3],
    read: (m) => [m.iridescenceIor ?? 1.3],
    patch: (v) => ({ iridescenceIor: scalar(v) }),
  },
  iridescenceThicknessRange: {
    kind: 'vec2', clamp: [0, INF],
    read: (m) => [...(m.iridescenceThicknessRange ?? [100, 400])],
    patch: (v) => ({ iridescenceThicknessRange: vec2(v) }),
  },
  anisotropy: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.anisotropy ?? 0],
    patch: (v) => ({ anisotropy: scalar(v) }),
  },
  anisotropyRotation: {
    kind: 'scalar', clamp: [-INF, INF],
    read: (m) => [m.anisotropyRotation ?? 0],
    patch: (v) => ({ anisotropyRotation: scalar(v) }),
  },
  normalScale: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.normalScale ?? 1],
    patch: (v) => ({ normalScale: scalar(v) }),
  },
  bumpScale: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.bumpScale ?? 1],
    patch: (v) => ({ bumpScale: scalar(v) }),
  },
  clearcoatNormalScale: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.clearcoatNormalScale ?? 1],
    patch: (v) => ({ clearcoatNormalScale: scalar(v) }),
  },
  aoMapIntensity: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.aoMapIntensity ?? 1],
    patch: (v) => ({ aoMapIntensity: scalar(v) }),
  },
  lightMapIntensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.lightMapIntensity ?? 1],
    patch: (v) => ({ lightMapIntensity: scalar(v) }),
  },
  envMapIntensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.envMapIntensity ?? 1],
    patch: (v) => ({ envMapIntensity: scalar(v) }),
  },
  displacementScale: {
    kind: 'scalar', clamp: [-INF, INF],
    read: (m) => [m.displacementScale ?? 1],
    patch: (v) => ({ displacementScale: scalar(v) }),
  },
  displacementBias: {
    kind: 'scalar', clamp: [-INF, INF],
    read: (m) => [m.displacementBias ?? 0],
    patch: (v) => ({ displacementBias: scalar(v) }),
  },
};

export const EMITTER_PARAM_DESCRIPTORS: Readonly<Record<string, EmitterParamDescriptor>> = {
  color: {
    kind: 'rgb', clamp: [0, INF],
    read: (e) => [...e.color],
    patch: (v) => ({ color: vec3(v) }),
  },
  intensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (e) => [e.intensity],
    patch: (v) => ({ intensity: scalar(v) }),
  },
};

/** RGB material fields (derived from the descriptor table). */
export const MATERIAL_RGB_FIELDS: ReadonlySet<string> = new Set(
  Object.keys(MATERIAL_PARAM_DESCRIPTORS).filter((f) => MATERIAL_PARAM_DESCRIPTORS[f]!.kind === 'rgb'),
);
export const MATERIAL_VEC2_FIELDS: ReadonlySet<string> = new Set(
  Object.keys(MATERIAL_PARAM_DESCRIPTORS).filter((f) => MATERIAL_PARAM_DESCRIPTORS[f]!.kind === 'vec2'),
);
export const MATERIAL_SCALAR_FIELDS: ReadonlySet<string> = new Set(
  Object.keys(MATERIAL_PARAM_DESCRIPTORS).filter((f) => MATERIAL_PARAM_DESCRIPTORS[f]!.kind === 'scalar'),
);
export const EMITTER_RGB_FIELDS: ReadonlySet<string> = new Set(
  Object.keys(EMITTER_PARAM_DESCRIPTORS).filter((f) => EMITTER_PARAM_DESCRIPTORS[f]!.kind === 'rgb'),
);
export const EMITTER_SCALAR_FIELDS: ReadonlySet<string> = new Set(
  Object.keys(EMITTER_PARAM_DESCRIPTORS).filter((f) => EMITTER_PARAM_DESCRIPTORS[f]!.kind === 'scalar'),
);

/** Field-aware default [min, max] clamp range, used when a parameter doesn't
 *  supply its own `min`/`max`. Resolved off the descriptor table; unknown fields
 *  fall back to [0, Infinity] (non-negative, unbounded above). */
export function defaultClampRange(field: string): [number, number] {
  const m = MATERIAL_PARAM_DESCRIPTORS[field];
  if (m) return [m.clamp[0], m.clamp[1]];
  const e = EMITTER_PARAM_DESCRIPTORS[field];
  if (e) return [e.clamp[0], e.clamp[1]];
  return [0, Infinity];
}

// ── flat-vector layout + clamping ──────────────────────────────────────────────

/** A flat-vector slot: where the parameter lives + the FIELD-aware default
 *  clamp range to use when the param doesn't supply its own min/max. */
export interface ParamLayoutEntry {
  readonly offset: number;
  readonly length: number;
  /** Default lower clamp (overridden by `InverseParam.min`). */
  readonly defaultMin: number;
  /** Default upper clamp (overridden by `InverseParam.max`). */
  readonly defaultMax: number;
}

/** Clamp every component of a flat parameter vector to its per-parameter
 *  [min, max] (applied after an optimizer step). The caller supplies a
 *  FIELD-aware default range per slot — a bare param kind cannot distinguish a
 *  roughness scalar (saturates at 1) from an emitter-intensity scalar
 *  (unbounded above), so the default range is resolved at the call site that
 *  knows the field, not guessed from the kind here. */
export function clampParams(
  flat: Float32Array,
  params: readonly InverseParam[],
  layout: readonly ParamLayoutEntry[],
): void {
  for (let i = 0; i < params.length; i++) {
    const p = params[i]!;
    const { offset, length, defaultMin, defaultMax } = layout[i]!;
    const lo = p.min ?? defaultMin;
    const hi = p.max ?? defaultMax;
    for (let c = 0; c < length; c++) {
      flat[offset + c] = Math.min(Math.max(flat[offset + c]!, lo), hi);
    }
  }
}

// ── validation + scene read/patch (descriptor-driven) ──────────────────────────

/** One resolved optimized parameter slot in the flat parameter vector. */
export interface ParamSlot {
  readonly param: InverseParam;
  readonly target: ResolvedParamTarget;
  readonly offset: number;
  readonly length: number;
}

export function findPrimitive(scene: Scene, id: string): ScenePrimitive | null {
  return scene.primitives.find((p) => p.id === id) ?? null;
}

function assertKind(param: InverseParam, expected: ParamFieldKind, backend: string): void {
  if (param.kind !== expected) {
    throw new Error(
      `createInverseSession: parameter "${param.path}" is declared kind '${param.kind}' ` +
        `but the resolved field is '${expected}'.`,
    );
  }
  void backend;
}

/**
 * Validate a resolved parameter against the live scene + descriptor table.
 * `opts.materialSupportDetails` / `opts.emitterSupportDetails` are an OPTIONAL
 * per-backend capability gate — a backend (pt-webgpu) whose active runtime
 * profile reports a field/emitter kind as `unsupported` rejects it here; a
 * backend that passes no support details (pt-webgl2) applies no such gate. This
 * is the per-backend availability flag the shared table carries.
 */
export function validateParam(
  scene: Scene,
  param: InverseParam,
  target: ResolvedParamTarget,
  opts: {
    backend: string;
    materialSupportDetails?: Readonly<Partial<Record<string, BackendSupportMode>>>;
    emitterSupportDetails?: Readonly<Partial<Record<string, BackendSupportMode>>>;
  },
): void {
  const { backend } = opts;
  if (target.domain === 'materials') {
    const prim = findPrimitive(scene, target.id);
    if (prim == null) {
      throw new Error(
        `createInverseSession: no primitive with id "${target.id}" for path "${param.path}".`,
      );
    }
    const isRgb = MATERIAL_RGB_FIELDS.has(target.field);
    const isVec2 = MATERIAL_VEC2_FIELDS.has(target.field);
    const isScalar = MATERIAL_SCALAR_FIELDS.has(target.field);
    if (!isRgb && !isVec2 && !isScalar) {
      throw new Error(
        `createInverseSession: material field "${target.field}" (path "${param.path}") is not ` +
          `optimizable. Supported: ${[
            ...MATERIAL_RGB_FIELDS,
            ...MATERIAL_VEC2_FIELDS,
            ...MATERIAL_SCALAR_FIELDS,
          ].join(', ')}.`,
      );
    }
    if (opts.materialSupportDetails?.[target.field] === 'unsupported') {
      throw new Error(
        `createInverseSession: material field "${target.field}" (path "${param.path}") is not ` +
          `optimizable on the active ${backend} runtime profile because that profile reports ` +
          'the field as unsupported.',
      );
    }
    assertKind(param, isRgb ? 'rgb' : isVec2 ? 'vec2' : 'scalar', backend);
  } else {
    const emitter = scene.emitters.find((e) => e.id === target.id);
    if (emitter == null) {
      throw new Error(
        `createInverseSession: no emitter with id "${target.id}" for path "${param.path}".`,
      );
    }
    if (opts.emitterSupportDetails?.[emitter.kind] === 'unsupported') {
      throw new Error(
        `createInverseSession: emitter kind "${emitter.kind}" (path "${param.path}") is not ` +
          `optimizable on the active ${backend} runtime profile because that profile reports ` +
          'the emitter kind as unsupported.',
      );
    }
    const isRgb = EMITTER_RGB_FIELDS.has(target.field);
    const isScalar = EMITTER_SCALAR_FIELDS.has(target.field);
    if (!isRgb && !isScalar) {
      throw new Error(
        `createInverseSession: emitter field "${target.field}" (path "${param.path}") is not ` +
          `optimizable. Supported: ${[...EMITTER_RGB_FIELDS, ...EMITTER_SCALAR_FIELDS].join(', ')}.`,
      );
    }
    assertKind(param, isRgb ? 'rgb' : 'scalar', backend);
  }
}

export function validateInitialSceneValue(
  slot: ParamSlot,
  value: readonly number[],
  fromExplicitInitial: boolean,
  backend: string,
): void {
  if (slot.target.domain === 'materials' && slot.target.field === 'attenuationDistance') {
    const distance = value[0];
    if (Number.isFinite(distance) && distance! > 0) return;
    const source = fromExplicitInitial ? 'initial' : 'scene';
    throw new Error(
      `createInverseSession: parameter "${slot.param.path}" requires a finite positive ${source} ` +
        'attenuationDistance. Undefined or Infinity means "no finite absorbing medium" in the ' +
        `renderer, so ${backend} cannot forward-difference this parameter without an explicit ` +
        'finite seed. Set parameter.initial to start fitting a finite medium.',
    );
  }
  if (slot.target.domain === 'materials' && slot.target.field === 'dispersionAbbeNumber') {
    const abbeNumber = value[0];
    if (Number.isFinite(abbeNumber) && abbeNumber! > 0) return;
    const source = fromExplicitInitial ? 'initial' : 'scene';
    throw new Error(
      `createInverseSession: parameter "${slot.param.path}" requires a finite positive ${source} ` +
        'dispersionAbbeNumber. Undefined means "dispersion disabled" in the renderer, so ' +
        `${backend} needs an explicit positive parameter.initial before fitting dispersion.`,
    );
  }
  for (const component of value) {
    if (!Number.isFinite(component)) {
      throw new Error(
        `createInverseSession: parameter "${slot.param.path}" initial value must be finite.`,
      );
    }
  }
}

export function readSceneValue(scene: Scene, target: ResolvedParamTarget, length: number): number[] {
  if (target.domain === 'materials') {
    const prim = findPrimitive(scene, target.id)!;
    const descriptor = MATERIAL_PARAM_DESCRIPTORS[target.field];
    if (descriptor) {
      const value = descriptor.read(prim.material);
      if (value.length !== length) {
        throw new Error(
          `inverse: material field "${target.field}" returned ${value.length} components; ` +
            `the resolved parameter layout requires ${length}.`,
        );
      }
      return value;
    }
  } else {
    const e = scene.emitters.find((em) => em.id === target.id)!;
    const descriptor = EMITTER_PARAM_DESCRIPTORS[target.field];
    if (descriptor) {
      const value = descriptor.read(e);
      if (value.length !== length) {
        throw new Error(
          `inverse: emitter field "${target.field}" returned ${value.length} components; ` +
            `the resolved parameter layout requires ${length}.`,
        );
      }
      return value;
    }
  }
  throw new Error(
    `inverse: resolved ${target.domain} field "${target.field}" has no descriptor.`,
  );
}

export function materialPatch(field: string, value: readonly number[]): Partial<MaterialSpec> {
  const descriptor = MATERIAL_PARAM_DESCRIPTORS[field];
  if (!descriptor) throw new Error(`inverse: unsupported material field "${field}".`);
  return descriptor.patch(value);
}

export function emitterPatch(field: string, value: readonly number[]): Partial<SceneEmitter> {
  const descriptor = EMITTER_PARAM_DESCRIPTORS[field];
  if (!descriptor) throw new Error(`inverse: unsupported emitter field "${field}".`);
  return descriptor.patch(value);
}
