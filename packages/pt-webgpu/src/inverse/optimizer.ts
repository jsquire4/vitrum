/**
 * optimizer.ts — pure CPU helpers for the inverse-rendering loop (WS5).
 *
 * Backend-agnostic, GPU-free, and fully unit-testable: image-space loss, a
 * small-vector Adam optimizer (Kingma & Ba 2015), and parameter-path resolution
 * against a vitrum Scene. The GPU session (`PtWebgpuInverseSession`) composes
 * these with `renderFrame` + readback; keeping the math here means the loop
 * logic is provable without a device.
 *
 * Ref: Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
 */

import type { InverseParam, InverseTargetImage } from '@vitrum/core';

// ── image-space loss ─────────────────────────────────────────────────────────

/**
 * Read an interleaved image sample, mapping any non-finite value (NaN / ±Inf) to
 * 0. A path tracer legitimately produces firefly pixels that reach ±Inf once
 * encoded into the rgba16float accumulation target, and `NaN` can appear from a
 * degenerate sample; neither is valid radiance. Without this guard a SINGLE bad
 * pixel poisons the whole mean image loss (`Inf - t = Inf`, `Inf*Inf = Inf`,
 * `Inf/N = Inf`, and any NaN propagates), which in turn NaNs the
 * finite-difference gradient and the Adam step — silently stalling the optimizer
 * at its initial value. Mapping to 0 keeps the loss finite and comparable across
 * probe renders (N is unchanged), so the gradient stays meaningful. Surfaced by
 * the V24 real-GPU inverse-session run (the unit tests use a finite fake forward
 * model and never hit a firefly).
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
    this.#cfg = cfg;
    this.#m = new Float32Array(length);
    this.#v = new Float32Array(length);
  }

  /** Apply one Adam update: params ← params − lr·m̂/(√v̂+ε). Mutates `params`. */
  step(params: Float32Array, grad: Float32Array): void {
    this.#t += 1;
    const { learningRate, beta1, beta2, epsilon } = this.#cfg;
    const bc1 = 1 - Math.pow(beta1, this.#t);
    const bc2 = 1 - Math.pow(beta2, this.#t);
    for (let i = 0; i < params.length; i++) {
      const g = grad[i] ?? 0;
      this.#m[i] = beta1 * this.#m[i]! + (1 - beta1) * g;
      this.#v[i] = beta2 * this.#v[i]! + (1 - beta2) * g * g;
      const mHat = this.#m[i]! / bc1;
      const vHat = this.#v[i]! / bc2;
      params[i] = params[i]! - (learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
    }
  }
}

// ── parameter packing / resolution ────────────────────────────────────────────

/** Components for an {@link InverseParam} kind. scalar → 1, vec2 → 2, rgb → 3. */
export function paramLength(p: InverseParam): number {
  switch (p.kind) {
    case 'scalar': return 1;
    case 'vec2': return 2;
    case 'rgb': return 3;
    case 'texture':
      throw new Error(
        `inverse: parameter kind 'texture' (path "${p.path}") is reserved for ` +
          'Phase 2 and is not yet differentiable in pt-webgpu.',
      );
    default: {
      const _exhaustive: never = p.kind;
      throw new Error(`inverse: unknown parameter kind ${String(_exhaustive)}`);
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
  return { domain, id, field };
}

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
