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

import type { InverseParam, InverseTargetImage } from './inverse.js';
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

/** Components for an {@link InverseParam} kind. scalar → 1, vec2 → 2, rgb → 3.
 *  `texture` is reserved (Phase 2) and not yet differentiable on any backend —
 *  `backend` names the caller for the thrown-error attribution. */
export function paramLength(p: InverseParam, backend: string): number {
  switch (p.kind) {
    case 'scalar': return 1;
    case 'vec2': return 2;
    case 'rgb': return 3;
    case 'texture':
      throw new Error(
        `inverse: parameter kind 'texture' (path "${p.path}") is reserved for ` +
          `Phase 2 (texture optimization) and is not yet differentiable in ${backend}.`,
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

/** Fill an rgb triple from a (possibly short) value list, falling back per
 *  component. For the full-length rgb slots the backends always pass, this is
 *  identical to a bare cast; the fallback guards a malformed shorter input. */
function vec3(value: readonly number[], fallback: readonly [number, number, number]): Vec3 {
  return [
    value[0] ?? fallback[0],
    value[1] ?? fallback[1],
    value[2] ?? fallback[2],
  ] as unknown as Vec3;
}

export const MATERIAL_PARAM_DESCRIPTORS: Readonly<Record<string, MaterialParamDescriptor>> = {
  baseColor: {
    kind: 'rgb', clamp: [0, 1],
    read: (m) => [...m.baseColor],
    patch: (v) => ({ baseColor: vec3(v, [1, 1, 1]) }),
  },
  roughness: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.roughness],
    patch: (v) => ({ roughness: v[0]! }),
  },
  metallic: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.metallic],
    patch: (v) => ({ metallic: v[0]! }),
  },
  emissive: {
    kind: 'rgb', clamp: [0, INF],
    read: (m) => [...(m.emissive ?? [0, 0, 0])],
    patch: (v) => ({ emissive: vec3(v, [0, 0, 0]) }),
  },
  emissiveIntensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.emissiveIntensity ?? 1],
    patch: (v) => ({ emissiveIntensity: v[0]! }),
  },
  opacity: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.opacity ?? 1],
    patch: (v) => ({ opacity: v[0]! }),
  },
  alphaCutoff: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.alphaCutoff ?? 0.5],
    patch: (v) => ({ alphaCutoff: v[0]! }),
  },
  ior: {
    kind: 'scalar', clamp: [1, 2.5],
    read: (m) => [m.ior ?? 1.5],
    patch: (v) => ({ ior: v[0]! }),
  },
  transmission: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.transmission ?? 0],
    patch: (v) => ({ transmission: v[0]! }),
  },
  thickness: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.thickness ?? 0],
    patch: (v) => ({ thickness: v[0]! }),
  },
  attenuationColor: {
    kind: 'rgb', clamp: [1e-4, 1],
    read: (m) => [...(m.attenuationColor ?? [1, 1, 1])],
    patch: (v) => ({ attenuationColor: vec3(v, [1, 1, 1]) }),
  },
  attenuationDistance: {
    kind: 'scalar', clamp: [1e-6, INF],
    read: (m) => [m.attenuationDistance ?? Number.POSITIVE_INFINITY],
    patch: (v) => ({ attenuationDistance: v[0]! }),
  },
  dispersionAbbeNumber: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.dispersionAbbeNumber ?? 0],
    patch: (v) => ({ dispersionAbbeNumber: v[0]! }),
  },
  scatteringCoefficient: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.scatteringCoefficient ?? 0],
    patch: (v) => ({ scatteringCoefficient: v[0]! }),
  },
  scatteringAnisotropy: {
    kind: 'scalar', clamp: [-0.95, 0.95],
    read: (m) => [m.scatteringAnisotropy ?? 0],
    patch: (v) => ({ scatteringAnisotropy: v[0]! }),
  },
  scatteringCoefficientRGB: {
    kind: 'rgb', clamp: [0, INF],
    read: (m) => [...(m.scatteringCoefficientRGB ?? [0, 0, 0])],
    patch: (v) => ({ scatteringCoefficientRGB: vec3(v, [0, 0, 0]) }),
  },
  specularColor: {
    kind: 'rgb', clamp: [0, 1],
    read: (m) => [...(m.specularColor ?? [1, 1, 1])],
    patch: (v) => ({ specularColor: vec3(v, [1, 1, 1]) }),
  },
  specularIntensity: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.specularIntensity ?? 1],
    patch: (v) => ({ specularIntensity: v[0]! }),
  },
  clearcoat: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.clearcoat ?? 0],
    patch: (v) => ({ clearcoat: v[0]! }),
  },
  clearcoatRoughness: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.clearcoatRoughness ?? 0],
    patch: (v) => ({ clearcoatRoughness: v[0]! }),
  },
  sheen: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.sheen ?? 0],
    patch: (v) => ({ sheen: v[0]! }),
  },
  sheenColor: {
    kind: 'rgb', clamp: [0, INF],
    read: (m) => [...(m.sheenColor ?? [1, 1, 1])],
    patch: (v) => ({ sheenColor: vec3(v, [1, 1, 1]) }),
  },
  sheenRoughness: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.sheenRoughness ?? 0],
    patch: (v) => ({ sheenRoughness: v[0]! }),
  },
  iridescence: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.iridescence ?? 0],
    patch: (v) => ({ iridescence: v[0]! }),
  },
  iridescenceIor: {
    kind: 'scalar', clamp: [1, 3],
    read: (m) => [m.iridescenceIor ?? 1.3],
    patch: (v) => ({ iridescenceIor: v[0]! }),
  },
  iridescenceThicknessRange: {
    kind: 'vec2', clamp: [0, INF],
    read: (m) => [...(m.iridescenceThicknessRange ?? [100, 400])],
    patch: (v) => ({
      iridescenceThicknessRange: [
        Math.max(v[0] ?? 100, 0),
        Math.max(v[1] ?? 400, 0),
      ] as unknown as Vec2,
    }),
  },
  anisotropy: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.anisotropy ?? 0],
    patch: (v) => ({ anisotropy: v[0]! }),
  },
  anisotropyRotation: {
    kind: 'scalar', clamp: [-INF, INF],
    read: (m) => [m.anisotropyRotation ?? 0],
    patch: (v) => ({ anisotropyRotation: v[0]! }),
  },
  normalScale: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.normalScale ?? 1],
    patch: (v) => ({ normalScale: v[0]! }),
  },
  bumpScale: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.bumpScale ?? 1],
    patch: (v) => ({ bumpScale: v[0]! }),
  },
  clearcoatNormalScale: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.clearcoatNormalScale ?? 1],
    patch: (v) => ({ clearcoatNormalScale: v[0]! }),
  },
  aoMapIntensity: {
    kind: 'scalar', clamp: [0, 1],
    read: (m) => [m.aoMapIntensity ?? 1],
    patch: (v) => ({ aoMapIntensity: v[0]! }),
  },
  lightMapIntensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.lightMapIntensity ?? 1],
    patch: (v) => ({ lightMapIntensity: v[0]! }),
  },
  envMapIntensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (m) => [m.envMapIntensity ?? 1],
    patch: (v) => ({ envMapIntensity: v[0]! }),
  },
  displacementScale: {
    kind: 'scalar', clamp: [-INF, INF],
    read: (m) => [m.displacementScale ?? 1],
    patch: (v) => ({ displacementScale: v[0]! }),
  },
  displacementBias: {
    kind: 'scalar', clamp: [-INF, INF],
    read: (m) => [m.displacementBias ?? 0],
    patch: (v) => ({ displacementBias: v[0]! }),
  },
};

export const EMITTER_PARAM_DESCRIPTORS: Readonly<Record<string, EmitterParamDescriptor>> = {
  color: {
    kind: 'rgb', clamp: [0, INF],
    read: (e) => [...e.color],
    patch: (v) => ({ color: vec3(v, [1, 1, 1]) }),
  },
  intensity: {
    kind: 'scalar', clamp: [0, INF],
    read: (e) => [e.intensity],
    patch: (v) => ({ intensity: v[0]! }),
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
 * `opts.backend` names the caller for `texture`-kind error attribution.
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
    /** Whether a `texture`-kind param throws in this validator (pt-webgpu) or is
     *  deferred to `paramLength` (pt-webgl2). Both ultimately throw for texture;
     *  this only controls which site raises. Default: false. */
    throwOnTextureKind?: boolean;
  },
): void {
  const { backend } = opts;
  if (param.kind === 'texture') {
    if (opts.throwOnTextureKind) {
      throw new Error(
        `createInverseSession: parameter kind 'texture' (path "${param.path}") is reserved ` +
          `for Phase 2 (texture optimization) and is not yet differentiable in ${backend}.`,
      );
    }
    // Non-throwing path: let paramLength raise (pt-webgl2 legacy behavior).
    paramLength(param, backend);
  }
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
    if (descriptor) return descriptor.read(prim.material);
  } else {
    const e = scene.emitters.find((em) => em.id === target.id)!;
    const descriptor = EMITTER_PARAM_DESCRIPTORS[target.field];
    if (descriptor) return descriptor.read(e);
  }
  // unreachable — validateParam already rejected unknown fields
  return new Array<number>(length).fill(0);
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
