import type {
  InverseGradientMethod,
  InverseParam,
  InverseSession,
  InverseSessionDiagnostic,
  InverseSessionOptions,
  InverseStepResult,
  MaterialSpec,
  Scene,
  SceneEmitter,
  Vec2,
  Vec3,
} from '@vitrum/core';

export interface WebGl2InverseEngineHooks {
  getScene(): Scene;
  renderAndReadback(
    width: number,
    height: number,
    samples: number,
  ): Promise<{ rgba: Float32Array; channels: 4 }>;
  patchMaterial(primitiveId: string, patch: Partial<MaterialSpec>): void;
  patchEmitter(emitterId: string, patch: Partial<SceneEmitter>): void;
}

interface ResolvedParamTarget {
  readonly domain: 'materials' | 'emitters';
  readonly id: string;
  readonly field: string;
}

interface ParamSlot {
  readonly param: InverseParam;
  readonly target: ResolvedParamTarget;
  readonly offset: number;
  readonly length: number;
}

interface AdamConfig {
  readonly learningRate: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly epsilon: number;
}

const DEFAULT_ADAM: AdamConfig = {
  learningRate: 1e-2,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
};

const MATERIAL_RGB_FIELDS = new Set([
  'baseColor',
  'emissive',
  'attenuationColor',
  'specularColor',
  'sheenColor',
  'scatteringCoefficientRGB',
]);

const MATERIAL_VEC2_FIELDS = new Set([
  'iridescenceThicknessRange',
]);

const MATERIAL_SCALAR_FIELDS = new Set([
  'roughness',
  'metallic',
  'emissiveIntensity',
  'opacity',
  'alphaCutoff',
  'ior',
  'transmission',
  'thickness',
  'attenuationDistance',
  'specularIntensity',
  'clearcoat',
  'clearcoatRoughness',
  'sheen',
  'sheenRoughness',
  'iridescence',
  'iridescenceIor',
  'anisotropy',
  'anisotropyRotation',
  'normalScale',
  'bumpScale',
  'clearcoatNormalScale',
  'aoMapIntensity',
  'lightMapIntensity',
  'envMapIntensity',
  'dispersionAbbeNumber',
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'displacementScale',
  'displacementBias',
]);

const EMITTER_RGB_FIELDS = new Set(['color']);
const EMITTER_SCALAR_FIELDS = new Set(['intensity']);

function finiteSample(buf: Float32Array, i: number): number {
  const v = buf[i];
  return v !== undefined && Number.isFinite(v) ? v : 0;
}

function lossValue(
  rendered: Float32Array,
  renderChannels: 4,
  target: InverseSessionOptions['target'],
  kind: 'l2' | 'l1',
): number {
  const targetChannels = target.channels ?? 3;
  const n = target.width * target.height;
  const sampleCount = n * 3;
  let loss = 0;
  for (let p = 0; p < n; p += 1) {
    for (let c = 0; c < 3; c += 1) {
      const diff = finiteSample(rendered, p * renderChannels + c) -
        finiteSample(target.data, p * targetChannels + c);
      loss += kind === 'l2' ? diff * diff : Math.abs(diff);
    }
  }
  return loss / sampleCount;
}

class Adam {
  readonly #cfg: AdamConfig;
  readonly #m: Float32Array;
  readonly #v: Float32Array;
  #t = 0;

  constructor(length: number, cfg: AdamConfig) {
    this.#cfg = cfg;
    this.#m = new Float32Array(length);
    this.#v = new Float32Array(length);
  }

  step(params: Float32Array, grad: Float32Array): void {
    this.#t += 1;
    const { learningRate, beta1, beta2, epsilon } = this.#cfg;
    const bc1 = 1 - Math.pow(beta1, this.#t);
    const bc2 = 1 - Math.pow(beta2, this.#t);
    for (let i = 0; i < params.length; i += 1) {
      const g = grad[i] ?? 0;
      this.#m[i] = beta1 * this.#m[i]! + (1 - beta1) * g;
      this.#v[i] = beta2 * this.#v[i]! + (1 - beta2) * g * g;
      const mHat = this.#m[i]! / bc1;
      const vHat = this.#v[i]! / bc2;
      params[i] = params[i]! - (learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
    }
  }
}

export class WebGl2FiniteDifferenceInverseSession implements InverseSession {
  readonly #hooks: WebGl2InverseEngineHooks;
  readonly #target: InverseSessionOptions['target'];
  readonly #lossKind: 'l2' | 'l1';
  readonly #samplesPerStep: number;
  readonly #fdEpsilon: number;
  readonly #diagnostics: readonly InverseSessionDiagnostic[];
  readonly #slots: ParamSlot[] = [];
  readonly #flat: Float32Array;
  readonly #adam: Adam;
  #stepIndex = 0;
  #disposed = false;

  constructor(hooks: WebGl2InverseEngineHooks, opts: InverseSessionOptions) {
    if (opts.parameters.length === 0) {
      throw new Error('createInverseSession: at least one parameter is required.');
    }
    if (opts.target.width <= 0 || opts.target.height <= 0) {
      throw new Error('createInverseSession: target image must have positive dimensions.');
    }
    const loss = opts.loss ?? 'l2';
    if (loss === 'ssim' || loss === 'lpips') {
      throw new Error(
        `createInverseSession: loss '${loss}' is a reserved perceptual loss and ` +
          "is not yet implemented in pt-webgl2 (use 'l2' or 'l1').",
      );
    }
    const diagnostics: InverseSessionDiagnostic[] = [];
    if (opts.method === 'path-replay') {
      diagnostics.push({
        severity: 'info',
        code: 'path-replay-hook-missing',
        message:
          '[vitrum/pt-webgl2] InverseSession requested path-replay, but pt-webgl2 ' +
          'only exposes finite-difference gradients; using finite-difference.',
      });
      try {
        opts.onDiagnostic?.(diagnostics[0]!);
      } catch {
        // Host diagnostic callbacks must not abort inverse-session creation.
      }
    }

    this.#hooks = hooks;
    this.#target = opts.target;
    this.#lossKind = loss === 'l1' ? 'l1' : 'l2';
    this.#diagnostics = diagnostics;
    this.#samplesPerStep = Math.max(1, Math.floor(opts.samplesPerStep ?? 8));
    const optimizerCfg = opts.optimizer ?? {};
    this.#fdEpsilon = optimizerCfg.fdEpsilon ?? 1e-3;

    const scene = hooks.getScene();
    let offset = 0;
    for (const param of opts.parameters) {
      const target = parseParamPath(param.path);
      validateParam(scene, param, target);
      const length = paramLength(param);
      this.#slots.push({ param, target, offset, length });
      offset += length;
    }
    this.#flat = new Float32Array(offset);

    for (const slot of this.#slots) {
      const initial = slot.param.initial ?? readSceneValue(scene, slot.target, slot.length);
      if (initial.length !== slot.length) {
        throw new Error(
          `createInverseSession: parameter "${slot.param.path}" initial value has ` +
            `length ${initial.length}, expected ${slot.length}.`,
        );
      }
      for (const value of initial) {
        if (!Number.isFinite(value)) {
          throw new Error(
            `createInverseSession: parameter "${slot.param.path}" initial value must be finite.`,
          );
        }
      }
      this.#flat.set(initial, slot.offset);
    }

    this.#adam = new Adam(this.#flat.length, {
      learningRate: optimizerCfg.learningRate ?? DEFAULT_ADAM.learningRate,
      beta1: optimizerCfg.beta1 ?? DEFAULT_ADAM.beta1,
      beta2: optimizerCfg.beta2 ?? DEFAULT_ADAM.beta2,
      epsilon: optimizerCfg.epsilon ?? DEFAULT_ADAM.epsilon,
    });
    this.#applyFlatToScene();
  }

  get parameterCount(): number {
    return this.#slots.length;
  }

  get method(): InverseGradientMethod {
    return 'finite-difference';
  }

  get diagnostics(): readonly InverseSessionDiagnostic[] {
    return this.#diagnostics;
  }

  currentValues(): readonly (readonly number[])[] {
    return this.#slots.map((slot) =>
      Array.from(this.#flat.subarray(slot.offset, slot.offset + slot.length)),
    );
  }

  async step(): Promise<InverseStepResult> {
    if (this.#disposed) {
      throw new Error('InverseSession.step: session is disposed.');
    }
    const { width, height } = this.#target;
    this.#applyFlatToScene();
    const base = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
    const baselineLoss = lossValue(base.rgba, base.channels, this.#target, this.#lossKind);
    const grad = new Float32Array(this.#flat.length);
    const eps = this.#fdEpsilon;
    for (const slot of this.#slots) {
      for (let local = 0; local < slot.length; local += 1) {
        const flatIndex = slot.offset + local;
        const original = this.#flat[flatIndex]!;
        this.#flat[flatIndex] = original + eps;
        this.#applyFlatToScene();
        const probe = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
        const probeLoss = lossValue(probe.rgba, probe.channels, this.#target, this.#lossKind);
        grad[flatIndex] = (probeLoss - baselineLoss) / eps;
        this.#flat[flatIndex] = original;
      }
    }

    const gradByParam = this.#slots.map((slot) =>
      Array.from(grad.subarray(slot.offset, slot.offset + slot.length)),
    );
    this.#adam.step(this.#flat, grad);
    clampParams(this.#flat, this.#slots);
    this.#applyFlatToScene();

    const result: InverseStepResult = {
      step: this.#stepIndex,
      loss: baselineLoss,
      values: this.currentValues(),
      gradient: gradByParam,
    };
    this.#stepIndex += 1;
    return result;
  }

  dispose(): void {
    this.#disposed = true;
  }

  #applyFlatToScene(): void {
    for (const slot of this.#slots) {
      const value = Array.from(this.#flat.subarray(slot.offset, slot.offset + slot.length));
      if (slot.target.domain === 'materials') {
        this.#hooks.patchMaterial(slot.target.id, materialPatch(slot.target.field, value));
      } else {
        this.#hooks.patchEmitter(slot.target.id, emitterPatch(slot.target.field, value));
      }
    }
  }
}

function paramLength(param: InverseParam): number {
  switch (param.kind) {
    case 'scalar':
      return 1;
    case 'vec2':
      return 2;
    case 'rgb':
      return 3;
    case 'texture':
      throw new Error(
        `createInverseSession: parameter kind 'texture' (path "${param.path}") is reserved ` +
          'for future texture optimization and is not yet differentiable in pt-webgl2.',
      );
  }
}

function parseParamPath(path: string): ResolvedParamTarget {
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

function validateParam(scene: Scene, param: InverseParam, target: ResolvedParamTarget): void {
  if (param.kind === 'texture') {
    paramLength(param);
  }
  if (target.domain === 'materials') {
    const primitive = scene.primitives.find((p) => p.id === target.id);
    if (primitive == null) {
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
    assertKind(param, isRgb ? 'rgb' : isVec2 ? 'vec2' : 'scalar');
    return;
  }

  const emitter = scene.emitters.find((e) => e.id === target.id);
  if (emitter == null) {
    throw new Error(
      `createInverseSession: no emitter with id "${target.id}" for path "${param.path}".`,
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
  assertKind(param, isRgb ? 'rgb' : 'scalar');
}

function assertKind(param: InverseParam, expected: 'rgb' | 'vec2' | 'scalar'): void {
  if (param.kind !== expected) {
    throw new Error(
      `createInverseSession: parameter "${param.path}" is declared kind '${param.kind}' ` +
        `but the resolved field is '${expected}'.`,
    );
  }
}

function readSceneValue(scene: Scene, target: ResolvedParamTarget, length: number): number[] {
  if (target.domain === 'materials') {
    const primitive = scene.primitives.find((p) => p.id === target.id)!;
    const material = primitive.material;
    switch (target.field) {
      case 'baseColor': return [...material.baseColor];
      case 'roughness': return [material.roughness];
      case 'metallic': return [material.metallic];
      case 'emissive': return [...(material.emissive ?? [0, 0, 0])];
      case 'emissiveIntensity': return [material.emissiveIntensity ?? 1];
      case 'opacity': return [material.opacity ?? 1];
      case 'alphaCutoff': return [material.alphaCutoff ?? 0.5];
      case 'ior': return [material.ior ?? 1.5];
      case 'transmission': return [material.transmission ?? 0];
      case 'thickness': return [material.thickness ?? 0];
      case 'attenuationColor': return [...(material.attenuationColor ?? [1, 1, 1])];
      case 'attenuationDistance': return [finiteDefault(material.attenuationDistance, 1)];
      case 'dispersionAbbeNumber': return [material.dispersionAbbeNumber ?? 0];
      case 'scatteringCoefficient': return [material.scatteringCoefficient ?? 0];
      case 'scatteringAnisotropy': return [material.scatteringAnisotropy ?? 0];
      case 'scatteringCoefficientRGB': return [...(material.scatteringCoefficientRGB ?? [0, 0, 0])];
      case 'specularColor': return [...(material.specularColor ?? [1, 1, 1])];
      case 'specularIntensity': return [material.specularIntensity ?? 1];
      case 'clearcoat': return [material.clearcoat ?? 0];
      case 'clearcoatRoughness': return [material.clearcoatRoughness ?? 0];
      case 'sheen': return [material.sheen ?? 0];
      case 'sheenColor': return [...(material.sheenColor ?? [1, 1, 1])];
      case 'sheenRoughness': return [material.sheenRoughness ?? 0];
      case 'iridescence': return [material.iridescence ?? 0];
      case 'iridescenceIor': return [material.iridescenceIor ?? 1.3];
      case 'iridescenceThicknessRange': return [...(material.iridescenceThicknessRange ?? [100, 400])];
      case 'anisotropy': return [material.anisotropy ?? 0];
      case 'anisotropyRotation': return [material.anisotropyRotation ?? 0];
      case 'normalScale': return [material.normalScale ?? 1];
      case 'bumpScale': return [material.bumpScale ?? 1];
      case 'clearcoatNormalScale': return [material.clearcoatNormalScale ?? 1];
      case 'aoMapIntensity': return [material.aoMapIntensity ?? 1];
      case 'lightMapIntensity': return [material.lightMapIntensity ?? 1];
      case 'envMapIntensity': return [material.envMapIntensity ?? 1];
      case 'displacementScale': return [material.displacementScale ?? 1];
      case 'displacementBias': return [material.displacementBias ?? 0];
    }
  } else {
    const emitter = scene.emitters.find((e) => e.id === target.id)!;
    switch (target.field) {
      case 'color': return [...emitter.color];
      case 'intensity': return [emitter.intensity];
    }
  }
  return new Array<number>(length).fill(0);
}

function finiteDefault(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function defaultClampRange(field: string): [number, number] {
  switch (field) {
    case 'baseColor':
    case 'specularColor':
    case 'sheenColor':
    case 'roughness':
    case 'metallic':
    case 'opacity':
    case 'alphaCutoff':
    case 'transmission':
    case 'specularIntensity':
    case 'clearcoat':
    case 'clearcoatRoughness':
    case 'sheen':
    case 'sheenRoughness':
    case 'iridescence':
    case 'anisotropy':
    case 'aoMapIntensity':
      return [0, 1];
    case 'attenuationColor':
      return [1e-4, 1];
    case 'ior':
      return [1, 2.5];
    case 'attenuationDistance':
      return [1e-6, Infinity];
    case 'scatteringAnisotropy':
      return [-0.95, 0.95];
    case 'iridescenceIor':
      return [1, 3];
    case 'anisotropyRotation':
    case 'displacementScale':
    case 'displacementBias':
      return [-Infinity, Infinity];
    case 'emissive':
    case 'emissiveIntensity':
    case 'thickness':
    case 'normalScale':
    case 'bumpScale':
    case 'clearcoatNormalScale':
    case 'lightMapIntensity':
    case 'envMapIntensity':
    case 'color':
    case 'intensity':
    case 'iridescenceThicknessRange':
    case 'dispersionAbbeNumber':
    case 'scatteringCoefficient':
    case 'scatteringCoefficientRGB':
      return [0, Infinity];
    default:
      return [0, Infinity];
  }
}

function clampParams(flat: Float32Array, slots: readonly ParamSlot[]): void {
  for (const slot of slots) {
    const [defaultMin, defaultMax] = defaultClampRange(slot.target.field);
    const lo = slot.param.min ?? defaultMin;
    const hi = slot.param.max ?? defaultMax;
    for (let c = 0; c < slot.length; c += 1) {
      const i = slot.offset + c;
      flat[i] = Math.min(Math.max(flat[i]!, lo), hi);
    }
  }
}

function materialPatch(field: string, value: readonly number[]): Partial<MaterialSpec> {
  switch (field) {
    case 'baseColor': return { baseColor: vec3(value, [1, 1, 1]) };
    case 'roughness': return { roughness: value[0]! };
    case 'metallic': return { metallic: value[0]! };
    case 'emissive': return { emissive: vec3(value, [0, 0, 0]) };
    case 'emissiveIntensity': return { emissiveIntensity: value[0]! };
    case 'opacity': return { opacity: value[0]! };
    case 'alphaCutoff': return { alphaCutoff: value[0]! };
    case 'ior': return { ior: value[0]! };
    case 'transmission': return { transmission: value[0]! };
    case 'thickness': return { thickness: value[0]! };
    case 'attenuationColor': return { attenuationColor: vec3(value, [1, 1, 1]) };
    case 'attenuationDistance': return { attenuationDistance: value[0]! };
    case 'dispersionAbbeNumber': return { dispersionAbbeNumber: value[0]! };
    case 'scatteringCoefficient': return { scatteringCoefficient: value[0]! };
    case 'scatteringAnisotropy': return { scatteringAnisotropy: value[0]! };
    case 'scatteringCoefficientRGB': return { scatteringCoefficientRGB: vec3(value, [0, 0, 0]) };
    case 'specularColor': return { specularColor: vec3(value, [1, 1, 1]) };
    case 'specularIntensity': return { specularIntensity: value[0]! };
    case 'clearcoat': return { clearcoat: value[0]! };
    case 'clearcoatRoughness': return { clearcoatRoughness: value[0]! };
    case 'sheen': return { sheen: value[0]! };
    case 'sheenColor': return { sheenColor: vec3(value, [1, 1, 1]) };
    case 'sheenRoughness': return { sheenRoughness: value[0]! };
    case 'iridescence': return { iridescence: value[0]! };
    case 'iridescenceIor': return { iridescenceIor: value[0]! };
    case 'iridescenceThicknessRange':
      return {
        iridescenceThicknessRange: [
          Math.max(value[0] ?? 100, 0),
          Math.max(value[1] ?? 400, 0),
        ] as unknown as Vec2,
      };
    case 'anisotropy': return { anisotropy: value[0]! };
    case 'anisotropyRotation': return { anisotropyRotation: value[0]! };
    case 'normalScale': return { normalScale: value[0]! };
    case 'bumpScale': return { bumpScale: value[0]! };
    case 'clearcoatNormalScale': return { clearcoatNormalScale: value[0]! };
    case 'aoMapIntensity': return { aoMapIntensity: value[0]! };
    case 'lightMapIntensity': return { lightMapIntensity: value[0]! };
    case 'envMapIntensity': return { envMapIntensity: value[0]! };
    case 'displacementScale': return { displacementScale: value[0]! };
    case 'displacementBias': return { displacementBias: value[0]! };
    default: throw new Error(`inverse: unsupported material field "${field}".`);
  }
}

function emitterPatch(field: string, value: readonly number[]): Partial<SceneEmitter> {
  switch (field) {
    case 'color': return { color: vec3(value, [1, 1, 1]) };
    case 'intensity': return { intensity: value[0]! };
    default: throw new Error(`inverse: unsupported emitter field "${field}".`);
  }
}

function vec3(value: readonly number[], fallback: readonly [number, number, number]): Vec3 {
  return [
    value[0] ?? fallback[0],
    value[1] ?? fallback[1],
    value[2] ?? fallback[2],
  ] as unknown as Vec3;
}
