import type {
  InverseGradientMethod,
  InverseSession,
  InverseSessionDiagnostic,
  InverseSessionOptions,
  InverseStepResult,
  MaterialSpec,
  Scene,
  SceneEmitter,
} from '@vitrum/core';
import {
  Adam,
  DEFAULT_ADAM,
  type AdamConfig,
  lossValue,
  parseParamPath,
  paramLength as sharedParamLength,
  clampParams,
  defaultClampRange,
  validateParam as sharedValidateParam,
  validateInitialSceneValue as sharedValidateInitialSceneValue,
  readSceneValue,
  materialPatch,
  emitterPatch,
  type ParamSlot,
} from '@vitrum/core/inverse-scaffolding';

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
      // pt-webgl2 has no runtime-profile capability gate, so no support-details
      // are passed; the `texture` kind is left for paramLength to reject
      // (throwOnTextureKind: false) to preserve pt-webgl2's original raise site.
      sharedValidateParam(scene, param, target, { backend: 'pt-webgl2' });
      const length = sharedParamLength(param, 'pt-webgl2');
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
      sharedValidateInitialSceneValue(slot, initial, slot.param.initial != null, 'pt-webgl2');
      this.#flat.set(initial, slot.offset);
    }

    const cfg: AdamConfig = {
      learningRate: optimizerCfg.learningRate ?? DEFAULT_ADAM.learningRate,
      beta1: optimizerCfg.beta1 ?? DEFAULT_ADAM.beta1,
      beta2: optimizerCfg.beta2 ?? DEFAULT_ADAM.beta2,
      epsilon: optimizerCfg.epsilon ?? DEFAULT_ADAM.epsilon,
    };
    this.#adam = new Adam(this.#flat.length, cfg);
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
    this.#clampParams();
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

  #clampParams(): void {
    const params = this.#slots.map((s) => s.param);
    const layout = this.#slots.map((s) => {
      const [defaultMin, defaultMax] = defaultClampRange(s.target.field);
      return { offset: s.offset, length: s.length, defaultMin, defaultMax };
    });
    clampParams(this.#flat, params, layout);
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
