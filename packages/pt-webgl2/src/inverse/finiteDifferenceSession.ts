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
  type AdamSnapshot,
  assertFiniteArray,
  assertFiniteNumber,
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
  invokeInverseHook,
  normalizeInverseError,
  validateInverseReadback,
  validateInverseSessionOptions,
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
  #generation = 0;
  #stepInFlight = false;
  #poisoned: AggregateError | null = null;

  constructor(hooks: WebGl2InverseEngineHooks, opts: InverseSessionOptions) {
    const config = validateInverseSessionOptions(opts, 'pt-webgl2');
    const loss = config.loss;
    const diagnostics: InverseSessionDiagnostic[] = [];
    if (config.method === 'path-replay') {
      diagnostics.push({
        severity: 'info',
        code: 'path-replay-hook-missing',
        message:
          '[vitrum/pt-webgl2] InverseSession requested path-replay, but pt-webgl2 ' +
          'only exposes finite-difference gradients; using finite-difference.',
      });
      try {
        config.onDiagnostic?.(diagnostics[0]!);
      } catch {
        // Host diagnostic callbacks must not abort inverse-session creation.
      }
    }

    this.#hooks = hooks;
    this.#target = config.target;
    this.#lossKind = loss;
    this.#diagnostics = diagnostics;
    this.#samplesPerStep = config.samplesPerStep;
    this.#fdEpsilon = config.optimizer.fdEpsilon;

    const scene = invokeInverseHook(
      'createInverseSession getScene',
      () => hooks.getScene(),
    );
    let offset = 0;
    for (const param of config.parameters) {
      const target = parseParamPath(param.path);
      sharedValidateParam(scene, param, target, { backend: 'pt-webgl2' });
      const length = sharedParamLength(param, 'pt-webgl2');
      this.#slots.push({ param, target, offset, length });
      offset += length;
    }
    this.#flat = new Float32Array(offset);

    const originalSceneValues = this.#captureSceneValues(scene);
    for (const slot of this.#slots) {
      const initial = slot.param.initial ?? readSceneValue(scene, slot.target, slot.length);
      sharedValidateInitialSceneValue(slot, initial, slot.param.initial != null, 'pt-webgl2');
      this.#flat.set(initial, slot.offset);
    }
    assertFiniteArray(this.#flat, 'createInverseSession parameter values');

    this.#adam = new Adam(this.#flat.length, config.optimizer);
    try {
      this.#applyFlatToScene();
    } catch (error) {
      const cause = normalizeInverseError(
        error,
        'createInverseSession initial scene update',
      );
      const rollbackErrors = this.#restoreSceneValues(originalSceneValues);
      if (rollbackErrors.length > 0) {
        throw new AggregateError(
          [cause, ...rollbackErrors],
          'createInverseSession: initial scene update and rollback both failed.',
        );
      }
      throw cause;
    }
  }

  get parameterCount(): number {
    return this.#slots.length;
  }

  get method(): InverseGradientMethod {
    return 'finite-difference';
  }

  get parameterMethods(): readonly InverseGradientMethod[] {
    return this.#slots.map(() => 'finite-difference');
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
    if (this.#poisoned != null) {
      throw this.#poisoned;
    }
    if (this.#stepInFlight) {
      throw new Error('InverseSession.step: a step is already in progress.');
    }
    this.#stepInFlight = true;
    try {
      const generation = this.#generation;
      const flatSnapshot = this.#flat.slice();
      const adamSnapshot = this.#adam.snapshot();
      const sceneSnapshot = this.#captureSceneValues(invokeInverseHook(
        'InverseSession.step getScene',
        () => this.#hooks.getScene(),
      ));
      try {
        const { width, height } = this.#target;
        this.#applyFlatToScene();
        const base = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
        this.#assertGeneration(generation);
        validateInverseReadback(base.rgba, base.channels, width, height, 'baseline readback');
        const baselineLoss = lossValue(base.rgba, base.channels, this.#target, this.#lossKind);
        assertFiniteNumber(baselineLoss, 'baseline loss');

        const grad = new Float32Array(this.#flat.length);
        for (const slot of this.#slots) {
          for (let local = 0; local < slot.length; local += 1) {
            const flatIndex = slot.offset + local;
            grad[flatIndex] = await this.#probeGradientComponent(
              flatIndex,
              baselineLoss,
              width,
              height,
              generation,
            );
            this.#assertGeneration(generation);
          }
        }
        assertFiniteArray(grad, 'InverseSession.step gradient');

        const gradByParam = this.#slots.map((slot) =>
          Array.from(grad.subarray(slot.offset, slot.offset + slot.length)),
        );
        this.#adam.step(this.#flat, grad);
        this.#clampParams();
        assertFiniteArray(this.#flat, 'InverseSession.step parameter values');
        this.#applyFlatToScene();
        this.#assertGeneration(generation);

        const result: InverseStepResult = {
          step: this.#stepIndex,
          loss: baselineLoss,
          values: this.currentValues(),
          gradient: gradByParam,
        };
        this.#stepIndex += 1;
        return result;
      } catch (error) {
        this.#rollbackStep(error, flatSnapshot, adamSnapshot, sceneSnapshot);
      }
    } finally {
      this.#stepInFlight = false;
    }
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#generation += 1;
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
      this.#patchSlot(slot, value);
    }
  }

  async #probeGradientComponent(
    flatIndex: number,
    baselineLoss: number,
    width: number,
    height: number,
    generation: number,
  ): Promise<number> {
    const original = this.#flat[flatIndex]!;
    let failed = false;
    let failure: unknown;
    let gradient = 0;
    this.#flat[flatIndex] = original + this.#fdEpsilon;
    try {
      assertFiniteNumber(this.#flat[flatIndex], 'finite-difference perturbation');
      this.#applyFlatToScene();
      const probe = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
      this.#assertGeneration(generation);
      validateInverseReadback(probe.rgba, probe.channels, width, height, 'probe readback');
      const probeLoss = lossValue(probe.rgba, probe.channels, this.#target, this.#lossKind);
      assertFiniteNumber(probeLoss, 'probe loss');
      gradient = (probeLoss - baselineLoss) / this.#fdEpsilon;
      assertFiniteNumber(gradient, 'finite-difference gradient');
    } catch (error) {
      failed = true;
      failure = normalizeInverseError(error, 'InverseSession finite-difference probe');
    }
    this.#flat[flatIndex] = original;
    try {
      this.#applyFlatToScene();
    } catch (restoreError) {
      const restoreCause = normalizeInverseError(
        restoreError,
        'InverseSession finite-difference probe restoration',
      );
      if (failed) {
        throw new AggregateError(
          [failure, restoreCause],
          'InverseSession.step: finite-difference probe and restoration both failed.',
        );
      }
      throw restoreCause;
    }
    if (failed) throw failure;
    return gradient;
  }

  #assertGeneration(generation: number): void {
    if (this.#disposed || generation !== this.#generation) {
      throw new Error('InverseSession.step: session was disposed while the step was in progress.');
    }
  }

  #captureSceneValues(scene: Scene): number[][] {
    return this.#slots.map((slot) =>
      readSceneValue(scene, slot.target, slot.length).slice(),
    );
  }

  #restoreSceneValues(values: readonly (readonly number[])[]): unknown[] {
    const errors: unknown[] = [];
    for (let i = 0; i < this.#slots.length; i++) {
      try {
        this.#patchSlot(this.#slots[i]!, values[i]!);
      } catch (error) {
        errors.push(normalizeInverseError(error, 'InverseSession scene rollback'));
      }
    }
    return errors;
  }

  #rollbackStep(
    error: unknown,
    flatSnapshot: Float32Array,
    adamSnapshot: AdamSnapshot,
    sceneSnapshot: readonly (readonly number[])[],
  ): never {
    const cause = normalizeInverseError(error, 'InverseSession.step');
    this.#flat.set(flatSnapshot);
    const rollbackErrors: unknown[] = [];
    try {
      this.#adam.restore(adamSnapshot);
    } catch (rollbackError) {
      rollbackErrors.push(normalizeInverseError(
        rollbackError,
        'InverseSession Adam rollback',
      ));
    }
    rollbackErrors.push(...this.#restoreSceneValues(sceneSnapshot));
    if (rollbackErrors.length > 0) {
      const aggregate = new AggregateError(
        [cause, ...rollbackErrors],
        'InverseSession.step: rollback failed; the session is poisoned.',
      );
      this.#poisoned = aggregate;
      throw aggregate;
    }
    throw cause;
  }

  #patchSlot(slot: ParamSlot, value: readonly number[]): void {
    if (slot.target.domain === 'materials') {
      this.#hooks.patchMaterial(slot.target.id, materialPatch(slot.target.field, value));
    } else {
      this.#hooks.patchEmitter(slot.target.id, emitterPatch(slot.target.field, value));
    }
  }
}
