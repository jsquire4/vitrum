// progressiveHandoff.ts — Walkaround↔PT progressive-refinement handoff (P8).
//
// A host-side coordinator over TWO engines sharing ONE scene: a REAL-TIME engine
// (e.g. @vitrum/walkaround-hybrid, smooth GI while the camera moves) and a
// CONVERGED engine (e.g. @vitrum/pt-webgpu, ground-truth path tracing that
// accumulates to a terminal image). It drives the real-time engine while the
// camera is in motion, and — once the camera has been still for a few frames —
// hands off to the converged engine, which accumulates and progressively
// refines (Blender-viewport-style). Any camera motion snaps back to real-time.
//
// HOST-OWNS-LIFECYCLE / HOST-OWNS-CADENCE: this owns NEITHER engine NOR the frame
// loop. The host constructs both engines (sharing the scene), calls `frame(input)`
// once per RAF tick, and presents `result.output` (the active engine's
// FrameOutput). The coordinator only decides WHICH engine renders this frame and
// resets the converged accumulator at the right moment. It is pure host logic —
// no GPU calls of its own — so it is fully unit-testable with stub engines.
//
// This v1 switches the DISPLAY to the converged engine as soon as it starts
// accumulating (progressive refinement — the standard interactive-PT UX). A
// "settle behind the real-time image then cross-fade" variant (to hide the
// real-time→1-sample 'pop') is a documented follow-on; it would render both
// engines during the transition window.

import {
  patchEmitterInScene,
  patchPrimitiveInScene,
  resolveFrameCameraPosition,
  validateScene as validateCoreScene,
  QUALITY_FINAL,
  QUALITY_PREVIEW,
} from '@vitrum/core';
import type {
  CapturedFrame,
  CaptureFrameOptions,
  Engine,
  FrameInput,
  FrameOutput,
  Scene,
  SceneEmitter,
  SceneEnvironment,
  ScenePrimitive,
  ScenePrimitivePatch,
} from '@vitrum/core';

/** Which engine the coordinator is presenting. */
export type HandoffPhase =
  /** Camera moving (or just moved) — real-time engine. */
  | 'realtime'
  /** Camera still but not yet past the handoff threshold — still real-time. */
  | 'settling'
  /** Camera still past the threshold; the converged engine is accumulating
   *  BEHIND the still-displayed real-time image (settleBehindRealtime only). */
  | 'prerolling'
  /** Camera still past the threshold — converged engine, displayed + accumulating. */
  | 'converging';

export interface ProgressiveHandoffControllerTarget {
  setScene(scene: Scene): void;
  updatePrimitive?(id: string, patch: ScenePrimitivePatch): void;
  reset?(): void;
}

export interface ProgressiveHandoffControllerAdvanceOptions {
  readonly engine?: ProgressiveHandoffControllerTarget;
  readonly loop?: boolean;
}

/** Structural animation-controller seam. `@vitrum/gltf-adapter`'s
 *  GltfSceneController satisfies this without making `@vitrum/engine` import the
 *  adapter at runtime. */
export interface ProgressiveHandoffController {
  readonly animations?: readonly unknown[];
  advance(deltaSeconds: number, options?: ProgressiveHandoffControllerAdvanceOptions): unknown;
}

export interface ProgressiveHandoffControllerDeltaState {
  readonly phase: HandoffPhase;
  readonly stillFrames: number;
}

export type ProgressiveHandoffControllerDelta =
  | number
  | ((input: FrameInput, state: ProgressiveHandoffControllerDeltaState) => number);

export interface ProgressiveHandoffOptions {
  /** Smooth real-time GI engine — driven while the camera moves. */
  readonly realtime: Engine;
  /** Converged path-tracing engine — driven (accumulating) while the camera is still. */
  readonly converged: Engine;
  /** Optional authoritative scene snapshot. When supplied, scene mutations can
   *  fall back to `setScene()` on both engines if an incremental method is absent
   *  or rejects, keeping the two-engine pair synchronized. */
  readonly scene?: Scene;
  /** Consecutive still frames before handing off to the converged engine. The
   *  intervening frames keep showing the real-time engine ("settling"), so a
   *  brief pause doesn't thrash the handoff. Default 6; clamped to >= 1. */
  readonly stillFramesBeforeHandoff?: number;
  /** Max-abs camera-delta (view/proj matrices + position) below which a frame
   *  counts as "still". Default 1e-5. */
  readonly cameraEpsilon?: number;
  /** Hide the real-time → 1-sample "pop": once the threshold is reached, the
   *  converged engine accumulates BEHIND the still-displayed real-time image; the
   *  display only switches to the converged engine once it is clean enough (see
   *  {@link convergedDisplaySamples}). Costs both engines rendering during the
   *  pre-roll window. Default false (switch + refine immediately — the standard
   *  interactive-PT UX, and what makes this backward-compatible). */
  readonly settleBehindRealtime?: boolean;
  /** With {@link settleBehindRealtime}: switch the display to the converged engine
   *  once it reports `isConverged` OR has accumulated this many samples (whichever
   *  is first). Default 64 — a reasonably clean preview without waiting for the
   *  full sample target. Ignored when settleBehindRealtime is false. */
  readonly convergedDisplaySamples?: number;
  /** P8 increment 2: on each handoff, SEED the converged engine's accumulator from
   *  the real-time engine's last frame (a DECAYING PREVIEW PRIOR —
   *  `seedAccumulator`) so the first progressive frames start from the smooth
   *  real-time output instead of a 1-sample blizzard. Before that seeded cohort
   *  could be published as terminal, the coordinator resets it and begins the
   *  canonical unseeded accumulation. Requires the real-time engine to
   *  expose `getProgressiveSeedTexture()`, the converged engine `seedAccumulator()`,
   *  and BOTH to share one GPUDevice (use `createProgressiveEngine`); a no-op
   *  otherwise. Default false (resets to black, the v1 behaviour).
   *
   *  NOTE — default divergence (intentional): the `createProgressiveEngine` FACADE
   *  defaults this to `true`, because it mints the shared device itself so seeding
   *  is always safe and is its whole reason to exist. This low-level coordinator
   *  defaults it `false` because it can be constructed over engines that do NOT
   *  share a device (where seeding would be a cross-device no-op), so off is the
   *  safe primitive-level default. */
  readonly seedFromRealtime?: boolean;
  /** Virtual-sample weight of the seed prior (passed to `seedAccumulator`). Higher =
   *  trust the real-time seed longer before PT samples dominate; it decays as
   *  W/(W+M) so it never biases the converged mean. Default 4. */
  readonly seedWeight?: number;
  /** Optional scene animation controller (for example a glTF controller). It is
   *  advanced once at the start of each {@link frame} call and receives a
   *  synthetic patch target that forwards `setScene` / `updatePrimitive` to BOTH
   *  engines through this coordinator. This keeps animated realtime + converged
   *  engines synchronized and naturally invalidates temporal/converged history
   *  through the existing mutation reset path. */
  readonly controller?: ProgressiveHandoffController;
  /** Seconds passed to `controller.advance()` per frame. Default 1/60. Hosts
   *  with an external clock can provide a function of the current FrameInput and
   *  previous handoff state. */
  readonly controllerDeltaSeconds?: ProgressiveHandoffControllerDelta;
  /** Forwarded to `controller.advance(..., { loop })`. Default true. */
  readonly controllerLoop?: boolean;
}

export interface HandoffFrameResult {
  readonly phase: HandoffPhase;
  /** The engine that rendered this frame (its output is `output`). */
  readonly active: Engine;
  /** The frame to present. */
  readonly output: FrameOutput;
  /** During `prerolling` ONLY: the converged engine's accumulating output (the
   *  image being built BEHIND the displayed real-time one). A host that wants a
   *  true alpha cross-fade — rather than the default hard switch once clean — can
   *  blend `output` (real-time) with this. Undefined in every other phase. */
  readonly behindOutput?: FrameOutput;
  /** Consecutive still frames so far (0 the moment the camera moves). */
  readonly stillFrames: number;
}

interface CameraSnapshot {
  readonly view: Float32Array;
  readonly proj: Float32Array;
  readonly pos: Float32Array;
}

function finiteVector(
  value: ArrayLike<number>,
  expectedLength: number,
  label: string,
): Float32Array {
  const sourceLength = value.length;
  if (sourceLength !== expectedLength) {
    throw new RangeError(
      `ProgressiveHandoffCoordinator.frame: ${label} must contain exactly ${expectedLength} values.`,
    );
  }
  const copy = new Float32Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    const sourceValue = Number(value[i]);
    const packed = Math.fround(sourceValue);
    if (!Number.isFinite(sourceValue) || !Number.isFinite(packed)) {
      throw new TypeError(
        `ProgressiveHandoffCoordinator.frame: ${label}[${i}] must be finite.`,
      );
    }
    copy[i] = packed;
  }
  return copy;
}

function snapshot(input: FrameInput): CameraSnapshot {
  const cameraPosition = resolveFrameCameraPosition(
    input,
    'ProgressiveHandoffCoordinator.frame',
  );
  return {
    view: finiteVector(input.viewMatrix, 16, 'viewMatrix'),
    proj: finiteVector(input.projMatrix, 16, 'projMatrix'),
    pos: finiteVector(cameraPosition, 3, 'cameraPosition'),
  };
}

function finiteAtLeast(value: number | undefined, fallback: number, minimum: number, label: string): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < minimum) {
    throw new RangeError(
      `ProgressiveHandoffCoordinator: ${label} must be a finite number >= ${minimum}.`,
    );
  }
  return resolved;
}

function positiveInteger(value: number | undefined, fallback: number, label: string): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < 1) {
    throw new RangeError(
      `ProgressiveHandoffCoordinator: ${label} must be a positive safe integer.`,
    );
  }
  return resolved;
}

function maxAbsDelta(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = Math.abs((a[i] ?? 0) - (b[i] ?? 0));
    if (d > m) m = d;
  }
  return m;
}

/** Did the camera move beyond `eps` between two snapshots? */
function cameraMoved(prev: CameraSnapshot, next: CameraSnapshot, eps: number): boolean {
  return (
    maxAbsDelta(prev.view, next.view) > eps ||
    maxAbsDelta(prev.proj, next.proj) > eps ||
    maxAbsDelta(prev.pos, next.pos) > eps
  );
}

function flattenAggregateErrors(error: unknown): unknown[] {
  return error instanceof AggregateError
    ? Array.from(error.errors as Iterable<unknown>)
    : [error];
}

function errorDescription(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}


export type ProgressiveHandoffConfiguration = Pick<
  ProgressiveHandoffOptions,
  | 'stillFramesBeforeHandoff'
  | 'cameraEpsilon'
  | 'convergedDisplaySamples'
  | 'seedWeight'
  | 'controllerDeltaSeconds'
>;

/** Validate every numeric handoff option without touching an engine or GPU.
 * The high-level factory calls this before adapter acquisition; the coordinator
 * calls it again so direct low-level construction has the same contract. */
export function assertProgressiveHandoffConfiguration(
  opts: ProgressiveHandoffConfiguration,
): void {
  positiveInteger(opts.stillFramesBeforeHandoff, 6, 'stillFramesBeforeHandoff');
  finiteAtLeast(opts.cameraEpsilon, 1e-5, 0, 'cameraEpsilon');
  positiveInteger(opts.convergedDisplaySamples, 64, 'convergedDisplaySamples');
  finiteAtLeast(opts.seedWeight, 4, 0, 'seedWeight');
  if (
    typeof opts.controllerDeltaSeconds === 'number' &&
    !Number.isFinite(opts.controllerDeltaSeconds)
  ) {
    throw new TypeError(
      'ProgressiveHandoffCoordinator: controllerDeltaSeconds must be finite when supplied as a number.',
    );
  }
}

function frameInputWithDefaultQuality(
  input: FrameInput,
  quality: NonNullable<FrameInput['quality']>,
): FrameInput {
  return input.quality != null ? input : { ...input, quality };
}

/**
 * Coordinator that hands the displayed frame off from a real-time engine to a
 * converged path tracer once the camera settles. See file header.
 */
export class ProgressiveHandoffCoordinator {
  readonly #realtime: Engine;
  readonly #converged: Engine;
  readonly #threshold: number;
  readonly #eps: number;
  readonly #settleBehind: boolean;
  readonly #displaySamples: number;
  readonly #seedFromRealtime: boolean;
  readonly #seedWeight: number;
  readonly #controller: ProgressiveHandoffController | undefined;
  readonly #controllerDeltaSeconds: ProgressiveHandoffControllerDelta;
  readonly #controllerLoop: boolean;
  readonly #controllerTarget: ProgressiveHandoffControllerTarget = {
    setScene: (scene) => this.setScene(scene),
    updatePrimitive: (id, patch) => this.updatePrimitive(id, patch),
    reset: () => this.#resetEnginesAfterControllerMutation(),
  };

  #prev: CameraSnapshot | null = null;
  #stillFrames = 0;
  #phase: HandoffPhase = 'realtime';
  /** The engine whose output was DISPLAYED by the most recent {@link frame} call
   *  (the realtime engine in realtime/settling/prerolling, the converged engine
   *  in converging). Drives {@link getPresentationSource} so a canvas-owning host
   *  presents the converged (offscreen) output after handoff and lets the swapchain
   *  realtime engine present itself before it. */
  #lastActive: Engine | null = null;
  /** The converged accumulator holds a DIFFERENT camera's samples (or none);
   *  reset it before the first converged frame of a settle. */
  #convergedStale = true;
  /** The current converged cohort contains realtime preview radiance and may be
   * displayed progressively, but must be retired before terminal publication. */
  #seededPreviewActive = false;
  #scene: Scene | null;
  /** Last size successfully accepted by both phases. Null until the first
   *  coordinator-routed resize, so a failed first resize cannot be rolled back
   *  and therefore becomes terminal. */
  #size: { readonly width: number; readonly height: number } | null = null;
  /**
   * An unrecoverable split between the two engines. The coordinator does not
   * own either backend, so it cannot safely guess which phase is authoritative
   * after rollback is impossible or fails. Every operation that could render,
   * capture, present, or mutate is blocked from this point on; callers must
   * recreate the pair.
   */
  #terminalError: AggregateError | null = null;

  constructor(opts: ProgressiveHandoffOptions) {
    this.#realtime = opts.realtime;
    this.#converged = opts.converged;
    assertProgressiveHandoffConfiguration(opts);
    this.#threshold = positiveInteger(opts.stillFramesBeforeHandoff, 6, 'stillFramesBeforeHandoff');
    this.#eps = finiteAtLeast(opts.cameraEpsilon, 1e-5, 0, 'cameraEpsilon');
    this.#settleBehind = opts.settleBehindRealtime ?? false;
    this.#displaySamples = positiveInteger(opts.convergedDisplaySamples, 64, 'convergedDisplaySamples');
    this.#seedFromRealtime = opts.seedFromRealtime ?? false;
    this.#seedWeight = finiteAtLeast(opts.seedWeight, 4, 0, 'seedWeight');
    this.#controller = opts.controller;
    this.#controllerDeltaSeconds = opts.controllerDeltaSeconds ?? (1 / 60);
    this.#controllerLoop = opts.controllerLoop ?? true;
    if (opts.scene !== undefined) validateCoreScene(opts.scene);
    this.#scene = opts.scene ?? null;
  }

  /** The phase presented by the most recent {@link frame} call. */
  get phase(): HandoffPhase {
    return this.#phase;
  }

  /** Consecutive still frames as of the last {@link frame} call. */
  get stillFrames(): number {
    return this.#stillFrames;
  }

  /** Authoritative scene shared by both engines, including coordinator-routed patches. */
  getScene(): Scene | null {
    return this.#scene;
  }

  /** Non-null once the two phases could no longer be proven synchronized. */
  get synchronizationError(): AggregateError | null {
    return this.#terminalError;
  }

  /**
   * Force back to real-time and invalidate the converged accumulation. Call when
   * the shared scene changes (both engines must re-`setScene` first) so the next
   * settle re-accumulates from scratch.
   */
  reset(): void {
    this.#prev = null;
    this.#stillFrames = 0;
    this.#phase = 'realtime';
    this.#convergedStale = true;
    this.#seededPreviewActive = false;
  }

  #resetEnginesAfterControllerMutation(): void {
    this.#assertSynchronized('controller reset');
    const errors: unknown[] = [];
    for (const engine of [this.#realtime, this.#converged]) {
      try {
        engine.reset();
      } catch (error) {
        errors.push(error);
      }
    }
    this.reset();
    if (errors.length > 0) {
      throw this.#enterTerminalState('controller reset', errors);
    }
  }

  #assertSynchronized(_operation: string): void {
    if (this.#terminalError == null) return;
    throw this.#terminalError;
  }

  #enterTerminalState(operation: string, errors: readonly unknown[]): AggregateError {
    const primary = errors[0];
    const terminal = new AggregateError(
      errors,
      `ProgressiveHandoffCoordinator entered a terminal synchronization-error state during ${operation}; ` +
        'the realtime and converged engines can no longer be proven equivalent and must be recreated.' +
        (primary === undefined ? '' : ` Primary failure: ${errorDescription(primary)}`),
    );
    this.#terminalError = terminal;
    this.reset();
    return terminal;
  }

  // ── Scene authority ───────────────────────────────────────────────────────
  // The handoff requires BOTH engines to hold the SAME scene; forwarding the
  // mutations here keeps them correct-by-construction (the host can't sync one
  // engine and forget the other) and invalidates the converged accumulation +
  // returns to real-time, since any scene change makes the converged image (and
  // both engines' cached GI) stale. Optional Engine methods are forwarded only
  // when the underlying engine implements them.

  /** Set the scene on both engines and restart at real-time. */
  setScene(scene: Scene): void {
    this.#assertSynchronized('setScene');
    validateCoreScene(scene);
    const previous = this.#scene;
    this.#installSceneOnBoth(scene, previous, 'setScene');
    this.#scene = scene;
    this.reset();
  }

  /**
   * Publish one scene to both engines without touching the coordinator's
   * authoritative `#scene`. If either phase rejects and a previous snapshot is
   * available, both engines are explicitly restored before this returns by
   * throwing. This is the transaction boundary used by public setScene and by
   * incremental-mutation recovery.
   */
  #installSceneOnBoth(
    scene: Scene,
    previous: Scene | null,
    operation: string,
  ): void {
    let forwardError: unknown = null;
    try {
      this.#realtime.setScene(scene);
    } catch (error) {
      forwardError = error;
    }
    if (forwardError == null) {
      try {
        this.#converged.setScene(scene);
      } catch (error) {
        forwardError = error;
      }
    }
    if (forwardError == null) return;
    if (previous == null) {
      throw this.#enterTerminalState(
        operation,
        [forwardError],
      );
    }

    const restorationErrors: unknown[] = [];
    for (const engine of [this.#realtime, this.#converged]) {
      try {
        engine.setScene(previous);
      } catch (error) {
        restorationErrors.push(error);
      }
    }
    // Even a successful restoration invalidates both accumulators. If a
    // backend could not restore, force the coordinator away from converging so
    // no stale phase is presented as authoritative.
    this.reset();
    if (restorationErrors.length > 0) {
      throw this.#enterTerminalState(
        `${operation} rollback`,
        [forwardError, ...restorationErrors],
      );
    }
    throw new AggregateError(
      [forwardError],
      `ProgressiveHandoffCoordinator.${operation} failed (${errorDescription(forwardError)}); ` +
        'both engines were restored to the previous scene.',
    );
  }

  /**
   * Shared skeleton for the primitive mutators (update/add/remove). Each op
   * differs only in (a) the precomputed authoritative `nextScene`, (b) the
   * optional Engine method probed on each sub-engine, (c) the closure that
   * invokes both, and (d) the method name for the "both engines must implement"
   * error. The applied policy — try both engines and commit `nextScene`, else
   * fall back to `setScene(nextScene)` (when an authoritative scene exists) or
   * throw — is identical across all three and lives here once. `reset()` is
   * called by the caller on the success path so the mutator body reads clearly.
   *
   * @returns true when the op ran on both engines (caller should `reset()`);
   *          false when it delegated to `setScene` (which already reset).
   */
  #applyToBothEngines(
    method: 'updatePrimitive' | 'addPrimitive' | 'removePrimitive' | 'updateEmitter' | 'updateEnvironment',
    bothSupported: boolean,
    performRealtime: () => void,
    performConverged: () => void,
    nextScene: Scene | null,
  ): boolean {
    this.#assertSynchronized(method);
    if (!bothSupported) {
      if (nextScene != null) {
        this.setScene(nextScene);
        return false;
      }
      throw new Error(
        `ProgressiveHandoffCoordinator.${method}: both engines must implement ${method} ` +
          'unless the coordinator was constructed with an authoritative scene fallback.',
      );
    }
    try {
      performRealtime();
      performConverged();
      if (nextScene != null) this.#scene = nextScene;
    } catch (forwardError) {
      if (nextScene != null) {
        const previous = this.#scene;
        try {
          this.#installSceneOnBoth(
            nextScene,
            previous,
            `${method} fallback rebuild`,
          );
        } catch (rebuildError) {
          const combined = new AggregateError(
            [forwardError, ...flattenAggregateErrors(rebuildError)],
            `ProgressiveHandoffCoordinator.${method} failed on the incremental path ` +
              `(${errorDescription(forwardError)}) and its full-scene recovery did not commit.`,
          );
          if (this.#terminalError != null) this.#terminalError = combined;
          throw combined;
        }
        this.#scene = nextScene;
        this.reset();
        return false;
      }
      throw this.#enterTerminalState(method, [forwardError]);
    }
    return true;
  }

  /** Patch a primitive on both engines (where supported) and restart at real-time. */
  updatePrimitive(id: string, patch: ScenePrimitivePatch): void {
    const nextScene = this.#scene != null ? patchPrimitiveInScene(this.#scene, id, patch) : null;
    const bothSupported =
      typeof this.#realtime.updatePrimitive === 'function' &&
      typeof this.#converged.updatePrimitive === 'function';
    if (this.#applyToBothEngines(
      'updatePrimitive',
      bothSupported,
      () => this.#realtime.updatePrimitive!(id, patch),
      () => this.#converged.updatePrimitive!(id, patch),
      nextScene,
    )) {
      this.reset();
    }
  }

  /** Add a primitive to both engines (where supported) and restart at real-time. */
  addPrimitive(primitive: ScenePrimitive): void {
    let nextScene: Scene | null = null;
    if (this.#scene != null) {
      if (this.#scene.primitives.some((p) => String(p.id) === String(primitive.id))) {
        throw new Error(`addPrimitive: primitive "${String(primitive.id)}" already exists in current scene`);
      }
      nextScene = { ...this.#scene, primitives: [...this.#scene.primitives, primitive] };
    }
    const bothSupported =
      typeof this.#realtime.addPrimitive === 'function' &&
      typeof this.#converged.addPrimitive === 'function';
    if (this.#applyToBothEngines(
      'addPrimitive',
      bothSupported,
      () => this.#realtime.addPrimitive!(primitive),
      () => this.#converged.addPrimitive!(primitive),
      nextScene,
    )) {
      this.reset();
    }
  }

  /** Remove a primitive from both engines (where supported) and restart at real-time. */
  removePrimitive(id: ScenePrimitive['id']): void {
    let nextScene: Scene | null = null;
    if (this.#scene != null) {
      if (!this.#scene.primitives.some((p) => String(p.id) === String(id))) {
        throw new Error(`removePrimitive: primitive "${String(id)}" not found in current scene`);
      }
      nextScene = {
        ...this.#scene,
        primitives: this.#scene.primitives.filter((p) => String(p.id) !== String(id)),
      };
    }
    const bothSupported =
      typeof this.#realtime.removePrimitive === 'function' &&
      typeof this.#converged.removePrimitive === 'function';
    if (this.#applyToBothEngines(
      'removePrimitive',
      bothSupported,
      () => this.#realtime.removePrimitive!(id),
      () => this.#converged.removePrimitive!(id),
      nextScene,
    )) {
      this.reset();
    }
  }


  /** Patch an emitter on both engines (or rebuild from the authoritative scene). */
  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    const nextScene = this.#scene != null ? patchEmitterInScene(this.#scene, id, patch) : null;
    const bothSupported =
      typeof this.#realtime.updateEmitter === 'function' &&
      typeof this.#converged.updateEmitter === 'function';
    if (this.#applyToBothEngines(
      'updateEmitter',
      bothSupported,
      () => this.#realtime.updateEmitter!(id, patch),
      () => this.#converged.updateEmitter!(id, patch),
      nextScene,
    )) {
      this.reset();
    }
  }

  /** Replace the environment on both engines (or rebuild from the authoritative scene). */
  updateEnvironment(environment: SceneEnvironment | null): void {
    const normalized = environment ?? { kind: 'none' as const };
    const nextScene = this.#scene != null
      ? { ...this.#scene, environment: normalized }
      : null;
    const bothSupported =
      typeof this.#realtime.updateEnvironment === 'function' &&
      typeof this.#converged.updateEnvironment === 'function';
    if (this.#applyToBothEngines(
      'updateEnvironment',
      bothSupported,
      () => this.#realtime.updateEnvironment!(environment),
      () => this.#converged.updateEnvironment!(environment),
      nextScene,
    )) {
      this.reset();
    }
  }

  /**
   * Apply backend runtime-lighting controls to both presentation phases.
   *
   * Unlike emitter/environment scene edits, `updateLighting` is an opaque
   * backend option record and cannot be reconstructed from the authoritative
   * core Scene for rollback. Both methods are therefore required up front. If
   * either backend rejects after fan-out begins, the pair enters the terminal
   * synchronization state rather than continuing with potentially different
   * lighting in the realtime and converged phases.
   */
  updateLighting(opts: Readonly<Record<string, unknown>>): void {
    this.#assertSynchronized('updateLighting');
    if (opts == null || typeof opts !== 'object' || Array.isArray(opts)) {
      throw new TypeError('ProgressiveHandoffCoordinator.updateLighting: opts must be an object.');
    }
    if (
      typeof this.#realtime.updateLighting !== 'function' ||
      typeof this.#converged.updateLighting !== 'function'
    ) {
      throw new Error(
        'ProgressiveHandoffCoordinator.updateLighting: both engines must implement updateLighting.',
      );
    }

    const errors: unknown[] = [];
    for (const engine of [this.#realtime, this.#converged]) {
      try {
        engine.updateLighting!(opts);
      } catch (error) {
        errors.push(error);
      }
    }
    this.reset();
    if (errors.length > 0) {
      throw this.#enterTerminalState('updateLighting', errors);
    }
  }

  /** Resize both phases after validating once at the coordinator boundary. */
  setSize(width: number, height: number): void {
    this.#assertSynchronized('setSize');
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) {
      throw new RangeError(
        'ProgressiveHandoffCoordinator.setSize: width and height must be positive safe integers.',
      );
    }
    const previous = this.#size;
    const forwardErrors: unknown[] = [];
    for (const engine of [this.#realtime, this.#converged]) {
      try {
        engine.setSize?.(width, height);
      } catch (error) {
        // Resize is a facade fan-out: one backend rejecting must not prevent
        // the other presentation phase from observing the host resize.
        forwardErrors.push(error);
      }
    }
    if (forwardErrors.length === 0) {
      this.#size = { width, height };
      this.reset();
      return;
    }

    if (previous == null) {
      throw this.#enterTerminalState('setSize', forwardErrors);
    }

    const restorationErrors: unknown[] = [];
    for (const engine of [this.#realtime, this.#converged]) {
      try {
        engine.setSize?.(previous.width, previous.height);
      } catch (error) {
        restorationErrors.push(error);
      }
    }
    this.reset();
    if (restorationErrors.length > 0) {
      throw this.#enterTerminalState(
        'setSize rollback',
        [...forwardErrors, ...restorationErrors],
      );
    }
    throw new AggregateError(
      forwardErrors,
      `ProgressiveHandoffCoordinator.setSize failed (${errorDescription(forwardErrors[0])}); ` +
        'both phases were restored to the previous size.',
    );
  }

  /** Capture the phase that was actually displayed by the latest frame. Before
   * the first frame, the realtime phase is the defined presentation source. */
  captureFrame(options?: CaptureFrameOptions): Promise<CapturedFrame | null> {
    this.#assertSynchronized('captureFrame');
    const active = this.#lastActive ?? this.#realtime;
    return active.captureFrame?.(options) ?? Promise.resolve(null);
  }
  /**
   * Drive one frame: detect camera motion, pick the engine, render, return its
   * output. The host presents `result.output`.
   */
  frame(input: FrameInput): HandoffFrameResult {
    this.#assertSynchronized('frame');
    this.#advanceController(input);

    // Treat the coordinator state as a presentation publication record.  The
    // engines may throw while resetting, seeding, or rendering; none of those
    // attempts produced a frame the host can safely present.  Compute the next
    // state locally and publish it only after every render needed by the chosen
    // result succeeds.  In particular, a failed first converged attempt leaves
    // #convergedStale armed so the retry resets/seeds from a known boundary.
    const cam = snapshot(input);
    const moved = this.#prev === null || cameraMoved(this.#prev, cam, this.#eps);
    const nextStillFrames = moved ? 0 : this.#stillFrames + 1;
    const nextConvergedStale = moved ? true : this.#convergedStale;
    let nextSeededPreviewActive = moved ? false : this.#seededPreviewActive;
    const publish = (
      phase: HandoffPhase,
      active: Engine,
      convergedStale: boolean,
      seededPreviewActive: boolean,
    ): void => {
      this.#prev = cam;
      this.#stillFrames = nextStillFrames;
      this.#phase = phase;
      this.#lastActive = active;
      this.#convergedStale = convergedStale;
      this.#seededPreviewActive = seededPreviewActive;
    };

    if (nextStillFrames >= this.#threshold) {
      // Hand off to the converged engine. Reset its accumulator on the FIRST
      // converged frame of this settle so it accumulates the current camera.
      if (nextConvergedStale) {
        this.#converged.reset();
        // P8 increment 2: seed the converged accumulator from the real-time engine's
        // last (still-camera) frame — a decaying prior that hides the 1-sample pop
        // without biasing the converged mean. Runs AFTER reset (the seed is the sole
        // prior). No-op unless seedFromRealtime + both engines' capabilities (+ a
        // shared device at runtime).
        nextSeededPreviewActive =
          this.#seedFromRealtime && this.#seedConvergedFromRealtime(input);
      }
      // Always advance the converged engine (it accumulates either way).
      let convOutput = this.#converged.renderFrame(
        frameInputWithDefaultQuality(input, QUALITY_FINAL),
      );
      if (
        nextSeededPreviewActive &&
        (convOutput.isConverged ||
          convOutput.samplesAccumulated >= this.#displaySamples)
      ) {
        // A realtime seed is a presentation prior, not a physical sample. Its
        // finite weight would remain in a finite-SPP terminal mean. Retire the
        // preview cohort before it can become authoritative, then start the
        // canonical accumulator with one real PT sample on this same frame.
        try {
          this.#converged.reset();
          convOutput = this.#converged.renderFrame(
            frameInputWithDefaultQuality(input, QUALITY_FINAL),
          );
          nextSeededPreviewActive = false;
        } catch (error) {
          // Reset may already have changed backend state. Force the next retry
          // through a fresh cohort instead of treating either generation as
          // publishable.
          this.#convergedStale = true;
          this.#seededPreviewActive = false;
          throw error;
        }
      }
      const convReady =
        convOutput.isConverged || convOutput.samplesAccumulated >= this.#displaySamples;
      if (!this.#settleBehind || convReady) {
        publish('converging', this.#converged, false, nextSeededPreviewActive);
        return { phase: 'converging', active: this.#converged, output: convOutput, stillFrames: nextStillFrames };
      }
      // Pre-roll: the converged engine accumulated above (behind the scenes);
      // keep DISPLAYING the smooth real-time image until it is clean enough,
      // hiding the real-time → 1-sample pop.
      const rtOutput = this.#realtime.renderFrame(
        frameInputWithDefaultQuality(input, QUALITY_PREVIEW),
      );
      publish('prerolling', this.#realtime, false, nextSeededPreviewActive);
      return {
        phase: 'prerolling',
        active: this.#realtime,
        output: rtOutput,
        behindOutput: convOutput,
        stillFrames: nextStillFrames,
      };
    }

    // Real-time: moving, or still-but-settling (below the threshold).
    const nextPhase: HandoffPhase = nextStillFrames > 0 ? 'settling' : 'realtime';
    const output = this.#realtime.renderFrame(
      frameInputWithDefaultQuality(input, QUALITY_PREVIEW),
    );
    publish(nextPhase, this.#realtime, nextConvergedStale, nextSeededPreviewActive);
    return { phase: nextPhase, active: this.#realtime, output, stillFrames: nextStillFrames };
  }

  /**
   * Presentation SOURCE for a canvas-owning host (V1-1 / R2). Delegates to the
   * engine that was DISPLAYED by the most recent {@link frame} call:
   *  - realtime/settling/prerolling → the realtime (swapchain) engine, which
   *    presents itself, so this returns `null` (nothing for the host to blit).
   *  - converging → the converged (offscreen pt-webgpu) engine's source, so the
   *    host blits its texture to the canvas — this is what unfreezes the display
   *    after handoff.
   *
   * Returns `null` before the first frame or whenever the active engine does not
   * expose `getPresentationSource` / has nothing to present.
   */
  getPresentationSource(): { device: unknown; texture: import('@vitrum/core').BackendTexture } | null {
    this.#assertSynchronized('getPresentationSource');
    const active = this.#lastActive;
    if (active == null) return null;
    if (typeof active.getPresentationSource !== 'function') return null;
    return active.getPresentationSource();
  }

  /** Seed the converged accumulator from the real-time engine's last frame (the
   *  smooth still-camera image). No-op unless BOTH engines expose the optional
   *  source/sink methods (`getProgressiveSeedTexture` / `seedAccumulator`); at
   *  runtime they must also share one GPUDevice (else the cross-device texture bind
   *  throws — the host wires this via `createProgressiveEngine`).
   *
   *  `input` is the current frame input; its viewport carries the DESTINATION
   *  accumulator dims (the converged engine accumulates at `viewport × its internal
   *  resolutionFactor`).  Per the contract, `seedAccumulator` `opts.width`/`height`
   *  are the DESTINATION dims (not the source), so we derive them from the viewport
   *  rather than the source texture dims — which may differ when source and dest have
   *  different resolutionFactors.  The backend resamples the seed texture to fit. */
  #seedConvergedFromRealtime(input: FrameInput): boolean {
    const src = this.#realtime.getProgressiveSeedTexture?.();
    if (
      src == null ||
      this.#seedWeight <= 0 ||
      typeof this.#converged.seedAccumulator !== 'function'
    ) {
      return false;
    }
    const destWidth = input.viewport?.width ?? src.width;
    const destHeight = input.viewport?.height ?? src.height;
    this.#converged.seedAccumulator(src.texture, {
      weight: this.#seedWeight,
      width: destWidth,
      height: destHeight,
    });
    return true;
  }

  #advanceController(input: FrameInput): void {
    const controller = this.#controller;
    if (controller == null) return;
    if (controller.animations != null && controller.animations.length === 0) return;

    const delta = typeof this.#controllerDeltaSeconds === 'function'
      ? this.#controllerDeltaSeconds(input, {
        phase: this.#phase,
        stillFrames: this.#stillFrames,
      })
      : this.#controllerDeltaSeconds;
    if (!Number.isFinite(delta)) {
      throw new TypeError(
        'ProgressiveHandoffCoordinator.frame: controllerDeltaSeconds must resolve to a finite number.',
      );
    }
    if (delta === 0) return;
    controller.advance(delta, {
      engine: this.#controllerTarget,
      loop: this.#controllerLoop,
    });
  }
}
