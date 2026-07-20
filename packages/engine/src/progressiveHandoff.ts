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

import { patchPrimitiveInScene } from '@vitrum/core';
import type { Engine, FrameInput, FrameOutput, Scene, ScenePrimitive } from '@vitrum/core';

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
  updatePrimitive?(id: string, patch: Partial<ScenePrimitive>): void;
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
   *  the real-time engine's last frame (a DECAYING PRIOR — `seedAccumulator`) so the
   *  converged image starts from the smooth real-time output instead of a 1-sample
   *  blizzard, WITHOUT biasing the converged mean. Requires the real-time engine to
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

function snapshot(input: FrameInput): CameraSnapshot {
  return {
    view: Float32Array.from(input.viewMatrix as unknown as ArrayLike<number>),
    proj: Float32Array.from(input.projMatrix as unknown as ArrayLike<number>),
    pos: Float32Array.from(input.cameraPosition as unknown as ArrayLike<number>),
  };
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
  #scene: Scene | null;

  constructor(opts: ProgressiveHandoffOptions) {
    this.#realtime = opts.realtime;
    this.#converged = opts.converged;
    this.#threshold = Math.max(1, Math.floor(opts.stillFramesBeforeHandoff ?? 6));
    this.#eps = opts.cameraEpsilon ?? 1e-5;
    this.#settleBehind = opts.settleBehindRealtime ?? false;
    this.#displaySamples = Math.max(1, Math.floor(opts.convergedDisplaySamples ?? 64));
    this.#seedFromRealtime = opts.seedFromRealtime ?? false;
    this.#seedWeight = Math.max(0, opts.seedWeight ?? 4);
    this.#controller = opts.controller;
    this.#controllerDeltaSeconds = opts.controllerDeltaSeconds ?? (1 / 60);
    this.#controllerLoop = opts.controllerLoop ?? true;
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
  }

  #resetEnginesAfterControllerMutation(): void {
    this.#realtime.reset();
    this.#converged.reset();
    this.reset();
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
    this.#realtime.setScene(scene);
    this.#converged.setScene(scene);
    this.#scene = scene;
    this.reset();
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
    method: 'updatePrimitive' | 'addPrimitive' | 'removePrimitive',
    bothSupported: boolean,
    perform: () => void,
    nextScene: Scene | null,
  ): boolean {
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
      perform();
      if (nextScene != null) this.#scene = nextScene;
    } catch (err) {
      if (nextScene != null) {
        this.setScene(nextScene);
        return false;
      }
      throw err;
    }
    return true;
  }

  /** Patch a primitive on both engines (where supported) and restart at real-time. */
  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    const nextScene = this.#scene != null ? patchPrimitiveInScene(this.#scene, id, patch) : null;
    const bothSupported =
      typeof this.#realtime.updatePrimitive === 'function' &&
      typeof this.#converged.updatePrimitive === 'function';
    if (this.#applyToBothEngines('updatePrimitive', bothSupported, () => {
      this.#realtime.updatePrimitive!(id, patch);
      this.#converged.updatePrimitive!(id, patch);
    }, nextScene)) {
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
    if (this.#applyToBothEngines('addPrimitive', bothSupported, () => {
      this.#realtime.addPrimitive!(primitive);
      this.#converged.addPrimitive!(primitive);
    }, nextScene)) {
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
    if (this.#applyToBothEngines('removePrimitive', bothSupported, () => {
      this.#realtime.removePrimitive!(id);
      this.#converged.removePrimitive!(id);
    }, nextScene)) {
      this.reset();
    }
  }

  /**
   * Drive one frame: detect camera motion, pick the engine, render, return its
   * output. The host presents `result.output`.
   */
  frame(input: FrameInput): HandoffFrameResult {
    this.#advanceController(input);

    const cam = snapshot(input);
    const moved = this.#prev === null || cameraMoved(this.#prev, cam, this.#eps);
    this.#prev = cam;

    if (moved) {
      this.#stillFrames = 0;
      // The converged accumulator (if any) is now for a stale camera.
      this.#convergedStale = true;
    } else {
      this.#stillFrames += 1;
    }

    if (this.#stillFrames >= this.#threshold) {
      // Hand off to the converged engine. Reset its accumulator on the FIRST
      // converged frame of this settle so it accumulates the current camera.
      if (this.#convergedStale) {
        this.#converged.reset();
        // P8 increment 2: seed the converged accumulator from the real-time engine's
        // last (still-camera) frame — a decaying prior that hides the 1-sample pop
        // without biasing the converged mean. Runs AFTER reset (the seed is the sole
        // prior). No-op unless seedFromRealtime + both engines' capabilities (+ a
        // shared device at runtime).
        if (this.#seedFromRealtime) this.#seedConvergedFromRealtime(input);
        this.#convergedStale = false;
      }
      // Always advance the converged engine (it accumulates either way).
      const convOutput = this.#converged.renderFrame(input);
      const convReady =
        convOutput.isConverged || convOutput.samplesAccumulated >= this.#displaySamples;
      if (!this.#settleBehind || convReady) {
        this.#phase = 'converging';
        this.#lastActive = this.#converged;
        return { phase: 'converging', active: this.#converged, output: convOutput, stillFrames: this.#stillFrames };
      }
      // Pre-roll: the converged engine accumulated above (behind the scenes);
      // keep DISPLAYING the smooth real-time image until it is clean enough,
      // hiding the real-time → 1-sample pop.
      this.#phase = 'prerolling';
      const rtOutput = this.#realtime.renderFrame(input);
      this.#lastActive = this.#realtime;
      return {
        phase: 'prerolling',
        active: this.#realtime,
        output: rtOutput,
        behindOutput: convOutput,
        stillFrames: this.#stillFrames,
      };
    }

    // Real-time: moving, or still-but-settling (below the threshold).
    this.#phase = this.#stillFrames > 0 ? 'settling' : 'realtime';
    const output = this.#realtime.renderFrame(input);
    this.#lastActive = this.#realtime;
    return { phase: this.#phase, active: this.#realtime, output, stillFrames: this.#stillFrames };
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
  #seedConvergedFromRealtime(input: FrameInput): void {
    const src = this.#realtime.getProgressiveSeedTexture?.();
    if (src == null) return;
    const destWidth = input.viewport?.width ?? src.width;
    const destHeight = input.viewport?.height ?? src.height;
    this.#converged.seedAccumulator?.(src.texture, {
      weight: this.#seedWeight,
      width: destWidth,
      height: destHeight,
    });
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
