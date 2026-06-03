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

import type { Engine, FrameInput, FrameOutput } from '@vitrum/core';

export interface ProgressiveHandoffOptions {
  /** Smooth real-time GI engine — driven while the camera moves. */
  readonly realtime: Engine;
  /** Converged path-tracing engine — driven (accumulating) while the camera is still. */
  readonly converged: Engine;
  /** Consecutive still frames before handing off to the converged engine. The
   *  intervening frames keep showing the real-time engine ("settling"), so a
   *  brief pause doesn't thrash the handoff. Default 6; clamped to >= 1. */
  readonly stillFramesBeforeHandoff?: number;
  /** Max-abs camera-delta (view/proj matrices + position) below which a frame
   *  counts as "still". Default 1e-5. */
  readonly cameraEpsilon?: number;
}

/** Which engine the coordinator is presenting. */
export type HandoffPhase =
  /** Camera moving (or just moved) — real-time engine. */
  | 'realtime'
  /** Camera still but not yet past the handoff threshold — still real-time. */
  | 'settling'
  /** Camera still past the threshold — converged engine, accumulating. */
  | 'converging';

export interface HandoffFrameResult {
  readonly phase: HandoffPhase;
  /** The engine that rendered this frame (its output is `output`). */
  readonly active: Engine;
  readonly output: FrameOutput;
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

  #prev: CameraSnapshot | null = null;
  #stillFrames = 0;
  #phase: HandoffPhase = 'realtime';
  /** The converged accumulator holds a DIFFERENT camera's samples (or none);
   *  reset it before the first converged frame of a settle. */
  #convergedStale = true;

  constructor(opts: ProgressiveHandoffOptions) {
    this.#realtime = opts.realtime;
    this.#converged = opts.converged;
    this.#threshold = Math.max(1, Math.floor(opts.stillFramesBeforeHandoff ?? 6));
    this.#eps = opts.cameraEpsilon ?? 1e-5;
  }

  /** The phase presented by the most recent {@link frame} call. */
  get phase(): HandoffPhase {
    return this.#phase;
  }

  /** Consecutive still frames as of the last {@link frame} call. */
  get stillFrames(): number {
    return this.#stillFrames;
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

  /**
   * Drive one frame: detect camera motion, pick the engine, render, return its
   * output. The host presents `result.output`.
   */
  frame(input: FrameInput): HandoffFrameResult {
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
        this.#convergedStale = false;
      }
      this.#phase = 'converging';
      const output = this.#converged.renderFrame(input);
      return { phase: 'converging', active: this.#converged, output, stillFrames: this.#stillFrames };
    }

    // Real-time: moving, or still-but-settling (below the threshold).
    this.#phase = this.#stillFrames > 0 ? 'settling' : 'realtime';
    const output = this.#realtime.renderFrame(input);
    return { phase: this.#phase, active: this.#realtime, output, stillFrames: this.#stillFrames };
  }
}

// Re-export the camera-delta helpers so hosts / tests can reuse the exact
// motion-detection the coordinator uses.
export const __testing = { maxAbsDelta, cameraMoved };
