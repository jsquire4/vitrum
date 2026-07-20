// numberRing.ts — fixed-size circular moving-average buffer.
//
// D2-5 — one implementation shared by both HUDs (React `FrameTimeHUD` +
// vanilla `addFrameTimeHud`). The canonical class keeps a running `sum` so
// `mean()` is O(1) (previously the vanilla `NumberRing.mean()` re-summed the
// window every call). `RingBuffer` is a back-compat alias for the React HUD's
// prior local class (same surface: `push`/`mean`/`filled`/`capacity`).

import type { FrameStats } from '../types.js';
import type { DebuggableEngine } from '../types.js';

/**
 * Fixed-capacity ring buffer that maintains a running sum for O(1) `mean()`.
 * Overwrites the oldest sample once full.
 */
export class NumberRing {
  readonly capacity: number;
  #buf: Float64Array;
  #head = 0;
  #count = 0;
  #sum = 0;

  constructor(size: number) {
    this.capacity = Math.max(1, size);
    this.#buf = new Float64Array(this.capacity);
  }

  push(value: number): void {
    const old = this.#buf[this.#head] ?? 0;
    this.#sum -= old;
    this.#buf[this.#head] = value;
    this.#sum += value;
    this.#head = (this.#head + 1) % this.capacity;
    if (this.#count < this.capacity) this.#count++;
  }

  mean(): number {
    return this.#count === 0 ? 0 : this.#sum / this.#count;
  }

  get filled(): number {
    return this.#count;
  }
}

/** Back-compat alias — the React `FrameTimeHUD` re-exports this name. */
export const RingBuffer = NumberRing;
export type RingBuffer = NumberRing;

/**
 * D2-6 — shared frame-time observer used by both HUDs. Subscribes to
 * `engine.onFrame` when present (T3.E), otherwise falls back to rAF wall-clock
 * deltas, pushes each sample into `ring`, and invokes `onSample(stats)` after
 * each push (the HUD renders `stats` + `ring.mean()`). Returns an unsubscribe
 * that drains the active source (onFrame unsub or the pending rAF).
 */
export function observeFrameTime(
  engine: DebuggableEngine,
  ring: NumberRing,
  onSample: (stats: FrameStats) => void,
): () => void {
  const record = (stats: FrameStats): void => {
    ring.push(stats.frameTimeMs);
    onSample(stats);
  };

  // ── Path A: engine.onFrame present (T3.E) ──────────────────────────────
  if (typeof engine.onFrame === 'function') {
    return engine.onFrame(record);
  }

  // ── Path B: rAF fallback — synthetic FrameStats from wall-clock delta. ──
  let lastTime: number | null = null;
  let rafId: number | null = null;
  const tick = (now: number): void => {
    if (lastTime !== null) {
      record({ frameTimeMs: now - lastTime });
    }
    lastTime = now;
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
  return () => {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  };
}
