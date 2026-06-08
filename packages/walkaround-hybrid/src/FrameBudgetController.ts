/**
 * FrameBudgetController — closed-loop adaptive quality (review gap D1,
 * `plan/integration-depth-plan-2026-06-03.md` §"Phase IV.1").
 *
 * The realtime backend has the quality LEVERS — `resolutionFactor` (internal
 * render scale, honoured per-frame via `FrameInput.quality.resolutionFactor`)
 * `ddgiUpdateDivisor` (the DDGI probe round-robin stride, runtime-settable
 * via `HybridEngine.setDdgiUpdateDivisor`), and, when PPG is enabled, the
 * PPG update-pass dispatch interval — but, before this controller, NO
 * closed loop drove them from MEASURED frame time. "Runs at 30–60 fps on Class
 * B" was an external claim, not a property the library measured + maintained.
 *
 * This controller closes that loop. It is a PURE, deterministic control law:
 * `update(measuredMs)` is a function of `measuredMs` + the controller's own
 * internal state ONLY — no GPU, no clock, no I/O — so the full adaptation policy
 * is unit-testable on CPU without a device (which is the whole point: a
 * deterministic ms-injection test, not a flaky live-perf gate).
 *
 * Deliberately NOT in scope (it adapts the two continuous COST knobs, not the
 * algorithm graph):
 *   - It does NOT toggle passes / pick denoisers / switch quality TIERS / enable
 *     or disable RC·PPG·NRC. Those are discrete structural choices (pass-graph
 *     compile decisions, resource allocation) that the host makes deliberately,
 *     not something to flip from a frame-time wobble.
 *   - It does NOT schedule itself. Consistent with the engine's "host owns
 *     cadence" principle, the host (or a per-frame hook) feeds it `measuredMs`
 *     and applies the returned knobs. The controller holds no timer.
 *   - It does NOT own the GPU-timing source. The caller reads ms however it
 *     likes (`HybridEngine.readGpuTimingsOnce()` for confirmed GPU ms, or the
 *     wall-clock `FrameStats.frameTimeMs` from an `onFrame` subscriber) and
 *     passes the number in. Keeping the source out of the controller is what
 *     makes it GPU-free testable.
 *
 * ── Control law (why it won't oscillate) ─────────────────────────────────────
 *
 *  1. EMA SMOOTHING. The raw per-frame ms is noisy (GC pauses, OS scheduling,
 *     a single heavy frame). We low-pass it: `ema ← α·measured + (1−α)·ema`.
 *     The controller reacts to SUSTAINED load, not single-frame spikes, so a
 *     lone janky frame does not yank the resolution down.
 *
 *  2. HYSTERESIS BAND (the core anti-oscillation mechanism). There is a
 *     dead-zone around the target. We only adapt:
 *        - DOWN (cheaper) when `ema > target · (1 + overBudgetBand)`,
 *        - UP   (richer)  when `ema < target · (1 − underBudgetBand)`.
 *     Between those two thresholds → NO change. Because the up-threshold sits
 *     strictly below the down-threshold, a knob change that moves ms across the
 *     target lands INSIDE the dead-zone and cannot immediately trigger the
 *     reverse action. That gap is the oscillation guard: a limit cycle would
 *     require ms to cross the entire band twice per period, which a single
 *     bounded step cannot do.
 *
 *  3. ASYMMETRIC STEP + UP-COOLDOWN. Dropping quality is urgent (the user is
 *     dropping frames NOW), so the down-step is the full `resolutionStep` with
 *     no cooldown. Restoring quality is speculative (headroom might be
 *     transient), so up-moves use the same bounded step but are rate-limited:
 *     at most one up-move every `upCooldownFrames` controller ticks. This makes
 *     the loop fall fast and rise slow — the standard dynamic-resolution
 *     shape — and prevents a tug-of-war at the threshold.
 *
 *  4. BOUNDED SINGLE STEP. Each `update` moves a knob by at most one step
 *     (`resolutionStep` for resolution; ±1 for the DDGI stride). No knob can
 *     jump from floor to ceiling in one tick, so the response is a gradual ramp
 *     the EMA can track — never a slam that overshoots and forces a correction.
 *
 *  5. CLAMPS. `resolutionFactor ∈ [minResolutionFactor, maxResolutionFactor]`,
 *     `ddgiStride ∈ [minDdgiStride, maxDdgiStride]` (integer). The knobs can
 *     never leave their valid ranges however long the loop runs.
 *
 *  6. ORDERED LEVERS (knobs don't fight). Resolution is the PRIMARY lever
 *     (largest, smoothest cost/ms tradeoff). Optional PPG cadence is next when
 *     enabled: the learned tree still feeds guided sampling every frame, so
 *     widening the train interval is a cheaper freshness trade than degrading
 *     probe updates. DDGI stride is the final fallback. At most one lever moves
 *     per tick, and they never push in opposite directions in the same frame.
 *
 * The defaults target ~60 fps (16.6 ms) with a symmetric ±12 % dead-zone, a
 * 0.05 resolution step, a 0.5 floor / 1.0 ceiling on resolution, a 2…32 DDGI
 * stride range (matching the quality-preset 2→32 spread), a 1…4 PPG train
 * interval range (inactive unless explicitly enabled), and a 30-tick
 * up-cooldown (~0.5 s at 60 fps). All are overridable.
 */

/** Tunable knobs for the control law. All optional; see {@link DEFAULT_FRAME_BUDGET_CONFIG}. */
export interface FrameBudgetControllerConfig {
  /** Target frame time in ms (the setpoint). e.g. 16.6 ≈ 60 fps, 33.3 ≈ 30 fps. */
  readonly targetMs: number;
  /** EMA smoothing factor in (0, 1]. Higher ⇒ reacts faster but noisier; lower
   *  ⇒ smoother but laggier. Default 0.2 (≈ a 5-frame effective window). */
  readonly emaAlpha: number;
  /** Upper dead-zone, as a fraction of target. Adapt DOWN only when
   *  `ema > target·(1+overBudgetBand)`. Default 0.12 (+12 %). */
  readonly overBudgetBand: number;
  /** Lower dead-zone, as a fraction of target. Adapt UP only when
   *  `ema < target·(1−underBudgetBand)`. Default 0.12 (−12 %). MUST keep the up-
   *  threshold below the down-threshold (validated in the ctor) so the dead-zone
   *  is non-empty — that is the oscillation guard. */
  readonly underBudgetBand: number;
  /** Resolution-factor change per adaptation step. Default 0.05. */
  readonly resolutionStep: number;
  /** Minimum (floor) internal-resolution factor. Default 0.5. */
  readonly minResolutionFactor: number;
  /** Maximum (ceiling) internal-resolution factor. Default 1.0. */
  readonly maxResolutionFactor: number;
  /** Minimum DDGI probe-update stride (snappiest GI). Default 2 (preset ultra). */
  readonly minDdgiStride: number;
  /** Maximum DDGI probe-update stride (cheapest GI). Default 32 (preset low). */
  readonly maxDdgiStride: number;
  /** Whether the controller may adapt PPG train-pass cadence. Defaults false
   *  so engines without PPG never spend controller ticks on a no-op lever. */
  readonly adaptPpgDispatchInterval: boolean;
  /** Minimum PPG train-pass interval. Default 1 (train every frame). */
  readonly minPpgDispatchInterval: number;
  /** Maximum PPG train-pass interval. Default 4 (quality-preset low). */
  readonly maxPpgDispatchInterval: number;
  /** Minimum controller ticks between consecutive UP moves (rate-limits the
   *  speculative quality-restore so it rises slowly). Default 30 (~0.5 s @60fps).
   *  Down moves are never rate-limited. */
  readonly upCooldownFrames: number;
}

export const DEFAULT_FRAME_BUDGET_CONFIG: FrameBudgetControllerConfig = Object.freeze({
  targetMs: 16.6,
  emaAlpha: 0.2,
  overBudgetBand: 0.12,
  underBudgetBand: 0.12,
  resolutionStep: 0.05,
  minResolutionFactor: 0.5,
  maxResolutionFactor: 1.0,
  minDdgiStride: 2,
  maxDdgiStride: 32,
  adaptPpgDispatchInterval: false,
  minPpgDispatchInterval: 1,
  maxPpgDispatchInterval: 4,
  upCooldownFrames: 30,
});

/** The lever the controller moved on a given tick (for telemetry / tests). */
export type FrameBudgetAction =
  | 'none'
  | 'resolution-down'
  | 'resolution-up'
  | 'ppg-interval-up'
  | 'ppg-interval-down'
  | 'ddgi-stride-up'
  | 'ddgi-stride-down';

/** Result of one {@link FrameBudgetController.update} tick: the knob values the
 *  host should apply this frame, plus the smoothed ms + the action taken (the
 *  latter two are diagnostic — `resolutionFactor` + `ddgiStride` are the
 *  load-bearing outputs). */
export interface FrameBudgetDecision {
  /** Internal-resolution factor to feed as `FrameInput.quality.resolutionFactor`
   *  (or via the engine convenience hook). Always in
   *  [minResolutionFactor, maxResolutionFactor]. */
  readonly resolutionFactor: number;
  /** DDGI probe-update divisor to apply via `HybridEngine.setDdgiUpdateDivisor`.
   *  Always an integer in [minDdgiStride, maxDdgiStride]. */
  readonly ddgiStride: number;
  /** PPG train-pass dispatch interval to apply via
   *  `HybridEngine.setPpgDispatchInterval`. Always an integer in
   *  [minPpgDispatchInterval, maxPpgDispatchInterval]. */
  readonly ppgDispatchInterval: number;
  /** The EMA-smoothed frame time the decision was based on (ms). */
  readonly smoothedMs: number;
  /** Which lever (if any) moved this tick. */
  readonly action: FrameBudgetAction;
}

const EPS = 1e-9;

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/**
 * A deterministic frame-budget controller. Construct once, then call
 * {@link update} once per frame with a measured ms; apply the returned knobs.
 * Holds NO GPU/clock/I/O state — purely a function of injected ms + internal
 * EMA / knob / cooldown state.
 */
export class FrameBudgetController {
  private readonly cfg: FrameBudgetControllerConfig;
  /** EMA-smoothed frame time; `null` until the first sample seeds it. */
  private ema: number | null = null;
  private resolutionFactor: number;
  private ddgiStride: number;
  private ppgDispatchInterval: number;
  /** Tick counter, used only for the up-move cooldown. */
  private tick = 0;
  /** Tick index of the last UP move (resolution-up, ppg-interval-down, or
   *  ddgi-stride-down); the cooldown compares against this. `-Infinity` ⇒ no
   *  cooldown initially. */
  private lastUpMoveTick = Number.NEGATIVE_INFINITY;

  /**
   * @param config        Partial override of {@link DEFAULT_FRAME_BUDGET_CONFIG}.
   * @param initial       Starting knob values. Default: full resolution (1.0)
   *                      and the minimum DDGI/PPG cadence values (snappiest),
   *                      i.e. start at best quality and let the loop back off under load.
   *                      Clamped into range at construction.
   */
  constructor(
    config: Partial<FrameBudgetControllerConfig> = {},
    initial?: { resolutionFactor?: number; ddgiStride?: number; ppgDispatchInterval?: number },
  ) {
    const cfg = { ...DEFAULT_FRAME_BUDGET_CONFIG, ...config };

    // Validate the dead-zone is non-empty AND brackets the target: the up-
    // threshold (target·(1−under)) must be strictly below the down-threshold
    // (target·(1+over)). With non-negative bands this always holds, but a
    // caller could pass a negative band and collapse/invert the dead-zone,
    // which would let one tick trigger both directions ⇒ oscillation. Guard it.
    if (!(cfg.targetMs > 0) || !Number.isFinite(cfg.targetMs)) {
      throw new RangeError(`[FrameBudgetController] targetMs must be a positive finite ms; got ${cfg.targetMs}`);
    }
    if (!(cfg.emaAlpha > 0) || cfg.emaAlpha > 1) {
      throw new RangeError(`[FrameBudgetController] emaAlpha must be in (0, 1]; got ${cfg.emaAlpha}`);
    }
    const downThresh = cfg.targetMs * (1 + cfg.overBudgetBand);
    const upThresh = cfg.targetMs * (1 - cfg.underBudgetBand);
    if (!(upThresh < downThresh)) {
      throw new RangeError(
        `[FrameBudgetController] dead-zone is empty/inverted: up-threshold ${upThresh.toFixed(3)}ms ` +
          `must be < down-threshold ${downThresh.toFixed(3)}ms (overBudgetBand=${cfg.overBudgetBand}, ` +
          `underBudgetBand=${cfg.underBudgetBand}). A non-positive band collapses the hysteresis guard.`,
      );
    }
    if (!(cfg.minResolutionFactor > 0) || !(cfg.maxResolutionFactor >= cfg.minResolutionFactor)) {
      throw new RangeError(
        `[FrameBudgetController] need 0 < minResolutionFactor ≤ maxResolutionFactor; got ` +
          `[${cfg.minResolutionFactor}, ${cfg.maxResolutionFactor}]`,
      );
    }
    if (!(cfg.minDdgiStride >= 1) || !(cfg.maxDdgiStride >= cfg.minDdgiStride)) {
      throw new RangeError(
        `[FrameBudgetController] need 1 ≤ minDdgiStride ≤ maxDdgiStride; got ` +
          `[${cfg.minDdgiStride}, ${cfg.maxDdgiStride}]`,
      );
    }
    if (!(cfg.minPpgDispatchInterval >= 1) ||
        !(cfg.maxPpgDispatchInterval >= cfg.minPpgDispatchInterval)) {
      throw new RangeError(
        `[FrameBudgetController] need 1 ≤ minPpgDispatchInterval ≤ maxPpgDispatchInterval; got ` +
          `[${cfg.minPpgDispatchInterval}, ${cfg.maxPpgDispatchInterval}]`,
      );
    }
    this.cfg = Object.freeze(cfg);

    this.resolutionFactor = clamp(
      initial?.resolutionFactor ?? cfg.maxResolutionFactor,
      cfg.minResolutionFactor,
      cfg.maxResolutionFactor,
    );
    this.ddgiStride = clamp(
      Math.round(initial?.ddgiStride ?? cfg.minDdgiStride),
      cfg.minDdgiStride,
      cfg.maxDdgiStride,
    );
    this.ppgDispatchInterval = clamp(
      Math.round(initial?.ppgDispatchInterval ?? cfg.minPpgDispatchInterval),
      cfg.minPpgDispatchInterval,
      cfg.maxPpgDispatchInterval,
    );
  }

  /** Current knob values + smoothed ms WITHOUT advancing the loop (read-only
   *  snapshot — useful for telemetry between frames). */
  snapshot(): FrameBudgetDecision {
    return {
      resolutionFactor: this.resolutionFactor,
      ddgiStride: this.ddgiStride,
      ppgDispatchInterval: this.ppgDispatchInterval,
      smoothedMs: this.ema ?? 0,
      action: 'none',
    };
  }

  /** The resolved (frozen) config this controller runs with. */
  get config(): FrameBudgetControllerConfig {
    return this.cfg;
  }

  /**
   * Feed one measured frame time (ms) and get back the knob values to apply.
   * DETERMINISTIC: identical (measuredMs sequence) ⇒ identical decisions.
   *
   * Non-finite / non-positive `measuredMs` are ignored (the EMA + knobs are
   * held, the current values returned with `action:'none'`) — a bogus timing
   * read must not perturb the loop.
   */
  update(measuredMs: number): FrameBudgetDecision {
    if (!Number.isFinite(measuredMs) || measuredMs <= 0) {
      return this.snapshot();
    }
    this.tick++;

    // 1. EMA smoothing — seed on the first sample, low-pass thereafter.
    this.ema = this.ema === null
      ? measuredMs
      : this.cfg.emaAlpha * measuredMs + (1 - this.cfg.emaAlpha) * this.ema;
    const ema = this.ema;

    const downThresh = this.cfg.targetMs * (1 + this.cfg.overBudgetBand);
    const upThresh = this.cfg.targetMs * (1 - this.cfg.underBudgetBand);

    let action: FrameBudgetAction = 'none';

    if (ema > downThresh + EPS) {
      // ── OVER BUDGET → make it cheaper. Primary lever first (resolution),
      //    then optional PPG cadence, then DDGI stride. No cooldown: dropping
      //    quality is urgent.
      if (this.resolutionFactor > this.cfg.minResolutionFactor + EPS) {
        this.resolutionFactor = clamp(
          this.resolutionFactor - this.cfg.resolutionStep,
          this.cfg.minResolutionFactor,
          this.cfg.maxResolutionFactor,
        );
        action = 'resolution-down';
      } else if (
        this.cfg.adaptPpgDispatchInterval &&
        this.ppgDispatchInterval < this.cfg.maxPpgDispatchInterval
      ) {
        this.ppgDispatchInterval = clamp(
          this.ppgDispatchInterval + 1,
          this.cfg.minPpgDispatchInterval,
          this.cfg.maxPpgDispatchInterval,
        );
        action = 'ppg-interval-up';
      } else if (this.ddgiStride < this.cfg.maxDdgiStride) {
        this.ddgiStride = clamp(this.ddgiStride + 1, this.cfg.minDdgiStride, this.cfg.maxDdgiStride);
        action = 'ddgi-stride-up';
      }
      // else: both knobs already at their cheapest — nothing more to give.
    } else if (ema < upThresh - EPS) {
      // ── UNDER BUDGET → spend the headroom (richer). Rate-limited by the
      //    up-cooldown so quality rises slowly (headroom may be transient).
      //    Restore the PRIMARY lever first (resolution up to ceiling), then PPG
      //    cadence, then DDGI stride toward snappier GI.
      const cooldownElapsed = this.tick - this.lastUpMoveTick >= this.cfg.upCooldownFrames;
      if (cooldownElapsed) {
        if (this.resolutionFactor < this.cfg.maxResolutionFactor - EPS) {
          this.resolutionFactor = clamp(
            this.resolutionFactor + this.cfg.resolutionStep,
            this.cfg.minResolutionFactor,
            this.cfg.maxResolutionFactor,
          );
          action = 'resolution-up';
          this.lastUpMoveTick = this.tick;
        } else if (
          this.cfg.adaptPpgDispatchInterval &&
          this.ppgDispatchInterval > this.cfg.minPpgDispatchInterval
        ) {
          this.ppgDispatchInterval = clamp(
            this.ppgDispatchInterval - 1,
            this.cfg.minPpgDispatchInterval,
            this.cfg.maxPpgDispatchInterval,
          );
          action = 'ppg-interval-down';
          this.lastUpMoveTick = this.tick;
        } else if (this.ddgiStride > this.cfg.minDdgiStride) {
          this.ddgiStride = clamp(this.ddgiStride - 1, this.cfg.minDdgiStride, this.cfg.maxDdgiStride);
          action = 'ddgi-stride-down';
          this.lastUpMoveTick = this.tick;
        }
        // else: both knobs already at their richest — nothing to restore.
      }
    }
    // else: inside the dead-zone — hold (action stays 'none'). This is the band
    // that absorbs a knob change which moved ms across the target, breaking the
    // limit cycle.

    return {
      resolutionFactor: this.resolutionFactor,
      ddgiStride: this.ddgiStride,
      ppgDispatchInterval: this.ppgDispatchInterval,
      smoothedMs: ema,
      action,
    };
  }

  /** Reset the EMA + cooldown (e.g. after a scene/quality-tier change makes the
   *  prior frame-time history meaningless). Knob values are retained unless
   *  `resetKnobs` is set, in which case they snap back to the configured
   *  ceiling resolution / floor stride. */
  reset(resetKnobs = false): void {
    this.ema = null;
    this.tick = 0;
    this.lastUpMoveTick = Number.NEGATIVE_INFINITY;
    if (resetKnobs) {
      this.resolutionFactor = this.cfg.maxResolutionFactor;
      this.ddgiStride = this.cfg.minDdgiStride;
      this.ppgDispatchInterval = this.cfg.minPpgDispatchInterval;
    }
  }
}
