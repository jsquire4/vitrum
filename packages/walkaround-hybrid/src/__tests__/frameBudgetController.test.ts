import { describe, expect, it } from 'vitest';
import {
  FrameBudgetController,
  DEFAULT_FRAME_BUDGET_CONFIG,
  type FrameBudgetControllerConfig,
} from '../FrameBudgetController.js';

/**
 * Deterministic, GPU-free control-law tests for the adaptive frame-budget
 * controller (Phase IV.1 / review gap D1). Every test injects a synthetic
 * measured-ms sequence and asserts the resulting knob trajectory — no device,
 * no clock, fully reproducible (this is the whole point of keeping the control
 * logic separate from the GPU-timing source).
 */

// A test config with a fast EMA (alpha=1 ⇒ ema == latest sample) so individual
// steps are easy to reason about, and a 1-tick up-cooldown so up-moves happen
// promptly in the small fixtures. The hysteresis band is kept (the default ±12%).
const FAST: Partial<FrameBudgetControllerConfig> = {
  targetMs: 16.0,
  emaAlpha: 1.0, // no smoothing lag — ema tracks the latest sample exactly
  upCooldownFrames: 1,
};

/** Drive a controller with a constant ms for N ticks; return the final decision. */
function settle(c: FrameBudgetController, ms: number, ticks: number) {
  let d = c.snapshot();
  for (let i = 0; i < ticks; i++) d = c.update(ms);
  return d;
}

describe('FrameBudgetController — control law', () => {
  it('OVER budget lowers resolutionFactor toward the floor, one step per tick', () => {
    const c = new FrameBudgetController(FAST); // starts at res=1.0, stride=2
    // 40 ms ≫ 16 ms target → sustained over budget. Each tick drops one step.
    const d1 = c.update(40);
    expect(d1.action).toBe('resolution-down');
    expect(d1.resolutionFactor).toBeCloseTo(0.95, 6);
    const d2 = c.update(40);
    expect(d2.resolutionFactor).toBeCloseTo(0.9, 6);
    // Keep pushing — it must clamp at the 0.5 floor and never go below.
    const dFinal = settle(c, 40, 50);
    expect(dFinal.resolutionFactor).toBeCloseTo(DEFAULT_FRAME_BUDGET_CONFIG.minResolutionFactor, 6);
    expect(dFinal.resolutionFactor).toBe(0.5);
  });

  it('UNDER budget raises resolutionFactor toward the ceiling (after cooldown), clamped at 1.0', () => {
    // Start already downscaled so there is headroom to climb.
    const c = new FrameBudgetController(FAST, { resolutionFactor: 0.7 });
    const d1 = c.update(5); // 5 ms ≪ 16 ms → under budget
    expect(d1.action).toBe('resolution-up');
    expect(d1.resolutionFactor).toBeCloseTo(0.75, 6);
    // Drives all the way up and clamps at the 1.0 ceiling.
    const dFinal = settle(c, 5, 50);
    expect(dFinal.resolutionFactor).toBe(1.0);
    // At the ceiling, with stride already at min, the only action left is none.
    const dHold = c.update(5);
    expect(dHold.resolutionFactor).toBe(1.0);
    expect(dHold.action).toBe('none');
  });

  it('inside the hysteresis dead-zone it HOLDS (no knob change)', () => {
    const c = new FrameBudgetController(FAST, { resolutionFactor: 0.8 });
    // Target 16; band ±12% ⇒ dead-zone (14.08, 17.92). Exactly on target:
    const onTarget = c.update(16.0);
    expect(onTarget.action).toBe('none');
    expect(onTarget.resolutionFactor).toBe(0.8);
    // Just inside the upper edge (17 < 17.92) and lower edge (15 > 14.08): hold.
    expect(c.update(17.0).action).toBe('none');
    expect(c.update(15.0).action).toBe('none');
    expect(c.snapshot().resolutionFactor).toBe(0.8);
  });

  it('does NOT oscillate on a knob change that crosses the target (the dead-zone absorbs it)', () => {
    // Model the closed loop: ms is a decreasing function of resolutionFactor.
    // Pick a mapping where the equilibrium lands INSIDE the dead-zone so a
    // correct controller settles instead of bang-banging across the target.
    //
    //   ms(res) = 22 * res   →  dead-zone (14.08, 17.92) ms. Descending from
    //   res=1.0 (22 ms, over): 0.85⇒18.7 (over), 0.80⇒17.6 (IN-BAND → settle).
    //   So the equilibrium is res=0.80, the first step whose ms enters the band.
    const c = new FrameBudgetController(FAST); // res starts 1.0
    const msOf = (res: number) => 22 * res;

    const actions: string[] = [];
    let res = c.snapshot().resolutionFactor;
    for (let i = 0; i < 40; i++) {
      const d = c.update(msOf(res));
      res = d.resolutionFactor;
      actions.push(d.action);
    }

    // It should converge: the tail of the action stream is all 'none' (settled),
    // NOT an alternating up/down limit cycle.
    const tail = actions.slice(-10);
    expect(tail.every((a) => a === 'none')).toBe(true);

    // And it must never exhibit an immediate reversal anywhere in the run
    // (a down directly followed by an up, or vice-versa) — that is the
    // oscillation signature the hysteresis band exists to prevent.
    const moves = actions.filter((a) => a !== 'none');
    for (let i = 1; i < moves.length; i++) {
      const a = moves[i - 1];
      const b = moves[i];
      const reversal =
        (a === 'resolution-down' && b === 'resolution-up') ||
        (a === 'resolution-up' && b === 'resolution-down');
      expect(reversal).toBe(false);
    }
    // Equilibrium: res=0.80, where ms(0.80)=17.6 first enters the dead-zone.
    expect(res).toBeCloseTo(0.8, 6);
    // Sanity: the settled ms is genuinely inside the band (not over/under).
    expect(22 * res).toBeGreaterThan(16 * (1 - 0.12)); // > 14.08
    expect(22 * res).toBeLessThan(16 * (1 + 0.12)); // < 17.92
  });

  it('EMA smoothing rejects a single-frame spike (no step on one lone bad frame)', () => {
    // Default-ish smoothing (alpha 0.2) but keep target 16. Sit on target, then
    // one 60 ms spike — the EMA barely moves (16 → 24.8) which IS over the
    // 17.92 upper threshold for a STRONG spike, so use a milder spike that the
    // EMA keeps in-band to prove smoothing.
    const c = new FrameBudgetController({ targetMs: 16.0, emaAlpha: 0.2, upCooldownFrames: 1 });
    // Seed the EMA at the target.
    for (let i = 0; i < 5; i++) c.update(16.0);
    expect(c.snapshot().resolutionFactor).toBe(1.0);
    // A single 24 ms frame: ema ← 0.2*24 + 0.8*16 = 17.6, still < 17.92 → no drop.
    const d = c.update(24.0);
    expect(d.smoothedMs).toBeCloseTo(17.6, 6);
    expect(d.action).toBe('none');
    expect(d.resolutionFactor).toBe(1.0);
    // But SUSTAINED 24 ms eventually pushes the EMA past the threshold → it adapts.
    const sustained = settle(c, 24.0, 20);
    expect(sustained.resolutionFactor).toBeLessThan(1.0);
  });

  it('two-lever ordering: DDGI stride only moves once resolution is pinned', () => {
    const c = new FrameBudgetController(FAST); // res=1.0, stride=2 (min)
    // Over budget: it must exhaust the resolution lever (1.0 → 0.5) BEFORE it
    // ever touches the stride.
    const seen: string[] = [];
    for (let i = 0; i < 20; i++) seen.push(c.update(40).action);
    // The first run of actions are all resolution-down; stride-up appears only
    // after resolution has bottomed out.
    const firstStrideIdx = seen.indexOf('ddgi-stride-up');
    expect(firstStrideIdx).toBeGreaterThan(0);
    // Every action before the first stride move is a resolution-down.
    for (let i = 0; i < firstStrideIdx; i++) expect(seen[i]).toBe('resolution-down');
    // By the end, resolution is at the floor and the stride has climbed past min.
    const snap = c.snapshot();
    expect(snap.resolutionFactor).toBe(0.5);
    expect(snap.ddgiStride).toBeGreaterThan(2);
    // Stride clamps at the configured max (32 default).
    const maxed = settle(c, 40, 60);
    expect(maxed.ddgiStride).toBe(DEFAULT_FRAME_BUDGET_CONFIG.maxDdgiStride);
  });

  it('PPG cadence lever is inactive by default, so legacy two-lever behaviour is unchanged', () => {
    const c = new FrameBudgetController(FAST, {
      resolutionFactor: 0.5,
      ddgiStride: 2,
      ppgDispatchInterval: 1,
    });

    const d = c.update(40);

    expect(d.action).toBe('ddgi-stride-up');
    expect(d.ddgiStride).toBe(3);
    expect(d.ppgDispatchInterval).toBe(1);
  });

  it('when enabled, PPG cadence backs off before DDGI stride', () => {
    const c = new FrameBudgetController({
      ...FAST,
      adaptPpgDispatchInterval: true,
      maxPpgDispatchInterval: 3,
    }, {
      resolutionFactor: 0.5,
      ddgiStride: 2,
      ppgDispatchInterval: 1,
    });

    const ppg1 = c.update(40);
    expect(ppg1.action).toBe('ppg-interval-up');
    expect(ppg1.ppgDispatchInterval).toBe(2);
    expect(ppg1.ddgiStride).toBe(2);

    const ppg2 = c.update(40);
    expect(ppg2.action).toBe('ppg-interval-up');
    expect(ppg2.ppgDispatchInterval).toBe(3);
    expect(ppg2.ddgiStride).toBe(2);

    const ddgi = c.update(40);
    expect(ddgi.action).toBe('ddgi-stride-up');
    expect(ddgi.ppgDispatchInterval).toBe(3);
    expect(ddgi.ddgiStride).toBe(3);
  });

  it('stride is restored (lowered) under budget only after resolution is at the ceiling', () => {
    // Start pinned at the cheapest: floor resolution + a raised stride.
    const c = new FrameBudgetController(FAST, { resolutionFactor: 0.5, ddgiStride: 10 });
    const seen: string[] = [];
    for (let i = 0; i < 40; i++) seen.push(c.update(4).action);
    // Resolution climbs to 1.0 first; only then does the stride come back down.
    const firstStrideDown = seen.indexOf('ddgi-stride-down');
    const lastResUp = seen.lastIndexOf('resolution-up');
    expect(firstStrideDown).toBeGreaterThan(lastResUp);
    const snap = c.snapshot();
    expect(snap.resolutionFactor).toBe(1.0);
    expect(snap.ddgiStride).toBe(DEFAULT_FRAME_BUDGET_CONFIG.minDdgiStride);
  });

  it('when enabled, PPG cadence restores before DDGI stride under budget', () => {
    const c = new FrameBudgetController({
      ...FAST,
      adaptPpgDispatchInterval: true,
      maxPpgDispatchInterval: 4,
    }, {
      resolutionFactor: 1.0,
      ppgDispatchInterval: 3,
      ddgiStride: 8,
    });

    const ppg1 = c.update(4);
    expect(ppg1.action).toBe('ppg-interval-down');
    expect(ppg1.ppgDispatchInterval).toBe(2);
    expect(ppg1.ddgiStride).toBe(8);

    const ppg2 = c.update(4);
    expect(ppg2.action).toBe('ppg-interval-down');
    expect(ppg2.ppgDispatchInterval).toBe(1);
    expect(ppg2.ddgiStride).toBe(8);

    const ddgi = c.update(4);
    expect(ddgi.action).toBe('ddgi-stride-down');
    expect(ddgi.ppgDispatchInterval).toBe(1);
    expect(ddgi.ddgiStride).toBe(7);
  });

  it('up-cooldown rate-limits quality restoration', () => {
    // cooldown of 5 ticks: under budget, an up-move happens at most every 5th tick.
    const c = new FrameBudgetController({ targetMs: 16, emaAlpha: 1.0, upCooldownFrames: 5 }, {
      resolutionFactor: 0.6,
    });
    const actions: string[] = [];
    for (let i = 0; i < 12; i++) actions.push(c.update(4).action);
    // Tick 1 moves up (lastUpMoveTick=-inf), then ticks 2-5 are gated, tick 6
    // moves, ticks 7-10 gated, tick 11 moves. → up-moves at indices 0, 5, 10.
    const upIdx = actions.map((a, i) => (a === 'resolution-up' ? i : -1)).filter((i) => i >= 0);
    expect(upIdx).toEqual([0, 5, 10]);
  });

  it('is DETERMINISTIC: identical ms sequences ⇒ identical decisions', () => {
    const seq = [40, 40, 9, 9, 16, 30, 30, 5, 5, 5, 20, 20];
    const a = new FrameBudgetController(FAST);
    const b = new FrameBudgetController(FAST);
    for (const ms of seq) {
      const da = a.update(ms);
      const db = b.update(ms);
      expect(db).toEqual(da);
    }
  });

  it('ignores non-finite / non-positive measured ms (holds the loop)', () => {
    const c = new FrameBudgetController(FAST, { resolutionFactor: 0.8 });
    // Seed the EMA.
    c.update(16);
    const before = c.snapshot();
    for (const bad of [NaN, Infinity, -Infinity, 0, -5]) {
      const d = c.update(bad);
      expect(d.action).toBe('none');
      expect(d.resolutionFactor).toBe(before.resolutionFactor);
      expect(d.ddgiStride).toBe(before.ddgiStride);
    }
  });

  it('reset() clears EMA + cooldown; reset(true) also restores best-quality knobs', () => {
    const c = new FrameBudgetController(FAST, { resolutionFactor: 0.7, ddgiStride: 8 });
    settle(c, 40, 30); // drive it to the floor
    expect(c.snapshot().resolutionFactor).toBe(0.5);
    c.reset(); // knobs retained
    expect(c.snapshot().resolutionFactor).toBe(0.5);
    expect(c.snapshot().smoothedMs).toBe(0); // EMA cleared
    c.reset(true); // knobs restored to ceiling/floor-stride
    expect(c.snapshot().resolutionFactor).toBe(1.0);
    expect(c.snapshot().ddgiStride).toBe(DEFAULT_FRAME_BUDGET_CONFIG.minDdgiStride);
  });

  it('rejects an empty/inverted hysteresis dead-zone at construction (oscillation guard)', () => {
    // Negative band ⇒ up-threshold ≥ down-threshold ⇒ one tick could trigger
    // both directions. The ctor must reject it.
    expect(() => new FrameBudgetController({ overBudgetBand: -0.5, underBudgetBand: -0.5 })).toThrow(
      /dead-zone is empty\/inverted/,
    );
    // Bad target / alpha / ranges also throw.
    expect(() => new FrameBudgetController({ targetMs: 0 })).toThrow(/targetMs/);
    expect(() => new FrameBudgetController({ emaAlpha: 0 })).toThrow(/emaAlpha/);
    expect(() => new FrameBudgetController({ emaAlpha: 1.5 })).toThrow(/emaAlpha/);
    expect(() => new FrameBudgetController({ minResolutionFactor: 0 })).toThrow(/minResolutionFactor/);
    expect(() => new FrameBudgetController({ minDdgiStride: 0 })).toThrow(/minDdgiStride/);
    expect(() => new FrameBudgetController({ minDdgiStride: 8, maxDdgiStride: 4 })).toThrow(/minDdgiStride/);
  });

  it('clamps the initial knobs into range at construction', () => {
    const c = new FrameBudgetController(FAST, { resolutionFactor: 5.0, ddgiStride: 999 });
    const s = c.snapshot();
    expect(s.resolutionFactor).toBe(DEFAULT_FRAME_BUDGET_CONFIG.maxResolutionFactor); // 1.0
    expect(s.ddgiStride).toBe(DEFAULT_FRAME_BUDGET_CONFIG.maxDdgiStride); // 32
    const c2 = new FrameBudgetController(FAST, { resolutionFactor: 0.01, ddgiStride: -3 });
    const s2 = c2.snapshot();
    expect(s2.resolutionFactor).toBe(DEFAULT_FRAME_BUDGET_CONFIG.minResolutionFactor); // 0.5
    expect(s2.ddgiStride).toBe(DEFAULT_FRAME_BUDGET_CONFIG.minDdgiStride); // 2
  });

  it('default config targets ~60fps and matches the preset 2→32 DDGI stride spread', () => {
    expect(DEFAULT_FRAME_BUDGET_CONFIG.targetMs).toBeCloseTo(16.6, 6);
    expect(DEFAULT_FRAME_BUDGET_CONFIG.minDdgiStride).toBe(2); // = ultra preset
    expect(DEFAULT_FRAME_BUDGET_CONFIG.maxDdgiStride).toBe(32); // = low preset
    expect(DEFAULT_FRAME_BUDGET_CONFIG.minResolutionFactor).toBe(0.5); // = low preset
    expect(DEFAULT_FRAME_BUDGET_CONFIG.maxResolutionFactor).toBe(1.0); // = ultra preset
  });

  // ── tickFrameBudget application pin (item 13) ─────────────────────────────
  // The FrameBudgetDecision contract is what HybridEngine.tickFrameBudget uses
  // to call setPpgDispatchInterval and setDdgiUpdateDivisor.  These tests pin
  // that every decision includes ppgDispatchInterval (the field tickFrameBudget
  // forwards to setPpgDispatchInterval) and that the adaptPpgDispatchInterval
  // gate correctly suppresses PPG lever motion when disabled.

  it('every decision includes ppgDispatchInterval (contract for tickFrameBudget application)', () => {
    // Verify the decision object always has the ppgDispatchInterval field so
    // HybridEngine.tickFrameBudget can unconditionally forward it.
    const c = new FrameBudgetController(FAST);
    const d0 = c.snapshot();
    expect(d0).toHaveProperty('ppgDispatchInterval');
    expect(typeof d0.ppgDispatchInterval).toBe('number');

    const d1 = c.update(40); // over budget
    expect(d1).toHaveProperty('ppgDispatchInterval');
    const d2 = c.update(4);  // under budget
    expect(d2).toHaveProperty('ppgDispatchInterval');
    const d3 = c.update(16); // on-target
    expect(d3).toHaveProperty('ppgDispatchInterval');
  });

  it('adaptPpgDispatchInterval:false (default) — ppgDispatchInterval is always minPpgDispatchInterval and never moves', () => {
    // When PPG adaptation is gated off (the default, matching engines without
    // PPG), ppgDispatchInterval must stay at the min regardless of budget
    // pressure.  HybridEngine.enableFrameBudget sets the gate from
    // cfg.ppgEnabled, so this pin ensures a non-PPG engine never gets
    // its PPG knob moved.
    const c = new FrameBudgetController({
      ...FAST,
      adaptPpgDispatchInterval: false,
      minPpgDispatchInterval: 1,
      maxPpgDispatchInterval: 4,
    }, { resolutionFactor: 0.5, ddgiStride: 32, ppgDispatchInterval: 1 });

    for (let i = 0; i < 20; i++) {
      const d = c.update(40); // sustained over-budget
      expect(d.ppgDispatchInterval).toBe(1);
      // Only ddgi-stride-up is permissible (resolution already floored).
      expect(d.action).not.toBe('ppg-interval-up');
    }
    for (let i = 0; i < 20; i++) {
      const d = c.update(4);  // sustained under-budget
      expect(d.ppgDispatchInterval).toBe(1);
      expect(d.action).not.toBe('ppg-interval-down');
    }
  });
});
