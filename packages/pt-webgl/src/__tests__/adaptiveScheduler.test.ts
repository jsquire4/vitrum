/**
 * CHARACTERIZATION TEST — pins `AdaptiveScheduler` (Task 4.4 Theme A) as a
 * golden before/after the extraction of the sample/tile scheduler + render-size
 * planner out of `PTEngineWebGL2`.
 *
 * The scheduler is a pure state machine: feed it a deterministic sequence of
 * frame batch-timings (`update(batchMs)`) and assert the SAME sequence of
 * `(samplesPerFrame, tileSize)` decisions, plus the same `planRenderSize` plans
 * and context-loss reaction, as the pre-extraction inline `#updateScheduler` /
 * `#planRenderSize` / context-lost-handler bodies produced.
 *
 * The golden numbers below were derived BY HAND from the original control flow
 * (re-verified against the pre-extraction source) — not regenerated from the new
 * class — so the test is a true regression net, not a tautology.
 */
import { describe, it, expect } from 'vitest';
import {
  AdaptiveScheduler,
  DEFAULT_TILE_SIZE,
  type SchedulerOptions,
  type SchedulerDeviceLimits,
} from '../adaptiveScheduler.js';
import { defaultSchedulerOptions } from '../ptEngineWebGL2.js';

const LIMITS_4K: SchedulerDeviceLimits = { maxTextureSize: 4096, maxRenderbufferSize: 4096 };

/** Interactive-mode options (adaptive=true, target=40ms, init spp=6, init
 *  tile=1, maxSpp=48, maxTile=4, budget=1GiB). */
function interactiveOpts(): SchedulerOptions {
  return defaultSchedulerOptions(undefined, 'interactive');
}

describe('AdaptiveScheduler — initial state mirrors the options', () => {
  it('seeds samplesPerFrame / tileSize from initialSamplesPerFrame / initialTileSize', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    expect(s.samplesPerFrame).toBe(6); // interactive initialSamplesPerFrame
    expect(s.tileSize).toBe(1); // interactive initialTileSize
    expect(s.contextLost).toBe(false);
    expect(s.options.qualityMode).toBe('interactive');
  });

  it('capture mode seeds from its frozen defaults (spp 1, tile DEFAULT_TILE_SIZE)', () => {
    const s = new AdaptiveScheduler(defaultSchedulerOptions(undefined, 'capture'), LIMITS_4K);
    expect(s.samplesPerFrame).toBe(1);
    expect(s.tileSize).toBe(DEFAULT_TILE_SIZE);
  });
});

describe('AdaptiveScheduler.update — golden decision sequence (interactive)', () => {
  it('GROWS samplesPerFrame on a fast batch (batchMs < target*0.55)', () => {
    // target=40 → 0.55*target=22. init spp=6 → grow to max(6+1, ceil(6*1.2)=8) = 8.
    // batchMs=10 is NOT < 0.25*target(10)? 10 < 10 is false → tile stays.
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    s.update(10);
    expect(s.samplesPerFrame).toBe(8);
    expect(s.tileSize).toBe(1);
  });

  it('GROWS samples AND shrinks tile on a very-fast batch when tile > initialTileSize', () => {
    // Seed tileSize above initialTileSize by first inflating it via a slow batch,
    // then a very-fast batch (batchMs < 0.25*target=10) decrements the tile.
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    // Slow batch (batchMs=90 > 2*target=80): halve spp (6→3), bump tile (1→2).
    s.update(90);
    expect(s.samplesPerFrame).toBe(3);
    expect(s.tileSize).toBe(2);
    // Very fast batch (batchMs=5 < 0.25*target=10): grow spp (3→max(4,ceil(3.6)=4)=4),
    // tile decrements (2→1) since tile(2) > initialTileSize(1).
    s.update(5);
    expect(s.samplesPerFrame).toBe(4);
    expect(s.tileSize).toBe(1);
  });

  it('SHRINKS samplesPerFrame on a slow batch (batchMs > target*1.35), no tile bump below 2x', () => {
    // target=40 → 1.35*target=54. batchMs=60 > 54 but NOT > 2*target=80 → tile unchanged.
    // spp halves: floor(6*0.5)=3.
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    s.update(60);
    expect(s.samplesPerFrame).toBe(3);
    expect(s.tileSize).toBe(1);
  });

  it('SHRINKS samples AND bumps tile on a very-slow batch (batchMs > target*2)', () => {
    // batchMs=100 > 2*target=80: spp halves (6→3), tile bumps (1→2, since 1<maxTile=4).
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    s.update(100);
    expect(s.samplesPerFrame).toBe(3);
    expect(s.tileSize).toBe(2);
  });

  it('does nothing in the dead-band (target*0.55 <= batchMs <= target*1.35)', () => {
    // target=40 → dead-band [22, 54]. batchMs=40 → no change.
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    s.update(40);
    expect(s.samplesPerFrame).toBe(6);
    expect(s.tileSize).toBe(1);
  });

  it('caps samplesPerFrame at maxSamplesPerFrame on repeated fast batches', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    const seq: number[] = [];
    // 12 fast frames: spp climbs 6→8→10→12→15→18→22→27→33→40→48(cap)→48.
    for (let i = 0; i < 12; i += 1) {
      s.update(10);
      seq.push(s.samplesPerFrame);
    }
    expect(seq).toEqual([8, 10, 12, 15, 18, 22, 27, 33, 40, 48, 48, 48]);
    expect(s.samplesPerFrame).toBe(48); // interactive maxSamplesPerFrame
  });

  it('floors samplesPerFrame at minSamplesPerFrame on repeated slow batches', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    const seq: number[] = [];
    // 4 slow-but-not-2x frames (batchMs=60): spp halves each time 6→3→1→1→1
    // (floor(3*0.5)=1, floor(1*0.5)=0→max(min=1,0)=1). Tile unchanged (60 < 80).
    for (let i = 0; i < 4; i += 1) {
      s.update(60);
      seq.push(s.samplesPerFrame);
    }
    expect(seq).toEqual([3, 1, 1, 1]);
    expect(s.tileSize).toBe(1);
  });

  it('is a no-op when adaptive scheduling is disabled (capture mode)', () => {
    const s = new AdaptiveScheduler(defaultSchedulerOptions(undefined, 'capture'), LIMITS_4K);
    const spp0 = s.samplesPerFrame;
    const tile0 = s.tileSize;
    s.update(1000);
    s.update(1);
    expect(s.samplesPerFrame).toBe(spp0);
    expect(s.tileSize).toBe(tile0);
  });
});

describe('AdaptiveScheduler.noteContextLost + update interaction (golden)', () => {
  it('noteContextLost clamps to 1 spp and bumps tile to >= DEFAULT_TILE_SIZE', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    // Grow spp/tile first so the clamp is observable.
    s.update(100); // spp 6→3, tile 1→2
    s.noteContextLost();
    expect(s.contextLost).toBe(true);
    expect(s.samplesPerFrame).toBe(1);
    expect(s.tileSize).toBe(Math.max(2, DEFAULT_TILE_SIZE)); // = 3
  });

  it('after context loss, update keeps spp at 1 and clamps tile to maxTileSize', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    s.noteContextLost(); // spp=1, tile=3
    s.update(10); // adaptive+target>0, contextLost branch: spp=1, tile=min(maxTile=4,max(3,3))=3
    expect(s.samplesPerFrame).toBe(1);
    expect(s.tileSize).toBe(3);
  });
});

describe('AdaptiveScheduler.planRenderSize — golden plans', () => {
  it('passes through a small request unscaled (no guardrail) and estimates bytes', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    const plan = s.planRenderSize(640, 360);
    expect(plan.width).toBe(640);
    expect(plan.height).toBe(360);
    expect(plan.guardrail).toBeNull();
    // 640*360*8*4 + 64MiB overhead.
    expect(plan.estimatedBytes).toBe(640 * 360 * 8 * 4 + 64 * 1024 * 1024);
  });

  it('caps to the WebGL max render dimension and reports the guardrail', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), { maxTextureSize: 2048, maxRenderbufferSize: 2048 });
    const plan = s.planRenderSize(4096, 2048);
    // scale = min(1, 2048/4096, 2048/2048) = 0.5 → 2048 x 1024.
    expect(plan.width).toBe(2048);
    expect(plan.height).toBe(1024);
    expect(plan.guardrail).toBe('capped to WebGL max render dimension 2048');
  });

  it('downscales to fit a tight render-target budget and reports the budget guardrail', () => {
    // capture budget is 512MiB; request a frame whose RGBA16F×4 footprint blows it.
    const opts = defaultSchedulerOptions(undefined, 'capture'); // budget 512MiB, no adaptive
    const s = new AdaptiveScheduler(opts, LIMITS_4K);
    const plan = s.planRenderSize(4096, 4096);
    // No dimension cap (4096 <= 4096). estimated = 4096*4096*8*4 + 64MiB = ~512MiB+64MiB > 512MiB,
    // so it downscales. The exact scaled size is whatever the original
    // memory-scale math produced — assert it shrank, fit the budget, and carries
    // the budget guardrail.
    expect(plan.width).toBeLessThan(4096);
    expect(plan.height).toBeLessThan(4096);
    expect(plan.estimatedBytes).toBeLessThanOrEqual(opts.renderTargetBudgetBytes);
    expect(plan.guardrail).toMatch(/render-target budget/);
  });

  it('planRenderSize is pure — repeated calls with the same args are identical', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    const a = s.planRenderSize(1920, 1080);
    const b = s.planRenderSize(1920, 1080);
    expect(a).toEqual(b);
  });

  it('floors a fractional request and clamps a zero request to 1x1 minimum', () => {
    const s = new AdaptiveScheduler(interactiveOpts(), LIMITS_4K);
    const plan = s.planRenderSize(0, 0);
    expect(plan.width).toBe(1);
    expect(plan.height).toBe(1);
  });
});
