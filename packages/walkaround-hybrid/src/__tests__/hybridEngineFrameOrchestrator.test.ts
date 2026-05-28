import { describe, expect, it } from 'vitest';
import {
  fingerprintHybridPipelineRebuildKey,
  HYBRID_FRAME_SKIP_OUTPUT,
  RESOLUTION_FACTOR_DEBOUNCE_MS,
  resolveInternalRenderSize,
} from '../HybridEngineFrameOrchestrator.js';

describe('HybridEngineFrameOrchestrator', () => {
  it('fingerprintHybridPipelineRebuildKey is stable', () => {
    expect(fingerprintHybridPipelineRebuildKey(null)).toBe('__null');
    expect(fingerprintHybridPipelineRebuildKey(42)).toBe('__n:42');
    expect(fingerprintHybridPipelineRebuildKey('scene-v2')).toBe('__s:scene-v2');
  });

  it('HYBRID_FRAME_SKIP_OUTPUT is skipped kind', () => {
    expect(HYBRID_FRAME_SKIP_OUTPUT.kind).toBe('skipped');
    expect(HYBRID_FRAME_SKIP_OUTPUT.samplesAccumulated).toBe(0);
  });
});

describe('resolveInternalRenderSize — §5.1 resolutionFactor wiring', () => {
  const swap = { swapW: 1920, swapH: 1080 };

  it('factor 0.5 from full-res current ⇒ resize to round(swap*0.5)', () => {
    const r = resolveInternalRenderSize({
      ...swap,
      factor: 0.5,
      currentW: 1920,
      currentH: 1080,
      nowMs: 1000,
      lastResizeTs: 0, // first-ever resize always allowed
    });
    expect(r.shouldResize).toBe(true);
    expect(r.targetW).toBe(960);
    expect(r.targetH).toBe(540);
  });

  it('re-passing the same factor (already applied) ⇒ no resize (idempotent)', () => {
    const r = resolveInternalRenderSize({
      ...swap,
      factor: 0.5,
      currentW: 960, // already at the 0.5 size
      currentH: 540,
      nowMs: 5000,
      lastResizeTs: 1000,
    });
    expect(r.shouldResize).toBe(false);
    expect(r.targetW).toBe(960);
    expect(r.targetH).toBe(540);
  });

  it('factor omitted ⇒ target == swap, no resize when already full (regression guard)', () => {
    const r = resolveInternalRenderSize({
      ...swap,
      factor: undefined,
      currentW: 1920,
      currentH: 1080,
      nowMs: 1000,
      lastResizeTs: 0,
    });
    expect(r.targetW).toBe(1920);
    expect(r.targetH).toBe(1080);
    expect(r.shouldResize).toBe(false);
  });

  it('a changed factor within the debounce window is suppressed', () => {
    const r = resolveInternalRenderSize({
      ...swap,
      factor: 0.67,
      currentW: 960,
      currentH: 540,
      nowMs: 1000 + RESOLUTION_FACTOR_DEBOUNCE_MS - 10, // < debounce since last
      lastResizeTs: 1000,
    });
    // Target is computed but the resize is debounced (not applied this frame).
    expect(r.targetW).toBe(Math.round(1920 * 0.67));
    expect(r.shouldResize).toBe(false);
  });

  it('a changed factor after the debounce window is applied', () => {
    const r = resolveInternalRenderSize({
      ...swap,
      factor: 0.67,
      currentW: 960,
      currentH: 540,
      nowMs: 1000 + RESOLUTION_FACTOR_DEBOUNCE_MS + 1,
      lastResizeTs: 1000,
    });
    expect(r.shouldResize).toBe(true);
    expect(r.targetW).toBe(Math.round(1920 * 0.67));
  });

  it('clamps factor > 1 to 1.0 and factor <= 0 to 1.0', () => {
    const tooBig = resolveInternalRenderSize({
      ...swap, factor: 2.0, currentW: 1920, currentH: 1080, nowMs: 0, lastResizeTs: 0,
    });
    expect(tooBig.targetW).toBe(1920);
    const zero = resolveInternalRenderSize({
      ...swap, factor: 0, currentW: 1920, currentH: 1080, nowMs: 0, lastResizeTs: 0,
    });
    expect(zero.targetW).toBe(1920);
    const negative = resolveInternalRenderSize({
      ...swap, factor: -0.5, currentW: 1920, currentH: 1080, nowMs: 0, lastResizeTs: 0,
    });
    expect(negative.targetW).toBe(1920);
  });

  it('never returns a zero internal dimension (floor at 1 px)', () => {
    const r = resolveInternalRenderSize({
      swapW: 1, swapH: 1, factor: 0.01, currentW: 1, currentH: 1, nowMs: 0, lastResizeTs: 0,
    });
    expect(r.targetW).toBeGreaterThanOrEqual(1);
    expect(r.targetH).toBeGreaterThanOrEqual(1);
  });
});
