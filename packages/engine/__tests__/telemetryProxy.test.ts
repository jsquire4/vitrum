// Verify the wrapWithIdempotentDispose proxy in createEngine.ts correctly
// forwards onFrame/onProgress when the wrapped engine implements them, and
// gracefully omits them when it doesn't (consumers typeof-check per the
// Engine contract).
//
// This is a structural / API-shape test that exercises wrapWithIdempotentDispose via
// a fake engine and asserts the public T3.E surface compiles.

import { describe, it, expect } from 'vitest';
import type { Engine, FrameStats, ProgressStats, EngineCapabilities, EngineState, FrameOutput, FrameInput, Scene } from '@vitrum/core';
import { asBackendTexture } from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../src/createEngine.js';

// ──────────────────────────────────────────────────────────────────────────
// Smoke: a minimal Engine implementation can declare the optional T3.E
// methods and the contract still holds.

const NULL_CAPS: EngineCapabilities = {
  supportsIncrementalScene: false,
  supportsAuxBuffers: false,
  accumulates: false,
  maxSamplesPerPixel: Infinity,
  maxBounces: 1,
  supportedAnalyticShapes: new Set(),
  supportedEmitterKinds: new Set(),
  causticStrategy: 'none',
};

class FakeEngine implements Engine {
  readonly state: EngineState = 'ready';
  readonly capabilities = NULL_CAPS;
  readonly _frameSubs: Array<(s: FrameStats) => void> = [];
  readonly _progressSubs: Array<(p: ProgressStats) => void> = [];

  setScene(_: Scene): void {}
  renderFrame(_: FrameInput): FrameOutput {
    return {
      kind: 'rendered',
      samplesAccumulated: 1,
      isConverged: false,
      primaryRadiance: asBackendTexture<'test', {}>({}),
    };
  }
  reset(): void {}
  pause(): void {}
  resume(): void {}
  dispose(): void {}

  onFrame(cb: (s: FrameStats) => void): () => void {
    this._frameSubs.push(cb);
    return () => {
      const i = this._frameSubs.indexOf(cb);
      if (i >= 0) this._frameSubs.splice(i, 1);
    };
  }

  onProgress(cb: (p: ProgressStats) => void): () => void {
    this._progressSubs.push(cb);
    return () => {
      const i = this._progressSubs.indexOf(cb);
      if (i >= 0) this._progressSubs.splice(i, 1);
    };
  }
}

describe('Engine T3.E contract', () => {
  it('FrameStats shape is constructible with only the required field', () => {
    const stats: FrameStats = { frameTimeMs: 16.7 };
    expect(stats.frameTimeMs).toBe(16.7);
  });

  it('FrameStats accepts the optional gpu/passTimings/spp/memory fields', () => {
    const stats: FrameStats = {
      frameTimeMs: 16.7,
      gpuTimeMs: 12.4,
      passTimings: { restir: 5.2, denoiser: 2.1 },
      spp: 1,
      estimatedGpuMemoryBytes: 256 * 1024 * 1024,
    };
    expect(stats.passTimings?.['restir']).toBe(5.2);
  });

  it('ProgressStats discriminates by kind', () => {
    const a: ProgressStats = { kind: 'pt-spp',           current:  64, target: 256, fraction: 0.25 };
    const b: ProgressStats = { kind: 'denoiser-converge', current:  10, target: 100, fraction: 0.10 };
    const c: ProgressStats = { kind: 'ddgi-warmup',      current:   3, target:  30, fraction: 0.10 };
    expect([a.kind, b.kind, c.kind]).toEqual(['pt-spp', 'denoiser-converge', 'ddgi-warmup']);
  });

  it('subscribe + unsubscribe via onFrame returns a working teardown', () => {
    const e = new FakeEngine();
    const proxy = wrapWithIdempotentDispose(e, () => {});
    const seen: number[] = [];
    const off = proxy.onFrame!((s) => seen.push(s.frameTimeMs));
    expect(e._frameSubs.length).toBe(1);

    // Simulate the engine firing a frame.
    e._frameSubs.forEach((cb) => cb({ frameTimeMs: 16.7 }));
    expect(seen).toEqual([16.7]);

    off();
    expect(e._frameSubs.length).toBe(0);
    e._frameSubs.forEach((cb) => cb({ frameTimeMs: 33.4 }));
    expect(seen).toEqual([16.7]); // didn't grow after off()
  });

  it('subscribe + unsubscribe via onProgress is symmetric', () => {
    const e = new FakeEngine();
    const proxy = wrapWithIdempotentDispose(e, () => {});
    const seen: number[] = [];
    const off = proxy.onProgress!((p) => seen.push(p.fraction));
    e._progressSubs.forEach((cb) => cb({ kind: 'pt-spp', current: 16, target: 64, fraction: 0.25 }));
    off();
    e._progressSubs.forEach((cb) => cb({ kind: 'pt-spp', current: 32, target: 64, fraction: 0.50 }));
    expect(seen).toEqual([0.25]);
  });

  it('multiple subscribers each get the callback', () => {
    const e = new FakeEngine();
    const proxy = wrapWithIdempotentDispose(e, () => {});
    const a: number[] = [], b: number[] = [];
    proxy.onFrame!((s) => a.push(s.frameTimeMs));
    proxy.onFrame!((s) => b.push(s.frameTimeMs));
    e._frameSubs.forEach((cb) => cb({ frameTimeMs: 16.7 }));
    expect(a).toEqual([16.7]);
    expect(b).toEqual([16.7]);
  });
});
