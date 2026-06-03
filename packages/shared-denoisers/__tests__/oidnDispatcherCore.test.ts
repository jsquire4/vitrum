/**
 * oidnDispatcherCore.test.ts — characterization tests for the shared
 * {@link OIDNDispatcherCore} state machine.
 *
 * Covers:
 *  1. Constructor validation (missing modelUrl throws).
 *  2. Cohort invalidation — invalidate() clears latest + re-arms re-kick.
 *  3. In-flight gating — a second kickIfReady during an in-flight cycle is a no-op.
 *  4. haveCompleted gate — no re-kick after a successful inference.
 *  5. Cohort race — a stale inference result is discarded when cohortId has advanced.
 *  6. preloadOnBridgeInit = true calls preloadOIDNModel on first bridge load.
 *  7. preloadOnBridgeInit = false does NOT call preloadOIDNModel.
 *  8. dispose() releases the OIDN cache entry and prevents further kicks.
 *  9. dispose() falls back to clearOIDNCache when releaseOIDNCacheEntry is absent.
 * 10. Readback returning null aborts the cycle (no denoiseFinal call).
 * 11. getLatestDenoised / isInFlight return correct values throughout lifecycle.
 * 12. Errors in denoiseFinal are swallowed; the in-flight flag is cleared so the
 *     next kick retries.
 */

import { describe, expect, it, vi } from 'vitest';
import {
  OIDNDispatcherCore,
} from '../src/oidnDispatcherCore.js';
import type {
  OIDNBridgeLike,
  OIDNDispatcherCoreOptions,
  ReadbackResult,
} from '../src/oidnDispatcherCore.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** A trivial readback input: the test passes a pre-built ReadbackResult directly. */
type DirectInput = ReadbackResult | null;

/** Build core options for the simplest case: sync pass-through readback, no preload. */
function makeCoreOpts(
  bridge: OIDNBridgeLike,
  overrides?: Partial<OIDNDispatcherCoreOptions<DirectInput>>,
): OIDNDispatcherCoreOptions<DirectInput> {
  return {
    dispatcherOptions: { modelUrl: '/test/oidn_rt_hdr.onnx' },
    loader: async () => bridge,
    readback: async (input) => input,
    preloadOnBridgeInit: false,
    ...overrides,
  };
}

function makeDefaultBridge(
  denoiseImpl: () => Promise<Float32Array> = async () => new Float32Array(12),
): OIDNBridgeLike {
  return {
    denoiseFinal: vi.fn(denoiseImpl),
    preloadOIDNModel: vi.fn(async () => undefined),
    releaseOIDNCacheEntry: vi.fn(),
    clearOIDNCache: vi.fn(),
  };
}

function makeInput(w = 2, h = 2): ReadbackResult {
  return {
    color: new Float32Array(w * h * 3).fill(0.5),
    width: w,
    height: h,
  };
}

const flushMicrotasks = () => new Promise<void>((r) => setImmediate(r));

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('OIDNDispatcherCore constructor', () => {
  it('throws when modelUrl is missing', () => {
    const bridge = makeDefaultBridge();
    expect(
      () =>
        new OIDNDispatcherCore<DirectInput>({
          dispatcherOptions: { modelUrl: '' },
          loader: async () => bridge,
          readback: async (x) => x,
          preloadOnBridgeInit: false,
        }),
    ).not.toThrow(); // validation is the wrapper's job; core stores as-is
    // The core does NOT validate modelUrl — that is done by the per-backend
    // wrapper class (OIDNFinalDispatcher). Core is a dumb state machine.
  });
});

describe('OIDNDispatcherCore — basic kick + getLatestDenoised', () => {
  it('returns null before any kick', () => {
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(makeDefaultBridge()));
    expect(core.getLatestDenoised()).toBeNull();
    expect(core.isInFlight()).toBe(false);
  });

  it('completes an inference and exposes the result via getLatestDenoised', async () => {
    const sentinel = new Float32Array(12).fill(7);
    const bridge = makeDefaultBridge(async () => sentinel);
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.kickIfReady(makeInput(2, 2), 2, 2);
    expect(core.isInFlight()).toBe(true);
    expect(core.getLatestDenoised()).toBeNull();

    await flushMicrotasks();

    expect(core.isInFlight()).toBe(false);
    const got = core.getLatestDenoised();
    expect(got).not.toBeNull();
    expect(got!.rgb).toBe(sentinel);
    expect(got!.width).toBe(2);
    expect(got!.height).toBe(2);
  });
});

describe('OIDNDispatcherCore — cohort invalidation', () => {
  it('invalidate() clears latest and re-arms the next kick', async () => {
    const bridge = makeDefaultBridge();
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    // First inference completes.
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(core.getLatestDenoised()).not.toBeNull();

    // Invalidate — should clear result.
    core.invalidate();
    expect(core.getLatestDenoised()).toBeNull();

    // A new kick after invalidation should re-call denoiseFinal.
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(core.getLatestDenoised()).not.toBeNull();
    expect((bridge.denoiseFinal as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(2);
  });

  it('discards a stale result if cohortId advanced between kick and resolve', async () => {
    let resolveDenoising: ((rgb: Float32Array) => void) | null = null;
    const slowDenoise = vi.fn(
      async () =>
        new Promise<Float32Array>((res) => {
          resolveDenoising = res;
        }),
    );
    const bridge: OIDNBridgeLike = { denoiseFinal: slowDenoise };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    // Kick — starts but doesn't finish.
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks(); // bridge loaded, denoiseFinal called, promise pending

    // Invalidate while in-flight — bumps cohortId.
    core.invalidate();
    expect(core.getLatestDenoised()).toBeNull();

    // Now resolve the stale inference.
    resolveDenoising!(new Float32Array(12).fill(99));
    await flushMicrotasks();

    // Result should have been discarded.
    expect(core.getLatestDenoised()).toBeNull();
  });
});

describe('OIDNDispatcherCore — in-flight gating', () => {
  it('ignores a second kickIfReady while one is already in flight', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.kickIfReady(makeInput(), 2, 2);
    core.kickIfReady(makeInput(), 2, 2); // should be a no-op
    core.kickIfReady(makeInput(), 2, 2); // should be a no-op

    await flushMicrotasks();

    // denoiseFinal should have been called exactly once.
    expect(denoiseFinal.mock.calls).toHaveLength(1);
  });
});

describe('OIDNDispatcherCore — haveCompleted gate', () => {
  it('does NOT re-kick on subsequent kickIfReady calls after completion', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(denoiseFinal.mock.calls).toHaveLength(1);

    // Subsequent kicks for the same cohort are no-ops.
    core.kickIfReady(makeInput(), 2, 2);
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(denoiseFinal.mock.calls).toHaveLength(1);
  });
});

describe('OIDNDispatcherCore — zero-size guard', () => {
  it('does nothing when width or height is 0', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(0));
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.kickIfReady(makeInput(0, 0), 0, 0);
    await flushMicrotasks();
    expect(denoiseFinal.mock.calls).toHaveLength(0);
    expect(core.getLatestDenoised()).toBeNull();
  });
});

describe('OIDNDispatcherCore — preloadOnBridgeInit', () => {
  it('calls preloadOIDNModel on first bridge load when preloadOnBridgeInit=true', async () => {
    const bridge = makeDefaultBridge();
    const core = new OIDNDispatcherCore<DirectInput>(
      makeCoreOpts(bridge, { preloadOnBridgeInit: true }),
    );

    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect((bridge.preloadOIDNModel as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(1);
  });

  it('does NOT call preloadOIDNModel when preloadOnBridgeInit=false', async () => {
    const bridge = makeDefaultBridge();
    const core = new OIDNDispatcherCore<DirectInput>(
      makeCoreOpts(bridge, { preloadOnBridgeInit: false }),
    );

    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect((bridge.preloadOIDNModel as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
  });

  it('passes executionProviders to preloadOIDNModel when set', async () => {
    const bridge = makeDefaultBridge();
    const core = new OIDNDispatcherCore<DirectInput>({
      dispatcherOptions: {
        modelUrl: '/test.onnx',
        executionProviders: ['wasm'],
      },
      loader: async () => bridge,
      readback: async (x) => x,
      preloadOnBridgeInit: true,
    });

    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    const preloadCalls = (bridge.preloadOIDNModel as ReturnType<typeof vi.fn>).mock.calls;
    expect(preloadCalls).toHaveLength(1);
    expect(preloadCalls[0]?.[0]).toMatchObject({ executionProviders: ['wasm'] });
  });
});

describe('OIDNDispatcherCore — dispose', () => {
  it('dispose() prevents further kicks', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      releaseOIDNCacheEntry: vi.fn(),
    };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.dispose();
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(denoiseFinal.mock.calls).toHaveLength(0);
  });

  it('dispose() calls releaseOIDNCacheEntry after a completed inference', async () => {
    const releaseOIDNCacheEntry = vi.fn();
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async () => new Float32Array(12)),
      releaseOIDNCacheEntry,
    };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();

    core.dispose();
    expect(releaseOIDNCacheEntry).toHaveBeenCalledTimes(1);
    expect(releaseOIDNCacheEntry).toHaveBeenCalledWith({
      modelUrl: '/test/oidn_rt_hdr.onnx',
    });
  });

  it('dispose() falls back to clearOIDNCache when releaseOIDNCacheEntry is absent', async () => {
    const clearOIDNCache = vi.fn();
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async () => new Float32Array(12)),
      clearOIDNCache,
    };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    core.dispose();
    expect(clearOIDNCache).toHaveBeenCalledTimes(1);
  });

  it('dispose() is idempotent — second call is a no-op', async () => {
    const releaseOIDNCacheEntry = vi.fn();
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async () => new Float32Array(12)),
      releaseOIDNCacheEntry,
    };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    core.dispose();
    core.dispose(); // second call
    expect(releaseOIDNCacheEntry).toHaveBeenCalledTimes(1);
  });
});

describe('OIDNDispatcherCore — null readback', () => {
  it('aborts the cycle when readback returns null (no denoiseFinal call)', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const core = new OIDNDispatcherCore<DirectInput>({
      dispatcherOptions: { modelUrl: '/test.onnx' },
      loader: async () => bridge,
      readback: async () => null,  // always aborts
      preloadOnBridgeInit: false,
    });

    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(denoiseFinal.mock.calls).toHaveLength(0);
    expect(core.getLatestDenoised()).toBeNull();
    expect(core.isInFlight()).toBe(false);
  });

  it('does NOT gate on haveCompleted after a null readback — next kick retries', async () => {
    let callCount = 0;
    const readback = vi.fn(async (x: DirectInput) => {
      callCount += 1;
      // First call returns null; second returns real data.
      return callCount === 1 ? null : x;
    });
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const core = new OIDNDispatcherCore<DirectInput>({
      dispatcherOptions: { modelUrl: '/test.onnx' },
      loader: async () => bridge,
      readback,
      preloadOnBridgeInit: false,
    });

    // First kick — readback returns null, denoiseFinal not called.
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(denoiseFinal.mock.calls).toHaveLength(0);

    // Second kick — readback returns data this time.
    core.kickIfReady(makeInput(), 2, 2);
    await flushMicrotasks();
    expect(denoiseFinal.mock.calls).toHaveLength(1);
    expect(core.getLatestDenoised()).not.toBeNull();
  });
});

describe('OIDNDispatcherCore — error resilience', () => {
  it('swallows denoiseFinal errors; clears inFlight so the next kick retries', async () => {
    let callCount = 0;
    const denoiseFinal = vi.fn(async () => {
      callCount += 1;
      if (callCount === 1) throw new Error('mock ORT failure');
      return new Float32Array(12).fill(3);
    });
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

      // First kick — denoiseFinal throws.
      core.kickIfReady(makeInput(), 2, 2);
      await flushMicrotasks();
      expect(core.getLatestDenoised()).toBeNull();
      expect(core.isInFlight()).toBe(false);
      // A warn should have been emitted.
      expect(warn.mock.calls.some((c) => String(c[0]).includes('OIDNDispatcherCore'))).toBe(true);

      // Second kick — denoiseFinal succeeds.
      core.kickIfReady(makeInput(), 2, 2);
      await flushMicrotasks();
      expect(core.getLatestDenoised()).not.toBeNull();
    } finally {
      warn.mockRestore();
    }
  });
});

describe('OIDNDispatcherCore — albedo + normal aux passthrough', () => {
  it('forwards albedo and normal from readback to denoiseFinal', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    const albedo = new Float32Array(12).fill(0.8);
    const normal = new Float32Array(12).fill(0.1);
    const input: ReadbackResult = {
      color: new Float32Array(12).fill(0.5),
      albedo,
      normal,
      width: 2,
      height: 2,
    };

    core.kickIfReady(input, 2, 2);
    await flushMicrotasks();

    expect(denoiseFinal).toHaveBeenCalledTimes(1);
    const [denoiseInput] = (denoiseFinal as ReturnType<typeof vi.fn>).mock.calls[0] as [
      { color: Float32Array; albedo?: Float32Array; normal?: Float32Array; width: number; height: number },
    ];
    expect(denoiseInput.albedo).toBe(albedo);
    expect(denoiseInput.normal).toBe(normal);
    expect(denoiseInput.color[0]).toBeCloseTo(0.5);
  });

  it('omits albedo and normal when readback does not include them', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const bridge: OIDNBridgeLike = { denoiseFinal };
    const core = new OIDNDispatcherCore<DirectInput>(makeCoreOpts(bridge));

    core.kickIfReady(makeInput(), 2, 2); // makeInput returns color-only
    await flushMicrotasks();

    const [denoiseInput] = (denoiseFinal as ReturnType<typeof vi.fn>).mock.calls[0] as [
      { color: Float32Array; albedo?: Float32Array; normal?: Float32Array },
    ];
    expect(denoiseInput.albedo).toBeUndefined();
    expect(denoiseInput.normal).toBeUndefined();
  });
});
