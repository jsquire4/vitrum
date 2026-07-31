import { describe, expect, it, vi } from 'vitest';
import {
  computeDdgiWarmupProgress,
  computeDenoiserConvergeProgress,
} from '../HybridEngineFrameOrchestrator.js';
import { HybridEngine } from '../HybridEngine.js';
import type { HybridEngineOptions } from '../HybridEngine.js';
import type { ProgressStats } from '@vitrum/core';

// ─────────────────────────────────────────────────────────────────────────────
// Pure metric functions — the honest, GPU-free heart of the two progress signals.
// These pin the ramp + reset + stop-when-converged semantics without needing a
// live GPUDevice (mirrors the `resolveInternalRenderSize` pure-function tests).
// ─────────────────────────────────────────────────────────────────────────────

describe('computeDdgiWarmupProgress — ddgi-warmup metric', () => {
  it('ramps fraction 0→1 over `stride` frames after a fresh build', () => {
    const stride = 8;
    // Frame 0 (just built / invalidated): nothing accumulated yet.
    const f0 = computeDdgiWarmupProgress({ frame: 0, stride, ready: false });
    expect(f0).not.toBeNull();
    expect(f0!.kind).toBe('ddgi-warmup');
    expect(f0!.current).toBe(0);
    expect(f0!.target).toBe(stride);
    expect(f0!.fraction).toBe(0);

    // Halfway through the round-robin sweep.
    const fHalf = computeDdgiWarmupProgress({ frame: 4, stride, ready: false });
    expect(fHalf!.fraction).toBeCloseTo(0.5, 6);

    // One frame short of warm.
    const fAlmost = computeDdgiWarmupProgress({ frame: 7, stride, ready: false });
    expect(fAlmost!.fraction).toBeCloseTo(7 / 8, 6);
    expect(fAlmost!.fraction).toBeLessThan(1);
  });

  it('stops emitting (returns null) once the grid is warm', () => {
    // DDGI flips `ready` true at frame >= stride; the metric must go quiet.
    expect(
      computeDdgiWarmupProgress({ frame: 8, stride: 8, ready: true }),
    ).toBeNull();
    expect(
      computeDdgiWarmupProgress({ frame: 99, stride: 8, ready: true }),
    ).toBeNull();
  });

  it('clamps fraction to [0,1] and floors stride at 1', () => {
    // frame > stride but not yet flagged ready ⇒ clamp at 1 (defensive).
    const over = computeDdgiWarmupProgress({ frame: 12, stride: 8, ready: false });
    expect(over!.fraction).toBe(1);
    // A degenerate stride < 1 has no real warm-up window ⇒ target floors at 1.
    const degenerate = computeDdgiWarmupProgress({ frame: 0, stride: 0, ready: false });
    expect(degenerate!.target).toBe(1);
    expect(degenerate!.fraction).toBe(0);
  });
});

describe('computeDenoiserConvergeProgress — denoiser-converge metric', () => {
  it('ramps fraction 0→1 over the ~1/alpha effective window', () => {
    const alpha = 0.01; // ~100-frame effective window
    const window = 100;

    const f0 = computeDenoiserConvergeProgress({ accumFrameIndex: 0, alpha });
    expect(f0).not.toBeNull();
    expect(f0!.kind).toBe('denoiser-converge');
    expect(f0!.target).toBe(window);
    expect(f0!.current).toBe(0);
    expect(f0!.fraction).toBe(0);

    const fHalf = computeDenoiserConvergeProgress({ accumFrameIndex: 50, alpha });
    expect(fHalf!.fraction).toBeCloseTo(0.5, 6);

    const fAlmost = computeDenoiserConvergeProgress({ accumFrameIndex: 99, alpha });
    expect(fAlmost!.fraction).toBeCloseTo(0.99, 6);
    expect(fAlmost!.fraction).toBeLessThan(1);
  });

  it('resets when accumFrameIndex drops to 0 (camera motion path)', () => {
    const alpha = 0.01;
    // Converging at frame 60...
    const mid = computeDenoiserConvergeProgress({ accumFrameIndex: 60, alpha });
    expect(mid!.fraction).toBeCloseTo(0.6, 6);
    // ...camera moves → pipeline zeroes _accumFrameIndex → fraction snaps to 0.
    const afterMotion = computeDenoiserConvergeProgress({ accumFrameIndex: 0, alpha });
    expect(afterMotion!.fraction).toBe(0);
  });

  it('stops emitting (returns null) once the window is full', () => {
    expect(
      computeDenoiserConvergeProgress({ accumFrameIndex: 100, alpha: 0.01 }),
    ).toBeNull();
    expect(
      computeDenoiserConvergeProgress({ accumFrameIndex: 250, alpha: 0.01 }),
    ).toBeNull();
  });

  it('returns null for non-accumulating / degenerate alpha', () => {
    // alpha >= 1 ⇒ every frame fully fresh ⇒ no temporal convergence concept.
    expect(
      computeDenoiserConvergeProgress({ accumFrameIndex: 3, alpha: 1 }),
    ).toBeNull();
    // alpha <= 0 ⇒ pure history hold ⇒ no finite window to report.
    expect(
      computeDenoiserConvergeProgress({ accumFrameIndex: 3, alpha: 0 }),
    ).toBeNull();
    expect(
      computeDenoiserConvergeProgress({ accumFrameIndex: 3, alpha: NaN }),
    ).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// HybridEngine.onProgress wiring — subscribe / unsubscribe / no-callback.
// Constructed with a minimal duck-typed device; we never call renderFrame here
// (that needs a real GPU pipeline) so no async init / WGSL compile is exercised.
// ─────────────────────────────────────────────────────────────────────────────

/** Minimal duck-typed GPUDevice good enough for the HybridEngine constructor
 *  (DDGI + RC wiring + capabilities). `createCommandEncoder` presence is the
 *  factory's duck-type gate; the constructor itself never dispatches GPU work. */
function makeStubDevice(): GPUDevice {
  const noop = () => undefined;
  return {
    createCommandEncoder: noop,
    createBuffer: () => ({ destroy: noop }),
    createTexture: () => ({ createView: noop, destroy: noop }),
    createBindGroupLayout: noop,
    createBindGroup: noop,
    createShaderModule: noop,
    createComputePipeline: noop,
    queue: { submit: noop, writeBuffer: noop },
    features: new Set<string>(),
    limits: {},
    addEventListener: noop,
    removeEventListener: noop,
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeEngineOpts(): HybridEngineOptions {
  return {
    device: makeStubDevice(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

describe('HybridEngine.onProgress — subscription wiring', () => {
  it('is a function (closes the contract zero-producer gap)', () => {
    const engine = new HybridEngine(makeEngineOpts());
    expect(typeof engine.onProgress).toBe('function');
  });

  it('registers a subscriber and returns an unsubscribe function', () => {
    const engine = new HybridEngine(makeEngineOpts());
    const cb = vi.fn();
    const unsub = engine.onProgress(cb);
    expect(typeof unsub).toBe('function');
    // The subscriber landed in the internal list.
    const subs = engine['_progressSubs'] as ReadonlyArray<unknown>;
    expect(subs.length).toBe(1);
    unsub();
    expect(subs.length).toBe(0);
  });

  it('no progress subscriber ⇒ progress list is empty (no emission, no work)', () => {
    const engine = new HybridEngine(makeEngineOpts());
    const subs = engine['_progressSubs'] as ReadonlyArray<unknown>;
    expect(subs.length).toBe(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end emission path WITHOUT a GPU: drive emitProgressTelemetry's inputs
// directly via the engine's frame-deps builder + a stubbed pipeline/DDGI so we
// pin "emits while ramping, goes quiet when converged, silent with no callback".
// ─────────────────────────────────────────────────────────────────────────────

import { runHybridEngineFrame } from '../HybridEngineFrameOrchestrator.js';
import type { HybridEngineFrameDeps } from '../HybridEngineFrameOrchestrator.js';
import { asMat4, type FrameInput } from '@vitrum/core';

/** Build a frame-deps object whose DDGI / pipeline are stubs exposing exactly
 *  the warm-up + accumulator state the progress path reads. */
function makeFrameDeps(opts: {
  progressSubs: Array<(p: ProgressStats) => void>;
  ddgiFrame: number;
  ddgiStride: number;
  ddgiReady: boolean;
  accumFrameIndex: number;
  alpha: number;
}): HybridEngineFrameDeps {
  let lastTs = 0;
  const stubPipeline = {
    accumFrameIndex: opts.accumFrameIndex,
    temporalAccumAlpha: opts.alpha,
    lastGpuTimings: {},
    renderFrame: () => undefined,
    presentLastFrame: () => undefined,
    setDDGIInputs: () => undefined,
    setRCInputs: () => undefined,
  };
  const stubDdgi = {
    warmupFrame: opts.ddgiFrame,
    warmupStride: opts.ddgiStride,
    ready: opts.ddgiReady,
    updateFrame: () => Promise.resolve(),
    // Forwarding-façade methods (interface-hygiene refactor 2026-06-02)
    setSunIntensityMultiplier: () => undefined,
    setSkyParams: () => undefined,
    setGlassMixScale: () => undefined,
    setIndirectFeedback: () => undefined,
    getReadAtlasGPUTextures: () => null,
    // Live BVH propagation (THREE-decouple): propagateBvhToGiSubsystems syncs the
    // shared scene-BVH buffers into DDGI each frame when syncDdgi + bvhBuffers.
    syncRestirBvhBuffers: () => undefined,
    gridParams: {},
  };
  return {
    subsystems: {
      pipeline: stubPipeline as unknown as HybridEngineFrameDeps['subsystems']['pipeline'],
      bvhBuffers: { totalEmissivePower: 1, emitters: { count: 0 }, bvhMode: 'merged' } as unknown as HybridEngineFrameDeps['subsystems']['bvhBuffers'],
      ddgi: stubDdgi as unknown as HybridEngineFrameDeps['subsystems']['ddgi'],
      // No traversal scene ⇒ DDGI updateFrame is skipped, but its warmup
      // accessors are still read by the progress path.
      rc: null,
      skinning: null,
      lastScene: null,
    },
    lighting: {
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
      ddgiLights: [],
    },
    filter: {
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    telemetry: {
      frameSubs: [],
      progressSubs: opts.progressSubs,
      verbose: false,
      debugTimings: [],
      debugSurface: { estimatedGpuMemoryBytes: () => undefined } as unknown as HybridEngineFrameDeps['telemetry']['debugSurface'],
      dbg: null,
      getDenoiserState: () => null,
    },
    dims: { width: 64, height: 64, internalWidth: 64, internalHeight: 64 },
    control: {
      targetFrameIntervalMs: null,
      getLastFrameTs: () => lastTs,
      setLastFrameTs: (t: number) => { lastTs = t; },
      applyResolutionFactor: () => ({ width: 64, height: 64 }),
      runSkinning: () => undefined,
      presentLastFrame: () => undefined,
    },
    flags: {
      state: 'ready',
      debug: false,
      isLayerEnabled: () => true,
      device: makeStubDevice(),
      maxBounces: 2,
      tunables: { glassMixScale: 1, triIntersectEpsilon: 1e-5 } as unknown as HybridEngineFrameDeps['flags']['tunables'],
      rayOriginBias: 1e-3,
      rcWeight: 0,
    },
  };
}

const FRAME_MATRIX = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

const FRAME_INPUT: FrameInput = {
  viewMatrix: FRAME_MATRIX,
  projMatrix: FRAME_MATRIX,
  frameSeed: 1,
  swapChainView: {} as unknown as GPUTextureView,
} as unknown as FrameInput;

describe('HybridEngine frame loop — progress emission', () => {
  it('emits ddgi-warmup ramping toward 1 over the warm-up window, then stops', () => {
    const events: ProgressStats[] = [];
    const subs = [(p: ProgressStats) => events.push(p)];

    // Mid-warmup: frame 4 of stride 8, not ready.
    runHybridEngineFrame(
      makeFrameDeps({
        progressSubs: subs,
        ddgiFrame: 4, ddgiStride: 8, ddgiReady: false,
        accumFrameIndex: 200, alpha: 0.01, // denoiser already converged → silent
      }),
      FRAME_INPUT,
    );
    const warm = events.filter((e) => e.kind === 'ddgi-warmup');
    expect(warm.length).toBe(1);
    expect(warm[0]!.fraction).toBeCloseTo(0.5, 6);

    // Warm: ready=true → no ddgi-warmup event.
    events.length = 0;
    runHybridEngineFrame(
      makeFrameDeps({
        progressSubs: subs,
        ddgiFrame: 8, ddgiStride: 8, ddgiReady: true,
        accumFrameIndex: 200, alpha: 0.01,
      }),
      FRAME_INPUT,
    );
    expect(events.filter((e) => e.kind === 'ddgi-warmup').length).toBe(0);
  });

  it('emits denoiser-converge ramping, then resets to 0 on motion (accumFrameIndex=0)', () => {
    const events: ProgressStats[] = [];
    const subs = [(p: ProgressStats) => events.push(p)];

    // Converging at frame 60 of a 100-frame window.
    runHybridEngineFrame(
      makeFrameDeps({
        progressSubs: subs,
        ddgiFrame: 8, ddgiStride: 8, ddgiReady: true, // ddgi silent
        accumFrameIndex: 60, alpha: 0.01,
      }),
      FRAME_INPUT,
    );
    let conv = events.filter((e) => e.kind === 'denoiser-converge');
    expect(conv.length).toBe(1);
    expect(conv[0]!.fraction).toBeCloseTo(0.6, 6);

    // Camera moved → pipeline reset accumFrameIndex to 0 → fraction snaps to 0.
    events.length = 0;
    runHybridEngineFrame(
      makeFrameDeps({
        progressSubs: subs,
        ddgiFrame: 8, ddgiStride: 8, ddgiReady: true,
        accumFrameIndex: 0, alpha: 0.01,
      }),
      FRAME_INPUT,
    );
    conv = events.filter((e) => e.kind === 'denoiser-converge');
    expect(conv.length).toBe(1);
    expect(conv[0]!.fraction).toBe(0);

    // Fully converged → silent.
    events.length = 0;
    runHybridEngineFrame(
      makeFrameDeps({
        progressSubs: subs,
        ddgiFrame: 8, ddgiStride: 8, ddgiReady: true,
        accumFrameIndex: 100, alpha: 0.01,
      }),
      FRAME_INPUT,
    );
    expect(events.filter((e) => e.kind === 'denoiser-converge').length).toBe(0);
  });

  it('no emission when no progress callback is registered', () => {
    // progressSubs empty ⇒ emitProgressTelemetry short-circuits before reading
    // any state. We assert by giving the stub DDGI/pipeline values that WOULD
    // emit if a subscriber existed, and confirming the frame still completes.
    const deps = makeFrameDeps({
      progressSubs: [],
      ddgiFrame: 4, ddgiStride: 8, ddgiReady: false,
      accumFrameIndex: 60, alpha: 0.01,
    });
    const out = runHybridEngineFrame(deps, FRAME_INPUT);
    expect(out.kind).toBe('rendered');
    // No subscriber list to inspect for events; the contract is "no callback ⇒
    // no-op". The pure-function tests above already pin the would-be values.
  });
});
