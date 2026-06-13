import { describe, expect, it } from 'vitest';
import type { FrameInput, Scene } from '@vitrum/core';
import {
  fingerprintHybridPipelineRebuildKey,
  HYBRID_FRAME_SKIP_OUTPUT,
  RESOLUTION_FACTOR_DEBOUNCE_MS,
  runHybridEngineFrame,
  resolveInternalRenderSize,
  type HybridEngineFrameDeps,
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

function identityMat4(): Float32Array {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

const FRAME_INPUT = {
  viewMatrix: identityMat4(),
  projMatrix: identityMat4(),
  cameraPosition: [0, 0, 0],
  frameSeed: 17,
  swapChainView: {} as GPUTextureView,
  swapChainFormat: 'bgra8unorm',
} as unknown as FrameInput;

function makeRcFrameDeps(args: {
  scene: Scene | null;
  primaryLightIntensity: number;
  capture: {
    sunColor: readonly [number, number, number] | null;
    sunCastShadowDisabled?: boolean | null;
  };
}): HybridEngineFrameDeps {
  let lastTs = 0;
  const pipeline = {
    accumFrameIndex: 200,
    temporalAccumAlpha: 0.01,
    lastGpuTimings: {},
    renderFrame: () => undefined,
    setDDGIInputs: () => undefined,
    setRCInputs: () => undefined,
    getAuxBufferTextures: () => null,
    getEmitterBufferAndCount: () => null,
    getEnvBindings: () => null,
  };
  const ddgi = {
    warmupFrame: 0,
    warmupStride: 1,
    ready: false,
    syncRestirBvhBuffers: () => undefined,
    updateFrame: () => Promise.resolve(),
    setSkyParams: () => undefined,
    setGlassMixScale: () => undefined,
    getReadAtlasGPUTextures: () => null,
    gridParams: {},
  };
  const rc = {
    syncRestirBvhBuffers: () => undefined,
    updateLights: () => undefined,
    dispatchFrame: (inputs: {
      sunColor: readonly [number, number, number];
      sunCastShadowDisabled?: boolean;
    }) => {
      args.capture.sunColor = inputs.sunColor;
      args.capture.sunCastShadowDisabled = inputs.sunCastShadowDisabled ?? false;
    },
    buildRCInputs: () => null,
  };

  return {
    subsystems: {
      pipeline: pipeline as unknown as HybridEngineFrameDeps['subsystems']['pipeline'],
      bvhBuffers: {
        totalEmissivePower: 1,
        emitters: { count: 0 },
        bvhMode: 'merged',
      } as unknown as HybridEngineFrameDeps['subsystems']['bvhBuffers'],
      ddgi: ddgi as unknown as HybridEngineFrameDeps['subsystems']['ddgi'],
      rc: rc as unknown as HybridEngineFrameDeps['subsystems']['rc'],
      skinning: null,
      lastScene: args.scene,
    },
    lighting: {
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: args.primaryLightIntensity,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    },
    filter: {
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
      restirPtReuse: 0,
      nrcEnabled: 0,
    },
    telemetry: {
      frameSubs: [],
      progressSubs: [],
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
      ddgiOn: false,
      isLayerEnabled: () => false,
      device: {} as GPUDevice,
      tunables: {
        emitterDist2Floor: 0.01,
        directFireflyClamp: 4,
        causticBoost: 1,
        causticVisClamp: 1,
        temporalMClampDI: 20,
        spatialReuseRadiusPx: 30,
        spatialDepthTolFloor: 0.05,
        restirGiWCap: 16,
        restirGiIrrClamp: 5,
        restirGiMClamp: 50,
        restirGiSpatialRadiusPx: 12,
        restirGiSpatialNormalDotMin: 0.9,
        restirGiSpatialCoplanarTol: 0.05,
        gtaoRadiusPx: 32,
        gtaoIntensity: 2,
        gtaoDepthThreshold: 2,
        gtaoBilateralDepthSigma: 0.25,
        adaptiveSamplingThresholdLow: 0.01,
        adaptiveSamplingThresholdHigh: 0.1,
        triIntersectEpsilon: 1e-5,
        glassMixScale: 0.7,
      },
      rcWeight: 0.5,
    },
  };
}

describe('HybridEngineFrameOrchestrator — RC sun input', () => {
  it('uses scene directional emitter RGB and intensity for RC sun color', () => {
    const capture = { sunColor: null as readonly [number, number, number] | null };
    const scene: Scene = {
      primitives: [],
      emitters: [{
        kind: 'directional',
        id: 'warm-blue-sun',
        direction: [0, -1, 0],
        color: [0.25, 0.5, 1],
        intensity: 4,
      }],
      environment: { kind: 'none' },
    };

    runHybridEngineFrame(
      makeRcFrameDeps({ scene, primaryLightIntensity: 10, capture }),
      FRAME_INPUT,
    );

    expect(capture.sunColor).toEqual([1, 2, 4]);
  });

  it('keeps the legacy grey primaryLightIntensity fallback when no scene directional exists', () => {
    const capture = { sunColor: null as readonly [number, number, number] | null };

    runHybridEngineFrame(
      makeRcFrameDeps({ scene: null, primaryLightIntensity: 3.5, capture }),
      FRAME_INPUT,
    );

    expect(capture.sunColor).toEqual([3.5, 3.5, 3.5]);
  });

  it('forwards directional emitter castShadow:false as the RC sun shadow flag', () => {
    const capture = {
      sunColor: null as readonly [number, number, number] | null,
      sunCastShadowDisabled: null as boolean | null,
    };
    const scene: Scene = {
      primitives: [],
      emitters: [{
        kind: 'directional',
        id: 'soft-no-shadow-sun',
        direction: [0, -1, 0],
        color: [1, 1, 1],
        intensity: 1,
        castShadow: false,
      }],
      environment: { kind: 'none' },
    };

    runHybridEngineFrame(
      makeRcFrameDeps({ scene, primaryLightIntensity: 10, capture }),
      FRAME_INPUT,
    );

    expect(capture.sunCastShadowDisabled).toBe(true);
  });
});
