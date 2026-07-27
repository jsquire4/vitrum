import { describe, expect, it } from 'vitest';
import type { FrameInput, Scene } from '@vitrum/core';
import type { DDGILight } from '../ddgi/types.js';
import {
  fingerprintHybridPipelineRebuildKey,
  HYBRID_FRAME_SKIP_OUTPUT,
  RESOLUTION_FACTOR_DEBOUNCE_MS,
  refractiveTraceCausticGate,
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

  it('gates refractive caustics on both the explicit strategy and a transmissive scene', () => {
    const transmissive: Scene = {
      primitives: [{
        kind: 'mesh', id: 'glass',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1], roughness: 0, metallic: 0,
          transmission: 1, ior: 1.5,
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const opaque: Scene = {
      ...transmissive,
      primitives: transmissive.primitives.map((primitive) => ({
        ...primitive,
        material: { ...primitive.material, transmission: 0 },
      })),
    };

    expect(refractiveTraceCausticGate('none', transmissive)).toBe(0);
    expect(refractiveTraceCausticGate('manifold-nee', transmissive)).toBe(2);
    expect(refractiveTraceCausticGate('photon-map', transmissive)).toBe(0);
    expect(refractiveTraceCausticGate('refractive-trace', opaque)).toBe(0);
    expect(refractiveTraceCausticGate('refractive-trace', transmissive)).toBe(1);

    const faceBlocked: Scene = {
      ...transmissive,
      primitives: transmissive.primitives.map((primitive) => ({
        ...primitive,
        material: {
          ...primitive.material,
          frontLayer: { transmission: [1, 0, 0] },
          backLayer: { transmission: [0, 1, 1] },
        },
      })),
    };
    const faceOverlap: Scene = {
      ...faceBlocked,
      primitives: faceBlocked.primitives.map((primitive) => ({
        ...primitive,
        material: {
          ...primitive.material,
          backLayer: { transmission: [1, 0, 0] },
        },
      })),
    };
    expect(refractiveTraceCausticGate('refractive-trace', faceBlocked)).toBe(0);
    expect(refractiveTraceCausticGate('refractive-trace', faceOverlap)).toBe(1);
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
  primaryLightDir?: readonly [number, number, number];
  capture: {
    sunColor: readonly [number, number, number] | null;
    sunCastShadowDisabled?: boolean | null;
    emittersBuf?: GPUBuffer | null;
    emitterCount?: number | null;
    envTextureView?: GPUTextureView | null;
    envSampler?: GPUSampler | null;
    envRotationY?: number | null;
    envIntensity?: number | null;
    scalarSkyRadiance?: readonly [number, number, number] | null;
    hasDirectionalEnvironment?: boolean | null;
    materialTextureAtlasView?: GPUTextureView | null;
    materialMapMetaTextureView?: GPUTextureView | null;
    bvhTangentTextureView?: GPUTextureView | null;
    bvhVertexColorTextureView?: GPUTextureView | null;
    lights?: readonly DDGILight[] | undefined;
  };
  rcEmitters?: { readonly buffer: GPUBuffer; readonly count: number } | null;
  rcEnvBindings?: {
    readonly textureView: GPUTextureView;
    readonly sampler: GPUSampler;
    readonly rotationY: number;
    readonly intensity: number;
    readonly hasDirectionalEnvironment: boolean;
  } | null;
  skyTint?: [number, number, number];
  skyIrradiance?: number;
  rcMaterialAtlasBindings?: {
    readonly materialTextureAtlasView: GPUTextureView;
    readonly materialMapMetaTextureView: GPUTextureView;
    readonly bvhTangentTextureView: GPUTextureView;
    readonly bvhVertexColorTextureView: GPUTextureView;
  } | null;
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
    getEmitterBufferAndCount: () => args.rcEmitters ?? null,
    getEmitterSamplingBufferAndCount: () => args.rcEmitters == null
      ? null
      : { ...args.rcEmitters, emitterDataOffset: 0, emitterAliasOffset: 256 },
    getEnvBindings: () => args.rcEnvBindings ?? null,
    getMaterialAtlasBindings: () => args.rcMaterialAtlasBindings ?? null,
    getSceneGeometryBufferBindings: () => null,
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
    updateLights: (lights: readonly DDGILight[]) => { args.capture.lights = lights; },
    dispatchFrame: (inputs: {
      sunColor: readonly [number, number, number];
      sunCastShadowDisabled?: boolean;
      emittersBuf?: GPUBuffer;
      emitterCount?: number;
      envTextureView?: GPUTextureView;
      envSampler?: GPUSampler;
      envRotationY?: number;
      envIntensity?: number;
      scalarSkyRadiance?: readonly [number, number, number];
      hasDirectionalEnvironment?: boolean;
      materialTextureAtlasView?: GPUTextureView;
      materialMapMetaTextureView?: GPUTextureView;
      bvhTangentTextureView?: GPUTextureView;
      bvhVertexColorTextureView?: GPUTextureView;
    }) => {
      args.capture.sunColor = inputs.sunColor;
      args.capture.sunCastShadowDisabled = inputs.sunCastShadowDisabled ?? false;
      args.capture.emittersBuf = inputs.emittersBuf ?? null;
      args.capture.emitterCount = inputs.emitterCount ?? null;
      args.capture.envTextureView = inputs.envTextureView ?? null;
      args.capture.envSampler = inputs.envSampler ?? null;
      args.capture.envRotationY = inputs.envRotationY ?? null;
      args.capture.envIntensity = inputs.envIntensity ?? null;
      args.capture.scalarSkyRadiance = inputs.scalarSkyRadiance ?? null;
      args.capture.hasDirectionalEnvironment =
        inputs.hasDirectionalEnvironment ?? null;
      args.capture.materialTextureAtlasView = inputs.materialTextureAtlasView ?? null;
      args.capture.materialMapMetaTextureView = inputs.materialMapMetaTextureView ?? null;
      args.capture.bvhTangentTextureView = inputs.bvhTangentTextureView ?? null;
      args.capture.bvhVertexColorTextureView = inputs.bvhVertexColorTextureView ?? null;
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
      primaryLightDir: [...(args.primaryLightDir ?? [0, -1, 0])] as [number, number, number],
      primaryLightIntensity: args.primaryLightIntensity,
      skyTint: args.skyTint ?? [1, 1, 1],
      skyIrradiance: args.skyIrradiance ?? 1,
    },
    filter: {
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
      grisReuse: 0,
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
  it('orients every RC sun to the runtime primary direction and clears stale lights', () => {
    const capture = {
      sunColor: null as readonly [number, number, number] | null,
      lights: undefined as readonly DDGILight[] | undefined,
    };
    const scene: Scene = {
      primitives: [],
      emitters: [
        { kind: 'directional', id: 'sun-a', direction: [0, 1, 0], color: [1, 1, 1], intensity: 1 },
        { kind: 'directional', id: 'sun-b', direction: [0, 0, 1], color: [1, 0.5, 0.25], intensity: 2 },
      ],
      environment: { kind: 'none' },
    };
    runHybridEngineFrame(
      makeRcFrameDeps({ scene, primaryLightDir: [1, 0, 0], primaryLightIntensity: 1, capture }),
      FRAME_INPUT,
    );
    expect(capture.lights).toHaveLength(2);
    expect(capture.lights?.map((light) => light.direction)).toEqual([
      { x: -1, y: -0, z: -0 },
      { x: -1, y: -0, z: -0 },
    ]);

    runHybridEngineFrame(
      makeRcFrameDeps({ scene: null, primaryLightIntensity: 1, capture }),
      FRAME_INPUT,
    );
    expect(capture.lights).toEqual([]);
  });

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

  it('forwards the pipeline emitter buffer and environment bindings into RC dispatch', () => {
    const capture: {
      sunColor: readonly [number, number, number] | null;
      emittersBuf?: GPUBuffer | null;
      emitterCount?: number | null;
      envTextureView?: GPUTextureView | null;
      envSampler?: GPUSampler | null;
      envRotationY?: number | null;
      envIntensity?: number | null;
      scalarSkyRadiance?: readonly [number, number, number] | null;
      hasDirectionalEnvironment?: boolean | null;
      materialTextureAtlasView?: GPUTextureView | null;
      materialMapMetaTextureView?: GPUTextureView | null;
      bvhTangentTextureView?: GPUTextureView | null;
      bvhVertexColorTextureView?: GPUTextureView | null;
    } = { sunColor: null };
    const emittersBuf = { label: 'rc-emitters' } as unknown as GPUBuffer;
    const envTextureView = { label: 'rc-env-view' } as unknown as GPUTextureView;
    const envSampler = { label: 'rc-env-sampler' } as unknown as GPUSampler;
    const materialTextureAtlasView = { label: 'rc-material-atlas-view' } as unknown as GPUTextureView;
    const materialMapMetaTextureView = { label: 'rc-material-meta-view' } as unknown as GPUTextureView;
    const bvhTangentTextureView = { label: 'rc-tangent-view' } as unknown as GPUTextureView;
    const bvhVertexColorTextureView = { label: 'rc-vertex-color-view' } as unknown as GPUTextureView;

    runHybridEngineFrame(
      makeRcFrameDeps({
        scene: { primitives: [], emitters: [], environment: { kind: 'none' } },
        primaryLightIntensity: 1,
        capture,
        rcEmitters: { buffer: emittersBuf, count: 7 },
        rcEnvBindings: {
          textureView: envTextureView,
          sampler: envSampler,
          rotationY: Math.PI / 3,
          intensity: 2.5,
          hasDirectionalEnvironment: true,
        },
        rcMaterialAtlasBindings: {
          materialTextureAtlasView,
          materialMapMetaTextureView,
          bvhTangentTextureView,
          bvhVertexColorTextureView,
        },
      }),
      FRAME_INPUT,
    );

    expect(capture.emittersBuf).toBe(emittersBuf);
    expect(capture.emitterCount).toBe(7);
    expect(capture.envTextureView).toBe(envTextureView);
    expect(capture.envSampler).toBe(envSampler);
    expect(capture.envRotationY).toBeCloseTo(Math.PI / 3);
    expect(capture.envIntensity).toBe(2.5);
    expect(capture.scalarSkyRadiance).toEqual([1, 1, 1]);
    expect(capture.hasDirectionalEnvironment).toBe(true);
    expect(capture.materialTextureAtlasView).toBe(materialTextureAtlasView);
    expect(capture.materialMapMetaTextureView).toBe(materialMapMetaTextureView);
    expect(capture.bvhTangentTextureView).toBe(bvhTangentTextureView);
    expect(capture.bvhVertexColorTextureView).toBe(bvhVertexColorTextureView);
  });

  it('forwards authored scalar sky radiance when the bound env is only a placeholder', () => {
    const capture = {
      sunColor: null as readonly [number, number, number] | null,
      scalarSkyRadiance: null as readonly [number, number, number] | null,
      hasDirectionalEnvironment: null as boolean | null,
    };
    runHybridEngineFrame(
      makeRcFrameDeps({
        scene: { primitives: [], emitters: [], environment: { kind: 'none' } },
        primaryLightIntensity: 1,
        skyTint: [0.25, 0.5, 1],
        skyIrradiance: 2,
        capture,
        rcEnvBindings: {
          textureView: { label: 'placeholder-view' } as unknown as GPUTextureView,
          sampler: { label: 'placeholder-sampler' } as unknown as GPUSampler,
          rotationY: 0,
          intensity: 0,
          hasDirectionalEnvironment: false,
        },
      }),
      FRAME_INPUT,
    );

    expect(capture.scalarSkyRadiance).toEqual([0.5, 1, 2]);
    expect(capture.hasDirectionalEnvironment).toBe(false);
  });
});
