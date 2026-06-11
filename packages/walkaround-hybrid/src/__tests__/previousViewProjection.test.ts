import { describe, expect, it } from 'vitest';
import type { FrameInput } from '@vitrum/core';
import {
  runHybridEngineFrame,
  type HybridEngineFrameDeps,
} from '../HybridEngineFrameOrchestrator.js';
import type { PipelineFrameInputs } from '../pipeline/WalkaroundGPUPipeline.js';
import { updateUBO } from '../pipeline/uboUpdater.js';
import { MOTION_VECTORS_WGSL } from '../shaders/motionVectors.wgsl.js';
import { TEMPORAL_WGSL } from '../shaders/temporal.wgsl.js';
import { TEMPORAL_GI_COMMON_WGSL } from '../shaders/temporalGiCommon.wgsl.js';
import { WALKAROUND_UBO_WGSL } from '../shaders/walkaroundUbo.wgsl.js';

function identityMat4(): Float32Array {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

function diagonalMat4(x: number, y: number, z: number): Float32Array {
  const m = new Float32Array(16);
  m[0] = x;
  m[5] = y;
  m[10] = z;
  m[15] = 1;
  return m;
}

function makeFrameDeps(capture: { inputs: PipelineFrameInputs | null }): HybridEngineFrameDeps {
  let lastTs = 0;
  const stubPipeline = {
    accumFrameIndex: 0,
    temporalAccumAlpha: 0.01,
    lastGpuTimings: {},
    renderFrame: (inputs: PipelineFrameInputs) => {
      capture.inputs = inputs;
      return true;
    },
    setDDGIInputs: () => undefined,
    setRCInputs: () => undefined,
    getAuxBufferTextures: () => null,
  };
  const stubDdgi = {
    warmupFrame: 0,
    warmupStride: 1,
    ready: false,
    updateFrame: () => Promise.resolve(),
    setSunIntensityMultiplier: () => undefined,
    setSkyParams: () => undefined,
    setGlassMixScale: () => undefined,
    getReadAtlasGPUTextures: () => null,
    syncRestirBvhBuffers: () => undefined,
    gridParams: {},
  };

  return {
    subsystems: {
      pipeline: stubPipeline as unknown as HybridEngineFrameDeps['subsystems']['pipeline'],
      bvhBuffers: { totalEmissivePower: 1, emitters: { count: 0 }, bvhMode: 'merged' } as unknown as HybridEngineFrameDeps['subsystems']['bvhBuffers'],
      ddgi: stubDdgi as unknown as HybridEngineFrameDeps['subsystems']['ddgi'],
      rc: null,
      skinning: null,
      lastScene: null,
    },
    lighting: {
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
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
      rcWeight: 0,
    },
  };
}

function makePipelineInputs(prevViewProjMatrix: Float32Array): PipelineFrameInputs {
  const m = identityMat4();
  return {
    camera: { viewMatrix: m, projMatrix: m, prevViewProjMatrix, cameraPos: [0, 0, 0] },
    screen: { screenWidth: 64, screenHeight: 64, frameSeed: 7, swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm' },
    lighting: {
      totalEmissivePower: 1,
      emitterCount: 4,
      primaryLightDir: [0, 1, 0],
      primaryLightIntensity: 1,
      skyTint: [0, 0, 0],
      skyIrradiance: 0,
      emitterDist2Floor: 0.01,
      directFireflyClamp: 4,
      causticBoost: 1,
      causticVisClamp: 1,
      lightTreeEnabled: 1,
      lightTreeNodeCount: 7,
    },
    restirDI: { temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05 },
    restirGI: {
      restirGiWCap: 16,
      restirGiIrrClamp: 5,
      restirGiMClamp: 50,
      restirGiSpatialRadiusPx: 12,
      restirGiSpatialNormalDotMin: 0.9,
      restirGiSpatialCoplanarTol: 0.05,
    },
    gtao: {
      gtaoRadiusPx: 32,
      gtaoIntensity: 2,
      gtaoDepthThreshold: 2,
      gtaoBilateralDepthSigma: 0.25,
      adaptiveSamplingThresholdLow: 0.01,
      adaptiveSamplingThresholdHigh: 0.1,
    },
    filter: {
      triIntersectEpsilon: 1e-5,
      glassMixScale: 0.7,
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    nrc: {},
    composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
  };
}

function capturingDevice(backing: Uint8Array): GPUDevice {
  return {
    queue: {
      writeBuffer: (_buf: GPUBuffer, offset: number, data: ArrayBuffer) => {
        backing.set(new Uint8Array(data), offset);
      },
    },
  } as unknown as GPUDevice;
}

describe('walkaround previous view-projection reprojection', () => {
  it('forwards FrameInput.prevProjMatrix by composing prevProj * prevView for the pipeline', () => {
    const capture = { inputs: null as PipelineFrameInputs | null };
    const deps = makeFrameDeps(capture);
    const input = {
      viewMatrix: identityMat4(),
      projMatrix: identityMat4(),
      prevViewMatrix: diagonalMat4(5, 7, 11),
      prevProjMatrix: diagonalMat4(2, 3, 4),
      cameraPosition: [0, 0, 0],
      viewport: { width: 64, height: 64, devicePixelRatio: 1 },
      frameIndex: 1,
      frameSeed: 123,
      swapChainView: {} as GPUTextureView,
      swapChainFormat: 'bgra8unorm',
    } as unknown as FrameInput;

    const out = runHybridEngineFrame(deps, input);

    expect(out.kind).toBe('rendered');
    expect(capture.inputs?.camera.prevViewProjMatrix[0]).toBe(10);
    expect(capture.inputs?.camera.prevViewProjMatrix[5]).toBe(21);
    expect(capture.inputs?.camera.prevViewProjMatrix[10]).toBe(44);
    expect(capture.inputs?.camera.prevViewProjMatrix[15]).toBe(1);
  });

  it('packs prevViewProjMatrix at the existing offset-128 matrix slot', () => {
    const prevViewProj = new Float32Array(16);
    for (let i = 0; i < 16; i++) prevViewProj[i] = 100 + i;
    const backing = new Uint8Array(416);

    updateUBO(capturingDevice(backing), {} as GPUBuffer, makePipelineInputs(prevViewProj));

    const f32 = new Float32Array(backing.buffer);
    for (let i = 0; i < 16; i++) {
      expect(f32[32 + i]).toBe(100 + i);
    }
  });

  it('uses the previous view-projection matrix in all reprojection shaders', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('prevViewProjMatrix:');
    expect(MOTION_VECTORS_WGSL).toContain('ubo.prevViewProjMatrix * vec4f(worldPos, 1.0)');
    expect(TEMPORAL_WGSL).toContain('ubo.prevViewProjMatrix * vec4f(world, 1.0)');
    expect(TEMPORAL_GI_COMMON_WGSL).toContain('ubo.prevViewProjMatrix * vec4f(worldPos, 1.0)');
    expect([
      WALKAROUND_UBO_WGSL,
      MOTION_VECTORS_WGSL,
      TEMPORAL_WGSL,
      TEMPORAL_GI_COMMON_WGSL,
    ].join('\n')).not.toContain('prevViewMatrix');
  });
});
