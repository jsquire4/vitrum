import { describe, expect, it } from 'vitest';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';
import { packWalkaroundUBO } from '../src/pipeline/uboUpdater.js';

function validInputs(): PipelineFrameInputs {
  const matrix = new Float32Array(16);
  return {
    camera: {
      viewMatrix: matrix,
      projMatrix: matrix,
      prevViewProjMatrix: matrix,
      cameraPos: [0, 0, 0],
    },
    screen: {
      screenWidth: 64,
      screenHeight: 64,
      frameSeed: 1,
      swapChainView: {} as GPUTextureView,
      swapChainFormat: 'bgra8unorm',
    },
    lighting: {
      emitterCount: 0,
      primaryLightDir: [0, 1, 0],
      primaryLightIntensity: 1,
      skyTint: [0, 0, 0],
      skyIrradiance: 0,
      emitterDist2Floor: 0.01,
      directFireflyClamp: 4,
      causticBoost: 1,
      causticVisClamp: 1,
    },
    restirDI: {
      temporalMClampDI: 20,
      spatialReuseRadiusPx: 30,
      spatialDepthTolFloor: 0.05,
    },
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
      rayOriginBias: 1e-3,
      glassMixScale: 0.7,
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    composite: { tonemapMode: 0, exposure: 1, outputColorSpace: 0 },
  } as unknown as PipelineFrameInputs;
}

describe('packWalkaroundUBO Float32 upload boundary', () => {
  it('rejects a camera component that overflows while packing to f32', () => {
    const inputs = validInputs();
    inputs.camera.cameraPos[0] = Number.MAX_VALUE;

    expect(() => packWalkaroundUBO(inputs)).toThrow(
      /Float32 lane 48 \(byte offset 192\) must be finite/,
    );
  });

  it('rejects a runtime tunable that overflows while packing to f32', () => {
    const inputs = validInputs();
    inputs.lighting.directFireflyClamp = Number.MAX_VALUE;

    expect(() => packWalkaroundUBO(inputs)).toThrow(
      /Float32 lane 65 \(byte offset 260\) must be finite/,
    );
  });

  it('does not misclassify valid u32 bit patterns as invalid float lanes', () => {
    const inputs = validInputs();
    inputs.screen.frameSeed = 0x7fc00000; // quiet-NaN when viewed as f32
    inputs.screen.screenWidth = 0x7f800000; // +Infinity when viewed as f32
    inputs.lighting.emitterCount = 0xff800000; // -Infinity when viewed as f32

    const words = new Uint32Array(
      packWalkaroundUBO(
        inputs,
        undefined,
        undefined,
        undefined,
        0x7fc00000,
      ),
    );

    expect(words[51]).toBe(0x7fc00000);
    expect(words[52]).toBe(0x7f800000);
    expect(words[54]).toBe(0xff800000);
    expect(words[105]).toBe(0x7fc00000);
  });
});
