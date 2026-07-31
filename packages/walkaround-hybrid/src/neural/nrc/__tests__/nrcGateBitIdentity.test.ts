// NRC is a construction-time pipeline/resource choice. The former per-frame
// UBO mirror was never read by WGSL; its word is now the independent ReSTIR
// reservoir scale. These tests pin the live replacement and ensure later
// controls remain aligned.

import { describe, it, expect } from 'vitest';
import {
  WALKAROUND_DEFAULT_SUN_ANGULAR_RADIUS,
  WALKAROUND_UBO_SIZE_BYTES,
} from '../../../pipeline/constants.ts';
import { updateUBO } from '../../../pipeline/uboUpdater.ts';
import type { PipelineFrameInputs } from '../../../pipeline/WalkaroundGPUPipeline.ts';
import { WALKAROUND_UBO_WGSL } from '../../../shaders/walkaroundUbo.wgsl.ts';

// A fake GPUDevice whose queue.writeBuffer captures the bytes updateUBO writes.
function makeCapturingDevice(): { device: GPUDevice; captured: () => Uint8Array } {
  let last: Uint8Array | null = null;
  const queue = {
    writeBuffer: (_buf: unknown, _off: number, data: BufferSource) => {
      const ab = (data as ArrayBufferView).buffer ?? (data as ArrayBuffer);
      last = new Uint8Array(ab.slice(0));
    },
  };
  const device = { queue } as unknown as GPUDevice;
  return { device, captured: () => last ?? new Uint8Array() };
}

// Minimal-but-complete inputs for the UBO write. updateUBO only reads the
// UBO-relevant numeric/array fields (never swapChainView), so we cast a partial.
function baseInputs(): PipelineFrameInputs {
  const m = new Float32Array(16);
  for (let i = 0; i < 16; i++) m[i] = i * 0.5 + 1;
  return {
    camera: { viewMatrix: m, projMatrix: m, prevViewProjMatrix: m, cameraPos: [1, 2, 3] },
    screen: { screenWidth: 1920, screenHeight: 1080, frameSeed: 42, swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm' },
    lighting: {
      emitterCount: 7,
      primaryLightDir: [0, -1, 0], primaryLightIntensity: 3,
      skyTint: [0.6, 0.7, 0.9], skyIrradiance: 1.5,
      emitterDist2Floor: 0.01, directFireflyClamp: 4,
      causticBoost: 1, causticVisClamp: 1,
      lightTreeEnabled: 0, lightTreeNodeCount: 0,
    },
    restirDI: { temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05 },
    restirGI: {
      restirGiWCap: 16, restirGiIrrClamp: 5, restirGiMClamp: 50,
      restirGiSpatialRadiusPx: 12, restirGiSpatialNormalDotMin: 0.906,
      restirGiSpatialCoplanarTol: 0.05,
    },
    gtao: { gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2, gtaoBilateralDepthSigma: 0.25, adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1 },
    filter: {
      triIntersectEpsilon: 1e-5, rayOriginBias: 1e-3, glassMixScale: 0.7,
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
  };
}

const RESTIR_RESERVOIR_SCALE_U32_INDEX = 91; // offset 364 / 4

describe('NRC construction-time gate — no dead UBO mirror', () => {
  it('keeps only genuinely retired fields as explicit ABI pads in WGSL', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('_abiPadEmitterPower:');
    expect(WALKAROUND_UBO_WGSL).toContain('restirReservoirScale:');
    expect(WALKAROUND_UBO_WGSL).toContain('rayOriginBias:');
    expect(WALKAROUND_UBO_WGSL).not.toMatch(/\n\s+nrcEnabled\s*:/);
    expect(WALKAROUND_UBO_WGSL).not.toMatch(/\n\s+grisReuse\s*:/);
    expect(WALKAROUND_UBO_WGSL).not.toMatch(/\n\s+totalEmPower\s*:/);
  });

  it('writes the default reservoir scale, live ray bias, and retired emitter slot', () => {
    const capture = makeCapturingDevice();
    updateUBO(capture.device, {} as GPUBuffer, baseInputs());
    const bytes = capture.captured();
    expect(bytes.length).toBe(WALKAROUND_UBO_SIZE_BYTES);
    const u32 = new Uint32Array(bytes.buffer.slice(0));
    const f32 = new Float32Array(bytes.buffer.slice(0));
    expect(f32[55]).toBe(0);
    expect(u32[RESTIR_RESERVOIR_SCALE_U32_INDEX]).toBe(1);
    expect(f32[103]).toBeCloseTo(1e-3);
    expect(f32[104]).toBeCloseTo(WALKAROUND_DEFAULT_SUN_ANGULAR_RADIUS);
  });
});
