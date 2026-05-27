import { describe, expect, it } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

function makeStubDevice(): GPUDevice {
  return {
    createCommandEncoder: () => ({ finish: () => ({}) }),
    createBuffer: () => ({}),
    createTexture: () => ({}),
    queue: { submit: () => {} },
    limits: { maxStorageBuffersPerShaderStage: 16, maxStorageTexturesPerShaderStage: 8 },
  } as unknown as GPUDevice;
}

describe('pt-webgpu denoiser scope (WG-9 deferred)', () => {
  it("rejects 'svgf-real' — Schied SVGF is walkaround-hybrid only", async () => {
    await expect(
      createPTEngine_WebGPU({ device: makeStubDevice(), denoiser: 'svgf-real', traceTier: 'full' }),
    ).rejects.toThrow(/walkaround-hybrid only/i);
  });
});
