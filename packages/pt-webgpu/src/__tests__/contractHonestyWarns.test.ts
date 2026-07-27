/**
 * Contract-honesty tests for strict option boundaries. Unknown keys throw
 * synchronously; no typo or retired extension can be accepted and ignored.
 */
import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';
import { PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE } from '../webgpuLimits.js';

function makeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

describe('causticOptions strict boundary', () => {
  it('rejects an unrecognised key', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeDevice(),
      causticStrategy: 'manifold-nee',
      causticOptions: {
        mneeMaxIterations: 8,
        unknownCausticParam: 42,
      },
    })).rejects.toThrow(/causticOptions contains unknown key.*unknownCausticParam/);
  });

  it('does NOT warn when causticOptions contains only known keys', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await createPTEngine_WebGPU({
      device: makeDevice(),
      causticStrategy: 'manifold-nee',
      causticOptions: {
        mneeMaxIterations: 4,
        mneeMaxChainLength: 2,
      },
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('causticOptions'))).toBe(false);
    warn.mockRestore();
  });
});

describe('extension keys are rejected', () => {
  it('rejects vitrum.ptWebgpu.spectralHeroWavelength.* with a migration pointer', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.spectralHeroWavelength.enable': true },
    })).rejects.toThrow(/vitrum\.ptWebgpu\.spectralHeroWavelength.*spectral:true/);
  });

  it('rejects vitrum.ptWebgpu.bdpt.* with a migration pointer', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.bdpt.enable': true },
    })).rejects.toThrow(/vitrum\.ptWebgpu\.bdpt.*bdpt:true/);
  });

  it('rejects vitrum.ptWebgpu.oidn.* with a migration pointer', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.oidn.modelUrl': 'https://example.com/model.onnx' },
    })).rejects.toThrow(/vitrum\.ptWebgpu\.oidn.*denoiser:'oidn-final'/);
  });

  it('rejects truly unknown extension keys', async () => {
    await expect(createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.futureFeature.enable': true },
    })).rejects.toThrow(/unsupported key.*vitrum\.ptWebgpu\.futureFeature\.enable/);
  });
});
