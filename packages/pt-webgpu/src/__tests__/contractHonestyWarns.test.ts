/**
 * Contract-honesty tests for unknown-key warnings:
 *   - causticOptions: unknown keys emit a console.warn with the key name.
 *   - extensions: graduated legacy keys emit a warn with migration pointer.
 *   - extensions: truly unknown keys emit the generic unknown-key warn.
 */
import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
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

describe('causticOptions unknown key warning', () => {
  it('warns when causticOptions contains an unrecognised key', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    await createPTEngine_WebGPU({
      device: makeDevice(),
      causticOptions: {
        mneeMaxIterations: 8,
        unknownCausticParam: 42,
      },
      onWarning: (w) => structured.push(w),
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('unknownCausticParam'))).toBe(true);
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.unknown-caustic-options' &&
      Array.isArray(w.details?.keys) &&
      w.details.keys.includes('unknownCausticParam'),
    )).toBe(true);
    warn.mockRestore();
  });

  it('does NOT warn when causticOptions contains only known keys', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await createPTEngine_WebGPU({
      device: makeDevice(),
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

describe('graduated legacy extension key warnings', () => {
  it('warns with migration pointer for vitrum.ptWebgpu.spectralHeroWavelength.* key', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.spectralHeroWavelength.enable': true },
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(
      calls.some(
        (c) =>
          c.includes('vitrum.ptWebgpu.spectralHeroWavelength') &&
          c.includes('opts.spectral'),
      ),
    ).toBe(true);
    warn.mockRestore();
  });

  it('warns with migration pointer for vitrum.ptWebgpu.bdpt.* key', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.bdpt.enable': true },
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(
      calls.some(
        (c) =>
          c.includes('vitrum.ptWebgpu.bdpt') &&
          c.includes('opts.bdpt'),
      ),
    ).toBe(true);
    warn.mockRestore();
  });

  it('warns with migration pointer for vitrum.ptWebgpu.oidn.* key', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.oidn.modelUrl': 'https://example.com/model.onnx' },
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(
      calls.some(
        (c) =>
          c.includes('vitrum.ptWebgpu.oidn') &&
          c.includes("denoiser:'oidn-final'"),
      ),
    ).toBe(true);
    warn.mockRestore();
  });

  it('warns for truly unknown extension keys (not graduated)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    await createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.futureFeature.enable': true },
      onWarning: (w) => structured.push(w),
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('vitrum.ptWebgpu.futureFeature.enable'))).toBe(true);
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.unknown-extension-key' &&
      Array.isArray(w.details?.keys) &&
      w.details.keys.includes('vitrum.ptWebgpu.futureFeature.enable'),
    )).toBe(true);
    warn.mockRestore();
  });
});
