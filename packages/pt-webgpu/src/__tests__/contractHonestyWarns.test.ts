/**
 * Contract-honesty tests for unknown-key warnings:
 *   - causticOptions: unknown keys emit a console.warn with the key name.
 *   - extensions: graduated legacy keys emit a warn with migration pointer.
 *   - extensions: truly unknown keys emit the generic unknown-key warn.
 */
import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

function makeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

describe('causticOptions unknown key warning', () => {
  it('warns when causticOptions contains an unrecognised key', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await createPTEngine_WebGPU({
      device: makeDevice(),
      causticOptions: {
        mneeMaxIterations: 8,
        unknownCausticParam: 42,
      } as Parameters<typeof createPTEngine_WebGPU>[0]['causticOptions'] & Record<string, unknown>,
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('unknownCausticParam'))).toBe(true);
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
    await createPTEngine_WebGPU({
      device: makeDevice(),
      extensions: { 'vitrum.ptWebgpu.futureFeature.enable': true },
    });
    const calls = warn.mock.calls.map((c) => c.join(' '));
    expect(calls.some((c) => c.includes('vitrum.ptWebgpu.futureFeature.enable'))).toBe(true);
    warn.mockRestore();
  });
});
