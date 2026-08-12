import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

/**
 * pt-webgpu is a converged progressive path tracer; its only wired denoisers are
 * 'none' and 'oidn-final'. 'svgf-real' is a real-time 1-spp spatiotemporal filter
 * and is intentionally NOT wired here. Explicit unsupported requests fail
 * construction; only `auto` may deliberately resolve to another concrete mode.
 */
describe('pt-webgpu denoiser resolution is strict', () => {
  it("'svgf-real' rejects before allocation instead of degrading to no-denoise", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const device = makeStubDevice();
    await expect(createPTEngine_WebGPU({
      device,
      denoiser: 'svgf-real' as never,
    })).rejects.toThrow(/denoiser="svgf-real".*unsupported.*not degraded/s);
    expect(device.createBuffer).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it("denoiser:'auto' resolves to oidn-final via the default model URL", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      denoiser: 'auto',
    });
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes("denoiser:'auto' resolved to 'oidn-final'")),
    ).toBe(true);
    expect(
      warn.mock.calls.some((c) => String(c[0]).includes('default-oidn-model-url')),
    ).toBe(true);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('unsupported-denoiser'))).toBe(false);
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-oidn-final')).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it.each(['atrous', 'atrous-variance', 'bmfr', 'neural'] as const)(
    "'%s' rejects instead of degrading to no-denoise",
    async (denoiser) => {
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const device = makeStubDevice();
      await expect(createPTEngine_WebGPU({
        device,
        denoiser: denoiser as never,
      })).rejects.toThrow(
        new RegExp(`denoiser="${denoiser}".*unsupported.*not degraded`, 's'),
      );
      expect(device.createBuffer).not.toHaveBeenCalled();
      expect(warn).not.toHaveBeenCalled();
      warn.mockRestore();
    },
  );

  it("'none' produces no denoiser warning", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      denoiser: 'none',
    });
    expect(warn.mock.calls.some((c) => String(c[0]).includes('denoiser='))).toBe(false);
    expect(engine.capabilities.activeFeatures).toEqual(new Set());
    engine.dispose();
    warn.mockRestore();
  });
});
