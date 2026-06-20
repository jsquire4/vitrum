import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

/**
 * pt-webgpu is a converged progressive path tracer; its only wired denoisers are
 * 'none' and 'oidn-final'. 'svgf-real' is a real-time 1-spp spatiotemporal filter
 * and is intentionally NOT wired here — it joins the already-warned unsupported set
	 * (auto / atrous / atrous-variance / bmfr / neural) and degrades to no-denoise.
 */
describe('pt-webgpu unsupported denoisers degrade to no-denoise', () => {
  it("'svgf-real' warns (pointing at oidn-final) and degrades to no-denoise", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      denoiser: 'svgf-real',
    });
    const svgfWarn = warn.mock.calls.find((c) => String(c[0]).includes('denoiser="svgf-real"'));
    expect(svgfWarn).toBeDefined();
    expect(String(svgfWarn?.[0])).toContain('oidn-final');
    // svgf-real must NOT register as a wired/experimental feature.
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-svgf-real')).toBe(false);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-oidn-final')).toBe(false);
    engine.dispose();
    warn.mockRestore();
  });

  it.each(['auto', 'atrous', 'atrous-variance', 'bmfr', 'neural'] as const)(
    "'%s' warns and degrades to no-denoise",
    async (denoiser) => {
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const engine = await createPTEngine_WebGPU({
        device: makeStubDevice(),
        denoiser,
      });
      expect(
        warn.mock.calls.some((c) => String(c[0]).includes(`denoiser="${denoiser}"`)),
      ).toBe(true);
      expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-oidn-final')).toBe(false);
      engine.dispose();
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
    engine.dispose();
    warn.mockRestore();
  });
});
