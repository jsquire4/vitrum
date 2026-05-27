import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';
import {
  OIDNFinalDispatcher,
  type OIDNBridgeLike,
  type OidnReadbackResult,
} from '../denoise/oidnFinalDispatcher.js';

function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

const mockReadback =
  (colorFill = 0.5): import('../denoise/oidnFinalDispatcher.js').OidnReadbackFn =>
  async (_device, _sources, width, height) => {
    const color = new Float32Array(width * height * 3);
    color.fill(colorFill);
    return {
      color,
      albedo: new Float32Array(width * height * 3),
      normal: new Float32Array(width * height * 3),
      width,
      height,
    } satisfies OidnReadbackResult;
  };

describe('pt-webgpu oidn-final (WG-1)', () => {
  it("createPTEngine_WebGPU throws when 'oidn-final' lacks model URL", async () => {
    await expect(
      createPTEngine_WebGPU({
        device: makeStubDevice(),
        denoiser: 'oidn-final',
      }),
    ).rejects.toThrow(/oidnModelUrl/);
  });

  it('does not warn when denoiser is oidn-final', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgpu.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => ({
        denoiseFinal: vi.fn(async () => new Float32Array(0)),
      }),
      oidnReadbackFn: mockReadback(),
    });
    expect(
      warn.mock.calls.some((c) =>
        String(c[0]).includes('no denoiser integration'),
      ),
    ).toBe(false);
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-oidn-final')).toBe(true);
    engine.dispose();
    warn.mockRestore();
  });

  it('OIDNFinalDispatcher forwards albedo + normal aux to denoiseFinal', async () => {
    let resolveDenoise: ((rgb: Float32Array) => void) | null = null;
    const denoisedSentinel = new Float32Array(32 * 16 * 3).fill(9);
    const denoisePromise = new Promise<Float32Array>((res) => {
      resolveDenoise = res;
    });
    const denoiseFinal = vi.fn(async () => denoisePromise);
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      releaseOIDNCacheEntry: vi.fn(),
    };

    const dispatcher = new OIDNFinalDispatcher(
      { modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' },
      async () => bridge,
      mockReadback(0.25),
    );

    dispatcher.kickIfReady(
      makeStubDevice(),
      {
        color: {} as GPUTexture,
        albedo: {} as GPUTexture,
        normalDepth: {} as GPUTexture,
      },
      32,
      16,
    );

    await new Promise((r) => setImmediate(r));
    expect(denoiseFinal).toHaveBeenCalledTimes(1);
    expect(denoiseFinal).toHaveBeenCalledWith(
      expect.objectContaining({
        width: 32,
        height: 16,
        color: expect.any(Float32Array),
        albedo: expect.any(Float32Array),
        normal: expect.any(Float32Array),
      }),
      expect.objectContaining({ modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' }),
    );
    const inputs = (denoiseFinal as ReturnType<typeof vi.fn>).mock.calls[0]?.[0] as {
      color: Float32Array;
    };
    expect(inputs.color[0]).toBeCloseTo(0.25);

    resolveDenoise!(denoisedSentinel);
    await denoisePromise;
    await new Promise((r) => setImmediate(r));

    const got = dispatcher.getLatestDenoised();
    expect(got?.rgb).toBe(denoisedSentinel);
    dispatcher.dispose();
  });

  it('invalidate drops completed denoise until next kick', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(12));
    const dispatcher = new OIDNFinalDispatcher(
      { modelUrl: '/models/oidn_rt_hdr.onnx' },
      async () => ({ denoiseFinal }),
      mockReadback(),
    );
    dispatcher.kickIfReady(
      makeStubDevice(),
      { color: {} as GPUTexture },
      2,
      2,
    );
    await new Promise((r) => setImmediate(r));
    expect(dispatcher.getLatestDenoised()).not.toBeNull();
    dispatcher.invalidate();
    expect(dispatcher.getLatestDenoised()).toBeNull();
    dispatcher.dispose();
  });
});
