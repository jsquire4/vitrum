import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';
import { SVGFRealDispatcher } from '../denoise/svgfRealDispatcher.js';

function makeStubDevice(): GPUDevice {
  return {
    queue: { submit: vi.fn() },
    createBuffer: vi.fn(() => ({
      destroy: vi.fn(),
      mapAsync: vi.fn(async () => undefined),
      getMappedRange: () => new ArrayBuffer(64),
      unmap: vi.fn(),
    })),
    createCommandEncoder: vi.fn(() => ({
      copyTextureToBuffer: vi.fn(),
      finish: () => ({}),
    })),
  } as unknown as GPUDevice;
}

describe('pt-webgpu svgf-real (WG-9)', () => {
  it("rejects 'svgf-real' on lite tier", async () => {
    const { resolvePtWebgpuTraceTier } = await import('../traceTier.js');
    const tier = resolvePtWebgpuTraceTier(makeStubDevice());
    if (tier !== 'lite') return;

    await expect(
      createPTEngine_WebGPU({ device: makeStubDevice(), denoiser: 'svgf-real', traceTier: 'lite' }),
    ).rejects.toThrow(/svgf-real/);
  });

  it("wires SVGFRealDispatcher when denoiser is 'svgf-real' on full tier", async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      denoiser: 'svgf-real',
      traceTier: 'full',
    });
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-svgf-real')).toBe(true);
    engine.dispose();
  });

  it('SVGFRealDispatcher runs mocked readback + runSVGFRealWebGPU', async () => {
    const w = 2;
    const h = 2;
    const pixels = w * h;
    const readback = vi.fn(async () => ({
      color: new Float32Array(pixels * 3).fill(0.5),
      albedo: new Float32Array(pixels * 3).fill(0.8),
      normal: new Float32Array(pixels * 3).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
      width: w,
      height: h,
    }));
    const runSvgf = vi.fn(async () => new Float32Array(pixels * 3).fill(0.25));

    const dispatcher = new SVGFRealDispatcher({ atrousIterations: 2 }, readback, runSvgf);
    dispatcher.kickIfReady(
      makeStubDevice(),
      { color: {} as GPUTexture, albedo: {} as GPUTexture, normalDepth: {} as GPUTexture },
      w,
      h,
    );
    await vi.waitFor(() => {
      expect(runSvgf).toHaveBeenCalled();
    });
    expect(dispatcher.getLatestDenoised()?.width).toBe(w);
    dispatcher.dispose();
  });
});
