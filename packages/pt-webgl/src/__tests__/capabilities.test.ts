import { beforeAll, describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

vi.mock('three-gpu-pathtracer', () => {
  class WebGLPathTracer {
    readonly target = { texture: {} };
    samples = 0;
    bounces = 0;
    transmissiveBounces = 0;
    filterGlossyFactor = 0;
    renderDelay = 0;
    minSamples = 0;
    dynamicLowRes = false;
    multipleImportanceSampling = false;
    readonly tiles = { set: vi.fn() };
    readonly _pathTracer = {
      material: {
        uniforms: {
          uCausticStrategy: { value: -1 },
          uMneeMaxIterations: { value: 0 },
          uMneeMaxChainLength: { value: 0 },
          uCmfX: { value: null },
          uCmfY: { value: null },
          uCmfZ: { value: null },
          uYCmfCdf: { value: null },
          uYCmfIntegral: { value: 0 },
        },
      },
    };

    setScene(): void {}
    setCamera(): void {}
    renderSample(): void {}
    reset(): void {}
    dispose(): void {}
  }

  return { WebGLPathTracer };
});

class FakeWebGL2RenderingContext {
  readonly MAX_FRAGMENT_UNIFORM_VECTORS = 0x8dfd;

  getParameter(param: number): number {
    if (param === this.MAX_FRAGMENT_UNIFORM_VECTORS) {
      return 512;
    }
    return 0;
  }
}

function makeRendererStub() {
  return {
    getContext: () => new FakeWebGL2RenderingContext(),
    setSize: () => undefined,
  };
}

describe('pt-webgl capabilities', () => {
  beforeAll(() => {
    (globalThis as unknown as { WebGL2RenderingContext: typeof FakeWebGL2RenderingContext })
      .WebGL2RenderingContext = FakeWebGL2RenderingContext;
  });

  it('reports conservative caustic capability even when strategy is requested', async () => {
    const engine = await createPTEngine_WebGL2({
      device: makeRendererStub() as never,
      causticStrategy: 'manifold-nee',
    });
    expect(engine.capabilities.causticStrategy).toBe('none');
  });

  it('warns when non-none caustic strategy is requested', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      await createPTEngine_WebGL2({
        device: makeRendererStub() as never,
        causticStrategy: 'photon-map',
      });
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('causticStrategy="photon-map" requested'),
      );
    } finally {
      warnSpy.mockRestore();
    }
  });
});
