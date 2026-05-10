import { beforeAll, describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

vi.mock('@vitrum/three-bindings', () => ({
  vitrumSceneToThree: () => ({
    traverse: () => undefined,
  }),
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
          uSpectralRendering: { value: -1 },
          uRadianceClamp: { value: -1 },
        },
      },
    };

    setScene(): void {}
    setCamera(): void {}
    renderSample(): void {
      this.samples += 1;
    }
    reset(): void {}
    dispose(): void {}
  }

  return { WebGLPathTracer };
});

class FakeWebGL2RenderingContext {
  readonly MAX_FRAGMENT_UNIFORM_VECTORS = 0x8dfd;
  readonly MAX_TEXTURE_SIZE = 0x0d33;
  readonly MAX_RENDERBUFFER_SIZE = 0x84e8;
  readonly RENDERER = 0x1f01;

  getExtension(_name: string): null {
    return null;
  }

  getParameter(param: number): number | string {
    if (param === this.MAX_FRAGMENT_UNIFORM_VECTORS) {
      return 512;
    }
    if (param === this.MAX_TEXTURE_SIZE || param === this.MAX_RENDERBUFFER_SIZE) {
      return 8192;
    }
    if (param === this.RENDERER) {
      return 'Fake WebGL2';
    }
    return 0;
  }
}

function makeRendererStub(options?: { maxSize?: number }) {
  const setSize = vi.fn();
  return {
    getContext: () => new FakeWebGL2RenderingContext(),
    domElement: { addEventListener: vi.fn() },
    setSize,
    _setSize: setSize,
    _maxSize: options?.maxSize,
  };
}

function makeFrame(width: number, height: number) {
  return {
    viewMatrix: new Float32Array(16),
    projMatrix: new Float32Array(16),
    cameraPosition: [0, 0, 0] as const,
    viewport: { width, height, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
    quality: { samplesTarget: 16 },
  };
}

describe('pt-webgl capabilities', () => {
  beforeAll(() => {
    (globalThis as unknown as { WebGL2RenderingContext: typeof FakeWebGL2RenderingContext })
      .WebGL2RenderingContext = FakeWebGL2RenderingContext;
  });

  it('reports requested caustic capability when strategy is configured', async () => {
    const engine = await createPTEngine_WebGL2({
      device: makeRendererStub() as never,
      causticStrategy: 'manifold-nee',
    });
    expect(engine.capabilities.causticStrategy).toBe('manifold-nee');
  });

  it('supports photon-map capability reporting path', async () => {
    const engine = await createPTEngine_WebGL2({
      device: makeRendererStub() as never,
      causticStrategy: 'photon-map',
    });
    expect(engine.capabilities.causticStrategy).toBe('photon-map');
  });

  it('downscales render targets to the configured memory budget', async () => {
    const renderer = makeRendererStub();
    const engine = await createPTEngine_WebGL2({
      device: renderer as never,
      extensions: {
        'vitrum.ptWebgl.qualityMode': 'interactive',
        'vitrum.ptWebgl.renderTargetBudgetBytes': 128 * 1024 * 1024,
      },
    });

    engine.setScene({} as never);
    const out = engine.renderFrame(makeFrame(8192, 8192));
    const telemetry = (out as { telemetry?: { renderWidth: number; renderHeight: number; guardrail: string | null } }).telemetry;

    expect(telemetry?.renderWidth).toBeLessThan(8192);
    expect(telemetry?.renderHeight).toBeLessThan(8192);
    expect(telemetry?.guardrail).toContain('render-target budget');
    expect(renderer._setSize).toHaveBeenCalledWith(telemetry?.renderWidth, telemetry?.renderHeight, false);
  });

  it('adapts sample batches upward when the GPU budget has headroom', async () => {
    const engine = await createPTEngine_WebGL2({
      device: makeRendererStub() as never,
      extensions: {
        'vitrum.ptWebgl.qualityMode': 'interactive',
        'vitrum.ptWebgl.samplesPerFrame': 1,
        'vitrum.ptWebgl.maxSamplesPerFrame': 4,
        'vitrum.ptWebgl.targetBatchMs': 1000,
      },
    });

    engine.setScene({} as never);
    const first = engine.renderFrame(makeFrame(320, 180)) as { telemetry?: { samplesPerFrame: number } };
    const second = engine.renderFrame({ ...makeFrame(320, 180), frameIndex: 1 }) as { telemetry?: { samplesPerFrame: number } };

    expect(first.telemetry?.samplesPerFrame).toBe(1);
    expect(second.telemetry?.samplesPerFrame).toBeGreaterThanOrEqual(2);
  });
});
