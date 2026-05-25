import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

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
    tileRepeatFactors: Uint8Array | null = null;
    configureAdditiveAccumulation = vi.fn();
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
          uXCmfCdf: { value: null },
          uYCmfCdf: { value: null },
          uZCmfCdf: { value: null },
          uXCmfIntegral: { value: 0 },
          uYCmfIntegral: { value: 0 },
          uZCmfIntegral: { value: 0 },
          uSpectralRendering: { value: -1 },
          uRadianceClamp: { value: -1 },
          uBdptEnabled: { value: false },
          uBdptLightPathTex: { value: null },
          uBdptMaxLightBounces: { value: 0 },
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

// FakeWebGL2RenderingContext + makeRendererStub now live in ./testUtils.ts.

function makeFrame(width: number, height: number) {
  return {
    viewMatrix: asMat4(new Float32Array(16)),
    projMatrix: asMat4(new Float32Array(16)),
    cameraPosition: [0, 0, 0] as const,
    viewport: { width, height, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
    quality: { samplesTarget: 16 },
  };
}

describe('pt-webgl capabilities', () => {
  let teardownGlobalStub: (() => void) | null = null;
  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    // 6.5: explicitly clear the global so subsequent test files don't see
    // a stale stub on globalThis.
    teardownGlobalStub?.();
    teardownGlobalStub = null;
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
