/**
 * oidnFinalIntegration.test.ts — W11 follow-up integration test.
 *
 * Verifies the pt-webgl side of the `denoiser: 'oidn-final'` wire:
 *
 *  1. createPTEngine_WebGL2 throws when the host requests 'oidn-final'
 *     without supplying extensions['vitrum.ptWebgl.oidnModelUrl'].
 *  2. When the engine is constructed with 'oidn-final' + a model URL and
 *     a converged frame is rendered, the dispatcher calls denoiseFinal
 *     once with the readback'd HDR RGB + the supplied modelUrl.
 *  3. After the async denoiseFinal resolves, getDenoisedFrame() returns
 *     the result (same width/height + same Float32Array).
 *  4. setScene / reset / updateEnvironment invalidate the cache — a new
 *     converged frame re-kicks denoiseFinal.
 *
 * The OIDN bridge is mocked at the engine layer via the
 * `oidnBridgeLoader` option (test-only) — no need to vi.mock the
 * @vitrum/shared-denoisers module, which keeps this test isolated from
 * the bridge's onnxruntime-web peer-dep concerns.
 *
 * Mirrors the mocking pattern in updateEnvironment.test.ts: WebGLPathTracer
 * is stubbed so we don't need a live WebGL2 context; vitrumSceneToThree
 * is stubbed to return a fresh THREE.Scene.
 */

import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { Scene as ThreeScene, Texture } from 'three';
import type { WebGLRenderTarget } from 'three';
import type { FrameInput } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import type { OIDNBridgeLike } from '../oidnFinalDispatcher.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

vi.mock('@vitrum/three-bindings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/three-bindings')>();
  return {
    ...actual,
    vitrumSceneToThree: vi.fn(() => new ThreeScene()),
  };
});

// WebGLPathTracer stub — same shape as updateEnvironment.test.ts. We add an
// `additiveAccumulation` getter on the target's texture so readback works.
vi.mock('three-gpu-pathtracer', () => {
  class WebGLPathTracer {
    readonly target = { texture: {} } as { texture: unknown };
    samples = 0;
    bounces = 0;
    transmissiveBounces = 0;
    filterGlossyFactor = 0;
    renderDelay = 0;
    minSamples = 0;
    dynamicLowRes = false;
    multipleImportanceSampling = false;
    tileRepeatFactors: Uint8Array | null = null;
    scene: unknown = null;
    configureAdditiveAccumulation = vi.fn();
    readonly tiles = {
      x: 1,
      y: 1,
      set: vi.fn(function (this: { x: number; y: number }, x: number, y: number) {
        this.x = x;
        this.y = y;
      }),
    };
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

    setScene = vi.fn((scene: unknown) => {
      this.scene = scene;
    });
    setCamera = vi.fn();
    /** Each renderSample increments samples by 1 so the test can drive
     *  the engine to convergence by calling renderFrame the desired
     *  number of times. */
    renderSample = vi.fn(() => {
      this.samples += 1;
    });
    reset = vi.fn(() => {
      this.samples = 0;
    });
    dispose = vi.fn();
    updateEnvironment = vi.fn();
  }
  return { WebGLPathTracer };
});

/** Minimal FrameInput satisfying PTEngineWebGL2.renderFrame's contract.
 *  frameIndex is a free parameter so callers can avoid the
 *  cameraSignature short-circuit on back-to-back identical frames. */
let _frameCounter = 0;
function makeFrameInput(samplesTarget: number): FrameInput {
  _frameCounter += 1;
  return {
    frameIndex: _frameCounter,
    frameSeed: _frameCounter,
    viewMatrix: asMat4(new Float32Array(16)),
    projMatrix: asMat4(new Float32Array(16)),
    cameraPosition: [0, 0, 0],
    viewport: { width: 32, height: 16, devicePixelRatio: 1 },
    quality: { samplesTarget, bounces: 2, resolutionFactor: 1 },
  };
}

/** Make a renderer stub that also satisfies the OIDN dispatcher's
 *  readback — `readRenderTargetPixels` fills the supplied Float32 buffer
 *  with a predictable HDR pattern so the test can assert on the values
 *  passed into denoiseFinal. */
function makeRendererForOIDN() {
  const stub = makeRendererStub();
  // Sentinel HDR value — pick a recognizable pattern: R=0.5, G=0.25, B=0.125, A=1.
  const readRenderTargetPixels = vi.fn(
    (_t: WebGLRenderTarget, _x: number, _y: number, w: number, h: number, buf: Float32Array) => {
      const pixelCount = w * h;
      for (let i = 0; i < pixelCount; i += 1) {
        const j = i * 4;
        buf[j] = 0.5;
        buf[j + 1] = 0.25;
        buf[j + 2] = 0.125;
        buf[j + 3] = 1;
      }
    },
  );
  return Object.assign(stub, { readRenderTargetPixels });
}

describe('PTEngineWebGL2 oidn-final wire', () => {
  let teardownGlobalStub: (() => void) | null = null;
  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("createPTEngine_WebGL2 throws when 'oidn-final' is requested without a model URL", async () => {
    await expect(
      createPTEngine_WebGL2({
        device: makeRendererForOIDN() as never,
        denoiser: 'oidn-final',
      }),
    ).rejects.toThrow(/oidnModelUrl/);
  });

  it("createPTEngine_WebGL2 throws when 'oidn-final' is requested with an empty model URL", async () => {
    await expect(
      createPTEngine_WebGL2({
        device: makeRendererForOIDN() as never,
        denoiser: 'oidn-final',
        extensions: { 'vitrum.ptWebgl.oidnModelUrl': '' },
      }),
    ).rejects.toThrow(/oidnModelUrl/);
  });

  it("succeeds with denoiser 'oidn-final' + a model URL + a synthetic bridge loader", async () => {
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async () => new Float32Array(0)),
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
    });
    expect(engine.state).toBe('ready');
    engine.dispose();
  });

  it('kicks denoiseFinal once when a converged frame is rendered', async () => {
    let resolveDenoise: ((rgb: Float32Array) => void) | null = null;
    const denoisedSentinel = new Float32Array(32 * 16 * 3).fill(7);
    const denoisePromise = new Promise<Float32Array>((res) => {
      resolveDenoise = res;
    });
    const denoiseFinal = vi.fn(async () => denoisePromise);
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };

    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
      maxSamplesPerPixel: 8,
    });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    // samplesTarget=1 → first renderFrame converges (renderSample bumps
    // samples from 0 → 1 ≥ target).
    const frameOut = engine.renderFrame(makeFrameInput(1));
    expect(frameOut.isConverged).toBe(true);
    expect(frameOut.samplesAccumulated).toBe(1);

    // The dispatcher kicked synchronously (readback) and then spawns an
    // async chain that awaits the bridge loader before calling
    // denoiseFinal. Flush the microtask queue so the loader resolves
    // and denoiseFinal lands BEFORE the promise resolves below.
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(1);
    const firstCall = denoiseFinal.mock.calls[0] as unknown as [
      { color: Float32Array; width: number; height: number },
      { modelUrl: string },
    ];
    const inputs = firstCall[0];
    const opts = firstCall[1];
    expect(opts.modelUrl).toBe('/models/oidn_rt_hdr.onnx');
    expect(inputs.width).toBe(32);
    expect(inputs.height).toBe(16);
    // First pixel of the readback sentinel — should be (0.5, 0.25, 0.125).
    expect(inputs.color[0]).toBeCloseTo(0.5);
    expect(inputs.color[1]).toBeCloseTo(0.25);
    expect(inputs.color[2]).toBeCloseTo(0.125);

    // Before the inference resolves, getDenoisedFrame returns null.
    expect((engine as unknown as { getDenoisedFrame: () => unknown }).getDenoisedFrame()).toBeNull();

    // Resolve the inference and wait a microtask.
    resolveDenoise!(denoisedSentinel);
    await denoisePromise;
    // Allow the finally() chain in OIDNFinalDispatcher to settle.
    await new Promise((res) => setImmediate(res));

    const got = (engine as unknown as {
      getDenoisedFrame: () => { rgb: Float32Array; width: number; height: number } | null;
    }).getDenoisedFrame();
    expect(got).not.toBeNull();
    expect(got!.width).toBe(32);
    expect(got!.height).toBe(16);
    expect(got!.rgb).toBe(denoisedSentinel);

    engine.dispose();
  });

  it('does NOT kick on non-converged frames', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(0));
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
      maxSamplesPerPixel: 8,
    });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    // samplesTarget=4 with samplesPerFrame=1 (capture mode default) — the
    // first call accumulates 1 sample, still 3 short of the target.
    const out = engine.renderFrame(makeFrameInput(4));
    expect(out.isConverged).toBe(false);
    expect(denoiseFinal).not.toHaveBeenCalled();

    engine.dispose();
  });

  it('does NOT re-kick on subsequent converged frames without invalidation', async () => {
    let resolveDenoise: ((rgb: Float32Array) => void) | null = null;
    const denoisePromise = new Promise<Float32Array>((res) => {
      resolveDenoise = res;
    });
    const denoiseFinal = vi.fn(async () => denoisePromise);
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
      maxSamplesPerPixel: 8,
    });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    // Resolve so the dispatcher transitions to "completed" state.
    resolveDenoise!(new Float32Array(32 * 16 * 3));
    await denoisePromise;
    await new Promise((res) => setImmediate(res));

    // Second converged frame — same cohort, no re-kick.
    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    engine.dispose();
  });

  it('invalidates the cache + re-kicks on reset()', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(32 * 16 * 3));
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
      maxSamplesPerPixel: 8,
    });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    // reset() clears the accumulator AND the OIDN cache. A renderFrame
    // that re-converges will re-kick.
    engine.reset();
    expect(
      (engine as unknown as { getDenoisedFrame: () => unknown }).getDenoisedFrame(),
    ).toBeNull();

    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(2);

    engine.dispose();
  });

  it('invalidates the cache + re-kicks on setScene()', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(32 * 16 * 3));
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
      maxSamplesPerPixel: 8,
    });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(2);

    engine.dispose();
  });

  it('invalidates the cache + re-kicks on updateEnvironment()', async () => {
    const denoiseFinal = vi.fn(async () => new Float32Array(32 * 16 * 3));
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
      maxSamplesPerPixel: 8,
    });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    (engine as unknown as {
      updateEnvironment: (env: { kind: 'hdri'; hdri: Texture; intensity: number }) => void;
    }).updateEnvironment({ kind: 'hdri', hdri: new Texture(), intensity: 1 });
    engine.renderFrame(makeFrameInput(1));
    await new Promise((res) => setImmediate(res));
    expect(denoiseFinal).toHaveBeenCalledTimes(2);

    engine.dispose();
  });

  it('dispose() releases the OIDN session cache entry via releaseOIDNCacheEntry', async () => {
    const releaseOIDNCacheEntry = vi.fn();
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async () => new Float32Array(32 * 16 * 3)),
      preloadOIDNModel: vi.fn(async () => undefined),
      releaseOIDNCacheEntry,
    };
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
      denoiser: 'oidn-final',
      extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
      oidnBridgeLoader: async () => bridge,
      maxSamplesPerPixel: 8,
    });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
    engine.renderFrame(makeFrameInput(1));
    // Wait for the inference cycle to resolve so the bridge has been loaded.
    await new Promise((res) => setImmediate(res));

    engine.dispose();
    expect(releaseOIDNCacheEntry).toHaveBeenCalledTimes(1);
    expect(releaseOIDNCacheEntry).toHaveBeenCalledWith({
      modelUrl: '/models/oidn_rt_hdr.onnx',
    });
  });

  it("getDenoisedFrame returns null when 'oidn-final' was not selected", async () => {
    const engine = await createPTEngine_WebGL2({
      device: makeRendererForOIDN() as never,
    });
    expect(
      (engine as unknown as { getDenoisedFrame: () => unknown }).getDenoisedFrame(),
    ).toBeNull();
    engine.dispose();
  });

  it('swallows denoiseFinal errors so the engine keeps running', async () => {
    const denoiseFinal = vi.fn(async () => {
      throw new Error('mock ORT load failure');
    });
    const bridge: OIDNBridgeLike = {
      denoiseFinal,
      preloadOIDNModel: vi.fn(async () => undefined),
      clearOIDNCache: vi.fn(),
    };
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const engine = await createPTEngine_WebGL2({
        device: makeRendererForOIDN() as never,
        denoiser: 'oidn-final',
        extensions: { 'vitrum.ptWebgl.oidnModelUrl': '/models/oidn_rt_hdr.onnx' },
        oidnBridgeLoader: async () => bridge,
        maxSamplesPerPixel: 8,
      });
      engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

      const out = engine.renderFrame(makeFrameInput(1));
      expect(out.isConverged).toBe(true);
      await new Promise((res) => setImmediate(res));

      expect(denoiseFinal).toHaveBeenCalledTimes(1);
      expect(
        (engine as unknown as { getDenoisedFrame: () => unknown }).getDenoisedFrame(),
      ).toBeNull();
      // Subsequent frame should not re-kick (still in-flight semantics? no
      // — the dispatcher cleared the in-flight flag in finally, but
      // haveCompleted is false because the inference threw. The behaviour
      // we want: the next kick attempt will re-fire so the host can retry
      // after fixing the model URL / network.)
      engine.renderFrame(makeFrameInput(1));
      await new Promise((res) => setImmediate(res));
      expect(denoiseFinal).toHaveBeenCalledTimes(2);

      engine.dispose();
    } finally {
      warn.mockRestore();
    }
  });
});
