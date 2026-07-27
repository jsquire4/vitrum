import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene, FrameInput, EngineWarning } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';
import {
  OIDNFinalDispatcher,
  type OIDNBridgeLike,
} from '../denoise/oidnFinalDispatcher.js';

const GREY: MaterialSpec = { baseColor: [0.6, 0.6, 0.6], roughness: 1, metallic: 0 };

function triScene(): Scene {
  const prim: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } };
}

function frame(spp: number, width = 64, height = 64): FrameInput {
  return {
    viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]) as never,
    projMatrix: new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]) as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width, height, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: spp },
  };
}

function flushAsync(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

interface TrackedGl {
  readonly gl: WebGL2RenderingContext;
  readonly bindings: Array<WebGLTexture | null>;
  readonly uploads: Array<{ readonly texture: WebGLTexture | null; readonly args: readonly unknown[] }>;
  readonly deleted: Array<WebGLTexture | null>;
  readonly drawCount: () => number;
  readonly failNextFloatRgbaUpload: () => void;
}

function createTrackedGl(record = new Map<string, unknown>()): TrackedGl {
  const base = createMockGl(record);
  const bindings: Array<WebGLTexture | null> = [];
  const uploads: Array<{ texture: WebGLTexture | null; args: readonly unknown[] }> = [];
  const deleted: Array<WebGLTexture | null> = [];
  let currentTexture: WebGLTexture | null = null;
  let draws = 0;
  let failNextFloatRgbaUpload = false;
  const gl = new Proxy(base, {
    get(target, prop, receiver): unknown {
      if (prop === 'getProgramParameter') {
        return (program: WebGLProgram, pname: GLenum): unknown => {
          if (pname === base.ACTIVE_UNIFORMS) return 1;
          const fn = Reflect.get(target, prop, receiver) as (p: WebGLProgram, n: GLenum) => unknown;
          return fn(program, pname);
        };
      }
      if (prop === 'getActiveUniform') {
        return (): WebGLActiveInfo => ({ name: 'uAccumTex', size: 1, type: base.SAMPLER_2D });
      }
      if (prop === 'bindTexture') {
        return (_target: GLenum, texture: WebGLTexture | null): void => {
          currentTexture = texture;
          bindings.push(texture);
        };
      }
      if (prop === 'texImage2D') {
        return (...args: unknown[]): void => {
          const data = args[8];
          const isFloatRgbaPayload =
            args[2] === base.RGBA32F && args[6] === base.RGBA && args[7] === base.FLOAT &&
            data instanceof Float32Array && data.length === Number(args[3]) * Number(args[4]) * 4;
          if (failNextFloatRgbaUpload && isFloatRgbaPayload) {
            failNextFloatRgbaUpload = false;
            throw new Error('synthetic OIDN upload failure');
          }
          uploads.push({ texture: currentTexture, args });
          const fn = Reflect.get(target, prop, receiver) as (...values: unknown[]) => void;
          fn(...args);
        };
      }
      if (prop === 'deleteTexture') {
        return (texture: WebGLTexture | null): void => {
          deleted.push(texture);
          const fn = Reflect.get(target, prop, receiver);
          fn(texture);
        };
      }
      if (prop === 'drawArrays') {
        return (...args: unknown[]): void => {
          draws += 1;
          const fn = Reflect.get(target, prop, receiver) as (...values: unknown[]) => void;
          fn(...args);
        };
      }
      return Reflect.get(target, prop, receiver);
    },
  });
  return {
    gl,
    bindings,
    uploads,
    deleted,
    drawCount: () => draws,
    failNextFloatRgbaUpload: () => { failNextFloatRgbaUpload = true; },
  };
}

function oidnUploads(tracked: TrackedGl, width: number, height: number, after = 0) {
  return tracked.uploads.slice(after).filter(({ args }) => {
    const data = args[8];
    return args[2] === tracked.gl.RGBA32F && args[3] === width && args[4] === height &&
      args[6] === tracked.gl.RGBA && args[7] === tracked.gl.FLOAT &&
      data instanceof Float32Array && data.length === width * height * 4 && data[3] === 1;
  });
}

describe('pt-webgl2 OIDN final denoiser', () => {
  it('requires an OIDN model URL when oidn-final is selected', async () => {
    await expect(createPTEngine_WebGL2({
      device: createMockGl(),
      denoiser: 'oidn-final',
    })).rejects.toThrow(/oidn: \{ modelUrl \}/);
  });

  it('kicks OIDN on convergence, reports state, exposes latest result, and invalidates on reset', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const warnings: EngineWarning[] = [];
    const stats: unknown[] = [];
    const denoiserStates: unknown[] = [];
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async (inputs) => new Float32Array(inputs.color.length).fill(0.25)),
      releaseOIDNCacheEntry: vi.fn(),
    };

    try {
      const engine = await createPTEngine_WebGL2({
        device: createMockGl(),
        denoiser: 'oidn-final',
        oidn: { modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' },
        oidnBridgeLoader: async () => bridge,
        onWarning: (w) => warnings.push(w),
      });
      engine.onFrame?.((s) => {
        stats.push(s);
        denoiserStates.push(s.denoiserState);
      });
      engine.setScene(triScene());

      const first = engine.renderFrame(frame(1));
      expect(first.kind).toBe('rendered');
      expect(first.isConverged).toBe(true);
      expect(denoiserStates.at(-1)).toMatchObject({ status: 'in-flight' });
      expect(warnings.some((w) => w.code === 'pt-webgl2.unsupported-denoiser')).toBe(false);

      await flushAsync();
      await flushAsync();

      expect(bridge.denoiseFinal).toHaveBeenCalledTimes(1);
      expect(engine.getLatestDenoised()?.width).toBe(64);
      expect(engine.getLatestDenoised()?.height).toBe(64);
      expect(engine.getLatestDenoised()?.rgb[0]).toBeCloseTo(0.25);

      engine.renderFrame(frame(1));
      expect(denoiserStates.at(-1)).toMatchObject({ status: 'ready' });

      engine.reset();
      expect(engine.getLatestDenoised()).toBeNull();
      engine.dispose();
      expect(bridge.releaseOIDNCacheEntry).toHaveBeenCalled();
      expect(stats.length).toBeGreaterThan(0);

      expect(warn.mock.calls.flat().map(String).some((m) => m.includes('unsupported-denoiser'))).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('presents queued OIDN output on render cadence with correct orientation and retains it across presentation changes', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const record = new Map<string, unknown>();
    const tracked = createTrackedGl(record);
    const topLeftRgb = new Float32Array([
      1, 10, 100, 2, 20, 200,
      3, 30, 300, 4, 40, 400,
    ]);
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async () => topLeftRgb.slice()),
      releaseOIDNCacheEntry: vi.fn(),
    };

    try {
      const engine = await createPTEngine_WebGL2({
        device: tracked.gl,
        denoiser: 'oidn-final',
        oidn: { modelUrl: '/models/oidn.onnx' },
        oidnBridgeLoader: async () => bridge,
      });
      engine.setScene(triScene());
      const first = engine.renderFrame(frame(1, 2, 2));
      expect(first.kind).toBe('rendered');
      const uploadsAtConvergence = tracked.uploads.length;
      const drawsAtConvergence = tracked.drawCount();

      await flushAsync();
      await flushAsync();
      expect(engine.getLatestDenoised()?.rgb).toEqual(topLeftRgb);
      expect(tracked.uploads).toHaveLength(uploadsAtConvergence);
      expect(tracked.drawCount()).toBe(drawsAtConvergence);

      const consumed = engine.renderFrame(frame(1, 2, 2));
      expect(consumed.kind).toBe('rendered');
      if (first.kind === 'rendered' && consumed.kind === 'rendered') {
        expect(consumed.primaryRadiance).toBe(first.primaryRadiance);
      }
      const firstUploads = oidnUploads(tracked, 2, 2, uploadsAtConvergence);
      expect(firstUploads).toHaveLength(1);
      expect(Array.from(firstUploads[0]!.args[8] as Float32Array)).toEqual([
        3, 30, 300, 1, 4, 40, 400, 1,
        1, 10, 100, 1, 2, 20, 200, 1,
      ]);
      const firstSource = firstUploads[0]!.texture;

      const bindingsBefore = tracked.bindings.length;
      const adjusted = engine.renderFrame({
        ...frame(1, 2, 2),
        quality: {
          samplesTarget: 1,
          tonemap: 'none',
          exposure: 2,
          outputColorSpace: 'linear',
        },
      });
      expect(adjusted.kind).toBe('rendered');
      if (first.kind === 'rendered' && adjusted.kind === 'rendered') {
        expect(adjusted.primaryRadiance).toBe(first.primaryRadiance);
      }
      expect(tracked.bindings.slice(bindingsBefore)).toContain(firstSource);
      expect(record.get('uTonemapMode')).toBe(4);
      expect(record.get('uExposure')).toBe(2);
      expect(record.get('uOutputColorSpace')).toBe(1);

      const uploadsBeforeExtension = tracked.uploads.length;
      const extended = engine.renderFrame(frame(2, 2, 2));
      expect(extended).toMatchObject({ kind: 'rendered', samplesAccumulated: 2, isConverged: true });
      expect(tracked.deleted).toContain(firstSource);
      await flushAsync();
      await flushAsync();
      expect(bridge.denoiseFinal).toHaveBeenCalledTimes(2);
      expect(tracked.uploads).toHaveLength(uploadsBeforeExtension);

      engine.renderFrame(frame(2, 2, 2));
      const secondUploads = oidnUploads(tracked, 2, 2, uploadsBeforeExtension);
      expect(secondUploads).toHaveLength(1);
      const secondSource = secondUploads[0]!.texture;
      expect(secondSource).not.toBe(firstSource);
      engine.reset();
      expect(tracked.deleted).toContain(secondSource);
      engine.dispose();
    } finally {
      warn.mockRestore();
    }
  });

  it('invalidates an accepted result when GL presentation fails and retries on the converged fast-out', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const tracked = createTrackedGl();
    const errors: string[] = [];
    let denoiseCycle = 0;
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(async () => new Float32Array(2 * 2 * 3).fill(++denoiseCycle / 10)),
      releaseOIDNCacheEntry: vi.fn(),
    };

    try {
      const engine = await createPTEngine_WebGL2({
        device: tracked.gl,
        denoiser: 'oidn-final',
        oidn: { modelUrl: '/models/oidn.onnx' },
        oidnBridgeLoader: async () => bridge,
      });
      engine.onError?.((error) => errors.push(error.message));
      engine.setScene(triScene());
      engine.renderFrame(frame(1, 2, 2));
      await flushAsync();
      await flushAsync();
      expect(bridge.denoiseFinal).toHaveBeenCalledTimes(1);

      tracked.failNextFloatRgbaUpload();
      engine.renderFrame(frame(1, 2, 2));
      expect(engine.getLatestDenoised()).toBeNull();
      expect(errors.at(-1)).toContain('synthetic OIDN upload failure');

      await flushAsync();
      await flushAsync();
      expect(bridge.denoiseFinal).toHaveBeenCalledTimes(2);
      expect(engine.getLatestDenoised()?.rgb[0]).toBeCloseTo(0.2);
      const uploadsBeforeRetryPresentation = tracked.uploads.length;
      engine.renderFrame(frame(1, 2, 2));
      expect(oidnUploads(tracked, 2, 2, uploadsBeforeRetryPresentation)).toHaveLength(1);
      engine.dispose();
    } finally {
      warn.mockRestore();
    }
  });

  it('retries a reset cohort after an older in-flight inference resolves stale', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const tracked = createTrackedGl();
    const pending: Array<(rgb: Float32Array) => void> = [];
    const bridge: OIDNBridgeLike = {
      denoiseFinal: vi.fn(() => new Promise<Float32Array>((resolve) => {
        pending.push(resolve);
      })),
      releaseOIDNCacheEntry: vi.fn(),
    };

    try {
      const engine = await createPTEngine_WebGL2({
        device: tracked.gl,
        denoiser: 'oidn-final',
        oidn: { modelUrl: '/models/oidn.onnx' },
        oidnBridgeLoader: async () => bridge,
      });
      engine.setScene(triScene());
      engine.renderFrame(frame(1));
      const uploadsAtInitialConvergence = tracked.uploads.length;
      await flushAsync();
      await flushAsync();
      expect(bridge.denoiseFinal).toHaveBeenCalledTimes(1);

      engine.reset();
      engine.renderFrame(frame(1));
      expect(bridge.denoiseFinal).toHaveBeenCalledTimes(1);

      pending[0]!(new Float32Array(64 * 64 * 3).fill(0.1));
      await flushAsync();
      await flushAsync();
      expect(engine.getLatestDenoised()).toBeNull();
      expect(oidnUploads(tracked, 64, 64, uploadsAtInitialConvergence)).toHaveLength(0);

      engine.renderFrame(frame(1));
      await flushAsync();
      await flushAsync();
      expect(bridge.denoiseFinal).toHaveBeenCalledTimes(2);

      pending[1]!(new Float32Array(64 * 64 * 3).fill(0.75));
      await flushAsync();
      await flushAsync();
      expect(engine.getLatestDenoised()?.rgb[0]).toBeCloseTo(0.75);
      expect(oidnUploads(tracked, 64, 64, uploadsAtInitialConvergence)).toHaveLength(0);
      engine.renderFrame(frame(1));
      expect(oidnUploads(tracked, 64, 64, uploadsAtInitialConvergence)).toHaveLength(1);
      engine.dispose();
    } finally {
      warn.mockRestore();
    }
  });

  it('owns a session lease and defers exactly-once release until inference settles', async () => {
    let resolveDenoise!: (rgb: Float32Array) => void;
    const release = vi.fn();
    const bridge: OIDNBridgeLike = {
      acquireOIDNSession: vi.fn(async () => ({ release })),
      denoiseFinal: vi.fn(() => new Promise<Float32Array>((resolve) => {
        resolveDenoise = resolve;
      })),
      releaseOIDNCacheEntry: vi.fn(),
    };
    const dispatcher = new OIDNFinalDispatcher(
      { modelUrl: '/models/lease.onnx' },
      async () => bridge,
    );

    dispatcher.kickIfReady({
      color: new Float32Array(12),
      width: 2,
      height: 2,
    });
    await vi.waitFor(() => expect(bridge.denoiseFinal).toHaveBeenCalledTimes(1));
    expect(bridge.acquireOIDNSession).toHaveBeenCalledTimes(1);

    dispatcher.dispose();
    dispatcher.dispose();
    expect(release).not.toHaveBeenCalled();
    resolveDenoise(new Float32Array(12));
    await vi.waitFor(() => expect(release).toHaveBeenCalledTimes(1));
    expect(dispatcher.getLatestDenoised()).toBeNull();
    expect(bridge.releaseOIDNCacheEntry).not.toHaveBeenCalled();
  });
});
