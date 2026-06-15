import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene, FrameInput, EngineWarning } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';
import type { OIDNBridgeLike } from '../denoise/oidnFinalDispatcher.js';

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

function frame(spp: number): FrameInput {
  return {
    viewMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]) as never,
    projMatrix: new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]) as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 64, height: 64, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: spp },
  };
}

function flushAsync(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
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
});
