import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import type { PTEngineWebGL2Options } from '../index.js';
import { createMockGl } from './mockGl.js';

// The mock GL drives the full pipeline without a GPU — it verifies the engine
// ORCHESTRATION (accumulation loop, convergence, FrameOutput shape, lifecycle),
// NOT pixel correctness (that is the real-GPU capture-host A/B, plan 06).

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

function tri(id: string, x: number): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([x, -1, 0, x + 1, -1, 0, x + 1, 1, 0, x, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
}

function sceneWithEmitter(): Scene {
  return {
    primitives: [tri('tri', 0)],
    emitters: [
      {
        kind: 'point',
        id: 'point-a',
        position: [1, 2, 3],
        color: [1, 1, 1],
        intensity: 2,
      },
    ],
    environment: { kind: 'none' },
  };
}

function hdriScene(): Scene {
  return {
    primitives: [tri('tri', 0)],
    emitters: [],
    environment: {
      kind: 'hdri',
      hdri: {
        width: 1,
        height: 1,
        data: new Float32Array([0.25, 0.5, 1]),
      },
      intensity: 1,
    },
  };
}

function opts(): PTEngineWebGL2Options {
  return { device: createMockGl() };
}

function frame(spp: number): FrameInput {
  // an identity-ish view + a finite invertible projection (packFrameParams inverts both)
  const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
  const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
  return {
    viewMatrix: view as never,
    projMatrix: proj as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 64, height: 64, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: spp },
  };
}

describe('PTEngineWebGL2 — contract conformance + accumulation orchestration', () => {
  it('factory returns an engine in state "ready"; rejects a non-WebGL2 device', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(e.state).toBe('ready');
    await expect(createPTEngine_WebGL2({ device: {} as never })).rejects.toThrow(/WebGL2RenderingContext/);
  });

  it('advertises the contract capabilities (offscreen-texture, accumulates, caustic field)', async () => {
    const c = (await createPTEngine_WebGL2(opts())).capabilities;
    expect(c.presentationMode).toBe('offscreen-texture');
    expect(c.accumulates).toBe(true);
    expect(c.causticStrategy).toBe('none');
    expect(c.supportedPrimitiveKinds?.has('mesh')).toBe(true);
    expect(c.supportsIncrementalScene).toBe(true);
    expect(c.incrementalPatchSupport).toEqual({
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    });
    expect(c.supportsAddRemovePrimitive).toBe(true);
    expect(c.supportDetails?.mutations.material).toBe('fallback-rebuild');
    expect(c.supportDetails?.mutations.environment).toBe('fallback-rebuild');
    expect(c.supportDetails?.denoisers).toEqual({
      none: 'native',
      atrous: 'unsupported',
      'atrous-variance': 'unsupported',
      'svgf-real': 'unsupported',
      bmfr: 'unsupported',
      'oidn-final': 'unsupported',
      neural: 'unsupported',
    });
  });

  it('exposes fallback-rebuild scene mutation methods', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.updatePrimitive).toBe('function');
    expect(typeof e.updateEmitter).toBe('function');
    expect(typeof e.updateEnvironment).toBe('function');
    expect(typeof e.addPrimitive).toBe('function');
    expect(typeof e.removePrimitive).toBe('function');
  });

  it('setScene ingests via shared-bvh; getScene returns the filtered scene', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['tri']);
    expect(e._debugGeoPack?.triangleCount).toBe(2);
  });

  it('warns when displacement material fields are supplied', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [{
          ...prim,
          material: {
            ...prim.material,
            displacementMap: { handle: { id: 'height' } },
            displacementScale: 0.2,
            displacementBias: -0.1,
          },
        }],
      };
      e.setScene(scene);
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('displacementMap') &&
        m.includes('displacementScale') &&
        m.includes('displacementBias'),
      )).toBe(true);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-displacement-material' &&
        Array.isArray(w.details?.fields) &&
        w.details.fields.includes('displacementMap') &&
        w.details.fields.includes('displacementScale') &&
        w.details.fields.includes('displacementBias'),
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('warns when unsupported (anisotropy) material fields are supplied (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      const base = triScene();
      const prim = base.primitives[0] as MeshPrimitive;
      const scene: Scene = {
        ...base,
        primitives: [{
          ...prim,
          material: {
            ...prim.material,
            anisotropy: 0.8,
            anisotropyRotation: 0.5,
            anisotropyMap: { handle: { id: 'aniso' } },
          },
        }],
      };
      e.setScene(scene);
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-material-fields' &&
        Array.isArray(w.details?.fields) &&
        w.details.fields.includes('anisotropy') &&
        w.details.fields.includes('anisotropyRotation') &&
        w.details.fields.includes('anisotropyMap'),
      )).toBe(true);
      expect(warn.mock.calls.flat().map(String).some((m) =>
        m.includes('anisotropy') && m.includes('not rendered'),
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it('does NOT emit the unsupported-material-fields warning for a plain supported material (CAP-01)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const e = await createPTEngine_WebGL2({
        ...opts(),
        onWarning: (w) => structured.push(w),
      });
      e.setScene(triScene());
      expect(structured.some((w) => w.code === 'pt-webgl2.unsupported-material-fields')).toBe(false);
    } finally {
      warn.mockRestore();
    }
  });

  it('renderFrame skips before a scene, then renders + accumulates one sample per call', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(e.renderFrame(frame(16)).kind).toBe('skipped'); // no scene yet
    e.setScene(triScene());
    const f1 = e.renderFrame(frame(16));
    expect(f1.kind).toBe('rendered');
    expect(f1.samplesAccumulated).toBe(1);
    expect(f1.isConverged).toBe(false);
    if (f1.kind === 'rendered') expect(f1.primaryRadiance).toBeDefined();
    const f2 = e.renderFrame(frame(16));
    expect(f2.samplesAccumulated).toBe(2);
  });

  it('converges at samplesTarget and stops advancing', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    let out = e.renderFrame(frame(4));
    for (let i = 0; i < 5; i += 1) out = e.renderFrame(frame(4));
    expect(out.samplesAccumulated).toBe(4);
    expect(out.isConverged).toBe(true);
  });

  it('emits pt-spp progress telemetry', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    const seen: Array<{ current: number; target: number }> = [];
    e.onProgress?.((p) => seen.push({ current: p.current, target: p.target }));
    e.renderFrame(frame(16));
    expect(seen.at(-1)).toEqual({ current: 1, target: 16 });
  });

  it('reset() restarts accumulation; pause/resume/dispose drive the state machine', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    e.renderFrame(frame(16));
    e.reset();
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(1);
    e.pause();
    expect(e.state).toBe('paused');
    e.resume();
    expect(e.state).toBe('ready');
    e.dispose();
    expect(e.state).toBe('disposed');
    expect(() => e.setScene(triScene())).toThrow(/disposed/);
    e.dispose(); // idempotent
  });

  it('updatePrimitive rebuilds from the patched retained scene and resets accumulation', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    e.renderFrame(frame(16));
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(2);

    e.updatePrimitive?.('tri', { material: { roughness: 0.25 } } as never);
    const scene = e.getScene?.();
    const prim = scene?.primitives[0];
    expect(prim?.kind).toBe('mesh');
    if (prim?.kind === 'mesh') {
      expect(prim.material.roughness).toBe(0.25);
      expect(prim.material.baseColor).toEqual(GREY.baseColor);
    }
    expect(e.renderFrame(frame(16)).samplesAccumulated).toBe(1);
  });

  it('updateEmitter and updateEnvironment rebuild from patched scene snapshots', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(sceneWithEmitter());

    e.updateEmitter?.('point-a', { intensity: 4 });
    expect(e.getScene?.()?.emitters[0]?.intensity).toBe(4);

    e.updateEnvironment?.(hdriScene().environment);
    const scene = e.getScene?.();
    expect(scene?.environment.kind).toBe('hdri');
    expect(e._debugSceneTex?.envMap).toBe(true);
  });

  // Contract-honesty: EngineOptions.denoiser must not be silently ignored.
  // pt-webgl2 has no denoiser pipeline; non-null non-'none' values must warn once.
  it("denoiser: 'none' and absent denoiser are both silent", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await createPTEngine_WebGL2({ ...opts(), denoiser: 'none' });
      await createPTEngine_WebGL2(opts()); // absent
      const denoiserWarns = warn.mock.calls.filter((args) =>
        String(args[0]).includes('denoiser'),
      );
      expect(denoiserWarns).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it("denoiser: 'oidn-final' emits exactly one console.warn naming the value", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      await createPTEngine_WebGL2({
        ...opts(),
        denoiser: 'oidn-final',
        onWarning: (w) => structured.push(w),
      });
      const denoiserWarns = warn.mock.calls.filter((args) =>
        String(args[0]).includes('denoiser'),
      );
      expect(denoiserWarns).toHaveLength(1);
      expect(String(denoiserWarns[0]![0])).toContain('oidn-final');
      expect(String(denoiserWarns[0]![0])).toContain('pt-webgl2');
      expect(structured.some((w) =>
        w.code === 'pt-webgl2.unsupported-denoiser' &&
        w.details?.requested === 'oidn-final',
      )).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  it("denoiser: 'svgf-real' emits exactly one console.warn naming the value", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await createPTEngine_WebGL2({ ...opts(), denoiser: 'svgf-real' });
      const denoiserWarns = warn.mock.calls.filter((args) =>
        String(args[0]).includes('denoiser'),
      );
      expect(denoiserWarns).toHaveLength(1);
      expect(String(denoiserWarns[0]![0])).toContain('svgf-real');
      expect(String(denoiserWarns[0]![0])).toContain('pt-webgl2');
    } finally {
      warn.mockRestore();
    }
  });

  it('exposes onError subscription (item 28 — GPU error surface)', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.onError).toBe('function');
    const errors: unknown[] = [];
    const unsub = e.onError!((err) => errors.push(err));
    expect(typeof unsub).toBe('function');
    // Unsubscribe should not throw.
    expect(() => unsub()).not.toThrow();
    // After unsubscribe, further unsub calls are idempotent.
    expect(() => unsub()).not.toThrow();
    e.dispose();
  });

  it('exposes onWarning subscription (ENGINE-01 warning surface)', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(typeof e.onWarning).toBe('function');
    const warnings: EngineWarning[] = [];
    const unsub = e.onWarning!((warning) => warnings.push(warning));
    expect(typeof unsub).toBe('function');
    expect(() => unsub()).not.toThrow();
    e.dispose();
  });

  it('addPrimitive and removePrimitive rebuild, validate ids, and allow an empty scene', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene({ primitives: [tri('a', 0)], emitters: [], environment: { kind: 'none' } });

    e.addPrimitive?.(tri('b', 2));
    expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['a', 'b']);
    expect(() => e.addPrimitive?.(tri('b', 4))).toThrow(/already exists/);

    e.removePrimitive?.('a');
    expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['b']);
    expect(() => e.removePrimitive?.('missing')).toThrow(/not found/);

    e.removePrimitive?.('b');
    expect(e.getScene?.()?.primitives).toEqual([]);
    expect(e.renderFrame(frame(16)).kind).toBe('rendered');
  });
});
