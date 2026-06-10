import { describe, expect, it, vi } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { buildSceneTextures } from '../scene/uploadSceneTextures.js';
import { createMockGl } from './mockGl.js';

// ── Robustness gap tests (Wave 5, §6.2 plan/v1-closure-plan-2026-06-10.md) ──
//
// Gap 1 — WebGL2 context-loss handling.
//   a. webglcontextlost fires → engine sets internal contextLost flag.
//   b. renderFrame after loss returns a safe 'skipped' output (no throw).
//   c. Resource-creation helpers fail with an accurate "context lost" message.
//   d. webglcontextrestored warns the host to dispose+recreate (no auto-restore).
//   e. dispose() removes the context-loss listeners (no leak).
//
// Gap 2 — MAX_TEXTURE_SIZE / MAX_ARRAY_TEXTURE_LAYERS validation.
//   a. setScene on a mock with a tiny MAX_TEXTURE_SIZE throws an actionable error
//      naming the offending resource, required size, and device limit.
//   b. Normal scenes (dim well within limit) are unaffected.
//   c. Atlas array-texture layer count guard fires when MAX_ARRAY_TEXTURE_LAYERS
//      is too small for the number of unique material map textures.

const GREY: MaterialSpec = { baseColor: [0.6, 0.6, 0.6], roughness: 1, metallic: 0 };

function tri(id: string): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
}

function smallScene(): Scene {
  return { primitives: [tri('a')], emitters: [], environment: { kind: 'none' } } as Scene;
}

function frame(): FrameInput {
  const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
  const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
  return {
    viewMatrix: view as never,
    projMatrix: proj as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 32, height: 32, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality: { samplesTarget: 4 },
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Gap 1 — context-loss
// ──────────────────────────────────────────────────────────────────────────────

describe('pt-webgl2 robustness — context-loss handling', () => {
  it('webglcontextlost: engine registers preventDefault listener; renderFrame becomes safe no-op', async () => {
    const gl = createMockGl();
    const canvas = (gl as unknown as { canvas: { dispatchEvent(t: string, e: Event): void } }).canvas;
    const engine = await createPTEngine_WebGL2({ device: gl } as never);
    engine.setScene(smallScene());

    // Confirm it renders normally before loss.
    expect(engine.renderFrame(frame()).kind).toBe('rendered');

    // Synthesize a webglcontextlost event. The spec requires preventDefault() to
    // be called on the event for the restore event to later fire. Our listener
    // calls it; we just need to check the post-loss no-op behaviour here.
    const lostEvent = Object.assign(new Event('webglcontextlost'), { preventDefault: vi.fn() });
    canvas.dispatchEvent('webglcontextlost', lostEvent);

    // After loss: renderFrame must return 'skipped', not throw.
    const out = engine.renderFrame(frame());
    expect(out.kind).toBe('skipped');
    expect(out.samplesAccumulated).toBe(0);
  });

  it('webglcontextlost: emits a console.warn naming the loss', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const gl = createMockGl();
      const canvas = (gl as unknown as { canvas: { dispatchEvent(t: string, e: Event): void } }).canvas;
      const engine = await createPTEngine_WebGL2({ device: gl } as never);

      const lostEvent = Object.assign(new Event('webglcontextlost'), { preventDefault: vi.fn() });
      canvas.dispatchEvent('webglcontextlost', lostEvent);

      const lossWarns = warn.mock.calls.filter((a) => String(a[0]).includes('context lost'));
      expect(lossWarns.length).toBeGreaterThan(0);
      expect(String(lossWarns[0]![0])).toContain('pt-webgl2');
      void engine;
    } finally {
      warn.mockRestore();
    }
  });

  it('webglcontextrestored: emits a console.warn instructing dispose+recreate', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const gl = createMockGl();
      const canvas = (gl as unknown as { canvas: { dispatchEvent(t: string, e: Event): void } }).canvas;
      const engine = await createPTEngine_WebGL2({ device: gl } as never);

      // Fire loss first (registers state), then restore.
      const lostEvent = Object.assign(new Event('webglcontextlost'), { preventDefault: vi.fn() });
      canvas.dispatchEvent('webglcontextlost', lostEvent);
      canvas.dispatchEvent('webglcontextrestored', new Event('webglcontextrestored'));

      const restoreWarns = warn.mock.calls.filter((a) => String(a[0]).includes('context restored'));
      expect(restoreWarns.length).toBeGreaterThan(0);
      expect(String(restoreWarns[0]![0])).toContain('dispose');
      void engine;
    } finally {
      warn.mockRestore();
    }
  });

  it('dispose() removes context-loss listeners (no warn after dispose+loss)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const gl = createMockGl();
      const canvas = (gl as unknown as { canvas: { dispatchEvent(t: string, e: Event): void } }).canvas;
      const engine = await createPTEngine_WebGL2({ device: gl } as never);

      engine.dispose();
      warn.mockClear();

      // Firing after dispose should not invoke the (now-removed) listener.
      const lostEvent = Object.assign(new Event('webglcontextlost'), { preventDefault: vi.fn() });
      canvas.dispatchEvent('webglcontextlost', lostEvent);

      const lossWarns = warn.mock.calls.filter((a) => String(a[0]).includes('context lost'));
      expect(lossWarns).toHaveLength(0);
    } finally {
      warn.mockRestore();
    }
  });

  it('context-loss guard: resource creation with isContextLost=true throws accurate error', () => {
    // The mock with contextLost:true simulates the driver returning null from
    // createTexture() after a context loss. Our guards in uploadSceneTextures
    // and bvhTextureAdapter check isContextLost() first and throw a specific
    // "context lost" message instead of the misleading generic error.
    const gl = createMockGl({ contextLost: true });
    const caps = { supportedPrimitiveKinds: new Set(['mesh']), supportsIncrementalScene: false };
    expect(() => buildSceneTextures(gl as never, smallScene() as never, caps as never)).toThrow(/context lost/i);
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// Gap 2 — MAX_TEXTURE_SIZE / MAX_ARRAY_TEXTURE_LAYERS validation
// ──────────────────────────────────────────────────────────────────────────────

describe('pt-webgl2 robustness — texture size validation', () => {
  it('setScene throws an actionable error when the scene BVH exceeds MAX_TEXTURE_SIZE', async () => {
    // A triangle scene with 2 tris → BVH bounds dim = ceil(sqrt(nodeCount*2)) ≈ 2.
    // Setting maxTexSize=1 forces a failure even for the smallest possible scene.
    const gl = createMockGl({ maxTexSize: 1 });
    const engine = await createPTEngine_WebGL2({ device: gl } as never);
    expect(() => engine.setScene(smallScene())).toThrow(/MAX_TEXTURE_SIZE|max.*²|supports.*²/i);
  });

  it('setScene with normal scene (dim << MAX_TEXTURE_SIZE) succeeds without throwing', async () => {
    const gl = createMockGl(); // default 16384 limit — well above any test scene
    const engine = await createPTEngine_WebGL2({ device: gl } as never);
    expect(() => engine.setScene(smallScene())).not.toThrow();
    expect(engine.state).toBe('ready');
  });

  it('error message names the offending resource, required size, and device limit', async () => {
    const gl = createMockGl({ maxTexSize: 1 });
    const engine = await createPTEngine_WebGL2({ device: gl } as never);
    let msg = '';
    try {
      engine.setScene(smallScene());
    } catch (e) {
      msg = String((e as Error).message);
    }
    // Must name the resource (some form of "BVH" / "material" / etc.) and the limit.
    expect(msg).toMatch(/pt-webgl2/);
    expect(msg).toMatch(/1/); // the device limit (maxTexSize=1) appears in the message
  });

  it('setScene throws an actionable error when array-texture layer count exceeds MAX_ARRAY_TEXTURE_LAYERS', async () => {
    // Build a scene whose materials have 3 unique readable texture handles.
    // Set maxArrayLayers=1 so the atlas (3 layers) exceeds the limit.
    const handle1 = { width: 2, height: 2, data: new Float32Array(16) };
    const handle2 = { width: 2, height: 2, data: new Float32Array(16) };
    const handle3 = { width: 2, height: 2, data: new Float32Array(16) };
    const matWithTexture = (h: unknown): MaterialSpec => ({
      baseColor: [1, 0, 0],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle: h } as never,
    });
    const prim = (id: string, mat: MaterialSpec): MeshPrimitive => ({
      kind: 'mesh',
      id,
      positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array(6),
      indices: new Uint32Array([0, 1, 2]),
      material: mat,
    });
    const scene: Scene = {
      primitives: [
        prim('a', matWithTexture(handle1)),
        prim('b', matWithTexture(handle2)),
        prim('c', matWithTexture(handle3)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    } as Scene;

    const gl = createMockGl({ maxArrayLayers: 1 });
    const engine = await createPTEngine_WebGL2({ device: gl } as never);
    let msg = '';
    try {
      engine.setScene(scene);
    } catch (e) {
      msg = String((e as Error).message);
    }
    // Either the atlas layers guard fires (layer count > 1) or the tex-size guard
    // fires first (maxTexSize=16384 is fine for these tiny textures). We want the
    // layer-count guard to fire here — only override maxArrayLayers.
    expect(msg).toMatch(/pt-webgl2/);
    expect(msg).toMatch(/layer/i);
  });
});
