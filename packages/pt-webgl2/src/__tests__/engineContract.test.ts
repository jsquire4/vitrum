import { describe, expect, it } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
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
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } } as Scene;
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
    expect(c.supportsIncrementalScene).toBe(false);
    expect(c.incrementalPatchSupport).toEqual({
      transform: false,
      positions: false,
      material: false,
      emitter: false,
      topology: false,
    });
    expect(c.supportDetails?.mutations.material).toBe('unsupported');
    expect(c.supportDetails?.mutations.environment).toBe('unsupported');
  });

  it('setScene ingests via shared-bvh; getScene returns the filtered scene', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['tri']);
    expect(e._debugGeoPack?.triangleCount).toBe(2);
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
});
