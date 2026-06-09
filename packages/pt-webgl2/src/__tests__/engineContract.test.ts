import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import type { PTEngineWebGL2Options } from '../index.js';

// A minimal WebGL2RenderingContext stub — enough for the factory's device check +
// trace-tier probe (no rendering in Slice 0). Real GL is exercised on the capture host.
function stubGl(): WebGL2RenderingContext {
  return {
    createFramebuffer: () => ({}),
    getExtension: () => ({}),
    getParameter: () => 8,
    MAX_DRAW_BUFFERS: 0x8824,
  } as unknown as WebGL2RenderingContext;
}

const GREY: MaterialSpec = { baseColor: [0.6, 0.6, 0.6], roughness: 1, metallic: 0 };
function triScene(): Scene {
  const prim: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    normals: new Float32Array(12),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material: GREY,
  };
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } } as Scene;
}

function opts(): PTEngineWebGL2Options {
  return { device: stubGl() };
}

describe('PTEngineWebGL2 — contract conformance (Slice 0 spine)', () => {
  it('factory returns an engine in state "ready"', async () => {
    const e = await createPTEngine_WebGL2(opts());
    expect(e.state).toBe('ready');
  });

  it('advertises the contract capabilities (offscreen-texture, accumulates, caustic field)', async () => {
    const e = await createPTEngine_WebGL2(opts());
    const c = e.capabilities;
    expect(c.presentationMode).toBe('offscreen-texture');
    expect(c.accumulates).toBe(true);
    expect(c.causticStrategy).toBe('none');
    expect(c.supportedPrimitiveKinds?.has('mesh')).toBe(true);
  });

  it('setScene ingests via shared-bvh; getScene returns the filtered scene', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    expect(e.getScene?.()?.primitives.map((p) => p.id)).toEqual(['tri']);
    expect(e._debugGeoPack?.triangleCount).toBe(2);
    expect((e._debugGeoPack?.bvhNodes.length ?? 0) % 8).toBe(0);
  });

  it('renderFrame returns a legal FrameSkipped until the GL pipeline lands', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.setScene(triScene());
    const out = e.renderFrame({
      viewMatrix: new Float32Array(16) as never,
      projMatrix: new Float32Array(16) as never,
      cameraPosition: [0, 0, 5] as never,
      viewport: { width: 64, height: 64, devicePixelRatio: 1 },
      frameIndex: 0,
      frameSeed: 0,
    });
    expect(out.kind).toBe('skipped');
    expect(out.samplesAccumulated).toBe(0);
  });

  it('pause/resume/dispose drive the state machine; mutators throw post-dispose', async () => {
    const e = await createPTEngine_WebGL2(opts());
    e.pause();
    expect(e.state).toBe('paused');
    e.resume();
    expect(e.state).toBe('ready');
    e.dispose();
    expect(e.state).toBe('disposed');
    expect(() => e.setScene(triScene())).toThrow(/disposed/);
    e.dispose(); // idempotent
    expect(e.state).toBe('disposed');
  });

  it('rejects a non-WebGL2 device', async () => {
    await expect(createPTEngine_WebGL2({ device: {} as never })).rejects.toThrow(/WebGL2RenderingContext/);
  });
});
