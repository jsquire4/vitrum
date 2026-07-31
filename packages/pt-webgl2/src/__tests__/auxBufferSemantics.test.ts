import { describe, expect, it, vi } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

const MATERIAL: MaterialSpec = {
  baseColor: [0.5, 0.6, 0.7],
  roughness: 0.5,
  metallic: 0,
};

function scene(): Scene {
  const primitive: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(6),
    indices: new Uint32Array([0, 2, 1]),
    material: MATERIAL,
  };
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}

function frame(): FrameInput {
  return {
    viewMatrix: new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, -5, 1,
    ]) as never,
    projMatrix: new Float32Array([
      1.5, 0, 0, 0,
      0, 1.5, 0, 0,
      0, 0, -1.002, -1,
      0, 0, -0.2, 0,
    ]) as never,
    cameraPosition: [0, 0, 5] as never,
    viewport: { width: 32, height: 32, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 7,
    quality: { samplesTarget: 8 },
  };
}

describe('PTEngineWebGL2 auxiliary MRT semantics', () => {
  it('keeps forced lite as a real lower-memory output profile', async () => {
    const engine = await createPTEngine_WebGL2({
      device: createMockGl(),
      traceTier: 'lite',
    });
    engine.setScene(scene());

    const output = engine.renderFrame(frame());
    expect(output.kind).toBe('rendered');
    if (output.kind !== 'rendered') return;
    expect(output.normalDepth).toBeUndefined();
    expect(output.albedo).toBeUndefined();
    // The coarse core flag remains false on both tiers because it promises
    // variance and motion vectors in addition to these OIDN-style inputs.
    expect(engine.capabilities.supportsAuxBuffers).toBe(false);
    engine.dispose();
  });

  it('overwrites last-sample auxiliaries and shader-accumulates radiance portably', async () => {
    // createMockGl exposes EXT_float_blend but no OES_draw_buffers_indexed
    // methods. This pins correctness on core WebGL2 without indexed blend state.
    const record = new Map<string, unknown>();
    const gl = createMockGl(record);
    const engine = await createPTEngine_WebGL2({ device: gl, traceTier: 'full' });
    engine.setScene(scene());
    const enable = vi.fn();
    const disable = vi.fn();
    const drawArrays = vi.fn();
    const uniform1f = vi.fn();
    gl.enable = enable as never;
    gl.disable = disable as never;
    gl.drawArrays = drawArrays as never;
    gl.uniform1f = uniform1f as never;

    const first = engine.renderFrame(frame());
    const second = engine.renderFrame(frame());

    expect(first.kind).toBe('rendered');
    expect(second.kind).toBe('rendered');
    if (first.kind !== 'rendered' || second.kind !== 'rendered') return;
    expect(first.normalDepth).toBeDefined();
    expect(first.albedo).toBeDefined();
    expect(second.normalDepth).toBe(first.normalDepth);
    expect(second.albedo).toBe(first.albedo);
    expect(second.samplesAccumulated).toBe(2);

    // Each sample is: shader-accumulated MRT trace + candidate replay +
    // additive no-loop resolve + present. No third radiance target/composite
    // draw remains.
    expect(drawArrays).toHaveBeenCalledTimes(8);
    expect(enable.mock.calls.some(([cap]) => cap === gl.BLEND)).toBe(true);
    expect(disable.mock.calls.some(([cap]) => cap === gl.BLEND)).toBe(true);

    const opacityValues = uniform1f.mock.calls
      .filter(([location]) =>
        typeof location === 'object' &&
        location != null &&
        '__u' in location &&
        (location as { __u: string }).__u === 'opacity')
      .map(([, value]) => value);
    // Main and resolve share the progressive 1/(N+1) factor; the candidate
    // replay records an unweighted proposal.
    expect(opacityValues).toEqual([1, 1, 1, 0.5, 1, 0.5]);
  });
});
