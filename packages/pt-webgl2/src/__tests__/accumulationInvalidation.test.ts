import { describe, expect, it, vi } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

const GREY: MaterialSpec = {
  baseColor: [0.6, 0.6, 0.6],
  roughness: 1,
  metallic: 0,
};

function scene(): Scene {
  const primitive: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(6),
    indices: new Uint32Array([0, 2, 1]),
    material: GREY,
  };
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}

function frame(quality: FrameInput['quality'] = { samplesTarget: 16 }): FrameInput {
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
    viewport: { width: 64, height: 64, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 0,
    quality,
  };
}

function withViewTranslation(input: FrameInput, translation: number): FrameInput {
  const viewMatrix = new Float32Array(input.viewMatrix);
  viewMatrix[12] = translation;
  return { ...input, viewMatrix: viewMatrix as never };
}

function withProjectionScale(input: FrameInput, scale: number): FrameInput {
  const projMatrix = new Float32Array(input.projMatrix);
  projMatrix[0] = scale;
  return { ...input, projMatrix: projMatrix as never };
}

describe('PTEngineWebGL2 accumulation invalidation', () => {
  it('derives a distinct geometry seed for each accumulated sample and replays it after reset', async () => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({ device: createMockGl(record) });
    engine.setScene(scene());
    const repeated = { ...frame(), frameIndex: 11, frameSeed: 37 };

    engine.renderFrame(repeated);
    const firstSeed = record.get('seed');
    engine.renderFrame(repeated);
    const secondSeed = record.get('seed');

    expect(firstSeed).not.toBe(secondSeed);

    engine.reset();
    engine.renderFrame(repeated);
    expect(record.get('seed')).toBe(firstSeed);
  });

  it('restarts when either camera matrix changes', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(scene());
    const base = frame();

    expect(engine.renderFrame(base).samplesAccumulated).toBe(1);
    expect(engine.renderFrame(base).samplesAccumulated).toBe(2);
    expect(engine.renderFrame(withViewTranslation(base, 0.25)).samplesAccumulated).toBe(1);
    expect(engine.renderFrame(withProjectionScale(base, 1.75)).samplesAccumulated).toBe(1);
  });

  it('restarts for estimator changes but not for a samplesTarget increase', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(scene());

    expect(engine.renderFrame(frame({ samplesTarget: 8, bounces: 4 })).samplesAccumulated).toBe(1);
    expect(engine.renderFrame(frame({ samplesTarget: 8, bounces: 4 })).samplesAccumulated).toBe(2);
    expect(engine.renderFrame(frame({ samplesTarget: 8, bounces: 5 })).samplesAccumulated).toBe(1);
    expect(engine.renderFrame(frame({
      samplesTarget: 8,
      bounces: 5,
      filteredGlossyFactor: 0.25,
    })).samplesAccumulated).toBe(1);
    expect(engine.renderFrame(frame({
      samplesTarget: 16,
      bounces: 5,
      filteredGlossyFactor: 0.25,
    })).samplesAccumulated).toBe(2);
  });

  it('re-runs only presentation when presentation quality changes after convergence', async () => {
    const gl = createMockGl();
    const drawArrays = vi.fn();
    gl.drawArrays = drawArrays as never;
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(scene());

    const first = engine.renderFrame(frame({ samplesTarget: 1 }));
    expect(first.samplesAccumulated).toBe(1);
    expect(first.isConverged).toBe(true);
    const drawsAfterSample = drawArrays.mock.calls.length;

    const presented = engine.renderFrame(frame({
      samplesTarget: 1,
      exposure: 2,
      tonemap: 'none',
      outputColorSpace: 'linear',
    }));
    expect(presented.samplesAccumulated).toBe(1);
    expect(presented.isConverged).toBe(true);
    expect(drawArrays).toHaveBeenCalledTimes(drawsAfterSample + 1);
  });

  it('rejects non-finite and out-of-range runtime quality before uniforms', async () => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({
      device: createMockGl(record),
      maxBounces: 12,
    });
    engine.setScene(scene());
    const bad = frame({
      samplesTarget: Number.NaN,
      bounces: Number.POSITIVE_INFINITY,
      filteredGlossyFactor: -7,
      resolutionFactor: Number.NaN,
      exposure: Number.POSITIVE_INFINITY,
      tonemap: 'invalid' as never,
      outputColorSpace: 'invalid' as never,
    });
    const recordBefore = new Map(record);

    expect(() => engine.renderFrame(bad)).toThrow(
      /quality\.samplesTarget must be a positive safe integer \(got NaN\)/,
    );
    expect(record).toEqual(recordBefore);
  });

  it('rejects malformed camera numerics before any GL mutation or upload', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    engine.setScene(scene());
    const createFramebuffer = vi.fn(() => ({}));
    const createProgram = vi.fn(() => ({}));
    const texImage2D = vi.fn();
    const uniformMatrix4fv = vi.fn();
    const drawArrays = vi.fn();
    gl.createFramebuffer = createFramebuffer;
    gl.createProgram = createProgram;
    gl.texImage2D = texImage2D as never;
    gl.uniformMatrix4fv = uniformMatrix4fv as never;
    gl.drawArrays = drawArrays as never;

    const base = frame();
    const badViewLength = {
      ...base,
      viewMatrix: new Float32Array(15) as never,
    };
    const badProjLength = {
      ...base,
      projMatrix: new Float32Array(17) as never,
    };
    const badViewFinite = {
      ...base,
      viewMatrix: Float32Array.from(base.viewMatrix, (v, i) => i === 4 ? Number.NaN : v) as never,
    };
    const badProjFinite = {
      ...base,
      projMatrix: Float32Array.from(
        base.projMatrix,
        (v, i) => i === 7 ? Number.POSITIVE_INFINITY : v,
      ) as never,
    };
    const badPosition = {
      ...base,
      cameraPosition: [0, Number.NaN, 5] as never,
    };

    expect(() => engine.renderFrame(badViewLength)).toThrow(/viewMatrix.*exactly 16/);
    expect(() => engine.renderFrame(badProjLength)).toThrow(/projMatrix.*exactly 16/);
    expect(() => engine.renderFrame(badViewFinite)).toThrow(/viewMatrix\[4\].*finite/);
    expect(() => engine.renderFrame(badProjFinite)).toThrow(/projMatrix\[7\].*finite/);
    expect(() => engine.renderFrame(badPosition)).toThrow(/cameraPosition\[1\].*finite/);
    expect(createFramebuffer).not.toHaveBeenCalled();
    expect(createProgram).not.toHaveBeenCalled();
    expect(texImage2D).not.toHaveBeenCalled();
    expect(uniformMatrix4fv).not.toHaveBeenCalled();
    expect(drawArrays).not.toHaveBeenCalled();
  });

  it('contains throwing telemetry subscribers and continues notifying others', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(scene());
    const progress = vi.fn();
    const frames = vi.fn();
    engine.onProgress?.(() => { throw new Error('host progress failure'); });
    engine.onProgress?.(progress);
    engine.onFrame?.(() => { throw new Error('host frame failure'); });
    engine.onFrame?.(frames);

    expect(() => engine.renderFrame(frame())).not.toThrow();
    expect(progress).toHaveBeenCalledTimes(1);
    expect(frames).toHaveBeenCalledTimes(1);
  });
});
