import { asMat4, type FrameInput } from '@vitrum/core';
import { describe, expect, it, vi } from 'vitest';

import { HybridEngine } from '../HybridEngine.js';

function frame(width: number, height: number, devicePixelRatio = 1): FrameInput {
  const matrix = asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));
  return {
    viewMatrix: matrix,
    projMatrix: matrix,
    cameraPosition: [0, 0, 0],
    viewport: { width, height, devicePixelRatio },
    frameIndex: 1,
    frameSeed: 2,
  };
}

function makeEngine(resize: () => void): {
  readonly engine: HybridEngine;
  readonly reset: ReturnType<typeof vi.fn>;
} {
  const engine = Object.create(HybridEngine.prototype) as HybridEngine;
  const reset = vi.fn();
  Object.assign(engine as unknown as Record<string, unknown>, {
    _state: 'ready',
    _width: 16,
    _height: 16,
    _internalWidth: 8,
    _internalHeight: 8,
    _resolutionFactor: 0.5,
    _pipeline: { resize },
    _errorFrameCount: 7,
    _cfg: { staticPipelineRebuildKey: 'viewport-test' },
    _rebuildKeyFingerprintSeen: '__different',
    reset,
  });
  return { engine, reset };
}

describe('HybridEngine frame-input boundary and viewport transaction', () => {
  it('derives the canonical camera position when the deprecated field is omitted', () => {
    const resize = vi.fn();
    const { engine } = makeEngine(resize);
    const { cameraPosition: _legacyCameraPosition, ...input } = frame(16, 16);

    expect(() => engine.renderFrame(input)).not.toThrow();
    expect(
      (engine as unknown as {
        _lastFrameCamera: { cameraPosition: readonly number[] };
      })._lastFrameCamera.cameraPosition,
    ).toEqual([0, 0, 0]);
  });

  it('rejects a mismatched legacy camera position before resize or frame publication', () => {
    const resize = vi.fn();
    const { engine, reset } = makeEngine(resize);
    const input = {
      ...frame(32, 24),
      cameraPosition: [0.25, 0, 0],
    } satisfies FrameInput;

    expect(() => engine.renderFrame(input)).toThrow(/cameraPosition.*viewMatrix/i);
    expect(resize).not.toHaveBeenCalled();
    expect(reset).not.toHaveBeenCalled();
    const state = engine as unknown as Record<string, unknown>;
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._errorFrameCount).toBe(7);
    expect(state._lastFrameCamera).toBeUndefined();
  });

  it('resizes changed physical dimensions once and preserves resolutionFactor', () => {
    const resize = vi.fn();
    const { engine, reset } = makeEngine(resize);

    const output = engine.renderFrame(frame(32, 24));

    expect(output.kind).toBe('skipped');
    expect(resize).toHaveBeenCalledOnce();
    expect(resize).toHaveBeenCalledWith(16, 12);
    expect(reset).toHaveBeenCalledOnce();
    const state = engine as unknown as Record<string, unknown>;
    expect(state._width).toBe(32);
    expect(state._height).toBe(24);
    expect(state._internalWidth).toBe(16);
    expect(state._internalHeight).toBe(12);
    expect(state._resolutionFactor).toBe(0.5);
    expect(state._errorFrameCount).toBe(8);
  });

  it('does not reallocate when the per-frame viewport is unchanged', () => {
    const resize = vi.fn();
    const { engine } = makeEngine(resize);

    engine.renderFrame(frame(16, 16));

    expect(resize).not.toHaveBeenCalled();
  });

  it('leaves dimensions and frame state untouched when viewport resize fails', () => {
    const resizeFailure = new Error('injected viewport resize failure');
    const resize = vi.fn(() => {
      throw resizeFailure;
    });
    const { engine, reset } = makeEngine(resize);

    expect(() => engine.renderFrame(frame(32, 24))).toThrow(resizeFailure);

    expect(resize).toHaveBeenCalledOnce();
    expect(reset).not.toHaveBeenCalled();
    const state = engine as unknown as Record<string, unknown>;
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._internalWidth).toBe(8);
    expect(state._internalHeight).toBe(8);
    expect(state._resolutionFactor).toBe(0.5);
    expect(state._errorFrameCount).toBe(7);
    expect(state._rebuildKeyFingerprintSeen).toBe('__different');
  });

  it.each([
    ['zero width', frame(0, 24)],
    ['fractional width', frame(32.5, 24)],
    ['NaN height', frame(32, Number.NaN)],
    ['infinite height', frame(32, Number.POSITIVE_INFINITY)],
    ['unsafe width', frame(Number.MAX_SAFE_INTEGER + 1, 24)],
    ['zero DPR', frame(32, 24, 0)],
    ['NaN DPR', frame(32, 24, Number.NaN)],
    ['infinite DPR', frame(32, 24, Number.POSITIVE_INFINITY)],
  ])('rejects %s before resize or frame-state publication', (_label, input) => {
    const resize = vi.fn();
    const { engine, reset } = makeEngine(resize);

    expect(() => engine.renderFrame(input)).toThrow();

    expect(resize).not.toHaveBeenCalled();
    expect(reset).not.toHaveBeenCalled();
    const state = engine as unknown as Record<string, unknown>;
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._internalWidth).toBe(8);
    expect(state._internalHeight).toBe(8);
    expect(state._resolutionFactor).toBe(0.5);
    expect(state._errorFrameCount).toBe(7);
    expect(state._rebuildKeyFingerprintSeen).toBe('__different');
    expect(state._lastFrameCamera).toBeUndefined();
  });

  it.each([
    ['null input', null],
    ['array input', []],
    ['missing viewport', {}],
    ['null viewport', { viewport: null }],
    ['array viewport', { viewport: [] }],
  ])('rejects %s before reading viewport fields', (_label, input) => {
    const resize = vi.fn();
    const { engine } = makeEngine(resize);

    expect(() => engine.renderFrame(input as unknown as FrameInput)).toThrow();
    expect(resize).not.toHaveBeenCalled();
    expect((engine as unknown as Record<string, unknown>)._errorFrameCount).toBe(7);
  });

  it.each([
    ['short view matrix', { viewMatrix: new Float32Array(15) }],
    ['non-finite view matrix', { viewMatrix: new Float32Array(16).fill(Number.NaN) }],
    ['short projection matrix', { projMatrix: new Float32Array(15) }],
    ['short camera position', { cameraPosition: [0, 0] }],
    ['non-finite camera position', { cameraPosition: [0, Number.POSITIVE_INFINITY, 0] }],
    ['short previous view matrix', { prevViewMatrix: new Float32Array(15) }],
    ['short previous projection matrix', { prevProjMatrix: new Float32Array(15) }],
    ['negative frame index', { frameIndex: -1 }],
    ['fractional frame index', { frameIndex: 1.5 }],
    ['overflow frame index', { frameIndex: 0x100000000 }],
    ['NaN frame seed', { frameSeed: Number.NaN }],
  ])('rejects %s before frame-state publication', (_label, patch) => {
    const resize = vi.fn();
    const { engine, reset } = makeEngine(resize);
    const input = { ...frame(16, 16), ...patch } as unknown as FrameInput;

    expect(() => engine.renderFrame(input)).toThrow();

    expect(resize).not.toHaveBeenCalled();
    expect(reset).not.toHaveBeenCalled();
    const state = engine as unknown as Record<string, unknown>;
    expect(state._errorFrameCount).toBe(7);
    expect(state._rebuildKeyFingerprintSeen).toBe('__different');
    expect(state._lastFrameCamera).toBeUndefined();
  });

  it.each([
    ['null quality', null],
    ['array quality', []],
    ['NaN samplesTarget', { samplesTarget: Number.NaN }],
    ['fractional samplesTarget', { samplesTarget: 1.5 }],
    ['infinite bounces', { bounces: Number.POSITIVE_INFINITY }],
    ['fractional bounces', { bounces: 2.5 }],
    ['NaN resolutionFactor', { resolutionFactor: Number.NaN }],
    ['negative exposure', { exposure: -1 }],
    ['NaN exposure', { exposure: Number.NaN }],
    ['NaN filteredGlossyFactor', { filteredGlossyFactor: Number.NaN }],
    ['unknown tonemap', { tonemap: 'bad' }],
    ['unknown color space', { outputColorSpace: 'bad' }],
  ])('rejects invalid quality payload: %s', (_label, quality) => {
    const resize = vi.fn();
    const { engine, reset } = makeEngine(resize);
    const input = { ...frame(16, 16), quality } as unknown as FrameInput;

    expect(() => engine.renderFrame(input)).toThrow();
    expect(resize).not.toHaveBeenCalled();
    expect(reset).not.toHaveBeenCalled();
    expect((engine as unknown as Record<string, unknown>)._errorFrameCount).toBe(7);
  });

  it('accepts finite integer/out-of-range quality dials for backend clamping', () => {
    const resize = vi.fn();
    const { engine, reset } = makeEngine(resize);
    const input = {
      ...frame(16, 16),
      quality: {
        samplesTarget: -100,
        bounces: 100_000,
        resolutionFactor: -5,
        exposure: 100,
        filteredGlossyFactor: 100,
      },
    } satisfies FrameInput;

    expect(() => engine.renderFrame(input)).not.toThrow();
    expect(resize).not.toHaveBeenCalled();
    expect(reset).toHaveBeenCalledOnce();
  });

  it.each([
    [0, 24],
    [32.5, 24],
    [32, Number.NaN],
    [32, Number.POSITIVE_INFINITY],
    [Number.MAX_SAFE_INTEGER + 1, 24],
  ])('direct setSize(%s, %s) fails closed before allocation', (width, height) => {
    const resize = vi.fn();
    const { engine } = makeEngine(resize);

    expect(() => engine.setSize(width, height)).toThrow(/positive safe integer/);
    expect(resize).not.toHaveBeenCalled();
    const state = engine as unknown as Record<string, unknown>;
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._internalWidth).toBe(8);
    expect(state._internalHeight).toBe(8);
  });
});
