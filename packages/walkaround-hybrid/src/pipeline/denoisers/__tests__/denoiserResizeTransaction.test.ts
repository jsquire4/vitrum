import { beforeAll, describe, expect, it, vi } from 'vitest';

let Bmfr: typeof import('../bmfr.js').BmfrDenoiser;
let Oidn: typeof import('../oidnFinal.js').OIDNFinalDenoiser;

type MockTexture = { destroy: ReturnType<typeof vi.fn> };
type MockBuffer = { destroy: ReturnType<typeof vi.fn> };

beforeAll(async () => {
  Object.assign(globalThis, {
    GPUTextureUsage: {
      COPY_SRC: 1,
      COPY_DST: 2,
      TEXTURE_BINDING: 4,
      STORAGE_BINDING: 8,
    },
    GPUBufferUsage: {
      STORAGE: 1,
    },
  });
  ({ BmfrDenoiser: Bmfr } = await import('../bmfr.js'));
  ({ OIDNFinalDenoiser: Oidn } = await import('../oidnFinal.js'));
});

describe('denoiser resize transactions', () => {
  it('keeps both BMFR histories when the second replacement allocation fails', () => {
    const oldA: MockTexture = { destroy: vi.fn() };
    const oldB: MockTexture = { destroy: vi.fn() };
    const oldFits: MockBuffer = { destroy: vi.fn() };
    const candidateA: MockTexture = { destroy: vi.fn() };
    const createTexture = vi
      .fn()
      .mockReturnValueOnce(candidateA)
      .mockImplementationOnce(() => {
        throw new Error('injected BMFR history allocation failure');
      });
    const denoiser = new Bmfr();
    const state = denoiser as unknown as Record<string, unknown>;
    Object.assign(state, {
      _device: { createTexture, createBuffer: vi.fn(), limits: {} },
      _sized: {
        historyA: oldA,
        historyB: oldB,
        blockFits: oldFits,
        width: 16,
        height: 16,
      },
      _pingPong: 1,
      _historyValid: true,
    });

    expect(() => denoiser.resize(32, 24)).toThrow('injected BMFR history allocation failure');

    const sized = state._sized as {
      historyA: MockTexture;
      historyB: MockTexture;
      blockFits: MockBuffer;
    };
    expect(sized.historyA).toBe(oldA);
    expect(sized.historyB).toBe(oldB);
    expect(sized.blockFits).toBe(oldFits);
    expect(state._pingPong).toBe(1);
    expect(state._historyValid).toBe(true);
    expect(oldA.destroy).not.toHaveBeenCalled();
    expect(oldB.destroy).not.toHaveBeenCalled();
    expect(oldFits.destroy).not.toHaveBeenCalled();
    expect(candidateA.destroy).toHaveBeenCalledOnce();
  });

  it('keeps OIDN dimensions, generation, and output when replacement fails', () => {
    const old: MockTexture = { destroy: vi.fn() };
    const denoiser = new Oidn({ modelUrl: 'model.onnx' });
    const state = denoiser as unknown as Record<string, unknown>;
    const pending = { marker: true };
    Object.assign(state, {
      _device: {
        createTexture: vi.fn(() => {
          throw new Error('injected OIDN output allocation failure');
        }),
      },
      _denoisedOutputTexture: old,
      _width: 16,
      _height: 16,
      _resizeGeneration: 4,
      _pendingReadback: pending,
      _haveDenoisedOutput: true,
    });

    expect(() => denoiser.resize(32, 24)).toThrow('injected OIDN output allocation failure');

    expect(state._denoisedOutputTexture).toBe(old);
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._resizeGeneration).toBe(4);
    expect(state._pendingReadback).toBe(pending);
    expect(state._haveDenoisedOutput).toBe(true);
    expect(old.destroy).not.toHaveBeenCalled();
  });

  it('publishes OIDN replacement before retiring the old output', () => {
    const old: MockTexture = { destroy: vi.fn() };
    const replacement: MockTexture = { destroy: vi.fn() };
    const denoiser = new Oidn({ modelUrl: 'model.onnx' });
    const state = denoiser as unknown as Record<string, unknown>;
    Object.assign(state, {
      _device: { createTexture: vi.fn(() => replacement) },
      _denoisedOutputTexture: old,
      _width: 16,
      _height: 16,
      _resizeGeneration: 4,
    });

    denoiser.resize(32, 24);

    expect(state._denoisedOutputTexture).toBe(replacement);
    expect(state._width).toBe(32);
    expect(state._height).toBe(24);
    expect(state._resizeGeneration).toBe(5);
    expect(old.destroy).toHaveBeenCalledOnce();
    expect(replacement.destroy).not.toHaveBeenCalled();
  });
});
