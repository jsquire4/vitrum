import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { DenoiserInitContext } from '../index.js';

let Bmfr: typeof import('../bmfr.js').BmfrDenoiser;

beforeAll(async () => {
  Object.assign(globalThis, {
    GPUTextureUsage: {
      COPY_SRC: 1,
      COPY_DST: 2,
      TEXTURE_BINDING: 4,
      STORAGE_BINDING: 8,
    },
    GPUBufferUsage: {
      UNIFORM: 1,
      COPY_DST: 2,
      STORAGE: 4,
    },
  });
  ({ BmfrDenoiser: Bmfr } = await import('../bmfr.js'));
});

function context(device: object): DenoiserInitContext {
  return {
    device,
    width: 16,
    height: 16,
    bglCache: {},
    frameResources: {},
  } as unknown as DenoiserInitContext;
}

function pipeline() {
  return { getBindGroupLayout: vi.fn(() => ({})) };
}

function shader() {
  return {
    getCompilationInfo: vi.fn(async () => ({ messages: [] })),
  };
}

describe('BmfrDenoiser lifecycle transactions', () => {
  it('destroys the UBO and both candidate histories when block-fit allocation fails', async () => {
    const ubo = { destroy: vi.fn() };
    const historyA = { destroy: vi.fn(() => { throw new Error('hostile destroy'); }) };
    const historyB = { destroy: vi.fn() };
    const createBuffer = vi
      .fn()
      .mockReturnValueOnce(ubo)
      .mockImplementationOnce(() => {
        throw new Error('block-fit allocation failed');
      });
    const device = {
      limits: {},
      createShaderModule: vi.fn(shader),
      createComputePipelineAsync: vi.fn(async () => pipeline()),
      createBuffer,
      createTexture: vi
        .fn()
        .mockReturnValueOnce(historyA)
        .mockReturnValueOnce(historyB),
    };
    const denoiser = new Bmfr();

    await expect(denoiser.initialize(context(device))).rejects.toThrow(
      /block-fit allocation failed/,
    );
    expect(ubo.destroy).toHaveBeenCalledOnce();
    expect(historyA.destroy).toHaveBeenCalledOnce();
    expect(historyB.destroy).toHaveBeenCalledOnce();
    const state = denoiser as unknown as Record<string, unknown>;
    expect(state._ubo).toBeNull();
    expect(state._sized).toBeNull();
  });

  it('does not let hostile retirement reject a successful reinitialization', async () => {
    const previousUbo = { destroy: vi.fn(() => { throw new Error('ubo destroy'); }) };
    const previousHistoryA = {
      destroy: vi.fn(() => { throw new Error('history-a destroy'); }),
    };
    const previousHistoryB = { destroy: vi.fn() };
    const previousBlockFits = { destroy: vi.fn() };
    const nextUbo = { destroy: vi.fn() };
    const nextHistoryA = { destroy: vi.fn() };
    const nextHistoryB = { destroy: vi.fn() };
    const nextBlockFits = { destroy: vi.fn() };
    const device = {
      limits: {},
      createShaderModule: vi.fn(shader),
      createComputePipelineAsync: vi.fn(async () => pipeline()),
      createBuffer: vi.fn()
        .mockReturnValueOnce(nextUbo)
        .mockReturnValueOnce(nextBlockFits),
      createTexture: vi.fn()
        .mockReturnValueOnce(nextHistoryA)
        .mockReturnValueOnce(nextHistoryB),
    };
    const denoiser = new Bmfr();
    Object.assign(denoiser as unknown as Record<string, unknown>, {
      _ubo: previousUbo,
      _sized: {
        historyA: previousHistoryA,
        historyB: previousHistoryB,
        blockFits: previousBlockFits,
        width: 8,
        height: 8,
      },
    });

    await expect(denoiser.initialize(context(device))).resolves.toBeUndefined();
    expect(previousUbo.destroy).toHaveBeenCalledOnce();
    expect(previousHistoryA.destroy).toHaveBeenCalledOnce();
    expect(previousHistoryB.destroy).toHaveBeenCalledOnce();
    expect(previousBlockFits.destroy).toHaveBeenCalledOnce();
    const state = denoiser as unknown as Record<string, unknown>;
    expect(state._ubo).toBe(nextUbo);
    expect(state._sized).toMatchObject({
      historyA: nextHistoryA,
      historyB: nextHistoryB,
      blockFits: nextBlockFits,
    });
    denoiser.dispose();
  });

  it('publishes resize resources even when each old resource retirement is hostile', () => {
    const previousHistoryA = {
      destroy: vi.fn(() => { throw new Error('history-a destroy'); }),
    };
    const previousHistoryB = {
      destroy: vi.fn(() => { throw new Error('history-b destroy'); }),
    };
    const previousBlockFits = { destroy: vi.fn() };
    const nextHistoryA = { destroy: vi.fn() };
    const nextHistoryB = { destroy: vi.fn() };
    const nextBlockFits = { destroy: vi.fn() };
    const device = {
      limits: {},
      createTexture: vi.fn()
        .mockReturnValueOnce(nextHistoryA)
        .mockReturnValueOnce(nextHistoryB),
      createBuffer: vi.fn(() => nextBlockFits),
    };
    const denoiser = new Bmfr();
    Object.assign(denoiser as unknown as Record<string, unknown>, {
      _device: device,
      _sized: {
        historyA: previousHistoryA,
        historyB: previousHistoryB,
        blockFits: previousBlockFits,
        width: 8,
        height: 8,
      },
    });

    expect(() => denoiser.resize(16, 16)).not.toThrow();
    expect(previousHistoryA.destroy).toHaveBeenCalledOnce();
    expect(previousHistoryB.destroy).toHaveBeenCalledOnce();
    expect(previousBlockFits.destroy).toHaveBeenCalledOnce();
    expect((denoiser as unknown as { _sized: unknown })._sized).toMatchObject({
      historyA: nextHistoryA,
      historyB: nextHistoryB,
      blockFits: nextBlockFits,
      width: 16,
      height: 16,
    });
    denoiser.dispose();
  });

  it('does not publish resources when dispose supersedes async pipeline creation', async () => {
    const resolvers: Array<(value: object) => void> = [];
    const createComputePipelineAsync = vi.fn(
      () =>
        new Promise<object>((resolve) => {
          resolvers.push(resolve);
        }),
    );
    const device = {
      limits: {},
      createShaderModule: vi.fn(shader),
      createComputePipelineAsync,
      createBuffer: vi.fn(),
      createTexture: vi.fn(),
    };
    const denoiser = new Bmfr();
    const initializing = denoiser.initialize(context(device));
    await vi.waitFor(() => {
      expect(createComputePipelineAsync).toHaveBeenCalledTimes(2);
    });

    denoiser.dispose();
    for (const resolve of resolvers) resolve(pipeline());
    await expect(initializing).rejects.toThrow(/superseded/);
    expect(device.createBuffer).not.toHaveBeenCalled();
    expect(device.createTexture).not.toHaveBeenCalled();
  });

  it('is idempotent when disposed repeatedly', () => {
    const historyA = { destroy: vi.fn(() => { throw new Error('history destroy'); }) };
    const historyB = { destroy: vi.fn(() => { throw new Error('history destroy'); }) };
    const blockFits = { destroy: vi.fn() };
    const ubo = { destroy: vi.fn() };
    const denoiser = new Bmfr();
    Object.assign(denoiser as unknown as Record<string, unknown>, {
      _device: {},
      _ubo: ubo,
      _sized: {
        historyA,
        historyB,
        blockFits,
        width: 16,
        height: 16,
      },
    });

    expect(() => denoiser.dispose()).not.toThrow();
    denoiser.dispose();
    expect(historyA.destroy).toHaveBeenCalledOnce();
    expect(historyB.destroy).toHaveBeenCalledOnce();
    expect(blockFits.destroy).toHaveBeenCalledOnce();
    expect(ubo.destroy).toHaveBeenCalledOnce();
  });
});
