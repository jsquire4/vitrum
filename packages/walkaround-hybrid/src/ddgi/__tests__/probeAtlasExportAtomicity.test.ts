import { describe, expect, it, vi } from 'vitest';
import { SceneBvh } from '@vitrum/shared-bvh';

import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { ProbeGrid } from '../probeGrid.js';
import { ProbeUpdatePass } from '../probeUpdatePass.js';

installWebGPUPolyfills();

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('ProbeUpdatePass atlas export generation', () => {
  it('queues irradiance and visibility together and retains pre-map grid metadata', async () => {
    const gates: Array<ReturnType<typeof deferred<void>>> = [];
    const stagingBuffers: Array<{
      mapState: GPUBufferMapState;
      mapAsync: ReturnType<typeof vi.fn<[], Promise<void>>>;
      getMappedRange: ReturnType<typeof vi.fn<[], ArrayBuffer>>;
      unmap: ReturnType<typeof vi.fn<[], void>>;
      destroy: ReturnType<typeof vi.fn<[], void>>;
    }> = [];
    const copyTextureToBuffer = vi.fn();
    const finish = vi.fn(() => ({} as GPUCommandBuffer));
    const encoder = { copyTextureToBuffer, finish };
    const submit = vi.fn();
    const device = {
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        const gate = deferred<void>();
        gates.push(gate);
        const bytes = new ArrayBuffer(Number(descriptor.size));
        const staging = {
          mapState: 'unmapped' as GPUBufferMapState,
          mapAsync: vi.fn(async () => {
            staging.mapState = 'pending';
            await gate.promise;
            staging.mapState = 'mapped';
          }),
          getMappedRange: vi.fn(() => bytes),
          unmap: vi.fn(() => {
            staging.mapState = 'unmapped';
          }),
          destroy: vi.fn(),
        };
        stagingBuffers.push(staging);
        return staging;
      }),
      createCommandEncoder: vi.fn(() => encoder),
      queue: { submit },
    } as unknown as GPUDevice;

    const grid = new ProbeGrid();
    grid.dims = { x: 1, y: 1, z: 1 };
    grid.worldSpacing = 2;
    grid.allocateAtlases();
    const irradiance = {} as GPUTexture;
    const visibility = {} as GPUTexture;
    const pass = new ProbeUpdatePass(new SceneBvh(), grid);
    vi.spyOn(pass, 'getReadAtlasGPUTextures').mockReturnValue({
      irradiance,
      visibility,
    });

    const pending = pass.exportAtlasData(device);

    expect(device.createCommandEncoder).toHaveBeenCalledOnce();
    expect(copyTextureToBuffer).toHaveBeenCalledTimes(2);
    expect(copyTextureToBuffer.mock.calls[0]?.[0]).toEqual({
      texture: irradiance,
    });
    expect(copyTextureToBuffer.mock.calls[1]?.[0]).toEqual({
      texture: visibility,
    });
    expect(finish).toHaveBeenCalledOnce();
    expect(submit).toHaveBeenCalledOnce();
    expect(stagingBuffers).toHaveLength(2);
    expect(stagingBuffers.every((buffer) =>
      buffer.mapAsync.mock.calls.length === 1)).toBe(true);

    // Model a synchronous setScene/grid publication while both maps are pending.
    grid.dims = { x: 2, y: 3, z: 4 };
    grid.worldSpacing = 99;
    grid.allocateAtlases();
    gates.forEach((gate) => gate.resolve());

    const snapshot = await pending;
    expect(snapshot).toMatchObject({
      irrW: 5,
      irrH: 5,
      visW: 18,
      visH: 18,
      probeStateW: 1,
      probeStateH: 1,
    });
    expect(snapshot?.probeStateData).toEqual(new Float32Array(4));
    expect(stagingBuffers.every((buffer) =>
      buffer.destroy.mock.calls.length === 1)).toBe(true);
    pass.dispose();
  });
});
