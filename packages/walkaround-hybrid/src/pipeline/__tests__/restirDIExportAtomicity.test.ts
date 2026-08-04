import { describe, expect, it, vi } from 'vitest';

import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { RESERVOIR_DI_STRIDE_U32 } from '../../restir/reservoirDiLayout.js';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';

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

describe('WalkaroundGPUPipeline ReSTIR-DI export generation', () => {
  it('retains dimensions captured before deferred reservoir maps', async () => {
    const width = 8;
    const height = 6;
    const byteSize = width * height * RESERVOIR_DI_STRIDE_U32 * 4;
    const gates: Array<ReturnType<typeof deferred<void>>> = [];
    const sources = Array.from(
      { length: 3 },
      () => ({ size: byteSize } as GPUBuffer),
    );
    const copyBufferToBuffer = vi.fn();
    const device = {
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        const gate = deferred<void>();
        gates.push(gate);
        const bytes = new ArrayBuffer(Number(descriptor.size));
        const staging = {
          size: Number(descriptor.size),
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
        return staging;
      }),
      createCommandEncoder: vi.fn(() => ({
        copyBufferToBuffer,
        finish: vi.fn(() => ({} as GPUCommandBuffer)),
      })),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;
    const pipeline = new WalkaroundGPUPipeline(device, width, height);
    const internal = pipeline as unknown as {
      _initialized: boolean;
      _width: number;
      _height: number;
      _reservoirScale: number;
      _res: unknown;
    };
    internal._initialized = true;
    internal._res = {
      restirDI: {
        reservoirCurrentBuffer: sources[0],
        reservoirPreviousBuffer: sources[1],
        reservoirSpatialBuffer: sources[2],
      },
    };

    const pending = pipeline.exportRestirDIReservoirs(device);
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(3);
    expect(gates).toHaveLength(3);

    // Model the synchronous size/scale publication that follows a successful
    // setSize transaction while the already-submitted maps remain pending.
    internal._width = 20;
    internal._height = 10;
    internal._reservoirScale = 2;
    gates.forEach((gate) => gate.resolve());

    const snapshot = await pending;
    expect(snapshot).toMatchObject({
      width,
      height,
      strideU32: RESERVOIR_DI_STRIDE_U32,
    });
    expect(snapshot?.current).toHaveLength(
      width * height * RESERVOIR_DI_STRIDE_U32,
    );

    internal._initialized = false;
    internal._res = null;
    pipeline.dispose();
  });
});
