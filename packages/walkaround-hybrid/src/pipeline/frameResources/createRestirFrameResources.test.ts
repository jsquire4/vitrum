import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { RESERVOIR_GI_STRIDE_BYTES } from '../../gi/giLayout.js';
import { RESERVOIR_DI_STRIDE_BYTES } from '../../restir/reservoirDiLayout.js';
import { createRestirDIFrameResources } from './createRestirDIFrameResources.js';
import { createRestirGIFrameResources } from './createRestirGIFrameResources.js';

installWebGPUPolyfills();

function sizingDevice() {
  const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => ({
    size: Number(descriptor.size),
    destroy: vi.fn(),
  }));
  return {
    device: { createBuffer } as unknown as GPUDevice,
    createBuffer,
  };
}

describe('scaled ReSTIR frame-resource allocators', () => {
  it('allocates all DI cohorts at floor(internal/scale)', () => {
    const { device, createBuffer } = sizingDevice();
    createRestirDIFrameResources(device, 1919, 1079, 3);

    const expectedBytes = 639 * 359 * RESERVOIR_DI_STRIDE_BYTES;
    expect(createBuffer).toHaveBeenCalledTimes(3);
    expect(
      createBuffer.mock.calls.map(([descriptor]) => Number(descriptor.size)),
    ).toEqual([expectedBytes, expectedBytes, expectedBytes]);
  });

  it('allocates all GI cohorts at floor(internal/(2*scale))', () => {
    const { device, createBuffer } = sizingDevice();
    createRestirGIFrameResources(device, 1919, 1079, 3);

    const expectedBytes = 319 * 179 * RESERVOIR_GI_STRIDE_BYTES;
    expect(createBuffer).toHaveBeenCalledTimes(3);
    expect(
      createBuffer.mock.calls.map(([descriptor]) => Number(descriptor.size)),
    ).toEqual([expectedBytes, expectedBytes, expectedBytes]);
  });

  it('keeps the WebGPU 256-byte minimum at tiny scaled grids', () => {
    const di = sizingDevice();
    const gi = sizingDevice();
    createRestirDIFrameResources(di.device, 1, 1, 4);
    createRestirGIFrameResources(gi.device, 1, 1, 4);

    expect(
      di.createBuffer.mock.calls.map(([descriptor]) => Number(descriptor.size)),
    ).toEqual([256, 256, 256]);
    expect(
      gi.createBuffer.mock.calls.map(([descriptor]) => Number(descriptor.size)),
    ).toEqual([256, 256, 256]);
  });

  it.each([0, 1.5, 5, Number.NaN])(
    'rejects invalid scale %s before allocating',
    (reservoirScale) => {
      const di = sizingDevice();
      const gi = sizingDevice();
      expect(() =>
        createRestirDIFrameResources(di.device, 64, 64, reservoirScale),
      ).toThrow(/reservoirScale must be an integer in \[1, 4\]/);
      expect(() =>
        createRestirGIFrameResources(gi.device, 64, 64, reservoirScale),
      ).toThrow(/reservoirScale must be an integer in \[1, 4\]/);
      expect(di.createBuffer).not.toHaveBeenCalled();
      expect(gi.createBuffer).not.toHaveBeenCalled();
    },
  );
});
