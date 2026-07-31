import { describe, expect, it, vi } from 'vitest';
import { float32ToFloat16Bits } from '@vitrum/shared-denoisers';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

import { readDenoiserTrainingInputsWalkaround } from '../gpuReadback.js';

interface MockBuffer {
  readonly mapAsync: (mode?: GPUMapModeFlags) => Promise<void>;
  readonly getMappedRange: () => ArrayBuffer;
  readonly unmap: () => void;
  readonly destroy: () => void;
}

function rgba16fRows(
  width: number,
  height: number,
  pixels: readonly (readonly [number, number, number, number])[],
): ArrayBuffer {
  const bytesPerRow = 256;
  const bytes = new ArrayBuffer(bytesPerRow * height);
  const view = new DataView(bytes);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixel = pixels[y * width + x]!;
      const offset = y * bytesPerRow + x * 8;
      for (let channel = 0; channel < 4; channel += 1) {
        view.setUint16(
          offset + channel * 2,
          float32ToFloat16Bits(pixel[channel]!),
          true,
        );
      }
    }
  }
  return bytes;
}

function makeReadbackHarness(
  mappedRanges: readonly ArrayBuffer[],
  rejectMapAt: number | null = null,
): {
  readonly device: GPUDevice;
  readonly buffers: readonly MockBuffer[];
  readonly copies: ReturnType<typeof vi.fn>;
  readonly submit: ReturnType<typeof vi.fn>;
} {
  const buffers = mappedRanges.map((range, index): MockBuffer => ({
    mapAsync: vi.fn(() =>
      index === rejectMapAt
        ? Promise.reject(new Error(`map ${index} failed`))
        : Promise.resolve(),
    ),
    getMappedRange: vi.fn(() => range),
    unmap: vi.fn(),
    destroy: vi.fn(),
  }));
  let nextBuffer = 0;
  const copies = vi.fn();
  const submit = vi.fn();
  return {
    buffers,
    copies,
    submit,
    device: {
      createBuffer: vi.fn(() => buffers[nextBuffer++] as unknown as GPUBuffer),
      createCommandEncoder: vi.fn(() => ({
        copyTextureToBuffer: copies,
        finish: vi.fn(() => ({ kind: 'finished-command-buffer' })),
      })),
      queue: { submit },
    } as unknown as GPUDevice,
  };
}

describe('readDenoiserTrainingInputsWalkaround', () => {
  it('copies one coherent cohort, strips row padding, and decodes signed normals', async () => {
    const width = 2;
    const height = 2;
    const radiance = rgba16fRows(width, height, [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ]);
    const albedo = rgba16fRows(width, height, [
      [0.25, 0.5, 0.75, 1],
      [1, 0, 0.5, 1],
      [0, 1, 0.25, 1],
      [0.75, 0.25, 1, 1],
    ]);
    const normal = rgba16fRows(width, height, [
      [0, 0.5, 1, 0.1],
      [0.25, 0.75, 0.5, 0.2],
      [1, 0, 0.5, 0.3],
      [0.5, 1, 0, 0.4],
    ]);
    const harness = makeReadbackHarness([radiance, albedo, normal]);
    const textures = {
      radiance: { id: 'radiance' } as unknown as GPUTexture,
      albedo: { id: 'albedo' } as unknown as GPUTexture,
      normalDepth: { id: 'normal-depth' } as unknown as GPUTexture,
    };

    const capture = await readDenoiserTrainingInputsWalkaround(
      harness.device,
      textures,
      width,
      height,
    );

    expect(capture).not.toBeNull();
    expect(capture!.radiance).toEqual(new Float32Array([
      1, 2, 3,
      5, 6, 7,
      9, 10, 11,
      13, 14, 15,
    ]));
    expect(capture!.albedo).toEqual(new Float32Array([
      0.25, 0.5, 0.75,
      1, 0, 0.5,
      0, 1, 0.25,
      0.75, 0.25, 1,
    ]));
    expect(capture!.worldNormal).toEqual(new Float32Array([
      -1, 0, 1,
      -0.5, 0.5, 0,
      1, -1, 0,
      0, 1, -1,
    ]));
    expect(harness.copies).toHaveBeenCalledTimes(3);
    expect(harness.copies.mock.calls.map((call) => call[0].texture)).toEqual([
      textures.radiance,
      textures.albedo,
      textures.normalDepth,
    ]);
    for (const call of harness.copies.mock.calls) {
      expect(call[1].bytesPerRow).toBe(256);
      expect(call[1].rowsPerImage).toBe(height);
      expect(call[2]).toEqual({ width, height, depthOrArrayLayers: 1 });
    }
    expect(harness.submit).toHaveBeenCalledTimes(1);
    for (const buffer of harness.buffers) {
      expect(buffer.mapAsync).toHaveBeenCalledWith(GPUMapMode.READ);
      expect(buffer.unmap).toHaveBeenCalledTimes(1);
      expect(buffer.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('waits for the complete map cohort and destroys every staging buffer on failure', async () => {
    const ranges = [
      rgba16fRows(1, 1, [[1, 1, 1, 1]]),
      rgba16fRows(1, 1, [[1, 1, 1, 1]]),
      rgba16fRows(1, 1, [[1, 1, 1, 1]]),
    ];
    const harness = makeReadbackHarness(ranges, 1);

    await expect(readDenoiserTrainingInputsWalkaround(
      harness.device,
      {
        radiance: {} as GPUTexture,
        albedo: {} as GPUTexture,
        normalDepth: {} as GPUTexture,
      },
      1,
      1,
    )).rejects.toThrow('map 1 failed');

    expect(harness.buffers[0]!.unmap).toHaveBeenCalledTimes(1);
    expect(harness.buffers[1]!.unmap).not.toHaveBeenCalled();
    expect(harness.buffers[2]!.unmap).toHaveBeenCalledTimes(1);
    for (const buffer of harness.buffers) {
      expect(buffer.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('rejects invalid dimensions without allocating GPU resources', async () => {
    const createBuffer = vi.fn();
    const device = { createBuffer } as unknown as GPUDevice;
    await expect(readDenoiserTrainingInputsWalkaround(
      device,
      {
        radiance: {} as GPUTexture,
        albedo: {} as GPUTexture,
        normalDepth: {} as GPUTexture,
      },
      0,
      16,
    )).resolves.toBeNull();
    expect(createBuffer).not.toHaveBeenCalled();
  });
});
