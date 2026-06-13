import { describe, expect, it, vi } from 'vitest';
import {
  refreshAnalyticLightsTexture,
  uploadAnalyticLightsTexture,
} from '../analyticLightsTexture.js';

function makeDevice() {
  const texture = {
    createView: vi.fn(() => ({})),
    destroy: vi.fn(),
  } as unknown as GPUTexture;
  const writes: Array<{
    data: Float32Array;
    layout: GPUImageDataLayout;
    size: GPUExtent3DDict;
  }> = [];
  const descriptors: GPUTextureDescriptor[] = [];
  const device = {
    createTexture: vi.fn((desc: GPUTextureDescriptor) => {
      descriptors.push(desc);
      return texture;
    }),
    queue: {
      writeTexture: vi.fn((_dst, data: ArrayBuffer, layout: GPUImageDataLayout, size: GPUExtent3DDict) => {
        writes.push({
          data: new Float32Array(data).slice(),
          layout,
          size,
        });
      }),
    },
  } as unknown as GPUDevice;
  return { device, texture, descriptors, writes };
}

describe('analyticLightsTexture', () => {
  it('writes an explicit zero-count header and ignores placeholder payload', () => {
    const { device, descriptors, writes } = makeDevice();
    const placeholder = new Float32Array(16).fill(9);

    uploadAnalyticLightsTexture(device, placeholder, 0);

    expect(descriptors[0]!).toMatchObject({
      format: 'rgba32float',
      size: { width: 4, height: 1, depthOrArrayLayers: 1 },
    });
    expect(writes[0]!.layout).toMatchObject({ bytesPerRow: 64, rowsPerImage: 1 });
    expect(writes[0]!.data[0]).toBe(0);
    expect(Array.from(writes[0]!.data.slice(4))).toEqual(new Array(12).fill(0));
  });

  it('stores packed lights after the header texel', () => {
    const { device, writes } = makeDevice();
    const packed = new Float32Array(Array.from({ length: 32 }, (_v, i) => i + 1));

    uploadAnalyticLightsTexture(device, packed, 2);

    expect(writes[0]!.data[0]).toBe(2);
    expect(Array.from(writes[0]!.data.slice(4, 36))).toEqual(Array.from(packed));
  });

  it('refresh validates capacity including the header texel', () => {
    const { device, writes } = makeDevice();
    const packed = new Float32Array(Array.from({ length: 16 }, (_v, i) => i + 1));
    const tex = uploadAnalyticLightsTexture(device, packed, 1);

    refreshAnalyticLightsTexture(device, tex, packed, 1);

    expect(writes).toHaveLength(2);
    expect(writes[1]!.data[0]).toBe(1);
    expect(Array.from(writes[1]!.data.slice(4, 20))).toEqual(Array.from(packed));
  });
});
