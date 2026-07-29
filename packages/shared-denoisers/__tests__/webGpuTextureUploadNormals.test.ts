import { describe, expect, it, vi } from 'vitest';
import { uploadUnitNormalsAsRgba32f } from '../src/webGpuTextureUpload.js';

describe('uploadUnitNormalsAsRgba32f', () => {
  it('affine-packs signed normals exactly once before shader-side decoding', () => {
    const writeTexture = vi.fn();
    const device = { queue: { writeTexture } } as unknown as GPUDevice;
    const texture = {} as GPUTexture;

    uploadUnitNormalsAsRgba32f(
      device,
      texture,
      new Float32Array([
        0, 1, 0,
        -1, 0, 1,
      ]),
      2,
      1,
    );

    expect(writeTexture).toHaveBeenCalledOnce();
    const backing = writeTexture.mock.calls[0]![1] as ArrayBuffer;
    const packed = new Float32Array(backing);
    expect(Array.from(packed.slice(0, 8))).toEqual([
      0.5, 1, 0.5, 0,
      0, 0.5, 1, 0,
    ]);
    for (let component = 0; component < 3; component += 1) {
      expect(packed[component]! * 2 - 1).toBeCloseTo(
        new Float32Array([0, 1, 0])[component]!,
      );
    }
  });
});
