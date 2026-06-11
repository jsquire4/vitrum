import { describe, expect, it } from 'vitest';

import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { createSvgfFrameResources } from '../frameResources/createSvgfFrameResources.js';

installWebGPUPolyfills();

type TextureRecord = {
  readonly texture: GPUTexture & { readonly label: string };
  readonly desc: GPUTextureDescriptor;
};

function makeDevice() {
  const textures: TextureRecord[] = [];
  const writes: Array<{
    readonly texture: GPUTexture & { readonly label: string };
    readonly data: BufferSource | SharedArrayBuffer;
    readonly layout: GPUImageDataLayout;
    readonly size: GPUExtent3D;
  }> = [];
  const device = {
    createTexture(desc: GPUTextureDescriptor): GPUTexture {
      const size = desc.size as readonly number[];
      const texture = {
        label: desc.label ?? '',
        width: size[0] ?? 1,
        height: size[1] ?? 1,
        createView: () => ({}),
        destroy() {},
      } as unknown as GPUTexture & { readonly label: string };
      textures.push({ texture, desc });
      return texture;
    },
    queue: {
      writeTexture(
        destination: GPUImageCopyTexture,
        data: BufferSource | SharedArrayBuffer,
        layout: GPUImageDataLayout,
        size: GPUExtent3D,
      ): void {
        writes.push({
          texture: destination.texture,
          data,
          layout,
          size,
        });
      },
    },
  } as unknown as GPUDevice;
  return { device, textures, writes };
}

function u32At(data: BufferSource | SharedArrayBuffer, byteOffset: number): number {
  if (ArrayBuffer.isView(data)) {
    return new DataView(data.buffer).getUint32(data.byteOffset + byteOffset, true);
  }
  return new DataView(data).getUint32(byteOffset, true);
}

describe('SVGF object-id frame resources', () => {
  it('allocates real current/previous object-id textures at full resolution', () => {
    const { device, textures, writes } = makeDevice();

    const resources = createSvgfFrameResources(device, 4, 2, false);

    expect(resources.svgfObjIdPlaceholderTexture.width).toBe(1);
    expect(resources.svgfPrevObjIdPlaceholderTexture.width).toBe(1);
    expect(resources.svgfCurrentObjectIdTexture.width).toBe(4);
    expect(resources.svgfCurrentObjectIdTexture.height).toBe(2);
    expect(resources.svgfPreviousObjectIdTexture.width).toBe(4);
    expect(resources.svgfPreviousObjectIdTexture.height).toBe(2);

    const current = textures.find((r) => r.texture === resources.svgfCurrentObjectIdTexture)!;
    const previous = textures.find((r) => r.texture === resources.svgfPreviousObjectIdTexture)!;
    expect(current.desc.format).toBe('r32uint');
    expect(current.desc.usage & GPUTextureUsage.STORAGE_BINDING).not.toBe(0);
    expect(current.desc.usage & GPUTextureUsage.TEXTURE_BINDING).not.toBe(0);
    expect(current.desc.usage & GPUTextureUsage.COPY_SRC).not.toBe(0);
    expect(previous.desc.format).toBe('r32uint');
    expect(previous.desc.usage & GPUTextureUsage.TEXTURE_BINDING).not.toBe(0);
    expect(previous.desc.usage & GPUTextureUsage.COPY_DST).not.toBe(0);

    const currentWrite = writes.find((w) => w.texture === resources.svgfCurrentObjectIdTexture)!;
    const previousWrite = writes.find((w) => w.texture === resources.svgfPreviousObjectIdTexture)!;
    expect(currentWrite.layout.bytesPerRow).toBe(256);
    expect(previousWrite.layout.bytesPerRow).toBe(256);
    expect(u32At(currentWrite.data, 0)).toBe(0);
    expect(u32At(previousWrite.data, 0)).toBe(1);
    expect(u32At(previousWrite.data, 256 + 4)).toBe(1);
  });
});
