import { describe, expect, it } from 'vitest';
import { canDecodeRawDdsPixels, decodeRawDdsPixels } from './ddsCodec.js';
import { decodeSceneTextures } from './texturePipeline.js';
import type { RawImageHandle } from './textures.js';
import type { MeshPrimitive, Scene, TextureRef } from '@vitrum/core';

const DDS_MAGIC = 0x2053_4444;

function fourCc(value: string): number {
  return (
    (value.charCodeAt(0) |
      (value.charCodeAt(1) << 8) |
      (value.charCodeAt(2) << 16) |
      (value.charCodeAt(3) << 24)) >>>
    0
  );
}

function compressedDds(
  width: number,
  height: number,
  codec: 'DXT1' | 'DXT3' | 'DXT5' | 'ATI2',
  block: Uint8Array,
): RawImageHandle {
  const bytes = new Uint8Array(128 + block.byteLength);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, DDS_MAGIC, true);
  view.setUint32(4, 124, true);
  view.setUint32(8, 0x0008_1007, true);
  view.setUint32(12, height, true);
  view.setUint32(16, width, true);
  view.setUint32(20, block.byteLength, true);
  view.setUint32(76, 32, true);
  view.setUint32(80, 0x4, true);
  view.setUint32(84, fourCc(codec), true);
  view.setUint32(108, 0x1000, true);
  bytes.set(block, 128);
  return {
    kind: 'raw-image',
    mimeType: 'image/vnd-ms.dds',
    data: bytes,
  };
}

function bc5Dds(red: number, green: number): RawImageHandle {
  const block = new Uint8Array(16);
  block[0] = red;
  block[1] = red;
  block[8] = green;
  block[9] = green;
  return compressedDds(4, 4, 'ATI2', block);
}

function dx10Bc1Dds(dxgiFormat: 71 | 72, alphaMode = 0): RawImageHandle {
  const block = new Uint8Array(8);
  const blockView = new DataView(block.buffer);
  // Mid-grey endpoint in RGB565; all indices select endpoint zero.
  blockView.setUint16(0, 0x8410, true);
  blockView.setUint16(2, 0x8410, true);

  const bytes = new Uint8Array(128 + 20 + block.byteLength);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, DDS_MAGIC, true);
  view.setUint32(4, 124, true);
  view.setUint32(8, 0x0008_1007, true);
  view.setUint32(12, 4, true);
  view.setUint32(16, 4, true);
  view.setUint32(20, block.byteLength, true);
  view.setUint32(76, 32, true);
  view.setUint32(80, 0x4, true);
  view.setUint32(84, fourCc('DX10'), true);
  view.setUint32(108, 0x1000, true);
  view.setUint32(128, dxgiFormat, true);
  view.setUint32(132, 3, true);
  view.setUint32(140, 1, true);
  view.setUint32(144, alphaMode, true);
  bytes.set(block, 148);
  return {
    kind: 'raw-image',
    mimeType: 'image/vnd-ms.dds',
    data: bytes,
  };
}

function bgra8Dds(): RawImageHandle {
  const pixels = new Uint8Array([0, 0, 255, 255, 0, 255, 0, 128]);
  const bytes = new Uint8Array(128 + pixels.byteLength);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, DDS_MAGIC, true);
  view.setUint32(4, 124, true);
  view.setUint32(8, 0x0000_100f, true);
  view.setUint32(12, 1, true);
  view.setUint32(16, 2, true);
  view.setUint32(20, 8, true);
  view.setUint32(76, 32, true);
  view.setUint32(80, 0x41, true);
  view.setUint32(88, 32, true);
  view.setUint32(92, 0x00ff_0000, true);
  view.setUint32(96, 0x0000_ff00, true);
  view.setUint32(100, 0x0000_00ff, true);
  view.setUint32(104, 0xff00_0000, true);
  view.setUint32(108, 0x1000, true);
  bytes.set(pixels, 128);
  return {
    kind: 'raw-image',
    mimeType: 'application/octet-stream',
    data: bytes,
  };
}

function bgra8Dds2x2(headerFlags = 0x0000_100f, pitch = 8): RawImageHandle {
  const pixels = new Uint8Array([0, 0, 255, 255, 0, 255, 0, 128, 255, 0, 0, 64, 255, 255, 255, 0]);
  const bytes = new Uint8Array(128 + pixels.byteLength);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, DDS_MAGIC, true);
  view.setUint32(4, 124, true);
  view.setUint32(8, headerFlags, true);
  view.setUint32(12, 2, true);
  view.setUint32(16, 2, true);
  view.setUint32(20, pitch, true);
  view.setUint32(76, 32, true);
  view.setUint32(80, 0x41, true);
  view.setUint32(88, 32, true);
  view.setUint32(92, 0x00ff_0000, true);
  view.setUint32(96, 0x0000_ff00, true);
  view.setUint32(100, 0x0000_00ff, true);
  view.setUint32(104, 0xff00_0000, true);
  view.setUint32(108, 0x1000, true);
  bytes.set(pixels, 128);
  return {
    kind: 'raw-image',
    mimeType: 'image/vnd-ms.dds',
    data: bytes,
  };
}

function malformedMaskedDds(
  bitsPerPixel: 8 | 16 | 24 | 32,
  pixelFlags: number,
  masks: {
    readonly r: number;
    readonly g: number;
    readonly b: number;
    readonly a: number;
  },
): RawImageHandle {
  const bytesPerPixel = bitsPerPixel / 8;
  const bytes = new Uint8Array(128 + bytesPerPixel);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, DDS_MAGIC, true);
  view.setUint32(4, 124, true);
  view.setUint32(8, 0x0000_100f, true);
  view.setUint32(12, 1, true);
  view.setUint32(16, 1, true);
  view.setUint32(20, bytesPerPixel, true);
  view.setUint32(76, 32, true);
  view.setUint32(80, pixelFlags, true);
  view.setUint32(88, bitsPerPixel, true);
  view.setUint32(92, masks.r, true);
  view.setUint32(96, masks.g, true);
  view.setUint32(100, masks.b, true);
  view.setUint32(104, masks.a, true);
  view.setUint32(108, 0x1000, true);
  bytes.fill(0xff, 128);
  return {
    kind: 'raw-image',
    mimeType: 'image/vnd-ms.dds',
    data: bytes,
  };
}

const CONTEXT = {
  materialField: 'baseColorMap',
  path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
  colorSpace: 'srgb',
  primitiveId: 'mesh-0',
  primitiveIndex: 0,
  maxDecodedTexturePixels: 1024,
} as const;

describe('built-in DDS decoder', () => {
  it('recognizes DDS by MIME type or magic', () => {
    expect(canDecodeRawDdsPixels(bgra8Dds())).toBe(true);
    expect(
      canDecodeRawDdsPixels({
        kind: 'raw-image',
        mimeType: 'image/vnd-ms.dds',
        data: new Uint8Array([0]),
      }),
    ).toBe(true);
  });

  it('decodes BC1/DXT1 color indices to RGBA8', async () => {
    const block = new Uint8Array(8);
    const view = new DataView(block.buffer);
    view.setUint16(0, 0xf800, true);
    view.setUint16(2, 0x07e0, true);
    // First row selects palette entries 0,1,2,3. Remaining texels select 0.
    view.setUint32(4, 0b11_10_01_00, true);
    const result = await decodeRawDdsPixels(compressedDds(4, 4, 'DXT1', block), CONTEXT);
    expect(result).toMatchObject({
      width: 4,
      height: 4,
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    });
    expect(Array.from(result.data).slice(0, 16)).toEqual([
      255, 0, 0, 255, 0, 255, 0, 255, 170, 85, 0, 255, 85, 170, 0, 255,
    ]);
  });

  it('decodes BC3/DXT5 alpha independently from its color block', async () => {
    const block = new Uint8Array(16);
    block[0] = 255;
    block[1] = 0;
    // Alpha indices stay zero: every texel receives alpha endpoint 255.
    const view = new DataView(block.buffer);
    view.setUint16(8, 0x001f, true);
    view.setUint16(10, 0x001f, true);
    const result = await decodeRawDdsPixels(compressedDds(2, 2, 'DXT5', block), {
      ...CONTEXT,
      colorSpace: 'linear',
    });
    expect(Array.from(result.data)).toEqual([
      0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    ]);
  });

  it.each([
    ['flat', 128, 128, 255],
    ['tilted', 191, 128, 238],
    ['overlength', 255, 255, 128],
  ] as const)(
    'reconstructs the positive Z channel for %s BC5 normal vectors',
    async (_name, red, green, blue) => {
      const result = await decodeRawDdsPixels(bc5Dds(red, green), {
        ...CONTEXT,
        materialField: 'normalMap',
        colorSpace: 'linear',
      });

      expect(Array.from(result.data).slice(0, 4)).toEqual([red, green, blue, 255]);
    },
  );

  it('preserves BC5 as two generic channels outside normal-map fields', async () => {
    const result = await decodeRawDdsPixels(bc5Dds(128, 128), {
      ...CONTEXT,
      materialField: 'roughnessMap',
      colorSpace: 'linear',
    });

    expect(Array.from(result.data).slice(0, 4)).toEqual([128, 128, 0, 255]);
  });

  it('keeps normal and generic semantics separate through the texture pipeline', async () => {
    const sharedHandle = bc5Dds(128, 128);
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'bc5-shared-texture',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            normalMap: { handle: sharedHandle },
            roughnessMap: { handle: sharedHandle },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, {
      target: 'cpu-linear',
    });

    expect(result.diagnostics).toEqual([]);
    expect(result.decodedCount).toBe(2);
    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    const normal = (material.normalMap as TextureRef).handle as {
      readonly data: Float32Array;
    };
    const roughness = (material.roughnessMap as TextureRef).handle as {
      readonly data: Float32Array;
    };
    expect(normal).not.toBe(roughness);
    expect(Array.from(normal.data.slice(0, 4))).toEqual([
      expect.closeTo(128 / 255, 6),
      expect.closeTo(128 / 255, 6),
      1,
      1,
    ]);
    expect(Array.from(roughness.data.slice(0, 4))).toEqual([
      expect.closeTo(128 / 255, 6),
      expect.closeTo(128 / 255, 6),
      0,
      1,
    ]);
  });

  it('keys a host decoder by the complete same-color-space material field', async () => {
    const sharedHandle = bgra8Dds();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'field-sensitive-host-decoder',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            roughnessMap: { handle: sharedHandle },
            metallicMap: { handle: sharedHandle },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const seenFields: string[] = [];

    const result = await decodeSceneTextures(scene, {
      target: 'cpu-linear',
      decodePixels: (_handle, context) => {
        seenFields.push(context.materialField);
        const value = context.materialField === 'roughnessMap' ? 64 : 192;
        return {
          width: 1,
          height: 1,
          data: new Uint8Array([value, value, value, 255]),
          channels: 4,
          dataType: 'uint8',
          colorSpace: 'linear',
        };
      },
    });

    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    const roughness = (material.roughnessMap as TextureRef).handle as {
      readonly data: Float32Array;
    };
    const metallic = (material.metallicMap as TextureRef).handle as {
      readonly data: Float32Array;
    };
    expect(seenFields).toEqual(['roughnessMap', 'metallicMap']);
    expect(roughness).not.toBe(metallic);
    expect(roughness.data[0]).toBeCloseTo(64 / 255, 6);
    expect(metallic.data[0]).toBeCloseTo(192 / 255, 6);
  });

  it('preserves DX10 source transfer metadata through pipeline normalization', async () => {
    const unorm = dx10Bc1Dds(71);
    const srgb = dx10Bc1Dds(72);
    const directUnorm = await decodeRawDdsPixels(unorm, CONTEXT);
    const directSrgb = await decodeRawDdsPixels(srgb, {
      ...CONTEXT,
      colorSpace: 'linear',
    });
    expect(directUnorm.colorSpace).toBe('linear');
    expect(directSrgb.colorSpace).toBe('srgb');

    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'dx10-unorm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            baseColorMap: { handle: unorm },
          },
        },
        {
          kind: 'mesh',
          id: 'dx10-srgb',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            baseColorMap: { handle: srgb },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, { target: 'cpu-linear' });
    const linearUnorm = (
      ((result.scene.primitives[0] as MeshPrimitive).material.baseColorMap as TextureRef)
        .handle as { readonly data: Float32Array }
    ).data[0]!;
    const linearSrgb = (
      ((result.scene.primitives[1] as MeshPrimitive).material.baseColorMap as TextureRef)
        .handle as { readonly data: Float32Array }
    ).data[0]!;
    expect(linearUnorm).toBeCloseTo(132 / 255, 6);
    expect(linearSrgb).toBeCloseTo(0.23074005, 6);
  });

  it('rejects DX10 premultiplied-alpha metadata instead of returning dark straight-alpha pixels', () => {
    expect(() => decodeRawDdsPixels(dx10Bc1Dds(72, 2), CONTEXT)).toThrow(
      /premultiplied alpha.*straight-alpha/i,
    );
  });

  it('rejects DX10 custom-alpha metadata whose application-defined semantics cannot be verified', () => {
    expect(() => decodeRawDdsPixels(dx10Bc1Dds(72, 4), CONTEXT)).toThrow(
      /application-defined custom alpha.*verifiable straight-alpha/i,
    );
  });

  it.each([
    ['straight', 1],
    ['opaque', 3],
  ] as const)('accepts DX10 %s-alpha metadata', async (_label, alphaMode) => {
    const result = await decodeRawDdsPixels(dx10Bc1Dds(72, alphaMode), CONTEXT);
    expect(result).toMatchObject({
      width: 4,
      height: 4,
      colorSpace: 'srgb',
    });
  });

  it('canonicalizes DX10 opaque-alpha pixels even when the BC1 block stores transparency', async () => {
    const handle = dx10Bc1Dds(72, 3);
    const bytes = new Uint8Array(handle.data);
    const view = new DataView(bytes.buffer);
    // c0 <= c1 and palette index 3 physically encode transparent BC1 texels.
    view.setUint16(148, 0, true);
    view.setUint16(150, 0xffff, true);
    view.setUint32(152, 0xffff_ffff, true);

    const result = await decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT);
    expect(Array.from(result.data).filter((_value, index) => index % 4 === 3)).toEqual(
      new Array(16).fill(255),
    );
  });

  it.each([
    ['lower-field reserved mode', 5],
    ['upper reserved bit', 0x8],
  ] as const)('rejects a DX10 %s', (_label, alphaMode) => {
    expect(() => decodeRawDdsPixels(dx10Bc1Dds(72, alphaMode), CONTEXT)).toThrow(
      /invalid reserved alpha-mode metadata/i,
    );
  });

  it('decodes legacy masked BGRA8 rows', async () => {
    const result = await decodeRawDdsPixels(bgra8Dds(), CONTEXT);
    expect(Array.from(result.data)).toEqual([255, 0, 0, 255, 0, 255, 0, 128]);
  });

  it('ignores a stale pitch field when DDSD_PITCH is absent', async () => {
    const result = await decodeRawDdsPixels(bgra8Dds2x2(0x0000_1007, 16), CONTEXT);
    expect(Array.from(result.data)).toEqual([
      255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 64, 255, 255, 255, 0,
    ]);
  });

  it('decodes rows using a declared padded base pitch', async () => {
    const tight = bgra8Dds2x2();
    const source = new Uint8Array(tight.data);
    const bytes = new Uint8Array(128 + 24);
    bytes.set(source.subarray(0, 128));
    const view = new DataView(bytes.buffer);
    view.setUint32(20, 12, true);
    bytes.set(source.subarray(128, 136), 128);
    bytes.set(source.subarray(136, 144), 140);
    const result = await decodeRawDdsPixels({ ...tight, data: bytes }, CONTEXT);
    expect(Array.from(result.data)).toEqual([
      255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 64, 255, 255, 255, 0,
    ]);
  });

  it('rejects a declared row pitch smaller than a packed row', () => {
    expect(() => decodeRawDdsPixels(bgra8Dds2x2(0x0000_100f, 4), CONTEXT)).toThrow(
      /row pitch 4.*8-byte packed row/i,
    );
  });

  it('rejects contradictory active pitch and linear-size flags', () => {
    expect(() => decodeRawDdsPixels(bgra8Dds2x2(0x0008_100f, 8), CONTEXT)).toThrow(
      /both pitch and linear-size semantics/i,
    );
  });

  it('ignores stale depth when DDSD_DEPTH and volume caps are absent', async () => {
    const handle = bgra8Dds();
    const bytes = new Uint8Array(handle.data);
    new DataView(bytes.buffer).setUint32(24, 7, true);
    const result = await decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT);
    expect(result).toMatchObject({ width: 2, height: 1 });
  });

  it('rejects a DDSD_DEPTH declaration even when the stale depth value is one', () => {
    const handle = bgra8Dds();
    const bytes = new Uint8Array(handle.data);
    const view = new DataView(bytes.buffer);
    view.setUint32(8, view.getUint32(8, true) | 0x0080_0000, true);
    view.setUint32(24, 1, true);
    expect(() => decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT)).toThrow(
      /cube\/volume texture/i,
    );
  });

  it('rejects a declared mip chain whose lower levels are missing', () => {
    const handle = compressedDds(4, 4, 'DXT1', new Uint8Array(8));
    const bytes = new Uint8Array(handle.data);
    const view = new DataView(bytes.buffer);
    view.setUint32(8, view.getUint32(8, true) | 0x0002_0000, true);
    view.setUint32(28, 2, true);
    expect(() => decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT)).toThrow(
      /declared mip payload is truncated/i,
    );
  });

  it('accepts a complete declared BC1 mip chain', async () => {
    const base = compressedDds(4, 4, 'DXT1', new Uint8Array(8));
    const bytes = new Uint8Array(base.data.byteLength + 8);
    bytes.set(new Uint8Array(base.data));
    const view = new DataView(bytes.buffer);
    view.setUint32(8, view.getUint32(8, true) | 0x0002_0000, true);
    view.setUint32(28, 2, true);
    const result = await decodeRawDdsPixels({ ...base, data: bytes }, CONTEXT);
    expect(result).toMatchObject({ width: 4, height: 4 });
  });

  it('recomputes BC mip offsets instead of trusting an oversized advisory linear size', async () => {
    const base = compressedDds(4, 4, 'DXT1', new Uint8Array(8));
    const bytes = new Uint8Array(base.data.byteLength + 8);
    bytes.set(new Uint8Array(base.data));
    const view = new DataView(bytes.buffer);
    view.setUint32(8, view.getUint32(8, true) | 0x0002_0000, true);
    view.setUint32(20, 64, true);
    view.setUint32(28, 2, true);
    const result = await decodeRawDdsPixels({ ...base, data: bytes }, CONTEXT);
    expect(result).toMatchObject({ width: 4, height: 4 });
  });

  it('treats an undeclared stale mipMapCount as inactive', async () => {
    const handle = compressedDds(4, 4, 'DXT1', new Uint8Array(8));
    const bytes = new Uint8Array(handle.data);
    new DataView(bytes.buffer).setUint32(28, 12, true);
    const result = await decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT);
    expect(result).toMatchObject({ width: 4, height: 4 });
  });

  it('does not activate a stale mipMapCount from legacy caps alone', async () => {
    const handle = compressedDds(4, 4, 'DXT1', new Uint8Array(8));
    const bytes = new Uint8Array(handle.data);
    const view = new DataView(bytes.buffer);
    view.setUint32(28, 2, true);
    view.setUint32(108, view.getUint32(108, true) | 0x0040_0000, true);
    const result = await decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT);
    expect(result).toMatchObject({ width: 4, height: 4 });
  });

  it('rejects a compressed linear size smaller than the base level', () => {
    const handle = compressedDds(4, 4, 'DXT1', new Uint8Array(8));
    const bytes = new Uint8Array(handle.data);
    new DataView(bytes.buffer).setUint32(20, 4, true);
    expect(() => decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT)).toThrow(
      /linear size 4.*8-byte compressed base level/i,
    );
  });

  it('rejects unsupported legacy pixel-format flag combinations', () => {
    const handle = bgra8Dds();
    const bytes = new Uint8Array(handle.data);
    const view = new DataView(bytes.buffer);
    view.setUint32(80, view.getUint32(80, true) | 0x200, true);
    expect(() => decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT)).toThrow(
      /unsupported legacy DDS pixel flags/i,
    );
  });

  it('requires active height and width header fields', () => {
    const handle = bgra8Dds();
    const bytes = new Uint8Array(handle.data);
    const view = new DataView(bytes.buffer);
    view.setUint32(8, view.getUint32(8, true) & ~0x4, true);
    expect(() => decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT)).toThrow(
      /declare valid height and width fields/i,
    );
  });

  it('rejects nonzero DX10 misc flags for the plain 2D decoder subset', () => {
    const handle = dx10Bc1Dds(72);
    const bytes = new Uint8Array(handle.data);
    new DataView(bytes.buffer).setUint32(136, 1, true);
    expect(() => decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT)).toThrow(
      /plain single 2D texture/i,
    );
  });

  it('rejects an individual legacy cube-face bit even when the base cubemap bit is absent', () => {
    const handle = bgra8Dds();
    const bytes = new Uint8Array(handle.data);
    new DataView(bytes.buffer).setUint32(112, 0x0000_0400, true);
    expect(() => decodeRawDdsPixels({ ...handle, data: bytes }, CONTEXT)).toThrow(
      /cube\/volume texture/i,
    );
  });

  it('enforces the pixel budget before allocating output', () => {
    expect(() =>
      decodeRawDdsPixels(bgra8Dds(), {
        ...CONTEXT,
        maxDecodedTexturePixels: 1,
      }),
    ).toThrow('exceed maxDecodedTexturePixels 1');
  });

  it('rejects a masked channel that extends beyond the declared pixel width', () => {
    expect(() =>
      decodeRawDdsPixels(
        malformedMaskedDds(8, 0x40, {
          r: 0x0000_0100,
          g: 0x0000_0006,
          b: 0x0000_0018,
          a: 0,
        }),
        CONTEXT,
      ),
    ).toThrow(/mask.*8-bit pixel/i);
  });

  it('rejects overlapping masked channels instead of duplicating their bits', () => {
    expect(() =>
      decodeRawDdsPixels(
        malformedMaskedDds(32, 0x41, {
          r: 0x0000_00ff,
          g: 0x0000_00ff,
          b: 0x00ff_0000,
          a: 0xff00_0000,
        }),
        CONTEXT,
      ),
    ).toThrow(/overlap/i);
  });

  it('rejects alpha-only pixels whose required alpha mask is absent', () => {
    expect(() =>
      decodeRawDdsPixels(
        malformedMaskedDds(8, 0x2, {
          r: 0,
          g: 0,
          b: 0,
          a: 0,
        }),
        { ...CONTEXT, colorSpace: 'linear' },
      ),
    ).toThrow(/alpha mask/i);
  });

  it('rejects RGB pixels that declare alpha data without an alpha mask', () => {
    expect(() =>
      decodeRawDdsPixels(
        malformedMaskedDds(32, 0x41, {
          r: 0x0000_00ff,
          g: 0x0000_ff00,
          b: 0x00ff_0000,
          a: 0,
        }),
        CONTEXT,
      ),
    ).toThrow(/declare alpha data.*no alpha mask/i);
  });

  it('rejects an alpha mask whose declaring flag is absent', () => {
    expect(() =>
      decodeRawDdsPixels(
        malformedMaskedDds(32, 0x40, {
          r: 0x0000_00ff,
          g: 0x0000_ff00,
          b: 0x00ff_0000,
          a: 0xff00_0000,
        }),
        CONTEXT,
      ),
    ).toThrow(/alpha mask without declaring alpha data/i);
  });

  it.each([
    ['RGB plus luminance', 0x0002_0040],
    ['RGB plus alpha-only', 0x42],
    ['alpha-only plus alpha-pixels', 0x3],
  ] as const)('rejects contradictory %s pixel flags', (_label, pixelFlags) => {
    expect(() =>
      decodeRawDdsPixels(
        malformedMaskedDds(32, pixelFlags, {
          r: 0x0000_00ff,
          g: 0x0000_ff00,
          b: 0x00ff_0000,
          a: 0xff00_0000,
        }),
        CONTEXT,
      ),
    ).toThrow(/contradictory or unsupported/i);
  });
});
