import { Buffer } from 'node:buffer';
import { describe, expect, it } from 'vitest';
import { canDecodeRawBasisKtx2Pixels, decodeRawBasisKtx2Pixels } from './basisKtx2Codec.js';
import type { RawImageHandle } from './textures.js';

// Real UASTC block produced by basis_universal 2.5, with KHR_texture_basisu's
// required 4x4 dimensions. Its first source texels are red/opaque,
// green/half-alpha, blue/quarter-alpha, and white/transparent.
const UASTC_4X4_BASE64 =
  'q0tUWCAyMLsNChoKAAAAAAEAAAAEAAAABAAAAAAAAAAAAAAAAQAAAAEAAAAAAAAAaAAAACwAAACUAAAALAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAsAAAAAAAAAAIAKACmAQIAAwMAABAAAAAAAAAAAAB/AwAAAAAAAAAA/////x8AAABLVFh3cml0ZXIAQmFzaXMgVW5pdmVyc2FsIDIuNTAAAAQAAAB/fwAAz/6x7h7gAR7h4f8J/H5+fg==';

// Apache-2.0 Khronos KTX-Software fixture
// tests/resources/ktx2/alpha_simple_blze.ktx2 at commit
// 5c17214576cbc9a610e987ee78604e40dcfb744f (SHA-256
// 59042e0998161c8dd3d5dc66b9f879fe30b7111aaa6214d910960c870d3dcb5b).
// It is a real two-slice 8x8 ETC1S/BasisLZ image whose decoded texels are all
// RGBA [171, 187, 204, 128].
const ETC1S_RGBA_8X8_BASE64 =
  'q0tUWCAyMLsNChoKAAAAAAEAAAAIAAAACAAAAAAAAAAAAAAAAQAAAAEAAAABAAAAaAAAADwAAACkAAAARAAAAOgAAAAAAAAAjAAAAAAAAAB0AQAAAAAAAAMAAAAAAAAAAAAAAAAAAAA8AAAAAAAAAAIAOACjAQIAAwMAAAgIAAAAAAAAAAA/AAAAAAAAAAAA/////0AAPw8AAAAAAAAAAP////9AAAAAS1RYd3JpdGVyAGt0eCBjcmVhdGUgdjUuMC5fX2RlZmF1bHRfXyAvIGxpYmt0eCB2NS4wLl9fZGVmYXVsdF9fAAIAAgAtAAAACQAAAC4AAAAAAAAAAAAAAAAAAAABAAAAAQAAAAIAAAABwAQAAAAAAAACBJgbIAAAAAjDNpE+kQBgAgAAAAAAAIEATAEQAAAAACBZwD2sqqqqUlVVVQUUwEQAAAAAAAASQQCYAAAAAAAAQBgCogQMAAAAg3Z7SQSiIABMAAgAAAAAIAIBBkwO';

function fixture(mimeType = 'image/ktx2'): RawImageHandle {
  return {
    kind: 'raw-image',
    mimeType,
    data: new Uint8Array(Buffer.from(UASTC_4X4_BASE64, 'base64')),
  };
}

function etc1sFixture(): RawImageHandle {
  return {
    kind: 'raw-image',
    mimeType: 'image/ktx2',
    data: new Uint8Array(Buffer.from(ETC1S_RGBA_8X8_BASE64, 'base64')),
  };
}

function mutatedFixture(mutate: (bytes: Uint8Array, view: DataView) => void): RawImageHandle {
  const handle = fixture();
  const data = new Uint8Array(handle.data);
  mutate(data, new DataView(data.buffer));
  return { ...handle, data };
}

function mutatedEtc1sFixture(mutate: (bytes: Uint8Array, view: DataView) => void): RawImageHandle {
  const handle = etc1sFixture();
  const data = new Uint8Array(handle.data);
  mutate(data, new DataView(data.buffer));
  return { ...handle, data };
}

function shortenedEtc1sSgdFixture(sgdLength: number): RawImageHandle {
  const handle = etc1sFixture();
  const source = new Uint8Array(handle.data);
  const sourceView = new DataView(source.buffer);
  const sgdOffset = Number(sourceView.getBigUint64(64, true));
  const sourceLevelOffset = Number(sourceView.getBigUint64(80, true));
  const levelLength = Number(sourceView.getBigUint64(88, true));
  const levelOffset = sgdOffset + sgdLength;
  const data = new Uint8Array(levelOffset + levelLength);
  data.set(source.subarray(0, Math.min(levelOffset, source.length)));
  data.set(source.subarray(sourceLevelOffset, sourceLevelOffset + levelLength), levelOffset);
  const view = new DataView(data.buffer);
  view.setBigUint64(72, BigInt(sgdLength), true);
  view.setBigUint64(80, BigInt(levelOffset), true);
  return { ...handle, data };
}

function twoLevelUastcFixture(): RawImageHandle {
  const handle = fixture();
  const source = new Uint8Array(handle.data);
  const sourceView = new DataView(source.buffer);
  const sourceDfdOffset = sourceView.getUint32(48, true);
  const sourceDfdLength = sourceView.getUint32(52, true);
  const sourceKvdOffset = sourceView.getUint32(56, true);
  const sourceKvdLength = sourceView.getUint32(60, true);
  const sourceLevelOffset = Number(sourceView.getBigUint64(80, true));
  const levelLength = Number(sourceView.getBigUint64(88, true));
  const dfdOffset = 128;
  const kvdOffset = dfdOffset + sourceDfdLength;
  const smallerLevelOffset = 224;
  const baseLevelOffset = smallerLevelOffset + levelLength;
  const data = new Uint8Array(baseLevelOffset + levelLength);
  data.set(source.subarray(0, 80));
  data.set(source.subarray(sourceDfdOffset, sourceDfdOffset + sourceDfdLength), dfdOffset);
  data.set(source.subarray(sourceKvdOffset, sourceKvdOffset + sourceKvdLength), kvdOffset);
  const encodedBlock = source.subarray(sourceLevelOffset, sourceLevelOffset + levelLength);
  data.set(encodedBlock, smallerLevelOffset);
  data.set(encodedBlock, baseLevelOffset);
  const view = new DataView(data.buffer);
  view.setUint32(40, 2, true);
  view.setUint32(48, dfdOffset, true);
  view.setUint32(56, kvdOffset, true);
  view.setBigUint64(80, BigInt(baseLevelOffset), true);
  view.setBigUint64(88, BigInt(levelLength), true);
  view.setBigUint64(96, BigInt(levelLength), true);
  view.setBigUint64(104, BigInt(smallerLevelOffset), true);
  view.setBigUint64(112, BigInt(levelLength), true);
  view.setBigUint64(120, BigInt(levelLength), true);
  return { ...handle, data };
}

const CONTEXT = {
  materialField: 'baseColorMap',
  path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
  colorSpace: 'srgb',
  primitiveId: 'mesh-0',
  primitiveIndex: 0,
  maxDecodedTexturePixels: 1024,
} as const;

describe('built-in KTX2/Basis Universal decoder', () => {
  it('recognizes KTX2 by signature even when the MIME label is generic', () => {
    expect(canDecodeRawBasisKtx2Pixels(fixture('application/octet-stream'))).toBe(true);
    expect(
      canDecodeRawBasisKtx2Pixels({
        kind: 'raw-image',
        mimeType: 'image/ktx2',
        data: new Uint8Array([0, 1, 2]),
      }),
    ).toBe(true);
  });

  it('transcodes a real UASTC KTX2 level to owned RGBA8 pixels', async () => {
    const decoded = await decodeRawBasisKtx2Pixels(fixture(), CONTEXT);
    expect(decoded).toMatchObject({
      width: 4,
      height: 4,
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    });
    expect(Array.from(decoded.data).slice(0, 16)).toEqual([
      255, 0, 0, 255, 0, 255, 0, 136, 0, 255, 0, 136, 0, 255, 0, 136,
    ]);
  });

  it('transcodes a real two-slice ETC1S/BasisLZ level and preserves alpha', async () => {
    const decoded = await decodeRawBasisKtx2Pixels(etc1sFixture(), CONTEXT);
    expect(decoded).toMatchObject({
      width: 8,
      height: 8,
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    });
    expect(Array.from(decoded.data)).toEqual(new Array(64).fill([171, 187, 204, 128]).flat());
  });

  it('canonicalizes UASTC RGB alpha to opaque instead of exposing latent block alpha', async () => {
    const decoded = await decodeRawBasisKtx2Pixels(
      mutatedFixture((bytes, view) => {
        const descriptorOffset = view.getUint32(48, true) + 4;
        bytes[descriptorOffset + 27] = 0;
      }),
      CONTEXT,
    );
    expect(Array.from(decoded.data).filter((_value, index) => index % 4 === 3)).toEqual(
      new Array(16).fill(255),
    );
  });

  it('maps ETC1S RRR+GGG second-slice alpha into G and canonicalizes B/A', async () => {
    const original = await decodeRawBasisKtx2Pixels(etc1sFixture(), CONTEXT);
    const decoded = await decodeRawBasisKtx2Pixels(
      mutatedEtc1sFixture((bytes, view) => {
        const descriptorOffset = view.getUint32(48, true) + 4;
        bytes[descriptorOffset + 9] = 0;
        bytes[descriptorOffset + 10] = 1;
        bytes[descriptorOffset + 27] = 3;
        bytes[descriptorOffset + 43] = 4;
      }),
      {
        ...CONTEXT,
        materialField: 'normalMap',
        colorSpace: 'linear',
      },
    );
    for (let offset = 0; offset < decoded.data.length; offset += 4) {
      expect(Array.from(decoded.data).slice(offset, offset + 4)).toEqual([
        original.data[offset],
        original.data[offset + 3],
        0,
        255,
      ]);
    }
  });

  it('accepts reverse-physical-order UASTC mip levels with zero alignment padding', async () => {
    const decoded = await decodeRawBasisKtx2Pixels(twoLevelUastcFixture(), CONTEXT);
    expect(decoded).toMatchObject({ width: 4, height: 4 });
  });

  it('rejects inter-level UASTC overlap in a multi-level container', async () => {
    const handle = twoLevelUastcFixture();
    const data = new Uint8Array(handle.data);
    const view = new DataView(data.buffer);
    view.setBigUint64(80, view.getBigUint64(104, true), true);
    await expect(decodeRawBasisKtx2Pixels({ ...handle, data }, CONTEXT)).rejects.toThrow(
      /physical order|overlap/i,
    );
  });

  it('rejects nonzero raw-UASTC mip padding', async () => {
    const handle = twoLevelUastcFixture();
    const data = new Uint8Array(handle.data);
    data[216] = 1;
    await expect(decodeRawBasisKtx2Pixels({ ...handle, data }, CONTEXT)).rejects.toThrow(
      /mip-level padding bytes must be zero/i,
    );
  });

  it('enforces the decoded-pixel ceiling before allocating RGBA output', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(fixture(), {
        ...CONTEXT,
        maxDecodedTexturePixels: 15,
      }),
    ).rejects.toThrow('exceed maxDecodedTexturePixels 15');
  });

  it('rejects a MIME-only false positive as an invalid KTX2 byte stream', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        {
          kind: 'raw-image',
          mimeType: 'image/ktx2',
          data: new Uint8Array([1, 2, 3]),
        },
        CONTEXT,
      ),
    ).rejects.toThrow('not a valid KTX2 byte stream');
  });

  it('rejects a one-layer array instead of decoding it as a 2D material image', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          // KTX2 layerCount. A non-array 2D texture must encode this as zero.
          view.setUint32(32, 1, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/layerCount 0/);
  });

  it('rejects a level count beyond the legal mip pyramid before reading its index', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          view.setUint32(40, 4, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/levelCount 4 exceeds.*3-level pyramid/i);
  });

  it.each([80, 104, 112, 148] as const)(
    'rejects a level payload offset %i that overlaps KTX2 metadata',
    async (byteOffset) => {
      await expect(
        decodeRawBasisKtx2Pixels(
          mutatedFixture((_bytes, view) => {
            view.setBigUint64(80, BigInt(byteOffset), true);
          }),
          CONTEXT,
        ),
      ).rejects.toThrow(/physical order|overlap|padding/i);
    },
  );

  it('rejects unsafe UInt64 level-index values before native parsing', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          view.setBigUint64(80, BigInt(Number.MAX_SAFE_INTEGER) + 1n, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/level 0 byteOffset exceeds JavaScript's safe integer range/i);
  });

  it('rejects an empty encoded level before native parsing', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          view.setBigUint64(88, 0n, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/level 0 has an empty encoded payload/i);
  });

  it('rejects a raw UASTC level whose indexed lengths disagree with its blocks', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          view.setBigUint64(96, 15n, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/UASTC level 0 byte lengths.*16-byte block payload/i);
  });

  it('rejects a misaligned raw UASTC mip payload', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          view.setBigUint64(80, 177n, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/physical order|padding/i);
  });

  it('rejects a DFD transfer function that conflicts with the material role', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          const dfdOffset = view.getUint32(48, true);
          // Basic DFD transferFunction: linear instead of the sRGB required by
          // this base-color context.
          bytes[dfdOffset + 14] = 1;
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/DFD.*sRGB/);
  });

  it('rejects a non-current Khronos basic DFD version', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          const descriptorOffset = view.getUint32(48, true) + 4;
          view.setUint16(descriptorOffset + 4, 1, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/basic version 2/);
  });

  it('rejects malformed trailing DFD blocks', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          const dfdOffset = view.getUint32(48, true);
          const expandedLength = view.getUint32(52, true) + 8;
          view.setUint32(52, expandedLength, true);
          view.setUint32(dfdOffset, expandedLength, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/Basis data-format descriptor is malformed/);
  });

  it.each([
    {
      label: 'texel-block dimensions',
      mutate: (bytes: Uint8Array, _view: DataView, descriptor: number) => {
        bytes[descriptor + 12] = 2;
      },
      error: /4x4x1 texel blocks/i,
    },
    {
      label: 'sample bit offset',
      mutate: (_bytes: Uint8Array, view: DataView, descriptor: number) => {
        view.setUint16(descriptor + 24, 1, true);
      },
      error: /sample 0 is malformed/i,
    },
    {
      label: 'sample bit length',
      mutate: (bytes: Uint8Array, _view: DataView, descriptor: number) => {
        bytes[descriptor + 26] = 126;
      },
      error: /sample 0 is malformed/i,
    },
    {
      label: 'sample qualifier',
      mutate: (bytes: Uint8Array, _view: DataView, descriptor: number) => {
        bytes[descriptor + 27] = bytes[descriptor + 27]! | 0x20;
      },
      error: /sample 0 is malformed/i,
    },
    {
      label: 'sample position',
      mutate: (bytes: Uint8Array, _view: DataView, descriptor: number) => {
        bytes[descriptor + 28] = 1;
      },
      error: /sample 0 is malformed/i,
    },
    {
      label: 'sample bounds',
      mutate: (_bytes: Uint8Array, view: DataView, descriptor: number) => {
        view.setUint32(descriptor + 36, 0, true);
      },
      error: /sample 0 is malformed/i,
    },
  ])('rejects malformed DFD $label', async ({ mutate, error }) => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          mutate(bytes, view, view.getUint32(48, true) + 4);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(error);
  });

  it('rejects invalid raw-UASTC bytesPlane metadata', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          const descriptor = view.getUint32(48, true) + 4;
          bytes.fill(0, descriptor + 16, descriptor + 24);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/invalid bytesPlane metadata/i);
  });

  it('rejects a non-color UASTC channel layout under an sRGB transfer', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          const descriptor = view.getUint32(48, true) + 4;
          bytes[descriptor + 27] = 4;
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/one\/two-component Basis textures must use a linear transfer/i);
  });

  it('accepts the optional LINEAR qualifier on a separate nonlinear ETC1S alpha slice', async () => {
    const decoded = await decodeRawBasisKtx2Pixels(
      mutatedEtc1sFixture((bytes, view) => {
        const descriptor = view.getUint32(48, true) + 4;
        bytes[descriptor + 43] = 0x1f;
      }),
      CONTEXT,
    );
    expect(decoded).toMatchObject({ width: 8, height: 8 });
  });

  it('accepts a linear DFD for a linear material role', async () => {
    const decoded = await decodeRawBasisKtx2Pixels(
      mutatedFixture((bytes, view) => {
        const dfdOffset = view.getUint32(48, true);
        bytes[dfdOffset + 13] = 0;
        bytes[dfdOffset + 14] = 1;
      }),
      {
        ...CONTEXT,
        materialField: 'normalMap',
        colorSpace: 'linear',
      },
    );

    expect(decoded).toMatchObject({
      width: 4,
      height: 4,
      colorSpace: 'linear',
    });
  });

  it('rejects non-rd KTX orientation metadata', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          const kvdOffset = view.getUint32(56, true);
          const kvdLength = view.getUint32(60, true);
          bytes.fill(0, kvdOffset, kvdOffset + kvdLength);

          view.setUint32(kvdOffset, 17, true);
          bytes.set(new TextEncoder().encode('KTXorientation\0ru'), kvdOffset + 4);

          view.setUint32(kvdOffset + 24, 16, true);
          bytes.set(new TextEncoder().encode('ignored\0padding!'), kvdOffset + 28);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/orientation.*rd/i);
  });

  it('rejects non-zero KTX key/value padding bytes', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          const kvdOffset = view.getUint32(56, true);
          const firstEntryLength = view.getUint32(kvdOffset, true);
          bytes[kvdOffset + 4 + firstEntryLength] = 1;
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/padding bytes must be zero/);
  });

  it('rejects UASTC supercompression global data instead of overlapping it with pixels', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((_bytes, view) => {
          view.setBigUint64(64, 192n, true);
          view.setBigUint64(72, 16n, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/UASTC payload must not declare supercompression global data/i);
  });

  it('rejects unsafe UInt64 BasisLZ SGD metadata', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedEtc1sFixture((_bytes, view) => {
          view.setBigUint64(64, BigInt(Number.MAX_SAFE_INTEGER) + 1n, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/global-data offset exceeds JavaScript's safe integer range/i);
  });

  it('rejects a BasisLZ level with nonzero uncompressedByteLength', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedEtc1sFixture((_bytes, view) => {
          view.setBigUint64(96, 1n, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/BasisLZ level 0 must use uncompressedByteLength 0/i);
  });

  it.each([1, 39])(
    'rejects an in-container but truncated %i-byte BasisLZ SGD',
    async (sgdLength) => {
      await expect(
        decodeRawBasisKtx2Pixels(shortenedEtc1sSgdFixture(sgdLength), CONTEXT),
      ).rejects.toThrow(/global data is too short for its image descriptors/i);
    },
  );

  it('rejects a BasisLZ SGD whose component lengths do not consume the section', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedEtc1sFixture((_bytes, view) => {
          const sgdOffset = Number(view.getBigUint64(64, true));
          view.setUint32(sgdOffset + 4, view.getUint32(sgdOffset + 4, true) + 1, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/global data header\/length is malformed/i);
  });

  it('rejects a BasisLZ RGB slice that exceeds its indexed level', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedEtc1sFixture((_bytes, view) => {
          const imageDescriptor = Number(view.getBigUint64(64, true)) + 20;
          view.setUint32(imageDescriptor + 8, 4, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/level 0 RGB slice is malformed/i);
  });

  it('rejects unsupported BasisLZ image flags for static glTF material images', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedEtc1sFixture((_bytes, view) => {
          const imageDescriptor = Number(view.getBigUint64(64, true)) + 20;
          view.setUint32(imageDescriptor, 2, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/level 0 uses unsupported image flags/i);
  });

  it('rejects a two-sample ETC1S DFD whose SGD omits its alpha slice', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedEtc1sFixture((_bytes, view) => {
          const imageDescriptor = Number(view.getBigUint64(64, true)) + 20;
          view.setUint32(imageDescriptor + 12, 0, true);
          view.setUint32(imageDescriptor + 16, 0, true);
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/alpha slice disagrees with its DFD/i);
  });

  it('rejects premultiplied-alpha Basis material images', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          const dfdOffset = view.getUint32(48, true);
          bytes[dfdOffset + 15] = 1;
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/zero DFD flags.*premultiplied/i);
  });

  it('rejects reserved DFD flag bits, not only premultiplied alpha', async () => {
    await expect(
      decodeRawBasisKtx2Pixels(
        mutatedFixture((bytes, view) => {
          const dfdOffset = view.getUint32(48, true);
          bytes[dfdOffset + 15] = 2;
        }),
        CONTEXT,
      ),
    ).rejects.toThrow(/zero DFD flags.*reserved/i);
  });
});
