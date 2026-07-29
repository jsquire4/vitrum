// Built-in DDS base-level decoder for MSFT_texture_dds.
//
// The vendor extension does not constrain the DDS pixel format. This decoder
// covers the portable formats found in WebGL-era glTF assets: BC1/DXT1,
// BC2/DXT3, BC3/DXT5, BC4, BC5, and masked 8/16/24/32-bit RGB(A). DX10
// containers for the equivalent UNORM/SRGB formats are accepted. Unsupported
// HDR/signed/BC6H/BC7/array/cube/volume payloads fail explicitly so the adapter
// never returns invented pixels.

import type { RawImageHandle } from './textures.js';
import type { DecodeGltfTexturePixelsFn, GltfDecodedTexturePixels } from './texturePipeline.js';
import { localUint8ArrayView } from './intrinsicTypedArrays.js';

const DDS_MAGIC = 0x2053_4444;
const DDS_HEADER_BYTES = 128;
const DDS_DX10_HEADER_BYTES = 20;
const DDS_PIXEL_FORMAT_OFFSET = 76;
const DDS_CAPS2_OFFSET = 112;
const DDSD_HEIGHT = 0x2;
const DDSD_WIDTH = 0x4;
const DDSD_PITCH = 0x8;
const DDSD_MIPMAPCOUNT = 0x0002_0000;
const DDSD_LINEARSIZE = 0x0008_0000;
const DDSD_DEPTH = 0x0080_0000;
const DDSCAPS2_CUBEMAP_ALLFACES = 0x0000_fe00;
const DDSCAPS2_VOLUME = 0x0020_0000;

const DDPF_ALPHAPIXELS = 0x1;
const DDPF_ALPHA = 0x2;
const DDPF_FOURCC = 0x4;
const DDPF_RGB = 0x40;
const DDPF_LUMINANCE = 0x0002_0000;

const FOURCC_DXT1 = fourCc('DXT1');
const FOURCC_DXT3 = fourCc('DXT3');
const FOURCC_DXT5 = fourCc('DXT5');
const FOURCC_ATI1 = fourCc('ATI1');
const FOURCC_ATI2 = fourCc('ATI2');
const FOURCC_BC4U = fourCc('BC4U');
const FOURCC_BC5U = fourCc('BC5U');
const FOURCC_DX10 = fourCc('DX10');

const D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3;
const DDS_ALPHA_MODE_MASK = 0x7;
const DDS_ALPHA_MODE_PREMULTIPLIED = 0x2;
const DDS_ALPHA_MODE_OPAQUE = 0x3;
const DDS_ALPHA_MODE_CUSTOM = 0x4;

const DXGI_R8G8B8A8_UNORM = 28;
const DXGI_R8G8B8A8_UNORM_SRGB = 29;
const DXGI_BC1_UNORM = 71;
const DXGI_BC1_UNORM_SRGB = 72;
const DXGI_BC2_UNORM = 74;
const DXGI_BC2_UNORM_SRGB = 75;
const DXGI_BC3_UNORM = 77;
const DXGI_BC3_UNORM_SRGB = 78;
const DXGI_BC4_UNORM = 80;
const DXGI_BC5_UNORM = 83;
const DXGI_B8G8R8A8_UNORM = 87;
const DXGI_B8G8R8A8_UNORM_SRGB = 91;

const MAX_RGBA8_PIXELS = 0x3fff_ffff;

type DdsFormat =
  | { readonly kind: 'bc1' | 'bc2' | 'bc3' | 'bc4' | 'bc5' }
  | {
      readonly kind: 'masked';
      readonly bitsPerPixel: 8 | 16 | 24 | 32;
      readonly rMask: number;
      readonly gMask: number;
      readonly bMask: number;
      readonly aMask: number;
      readonly luminance: boolean;
      readonly alphaOnly: boolean;
    };

interface ParsedDds {
  readonly width: number;
  readonly height: number;
  readonly dataOffset: number;
  readonly mipCount: number;
  /** Present only when DDSD_PITCH makes the header field meaningful. */
  readonly rowPitch?: number;
  /** Present only when DDSD_LINEARSIZE makes the header field meaningful. */
  readonly linearSize?: number;
  readonly format: DdsFormat;
  readonly forceOpaqueAlpha: boolean;
  /** Present only when a DX10 format declares the source transfer function. */
  readonly colorSpace?: 'srgb' | 'linear';
}

type ParsedDdsHeader = Pick<ParsedDds, 'width' | 'height' | 'mipCount' | 'rowPitch' | 'linearSize'>;

function fourCc(value: string): number {
  return (
    (value.charCodeAt(0) |
      (value.charCodeAt(1) << 8) |
      (value.charCodeAt(2) << 16) |
      (value.charCodeAt(3) << 24)) >>>
    0
  );
}

function viewFor(bytes: Uint8Array, path: string): DataView {
  if (bytes.byteLength < DDS_HEADER_BYTES) {
    throw new Error(`[vitrum/gltf-adapter] ${path} DDS header is truncated.`);
  }
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

export function isDdsBytes(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 4) return false;
  return (
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true) === DDS_MAGIC
  );
}

export function canDecodeRawDdsPixels(handle: RawImageHandle): boolean {
  const bytes = localUint8ArrayView(handle.data);
  if (bytes === null) return false;
  const mimeType = handle.mimeType.trim().toLowerCase();
  return mimeType === 'image/vnd-ms.dds' || isDdsBytes(bytes);
}

function assertPixelBudget(
  width: number,
  height: number,
  path: string,
  maxDecodedTexturePixels: number,
): number {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS header has invalid dimensions ${width}x${height}.`,
    );
  }
  const pixels = BigInt(width) * BigInt(height);
  if (pixels > BigInt(MAX_RGBA8_PIXELS)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS dimensions ${width}x${height} exceed the RGBA8 decoder address space.`,
    );
  }
  if (maxDecodedTexturePixels > 0 && pixels > BigInt(maxDecodedTexturePixels)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS dimensions ${width}x${height} ` +
        `exceed maxDecodedTexturePixels ${maxDecodedTexturePixels}.`,
    );
  }
  return Number(pixels);
}

function parseDds(bytes: Uint8Array, path: string): ParsedDds {
  const view = viewFor(bytes, path);
  if (view.getUint32(0, true) !== DDS_MAGIC) {
    throw new Error(`[vitrum/gltf-adapter] ${path} is not a DDS byte stream.`);
  }
  if (view.getUint32(4, true) !== 124 || view.getUint32(DDS_PIXEL_FORMAT_OFFSET, true) !== 32) {
    throw new Error(`[vitrum/gltf-adapter] ${path} has an invalid DDS header size.`);
  }

  const headerFlags = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const width = view.getUint32(16, true);
  const pitchOrLinearSize = view.getUint32(20, true);
  const depth = view.getUint32(24, true);
  const rawMipCount = view.getUint32(28, true);
  const caps2 = view.getUint32(DDS_CAPS2_OFFSET, true);
  if ((headerFlags & (DDSD_HEIGHT | DDSD_WIDTH)) !== (DDSD_HEIGHT | DDSD_WIDTH)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS header must declare valid height and width fields.`,
    );
  }
  const pitchDeclared = (headerFlags & DDSD_PITCH) !== 0;
  const linearSizeDeclared = (headerFlags & DDSD_LINEARSIZE) !== 0;
  if (pitchDeclared && linearSizeDeclared) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS header contradictorily declares both pitch and linear-size semantics.`,
    );
  }
  // Legacy writers may omit caps/pixel-format/mipmap validity bits, so only
  // DDSD_MIPMAPCOUNT makes the otherwise stale union field authoritative.
  const mipChainDeclared = (headerFlags & DDSD_MIPMAPCOUNT) !== 0;
  const mipCount = mipChainDeclared ? rawMipCount : 1;
  const maximumMipCount =
    width > 0 && height > 0 ? Math.floor(Math.log2(Math.max(width, height))) + 1 : 0;
  if (mipChainDeclared && (mipCount < 1 || mipCount > maximumMipCount)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS header declares invalid mipMapCount ${mipCount} ` +
        `for ${width}x${height}.`,
    );
  }
  if (
    (headerFlags & DDSD_DEPTH) !== 0 ||
    (caps2 & (DDSCAPS2_CUBEMAP_ALLFACES | DDSCAPS2_VOLUME)) !== 0
  ) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} is a DDS cube/volume texture ` +
        `(depth=${depth}); MSFT_texture_dds requires a 2D material image.`,
    );
  }
  const parsedHeader: ParsedDdsHeader = {
    width,
    height,
    mipCount,
    ...(pitchDeclared ? { rowPitch: pitchOrLinearSize } : {}),
    ...(linearSizeDeclared ? { linearSize: pitchOrLinearSize } : {}),
  };

  const pixelFlags = view.getUint32(80, true);
  const legacyFourCc = view.getUint32(84, true);
  const rgbBits = view.getUint32(88, true);
  const rMask = view.getUint32(92, true);
  const gMask = view.getUint32(96, true);
  const bMask = view.getUint32(100, true);
  const aMask = view.getUint32(104, true);

  if ((pixelFlags & DDPF_FOURCC) !== 0) {
    const unsupportedFourCcFlags = pixelFlags & ~(DDPF_FOURCC | DDPF_ALPHAPIXELS);
    if (unsupportedFourCcFlags !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} combines DDS FourCC data with unsupported pixel flags ` +
          `0x${unsupportedFourCcFlags.toString(16)}.`,
      );
    }
    if (pitchDeclared && legacyFourCc !== FOURCC_DX10) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} block-compressed DDS data cannot declare an uncompressed row pitch.`,
      );
    }
    if (legacyFourCc === FOURCC_DXT1) {
      return {
        ...parsedHeader,
        dataOffset: DDS_HEADER_BYTES,
        format: { kind: 'bc1' },
        forceOpaqueAlpha: false,
      };
    }
    if (legacyFourCc === FOURCC_DXT3) {
      return {
        ...parsedHeader,
        dataOffset: DDS_HEADER_BYTES,
        format: { kind: 'bc2' },
        forceOpaqueAlpha: false,
      };
    }
    if (legacyFourCc === FOURCC_DXT5) {
      return {
        ...parsedHeader,
        dataOffset: DDS_HEADER_BYTES,
        format: { kind: 'bc3' },
        forceOpaqueAlpha: false,
      };
    }
    if (legacyFourCc === FOURCC_ATI1 || legacyFourCc === FOURCC_BC4U) {
      return {
        ...parsedHeader,
        dataOffset: DDS_HEADER_BYTES,
        format: { kind: 'bc4' },
        forceOpaqueAlpha: false,
      };
    }
    if (legacyFourCc === FOURCC_ATI2 || legacyFourCc === FOURCC_BC5U) {
      return {
        ...parsedHeader,
        dataOffset: DDS_HEADER_BYTES,
        format: { kind: 'bc5' },
        forceOpaqueAlpha: false,
      };
    }
    if (legacyFourCc !== FOURCC_DX10) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} uses unsupported DDS FourCC 0x${legacyFourCc.toString(16)}.`,
      );
    }
    return parseDx10Dds(bytes, view, parsedHeader, path);
  }

  if (linearSizeDeclared) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} uncompressed DDS data cannot declare compressed linear-size semantics.`,
    );
  }
  const supportedMaskedFlags = DDPF_ALPHAPIXELS | DDPF_ALPHA | DDPF_RGB | DDPF_LUMINANCE;
  if ((pixelFlags & ~supportedMaskedFlags) !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} uses unsupported legacy DDS pixel flags ` +
        `0x${(pixelFlags & ~supportedMaskedFlags).toString(16)}.`,
    );
  }
  const isRgb = (pixelFlags & DDPF_RGB) !== 0;
  const isLuminance = (pixelFlags & DDPF_LUMINANCE) !== 0;
  const isAlphaOnly = (pixelFlags & DDPF_ALPHA) !== 0;
  const hasAlphaPixels = (pixelFlags & DDPF_ALPHAPIXELS) !== 0;
  const primaryFormatCount = Number(isRgb) + Number(isLuminance) + Number(isAlphaOnly);
  if (primaryFormatCount !== 1 || (isAlphaOnly && hasAlphaPixels)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} uses contradictory or unsupported legacy DDS pixel flags ` +
        `0x${pixelFlags.toString(16)}; expected exactly one of RGB, luminance, or alpha-only.`,
    );
  }
  if (hasAlphaPixels && aMask === 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS pixels declare alpha data but provide no alpha mask.`,
    );
  }
  if (!hasAlphaPixels && !isAlphaOnly && aMask !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS pixels provide an alpha mask without declaring alpha data.`,
    );
  }
  if (rgbBits !== 8 && rgbBits !== 16 && rgbBits !== 24 && rgbBits !== 32) {
    throw new Error(`[vitrum/gltf-adapter] ${path} uses unsupported ${rgbBits}-bit DDS pixels.`);
  }
  validateMaskedFormat(
    {
      bitsPerPixel: rgbBits,
      rMask,
      gMask,
      bMask,
      aMask: hasAlphaPixels || isAlphaOnly ? aMask : 0,
      luminance: isLuminance,
      alphaOnly: isAlphaOnly,
    },
    path,
  );
  return {
    ...parsedHeader,
    dataOffset: DDS_HEADER_BYTES,
    format: {
      kind: 'masked',
      bitsPerPixel: rgbBits,
      rMask,
      gMask,
      bMask,
      aMask: hasAlphaPixels || isAlphaOnly ? aMask : 0,
      luminance: isLuminance,
      alphaOnly: isAlphaOnly,
    },
    forceOpaqueAlpha: false,
  };
}

function maskIsContiguous(mask: number): boolean {
  if (mask === 0) return true;
  const shift = trailingZeroBits(mask);
  const normalized = (mask >>> shift) >>> 0;
  return (normalized & (normalized + 1)) === 0;
}

function validateMaskedFormat(
  format: Omit<Extract<DdsFormat, { readonly kind: 'masked' }>, 'kind'>,
  path: string,
): void {
  const { bitsPerPixel, rMask, gMask, bMask, aMask } = format;
  const masks = [
    ['red', rMask],
    ['green', gMask],
    ['blue', bMask],
    ['alpha', aMask],
  ] as const;
  const pixelMask = bitsPerPixel === 32 ? 0xffff_ffff : 2 ** bitsPerPixel - 1;
  for (const [name, mask] of masks) {
    if (mask > pixelMask) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} DDS ${name} mask 0x${mask.toString(16)} ` +
          `extends beyond its ${bitsPerPixel}-bit pixel.`,
      );
    }
    if (!maskIsContiguous(mask)) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} DDS ${name} mask 0x${mask.toString(16)} is not contiguous.`,
      );
    }
  }

  if (format.alphaOnly) {
    if (aMask === 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} DDS alpha-only pixels require a non-zero alpha mask.`,
      );
    }
    if (rMask !== 0 || gMask !== 0 || bMask !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} DDS alpha-only pixels contain unexpected color masks.`,
      );
    }
  } else if (format.luminance) {
    if (rMask === 0 || gMask !== 0 || bMask !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} DDS luminance pixels require one non-zero luminance mask.`,
      );
    }
  } else if (rMask === 0 || gMask === 0 || bMask === 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS RGB pixels require non-zero red, green, and blue masks.`,
    );
  }

  let occupied = 0;
  for (const [, mask] of masks) {
    if (mask === 0) continue;
    if ((occupied & mask) !== 0) {
      throw new Error(`[vitrum/gltf-adapter] ${path} DDS channel masks overlap.`);
    }
    occupied = (occupied | mask) >>> 0;
  }
}

function parseDx10Dds(
  bytes: Uint8Array,
  view: DataView,
  header: ParsedDdsHeader,
  path: string,
): ParsedDds {
  const dataOffset = DDS_HEADER_BYTES + DDS_DX10_HEADER_BYTES;
  if (bytes.byteLength < dataOffset) {
    throw new Error(`[vitrum/gltf-adapter] ${path} DDS DX10 header is truncated.`);
  }
  const dxgiFormat = view.getUint32(DDS_HEADER_BYTES, true);
  const resourceDimension = view.getUint32(DDS_HEADER_BYTES + 4, true);
  const miscFlag = view.getUint32(DDS_HEADER_BYTES + 8, true);
  const arraySize = view.getUint32(DDS_HEADER_BYTES + 12, true);
  const miscFlags2 = view.getUint32(DDS_HEADER_BYTES + 16, true);
  if (
    resourceDimension !== D3D10_RESOURCE_DIMENSION_TEXTURE2D ||
    arraySize !== 1 ||
    miscFlag !== 0
  ) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS DX10 payload is not a plain single 2D texture.`,
    );
  }
  const alphaMode = miscFlags2 & DDS_ALPHA_MODE_MASK;
  if ((miscFlags2 & ~DDS_ALPHA_MODE_MASK) !== 0 || alphaMode > DDS_ALPHA_MODE_CUSTOM) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS DX10 payload has invalid reserved alpha-mode metadata ` +
        `0x${miscFlags2.toString(16)}.`,
    );
  }
  if (alphaMode === DDS_ALPHA_MODE_PREMULTIPLIED || alphaMode === DDS_ALPHA_MODE_CUSTOM) {
    const unsupportedSemantics =
      alphaMode === DDS_ALPHA_MODE_PREMULTIPLIED
        ? 'premultiplied alpha'
        : 'application-defined custom alpha';
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS DX10 payload uses ${unsupportedSemantics}, ` +
        'which cannot be returned with verifiable straight-alpha glTF material semantics.',
    );
  }

  let format: DdsFormat;
  let colorSpace: 'srgb' | 'linear';
  switch (dxgiFormat) {
    case DXGI_BC1_UNORM:
      format = { kind: 'bc1' };
      colorSpace = 'linear';
      break;
    case DXGI_BC1_UNORM_SRGB:
      format = { kind: 'bc1' };
      colorSpace = 'srgb';
      break;
    case DXGI_BC2_UNORM:
      format = { kind: 'bc2' };
      colorSpace = 'linear';
      break;
    case DXGI_BC2_UNORM_SRGB:
      format = { kind: 'bc2' };
      colorSpace = 'srgb';
      break;
    case DXGI_BC3_UNORM:
      format = { kind: 'bc3' };
      colorSpace = 'linear';
      break;
    case DXGI_BC3_UNORM_SRGB:
      format = { kind: 'bc3' };
      colorSpace = 'srgb';
      break;
    case DXGI_BC4_UNORM:
      format = { kind: 'bc4' };
      colorSpace = 'linear';
      break;
    case DXGI_BC5_UNORM:
      format = { kind: 'bc5' };
      colorSpace = 'linear';
      break;
    case DXGI_R8G8B8A8_UNORM:
      format = {
        kind: 'masked',
        bitsPerPixel: 32,
        rMask: 0x0000_00ff,
        gMask: 0x0000_ff00,
        bMask: 0x00ff_0000,
        aMask: 0xff00_0000,
        luminance: false,
        alphaOnly: false,
      };
      colorSpace = 'linear';
      break;
    case DXGI_R8G8B8A8_UNORM_SRGB:
      format = {
        kind: 'masked',
        bitsPerPixel: 32,
        rMask: 0x0000_00ff,
        gMask: 0x0000_ff00,
        bMask: 0x00ff_0000,
        aMask: 0xff00_0000,
        luminance: false,
        alphaOnly: false,
      };
      colorSpace = 'srgb';
      break;
    case DXGI_B8G8R8A8_UNORM:
      format = {
        kind: 'masked',
        bitsPerPixel: 32,
        rMask: 0x00ff_0000,
        gMask: 0x0000_ff00,
        bMask: 0x0000_00ff,
        aMask: 0xff00_0000,
        luminance: false,
        alphaOnly: false,
      };
      colorSpace = 'linear';
      break;
    case DXGI_B8G8R8A8_UNORM_SRGB:
      format = {
        kind: 'masked',
        bitsPerPixel: 32,
        rMask: 0x00ff_0000,
        gMask: 0x0000_ff00,
        bMask: 0x0000_00ff,
        aMask: 0xff00_0000,
        luminance: false,
        alphaOnly: false,
      };
      colorSpace = 'srgb';
      break;
    default:
      throw new Error(
        `[vitrum/gltf-adapter] ${path} uses unsupported DDS DXGI format ${dxgiFormat}.`,
      );
  }
  if (format.kind === 'masked' && header.linearSize !== undefined) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} uncompressed DDS DX10 data cannot declare compressed linear-size semantics.`,
    );
  }
  if (format.kind !== 'masked' && header.rowPitch !== undefined) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} block-compressed DDS DX10 data cannot declare an uncompressed row pitch.`,
    );
  }
  return {
    ...header,
    dataOffset,
    format,
    forceOpaqueAlpha: alphaMode === DDS_ALPHA_MODE_OPAQUE,
    colorSpace,
  };
}

function rgb565(value: number): readonly [number, number, number] {
  const r5 = (value >>> 11) & 0x1f;
  const g6 = (value >>> 5) & 0x3f;
  const b5 = value & 0x1f;
  return [(r5 << 3) | (r5 >>> 2), (g6 << 2) | (g6 >>> 4), (b5 << 3) | (b5 >>> 2)];
}

function colorPalette(
  c0: number,
  c1: number,
  allowBc1Transparency: boolean,
): readonly (readonly [number, number, number, number])[] {
  const a = rgb565(c0);
  const b = rgb565(c1);
  if (allowBc1Transparency && c0 <= c1) {
    return [
      [a[0], a[1], a[2], 255],
      [b[0], b[1], b[2], 255],
      [
        Math.floor((a[0] + b[0]) / 2),
        Math.floor((a[1] + b[1]) / 2),
        Math.floor((a[2] + b[2]) / 2),
        255,
      ],
      [0, 0, 0, 0],
    ];
  }
  return [
    [a[0], a[1], a[2], 255],
    [b[0], b[1], b[2], 255],
    [
      Math.floor((2 * a[0] + b[0]) / 3),
      Math.floor((2 * a[1] + b[1]) / 3),
      Math.floor((2 * a[2] + b[2]) / 3),
      255,
    ],
    [
      Math.floor((a[0] + 2 * b[0]) / 3),
      Math.floor((a[1] + 2 * b[1]) / 3),
      Math.floor((a[2] + 2 * b[2]) / 3),
      255,
    ],
  ];
}

function alphaPalette(a0: number, a1: number): readonly number[] {
  if (a0 > a1) {
    return [
      a0,
      a1,
      Math.floor((6 * a0 + a1) / 7),
      Math.floor((5 * a0 + 2 * a1) / 7),
      Math.floor((4 * a0 + 3 * a1) / 7),
      Math.floor((3 * a0 + 4 * a1) / 7),
      Math.floor((2 * a0 + 5 * a1) / 7),
      Math.floor((a0 + 6 * a1) / 7),
    ];
  }
  return [
    a0,
    a1,
    Math.floor((4 * a0 + a1) / 5),
    Math.floor((3 * a0 + 2 * a1) / 5),
    Math.floor((2 * a0 + 3 * a1) / 5),
    Math.floor((a0 + 4 * a1) / 5),
    0,
    255,
  ];
}

function littleEndianBits(bytes: Uint8Array, offset: number, byteCount: number): bigint {
  let value = 0n;
  for (let i = 0; i < byteCount; i += 1) {
    value |= BigInt(bytes[offset + i]!) << BigInt(i * 8);
  }
  return value;
}

function writeBlockPixel(
  output: Uint8Array,
  width: number,
  height: number,
  blockX: number,
  blockY: number,
  localIndex: number,
  rgba: readonly [number, number, number, number],
): void {
  const x = blockX * 4 + (localIndex & 3);
  const y = blockY * 4 + (localIndex >>> 2);
  if (x >= width || y >= height) return;
  const dst = (y * width + x) * 4;
  output[dst] = rgba[0];
  output[dst + 1] = rgba[1];
  output[dst + 2] = rgba[2];
  output[dst + 3] = rgba[3];
}

function requireBytes(bytes: Uint8Array, offset: number, count: number, path: string): void {
  if (
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(count) ||
    offset < 0 ||
    count < 0 ||
    offset + count > bytes.byteLength
  ) {
    throw new Error(`[vitrum/gltf-adapter] ${path} DDS base level is truncated.`);
  }
}

function mipDimension(size: number, level: number): number {
  return Math.max(1, Math.floor(size / 2 ** level));
}

function bcLevelByteLength(width: number, height: number, blockBytes: number): bigint {
  const blocksX = Math.max(1, Math.ceil(width / 4));
  const blocksY = Math.max(1, Math.ceil(height / 4));
  return BigInt(blocksX) * BigInt(blocksY) * BigInt(blockBytes);
}

function maskedRowStride(
  parsed: ParsedDds & {
    readonly format: Extract<DdsFormat, { readonly kind: 'masked' }>;
  },
  path: string,
): number {
  const bytesPerPixel = parsed.format.bitsPerPixel / 8;
  const minimumStride = parsed.width * bytesPerPixel;
  if (parsed.rowPitch !== undefined && parsed.rowPitch < minimumStride) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} DDS row pitch ${parsed.rowPitch} is smaller than ` +
        `the ${minimumStride}-byte packed row.`,
    );
  }
  return parsed.rowPitch ?? minimumStride;
}

function validateDdsPayload(bytes: Uint8Array, parsed: ParsedDds, path: string): void {
  let requiredPayloadBytes = 0n;
  if (parsed.format.kind === 'masked') {
    const masked = parsed as ParsedDds & {
      readonly format: Extract<DdsFormat, { readonly kind: 'masked' }>;
    };
    const bytesPerPixel = masked.format.bitsPerPixel / 8;
    requiredPayloadBytes = BigInt(maskedRowStride(masked, path)) * BigInt(masked.height);
    for (let level = 1; level < masked.mipCount; level += 1) {
      const width = mipDimension(masked.width, level);
      const height = mipDimension(masked.height, level);
      requiredPayloadBytes += BigInt(width) * BigInt(height) * BigInt(bytesPerPixel);
    }
  } else {
    const blockBytes = parsed.format.kind === 'bc1' || parsed.format.kind === 'bc4' ? 8 : 16;
    const baseBytes = bcLevelByteLength(parsed.width, parsed.height, blockBytes);
    if (parsed.linearSize !== undefined && BigInt(parsed.linearSize) < baseBytes) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} DDS linear size ${parsed.linearSize} is smaller than ` +
          `the ${baseBytes.toString()}-byte compressed base level.`,
      );
    }
    // Microsoft recommends recomputing BC surface sizes because legacy writers
    // frequently leave an oversized advisory dwLinearSize. Never let that
    // advisory value mask a missing lower mip.
    requiredPayloadBytes = baseBytes;
    for (let level = 1; level < parsed.mipCount; level += 1) {
      requiredPayloadBytes += bcLevelByteLength(
        mipDimension(parsed.width, level),
        mipDimension(parsed.height, level),
        blockBytes,
      );
    }
  }

  if (BigInt(parsed.dataOffset) + requiredPayloadBytes > BigInt(bytes.byteLength)) {
    throw new Error(`[vitrum/gltf-adapter] ${path} DDS declared mip payload is truncated.`);
  }
}

function decodeBc(
  bytes: Uint8Array,
  parsed: ParsedDds,
  output: Uint8Array,
  path: string,
  reconstructNormalZ: boolean,
): void {
  const { width, height } = parsed;
  const blocksX = Math.ceil(width / 4);
  const blocksY = Math.ceil(height / 4);
  const blockBytes = parsed.format.kind === 'bc1' || parsed.format.kind === 'bc4' ? 8 : 16;
  const totalBytes = blocksX * blocksY * blockBytes;
  requireBytes(bytes, parsed.dataOffset, totalBytes, path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  for (let by = 0; by < blocksY; by += 1) {
    for (let bx = 0; bx < blocksX; bx += 1) {
      const block = parsed.dataOffset + (by * blocksX + bx) * blockBytes;
      if (parsed.format.kind === 'bc4' || parsed.format.kind === 'bc5') {
        const r = decodeBcAlphaBlock(bytes, block);
        const g = parsed.format.kind === 'bc5' ? decodeBcAlphaBlock(bytes, block + 8) : null;
        for (let i = 0; i < 16; i += 1) {
          const red = r[i]!;
          const green = g?.[i] ?? 0;
          writeBlockPixel(output, width, height, bx, by, i, [
            red,
            green,
            parsed.format.kind === 'bc5' && reconstructNormalZ
              ? reconstructPositiveNormalZ(red, green)
              : 0,
            255,
          ]);
        }
        continue;
      }

      let colorOffset = block;
      let alphaValues: readonly number[] | null = null;
      if (parsed.format.kind === 'bc2') {
        const alphaBits = littleEndianBits(bytes, block, 8);
        alphaValues = Array.from(
          { length: 16 },
          (_, i) => Number((alphaBits >> BigInt(i * 4)) & 0xfn) * 17,
        );
        colorOffset += 8;
      } else if (parsed.format.kind === 'bc3') {
        alphaValues = decodeBcAlphaBlock(bytes, block);
        colorOffset += 8;
      }
      const c0 = view.getUint16(colorOffset, true);
      const c1 = view.getUint16(colorOffset + 2, true);
      const palette = colorPalette(c0, c1, parsed.format.kind === 'bc1');
      const indices = view.getUint32(colorOffset + 4, true);
      for (let i = 0; i < 16; i += 1) {
        const source = palette[(indices >>> (i * 2)) & 3]!;
        writeBlockPixel(output, width, height, bx, by, i, [
          source[0],
          source[1],
          source[2],
          alphaValues?.[i] ?? source[3],
        ]);
      }
    }
  }
}

function reconstructPositiveNormalZ(red: number, green: number): number {
  const x = (red / 255) * 2 - 1;
  const y = (green / 255) * 2 - 1;
  const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
  return Math.round((z * 0.5 + 0.5) * 255);
}

function isNormalMapField(
  field: Parameters<DecodeGltfTexturePixelsFn>[1]['materialField'],
): boolean {
  return field === 'normalMap' || field === 'clearcoatNormalMap';
}

function decodeBcAlphaBlock(bytes: Uint8Array, offset: number): readonly number[] {
  const palette = alphaPalette(bytes[offset]!, bytes[offset + 1]!);
  const indices = littleEndianBits(bytes, offset + 2, 6);
  return Array.from({ length: 16 }, (_, i) => palette[Number((indices >> BigInt(i * 3)) & 0x7n)]!);
}

function trailingZeroBits(mask: number): number {
  if (mask === 0) return 0;
  let shift = 0;
  let value = mask >>> 0;
  while ((value & 1) === 0) {
    value >>>= 1;
    shift += 1;
  }
  return shift;
}

function maskedChannel(pixel: number, mask: number, fallback: number): number {
  if (mask === 0) return fallback;
  const shift = trailingZeroBits(mask);
  const maximum = (mask >>> shift) >>> 0;
  const value = ((pixel & mask) >>> shift) >>> 0;
  return Math.round((value * 255) / maximum);
}

function readPackedPixel(bytes: Uint8Array, offset: number, byteCount: number): number {
  let value = 0;
  for (let i = 0; i < byteCount; i += 1) {
    value += bytes[offset + i]! * 2 ** (i * 8);
  }
  return value >>> 0;
}

function decodeMasked(
  bytes: Uint8Array,
  parsed: ParsedDds & { readonly format: Extract<DdsFormat, { readonly kind: 'masked' }> },
  output: Uint8Array,
  path: string,
): void {
  const bytesPerPixel = parsed.format.bitsPerPixel / 8;
  const rowStride = maskedRowStride(parsed, path);
  requireBytes(bytes, parsed.dataOffset, rowStride * parsed.height, path);

  for (let y = 0; y < parsed.height; y += 1) {
    for (let x = 0; x < parsed.width; x += 1) {
      const pixel = readPackedPixel(
        bytes,
        parsed.dataOffset + y * rowStride + x * bytesPerPixel,
        bytesPerPixel,
      );
      const dst = (y * parsed.width + x) * 4;
      if (parsed.format.alphaOnly) {
        output[dst] = 255;
        output[dst + 1] = 255;
        output[dst + 2] = 255;
        output[dst + 3] = maskedChannel(pixel, parsed.format.aMask, 255);
        continue;
      }
      const r = maskedChannel(pixel, parsed.format.rMask, 0);
      const g = parsed.format.luminance ? r : maskedChannel(pixel, parsed.format.gMask, 0);
      const b = parsed.format.luminance ? r : maskedChannel(pixel, parsed.format.bMask, 0);
      output[dst] = r;
      output[dst + 1] = g;
      output[dst + 2] = b;
      output[dst + 3] = maskedChannel(pixel, parsed.format.aMask, 255);
    }
  }
}

export const decodeRawDdsPixels: DecodeGltfTexturePixelsFn = (
  handle,
  context,
): GltfDecodedTexturePixels => {
  const source = localUint8ArrayView(handle.data);
  if (source === null) {
    throw new Error(
      `[vitrum/gltf-adapter] ${context.path} DDS data must be an intrinsic Uint8Array.`,
    );
  }
  const bytes = new Uint8Array(source);
  const parsed = parseDds(bytes, context.path);
  const pixelCount = assertPixelBudget(
    parsed.width,
    parsed.height,
    context.path,
    context.maxDecodedTexturePixels,
  );
  // Reject truncated or contradictory payload declarations before reserving
  // the potentially much larger RGBA8 output.
  validateDdsPayload(bytes, parsed, context.path);
  const output = new Uint8Array(pixelCount * 4);
  if (parsed.format.kind === 'masked') {
    decodeMasked(
      bytes,
      parsed as ParsedDds & {
        readonly format: Extract<DdsFormat, { readonly kind: 'masked' }>;
      },
      output,
      context.path,
    );
  } else {
    decodeBc(bytes, parsed, output, context.path, isNormalMapField(context.materialField));
  }
  if (parsed.forceOpaqueAlpha) {
    for (let alpha = 3; alpha < output.length; alpha += 4) {
      output[alpha] = 255;
    }
  }
  return {
    width: parsed.width,
    height: parsed.height,
    data: output,
    channels: 4,
    dataType: 'uint8',
    colorSpace: parsed.colorSpace ?? context.colorSpace,
  };
};
