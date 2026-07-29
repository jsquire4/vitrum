// Built-in KTX2/Basis Universal -> RGBA8 decoder.
//
// The vendored binary is pinned byte-for-byte to the
// @h00w/basis-universal-transcoder 2.1.0 C-ABI WebAssembly release. We
// intentionally instantiate that ABI directly instead of importing its
// JavaScript wrapper: v2.1.0's published ESM wrapper references CommonJS-only
// `require`/`__dirname` on Node. Keeping the small ABI adapter here makes the
// same decoder work in standards-based browsers and native ESM Node without
// mutating process globals.
//
// Basis Universal: Binomial LLC, Apache-2.0.
// https://github.com/BinomialLLC/basis_universal

import type { RawImageHandle } from './textures.js';
import type { DecodeGltfTexturePixelsFn, GltfDecodedTexturePixels } from './texturePipeline.js';
import { localUint8ArrayView } from './intrinsicTypedArrays.js';

const BASIS_TRANSCODER_WASM_URL = new URL('./assets/basis_capi_transcoder.wasm', import.meta.url);

const KTX2_SIGNATURE = Object.freeze([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
] as const);

// basis_universal transcoder_texture_format::cTFRGBA32.
const BASIS_RGBA32_FORMAT = 13;
const KTX2_HEADER_BYTES = 80;
const KTX2_LEVEL_INDEX_ENTRY_BYTES = 24;
const KTX2_HEADER_U32_LENGTH = 20;
const KTX2_LEVEL_INFO_U32_LENGTH = 14;
const KTX2_LEVEL_INFO_BYTES = KTX2_LEVEL_INFO_U32_LENGTH * Uint32Array.BYTES_PER_ELEMENT;
const KTX2_VK_FORMAT_UNDEFINED = 0;
const KTX2_SUPERCOMPRESSION_NONE = 0;
const KTX2_SUPERCOMPRESSION_BASIS_LZ = 1;
const KTX2_SUPERCOMPRESSION_ZSTD = 2;
const KHR_DF_MODEL_ETC1S = 163;
const KHR_DF_MODEL_UASTC = 166;
const KHR_DF_BASIC_VERSION = 2;
const KHR_DF_PRIMARIES_UNSPECIFIED = 0;
const KHR_DF_PRIMARIES_BT709 = 1;
const KHR_DF_TRANSFER_LINEAR = 1;
const KHR_DF_TRANSFER_SRGB = 2;
const KHR_DF_CHANNEL_ETC1S_RGB = 0;
const KHR_DF_CHANNEL_ETC1S_RRR = 3;
const KHR_DF_CHANNEL_ETC1S_GGG = 4;
const KHR_DF_CHANNEL_ETC1S_AAA = 15;
const KHR_DF_CHANNEL_UASTC_RGB = 0;
const KHR_DF_CHANNEL_UASTC_RGBA = 3;
const KHR_DF_CHANNEL_UASTC_RRR = 4;
const KHR_DF_CHANNEL_UASTC_RG = 6;
const KTX2_DFD_MIN_BYTES = 44;
const KTX2_DFD_BASIC_HEADER_BYTES = 24;
const KTX2_DFD_SAMPLE_BYTES = 16;
const KTX2_BASIS_LZ_SGD_HEADER_BYTES = 20;
const KTX2_BASIS_LZ_IMAGE_DESC_BYTES = 20;
const KHR_DF_SAMPLE_DATATYPE_LINEAR = 0x10;
const MAX_WASM32_ALLOCATION_BYTES = 0x7fff_ffff;
const WASM_PAGE_BYTES = 64 * 1024;
const MAX_RGBA8_PIXELS = 0x1fff_ffff;

type WasmNumberFunction = (...args: number[]) => number;

interface BasisWasmExports {
  readonly memory: WebAssembly.Memory;
  readonly initializeRuntime: () => void;
  readonly malloc: (size: number) => number;
  readonly free: (pointer: number) => void;
  readonly initializeBasis: () => void;
  readonly computeTranscodedSize: (format: number, width: number, height: number) => number;
  readonly createKtx2Transcoder: () => number;
  readonly deleteKtx2Transcoder: (transcoder: number) => void;
  readonly initializeKtx2Transcoder: (
    transcoder: number,
    data: number,
    byteLength: number,
  ) => number;
  readonly getKtx2Header: (transcoder: number) => number;
  readonly startKtx2Transcoding: (transcoder: number) => number;
  readonly getKtx2LevelInfo: (
    transcoder: number,
    result: number,
    level: number,
    layer: number,
    face: number,
  ) => number;
  readonly transcodeKtx2Level: (
    transcoder: number,
    level: number,
    layer: number,
    face: number,
    output: number,
    outputPixels: number,
    format: number,
    decodeFlags: number,
    outputRowPitchPixels: number,
    outputRows: number,
    channel0: number,
    channel1: number,
    state: number,
  ) => number;
}

let basisModulePromise: Promise<BasisWasmExports> | null = null;

export function isBasisKtx2Bytes(bytes: Uint8Array): boolean {
  if (bytes.length < KTX2_SIGNATURE.length) return false;
  return KTX2_SIGNATURE.every((byte, index) => bytes[index] === byte);
}

export function canDecodeRawBasisKtx2Pixels(handle: RawImageHandle): boolean {
  const data = localUint8ArrayView(handle.data);
  if (data === null) return false;
  const mimeType = handle.mimeType.trim().toLowerCase();
  return mimeType === 'image/ktx2' || isBasisKtx2Bytes(data);
}

function asFunction(exports: WebAssembly.Exports, name: string): WasmNumberFunction {
  const value = exports[name];
  if (typeof value !== 'function') {
    throw new Error(`Basis Universal WASM ABI mismatch: export "${name}" is missing.`);
  }
  return value as WasmNumberFunction;
}

function asMemory(exports: WebAssembly.Exports, name: string): WebAssembly.Memory {
  const value = exports[name];
  if (!(value instanceof WebAssembly.Memory)) {
    throw new Error(`Basis Universal WASM ABI mismatch: memory export "${name}" is missing.`);
  }
  return value;
}

/**
 * Map the minified C-ABI export names emitted by the exact pinned v2.1.0
 * dependency. A dependency upgrade must update this adapter and its real KTX2
 * decode fixture together; silent ABI drift is rejected at initialization.
 */
function bindBasisExports(exports: WebAssembly.Exports): BasisWasmExports {
  return {
    memory: asMemory(exports, 'm'),
    initializeRuntime: asFunction(exports, 'n'),
    malloc: asFunction(exports, 'o'),
    free: asFunction(exports, 'p'),
    initializeBasis: asFunction(exports, 'q'),
    computeTranscodedSize: asFunction(exports, 'r'),
    createKtx2Transcoder: asFunction(exports, 'B'),
    deleteKtx2Transcoder: asFunction(exports, 'C'),
    initializeKtx2Transcoder: asFunction(exports, 'D'),
    getKtx2Header: asFunction(exports, 'E'),
    startKtx2Transcoding: asFunction(exports, 'G'),
    getKtx2LevelInfo: asFunction(exports, 'H'),
    transcodeKtx2Level: asFunction(exports, 'I'),
  };
}

async function readBasisWasmBinary(): Promise<ArrayBuffer> {
  if (BASIS_TRANSCODER_WASM_URL.protocol === 'file:') {
    // Keep the Node fallback out of browser bundles. The statically constructed
    // URL above remains visible to Vite/Rollup/Webpack so the package-owned
    // WASM is emitted as an asset in browser builds.
    const nodeFsSpecifier = `node:${'fs/promises'}`;
    const nodeFs = (await import(/* @vite-ignore */ nodeFsSpecifier)) as {
      readonly readFile: (url: URL) => Promise<Uint8Array>;
    };
    const bytes = await nodeFs.readFile(BASIS_TRANSCODER_WASM_URL);
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    return copy.buffer;
  }

  const response = await fetch(BASIS_TRANSCODER_WASM_URL);
  if (!response.ok) {
    throw new Error(
      `Basis Universal WASM fetch failed ` +
        `(${response.status} ${response.statusText}) from ` +
        BASIS_TRANSCODER_WASM_URL.href,
    );
  }
  return response.arrayBuffer();
}

async function instantiateBasisModule(): Promise<BasisWasmExports> {
  const wasm = await readBasisWasmBinary();
  let memory: WebAssembly.Memory | null = null;

  const noop = (): number => 0;
  const imports: WebAssembly.Imports = {
    a: {
      // The pinned C-ABI binary retains Emscripten registration imports, but
      // this adapter calls only plain C exports and therefore needs no embind
      // type registry. The remaining imports implement abort, heap growth, and
      // stdout/stderr writes.
      a: noop,
      b: noop,
      c: noop,
      d: noop,
      e: noop,
      f: noop,
      g: noop,
      h: noop,
      i: () => {
        throw new Error('Basis Universal WASM aborted.');
      },
      j: (requestedSize: number): number => {
        if (memory === null) return 0;
        try {
          const currentSize = memory.buffer.byteLength;
          if (requestedSize <= currentSize) return 1;
          memory.grow(Math.ceil((requestedSize - currentSize) / WASM_PAGE_BYTES));
          return 1;
        } catch {
          return 0;
        }
      },
      k: noop,
      l: noop,
    },
  };

  const instantiated = await WebAssembly.instantiate(wasm, imports);
  const bound = bindBasisExports(instantiated.instance.exports);
  memory = bound.memory;
  bound.initializeRuntime();
  bound.initializeBasis();
  return bound;
}

async function basisModule(): Promise<BasisWasmExports> {
  if (basisModulePromise !== null) return basisModulePromise;
  const pending = instantiateBasisModule();
  basisModulePromise = pending;
  void pending.catch(() => {
    if (basisModulePromise === pending) basisModulePromise = null;
  });
  return pending;
}

function checkedPixelCount(
  width: number,
  height: number,
  path: string,
  maxDecodedTexturePixels: number,
): number {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 header has invalid dimensions ${width}x${height}.`,
    );
  }
  const pixels = BigInt(width) * BigInt(height);
  if (pixels > BigInt(MAX_RGBA8_PIXELS)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 dimensions ${width}x${height} exceed the RGBA8 decoder address space.`,
    );
  }
  if (maxDecodedTexturePixels > 0 && pixels > BigInt(maxDecodedTexturePixels)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 dimensions ${width}x${height} ` +
        `exceed maxDecodedTexturePixels ${maxDecodedTexturePixels}.`,
    );
  }
  return Number(pixels);
}

function copiedU32(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
  label: string,
): Uint32Array {
  if (
    !Number.isSafeInteger(pointer) ||
    pointer <= 0 ||
    pointer + length * Uint32Array.BYTES_PER_ELEMENT > memory.buffer.byteLength
  ) {
    throw new Error(`Basis Universal WASM returned an invalid ${label} pointer.`);
  }
  return new Uint32Array(new Uint32Array(memory.buffer, pointer, length));
}

interface KhrTextureBasisuMetadata {
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly layerCount: number;
  readonly faceCount: number;
  readonly levelCount: number;
  readonly rgbaMapping: BasisRgbaMapping;
}

type BasisRgbaMapping = 'rgba' | 'rgb' | 'r' | 'rg' | 'etc1s-rg';

interface Ktx2LevelIndex {
  readonly index: number;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly uncompressedByteLength: number;
}

interface BasisDfdMetadata {
  readonly model: typeof KHR_DF_MODEL_ETC1S | typeof KHR_DF_MODEL_UASTC;
  readonly sampleCount: 1 | 2;
  readonly rgbaMapping: BasisRgbaMapping;
}

function checkedContainerRange(
  bytes: Uint8Array,
  offset: number,
  length: number,
  path: string,
  label: string,
): readonly [number, number] {
  const end = offset + length;
  if (
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(length) ||
    offset < 0 ||
    length < 0 ||
    !Number.isSafeInteger(end) ||
    end > bytes.byteLength
  ) {
    throw new Error(`[vitrum/gltf-adapter] ${path} KTX2 ${label} range is truncated or invalid.`);
  }
  return [offset, end];
}

function checkedUint64(view: DataView, offset: number, path: string, label: string): number {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 ${label} exceeds JavaScript's safe integer range.`,
    );
  }
  return Number(value);
}

function bytesEqual(bytes: Uint8Array, start: number, expected: readonly number[]): boolean {
  return expected.every((value, index) => bytes[start + index] === value);
}

function validateBasisDfdLayout(
  bytes: Uint8Array,
  view: DataView,
  descriptorOffset: number,
  descriptorBlockBytes: number,
  model: number,
  transfer: number,
  supercompression: number,
  path: string,
): BasisDfdMetadata {
  const sampleBytes = descriptorBlockBytes - KTX2_DFD_BASIC_HEADER_BYTES;
  if (sampleBytes < KTX2_DFD_SAMPLE_BYTES || sampleBytes % KTX2_DFD_SAMPLE_BYTES !== 0) {
    throw new Error(`[vitrum/gltf-adapter] ${path} KTX2 Basis DFD has an invalid sample table.`);
  }
  const count = sampleBytes / KTX2_DFD_SAMPLE_BYTES;
  if (count !== 1 && count !== 2) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 Basis DFD has unsupported sample count ${count}.`,
    );
  }
  if (!bytesEqual(bytes, descriptorOffset + 12, [3, 3, 0, 0])) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 Basis DFD must describe 4x4x1 texel blocks.`,
    );
  }

  const channels: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const sampleOffset =
      descriptorOffset + KTX2_DFD_BASIC_HEADER_BYTES + index * KTX2_DFD_SAMPLE_BYTES;
    const expectedBitOffset = model === KHR_DF_MODEL_ETC1S ? index * 64 : 0;
    const expectedBitLength = model === KHR_DF_MODEL_ETC1S ? 63 : 127;
    const channelType = bytes[sampleOffset + 3]!;
    const channel = channelType & 0x0f;
    const qualifiers = channelType & 0xf0;
    const linearAlphaQualifier =
      model === KHR_DF_MODEL_ETC1S &&
      index === 1 &&
      channel === KHR_DF_CHANNEL_ETC1S_AAA &&
      transfer === KHR_DF_TRANSFER_SRGB &&
      qualifiers === KHR_DF_SAMPLE_DATATYPE_LINEAR;
    if (
      view.getUint16(sampleOffset, true) !== expectedBitOffset ||
      bytes[sampleOffset + 2] !== expectedBitLength ||
      (qualifiers !== 0 && !linearAlphaQualifier) ||
      !bytesEqual(bytes, sampleOffset + 4, [0, 0, 0, 0]) ||
      view.getUint32(sampleOffset + 8, true) !== 0 ||
      view.getUint32(sampleOffset + 12, true) !== 0xffff_ffff
    ) {
      throw new Error(`[vitrum/gltf-adapter] ${path} KTX2 Basis DFD sample ${index} is malformed.`);
    }
    channels.push(channel);
  }

  let rgbaMapping: BasisRgbaMapping | null = null;
  if (model === KHR_DF_MODEL_ETC1S) {
    if (count === 1 && channels[0] === KHR_DF_CHANNEL_ETC1S_RGB) {
      rgbaMapping = 'rgb';
    } else if (count === 1 && channels[0] === KHR_DF_CHANNEL_ETC1S_RRR) {
      rgbaMapping = 'r';
    } else if (
      count === 2 &&
      channels[0] === KHR_DF_CHANNEL_ETC1S_RGB &&
      channels[1] === KHR_DF_CHANNEL_ETC1S_AAA
    ) {
      rgbaMapping = 'rgba';
    } else if (
      count === 2 &&
      channels[0] === KHR_DF_CHANNEL_ETC1S_RRR &&
      channels[1] === KHR_DF_CHANNEL_ETC1S_GGG
    ) {
      rgbaMapping = 'etc1s-rg';
    }
  } else if (model === KHR_DF_MODEL_UASTC && count === 1) {
    if (channels[0] === KHR_DF_CHANNEL_UASTC_RGB) {
      rgbaMapping = 'rgb';
    } else if (channels[0] === KHR_DF_CHANNEL_UASTC_RGBA) {
      rgbaMapping = 'rgba';
    } else if (channels[0] === KHR_DF_CHANNEL_UASTC_RRR) {
      rgbaMapping = 'r';
    } else if (channels[0] === KHR_DF_CHANNEL_UASTC_RG) {
      rgbaMapping = 'rg';
    }
  }
  if (rgbaMapping === null) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 DFD is not a KHR_texture_basisu ETC1S/UASTC channel layout.`,
    );
  }
  if (transfer === KHR_DF_TRANSFER_SRGB && rgbaMapping !== 'rgb' && rgbaMapping !== 'rgba') {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 one/two-component Basis textures must use a linear transfer function.`,
    );
  }

  const planes = Array.from(bytes.subarray(descriptorOffset + 16, descriptorOffset + 24));
  const canonicalPlanes =
    model === KHR_DF_MODEL_UASTC
      ? [16, 0, 0, 0, 0, 0, 0, 0]
      : count === 1
        ? [8, 0, 0, 0, 0, 0, 0, 0]
        : [8, 8, 0, 0, 0, 0, 0, 0];
  const allZeroPlanes = planes.every((value) => value === 0);
  if (
    !planes.every((value, index) => value === canonicalPlanes[index]) &&
    !(supercompression !== KTX2_SUPERCOMPRESSION_NONE && allZeroPlanes)
  ) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 Basis DFD has invalid bytesPlane metadata.`,
    );
  }

  return {
    model: model as BasisDfdMetadata['model'],
    sampleCount: count,
    rgbaMapping,
  };
}

function asciiValue(bytes: Uint8Array, start: number, end: number): string {
  let valueEnd = end;
  while (valueEnd > start && bytes[valueEnd - 1] === 0) valueEnd -= 1;
  let value = '';
  for (let index = start; index < valueEnd; index += 1) {
    const byte = bytes[index]!;
    if (byte < 0x20 || byte > 0x7e) return '';
    value += String.fromCharCode(byte);
  }
  return value;
}

function validateKhrTextureBasisuKeyValues(bytes: Uint8Array, view: DataView, path: string): void {
  const kvdOffset = view.getUint32(56, true);
  const kvdLength = view.getUint32(60, true);
  if (kvdLength === 0) {
    if (kvdOffset !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 key/value offset must be zero when the section is empty.`,
      );
    }
    return;
  }
  const [, kvdEnd] = checkedContainerRange(bytes, kvdOffset, kvdLength, path, 'key/value data');
  let cursor = kvdOffset;
  const seen = new Set<string>();
  while (cursor < kvdEnd) {
    if (cursor + Uint32Array.BYTES_PER_ELEMENT > kvdEnd) {
      throw new Error(`[vitrum/gltf-adapter] ${path} KTX2 key/value entry header is truncated.`);
    }
    const entryLength = view.getUint32(cursor, true);
    const entryStart = cursor + Uint32Array.BYTES_PER_ELEMENT;
    const entryEnd = entryStart + entryLength;
    if (entryLength === 0 || !Number.isSafeInteger(entryEnd) || entryEnd > kvdEnd) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 key/value entry is truncated or invalid.`,
      );
    }
    let separator = entryStart;
    while (separator < entryEnd && bytes[separator] !== 0) separator += 1;
    if (separator === entryStart || separator >= entryEnd) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 key/value entry has no valid key terminator.`,
      );
    }
    const key = asciiValue(bytes, entryStart, separator);
    const value = asciiValue(bytes, separator + 1, entryEnd);
    if ((key === 'KTXorientation' || key === 'KTXswizzle') && seen.has(key)) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 metadata contains duplicate ${key} entries.`,
      );
    }
    seen.add(key);
    if (key === 'KTXorientation' && value !== 'rd') {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 orientation must be "rd" for KHR_texture_basisu; received "${value}".`,
      );
    }
    if (key === 'KTXswizzle' && value !== 'rgba') {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 swizzle must be "rgba" for KHR_texture_basisu; received "${value}".`,
      );
    }
    const paddedLength = (entryLength + 3) & ~3;
    const next = entryStart + paddedLength;
    if (!Number.isSafeInteger(next) || next <= cursor || next > kvdEnd) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 key/value padding is truncated or invalid.`,
      );
    }
    for (let padding = entryEnd; padding < next; padding += 1) {
      if (bytes[padding] !== 0) {
        throw new Error(`[vitrum/gltf-adapter] ${path} KTX2 key/value padding bytes must be zero.`);
      }
    }
    cursor = next;
  }
}

function alignUp(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function validateZeroPadding(
  bytes: Uint8Array,
  start: number,
  end: number,
  path: string,
  label: string,
): void {
  for (let offset = start; offset < end; offset += 1) {
    if (bytes[offset] !== 0) {
      throw new Error(`[vitrum/gltf-adapter] ${path} KTX2 ${label} padding bytes must be zero.`);
    }
  }
}

function validateBasisLzGlobalData(
  bytes: Uint8Array,
  view: DataView,
  sgdOffset: number,
  sgdLength: number,
  levels: readonly Ktx2LevelIndex[],
  dfd: BasisDfdMetadata,
  path: string,
): void {
  if (sgdLength < KTX2_BASIS_LZ_SGD_HEADER_BYTES + levels.length * KTX2_BASIS_LZ_IMAGE_DESC_BYTES) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ global data is too short for its image descriptors.`,
    );
  }
  const endpointCount = view.getUint16(sgdOffset, true);
  const selectorCount = view.getUint16(sgdOffset + 2, true);
  const endpointsByteLength = view.getUint32(sgdOffset + 4, true);
  const selectorsByteLength = view.getUint32(sgdOffset + 8, true);
  const tablesByteLength = view.getUint32(sgdOffset + 12, true);
  const extendedByteLength = view.getUint32(sgdOffset + 16, true);
  const expectedLength =
    BigInt(KTX2_BASIS_LZ_SGD_HEADER_BYTES) +
    BigInt(levels.length * KTX2_BASIS_LZ_IMAGE_DESC_BYTES) +
    BigInt(endpointsByteLength) +
    BigInt(selectorsByteLength) +
    BigInt(tablesByteLength) +
    BigInt(extendedByteLength);
  if (
    endpointCount === 0 ||
    selectorCount === 0 ||
    endpointsByteLength === 0 ||
    selectorsByteLength === 0 ||
    tablesByteLength === 0 ||
    extendedByteLength !== 0 ||
    expectedLength !== BigInt(sgdLength)
  ) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ global data header/length is malformed.`,
    );
  }

  for (const level of levels) {
    const descriptor =
      sgdOffset + KTX2_BASIS_LZ_SGD_HEADER_BYTES + level.index * KTX2_BASIS_LZ_IMAGE_DESC_BYTES;
    const imageFlags = view.getUint32(descriptor, true);
    const rgbOffset = view.getUint32(descriptor + 4, true);
    const rgbLength = view.getUint32(descriptor + 8, true);
    const alphaOffset = view.getUint32(descriptor + 12, true);
    const alphaLength = view.getUint32(descriptor + 16, true);
    const rgbEnd = BigInt(rgbOffset) + BigInt(rgbLength);
    const alphaEnd = BigInt(alphaOffset) + BigInt(alphaLength);
    if (imageFlags !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ level ${level.index} uses unsupported image flags.`,
      );
    }
    if (rgbLength === 0 || rgbEnd > BigInt(level.byteLength)) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ level ${level.index} RGB slice is malformed.`,
      );
    }
    if (
      dfd.sampleCount === 1
        ? alphaOffset !== 0 || alphaLength !== 0
        : alphaLength === 0 || alphaEnd > BigInt(level.byteLength)
    ) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ level ${level.index} alpha slice disagrees with its DFD.`,
      );
    }
  }
}

function expectedUastcLevelBytes(width: number, height: number, level: number): bigint {
  const levelWidth = Math.max(1, Math.floor(width / 2 ** level));
  const levelHeight = Math.max(1, Math.floor(height / 2 ** level));
  return BigInt(Math.ceil(levelWidth / 4)) * BigInt(Math.ceil(levelHeight / 4)) * 16n;
}

function validateKhrTextureBasisuContainer(
  bytes: Uint8Array,
  path: string,
  colorSpace: 'srgb' | 'linear',
): KhrTextureBasisuMetadata {
  if (bytes.byteLength < KTX2_HEADER_BYTES) {
    throw new Error(`[vitrum/gltf-adapter] ${path} KTX2 header is truncated.`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vkFormat = view.getUint32(12, true);
  const typeSize = view.getUint32(16, true);
  const width = view.getUint32(20, true);
  const height = view.getUint32(24, true);
  const depth = view.getUint32(28, true);
  const layerCount = view.getUint32(32, true);
  const faceCount = view.getUint32(36, true);
  const levelCount = view.getUint32(40, true);
  const supercompression = view.getUint32(44, true);

  if (vkFormat !== KTX2_VK_FORMAT_UNDEFINED || typeSize !== 1) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 is not a Basis Universal payload ` +
        `(vkFormat=${vkFormat}, typeSize=${typeSize}).`,
    );
  }
  if (depth !== 0 || layerCount !== 0 || faceCount !== 1 || levelCount < 1) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KHR_texture_basisu material images require ` +
        `depth 0, layerCount 0, faceCount 1, and at least one level ` +
        `(depth=${depth}, layerCount=${layerCount}, faceCount=${faceCount}, levels=${levelCount}).`,
    );
  }
  if (width === 0 || height === 0 || width % 4 !== 0 || height % 4 !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KHR_texture_basisu dimensions must be positive multiples of 4; ` +
        `received ${width}x${height}.`,
    );
  }
  const maximumMipCount = Math.floor(Math.log2(Math.max(width, height))) + 1;
  if (levelCount > maximumMipCount) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 levelCount ${levelCount} exceeds the ` +
        `${maximumMipCount}-level pyramid possible for ${width}x${height}.`,
    );
  }
  const [, levelIndexEnd] = checkedContainerRange(
    bytes,
    KTX2_HEADER_BYTES,
    levelCount * KTX2_LEVEL_INDEX_ENTRY_BYTES,
    path,
    'level index',
  );
  const levels: Ktx2LevelIndex[] = Array.from({ length: levelCount }, (_unused, index) => {
    const entry = KTX2_HEADER_BYTES + index * KTX2_LEVEL_INDEX_ENTRY_BYTES;
    return {
      index,
      byteOffset: checkedUint64(view, entry, path, `level ${index} byteOffset`),
      byteLength: checkedUint64(view, entry + 8, path, `level ${index} byteLength`),
      uncompressedByteLength: checkedUint64(
        view,
        entry + 16,
        path,
        `level ${index} uncompressedByteLength`,
      ),
    };
  });

  const dfdOffset = view.getUint32(48, true);
  const dfdLength = view.getUint32(52, true);
  const [dfdStart, dfdEnd] = checkedContainerRange(
    bytes,
    dfdOffset,
    dfdLength,
    path,
    'data-format descriptor',
  );
  if (dfdStart !== levelIndexEnd) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 data-format descriptor must immediately follow the level index.`,
    );
  }
  if (dfdLength < KTX2_DFD_MIN_BYTES) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 data-format descriptor is too short for a Basis payload.`,
    );
  }
  const declaredDfdBytes = view.getUint32(dfdStart, true);
  const descriptorOffset = dfdStart + Uint32Array.BYTES_PER_ELEMENT;
  const descriptorWord0 = view.getUint32(descriptorOffset, true);
  const descriptorVersion = view.getUint16(descriptorOffset + 4, true);
  const descriptorBlockBytes = view.getUint16(descriptorOffset + 6, true);
  if (descriptorVersion !== KHR_DF_BASIC_VERSION) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 Basis data-format descriptor must use basic version 2.`,
    );
  }
  if (
    declaredDfdBytes !== dfdLength ||
    descriptorWord0 !== 0 ||
    descriptorBlockBytes < KTX2_DFD_BASIC_HEADER_BYTES + KTX2_DFD_SAMPLE_BYTES ||
    descriptorBlockBytes % Uint32Array.BYTES_PER_ELEMENT !== 0 ||
    descriptorOffset + descriptorBlockBytes !== dfdEnd
  ) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 Basis data-format descriptor is malformed.`,
    );
  }

  const model = bytes[descriptorOffset + 8]!;
  const primaries = bytes[descriptorOffset + 9]!;
  const transfer = bytes[descriptorOffset + 10]!;
  const flags = bytes[descriptorOffset + 11]!;
  const isEtc1s = model === KHR_DF_MODEL_ETC1S;
  const isUastc = model === KHR_DF_MODEL_UASTC;
  if (!isEtc1s && !isUastc) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 DFD is not a KHR_texture_basisu ETC1S/UASTC channel layout.`,
    );
  }
  if (
    (isEtc1s && supercompression !== KTX2_SUPERCOMPRESSION_BASIS_LZ) ||
    (isUastc &&
      supercompression !== KTX2_SUPERCOMPRESSION_NONE &&
      supercompression !== KTX2_SUPERCOMPRESSION_ZSTD)
  ) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 supercompression ${supercompression} does not match its Basis DFD model.`,
    );
  }
  if (flags !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KHR_texture_basisu material images must use zero DFD flags ` +
        `(premultiplied/reserved flags are unsupported).`,
    );
  }
  const expectedPrimaries =
    colorSpace === 'srgb' ? KHR_DF_PRIMARIES_BT709 : KHR_DF_PRIMARIES_UNSPECIFIED;
  const expectedTransfer = colorSpace === 'srgb' ? KHR_DF_TRANSFER_SRGB : KHR_DF_TRANSFER_LINEAR;
  if (primaries !== expectedPrimaries || transfer !== expectedTransfer) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 DFD must declare ` +
        `${colorSpace === 'srgb' ? 'BT.709 primaries with the sRGB transfer function' : 'unspecified primaries with a linear transfer function'} ` +
        `for this material role; received primaries=${primaries}, transfer=${transfer}.`,
    );
  }
  const dfd = validateBasisDfdLayout(
    bytes,
    view,
    descriptorOffset,
    descriptorBlockBytes,
    model,
    transfer,
    supercompression,
    path,
  );

  const kvdOffset = view.getUint32(56, true);
  const kvdLength = view.getUint32(60, true);
  let metadataEnd = dfdEnd;
  if (kvdLength === 0) {
    if (kvdOffset !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 key/value offset must be zero when the section is empty.`,
      );
    }
  } else {
    if (kvdOffset !== dfdEnd) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 key/value data must immediately follow its DFD.`,
      );
    }
    const [, kvdEnd] = checkedContainerRange(bytes, kvdOffset, kvdLength, path, 'key/value data');
    metadataEnd = kvdEnd;
  }
  validateKhrTextureBasisuKeyValues(bytes, view, path);

  const sgdOffset = checkedUint64(view, 64, path, 'supercompression global-data offset');
  const sgdLength = checkedUint64(view, 72, path, 'supercompression global-data length');
  if (isEtc1s) {
    if (sgdLength === 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ payload requires non-empty supercompression global data.`,
      );
    }
    const expectedSgdOffset = alignUp(metadataEnd, 8);
    if (sgdOffset !== expectedSgdOffset || sgdOffset % 8 !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ global data must start on the next 8-byte metadata boundary.`,
      );
    }
    validateZeroPadding(bytes, metadataEnd, sgdOffset, path, 'global-data');
    const [, sgdEnd] = checkedContainerRange(
      bytes,
      sgdOffset,
      sgdLength,
      path,
      'supercompression global data',
    );
    metadataEnd = sgdEnd;
  } else if (sgdOffset !== 0 || sgdLength !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} KTX2 UASTC payload must not declare supercompression global data.`,
    );
  }

  for (const level of levels) {
    if (level.byteLength === 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 level ${level.index} has an empty encoded payload.`,
      );
    }
    checkedContainerRange(
      bytes,
      level.byteOffset,
      level.byteLength,
      path,
      `level ${level.index} payload`,
    );
    if (isEtc1s) {
      if (level.uncompressedByteLength !== 0) {
        throw new Error(
          `[vitrum/gltf-adapter] ${path} KTX2 BasisLZ level ${level.index} must use uncompressedByteLength 0.`,
        );
      }
      continue;
    }
    const expectedBytes = expectedUastcLevelBytes(width, height, level.index);
    if (
      BigInt(level.uncompressedByteLength) !== expectedBytes ||
      (supercompression === KTX2_SUPERCOMPRESSION_NONE &&
        BigInt(level.byteLength) !== expectedBytes)
    ) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 UASTC level ${level.index} byte lengths ` +
          `do not match its ${expectedBytes.toString()}-byte block payload.`,
      );
    }
  }

  const levelAlignment = supercompression === KTX2_SUPERCOMPRESSION_NONE ? 16 : 1;
  let expectedLevelOffset = alignUp(metadataEnd, levelAlignment);
  validateZeroPadding(bytes, metadataEnd, expectedLevelOffset, path, 'mip-level');
  for (let index = levels.length - 1; index >= 0; index -= 1) {
    const level = levels[index]!;
    if (level.byteOffset !== expectedLevelOffset) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 mip levels are out of physical order, overlap, or contain invalid padding at level ${index}.`,
      );
    }
    const levelEnd = level.byteOffset + level.byteLength;
    expectedLevelOffset = alignUp(levelEnd, levelAlignment);
    if (index > 0) {
      validateZeroPadding(bytes, levelEnd, expectedLevelOffset, path, 'mip-level');
    } else if (levelEnd !== bytes.byteLength) {
      throw new Error(
        `[vitrum/gltf-adapter] ${path} KTX2 base level must end at the end of the container.`,
      );
    }
  }

  if (isEtc1s) {
    validateBasisLzGlobalData(bytes, view, sgdOffset, sgdLength, levels, dfd, path);
  }
  return {
    width,
    height,
    depth,
    layerCount,
    faceCount,
    levelCount,
    rgbaMapping: dfd.rgbaMapping,
  };
}

function applyBasisRgbaMapping(data: Uint8Array, mapping: BasisRgbaMapping): void {
  if (mapping === 'rgba') return;
  for (let offset = 0; offset < data.length; offset += 4) {
    if (mapping === 'etc1s-rg') {
      data[offset + 1] = data[offset + 3]!;
      data[offset + 2] = 0;
    } else if (mapping === 'r') {
      data[offset + 1] = 0;
      data[offset + 2] = 0;
    } else if (mapping === 'rg') {
      data[offset + 2] = 0;
    }
    data[offset + 3] = 255;
  }
}

export const decodeRawBasisKtx2Pixels: DecodeGltfTexturePixelsFn = async (
  handle,
  context,
): Promise<GltfDecodedTexturePixels> => {
  const source = localUint8ArrayView(handle.data);
  if (source === null || !isBasisKtx2Bytes(source)) {
    throw new Error(`[vitrum/gltf-adapter] ${context.path} is not a valid KTX2 byte stream.`);
  }
  if (source.byteLength > MAX_WASM32_ALLOCATION_BYTES) {
    throw new Error(
      `[vitrum/gltf-adapter] ${context.path} KTX2 payload exceeds the 32-bit Basis WASM address space.`,
    );
  }

  // Perform all cheap structural and output-budget checks before duplicating
  // the encoded payload or initializing native state.
  const container = validateKhrTextureBasisuContainer(source, context.path, context.colorSpace);
  const pixelCount = checkedPixelCount(
    container.width,
    container.height,
    context.path,
    context.maxDecodedTexturePixels,
  );
  // Own the input bytes for the complete synchronous native call sequence.
  const input = new Uint8Array(source);
  const basis = await basisModule();
  let transcoder = 0;
  let inputPointer = 0;
  let levelInfoPointer = 0;
  let outputPointer = 0;

  try {
    transcoder = basis.createKtx2Transcoder();
    if (transcoder === 0) {
      throw new Error('Basis Universal could not allocate a KTX2 transcoder.');
    }

    inputPointer = basis.malloc(input.byteLength);
    if (inputPointer === 0 && input.byteLength > 0) {
      throw new Error('Basis Universal could not allocate the KTX2 input buffer.');
    }
    new Uint8Array(basis.memory.buffer, inputPointer, input.byteLength).set(input);

    if (basis.initializeKtx2Transcoder(transcoder, inputPointer, input.byteLength) === 0) {
      throw new Error('Basis Universal rejected the KTX2 container.');
    }

    const header = copiedU32(
      basis.memory,
      basis.getKtx2Header(transcoder),
      KTX2_HEADER_U32_LENGTH,
      'KTX2 header',
    );
    const width = header[5]!;
    const height = header[6]!;
    const depth = header[7]!;
    const layers = header[8]!;
    const faces = header[9]!;
    const levels = header[10]!;
    if (
      width !== container.width ||
      height !== container.height ||
      depth !== container.depth ||
      layers !== container.layerCount ||
      faces !== container.faceCount ||
      levels !== container.levelCount
    ) {
      throw new Error(
        `[vitrum/gltf-adapter] ${context.path} KTX2 parser and Basis Universal header metadata disagree.`,
      );
    }

    if (basis.startKtx2Transcoding(transcoder) === 0) {
      throw new Error('Basis Universal could not start KTX2 transcoding.');
    }

    levelInfoPointer = basis.malloc(KTX2_LEVEL_INFO_BYTES);
    if (levelInfoPointer === 0) {
      throw new Error('Basis Universal could not allocate level metadata.');
    }
    if (basis.getKtx2LevelInfo(transcoder, levelInfoPointer, 0, 0, 0) === 0) {
      throw new Error('Basis Universal could not read KTX2 mip level 0.');
    }
    const levelInfo = copiedU32(
      basis.memory,
      levelInfoPointer,
      KTX2_LEVEL_INFO_U32_LENGTH,
      'KTX2 level-info',
    );
    if (levelInfo[3] !== width || levelInfo[4] !== height) {
      throw new Error('Basis Universal KTX2 header/level dimensions disagree.');
    }

    const expectedBytes = pixelCount * 4;
    const outputBytes = basis.computeTranscodedSize(BASIS_RGBA32_FORMAT, width, height);
    if (outputBytes !== expectedBytes) {
      throw new Error(
        `Basis Universal reported ${outputBytes} RGBA8 bytes for ${width}x${height}; expected ${expectedBytes}.`,
      );
    }
    outputPointer = basis.malloc(outputBytes);
    if (outputPointer === 0 && outputBytes > 0) {
      throw new Error('Basis Universal could not allocate the RGBA8 output buffer.');
    }

    const succeeded = basis.transcodeKtx2Level(
      transcoder,
      0,
      0,
      0,
      outputPointer,
      pixelCount,
      BASIS_RGBA32_FORMAT,
      0,
      width,
      height,
      -1,
      -1,
      0,
    );
    if (succeeded === 0) {
      throw new Error('Basis Universal failed to transcode KTX2 mip level 0.');
    }

    const data = new Uint8Array(new Uint8Array(basis.memory.buffer, outputPointer, outputBytes));
    applyBasisRgbaMapping(data, container.rgbaMapping);
    return {
      width,
      height,
      data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } finally {
    // The native transcoder retains the encoded input pointer after init().
    // Destroy it while that backing allocation is still live, matching the
    // pinned wrapper's dispose order, then release the adapter-owned buffers.
    if (transcoder !== 0) basis.deleteKtx2Transcoder(transcoder);
    if (outputPointer !== 0) basis.free(outputPointer);
    if (levelInfoPointer !== 0) basis.free(levelInfoPointer);
    if (inputPointer !== 0) basis.free(inputPointer);
  }
};
