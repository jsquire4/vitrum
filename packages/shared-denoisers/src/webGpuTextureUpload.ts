/**
 * webGpuTextureUpload.ts — Shared WebGPU texture upload / fill / readback helpers.
 *
 * Consolidates row-padded staging buffer construction so every one-shot denoiser
 * (atrousVarianceWebGPU, svgfRealWebGPU, hdrLuminanceBilateralWebGPU) funnels
 * through the same primitive. WebGPU `writeTexture` requires `bytesPerRow` to
 * be a multiple of 256; this module owns that alignment math.
 *
 * Previously: each one-shot denoiser inlined its own copy of these helpers.
 * Extracted by sweep H8 / W4-A7 refactor.
 */

import { float32ToFloat16Bits, float16BitsToFloat32 } from './halfFloat.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';

// Bytes-per-pixel constants for the formats we upload to — all file-local.
const RGBA32F_BPP = 16 as const;
const RGBA16F_BPP = 8  as const;
const RG32F_BPP   = 8  as const;
const R32F_BPP    = 4  as const;
const R32U_BPP    = 4  as const;
const R16U_BPP    = 2  as const;

/**
 * Generic stride-aware texture upload helper.
 *
 * Allocates a row-padded staging buffer (driver requires 256-byte aligned
 * bytesPerRow on writeTexture) using the supplied typed-array ctor, lets
 * the caller fill it, then forwards to writeTexture.
 *
 * The `fill` callback receives the destination row stride in *elements*
 * (not bytes), so it can compute per-row offsets without re-deriving the
 * alignment math at every call site.
 *
 * For byte-level uploads (e.g. f16 packed via DataView), pass Uint8Array as
 * the ctor and bytesPerElement=1; the rowStride argument is then in bytes.
 */
function uploadTexture2D<T extends Float32Array | Uint8Array | Uint32Array>(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  bpp: number,
  TypedArrayCtor: new (lengthInElements: number) => T,
  bytesPerElement: number,
  fill: (buf: T, rowStrideElements: number) => void,
): void {
  const bpr = alignedTextureCopyBytesPerRow(width, bpp);
  const rowStrideElements = bpr / bytesPerElement;
  const upload = new TypedArrayCtor(rowStrideElements * height);
  fill(upload, rowStrideElements);
  device.queue.writeTexture(
    { texture },
    upload.buffer,
    { bytesPerRow: bpr, rowsPerImage: height },
    [width, height],
  );
}

/** Tight RGB (length w*h*3) → rgba16float texture; alpha forced to 1. */
export function uploadRgbAsRgba16f(
  device: GPUDevice,
  texture: GPUTexture,
  rgb: Float32Array,
  width: number,
  height: number,
): void {
  uploadTexture2D(device, texture, width, height, RGBA16F_BPP, Uint8Array, 1, (buf, rowStrideBytes) => {
    const dv = new DataView(buf.buffer);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const si = (y * width + x) * 3;
        const byte = y * rowStrideBytes + x * 8;
        dv.setUint16(byte + 0, float32ToFloat16Bits(rgb[si]     ?? 0), true);
        dv.setUint16(byte + 2, float32ToFloat16Bits(rgb[si + 1] ?? 0), true);
        dv.setUint16(byte + 4, float32ToFloat16Bits(rgb[si + 2] ?? 0), true);
        dv.setUint16(byte + 6, float32ToFloat16Bits(1),                true);
      }
    }
  });
}

/**
 * Tight RGB (length w*h*3) → rgba32float texture with configurable per-channel
 * fallback values and alpha fill.
 *
 * Unlike `uploadRgbAsRgba32f` (which hard-codes fallback=0 and alpha=1), this
 * variant lets the caller supply a `fallbackRgba` tuple that is used:
 *   • per R/G/B channel when the source array has no value at that index
 *     (the `?? fallback[ch]` guard, matching the SVGF normal-texture upload
 *     convention where the packed-normal rest pose is (0.5, 0.5, 1.0, 0.0))
 *   • as the constant alpha written to every texel (`.w` is always `fallbackRgba[3]`
 *     regardless of source length)
 *
 * Default rest pose: `[0.5, 0.5, 1.0, 0.0]` (packed octahedral +Z normal,
 * alpha=0 — the SVGF gbuffer normal texture convention).
 */
export function uploadRgbAsRgba32fPacked(
  device: GPUDevice,
  texture: GPUTexture,
  rgb: Float32Array,
  width: number,
  height: number,
  fallbackRgba: readonly [number, number, number, number] = [0.5, 0.5, 1.0, 0.0],
): void {
  uploadTexture2D(device, texture, width, height, RGBA32F_BPP, Float32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      const row = y * rowStride;
      for (let x = 0; x < width; x += 1) {
        const si = (y * width + x) * 3;
        const o  = row + x * 4;
        buf[o]     = rgb[si]     ?? fallbackRgba[0];
        buf[o + 1] = rgb[si + 1] ?? fallbackRgba[1];
        buf[o + 2] = rgb[si + 2] ?? fallbackRgba[2];
        buf[o + 3] = fallbackRgba[3];
      }
    }
  });
}

/** Tight RGB (length w*h*3) → rgba32float texture; alpha forced to 1. */
export function uploadRgbAsRgba32f(
  device: GPUDevice,
  texture: GPUTexture,
  rgb: Float32Array,
  width: number,
  height: number,
): void {
  uploadTexture2D(device, texture, width, height, RGBA32F_BPP, Float32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      const row = y * rowStride;
      for (let x = 0; x < width; x += 1) {
        const si = (y * width + x) * 3;
        const o  = row + x * 4;
        buf[o]     = rgb[si]     ?? 0;
        buf[o + 1] = rgb[si + 1] ?? 0;
        buf[o + 2] = rgb[si + 2] ?? 0;
        buf[o + 3] = 1;
      }
    }
  });
}

/** Linear depth → rgba32float texel `.r` (matches SVGF gbufferDepth sampling). */
export function uploadLinearDepthAsRgba32f(
  device: GPUDevice,
  texture: GPUTexture,
  depth: Float32Array,
  width: number,
  height: number,
): void {
  uploadTexture2D(device, texture, width, height, RGBA32F_BPP, Float32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      const row = y * rowStride;
      for (let x = 0; x < width; x += 1) {
        const si = y * width + x;
        const o  = row + x * 4;
        buf[o]     = depth[si] ?? 0;
        buf[o + 1] = 0;
        buf[o + 2] = 0;
        buf[o + 3] = 0;
      }
    }
  });
}

/** Tight interleaved RG floats per pixel (length w*h*2) → rg32float texture. */
export function uploadInterleavedRgAsRg32f(
  device: GPUDevice,
  texture: GPUTexture,
  rg: Float32Array,
  width: number,
  height: number,
): void {
  uploadTexture2D(device, texture, width, height, RG32F_BPP, Float32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      const row = y * rowStride;
      for (let x = 0; x < width; x += 1) {
        const si = (y * width + x) * 2;
        const o  = row + x * 2;
        buf[o]     = rg[si]     ?? 0;
        buf[o + 1] = rg[si + 1] ?? 0;
      }
    }
  });
}

/** Tight scalar R floats per pixel (length w*h) → r32float texture. */
export function uploadR32f(
  device: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  width: number,
  height: number,
): void {
  uploadTexture2D(device, texture, width, height, R32F_BPP, Float32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        buf[y * rowStride + x] = data[y * width + x] ?? 0;
      }
    }
  });
}

/** Tight scalar R u32 per pixel (length w*h) → r32uint texture. */
export function uploadR32Uint(
  device: GPUDevice,
  texture: GPUTexture,
  data: Uint32Array,
  width: number,
  height: number,
): void {
  uploadTexture2D(device, texture, width, height, R32U_BPP, Uint32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        buf[y * rowStride + x] = data[y * width + x] ?? 0;
      }
    }
  });
}

/** Tight scalar R u16 per pixel (length w*h) → r16uint texture. */
export function uploadR16Uint(
  device: GPUDevice,
  texture: GPUTexture,
  data: Uint32Array,
  width: number,
  height: number,
): void {
  uploadTexture2D(device, texture, width, height, R16U_BPP, Uint8Array, 1, (buf, rowStrideBytes) => {
    const dv = new DataView(buf.buffer);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        dv.setUint16(y * rowStrideBytes + x * 2, (data[y * width + x] ?? 0) & 0xFFFF, true);
      }
    }
  });
}

/** Fill an rgba32float texture with a single constant texel. */
export function fillRgba32f(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  rgba: readonly [number, number, number, number],
): void {
  uploadTexture2D(device, texture, width, height, RGBA32F_BPP, Float32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      const row = y * rowStride;
      for (let x = 0; x < width; x += 1) {
        const o = row + x * 4;
        buf[o]     = rgba[0]!;
        buf[o + 1] = rgba[1]!;
        buf[o + 2] = rgba[2]!;
        buf[o + 3] = rgba[3]!;
      }
    }
  });
}

/** Fill an rg32float texture with a single constant texel. */
export function fillRg32f(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  r: number,
  g: number,
): void {
  uploadTexture2D(device, texture, width, height, RG32F_BPP, Float32Array, 4, (buf, rowStride) => {
    for (let y = 0; y < height; y += 1) {
      const row = y * rowStride;
      for (let x = 0; x < width; x += 1) {
        const o = row + x * 2;
        buf[o]     = r;
        buf[o + 1] = g;
      }
    }
  });
}

/** Fill an r16uint texture with a single constant value (low 16 bits used). */
export function fillR16Uint(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  value: number,
): void {
  uploadTexture2D(device, texture, width, height, R16U_BPP, Uint8Array, 1, (buf, rowStrideBytes) => {
    const dv = new DataView(buf.buffer);
    const v  = value & 0xFFFF;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        dv.setUint16(y * rowStrideBytes + x * 2, v, true);
      }
    }
  });
}

/**
 * Read an rgba32float texture back to tight RGB (length w*h*3, alpha discarded).
 * Mirrors `readRgba16fToRgb` but for 32-bit float textures (4 floats per texel,
 * 16 bytes/pixel). Submits its own copy command + awaits map.
 */
export async function readRgba32fToRgb(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
): Promise<Float32Array> {
  const bpr = alignedTextureCopyBytesPerRow(width, RGBA32F_BPP);
  const buf = device.createBuffer({
    size: bpr * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [width, height]);
  device.queue.submit([encoder.finish()]);
  try {
    await buf.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(buf.getMappedRange());
    const out = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y += 1) {
      const rowOff = (y * bpr) / 4;
      for (let x = 0; x < width; x += 1) {
        const di = (y * width + x) * 3;
        const si = rowOff + x * 4;
        out[di]     = mapped[si]     ?? 0;
        out[di + 1] = mapped[si + 1] ?? 0;
        out[di + 2] = mapped[si + 2] ?? 0;
      }
    }
    buf.unmap();
    return out;
  } finally {
    buf.destroy();
  }
}

/**
 * Read an rgba32float texture back to tight interleaved RG (length w*h*2;
 * B/A discarded). Used by the one-shot SVGF chaining path to recover the
 * blended moments (M1, M2) from the rgba32float moments-out texture so the
 * caller can feed them back as `momentsIn` (rg32float) next frame.
 * Submits its own copy command + awaits map; wraps map/read in try/finally so
 * the staging buffer is freed even if mapAsync rejects.
 */
export async function readRgba32fToRg(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
): Promise<Float32Array> {
  const bpr = alignedTextureCopyBytesPerRow(width, RGBA32F_BPP);
  const buf = device.createBuffer({
    size: bpr * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [width, height]);
  device.queue.submit([encoder.finish()]);
  try {
    await buf.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(buf.getMappedRange());
    const out = new Float32Array(width * height * 2);
    for (let y = 0; y < height; y += 1) {
      const rowOff = (y * bpr) / 4;
      for (let x = 0; x < width; x += 1) {
        const di = (y * width + x) * 2;
        const si = rowOff + x * 4;
        out[di]     = mapped[si]     ?? 0;
        out[di + 1] = mapped[si + 1] ?? 0;
      }
    }
    buf.unmap();
    return out;
  } finally {
    buf.destroy();
  }
}

/**
 * Read an r32uint texture back to a tight scalar u32 buffer (length w*h).
 * Used by the one-shot SVGF chaining path to recover the per-pixel history
 * length so the caller can feed it back as `historyLengthIn` next frame.
 * Submits its own copy command + awaits map; wraps map/read in try/finally.
 */
export async function readR32UintToU32(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
): Promise<Uint32Array> {
  const bpr = alignedTextureCopyBytesPerRow(width, R32U_BPP);
  const buf = device.createBuffer({
    size: bpr * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [width, height]);
  device.queue.submit([encoder.finish()]);
  try {
    await buf.mapAsync(GPUMapMode.READ);
    const mapped = new Uint32Array(buf.getMappedRange());
    const out = new Uint32Array(width * height);
    const rowStride = bpr / 4;
    for (let y = 0; y < height; y += 1) {
      const rowOff = y * rowStride;
      for (let x = 0; x < width; x += 1) {
        out[y * width + x] = mapped[rowOff + x] ?? 0;
      }
    }
    buf.unmap();
    return out;
  } finally {
    buf.destroy();
  }
}

/**
 * Read an rgba16float texture back to tight RGB (length w*h*3, alpha discarded).
 * Submits its own copy command + awaits map; caller need not provide an encoder.
 */
export async function readRgba16fToRgb(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
): Promise<Float32Array> {
  const bpr = alignedTextureCopyBytesPerRow(width, RGBA16F_BPP);
  const buf = device.createBuffer({
    size: bpr * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [width, height]);
  device.queue.submit([encoder.finish()]);
  try {
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(buf.getMappedRange());
    const dv  = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const out = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const byte = y * bpr + x * 8;
        const di   = (y * width + x) * 3;
        out[di]     = float16BitsToFloat32(dv.getUint16(byte + 0, true));
        out[di + 1] = float16BitsToFloat32(dv.getUint16(byte + 2, true));
        out[di + 2] = float16BitsToFloat32(dv.getUint16(byte + 4, true));
      }
    }
    buf.unmap();
    return out;
  } finally {
    buf.destroy();
  }
}
