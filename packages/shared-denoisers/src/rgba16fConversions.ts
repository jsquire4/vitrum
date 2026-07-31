/**
 * RGBA16F ↔ Float32 RGB conversion helpers for GPU readback / upload paths.
 *
 * These wrap `float16BitsToFloat32` / `float32ToFloat16Bits` (halfFloat.ts)
 * and `alignedTextureCopyBytesPerRow` (webGpuTextureCopy.ts).  Extracted from
 * `walkaround-hybrid/pipeline/denoisers/oidnFinal.ts` so they can be shared
 * with any backend that needs the same rgba16float ↔ Float32-RGB round-trip
 * (the OIDN inference path in pt-webgpu being the immediate next consumer).
 *
 * Format conventions:
 *   - GPU side: `rgba16float`, row-major, bytesPerRow 256-byte aligned.
 *   - CPU side: `Float32Array` with 3 channels (RGB, no alpha) per pixel,
 *     row-major, tightly packed — the layout expected by OIDNDenoiseInputs.
 */

import { float16BitsToFloat32, float32ToFloat16Bits } from './halfFloat.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';
import {
  assertFiniteFloat16Slice,
  assertOneShotDimensions,
} from './webGpuOneShotValidation.js';

/**
 * Read 4 channels of a row-major rgba16float GPU readback buffer into a
 * Float32 RGB (3-channel) layout.  `decode` is applied per-pixel
 * post-extraction — pass `(r, g, b) => [r*2-1, g*2-1, b*2-1]` for the
 * normal channel to convert from `[0, 1]` packed normals back to `[-1, 1]`.
 *
 * `bytesPerRow` must be the 256-byte-aligned row pitch used in
 * `copyTextureToBuffer` (i.e. the value from
 * {@link alignedTextureCopyBytesPerRow}`(width, 8)`).
 */
export function rgba16fBufferToRgbF32(
  src: ArrayBuffer,
  bytesPerRow: number,
  width: number,
  height: number,
  decode?: (r: number, g: number, b: number) => [number, number, number],
): Float32Array {
  const dst = new Float32Array(width * height * 3);
  const view = new DataView(src);
  for (let y = 0; y < height; y++) {
    const rowOff = y * bytesPerRow;
    for (let x = 0; x < width; x++) {
      const texOff = rowOff + x * 8; // 4 channels × 2 bytes per f16
      const r = float16BitsToFloat32(view.getUint16(texOff,     true));
      const g = float16BitsToFloat32(view.getUint16(texOff + 2, true));
      const b = float16BitsToFloat32(view.getUint16(texOff + 4, true));
      const [or, og, ob] = decode ? decode(r, g, b) : [r, g, b];
      const dstIdx = (y * width + x) * 3;
      dst[dstIdx    ] = or;
      dst[dstIdx + 1] = og;
      dst[dstIdx + 2] = ob;
    }
  }
  return dst;
}

/**
 * Pack a Float32 RGB (HxWx3, tightly packed) buffer into a row-aligned
 * rgba16float layout suitable for `GPUQueue.writeTexture` into a
 * `rgba16float` texture.  The alpha channel is set to 1.0.
 *
 * Returns `{ buffer, bytesPerRow }` — pass both directly to
 * `queue.writeTexture`:
 * ```ts
 * const { buffer, bytesPerRow } = rgbF32ToRgba16fRowAligned(rgb, W, H);
 * device.queue.writeTexture(
 *   { texture },
 *   buffer,
 *   { offset: 0, bytesPerRow },
 *   { width: W, height: H, depthOrArrayLayers: 1 },
 * );
 * ```
 */
export function rgbF32ToRgba16fRowAligned(
  src: Float32Array,
  width: number,
  height: number,
): { buffer: ArrayBuffer; bytesPerRow: number } {
  const label = 'rgbF32ToRgba16fRowAligned';
  const pixelCount = assertOneShotDimensions(label, width, height);
  const requiredLength = pixelCount * 3;
  if (src.length !== requiredLength) {
    throw new Error(
      `${label}: src length must equal ${requiredLength}; received ${src.length}`,
    );
  }
  assertFiniteFloat16Slice(label, 'src', src, requiredLength);

  const bytesPerRow = alignedTextureCopyBytesPerRow(width, 8);
  // Allocate as ArrayBuffer (not ArrayBufferLike via `new Uint8Array(N).buffer`)
  // so the return type stays narrow enough for GPUAllowSharedBufferSource —
  // TS 5.5+ widens `Uint8Array<...>` to `Uint8Array<ArrayBufferLike>` which
  // GPUQueue.writeTexture's strict overload rejects.
  const buf = new ArrayBuffer(bytesPerRow * height);
  const view = new DataView(buf);
  const oneF16 = float32ToFloat16Bits(1.0);
  for (let y = 0; y < height; y++) {
    const rowOff = y * bytesPerRow;
    for (let x = 0; x < width; x++) {
      const srcIdx = (y * width + x) * 3;
      const texOff = rowOff + x * 8;
      view.setUint16(texOff,     float32ToFloat16Bits(src[srcIdx    ] ?? 0), true);
      view.setUint16(texOff + 2, float32ToFloat16Bits(src[srcIdx + 1] ?? 0), true);
      view.setUint16(texOff + 4, float32ToFloat16Bits(src[srcIdx + 2] ?? 0), true);
      view.setUint16(texOff + 6, oneF16,                                      true);
    }
  }
  return { buffer: buf, bytesPerRow };
}
