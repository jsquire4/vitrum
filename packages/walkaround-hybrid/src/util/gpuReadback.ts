/**
 * GPU→CPU readback utilities for walkaround-hybrid.
 *
 * Extracted from `HybridEngine.ts` (R3 B-chain decomposition sweep, step 2).
 * No class dependencies.
 */

import {
  alignedTextureCopyBytesPerRow,
  float16BitsToFloat32,
} from '@vitrum/shared-denoisers';

/**
 * Read a single rgba16float GPUTexture to a Float32 RGBA array, row-major,
 * top-left origin. Returns `null` if dimensions are invalid. Pipeline stall
 * (copyTextureToBuffer + mapAsync).
 *
 * Used by HybridEngine.captureFrame for 'linear' readback.
 */
export async function readRgba16fWalkaround(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
): Promise<Float32Array | null> {
  if (width <= 0 || height <= 0) return null;
  const bytesPerRow = alignedTextureCopyBytesPerRow(width, 8); // 4 ch × 2 B per f16
  const readSize = bytesPerRow * height;
  const staging = device.createBuffer({
    label: 'vitrum.walkaround-hybrid.captureFrame.staging',
    size: readSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({
      label: 'vitrum.walkaround-hybrid.captureFrame.encoder',
    });
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: staging, bytesPerRow },
      { width, height, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new DataView(staging.getMappedRange().slice(0));
    staging.unmap();
    // Decode rgba16float → float32 RGBA, 4 channels per pixel.
    const out = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      const rowOff = y * bytesPerRow;
      for (let x = 0; x < width; x++) {
        const texOff = rowOff + x * 8;
        const di = (y * width + x) * 4;
        out[di]     = float16BitsToFloat32(src.getUint16(texOff,     true));
        out[di + 1] = float16BitsToFloat32(src.getUint16(texOff + 2, true));
        out[di + 2] = float16BitsToFloat32(src.getUint16(texOff + 4, true));
        out[di + 3] = float16BitsToFloat32(src.getUint16(texOff + 6, true));
      }
    }
    return out;
  } finally {
    staging.destroy();
  }
}
