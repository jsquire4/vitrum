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
import type { HybridDenoiserTrainingCapture } from '../HybridEnginePublic.js';

interface MappedRgba16fReadback {
  readonly buffer: GPUBuffer;
  mapped: boolean;
}

function decodeRgba16fRgb(
  source: DataView,
  width: number,
  height: number,
  bytesPerRow: number,
  transform: (r: number, g: number, b: number) => readonly [number, number, number],
): Float32Array {
  const output = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * bytesPerRow;
    for (let x = 0; x < width; x += 1) {
      const textureOffset = rowOffset + x * 8;
      const destinationOffset = (y * width + x) * 3;
      const decoded = transform(
        float16BitsToFloat32(source.getUint16(textureOffset, true)),
        float16BitsToFloat32(source.getUint16(textureOffset + 2, true)),
        float16BitsToFloat32(source.getUint16(textureOffset + 4, true)),
      );
      output[destinationOffset] = decoded[0];
      output[destinationOffset + 1] = decoded[1];
      output[destinationOffset + 2] = decoded[2];
    }
  }
  return output;
}

/**
 * Atomically snapshot the three live rgba16float inputs consumed by the neural
 * denoiser. All copies are encoded in one command buffer, so no later renderer
 * submission can land between the radiance, albedo, and normal snapshots.
 *
 * WebGPU texture-to-buffer copies require a 256-byte-aligned row pitch. Padding
 * is removed while decoding into tightly packed CPU RGB arrays. All staging
 * buffers are unmapped and destroyed before this promise settles, including
 * partial allocation, submit, map, or decode failures.
 */
export async function readDenoiserTrainingInputsWalkaround(
  device: GPUDevice,
  textures: {
    readonly radiance: GPUTexture;
    readonly albedo: GPUTexture;
    readonly normalDepth: GPUTexture;
  },
  width: number,
  height: number,
): Promise<HybridDenoiserTrainingCapture | null> {
  if (width <= 0 || height <= 0) return null;
  const bytesPerRow = alignedTextureCopyBytesPerRow(width, 8);
  const readSize = bytesPerRow * height;
  const staging: MappedRgba16fReadback[] = [];
  try {
    for (const label of ['radiance', 'albedo', 'normal-depth'] as const) {
      staging.push({
        buffer: device.createBuffer({
          label: `vitrum.walkaround-hybrid.denoiser-training.${label}.staging`,
          size: readSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        mapped: false,
      });
    }

    const encoder = device.createCommandEncoder({
      label: 'vitrum.walkaround-hybrid.denoiser-training.encoder',
    });
    const sources = [textures.radiance, textures.albedo, textures.normalDepth];
    for (let index = 0; index < sources.length; index += 1) {
      encoder.copyTextureToBuffer(
        { texture: sources[index]! },
        {
          buffer: staging[index]!.buffer,
          bytesPerRow,
          rowsPerImage: height,
        },
        { width, height, depthOrArrayLayers: 1 },
      );
    }
    device.queue.submit([encoder.finish()]);

    const mapResults = await Promise.allSettled(
      staging.map(async (entry) => {
        await entry.buffer.mapAsync(GPUMapMode.READ);
        entry.mapped = true;
      }),
    );
    const failedMap = mapResults.find(
      (result): result is PromiseRejectedResult => result.status === 'rejected',
    );
    if (failedMap != null) throw failedMap.reason;

    const identity = (r: number, g: number, b: number): readonly [number, number, number] =>
      [r, g, b];
    const decodeWorldNormal = (
      r: number,
      g: number,
      b: number,
    ): readonly [number, number, number] => [
      r * 2 - 1,
      g * 2 - 1,
      b * 2 - 1,
    ];
    return {
      width,
      height,
      radiance: decodeRgba16fRgb(
        new DataView(staging[0]!.buffer.getMappedRange()),
        width,
        height,
        bytesPerRow,
        identity,
      ),
      albedo: decodeRgba16fRgb(
        new DataView(staging[1]!.buffer.getMappedRange()),
        width,
        height,
        bytesPerRow,
        identity,
      ),
      worldNormal: decodeRgba16fRgb(
        new DataView(staging[2]!.buffer.getMappedRange()),
        width,
        height,
        bytesPerRow,
        decodeWorldNormal,
      ),
    };
  } finally {
    for (const entry of staging) {
      if (entry.mapped) {
        try {
          entry.buffer.unmap();
        } catch {
          // Preserve the capture result or primary failure.
        }
      }
      try {
        entry.buffer.destroy();
      } catch {
        // Preserve the capture result or primary failure.
      }
    }
  }
}

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
