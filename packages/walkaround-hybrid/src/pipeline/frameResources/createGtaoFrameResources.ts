/**
 * GTAO half/full-res textures + UBO (W4a — resourceManager split).
 */

import type { GTAOFrameResources } from '../resourceManager.js';

export function createGtaoFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
): GTAOFrameResources {
  const halfW = Math.max(1, Math.floor(width / 2));
  const halfH = Math.max(1, Math.floor(height / 2));

  const aoHalfTexture = device.createTexture({
    label: 'gtao-half',
    size: [halfW, halfH],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const aoFullTexture = device.createTexture({
    label: 'gtao-full',
    size: [width, height],
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });

  const bytesPerTexel = 8;
  const rowBytes = Math.max(256, Math.ceil((width * bytesPerTexel) / 256) * 256);
  const buf = new Uint8Array(rowBytes * height);
  for (let y = 0; y < height; y++) {
    const rowOff = y * rowBytes;
    for (let x = 0; x < width; x++) {
      const o = rowOff + x * bytesPerTexel;
      buf[o + 1] = 0x3c;
      buf[o + 3] = 0x3c;
      buf[o + 5] = 0x3c;
    }
  }
  device.queue.writeTexture(
    { texture: aoFullTexture },
    buf,
    { offset: 0, bytesPerRow: rowBytes },
    { width, height, depthOrArrayLayers: 1 },
  );

  const gtaoUboBuffer = device.createBuffer({
    label: 'gtao-ubo',
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return { aoHalfTexture, aoFullTexture, gtaoUboBuffer };
}
