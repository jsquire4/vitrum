/**
 * GTAO low-res + full-res textures + UBO (W4a — resourceManager split).
 *
 * `downscale` controls the AO compute resolution: the `aoHalfTexture`
 * (historically half-res) is allocated at `W/downscale × H/downscale`.
 *   - `downscale = 2` → half-res (`gtaoMode:'on'`, Sprint-15 default).
 *   - `downscale = 4` → quarter-res (`gtaoMode:'quarter'`, a real step below
 *     'on': 1/4 of each axis = 1/16 the AO compute footprint).
 * The bilateral upsample (`gtaoUpsample.wgsl`) reconstructs the full-res
 * `aoFullTexture` from this low-res map at any downscale factor; the factor
 * is threaded into both shaders via the GTAO UBO so the `÷N`/`×N` mapping is
 * data-driven rather than the prior hardcoded `÷2`.
 */

import type { GTAOFrameResources } from '../resourceManager.js';

export function createGtaoFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
  downscale: number = 2,
): GTAOFrameResources {
  // Clamp the downscale to a sane integer ≥ 1 so a stray 0 / fractional
  // host value can never produce a zero-sized texture.
  const ds = Math.max(1, Math.floor(downscale));
  const lowW = Math.max(1, Math.floor(width / ds));
  const lowH = Math.max(1, Math.floor(height / ds));

  const aoHalfTexture = device.createTexture({
    label: ds === 2 ? 'gtao-half' : `gtao-low-${ds}x`,
    size: [lowW, lowH],
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
