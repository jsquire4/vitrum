/**
 * DDGI placeholder atlas + grid UBO (W4a — resourceManager split).
 */

import type { DDGIFrameResources } from '../resourceManager.js';
import { buildDDGIPlaceholderUBO } from '../resourceManager.js';

export function createDdgiFrameResources(device: GPUDevice): DDGIFrameResources {
  const ddgiPlaceholderRgba16f = device.createTexture({
    label: 'ddgi-placeholder-irr',
    size: [1, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const ddgiPlaceholderVisRgba16f = device.createTexture({
    label: 'ddgi-placeholder-vis',
    size: [1, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const ddgiUboBuffer = device.createBuffer({
    label: 'ddgi-ubo',
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ddgiUboBuffer, 0, buildDDGIPlaceholderUBO().buffer);

  return {
    ddgiPlaceholderRgba16f,
    ddgiPlaceholderVisRgba16f,
    ddgiUboBuffer,
  };
}
