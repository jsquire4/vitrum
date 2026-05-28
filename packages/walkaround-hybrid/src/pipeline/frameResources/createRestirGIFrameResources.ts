/**
 * ReSTIR GI reservoir GPU buffers (W4a — resourceManager split).
 */

import type { RestirGIFrameResources } from '../resourceManager.js';

const RESERVOIR_GI_STRIDE_BYTES = 80;

export function createRestirGIFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
): RestirGIFrameResources {
  const halfW = Math.max(1, Math.floor(width / 2));
  const halfH = Math.max(1, Math.floor(height / 2));
  const reservoirGiSize = halfW * halfH * RESERVOIR_GI_STRIDE_BYTES;
  const size = Math.max(256, reservoirGiSize);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  return {
    reservoirGiCurrentBuffer: device.createBuffer({ label: 'reservoir-gi-current', size, usage }),
    reservoirGiPreviousBuffer: device.createBuffer({ label: 'reservoir-gi-previous', size, usage }),
    reservoirGiSpatialBuffer: device.createBuffer({ label: 'reservoir-gi-spatial', size, usage }),
  };
}
