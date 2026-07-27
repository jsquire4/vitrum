/**
 * ReSTIR DI reservoir GPU buffers (W4a — resourceManager split).
 */

import type { RestirDIFrameResources } from '../resourceManager.js';
import { RESERVOIR_DI_STRIDE_BYTES } from '../../restir/reservoirDiLayout.js';

export function createRestirDIFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
): RestirDIFrameResources {
  const totalReservoirBytes = Math.max(
    width * height * RESERVOIR_DI_STRIDE_BYTES,
    256,
  );
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  const reservoirCurrentBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage,
  });
  const reservoirPreviousBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const reservoirSpatialBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return {
    reservoirCurrentBuffer,
    reservoirPreviousBuffer,
    reservoirSpatialBuffer,
  };
}
