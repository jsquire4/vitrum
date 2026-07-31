/**
 * ReSTIR DI reservoir GPU buffers (W4a — resourceManager split).
 */

import type { RestirDIFrameResources } from '../resourceManager.js';
import { RESERVOIR_DI_STRIDE_BYTES } from '../../restir/reservoirDiLayout.js';
import { assertFrameResourceReservoirScale } from '../frameResourcePlan.js';

export function createRestirDIFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
  reservoirScale = 1,
): RestirDIFrameResources {
  assertFrameResourceReservoirScale(reservoirScale);
  const reservoirWidth = Math.max(1, Math.floor(width / reservoirScale));
  const reservoirHeight = Math.max(1, Math.floor(height / reservoirScale));
  const totalReservoirBytes = Math.max(
    reservoirWidth * reservoirHeight * RESERVOIR_DI_STRIDE_BYTES,
    256,
  );
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  const reservoirCurrentBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage,
  });
  const reservoirPreviousBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage,
  });
  const reservoirSpatialBuffer = device.createBuffer({
    size: totalReservoirBytes,
    usage,
  });

  return {
    reservoirCurrentBuffer,
    reservoirPreviousBuffer,
    reservoirSpatialBuffer,
  };
}
