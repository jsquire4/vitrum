/**
 * ReSTIR GI reservoir GPU buffers (W4a — resourceManager split).
 *
 * All three buffers use the sole live 28-u32 generalized-reuse ABI. The
 * producer writes the appended metadata and the canonical temporal/spatial
 * passes consume it; compact 20-u32 data is snapshot-migration-only.
 */

import type { RestirGIFrameResources } from '../resourceManager.js';
import { RESERVOIR_GI_STRIDE_BYTES } from '../../gi/giLayout.js';
import { assertFrameResourceReservoirScale } from '../frameResourcePlan.js';

export function createRestirGIFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
  reservoirScale = 1,
): RestirGIFrameResources {
  assertFrameResourceReservoirScale(reservoirScale);
  const halfW = Math.max(1, Math.floor(width / (2 * reservoirScale)));
  const halfH = Math.max(1, Math.floor(height / (2 * reservoirScale)));
  const reservoirGiSize = halfW * halfH * RESERVOIR_GI_STRIDE_BYTES;
  const size = Math.max(256, reservoirGiSize);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  return {
    reservoirGiCurrentBuffer: device.createBuffer({ label: 'reservoir-gi-current', size, usage }),
    reservoirGiPreviousBuffer: device.createBuffer({ label: 'reservoir-gi-previous', size, usage }),
    reservoirGiSpatialBuffer: device.createBuffer({ label: 'reservoir-gi-spatial', size, usage }),
  };
}
