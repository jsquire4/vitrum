/**
 * ReSTIR GI reservoir GPU buffers (W4a — resourceManager split).
 */

import type { RestirGIFrameResources } from '../resourceManager.js';

// GRIS Phase-0: widened ReservoirGI → ReservoirPT (30 × u32 = 120 bytes).
// The [0..19] / 80-byte prefix is byte-identical to the Sprint-16/17 layout;
// indices [20..29] cache the reconnection-shift data the Phase-1/2 GRIS reuse
// will read (Lin 2022). Must stay in lockstep with RESERVOIR_GI_STRIDE in
// shaders/reservoirGi.wgsl.ts.
const RESERVOIR_GI_STRIDE_BYTES = 120;

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
