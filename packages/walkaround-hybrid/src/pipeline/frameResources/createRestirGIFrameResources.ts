/**
 * ReSTIR GI reservoir GPU buffers (W4a — resourceManager split).
 */

import type { RestirGIFrameResources } from '../resourceManager.js';
import { RESERVOIR_GI_STRIDE } from '../../ppg/ppgConstants.js';

// GRIS Phase-0: widened ReservoirGI → ReservoirPT (30 × u32 = 120 bytes).
// The [0..19] / 80-byte prefix is byte-identical to the Sprint-16/17 layout;
// indices [20..29] cache the reconnection-shift data the Phase-1/2 GRIS reuse
// will read (Lin 2022). RESERVOIR_GI_STRIDE is the single TS source of truth;
// the WGSL-side const in reservoirGi.wgsl.ts must stay in lockstep with it.
const RESERVOIR_GI_STRIDE_BYTES = RESERVOIR_GI_STRIDE * 4; // 4 bytes per u32

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
