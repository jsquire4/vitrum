/**
 * ReSTIR GI reservoir GPU buffers (W4a — resourceManager split).
 */

import type { RestirGIFrameResources } from '../resourceManager.js';
import { reservoirGiStrideBytesForRestirPtReuse } from '../../gi/giLayout.js';

export interface RestirGIFrameResourceOptions {
  /**
	 * GRIS / ReSTIR-PT reconnection-shift reuse widens each half-res reservoir
	 * from the base 20-u32 Sprint-16/17 layout to the 30-u32 ReservoirPT layout.
	 * The appended cache fields are read by the GRIS reuse variants; the default
	 * path stays compact because those variants are not compiled.
	 */
  readonly restirPtReuse?: boolean;
}

export function createRestirGIFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
  options?: RestirGIFrameResourceOptions,
): RestirGIFrameResources {
  const halfW = Math.max(1, Math.floor(width / 2));
  const halfH = Math.max(1, Math.floor(height / 2));
  const reservoirGiSize = halfW * halfH * reservoirGiStrideBytesForRestirPtReuse(options?.restirPtReuse === true);
  const size = Math.max(256, reservoirGiSize);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  return {
    reservoirGiCurrentBuffer: device.createBuffer({ label: 'reservoir-gi-current', size, usage }),
    reservoirGiPreviousBuffer: device.createBuffer({ label: 'reservoir-gi-previous', size, usage }),
    reservoirGiSpatialBuffer: device.createBuffer({ label: 'reservoir-gi-spatial', size, usage }),
  };
}
