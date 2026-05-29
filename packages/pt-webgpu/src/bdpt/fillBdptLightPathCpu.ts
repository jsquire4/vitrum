/**
 * CPU reference fill of the BDPT light-path scratch buffer (bounce 0 only).
 * Production uses GPU `bdptExtendLightSubpath` (col 0 + extension); kept for tests/oracles.
 *
 * The light path is a read_write storage BUFFER of vec4f (NOT a storage texture —
 * `rgba32float` read_write storage textures are not in core WebGPU; gpuweb #4651).
 * Flattened row-minor as `idx = col * 3 + row` (matches WGSL `bdptLightPathIndex`):
 * per light-vertex column, row 0 = pos (+ kind in .w), row 1 = normal + pdfFwd,
 * row 2 = throughput + pdfRev.
 */

import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  bdptEmitterCount,
  bdptEmitterPower,
  bdptPickEmitterFlat,
  sampleBdptBounce0Cpu,
} from './bdptEmitterPickCpu.js';

const KIND_INVALID = 3;
const KIND_LIGHT = 0;
const LIGHT_PATH_ROWS = 3;

/** vec4f index into the flat light-path buffer for (col, row). */
export function bdptLightPathColumnIndex(col: number, row: number): number {
  return (col * LIGHT_PATH_ROWS + row) * 4;
}

/** Flat vec4f light-path buffer contents (`col * 3 + row` ordering). */
export function packBdptLightPathColumns(
  width: number,
  bounce0: ReturnType<typeof sampleBdptBounce0Cpu>,
): Float32Array {
  const data = new Float32Array(width * LIGHT_PATH_ROWS * 4);
  for (let col = 0; col < width; col += 1) {
    data[bdptLightPathColumnIndex(col, 0) + 3] = KIND_INVALID;
  }
  if (bounce0 == null) return data;
  const col = 0;
  const o0 = bdptLightPathColumnIndex(col, 0);
  const o1 = bdptLightPathColumnIndex(col, 1);
  const o2 = bdptLightPathColumnIndex(col, 2);
  data[o0 + 0] = bounce0.emitPos[0];
  data[o0 + 1] = bounce0.emitPos[1];
  data[o0 + 2] = bounce0.emitPos[2];
  data[o0 + 3] = KIND_LIGHT;
  data[o1 + 0] = bounce0.emitNormal[0];
  data[o1 + 1] = bounce0.emitNormal[1];
  data[o1 + 2] = bounce0.emitNormal[2];
  data[o1 + 3] = bounce0.pdfJoint;
  data[o2 + 0] = bounce0.emitRad[0];
  data[o2 + 1] = bounce0.emitRad[1];
  data[o2 + 2] = bounce0.emitRad[2];
  data[o2 + 3] = bounce0.pdfHemi;
  return data;
}

/**
 * Fill the light-path scratch buffer (bounce-0 column) from a CPU-picked emitter.
 *
 * @internal Kept CPU oracle for the GPU BDPT light-path fill; not public API.
 */
export function fillBdptLightPathCpu(
  device: GPUDevice,
  buffer: GPUBuffer,
  maxLightBounces: number,
  sceneBuffers: UploadedSceneBuffers,
  frameSeed: number,
): void {
  const width = maxLightBounces;
  const emitterCount = bdptEmitterCount(sceneBuffers);
  if (emitterCount === 0) {
    writeBuffer(device, buffer, packBdptLightPathColumns(width, null));
    return;
  }

  let totalPower = 0;
  for (let i = 0; i < emitterCount; i += 1) {
    totalPower += bdptEmitterPower(sceneBuffers, i);
  }
  const uPick = ((frameSeed * 2654435761) >>> 0) / 2 ** 32;
  const uHemi = (((frameSeed + 1) * 1597334677) >>> 0) / 2 ** 32;
  const flat = bdptPickEmitterFlat(sceneBuffers, uPick * totalPower, totalPower, emitterCount);
  const discretePdf = bdptEmitterPower(sceneBuffers, flat) / Math.max(totalPower, 1e-20);
  const sample = sampleBdptBounce0Cpu(sceneBuffers, flat, discretePdf, uHemi);
  writeBuffer(device, buffer, packBdptLightPathColumns(width, sample));
}

function writeBuffer(device: GPUDevice, buffer: GPUBuffer, data: Float32Array): void {
  device.queue.writeBuffer(buffer, 0, new Float32Array(data));
}
