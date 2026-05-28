/**
 * CPU reference fill of the BDPT light-path texture (bounce 0 only).
 * Production uses GPU `bdptExtendLightSubpath` (col 0 + extension); kept for tests/oracles.
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

/** Row-major RGBA32F light-path texture bytes (height=3, width=maxLightBounces). */
export function packBdptLightPathColumns(
  width: number,
  bounce0: ReturnType<typeof sampleBdptBounce0Cpu>,
): Float32Array {
  const data = new Float32Array(width * 4 * 3);
  for (let col = 0; col < width; col += 1) {
    data[col * 4 + 3] = KIND_INVALID;
  }
  if (bounce0 == null) return data;
  const col = 0;
  const o0 = col * 4;
  const o1 = width * 4 + col * 4;
  const o2 = width * 8 + col * 4;
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
 * Pack one vertex column into a Float32Array row-major [width*4 * 3 rows].
 *
 * @internal Kept CPU oracle for the GPU BDPT light-path fill; not public API.
 */
export function fillBdptLightPathCpu(
  device: GPUDevice,
  texture: GPUTexture,
  maxLightBounces: number,
  sceneBuffers: UploadedSceneBuffers,
  frameSeed: number,
): void {
  const width = maxLightBounces;
  const emitterCount = bdptEmitterCount(sceneBuffers);
  if (emitterCount === 0) {
    writeTexture(device, texture, width, packBdptLightPathColumns(width, null));
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
  writeTexture(device, texture, width, packBdptLightPathColumns(width, sample));
}

function writeTexture(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  data: Float32Array,
): void {
  const upload = new Float32Array(data);
  device.queue.writeTexture(
    { texture },
    upload,
    { bytesPerRow: width * 16, rowsPerImage: 3 },
    { width, height: 3, depthOrArrayLayers: 1 },
  );
}
