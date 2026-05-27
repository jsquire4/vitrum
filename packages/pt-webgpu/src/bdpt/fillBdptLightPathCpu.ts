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

/** Pack one vertex column into a Float32Array row-major [width*4 * 3 rows]. */
export function fillBdptLightPathCpu(
  device: GPUDevice,
  texture: GPUTexture,
  maxLightBounces: number,
  sceneBuffers: UploadedSceneBuffers,
  frameSeed: number,
): void {
  const width = maxLightBounces;
  const data = new Float32Array(width * 4 * 3);
  for (let col = 0; col < width; col += 1) {
    const o0 = col * 4;
    data[o0 + 3] = KIND_INVALID;
  }

  const emitterCount = bdptEmitterCount(sceneBuffers);
  if (emitterCount === 0) {
    writeTexture(device, texture, width, data);
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

  const col = 0;
  const o0 = col * 4;
  const o1 = width * 4 + col * 4;
  const o2 = width * 8 + col * 4;

  if (sample != null) {
    data[o0 + 0] = sample.emitPos[0];
    data[o0 + 1] = sample.emitPos[1];
    data[o0 + 2] = sample.emitPos[2];
    data[o0 + 3] = KIND_LIGHT;
    data[o1 + 0] = sample.emitNormal[0];
    data[o1 + 1] = sample.emitNormal[1];
    data[o1 + 2] = sample.emitNormal[2];
    data[o1 + 3] = sample.pdfJoint;
    data[o2 + 0] = sample.emitRad[0];
    data[o2 + 1] = sample.emitRad[1];
    data[o2 + 2] = sample.emitRad[2];
    data[o2 + 3] = sample.pdfHemi;
  }

  writeTexture(device, texture, width, data);
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
