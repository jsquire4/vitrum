/**
 * UBO updater — writes per-frame camera + lighting data into the 256-byte
 * WalkaroundUBO uniform buffer.
 *
 * UBO layout (all f32 / u32, 256 bytes total):
 *   offset   0: viewMatrix            (mat4×4f = 64 bytes)
 *   offset  64: projMatrix            (64 bytes)
 *   offset 128: prevViewMatrix        (64 bytes)
 *   offset 192: cameraPos             (vec3f = 12 bytes)
 *   offset 204: frameSeed             (u32 = 4 bytes)
 *   offset 208: screenSize            (vec2u = 8 bytes)
 *   offset 216: emitterCount          (u32 = 4 bytes)
 *   offset 220: totalEmPower          (f32 = 4 bytes)
 *   offset 224: primaryLightDir       (vec3f = 12 bytes)
 *   offset 236: primaryLightIntensity (f32 = 4 bytes)
 *   offset 240: skyTint               (vec3f = 12 bytes)
 *   offset 252: skyIrradiance         (f32 = 4 bytes)
 * Total: 256 bytes (fully consumed).
 */

import type { PipelineFrameInputs } from './WalkaroundGPUPipeline.js';

export function updateUBO(
  device: GPUDevice,
  uboBuffer: GPUBuffer,
  inputs: PipelineFrameInputs,
): void {
  const data = new ArrayBuffer(256);
  const f32  = new Float32Array(data);
  const u32  = new Uint32Array(data);

  f32.set(inputs.viewMatrix,     0);    //  0..15 (64 bytes)
  f32.set(inputs.projMatrix,    16);    // 16..31 (64 bytes)
  f32.set(inputs.prevViewMatrix, 32);   // 32..47 (64 bytes)
  f32[48] = inputs.cameraPos[0];
  f32[49] = inputs.cameraPos[1];
  f32[50] = inputs.cameraPos[2];
  u32[51] = inputs.frameSeed >>> 0;
  u32[52] = inputs.screenWidth;
  u32[53] = inputs.screenHeight;
  u32[54] = inputs.emitterCount;
  f32[55] = inputs.totalEmissivePower;
  f32[56] = inputs.primaryLightDir[0];
  f32[57] = inputs.primaryLightDir[1];
  f32[58] = inputs.primaryLightDir[2];
  f32[59] = inputs.primaryLightIntensity;
  f32[60] = inputs.skyTint[0];
  f32[61] = inputs.skyTint[1];
  f32[62] = inputs.skyTint[2];
  f32[63] = inputs.skyIrradiance;

  device.queue.writeBuffer(uboBuffer, 0, data);
}
