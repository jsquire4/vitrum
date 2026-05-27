/**
 * CPU fill of the BDPT light-path texture (bounce 0 from scene emitters).
 * Bounces 1..N-1 are left invalid until the GPU light-subpath pass lands;
 * the connection pass skips invalid vertices (kind = 3).
 */

import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';

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
  const invalid = () => {
    for (let col = 0; col < width; col += 1) {
      const o0 = col * 4;
      data[o0 + 0] = 0;
      data[o0 + 1] = 0;
      data[o0 + 2] = 0;
      data[o0 + 3] = KIND_INVALID;
    }
  };
  invalid();

  // Bounce 0: prefer first mesh-area emitter, else first point light, else directional proxy.
  const col = 0;
  const o0 = col * 4;
  const o1 = width * 4 + col * 4;
  const o2 = width * 8 + col * 4;

  if (sceneBuffers.meshAreaLightCount > 0) {
    const m = sceneBuffers.meshAreaLightsData;
    const ax = m[0]!;
    const ay = m[1]!;
    const az = m[2]!;
    const bx = m[4]!;
    const by = m[5]!;
    const bz = m[6]!;
    const cx = m[8]!;
    const cy = m[9]!;
    const cz = m[10]!;
    const pos = [(ax + bx + cx) / 3, (ay + by + cy) / 3, (az + bz + cz) / 3];
    const e1 = [bx - ax, by - ay, bz - az];
    const e2 = [cx - ax, cy - ay, cz - az];
    const n = [
      e1[1]! * e2[2]! - e1[2]! * e2[1]!,
      e1[2]! * e2[0]! - e1[0]! * e2[2]!,
      e1[0]! * e2[1]! - e1[1]! * e2[0]!,
    ];
    const rad = [m[12]!, m[13]!, m[14]!];
    const len = Math.hypot(n[0], n[1], n[2]) || 1;
    const normal = [n[0] / len, n[1] / len, n[2] / len];
    const pdfFwd = 1.0;
    const throughput = [rad[0] / pdfFwd, rad[1] / pdfFwd, rad[2] / pdfFwd];
    data[o0 + 0] = pos[0]!;
    data[o0 + 1] = pos[1]!;
    data[o0 + 2] = pos[2]!;
    data[o0 + 3] = KIND_LIGHT;
    data[o1 + 0] = normal[0]!;
    data[o1 + 1] = normal[1]!;
    data[o1 + 2] = normal[2]!;
    data[o1 + 3] = pdfFwd;
    data[o2 + 0] = throughput[0]!;
    data[o2 + 1] = throughput[1]!;
    data[o2 + 2] = throughput[2]!;
    data[o2 + 3] = pdfFwd;
  } else if (sceneBuffers.pointLightCount > 0) {
    const p = sceneBuffers.pointLightsData;
    const pos = [p[0]!, p[1]!, p[2]!];
    const rad = [p[4]!, p[5]!, p[6]!];
    const pdfFwd = 1.0;
    data[o0 + 0] = pos[0]!;
    data[o0 + 1] = pos[1]!;
    data[o0 + 2] = pos[2]!;
    data[o0 + 3] = KIND_LIGHT;
    data[o1 + 0] = 0;
    data[o1 + 1] = 1;
    data[o1 + 2] = 0;
    data[o1 + 3] = pdfFwd;
    data[o2 + 0] = rad[0]! / pdfFwd;
    data[o2 + 1] = rad[1]! / pdfFwd;
    data[o2 + 2] = rad[2]! / pdfFwd;
    data[o2 + 3] = pdfFwd;
  } else if (sceneBuffers.directionalIrradiance[0] + sceneBuffers.directionalIrradiance[1] + sceneBuffers.directionalIrradiance[2] > 1e-6) {
    const dir = sceneBuffers.directionalLight;
    const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const lightDir = [dir[0] / len, dir[1] / len, dir[2] / len];
    const emitPos = [
      -lightDir[0]! * 50 + (frameSeed % 7) * 1e-4,
      -lightDir[1]! * 50,
      -lightDir[2]! * 50,
    ];
    const irr = sceneBuffers.directionalIrradiance;
    const pdfFwd = 1.0;
    data[o0 + 0] = emitPos[0]!;
    data[o0 + 1] = emitPos[1]!;
    data[o0 + 2] = emitPos[2]!;
    data[o0 + 3] = KIND_LIGHT;
    data[o1 + 0] = lightDir[0]!;
    data[o1 + 1] = lightDir[1]!;
    data[o1 + 2] = lightDir[2]!;
    data[o1 + 3] = pdfFwd;
    data[o2 + 0] = irr[0]! / pdfFwd;
    data[o2 + 1] = irr[1]! / pdfFwd;
    data[o2 + 2] = irr[2]! / pdfFwd;
    data[o2 + 3] = pdfFwd;
  }

  device.queue.writeTexture(
    { texture },
    data,
    { bytesPerRow: width * 16, rowsPerImage: 3 },
    { width, height: 3, depthOrArrayLayers: 1 },
  );
}
