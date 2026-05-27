import {
  float16BitsToFloat32 as f16ToF32,
} from '@vitrum/shared-denoisers';
import { alignedTextureCopyBytesPerRow } from '@vitrum/shared-denoisers';

/** Row-major rgba16float → interleaved RGB Float32 (OIDN layout). */
export function rgba16fBufferToRgbF32(
  src: ArrayBuffer,
  bytesPerRow: number,
  width: number,
  height: number,
  decode?: (r: number, g: number, b: number) => [number, number, number],
): Float32Array {
  const dst = new Float32Array(width * height * 3);
  const view = new DataView(src);
  for (let y = 0; y < height; y++) {
    const rowOff = y * bytesPerRow;
    for (let x = 0; x < width; x++) {
      const texOff = rowOff + x * 8;
      const r = f16ToF32(view.getUint16(texOff, true));
      const g = f16ToF32(view.getUint16(texOff + 2, true));
      const b = f16ToF32(view.getUint16(texOff + 4, true));
      const [or, og, ob] = decode ? decode(r, g, b) : [r, g, b];
      const dstIdx = (y * width + x) * 3;
      dst[dstIdx] = or;
      dst[dstIdx + 1] = og;
      dst[dstIdx + 2] = ob;
    }
  }
  return dst;
}

export interface OidnTextureSources {
  readonly color: GPUTexture;
  readonly albedo?: GPUTexture | null;
  readonly normalDepth?: GPUTexture | null;
}

export interface OidnReadbackResult {
  readonly color: Float32Array;
  readonly albedo?: Float32Array;
  readonly normal?: Float32Array;
  readonly width: number;
  readonly height: number;
}

export type OidnReadbackFn = (
  device: GPUDevice,
  sources: OidnTextureSources,
  width: number,
  height: number,
) => Promise<OidnReadbackResult>;

/**
 * GPU → CPU readback of HDR color and optional G-buffer aux for OIDN.
 * Submits its own command buffer (call after the path-trace pass has landed).
 */
export async function readOidnInputsFromTextures(
  device: GPUDevice,
  sources: OidnTextureSources,
  width: number,
  height: number,
): Promise<OidnReadbackResult> {
  if (width <= 0 || height <= 0) {
    return { color: new Float32Array(0), width, height };
  }
  const bytesPerRow = alignedTextureCopyBytesPerRow(width, 8);
  const readSize = bytesPerRow * height;

  const colorReadback = device.createBuffer({
    label: 'vitrum.pt-webgpu.oidn-readback-color',
    size: readSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder({ label: 'vitrum.pt-webgpu.oidn-readback.encoder' });
  encoder.copyTextureToBuffer(
    { texture: sources.color },
    { buffer: colorReadback, bytesPerRow },
    { width, height, depthOrArrayLayers: 1 },
  );

  let albedoReadback: GPUBuffer | null = null;
  let normalReadback: GPUBuffer | null = null;
  if (sources.albedo != null) {
    albedoReadback = device.createBuffer({
      label: 'vitrum.pt-webgpu.oidn-readback-albedo',
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: sources.albedo },
      { buffer: albedoReadback, bytesPerRow },
      { width, height, depthOrArrayLayers: 1 },
    );
  }
  if (sources.normalDepth != null) {
    normalReadback = device.createBuffer({
      label: 'vitrum.pt-webgpu.oidn-readback-normal',
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: sources.normalDepth },
      { buffer: normalReadback, bytesPerRow },
      { width, height, depthOrArrayLayers: 1 },
    );
  }

  device.queue.submit([encoder.finish()]);

  const mapPromises: Promise<void>[] = [colorReadback.mapAsync(GPUMapMode.READ)];
  if (albedoReadback != null) mapPromises.push(albedoReadback.mapAsync(GPUMapMode.READ));
  if (normalReadback != null) mapPromises.push(normalReadback.mapAsync(GPUMapMode.READ));
  await Promise.all(mapPromises);

  const color = rgba16fBufferToRgbF32(
    colorReadback.getMappedRange().slice(0),
    bytesPerRow,
    width,
    height,
  );
  colorReadback.unmap();

  let albedo: Float32Array | undefined;
  if (albedoReadback != null) {
    albedo = rgba16fBufferToRgbF32(
      albedoReadback.getMappedRange().slice(0),
      bytesPerRow,
      width,
      height,
    );
    albedoReadback.unmap();
    albedoReadback.destroy();
  }

  let normal: Float32Array | undefined;
  if (normalReadback != null) {
    normal = rgba16fBufferToRgbF32(
      normalReadback.getMappedRange().slice(0),
      bytesPerRow,
      width,
      height,
      (r, g, b) => [r * 2 - 1, g * 2 - 1, b * 2 - 1],
    );
    normalReadback.unmap();
    normalReadback.destroy();
  }

  colorReadback.destroy();

  return { color, ...(albedo !== undefined ? { albedo } : {}), ...(normal !== undefined ? { normal } : {}), width, height };
}
