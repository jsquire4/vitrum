import {
  alignedTextureCopyBytesPerRow,
  decodeNormalDepthWorldNormal,
  float16BitsToFloat32 as f16ToF32,
  rgba16fBufferToRgbF32,
} from '@vitrum/shared-denoisers';

/**
 * Row-major rgba16float GPU buffer → interleaved RGBA Float32, top-left origin.
 *
 * The GPU readback buffer has `bytesPerRow` padding per row (256-byte aligned);
 * this function strips the padding and unpacks all four channels as Float32.
 * Row 0 maps to the top of the image (WebGPU uses top-left origin natively, so
 * no flip is needed here — unlike the WebGL2 path which uses bottom-left origin).
 */
export function rgba16fBufferToRgbaF32(
  src: ArrayBuffer,
  bytesPerRow: number,
  width: number,
  height: number,
): Float32Array {
  const dst = new Float32Array(width * height * 4);
  const view = new DataView(src);
  for (let y = 0; y < height; y++) {
    const rowOff = y * bytesPerRow;
    for (let x = 0; x < width; x++) {
      const texOff = rowOff + x * 8; // 4 channels × 2 bytes per f16
      const dstIdx = (y * width + x) * 4;
      dst[dstIdx]     = f16ToF32(view.getUint16(texOff,     true));
      dst[dstIdx + 1] = f16ToF32(view.getUint16(texOff + 2, true));
      dst[dstIdx + 2] = f16ToF32(view.getUint16(texOff + 4, true));
      dst[dstIdx + 3] = f16ToF32(view.getUint16(texOff + 6, true));
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
/**
 * H14-D — GPU → CPU readback of HDR color and optional G-buffer aux for OIDN.
 * Submits its own command buffer (call after the path-trace pass has landed).
 *
 * All created GPUBuffers are tracked in a `created` array and destroyed in a
 * `finally` block, so a rejected `mapAsync` (device lost, out of memory, etc.)
 * never leaks GPU memory. The destroy-once guard (set membership) prevents
 * double-destroy if a buffer appears in both the eager destroy path (albedo /
 * normal after unmap) and the finally cleanup.
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

  // Track every buffer created so the finally block can destroy any that
  // survive a rejection without double-destroying eagerly-cleaned ones.
  const created: GPUBuffer[] = [];
  const destroyed = new Set<GPUBuffer>();
  const safeDestroy = (buf: GPUBuffer | null) => {
    if (buf != null && !destroyed.has(buf)) {
      destroyed.add(buf);
      buf.destroy();
    }
  };

  const colorReadback = device.createBuffer({
    label: 'vitrum.pt-webgpu.oidn-readback-color',
    size: readSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  created.push(colorReadback);

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
    created.push(albedoReadback);
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
    created.push(normalReadback);
    encoder.copyTextureToBuffer(
      { texture: sources.normalDepth },
      { buffer: normalReadback, bytesPerRow },
      { width, height, depthOrArrayLayers: 1 },
    );
  }

  device.queue.submit([encoder.finish()]);

  try {
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
      safeDestroy(albedoReadback);
    }

    let normal: Float32Array | undefined;
    if (normalReadback != null) {
      normal = rgba16fBufferToRgbF32(
        normalReadback.getMappedRange().slice(0),
        bytesPerRow,
        width,
        height,
        decodeNormalDepthWorldNormal,
      );
      normalReadback.unmap();
      safeDestroy(normalReadback);
    }

    safeDestroy(colorReadback);
    return { color, ...(albedo !== undefined ? { albedo } : {}), ...(normal !== undefined ? { normal } : {}), width, height };
  } finally {
    // Destroy any buffers that weren't already cleaned up in the happy path
    // (e.g. when mapAsync rejects due to device loss or OOM).
    for (const buf of created) safeDestroy(buf);
  }
}

/**
 * GPU → CPU readback of a single rgba16float texture as a Float32 RGBA array,
 * row-major, top-left origin.
 *
 * Used by `captureFrame` in the pt-webgpu backend — reads `accumTexture` (the
 * Welford running-mean linear HDR radiance, written by `accumulateFrame`) for
 * `colorSpace:'linear'`, or `presentTexture` (the tonemapped output, written by
 * the present pass) for `colorSpace:'output'`.
 *
 * Returns `null` when `width` or `height` is ≤ 0 (engine not yet initialised).
 */
export async function readRgba16fTextureToF32(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
): Promise<Float32Array | null> {
  if (width <= 0 || height <= 0) return null;

  const bytesPerRow = alignedTextureCopyBytesPerRow(width, 8); // 4 ch × 2 B per f16
  const readSize = bytesPerRow * height;

  const stagingBuffer = device.createBuffer({
    label: 'vitrum.pt-webgpu.captureFrame.staging',
    size: readSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.captureFrame.encoder',
    });
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: stagingBuffer, bytesPerRow },
      { width, height, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = rgba16fBufferToRgbaF32(
      stagingBuffer.getMappedRange().slice(0),
      bytesPerRow,
      width,
      height,
    );
    stagingBuffer.unmap();
    return result;
  } finally {
    stagingBuffer.destroy();
  }
}
