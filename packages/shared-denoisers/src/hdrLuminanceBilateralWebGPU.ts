/**
 * One-shot HDR bilateral denoise on CPU-backed linear RGB (post path trace).
 * Uses WebGPU compute — separate from the GL bilateral preview in Cornell.
 */

import {
  HDR_LUMINANCE_BILATERAL_ENTRY,
  HDR_LUMINANCE_BILATERAL_WGSL,
  HDR_LUMINANCE_BILATERAL_WORKGROUP_SIZE,
} from './wgsl/hdrLuminanceBilateral.wgsl.js';
import { getSharedWebGPUDevice } from './sharedWebGpuDevice.js';
import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';

/** Default luminance edge-stop σ for `runHdrLuminanceBilateralWebGPU` (matches Cornell `vitrumWgslSigma`). */
export const HDR_LUMINANCE_BILATERAL_DEFAULT_SIGMA_LUMINANCE = 0.06 as const;

export interface HdrLuminanceBilateralWebGPUOptions {
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;
  /** Luminance edge-stop σ; larger → more blur. Typical 0.02–0.15 HDR linear. */
  readonly sigmaLuminance?: number;
  /** Use this device; never destroyed by this call. */
  readonly device?: GPUDevice;
  /**
   * When true (default), uses a cached process-wide device via getSharedWebGPUDevice.
   * When false, allocates a dedicated device per call and destroys it afterward.
   */
  readonly reuseSharedWebGpuDevice?: boolean;
}

const bilateralPipelineByDevice = new WeakMap<GPUDevice, GPUComputePipeline>();

function bilateralComputePipeline(device: GPUDevice): GPUComputePipeline {
  let pipeline = bilateralPipelineByDevice.get(device);
  if (pipeline == null) {
    const shaderModule = device.createShaderModule({ label: 'hdr-lum-bilateral', code: HDR_LUMINANCE_BILATERAL_WGSL });
    pipeline = device.createComputePipeline({
      label: 'hdr-lum-bilateral-pipeline',
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: HDR_LUMINANCE_BILATERAL_ENTRY },
    });
    bilateralPipelineByDevice.set(device, pipeline);
  }
  return pipeline;
}

/**
 * Runs one luminance bilateral pass on linear HDR RGB (flattened length w*h*3).
 * By default reuses getSharedWebGPUDevice and caches the compute pipeline per device.
 */
export async function runHdrLuminanceBilateralWebGPU(
  opts: HdrLuminanceBilateralWebGPUOptions,
): Promise<Float32Array> {
  const { rgb, width: w, height: h } = opts;
  const sigma = opts.sigmaLuminance ?? HDR_LUMINANCE_BILATERAL_DEFAULT_SIGMA_LUMINANCE;
  const reuseShared = opts.reuseSharedWebGpuDevice !== false && opts.device == null;
  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error('runHdrLuminanceBilateralWebGPU: WebGPU not available in this browser');
  }
  if (w <= 0 || h <= 0 || rgb.length < w * h * 3) {
    throw new Error('runHdrLuminanceBilateralWebGPU: invalid rgb buffer or dimensions');
  }

  let device: GPUDevice;
  let destroyEphemeral: (() => void) | null = null;
  if (opts.device != null) {
    device = opts.device;
  } else if (reuseShared) {
    device = await getSharedWebGPUDevice();
  } else {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter == null) {
      throw new Error('runHdrLuminanceBilateralWebGPU: failed to request GPU adapter');
    }
    device = await adapter.requestDevice();
    destroyEphemeral = () => {
      device.destroy();
    };
  }

  const pipeline = bilateralComputePipeline(device);

  const texIn = device.createTexture({
    label: 'hdr-bilateral-in',
    size: [w, h],
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const texOut = device.createTexture({
    label: 'hdr-bilateral-out',
    size: [w, h],
    format: 'rgba32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  // rgba32float texels are 16 bytes (4 × f32).
  const RGBA32F_BPP = 16;
  const bytesPerRow = alignedTextureCopyBytesPerRow(w, RGBA32F_BPP);
  const uploadSize = bytesPerRow * h;
  const upload = new Float32Array(uploadSize / 4);
  for (let y = 0; y < h; y += 1) {
    const rowOff = (y * bytesPerRow) / 4;
    for (let x = 0; x < w; x += 1) {
      const si = (y * w + x) * 3;
      upload[rowOff + x * 4 + 0] = rgb[si] ?? 0;
      upload[rowOff + x * 4 + 1] = rgb[si + 1] ?? 0;
      upload[rowOff + x * 4 + 2] = rgb[si + 2] ?? 0;
      upload[rowOff + x * 4 + 3] = 1;
    }
  }

  device.queue.writeTexture({ texture: texIn }, upload.buffer, {
    bytesPerRow,
    rowsPerImage: h,
  }, [w, h]);

  // BilateralParams UBO: 4 × f32 = 16 bytes.
  const HDR_BILATERAL_UBO_SIZE_BYTES = 16;
  const ubo = device.createBuffer({
    label: 'hdr-bilateral-ubo',
    size: HDR_BILATERAL_UBO_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uboData = new Float32Array(4);
  uboData[0] = sigma;
  device.queue.writeBuffer(ubo, 0, uboData.buffer);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: texIn.createView() },
      { binding: 1, resource: texOut.createView() },
      { binding: 2, resource: { buffer: ubo } },
    ],
  });

  const wg = HDR_LUMINANCE_BILATERAL_WORKGROUP_SIZE;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
  pass.end();

  // Readback uses the same stride as the upload — single source.
  const readbackBuffer = device.createBuffer({
    size: bytesPerRow * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  encoder.copyTextureToBuffer(
    { texture: texOut },
    { buffer: readbackBuffer, bytesPerRow },
    [w, h],
  );
  device.queue.submit([encoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const mapped = new Float32Array(readbackBuffer.getMappedRange());

  const out = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y += 1) {
    const rowOff = (y * bytesPerRow) / 4;
    for (let x = 0; x < w; x += 1) {
      const di = (y * w + x) * 3;
      const si = rowOff + x * 4;
      out[di] = mapped[si] ?? 0;
      out[di + 1] = mapped[si + 1] ?? 0;
      out[di + 2] = mapped[si + 2] ?? 0;
    }
  }
  readbackBuffer.unmap();

  texIn.destroy();
  texOut.destroy();
  ubo.destroy();
  readbackBuffer.destroy();
  destroyEphemeral?.();

  return out;
}
