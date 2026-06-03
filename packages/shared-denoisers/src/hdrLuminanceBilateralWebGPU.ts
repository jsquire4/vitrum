/**
 * One-shot HDR bilateral denoise on CPU-backed linear RGB (post path trace).
 * Uses WebGPU compute — separate from the GL bilateral preview in Cornell.
 */

import {
  HDR_LUMINANCE_BILATERAL_ENTRY,
  HDR_LUMINANCE_BILATERAL_WGSL,
  HDR_LUMINANCE_BILATERAL_WORKGROUP_SIZE,
} from './wgsl/hdrLuminanceBilateral.wgsl.js';
import { acquireDenoiseDevice, makePerDevicePipelineCache } from './sharedWebGpuDevice.js';
import { uploadRgbAsRgba32f, readRgba32fToRgb } from './webGpuTextureUpload.js';

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
   * When true, uses a cached process-wide device via getSharedWebGPUDevice.
   * When false (default), allocates a dedicated device per call and destroys it afterward.
   */
  readonly reuseSharedWebGpuDevice?: boolean;
}

const bilateralComputePipeline = makePerDevicePipelineCache<GPUComputePipeline>(
  (device) => {
    const shaderModule = device.createShaderModule({ label: 'hdr-lum-bilateral', code: HDR_LUMINANCE_BILATERAL_WGSL });
    return device.createComputePipeline({
      label: 'hdr-lum-bilateral-pipeline',
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: HDR_LUMINANCE_BILATERAL_ENTRY },
    });
  },
);

/**
 * Runs one luminance bilateral pass on linear HDR RGB (flattened length w*h*3).
 * By default allocates an ephemeral device per call. Pass reuseSharedWebGpuDevice: true to reuse the cached process-wide device.
 */
export async function runHdrLuminanceBilateralWebGPU(
  opts: HdrLuminanceBilateralWebGPUOptions,
): Promise<Float32Array> {
  const { rgb, width: w, height: h } = opts;
  const sigma = opts.sigmaLuminance ?? HDR_LUMINANCE_BILATERAL_DEFAULT_SIGMA_LUMINANCE;
  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error('runHdrLuminanceBilateralWebGPU: WebGPU not available in this browser');
  }
  if (w <= 0 || h <= 0 || rgb.length < w * h * 3) {
    throw new Error('runHdrLuminanceBilateralWebGPU: invalid rgb buffer or dimensions');
  }

  const { device, dispose: destroyEphemeral } = await acquireDenoiseDevice({
    device: opts.device,
    reuseSharedWebGpuDevice: opts.reuseSharedWebGpuDevice,
    errorLabel: 'runHdrLuminanceBilateralWebGPU',
  });

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

  // Upload tight RGB as rgba32float (alpha=1) via the shared helper.
  uploadRgbAsRgba32f(device, texIn, rgb, w, h);

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

  device.queue.submit([encoder.finish()]);

  const out = await readRgba32fToRgb(device, texOut, w, h);

  texIn.destroy();
  texOut.destroy();
  ubo.destroy();
  destroyEphemeral();

  return out;
}
