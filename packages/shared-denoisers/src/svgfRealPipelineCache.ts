/**
 * svgfRealPipelineCache.ts — Per-GPUDevice cache of the four SVGF compute
 * pipelines (reproject, variance-from-moments, 7×7 fallback, à-trous).
 *
 * Compute pipelines are device-scoped resources; keeping them in a WeakMap
 * keyed by GPUDevice lets every `runSVGFRealWebGPU` call reuse the same
 * compiled shader modules without leaking when the device is destroyed.
 *
 * Extracted from svgfRealWebGPU.ts by the W4-A7 refactor (sweep H8).
 */

import { SVGF_REPROJECTION_WGSL } from './wgsl/svgfReprojection.wgsl.js';
import { SVGF_VARIANCE_FROM_MOMENTS_WGSL } from './wgsl/svgfVarianceFromMoments.wgsl.js';
import { SVGF_7X7_SPATIAL_FALLBACK_WGSL } from './wgsl/svgf7x7SpatialFallback.wgsl.js';
import { ATROUS_VARIANCE_WGSL } from './wgsl/atrousVariance.wgsl.js';

export interface SVGFRealPipelineBundle {
  readonly reprojPipeline:    GPUComputePipeline;
  readonly momentsPipeline:   GPUComputePipeline;
  readonly fallbackPipeline:  GPUComputePipeline;
  readonly atrousPipeline:    GPUComputePipeline;
}

const svgfRealPipelinesByDevice = new WeakMap<GPUDevice, SVGFRealPipelineBundle>();

/**
 * Returns the cached SVGF compute pipelines for `device`, compiling them on
 * first access. Subsequent calls for the same device are O(1).
 */
export function svgfRealPipelines(device: GPUDevice): SVGFRealPipelineBundle {
  let bundle = svgfRealPipelinesByDevice.get(device);
  if (bundle == null) {
    const reprojSM         = device.createShaderModule({ label: 'svgf-reproj',                 code: SVGF_REPROJECTION_WGSL });
    const momentsSM        = device.createShaderModule({ label: 'svgf-moments',                code: SVGF_VARIANCE_FROM_MOMENTS_WGSL });
    const fallbackSM       = device.createShaderModule({ label: 'svgf-7x7',                    code: SVGF_7X7_SPATIAL_FALLBACK_WGSL });
    const atrousVarianceSM = device.createShaderModule({ label: 'svgf-real-atrous-variance',   code: ATROUS_VARIANCE_WGSL });

    bundle = {
      reprojPipeline: device.createComputePipeline({
        label: 'svgf-real-reproj',
        layout: 'auto',
        compute: { module: reprojSM, entryPoint: 'svgfReprojMain' },
      }),
      momentsPipeline: device.createComputePipeline({
        label: 'svgf-real-moments',
        layout: 'auto',
        compute: { module: momentsSM, entryPoint: 'svgfVarianceFromMomentsMain' },
      }),
      fallbackPipeline: device.createComputePipeline({
        label: 'svgf-real-7x7',
        layout: 'auto',
        compute: { module: fallbackSM, entryPoint: 'svgf7x7FallbackMain' },
      }),
      atrousPipeline: device.createComputePipeline({
        label: 'svgf-real-atrous',
        layout: 'auto',
        compute: { module: atrousVarianceSM, entryPoint: 'svgfAtrousMain' },
      }),
    };
    svgfRealPipelinesByDevice.set(device, bundle);
  }
  return bundle;
}
