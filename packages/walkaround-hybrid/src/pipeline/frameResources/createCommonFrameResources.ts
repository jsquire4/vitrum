/**
 * Cross-cutting HDR / temporal / variance / motion resources (W4a split).
 */

import type { CommonFrameResources } from '../resourceManager.js';
import { createVarianceBuffer } from '../resourceManager.js';
import { WALKAROUND_UBO_SIZE_BYTES } from '../constants.js';

export function createCommonFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
  options?: {
    /** Allocate the second full-res Welford side.
     *  Production pipelines keep this true for every denoiser so the shared
     *  variance tracker can drive adaptive sampling and FrameOutput. Direct
     *  resource-harness callers may still request a 1x1 placeholder. */
    readonly welfordPingPong?: boolean;
    /** Allocate the à-trous-only scalar variance estimate at full resolution.
     *  Other denoisers consume the shared Welford state directly and retain a
     *  1x1 placeholder for this otherwise-unused texture. */
    readonly atrousVarianceEstimate?: boolean;
    /** Allocate the checkerboard prefill's current-radiance snapshot at full resolution. */
    readonly checkerboardSnapshot?: boolean;
  },
): CommonFrameResources {
  const hdrColorTexture = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const hdrIndirectTexture = device.createTexture({
    label: 'hdrIndirect',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const combinedDenoisedTexture = device.createTexture({
    label: 'combinedDenoised',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const transparentCompositeTexture = device.createTexture({
    label: 'transparentComposite',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const hdrTotalTexture = device.createTexture({
    label: 'hdrTotal',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const indirectDenoisedPingTexture = device.createTexture({
    label: 'indirectDenoisedPing',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const indirectDenoisedPongTexture = device.createTexture({
    label: 'indirectDenoisedPong',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const indirectAccumPingTexture = device.createTexture({
    label: 'indirectAccumPing',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const indirectAccumPongTexture = device.createTexture({
    label: 'indirectAccumPong',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const gNormalDepthTexture = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });

  const denoisedPingTexture = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const denoisedPongTexture = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  const accumTextureA = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const accumTextureB = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  const uboBuffer = device.createBuffer({
    size: WALKAROUND_UBO_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const nearestSampler = device.createSampler({
    magFilter: 'nearest',
    minFilter: 'nearest',
  });

  const varianceBuffer = createVarianceBuffer(device, width, height);
  const fullWelfordPingPong = options?.welfordPingPong !== false;
  const varianceAuxW = fullWelfordPingPong ? width : 1;
  const varianceAuxH = fullWelfordPingPong ? height : 1;
  const varianceBufferAux = createVarianceBuffer(device, varianceAuxW, varianceAuxH);
  const fullAtrousVarianceEstimate =
    options?.atrousVarianceEstimate ?? fullWelfordPingPong;
  const atrousVarianceEstimateTexture = device.createTexture({
    label: 'atrous-variance-estimate',
    size: [
      fullAtrousVarianceEstimate ? width : 1,
      fullAtrousVarianceEstimate ? height : 1,
    ],
    format: 'rgba32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const motionVectorTexture = device.createTexture({
    label: 'motion-vectors-zero',
    size: [width, height],
    format: 'rgba32float',
    usage:
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING,
  });
  const rowBytes = 16 * width;
  const bytesPerRow = Math.max(256, Math.ceil(rowBytes / 256) * 256);
  const motionZero = new Uint8Array(bytesPerRow * height);
  device.queue.writeTexture(
    { texture: motionVectorTexture },
    motionZero,
    { offset: 0, bytesPerRow },
    { width, height, depthOrArrayLayers: 1 },
  );
  const checkerboardSnapshotW = options?.checkerboardSnapshot === true ? width : 1;
  const checkerboardSnapshotH = options?.checkerboardSnapshot === true ? height : 1;
  const checkerboardRadianceSnapshotTexture = device.createTexture({
    label: 'checkerboard-current-radiance-snapshot',
    size: [checkerboardSnapshotW, checkerboardSnapshotH],
    format: 'rgba16float',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });

  const tierTexture = device.createTexture({
    label: 'sample-tier',
    size: [width, height],
    format: 'r32uint',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const resolvedTexture = device.createTexture({
    label: 'resolved-radiance',
    size: [width, height],
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });

  const albedoTexture = device.createTexture({
    label: 'albedo-demodulation',
    size: [width, height],
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });

  return {
    hdrColorTexture,
    gNormalDepthTexture,
    denoisedPingTexture,
    denoisedPongTexture,
    accumTextureA,
    accumTextureB,
    uboBuffer,
    nearestSampler,
    motionVectorTexture,
    checkerboardRadianceSnapshotTexture,
    tierTexture,
    resolvedTexture,
    hdrTotalTexture,
    albedoTexture,
    hdrIndirectTexture,
    combinedDenoisedTexture,
    transparentCompositeTexture,
    indirectDenoisedPingTexture,
    indirectDenoisedPongTexture,
    indirectAccumPingTexture,
    indirectAccumPongTexture,
    varianceBuffer,
    varianceBufferAux,
    atrousVarianceEstimateTexture,
  };
}
