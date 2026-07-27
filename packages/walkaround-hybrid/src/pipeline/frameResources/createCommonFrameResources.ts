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
    /** Full-res Welford ping-pong + estimate textures are only needed by
     *  `atrous-variance`. Other denoisers keep legal 1x1 placeholders because
     *  SampleBudgetPass reads only `varianceBuffer` when no denoiser exposes a
     *  Welford ping index. Defaults to true for legacy direct callers. */
    readonly welfordPingPong?: boolean;
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

  const placeholderTexture = device.createTexture({
    size: [1, 1],
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const placeholderData = new Float32Array([0.5, 0.5, 1.0, 0.0]);
  device.queue.writeTexture({ texture: placeholderTexture }, placeholderData, { bytesPerRow: 16 }, [1, 1]);

  const uboBuffer = device.createBuffer({
    size: WALKAROUND_UBO_SIZE_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const nearestSampler = device.createSampler({
    magFilter: 'nearest',
    minFilter: 'nearest',
  });
  const compositeSampler = device.createSampler({
    magFilter: 'nearest',
    minFilter: 'nearest',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const varianceBuffer = createVarianceBuffer(device, width, height);
  const fullWelfordPingPong = options?.welfordPingPong !== false;
  const varianceAuxW = fullWelfordPingPong ? width : 1;
  const varianceAuxH = fullWelfordPingPong ? height : 1;
  const varianceBufferAux = createVarianceBuffer(device, varianceAuxW, varianceAuxH);
  const atrousVarianceEstimateTexture = device.createTexture({
    label: 'atrous-variance-estimate',
    size: [varianceAuxW, varianceAuxH],
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
    placeholderTexture,
    uboBuffer,
    nearestSampler,
    compositeSampler,
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
