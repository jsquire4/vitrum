/**
 * SVGF-real persistent textures + object-id placeholders (W4a split).
 */

import type { SVGFFrameResources } from '../resourceManager.js';

export function createSvgfFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
): SVGFFrameResources {
  const svgfObjIdPlaceholderTexture = device.createTexture({
    label: 'svgf-real-objid-placeholder',
    size: [1, 1],
    format: 'r32uint',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: svgfObjIdPlaceholderTexture },
    new Uint32Array([0]),
    { bytesPerRow: 4 },
    [1, 1],
  );
  const svgfPrevObjIdPlaceholderTexture = device.createTexture({
    label: 'svgf-real-prev-objid-placeholder',
    size: [1, 1],
    format: 'r32uint',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: svgfPrevObjIdPlaceholderTexture },
    new Uint32Array([1]),
    { bytesPerRow: 4 },
    [1, 1],
  );
  const svgfPrevNormalDepthTexture = device.createTexture({
    label: 'svgf-real-prev-normal-depth',
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const svgfHistUsage =
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
  const svgfHistoryLengthTextureA = device.createTexture({
    label: 'svgf-real-history-length-a',
    size: [width, height],
    format: 'r32uint',
    usage: svgfHistUsage,
  });
  const svgfHistoryLengthTextureB = device.createTexture({
    label: 'svgf-real-history-length-b',
    size: [width, height],
    format: 'r32uint',
    usage: svgfHistUsage,
  });
  {
    const bpr = Math.max(256, Math.ceil(width * 4 / 256) * 256);
    const zeroBuf = new Uint8Array(bpr * height);
    device.queue.writeTexture(
      { texture: svgfHistoryLengthTextureA },
      zeroBuf,
      { bytesPerRow: bpr },
      { width, height, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: svgfHistoryLengthTextureB },
      zeroBuf,
      { bytesPerRow: bpr },
      { width, height, depthOrArrayLayers: 1 },
    );
  }

  const svgfMomUsage =
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
  const svgfMomentsTextureA = device.createTexture({
    label: 'svgf-real-moments-a',
    size: [width, height],
    format: 'rg32float',
    usage: svgfMomUsage,
  });
  const svgfMomentsTextureB = device.createTexture({
    label: 'svgf-real-moments-b',
    size: [width, height],
    format: 'rg32float',
    usage: svgfMomUsage,
  });
  const svgfRadUsage =
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_DST |
    GPUTextureUsage.COPY_SRC;
  const svgfPrevRadianceTextureA = device.createTexture({
    label: 'svgf-real-prev-radiance-a',
    size: [width, height],
    format: 'rgba16float',
    usage: svgfRadUsage,
  });
  const svgfPrevRadianceTextureB = device.createTexture({
    label: 'svgf-real-prev-radiance-b',
    size: [width, height],
    format: 'rgba16float',
    usage: svgfRadUsage,
  });
  const svgfVarianceTexture = device.createTexture({
    label: 'svgf-real-variance',
    size: [width, height],
    format: 'rg32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const svgfVarianceMomentsIntermedTexture = device.createTexture({
    label: 'svgf-real-variance-moments-intermed',
    size: [width, height],
    format: 'rg32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  return {
    svgfObjIdPlaceholderTexture,
    svgfPrevObjIdPlaceholderTexture,
    svgfPrevNormalDepthTexture,
    svgfHistoryLengthTextureA,
    svgfHistoryLengthTextureB,
    svgfMomentsTextureA,
    svgfMomentsTextureB,
    svgfPrevRadianceTextureA,
    svgfPrevRadianceTextureB,
    svgfVarianceTexture,
    svgfVarianceMomentsIntermedTexture,
  };
}
