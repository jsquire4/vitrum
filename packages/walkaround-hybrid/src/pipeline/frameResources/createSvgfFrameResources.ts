/**
 * SVGF-real persistent textures + object-id placeholders (W4a split).
 *
 * Perf hygiene (G-P2.6): the ~10 full-resolution persistent textures here
 * (~80-90 MB @1080p — history/moments/prev-radiance/variance ping-pong pairs)
 * are read EXCLUSIVELY by `SVGFRealDenoiser.dispatch`, which only runs when the
 * active denoiser is `svgf-real`. When any other denoiser is selected (the
 * default `atrous-variance`, `oidn-final`, `neural`, `bmfr`, `none`, `atrous`)
 * these textures are never bound and the full-res allocation is pure waste.
 *
 * `svgfEnabled` gates the FULL-RES allocation: when false we still return the
 * exact same 11-field struct shape (the `FrameResources.svgf` contract — see
 * frameResourcesShape.test.ts) but every full-res texture collapses to a 1×1
 * placeholder of the SAME format/usage. The render is byte-identical because
 * nothing reads these fields off the svgf-real dispatch path; only the GPU
 * memory footprint drops. The two 1×1 object-id placeholders are always tiny
 * regardless, so they are allocated unconditionally.
 */

import type { SVGFFrameResources } from '../resourceManager.js';

export function createSvgfFrameResources(
  device: GPUDevice,
  width: number,
  height: number,
  /** Allocate the full-res persistent textures. When false (the active denoiser
   *  is not `svgf-real`) the full-res textures collapse to 1×1 placeholders —
   *  byte-identical render, ~80-90 MB @1080p reclaimed. Defaults to `true` so
   *  callers that omit it keep the legacy full-allocation behavior. */
  svgfEnabled = true,
): SVGFFrameResources {
  // When SVGF-real is not the active denoiser, the persistent history textures
  // are never bound — size them 1×1 to reclaim the full-res footprint while
  // preserving the struct shape + format/usage of every field.
  const w = svgfEnabled ? width : 1;
  const h = svgfEnabled ? height : 1;
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
    size: [w, h],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const svgfHistUsage =
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
  const svgfHistoryLengthTextureA = device.createTexture({
    label: 'svgf-real-history-length-a',
    size: [w, h],
    format: 'r32uint',
    usage: svgfHistUsage,
  });
  const svgfHistoryLengthTextureB = device.createTexture({
    label: 'svgf-real-history-length-b',
    size: [w, h],
    format: 'r32uint',
    usage: svgfHistUsage,
  });
  {
    const bpr = Math.max(256, Math.ceil(w * 4 / 256) * 256);
    const zeroBuf = new Uint8Array(bpr * h);
    device.queue.writeTexture(
      { texture: svgfHistoryLengthTextureA },
      zeroBuf,
      { bytesPerRow: bpr },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: svgfHistoryLengthTextureB },
      zeroBuf,
      { bytesPerRow: bpr },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
  }

  const svgfMomUsage =
    GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
  const svgfMomentsTextureA = device.createTexture({
    label: 'svgf-real-moments-a',
    size: [w, h],
    format: 'rg32float',
    usage: svgfMomUsage,
  });
  const svgfMomentsTextureB = device.createTexture({
    label: 'svgf-real-moments-b',
    size: [w, h],
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
    size: [w, h],
    format: 'rgba16float',
    usage: svgfRadUsage,
  });
  const svgfPrevRadianceTextureB = device.createTexture({
    label: 'svgf-real-prev-radiance-b',
    size: [w, h],
    format: 'rgba16float',
    usage: svgfRadUsage,
  });
  const svgfVarianceTexture = device.createTexture({
    label: 'svgf-real-variance',
    size: [w, h],
    format: 'rg32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const svgfVarianceMomentsIntermedTexture = device.createTexture({
    label: 'svgf-real-variance-moments-intermed',
    size: [w, h],
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
