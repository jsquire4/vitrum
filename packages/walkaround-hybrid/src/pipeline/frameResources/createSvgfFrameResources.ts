/**
 * SVGF-real persistent textures + object-id resources (W4a split).
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
 * memory footprint drops. Object-id textures follow the same size gate; shade
 * writes them through a dimension-guarded helper so the inactive 1×1 storage
 * texture remains a legal frame-layout placeholder.
 */

import type { SVGFFrameResources } from '../resourceManager.js';

function writeR32UintFill(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  value: number,
): void {
  const bytesPerRow = Math.max(256, Math.ceil((width * 4) / 256) * 256);
  const data = new Uint8Array(bytesPerRow * height);
  if (value !== 0) {
    const view = new DataView(data.buffer);
    for (let y = 0; y < height; y += 1) {
      const row = y * bytesPerRow;
      for (let x = 0; x < width; x += 1) {
        view.setUint32(row + x * 4, value, true);
      }
    }
  }
  device.queue.writeTexture(
    { texture },
    data,
    { offset: 0, bytesPerRow },
    { width, height, depthOrArrayLayers: 1 },
  );
}

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
  const svgfCurrentObjectIdTexture = device.createTexture({
    label: 'svgf-real-current-object-id',
    size: [w, h],
    format: 'r32uint',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  });
  const svgfPreviousObjectIdTexture = device.createTexture({
    label: 'svgf-real-previous-object-id',
    size: [w, h],
    format: 'r32uint',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  writeR32UintFill(device, svgfCurrentObjectIdTexture, w, h, 0);
  writeR32UintFill(device, svgfPreviousObjectIdTexture, w, h, 1);
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
    format: 'rgba32float',
    usage: svgfMomUsage,
  });
  const svgfMomentsTextureB = device.createTexture({
    label: 'svgf-real-moments-b',
    size: [w, h],
    format: 'rgba32float',
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
    format: 'rgba32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const svgfVarianceMomentsIntermedTexture = device.createTexture({
    label: 'svgf-real-variance-moments-intermed',
    size: [w, h],
    format: 'rgba32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  return {
    svgfCurrentObjectIdTexture,
    svgfPreviousObjectIdTexture,
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
