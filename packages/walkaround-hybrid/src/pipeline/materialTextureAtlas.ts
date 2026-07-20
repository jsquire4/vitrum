/**
 * GPU-upload half of the walkaround material-texture atlas.
 *
 * The pure-CPU pack half (`packMaterialTextureAtlas` + the meta layout
 * constants, `AtlasMapField`/`AtlasColorSpace` types,
 * `MaterialTextureAtlasDiagnostic`, `MaterialTextureAtlasPayload`) moved to
 * `../bvh/materialTextureAtlasPack.ts` (I3-2) so subsystem BVH builders can pack
 * an atlas without an upward `subsystem → pipeline` value edge. This module
 * keeps the `GPUDevice`-bound upload (`uploadMaterialTextureAtlas` +
 * `MaterialTextureAtlasGpu`) and re-exports the CPU half so existing
 * `from './materialTextureAtlas.js'` imports keep resolving.
 */

import type { MaterialTextureAtlasPayload } from '../bvh/materialTextureAtlasPack.js';

export {
  BASE_COLOR_MAP_META_TEX_WIDTH,
  MATERIAL_MAP_META_TEXELS_PER_TRI,
  MATERIAL_MAP_META_TEXEL_OFFSETS,
  packMaterialTextureAtlas,
  type AtlasMapField,
  type AtlasColorSpace,
  type MaterialTextureAtlasDiagnostic,
  type MaterialTextureAtlasPayload,
} from '../bvh/materialTextureAtlasPack.js';

const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

export interface MaterialTextureAtlasGpu {
  readonly atlasTexture: GPUTexture;
  readonly atlasTextureView: GPUTextureView;
  readonly baseColorMetaTexture: GPUTexture;
  readonly baseColorMetaTextureView: GPUTextureView;
  readonly atlasDim: number;
  readonly atlasLayerCount: number;
  readonly baseColorMetaWidth: number;
  readonly baseColorMetaHeight: number;
}

export function uploadMaterialTextureAtlas(
  device: GPUDevice,
  payload: MaterialTextureAtlasPayload,
): MaterialTextureAtlasGpu {
  const atlasTexture = device.createTexture({
    label: 'vitrum.materialTextureAtlas.baseColor.rgba32float-array',
    size: {
      width: payload.atlasDim,
      height: payload.atlasDim,
      depthOrArrayLayers: payload.atlasLayerCount,
    },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  writeRgba32FloatTexture(
    device,
    atlasTexture,
    payload.atlasData,
    payload.atlasDim,
    payload.atlasDim,
    payload.atlasLayerCount,
  );

  const baseColorMetaTexture = device.createTexture({
    label: 'vitrum.materialTextureAtlas.baseColorMeta.rgba32float',
    size: {
      width: payload.baseColorMetaWidth,
      height: payload.baseColorMetaHeight,
      depthOrArrayLayers: 1,
    },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  writeRgba32FloatTexture(
    device,
    baseColorMetaTexture,
    payload.baseColorMetaData,
    payload.baseColorMetaWidth,
    payload.baseColorMetaHeight,
    1,
  );

  return {
    atlasTexture,
    atlasTextureView: atlasTexture.createView({ dimension: '2d-array' }),
    baseColorMetaTexture,
    baseColorMetaTextureView: baseColorMetaTexture.createView(),
    atlasDim: payload.atlasDim,
    atlasLayerCount: payload.atlasLayerCount,
    baseColorMetaWidth: payload.baseColorMetaWidth,
    baseColorMetaHeight: payload.baseColorMetaHeight,
  };
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function writeRgba32FloatTexture(
  device: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  width: number,
  height: number,
  depthOrArrayLayers: number,
): void {
  const rowBytes = width * 4 * 4;
  const bytesPerRow = alignTo(rowBytes, 256);
  const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let upload: Uint8Array;
  if (bytesPerRow === rowBytes) {
    upload = source;
  } else {
    upload = new Uint8Array(bytesPerRow * height * depthOrArrayLayers);
    for (let layer = 0; layer < depthOrArrayLayers; layer += 1) {
      for (let y = 0; y < height; y += 1) {
        const srcOffset = (layer * height + y) * rowBytes;
        const dstOffset = (layer * height + y) * bytesPerRow;
        upload.set(source.subarray(srcOffset, srcOffset + rowBytes), dstOffset);
      }
    }
  }
  const uploadBuffer = upload.buffer.slice(upload.byteOffset, upload.byteOffset + upload.byteLength) as ArrayBuffer;
  device.queue.writeTexture(
    { texture },
    uploadBuffer,
    { bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers },
  );
}
