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

import type {
  MaterialTextureAtlasGpuSourceLayer,
  MaterialTextureAtlasPayload,
} from '../bvh/materialTextureAtlasPack.js';
import { isWalkaroundWebGpuTextureSource } from '../materialTextureSource.js';

export {
  BASE_COLOR_MAP_META_TEX_WIDTH,
  MATERIAL_MAP_META_TEXELS_PER_TRI,
  MATERIAL_MAP_META_TEXEL_OFFSETS,
  packMaterialTextureAtlas,
  type AtlasMapField,
  type AtlasColorSpace,
  type MaterialTextureAtlasDiagnostic,
  type MaterialTextureAtlasGpuSourceLayer,
  type MaterialTextureAtlasPayload,
} from '../bvh/materialTextureAtlasPack.js';

const TEX_BINDING = 0x04;
const COPY_DST = 0x02;
const STORAGE_BINDING = 0x08;
const UNIFORM_BUFFER = 0x40;
const COMPUTE_STAGE = 0x04;

const GPU_ATLAS_SOURCE_FORMATS: ReadonlySet<string> = new Set([
  'r8unorm',
  'r8snorm',
  'rg8unorm',
  'rg8snorm',
  'rgba8unorm',
  'rgba8unorm-srgb',
  'rgba8snorm',
  'bgra8unorm',
  'bgra8unorm-srgb',
  'r16float',
  'rg16float',
  'rgba16float',
  'r32float',
  'rg32float',
  'rgba32float',
  'rgb10a2unorm',
  'rg11b10ufloat',
  'rgb9e5ufloat',
  'rgba16unorm',
  'rgba16snorm',
]);

export const MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL = /* wgsl */ `
struct CopyParams {
  targetLayer: u32,
  decodeSrgb: u32,
  _pad: vec2u,
};

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var targetTexture: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: CopyParams;

fn srgbChannelToLinear(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  if (c <= 0.04045) {
    return c / 12.92;
  }
  return pow((c + 0.055) / 1.055, 2.4);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let targetSize = textureDimensions(targetTexture);
  if (gid.x >= targetSize.x || gid.y >= targetSize.y) {
    return;
  }
  let sourceSize = textureDimensions(sourceTexture);
  let sourceCoord = min(
    vec2u((vec2f(gid.xy) + vec2f(0.5)) * vec2f(sourceSize) / vec2f(targetSize)),
    sourceSize - vec2u(1u),
  );
  var value = textureLoad(sourceTexture, vec2i(sourceCoord), 0);
  if (params.decodeSrgb != 0u) {
    value = vec4f(
      srgbChannelToLinear(value.r),
      srgbChannelToLinear(value.g),
      srgbChannelToLinear(value.b),
      value.a,
    );
  }
  textureStore(targetTexture, vec2i(gid.xy), i32(params.targetLayer), value);
}
`;

export const MATERIAL_ATLAS_GENERATE_MIP_WGSL = /* wgsl */ `
@group(0) @binding(0) var sourceTexture: texture_2d_array<f32>;
@group(0) @binding(1) var targetTexture: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let targetSize = textureDimensions(targetTexture);
  if (gid.x >= targetSize.x || gid.y >= targetSize.y) {
    return;
  }
  let sourceSize = textureDimensions(sourceTexture);
  let base = gid.xy * 2u;
  let maxCoord = sourceSize - vec2u(1u);
  let p00 = min(base, maxCoord);
  let p10 = min(base + vec2u(1u, 0u), maxCoord);
  let p01 = min(base + vec2u(0u, 1u), maxCoord);
  let p11 = min(base + vec2u(1u, 1u), maxCoord);
  let layer = i32(gid.z);
  let value = 0.25 * (
    textureLoad(sourceTexture, vec2i(p00), layer, 0) +
    textureLoad(sourceTexture, vec2i(p10), layer, 0) +
    textureLoad(sourceTexture, vec2i(p01), layer, 0) +
    textureLoad(sourceTexture, vec2i(p11), layer, 0)
  );
  textureStore(targetTexture, vec2i(gid.xy), layer, value);
}
`;

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
  validateMaterialTextureAtlasUpload(device, payload);
  const atlasTexture = device.createTexture({
    label: 'vitrum.materialTextureAtlas.baseColor.rgba32float-array',
    size: {
      width: payload.atlasDim,
      height: payload.atlasDim,
      depthOrArrayLayers: payload.atlasLayerCount,
    },
    mipLevelCount: payload.atlasMipLevelCount,
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST | STORAGE_BINDING,
  });
  let baseColorMetaTexture: GPUTexture | null = null;
  const transientBuffers: GPUBuffer[] = [];
  try {
    writeRgba32FloatTexture(
      device,
      atlasTexture,
      payload.atlasData,
      payload.atlasDim,
      payload.atlasDim,
      payload.atlasLayerCount,
    );

    baseColorMetaTexture = device.createTexture({
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

    if (payload.gpuSourceLayers.length > 0 || payload.atlasMipLevelCount > 1) {
      const encoder = device.createCommandEncoder({
        label: 'vitrum.materialTextureAtlas.upload-and-mips',
      });

      if (payload.gpuSourceLayers.length > 0) {
        const copyBindGroupLayout = device.createBindGroupLayout({
          label: 'vitrum.materialTextureAtlas.gpu-source-convert.bindings',
          entries: [
            {
              binding: 0,
              visibility: COMPUTE_STAGE,
              texture: {
                sampleType: 'unfilterable-float',
                viewDimension: '2d',
                multisampled: false,
              },
            },
            {
              binding: 1,
              visibility: COMPUTE_STAGE,
              storageTexture: {
                access: 'write-only',
                format: 'rgba32float',
                viewDimension: '2d-array',
              },
            },
            {
              binding: 2,
              visibility: COMPUTE_STAGE,
              buffer: { type: 'uniform' },
            },
          ],
        });
        const copyPipeline = device.createComputePipeline({
          label: 'vitrum.materialTextureAtlas.gpu-source-convert',
          layout: device.createPipelineLayout({
            label: 'vitrum.materialTextureAtlas.gpu-source-convert.pipeline-layout',
            bindGroupLayouts: [copyBindGroupLayout],
          }),
          compute: {
            module: device.createShaderModule({
              label: 'vitrum.materialTextureAtlas.gpu-source-convert.wgsl',
              code: MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
            }),
            entryPoint: 'main',
          },
        });
        const targetMipZeroView = atlasTexture.createView({
          label: 'vitrum.materialTextureAtlas.mip0.storage',
          dimension: '2d-array',
          baseMipLevel: 0,
          mipLevelCount: 1,
          baseArrayLayer: 0,
          arrayLayerCount: payload.atlasLayerCount,
        });
        for (const entry of payload.gpuSourceLayers) {
          const paramsBuffer = createCopyParamsBuffer(device, entry);
          transientBuffers.push(paramsBuffer);
          const sourceView = entry.source.texture.createView({
            label: `vitrum.materialTextureAtlas.source.${entry.layer}`,
            dimension: '2d',
            baseMipLevel: entry.source.baseMipLevel,
            mipLevelCount: 1,
            baseArrayLayer: entry.source.arrayLayer,
            arrayLayerCount: 1,
          });
          const bindGroup = device.createBindGroup({
            label: `vitrum.materialTextureAtlas.gpu-source.${entry.layer}`,
            layout: copyBindGroupLayout,
            entries: [
              { binding: 0, resource: sourceView },
              { binding: 1, resource: targetMipZeroView },
              { binding: 2, resource: { buffer: paramsBuffer, size: 16 } },
            ],
          });
          const pass = encoder.beginComputePass({
            label: `vitrum.materialTextureAtlas.gpu-source.${entry.layer}`,
          });
          pass.setPipeline(copyPipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(
            Math.ceil(payload.atlasDim / 8),
            Math.ceil(payload.atlasDim / 8),
            1,
          );
          pass.end();
        }
      }

      if (payload.atlasMipLevelCount > 1) {
        const mipBindGroupLayout = device.createBindGroupLayout({
          label: 'vitrum.materialTextureAtlas.generate-mips.bindings',
          entries: [
            {
              binding: 0,
              visibility: COMPUTE_STAGE,
              texture: {
                sampleType: 'unfilterable-float',
                viewDimension: '2d-array',
                multisampled: false,
              },
            },
            {
              binding: 1,
              visibility: COMPUTE_STAGE,
              storageTexture: {
                access: 'write-only',
                format: 'rgba32float',
                viewDimension: '2d-array',
              },
            },
          ],
        });
        const mipPipeline = device.createComputePipeline({
          label: 'vitrum.materialTextureAtlas.generate-mips',
          layout: device.createPipelineLayout({
            label: 'vitrum.materialTextureAtlas.generate-mips.pipeline-layout',
            bindGroupLayouts: [mipBindGroupLayout],
          }),
          compute: {
            module: device.createShaderModule({
              label: 'vitrum.materialTextureAtlas.generate-mips.wgsl',
              code: MATERIAL_ATLAS_GENERATE_MIP_WGSL,
            }),
            entryPoint: 'main',
          },
        });
        for (let mipLevel = 1; mipLevel < payload.atlasMipLevelCount; mipLevel += 1) {
          const sourceView = atlasTexture.createView({
            label: `vitrum.materialTextureAtlas.mip${mipLevel - 1}.sampled`,
            dimension: '2d-array',
            baseMipLevel: mipLevel - 1,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: payload.atlasLayerCount,
          });
          const targetView = atlasTexture.createView({
            label: `vitrum.materialTextureAtlas.mip${mipLevel}.storage`,
            dimension: '2d-array',
            baseMipLevel: mipLevel,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: payload.atlasLayerCount,
          });
          const bindGroup = device.createBindGroup({
            label: `vitrum.materialTextureAtlas.generate-mip.${mipLevel}`,
            layout: mipBindGroupLayout,
            entries: [
              { binding: 0, resource: sourceView },
              { binding: 1, resource: targetView },
            ],
          });
          const pass = encoder.beginComputePass({
            label: `vitrum.materialTextureAtlas.generate-mip.${mipLevel}`,
          });
          pass.setPipeline(mipPipeline);
          pass.setBindGroup(0, bindGroup);
          const mipDimension = Math.max(1, payload.atlasDim >> mipLevel);
          pass.dispatchWorkgroups(
            Math.ceil(mipDimension / 8),
            Math.ceil(mipDimension / 8),
            payload.atlasLayerCount,
          );
          pass.end();
        }
      }

      device.queue.submit([encoder.finish()]);
    }

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
  } catch (error) {
    atlasTexture.destroy();
    baseColorMetaTexture?.destroy();
    throw error;
  } finally {
    for (const buffer of transientBuffers) buffer.destroy();
  }
}

function createCopyParamsBuffer(
  device: GPUDevice,
  entry: MaterialTextureAtlasGpuSourceLayer,
): GPUBuffer {
  const buffer = device.createBuffer({
    label: `vitrum.materialTextureAtlas.gpu-source.${entry.layer}.params`,
    size: 16,
    usage: UNIFORM_BUFFER,
    mappedAtCreation: true,
  });
  try {
    new Uint32Array(buffer.getMappedRange()).set([
      entry.layer,
      entry.decodeSrgb ? 1 : 0,
      0,
      0,
    ]);
    buffer.unmap();
    return buffer;
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer; received ${String(value)}.`);
  }
}

function validateMaterialTextureAtlasUpload(
  device: GPUDevice,
  payload: MaterialTextureAtlasPayload,
): void {
  assertPositiveInteger(payload.atlasDim, 'material atlas dimension');
  assertPositiveInteger(payload.atlasLayerCount, 'material atlas layer count');
  assertPositiveInteger(payload.atlasMipLevelCount, 'material atlas mip count');
  assertPositiveInteger(payload.baseColorMetaWidth, 'material atlas metadata width');
  assertPositiveInteger(payload.baseColorMetaHeight, 'material atlas metadata height');

  const expectedMipCount = Math.floor(Math.log2(payload.atlasDim)) + 1;
  if (payload.atlasMipLevelCount !== expectedMipCount) {
    throw new RangeError(
      `material atlas mip count ${payload.atlasMipLevelCount} does not match ` +
      `${expectedMipCount} levels required by dimension ${payload.atlasDim}.`,
    );
  }
  const expectedAtlasValues = payload.atlasDim * payload.atlasDim * 4 * payload.atlasLayerCount;
  if (!Number.isSafeInteger(expectedAtlasValues) || payload.atlasData.length !== expectedAtlasValues) {
    throw new RangeError(
      `material atlas base data has ${payload.atlasData.length} values; expected ` +
      `${String(expectedAtlasValues)}.`,
    );
  }
  const expectedMetaValues = payload.baseColorMetaWidth * payload.baseColorMetaHeight * 4;
  if (!Number.isSafeInteger(expectedMetaValues) || payload.baseColorMetaData.length !== expectedMetaValues) {
    throw new RangeError(
      `material atlas metadata has ${payload.baseColorMetaData.length} values; expected ` +
      `${String(expectedMetaValues)}.`,
    );
  }

  const limits = device.limits as GPUSupportedLimits | undefined;
  if (
    Number.isFinite(limits?.maxTextureDimension2D) &&
    payload.atlasDim > (limits?.maxTextureDimension2D ?? Number.POSITIVE_INFINITY)
  ) {
    throw new RangeError(
      `material atlas dimension ${payload.atlasDim} exceeds maxTextureDimension2D ` +
      `${String(limits?.maxTextureDimension2D)}.`,
    );
  }
  if (
    Number.isFinite(limits?.maxTextureArrayLayers) &&
    payload.atlasLayerCount > (limits?.maxTextureArrayLayers ?? Number.POSITIVE_INFINITY)
  ) {
    throw new RangeError(
      `material atlas layer count ${payload.atlasLayerCount} exceeds maxTextureArrayLayers ` +
      `${String(limits?.maxTextureArrayLayers)}.`,
    );
  }

  const occupiedLayers = new Set<number>();
  for (const entry of payload.gpuSourceLayers) {
    if (!Number.isSafeInteger(entry.layer) || entry.layer < 0 || entry.layer >= payload.atlasLayerCount) {
      throw new RangeError(
        `material atlas GPU source layer ${String(entry.layer)} is outside ` +
        `${payload.atlasLayerCount} atlas layers.`,
      );
    }
    if (occupiedLayers.has(entry.layer)) {
      throw new RangeError(`material atlas GPU source layer ${entry.layer} is assigned more than once.`);
    }
    occupiedLayers.add(entry.layer);

    const source = entry.source;
    if (!isWalkaroundWebGpuTextureSource(source)) {
      throw new TypeError('material atlas GPU source is not a nominal walkaround texture descriptor.');
    }
    if (source.device !== device) {
      throw new TypeError(
        `material atlas GPU source layer ${entry.layer} belongs to a different GPUDevice.`,
      );
    }
    if (source.ownership !== 'host' || source.dimension !== '2d') {
      throw new TypeError(
        `material atlas GPU source layer ${entry.layer} has invalid ownership or dimension metadata.`,
      );
    }
    if (source.texture.format !== source.format) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} declares ${source.format}, but the ` +
        `GPUTexture reports ${source.texture.format}.`,
      );
    }
    if (!GPU_ATLAS_SOURCE_FORMATS.has(source.format)) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} uses unsupported format ${source.format}.`,
      );
    }
    if (source.texture.dimension !== '2d' || source.texture.sampleCount !== 1) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} must be a non-multisampled 2d texture.`,
      );
    }
    if ((source.texture.usage & TEX_BINDING) === 0) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} lacks GPUTextureUsage.TEXTURE_BINDING.`,
      );
    }
    if (
      source.baseMipLevel >= source.texture.mipLevelCount ||
      source.arrayLayer >= source.texture.depthOrArrayLayers
    ) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} selects an out-of-range mip or array layer.`,
      );
    }
    const expectedWidth = Math.max(1, Math.floor(source.texture.width / (2 ** source.baseMipLevel)));
    const expectedHeight = Math.max(1, Math.floor(source.texture.height / (2 ** source.baseMipLevel)));
    if (source.width !== expectedWidth || source.height !== expectedHeight) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} dimensions ${source.width}x${source.height} ` +
        `do not match selected subresource ${expectedWidth}x${expectedHeight}.`,
      );
    }
    if (source.width > payload.atlasDim || source.height > payload.atlasDim) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} exceeds atlas dimension ${payload.atlasDim}.`,
      );
    }
    const nativeSrgb = source.format.endsWith('-srgb');
    if (nativeSrgb && source.colorSpace !== 'srgb') {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} declares linear values in native-sRGB ` +
        `format ${source.format}.`,
      );
    }
    const expectedDecodeSrgb = source.colorSpace === 'srgb' && !nativeSrgb;
    if (entry.decodeSrgb !== expectedDecodeSrgb) {
      throw new RangeError(
        `material atlas GPU source layer ${entry.layer} has inconsistent sRGB decode metadata.`,
      );
    }
  }
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
