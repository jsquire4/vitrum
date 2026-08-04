/**
 * Device-aware GPU upload for the packed walkaround material atlas.
 *
 * Every logical map keeps its native width/height. The CPU packer selects a
 * fidelity-matched codec (one r32uint plane for 8-bit values, two for
 * 16-bit/half values, four for full floats). This module rectangle-packs only
 * the logical mip levels requested by authored sampler policy into a single
 * r32uint 2D-array binding. Per-map/mip addresses are written into a private
 * clone of the existing RGBA32F metadata texture before either GPU texture is
 * published.
 */

import type {
  MaterialTextureAtlasGpuSourceLayer,
  MaterialTextureAtlasLayer,
  MaterialTextureAtlasPayload,
} from '../bvh/materialTextureAtlasPack.js';
import {
  MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT,
  MATERIAL_ATLAS_ENCODING_RGBA16_SNORM,
  MATERIAL_ATLAS_ENCODING_RGBA16_UNORM,
  MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
  MATERIAL_ATLAS_ENCODING_RGBA8_SNORM,
  MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
  generateMaterialTextureAtlasMip,
  materialTextureAtlasEncodingForGpuFormat,
  materialTextureAtlasEncodingPlaneCount,
} from '../bvh/materialTextureAtlasCodec.js';
import {
  isWalkaroundWebGpuTextureSource,
  walkaroundTextureFormatChannelCount,
} from '../materialTextureSource.js';
import {
  planMaterialTextureAtlasLayout,
  type MaterialTextureAtlasMipPlacement,
} from './materialTextureAtlasLayout.js';

export {
  BASE_COLOR_MAP_META_TEX_WIDTH,
  MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER,
  MATERIAL_ATLAS_MAX_MIP_LEVELS,
  MATERIAL_META_HEADER_TEXELS,
  MATERIAL_META_MAX_EXACT_UINT,
  MATERIAL_TEXTURE_ATLAS_CPU_TRANSACTION_BUDGET_BYTES,
  MATERIAL_MAP_META_TEXELS_PER_TRI,
  MATERIAL_MAP_META_TEXEL_OFFSETS,
  materialTextureAtlasFingerprintParts,
  packMaterialTextureAtlas,
  type AtlasMapField,
  type AtlasColorSpace,
  type MaterialTextureAtlasCpuSourceLayer,
  type MaterialTextureAtlasDiagnostic,
  type MaterialTextureAtlasGpuBackedLayer,
  type MaterialTextureAtlasGpuSourceLayer,
  type MaterialTextureAtlasLayer,
  type MaterialTextureAtlasPayload,
} from '../bvh/materialTextureAtlasPack.js';
export {
  estimateMaterialTextureAtlasCpuTransactionPeakBytes,
  type MaterialTextureAtlasLayoutOptions,
} from './materialTextureAtlasLayout.js';
export {
  MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
  MATERIAL_ATLAS_ENCODING_RGBA8_SNORM,
  MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT,
  MATERIAL_ATLAS_ENCODING_RGBA16_UNORM,
  MATERIAL_ATLAS_ENCODING_RGBA16_SNORM,
  MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
  materialTextureAtlasEncodingPlaneCount,
  unpackMaterialTextureAtlasPixels,
  type MaterialTextureAtlasEncoding,
} from '../bvh/materialTextureAtlasCodec.js';
export {
  planMaterialTextureAtlasLayout,
  type MaterialTextureAtlasLayoutPlan,
  type MaterialTextureAtlasMipPlacement,
} from './materialTextureAtlasLayout.js';

const COPY_SRC = 0x01;
const COPY_DST = 0x02;
const TEX_BINDING = 0x04;
const STORAGE_BINDING = 0x08;
const UNIFORM_BUFFER = 0x40;
const COMPUTE_STAGE = 0x04;
export const DEFAULT_MATERIAL_ATLAS_TRANSACTION_BUDGET_BYTES =
  512 * 1024 * 1024;

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

const MATERIAL_ATLAS_CODEC_WGSL = /* wgsl */ `
const ATLAS_RGBA8_UNORM: u32 = ${MATERIAL_ATLAS_ENCODING_RGBA8_UNORM}u;
const ATLAS_RGBA8_SNORM: u32 = ${MATERIAL_ATLAS_ENCODING_RGBA8_SNORM}u;
const ATLAS_RGBA16_FLOAT: u32 = ${MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT}u;
const ATLAS_RGBA16_UNORM: u32 = ${MATERIAL_ATLAS_ENCODING_RGBA16_UNORM}u;
const ATLAS_RGBA16_SNORM: u32 = ${MATERIAL_ATLAS_ENCODING_RGBA16_SNORM}u;
const ATLAS_RGBA32_FLOAT: u32 = ${MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT}u;

fn atlasPackUnorm16(value: f32) -> u32 {
  return u32(round(clamp(value, 0.0, 1.0) * 65535.0));
}

fn atlasPackSnorm16(value: f32) -> u32 {
  return bitcast<u32>(i32(round(clamp(value, -1.0, 1.0) * 32767.0))) & 0xffffu;
}

fn atlasEncodePlane(value: vec4f, encoding: u32, plane: u32) -> u32 {
  if (encoding == ATLAS_RGBA8_UNORM) {
    return pack4x8unorm(value);
  }
  if (encoding == ATLAS_RGBA8_SNORM) {
    return pack4x8snorm(value);
  }
  if (encoding == ATLAS_RGBA16_FLOAT) {
    return select(pack2x16float(value.rg), pack2x16float(value.ba), plane != 0u);
  }
  if (encoding == ATLAS_RGBA16_UNORM) {
    let pair = select(value.rg, value.ba, plane != 0u);
    return atlasPackUnorm16(pair.x) | (atlasPackUnorm16(pair.y) << 16u);
  }
  if (encoding == ATLAS_RGBA16_SNORM) {
    let pair = select(value.rg, value.ba, plane != 0u);
    return atlasPackSnorm16(pair.x) | (atlasPackSnorm16(pair.y) << 16u);
  }
  return bitcast<u32>(value[plane]);
}

fn atlasSigned8(value: u32) -> i32 {
  let byte = value & 0xffu;
  return select(i32(byte), i32(byte) - 256, byte >= 128u);
}

fn atlasSigned16(value: u32) -> i32 {
  let word = value & 0xffffu;
  return select(i32(word), i32(word) - 65536, word >= 32768u);
}

fn atlasDecodePacked(
  source: texture_2d_array<u32>,
  coord: vec2i,
  encoding: u32,
  planeCount: u32,
) -> vec4f {
  let p0 = textureLoad(source, coord, 0, 0).r;
  if (encoding == ATLAS_RGBA8_UNORM) {
    return unpack4x8unorm(p0);
  }
  if (encoding == ATLAS_RGBA8_SNORM) {
    return unpack4x8snorm(p0);
  }
  let p1 = textureLoad(source, coord, 1, 0).r;
  if (encoding == ATLAS_RGBA16_FLOAT) {
    return vec4f(unpack2x16float(p0), unpack2x16float(p1));
  }
  if (encoding == ATLAS_RGBA16_UNORM) {
    return vec4f(
      f32(p0 & 0xffffu),
      f32(p0 >> 16u),
      f32(p1 & 0xffffu),
      f32(p1 >> 16u),
    ) / 65535.0;
  }
  if (encoding == ATLAS_RGBA16_SNORM) {
    return max(
      vec4f(
        f32(atlasSigned16(p0)),
        f32(atlasSigned16(p0 >> 16u)),
        f32(atlasSigned16(p1)),
        f32(atlasSigned16(p1 >> 16u)),
      ) / 32767.0,
      vec4f(-1.0),
    );
  }
  let p2 = textureLoad(source, coord, 2, 0).r;
  let p3 = textureLoad(source, coord, 3, 0).r;
  return vec4f(
    bitcast<f32>(p0),
    bitcast<f32>(p1),
    bitcast<f32>(p2),
    bitcast<f32>(p3),
  );
}

fn atlasSrgbToLinear(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(c / 12.92, pow((c + 0.055) / 1.055, 2.4), c > 0.04045);
}

fn atlasLinearToSrgb(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055, c > 0.0031308);
}

fn atlasDecodeSrgb(value: vec4f) -> vec4f {
  return vec4f(
    atlasSrgbToLinear(value.r),
    atlasSrgbToLinear(value.g),
    atlasSrgbToLinear(value.b),
    value.a,
  );
}

fn atlasEncodeSrgb(value: vec4f) -> vec4f {
  return vec4f(
    atlasLinearToSrgb(value.r),
    atlasLinearToSrgb(value.g),
    atlasLinearToSrgb(value.b),
    value.a,
  );
}
`;

export const MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL = /* wgsl */ `
${MATERIAL_ATLAS_CODEC_WGSL}

struct CopyParams {
  width: u32,
  height: u32,
  encoding: u32,
  planeCount: u32,
  encodeSrgb: u32,
  sourceChannels: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var targetTexture: texture_storage_2d_array<r32uint, write>;
@group(0) @binding(2) var<uniform> params: CopyParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  // Normalize WebGPU's native missing-channel defaults to the public raw-map
  // convention shared by CPU atlas sources and emitter classification:
  // R→RRR, RG→RG0, RGB→RGB, and alpha=1 whenever it was not authored.
  var value = textureLoad(sourceTexture, vec2i(gid.xy), 0);
  if (params.sourceChannels == 1u) {
    value = vec4f(value.rrr, 1.0);
  } else if (params.sourceChannels == 2u) {
    value = vec4f(value.rg, 0.0, 1.0);
  } else if (params.sourceChannels == 3u) {
    value = vec4f(value.rgb, 1.0);
  }
  // Native *-srgb sources are decoded by textureLoad. Re-encode before the
  // RGBA8 codec so later shader-side decode reconstructs the source's sampled
  // linear value instead of quantizing it into linear 8-bit space.
  if (params.encodeSrgb != 0u) {
    value = atlasEncodeSrgb(value);
  }
  for (var plane = 0u; plane < params.planeCount; plane = plane + 1u) {
    textureStore(
      targetTexture,
      vec2i(gid.xy),
      i32(plane),
      vec4u(atlasEncodePlane(value, params.encoding, plane), 0u, 0u, 0u),
    );
  }
}
`;

export const MATERIAL_ATLAS_GENERATE_MIP_WGSL = /* wgsl */ `
${MATERIAL_ATLAS_CODEC_WGSL}

struct MipParams {
  targetWidth: u32,
  targetHeight: u32,
  sourceWidth: u32,
  sourceHeight: u32,
  encoding: u32,
  planeCount: u32,
  decodeSrgb: u32,
  _pad0: u32,
};

@group(0) @binding(0) var sourceTexture: texture_2d_array<u32>;
@group(0) @binding(1) var targetTexture: texture_storage_2d_array<r32uint, write>;
@group(0) @binding(2) var<uniform> params: MipParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.targetWidth || gid.y >= params.targetHeight) {
    return;
  }
  let sourceSize = vec2u(params.sourceWidth, params.sourceHeight);
  let sourceStart = vec2f(gid.xy) * vec2f(sourceSize) /
    vec2f(f32(params.targetWidth), f32(params.targetHeight));
  let sourceEnd = vec2f(gid.xy + vec2u(1u)) * vec2f(sourceSize) /
    vec2f(f32(params.targetWidth), f32(params.targetHeight));
  let first = vec2u(floor(sourceStart));
  let last = vec2u(ceil(sourceEnd)) - vec2u(1u);
  var mean = vec4f(0.0);
  var weightSum = 0.0;
  for (var sy = first.y; sy <= last.y; sy = sy + 1u) {
    let wy = max(0.0, min(sourceEnd.y, f32(sy + 1u)) - max(sourceStart.y, f32(sy)));
    for (var sx = first.x; sx <= last.x; sx = sx + 1u) {
      let wx = max(0.0, min(sourceEnd.x, f32(sx + 1u)) - max(sourceStart.x, f32(sx)));
      let weight = wx * wy;
      var value = atlasDecodePacked(
        sourceTexture,
        vec2i(i32(sx), i32(sy)),
        params.encoding,
        params.planeCount,
      );
      if (params.decodeSrgb != 0u) {
        value = atlasDecodeSrgb(value);
      }
      let nextWeightSum = weightSum + weight;
      let blend = weight / max(nextWeightSum, 1e-8);
      mean = mix(mean, value, blend);
      weightSum = nextWeightSum;
    }
  }
  var value = mean;
  if (params.decodeSrgb != 0u) {
    value = atlasEncodeSrgb(value);
  }
  for (var plane = 0u; plane < params.planeCount; plane = plane + 1u) {
    textureStore(
      targetTexture,
      vec2i(gid.xy),
      i32(plane),
      vec4u(atlasEncodePlane(value, params.encoding, plane), 0u, 0u, 0u),
    );
  }
}
`;

export interface MaterialTextureAtlasGpu {
  readonly atlasTexture: GPUTexture;
  readonly atlasTextureView: GPUTextureView;
  readonly baseColorMetaTexture: GPUTexture;
  readonly baseColorMetaTextureView: GPUTextureView;
  readonly atlasWidth: number;
  readonly atlasHeight: number;
  readonly atlasArrayLayerCount: number;
  readonly baseColorMetaWidth: number;
  readonly baseColorMetaHeight: number;
  readonly atlasBytes: number;
  readonly metadataBytes: number;
  /** Steady atlas + metadata bytes. */
  readonly allocatedBytes: number;
  readonly uploadScratchBytes: number;
  readonly uploadUniformBytes: number;
  /** Peak for this candidate without a retained previous generation. */
  readonly candidatePeakBytes: number;
  /** Actual preflighted peak including caller-declared retained bytes. */
  readonly transactionalPeakBytes: number;
  readonly rawCodecBytes: number;
}

export interface MaterialTextureAtlasUploadOptions {
  /**
   * Bytes from a previous live atlas generation that remain allocated until
   * this candidate is fully published. Mutation paths pass the current
   * generation's `allocatedBytes` so the fail-closed budget covers both.
   */
  readonly retainedBytes?: number;
  /**
   * Number of simultaneously resident copies owned by the renderer. Hybrid
   * main shading + DDGI each retain one copy; RC borrows the main views.
   */
  readonly replicatedResidentCopies?: number;
  /** Defaults to 512 MiB. */
  readonly maxTransactionBytes?: number;
  /** Complete CPU staging/clone/mip peak; defaults to 512 MiB. */
  readonly maxCpuTransactionBytes?: number;
}

interface UploadPipelines {
  readonly copyLayout: GPUBindGroupLayout;
  readonly copyPipeline: GPUComputePipeline;
  readonly mipLayout: GPUBindGroupLayout;
  readonly mipPipeline: GPUComputePipeline;
}

export function uploadMaterialTextureAtlas(
  device: GPUDevice,
  payload: MaterialTextureAtlasPayload,
  options: MaterialTextureAtlasUploadOptions = {},
): MaterialTextureAtlasGpu {
  validateMaterialTextureAtlasPayload(device, payload);
  // Device-aware layout and metadata-address population are pure CPU work and
  // complete before the first candidate GPU resource is created.
  const plan = planMaterialTextureAtlasLayout(payload, device.limits, {
    ...(options.maxCpuTransactionBytes == null
      ? {}
      : { maxCpuTransactionBytes: options.maxCpuTransactionBytes }),
  });
  const retainedBytes = finiteNonNegativeInteger(
    options.retainedBytes ?? 0,
    'material atlas retained bytes',
  );
  const maxTransactionBytes = finiteNonNegativeInteger(
    options.maxTransactionBytes ??
      DEFAULT_MATERIAL_ATLAS_TRANSACTION_BUDGET_BYTES,
    'material atlas transaction budget',
  );
  const replicatedResidentCopies = finiteNonNegativeInteger(
    options.replicatedResidentCopies ?? 1,
    'material atlas replicated resident copies',
  );
  if (replicatedResidentCopies < 1) {
    throw new RangeError(
      'material atlas replicated resident copies must be at least one.',
    );
  }
  // Callers pass their own old generation in retainedBytes. Every additional
  // renderer-owned copy is the same logical payload but may still be on the
  // previous generation, so reserve the larger of old/current candidate size.
  const peerRetainedBytes =
    (replicatedResidentCopies - 1) *
    Math.max(retainedBytes, plan.allocatedBytes);
  const transactionalPeakBytes =
    retainedBytes + peerRetainedBytes + plan.candidatePeakBytes;
  if (
    !Number.isSafeInteger(transactionalPeakBytes) ||
    transactionalPeakBytes > maxTransactionBytes
  ) {
    throw new RangeError(
      `Material atlas transaction requires ${String(transactionalPeakBytes)} GPU bytes ` +
      `(${retainedBytes} retained + ${peerRetainedBytes} peer replicas + ` +
      `${plan.allocatedBytes} steady candidate + ` +
      `${plan.uploadScratchBytes} upload scratch + ${plan.uploadUniformBytes} uniforms), ` +
      `above the ${maxTransactionBytes}-byte transaction budget.`,
    );
  }
  const atlasTexture = device.createTexture({
    label: 'vitrum.materialTextureAtlas.packed.r32uint-array',
    size: {
      width: plan.width,
      height: plan.height,
      depthOrArrayLayers: plan.arrayLayerCount,
    },
    mipLevelCount: 1,
    format: 'r32uint',
    usage: TEX_BINDING | COPY_SRC | COPY_DST | STORAGE_BINDING,
  });
  let baseColorMetaTexture: GPUTexture | null = null;
  const transientBuffers: GPUBuffer[] = [];
  const transientTextures: GPUTexture[] = [];
  try {
    if (payload.atlasLayers.length === 0) {
      writeUint32PlaneRegion(
        device,
        atlasTexture,
        new Uint32Array([0xffffffff]),
        1,
        1,
        { x: 0, y: 0, baseArrayLayer: 0 },
      );
    }

    const placements = new Map(
      plan.placements.map((placement) => [
        `${placement.layer}:${placement.mipLevel}`,
        placement,
      ]),
    );
    const gpuLayers = payload.atlasLayers.filter(
      (layer): layer is Extract<MaterialTextureAtlasLayer, { kind: 'gpu' }> =>
        layer.kind === 'gpu',
    );
    let pipelines: UploadPipelines | null = null;
    let encoder: GPUCommandEncoder | null = null;
    let scratch: GPUTexture | null = null;
    if (gpuLayers.length > 0) {
      pipelines = createUploadPipelines(device);
      encoder = device.createCommandEncoder({
        label: 'vitrum.materialTextureAtlas.gpu-source-upload',
      });
      const scratchWidth = Math.max(...gpuLayers.map((layer) => layer.width));
      const scratchHeight = Math.max(...gpuLayers.map((layer) => layer.height));
      const scratchMipLevelCount = Math.max(
        ...gpuLayers.map((layer) => layer.mipLevelCount),
      );
      const scratchPlaneCount = Math.max(
        ...gpuLayers.map((layer) =>
          materialTextureAtlasEncodingPlaneCount(layer.encoding)),
      );
      scratch = device.createTexture({
        label: 'vitrum.materialTextureAtlas.gpu-source.scratch',
        size: {
          width: scratchWidth,
          height: scratchHeight,
          depthOrArrayLayers: scratchPlaneCount,
        },
        mipLevelCount: scratchMipLevelCount,
        format: 'r32uint',
        usage: TEX_BINDING | COPY_SRC | STORAGE_BINDING,
      });
      transientTextures.push(scratch);
    }

    for (const layer of payload.atlasLayers) {
      if (layer.kind === 'cpu') {
        uploadCpuLayer(device, atlasTexture, layer, placements);
      } else {
        uploadGpuLayer(
          device,
          encoder!,
          pipelines!,
          atlasTexture,
          scratch!,
          layer,
          placements,
          transientBuffers,
        );
      }
    }

    baseColorMetaTexture = device.createTexture({
      label: 'vitrum.materialTextureAtlas.metadata.rgba32float',
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
      plan.metadata,
      payload.baseColorMetaWidth,
      payload.baseColorMetaHeight,
    );

    if (encoder != null) {
      device.queue.submit([encoder.finish()]);
    }
    return {
      atlasTexture,
      atlasTextureView: atlasTexture.createView({ dimension: '2d-array' }),
      baseColorMetaTexture,
      baseColorMetaTextureView: baseColorMetaTexture.createView(),
      atlasWidth: plan.width,
      atlasHeight: plan.height,
      atlasArrayLayerCount: plan.arrayLayerCount,
      baseColorMetaWidth: payload.baseColorMetaWidth,
      baseColorMetaHeight: payload.baseColorMetaHeight,
      atlasBytes: plan.atlasBytes,
      metadataBytes: plan.metadataBytes,
      allocatedBytes: plan.allocatedBytes,
      uploadScratchBytes: plan.uploadScratchBytes,
      uploadUniformBytes: plan.uploadUniformBytes,
      candidatePeakBytes: plan.candidatePeakBytes,
      transactionalPeakBytes,
      rawCodecBytes: plan.rawCodecBytes,
    };
  } catch (error) {
    atlasTexture.destroy();
    baseColorMetaTexture?.destroy();
    throw error;
  } finally {
    for (const buffer of transientBuffers) destroyBestEffort(buffer);
    for (const texture of transientTextures) destroyBestEffort(texture);
  }
}

export function destroyMaterialTextureAtlasGpu(
  atlas: MaterialTextureAtlasGpu | null | undefined,
): void {
  destroyBestEffort(atlas?.atlasTexture);
  destroyBestEffort(atlas?.baseColorMetaTexture);
}

function destroyBestEffort(resource: { destroy(): void } | null | undefined): void {
  try {
    resource?.destroy();
  } catch {
    // Candidate retirement must continue even when a hostile/mock resource
    // throws from destroy().
  }
}

function createUploadPipelines(device: GPUDevice): UploadPipelines {
  const copyLayout = device.createBindGroupLayout({
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
          format: 'r32uint',
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
      bindGroupLayouts: [copyLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label: 'vitrum.materialTextureAtlas.gpu-source-convert.wgsl',
        code: MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
      }),
      entryPoint: 'main',
    },
  });
  const mipLayout = device.createBindGroupLayout({
    label: 'vitrum.materialTextureAtlas.generate-mips.bindings',
    entries: [
      {
        binding: 0,
        visibility: COMPUTE_STAGE,
        texture: {
          sampleType: 'uint',
          viewDimension: '2d-array',
          multisampled: false,
        },
      },
      {
        binding: 1,
        visibility: COMPUTE_STAGE,
        storageTexture: {
          access: 'write-only',
          format: 'r32uint',
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
  const mipPipeline = device.createComputePipeline({
    label: 'vitrum.materialTextureAtlas.generate-mips',
    layout: device.createPipelineLayout({
      label: 'vitrum.materialTextureAtlas.generate-mips.pipeline-layout',
      bindGroupLayouts: [mipLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label: 'vitrum.materialTextureAtlas.generate-mips.wgsl',
        code: MATERIAL_ATLAS_GENERATE_MIP_WGSL,
      }),
      entryPoint: 'main',
    },
  });
  return { copyLayout, copyPipeline, mipLayout, mipPipeline };
}

function placementFor(
  placements: ReadonlyMap<string, MaterialTextureAtlasMipPlacement>,
  layer: number,
  mipLevel: number,
): MaterialTextureAtlasMipPlacement {
  const placement = placements.get(`${layer}:${mipLevel}`);
  if (placement == null) {
    throw new Error(`Missing material atlas placement for layer ${layer} mip ${mipLevel}.`);
  }
  return placement;
}

function uploadCpuLayer(
  device: GPUDevice,
  atlasTexture: GPUTexture,
  layer: Extract<MaterialTextureAtlasLayer, { kind: 'cpu' }>,
  placements: ReadonlyMap<string, MaterialTextureAtlasMipPlacement>,
): void {
  let width = layer.width;
  let height = layer.height;
  let data = layer.data;
  for (let mipLevel = 0; mipLevel < layer.mipLevelCount; mipLevel += 1) {
    const placement = placementFor(placements, layer.layer, mipLevel);
    const pixelCount = width * height;
    for (let plane = 0; plane < placement.planeCount; plane += 1) {
      writeUint32PlaneRegion(
        device,
        atlasTexture,
        data.subarray(plane * pixelCount, (plane + 1) * pixelCount),
        width,
        height,
        {
          x: placement.x,
          y: placement.y,
          baseArrayLayer: placement.baseArrayLayer + plane,
        },
      );
    }
    if (mipLevel + 1 < layer.mipLevelCount) {
      const next = generateMaterialTextureAtlasMip(
        data,
        width,
        height,
        layer.encoding,
        layer.decodeSrgb,
      );
      width = next.width;
      height = next.height;
      data = next.data;
    }
  }
}

function uploadGpuLayer(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  pipelines: UploadPipelines,
  atlasTexture: GPUTexture,
  scratch: GPUTexture,
  layer: Extract<MaterialTextureAtlasLayer, { kind: 'gpu' }>,
  placements: ReadonlyMap<string, MaterialTextureAtlasMipPlacement>,
  transientBuffers: GPUBuffer[],
): void {
  const planeCount = materialTextureAtlasEncodingPlaneCount(layer.encoding);
  const copyParams = createParamsBuffer(device, [
    layer.width,
    layer.height,
    layer.encoding,
    planeCount,
    layer.source.format.endsWith('-srgb') ? 1 : 0,
    walkaroundTextureFormatChannelCount(layer.source.format),
    0,
    0,
  ], `vitrum.materialTextureAtlas.source.${layer.layer}.params`);
  transientBuffers.push(copyParams);
  const sourceView = layer.source.texture.createView({
    label: `vitrum.materialTextureAtlas.source.${layer.layer}`,
    dimension: '2d',
    baseMipLevel: layer.source.baseMipLevel,
    mipLevelCount: 1,
    baseArrayLayer: layer.source.arrayLayer,
    arrayLayerCount: 1,
  });
  const scratchBaseView = scratch.createView({
    dimension: '2d-array',
    baseMipLevel: 0,
    mipLevelCount: 1,
    baseArrayLayer: 0,
    arrayLayerCount: planeCount,
  });
  const copyBindGroup = device.createBindGroup({
    label: `vitrum.materialTextureAtlas.source.${layer.layer}.copy`,
    layout: pipelines.copyLayout,
    entries: [
      { binding: 0, resource: sourceView },
      { binding: 1, resource: scratchBaseView },
      { binding: 2, resource: { buffer: copyParams, size: 32 } },
    ],
  });
  const copyPass = encoder.beginComputePass({
    label: `vitrum.materialTextureAtlas.source.${layer.layer}.copy`,
  });
  copyPass.setPipeline(pipelines.copyPipeline);
  copyPass.setBindGroup(0, copyBindGroup);
  copyPass.dispatchWorkgroups(Math.ceil(layer.width / 8), Math.ceil(layer.height / 8), 1);
  copyPass.end();

  for (let mipLevel = 1; mipLevel < layer.mipLevelCount; mipLevel += 1) {
    const width = Math.max(1, Math.floor(layer.width / (2 ** mipLevel)));
    const height = Math.max(1, Math.floor(layer.height / (2 ** mipLevel)));
    const sourceWidth = Math.max(
      1,
      Math.floor(layer.width / (2 ** (mipLevel - 1))),
    );
    const sourceHeight = Math.max(
      1,
      Math.floor(layer.height / (2 ** (mipLevel - 1))),
    );
    const params = createParamsBuffer(device, [
      width,
      height,
      sourceWidth,
      sourceHeight,
      layer.encoding,
      planeCount,
      layer.decodeSrgb ? 1 : 0,
      0,
    ], `vitrum.materialTextureAtlas.source.${layer.layer}.mip${mipLevel}.params`);
    transientBuffers.push(params);
    const sourceMipView = scratch.createView({
      dimension: '2d-array',
      baseMipLevel: mipLevel - 1,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: planeCount,
    });
    const targetMipView = scratch.createView({
      dimension: '2d-array',
      baseMipLevel: mipLevel,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: planeCount,
    });
    const bindGroup = device.createBindGroup({
      label: `vitrum.materialTextureAtlas.source.${layer.layer}.mip${mipLevel}`,
      layout: pipelines.mipLayout,
      entries: [
        { binding: 0, resource: sourceMipView },
        { binding: 1, resource: targetMipView },
        { binding: 2, resource: { buffer: params, size: 32 } },
      ],
    });
    const pass = encoder.beginComputePass({
      label: `vitrum.materialTextureAtlas.source.${layer.layer}.mip${mipLevel}`,
    });
    pass.setPipeline(pipelines.mipPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
    pass.end();
  }

  for (let mipLevel = 0; mipLevel < layer.mipLevelCount; mipLevel += 1) {
    const placement = placementFor(placements, layer.layer, mipLevel);
    for (let plane = 0; plane < planeCount; plane += 1) {
      encoder.copyTextureToTexture(
        {
          texture: scratch,
          mipLevel,
          origin: { x: 0, y: 0, z: plane },
        },
        {
          texture: atlasTexture,
          mipLevel: 0,
          origin: {
            x: placement.x,
            y: placement.y,
            z: placement.baseArrayLayer + plane,
          },
        },
        {
          width: placement.width,
          height: placement.height,
          depthOrArrayLayers: 1,
        },
      );
    }
  }
}

function createParamsBuffer(
  device: GPUDevice,
  words: readonly number[],
  label: string,
): GPUBuffer {
  const byteLength = Math.max(16, words.length * Uint32Array.BYTES_PER_ELEMENT);
  const buffer = device.createBuffer({
    label,
    size: byteLength,
    usage: UNIFORM_BUFFER,
    mappedAtCreation: true,
  });
  try {
    new Uint32Array(buffer.getMappedRange()).set(words);
    buffer.unmap();
    return buffer;
  } catch (error) {
    destroyBestEffort(buffer);
    throw error;
  }
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer; received ${String(value)}.`);
  }
}

function finiteNonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(
      `${label} must be a non-negative safe integer; received ${String(value)}.`,
    );
  }
  return value;
}

function validateMaterialTextureAtlasPayload(
  device: GPUDevice,
  payload: MaterialTextureAtlasPayload,
): void {
  assertPositiveInteger(payload.baseColorMetaWidth, 'material atlas metadata width');
  assertPositiveInteger(payload.baseColorMetaHeight, 'material atlas metadata height');
  const expectedMetaValues = payload.baseColorMetaWidth * payload.baseColorMetaHeight * 4;
  if (
    !Number.isSafeInteger(expectedMetaValues) ||
    payload.baseColorMetaData.length !== expectedMetaValues
  ) {
    throw new RangeError(
      `material atlas metadata has ${payload.baseColorMetaData.length} values; expected ` +
      `${String(expectedMetaValues)}.`,
    );
  }
  const maxDimension = Number(device.limits.maxTextureDimension2D);
  if (
    payload.baseColorMetaWidth > maxDimension ||
    payload.baseColorMetaHeight > maxDimension
  ) {
    throw new RangeError(
      `material atlas metadata dimensions ${payload.baseColorMetaWidth}x` +
      `${payload.baseColorMetaHeight} exceed maxTextureDimension2D ${maxDimension}.`,
    );
  }
  const gpuByLayer = new Map(payload.gpuSourceLayers.map((entry) => [entry.layer, entry]));
  if (gpuByLayer.size !== payload.gpuSourceLayers.length) {
    throw new RangeError('material atlas GPU source layer is assigned more than once.');
  }
  payload.atlasLayers.forEach((layer, index) => {
    if (layer.layer !== index) {
      throw new RangeError(
        `material atlas logical layer ${layer.layer} is not contiguous at index ${index}.`,
      );
    }
    assertPositiveInteger(layer.width, `material atlas layer ${index} width`);
    assertPositiveInteger(layer.height, `material atlas layer ${index} height`);
    assertPositiveInteger(layer.mipLevelCount, `material atlas layer ${index} mip count`);
    const expectedMipCount = Math.floor(Math.log2(Math.max(layer.width, layer.height))) + 1;
    if (layer.mipLevelCount > expectedMipCount) {
      throw new RangeError(
        `material atlas layer ${index} mip count ${layer.mipLevelCount} exceeds ` +
        `${expectedMipCount} levels for ${layer.width}x${layer.height}.`,
      );
    }
    const planeCount = materialTextureAtlasEncodingPlaneCount(layer.encoding);
    if (layer.kind === 'cpu') {
      const expectedWords = layer.width * layer.height * planeCount;
      if (!Number.isSafeInteger(expectedWords) || layer.data.length !== expectedWords) {
        throw new RangeError(
          `material atlas CPU layer ${index} has ${layer.data.length} words; expected ` +
          `${String(expectedWords)}.`,
        );
      }
      if (gpuByLayer.has(index)) {
        throw new RangeError(`material atlas CPU layer ${index} also has a GPU source entry.`);
      }
      return;
    }
    const gpuEntry = gpuByLayer.get(index);
    if (
      gpuEntry == null ||
      gpuEntry.source !== layer.source ||
      gpuEntry.encoding !== layer.encoding ||
      gpuEntry.decodeSrgb !== layer.decodeSrgb ||
      gpuEntry.mipLevelCount !== layer.mipLevelCount
    ) {
      throw new RangeError(`material atlas GPU layer ${index} compatibility entry drifted.`);
    }
    validateGpuSource(device, gpuEntry);
  });
  if (
    payload.gpuSourceLayers.length !==
    payload.atlasLayers.filter((layer) => layer.kind === 'gpu').length
  ) {
    throw new RangeError('material atlas has orphaned GPU source entries.');
  }
}

function validateGpuSource(
  device: GPUDevice,
  entry: MaterialTextureAtlasGpuSourceLayer,
): void {
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
  if (entry.encoding !== materialTextureAtlasEncodingForGpuFormat(source.format)) {
    throw new RangeError(
      `material atlas GPU source layer ${entry.layer} codec does not match ${source.format}.`,
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
  const nativeSrgb = source.format.endsWith('-srgb');
  if (nativeSrgb && source.colorSpace !== 'srgb') {
    throw new RangeError(
      `material atlas GPU source layer ${entry.layer} declares linear values in native-sRGB ` +
      `format ${source.format}.`,
    );
  }
  const expectedDecodeSrgb = source.colorSpace === 'srgb';
  if (entry.decodeSrgb !== expectedDecodeSrgb) {
    throw new RangeError(
      `material atlas GPU source layer ${entry.layer} has inconsistent sRGB decode metadata.`,
    );
  }
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function gpuUploadBytes(
  data: ArrayBufferView<ArrayBufferLike>,
): Uint8Array<ArrayBuffer> {
  const buffer = data.buffer;
  if (buffer instanceof ArrayBuffer) {
    return new Uint8Array(buffer, data.byteOffset, data.byteLength);
  }
  const copy = new Uint8Array(data.byteLength);
  copy.set(new Uint8Array(buffer, data.byteOffset, data.byteLength));
  return copy;
}

function writeUint32PlaneRegion(
  device: GPUDevice,
  texture: GPUTexture,
  data: Uint32Array,
  width: number,
  height: number,
  origin: { readonly x: number; readonly y: number; readonly baseArrayLayer: number },
): void {
  if (data.length !== width * height) {
    throw new RangeError('Packed material atlas plane upload has an invalid word count.');
  }
  const rowBytes = width * Uint32Array.BYTES_PER_ELEMENT;
  const bytesPerRow = alignTo(rowBytes, 256);
  const source = gpuUploadBytes(data);
  const upload = bytesPerRow === rowBytes
    ? source
    : paddedRows(source, rowBytes, bytesPerRow, height);
  device.queue.writeTexture(
    {
      texture,
      mipLevel: 0,
      origin: { x: origin.x, y: origin.y, z: origin.baseArrayLayer },
    },
    upload,
    { bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
}

function paddedRows(
  source: Uint8Array<ArrayBuffer>,
  rowBytes: number,
  bytesPerRow: number,
  height: number,
): Uint8Array<ArrayBuffer> {
  const upload = new Uint8Array(bytesPerRow * height);
  for (let y = 0; y < height; y += 1) {
    upload.set(
      source.subarray(y * rowBytes, (y + 1) * rowBytes),
      y * bytesPerRow,
    );
  }
  return upload;
}

function writeRgba32FloatTexture(
  device: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  width: number,
  height: number,
): void {
  const rowBytes = width * 4 * Float32Array.BYTES_PER_ELEMENT;
  const bytesPerRow = alignTo(rowBytes, 256);
  const source = gpuUploadBytes(data);
  const upload = bytesPerRow === rowBytes
    ? source
    : paddedRows(source, rowBytes, bytesPerRow, height);
  device.queue.writeTexture(
    { texture },
    upload,
    { bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
}
