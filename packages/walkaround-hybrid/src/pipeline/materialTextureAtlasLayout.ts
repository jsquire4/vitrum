import {
  MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER,
  MATERIAL_ATLAS_MAX_MIP_LEVELS,
  MATERIAL_META_MAX_EXACT_UINT,
  MATERIAL_META_HEADER_TEXELS,
  MATERIAL_TEXTURE_ATLAS_CPU_TRANSACTION_BUDGET_BYTES,
  type MaterialTextureAtlasLayer,
  type MaterialTextureAtlasPayload,
} from '../bvh/materialTextureAtlasPack.js';
import { materialTextureAtlasEncodingPlaneCount } from '../bvh/materialTextureAtlasCodec.js';

interface AtlasRect {
  readonly layer: number;
  readonly mipLevel: number;
  readonly width: number;
  readonly height: number;
  readonly planeCount: 1 | 2 | 4;
}

interface FreeRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface MaterialTextureAtlasMipPlacement {
  readonly layer: number;
  readonly mipLevel: number;
  readonly width: number;
  readonly height: number;
  readonly x: number;
  readonly y: number;
  readonly baseArrayLayer: number;
  readonly planeCount: 1 | 2 | 4;
}

export interface MaterialTextureAtlasLayoutPlan {
  readonly width: number;
  readonly height: number;
  readonly arrayLayerCount: number;
  /** Packed r32uint atlas texture bytes. */
  readonly atlasBytes: number;
  /** RGBA32F metadata texture bytes. */
  readonly metadataBytes: number;
  /** Steady candidate bytes: atlas + metadata. */
  readonly allocatedBytes: number;
  /** One reusable r32uint scratch mip-chain used for GPU-backed sources. */
  readonly uploadScratchBytes: number;
  /** Uniform-buffer descriptor bytes retained until the upload submit. */
  readonly uploadUniformBytes: number;
  /** Candidate-only transactional peak (steady + scratch + uniforms). */
  readonly candidatePeakBytes: number;
  readonly rawCodecBytes: number;
  readonly placements: readonly MaterialTextureAtlasMipPlacement[];
  /** Private clone with device-specific address records filled in. */
  readonly metadata: Float32Array;
}

export interface MaterialTextureAtlasLayoutOptions {
  /** Complete live CPU transaction ceiling; defaults to 512 MiB. */
  readonly maxCpuTransactionBytes?: number;
}

interface PackedGroup {
  readonly binCount: number;
  readonly placements: readonly GroupPlacement[];
}

interface GroupPlacement extends AtlasRect {
  readonly x: number;
  readonly y: number;
  readonly bin: number;
}

function mipDimension(base: number, level: number): number {
  return Math.max(1, Math.floor(base / (2 ** level)));
}

function alignTo256(value: number): number {
  return Math.ceil(value / 256) * 256;
}

function checkedCpuBytes(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(
      `Material atlas ${label} exceeds the safe integer byte domain.`,
    );
  }
  return value;
}

/**
 * Exact peak of the typed CPU allocations retained/live during layout and
 * upload: payload metadata + CPU codec planes, the device-address metadata
 * clone, one streaming mip generation, and one row-padding upload.
 */
export function estimateMaterialTextureAtlasCpuTransactionPeakBytes(
  payload: MaterialTextureAtlasPayload,
): number {
  const retainedBytes = checkedCpuBytes(
    payload.baseColorMetaData.byteLength +
      payload.atlasLayers.reduce(
        (sum, layer) =>
          sum + (layer.kind === 'cpu' ? layer.data.byteLength : 0),
        0,
      ),
    'retained CPU payload',
  );
  const metadataCloneBytes = payload.baseColorMetaData.byteLength;
  const baseBytes = checkedCpuBytes(
    retainedBytes + metadataCloneBytes,
    'metadata-clone peak',
  );
  let peak = baseBytes;

  for (const layer of payload.atlasLayers) {
    if (layer.kind !== 'cpu') continue;
    const planeCount = materialTextureAtlasEncodingPlaneCount(layer.encoding);
    let width = layer.width;
    let height = layer.height;
    let currentTransientBytes = 0;
    for (let mipLevel = 0; mipLevel < layer.mipLevelCount; mipLevel += 1) {
      const rowBytes = width * Uint32Array.BYTES_PER_ELEMENT;
      const paddedUploadBytes = alignTo256(rowBytes) === rowBytes
        ? 0
        : alignTo256(rowBytes) * height;
      peak = Math.max(
        peak,
        checkedCpuBytes(
          baseBytes + currentTransientBytes + paddedUploadBytes,
          'row-padding peak',
        ),
      );
      if (mipLevel + 1 >= layer.mipLevelCount) continue;
      const nextWidth = mipDimension(width, 1);
      const nextHeight = mipDimension(height, 1);
      const nextPixels = checkedCpuBytes(
        nextWidth * nextHeight,
        'generated mip pixel count',
      );
      const floatFilterBytes = checkedCpuBytes(
        nextPixels * 4 * Float32Array.BYTES_PER_ELEMENT,
        'generated mip float filter',
      );
      const nextPackedBytes = checkedCpuBytes(
        nextPixels * planeCount * Uint32Array.BYTES_PER_ELEMENT,
        'generated mip codec',
      );
      peak = Math.max(
        peak,
        checkedCpuBytes(
          baseBytes +
            currentTransientBytes +
            floatFilterBytes +
            nextPackedBytes,
          'mip-generation peak',
        ),
      );
      currentTransientBytes = nextPackedBytes;
      width = nextWidth;
      height = nextHeight;
    }
  }

  const metadataRowBytes =
    payload.baseColorMetaWidth * 4 * Float32Array.BYTES_PER_ELEMENT;
  const metadataPaddingBytes = alignTo256(metadataRowBytes) === metadataRowBytes
    ? 0
    : alignTo256(metadataRowBytes) * payload.baseColorMetaHeight;
  return Math.max(
    peak,
    checkedCpuBytes(
      baseBytes + metadataPaddingBytes,
      'metadata row-padding peak',
    ),
  );
}

function collectRects(layers: readonly MaterialTextureAtlasLayer[]): AtlasRect[] {
  const rects: AtlasRect[] = [];
  for (const layer of layers) {
    const planeCount = materialTextureAtlasEncodingPlaneCount(layer.encoding);
    for (let mipLevel = 0; mipLevel < layer.mipLevelCount; mipLevel += 1) {
      rects.push({
        layer: layer.layer,
        mipLevel,
        width: mipDimension(layer.width, mipLevel),
        height: mipDimension(layer.height, mipLevel),
        planeCount,
      });
    }
  }
  return rects;
}

function intersects(a: FreeRect, b: FreeRect): boolean {
  return (
    a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y
  );
}

function containedBy(a: FreeRect, b: FreeRect): boolean {
  return (
    a.x >= b.x &&
    a.y >= b.y &&
    a.x + a.width <= b.x + b.width &&
    a.y + a.height <= b.y + b.height
  );
}

function splitFreeRects(free: readonly FreeRect[], placed: FreeRect): FreeRect[] {
  const split: FreeRect[] = [];
  for (const rect of free) {
    if (!intersects(rect, placed)) {
      split.push(rect);
      continue;
    }
    const rectRight = rect.x + rect.width;
    const rectBottom = rect.y + rect.height;
    const placedRight = placed.x + placed.width;
    const placedBottom = placed.y + placed.height;
    if (placed.x > rect.x) {
      split.push({
        x: rect.x,
        y: rect.y,
        width: placed.x - rect.x,
        height: rect.height,
      });
    }
    if (placedRight < rectRight) {
      split.push({
        x: placedRight,
        y: rect.y,
        width: rectRight - placedRight,
        height: rect.height,
      });
    }
    if (placed.y > rect.y) {
      split.push({
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: placed.y - rect.y,
      });
    }
    if (placedBottom < rectBottom) {
      split.push({
        x: rect.x,
        y: placedBottom,
        width: rect.width,
        height: rectBottom - placedBottom,
      });
    }
  }
  return split.filter((candidate, index, all) => (
    candidate.width > 0 &&
    candidate.height > 0 &&
    !all.some((other, otherIndex) => (
      otherIndex !== index && containedBy(candidate, other)
    ))
  ));
}

function packGroup(
  rects: readonly AtlasRect[],
  pageWidth: number,
  pageHeight: number,
): PackedGroup | null {
  const sorted = [...rects].sort((a, b) => (
    b.width * b.height - a.width * a.height ||
    Math.max(b.width, b.height) - Math.max(a.width, a.height) ||
    a.layer - b.layer ||
    a.mipLevel - b.mipLevel
  ));
  const bins: FreeRect[][] = [];
  const placements: GroupPlacement[] = [];
  for (const rect of sorted) {
    if (rect.width > pageWidth || rect.height > pageHeight) return null;
    let best:
      | {
          readonly bin: number;
          readonly freeIndex: number;
          readonly areaWaste: number;
          readonly shortWaste: number;
          readonly y: number;
          readonly x: number;
        }
      | undefined;
    for (let bin = 0; bin < bins.length; bin += 1) {
      const free = bins[bin]!;
      for (let freeIndex = 0; freeIndex < free.length; freeIndex += 1) {
        const candidate = free[freeIndex]!;
        if (rect.width > candidate.width || rect.height > candidate.height) continue;
        const score = {
          bin,
          freeIndex,
          areaWaste: candidate.width * candidate.height - rect.width * rect.height,
          shortWaste: Math.min(
            candidate.width - rect.width,
            candidate.height - rect.height,
          ),
          y: candidate.y,
          x: candidate.x,
        };
        if (
          best == null ||
          score.areaWaste < best.areaWaste ||
          (score.areaWaste === best.areaWaste && score.shortWaste < best.shortWaste) ||
          (
            score.areaWaste === best.areaWaste &&
            score.shortWaste === best.shortWaste &&
            (
              score.bin < best.bin ||
              (score.bin === best.bin && (
                score.y < best.y ||
                (score.y === best.y && score.x < best.x)
              ))
            )
          )
        ) {
          best = score;
        }
      }
    }
    if (best == null) {
      best = {
        bin: bins.length,
        freeIndex: 0,
        areaWaste: pageWidth * pageHeight - rect.width * rect.height,
        shortWaste: Math.min(pageWidth - rect.width, pageHeight - rect.height),
        x: 0,
        y: 0,
      };
      bins.push([{ x: 0, y: 0, width: pageWidth, height: pageHeight }]);
    }
    const chosen = bins[best.bin]![best.freeIndex]!;
    const placed = {
      x: chosen.x,
      y: chosen.y,
      width: rect.width,
      height: rect.height,
    };
    bins[best.bin] = splitFreeRects(bins[best.bin]!, placed);
    placements.push({
      ...rect,
      x: placed.x,
      y: placed.y,
      bin: best.bin,
    });
  }
  return { binCount: bins.length, placements };
}

function candidateDimensions(rects: readonly AtlasRect[], maxDimension: number): number[] {
  if (rects.length === 0) return [1];
  const values = new Set<number>([1, maxDimension]);
  const widths = [...rects].map((rect) => rect.width).sort((a, b) => b - a);
  const heights = [...rects].map((rect) => rect.height).sort((a, b) => b - a);
  const maxRect = Math.max(widths[0]!, heights[0]!);
  values.add(widths[0]!);
  values.add(heights[0]!);
  let widthSum = 0;
  let heightSum = 0;
  for (let index = 0; index < Math.min(8, rects.length); index += 1) {
    widthSum = Math.min(maxDimension, widthSum + widths[index]!);
    heightSum = Math.min(maxDimension, heightSum + heights[index]!);
    values.add(widthSum);
    values.add(heightSum);
  }
  for (let multiplier = 2; multiplier <= 8; multiplier += 1) {
    values.add(Math.min(maxDimension, maxRect * multiplier));
  }
  const totalArea = rects.reduce((sum, rect) => sum + rect.width * rect.height, 0);
  const square = Math.min(maxDimension, Math.ceil(Math.sqrt(totalArea)));
  values.add(square);
  let powerOfTwo = 1;
  while (powerOfTwo < square && powerOfTwo < maxDimension) powerOfTwo *= 2;
  values.add(Math.min(maxDimension, powerOfTwo));
  return [...values]
    .filter((value) => Number.isSafeInteger(value) && value >= 1 && value <= maxDimension)
    .sort((a, b) => a - b);
}

function atlasAddressBaseTexel(metadata: Float32Array): number {
  const totalTexels = metadata.length / 4;
  const headerWord = (totalTexels - MATERIAL_META_HEADER_TEXELS) * 4;
  if (metadata[headerWord] !== 3) {
    throw new RangeError('Material texture metadata must use packed-atlas ABI v3.');
  }
  return exactMetadataUint(metadata[headerWord + 8], 'atlas address base');
}

function exactMetadataUint(value: number | undefined, label: string): number {
  if (
    value == null ||
    !Number.isInteger(value) ||
    value < 0 ||
    value > MATERIAL_META_MAX_EXACT_UINT
  ) {
    throw new RangeError(
      `Material texture metadata ${label} must be an exact non-negative binary32 integer.`,
    );
  }
  return value;
}

function writeAddressMetadata(
  payload: MaterialTextureAtlasPayload,
  placements: readonly MaterialTextureAtlasMipPlacement[],
): Float32Array {
  const metadata = new Float32Array(payload.baseColorMetaData);
  const base = atlasAddressBaseTexel(metadata);
  const placementByKey = new Map(
    placements.map((placement) => [
      `${placement.layer}:${placement.mipLevel}`,
      placement,
    ]),
  );
  for (const layer of payload.atlasLayers) {
    if (
      layer.mipLevelCount < 1 ||
      layer.mipLevelCount > MATERIAL_ATLAS_MAX_MIP_LEVELS
    ) {
      throw new RangeError(
        `Material texture atlas layer ${layer.layer} has an invalid mip count.`,
      );
    }
    const texel = base + layer.layer * MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER;
    const word = texel * 4;
    const planeCount = materialTextureAtlasEncodingPlaneCount(layer.encoding);
    metadata.set([
      exactMetadataUint(layer.encoding, 'encoding'),
      exactMetadataUint(layer.width, 'layer width'),
      exactMetadataUint(layer.height, 'layer height'),
      exactMetadataUint(layer.mipLevelCount, 'mip count'),
      layer.decodeSrgb ? 1 : 0,
      exactMetadataUint(planeCount, 'plane count'),
      0,
      0,
    ], word);
    for (let mipLevel = 0; mipLevel < layer.mipLevelCount; mipLevel += 1) {
      const placement = placementByKey.get(`${layer.layer}:${mipLevel}`);
      if (placement == null) {
        throw new Error(`Missing packed-atlas placement for layer ${layer.layer} mip ${mipLevel}.`);
      }
      metadata.set([
        exactMetadataUint(placement.x, 'mip x origin'),
        exactMetadataUint(placement.y, 'mip y origin'),
        exactMetadataUint(placement.baseArrayLayer, 'mip array layer'),
        0,
      ], word + (2 + mipLevel) * 4);
    }
  }
  return metadata;
}

export function planMaterialTextureAtlasLayout(
  payload: MaterialTextureAtlasPayload,
  limits: Pick<GPUSupportedLimits, 'maxTextureDimension2D' | 'maxTextureArrayLayers'>,
  options: MaterialTextureAtlasLayoutOptions = {},
): MaterialTextureAtlasLayoutPlan {
  const maxCpuTransactionBytes =
    options.maxCpuTransactionBytes ??
    MATERIAL_TEXTURE_ATLAS_CPU_TRANSACTION_BUDGET_BYTES;
  if (
    !Number.isSafeInteger(maxCpuTransactionBytes) ||
    maxCpuTransactionBytes < 0
  ) {
    throw new RangeError(
      'Material atlas CPU transaction budget must be a non-negative safe integer.',
    );
  }
  const cpuPeakBytes =
    estimateMaterialTextureAtlasCpuTransactionPeakBytes(payload);
  if (cpuPeakBytes > maxCpuTransactionBytes) {
    throw new RangeError(
      `Material atlas CPU transaction requires ${cpuPeakBytes} bytes, above ` +
      `the ${maxCpuTransactionBytes}-byte transaction budget.`,
    );
  }
  const maxDimension = Number(limits.maxTextureDimension2D);
  const maxArrayLayers = Number(limits.maxTextureArrayLayers);
  if (!Number.isSafeInteger(maxDimension) || maxDimension < 1) {
    throw new RangeError('Material atlas requires a positive maxTextureDimension2D limit.');
  }
  if (!Number.isSafeInteger(maxArrayLayers) || maxArrayLayers < 1) {
    throw new RangeError('Material atlas requires a positive maxTextureArrayLayers limit.');
  }
  const rects = collectRects(payload.atlasLayers);
  const metadataBytes = payload.baseColorMetaData.byteLength;
  const gpuLayers = payload.atlasLayers.filter(
    (layer): layer is Extract<MaterialTextureAtlasLayer, { kind: 'gpu' }> =>
      layer.kind === 'gpu',
  );
  const uploadUniformBytes = gpuLayers.reduce(
    (sum, layer) => sum + layer.mipLevelCount * 32,
    0,
  );
  const uploadScratchBytes = (() => {
    if (gpuLayers.length === 0) return 0;
    const width = Math.max(...gpuLayers.map((layer) => layer.width));
    const height = Math.max(...gpuLayers.map((layer) => layer.height));
    const mipLevelCount = Math.max(
      ...gpuLayers.map((layer) => layer.mipLevelCount),
    );
    const planeCount = Math.max(
      ...gpuLayers.map((layer) =>
        materialTextureAtlasEncodingPlaneCount(layer.encoding)),
    );
    let texels = 0;
    for (let level = 0; level < mipLevelCount; level += 1) {
      texels += mipDimension(width, level) * mipDimension(height, level);
    }
    return texels * planeCount * Uint32Array.BYTES_PER_ELEMENT;
  })();
  if (rects.length === 0) {
    const atlasBytes = Uint32Array.BYTES_PER_ELEMENT;
    const allocatedBytes = atlasBytes + metadataBytes;
    return {
      width: 1,
      height: 1,
      arrayLayerCount: 1,
      atlasBytes,
      metadataBytes,
      allocatedBytes,
      uploadScratchBytes,
      uploadUniformBytes,
      candidatePeakBytes:
        allocatedBytes + uploadScratchBytes + uploadUniformBytes,
      rawCodecBytes: 0,
      placements: [],
      metadata: new Float32Array(payload.baseColorMetaData),
    };
  }
  for (const rect of rects) {
    if (rect.width > maxDimension || rect.height > maxDimension) {
      throw new RangeError(
        `Material atlas layer ${rect.layer} mip ${rect.mipLevel} dimensions ` +
        `${rect.width}x${rect.height} exceed maxTextureDimension2D ${maxDimension}.`,
      );
    }
  }
  const candidates = candidateDimensions(rects, maxDimension);
  let best:
    | {
        readonly width: number;
        readonly height: number;
        readonly arrayLayerCount: number;
        readonly placements: readonly MaterialTextureAtlasMipPlacement[];
        readonly cost: number;
      }
    | undefined;
  for (const width of candidates) {
    for (const height of candidates) {
      const placements: MaterialTextureAtlasMipPlacement[] = [];
      let arrayLayerCount = 0;
      let valid = true;
      for (const planeCount of [1, 2, 4] as const) {
        const groupRects = rects.filter((rect) => rect.planeCount === planeCount);
        if (groupRects.length === 0) continue;
        const packed = packGroup(groupRects, width, height);
        if (packed == null) {
          valid = false;
          break;
        }
        const groupBaseLayer = arrayLayerCount;
        for (const placement of packed.placements) {
          placements.push({
            layer: placement.layer,
            mipLevel: placement.mipLevel,
            width: placement.width,
            height: placement.height,
            x: placement.x,
            y: placement.y,
            planeCount: placement.planeCount,
            baseArrayLayer: groupBaseLayer + placement.bin * planeCount,
          });
        }
        arrayLayerCount += packed.binCount * planeCount;
      }
      if (!valid || arrayLayerCount > maxArrayLayers) continue;
      const cost = width * height * arrayLayerCount;
      if (!Number.isSafeInteger(cost)) continue;
      if (
        best == null ||
        cost < best.cost ||
        (cost === best.cost && arrayLayerCount < best.arrayLayerCount) ||
        (
          cost === best.cost &&
          arrayLayerCount === best.arrayLayerCount &&
          (
            width * height < best.width * best.height ||
            (width * height === best.width * best.height && width < best.width)
          )
        )
      ) {
        best = { width, height, arrayLayerCount, placements, cost };
      }
    }
  }
  if (best == null) {
    throw new RangeError(
      `Material atlas cannot fit within maxTextureDimension2D=${maxDimension} and ` +
      `maxTextureArrayLayers=${maxArrayLayers}.`,
    );
  }
  const rawCodecBytes = rects.reduce(
    (sum, rect) => sum + rect.width * rect.height * rect.planeCount * 4,
    0,
  );
  const atlasBytes = best.cost * Uint32Array.BYTES_PER_ELEMENT;
  const allocatedBytes = atlasBytes + metadataBytes;
  return {
    width: best.width,
    height: best.height,
    arrayLayerCount: best.arrayLayerCount,
    atlasBytes,
    metadataBytes,
    allocatedBytes,
    uploadScratchBytes,
    uploadUniformBytes,
    candidatePeakBytes:
      allocatedBytes + uploadScratchBytes + uploadUniformBytes,
    rawCodecBytes,
    placements: best.placements,
    metadata: writeAddressMetadata(payload, best.placements),
  };
}
