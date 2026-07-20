const WEBGPU_MAX_TEXTURE_DIMENSION_2D_FLOOR = 8192;

/** `GPUTextureUsage.TEXTURE_BINDING | COPY_DST` — literals avoid a top-level
 *  `GPUTextureUsage` reference (Node vitest has no WebGPU globals). */
const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

/**
 * Create a `TEXTURE_BINDING | COPY_DST` 2D texture and write a fully-padded
 * `width×height` grid into it. Shared by the per-element walkaround texture
 * builders (beer / emissive / rough-metal / tangent / vertex-color / analytic
 * lights), which differ only in how they populate the padded typed array — the
 * `fill` callback owns that; this helper owns the identical create + writeTexture
 * skeleton (single source of truth for the usage flags, size shape, and
 * `bytesPerRow`/`rowsPerImage` computation).
 *
 * `PaddedArray` is a `Uint32Array` (r32uint) or `Float32Array` (rgba*float);
 * the caller supplies the constructor via `makePadded`. `bytesPerTexel`
 * MUST match the format (4 for r32uint, 16 for rgba32float) — it drives
 * `bytesPerRow`.
 */
export function uploadElementTexture<T extends Uint32Array | Float32Array>(
  device: GPUDevice,
  opts: {
    label: string;
    format: GPUTextureFormat;
    width: number;
    height: number;
    /** Bytes per texel for the format (r32uint → 4, rgba32float → 16). */
    bytesPerTexel: number;
    /** Allocate the zero-initialised padded backing array (width×height×N). */
    makePadded: (elementCount: number) => T;
    /** Number of typed-array elements per texel (1 for r32uint, 4 for rgba32float). */
    elementsPerTexel: number;
    /** Populate the padded array in place before upload. */
    fill: (padded: T) => void;
  },
): GPUTexture {
  const { label, format, width, height, bytesPerTexel, makePadded, elementsPerTexel, fill } = opts;
  const texture = device.createTexture({
    label,
    size: { width, height, depthOrArrayLayers: 1 },
    format,
    usage: TEX_BINDING | COPY_DST,
  });
  writeElementTexture(device, texture, { width, height, bytesPerTexel, makePadded, elementsPerTexel, fill });
  return texture;
}

/** Re-upload the full padded grid into an existing texture (refresh path). */
export function writeElementTexture<T extends Uint32Array | Float32Array>(
  device: GPUDevice,
  texture: GPUTexture,
  opts: {
    width: number;
    height: number;
    bytesPerTexel: number;
    makePadded: (elementCount: number) => T;
    elementsPerTexel: number;
    fill: (padded: T) => void;
  },
): void {
  const { width, height, bytesPerTexel, makePadded, elementsPerTexel, fill } = opts;
  const padded = makePadded(width * height * elementsPerTexel);
  fill(padded);
  device.queue.writeTexture(
    { texture },
    padded.buffer,
    { bytesPerRow: width * bytesPerTexel, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
}

export function normalizeBvhTextureTriangleCount(triCount: number): number {
  if (!Number.isFinite(triCount)) return 1;
  return Math.max(1, Math.floor(triCount));
}

export function maxTextureDimension2D(device: GPUDevice): number {
  const limit = device.limits?.maxTextureDimension2D;
  if (Number.isFinite(limit) && limit > 0) return Math.floor(limit);
  return WEBGPU_MAX_TEXTURE_DIMENSION_2D_FLOOR;
}

export function assertBvhTextureFitsDevice(
  kind: string,
  device: GPUDevice,
  width: number,
  height: number,
  triCount: number,
  // Noun for the per-texel element (triangles for BVH textures, vertices for
  // per-vertex streams like tangents/vertex-colors). Only affects the
  // diagnostic message wording, not the fit logic.
  unit = 'triangle',
  resourceNoun = 'BVH textures',
): void {
  const maxDim = maxTextureDimension2D(device);
  if (width <= maxDim && height <= maxDim) return;
  throw new RangeError(
    `[vitrum/walkaround-hybrid] ${kind} texture requires ${width}x${height} texels ` +
    `for ${normalizeBvhTextureTriangleCount(triCount)} ${unit}s, which exceeds ` +
    `device.limits.maxTextureDimension2D=${maxDim}. Reduce ${unit} count or split ` +
    `the scene before creating walkaround-hybrid ${resourceNoun}.`,
  );
}

export function assertBvhTextureRefreshCapacity(
  kind: string,
  width: number,
  height: number,
  triCount: number,
): void {
  const count = normalizeBvhTextureTriangleCount(triCount);
  const capacity = Math.max(0, width) * Math.max(0, height);
  if (count <= capacity) return;
  throw new RangeError(
    `[vitrum/walkaround-hybrid] ${kind} texture refresh needs ${count} triangles, ` +
    `but the existing ${width}x${height} texture only stores ${capacity}. Rebuild ` +
    'the BVH texture resources instead of refreshing in place.',
  );
}
