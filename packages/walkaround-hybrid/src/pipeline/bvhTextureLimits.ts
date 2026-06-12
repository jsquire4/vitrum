const WEBGPU_MAX_TEXTURE_DIMENSION_2D_FLOOR = 8192;

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
): void {
  const maxDim = maxTextureDimension2D(device);
  if (width <= maxDim && height <= maxDim) return;
  throw new RangeError(
    `[vitrum/walkaround-hybrid] ${kind} texture requires ${width}x${height} texels ` +
    `for ${normalizeBvhTextureTriangleCount(triCount)} triangles, which exceeds ` +
    `device.limits.maxTextureDimension2D=${maxDim}. Reduce triangle count or split ` +
    'the scene before creating walkaround-hybrid BVH textures.',
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
