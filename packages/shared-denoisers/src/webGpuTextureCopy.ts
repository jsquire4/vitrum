/** WebGPU copy texture layout: bytes per row must be a multiple of this value (spec). */
export const WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT = 256 as const;

/** Row pitch for `copyTextureToBuffer` / `writeTexture` with tightly packed `widthPx * bytesPerPixel` rows. */
export function alignedTextureCopyBytesPerRow(widthPx: number, bytesPerPixel: number): number {
  const rowBytes = widthPx * bytesPerPixel;
  return Math.ceil(rowBytes / WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT) * WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT;
}
