/** WebGPU copy texture layout: bytes per row must be a multiple of this value (spec).
 *  File-local — the public surface only exposes `alignedTextureCopyBytesPerRow`
 *  (verified zero external consumers in 2026-05-18 dead-code sweep). */
const WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT = 256 as const;

/** Row pitch for `copyTextureToBuffer` / `writeTexture` with tightly packed `widthPx * bytesPerPixel` rows. */
export function alignedTextureCopyBytesPerRow(widthPx: number, bytesPerPixel: number): number {
  const rowBytes = widthPx * bytesPerPixel;
  return Math.ceil(rowBytes / WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT) * WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT;
}
