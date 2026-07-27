/**
 * Canonical physical-transmission predicate for the compact walkaround BVH
 * material payload. Physical transmission is the independent 4-bit lane in
 * bits 7:4; alpha coverage/opacity lives in separate material-atlas metadata
 * and must never participate in this predicate.
 */
export const PACKED_MATERIAL_TRANSMISSION_SHIFT = 4;
export const PACKED_MATERIAL_TRANSMISSION_MASK = 0xF;

/**
 * Quantize a scalar physical-transmission factor into the compact 4-bit lane.
 * Zero is reserved exclusively for the non-transmissive class: every finite
 * positive input receives at least code 1, so quantization cannot silently
 * change a glass surface into an opaque one.
 */
export function quantizePackedMaterialTransmission(transmission: number): number {
  if (!Number.isFinite(transmission) || transmission <= 0) return 0;
  return Math.max(1, Math.min(
    PACKED_MATERIAL_TRANSMISSION_MASK,
    Math.round(Math.min(1, transmission) * PACKED_MATERIAL_TRANSMISSION_MASK),
  ));
}

export function packedMaterialHasTransmission(packed: number): boolean {
  return (
    (packed >>> PACKED_MATERIAL_TRANSMISSION_SHIFT) &
    PACKED_MATERIAL_TRANSMISSION_MASK
  ) !== 0;
}

export function buildMaterialTransmissionPredicatesWGSL(options: {
  readonly packedFunctionName?: string;
  readonly sampledFunctionName?: string;
}): string {
  const blocks: string[] = [];
  if (options.packedFunctionName != null) {
    blocks.push(`fn ${options.packedFunctionName}(packedMaterial: u32) -> bool {
  return ((packedMaterial >> ${PACKED_MATERIAL_TRANSMISSION_SHIFT}u) & 0x${PACKED_MATERIAL_TRANSMISSION_MASK.toString(16).toUpperCase()}u) != 0u;
}`);
  }
  if (options.sampledFunctionName != null) {
    blocks.push(`fn ${options.sampledFunctionName}(transmission: f32) -> bool {
  return transmission > 0.0;
}`);
  }
  return blocks.join('\n\n');
}
