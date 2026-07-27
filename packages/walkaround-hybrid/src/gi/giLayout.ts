/** Shared authority for the compact and GRIS DDGI-proxy reservoir strides. */
export const RESERVOIR_GI_BASE_STRIDE_U32 = 20;
export const RESERVOIR_GI_GRIS_STRIDE_U32 = 28;

export const RESERVOIR_GI_BASE_STRIDE_BYTES = RESERVOIR_GI_BASE_STRIDE_U32 * 4;
export const RESERVOIR_GI_GRIS_STRIDE_BYTES = RESERVOIR_GI_GRIS_STRIDE_U32 * 4;

export function reservoirGiStrideU32ForGrisReuse(grisReuse: boolean): number {
  return grisReuse ? RESERVOIR_GI_GRIS_STRIDE_U32 : RESERVOIR_GI_BASE_STRIDE_U32;
}

export function reservoirGiStrideBytesForGrisReuse(grisReuse: boolean): number {
  return reservoirGiStrideU32ForGrisReuse(grisReuse) * 4;
}
