/**
 * ReSTIR-GI reservoir layout selection.
 *
 * The base Sprint-16/17 reservoir is 20 u32 values (80 bytes). GRIS /
 * ReSTIR-PT reconnection reuse appends a 10-u32 shift cache, widening the live
 * layout to 30 u32 values (120 bytes). Keeping this selection in one TS module
 * prevents the shader stride, frame-resource allocation, PPG train kernel, and
 * GI snapshot restore checks from drifting.
 */

export const RESERVOIR_GI_BASE_STRIDE_U32 = 20;
export const RESERVOIR_GI_GRIS_STRIDE_U32 = 30;

export const RESERVOIR_GI_BASE_STRIDE_BYTES = RESERVOIR_GI_BASE_STRIDE_U32 * 4;
export const RESERVOIR_GI_GRIS_STRIDE_BYTES = RESERVOIR_GI_GRIS_STRIDE_U32 * 4;

export function reservoirGiStrideU32ForRestirPtReuse(restirPtReuse: boolean): number {
  return restirPtReuse ? RESERVOIR_GI_GRIS_STRIDE_U32 : RESERVOIR_GI_BASE_STRIDE_U32;
}

export function reservoirGiStrideBytesForRestirPtReuse(restirPtReuse: boolean): number {
  return reservoirGiStrideU32ForRestirPtReuse(restirPtReuse) * 4;
}
