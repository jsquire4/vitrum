/** Canonical ReSTIR-DI storage layout shared by WGSL and host allocation. */
export const RESERVOIR_DI_STRIDE_U32 = 8;
export const RESERVOIR_DI_STRIDE_BYTES =
  RESERVOIR_DI_STRIDE_U32 * Uint32Array.BYTES_PER_ELEMENT;
