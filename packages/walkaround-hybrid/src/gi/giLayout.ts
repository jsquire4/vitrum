/**
 * Historical compact ReSTIR-GI ABI. Production shaders and allocations must
 * never use this stride; it remains only so v3-v6 snapshots can be decoded and
 * migrated to a cold generalized-reuse history without losing DDGI/PPG state.
 */
export const RESERVOIR_GI_LEGACY_STRIDE_U32 = 20;

/** The sole live ReSTIR-GI / generalized reconnection-shift reservoir ABI. */
export const RESERVOIR_GI_STRIDE_U32 = 28;
export const RESERVOIR_GI_STRIDE_BYTES = RESERVOIR_GI_STRIDE_U32 * 4;
