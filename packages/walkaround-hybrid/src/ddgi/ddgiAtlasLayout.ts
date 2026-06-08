/**
 * DDGI atlas layout constants — single source of truth for both the
 * producer (probeGrid.ts allocateAtlases) and the two consumers
 * (ddgiSampleWgsl.ts TSL sampler, engines/restir/shaders/shade.wgsl.ts
 * WGSL sampler). Keeping the cell sizes + border in one place prevents
 * the producer/consumer pair from silently drifting apart.
 *
 * Layout invariant per probe:
 *  - Irradiance SH cell: IRR_CELL × IRR_CELL pixels of rgba16float, storing
 *    the 9 RGB L2 spherical-harmonic coefficients in a 3x3 texel block.
 *  - Visibility octahedral cell: VIS_CELL × VIS_CELL pixels of rgba16float
 *    (rg = mean / mean²; a denser texel grid is needed than for irradiance
 *    because Chebyshev shadows benefit from sharper depth comparisons).
 *  - Visibility cells are wrapped by BORDER pixels (split BORDER/2 each side)
 *    so bilinear sampling at the cell edge wraps around the octahedral seam
 *    correctly. SH irradiance has no seam and does not run a border pass; the
 *    reserve ring is unused.
 *  - In-atlas stride between adjacent probe cells = CELL + BORDER.
 *
 * The DDGI WGSL helpers in ddgi/wgsl/probeUpdateRays.wgsl.ts and the
 * sampler-string consumers encode this same arithmetic for the compute side;
 * if those helpers ever change, this module and the consumers must change in
 * lockstep.
 */

// Irradiance migrated to L2 SH (ddgiSH.wgsl.ts): each probe stores 9 RGB
// coefficients in a 3x3 interior block, so the cell shrank 8x8 -> 3x3 (4x less
// irradiance-atlas memory). SH has no octahedral seam, so the irradiance cell
// needs no border — the 2px ring is kept only so the shared (CELL + BORDER)
// stride arithmetic and the texel-centre sampler reads stay uniform with the
// visibility atlas; nothing writes or reads the irradiance border (the border
// pass is skipped for irradiance, see probeUpdatePass).
export const IRR_CELL = 3;
export const VIS_CELL = 16;
export const BORDER = 2;

export const IRR_STRIDE = IRR_CELL + BORDER; // 5
export const VIS_STRIDE = VIS_CELL + BORDER; // 18
