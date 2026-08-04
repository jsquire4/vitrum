/** Canonical numeric publication limits shared by DDGI CPU and WGSL paths. */
export const DDGI_F32_MAX = 3.4028234663852886e38;
export const DDGI_F32_MIN_POSITIVE = 1.401298464324817e-45;
export const DDGI_U32_MAX = 0xffff_ffff;

/** Largest finite value the f16-mantissa block codec decodes without overflow. */
export const DDGI_ATLAS_SAFE_BLOCK_MAX = 3.4011621342146535e38;
/** f32 bits 0x5f7fefff; its f32 square stays within SAFE_BLOCK_MAX. */
export const DDGI_VISIBILITY_DISTANCE_MAX = 1.8442239374570553e19;
/** f32 bits 0x1efa0001; (.001 * spacing)^2 remains positive f32. */
export const DDGI_PROBE_SPACING_MIN = 2.646978121728402e-20;
/** f32 bits 0x5d7fefff; spacing * 16 equals VISIBILITY_DISTANCE_MAX. */
export const DDGI_PROBE_SPACING_MAX = 1.1526399609106596e18;

/** Exact f32 factors used by WGSL for the normal/ray-origin bias path. */
export const DDGI_NORMAL_BIAS_FACTOR_F32 = Math.fround(0.001);
export const DDGI_DIAGONAL_COMPONENT_F32 = Math.fround(1 / Math.sqrt(3));
