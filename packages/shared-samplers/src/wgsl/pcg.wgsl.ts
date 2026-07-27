/** Canonical PCG32 random primitives shared across WGSL backends. */
export const PCG_MODULE_NAME = 'pcg';

export const PCG_WGSL = /* wgsl */ `
fn pcgInit(px: u32, py: u32, frameSeed: u32) -> u32 {
  var state = px * 1664525u + py * 1013904223u + frameSeed * 22695477u;
  state ^= state >> 17u;
  state ^= state << 31u;
  state ^= state >> 11u;
  return state;
}

fn pcgNext(state: ptr<function, u32>) -> u32 {
  (*state) = (*state) * 747796405u + 2891336453u;
  var word = (((*state) >> (((*state) >> 28u) + 4u)) ^ (*state)) * 277803737u;
  word = (word >> 22u) ^ word;
  return word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
  return f32(pcgNext(state) >> 8u) / 16777216.0;
}

fn rand2(state: ptr<function, u32>) -> vec2f {
  return vec2f(rand_f32(state), rand_f32(state));
}

fn rand3(state: ptr<function, u32>) -> vec3f {
  return vec3f(rand_f32(state), rand_f32(state), rand_f32(state));
}
`;

/**
 * Stateless single-shot PCG32 hash → f32 in [0,1).
 *
 * Shares the PCG32 mixer (multiplier 747796405u/2891336453u, output mix
 * 277803737u) with `pcgNext`, but takes a bare seed instead of a mutable state
 * pointer. Kept as a SEPARATE export — NOT appended to `PCG_WGSL` — so:
 *   (a) consumers needing only the stateless hash (e.g. walkaround-rc probe
 *       jitter) don't drag in the full stateful RNG set, and
 *   (b) `PCG_WGSL`'s composed-shader bytes stay byte-stable for its existing
 *       consumers (pt-webgpu, walkaround-hybrid), so their WGSL contract pins
 *       don't churn.
 */
export const PCG_HASH_TO_F32_WGSL = /* wgsl */ `
fn pcgHashToF32(seed: u32) -> f32 {
  var s = seed * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return f32(((word >> 22u) ^ word) >> 8u) / 16777216.0;
}
`;
