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

// Realized probability for a weighted Bernoulli driven by rand_f32. The RNG
// has exactly 2^24 equiprobable values, so an arbitrary f32 threshold is not
// generally its own realized probability. Keep exact endpoints; for every
// interior physical branch preserve at least one bucket on both sides and
// round to the nearest representable k/2^24. Callers must use this returned
// value for both the rand_f32(state) < p comparison and its matching
// PDF/throughput factor.
fn represented_bernoulli_probability_f32(probability: f32) -> f32 {
  if (!(probability > 0.0)) { return 0.0; }
  if (probability >= 1.0) { return 1.0; }
  let bucket = clamp(
    floor(probability * 16777216.0 + 0.5),
    1.0,
    16777215.0,
  );
  return bucket / 16777216.0;
}

// Exact uniform integer in [0, bound). Multiplication of rand_f32 by an
// arbitrary bound is biased because rand_f32 has only 2^24 equiprobable values.
// Rejection removes the incomplete residue prefix from the full PCG32 domain.
fn rand_bounded_u32(state: ptr<function, u32>, bound: u32) -> u32 {
  if (bound <= 1u) { return 0u; }
  let threshold = (0u - bound) % bound;
  loop {
    let value = pcgNext(state);
    if (value >= threshold) { return value % bound; }
  }
  return 0u; // syntactic totality for validators; the loop returns on acceptance
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
