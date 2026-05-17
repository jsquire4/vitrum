/**
 * Canonical PCG random-number generator WGSL primitives — single source of
 * truth across @vitrum/walkaround-hybrid and @vitrum/pt-webgpu.
 *
 * W2-C6 dedup (premium-grade-refactor-20260517 §W2): pcgInit / pcgNext and
 * the `rand_f32*` helpers were declared independently in
 *   - packages/walkaround-hybrid/src/shaders/common.wgsl.ts
 *   - packages/pt-webgpu/src/wgsl/common.wgsl.ts
 * with byte-identical bodies (no algorithmic drift; the only difference was
 * walkaround additionally exposing `rand2` / `rand3` 2-tuple / 3-tuple
 * helpers under shorter names).  This module is the canonical declaration;
 * consumers must NOT inline a copy.
 *
 * Functions exported:
 *   fn pcgInit(px: u32, py: u32, frameSeed: u32) -> u32
 *       Seed the PCG state from the per-pixel coordinate and a per-frame
 *       seed.  The leading affine combination decorrelates adjacent pixels
 *       across frames; the three XOR-shifts diffuse the bits before the
 *       first `pcgNext` call to avoid the low-entropy initial state biasing
 *       any one-shot consumer (e.g. a single `rand_f32` call per pixel).
 *
 *   fn pcgNext(state: ptr<function, u32>) -> u32
 *       PCG XSH-RR variant — O'Neill 2014 "PCG: A Family of Simple Fast
 *       Space-Efficient Statistically Good Algorithms for Random Number
 *       Generation."  Advances `state` in place and returns the next u32.
 *
 *   fn rand_f32(state: ptr<function, u32>) -> f32
 *       One uniform `[0, 1)` draw.
 *
 *   fn rand_f32_2(state: ptr<function, u32>) -> vec2f
 *       Two uniform `[0, 1)` draws as a vec2f.
 *
 *   fn rand_f32_3(state: ptr<function, u32>) -> vec3f
 *       Three uniform `[0, 1)` draws as a vec3f.
 *
 * Canonical name choices: the task spec (plan/premium-grade-refactor-
 * 20260517.md §W2 C6) names the n-tuple helpers `rand_f32_2` / `rand_f32_3`
 * for parallel-stack consistency with `rand_f32`.  Walkaround's pre-W2-C6
 * `rand2` / `rand3` are aliased to the canonical names so existing call
 * sites compile unchanged.
 *
 * References:
 *   - O'Neill, M.E. (2014). "PCG: A Family of Simple Fast Space-Efficient
 *     Statistically Good Algorithms for Random Number Generation."
 *     https://www.pcg-random.org/paper.html
 */

export const PCG_WGSL = /* wgsl */ `

// ============================================================
// PCG random number generator (canonical — @vitrum/shared-samplers)
// ============================================================
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
  return f32(pcgNext(state)) / f32(0xFFFFFFFFu);
}

fn rand_f32_2(state: ptr<function, u32>) -> vec2f {
  return vec2f(rand_f32(state), rand_f32(state));
}

fn rand_f32_3(state: ptr<function, u32>) -> vec3f {
  return vec3f(rand_f32(state), rand_f32(state), rand_f32(state));
}

`;
