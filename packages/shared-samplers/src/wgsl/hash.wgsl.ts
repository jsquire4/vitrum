/**
 * Canonical per-pixel deterministic hash WGSL primitives — single source of
 * truth across @vitrum/walkaround-hybrid.
 *
 * W2-C15 dedup (premium-grade-refactor-20260517 §W2): three shader files
 * open-coded the classic `fract(sin(...) * 43758.5453)` per-pixel hash with
 * cosmetic variation (the seed constant pair, the wrapper function name):
 *   - packages/walkaround-hybrid/src/shaders/shade.wgsl.ts     (sun-cone jitter)
 *   - packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts      (slice jitter)
 *   - (surfaceTextures.wgsl.ts also uses the constant 43758.5453 but with
 *      a different seed pair (127.1, 311.7) over continuous vec2f UVs for
 *      value-noise — semantically distinct, not consolidated here.)
 *
 * Functions exported:
 *   const HASH_SEED: f32 = 43758.5453
 *       The Mittring "trick of the trade" magic seed — irrational enough
 *       that adjacent pixels produce widely-spread hash output.
 *
 *   fn pixelHash21(p: vec2u) -> f32
 *       2D pixel coord -> 1D scalar hash in roughly `[0, 1)`.  The seed
 *       pair (12.9898, 78.233) is the de-facto standard from shadertoy
 *       and matches the pre-W2-C15 shade/gtao open-codings exactly.
 *
 *   fn pixelHash22(p: vec2u) -> vec2f
 *       2D pixel coord -> 2D vector hash.  The two channels use
 *       independent seed-pair-and-seed constellations so the output
 *       components are mutually uncorrelated (matters when a caller
 *       needs an independent (u, v) pair from the same pixel, e.g.
 *       sun-cone jitter in shade.wgsl).  Component-zero uses the
 *       canonical pair / HASH_SEED; component-one uses the
 *       pre-W2-C15 shade.wgsl seed (93.989, 67.345) and seed
 *       constant 24634.6345.
 *
 *   fn pixelHash11(x: f32) -> f32
 *       Scalar-to-scalar hash for callers that need a single-input
 *       deterministic source (e.g. cascade-index dithering).
 *
 * Behaviour preservation: every consumer that previously open-coded a
 * `sin(...) * 43758.5453` form gets the SAME bit-pattern out of the
 * canonical helper for the same pixel coord; the only change is removal
 * of the duplicate function declaration.
 *
 * References:
 *   - "Hash without sine" thread (shadertoy): the (12.9898, 78.233) /
 *     43758.5453 constellation traces back to Mittring's 2007 GDC
 *     "Finding Next Gen — CryEngine 2" presentation and was popularised
 *     for shader use by Inigo Quilez and shadertoy users circa 2013.
 */

export const HASH_WGSL = /* wgsl */ `

// ============================================================
// Per-pixel deterministic hash (canonical — @vitrum/shared-samplers)
// ============================================================
const HASH_SEED: f32 = 43758.5453;

fn pixelHash21(p: vec2u) -> f32 {
  return fract(sin(f32(p.x) * 12.9898 + f32(p.y) * 78.233) * HASH_SEED);
}

fn pixelHash22(p: vec2u) -> vec2f {
  let h0 = fract(sin(f32(p.x) * 12.9898 + f32(p.y) * 78.233) * HASH_SEED);
  let h1 = fract(sin(f32(p.x) * 93.989  + f32(p.y) * 67.345) * 24634.6345);
  return vec2f(h0, h1);
}

fn pixelHash11(x: f32) -> f32 {
  return fract(sin(x * 12.9898) * HASH_SEED);
}

`;
