import {
  PCG_WGSL,
  SOBOL_DIRECTION_NUMBERS_WGSL,
} from '@vitrum/shared-samplers';
import { SAFE_INV_DIR_WGSL, MOLLER_TRUMBORE_WGSL } from '@vitrum/shared-bvh';

export type PtWebgpuSamplingMode = 'pcg' | 'sobol';

/**
 * pt-webgpu low-discrepancy RNG module.
 *
 * It intentionally preserves the existing pt-webgpu RNG symbol surface
 * (`pcgInit`, `pcgNext`, `rand_f32`, `rand2`) so the path-trace,
 * ReSTIR-PT, SPPM, BDPT, and adjoint call sites can switch as a composed
 * whole. It combines the shared Joe-Kuo D(6) direction-number prefix with
 * JCGT/Laine-Karras hash-based Owen scrambling while remaining binding-free.
 *
 * Dimensions 0 through 511 use fixed-stream Owen scrambles over one
 * CPU/WGSL-canonical table, covering the audited 324-draw path budget with
 * headroom. Dimension 512 and later use an independent PCG continuation seeded
 * from the full pixel/frame/path identity and the crossing dimension. The
 * 32-bit counter never wraps into the Sobol prefix, and every 65,536-sample
 * block receives a distinct fixed scramble key.
 */
export const PT_WEBGPU_SOBOL_RNG_WGSL = /* wgsl */ `
struct PtRngState {
  sampleIndex: u32,
  dimension: u32,
  pixelX: u32,
  pixelY: u32,
  sequenceKey: u32,
  rotationTile: u32,
  fallbackState: u32,
};

${SOBOL_DIRECTION_NUMBERS_WGSL}

const PT_SOBOL_FACTOR = 0.000000059604644775390625; // 1 / 2^24
const PT_SOBOL_MAX_POINTS = 65536u;
const PT_SOBOL_DIMENSION_COUNT = SOBOL_DIRECTION_DIMENSION_COUNT;

const PT_SOBOL_BLUE_NOISE_RANK_8X8 = array<u32, 64>(
  0u, 63u, 12u, 60u, 3u, 55u, 15u, 62u,
  53u, 23u, 57u, 17u, 44u, 19u, 32u, 22u,
  10u, 40u, 5u, 41u, 8u, 35u, 7u, 47u,
  45u, 28u, 48u, 25u, 54u, 29u, 36u, 24u,
  2u, 38u, 13u, 46u, 1u, 37u, 14u, 51u,
  58u, 30u, 49u, 16u, 59u, 20u, 43u, 18u,
  11u, 56u, 6u, 34u, 9u, 39u, 4u, 50u,
  52u, 31u, 33u, 27u, 42u, 26u, 61u, 21u
);

fn ptSobolReverseBits32(xIn: u32) -> u32 {
  var x = xIn;
  x = ((x & 0xaaaaaaaau) >> 1u) | ((x & 0x55555555u) << 1u);
  x = ((x & 0xccccccccu) >> 2u) | ((x & 0x33333333u) << 2u);
  x = ((x & 0xf0f0f0f0u) >> 4u) | ((x & 0x0f0f0f0fu) << 4u);
  x = ((x & 0xff00ff00u) >> 8u) | ((x & 0x00ff00ffu) << 8u);
  return (x >> 16u) | (x << 16u);
}

fn ptSobolHash(xIn: u32) -> u32 {
  var x = xIn;
  x ^= x >> 16u;
  x *= 0x85ebca6bu;
  x ^= x >> 13u;
  x *= 0xc2b2ae35u;
  x ^= x >> 16u;
  return x;
}

fn ptSobolHashCombine(seed: u32, v: u32) -> u32 {
  return seed ^ (v + ((seed << 6u) + (seed >> 2u)));
}

fn ptSobolLaineKarrasPermutation(xIn: u32, seed: u32) -> u32 {
  var x = xIn + seed;
  x ^= x * 0x6c50b47cu;
  x ^= x * 0xb82f1e52u;
  x ^= x * 0xc7afe638u;
  x ^= x * 0x8d22f6e6u;
  return x;
}

fn ptSobolNestedUniformScrambleBase2(x: u32, seed: u32) -> u32 {
  return ptSobolReverseBits32(ptSobolLaineKarrasPermutation(x, seed));
}

fn ptSobolMaskedComponent(index: u32, dim: u32) -> u32 {
  var out = 0u;
  for (var bit = 0u; bit < SOBOL_DIRECTION_BITS; bit = bit + 1u) {
    if (((index >> bit) & 1u) != 0u) {
      out ^= sobolDirectionComponent(dim, bit);
    }
  }
  return out;
}

fn ptSobolTextureComponent(index: u32, dim: u32) -> u32 {
  return ptSobolMaskedComponent(index % PT_SOBOL_MAX_POINTS, dim) & 0x00ffffffu;
}

fn ptSobolBlueNoiseRotation(tile: u32, dim: u32) -> u32 {
  let rank = PT_SOBOL_BLUE_NOISE_RANK_8X8[tile & 63u];
  // Rank zero is the zero-offset stratum, not an unscrambled pixel: the full
  // pixel identity still feeds both fixed-stream Owen seeds in ptSobolNextU32.
  if (rank == 0u) {
    return 0u;
  }
  return ptSobolHash(ptSobolHashCombine(rank, dim)) & 0x00ffffffu;
}

fn pcgInit(px: u32, py: u32, frameKey: u32) -> PtRngState {
  let pixelSeed = ptSobolHash(ptSobolHashCombine(ptSobolHash(px), py));
  let sampleIndex = frameKey & 0x0000ffffu;
  let sequenceKey = frameKey >> 16u;
  var state: PtRngState;
  state.sampleIndex = sampleIndex;
  state.dimension = 0u;
  state.pixelX = px;
  state.pixelY = py;
  state.sequenceKey = sequenceKey;
  state.rotationTile = (px & 7u) | ((py & 7u) << 3u);
  state.fallbackState = ptSobolHash(
    ptSobolHashCombine(
      ptSobolHashCombine(ptSobolHashCombine(pixelSeed, sequenceKey), sampleIndex),
      PT_SOBOL_DIMENSION_COUNT,
    ),
  );
  return state;
}

fn ptSobolFallbackNext(state: ptr<function, PtRngState>) -> u32 {
  (*state).fallbackState = (*state).fallbackState * 747796405u + 2891336453u;
  var word = (((*state).fallbackState >> (((*state).fallbackState >> 28u) + 4u)) ^
    (*state).fallbackState) * 277803737u;
  word = (word >> 22u) ^ word;
  return word;
}

fn ptSobolNextU32(state: ptr<function, PtRngState>) -> u32 {
  let dim = (*state).dimension;
  if (dim >= PT_SOBOL_DIMENSION_COUNT) {
    if (dim != 0xffffffffu) {
      (*state).dimension = dim + 1u;
    }
    return ptSobolFallbackNext(state);
  }
  let pathIndex = (*state).sampleIndex;
  let pixelSeed = ptSobolHash(
    ptSobolHashCombine(ptSobolHash((*state).pixelX), (*state).pixelY),
  );
  let streamSeed = ptSobolHash(ptSobolHashCombine(pixelSeed, (*state).sequenceKey));
  let seed = ptSobolHash(ptSobolHashCombine(streamSeed, dim));
  let shuffleSeed = ptSobolHashCombine(streamSeed, 0u);
  let shuffledIndex = ptSobolNestedUniformScrambleBase2(
    ptSobolReverseBits32(pathIndex),
    shuffleSeed,
  ) % PT_SOBOL_MAX_POINTS;
  var result = ptSobolTextureComponent(shuffledIndex, dim);
  let componentSeed = ptSobolHashCombine(seed, 1u + (dim & 3u));
  result = ptSobolNestedUniformScrambleBase2(result, componentSeed);
  let rotated24 = (((result >> 8u) & 0x00ffffffu) +
    ptSobolBlueNoiseRotation((*state).rotationTile, dim)) & 0x00ffffffu;
  (*state).dimension = dim + 1u;
  return rotated24 << 8u;
}

fn pcgNext(state: ptr<function, PtRngState>) -> u32 {
  return ptSobolNextU32(state);
}

fn rand_f32(state: ptr<function, PtRngState>) -> f32 {
  return f32(ptSobolNextU32(state) >> 8u) * PT_SOBOL_FACTOR;
}

fn rand2(state: ptr<function, PtRngState>) -> vec2f {
  return vec2f(rand_f32(state), rand_f32(state));
}
`;

const PCG_FRAME_KEY_WGSL = /* wgsl */ `
fn ptRngFrameKey(frameSeed: u32, frameIndex: u32) -> u32 {
  return frameSeed ^ frameIndex;
}
`;

export const SOBOL_FRAME_KEY_WGSL = /* wgsl */ `
fn ptRngFrameKey(frameSeed: u32, frameIndex: u32) -> u32 {
  let sampleBlock = frameIndex >> 16u;
  let seedKey = ptSobolHash(frameSeed) & 0x0000ffffu;
  let blockKey = (seedKey + sampleBlock * 0x00009e37u) & 0x0000ffffu;
  return (blockKey << 16u) | (frameIndex & 0x0000ffffu);
}
`;

/** Compose the exact RNG implementation and frame-key mapping for a mode. */
export function composePtWebgpuRngWgsl(
  sampling: PtWebgpuSamplingMode = 'pcg',
): string {
  const rng = sampling === 'sobol'
    ? PT_WEBGPU_SOBOL_RNG_WGSL
    : `alias PtRngState = u32;\n${PCG_WGSL}`;
  const frameKey = sampling === 'sobol'
    ? SOBOL_FRAME_KEY_WGSL
    : PCG_FRAME_KEY_WGSL;
  // Preserve the historical blank line between the two modules so every
  // existing production composition remains byte-for-byte stable.
  return `${rng}\n\n${frameKey}`;
}

/**
 * Early shared WGSL include for pt-webgpu.
 *
 * This captures transferable, renderer-agnostic pieces from stainedGlass:
 * - PCG RNG
 * - BVH node/ray/hit structs aligned with three-mesh-bvh's packed layout
 * - Triangle intersection and basic utilities
 *
 * ReSTIR/DDGI-specific reservoir and lighting logic is intentionally excluded.
 */
export function composePtWebgpuCommonWgsl(
  sampling: PtWebgpuSamplingMode = 'pcg',
): string {
  const rng = composePtWebgpuRngWgsl(sampling);
  return /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
const INV_2PI = 0.15915494309189535;
const INFINITY = 1e20;

struct BVHNode {
  boundsMin: array<f32, 3>,
  boundsMax: array<f32, 3>,
  rightChildOrTriOffset: u32,
  splitAxisOrTriCount: u32,
};

struct Ray {
  origin: vec3f,
  direction: vec3f,
};

struct HitResult {
  didHit: bool,
  dist: f32,
  triIndex: u32,
  bary: vec3f,
  normal: vec3f,
};

${rng}

fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  if (len < 1e-8) {
    return vec3f(0.0, 1.0, 0.0);
  }
  return v / len;
}

${SAFE_INV_DIR_WGSL}

${MOLLER_TRUMBORE_WGSL}

// intersectTriangle: pt-webgpu's scalar-returning wrapper over the canonical
// shared Moller-Trumbore core (mollerTrumboreCore, prepended above as
// MOLLER_TRUMBORE_WGSL from @vitrum/shared-bvh). The MATH is now single-sourced
// in shared-bvh -- both this wrapper and the canonical struct-returning
// intersectTriangle in BVH_INTERSECT_WGSL delegate to the same core.
//
// pt-webgpu keeps its own thin f32-returning wrapper (rather than composing the
// full BVH_INTERSECT_WGSL) because its traversal kernels define their own
// BVHNode / Ray / SceneHit / HitResult structs, which would collide with the
// ones BVH_INTERSECT_WGSL declares. The wrapper just unpacks the core's
// TriHit: returns the hit distance t on a hit, INFINITY on a miss. The
// three pt-webgpu call sites (traceMeshBvh in intersection/intersectionLite,
// intersectMeshAreaLightRay in connect) compare the returned f32 against their
// own t-bounds, so the f32 contract is preserved.
//
// NUMERICS NOTE (V7): switching to the canonical core changes edge behaviour --
// the core uses triEps-tolerant SIGNED barycentric tests (u/v/w < -triEps)
// instead of the old strict u<0||u>1 / v<0||u+v>1 tests, so hits grazing a
// triangle edge by less than triEps are now accepted (closes shared-edge
// cracks). The hit distance t for interior hits is algebraically unchanged.
// The __tests__/cpuTracer.ts oracle mirrors this same core so it stays in sync.
fn intersectTriangle(origin: vec3f, dir: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let core = mollerTrumboreCore(origin, dir, a, b, c, params.triIntersectEpsilon);
  if (!core.hit) {
    return INFINITY;
  }
  return core.t;
}
`;
}

export const PT_WEBGPU_COMMON_WGSL = composePtWebgpuCommonWgsl('pcg');
