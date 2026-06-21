import { PCG_WGSL } from '@vitrum/shared-samplers';
import { SAFE_INV_DIR_WGSL, MOLLER_TRUMBORE_WGSL } from '@vitrum/shared-bvh';

export type PtWebgpuSamplingMode = 'pcg' | 'sobol';

/**
 * Opt-in pt-webgpu low-discrepancy RNG module.
 *
 * It intentionally preserves the existing pt-webgpu RNG symbol surface
 * (`pcgInit`, `pcgNext`, `rand_f32`, `rand2`, `rand3`) so the path-trace,
 * ReSTIR-PT, SPPM, BDPT, and adjoint call sites can switch as a composed
 * whole. This mirrors the pt-webgl2 Sobol texture path's first-four direction
 * set plus JCGT/Laine-Karras hash-based Owen scrambling, but keeps it
 * binding-free for WebGPU.
 *
 * Promotion caveat: higher dimensions are hash-decorrelated over the first four
 * direction tables and the stream uses a small tiled ranked rotation. The
 * bounce/lobe/light dimension assignment is pinned by source-level regression
 * tests; measured equal-time RMSE promotion evidence is tracked separately.
 */
export const PT_WEBGPU_SOBOL_RNG_WGSL = /* wgsl */ `
const PT_SOBOL_FACTOR = 0.000000059604644775390625; // 1 / 2^24
const PT_SOBOL_MAX_POINTS = 65536u;

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

const PT_SOBOL_DIRECTIONS_1 = array<u32, 32>(
  0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u,
  0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
  0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u,
  0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
  0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u,
  0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
  0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u,
  0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu
);

const PT_SOBOL_DIRECTIONS_2 = array<u32, 32>(
  0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u,
  0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u,
  0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u,
  0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u,
  0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u,
  0x6868e800u, 0x9c9c5c00u, 0xeeee8e00u, 0x5555c500u,
  0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u,
  0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u
);

const PT_SOBOL_DIRECTIONS_3 = array<u32, 32>(
  0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u,
  0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u,
  0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u,
  0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u,
  0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u,
  0xa0858800u, 0x914e5400u, 0xdbe79e00u, 0x25db6d00u,
  0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u,
  0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u
);

const PT_SOBOL_DIRECTIONS_4 = array<u32, 32>(
  0x80000000u, 0x40000000u, 0x20000000u, 0xb0000000u,
  0xf8000000u, 0xdc000000u, 0x7a000000u, 0x9d000000u,
  0x5a800000u, 0x2fc00000u, 0xa1600000u, 0xf0b00000u,
  0xda880000u, 0x6fc40000u, 0x81620000u, 0x40bb0000u,
  0x22878000u, 0xb3c9c000u, 0xfb65a000u, 0xddb2d000u,
  0x78022800u, 0x9c0b3c00u, 0x5a0fb600u, 0x2d0ddb00u,
  0xa2878080u, 0xf3c9c040u, 0xdb65a020u, 0x6db2d0b0u,
  0x800228f8u, 0x400b3cdcu, 0x200fb67au, 0xb00ddb9du
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

fn ptSobolDirection(dim: u32, bit: u32) -> u32 {
  let d = dim & 3u;
  if (d == 0u) {
    return PT_SOBOL_DIRECTIONS_1[bit];
  }
  if (d == 1u) {
    return PT_SOBOL_DIRECTIONS_2[bit];
  }
  if (d == 2u) {
    return PT_SOBOL_DIRECTIONS_3[bit];
  }
  return PT_SOBOL_DIRECTIONS_4[bit];
}

fn ptSobolMasked(index: u32, dim: u32) -> u32 {
  var out = 0u;
  for (var bit = 0u; bit < 32u; bit = bit + 1u) {
    if (((index >> bit) & 1u) != 0u) {
      out ^= ptSobolDirection(dim, bit);
    }
  }
  return out;
}

fn ptSobolTextureComponent(index: u32, dim: u32) -> u32 {
  return ptSobolReverseBits32(ptSobolMasked(index % PT_SOBOL_MAX_POINTS, dim)) & 0x00ffffffu;
}

fn ptSobolBlueNoiseRotation(tile: u32, dim: u32) -> u32 {
  let rank = PT_SOBOL_BLUE_NOISE_RANK_8X8[tile & 63u];
  if (rank == 0u) {
    return 0u;
  }
  return ptSobolHash(ptSobolHashCombine(rank, dim)) & 0x00ffffffu;
}

fn pcgInit(px: u32, py: u32, frameSeed: u32) -> u32 {
  let pixelSeed = ptSobolHash(ptSobolHashCombine(ptSobolHash(px), py));
  let sampleIndex = frameSeed & 0x0000ffffu;
  let rotationTile = ptSobolHash(ptSobolHashCombine(pixelSeed, frameSeed >> 16u)) & 0xffu;
  return (sampleIndex << 16u) | (rotationTile << 8u);
}

fn ptSobolNextU32(state: ptr<function, u32>) -> u32 {
  let pathIndex = ((*state) >> 16u) & 0x0000ffffu;
  let rotationTile = ((*state) >> 8u) & 0xffu;
  let dim = (*state) & 0xffu;
  let seed = ptSobolHash(ptSobolHashCombine(pathIndex, dim));
  let shuffleSeed = ptSobolHashCombine(seed, 0u);
  let shuffledIndex = ptSobolNestedUniformScrambleBase2(
    ptSobolReverseBits32(pathIndex),
    shuffleSeed,
  ) % PT_SOBOL_MAX_POINTS;
  var result = ptSobolTextureComponent(shuffledIndex, dim);
  let componentSeed = ptSobolHashCombine(seed, 1u + (dim & 3u));
  result = ptSobolNestedUniformScrambleBase2(result, componentSeed);
  let rotated24 = (((result >> 8u) & 0x00ffffffu) + ptSobolBlueNoiseRotation(rotationTile, dim)) & 0x00ffffffu;
  (*state) = (pathIndex << 16u) | (rotationTile << 8u) | ((dim + 1u) & 0xffu);
  return rotated24 << 8u;
}

fn pcgNext(state: ptr<function, u32>) -> u32 {
  return ptSobolNextU32(state);
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
  return f32(ptSobolNextU32(state) >> 8u) * PT_SOBOL_FACTOR;
}

fn rand2(state: ptr<function, u32>) -> vec2f {
  return vec2f(rand_f32(state), rand_f32(state));
}

fn rand3(state: ptr<function, u32>) -> vec3f {
  return vec3f(rand_f32(state), rand_f32(state), rand_f32(state));
}
`;

const PCG_FRAME_KEY_WGSL = /* wgsl */ `
fn ptRngFrameKey(frameSeed: u32, frameIndex: u32) -> u32 {
  return frameSeed ^ frameIndex;
}
`;

const SOBOL_FRAME_KEY_WGSL = /* wgsl */ `
fn ptRngFrameKey(frameSeed: u32, frameIndex: u32) -> u32 {
  return (ptSobolHash(frameSeed) & 0xffff0000u) | (frameIndex & 0x0000ffffu);
}
`;

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
  const rng = sampling === 'sobol' ? PT_WEBGPU_SOBOL_RNG_WGSL : PCG_WGSL;
  const frameKey = sampling === 'sobol' ? SOBOL_FRAME_KEY_WGSL : PCG_FRAME_KEY_WGSL;
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

${frameKey}

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
