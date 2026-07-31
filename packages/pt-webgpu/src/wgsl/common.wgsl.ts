import { PCG_WGSL, SOBOL_DIRECTION_NUMBERS_WGSL } from '@vitrum/shared-samplers';
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
export function composePtWebgpuRngWgsl(sampling: PtWebgpuSamplingMode = 'pcg'): string {
  const rng =
    sampling === 'sobol' ? PT_WEBGPU_SOBOL_RNG_WGSL : `alias PtRngState = u32;\n${PCG_WGSL}`;
  const frameKey = sampling === 'sobol' ? SOBOL_FRAME_KEY_WGSL : PCG_FRAME_KEY_WGSL;
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
export function composePtWebgpuCommonWgsl(sampling: PtWebgpuSamplingMode = 'pcg'): string {
  const rng = composePtWebgpuRngWgsl(sampling);
  return /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
const INV_2PI = 0.15915494309189535;
const INFINITY = 3.402823466e38;

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
  let scale = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (!(scale > 0.0) || scale > 3.402823e38) {
    return vec3f(0.0, 1.0, 0.0);
  }
  let scaled = v / scale;
  return scaled / length(scaled);
}

fn safe_length(v: vec3f) -> f32 {
  let scale = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (!(scale > 0.0) || scale > INFINITY) {
    return 0.0;
  }
  let result = scale * length(v / scale);
  if (!(result > 0.0) || result > INFINITY) {
    return 0.0;
  }
  return result;
}

struct AreaVectorMeasure {
  normal: vec3f,
  area: f32,
  edgeScale: f32,
  valid: u32,
};

// Scale-equilibrated area-vector contract shared by every finite-area emitter
// path. The cross product is formed only after both axes are divided by one
// shared max-component scale. Its direction is normalized in that O(1) domain,
// then coefficient·|u×v| is recovered with one multiplication by edgeScale at
// a time. Exact collinearity remains invalid; NaN, infinity, underflow to zero,
// and overflow fail closed through valid=0.
fn measureAreaVector(
  u: vec3f,
  v: vec3f,
  coefficient: f32,
) -> AreaVectorMeasure {
  var result: AreaVectorMeasure;
  result.normal = vec3f(0.0, 1.0, 0.0);
  result.area = 0.0;
  result.edgeScale = 0.0;
  result.valid = 0u;
  if (
    !all(u == u) || !all(v == v) || !(coefficient > 0.0) ||
    any(abs(u) > vec3f(3.402823e38)) ||
    any(abs(v) > vec3f(3.402823e38)) ||
    coefficient > 3.402823e38
  ) {
    return result;
  }
  let edgeScale = max(
    max(abs(u.x), max(abs(u.y), abs(u.z))),
    max(abs(v.x), max(abs(v.y), abs(v.z))),
  );
  if (!(edgeScale > 0.0) || edgeScale > 3.402823e38) {
    return result;
  }
  let areaVector = cross(u / edgeScale, v / edgeScale);
  let crossScale = max(
    abs(areaVector.x),
    max(abs(areaVector.y), abs(areaVector.z)),
  );
  if (!(crossScale > 0.0) || crossScale > 3.402823e38) {
    return result;
  }
  let areaDirection = areaVector / crossScale;
  let directionLength = length(areaDirection);
  if (!(directionLength > 0.0) || directionLength > 3.402823e38) {
    return result;
  }
  let normal = areaDirection / directionLength;
  var area = coefficient * (crossScale * directionLength);
  area = (area * edgeScale) * edgeScale;
  let inverseArea = 1.0 / area;
  if (
    !(area > 0.0) || area > 3.402823e38 ||
    !(inverseArea > 0.0) || inverseArea > 3.402823e38 ||
    !all(normal == normal) || any(abs(normal) > vec3f(3.402823e38))
  ) {
    return result;
  }
  result.normal = normal;
  result.area = area;
  result.edgeScale = edgeScale;
  result.valid = 1u;
  return result;
}

// Solve rel = u·s + v·t by projecting onto the coordinate pair whose 2x2
// determinant is the dominant component of the already scale-safe area normal.
// This avoids the squared cross magnitude in the Gram-system inverse.
// Return (s, t, valid).
fn solveAreaVectorCoordinates(
  u: vec3f,
  v: vec3f,
  rel: vec3f,
  measure: AreaVectorMeasure,
) -> vec3f {
  if (measure.valid == 0u) {
    return vec3f(0.0);
  }
  let scaledU = u / measure.edgeScale;
  let scaledV = v / measure.edgeScale;
  let scaledRel = rel / measure.edgeScale;
  let absNormal = abs(measure.normal);
  var det = 0.0;
  var sNumerator = 0.0;
  var tNumerator = 0.0;
  if (absNormal.x >= absNormal.y && absNormal.x >= absNormal.z) {
    det = scaledU.y * scaledV.z - scaledU.z * scaledV.y;
    sNumerator = scaledRel.y * scaledV.z - scaledRel.z * scaledV.y;
    tNumerator = scaledU.y * scaledRel.z - scaledU.z * scaledRel.y;
  } else if (absNormal.y >= absNormal.z) {
    det = scaledU.z * scaledV.x - scaledU.x * scaledV.z;
    sNumerator = scaledRel.z * scaledV.x - scaledRel.x * scaledV.z;
    tNumerator = scaledU.z * scaledRel.x - scaledU.x * scaledRel.z;
  } else {
    det = scaledU.x * scaledV.y - scaledU.y * scaledV.x;
    sNumerator = scaledRel.x * scaledV.y - scaledRel.y * scaledV.x;
    tNumerator = scaledU.x * scaledRel.y - scaledU.y * scaledRel.x;
  }
  if (det == 0.0) {
    return vec3f(0.0);
  }
  let coordinates = vec2f(sNumerator, tNumerator) / det;
  if (
    !all(coordinates == coordinates) ||
    any(abs(coordinates) > vec2f(3.402823e38))
  ) {
    return vec3f(0.0);
  }
  return vec3f(coordinates, 1.0);
}

fn finite_homogeneous_point_common(value: vec4f) -> vec4f {
  let scale = max(max(abs(value.x), abs(value.y)), max(abs(value.z), abs(value.w)));
  if (!(scale > 0.0) || scale > 3.402823e38) {
    return vec4f(0.0);
  }
  let normalized = value / scale;
  if (normalized.w == 0.0) {
    return vec4f(0.0);
  }
  let point = normalized.xyz / normalized.w;
  if (
    !all(point == point) ||
    any(abs(point) > vec3f(3.402823e38))
  ) {
    return vec4f(0.0);
  }
  return vec4f(point, 1.0);
}

fn unproject_ray_common(invViewProjection: mat4x4f, ndc: vec2f) -> Ray {
  var ray: Ray;
  ray.origin = vec3f(0.0);
  ray.direction = vec3f(0.0);
  var farH = invViewProjection * vec4f(ndc, 1.0, 1.0);
  var nearH = invViewProjection * vec4f(ndc, -1.0, 1.0);
  let farScale = max(max(abs(farH.x), abs(farH.y)), max(abs(farH.z), abs(farH.w)));
  let nearScale = max(max(abs(nearH.x), abs(nearH.y)), max(abs(nearH.z), abs(nearH.w)));
  if (
    !(farScale > 0.0) || farScale > 3.402823e38 ||
    !(nearScale > 0.0) || nearScale > 3.402823e38
  ) {
    return ray;
  }
  farH /= farScale;
  nearH /= nearScale;
  let nearPoint = finite_homogeneous_point_common(nearH);
  if (nearPoint.w == 0.0) {
    return ray;
  }
  var orientation = 1.0;
  if (farH.w != 0.0) {
    orientation = sign(farH.w * nearH.w);
  }
  let directionNumerator =
    (farH.xyz * nearH.w - nearH.xyz * farH.w) * orientation;
  let directionScale = max(
    abs(directionNumerator.x),
    max(abs(directionNumerator.y), abs(directionNumerator.z)),
  );
  if (!(directionScale > 0.0) || directionScale > 3.402823e38) {
    return ray;
  }
  ray.origin = nearPoint.xyz;
  ray.direction = safe_normalize(directionNumerator);
  return ray;
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
// TriHit: returns the hit distance t on a hit, INFINITY on a miss. This scalar
// adapter remains only for call sites that need distance but no hit attributes;
// mesh traversal consumes mollerTrumboreCore directly so barycentrics,
// determinant side, and the scale-safe normal cannot drift from acceptance.
//
// NUMERICS NOTE (V7): switching to the canonical core changes edge behaviour --
// the core uses fixed, dimensionless SIGNED barycentric tests
// (u/v/w < -MOLLER_TRUMBORE_BARYCENTRIC_EPSILON)
// instead of the old strict u<0||u>1 / v<0||u+v>1 tests, so hits grazing a
// triangle edge by less than that tolerance are now accepted (closes shared-edge
// cracks). The hit distance t for interior hits is algebraically unchanged.
// The __tests__/cpuTracer.ts oracle mirrors this same core so it stays in sync.
fn intersectTriangle(
  origin: vec3f,
  dir: vec3f,
  a: vec3f,
  b: vec3f,
  c: vec3f,
  tMin: f32,
) -> f32 {
  let core = mollerTrumboreCore(origin, dir, a, b, c, tMin);
  if (!core.hit) {
    return INFINITY;
  }
  return core.t;
}
`;
}

export const PT_WEBGPU_COMMON_WGSL = composePtWebgpuCommonWgsl('pcg');
