/**
 * Common WGSL code shared across all ReSTIR compute passes.
 * Exported as a TypeScript string so the bundler inlines it without a GLSL plugin.
 *
 * Includes:
 *   - PCG random number generator
 *   - BRDF utilities (GGX BRDF evaluation + sampling)
 *   - BVH struct definitions (matching three-mesh-bvh's WGSL layout)
 *   - Reservoir struct + pack/unpack helpers
 *   - Emitter struct + sampling helpers
 *   - G-buffer unpack helpers
 *   - WelfordVariance struct + helpers (imported from @vitrum/shared-denoisers)
 *
 * References:
 *   - three-mesh-bvh/src/webgpu/common_functions.wgsl.js — BVHNode struct
 *   - C-none/Web-RTRT reservoir.wgsl — encode/decode helpers
 */

import { WELFORD_VARIANCE_WGSL } from '@vitrum/shared-denoisers';
import type { WgslModule } from '../pipeline/wgslComposer.js';

export const COMMON_WGSL = /* wgsl */ `

// ============================================================
// Constants
// ============================================================
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
const INFINITY = 1e20;
const BVH_STACK_DEPTH = 60u;
const LEAFNODE_FLAG = 0xFFFF0000u;

// Audit M12 follow-up: the emitter-geometry-term distance² floor is now
// a runtime UBO value (ubo.emitterDist2Floor) rather than a compile-
// time constant. Hosts set it via HybridEngineOptions.emitterDist2Floor
// and a sensible default is computed from the scene AABB:
//   max(sceneDiag × 1e-3, 0.0001)²
// Cornell preserves 0.01 (≈10 cm minimum effective distance).
//
// CRITICAL: the floor must be applied identically in BOTH the RIS
// reservoir construction (ris.wgsl computePHat + M_LIGHT loop) AND
// shade direct-light evaluation (shade.wgsl) so the importance-sampled
// pHat matches the evaluated pHat. Both call:
//   emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor)
// with the same UBO value.

// Sprint 18 follow-up — ReSTIR-GI per-pixel unbiased weight (W) cap.
//
// W = w_sum / (M · p̂(z)) is unbounded when the chosen sample's p̂ at the
// visible point is small (grazing cosTheta from the visible normal to the
// reconnection direction, or near-zero luminance after a sky-miss bounce).
// A single pixel with a tiny p̂ produces a huge spike that:
//   1. Passes through the per-channel atrous-indirect chain (the spike is
//      a 1-pixel impulse, which the 5×5 kernel only attenuates to ~1/13 of
//      its peak per step; spread across 4 iterations the spike becomes a
//      multi-pixel halo rather than disappearing).
//   2. Defeats the temporal accumulator: alpha=0.02 means the spike's
//      contribution to the running average is 2% per frame, but new spikes
//      arrive on different pixels every ~5 frames, so the average never
//      settles to a stable fixed point — the user sees a never-converging
//      image with shifting bright dots and wavy bands.
//
// Capping W at 16 keeps single-sample contribution bounded without biasing
// the indirect magnitude.  Previous tighter cap of 4 was masking a real
// firefly source (near-light DDGI atlas spikes) while also truncating the
// heavy tail of legitimately bright samples — the result was a uniformly
// dim indirect signal (~17× too low on Cornell white walls) AND visible
// fireflies, because the cap reduced spike *amplitude* without addressing
// spike *prevalence*.  The proper fix is upstream: cap irrAtXs at the
// reconnection vertex (see risGi.wgsl).  W can then breathe at 16, which
// admits the legitimate variance the unbiased estimator needs while still
// bounding pathological cases for variance-bounded convergence.
const RESTIR_GI_W_CAP: f32 = 16.0;

// ============================================================
// WalkaroundUBO — canonical per-frame uniform layout shared by every
// ReSTIR compute pass (ris/temporal/spatial/shade). Defined here so
// the four passes do not drift; each shader binds with:
//   @group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
// (Atrous/welford/svgf do not bind @group(2); the struct definition is
// inert in those modules.)
//
// Layout offsets are pinned and host-side packers (pipeline/uboUpdater.ts)
// rely on them. Bump documentation if the layout changes.
// ============================================================
struct WalkaroundUBO {
  viewMatrix:                 mat4x4f, //  offset 0
  projMatrix:                 mat4x4f, //  offset 64
  prevViewMatrix:             mat4x4f, //  offset 128
  cameraPos:                  vec3f,   //  offset 192
  frameSeed:                  u32,     //  offset 204
  screenSize:                 vec2u,   //  offset 208
  emitterCount:               u32,     //  offset 216
  totalEmPower:               f32,     //  offset 220
  sunDirection:               vec3f,   //  offset 224
  sunIntensity:               f32,     //  offset 236 — matches BVH build
  skyTint:                    vec3f,   //  offset 240 — diffuse sky dome RGB
  skyIrradiance:              f32,     //  offset 252 — sky dome brightness
  // Library-generality tunables (audit follow-up). All scalar f32/u32
  // packed tightly after the existing 256-byte block. Defaults preserve
  // Cornell behaviour; hosts override via HybridEngineOptions.
  emitterDist2Floor:          f32,     //  offset 256 — audit M12
  directFireflyClamp:         f32,     //  offset 260 — audit B4
  causticBoost:               f32,     //  offset 264 — audit B1
  causticVisClamp:            f32,     //  offset 268 — audit B1
  temporalMClampDI:           u32,     //  offset 272 — audit M6
  spatialReuseRadiusPx:       f32,     //  offset 276 — audit M7
  spatialDepthTolFloor:       f32,     //  offset 280 — audit M8
  triIntersectEpsilon:        f32,     //  offset 284 — D12: Möller-Trumbore coplanarity floor
  // Padding to maintain 16-byte struct alignment (struct size must be a
  // multiple of 16). triIntersectEpsilon occupies 284-287; the previous
  // single _pad is now split into 4 words to reach 304 bytes (304 % 16 == 0).
  _pad:                       u32,     //  offset 288
  _pad2:                      u32,     //  offset 292
  _pad3:                      u32,     //  offset 296
  _pad4:                      u32,     //  offset 300
};

// Emitter geometry term G with a configurable dist² clamp applied at
// every call site. Use this everywhere instead of inlining
// nlDotL / dist² directly.  The floor is now read from the WalkaroundUBO
// (audit M12) rather than the old scene-scale-baked constant.
fn emitterGeometry(nlDotL: f32, dist2: f32, dist2Floor: f32) -> f32 {
  let dist2_clamped = max(dist2, dist2Floor);
  return nlDotL / dist2_clamped;
}

// ============================================================
// BVH structs (matches three-mesh-bvh WGSL layout)
// ============================================================

// BVHNode: 32 bytes -- exactly matches three-mesh-bvh's raw node layout.
// From node_modules/three-mesh-bvh/src/core/Constants.js:
//   BYTES_PER_NODE = 6 * 4 + 4 + 4 = 32
// Binary layout:
//   bytes  0-11:  boundsMin[0..2]  (3 x f32, NO padding between min+max)
//   bytes 12-23:  boundsMax[0..2]  (3 x f32)
//   bytes 24-27:  rightChildOrTriangleOffset  (u32)
//   bytes 28-31:  splitAxisOrTriangleCount    (u32, 0xFFFF0000|count for leaves)
//
// IMPORTANT: we use array<f32,3> (align 4, stride 12) NOT vec3f (align 16)
// so that boundsMin and boundsMax are packed without padding, and the
// right/split fields start at byte 24 -- matching the C++ layout.
struct BVHNode {
  boundsMin: array<f32, 3>,              // bytes 0-11  (no padding)
  boundsMax: array<f32, 3>,              // bytes 12-23 (no padding)
  rightChildOrTriOffset: u32,            // bytes 24-27
  splitAxisOrTriCount: u32,              // bytes 28-31
};

struct Ray {
  origin:    vec3f,
  direction: vec3f,
};

struct HitResult {
  didHit:         bool,
  dist:           f32,
  triIndex:       u32,       // triangle index in bvhIndex
  bary:           vec3f,     // barycentric coords (u,v,w)
  normal:         vec3f,
  matColorPacked: u32,       // RGB888 + (trans4|texType4) packed from bvhIndex[triIdx].w
  uv:             vec2f,     // interpolated UV at the hit point (0..1)
};

// ============================================================
// Emitter struct (80 bytes per emitter, 16-byte aligned)
// ============================================================
struct EmitterTri {
  vA:        vec3f,   // bytes 0-11
  _padA:     f32,     // bytes 12-15
  vB:        vec3f,   // bytes 16-27
  _padB:     f32,     // bytes 28-31
  vC:        vec3f,   // bytes 32-43
  _padC:     f32,     // bytes 44-47
  normal:    vec3f,   // bytes 48-59
  area:      f32,     // bytes 60-63
  Le:        vec3f,   // bytes 64-75
  intensity: f32,     // bytes 76-79
};

// ============================================================
// Per-pixel G-buffer data
// ============================================================
struct GBufferSample {
  pos:       vec3f,
  normal:    vec3f,
  albedo:    vec3f,
  roughness: f32,
  metalness: f32,
  linearDepth: f32,
  wo:        vec3f,   // outgoing direction to camera
  isSky:     bool,
};

// ============================================================
// ReSTIR DI Reservoir (16 bytes)
// ============================================================
struct ReservoirDI {
  lightId: u32,
  M:       u32,
  w_sum:   f32,
  W:       f32,
};

fn emptyReservoirDI() -> ReservoirDI {
  return ReservoirDI(0u, 0u, 0.0, 0.0);
}

fn updateReservoirDI(r: ptr<function, ReservoirDI>, lid: u32, w: f32, rng: ptr<function, u32>) {
  (*r).M += 1u;
  (*r).w_sum += w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).lightId = lid;
  }
}

// ============================================================
// ReservoirDI pack/unpack helpers — canonical, used by ris/temporal/spatial.
// 16 bytes = 4 × u32 per pixel. lightId, M are u32; w_sum and W are
// bit-cast to/from u32 to preserve f32 precision through the storage buffer.
// ============================================================
const RESERVOIR_DI_STRIDE = 4u;

fn loadReservoirDI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32) -> ReservoirDI {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  return ReservoirDI(buf[base], buf[base + 1u], bitcast<f32>(buf[base + 2u]), bitcast<f32>(buf[base + 3u]));
}

fn loadReservoirDI_ro(buf: ptr<storage, array<u32>, read>, pixelIdx: u32) -> ReservoirDI {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  return ReservoirDI(buf[base], buf[base + 1u], bitcast<f32>(buf[base + 2u]), bitcast<f32>(buf[base + 3u]));
}

fn storeReservoirDI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32, r: ReservoirDI) {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  buf[base + 0u] = r.lightId;
  buf[base + 1u] = r.M;
  buf[base + 2u] = bitcast<u32>(r.w_sum);
  buf[base + 3u] = bitcast<u32>(r.W);
}

// ============================================================
// PrimarySurface — derived from re-casting the primary ray through the BVH.
// Replaces the pre-fix placeholder G-buffer reads that returned constant
// values for all pixels. Shared by temporal and spatial passes; shade.wgsl
// reads the same fields inline.
// ============================================================
struct PrimarySurface {
  hit:    bool,
  pos:    vec3f,
  normal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  depth:  f32,
};

// ============================================================
// ReSTIR GI Reservoir (80 bytes, co-located at pixel offset after DI)
// ============================================================
struct ReservoirGI {
  xv:      vec3f,   // visible point (primary hit)
  _pad0:   f32,
  nv:      vec3f,   // normal at xv
  W:       f32,
  xs:      vec3f,   // sample point (secondary bounce hit)
  w_sum:   f32,
  ns:      vec3f,   // normal at xs
  M:       u32,
  Lo:      vec3f,   // outgoing radiance at xs
  lightId: u32,
};

fn emptyReservoirGI() -> ReservoirGI {
  var r: ReservoirGI;
  r.xv = vec3f(0.0); r.nv = vec3f(0,1,0);
  r.xs = vec3f(0.0); r.ns = vec3f(0,1,0);
  r.Lo = vec3f(0.0); r.W = 0.0; r.w_sum = 0.0; r.M = 0u;
  r.lightId = 0u; r._pad0 = 0.0;
  return r;
}

// Sprint 16 — ReservoirGI byte layout (80 bytes = 20 × u32):
//   [0..2]  xv.xyz       [3]    _pad0
//   [4..6]  nv.xyz       [7]    W
//   [8..10] xs.xyz       [11]   w_sum
//   [12..14] ns.xyz      [15]   M
//   [16..18] Lo.xyz      [19]   lightId
// Strided storage in array<u32> (4-byte elements) — stride = 20 u32.
const RESERVOIR_GI_STRIDE: u32 = 20u;

fn loadReservoirGI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32) -> ReservoirGI {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  var r: ReservoirGI;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.lightId = buf[b + 19u];
  return r;
}

fn loadReservoirGI_ro(buf: ptr<storage, array<u32>, read>, pixelIdx: u32) -> ReservoirGI {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  var r: ReservoirGI;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.lightId = buf[b + 19u];
  return r;
}

fn storeReservoirGI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32, r: ReservoirGI) {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  buf[b + 0u]  = bitcast<u32>(r.xv.x);
  buf[b + 1u]  = bitcast<u32>(r.xv.y);
  buf[b + 2u]  = bitcast<u32>(r.xv.z);
  buf[b + 3u]  = bitcast<u32>(r._pad0);
  buf[b + 4u]  = bitcast<u32>(r.nv.x);
  buf[b + 5u]  = bitcast<u32>(r.nv.y);
  buf[b + 6u]  = bitcast<u32>(r.nv.z);
  buf[b + 7u]  = bitcast<u32>(r.W);
  buf[b + 8u]  = bitcast<u32>(r.xs.x);
  buf[b + 9u]  = bitcast<u32>(r.xs.y);
  buf[b + 10u] = bitcast<u32>(r.xs.z);
  buf[b + 11u] = bitcast<u32>(r.w_sum);
  buf[b + 12u] = bitcast<u32>(r.ns.x);
  buf[b + 13u] = bitcast<u32>(r.ns.y);
  buf[b + 14u] = bitcast<u32>(r.ns.z);
  buf[b + 15u] = r.M;
  buf[b + 16u] = bitcast<u32>(r.Lo.x);
  buf[b + 17u] = bitcast<u32>(r.Lo.y);
  buf[b + 18u] = bitcast<u32>(r.Lo.z);
  buf[b + 19u] = r.lightId;
}

fn updateReservoirGI(
  r: ptr<function, ReservoirGI>,
  xs: vec3f, ns: vec3f, Lo: vec3f,
  w: f32,
  rng: ptr<function, u32>,
) {
  (*r).M = (*r).M + 1u;
  (*r).w_sum = (*r).w_sum + w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).xs = xs;
    (*r).ns = ns;
    (*r).Lo = Lo;
  }
}

// ============================================================
// PCG random number generator
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

fn rand2(state: ptr<function, u32>) -> vec2f {
  return vec2f(rand_f32(state), rand_f32(state));
}

fn rand3(state: ptr<function, u32>) -> vec3f {
  return vec3f(rand_f32(state), rand_f32(state), rand_f32(state));
}

// ============================================================
// Utility
// ============================================================
fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  if (len < 1e-8) { return vec3f(0.0, 1.0, 0.0); }
  return v / len;
}

// Build an orthonormal basis around a normal.
fn buildONB(n: vec3f, T: ptr<function, vec3f>, B: ptr<function, vec3f>) {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  *T = normalize(cross(up, n));
  *B = cross(n, *T);
}

// Cosine-hemisphere sample in local space, returns world-space direction.
fn sampleCosineHemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f {
  let xi = rand2(rng);
  let r = sqrt(xi.x);
  let phi = 2.0 * PI * xi.y;
  let localDir = vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - xi.x)));
  var T: vec3f; var B: vec3f;
  buildONB(n, &T, &B);
  return localDir.x * T + localDir.y * B + localDir.z * n;
}

fn cosineHemispherePdf(n: vec3f, wi: vec3f) -> f32 {
  return max(0.0, dot(n, wi)) * INV_PI;
}

// ============================================================
// GGX BRDF (simplified Lambertian + GGX specular)
// ============================================================

// Schlick Fresnel
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
  let c = 1.0 - cosTheta;
  return F0 + (1.0 - F0) * (c * c * c * c * c);
}

// GGX NDF
fn distributionGGX(NdotH: f32, rough: f32) -> f32 {
  let a = rough * rough;
  let a2 = a * a;
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Smith G1 (Schlick approximation)
fn geometrySchlickGGX(NdotV: f32, rough: f32) -> f32 {
  let r = rough + 1.0;
  let k = r * r / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(NdotV: f32, NdotL: f32, rough: f32) -> f32 {
  return geometrySchlickGGX(NdotV, rough) * geometrySchlickGGX(NdotL, rough);
}

// Evaluate GGX BRDF (diffuse + specular).
// albedo: base color, rough: roughness, metalness baked into F0.
fn evalGGX(albedo: vec3f, rough: f32, metal: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let h = safe_normalize(wo + wi);
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  let VdotH = max(0.0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }

  let F0 = mix(vec3f(0.04), albedo, metal);
  let F   = fresnelSchlick(VdotH, F0);
  let D   = distributionGGX(NdotH, max(0.01, rough));
  let G   = geometrySmith(NdotV, NdotL, max(0.01, rough));

  let specular = (D * G * F) / (4.0 * NdotV * NdotL);
  let diffuse  = (1.0 - F) * (1.0 - metal) * albedo * INV_PI;
  return (diffuse + specular) * NdotL;
}

// ============================================================
// BVH ray traversal (adapted from three-mesh-bvh WGSL)
// ============================================================

// Williams 2005 §4 IEEE-safe inverse-direction helper.
// When a direction component is exactly zero, 1/0 = ±Inf is IEEE-valid but
// 0 * ±Inf = NaN can poison the slab test when the ray origin coincides
// with an AABB face.  We substitute a finite large value instead.
// WGSL sign(0) == 0, so for a zero component sign(d.x) * 1e30 == 0, and
// (bMin - origin) * 0 == 0 — the axis contributes zero to tNear/tFar,
// which is correct: a zero-direction ray cannot enter/exit the slab through
// that axis; entry/exit are determined by the other two axes.
fn safeInvDir(d: vec3f) -> vec3f {
  return vec3f(
    select(1.0 / d.x, sign(d.x) * 1e30, abs(d.x) < 1e-30),
    select(1.0 / d.y, sign(d.y) * 1e30, abs(d.y) < 1e-30),
    select(1.0 / d.z, sign(d.z) * 1e30, abs(d.z) < 1e-30),
  );
}

// Returns intersection distance (INFINITY if no hit) -- shadow ray.
// bvh_index is array<vec4u>: .xyz = vertex indices,
//                            .w = packed RGB888 + (trans4 | texType4) byte.
// bvh_position is array<vec4f>: .xyz = world-space position, .w = packed UV.
fn bvhIntersectAny(
  bvh_index:    ptr<storage, array<vec4u>,    read>,
  bvh_position: ptr<storage, array<vec4f>,    read>,
  bvh:          ptr<storage, array<BVHNode>,  read>,
  origin: vec3f,
  dir:    vec3f,
  tMax:   f32,
  triEps: f32,
) -> bool {
  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u; stackPtr++;

  while (stackPtr > 0u) {
    stackPtr--;
    let nodeIdx = stack[stackPtr];
    let node = (*bvh)[nodeIdx];

    // Slab test for bounds -- read array<f32,3> fields by index.
    let nMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let nMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    let invDir = safeInvDir(dir);
    let t1 = (nMin - origin) * invDir;
    let t2 = (nMax - origin) * invDir;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tNear > tFar || tFar < 0.0 || tNear > tMax) { continue; }

    // Leaf test.
    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & 0xFFFF0000u) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000FFFFu;
      let offset = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i++) {
        let triIdx = offset + i;
        let idxEntry = (*bvh_index)[triIdx];
        let idx = idxEntry.xyz;
        // Skip transmissive (glass) triangles in shadow rays so light passes
        // through them.  Transmission lives in bits [7:4] of idxEntry.w
        // (4-bit unorm); glass has transmission > ~0.3 → packed > 4.
        let trans4 = (idxEntry.w >> 4u) & 0xFu;
        if (trans4 > 4u) { continue; }
        let a = (*bvh_position)[idx.x].xyz;
        let b = (*bvh_position)[idx.y].xyz;
        let c = (*bvh_position)[idx.z].xyz;
        let t = intersectTriangle(origin, dir, a, b, c, triEps);
        if (t > 1e-4 && t < tMax) { return true; }
      }
    } else {
      // Interior node: ordered traversal (Wald 2007 / PBR4e §7.3.3).
      // three-mesh-bvh stores rightChildOrTriOffset as a RELATIVE offset (in
      // node units) from the current node, NOT an absolute node index.  The
      // left child is always nodeIdx+1 (the immediately-following node).
      // splitAxisOrTriCount low 2 bits encode the split axis (0=X,1=Y,2=Z).
      // Push far child first so near child is popped (and tested) first.
      // Guard uses 'stackPtr + 1u < 64u' (i.e. index 63 is the last valid slot).
      let rightChild  = nodeIdx + node.rightChildOrTriOffset;
      let axis        = splitOrCount & 0x3u;
      let leftToRight = dir[axis] >= 0.0;
      let nearChild   = select(rightChild,   nodeIdx + 1u, leftToRight);
      let farChild    = select(nodeIdx + 1u, rightChild,   leftToRight);
      if (stackPtr + 1u < 64u) {
        stack[stackPtr] = farChild;  stackPtr++;
        stack[stackPtr] = nearChild; stackPtr++;
      } else {
        // Stack overflow: abandon traversal and return current best
        // (false = not-yet-occluded) rather than silently dropping the
        // far subtree.  At depth 64 a balanced BVH spans 2^64 triangles,
        // so this branch is unreachable for any real scene; the guard
        // exists for invariant clarity, not as a performance path.
        return false;
      }
    }
  }
  return false;
}

// bvhTraceTintedVisibility lives in surfaceTextures.wgsl.ts (shade-only).

// Returns closest hit.
// bvh_index is array<vec4u>: .xyz = vertex indices,
//                            .w = packed RGB888 + (trans4 | texType4) byte.
// bvh_position is array<vec4f>: .xyz = world-space position, .w = packed UV.
fn bvhIntersectFirstHit(
  bvh_index:    ptr<storage, array<vec4u>,    read>,
  bvh_position: ptr<storage, array<vec4f>,    read>,
  bvh:          ptr<storage, array<BVHNode>,  read>,
  ray: Ray,
  triEps: f32,
) -> HitResult {
  var result: HitResult;
  result.didHit = false;
  result.dist = INFINITY;
  result.uv = vec2f(0.0);

  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u; stackPtr++;

  while (stackPtr > 0u) {
    stackPtr--;
    let nodeIdx = stack[stackPtr];
    let node = (*bvh)[nodeIdx];

    let nMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let nMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    let invDir = safeInvDir(ray.direction);
    let t1 = (nMin - ray.origin) * invDir;
    let t2 = (nMax - ray.origin) * invDir;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tNear > tFar || tFar < 0.0 || tNear > result.dist) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & 0xFFFF0000u) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000FFFFu;
      let offset = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i++) {
        let triIdx = offset + i;
        let idxEntry = (*bvh_index)[triIdx];
        let idx = idxEntry.xyz;  // vertex indices in .xyz
        let pa4 = (*bvh_position)[idx.x];
        let pb4 = (*bvh_position)[idx.y];
        let pc4 = (*bvh_position)[idx.z];
        let a = pa4.xyz;
        let b = pb4.xyz;
        let c = pc4.xyz;
        let t = intersectTriangle(ray.origin, ray.direction, a, b, c, triEps);
        if (t > 1e-4 && t < result.dist) {
          result.didHit = true;
          result.dist = t;
          result.triIndex = triIdx;
          result.matColorPacked = idxEntry.w;  // RGB888 + (trans4|texType4)
          // Compute barycentric.
          let p = ray.origin + ray.direction * t;
          let ab = b - a; let ac = c - a; let ap = p - a;
          let d00 = dot(ab, ab); let d01 = dot(ab, ac); let d11 = dot(ac, ac);
          let d20 = dot(ap, ab); let d21 = dot(ap, ac);
          let denom = d00 * d11 - d01 * d01;
          var u = (d11 * d20 - d01 * d21) / denom;
          var v = (d00 * d21 - d01 * d20) / denom;
          u = clamp(u, 0.0, 1.0); v = clamp(v, 0.0, 1.0);
          let bw = 1.0 - u - v;
          result.bary = vec3f(bw, u, v);
          result.normal = safe_normalize(cross(ab, ac));
          // Decode + interpolate per-vertex UV (packed 16:16 unorm in .w of
          // each bvh_position entry).
          let uvA = unpack2x16unorm(bitcast<u32>(pa4.w));
          let uvB = unpack2x16unorm(bitcast<u32>(pb4.w));
          let uvC = unpack2x16unorm(bitcast<u32>(pc4.w));
          result.uv = bw * uvA + u * uvB + v * uvC;
        }
      }
    } else {
      // Interior node: ordered traversal (Wald 2007 / PBR4e §7.3.3).
      // rightChildOrTriOffset is a RELATIVE offset (node units) from this node.
      // splitAxisOrTriCount low 2 bits encode the split axis (0=X,1=Y,2=Z).
      // Push far child first so near child is popped (and tested) first.
      // Guard uses 'stackPtr + 1u < 64u' (i.e. index 63 is the last valid slot).
      let rightChild  = nodeIdx + node.rightChildOrTriOffset;
      let axis        = splitOrCount & 0x3u;
      let leftToRight = ray.direction[axis] >= 0.0;
      let nearChild   = select(rightChild,   nodeIdx + 1u, leftToRight);
      let farChild    = select(nodeIdx + 1u, rightChild,   leftToRight);
      if (stackPtr + 1u < 64u) {
        stack[stackPtr] = farChild;  stackPtr++;
        stack[stackPtr] = nearChild; stackPtr++;
      } else {
        // Stack overflow: bail out with current best-hit rather than
        // silently dropping the far subtree.  At depth 64 a balanced BVH
        // spans 2^64 triangles, so this branch is unreachable for any
        // real scene; the guard exists for invariant clarity.
        return result;
      }
    }
  }
  return result;
}

// Decode RGB888 + (trans4|texType4) packed material data from bvhIndex[triIdx].w.
// Returns vec4f(r, g, b, transmission) in [0, 1].  The texture-type id is
// retrieved separately via decodeSurfaceTextureId.
fn decodeMaterialColor(packed: u32) -> vec4f {
  let r = f32((packed >> 24u) & 0xFFu) / 255.0;
  let g = f32((packed >> 16u) & 0xFFu) / 255.0;
  let b = f32((packed >>  8u) & 0xFFu) / 255.0;
  // Transmission is a 4-bit unorm in bits [7:4] of the low byte.
  let t = f32((packed >> 4u) & 0xFu) / 15.0;
  return vec4f(r, g, b, t);
}

// Decode the authored surface-texture id from bvhIndex[triIdx].w.
// Uses only 3 bits (bits 0-2) — bit 3 of the low nybble is isMetal.
//   0=smooth 1=hammered 2=ripple 3=granite
//   4=baroque 5=waterglass 6=catspaw 7=flemish
fn decodeSurfaceTextureId(packed: u32) -> u32 {
  return packed & 0x7u;
}

// Decode the isMetal flag — true for came / solder / metallic surfaces
// (metalness > 0.5 in the source material). Used to skip the noisy
// Lo_direct ReSTIR DI sampling on thin metallic geometry where the
// single-sample variance produces visible firefly speckle that atrous
// can't smooth across the thin came strips.
fn decodeIsMetal(packed: u32) -> bool {
  return ((packed >> 3u) & 0x1u) != 0u;
}

// Moller-Trumbore triangle intersection; returns t or INFINITY.
// Caller supplies the coplanarity floor as a parameter — typically threaded
// from WalkaroundUBO.triIntersectEpsilon (D12), default 1e-5 (metre-scale).
// Threading via parameter keeps COMMON_WGSL compilable when concatenated
// with shaders that bind a different UBO struct (atrous binds AtrousUBO,
// which has no triIntersectEpsilon member). Without parameterization the
// atrous shader fails to compile at COMMON_WGSL + ATROUS_WGSL link time.
fn intersectTriangle(origin: vec3f, dir: vec3f, a: vec3f, b: vec3f, c: vec3f, triEps: f32) -> f32 {
  let e1 = b - a;
  let e2 = c - a;
  let h = cross(dir, e2);
  let det = dot(e1, h);
  if (abs(det) < triEps) { return INFINITY; }
  let invDet = 1.0 / det;
  let s = origin - a;
  let u = dot(s, h) * invDet;
  if (u < 0.0 || u > 1.0) { return INFINITY; }
  let q = cross(s, e1);
  let v = dot(dir, q) * invDet;
  if (v < 0.0 || u + v > 1.0) { return INFINITY; }
  let t = dot(e2, q) * invDet;
  if (t < triEps) { return INFINITY; }
  return t;
}

// ============================================================
// Emitter sampling helpers
// ============================================================

// Sample a point on an emitter triangle; returns {pos, normal, area, Le, pdfArea}.
struct EmitterSample {
  pos:     vec3f,
  normal:  vec3f,
  Le:      vec3f,
  area:    f32,
  pdfArea: f32,   // uniform-area pdf = 1/area
};

fn sampleEmitterPoint(e: EmitterTri, xi: vec2f) -> EmitterSample {
  // Uniform sampling of a triangle: (1-sqrt(xi.x))*vA + sqrt(xi.x)*(1-xi.y)*vB + sqrt(xi.x)*xi.y*vC
  let s = sqrt(xi.x);
  let u = 1.0 - s;
  let v = s * xi.y;
  let w = s * (1.0 - xi.y);
  let pos = u * e.vA + v * e.vB + w * e.vC;
  var result: EmitterSample;
  result.pos     = pos;
  result.normal  = e.normal;
  result.Le      = e.Le;
  result.area    = e.area;
  result.pdfArea = 1.0 / e.area;
  return result;
}

// Binary search over emitter CDF for importance sampling.
fn sampleEmitterIdx(
  cdf: ptr<storage, array<f32>, read>,
  emitterCount: u32,
  xi: f32,
) -> u32 {
  var lo = 0u;
  var hi = emitterCount;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if ((*cdf)[mid] < xi) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  return min(lo, emitterCount - 1u);
}

// ============================================================
// Jacobian reconnection shift
// ============================================================
fn jacobianReconnectionShift(
  xv_r: vec3f, nv_r: vec3f,  // current pixel primary hit + normal
  xv_q: vec3f,               // neighbor pixel primary hit (source)
  xs:   vec3f, ns: vec3f,    // reconnection vertex + normal (invariant)
) -> f32 {
  let dq = xv_q - xs;
  let dr = xv_r - xs;
  let dq_len2 = dot(dq, dq);
  let dr_len2 = dot(dr, dr);

  if (dr_len2 < 1e-8 || dq_len2 < 1e-8) { return 0.0; }

  let inv_dq_len = inverseSqrt(dq_len2);
  let inv_dr_len = inverseSqrt(dr_len2);

  let cos_theta_q = dot(ns, dq * inv_dq_len);
  let cos_theta_r = dot(ns, dr * inv_dr_len);

  if (cos_theta_q <= 1e-4 || cos_theta_r <= 1e-4) { return 0.0; }

  // Eq. 11 reconnection shift: cosine ratio x inverse-square distance ratio.
  let J = (cos_theta_r / cos_theta_q) * (dq_len2 / dr_len2);
  return clamp(J, 0.1, 10.0);
}

// ============================================================
// Camera helpers (shared by RIS / temporal / spatial / shade)
// ============================================================
// Invert a 4x4 matrix (standard cofactor method).  Used to unproject screen
// coords → world rays for primary-ray-cast mode.
fn invertMat4_common(m: mat4x4f) -> mat4x4f {
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
  let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
  let b00 = a00*a11-a01*a10; let b01 = a00*a12-a02*a10; let b02 = a00*a13-a03*a10;
  let b03 = a01*a12-a02*a11; let b04 = a01*a13-a03*a11; let b05 = a02*a13-a03*a12;
  let b06 = a20*a31-a21*a30; let b07 = a20*a32-a22*a30; let b08 = a20*a33-a23*a30;
  let b09 = a21*a32-a22*a31; let b10 = a21*a33-a23*a31; let b11 = a22*a33-a23*a32;
  let det = b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (abs(det) < 1e-10) { return mat4x4f(); }
  let inv = 1.0/det;
  return mat4x4f(
    vec4f((a11*b11-a12*b10+a13*b09)*inv, (-a01*b11+a02*b10-a03*b09)*inv,
           (a31*b05-a32*b04+a33*b03)*inv,  (-a21*b05+a22*b04-a23*b03)*inv),
    vec4f((-a10*b11+a12*b08-a13*b07)*inv, (a00*b11-a02*b08+a03*b07)*inv,
           (-a30*b05+a32*b02-a33*b01)*inv,  (a20*b05-a22*b02+a23*b01)*inv),
    vec4f((a10*b10-a11*b08+a13*b06)*inv, (-a00*b10+a01*b08-a03*b06)*inv,
           (a30*b04-a31*b02+a33*b00)*inv,  (-a20*b04+a21*b02-a23*b00)*inv),
    vec4f((-a10*b09+a11*b07-a12*b06)*inv, (a00*b09-a01*b07+a02*b06)*inv,
           (-a30*b03+a31*b01-a32*b00)*inv,  (a20*b03-a21*b01+a22*b00)*inv)
  );
}

// Generate a world-space primary ray for pixel (px, py) given the inverse
// view-projection matrix.  Ray origin = camera position; direction unprojects
// the pixel center through near→far in NDC.  Used by ALL passes that need
// to cast primary rays (RIS, shade, temporal, spatial).
fn generatePrimaryRay_common(
  px: u32, py: u32, w: u32, h: u32,
  camPos: vec3f, invVP: mat4x4f,
) -> Ray {
  let uv  = (vec2f(f32(px), f32(py)) + 0.5) / vec2f(f32(w), f32(h));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far4  = invVP * vec4f(ndc,  1.0, 1.0);
  let near4 = invVP * vec4f(ndc, -1.0, 1.0);
  let farW  = far4.xyz  / far4.w;
  let nearW = near4.xyz / near4.w;
  var ray: Ray;
  ray.origin    = camPos;
  ray.direction = safe_normalize(farW - nearW);
  return ray;
}

// ============================================================
// WelfordVariance — canonical struct + helpers imported from
// @vitrum/shared-denoisers (see welfordVariance.wgsl.ts).
// Single source of truth across all variance-aware passes.
// ============================================================
${WELFORD_VARIANCE_WGSL}

// Surface-texture pattern functions live in surfaceTextures.wgsl.ts (shade-only).

`;

/** W1-R6 — declarative include-graph entry. Common is the root of the
 *  dependency tree; everything else opts in via `requires: ['common']`. */
export const COMMON_MODULE: WgslModule = {
  name: 'common',
  source: COMMON_WGSL,
  requires: [],
};
