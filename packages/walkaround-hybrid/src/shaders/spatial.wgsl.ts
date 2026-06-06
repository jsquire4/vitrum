/**
 * Spatial reuse compute pass.
 *
 * Combines 5 spatially-neighboring reservoirs via Poisson-disk offsets (30px radius).
 * Reads from currentReservoir, writes to spatialReservoir.
 * Two separable passes are run by the orchestrator (using the same shader twice).
 *
 * Primary-ray-cast mode: no G-buffer rasterization.  We re-cast primary
 * rays here for the center pixel + each neighbor so the target function p̂ is
 * evaluated at the CORRECT surface, not at the world origin.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SPATIAL_WGSL = /* wgsl */ `

// Group 0: only the slots spatial actually reads / writes. The shared
// FrameBindGroup layout carries 10 entries for shade; WGSL allows the
// shader to declare a subset (W5-I1 cleanup 2026-05-18).
@group(0) @binding(5) var<storage, read_write> currentReservoir:  array<u32>;
@group(0) @binding(7) var<storage, read_write> spatialReservoir:  array<u32>;

// bvh_index is array<vec4u>: .xyz=vertex indices, .w=packed RGBA8 material color+transmission
@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;
@group(1) @binding(4) var<storage, read> emitterCdf:   array<f32>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(1) @binding(11) var<storage, read> bvh_normal: array<vec4f>;

// WalkaroundUBO struct defined in COMMON_WGSL.
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

// RESERVOIR_DI_STRIDE / loadReservoirDI_rw / storeReservoirDI_rw live in COMMON_WGSL.
// NEIGHBORS = 5 (restored — was briefly 3 for perf). The spatial-2
// pass was also restored alongside.
const NEIGHBORS = 5u;
// RADIUS is now read from ubo.spatialReuseRadiusPx (derived from
// HybridEngineOptions.spatialReuseRadiusFraction × screenHeight).
const M_SCALE = 4u;

// PrimarySurface struct defined in COMMON_WGSL.
// W2-C9 — primary-surface cast moved to restirCastPrimary.wgsl
// (canonical castPrimary(px, dims, camPos, invVP)).
// W2-C7 — p̂ moved to restirPHat.wgsl
// (canonical restir_di_compute_phat_from_surface(lid, surf)).

// Poisson disk offsets (normalized, scale by RADIUS in the shader).
fn poissonDisk(i: u32, rotation: f32) -> vec2f {
  var offsets: array<vec2f, 8> = array<vec2f, 8>(
    vec2f( 0.0,      1.0     ),
    vec2f( 0.866,    0.5     ),
    vec2f( 0.866,   -0.5     ),
    vec2f( 0.0,     -1.0     ),
    vec2f(-0.866,   -0.5     ),
    vec2f(-0.866,    0.5     ),
    vec2f( 0.354,    0.354   ),
    vec2f(-0.354,   -0.354   ),
  );
  let o = offsets[i % 8u];
  let s = sin(rotation);
  let c = cos(rotation);
  return vec2f(o.x * c - o.y * s, o.x * s + o.y * c);
}

@compute @workgroup_size(8, 8, 1)
fn spatialMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  if (any(gid.xy >= dims)) { return; }

  let pixelIdx = gid.y * dims.x + gid.x;
  var rng = pcgInit(gid.x ^ 54321u, gid.y ^ 98765u, ubo.frameSeed ^ 0xCAFEu);

  var r = loadReservoirDI_rw(&currentReservoir, pixelIdx);

  // M-scale down before spatial.
  if (r.M > M_SCALE) {
    r.w_sum = r.w_sum * f32(M_SCALE) / f32(r.M);
    r.M = M_SCALE;
  }

  // Re-cast the center pixel's primary ray to get the actual surface — needed
  // both for the similarity gate (we compare against neighbor surfaces, not
  // against placeholder textures) and for evaluating p̂ at the right pos/normal.
  let vp    = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let center = castPrimary(gid.xy, dims, ubo.cameraPos, invVP);
  if (!center.hit) {
    // Sky pixel — no reservoir to combine; pass current through unchanged.
    storeReservoirDI_rw(&spatialReservoir, pixelIdx, r);
    return;
  }

  let rotation = rand_f32(&rng) * 6.2831;

  for (var i = 0u; i < NEIGHBORS; i++) {
    let offset = poissonDisk(i, rotation);
    let nbrPx  = vec2i(gid.xy) + vec2i(vec2f(offset.x * ubo.spatialReuseRadiusPx, offset.y * ubo.spatialReuseRadiusPx));
    if (any(nbrPx < vec2i(0)) || any(nbrPx >= vec2i(dims))) { continue; }
    let nbrIdx = u32(nbrPx.y) * dims.x + u32(nbrPx.x);

    // Geometric similarity gate computed from BVH-cast primary surfaces.
    let nbr_surf = castPrimary(vec2u(nbrPx), dims, ubo.cameraPos, invVP);
    if (!nbr_surf.hit) { continue; }
    let depthDiff = abs(center.depth - nbr_surf.depth);
    // Relative 10% depth tolerance, with an absolute floor from the UBO.
    // spatialDepthTolFloor is derived from scene scale in uboUpdater.ts;
    // default ~0.001 preserves near-zero tolerance for Cornell-scale scenes.
    let depthTol  = max(ubo.spatialDepthTolFloor, 0.10 * center.depth);
    let normalDot = dot(center.normal, nbr_surf.normal);
    if (depthDiff > depthTol || normalDot < 0.9) { continue; }

    let nbr  = loadReservoirDI_rw(&currentReservoir, nbrIdx);
    let nbrM = max(1u, nbr.M / M_SCALE);

    // Re-evaluate p̂ at the CENTER surface for the neighbor's chosen light.
    let pHatNbrAtCenter = restir_di_compute_phat_from_surface(nbr.lightId, center);
    let w = pHatNbrAtCenter * nbr.W * f32(nbrM);

    r.M += nbrM;
    r.w_sum += w;
    if (rand_f32(&rng) * r.w_sum < w && w > 0.0) {
      r.lightId = nbr.lightId;
      // 2026-05-18 sweep finding #3 — carry the neighbor's sample-xi
      // forward so the next visibility test reconstructs the same point
      // on the light, not the centroid.
      r.xi      = nbr.xi;
    }
  }

  // Recompute W.
  let pHatZ = restir_di_compute_phat_from_surface(r.lightId, center);
  r.W = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 0.0);

  storeReservoirDI_rw(&spatialReservoir, pixelIdx, r);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  W2-C7+C9: depends on the canonical ReSTIR p̂ and primary-cast helpers
 *  (both of which transitively require `common`). The composer dedupes
 *  `common` so the emitted order is `common, restirPHat, restirCastPrimary,
 *  spatial`. */
export const SPATIAL_MODULE: WgslModule = {
  name: 'spatial',
  source: SPATIAL_WGSL,
  requires: ['restirPHat', 'restirCastPrimary'],
};
