/**
 * Sprint 17 — ReSTIR-GI spatial reuse.
 *
 * Per half-res pixel, pull K_SPATIAL = 5 random neighbours from within a
 * disc of radius SPATIAL_RADIUS px. For each accepted neighbour:
 *
 *   1. Geometric-consistency reject if normal mismatch or depth jump.
 *   2. Compute reconnection-shift jacobian J at the current pixel.
 *   3. Compute p̂(z_q) at the current pixel.
 *   4. Combine into RIS reservoir with weight  w_q = p̂(z_q) · W_q · M_q · J.
 *
 * Finalise W from the chosen-sample p̂. M clamps at 500 to bound variance.
 *
 * Run twice per frame (current → spatial, then spatial → current) with
 * different RNG seeds for full-resolution-equivalent coverage.
 *
 * Bindings (dedicated bind group, ping-pong via two distinct bind groups
 * that swap reservoirGiCurrent / reservoirGiSpatial between in and out):
 *   @group(0) @binding(0) input  reservoir (storage, read)
 *   @group(0) @binding(1) output reservoir (storage, read_write)
 *   @group(0) @binding(2) WalkaroundUBO    (uniform)
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SPATIAL_GI_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read>       sgi_resIn:  array<u32>;
@group(0) @binding(1) var<storage, read_write> sgi_resOut: array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

const K_SPATIAL_GI: u32 = 5u;
const SPATIAL_RADIUS_GI: f32 = 12.0;     // pixels (half-res space)
const M_CLAMP_SPATIAL: u32 = 500u;
const NORMAL_DOT_MIN_S: f32 = 0.906;     // cos(25°)
// Coplanar-distance tolerance — neighbour must lie within this perpendicular
// distance of the centre pixel's tangent plane.  Replaces the older
// camera-distance ratio test (DEPTH_REL_TOL_S) which rejected neighbours
// in corner geometry where the same wall recedes from the camera at a
// steep angle — verified via reservoir probe that the camera-ratio test
// was rejecting essentially all 5 neighbours on left-wall-near-back-corner
// pixels, locking each pixel into its own initial-RIS sample.  The plane
// test instead asks "are these points on the same surface" which is what
// the spatial filter actually needs.  0.05 world units = 5 cm tolerance.
const COPLANAR_TOL_S: f32 = 0.05;

fn sampleDiscPx(rng: ptr<function, u32>) -> vec2f {
  let r = SPATIAL_RADIUS_GI * sqrt(rand_f32(rng));
  let phi = 6.2831853 * rand_f32(rng);
  return vec2f(r * cos(phi), r * sin(phi));
}

@compute @workgroup_size(8, 8, 1)
fn spatialGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdx = gid.y * halfDims.x + gid.x;
  var rCenter = loadReservoirGI_ro(&sgi_resIn, pixelIdx);

  // No surface here — skip reuse, copy through.
  if (rCenter.M == 0u) {
    storeReservoirGI_rw(&sgi_resOut, pixelIdx, rCenter);
    return;
  }

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA127u),
    gid.y ^ (ubo.frameSeed * 0x271Au),
    ubo.frameSeed ^ 0xBCD3u,
  );

  var rOut = rCenter;
  let centerDepth = max(1e-3, length(rCenter.xv - ubo.cameraPos));

  for (var i: u32 = 0u; i < K_SPATIAL_GI; i = i + 1u) {
    let off = sampleDiscPx(&rng);
    let qx = i32(gid.x) + i32(round(off.x));
    let qy = i32(gid.y) + i32(round(off.y));
    if (qx < 0 || qy < 0
     || u32(qx) >= halfDims.x || u32(qy) >= halfDims.y) { continue; }
    if (qx == i32(gid.x) && qy == i32(gid.y)) { continue; }

    let qIdx = u32(qy) * halfDims.x + u32(qx);
    let rQ = loadReservoirGI_ro(&sgi_resIn, qIdx);
    if (rQ.M == 0u || rQ.W <= 0.0) { continue; }

    // Geometric-consistency: normal alignment + coplanarity to centre pixel.
    if (dot(rCenter.nv, rQ.nv) < NORMAL_DOT_MIN_S) { continue; }
    let planeDist = abs(dot(rQ.xv - rCenter.xv, rCenter.nv));
    if (planeDist > COPLANAR_TOL_S) { continue; }

    // Jacobian shift: rQ's reservoir holds (xs, ns, Lo) seen from rQ.xv;
    // re-weight it for evaluation at rCenter.xv.
    let J = jacobianReconnectionShift(rCenter.xv, rCenter.nv, rQ.xv, rQ.xs, rQ.ns);
    if (J <= 0.0) { continue; }

    // p̂ at center pixel.
    let toS = rQ.xs - rCenter.xv;
    let distS = length(toS);
    if (distS < 1e-4) { continue; }
    let wiZ = toS / distS;
    let cosThetaZ = max(0.0, dot(rCenter.nv, wiZ));
    let pHatZ = luminance(rQ.Lo) * cosThetaZ * INV_PI;
    if (pHatZ < 1e-9) { continue; }

    let Mq = min(rQ.M, M_CLAMP_SPATIAL);
    let w_q = pHatZ * rQ.W * f32(Mq) * J;
    let oldM = rOut.M;
    updateReservoirGI(&rOut, rQ.xs, rQ.ns, rQ.Lo, w_q, &rng);
    rOut.M = oldM + Mq;
  }

  // Finalise W from the chosen sample's p̂ at this pixel.
  if (rOut.M > 0u) {
    let toSf = rOut.xs - rOut.xv;
    let distSf = length(toSf);
    if (distSf > 1e-4) {
      let wiF = toSf / distSf;
      let cosThetaF = max(0.0, dot(rOut.nv, wiF));
      let pHatF = luminance(rOut.Lo) * cosThetaF * INV_PI;
      let W_raw = select(0.0, rOut.w_sum / (f32(rOut.M) * pHatF), pHatF > 1e-9);
      rOut.W = min(W_raw, RESTIR_GI_W_CAP);
    } else {
      rOut.W = 0.0;
    }
  }

  storeReservoirGI_rw(&sgi_resOut, pixelIdx, rOut);
}
`;

/** W1-R6 — declarative include-graph entry. */
export const SPATIAL_GI_MODULE: WgslModule = {
  name: 'spatialGi',
  source: SPATIAL_GI_WGSL,
  requires: ['common'],
};
