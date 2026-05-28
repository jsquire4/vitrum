/**
 * Temporal reuse compute pass.
 *
 * Projects current pixel through previous MVP to find the previous-frame
 * reservoir, then combines via GRIS with M-clamp = 20.
 *
 * Primary-ray-cast mode: no G-buffer rasterization.  We re-cast the
 * primary ray here to get the world-space hit `pos` and `normal`, then
 * reproject `pos` through the previous-frame view+projection matrix to find
 * the previous pixel.  This replaces the placeholder motion-vector texture
 * (which returned a constant offset, making temporal look-up land in the
 * wrong screen quadrant for ~all pixels).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const TEMPORAL_WGSL = /* wgsl */ `

// Group 0: only the slots temporal actually reads / writes. The shared
// FrameBindGroup layout carries 10 entries for shade; WGSL allows the
// shader to declare a subset (W5-I1 cleanup 2026-05-18).
@group(0) @binding(5) var<storage, read_write> currentReservoir:  array<u32>;
@group(0) @binding(6) var<storage, read>       previousReservoir: array<u32>;

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

// WalkaroundUBO struct defined in COMMON_WGSL.
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

// RESERVOIR_DI_STRIDE / loadReservoirDI_{rw,ro} / storeReservoirDI_rw live in COMMON_WGSL.
// M_CLAMP is read from the UBO (ubo.temporalMClampDI).

// PrimarySurface struct defined in COMMON_WGSL.
// W2-C9 — primary-surface cast moved to restirCastPrimary.wgsl
// (canonical castPrimary(px, dims, camPos, invVP)).

// Reproject a world-space position through the previous-frame view+proj
// matrix.  Returns the previous pixel as ivec2 or -1 outside the frustum.
// We assume the projection matrix is unchanged between frames (FOV+aspect
// are static), so we use the current projMatrix together with the stored
// prevViewMatrix — matching how WalkaroundStage feeds the UBO
// (prevProjMatrix=projMatrix in the captureFrame path).
fn reprojectToPrev(world: vec3f, dims: vec2u) -> vec2i {
  let prevView = ubo.prevViewMatrix;
  let proj     = ubo.projMatrix;
  let clip = proj * (prevView * vec4f(world, 1.0));
  if (clip.w <= 0.0) { return vec2i(-1, -1); }
  let ndc = clip.xyz / clip.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) { return vec2i(-1, -1); }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
  let px = vec2i(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y)));
  return px;
}

// W2-C7 — p̂ moved to restirPHat.wgsl
// (canonical restir_di_compute_phat_from_surface(lid, surf)).

@compute @workgroup_size(8, 8, 1)
fn temporalMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  if (any(gid.xy >= dims)) { return; }

  let pixelIdx = gid.y * dims.x + gid.x;
  var rng = pcgInit(gid.x ^ 12345u, gid.y ^ 67890u, ubo.frameSeed ^ 0xABCDu);

  var cur = loadReservoirDI_rw(&currentReservoir, pixelIdx);

  // Re-cast current pixel's primary ray to get the actual surface.
  let vp    = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let curSurf = castPrimary(gid.xy, dims, ubo.cameraPos, invVP);
  if (!curSurf.hit) {
    // Sky pixel — nothing to project; pass-through.
    storeReservoirDI_rw(&currentReservoir, pixelIdx, cur);
    return;
  }

  // Reproject this surface's world position through the previous-frame view
  // matrix to find the matching previous-frame pixel.  Replaces the pre-fix
  // placeholder motion-vector lookup, which read a constant (0.5, 0.5) and
  // sent prevPx far off-screen for ~half of the frame.
  let prevPx = reprojectToPrev(curSurf.pos, dims);
  if (any(prevPx < vec2i(0)) || any(prevPx >= vec2i(dims))) {
    storeReservoirDI_rw(&currentReservoir, pixelIdx, cur);
    return;
  }

  let prevIdx = u32(prevPx.y) * dims.x + u32(prevPx.x);
  var prev = loadReservoirDI_ro(&previousReservoir, prevIdx);

  // Note: there is no explicit disocclusion gate here.  The implicit gate is
  // the p̂ re-evaluation below — if the previous reservoir's lightId is
  // occluded or back-facing at the current surface, p̂≈0 and the sample
  // contributes ~nothing (w_prev → 0).

  // M-clamp previous reservoir — UBO-driven for scene-independence.
  prev.M = min(prev.M, ubo.temporalMClampDI);

  // Evaluate p̂ at CURRENT pixel for the previous reservoir's chosen light.
  let pHatPrevAtCur = restir_di_compute_phat_from_surface(prev.lightId, curSurf);
  let w_prev = pHatPrevAtCur * prev.W * f32(prev.M);

  // Combine reservoirs.
  var combined = cur;
  combined.M += prev.M;
  combined.w_sum += w_prev;
  if (rand_f32(&rng) * combined.w_sum < w_prev && w_prev > 0.0) {
    combined.lightId = prev.lightId;
    // 2026-05-18 sweep finding #3 — carry the chosen sample's xi forward
    // so a downstream visibility test (this frame's spatial pass + shade)
    // reconstructs the same point on the light, not the centroid.
    combined.xi      = prev.xi;
  }

  // Recompute W.
  let pHatZ = restir_di_compute_phat_from_surface(combined.lightId, curSurf);
  combined.W = select(0.0, combined.w_sum / (f32(combined.M) * pHatZ), pHatZ > 0.0);

  storeReservoirDI_rw(&currentReservoir, pixelIdx, combined);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  W2-C7+C9: depends on the canonical ReSTIR p̂ and primary-cast helpers
 *  (both of which transitively require `common`). The composer dedupes
 *  `common` so the emitted order is `common, restirPHat, restirCastPrimary,
 *  temporal`. */
export const TEMPORAL_MODULE: WgslModule = {
  name: 'temporal',
  source: TEMPORAL_WGSL,
  requires: ['restirPHat', 'restirCastPrimary'],
};
