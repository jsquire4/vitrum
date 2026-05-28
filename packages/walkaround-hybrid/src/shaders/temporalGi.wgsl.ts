/**
 * Sprint 17 — ReSTIR-GI temporal reuse.
 *
 * Reproject the current half-res pixel through the previous-frame camera
 * matrices to find the same world point in the previous reservoir, then
 * combine its sample with this frame's RIS-output reservoir under the
 * standard ReSTIR temporal-reuse rules:
 *
 *   - Geometric consistency: |Δdepth| < 0.1 × depth and |Δnormal| < 25°.
 *   - M-clamp at 20 to bound history accumulation (Bitterli 2020 §5.2).
 *   - Jacobian reconnection shift (common.wgsl jacobianReconnectionShift)
 *     re-weights the prev sample's contribution at this pixel's visible
 *     point so the integrand stays unbiased under shifted reconnections.
 *
 * Reads:  reservoirGiCurrent  (this frame's RIS output)
 *         reservoirGiPrevious (last frame's spatial output)
 *         gNormalDepth        (full-res, sampled at full-res pixel)
 *         WalkaroundUBO       (prevViewMatrix, viewMatrix, projMatrix, screenSize)
 * Writes: reservoirGiCurrent  (in-place — RIS output is consumed, temporally fused)
 *
 * Half-resolution: dispatches W/2 × H/2 invocations, like the RIS pass.
 *
 * Bindings (dedicated bind group, not shared with the frame BGL):
 *   @group(0) @binding(0) reservoirGiCurrent  (storage, read_write)
 *   @group(0) @binding(1) reservoirGiPrevious (storage, read)
 *   @group(0) @binding(2) WalkaroundUBO       (uniform)
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const TEMPORAL_GI_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> tgi_resCurrent: array<u32>;
@group(0) @binding(1) var<storage, read>       tgi_resPrev:    array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

// The temporal-GI M clamp (ubo.restirGiMClamp, Cornell default 50)
// controls how strongly the previous-frame reservoir dominates temporal
// reuse.  Higher = the chosen sample changes less often per-pixel → less
// per-frame pattern jitter (the temporal accumulator's per-frame
// contribution looks stabler).  Bitterli 2020 uses M=20 for ReSTIR-DI;
// Majercik 2021 §4.5 suggests ~30–100 for GI since the indirect signal
// varies less per pixel than DI light-source swaps.  Empirically 50 cuts
// visible pattern dance on Cornell static frames in half compared to 20
// without introducing motion lag (the camera-move reset path forces α=1
// and discards prev independently). Library consumers override via
// HybridEngineOptions.restirGiMClamp.
// Geometric-rejection thresholds for "is this the same world surface".
// 0.1 × current depth = 10 % depth tolerance — generous enough for sub-pixel
// jitter and 1-frame camera motion, tight enough to reject occlusion changes.
const DEPTH_REL_TOL: f32 = 0.1;
const NORMAL_DOT_MIN: f32 = 0.906; // cos(25°)

fn worldFromHalfPx_temporal(halfPx: vec2u, depth: f32, fullDims: vec2u) -> vec3f {
  // Full-res sample centre for this half-res pixel.
  let fullPx = halfPx * 2u + 1u;
  let uv = (vec2f(f32(fullPx.x), f32(fullPx.y)) + 0.5) / vec2f(f32(fullDims.x), f32(fullDims.y));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let invVP = invertMat4_common(ubo.projMatrix * ubo.viewMatrix);
  let far4 = invVP * vec4f(ndc, 1.0, 1.0);
  let near4 = invVP * vec4f(ndc, -1.0, 1.0);
  // Guard: invertMat4_common returns zero matrix for near-singular det; the
  // raw /w would then NaN-poison the ray. See generatePrimaryRay_common for
  // the canonical handling.
  let farW  = far4.xyz  / select(1.0, far4.w,  abs(far4.w)  > 1e-30);
  let nearW = near4.xyz / select(1.0, near4.w, abs(near4.w) > 1e-30);
  let dir = safe_normalize(farW - nearW);
  // Reconstruct world from linear depth = distance along ray from camera.
  return ubo.cameraPos + dir * depth;
}

fn projectToPrevHalfPx(worldPos: vec3f, halfDims: vec2u, fullDims: vec2u) -> vec2i {
  // Use previous-frame view matrix; projection assumed constant (typical
  // case for a non-zooming camera; even with FOV changes the reprojected
  // pixel is still a reasonable starting candidate).
  let prevClip = ubo.projMatrix * ubo.prevViewMatrix * vec4f(worldPos, 1.0);
  if (prevClip.w <= 1e-6) { return vec2i(-1, -1); }
  let ndc = prevClip.xyz / prevClip.w;
  if (any(abs(ndc.xy) > vec2f(1.0))) { return vec2i(-1, -1); }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let fullPxF = uv * vec2f(f32(fullDims.x), f32(fullDims.y));
  let halfPxF = fullPxF * 0.5;
  return vec2i(floor(halfPxF));
}

@compute @workgroup_size(8, 8, 1)
fn temporalGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdx = gid.y * halfDims.x + gid.x;
  var rCur = loadReservoirGI_rw(&tgi_resCurrent, pixelIdx);

  // Need a visible-surface point to reproject. If current's RIS pass wrote
  // an empty reservoir (primary miss / glass / metal), nothing to fuse.
  if (rCur.M == 0u) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Reproject through prev camera. Use the current visible-point xv as the
  // world anchor — same world point in both frames (camera moves, not scene).
  let prevHalfPx = projectToPrevHalfPx(rCur.xv, halfDims, fullDims);
  if (prevHalfPx.x < 0 || prevHalfPx.y < 0
   || u32(prevHalfPx.x) >= halfDims.x || u32(prevHalfPx.y) >= halfDims.y) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  let prevIdx = u32(prevHalfPx.y) * halfDims.x + u32(prevHalfPx.x);
  let rPrev = loadReservoirGI_ro(&tgi_resPrev, prevIdx);

  if (rPrev.M == 0u || rPrev.W <= 0.0) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Geometric-consistency test: compare current vs prev visible-point depth
  // and normal. Reject under occlusion or material swap.
  let dDepth = abs(length(rCur.xv - ubo.cameraPos) - length(rPrev.xv - ubo.cameraPos));
  let depthRef = max(1e-3, length(rCur.xv - ubo.cameraPos));
  if (dDepth / depthRef > DEPTH_REL_TOL) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  if (dot(rCur.nv, rPrev.nv) < NORMAL_DOT_MIN) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // M-clamp: bound prev history before contributing.
  let prevM = min(rPrev.M, ubo.restirGiMClamp);

  // Reconnection-shift jacobian: prev's reservoir holds the (xs, ns, Lo)
  // visible *from* rPrev.xv. We want to weight it as if observed from rCur.xv.
  let J = jacobianReconnectionShift(rCur.xv, rCur.nv, rPrev.xv, rPrev.xs, rPrev.ns);
  if (J <= 0.0) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Compute the prev sample's p̂ at the current pixel.
  let toS = rPrev.xs - rCur.xv;
  let distS2 = dot(toS, toS);
  if (distS2 < 1e-8) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  let wiZ = toS / sqrt(distS2);
  let cosThetaZ = max(0.0, dot(rCur.nv, wiZ));
  let pHatZ_prev = luminance(rPrev.Lo) * cosThetaZ * INV_PI;
  if (pHatZ_prev < 1e-9) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Combined-RIS weight: prev contributes w = p̂_at_cur × W × M × J.
  let w_prev = pHatZ_prev * rPrev.W * f32(prevM) * J;
  // Combine. Mirror the standard ReSTIR temporal-reuse formula.
  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0x71E5u),
    gid.y ^ (ubo.frameSeed * 0xE571u),
    ubo.frameSeed ^ 0x9B7Fu,
  );
  let M_total = rCur.M + prevM;
  updateReservoirGI(&rCur, rPrev.xs, rPrev.ns, rPrev.Lo, w_prev, &rng);
  rCur.M = M_total;

  // Finalise W with the chosen sample's p̂ at this pixel.
  if (rCur.M > 0u) {
    let toSf = rCur.xs - rCur.xv;
    let distSf = length(toSf);
    if (distSf > 1e-4) {
      let wiF = toSf / distSf;
      let cosThetaF = max(0.0, dot(rCur.nv, wiF));
      let pHatF = luminance(rCur.Lo) * cosThetaF * INV_PI;
      let W_raw = select(0.0, rCur.w_sum / (f32(rCur.M) * pHatF), pHatF > 1e-9);
      rCur.W = min(W_raw, ubo.restirGiWCap);
    } else {
      rCur.W = 0.0;
    }
  }

  storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  T9-stepC — narrowed from `['common']` to the modules this pass uses:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - `loadReservoirGI_*` / `storeReservoirGI_rw` / `updateReservoirGI`
 *                                            → reservoirGi
 *    - `pcgInit` / `luminance` / `safe_normalize` → sharedPrimitives
 *    - `jacobianReconnectionShift`           → jacobianShift
 *    - `invertMat4_common` (reprojection)    → cameraRays (uses `Ray` →
 *                                              sceneTraversal)
 *  Verified complete by the static ident-resolution gate. */
export const TEMPORAL_GI_MODULE: WgslModule = {
  name: 'temporalGi',
  source: TEMPORAL_GI_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'jacobianShift', 'cameraRays'],
};
