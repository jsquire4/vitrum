/**
 * resolve.wgsl — Checkerboard upsampling + temporal reprojection resolve pass.
 *
 * Sprint 9 — Walkaround adaptive sampling, sparse-shade companion pass.
 * For each output pixel, determines whether it is a "shaded" pixel
 * (covered by this frame's checkerboard write) or a "gap" pixel (skipped).
 * Shaded pixels copy through from the current radiance source; gap pixels
 * are filled by reprojecting from the previous frame's radiance buffer via
 * the motion-vector G-buffer slot.
 *
 * Checkerboard pattern:
 *   frameParity = frameIndex & 1u
 *   shaded pixel: (px + py) & 1u == frameParity
 *   gap pixel:    (px + py) & 1u != frameParity
 *
 * Motion vector / reprojection:
 *   The shader accepts a motion-vector texture (rgba32float, r=dx,
 *   g=dy in NDC space). If the binding is present and the motion-vector
 *   magnitude is non-zero, the previous-frame radiance is sampled at
 *   (px - round(dx * screenWidth), py - round(dy * screenHeight)).
 *   If motion vectors are unavailable (zero-vector fallback), the
 *   current-frame center pixel is used (equivalent to zero-motion
 *   assumption).
 *
 * Pipeline placement:
 *   slot N-2 — temporalAccum (writes writeAccum)
 *   slot N-1 — resolve       (this pass; reads writeAccum + readAccum,
 *                             writes resolvedTexture)
 *   slot N   — composite     (reads resolvedTexture)
 *
 * Ghosting note:
 *   Sprint 9 DoD: "ghosting acceptable per the existing variance-clamped
 *   AABB." No additional disocclusion or velocity clamping is applied in
 *   this shader. Sprint 10a SVGF will replace the simple reprojection
 *   with a variance-guided version once motion vectors are verified
 *   available from the host.
 *
 * Bindings use `texture_2d<f32>` (sampled, unfilterable-float) for all
 * reads — storage-read access is not portably supported across WebGPU
 * browsers for every relevant format. The
 * upstream textures (writeAccum, readAccum, motionVectorTexture) all
 * carry TEXTURE_BINDING usage for exactly this path.
 *
 * @version 2 (Sprint 9 wire-in, 2026-05-11)
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

// clampCoord is provided by the screenCoordHelpers module (D5.4 dedup).

export const RESOLVE_WGSL = /* wgsl */ `

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct ResolveUniforms {
  screenWidth:    u32,
  screenHeight:   u32,
  frameParity:    u32,   // frameIndex & 1 — which checkerboard phase is "shaded" this frame
  checkerboardOn: u32,   // 0 = passthrough (all pixels shaded); 1 = checkerboard sparse-shade
};

@group(0) @binding(0) var<uniform>  u_resolve: ResolveUniforms;

// Current-frame radiance (rgba16float). Source: writeAccum from temporalAccum.
@group(0) @binding(1) var           t_current_radiance: texture_2d<f32>;

// Previous-frame accumulated radiance (rgba16float). Source: readAccum.
@group(0) @binding(2) var           t_prev_radiance:    texture_2d<f32>;

// Motion vectors (rgba32float). r=dx, g=dy in NDC [-1..1] space. Zero-filled
// fallback when no motion-vector pass is wired (current host state).
@group(0) @binding(3) var           t_motion_vectors:   texture_2d<f32>;

// Full-resolution resolved output (rgba16float). Read by composite pass.
@group(0) @binding(4) var           t_resolved_out:     texture_storage_2d<rgba16float, write>;

// ── Helpers ───────────────────────────────────────────────────────────────────

// clampCoord is provided by screenCoordHelpers (D5.4 dedup — shared with cbPrefill).

// Determine whether a pixel is shaded this frame.
// When checkerboardOn is 0 (passthrough mode), all pixels are treated as
// shaded so the gap-fill branch is dormant — that's the wire-in state until
// shade.wgsl is upgraded to write sparsely. When checkerboardOn is 1, the
// checkerboard pattern controls which half is shaded vs gap.
fn isShadedPixel(px: u32, py: u32, frameParity: u32, checkerboardOn: u32) -> bool {
  if (checkerboardOn == 0u) { return true; }
  return ((px + py) & 1u) == frameParity;
}

// ── Compute entry point ───────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn resolveKernel(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let px = globalId.x;
  let py = globalId.y;
  let W = u_resolve.screenWidth;
  let H = u_resolve.screenHeight;

  // Guard against workgroup padding.
  if (px >= W || py >= H) { return; }

  var radiance: vec4<f32>;

  if (isShadedPixel(px, py, u_resolve.frameParity, u_resolve.checkerboardOn)) {
    // ── Shaded pixel: read directly from current-frame radiance ────────────
    radiance = textureLoad(t_current_radiance, vec2<i32>(i32(px), i32(py)), 0);
  } else {
    // ── Gap pixel: hybrid temporal-reproject ⊕ spatial-neighbour fill ────────
    //
    // TEMPORAL term: reproject the previous frame's radiance through the motion
    // vector (sharp when reprojection is valid). Motion-vector convention:
    // mv = (dx_ndc, dy_ndc); pixel offset = (round(dx*W/2), round(-dy*H/2))
    // (NDC-y up vs pixel-y down → negate dy).
    let mv     = textureLoad(t_motion_vectors, vec2<i32>(i32(px), i32(py)), 0).rg;
    let dxPx   = i32(round(mv.x * f32(W) * 0.5));
    let dyPx   = i32(round(-mv.y * f32(H) * 0.5));
    let prevXY = clampCoord(vec2<i32>(i32(px) - dxPx, i32(py) - dyPx), W, H);
    let temporal = textureLoad(t_prev_radiance, prevXY, 0);

    // SPATIAL term: average the 4 axis neighbours. In the checkerboard pattern
    // every gap pixel's 4-neighbours have the OPPOSITE parity → they were
    // SHADED this frame → their current radiance is fresh (no temporal lag, so
    // no ghosting under motion). This is the disocclusion-safe fallback: a gap
    // pixel that just appeared (no valid history) reconstructs from current
    // shaded neighbours rather than stale reprojected history.
    let xL = clampCoord(vec2<i32>(i32(px) - 1, i32(py)), W, H);
    let xR = clampCoord(vec2<i32>(i32(px) + 1, i32(py)), W, H);
    let yU = clampCoord(vec2<i32>(i32(px), i32(py) - 1), W, H);
    let yD = clampCoord(vec2<i32>(i32(px), i32(py) + 1), W, H);
    let spatial = (textureLoad(t_current_radiance, xL, 0)
                 + textureLoad(t_current_radiance, xR, 0)
                 + textureLoad(t_current_radiance, yU, 0)
                 + textureLoad(t_current_radiance, yD, 0)) * 0.25;

    // BLEND by motion magnitude: small motion (<~1px) trusts the sharp temporal
    // reprojection; as motion grows the reprojected history becomes unreliable
    // (ghosting/disocclusion), so fade to the always-current spatial average.
    // Full spatial by ~4px of motion. This is the cheap, binding-free analogue
    // of an SVGF disocclusion reject (Schied 2017) + checkerboard neighbour
    // fallback (El Mansouri 2016); a depth/normal-gated reject is a future
    // refinement once this is enabled.
    let motionPx   = length(vec2<f32>(f32(dxPx), f32(dyPx)));
    let wTemporal  = clamp(1.0 - motionPx * 0.25, 0.0, 1.0);
    radiance = mix(spatial, temporal, wTemporal);
  }

  // Write resolved radiance to the output texture.
  textureStore(t_resolved_out, vec2<u32>(px, py), radiance);
}
`;

/** D5.4 dedup — clampCoord shared via screenCoordHelpers. */
export const RESOLVE_MODULE: WgslModule = {
  name: 'resolve',
  source: RESOLVE_WGSL,
  requires: ['screenCoordHelpers'],
};
