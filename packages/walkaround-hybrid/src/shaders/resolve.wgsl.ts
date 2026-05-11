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
 *   The shader accepts a motion-vector storage texture (rg32float, r=dx,
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
 * reads — rgba16float and rg32float are tier-1 storage formats whose
 * read access is not portably supported across WebGPU browsers. The
 * upstream textures (writeAccum, readAccum, motionVectorTexture) all
 * carry TEXTURE_BINDING usage for exactly this path.
 *
 * @version 2 (Sprint 9 wire-in, 2026-05-11)
 */

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

// Motion vectors (rg32float). r=dx, g=dy in NDC [-1..1] space. Zero-filled
// fallback when no motion-vector pass is wired (current host state).
@group(0) @binding(3) var           t_motion_vectors:   texture_2d<f32>;

// Full-resolution resolved output (rgba16float). Read by composite pass.
@group(0) @binding(4) var           t_resolved_out:     texture_storage_2d<rgba16float, write>;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Clamp coordinates to valid texture bounds.
fn clampCoord(c: vec2<i32>, w: u32, h: u32) -> vec2<i32> {
  let cx = clamp(c.x, 0, i32(w) - 1);
  let cy = clamp(c.y, 0, i32(h) - 1);
  return vec2<i32>(cx, cy);
}

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
    // ── Gap pixel: reproject from previous frame ────────────────────────────
    //
    // 1. Read motion vector (NDC delta).
    // 2. Convert NDC delta → pixel offset.
    // 3. Clamp to valid texture bounds.
    // 4. Read previous-frame radiance at reprojected position.
    //
    // Motion-vector convention: mv = (dx_ndc, dy_ndc).
    // Pixel offset: (round(dx_ndc * W / 2), round(dy_ndc * H / 2)).
    // NDC x increases right; pixel x increases right. Sign is direct.
    // NDC y increases up; pixel y increases DOWN. Negate dy for pixel space.

    let mv     = textureLoad(t_motion_vectors, vec2<i32>(i32(px), i32(py)), 0).rg;
    let dxPx   = i32(round(mv.x * f32(W) * 0.5));
    let dyPx   = i32(round(-mv.y * f32(H) * 0.5));  // negate: NDC-y vs pixel-y
    let prevXY = clampCoord(vec2<i32>(i32(px) - dxPx, i32(py) - dyPx), W, H);

    radiance = textureLoad(t_prev_radiance, prevXY, 0);
  }

  // Write resolved radiance to the output texture.
  textureStore(t_resolved_out, vec2<u32>(px, py), radiance);
}
`;
