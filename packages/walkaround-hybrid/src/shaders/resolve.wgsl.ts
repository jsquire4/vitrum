/**
 * resolve.wgsl — Checkerboard upsampling + temporal reprojection resolve pass.
 *
 * Sprint 9 (walkaround only). For each output pixel, determines whether it
 * is a "shaded" pixel (covered by this frame's RIS dispatch) or a "gap" pixel
 * (skipped in the checkerboard pattern). Gap pixels are filled by reprojecting
 * from the previous frame's radiance buffer via motion vectors.
 *
 * Checkerboard pattern:
 *   frameParity = frameIndex & 1u
 *   shaded pixel: (px + py) & 1u == frameParity
 *   gap pixel:    (px + py) & 1u != frameParity
 *
 * Motion vector / reprojection:
 *   The shader accepts a motion-vector storage texture (rg32float, r=dx, g=dy
 *   in NDC space). If the binding is present and the motion-vector magnitude
 *   is non-zero, the previous-frame radiance is sampled at
 *   (px - round(dx * screenWidth), py - round(dy * screenHeight)).
 *   If motion vectors are unavailable (zero-vector fallback), the current-frame
 *   center pixel is used (equivalent to zero-motion assumption).
 *
 * Ghosting note:
 *   Sprint 9 DoD: "ghosting acceptable per the existing variance-clamped AABB."
 *   No additional disocclusion or velocity clamping is applied in this shader.
 *   Sprint 10a SVGF will replace the simple reprojection with a variance-guided
 *   version once motion vectors are verified available from the host.
 *
 * Integration status: DEFERRED — shader is complete and runnable but is NOT
 * wired into the dispatch pipeline. See plan/sprint-9-walkaround-integration.md.
 *
 * @version 1 (Sprint 9, 2026-05-09)
 */

export const RESOLVE_WGSL = /* wgsl */ `

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct ResolveUniforms {
  screenWidth:  u32,
  screenHeight: u32,
  frameParity:  u32,   // frameIndex & 1 — which checkerboard phase is "shaded" this frame
  _pad:         u32,
};

@group(0) @binding(0) var<uniform>  u_resolve: ResolveUniforms;

// Current-frame radiance (rgba16float). Written by the shade pass.
// Only shaded pixels have valid data this frame; gap pixels contain stale data.
@group(0) @binding(1) var           t_current_radiance: texture_storage_2d<rgba16float, read>;

// Previous-frame accumulated radiance (rgba16float). Written by temporal accumulator.
// All pixels are valid (full coverage from the previous frame's resolve pass).
@group(0) @binding(2) var           t_prev_radiance: texture_storage_2d<rgba16float, read>;

// Motion vectors (rg32float). r=dx, g=dy in NDC [-1..1] space.
// If not available, pass a 1×1 zero-filled texture and the shader falls back
// to the zero-motion assumption (reads prev pixel at the same screen position).
@group(0) @binding(3) var           t_motion_vectors: texture_storage_2d<rg32float, read>;

// Full-resolution resolved output (rgba16float). Read by next-frame composite pass.
@group(0) @binding(4) var           t_resolved_out: texture_storage_2d<rgba16float, write>;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Clamp coordinates to valid texture bounds.
fn clampCoord(c: vec2<i32>, w: u32, h: u32) -> vec2<u32> {
  let cx = u32(clamp(c.x, 0, i32(w) - 1));
  let cy = u32(clamp(c.y, 0, i32(h) - 1));
  return vec2<u32>(cx, cy);
}

// Determine whether a pixel is shaded this frame.
// shaded if (px + py) & 1 == frameParity.
fn isShadedPixel(px: u32, py: u32, frameParity: u32) -> bool {
  return ((px + py) & 1u) == frameParity;
}

// Read motion vector for pixel (px, py).
// Falls back to zero if the motion-vector texture is 1×1 (not wired).
fn readMotionVector(px: u32, py: u32, mvTexW: u32, mvTexH: u32) -> vec2<f32> {
  // 1×1 texture signals "not available" — return zero (no motion assumption).
  if (mvTexW <= 1u && mvTexH <= 1u) { return vec2<f32>(0.0, 0.0); }
  let raw = textureLoad(t_motion_vectors, vec2<u32>(px, py));
  return raw.rg;
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

  if (isShadedPixel(px, py, u_resolve.frameParity)) {
    // ── Shaded pixel: read directly from current-frame radiance ────────────
    radiance = textureLoad(t_current_radiance, vec2<u32>(px, py));
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

    // Query motion-vector texture dimensions for the 1×1 fallback check.
    let mvDims  = textureDimensions(t_motion_vectors);
    let mvTexW  = mvDims.x;
    let mvTexH  = mvDims.y;

    let mv     = readMotionVector(px, py, mvTexW, mvTexH);
    let dxPx   = i32(round(mv.x * f32(W) * 0.5));
    let dyPx   = i32(round(-mv.y * f32(H) * 0.5));  // negate: NDC-y vs pixel-y
    let prevXY = clampCoord(vec2<i32>(i32(px) - dxPx, i32(py) - dyPx), W, H);

    radiance = textureLoad(t_prev_radiance, prevXY);
  }

  // Write resolved radiance to the output texture.
  textureStore(t_resolved_out, vec2<u32>(px, py), radiance);
}
`;
