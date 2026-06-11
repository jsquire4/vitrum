/**
 * cbPrefill.wgsl — Checkerboard pre-denoiser gap-fill pass.
 *
 * Runs BEFORE the denoiser-adapter slot when checkerboard rendering is ON and
 * the active denoiser is one of the four "real" denoisers (svgf-real, bmfr,
 * neural, oidn-final). Those denoisers read `hdrColorTexture` directly, but
 * when checkerboard is active ShadePass writes only the active-parity half —
 * the complementary half holds the previous frame's stale radiance. This pass
 * fills the gap pixels with a motion-vector-reprojected estimate from the
 * previous-frame accumulator (`readAccum`), so the denoiser receives a
 * fully-populated `hdrColorTexture`.
 *
 * Binding layout (4 bindings, own `cb-prefill` BGL):
 *   0  CbPrefillUniforms UBO   (screenW/H, frameParity — 12 bytes, padded to 16)
 *   1  readAccum in            (rgba16float sampled — previous-frame radiance)
 *   2  motionVectorTexture in  (rg32float sampled)
 *   3  hdrColorTexture out     (rgba16float storage write — gap pixels filled)
 *
 * NOTE: hdrColorTexture is not bound as a sampled input here. Only gap pixels
 * are written (identified by (px+py)&1 != frameParity). Shaded pixels are left
 * exactly as ShadePass wrote them — this pass writes only to gap-parity coords.
 *
 * Reprojection convention matches resolve.wgsl (temporal term only — no
 * spatial blend with shaded neighbours, which would require a same-texture
 * read+write conflict):
 *   mv = (dx_ndc, dy_ndc)
 *   pixel_offset = (round(dx*W/2), round(-dy*H/2))
 *   gap_fill = readAccum[clamp(pos - offset)]
 *
 * When `checkerboardOn == 0` this pass gates off entirely (gated in the pass
 * class), so the default atrous/atrous-variance path is byte-identical.
 *
 * @version 1 (2026-06-10 — trust-remediation plan item 6)
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

// clampCoord is provided by the screenCoordHelpers module (D5.4 dedup).

const CB_PREFILL_WGSL = /* wgsl */ `

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct CbPrefillUniforms {
  screenWidth:  u32,
  screenHeight: u32,
  frameParity:  u32,   // frameIndex & 1 — which checkerboard phase is "shaded"
  _pad:         u32,   // align to 16 bytes
};

@group(0) @binding(0) var<uniform>  u_cb: CbPrefillUniforms;

// Previous-frame accumulated radiance (rgba16float, sampled).
// Source for temporal reprojection of gap pixels.
@group(0) @binding(1) var           t_prev_radiance: texture_2d<f32>;

// Motion vectors (rg32float, sampled). r=dx_ndc, g=dy_ndc.
@group(0) @binding(2) var           t_motion_vec:    texture_2d<f32>;

// hdrColorTexture (rgba16float, storage write).
// Only gap pixels are written; shaded pixels remain as written by ShadePass.
@group(0) @binding(3) var           t_hdr_out: texture_storage_2d<rgba16float, write>;

// ── Compute entry point ───────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn cbPrefillKernel(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let px = globalId.x;
  let py = globalId.y;
  let W = u_cb.screenWidth;
  let H = u_cb.screenHeight;

  // Guard against workgroup padding.
  if (px >= W || py >= H) { return; }

  // Shaded pixels were already written by ShadePass — leave them untouched.
  // Only gap-parity pixels need to be filled.
  if (((px + py) & 1u) == u_cb.frameParity) { return; }

  // Temporal reprojection: reproject the previous-frame accumulated radiance
  // through the motion vector to fill the gap pixel.
  // Convention: mv = (dx_ndc, dy_ndc); pixel_offset = (dx*W/2, -dy*H/2).
  // Negate dy because NDC-y is up while pixel-y is down.
  let mv    = textureLoad(t_motion_vec, vec2<i32>(i32(px), i32(py)), 0).rg;
  let dxPx  = i32(round(mv.x * f32(W) * 0.5));
  let dyPx  = i32(round(-mv.y * f32(H) * 0.5));
  let prevXY = clampCoord(vec2<i32>(i32(px) - dxPx, i32(py) - dyPx), W, H);
  let filled = textureLoad(t_prev_radiance, prevXY, 0);

  textureStore(t_hdr_out, vec2<u32>(px, py), filled);
}
`;

/** D5.4 dedup — clampCoord shared via screenCoordHelpers. */
export const CB_PREFILL_MODULE: WgslModule = {
  name: 'cb-prefill',
  source: CB_PREFILL_WGSL,
  requires: ['screenCoordHelpers'],
};
