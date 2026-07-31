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
 * Binding layout (5 bindings, own `cb-prefill` BGL):
 *   0  CbPrefillUniforms UBO   (screenW/H, frameParity — 12 bytes, padded to 16)
 *   1  readAccum in            (rgba16float sampled — previous-frame radiance)
 *   2  motionVectorTexture in  (rg32float sampled)
 *   3  current shaded snapshot (rgba16float sampled; immutable copy)
 *   4  hdrColorTexture out     (rgba16float storage write — gap pixels filled)
 *
 * NOTE: hdrColorTexture is not bound as a sampled input here. Only gap pixels
 * are written (identified by (px+py)&1 != frameParity). Shaded pixels are left
 * exactly as ShadePass wrote them — this pass writes only to gap-parity coords.
 *
 * Reprojection convention and color-box rejection match resolve.wgsl. The host
 * copies hdrColorTexture to binding 3 immediately before this dispatch, which
 * avoids a same-subresource read/write conflict while exposing the four fresh
 * active-parity neighbours for disocclusion-safe spatial reconstruction:
 *   mv = previousPixel - currentPixel (pixel units, y-down)
 *   gap_fill = readAccum[clamp(currentPixel + round(mv))]
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

// Motion vectors (rgba32float, sampled). rg = previous-current pixel delta.
@group(0) @binding(2) var           t_motion_vec:    texture_2d<f32>;

// Immutable copy of current hdrColorTexture made immediately before dispatch.
@group(0) @binding(3) var           t_current_shaded: texture_2d<f32>;

// hdrColorTexture (rgba16float, storage write).
// Only gap pixels are written; shaded pixels remain as written by ShadePass.
@group(0) @binding(4) var           t_hdr_out: texture_storage_2d<rgba16float, write>;

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
  // Convention: mv is already previous-current in framebuffer pixel units.
  let motionSample = textureLoad(t_motion_vec, vec2<i32>(i32(px), i32(py)), 0);
  let mv = motionSample.rg;
  let deltaPx = vec2i(round(mv));
  let prevXY = clampCoord(vec2<i32>(i32(px), i32(py)) + deltaPx, W, H);
  let temporal = textureLoad(t_prev_radiance, prevXY, 0);

  // Every axis neighbour has the active parity and is therefore fresh in the
  // immutable snapshot. Build the same variance-expanded temporal color box as
  // resolveKernel; out-of-box history is a disocclusion and gets zero weight.
  let xL = clampCoord(vec2<i32>(i32(px) - 1, i32(py)), W, H);
  let xR = clampCoord(vec2<i32>(i32(px) + 1, i32(py)), W, H);
  let yU = clampCoord(vec2<i32>(i32(px), i32(py) - 1), W, H);
  let yD = clampCoord(vec2<i32>(i32(px), i32(py) + 1), W, H);
  let cL = textureLoad(t_current_shaded, xL, 0);
  let cR = textureLoad(t_current_shaded, xR, 0);
  let cU = textureLoad(t_current_shaded, yU, 0);
  let cD = textureLoad(t_current_shaded, yD, 0);
  let spatial = (cL + cR + cU + cD) * 0.25;
  let nMin = min(min(cL, cR), min(cU, cD));
  let nMax = max(max(cL, cR), max(cU, cD));
  let mean2 = (cL * cL + cR * cR + cU * cU + cD * cD) * 0.25;
  let sigma = sqrt(max(mean2 - spatial * spatial, vec4f(0.0)));
  let clipPad = max(vec4f(0.02), sigma * 1.5);
  let temporalFinite = all(temporal == temporal)
    && all(abs(temporal) <= vec4f(65504.0));
  let temporalSafe = select(spatial, temporal, temporalFinite);
  let temporalClipped = clamp(temporalSafe, nMin - clipPad, nMax + clipPad);
  let historyDelta = abs(temporalSafe.rgb - temporalClipped.rgb);
  let maxHistoryDelta = max(max(historyDelta.x, historyDelta.y), historyDelta.z);
  let historyScale = max(
    1.0,
    max(max(abs(temporalClipped.r), abs(temporalClipped.g)), abs(temporalClipped.b)),
  );
  let historyAccepted =
    motionSample.a > 0.5 &&
    temporalFinite &&
    maxHistoryDelta <= 0.25 * historyScale;
  let motionTrust = clamp(1.0 - length(mv) * 0.25, 0.0, 1.0);
  let temporalWeight = select(0.0, motionTrust, historyAccepted);
  let filled = mix(spatial, temporalClipped, temporalWeight);

  textureStore(t_hdr_out, vec2<u32>(px, py), filled);
}
`;

/** D5.4 dedup — clampCoord shared via screenCoordHelpers. */
export const CB_PREFILL_MODULE: WgslModule = {
  name: 'cb-prefill',
  source: CB_PREFILL_WGSL,
  requires: ['screenCoordHelpers'],
};
