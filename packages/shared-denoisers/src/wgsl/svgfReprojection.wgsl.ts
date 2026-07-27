/**
 * svgfReprojection.wgsl.ts — Schied 2017 SVGF temporal-reprojection pass.
 *
 * Implements the full Stage 1 temporal accumulation:
 *   Eq. 1 — Bilinear gather of previous-frame radiance + moments at the
 *            reprojected screen-space position (from motion vectors).
 *   Eq. 2 — Per-tap disocclusion test: reject if relative depth deviation
 *            > σ_z, if normal dot-product < σ_n, or object-id mismatch.
 *   Eq. 3 — History length: h ← h_prev + 1 (accept) or h ← 1 (reject).
 *   Eq. 4 — EMA α-clamp: α = max(α_min, 1 / h), applied to
 *            color, M1 (first moment), and M2 (second moment).
 *
 * Bind group 0 layout (one entry point: svgfReprojMain):
 *   binding 0 — texture_2d<f32>                       currColor        (rgba16float noisy current frame)
 *   binding 1 — texture_2d<f32>                        prevColor        (rgba16float previous frame EMA output)
 *   binding 2 — texture_2d<f32>                        motionVec        (rg32float previous-minus-current pixel delta)
 *   binding 3 — texture_2d<f32>                        currDepth        (r32float or rgba16float .r linear depth)
 *   binding 4 — texture_2d<f32>                        currNormal       (rgba16float .xyz world-normal packed 0..1)
 *   binding 5 — texture_2d<u32>                        currObjId        (r8uint or r16uint object identifier)
 *   binding 6 — texture_2d<f32>                        prevDepth        (r32float, previous frame)
 *   binding 7 — texture_2d<f32>                        prevNormal       (rgba16float .xyz, previous frame)
 *   binding 8 — texture_2d<u32>                        prevObjId        (same format as currObjId)
 *   binding 9 — texture_2d<u32>                        historyLengthIn  (r16uint persistent per-pixel history)
 *   binding 10 — texture_2d<f32>                       momentsIn        (rg32float M1, M2 previous frame)
 *   binding 11 — texture_storage_2d<rgba16float, write> colorOut        (blended current-frame color)
 *   binding 12 — texture_storage_2d<r16uint, write>    historyOut       (updated per-pixel history h)
 *   binding 13 — texture_storage_2d<rgba32float, write> momentsOut      (updated M1, M2)
 *   binding 14 — var<uniform>                          reprojUBO        (SVGFReprojUBO)
 *
 * References:
 *   Schied, C., et al. "Spatiotemporal Variance-Guided Filtering" HPG 2017.
 *   §4, Equations 1–4.
 *
 * GPU memory budget note:
 *   At 1920×1080:
 *     historyLength (r16uint):  1920×1080×2  ≈  4 MB
 *     momentsHistory (rg32float): 1920×1080×8 ≈ 16 MB
 *     prevColor (rgba16float):  1920×1080×8  ≈ 16 MB
 *     motionVec (rg32float):    1920×1080×8  ≈ 16 MB
 *   Total new persistent textures for svgf-real: ~52 MB at 1080p.
 */

import { LUMINANCE_WGSL } from '@vitrum/shared-samplers';
import {
  STANDALONE_DEPTH_TEXTURE_LAYOUT,
  normalDepthWgslDepthComponent,
  type NormalDepthTextureLayout,
} from '../normalDepthEncoding.js';

/** Must match @workgroup_size in svgfReprojMain. */
export const SVGF_REAL_REPROJECTION_WORKGROUP_SIZE = 16 as const;

export function buildSvgfReprojectionWgsl(
  depthLayout: NormalDepthTextureLayout = STANDALONE_DEPTH_TEXTURE_LAYOUT,
): string {
  const depthComponent = normalDepthWgslDepthComponent(depthLayout);
  return /* wgsl */ `
${LUMINANCE_WGSL}
// ============================================================
// SVGFReprojUBO — tunable disocclusion + EMA constants
// ============================================================
struct SVGFReprojUBO {
  // σ_z: max relative depth deviation for acceptance (Schied Eq. 2).
  // Default 0.1 (10% relative difference). Scene-scale-independent.
  sigmaDepth:   f32,
  // σ_n: minimum dot-product of current vs prev normal for acceptance.
  // Default 0.95 (≈18° angle tolerance). Schied Eq. 2.
  sigmaNormal:  f32,
  // α_min: minimum EMA weight (Schied Eq. 4). Default 0.05 (Schied paper).
  // Prevents convergence from becoming arbitrarily slow at high history counts.
  alphaMin:     f32,
  // non-zero → reject previous-frame history for this dispatch. Walkaround uses
  // this after scene/material/light mutations that call requestAccumReset().
  forceReset:   u32,
};

// ============================================================
// Bindings
// ============================================================
@group(0) @binding(0)  var reproj_currColor:       texture_2d<f32>;
@group(0) @binding(1)  var reproj_prevColor:        texture_2d<f32>;
@group(0) @binding(2)  var reproj_motionVec:        texture_2d<f32>;
@group(0) @binding(3)  var reproj_currDepth:        texture_2d<f32>;
@group(0) @binding(4)  var reproj_currNormal:       texture_2d<f32>;
@group(0) @binding(5)  var reproj_currObjId:        texture_2d<u32>;
@group(0) @binding(6)  var reproj_prevDepth:        texture_2d<f32>;
@group(0) @binding(7)  var reproj_prevNormal:       texture_2d<f32>;
@group(0) @binding(8)  var reproj_prevObjId:        texture_2d<u32>;
@group(0) @binding(9)  var reproj_historyLengthIn:  texture_2d<u32>;
@group(0) @binding(10) var reproj_momentsIn:        texture_2d<f32>;
@group(0) @binding(11) var reproj_colorOut:         texture_storage_2d<rgba16float, write>;
// historyOut: was r16uint; base-spec WebGPU disallows r16uint as storage
// (needs texture-formats-tier1, which three.js's WebGPURenderer doesn't
// request). r32uint is base-spec storage-capable; counter stays well under
// u16 max so the wider format is just 2x memory, no behavioural change.
@group(0) @binding(12) var reproj_historyOut:       texture_storage_2d<r32uint, write>;
@group(0) @binding(13) var reproj_momentsOut:       texture_storage_2d<rgba32float, write>;
@group(0) @binding(14) var<uniform> reproj_ubo:     SVGFReprojUBO;

// ============================================================
// Helpers
// ============================================================

// luminance(c) is the canonical Rec.709 helper from LUMINANCE_WGSL
// (prepended above at module build time).

// Test a single previous-frame tap for Schied Eq. 2 disocclusion criteria.
// Returns true if the tap is VALID (should be included in bilinear blend).
fn tapValid(
  pCoord:      vec2u,
  dims:        vec2u,
  zCurr:       f32,
  nCurr:       vec3f,
  objIdCurr:   u32,
  sigmaDepth:  f32,
  sigmaNormal: f32,
) -> bool {
  if (any(pCoord >= dims)) { return false; }

  let zPrev = textureLoad(reproj_prevDepth,  pCoord, 0).${depthComponent};
  let nPrev = textureLoad(reproj_prevNormal, pCoord, 0).xyz * 2.0 - 1.0;
  let oPrev = textureLoad(reproj_prevObjId,  pCoord, 0).r;

  // Packed walkaround depth uses the sign bit as a glass-primary marker.
  // Glass radiance is deterministic and bypasses the spatial filter, so it
  // deliberately starts a fresh temporal history instead of relying on signed
  // values in the relative-depth expression below.
  if (zCurr < 0.0 || zPrev < 0.0) { return false; }

  // Depth test — relative deviation (Schied Eq. 2, first clause).
  // Use max(z,zPrev) in denominator so the test is symmetric.
  if (abs(zCurr - zPrev) > sigmaDepth * max(zCurr, zPrev) + 1e-4) { return false; }

  // Normal test — dot-product threshold (Eq. 2, second clause).
  if (dot(nCurr, nPrev) < sigmaNormal) { return false; }

  // Object-id test — prevents blending across independently moving objects.
  if (oPrev != objIdCurr) { return false; }

  return true;
}

// ============================================================
// svgfReprojMain — temporal reprojection + history + EMA
// ============================================================
@compute @workgroup_size(16, 16, 1)
fn svgfReprojMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(reproj_currColor);
  if (any(gid.xy >= dims)) { return; }

  let sigmaDepth  = reproj_ubo.sigmaDepth;
  let sigmaNormal = reproj_ubo.sigmaNormal;
  let alphaMin    = reproj_ubo.alphaMin;

  // Current-frame G-buffer values.
  let currColor = textureLoad(reproj_currColor,  gid.xy, 0).rgb;
  let zCurr     = textureLoad(reproj_currDepth,  gid.xy, 0).${depthComponent};
  // Normal is packed 0..1 → world-space by *2-1.
  let nCurr     = textureLoad(reproj_currNormal, gid.xy, 0).xyz * 2.0 - 1.0;
  let objIdCurr = textureLoad(reproj_currObjId,  gid.xy, 0).r;

  // Motion vector: screen-space pixel delta (float), in [−dims, dims] range.
  let mv        = textureLoad(reproj_motionVec, gid.xy, 0).xy;

  // Previous-frame position (bilinear footprint origin).
  let prevPosF  = vec2f(gid.xy) + mv;
  let prevPos   = vec2i(floor(prevPosF));
  let frac      = fract(prevPosF);

  let forceReset = reproj_ubo.forceReset != 0u;

  // Accumulate color and moments from valid taps.
  var accColor   = vec3f(0.0);
  var accM1      = 0.0;
  var accM2      = 0.0;
  var accHistory = 0.0;
  var accWeight  = 0.0;

  if (!forceReset) {
    // Bilinear gather: four taps.
    let coords = array<vec2i, 4>(
      prevPos,
      prevPos + vec2i(1, 0),
      prevPos + vec2i(0, 1),
      prevPos + vec2i(1, 1),
    );
    let bilerp = array<f32, 4>(
      (1.0 - frac.x) * (1.0 - frac.y),
      frac.x         * (1.0 - frac.y),
      (1.0 - frac.x) * frac.y,
      frac.x         * frac.y,
    );

    for (var i = 0; i < 4; i++) {
      let c = coords[i];
      if (c.x < 0 || c.y < 0) { continue; }
      let cu = vec2u(c);
      if (!tapValid(cu, dims, zCurr, nCurr, objIdCurr, sigmaDepth, sigmaNormal)) {
        continue;
      }
      let w = bilerp[i];
      let prevC = textureLoad(reproj_prevColor, cu, 0).rgb;
      let prevM = textureLoad(reproj_momentsIn, cu, 0).rg;
      let prevH = f32(textureLoad(reproj_historyLengthIn, cu, 0).r);

      accColor   += prevC * w;
      accM1      += prevM.r * w;
      accM2      += prevM.g * w;
      accHistory += prevH * w;
      accWeight  += w;
    }
  }

  // Determine if reprojection succeeded (any valid tap).
  let reprojValid = !forceReset && accWeight > 1e-6;

  var newHistory: u32;
  var alpha: f32;

  if (reprojValid) {
    // Normalise the bilinear blend.
    let invW = 1.0 / accWeight;
    accColor   *= invW;
    accM1      *= invW;
    accM2      *= invW;
    accHistory *= invW;

    // Eq. 3 — history increment.
    newHistory = u32(accHistory) + 1u;
    // Eq. 4 — EMA alpha clamp.
    alpha = max(alphaMin, 1.0 / f32(newHistory));
  } else {
    // Disocclusion: reset history to 1, use only current frame.
    newHistory = 1u;
    alpha = 1.0;
    accColor = vec3f(0.0);
    accM1    = 0.0;
    accM2    = 0.0;
  }

  // Eq. 4 — EMA blend: color_out = α·current + (1−α)·prev
  let blendedColor = alpha * currColor + (1.0 - alpha) * accColor;

  // Moment update: first and second luminance moments.
  let lCurr = luminance(currColor);
  let newM1  = alpha * lCurr        + (1.0 - alpha) * accM1;
  let newM2  = alpha * lCurr * lCurr + (1.0 - alpha) * accM2;

  // Write outputs.
  textureStore(reproj_colorOut,    gid.xy, vec4f(blendedColor, 1.0));
  textureStore(reproj_historyOut,  gid.xy, vec4u(newHistory, 0u, 0u, 0u));
  textureStore(reproj_momentsOut,  gid.xy, vec4f(newM1, newM2, 0.0, 0.0));
}
`;
}

/** Standalone ABI: current/previous depth are dedicated R textures. */
export const SVGF_REPROJECTION_WGSL = buildSvgfReprojectionWgsl();
