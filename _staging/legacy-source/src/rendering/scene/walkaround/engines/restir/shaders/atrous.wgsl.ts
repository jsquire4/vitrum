/**
 * À-trous wavelet denoiser — §7 of the walkaround plan.
 *
 * Edge-aware (edge-stopped) À-trous wavelet filter. 5 iterations with
 * stepWidth ∈ {1, 2, 4, 8, 16}. Guided by G-buffer normal + depth + albedo.
 *
 * References:
 *   Dammertz et al. "Edge-Avoiding À-Trous Wavelet Transform" HPG 2010.
 *   C-none/Web-RTRT atrous denoiser implementation.
 */

export const ATROUS_WGSL = /* wgsl */ `

@group(0) @binding(0) var inputColor:    texture_2d<f32>;
@group(0) @binding(1) var outputColor:   texture_storage_2d<rgba16float, write>;
// Normal+depth G-buffer authored by the shade pass.  Both binding 2 and
// binding 3 reference the SAME texture (kept as two slots for backward
// compatibility with the bind group layout — the host code happens to
// pass the same view to both).  We read .xyz for the encoded normal and
// .w for primary-hit distance (depth) — NOT .r as in the legacy layout.
@group(0) @binding(2) var gbufferNormal: texture_2d<f32>;
@group(0) @binding(3) var gbufferDepth:  texture_2d<f32>;

struct AtrousUBO {
  stepWidth: f32,
  sigmaN:    f32,
  sigmaZ:    f32,
  sigmaC:    f32,
};
@group(0) @binding(4) var<uniform> ubo: AtrousUBO;

// 5x5 B3 spline kernel weights (row-major).
const KERNEL: array<f32, 25> = array<f32, 25>(
   1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0,  1.0/256.0,
   4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0,  4.0/256.0,
   6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0,  6.0/256.0,
   4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0,  4.0/256.0,
   1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0,  1.0/256.0,
);

@compute @workgroup_size(16, 16, 1)
fn atrousMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(inputColor);
  if (any(gid.xy >= dims)) { return; }

  let cCenter = textureLoad(inputColor,    gid.xy, 0).rgb;
  let nCenter = textureLoad(gbufferNormal, gid.xy, 0).xyz * 2.0 - 1.0;
  // Depth lives in .w (alpha channel of the rgba16float G-buffer authored by
  // the shade pass).  Sky pixels write depth=0; non-sky pixels write
  // primary-hit distance, SIGN-FLIPPED for glass primary hits (the shade
  // pass encodes isGlass into the sign).
  let zCenter = textureLoad(gbufferDepth,  gid.xy, 0).w;

  // Glass primary hits skip atrous entirely — Lo_emit is deterministic
  // (albedo × trans × sun × |dot| × textureMod) with no Monte Carlo noise
  // to denoise. Running the wavelet filter on glass pixels only causes
  // adjacent cells with different authored colors to bleed across cell
  // boundaries (the in-cell ghost-cell artifact). zCenter < 0 = glass.
  if (zCenter < 0.0) {
    textureStore(outputColor, gid.xy, vec4f(cCenter, 1.0));
    return;
  }

  var sumColor  = vec3f(0.0);
  var sumWeight = 0.0;

  let sw = i32(ubo.stepWidth);

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let off = vec2i(dx, dy) * sw;
      let p   = vec2i(gid.xy) + off;
      if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) { continue; }
      let pu  = vec2u(p);

      let cP = textureLoad(inputColor,    pu, 0).rgb;
      let nP = textureLoad(gbufferNormal, pu, 0).xyz * 2.0 - 1.0;
      let zP = textureLoad(gbufferDepth,  pu, 0).w;

      let kIdx = u32((dy + 2) * 5 + (dx + 2));
      let h    = KERNEL[kIdx];

      // Edge-stopping weights (Dammertz et al. 2010).
      let dn = max(0.0, dot(nCenter, nP));
      let wn = pow(max(0.0, dn), ubo.sigmaN);
      let wz = exp(-abs(zP - zCenter) / (ubo.sigmaZ + 1e-6));
      // Chromaticity-based color edge-stop. Compare normalized colors
      // (color direction), NOT raw HDR magnitudes. Plain Euclidean
      // distance on HDR linear values fires hard on bright warm caustics
      // (delta ~5) but stays under threshold for dim cool caustics
      // (delta ~0.05) — even at σc=0.05 the dim caustics' edges blur
      // into the surrounding floor while bright ones stay sharp. By
      // normalizing both colors to unit luminance first, we measure
      // hue/saturation difference instead of brightness difference,
      // making σc behave consistently across all cell luminances.
      let lumW = vec3f(0.2126, 0.7152, 0.0722);
      let lumP = max(1e-3, dot(cP, lumW));
      let lumC = max(1e-3, dot(cCenter, lumW));
      let dc = length(cP / lumP - cCenter / lumC);
      let wc = exp(-dc * dc / (ubo.sigmaC * ubo.sigmaC + 1e-6));

      let w  = h * wn * wz * wc;
      sumColor  += cP * w;
      sumWeight += w;
    }
  }

  let result = select(cCenter, sumColor / sumWeight, sumWeight > 1e-6);
  textureStore(outputColor, gid.xy, vec4f(result, 1.0));
}
`;
