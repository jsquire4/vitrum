/**
 * Sprint 18 — per-channel SVGF combine pass.
 *
 * Reads the denoised direct-channel output (from the SVGF/atrous chain)
 * and the raw indirect-channel output (from shade), applies a broader
 * 5×5 bilateral blur to the indirect signal using the same depth +
 * normal edge stops as SVGF (just with looser sigmas — the indirect
 * signal is already temporally smoothed by ReSTIR-GI temporal+spatial
 * reuse, so it can tolerate more spatial blurring), then sums the two
 * into the combined output texture that temporalAccum consumes.
 *
 * The split lets us use **tight sigmas on direct** to preserve hard
 * shadow boundaries, while **broader sigmas on indirect** smooth out
 * residual GI noise without leaking light across geometry edges.
 *
 * Bindings:
 *   @group(0) @binding(0) denoisedDirectIn (sampled, unfilterable)
 *   @group(0) @binding(1) hdrIndirectIn    (sampled, unfilterable)
 *   @group(0) @binding(2) gNormalDepth     (sampled, unfilterable)
 *   @group(0) @binding(3) combinedOut      (rgba16float, write-only storage)
 */

export const INDIRECT_COMBINE_WGSL = /* wgsl */ `
@group(0) @binding(0) var ic_denoisedDirect: texture_2d<f32>;
@group(0) @binding(1) var ic_hdrIndirect:    texture_2d<f32>;
@group(0) @binding(2) var ic_gNormalDepth:   texture_2d<f32>;
@group(0) @binding(3) var ic_combinedOut:    texture_storage_2d<rgba16float, write>;

const IC_RADIUS:        i32 = 2;     // 5×5 kernel
const IC_SIGMA_DEPTH:   f32 = 0.05;  // broader than SVGF direct (0.01)
const IC_SIGMA_NORMAL:  f32 = 32.0;  // sharper falloff exponent → narrower normal stop

fn ic_decodeNormal(rgba: vec4f) -> vec3f {
  return normalize(rgba.xyz * 2.0 - 1.0);
}

fn ic_smoothIndirectAt(px: vec2i, dims: vec2i) -> vec3f {
  let centerND = textureLoad(ic_gNormalDepth, px, 0);
  let centerDepth = centerND.w;
  let centerNormal = ic_decodeNormal(centerND);
  let centerIndirect = textureLoad(ic_hdrIndirect, px, 0).rgb;
  if (centerDepth < 1e-4) { return centerIndirect; }

  var sum = vec3f(0.0);
  var totalW = 0.0;
  for (var dy: i32 = -IC_RADIUS; dy <= IC_RADIUS; dy = dy + 1) {
    for (var dx: i32 = -IC_RADIUS; dx <= IC_RADIUS; dx = dx + 1) {
      let q = vec2i(px.x + dx, px.y + dy);
      if (q.x < 0 || q.y < 0 || q.x >= dims.x || q.y >= dims.y) { continue; }
      let qND = textureLoad(ic_gNormalDepth, q, 0);
      let qDepth = qND.w;
      if (qDepth < 1e-4) { continue; }
      let qNormal = ic_decodeNormal(qND);

      // Depth edge-stop (Gaussian).
      let dd = abs(qDepth - centerDepth) / max(1e-3, centerDepth);
      let wD = exp(-(dd * dd) / (2.0 * IC_SIGMA_DEPTH * IC_SIGMA_DEPTH));

      // Normal edge-stop (cos-power).
      let cn = max(0.0, dot(centerNormal, qNormal));
      let wN = pow(cn, IC_SIGMA_NORMAL);

      // Gaussian spatial weight (sigma = IC_RADIUS so kernel covers ~2σ).
      let r2 = f32(dx * dx + dy * dy);
      let wS = exp(-r2 / f32(2 * IC_RADIUS * IC_RADIUS));

      let w = wD * wN * wS;
      sum = sum + textureLoad(ic_hdrIndirect, q, 0).rgb * w;
      totalW = totalW + w;
    }
  }
  if (totalW < 1e-4) { return centerIndirect; }
  return sum / totalW;
}

@compute @workgroup_size(16, 16, 1)
fn indirectCombineMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(ic_combinedOut);
  if (any(gid.xy >= dims)) { return; }
  let px = vec2i(gid.xy);

  let direct = textureLoad(ic_denoisedDirect, px, 0).rgb;
  let indirectSmooth = ic_smoothIndirectAt(px, vec2i(dims));
  let combined = direct + indirectSmooth;
  textureStore(ic_combinedOut, gid.xy, vec4f(combined, 1.0));
}
`;
