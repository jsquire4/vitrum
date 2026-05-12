/**
 * Sprint 18 — per-channel denoising combine pass.
 *
 * Both channels are denoised upstream:
 *   - direct: hdrColorTexture → welford → atrous-variance-variance → atrous-variance-atrous ×3
 *   - indirect: hdrIndirectTexture → atrous ×4 (steps 1, 2, 4, 8) with
 *               broader edge-stop sigmas tuned for ReSTIR-GI's pre-smoothed
 *               signal (see ATROUS_INDIRECT_SIGMAS in bindGroupBuilders).
 *
 * This pass simply sums the two denoised channels into combinedDenoisedTexture
 * which temporalAccum reads. No per-pixel filtering happens here — the heavy
 * lifting is in the upstream atrous chains.
 *
 * Bindings:
 *   @group(0) @binding(0) denoisedDirect (sampled, unfilterable)
 *   @group(0) @binding(1) denoisedIndirect (sampled, unfilterable)
 *   @group(0) @binding(2) gNormalDepth (sampled, unfilterable) — kept for
 *     BGL compatibility but unused by the simplified combine. The layout
 *     stays the same so the bind-group builder doesn't need a parallel path.
 *   @group(0) @binding(3) combinedOut (rgba16float, write-only storage)
 */

export const INDIRECT_COMBINE_WGSL = /* wgsl */ `
@group(0) @binding(0) var ic_denoisedDirect:   texture_2d<f32>;
@group(0) @binding(1) var ic_denoisedIndirect: texture_2d<f32>;
@group(0) @binding(2) var ic_gNormalDepth:     texture_2d<f32>;
@group(0) @binding(3) var ic_combinedOut:      texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn indirectCombineMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(ic_combinedOut);
  if (any(gid.xy >= dims)) { return; }

  let direct   = textureLoad(ic_denoisedDirect,   gid.xy, 0).rgb;
  let indirect = textureLoad(ic_denoisedIndirect, gid.xy, 0).rgb;
  textureStore(ic_combinedOut, gid.xy, vec4f(direct + indirect, 1.0));
}
`;
