/**
 * Sprint 18 — per-channel denoising combine pass.
 *
 * Both channels are denoised upstream:
 *   - direct: hdrColorTexture → welford → atrous-variance-variance → atrous-variance-atrous ×3
 *   - indirect: hdrIndirectTexture → atrous ×4 (steps 1, 2, 4, 8) with
 *               broader edge-stop sigmas tuned for ReSTIR-GI's pre-smoothed
 *               signal (see ATROUS_INDIRECT_SIGMAS in bindGroupBuilders).
 *
 * Item 24 — albedo demodulation re-modulation (Schied 2017 §4.1):
 * shade.wgsl writes the indirect channel WITHOUT the albedo factor (so the
 * à-trous chain filters pure lighting L). This pass re-multiplies the denoised
 * indirect by albedo before summing: `combined = direct + filtered_lighting × albedo`.
 * This preserves crisp material boundaries that would otherwise bleed into
 * the lighting during à-trous spatial filtering.
 *
 * Bindings:
 *   @group(0) @binding(0) denoisedDirect  (sampled, unfilterable)
 *   @group(0) @binding(1) denoisedIndirect (sampled, unfilterable) — demodulated lighting
 *   @group(0) @binding(2) combinedOut     (rgba16float, write-only storage)
 *   @group(0) @binding(3) albedo          (sampled, unfilterable) — Item 24 re-modulation
 *
 * W5-I2 cleanup (2026-05-18): the previous `gNormalDepth` binding at slot 2
 * was declared "for BGL compat, unused" and never read by the shader body —
 * dropped along with its host-side BGL entry + builder argument. Bindings
 * 3/4 renumbered to 2/3.
 *
 * Reference: Schied et al. 2017, "Spatiotemporal Variance-Guided Filtering",
 *   HPG §4.1: "We demodulate the lighting from the albedo of the first hit
 *   surface before filtering: L = c/ρ … After filtering, the albedo is
 *   remodulated: c_filtered = L_filtered × ρ."
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const INDIRECT_COMBINE_WGSL = /* wgsl */ `
@group(0) @binding(0) var ic_denoisedDirect:   texture_2d<f32>;
@group(0) @binding(1) var ic_denoisedIndirect: texture_2d<f32>;
@group(0) @binding(2) var ic_combinedOut:      texture_storage_2d<rgba16float, write>;
// Item 24 — albedo demodulation (Schied 2017 §4.1). Written by shade as the
// visible-point diffuse colour; used here to re-modulate the denoised lighting.
@group(0) @binding(3) var ic_albedo:           texture_2d<f32>;

@compute @workgroup_size(16, 16, 1)
fn indirectCombineMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(ic_combinedOut);
  if (any(gid.xy >= dims)) { return; }

  let direct          = textureLoad(ic_denoisedDirect,   gid.xy, 0).rgb;
  // ic_denoisedIndirect carries demodulated lighting (L without albedo).
  let filteredLighting = textureLoad(ic_denoisedIndirect, gid.xy, 0).rgb;
  // Re-modulate: filtered_lighting × albedo = physically correct denoised indirect.
  // Clamp albedo to [1e-3, 1] to avoid divide-by-near-zero artefacts from
  // black surfaces (black albedo → near-zero indirect anyway, so no visible change).
  let albedo          = max(vec3f(1e-3), textureLoad(ic_albedo, gid.xy, 0).rgb);
  let indirect        = filteredLighting * albedo;
  textureStore(ic_combinedOut, gid.xy, vec4f(direct + indirect, 1.0));
}
`;

/** W1-R6 — declarative include-graph entry. Self-contained. */
export const INDIRECT_COMBINE_MODULE: WgslModule = {
  name: 'indirectCombine',
  source: INDIRECT_COMBINE_WGSL,
  requires: [],
};
