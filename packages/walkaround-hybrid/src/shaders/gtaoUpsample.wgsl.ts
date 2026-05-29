/**
 * GTAO bilateral upsample — low-res per-channel AO → full-res per-channel AO.
 *
 * Reads the low-res per-channel multi-bounce AO map (rgba16float) produced by
 * `gtao.wgsl.ts` and a full-res gNormalDepth map. For each full-res pixel,
 * samples the four nearest AO-grid taps and weights them by depth+normal
 * similarity to the full-res pixel's surface. Standard "joint bilateral
 * upsample" pattern (Kopf et al. 2007) — preserves AO discontinuities at
 * geometric edges that the simple trilinear upsample would smear. The AO grid
 * is W/ds × H/ds (ds = gtaoDownscale: 2 for `gtaoMode:'on'`, 4 for
 * `gtaoMode:'quarter'`); the upsample ratio follows ds via the UBO so the same
 * machinery handles both half- and quarter-res inputs.
 *
 * Tier-G fix (Jiménez 2016 §5.2 per-channel multi-bounce): previously the
 * upsample collapsed the per-channel AO vec3 to a single luminance scalar
 * with weights (0.2126, 0.7152, 0.0722) and stored only `.r`. That defeated
 * the Jiménez §5.2 per-channel formulation — every consumer pixel got the
 * same scalar AO across R/G/B, equivalent to Bavoil-style scalar AO with a
 * one-time luminance-weighted brightening. The upsample now keeps three
 * independent AO channels through the bilateral filter so shade can darken
 * each colour channel by its own multi-bounce factor.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const GTAO_UPSAMPLE_WGSL = /* wgsl */ `

// Duplicate of gtao.wgsl's GTAOUniforms struct (both shaders bind the same
// uboBuffer; the duplicate WGSL declaration is required because each shader
// module is compiled independently — concatenating would conflict with
// gtao's @binding(0/1/2) declarations on the same group).
struct GTAOUniforms {
  tanFovHalf: f32,
  radiusPx:   f32,
  intensity:  f32,
  depthThresh: f32,
  bilateralDepthSigma: f32,
  // AO compute downscale factor (integer, stored as f32). 2 = half-res input,
  // 4 = quarter-res input. Maps full-res pixel to low-res tap (gid/ds) and
  // low-res tap to full-res sample centre (tap*ds + ds/2). Was the _pad0 slot.
  gtaoDownscale: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var up_aoHalf:      texture_2d<f32>;
@group(0) @binding(1) var up_normalDepth: texture_2d<f32>;
// aoFullOut: rgba16float storage texture. .rgb carries the per-channel
// Jiménez 2016 §5.2 multi-bounce AO (one factor per RGB channel); .a unused.
// Was previously stored as a scalar in .r only with .gba zero — that
// dropped the per-channel multi-bounce contribution and made shade darken
// every channel uniformly, equivalent to Bavoil-style scalar AO. shade.wgsl
// now reads .rgb and multiplies the vec3 AO into the radiance terms.
// rgba16float is base-spec storage-capable; r16float would require the
// optional texture-formats-tier1 feature that three.js's WebGPURenderer
// does not request.
@group(0) @binding(2) var up_aoFullOut:   texture_storage_2d<rgba16float, write>;
// Audit B3: bilateral depth sigma now read from the GTAO UBO (shared with
// gtao.wgsl's GTAOUniforms struct) so the host can scale it with the scene.
@group(0) @binding(3) var<uniform> up_gtao: GTAOUniforms;

fn similarityWeight(
  centerDepth: f32,
  centerNormal: vec3f,
  sampleDepth: f32,
  sampleNormal: vec3f,
) -> f32 {
  // Reject samples too far in depth or too different in normal — they
  // belong to a different surface and would bleed AO across geometric edges.
  // depthW = exp(-Δdepth / (2 σ²)), with σ = up_gtao.bilateralDepthSigma.
  let depthDelta = abs(centerDepth - sampleDepth);
  let sigma = max(1e-6, up_gtao.bilateralDepthSigma);
  let depthW = exp(-depthDelta / (2.0 * sigma * sigma));
  let nDot = max(0.0, dot(centerNormal, sampleNormal));
  let normW = pow(nDot, 16.0);
  return depthW * normW;
}

@compute @workgroup_size(8, 8, 1)
fn gtaoUpsampleMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = vec2u(textureDimensions(up_normalDepth));
  if (any(gid.xy >= fullDims)) { return; }

  // AO downscale factor (2 = half-res input, 4 = quarter-res). Clamp ≥ 1.
  let ds = max(1u, u32(up_gtao.gtaoDownscale));
  let halfDims = fullDims / ds;

  // Read center pixel's normal + depth.
  let center = textureLoad(up_normalDepth, gid.xy, 0);
  let centerNormal = center.xyz * 2.0 - 1.0;
  let centerDepth = abs(center.w);

  // Sky-miss pixels are fully lit.
  if (centerDepth < 1e-4) {
    textureStore(up_aoFullOut, gid.xy, vec4f(1.0));
    return;
  }

  // Low-res tap position: gid.xy / ds gives the integer AO-grid cell index.
  let halfPx = gid.xy / ds;

  // Sample 2×2 neighborhood of AO-grid taps. Use clamped coords for the edges.
  //
  // Tier-G fix: keep the per-channel multi-bounce vec3 AO through the
  // bilateral filter rather than collapsing it to a luminance scalar.
  // Jiménez 2016 §5.2 Eq. 16 produces one AO factor per RGB channel based
  // on the surface albedo (a red wall darkens only the red channel; the
  // green/blue inter-reflection terms are near 1.0). Reducing to luminance
  // here erased that per-channel structure and made every output pixel
  // darken uniformly — equivalent to Bavoil-style scalar AO with a
  // one-time luminance-weighted brightening.
  var sumAO: vec3f = vec3f(0.0);
  var sumW:  f32   = 0.0;

  for (var dy: u32 = 0u; dy < 2u; dy = dy + 1u) {
    for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
      let sampleHalf = vec2u(
        min(halfPx.x + dx, halfDims.x - 1u),
        min(halfPx.y + dy, halfDims.y - 1u),
      );
      // Read per-channel multi-bounce AO as-is.
      let aoMb = textureLoad(up_aoHalf, sampleHalf, 0).rgb;
      // Corresponding full-res sample point (center of the ds×ds AO cell).
      let sampleFull = sampleHalf * ds + ds / 2u;
      let nd = textureLoad(
        up_normalDepth,
        vec2u(min(sampleFull.x, fullDims.x - 1u),
              min(sampleFull.y, fullDims.y - 1u)),
        0,
      );
      let sNormal = nd.xyz * 2.0 - 1.0;
      let sDepth = abs(nd.w);
      let w = similarityWeight(centerDepth, centerNormal, sDepth, sNormal);
      sumAO = sumAO + aoMb * w;
      sumW = sumW + w;
    }
  }

  // If no half-res sample matches our surface (heavy edge), fall back to
  // the unweighted per-channel average — better to have *some* AO than zero.
  var ao: vec3f = vec3f(1.0);
  if (sumW > 1e-4) {
    ao = sumAO / sumW;
  } else {
    // Cheap unweighted per-channel average as backup.
    ao = (
      textureLoad(up_aoHalf, halfPx, 0).rgb +
      textureLoad(up_aoHalf, vec2u(min(halfPx.x + 1u, halfDims.x - 1u), halfPx.y), 0).rgb +
      textureLoad(up_aoHalf, vec2u(halfPx.x, min(halfPx.y + 1u, halfDims.y - 1u)), 0).rgb +
      textureLoad(up_aoHalf, vec2u(min(halfPx.x + 1u, halfDims.x - 1u),
                                   min(halfPx.y + 1u, halfDims.y - 1u)), 0).rgb
    ) * 0.25;
  }

  textureStore(up_aoFullOut, gid.xy, vec4f(clamp(ao, vec3f(0.0), vec3f(1.0)), 1.0));
}
`;

/** W1-R6 — declarative include-graph entry. Self-contained. */
export const GTAO_UPSAMPLE_MODULE: WgslModule = {
  name: 'gtaoUpsample',
  source: GTAO_UPSAMPLE_WGSL,
  requires: [],
};
