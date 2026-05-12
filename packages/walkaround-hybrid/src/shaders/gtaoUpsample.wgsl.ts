/**
 * GTAO bilateral upsample — half-res AO → full-res AO.
 *
 * Reads the half-res AO map (r16float) produced by `gtao.wgsl.ts` and a
 * full-res gNormalDepth map. For each full-res pixel, samples the four nearest
 * half-res taps and weights them by depth+normal similarity to the full-res
 * pixel's surface. Standard "joint bilateral upsample" pattern (Kopf et al.
 * 2007) — preserves AO discontinuities at geometric edges that the simple
 * trilinear upsample would smear.
 */

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
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var up_aoHalf:      texture_2d<f32>;
@group(0) @binding(1) var up_normalDepth: texture_2d<f32>;
@group(0) @binding(2) var up_aoFullOut:   texture_storage_2d<r16float, write>;
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

  let halfDims = fullDims / 2u;

  // Read center pixel's normal + depth.
  let center = textureLoad(up_normalDepth, gid.xy, 0);
  let centerNormal = center.xyz * 2.0 - 1.0;
  let centerDepth = abs(center.w);

  // Sky-miss pixels are fully lit.
  if (centerDepth < 1e-4) {
    textureStore(up_aoFullOut, gid.xy, vec4f(1.0));
    return;
  }

  // Half-res tap position: gid.xy / 2 gives the integer cell index.
  let halfPx = gid.xy / 2u;

  // Sample 2×2 neighborhood in half-res. Use clamped coords for the edges.
  var sumAO: f32 = 0.0;
  var sumW:  f32 = 0.0;

  // E1: aoHalf now carries per-channel multi-bounce AO (rgba16float).
  // Reduce to a scalar luminance weight before bilateral filtering so
  // the output aoFull remains r16float (shade reads a single channel).
  let lum = vec3f(0.2126, 0.7152, 0.0722);

  for (var dy: u32 = 0u; dy < 2u; dy = dy + 1u) {
    for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
      let sampleHalf = vec2u(
        min(halfPx.x + dx, halfDims.x - 1u),
        min(halfPx.y + dy, halfDims.y - 1u),
      );
      // Read per-channel multi-bounce AO and collapse to scalar luminance.
      let aoMb = textureLoad(up_aoHalf, sampleHalf, 0).rgb;
      let ao = dot(aoMb, lum);
      // Corresponding full-res sample point (center of the half-res cell).
      let sampleFull = sampleHalf * 2u + 1u;
      let nd = textureLoad(
        up_normalDepth,
        vec2u(min(sampleFull.x, fullDims.x - 1u),
              min(sampleFull.y, fullDims.y - 1u)),
        0,
      );
      let sNormal = nd.xyz * 2.0 - 1.0;
      let sDepth = abs(nd.w);
      let w = similarityWeight(centerDepth, centerNormal, sDepth, sNormal);
      sumAO = sumAO + ao * w;
      sumW = sumW + w;
    }
  }

  // If no half-res sample matches our surface (heavy edge), fall back to
  // the unweighted average — better to have *some* AO than zero.
  // E1: reduce per-channel multi-bounce vec3 to luminance scalar in fallback too.
  var ao: f32 = 1.0;
  if (sumW > 1e-4) {
    ao = sumAO / sumW;
  } else {
    // Cheap unweighted average as backup; reduce each tap to luminance first.
    ao = (
      dot(textureLoad(up_aoHalf, halfPx, 0).rgb, lum) +
      dot(textureLoad(up_aoHalf, vec2u(min(halfPx.x + 1u, halfDims.x - 1u), halfPx.y), 0).rgb, lum) +
      dot(textureLoad(up_aoHalf, vec2u(halfPx.x, min(halfPx.y + 1u, halfDims.y - 1u)), 0).rgb, lum) +
      dot(textureLoad(up_aoHalf, vec2u(min(halfPx.x + 1u, halfDims.x - 1u),
                                       min(halfPx.y + 1u, halfDims.y - 1u)), 0).rgb, lum)
    ) * 0.25;
  }

  textureStore(up_aoFullOut, gid.xy, vec4f(clamp(ao, 0.0, 1.0)));
}
`;
