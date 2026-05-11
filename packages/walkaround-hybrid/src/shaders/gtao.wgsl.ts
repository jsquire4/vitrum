/**
 * GTAO (Ground-Truth Ambient Occlusion) — half-resolution horizon-based AO.
 *
 * Reference: Jiménez et al. 2016, "Practical Realtime Strategies for Accurate
 * Indirect Occlusion," SIGGRAPH 2016.
 *
 * Inputs:
 *   - `gNormalDepth` (rgba16float): xyz = normal × 0.5 + 0.5, w = signed depth
 *     (abs = world-space hit distance; negative = glass; 0 = sky-miss).
 * Outputs:
 *   - `aoHalf` (r16float): per-pixel occlusion factor in [0, 1]. 1 = fully lit,
 *     0 = fully occluded.
 *
 * Half-res: dispatches over W/2 × H/2 invocations; each samples the *full-res*
 * gNormalDepth at the 2×2 quad center. Bilateral upsample (see
 * `gtaoUpsample.wgsl.ts`) reconstructs full-res AO from this half-res map.
 *
 * Algorithm summary:
 *   for each of NUM_DIRECTIONS angular slices on the screen-space tangent plane:
 *     for each of NUM_STEPS log-spaced step distances along ±direction:
 *       sample depth at p + dir × step (and p − dir × step)
 *       track the maximum horizon angle on each side (cos θ_horizon)
 *     integrate the slice's visible-arc fraction
 *   ao = average over slices, gamma-corrected.
 *
 * Per-pixel jitter on the base direction breaks aliased horizon-ray patterns;
 * the bilateral upsample averages neighbors before the shade pass consumes it,
 * so per-frame jitter doesn't introduce temporal flicker.
 */

export const GTAO_WGSL = /* wgsl */ `

struct GTAOUniforms {
  // tan(fov/2) along screen height; used to convert pixel offsets to view-space
  // angles for the horizon integration. Packed from the host's camera.
  tanFovHalf: f32,
  // Sampling radius in screen pixels (full-res space). 32 px is a typical
  // contact-AO radius; larger broadens the AO to medium-range occlusion.
  radiusPx:   f32,
  // AO intensity exponent. ao = pow(ao_raw, intensity). 1.0 = linear,
  // 2.0 = stronger contact darkening.
  intensity:  f32,
  // Maximum depth difference (world units) to consider a sample for the
  // horizon test. Larger gap = treat as background → no occlusion. Prevents
  // halos around foreground silhouettes.
  depthThresh: f32,
};

@group(0) @binding(0) var gtao_normalDepth: texture_2d<f32>;
@group(0) @binding(1) var gtao_aoOut:       texture_storage_2d<r16float, write>;
@group(0) @binding(2) var<uniform> gtao_ubo: GTAOUniforms;

const PI: f32 = 3.14159265359;
const NUM_DIRECTIONS: u32 = 4u;
const NUM_STEPS:      u32 = 6u;

fn hashPx(p: vec2u) -> f32 {
  let h = sin(f32(p.x) * 12.9898 + f32(p.y) * 78.233) * 43758.5453;
  return fract(h);
}

@compute @workgroup_size(8, 8, 1)
fn gtaoMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = vec2u(textureDimensions(gtao_normalDepth));
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  // Half-res sample point: centre of the 2x2 quad in full-res coords.
  let fullPx = gid.xy * 2u + 1u;
  let center = textureLoad(gtao_normalDepth, vec2i(fullPx), 0);
  let centerDepth = abs(center.w);

  // Sky-miss pixels (depth = 0) are not occluded.
  if (centerDepth < 1e-4) {
    textureStore(gtao_aoOut, gid.xy, vec4f(1.0));
    return;
  }

  let jitter = hashPx(gid.xy);

  var aoSum: f32 = 0.0;

  for (var d: u32 = 0u; d < NUM_DIRECTIONS; d = d + 1u) {
    let baseAngle = (f32(d) / f32(NUM_DIRECTIONS)) * PI;
    let theta = baseAngle + jitter * (PI / f32(NUM_DIRECTIONS));
    let dir = vec2f(cos(theta), sin(theta));

    // Track the highest cos(horizon) found on each side of this slice.
    // cos(horizon) > 0 means a sample is *closer* to camera than the center
    // (occluder above the surface plane). 1 = surface coplanar.
    var horizonPos: f32 = -1.0;
    var horizonNeg: f32 = -1.0;

    for (var s: u32 = 1u; s <= NUM_STEPS; s = s + 1u) {
      // Log-spaced step radius: closer samples get more weight.
      let t = f32(s) / f32(NUM_STEPS);
      let stepRadius = gtao_ubo.radiusPx * pow(2.0, t - 1.0);
      let offset = dir * stepRadius;

      let posPx = vec2i(fullPx) + vec2i(offset);
      let negPx = vec2i(fullPx) - vec2i(offset);

      if (all(posPx >= vec2i(0)) && all(posPx < vec2i(fullDims))) {
        let dP = abs(textureLoad(gtao_normalDepth, posPx, 0).w);
        if (dP > 1e-4) {
          let dz = centerDepth - dP;
          if (abs(dz) < gtao_ubo.depthThresh) {
            // cos(horizon) = dz / sample_distance; sample_distance is the
            // view-space distance from center to the sample, approximated by
            // step_pixels × tan(fov/2) / fullDims.y × centerDepth.
            let viewDist = stepRadius * gtao_ubo.tanFovHalf
                         * (2.0 / f32(fullDims.y)) * centerDepth;
            let cosH = dz / max(viewDist, 1e-4);
            horizonPos = max(horizonPos, cosH);
          }
        }
      }

      if (all(negPx >= vec2i(0)) && all(negPx < vec2i(fullDims))) {
        let dN = abs(textureLoad(gtao_normalDepth, negPx, 0).w);
        if (dN > 1e-4) {
          let dz = centerDepth - dN;
          if (abs(dz) < gtao_ubo.depthThresh) {
            let viewDist = stepRadius * gtao_ubo.tanFovHalf
                         * (2.0 / f32(fullDims.y)) * centerDepth;
            let cosH = dz / max(viewDist, 1e-4);
            horizonNeg = max(horizonNeg, cosH);
          }
        }
      }
    }

    // Slice visibility: visible arc fraction of the upper hemisphere on this
    // tangent slice. acos(horizon) in [0, π/2] when cosH = 0 → horizon at
    // ground plane (no occlusion); cosH → 1 → horizon overhead (full block).
    let h1 = acos(clamp(horizonPos, -1.0, 1.0));
    let h2 = acos(clamp(horizonNeg, -1.0, 1.0));
    aoSum = aoSum + (h1 + h2) / PI;
  }

  let aoRaw = clamp(aoSum / f32(NUM_DIRECTIONS), 0.0, 1.0);
  let ao = pow(aoRaw, gtao_ubo.intensity);
  textureStore(gtao_aoOut, gid.xy, vec4f(ao, 0.0, 0.0, 1.0));
}
`;
