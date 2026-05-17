/**
 * GTAO (Ground-Truth Ambient Occlusion) — half-resolution horizon-based AO.
 *
 * Reference: Jiménez et al. 2016, "Practical Realtime Strategies for Accurate
 * Indirect Occlusion," SIGGRAPH 2016. §4.2 Eq. 11 (slice integral).
 *
 * XeGTAO reference implementation verified against:
 *   Intel XeGTAO — https://github.com/GameTechDev/XeGTAO
 *   (XeGTAO.hlsli, `XeGTAO_MainPass` inner loop)
 *
 * Inputs:
 *   - `gNormalDepth` (rgba16float): xyz = normal × 0.5 + 0.5, w = signed depth
 *     (abs = world-space hit distance; negative = glass; 0 = sky-miss).
 * Outputs:
 *   - `aoHalf` (rgba16float): per-channel multi-bounce ambient occlusion factor.
 *     Each channel encodes the Jiménez 2016 §5.2 Eq. 16 albedo-aware brightened
 *     visibility for that colour channel.  The bilateral upsample reduces this
 *     to a single luminance-weighted scalar before shade reads it.
 *     1 = fully lit, 0 = fully occluded.  The multi-bounce term brightens
 *     intermediate albedo surfaces; at ρ = 1 each channel equals the scalar AO.
 *
 * Half-res: dispatches over W/2 × H/2 invocations; each samples the *full-res*
 * gNormalDepth at the 2×2 quad center. Bilateral upsample (see
 * `gtaoUpsample.wgsl.ts`) reconstructs full-res AO from this half-res map.
 *
 * Algorithm summary:
 *   Decode world-space surface normal from G-buffer.
 *   for each of NUM_DIRECTIONS angular slices on the screen-space tangent plane:
 *     project surface normal onto the slice plane → get projected-normal angle γ (n)
 *     for each of NUM_STEPS log-spaced step distances along ±direction:
 *       sample depth at p + dir × step (and p − dir × step)
 *       track the maximum horizon angle on each side (cos θ_horizon)
 *     clamp θ_h to ± π/2 of γ (upper hemisphere w.r.t. normal)
 *     integrate the Jiménez 2016 §4.2 Eq. 11 slice AO:
 *       iarc(h, n) = (cos(n) + 2·h·sin(n) − cos(2·h − n)) / 4
 *       localVisibility = |projNormal| · (iarc(h0, n) + iarc(h1, n))
 *   ao = average over slices, gamma-corrected.
 *
 * Previously this used the simplified Bavoil-style HBAO formula `(h1+h2)/π`
 * (angular fraction of hemisphere), which ignores the cosine weight relative to
 * the surface normal (the γ/n terms). The Jiménez integral is the correct
 * Lambertian-weighted visible-arc integral on the slice plane.
 *
 * Multi-bounce (Jiménez 2016 §5.2 / Eq. 16): now implemented. The per-channel
 * albedo G-buffer (hdrAlbedoOut from shade pass M9.C) is bound at @binding(3).
 * The scalar visibility `vis` is lifted to a per-channel `vec3f` by the Eq. 16
 * polynomial before being stored; `gtaoUpsample.wgsl.ts` reads it as vec3.
 *
 * Per-pixel jitter on the base direction breaks aliased horizon-ray patterns;
 * the bilateral upsample averages neighbors before the shade pass consumes it,
 * so per-frame jitter doesn't introduce temporal flicker.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

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
  // Audit B3: bilateral upsample depth-weight sigma (world units). Used by
  // gtaoUpsample.wgsl for the joint-bilateral half→full upsample. Previously
  // hardcoded to 4.0 in the shader (σ ≈ 0.25 m), which was Cornell-scale-only.
  // Hosts should set ~(sceneDiagonal * 0.01) so the half-life of the depth
  // weight is ~1% of the scene's longest axis.
  bilateralDepthSigma: f32,
  // Pad to 32 bytes (8-element struct) for WebGPU 16-byte UBO alignment.
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var gtao_normalDepth: texture_2d<f32>;
@group(0) @binding(1) var gtao_aoOut:       texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> gtao_ubo: GTAOUniforms;
// E1 — Jiménez 2016 §5.2 multi-bounce: per-channel diffuse albedo from the
// shade pass (hdrAlbedoOut, Item 24). Sampled at the same pixel to compute
// the albedo-weighted brightening polynomial for each colour channel.
@group(0) @binding(3) var gtao_albedo:      texture_2d<f32>;

const PI:      f32 = 3.14159265359;
const PI_HALF: f32 = 1.57079632679;
const NUM_DIRECTIONS: u32 = 4u;
const NUM_STEPS:      u32 = 6u;

fn hashPx(p: vec2u) -> f32 {
  let h = sin(f32(p.x) * 12.9898 + f32(p.y) * 78.233) * 43758.5453;
  return fract(h);
}

// Jiménez 2016 §4.2 Eq. 11 — closed-form slice integral for one horizon side.
//
// Derivation (XeGTAO-verified form):
//   The visible solid-angle arc on the slice plane, Lambertian-weighted by the
//   surface normal, integrates to:
//     iarc(h, n) = (cos(n) + 2·h·sin(n) − cos(2·h − n)) / 4
//   where:
//     h  = horizon angle for this side (in [-π, π], measured from view axis)
//     n  = projected-normal angle γ (angle of surface normal from view axis in
//          the slice plane)
//
// This is equivalent to Eq. 11 in the paper after expanding:
//   cos(γ)·(2θ_h − sin(2θ_h − 2γ)) + sin(γ)·sin²(θ_h − γ) all over 4,
// rewritten using the identity sin²(x) = (1 − cos(2x))/2 and combining.
//
// Reference: XeGTAO.hlsli (iarc0/iarc1 form) — Intel GameTechDev/XeGTAO, MIT licence.
fn gtaoSliceIntegral(h: f32, n: f32, cosN: f32) -> f32 {
  return (cosN + 2.0 * h * sin(n) - cos(2.0 * h - n)) * 0.25;
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

  // Decode world-space surface normal from G-buffer (stored as n*0.5+0.5).
  let surfNormal = normalize(center.xyz * 2.0 - 1.0);

  let jitter = hashPx(gid.xy);

  var aoSum: f32 = 0.0;

  for (var d: u32 = 0u; d < NUM_DIRECTIONS; d = d + 1u) {
    let baseAngle = (f32(d) / f32(NUM_DIRECTIONS)) * PI;
    let theta = baseAngle + jitter * (PI / f32(NUM_DIRECTIONS));
    let dir = vec2f(cos(theta), sin(theta));

    // ── Projected-normal angle γ (n) for this slice ──────────────────────
    //
    // The slice plane is defined by two axes:
    //   axisVec  = vec3(dir.x, dir.y, 0.0) — the 2D slice direction lifted
    //              into 3D; this is the "in-plane perpendicular" (XeGTAO's
    //              axisVec). We treat screen X/Y as lateral and depth as −Z.
    //   viewAxis = vec3(0.0, 0.0, -1.0)    — view direction (into screen).
    //              This is an approximation: correct for orthographic projection
    //              and a good central-pixel approximation for perspective.
    //
    // Project the surface normal onto the slice plane (perpendicular to axisVec)
    // and compute the signed angle between that projection and viewAxis.
    //   projNormal = surfNormal − axisVec · dot(surfNormal, axisVec)
    //   n = signNorm · acos(clamp(dot(projNormal, viewAxis) / |projNormal|))
    //
    // Matches XeGTAO.hlsli lines ~640–660.
    let axisVec = vec3f(dir.x, dir.y, 0.0);
    let viewAxis = vec3f(0.0, 0.0, -1.0);

    let projNormal = surfNormal - axisVec * dot(surfNormal, axisVec);
    let projNormalLen = max(length(projNormal), 1e-6);

    // orthoDir = component of dir perpendicular to viewAxis in the slice plane
    // (XeGTAO's orthoDirectionVec). Used only to determine the sign of n.
    // In our coord system, orthoDir = vec3(dir.x, dir.y, 0) - it is axisVec
    // itself here because axisVec lies in the XY plane (depth = 0).
    let signNorm = sign(dot(axisVec, projNormal));
    let cosN = clamp(dot(projNormal, viewAxis) / projNormalLen, -1.0, 1.0);
    // n = γ: signed angle of projected normal from view axis in slice plane.
    let n = signNorm * acos(cosN);

    // ── Horizon march: track max cos(θ_h) on each side ───────────────────
    // cos(θ_h) > 0 means the sample rises above the center depth (occluder
    // above the surface). horizonPos = positive-dir side, horizonNeg = neg side.
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

    // ── Jiménez 2016 §4.2 Eq. 11 slice integral ──────────────────────────
    //
    // Convert cos(θ_h) → θ_h angles, placing them in the signed-angle frame:
    //   h0 = horizon angle for the negative-direction side → maps to [-π, 0]
    //   h1 = horizon angle for the positive-direction side → maps to [0, +π]
    //
    // Clamp to ±π/2 around n so we only integrate the upper hemisphere
    // w.r.t. the surface normal (XeGTAO.hlsli commented-out clamp lines):
    //   h0 = n + clamp(h0 - n, -π/2, +π/2)
    //   h1 = n + clamp(h1 - n, -π/2, +π/2)
    let h0_raw = -acos(clamp(horizonNeg, -1.0, 1.0)); // negative side → [-π, 0]
    let h1_raw =  acos(clamp(horizonPos, -1.0, 1.0)); // positive side → [0, +π]

    let h0 = n + clamp(h0_raw - n, -PI_HALF, PI_HALF);
    let h1 = n + clamp(h1_raw - n, -PI_HALF, PI_HALF);

    // Weighted by how much of the normal lies in the slice plane.
    // projNormalLen → 1 when normal is in-plane; → 0 when normal is
    // perpendicular to the slice (slice contributes nothing for that normal).
    let localVis = projNormalLen * (gtaoSliceIntegral(h0, n, cosN)
                                  + gtaoSliceIntegral(h1, n, cosN));
    aoSum = aoSum + localVis;
  }

  let aoRaw = clamp(aoSum / f32(NUM_DIRECTIONS), 0.0, 1.0);
  let vis   = pow(aoRaw, gtao_ubo.intensity);

  // ── Jiménez 2016 §5.2 / Eq. 16 — multi-bounce ambient occlusion ──────────
  //
  // Per-channel albedo-aware brightening that approximates the "missing" energy
  // from inter-reflections.  With ρ = 1 (white surface), a_mb ≡ vis (identity).
  // With ρ → 0 (black surface), a_mb → 0 (no AO leakage on dark surfaces).
  //
  // Albedo is sampled from the shade pass's hdrAlbedoOut (written by M9.C).
  // hdrAlbedoOut is full-resolution; GTAO is half-resolution, so we sample at
  // the 2×2 quad centre (fullPx = gid.xy * 2 + 1), same as gNormalDepth.
  //
  // Coefficients from Jiménez 2016 Eq. 16 (table in §5.2):
  //   a_mb = ((2.0404·ρ − 0.3324)·v + (−4.7951·ρ + 0.6417))·v + (2.7552·ρ + 0.6903))·v
  //
  // Reference: Jiménez et al. 2016, §5.2 / Eq. 16.
  let albedoSample = textureLoad(gtao_albedo, vec2i(fullPx), 0).rgb;
  let albedo = clamp(albedoSample, vec3f(0.0), vec3f(1.0));
  let ca = 2.0404 * albedo - vec3f(0.3324);
  let cb = -4.7951 * albedo + vec3f(0.6417);
  let cc = 2.7552 * albedo + vec3f(0.6903);
  let aoMb = clamp(((ca * vis + cb) * vis + cc) * vis, vec3f(0.0), vec3f(1.0));

  textureStore(gtao_aoOut, gid.xy, vec4f(aoMb, 1.0));
}
`;

/** W1-R6 — declarative include-graph entry. Self-contained. */
export const GTAO_MODULE: WgslModule = {
  name: 'gtao',
  source: GTAO_WGSL,
  requires: [],
};
