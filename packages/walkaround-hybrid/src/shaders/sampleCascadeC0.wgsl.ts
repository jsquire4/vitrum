/**
 * sampleCascadeC0 — WGSL helper that reads Sannikov 2023 Radiance Cascades
 * level-0 storage at a world position + surface normal and returns a
 * Lambertian-irradiance estimate.
 *
 * Bind group / bindings used (added by SHADE_MODULE):
 *   @group(3) @binding(4) var<storage, read> rcCascade0: array<vec4f>;
 *   @group(3) @binding(5) var<uniform>       rcParams:   RCParams;
 *
 * Packed into the existing DDGI group(3) rather than a new group(4)
 * because adapter caps maxBindGroups = 4 on Lovelace and below
 * (see pipeline/bindGroupLayouts.ts:170). DDGI uses bindings 0-3.
 *
 * Algorithm (W8 Phase 3 → Phase-3b trilinear, 2026-05-28):
 *   1. Map worldPos → probe-grid UV in [0,1]³ via
 *      (worldPos - probeOriginWorld) / roomSize.
 *   2. TRILINEAR interpolation over the 8 surrounding cascade-0 probes.
 *      The producer stores each probe centred at `(p + 0.5) / count` in
 *      grid-UV (probeRayCast.wgsl.ts:220), so the probe-centre-relative
 *      grid coordinate is `g = probeUV·count − 0.5`. `floor(g)` gives the
 *      lower-corner probe; `frac(g)` gives the trilinear blend weights.
 *      The 8 corners are clamped to `[0, count−1]` so edge/out-of-bounds
 *      shading points degrade to a (partly-degenerate) blend of the
 *      boundary probes rather than reading out of range. A corner whose
 *      per-probe cosine weight `Wsum ≤ 1e-4` (no ray faces the receiver
 *      normal, or an uninitialised probe) is dropped from the blend and
 *      its trilinear weight is redistributed across the remaining corners
 *      (re-normalised), so a zero-radiance probe never leaks darkness into
 *      the interpolated estimate. When ALL 8 corners are degenerate the
 *      result is vec3f(0), matching the old nearest-probe Wsum guard.
 *   3. Each corner probe integrates the stored rays cosine-weighted by
 *      `dot(normal, ω_k)`, where
 *      `ω_k = octDecode((vec2f(gx, gy) + 0.5) / rayGridSize · 2 − 1)`
 *      matches the producer's ray-direction generation in
 *      probeRayCast.wgsl.ts:216-221 (unjittered ray-center direction).
 *      A degenerate blend (shading point exactly on one probe, all weight
 *      on a single corner) reduces bit-for-bit to the previous
 *      nearest-probe estimate.
 *
 * The producer stores `L_i(ω_k)` per ray (radiance through that ray
 * direction). The cosine-weighted sum approximates
 *   irradiance(N) = ∫_Ω L(ω) (N·ω)+ dω
 * for a Lambertian receiver. The Lambertian BRDF factor `1/π` is applied
 * here so the caller adds the result directly into `Lo_indirect`. Albedo
 * is omitted (Schied 2017 §4.1 demodulation — `indirectCombine`
 * re-multiplies after denoising).
 *
 * When `rcParams.enabled == 0u`, returns `vec3f(0)` — the caller can
 * conditionally weight it in the MIS sum. This makes the rcEnabled
 * toggle bit-identical to "RC off" without a separate shader compile.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SAMPLE_CASCADE_C0_WGSL = /* wgsl */`

// ─── Bind group: RC cascade-0 storage + params UBO ────────────────────
// Packed into group(3) (alongside DDGI's irradiance/visibility/sampler/grid
// at bindings 0..3) because adapter cap maxBindGroups = 4 on Lovelace.
@group(3) @binding(4) var<storage, read> rcCascade0: array<vec4f>;

struct RCParams {
  // World-space corner of the cascade-0 probe grid (= scene AABB min).
  probeOriginWorld: vec3f,
  // RC's MIS weight in the indirect sum. The ReSTIR-GI weight is
  // (1.0 - rcWeight); the two are forced to sum to 1.
  rcWeight: f32,
  // Room size (= scene AABB extent). Floored at 1e-6 host-side.
  roomSize: vec3f,
  // 1 when RC is active; 0 disables the sample (caller treats Lo_rc=0).
  enabled: u32,
  probeCount: vec3u,
  raysPerProbe: u32,
  rayGridSize: u32,
  // 12 bytes pad to reach 16-byte aligned struct end (64 bytes total).
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(3) @binding(5) var<uniform> rcParams: RCParams;

// ─── Per-probe cosine-weighted directional integral ───────────────────
// Integrates a single cascade-0 probe's stored rays, cosine-weighted by
// dot(normal, omega_k). Returns the irradiance estimate in .rgb and the
// cosine-weight mass in .w (so the trilinear caller can detect a
// degenerate / uninitialised probe and renormalise around it).
//
// A7 receiver fix (2026-06-10): the estimator is a Monte Carlo irradiance
// integral over the full sphere. The producer casts N rays uniformly
// distributed over 4π steradians (uniform octahedral tiling), each with
// solid angle Ω_k ≈ 4π/N. The irradiance estimate is:
//
//   E(n) = Σ_k L_k · max(0, n·ω_k) · Ω_k
//        = (4π/N) · Σ_k L_k · cos_k
//        = 4π · Le / raysPerProbe          (Le = Σ L_k·cos_k)
//
// The 1/π Lambertian factor is applied once by the caller after the blend,
// giving the final indirect signal (E/π).
//
// Previous formula (Le/Wsum * raysPerProbe * 0.5) computed the
// cosine-weighted MEAN radiance times N/2, which grows linearly with N and
// over-estimates by a factor of ≈ 2.55 at the default N=16 — any
// cascadeDims override silently changed scene brightness. The new formula is
// ray-count-independent: at N=16 AND N=64 a unit-radiance isotropic field
// returns E ≈ π (the correct Lambertian irradiance for L=1 uniform field).
//
// Wsum is still returned in .w so the trilinear caller can detect and skip
// degenerate / uninitialised probes (the probe normal-facing guard is
// unchanged — a probe with no rays facing the receiver normal returns 0).
//
// Reference: Veach 1997 §2.3 — MC estimator (1/N) · Σ f/pdf with
// pdf = 1/(4π) for uniform sphere → estimate = (4π/N) · Σ L cos.
const FOUR_PI_RC: f32 = 12.56637061436;   // 4π, precomputed to avoid /0 on pure-zero Wsum
fn rcProbeIrradiance(probeIdx: u32, normal: vec3f) -> vec4f {
  let base = probeIdx * rcParams.raysPerProbe;
  var Le: vec3f = vec3f(0.0);
  var Wsum: f32 = 0.0;
  let inv_rg = 1.0 / f32(rcParams.rayGridSize);
  for (var ri: u32 = 0u; ri < rcParams.raysPerProbe; ri = ri + 1u) {
    let gx = f32(ri % rcParams.rayGridSize);
    let gy = f32(ri / rcParams.rayGridSize);
    let rayUV = (vec2f(gx, gy) + 0.5) * inv_rg;
    // octDecode expects [-1,1] octahedral coords; matches producer
    // octDecode(rayUV * 2 - 1) exactly.
    let dir = octDecode(rayUV * 2.0 - 1.0);
    let cosTheta = max(0.0, dot(normal, dir));
    if (cosTheta < 1e-4) { continue; }
    let L = rcCascade0[base + ri].rgb;
    Le = Le + L * cosTheta;
    Wsum = Wsum + cosTheta;
  }
  if (Wsum > 1e-4) {
    // A7: E = (4π/N) · Σ L_k·cos_k = 4π · Le / raysPerProbe (N-independent).
    return vec4f(Le * FOUR_PI_RC / f32(rcParams.raysPerProbe), Wsum);
  }
  return vec4f(0.0);
}

// ─── sampleCascadeC0 ──────────────────────────────────────────────────
fn sampleCascadeC0(worldPos: vec3f, normal: vec3f) -> vec3f {
  if (rcParams.enabled == 0u) { return vec3f(0.0); }

  // 1. World → probe-grid UV in [0,1]³, then to probe-centre-relative grid
  //    coordinates. Producer probes sit at (p + 0.5)/count in grid-UV
  //    (probeRayCast.wgsl.ts:220), so g = probeUV·count − 0.5.
  let count = vec3f(rcParams.probeCount);
  let probeUV = (worldPos - rcParams.probeOriginWorld) / rcParams.roomSize;
  let g = probeUV * count - vec3f(0.5);

  // Lower-corner probe + fractional position (the trilinear weights).
  let g0 = floor(g);
  let f = g - g0;                          // ∈ [0,1) per axis (degenerate → 0)
  let cmax = count - vec3f(1.0);

  // 2. Trilinear blend of the 8 surrounding probes. A corner with no
  //    cosine mass (Wsum ≤ 1e-4 — uninitialised / normal-facing-away) is
  //    dropped and its weight redistributed via the running weightSum
  //    renormalisation, so zero radiance never leaks into the blend.
  var blendL: vec3f = vec3f(0.0);
  var weightSum: f32 = 0.0;
  for (var corner: u32 = 0u; corner < 8u; corner = corner + 1u) {
    let dx = f32(corner & 1u);
    let dy = f32((corner >> 1u) & 1u);
    let dz = f32((corner >> 2u) & 1u);
    // Trilinear weight = ∏ mix(1−f, f, d) over the three axes.
    let wTri = mix(1.0 - f.x, f.x, dx)
             * mix(1.0 - f.y, f.y, dy)
             * mix(1.0 - f.z, f.z, dz);
    if (wTri <= 0.0) { continue; }
    // Clamp the corner to valid grid bounds (edge handling).
    let pi = vec3u(clamp(g0 + vec3f(dx, dy, dz), vec3f(0.0), cmax));
    let probeIdx = pi.z * rcParams.probeCount.x * rcParams.probeCount.y
                 + pi.y * rcParams.probeCount.x
                 + pi.x;
    let probe = rcProbeIrradiance(probeIdx, normal);
    if (probe.w <= 1e-4) { continue; }     // degenerate corner — skip + renorm
    blendL = blendL + probe.rgb * wTri;
    weightSum = weightSum + wTri;
  }

  // Apply the Lambertian BRDF response (1/π) once, after renormalising the
  // blend by the surviving trilinear weight mass. We return the lighting
  // signal only (indirectCombine re-multiplies by albedo).
  if (weightSum > 1e-4) {
    return blendL * INV_PI / weightSum;
  }
  return vec3f(0.0);
}

`;

/**
 * W1-R6 — declarative include-graph entry. Requires `octahedralCore` for
 * `octDecode` and `common` for the `INV_PI` constant.
 */
export const SAMPLE_CASCADE_C0_MODULE: WgslModule = {
  name: 'sampleCascadeC0',
  source: SAMPLE_CASCADE_C0_WGSL,
  requires: ['common', 'octahedralCore'],
};
