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
 * Algorithm (W8 Phase 3, "Track A" minimum-viable):
 *   1. Map worldPos → probe-grid UV in [0,1]³ via
 *      (worldPos - probeOriginWorld) / roomSize.
 *   2. Quantise to the nearest probe (no trilinear interpolation yet —
 *      that's a Phase-3b follow-up; the rev-blame on the cascade producer
 *      already linearises the spatial signal via its own merge math).
 *   3. Integrate the 16 stored rays cosine-weighted by `dot(normal, ω_k)`,
 *      where `ω_k = octDecode((vec2f(gx, gy) + 0.5) / rayGridSize * 2 - 1)`
 *      matches the producer's ray-direction generation in
 *      probeRayCast.wgsl.ts:216-221 (unjittered ray-center direction).
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

// ─── sampleCascadeC0 ──────────────────────────────────────────────────
fn sampleCascadeC0(worldPos: vec3f, normal: vec3f) -> vec3f {
  if (rcParams.enabled == 0u) { return vec3f(0.0); }

  // 1. World → probe-grid UV in [0,1]³.
  let probeUV = (worldPos - rcParams.probeOriginWorld) / rcParams.roomSize;
  // Quantise to nearest probe — clamped to grid bounds.
  let pf = clamp(
    probeUV * vec3f(rcParams.probeCount),
    vec3f(0.0),
    vec3f(rcParams.probeCount) - vec3f(1.0),
  );
  let pi = vec3u(pf);
  let probeIdx = pi.z * rcParams.probeCount.x * rcParams.probeCount.y
               + pi.y * rcParams.probeCount.x
               + pi.x;
  let base = probeIdx * rcParams.raysPerProbe;

  // 2. Cosine-weighted directional integral over the 16 stored rays.
  //    Ray direction recovery matches probeRayCast.wgsl.ts:216-221
  //    (unjittered ray-center direction; the per-frame jitter averages
  //    out over the temporal accumulator above us).
  var Le: vec3f = vec3f(0.0);
  var Wsum: f32 = 0.0;
  let rg = f32(rcParams.rayGridSize);
  let inv_rg = 1.0 / rg;
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
  // Apply the Lambertian BRDF response (1/π). With Wsum being the sum of
  // raw cosines (≈ Σ 1·cos = N/2 for hemispheres uniformly sampled), the
  // normalisation Le / Wsum × 1 yields a unit-radiance estimate; multiply
  // by π × INV_PI = 1 net. We return Le · INV_PI (lighting signal only;
  // indirectCombine re-multiplies by albedo).
  if (Wsum > 1e-4) {
    return Le * INV_PI / Wsum * f32(rcParams.raysPerProbe) * 0.5;
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
