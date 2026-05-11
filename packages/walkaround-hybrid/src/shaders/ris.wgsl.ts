/**
 * RIS (Resampled Importance Sampling) compute pass.
 *
 * Samples M_LIGHT=64 direct light candidates + M_BRDF=1 BRDF candidate per pixel,
 * selects the best via weighted reservoir sampling (RIS), then applies a visibility
 * test to finalize the W weight.
 *
 * This pass does primary ray casting to find the hit surface (no separate G-buffer
 * raster pass needed) — primary-ray-cast mode using manual device.createShaderModule()
 * with full primary-ray cast instead of a rasterized G-buffer, which is simpler and
 * provably correct.
 *
 * Bind groups: see WalkaroundGPUPipeline bind group layouts.
 *   @group(0): frame (placeholder G-buffer textures + reservoirs)
 *   @group(1): scene (BVH + emitters)
 *   @group(2): ubo   (camera matrices + per-frame params)
 */

export const RIS_WGSL = /* wgsl */ `

// ============================================================
// Bind group declarations
// ============================================================

// Group 0: per-frame G-buffer + reservoirs
// (G-buffer textures are bound but not used in primary-ray-cast mode;
//  they are kept for bind group layout compatibility with other passes)
@group(0) @binding(0) var gDepth:     texture_2d<f32>;
@group(0) @binding(1) var gNormal:    texture_2d<f32>;
@group(0) @binding(2) var gAlbedo:    texture_2d<f32>;
@group(0) @binding(3) var gRough:     texture_2d<f32>;
@group(0) @binding(4) var motionVec:  texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> currentReservoir:  array<u32>;
@group(0) @binding(6) var<storage, read>       previousReservoir: array<u32>;
@group(0) @binding(7) var<storage, read_write> spatialReservoir:  array<u32>;
@group(0) @binding(8) var hdrColorOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var nearestSampler: sampler;

// Group 1: static scene BVH + emitters
// bvh_index is array<vec4u>: .xyz=vertex indices, .w=packed RGBA8 material color+transmission
@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;
@group(1) @binding(4) var<storage, read> emitterCdf:   array<f32>;

// Group 2: uniform buffer (WalkaroundUBO struct defined in COMMON_WGSL)
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

// Reservoir storage helpers (RESERVOIR_DI_STRIDE, loadReservoirDI_rw,
// storeReservoirDI_rw) live in COMMON_WGSL.

// invertMat4_common + generatePrimaryRay_common live in common.wgsl;
// they are prepended to RIS_WGSL at compile time (see
// WalkaroundGPUPipeline shader-module concat).

// ============================================================
// Emitter target function (unshadowed)
// ============================================================
fn computePHat(lid: u32, pos: vec3f, normal: vec3f, wo: vec3f, albedo: vec3f, rough: f32, metal: f32) -> f32 {
  let e = emitters[lid];
  let centroid = (e.vA + e.vB + e.vC) / 3.0;
  let toL = centroid - pos;
  let dist2 = dot(toL, toL);
  if (dist2 < 1e-8) { return 0.0; }
  let wi = toL / sqrt(dist2);
  let nDotL  = max(0.0, dot(normal, wi));
  let nlDotL = max(0.0, dot(-e.normal, wi));
  if (nDotL < 1e-6 || nlDotL < 1e-6) { return 0.0; }
  // evalGGX includes NdotL; G is the emitter geometry term only (nlDotL/dist²).
  // Use emitterGeometry helper from common.wgsl to apply the EMITTER_DIST2_FLOOR
  // clamp consistently with shade.wgsl (sweep finding Bug 3 — the RIS
  // reservoir was importance-sampling against an unclamped p̂ while shade
  // evaluated with the clamped one, causing the ratio mismatch to show
  // up as fireflies in temporal+spatial reuse).
  let G    = emitterGeometry(nlDotL, dist2);
  let brdf = evalGGX(albedo, rough, metal, normal, wo, wi);
  return luminance(e.Le * brdf * G);
}

// ============================================================
// RIS main kernel -- primary ray cast + reservoir sampling
// ============================================================
// M_LIGHT 64 (restored — was briefly 32 for perf). The per-pass
// timestamp telemetry showed the spatial pass was the actual
// bottleneck (~22ms × 2 passes); RIS was 7ms regardless. Halving
// M_LIGHT shaved ~3.5ms of RIS while doubling per-frame variance
// at the cell level — bad trade for fidelity. Back to 64 candidates
// for cleaner direct-light reservoirs feeding spatial+temporal.
const M_LIGHT = 64u;
const M_BRDF  = 1u;

@compute @workgroup_size(8, 8, 1)
fn risMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  if (any(gid.xy >= dims)) { return; }

  let pixelIdx = gid.y * dims.x + gid.x;
  var rng = pcgInit(gid.x ^ (ubo.frameSeed * 73856093u), gid.y ^ (ubo.frameSeed * 19349663u), ubo.frameSeed);

  // --- Primary ray cast to find the surface hit ---
  // Compute inverse view-projection matrix for ray generation.
  // UBO stores view and proj separately; we compose VP = proj * view, then invert.
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);

  let primaryRay = generatePrimaryRay_common(gid.x, gid.y, dims.x, dims.y, ubo.cameraPos, invVP);
  let hit = bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, primaryRay);

  if (!hit.didHit) {
    // Sky pixel -- write sky color directly to HDR output, empty reservoir.
    // skyTint × skyIrradiance from UBO (computeLightingState); replaces
    // hardcoded sky color.
    storeReservoirDI_rw(&currentReservoir, pixelIdx, emptyReservoirDI());
    let skyColor = ubo.skyTint * ubo.skyIrradiance;
    textureStore(hdrColorOut, gid.xy, vec4f(skyColor, 1.0));
    return;
  }

  // Surface hit -- extract position, normal, material color from packed bvh_index.
  let pos    = primaryRay.origin + primaryRay.direction * hit.dist;
  let normal = hit.normal;
  let wo     = -primaryRay.direction;

  // Decode per-triangle material color from bvhIndex[triIdx].w (RGBA8 packed).
  let matColor  = decodeMaterialColor(hit.matColorPacked);
  let isGlass   = matColor.a > 0.3;  // transmission > ~76/255
  // Use the actual BVH-baked material color for all surfaces.
  let albedo    = matColor.rgb;
  let roughness = select(0.85, 0.05, isGlass);
  let metalness = 0.0;

  var r = emptyReservoirDI();
  let totalPower = max(ubo.totalEmPower, 1e-8);
  let emCount = max(ubo.emitterCount, 1u);

  // --- M_LIGHT candidates from emitter distribution ---
  for (var i = 0u; i < M_LIGHT; i++) {
    let xi  = rand_f32(&rng);
    let lid = sampleEmitterIdx(&emitterCdf, emCount, xi);
    let e   = emitters[lid];
    let ls  = sampleEmitterPoint(e, rand2(&rng));

    let toL   = ls.pos - pos;
    let dist2 = dot(toL, toL);
    if (dist2 < 1e-8) { continue; }
    let wi     = toL / sqrt(dist2);
    let nDotL  = max(0.0, dot(normal, wi));
    let nlDotL = max(0.0, dot(-e.normal, wi));
    if (nDotL < 1e-6 || nlDotL < 1e-6) { continue; }

    // evalGGX includes NdotL; G is the emitter geometry term only.
    // Same emitterGeometry helper as computePHat above so the per-candidate
    // p̂ in the M_LIGHT loop matches the reservoir's selection p̂ matches
    // shade's evaluation p̂ (sweep finding Bug 3).
    let G    = emitterGeometry(nlDotL, dist2);
    let brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi);
    let pHat = luminance(ls.Le * brdf * G);

    // p(x): emitter pmf x per-triangle area pdf.
    let emitterPmf = max(1e-15, (luminance(e.Le) * e.area) / totalPower);
    let pX = max(1e-15, emitterPmf * ls.pdfArea);
    let w = select(0.0, pHat / pX, pHat > 0.0);
    updateReservoirDI(&r, lid, w, &rng);
  }

  // --- Visibility test on chosen candidate ---
  if (r.M > 0u && r.w_sum > 0.0) {
    let lid = r.lightId;
    let e   = emitters[lid];
    let ls  = sampleEmitterPoint(e, vec2f(0.5, 0.5));  // centroid
    let toL = ls.pos - pos;
    let dist = length(toL);
    let wi  = toL / dist;
    let shadowOrig = pos + normal * 1e-3;
    let occluded = bvhIntersectAny(&bvh_index, &bvh_position, &bvh, shadowOrig, wi, dist - 2e-3);
    if (occluded) {
      r.w_sum = 0.0;
      r.W     = 0.0;
    } else {
      let pHatZ = computePHat(lid, pos, normal, wo, albedo, roughness, metalness);
      r.W = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 0.0);
    }
  }

  storeReservoirDI_rw(&currentReservoir, pixelIdx, r);
}
`;
