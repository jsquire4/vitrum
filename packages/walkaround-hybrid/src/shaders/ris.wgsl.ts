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

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RIS_WGSL = /* wgsl */ `

// ============================================================
// Bind group declarations
// ============================================================

// Group 0: only the slots ris actually reads. The shared FrameBindGroup
// layout carries 10 entries (gDepth/Normal/Albedo/Rough/motionVec/
// 3 reservoirs/hdrColorOut/nearestSampler) for temporal/spatial/shade;
// WGSL allows the shader to declare a subset (W5-I1 cleanup 2026-05-18).
@group(0) @binding(5) var<storage, read_write> currentReservoir:  array<u32>;
@group(0) @binding(8) var hdrColorOut: texture_storage_2d<rgba16float, write>;

// Group 1: static scene BVH + emitters
// bvh_index is array<vec4u>: .xyz=vertex indices, .w=packed RGBA8 material color+transmission
@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;
@group(1) @binding(4) var<storage, read> emitterCdf:   array<f32>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;

// Group 2: uniform buffer (WalkaroundUBO struct defined in COMMON_WGSL)
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

// Reservoir storage helpers (RESERVOIR_DI_STRIDE, loadReservoirDI_rw,
// storeReservoirDI_rw) live in COMMON_WGSL.

// invertMat4_common + generatePrimaryRay_common live in common.wgsl;
// injected from common.wgsl via the W1-R6 WGSL include-graph
// (requires: ['restirPHat'] → 'common').

// W2-C7 — emitter target function p̂ moved to restirPHat.wgsl
// (canonical restir_di_compute_phat_from_surface(lid, surf)). RIS calls
// it once at the visibility-test stage with a PrimarySurface built from
// the inline primary-cast result (the M_LIGHT loop computes its own
// per-candidate p̂ inline because it uses the sampled emitter point
// ls.pos, not the centroid the canonical helper assumes).

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
  let hit = traceSceneFirstHit(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    primaryRay, ubo.triIntersectEpsilon);

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
    let xiEm = rand_f32(&rng);
    let lid = sampleEmitterIdx(&emitterCdf, emCount, xiEm);
    let e   = emitters[lid];
    let xiTri = rand2(&rng);
    let ls  = sampleEmitterPoint(e, xiTri);

    let toL   = ls.pos - pos;
    let dist2 = dot(toL, toL);
    if (dist2 < 1e-8) { continue; }
    let wi     = toL / sqrt(dist2);
    let nDotL  = max(0.0, dot(normal, wi));
    let nlDotL = max(0.0, dot(-e.normal, wi));
    if (nDotL < 1e-6 || nlDotL < 1e-6) { continue; }

    // evalGGX includes NdotL; G is the emitter geometry term only.
    // Same emitterGeometry helper as the canonical
    // restir_di_compute_phat_from_surface (restirPHat.wgsl), so the
    // per-candidate p̂ in the M_LIGHT loop matches the reservoir's
    // selection p̂ matches shade's evaluation p̂ (sweep finding Bug 3).
    let G    = emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor);
    let brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi);
    let pHat = luminance(ls.Le * brdf * G);

    // p(x): emitter pmf x per-triangle area pdf.
    let emitterPmf = max(1e-15, (luminance(e.Le) * e.area) / totalPower);
    let pX = max(1e-15, emitterPmf * ls.pdfArea);
    let w = select(0.0, pHat / pX, pHat > 0.0);
    updateReservoirDI(&r, lid, xiTri, w, &rng);
  }

  // --- Visibility test on chosen candidate ---
  if (r.M > 0u && r.w_sum > 0.0) {
    let lid = r.lightId;
    let e   = emitters[lid];
    // 2026-05-18 sweep finding #3 fix — sample the EXACT point that was
    // chosen by the WRS (r.xi), not the centroid. The centroid bias was
    // a real correctness gap: visibility at the centroid disagrees with
    // visibility at the sample for any emitter whose extent is comparable
    // to the occluder's.
    let ls  = sampleEmitterPoint(e, r.xi);
    let toL = ls.pos - pos;
    let dist = length(toL);
    let wi  = toL / dist;
    let shadowOrig = pos + normal * 1e-3;
    // skipGlass=true: matches pre-canonical ReSTIR shadow-ray glass filter
    // (light passes through glass; per-channel tinted-visibility handles tint).
    let occluded = traceSceneAny(
      ubo.bvhMode, ubo.tlasNodeCount,
      &bvh_index, &bvh_position, &bvh,
      &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
      &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
      shadowOrig, wi, dist - 2e-3, ubo.triIntersectEpsilon, true);
    if (occluded) {
      r.w_sum = 0.0;
      r.W     = 0.0;
    } else {
      // Build a PrimarySurface from the inline-cast values so the canonical
      // p̂ helper (Bitterli 2020 §4.3 — identical across RIS/temporal/spatial)
      // sees the same struct shape as the reuse passes.
      var surf: PrimarySurface;
      surf.hit    = true;
      surf.pos    = pos;
      surf.normal = normal;
      surf.wo     = wo;
      surf.albedo = albedo;
      surf.rough  = roughness;
      surf.metal  = metalness;
      surf.depth  = hit.dist;
      let pHatZ = restir_di_compute_phat_from_surface(lid, surf);
      r.W = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 0.0);
    }
  }

  storeReservoirDI_rw(&currentReservoir, pixelIdx, r);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  W2-C7: depends on the canonical ReSTIR p̂ helper (which transitively
 *  requires `common`). RIS does not require restirCastPrimary because it
 *  inlines its primary cast (the surface decode feeds the M_LIGHT loop's
 *  per-candidate BRDF evaluation; converting RIS to use the canonical
 *  PrimarySurface cast would require a larger restructure than C7+C9
 *  warrants — see restirCastPrimary.wgsl.ts header). The composer emits
 *  `common, restirPHat, ris`. */
export const RIS_MODULE: WgslModule = {
  name: 'ris',
  source: RIS_WGSL,
  requires: ['restirPHat'],
};
