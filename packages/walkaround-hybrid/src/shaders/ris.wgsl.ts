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
// WS1 (2026-05-29) — per-vertex world-space normals for the smooth shading
// normal. ris uses it for the BRDF / candidate p̂; the geometric normal is
// kept for the shadow-ray offset. (Beer texture binding 5 is shade-only — ris
// declares a subset of the scene BGL; WGSL permits that.)
@group(1) @binding(11) var<storage, read> bvh_normal: array<vec4f>;
// B1 — per-triangle roughness+metalness (r32uint texture). Decoded into the
// real GGX roughness/metal that feed evalGGX in the candidate p̂ (was hardcoded
// rough=0.85/0.05, metal=0).
@group(1) @binding(14) var bvh_material: texture_2d<u32>;

// Group 2: uniform buffer (WalkaroundUBO struct defined in COMMON_WGSL)
@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

// Reservoir storage helpers (RESERVOIR_DI_STRIDE, loadReservoirDI_rw,
// storeReservoirDI_rw) live in COMMON_WGSL.

// invertMat4_common + generatePrimaryRay_common live in common.wgsl;
// injected from common.wgsl via the W1-R6 WGSL include-graph
// (requires: ['restirPHat'] → 'common').

// W2-C7 — emitter target function p̂ moved to restirPHat.wgsl
// (canonical restir_di_compute_phat_xi(lid, xi, surf)). RIS calls
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
// Wave 4 — M_ENV: one importance-sampled HDRI candidate per pixel.
// HDRI maps are spatially smooth (sinθ-weighted CDF pre-baked); one
// sample captures the dominant bright region (e.g. a sun disk) with
// low variance. A second env candidate would increase cost by 0.8% of
// RIS while halving env variance — the gain doesn't justify the cost
// given temporal accumulation already suppresses env noise.
// Gate: envHasMap() → 0 contribution for no-HDRI scenes (p̂ = 0 →
// w = 0 → reservoir unchanged), so M_ENV = 1 is a no-op for all
// emitter-only scenes — byte-identical with the pre-Wave-4 kernel.
const M_ENV   = 1u;

@compute @workgroup_size(8, 8, 1)
fn risMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;

  // Checkerboard sparse-RIS (opt-in; OFF by default). RIS SEEDS the per-pixel
  // reservoir (primary BVH cast + M_LIGHT emitter candidates) — the single most
  // expensive initial-candidate stage. When checkerboardOn == 1u the host
  // COMPACTS the dispatch to ~half the threads (one per active-parity pixel), so
  // gid indexes the active-parity pixel set rather than the full-res grid.
  // Decode it back to the true pixel, the SAME decode shade/spatial use:
  //   px = gid.x*2 + ((gid.y + frameParity) & 1u),  py = gid.y
  // This lands EXACTLY on the (px+py)&1u == frameParity set ShadePass shades +
  // SpatialReservoirPass refines this frame (all three read the SAME
  // frameParity/checkerboardOn from the UBO), so RIS re-seeds precisely the
  // reservoirs shade consumes this frame. The complementary GAP-parity reservoir
  // slots are NOT written here — they retain the carried-forward reservoir RIS
  // wrote when they were last active-parity (the parity flips each frame), which
  // the FULL-RATE temporal pass then reads as its cur reservoir and keeps refining against
  // the reprojected history. So gap pixels keep a VALID reservoir for
  // temporal/spatial; they just miss ONE fresh candidate that frame
  // (effectively a half-rate candidate cadence reconstructed by temporal + the
  // denoiser). The compacted X grid (ceil(W/2) columns) can overshoot the row's
  // last active pixel on odd widths; that overshoot lands at px >= W and is
  // caught by the bounds guard. When OFF, pix == gid.xy and the dispatch is
  // full-res ⇒ bit-identical with the pre-checkerboard kernel.
  var pix = gid.xy;
  if (ubo.checkerboardOn == 1u) {
    let startCol = (gid.y + ubo.frameParity) & 1u;
    pix = vec2u(gid.x * 2u + startCol, gid.y);
  }
  if (any(pix >= dims)) { return; }

  let pixelIdx = pix.y * dims.x + pix.x;
  var rng = pcgInit(pix.x ^ (ubo.frameSeed * 73856093u), pix.y ^ (ubo.frameSeed * 19349663u), ubo.frameSeed);

  // --- Primary ray cast to find the surface hit ---
  // Compute inverse view-projection matrix for ray generation.
  // UBO stores view and proj separately; we compose VP = proj * view, then invert.
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);

  let primaryRay = generatePrimaryRay_common(pix.x, pix.y, dims.x, dims.y, ubo.cameraPos, invVP);
  let hit = traceSceneFirstHit(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    primaryRay, ubo.triIntersectEpsilon);

  if (!hit.didHit) {
    // Sky pixel -- write sky color directly to HDR output, empty reservoir.
    // B3 — directional IBL: when a pixel-backed HDRI is bound, sample the ACTUAL
    // map along the camera ray (rotationY-aware); envRadiance falls back to the
    // scalar skyTint × skyIrradiance when no map is present (no-HDRI byte-identity).
    storeReservoirDI_rw(&currentReservoir, pixelIdx, emptyReservoirDI());
    let skyColor = envRadiance(primaryRay.direction);
    textureStore(hdrColorOut, pix, vec4f(skyColor, 1.0));
    return;
  }

  // Surface hit -- extract position, normal, material color from packed bvh_index.
  let pos    = primaryRay.origin + primaryRay.direction * hit.dist;
  // WS1 — smooth shading normal for the BRDF / p̂; geometric normal for the
  // shadow-ray offset. V21 — applies in TLAS too (transform the LOCAL blend to
  // world by the hit instance inverse-transpose; cols passed by value, naga-safe).
  let geoNormal = hit.normal;
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = hit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < arrayLength(&tlasInstanceWorldToLocal);
  let n_i = select(0u, n_base, n_ok);
  let normal = smoothShadingNormal(
    hit, geoNormal,
    bvh_normal[hit.indices.x].xyz, bvh_normal[hit.indices.y].xyz, bvh_normal[hit.indices.z].xyz,
    n_ok,
    tlasInstanceWorldToLocal[n_i], tlasInstanceWorldToLocal[n_i + 1u], tlasInstanceWorldToLocal[n_i + 2u],
  );
  let wo     = -primaryRay.direction;

  // Decode per-triangle material color from bvhIndex[triIdx].w (RGBA8 packed).
  let matColor  = decodeMaterialColor(hit.matColorPacked);
  let isGlass   = matColor.a > 0.3;  // transmission > ~76/255
  // Use the actual BVH-baked material color for all surfaces.
  let albedo    = matColor.rgb;
  // B1 — real authored roughness/metalness from the per-tri bvh_material texture
  // (was hardcoded). The diffuse-default invariant packs 0.85 for unspecified
  // roughness / 0.05 for glass, so default-diffuse scenes are numerically
  // unchanged; authored glossy/metal surfaces now drive the GGX candidate p̂.
  let rmCoord   = vec2u(hit.indices.w % BVH_MATERIAL_TEX_WIDTH, hit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let rm        = decodeRoughMetal(textureLoad(bvh_material, vec2i(rmCoord), 0).r);
  let roughness = rm.x;
  let metalness = rm.y;

  var r = emptyReservoirDI();
  // Support-aware sample counts for mixed-measure reservoirs. Area emitters and
  // the BSDF->emitter candidate share finite-emitter support; the HDRI sentinel
  // lives on a disjoint directional domain. A selected candidate's W denominator
  // must count only candidates from its support, otherwise a single env sample is
  // averaged down by all finite-emitter candidates in the pool.
  var mAreaSupport = 0u;
  var mEnvSupport = 0u;
  let totalPower = max(ubo.totalEmPower, 1e-8);
  let emCount = max(ubo.emitterCount, 1u);

  // --- M_LIGHT candidates from emitter distribution ---
  // Light selection mode is data-driven via two UBO gates (priority order):
  //   regirEnabled == 1 ⇒ ReGIR grid (regir_sample_cell) — draws a survivor
  //       from the containing cell's pre-resampled reservoir; the survivor's
  //       stored per-cell pmf (q̂_c(e)/Ŝ) is the EXACT source pmf. O(1) per
  //       pixel regardless of light count. The grid itself was seeded by the
  //       light tree at grid-build time (see regir.wgsl).
  //   else lightTreeEnabled == 1 ⇒ spatially-aware light tree (sampleLightTree)
  //       — importance-samples emitters by power/dist², returning the exact
  //       per-pixel selection pmf;
  //   else ⇒ flat power CDF (sampleEmitterIdx) — the historical path.
  // In ALL modes the WRS source pmf used in the weight w = p̂ / p_source is
  // the EXACT pmf the selection drew from, so the estimator is unbiased in
  // every mode. When regirEnabled == 0 this reduces to the light-tree path
  // bit-identically (the regir branch is never taken).
  let useRegir = ubo.regirEnabled == 1u;
  let useLightTree = ubo.lightTreeEnabled == 1u;
  for (var i = 0u; i < M_LIGHT; i++) {
    // Select an emitter + record the EXACT selection pmf that produced it.
    var lid: u32;
    var emitterSelPmf: f32;
    if (useRegir) {
      let rs = regir_sample_cell(pos, &rng);
      // Empty survivor (cell had no positive-target emitter) or non-positive
      // pmf cannot contribute — skip rather than emit an infinite weight.
      if (rs.emitterIndex < 0 || rs.pSel <= 0.0) { continue; }
      lid = u32(rs.emitterIndex);
      // pSel = q̂_c(lid)/Ŝ is the EXACT per-cell selection pmf the grid stored;
      // dividing p̂ by it (× pdfArea below) keeps RIS unbiased.
      emitterSelPmf = rs.pSel;
    } else if (useLightTree) {
      let lt = sampleLightTree(pos, ubo.emitterDist2Floor, ubo.lightTreeNodeCount, &rng);
      // Degenerate guard: a malformed leaf (emitterIndex < 0) or non-positive
      // pdf cannot contribute — skip rather than emit an infinite weight.
      if (lt.emitterIndex < 0 || lt.pdf <= 0.0) { continue; }
      lid = u32(lt.emitterIndex);
      emitterSelPmf = lt.pdf;
    } else {
      let xiEm = rand_f32(&rng);
      lid = sampleEmitterIdx(&emitterCdf, emCount, xiEm);
      // Flat power CDF pmf: p(emitter) = luminance(Le)·area / totalPower.
      emitterSelPmf = (luminance(emitters[lid].Le) * emitters[lid].area) / totalPower;
    }
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
    // Same emitterGeometry helper as the canonical xi-aware p̂ helper
    // (restirPHat.wgsl), so the
    // per-candidate p̂ in the M_LIGHT loop matches the reservoir's
    // selection p̂ matches shade's evaluation p̂ (sweep finding Bug 3).
    let G    = emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor);
    let brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi);
    let pHat = luminance(ls.Le * brdf * G);

    // p(x) = emitter-selection pmf × per-triangle uniform-area pdf. The first
    // factor is the EXACT pmf the chosen sampler drew from (tree or flat CDF);
    // the area pdf factor is identical for both modes. This is the source pdf
    // the WRS weight divides p̂ by — getting it exactly right is what keeps
    // ReSTIR unbiased.
    let pX = max(1e-15, emitterSelPmf * ls.pdfArea);
    let w = select(0.0, pHat / pX, pHat > 0.0);
    updateReservoirDI(&r, lid, xiTri, w, &rng);
    mAreaSupport = mAreaSupport + 1u;
  }

  // --- M_BRDF candidate(s): GGX-VNDF-sampled BSDF candidate (B16) ────────────
  // The M_LIGHT loop importance-samples the EMITTERS; it under-samples the cases
  // where the BSDF lobe is narrow (glossy/metal) and concentrated away from the
  // emitter's solid angle, so a light-only RIS pool is high-variance on shiny
  // surfaces. This loop adds a BSDF-sampled candidate per pixel (talk-of-the-
  // trade light+BSDF MIS-style multi-strategy RIS): draw wi ∝ GGX-VNDF, find
  // which emitter triangle that ray hits, and fold the resulting on-emitter
  // sample into the SAME reservoir with the SAME (lid, xi) representation.
  //
  // UNBIASEDNESS (the load-bearing detail): RIS over heterogeneous candidate
  // strategies is unbiased as long as each candidate's WRS weight divides the
  // SHARED target p̂ by the SOURCE pdf of the strategy that generated it, all in
  // ONE measure. The light candidates use AREA measure (p = selPmf·pdfArea); the
  // BRDF candidate samples wi in SOLID-ANGLE measure, so we convert its pdf to
  // area measure via the geometry Jacobian  p_area = p_sa · dist² / cosθ_light.
  // p̂ is IDENTICAL to the light candidates (luminance(Le·brdf·G)), so the chosen
  // sample's W finalisation uses the stored xi, so the BRDF candidate is just
  // another contributor to w_sum / M over finite-emitter support.
  //
  // DIFFUSE-DEFAULT NON-BIAS: for a rough Lambertian (rough≈0.85) the VNDF lobe
  // is broad; the candidate still has the CORRECT source pdf so it cannot bias
  // the estimate. It adds at most M_BRDF=1 to r.M, which the unbiased estimator
  // already accounts for in W = w_sum/(M·p̂). It changes the rng stream + the
  // numeric result vs the pre-B16 kernel (this is a RENDER-CHANGING DI quality
  // improvement, NOT a byte-identity-preserving change — see V28 B16 A/B).
  let emCountB = arrayLength(&emitters);
  for (var bi = 0u; bi < M_BRDF; bi++) {
    // Draw a BSDF direction (VNDF) and its solid-angle pdf.
    let wiB = ggxSampleVndf(normal, wo, roughness, &rng);
    let nDotLB = dot(normal, wiB);
    if (nDotLB <= 1e-6) { continue; }
    let pdfSa = ggxVndfReflectionPdf(normal, wo, wiB, roughness);
    if (pdfSa <= 1e-8) { continue; }

    // Intersect the BSDF ray against every emitter triangle; keep the nearest
    // forward hit (emitter counts are small in these scenes — a linear test is
    // cheaper than recovering the emitter index from a full BVH closest-hit and
    // inverting its barycentrics). The hit barycentrics map directly to the
    // (lid, xi) the reservoir + visibility stage already understand.
    let brdfOrig = pos + geoNormal * 1e-3;
    var bestT = 1e30;
    var bestLid = 0u;
    var bestXi = vec2f(0.0);
    var bestLe = vec3f(0.0);
    var bestNl = vec3f(0.0);
    var found = false;
    for (var li = 0u; li < emCountB; li++) {
      let eb = emitters[li];
      let it = intersectTriangle(brdfOrig, wiB, eb.vA, eb.vB, eb.vC, ubo.triIntersectEpsilon);
      if (!it.didHit || it.dist <= 1e-4 || it.dist >= bestT) { continue; }
      // The emitter must face the incoming ray (front side emits).
      if (dot(eb.normal, wiB) >= 0.0) { continue; }
      bestT = it.dist;
      bestLid = li;
      // Invert sampleEmitterPoint: weights (u,v,w) = (1−s, s·xi.y, s·(1−xi.y))
      // with s = sqrt(xi.x). barycoord.(x,y,z) are the (A,B,C) vertex weights.
      let bA = it.barycoord.x;
      let bB = it.barycoord.y;
      let bC = it.barycoord.z;
      let sInv = clamp(1.0 - bA, 0.0, 1.0);
      let xiX = sInv * sInv;
      let bcSum = max(bB + bC, 1e-8);
      let xiY = clamp(bB / bcSum, 0.0, 1.0);
      bestXi = vec2f(xiX, xiY);
      bestLe = eb.Le;
      bestNl = eb.normal;
      found = true;
    }
    if (!found) { continue; }

    // Shared target p̂ = luminance(Le · brdf · G), evaluated for the on-emitter
    // sample the BSDF ray landed on — IDENTICAL form to the M_LIGHT candidates.
    let hitPos = brdfOrig + wiB * bestT;
    let toLB = hitPos - pos;
    let dist2B = max(dot(toLB, toLB), 1e-8);
    let nlDotLB = max(0.0, dot(-bestNl, wiB));
    if (nlDotLB < 1e-6) { continue; }
    let Gb = emitterGeometry(nlDotLB, dist2B, ubo.emitterDist2Floor);
    let brdfB = evalGGX(albedo, roughness, metalness, normal, wo, wiB);
    let pHatB = luminance(bestLe * brdfB * Gb);

    // Convert the BSDF solid-angle pdf to AREA measure so it shares the
    // light candidates' measure:  p_area = p_sa · dist² / cosθ_light.
    let pAreaB = pdfSa * dist2B / max(nlDotLB, 1e-6);
    let pXb = max(1e-15, pAreaB);
    let wB = select(0.0, pHatB / pXb, pHatB > 0.0);
    updateReservoirDI(&r, bestLid, bestXi, wB, &rng);
    mAreaSupport = mAreaSupport + 1u;
  }

  // --- M_ENV candidate(s): HDRI importance-sampled directional candidates (Wave 4) ──
  //
  // The M_LIGHT+M_BRDF loops handle area-emitter and BSDF-sampled candidates; both
  // contribute ZERO for HDRI-lit scenes with no mesh area lights. This loop adds one
  // importance-sampled env direction per pixel via the pre-baked sinθ-weighted CDF
  // (bindings 16-17). The env candidate is gated by envHasMap() so emitter-only
  // scenes are BYTE-IDENTICAL to the pre-Wave-4 kernel (p̂=0 → w=0 → reservoir
  // unchanged, M still incremented). The sentinel lightId (ENV_SAMPLE_SENTINEL) tells
  // downstream passes to decode xi → direction rather than indexing into emitters[].
  //
  // MEASURE CONSISTENCY: env samples live in SOLID-ANGLE measure (the env is at
  // infinity — there is no geometry term G = cosθ_light/dist²). The BRDF candidate
  // above CONVERTS to area measure to match the area-light candidates; the env
  // candidate stays in SA measure. Both representations are valid in a multi-strategy
  // RIS pool because each candidate divides p̂ by its OWN source pdf — unbiasedness
  // holds regardless of mixed measures. The shared canonical p̂ (restir_di_compute_phat_xi)
  // returns the SA-measure env p̂ (no G term) for sentinel lids, which is consistent
  // with the SA source pdf used here.
  //
  // VISIBILITY: shadow ray toward the sampled direction with tmax=1e20 (to infinity).
  // skipGlass=true (same as emitter shadow rays — glass is translucent to env light).
  for (var ei = 0u; ei < M_ENV; ei++) {
    let envS = envImportanceSample(&rng);
    if (envS.pdf <= 1e-8 || !envHasMap()) { continue; }
    let nDotL = max(0.0, dot(normal, envS.dir));
    if (nDotL < 1e-6) { continue; }
    let brdfE = evalGGX(albedo, roughness, metalness, normal, wo, envS.dir);
    // p̂ = luminance(envColor * brdf) — no G term (env is at infinity).
    let pHatE = luminance(envS.color * brdfE);
    // Source pdf: solid-angle pdf from the CDF importance sample (same measure as p̂).
    let pXe = max(1e-15, envS.pdf);
    let wE = select(0.0, pHatE / pXe, pHatE > 0.0);
    // Encode direction into xi: xi.x = theta/PI, xi.y = phi/(2PI)+0.5.
    let envXi = envDirToXi(envS.dir);
    updateReservoirDI(&r, ENV_SAMPLE_SENTINEL, envXi, wE, &rng);
    mEnvSupport = mEnvSupport + 1u;
  }

  // --- Visibility test on chosen candidate ---
  if (r.M > 0u && r.w_sum > 0.0) {
    let lid = r.lightId;
    // WS1 — offset along the GEOMETRIC normal (smooth normal can self-hit).
    let shadowOrig = pos + geoNormal * 1e-3;

    // Wave 4 — ENV_SAMPLE_SENTINEL: shadow ray to infinity along the stored dir.
    if (lid == ENV_SAMPLE_SENTINEL) {
      let envDir = envDirFromXi(r.xi);
      // SHADOW-01 — DI shadow rays honor primitive castShadow:false via the
      // bvh_material bit-0 mask (traceSceneAnyCastMask).
      let occluded = traceSceneAnyCastMask(
        ubo.bvhMode, ubo.tlasNodeCount,
        &bvh_index, &bvh_position, &bvh,
        &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
        &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
        shadowOrig, envDir, 1e20, ubo.triIntersectEpsilon, true,
        bvh_material, BVH_MATERIAL_TEX_WIDTH);
      if (occluded) {
        r.w_sum = 0.0;
        r.W     = 0.0;
      } else {
        var surf: PrimarySurface;
        surf.hit    = true;
        surf.pos    = pos;
        surf.normal = normal;
        surf.wo     = wo;
        surf.albedo = albedo;
        surf.rough  = roughness;
        surf.metal  = metalness;
        surf.depth  = hit.dist;
        let pHatZ = restir_di_compute_phat_xi(lid, r.xi, surf);
        let supportM = max(1u, mEnvSupport);
        r.M = supportM;
        r.W = select(0.0, r.w_sum / (f32(supportM) * pHatZ), pHatZ > 0.0);
      }
    } else {
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
      // skipGlass=true: matches pre-canonical ReSTIR shadow-ray glass filter
      // (light passes through glass; per-channel tinted-visibility handles tint).
      // SHADOW-01 — castShadow:false geometry is skipped via the bvh_material mask.
      let occluded = traceSceneAnyCastMask(
        ubo.bvhMode, ubo.tlasNodeCount,
        &bvh_index, &bvh_position, &bvh,
        &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
        &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
        shadowOrig, wi, dist - 2e-3, ubo.triIntersectEpsilon, true,
        bvh_material, BVH_MATERIAL_TEX_WIDTH);
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
        let pHatZ = restir_di_compute_phat_xi(lid, r.xi, surf);
        let supportM = max(1u, mAreaSupport);
        r.M = supportM;
        r.W = select(0.0, r.w_sum / (f32(supportM) * pHatZ), pHatZ > 0.0);
      }
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
  // restirPHat → common (p̂ helper + UBO/RNG). regir → lightTree adds the
  // @group(3) combined light-tree + ReGIR-grid storage buffer (one binding):
  // `sampleLightTree` (lightTreeEnabled path) + `regir_sample_cell`
  // (regirEnabled path) both read it. The composer emits
  // `common, restirPHat, lightTree, regir, ris`.
  // B3 — environmentSample adds the scene-group env bindings (15-19) + the
  // directional lookup/importance helpers; ordered after restirPHat→common so
  // WalkaroundUBO/safe_normalize/rand_f32 are in scope.
  requires: ['restirPHat', 'regir', 'environmentSample'],
};
