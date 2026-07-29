/**
 * RIS (Resampled Importance Sampling) compute pass.
 *
 * Samples M_LIGHT=64 direct-light candidates per pixel,
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
import { reservoirDiAccessorsWgsl } from './reservoirDi.wgsl.js';

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

${reservoirDiAccessorsWgsl({ storeReadWriteBinding: 'currentReservoir' })}

// Group 1: static scene BVH + emitters
// bvh_index is array<vec4u>: .xyz=vertex indices, .w=packed RGBA8 material color+transmission
@group(1) @binding(5) var bvh_beer: texture_2d<u32>;
// WS1 (2026-05-29) — per-vertex world-space normals for the smooth shading
// normal. ris uses it for the BRDF / candidate p̂; the geometric normal is
// kept for the shadow-ray offset. Beer binding 5 feeds RGB transparent-shadow
// visibility; the scalar reservoir stores luminance(visibility).
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
// ls.pos before the reservoir winner is known; finalization replays the
// selected sample through the canonical xi-aware helper).

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

fn restirDirectVisibilityScalar(tint: vec3f) -> f32 {
  return clamp(luminance(tint), 0.0, 1.0);
}

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
  let hit = traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(
    ubo.bvhMode, ubo.tlasNodeCount,

    primaryRay, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, 0u);

  if (!hit.didHit) {
    // Sky pixel -- write sky color directly to HDR output, empty reservoir.
    // B3 — directional IBL: when a pixel-backed HDRI is bound, sample the ACTUAL
    // map along the camera ray (rotationY-aware); envRadiance falls back to the
    // scalar skyTint × skyIrradiance when no map is present (no-HDRI byte-identity).
    storeReservoirDI_rw(pixelIdx, emptyReservoirDI());
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
  let n_ok = n_isTlas && n_base + 2u < tlasWorldToLocalColumnCount();
  let n_i = select(0u, n_base, n_ok);
  let smoothNormal = smoothShadingNormal(
    hit, geoNormal,
    sceneLoadBvhNormal(hit.indices.x).xyz, sceneLoadBvhNormal(hit.indices.y).xyz, sceneLoadBvhNormal(hit.indices.z).xyz,
    n_ok,
    tlasLoadWorldToLocalColumn(n_i), tlasLoadWorldToLocalColumn(n_i + 1u), tlasLoadWorldToLocalColumn(n_i + 2u),
  );
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  let normal = applyBumpMapForHit(hit, normalMapped);
  let wo     = -primaryRay.direction;

  // Decode per-triangle material color from bvhIndex[triIdx].w (RGBA8 packed).
  let scalarMatColor = decodeMaterialColor(hit.matColorPacked);
  let matColor  = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(hit, scalarMatColor.a),
  );
  let isGlass   = materialHasTransmission(matColor.a);
  // B1 — real authored roughness/metalness from the per-tri bvh_material texture
  // (was hardcoded). The diffuse-default invariant packs 0.85 for unspecified
  // roughness / 0.05 for glass, so default-diffuse scenes are numerically
  // unchanged; authored glossy/metal surfaces now drive the GGX candidate p̂.
  let rmCoord   = vec2u(hit.indices.w % BVH_MATERIAL_TEX_WIDTH, hit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let materialWord = textureLoad(bvh_material, vec2i(rmCoord), 0).r;
  let payload = sampleRestirDIMaterialPayloadForHit(hit, smoothNormal, normal, matColor.rgb, materialWord, wo);
  let albedo    = payload.albedo;
  let roughness = payload.rough;
  let metalness = payload.metal;
  let envMapIntensity = payload.envMapIntensity;

  var surf: PrimarySurface;
  surf.hit    = true;
  surf.pos    = pos;
  surf.normal = normal;
  surf.clearcoatNormal = payload.clearcoatNormal;
  surf.wo     = wo;
  surf.albedo = albedo;
  surf.rough  = roughness;
  surf.metal  = metalness;
  surf.isGlass = isGlass;
  surf.specular = payload.specular;
  surf.anisotropy = payload.anisotropy;
  surf.anisotropyTangent = payload.anisotropyTangent;
  surf.anisotropyBitangent = payload.anisotropyBitangent;
  surf.iridescence = payload.iridescence;
  surf.clearcoat = payload.clearcoat;
  surf.sheen = payload.sheen;
  surf.sheenRoughness = payload.sheenRoughness;
  surf.layerTransmission = payload.layerTransmission;
  surf.volumeScattering = payload.volumeScattering;
  surf.bulkThickness = payload.bulkThickness;
  surf.envMapIntensity = envMapIntensity;
  surf.depth  = hit.dist;
  surf.triangleId = hit.indices.w;
  surf.instanceId = select(0u, hit.instanceIndex, ubo.bvhMode == 1u);
  surf.materialKey =
    hit.matColorPacked ^
    (materialWord * 0x9e3779b9u) ^
    (hit.indices.w * 0x85ebca6bu);

  var r = emptyReservoirDI();
  // Support-aware sample counts for mixed-measure reservoirs. Area emitters
  // use finite-emitter support; the HDRI sentinel
  // lives on a disjoint directional domain. A selected candidate's W denominator
  // must count only candidates from its support, otherwise a single env sample is
  // averaged down by all finite-emitter candidates in the pool.
  var mAreaSupport = 0u;
  var mEnvSupport = 0u;
  // Generalized stratified RIS over the tagged union of finite-emitter and
  // directional-environment domains. Each domain's candidates estimate that
  // domain's integral as a mean; multiplying source weights by Mtotal/nDomain
  // lets the canonical W = w_sum/(Mtotal*pHat) represent their SUM rather than
  // a candidate-count-weighted average. Null draws remain in nDomain.
  let envStrategyActive = envHasMap();
  let scheduledAreaM = M_LIGHT;
  let scheduledEnvM = select(0u, M_ENV, envStrategyActive);
  let scheduledTotalM = scheduledAreaM + scheduledEnvM;
  let areaRisScale = f32(scheduledTotalM) / f32(max(1u, scheduledAreaM));
  let envRisScale = f32(scheduledTotalM) / f32(max(1u, scheduledEnvM));
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
    // RIS normalisation counts scheduled proposal draws, including null /
    // zero-weight outcomes.  Counting only candidates that survive the guards
    // below conditions the estimator on acceptance and inflates it by the
    // reciprocal acceptance probability.
    mAreaSupport = mAreaSupport + 1u;
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
      lid = sampleEmitterIdx(emCount, xiEm);
      // Flat power CDF pmf is the actual CDF segment sampled above. This matters
      // for UV-varying emissive maps: the CPU CDF can use map-aware selection
      // power while EmitterTri.Le stays scalar so sampleEmitterLeAtXi can apply
      // the exact hit texel at candidate time.
      emitterSelPmf = emitterCdfPmf(emCount, lid);
    }
    let e   = sceneLoadEmitter(lid);
    let xiTri = rand2(&rng);
    let ls  = sampleEmitterPoint(e, xiTri);

    let toL   = ls.pos - pos;
    let dist2 = dot(toL, toL);
    if (dist2 < 1e-8) { continue; }
    let wi     = toL / sqrt(dist2);
    let nDotL  = max(0.0, dot(normal, wi));
    let nlDotL = max(0.0, dot(-e.normal, wi));
    if (nDotL <= 0.0 || nlDotL <= 0.0) { continue; }

    // evalGGX includes NdotL; G is the emitter geometry term only.
    // Same emitterGeometry helper as the canonical xi-aware p̂ helper
    // (restirPHat.wgsl), so the
    // per-candidate p̂ in the M_LIGHT loop matches the reservoir's
    // selection p̂ matches shade's evaluation p̂ (sweep finding Bug 3).
    let G    = emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor);
    let brdf = restir_di_eval_surface_brdf(surf, wi);
    let Le = sampleEmitterLeAtXi(e, xiTri);
    let pHat = luminance(Le * brdf * G);

    // p(x) = emitter-selection pmf × per-triangle uniform-area pdf. The first
    // factor is the EXACT pmf the chosen sampler drew from (tree or flat CDF);
    // the area pdf factor is identical for both modes. This is the source pdf
    // the WRS weight divides p̂ by — getting it exactly right is what keeps
    // ReSTIR unbiased.
    let pX = emitterSelPmf * ls.pdfArea;
    var w = 0.0;
    if (pHat > 0.0 && pX > 0.0) { w = pHat / pX; }
    if (!reservoirDiFinite(w)) { w = 0.0; }
    updateReservoirDI(&r, lid, xiTri, w * areaRisScale, &rng);
  }

  // --- M_ENV candidate(s): HDRI importance-sampled directional candidates (Wave 4) ──
  //
  // M_LIGHT handles finite emitters and contributes zero in an environment-only
  // scene. This loop adds one
  // importance-sampled env direction per pixel via the pre-baked sinθ-weighted CDF
  // (bindings 16-17). The env candidate is gated by envHasMap() so emitter-only
  // scenes are BYTE-IDENTICAL to the pre-Wave-4 kernel (p̂=0 → w=0 → reservoir
  // unchanged). The sentinel lightId (ENV_SAMPLE_SENTINEL) tells downstream
  // passes to decode xi → direction rather than indexing into emitters[].
  //
  // MEASURE CONSISTENCY: env samples live in SOLID-ANGLE measure (the env is at
  // infinity — there is no geometry term G = cosθ_light/dist²). The BRDF candidate
  // above CONVERTS to area measure to match the area-light candidates; the env
  // candidate stays in SA measure. Each candidate divides p̂ by its OWN source
  // pdf. Generalized-RIS scaling above converts each support family's sample
  // mean into one tagged-union reservoir with Mtotal; the shared canonical p̂
  // returns the SA-measure env p̂ (no G term) for sentinel lids, consistent
  // with the SA source pdf used here.
  //
  // VISIBILITY: shadow ray toward the sampled direction with tmax=1e20 (to infinity).
  // skipGlass=true (same as emitter shadow rays — glass is translucent to env light).
  for (var ei = 0u; ei < M_ENV; ei++) {
    let envS = envImportanceSample(&rng);
    // The environment strategy is inactive when no map exists.  Once active,
    // every scheduled draw counts even when its pdf/cosine/target is zero.
    if (!envStrategyActive) { continue; }
    mEnvSupport = mEnvSupport + 1u;
    if (!(envS.pdf > 0.0)) { continue; }
    let nDotL = max(0.0, dot(normal, envS.dir));
    if (nDotL <= 0.0) { continue; }
    let brdfE = restir_di_eval_surface_brdf(surf, envS.dir);
    // p̂ = luminance(envColor * brdf) — no G term (env is at infinity).
    let pHatE = luminance(envS.color * envMapIntensity * brdfE);
    // Source pdf: solid-angle pdf from the CDF importance sample (same measure as p̂).
    var wE = select(0.0, pHatE / envS.pdf, pHatE > 0.0);
    if (!reservoirDiFinite(wE)) { wE = 0.0; }
    // Encode direction into xi: xi.x = theta/PI, xi.y = phi/(2PI)+0.5.
    let envXi = envDirToXi(envS.dir);
    updateReservoirDI(&r, ENV_SAMPLE_SENTINEL, envXi, wE * envRisScale, &rng);
  }

  // Persist attempted multiplicities before visibility. All-null and occluded
  // frames are real zero-valued estimates and must retain their M so temporal
  // and spatial reuse do not condition on proposal acceptance.
  r.areaM = mAreaSupport;
  r.envM = mEnvSupport;
  r.M = mAreaSupport + mEnvSupport;

  // --- Visibility test on chosen candidate ---
  if (r.M > 0u && r.w_sum > 0.0) {
    let lid = r.lightId;
    // WS1 — offset along the GEOMETRIC normal (smooth normal can self-hit).
    let shadowOrig = pos + geoNormal * 1e-3;

    // Wave 4 — ENV_SAMPLE_SENTINEL: shadow ray to infinity along the stored dir.
    if (lid == ENV_SAMPLE_SENTINEL) {
      let envDir = envDirFromXi(r.xi);
      // SHADOW-01 / ALPHA-03 — DI shadow rays honor primitive castShadow:false
      // and attenuate through readable alpha-blend coverage plus glass Beer tint.
      // Reservoir weights are scalar, so RGB visibility is scalarized by
      // luminance; shade recomputes the same tint for the final RGB contribution.
      let shadowTint = traceSceneAlphaTintTransmittanceTexturedWithOwnership(
        ubo.bvhMode, ubo.tlasNodeCount,

        shadowOrig, envDir, 1e20, ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
        ubo.sunAngular.z >= 1.5);
      let shadowT = restirDirectVisibilityScalar(shadowTint);
      if (shadowT <= 0.0) {
        r.w_sum = 0.0;
        r.W     = 0.0;
      } else {
        let pHatZ = restir_di_compute_phat_xi(lid, r.xi, surf);
        r.w_sum = r.w_sum * shadowT;
        r.W = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 0.0);
      }
    } else {
      let e   = sceneLoadEmitter(lid);
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
      // SHADOW-01 / ALPHA-03 — castShadow:false geometry is skipped and readable
      // alpha-blend coverage and glass Beer tint attenuate visibility via the
      // material atlas. Reservoir weights store luminance(visibility); shade
      // recomputes and re-colors the selected sample.
      // Emitter castShadow:false disables the emitter's own NEE shadow ray.
      var shadowT = 1.0;
      if (e.castShadowDisabled < 0.5) {
        let shadowTint = traceSceneAlphaTintTransmittanceTexturedWithOwnership(
          ubo.bvhMode, ubo.tlasNodeCount,

          shadowOrig, wi, dist - 2e-3, ubo.triIntersectEpsilon,
          bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
          ubo.sunAngular.z >= 1.5);
        shadowT = restirDirectVisibilityScalar(shadowTint);
      }
      if (shadowT <= 0.0) {
        r.w_sum = 0.0;
        r.W     = 0.0;
      } else {
        // Build a PrimarySurface from the inline-cast values so the canonical
        // p̂ helper (Bitterli 2020 §4.3 — identical across RIS/temporal/spatial)
        // sees the same struct shape as the reuse passes.
        let pHatZ = restir_di_compute_phat_xi(lid, r.xi, surf);
        r.w_sum = r.w_sum * shadowT;
        r.W = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 0.0);
      }
    }
  }

  storeReservoirDI_rw(pixelIdx, r);
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
  // WalkaroundUBO/safe_normalize/rand_f32 are in scope. surfaceTextures supplies
  // RGB transparent-shadow visibility for DI finalization.
  requires: ['restirPHat', 'materialAtlas', 'surfaceTextures', 'regir', 'environmentSample'],
};
