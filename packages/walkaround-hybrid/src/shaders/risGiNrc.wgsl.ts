/**
 * risGiNrc.wgsl.ts — NRC (Müller et al. 2021) variant of the Sprint-16
 * ReSTIR-GI RIS pass.
 *
 * Reference: Müller, Rousselle, Novák, Keller 2021, "Real-time Neural Radiance
 * Caching for Path Tracing", ACM TOG 40(4) §4 (encoding) + §5 (spread
 * termination + self-training).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * COMPILE-TIME NRC gate (nrcEnabled) — why this file exists alongside risGi.wgsl
 * ════════════════════════════════════════════════════════════════════════════
 * Live NRC is an OPT-IN feature gated by `HybridEngineOptions.nrcEnabled`. The
 * gate is resolved at PIPELINE-COMPILE time (fixed at engine creation), NOT at
 * runtime, because turning it on STRUCTURALLY changes the gi-ris pass: the ON
 * shader declares a `@group(4)` NRC bind group (MLP weights/biases, hash-grid
 * tables + level descriptors, the record-gather buffer, the encoding-config
 * uniform) so the inline MLP forward can run at the suffix vertex. The OFF
 * (default) shader is the verbatim Sprint-16 single-DDGI-estimate pass with NO
 * @group(4) and NONE of the NRC symbols.
 *
 * A runtime `ubo.nrcEnabled` flag is NOT sufficient: binding a fifth group /
 * changing the pipeline layout on the DEFAULT path alters the default pipeline
 * structure — exactly the class of change that regressed the default render to
 * an all-black frame for GRIS (f8df9a4). An opt-in feature must not change the
 * default pipeline at all, so the structure is gated at compile time:
 *   - {@link RIS_GI_MODULE}        (risGi.wgsl.ts) — OFF (default). 4 groups
 *                                   (frame/scene/ubo/hybrid), DDGI Lo estimate.
 *   - {@link buildRisGiNrcModule}  — ON. Adds @group(4) NRC + the inline MLP
 *                                   forward + spread-gated cache query + record
 *                                   gather. Composed ONLY when nrcEnabled.
 * {@link compilePipelines} composes whichever module matches the host flag and
 * builds the matching pipeline layout (4 groups OFF, 5 ON); {@link RISGIPass}
 * only calls `setBindGroup(4, …)` when ON.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The NRC suffix replacement (the only behavioural delta vs the OFF pass)
 * ════════════════════════════════════════════════════════════════════════════
 * The OFF pass computes the reconnection-vertex outgoing radiance Lo by sampling
 * the DDGI irradiance atlas (`Lo = irrAtXs · albedo · INV_PI`). The NRC pass
 * tracks Müller's path SPREAD along the single bounce (primary→suffix edge); at
 * the suffix vertex, when the spread heuristic fires (a(x) > c·a₀), it REPLACES
 * that DDGI estimate with the MLP's predicted outgoing radiance (the cache
 * query). Below the spread threshold it keeps the DDGI estimate verbatim, so a
 * scene/region where spread never exceeds c·a₀ is bit-identical to the OFF pass.
 * A self-training record (encoded input + the DDGI radiance as the target) is
 * written for the host to feed `FusedMlpTrainer.trainStep` (Müller §5
 * self-training: the cache learns the radiance the path actually carried).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { nrcEncodeHelpersWgsl } from '../neural/nrc/wgsl/nrcEncoding.wgsl.js';
import { nrcSpreadTerminationWgsl } from '../neural/nrc/wgsl/spreadTermination.wgsl.js';
import { nrcQueryWgsl, type NrcQueryWgslOptions } from '../neural/nrc/wgsl/nrcQuery.wgsl.js';

/** Config the NRC gi-ris module bakes its sizes from. Must agree with the host
 *  `NrcSubsystem` config (same hash-grid L/F, one-blob bins, MLP W/OUT/hidden). */
export interface RisGiNrcConfig extends NrcQueryWgslOptions {}

// The NRC gi-ris compute body. Identical to RIS_GI_WGSL EXCEPT:
//   • it tracks the primary-edge footprint a0 + the bounce-edge spread a(x),
//   • at the suffix vertex it cache-queries the MLP when the spread fires,
//   • it writes a self-training record per pixel.
// The @group(0..3) bindings are byte-identical to risGi.wgsl; @group(4) is the
// NRC group declared by the nrcQueryWgsl helpers prepended to this body.
export const RIS_GI_NRC_BODY = /* wgsl */ `

@group(0) @binding(10) var gi_gNormalDepth: texture_2d<f32>;
@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent: array<u32>;

@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
// WS1 (2026-05-29) — per-vertex world-space normals for the smooth shading
// normal. Byte-identical scene-group addition to risGi.wgsl (binding 11).
@group(1) @binding(11) var<storage, read> bvh_normal: array<vec4f>;

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
@group(2) @binding(2) var gi_tier: texture_2d<u32>;

@group(3) @binding(0) var ddgiIrradiance: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibility: texture_2d<f32>;
@group(3) @binding(2) var ddgiSampler:    sampler;
struct DDGIGridUBO {
  origin:    vec3f,
  spacing:   f32,
  dimsX:     u32,
  dimsY:     u32,
  dimsZ:     u32,
  _pad0:     u32,
  irrW:      f32,
  irrH:      f32,
  visW:      f32,
  visH:      f32,
};
@group(3) @binding(3) var<uniform> ddgiGrid: DDGIGridUBO;

const M_GI_BASE: u32 = 8u;
const RECONNECT_MAX_DIST: f32 = 100.0;
const NORMAL_BIAS_GI: f32 = 1e-3;

fn sampleDDGIAtPoint(worldPos: vec3f, surfaceNormal: vec3f) -> vec3f {
  return ddgiSample(
    worldPos, surfaceNormal,
    ddgiIrradiance, ddgiVisibility, ddgiSampler,
    ddgiGrid.origin.x, ddgiGrid.origin.y, ddgiGrid.origin.z,
    ddgiGrid.spacing,
    ddgiGrid.dimsX, ddgiGrid.dimsY, ddgiGrid.dimsZ,
    ddgiGrid.irrW, ddgiGrid.irrH, ddgiGrid.visW, ddgiGrid.visH,
  );
}

@compute @workgroup_size(8, 8, 1)
fn risGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdxGi = gid.y * halfDims.x + gid.x;
  let fullPx = gid.xy * 2u + 1u;

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA5A5u),
    gid.y ^ (ubo.frameSeed * 0x5A5Au),
    ubo.frameSeed ^ 0xC1A2u,
  );

  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let primaryRay = generatePrimaryRay_common(
    fullPx.x, fullPx.y, fullDims.x, fullDims.y, ubo.cameraPos, invVP,
  );
  let hit = traceSceneFirstHit(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    primaryRay, ubo.triIntersectEpsilon);
  if (!hit.didHit) {
    storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI());
    return;
  }

  let pos = primaryRay.origin + primaryRay.direction * hit.dist;
  // WS1 — smooth shading normal; geometric normal kept for the bounce offset.
  // V21 — applies in TLAS too (LOCAL blend → world by the instance inverse-transpose).
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
  // B1 — metals/glossy now get a (diffuse-target) GI reservoir; shade reflects
  // it via the GGX specular lobe. Glass still punts (refracted GI out of scope).
  // Mirrors risGi.wgsl. The Lambertian target p̂ is unchanged.
  let matColor = decodeMaterialColor(hit.matColorPacked);
  let isGlass = matColor.a > 0.3;
  if (isGlass) {
    storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI());
    return;
  }

  var r: ReservoirGI = emptyReservoirGI();
  r.xv = pos;
  r.nv = normal;

  let tier_raw = textureLoad(gi_tier, vec2i(fullPx), 0).r;
  let tier = clamp(tier_raw, 1u, 4u);
  let M_GI = M_GI_BASE * tier / 2u;

  let ppgGuidedOn = (ubo.ppgEnabled == 1u);
  let alpha = select(0.0, ubo.ppgMixAlpha, ppgGuidedOn);

  // ── NRC: primary-vertex footprint a0 (Müller §5). The primary edge is
  // camera→primary hit; its spread term uses the camera's per-pixel
  // solid-angle pdf (nrcCfg.cameraPixelPdf — host-computed each frame from
  // the projection matrix and render resolution; see NrcSubsystem.updateCameraPixelPdf)
  // and the primary-hit |cosθ|. a0 = (first-segment term)². The bounce edge's
  // spread is compared against c·a0 to decide cache termination.
  //
  // H26 camera-pdf fix: previously this used pdf=1.0 (pinhole unit-resolution
  // fallback). The Müller-correct value is the camera's per-pixel solid-angle
  // pdf = cot²(fovY/2)·W·H/4, which grows with resolution and FOV narrowing.
  // A tighter a0 (higher pdf → smaller spread term) means the bounce edges must
  // grow MORE before termination fires — correctly modelling the narrower camera
  // footprint at higher resolution or zoom.
  let cosThetaPrimary = max(1e-4, abs(dot(normal, primaryRay.direction)));
  let a0term = nrcSegmentSpreadTerm(hit.dist, nrcCfg.cameraPixelPdf, cosThetaPrimary);
  let a0 = a0term * a0term;

  for (var i: u32 = 0u; i < M_GI; i = i + 1u) {
    var wi: vec3f;
    if (alpha > 0.0) {
      let bern = rand_f32(&rng);
      if (bern < alpha) {
        wi = ppgSampleGuidedDir(pos, &rng);
      } else {
        wi = sampleCosineHemisphere(normal, &rng);
      }
    } else {
      wi = sampleCosineHemisphere(normal, &rng);
    }
    let cosTheta = max(0.0, dot(normal, wi));
    if (cosTheta < 1e-4) { continue; }

    // WS1 — offset the bounce-ray origin along the GEOMETRIC normal.
    let bounceRay = Ray(pos + geoNormal * NORMAL_BIAS_GI, wi);
    let bounceHit = traceSceneFirstHit(
      ubo.bvhMode, ubo.tlasNodeCount,
      &bvh_index, &bvh_position, &bvh,
      &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
      &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
      bounceRay, ubo.triIntersectEpsilon,
    );

    var xs:  vec3f;
    var ns:  vec3f;
    var Lo:  vec3f;

    if (bounceHit.didHit) {
      xs = bounceRay.origin + wi * bounceHit.dist;
      ns = bounceHit.normal;
      let irrAtXs = min(sampleDDGIAtPoint(xs, ns), vec3f(ubo.restirGiIrrClamp));
      let xsMat = decodeMaterialColor(bounceHit.matColorPacked);
      let ddgiLo = irrAtXs * xsMat.rgb * INV_PI;

      // ── NRC cache termination (Müller §5) ──
      // The bounce edge pos→xs accumulates spread. The candidate's source pdf
      // is the cosine-hemisphere pdf cosθ/π (the always-present component; the
      // guided dTree term only narrows it). When a(x) > c·a0 the suffix is
      // TERMINATED into the cache: the MLP prediction REPLACES the DDGI suffix
      // estimate. Below threshold the DDGI estimate is kept verbatim, so a
      // sub-threshold region is bit-identical to the OFF pass.
      //
      // H26 seeding fix: runningSum starts at 0.0 (Müller 2021 §5). The
      // previous seed of a0term was a tautology: nrcAccumulateSpread adds
      // the bounce-edge segment term to runningSum, then squares it for a(x).
      // Seeding with a0term made a(x) depend on a0term even before the first
      // bounce edge was measured — the threshold a(x) > c·a0 fired
      // immediately on the primary hit (k=0), turning every pixel into a
      // cache query. The correct seeding (0.0) means only the bounce edge's
      // spread term matters, matching the oracle in spreadTermination.ts.
      var runningSum: f32 = 0.0;
      let cosArrive = max(1e-4, abs(dot(ns, -wi)));
      let pSrcBounce = max(cosTheta * INV_PI, 1e-12);
      let aX = nrcAccumulateSpread(&runningSum, bounceHit.dist, pSrcBounce, cosArrive);
      if (nrcShouldTerminateIntoCache(aX, a0, nrcCfg.spreadC)) {
        let xsAlbedo = xsMat.rgb;
        // xsRough = 1.0 (full diffuse) for the NRC self-training feature. B1
        // added a per-tri roughness lane (bvh_material), so a real xs roughness
        // is now AVAILABLE in principle — but NRC (opt-in/experimental) does not
        // bind bvh_material on this pass, and the reconnection vertex xs is a
        // diffuse-bounce target regardless; the constant keeps the training
        // feature identical to the query, which is what matters for self-training
        // consistency. Wiring real xsRough is a follow-up scoped to the NRC track.
        let xsRough = 1.0;
        // Query the cache for outgoing radiance toward the visible point
        // (view dir at xs is −wi, the incident bounce direction reversed).
        Lo = nrcQueryRadiance(xs, ns, -wi, xsRough, xsAlbedo);
        // H27 — improved self-training target (Müller §5):
        // Replace bare DDGI irradiance (ddgiLo) with direct-sun + one-DDGI-bounce
        // combined Lo. This gives the NRC a physically grounded training signal that
        // includes direct sunlight at xs — not just diffuse irradiance from probes.
        //   directLo = (sun_contrib + DDGI_irradiance) × albedo × (1/π)
        // Shadow test: trace from xs toward sunDirection; skip if occluded.
        let sunDir = normalize(ubo.sunDirection);
        let sunNdotL = max(0.0, dot(ns, sunDir));
        var sunContrib = vec3f(0.0);
        if (sunNdotL > 1e-4) {
          let shadowOrig = xs + ns * NORMAL_BIAS_GI;
          let occluded = traceSceneAny(
            ubo.bvhMode, ubo.tlasNodeCount,
            &bvh_index, &bvh_position, &bvh,
            &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
            &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
            shadowOrig, sunDir, 1e20, ubo.triIntersectEpsilon, true,
          );
          if (!occluded) {
            sunContrib = vec3f(ubo.sunIntensity * sunNdotL);
          }
        }
        let directLo = (sunContrib + irrAtXs) * xsAlbedo * INV_PI;
        nrcWriteRecord(pixelIdxGi % nrcCfg.recordCap, xs, ns, -wi, xsRough, xsAlbedo, directLo);
      } else {
        Lo = ddgiLo;
      }
    } else {
      xs = pos + wi * RECONNECT_MAX_DIST;
      ns = -wi;
      // Wave 4 parity — directional IBL: sample the actual map along wi (same
      // as risGi.wgsl:264). envRadiance falls back to skyTint×skyIrradiance when
      // no HDRI is bound, preserving byte-identity with the scalar-sky path.
      Lo = envRadiance(wi);
    }

    let pHat = luminance(Lo) * cosTheta * INV_PI;
    if (pHat < 1e-9) { continue; }

    var w: f32;
    if (alpha > 0.0) {
      let pCos = cosTheta * INV_PI;
      let pGuide = ppgEvalPdf(pos, wi);
      let pSrc = alpha * pGuide + (1.0 - alpha) * pCos;
      w = select(0.0, pHat / pSrc, pSrc > 1e-12);
    } else {
      w = luminance(Lo);
    }
    updateReservoirGI(&r, xs, ns, Lo, w, &rng);
  }

  if (r.M > 0u && r.w_sum > 0.0) {
    let toS = r.xs - r.xv;
    let distS = length(toS);
    if (distS > 1e-4) {
      let wiZ = toS / distS;
      let shadowOrig = r.xv + r.nv * NORMAL_BIAS_GI;
      let occ = traceSceneAny(
        ubo.bvhMode, ubo.tlasNodeCount,
        &bvh_index, &bvh_position, &bvh,
        &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
        &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
        shadowOrig, wiZ, distS - 2e-3, ubo.triIntersectEpsilon, true,
      );
      if (occ) {
        r.w_sum = 0.0;
        r.W = 0.0;
      } else {
        let cosThetaZ = max(0.0, dot(r.nv, wiZ));
        let pHatZ = luminance(r.Lo) * cosThetaZ * INV_PI;
        let W_raw = select(0.0, r.w_sum / (f32(r.M) * pHatZ), pHatZ > 1e-9);
        r.W = min(W_raw, ubo.restirGiWCap);
      }
    } else {
      r.W = 0.0;
      r.w_sum = 0.0;
    }
  }

  // GRIS Phase-0 reconnection-shift cache (Lin et al. 2022 §5) — written, read
  // by no pass in Phase 0.  Shared with risGi.wgsl via refreshPhase0Cache.
  refreshPhase0Cache(&r);

  storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, r);
}
`;

/**
 * Build the NRC gi-ris module for a given encoding/MLP config. The source is:
 *   nrcEncodeHelpers (hash + one-blob + normalise)
 *   + nrcHashGridForward (NrcLevelDesc + trilinear forward)
 *   + nrcSpreadTermination (segment-spread + accumulate + predicate)
 *   + nrcQuery (@group(4) bindings + one-blob/oct/assemble/MLP-forward/record)
 *   + RIS_GI_NRC_BODY (the gi-ris compute body)
 * Its `requires` MATCH RIS_GI_MODULE exactly (walkaroundUbo / sceneTraversal /
 * reservoirGi / sharedPrimitives / materialDecode / cameraRays / ddgiSample /
 * ppgPdf) so the @group(0..3) closure is identical; the only structural delta
 * is the prepended NRC helpers + the @group(4) NRC bindings.
 *
 * Composed ONLY when nrcEnabled is compile-time true — see compilePipelines.
 */
export function buildRisGiNrcModule(cfg: RisGiNrcConfig): WgslModule {
  // nrcEncodeHelpers provides nrcSpatialHash3D / nrcNormalizeToAabb /
  // nrcOneBlobScalar (value/function-space args only — WGSL-portable). The
  // hash-grid trilinear forward is INLINED inside nrcQueryWgsl (it reads the
  // storage `nrcTables` directly; a helper taking a storage pointer is
  // WGSL-illegal without unrestricted_pointer_parameters). So we do NOT include
  // nrcHashGridForwardWgsl here.
  const source =
    nrcEncodeHelpersWgsl() +
    nrcSpreadTerminationWgsl() +
    nrcQueryWgsl(cfg) +
    RIS_GI_NRC_BODY;
  return {
    name: 'risGiNrc',
    source,
    // Wave 4 parity: `environmentSample` added so envRadiance() is in scope for
    // the GI-escape sky-miss branch (matching risGi.wgsl:264). The @group(1)
    // env bindings 15-19 are already present in the scene BGL for NRC passes.
    requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialDecode', 'cameraRays', 'ddgiSample', 'ppgPdf', 'environmentSample'],
  };
}
