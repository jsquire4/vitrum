/**
 * Sprint 16 — ReSTIR-GI initial-candidate RIS pass.
 *
 * Reference: Majercik et al. 2021, "Dynamic Diffuse Global Illumination
 * Resampling," SIGGRAPH 2021, §4.2 (initial-sample RIS).
 *
 * Per-pixel:
 *   1. Re-cast primary ray; on miss / glass / metal → empty reservoir.
 *   2. RIS over M_GI = 8 candidates. Each candidate samples a
 *      cosine-weighted hemisphere direction; the reconnection vertex
 *      is the first BVH hit along that direction (or sky).
 *      Outgoing radiance Lo at the reconnection vertex is computed by
 *      sampling the DDGI irradiance atlas, multiplied by the hit
 *      surface's albedo / π (Lambertian re-radiation).
 *   3. p̂ = luminance(Lo) × cos(N_visible, wi) × INV_PI
 *      pdf_source = cos(N_visible, wi) / π (cosine hemisphere)
 *      w_i = p̂ / pdf_source = luminance(Lo) × INV_PI² (cancels cos)
 *   4. Visibility test on the chosen sample (one extra BVH ray).
 *   5. W = w_sum / (M · p̂(z)) per the standard RIS estimator
 *      (Talbot 2005 + ReSTIR DI 2020).
 *
 * Half-resolution: dispatches W/2 × H/2 invocations. The visible point
 * is the center of each 2×2 quad in full-res coords. The shade pass
 * reconstructs full-res indirect via reservoir read at gid.xy / 2.
 *
 * Bindings:
 *   group(0) — frame (same as shade; uses gNormalDepth + reservoir)
 *   group(1) — scene (BVH + emitters; reuse existing layout)
 *   group(2) — ubo (camera matrices, frameSeed, aoFullTexture)
 *   group(3) — hybrid (DDGI atlas + sampler + grid params)
 *   The GI reservoir buffer is bound as @group(0) @binding(11), added
 *   to the frame BGL by the Sprint 16 pipeline machinery.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RIS_GI_WGSL = /* wgsl */ `

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

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
// Sprint 9 — adaptive sampling tier (r32uint, full-res). 1 = low variance,
// 2 = medium, 4 = high. Read at the centre of each half-res 2×2 quad to
// scale the RIS candidate count: high-variance pixels get more candidates
// where they're needed, low-variance pixels save the compute.
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

// Base RIS-GI candidate count. Scaled per pixel by adaptive-sampling tier:
// tier=1 → M_GI_eff = 4; tier=2 → 8 (default); tier=4 → 16.
const M_GI_BASE: u32 = 8u;
const RECONNECT_MAX_DIST: f32 = 100.0;
const NORMAL_BIAS_GI: f32 = 1e-3;

// sampleCosineHemisphere is the canonical helper from @vitrum/shared-samplers'
// bsdfPrimitives.wgsl, injected into the composed shade module via composeWgsl
// (SHARED_PRIMITIVES_MODULE → BSDF_PRIMITIVES_WGSL).

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

  // Sample point in full-res: centre of the 2×2 quad.
  let fullPx = gid.xy * 2u + 1u;

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA5A5u),
    gid.y ^ (ubo.frameSeed * 0x5A5Au),
    ubo.frameSeed ^ 0xC1A2u,
  );

  // Re-cast primary ray to find the visible surface.
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
  let normal = hit.normal;
  // Skip glass / metal — indirect for those goes through the
  // path-traced fork, not DDGI atlas sampling. ReSTIR-DI Lo_direct stays.
  let matColor = decodeMaterialColor(hit.matColorPacked);
  let isGlass = matColor.a > 0.3;
  let isMetal = decodeIsMetal(hit.matColorPacked);
  if (isGlass || isMetal) {
    storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI());
    return;
  }

  var r: ReservoirGI = emptyReservoirGI();
  r.xv = pos;
  r.nv = normal;

  // Adaptive-sampling tier read at the full-res quad centre. Clamped to
  // [1,4] in case the sample-budget pass emits a bad/uninitialised value
  // (first frame writes vec4u(2,0,0,0) by default). M_GI scales linearly.
  let tier_raw = textureLoad(gi_tier, vec2i(fullPx), 0).r;
  let tier = clamp(tier_raw, 1u, 4u);
  let M_GI = M_GI_BASE * tier / 2u;

  for (var i: u32 = 0u; i < M_GI; i = i + 1u) {
    // Cosine-weighted hemisphere candidate.
    let wi = sampleCosineHemisphere(normal, &rng);
    let cosTheta = max(0.0, dot(normal, wi));
    if (cosTheta < 1e-4) { continue; }

    // Trace from the visible point along wi. Reconnection vertex is the
    // first BVH hit (or sky-miss at RECONNECT_MAX_DIST).
    let bounceRay = Ray(pos + normal * NORMAL_BIAS_GI, wi);
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
      // Sample DDGI atlas at the reconnection vertex along its normal —
      // gives the incoming irradiance there. Modulate by the hit surface's
      // albedo / π for Lambertian outgoing radiance toward the visible pt.
      //
      // Defensive cap on the atlas read.  DDGI probes within ~1 spacing of
      // an area light catch Le directly during the probe trace, so atlas
      // reads of 5..8 are possible — but for a Cornell-scale scene with
      // Le=12, legitimate near-light wall irradiance is also in this band,
      // so we can't reject those samples without truncating real indirect.
      // The cap (Cornell-tuned default 5.0, exposed via ubo.restirGiIrrClamp)
      // admits the realistic indirect range while bounding pathological DDGI
      // atlas readings (which would otherwise produce ~10× per-channel spikes
      // in Lo). The previous tighter reject+cap (>2.0 reject, min 1.0) was
      // over-truncating: the magnitude audit showed it was a 5-10× *under*-
      // energizer of the indirect channel.
      let irrAtXs = min(sampleDDGIAtPoint(xs, ns), vec3f(ubo.restirGiIrrClamp));
      let xsMat = decodeMaterialColor(bounceHit.matColorPacked);
      Lo = irrAtXs * xsMat.rgb * INV_PI;
    } else {
      // Sky miss — sample the engine's sky as a direct contribution.
      xs = pos + wi * RECONNECT_MAX_DIST;
      ns = -wi;
      Lo = ubo.skyTint * ubo.skyIrradiance;
    }

    // p̂ at the visible point for this candidate.
    let pHat = luminance(Lo) * cosTheta * INV_PI;
    if (pHat < 1e-9) { continue; }
    // pdf_source = cosTheta / π (cosine hemisphere). w = p̂ / pdf:
    // = luminance(Lo) × cosTheta × INV_PI / (cosTheta × INV_PI)
    // = luminance(Lo)
    let w = luminance(Lo);
    updateReservoirGI(&r, xs, ns, Lo, w, &rng);
  }

  // Final visibility test on the chosen sample.
  if (r.M > 0u && r.w_sum > 0.0) {
    let toS = r.xs - r.xv;
    let distS = length(toS);
    if (distS > 1e-4) {
      let wiZ = toS / distS;
      let shadowOrig = r.xv + r.nv * NORMAL_BIAS_GI;
      // skipGlass=true: matches pre-canonical ReSTIR shadow-ray glass filter
      // (light passes through glass; per-channel tinted-visibility handles tint).
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
        // Cap W to bound firefly contribution from tiny pHat denominators
        // (grazing cos or near-zero Lo luminance). Cornell default 16.0
        // lives on ubo.restirGiWCap; see common.wgsl for the rationale.
        r.W = min(W_raw, ubo.restirGiWCap);
      }
    } else {
      r.W = 0.0;
      r.w_sum = 0.0;
    }
  }

  storeReservoirGI_rw(&reservoirGiCurrent, pixelIdxGi, r);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  T9-stepC — narrowed from `['common', 'ddgiSample']` to the modules this
 *  half-res GI-RIS pass actually references:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - primary cast (`traceScene*` / `BVHNode` / `Ray`) → sceneTraversal
 *    - `ReservoirGI` / `emptyReservoirGI` / `updateReservoirGI` /
 *      `storeReservoirGI_rw`                 → reservoirGi
 *    - `pcgInit` / `luminance` / `sampleCosineHemisphere` → sharedPrimitives
 *    - `decodeMaterialColor` / `decodeIsMetal` → materialDecode
 *    - `invertMat4_common` / `generatePrimaryRay_common` → cameraRays
 *    - `ddgiSample`                          → ddgiSample
 *  Drops emitterSampling / ggxBrdf / jacobianShift / welfordTail (unused).
 *  Verified complete by the static ident-resolution gate. */
export const RIS_GI_MODULE: WgslModule = {
  name: 'risGi',
  source: RIS_GI_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialDecode', 'cameraRays', 'ddgiSample'],
};
