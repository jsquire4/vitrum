/**
 * restirPtProducer.wgsl.ts — the ReSTIR-PT reconnection-sample PRODUCER pass.
 *
 * A SEPARATE `@compute` entry point (`restirPtProduce`) — it does NOT modify the
 * megakernel (`kernel.wgsl.ts`). It traces, per full-res pixel:
 *   1. the PRIMARY ray → the visible vertex xv (path prefix; prefix length 1),
 *   2. ONE BSDF-sampled bounce off xv → the reconnection vertex xs,
 *   3. the SUFFIX radiance Lo leaving xs back toward xv,
 * then seeds a 1-sample RIS reservoir, finalises W, and stores it.
 *
 * This is the hero-stack analogue of walkaround-hybrid's risGi producer
 * (`risGi.wgsl.ts` → `updateReservoirGI` + `finaliseGIReservoirW` +
 * `refreshPhase0Cache`); the reservoir ADT + target + finalize are mirrored from
 * `reservoirPtHero.wgsl.ts` (itself a port of `reservoirGi.wgsl.ts`).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The EXACT Lo definition (energy-consistency — read this)
 * ════════════════════════════════════════════════════════════════════════════
 * Lo is the OUTGOING RADIANCE LEAVING xs back toward xv — i.e. the radiance the
 * suffix path (everything from xs onward) carries along the direction xs → xv,
 * measured with the SUFFIX throughput initialised to 1 at xs (it does NOT carry
 * the prefix throughput, and it does NOT include the reconnection-edge BRDF/cos
 * at xv). Concretely Lo accumulates, with a suffix throughput that starts at 1:
 *   • the EMISSION of xs as seen from xv (xs's own emissive, if any), PLUS
 *   • NEE at xs (direct lighting: the analytic directional/point/env connection
 *     with the visible-vertex-independent BRDF at xs), PLUS
 *   • onward INDIRECT bounces from xs (a short BSDF-sampled walk with RR),
 *     including emissive-on-hit / env on those further vertices.
 *
 * WHY this exact split: the RESOLVE pass reconstructs the full path contribution
 * as  prefixThroughput · f_bsdf(xv; wo→wi_recon) · cos(nv, wi_recon) · Lo · W.
 * If Lo folded in the prefix f·cos (or the prefix throughput), resolve would
 * DOUBLE-apply it. Keeping Lo = "radiance leaving xs, suffix-throughput-1" makes
 * the reconnection split clean and matches Lin 2022's cached L_o at x_s.
 *
 * The producer's 1-sample RIS candidate weight is  w = p̂ / p_src  where
 *   p̂    = restirPtTargetAt(xv, nv, woV, mat, xs, Lo) = luminance(f_bsdf·cos·Lo)  (integrand-matching, B3)
 *          (the diffuse-cosine resampling proxy — a SCALAR heuristic), and
 *   p_src = the REAL directional pdf that GENERATED wi_recon at xv
 *          (brdfDirectionalPdf at the visible vertex).
 * After finalize W = w_sum/p̂ = 1/p_src, and resolve forms f·cos·Lo/p_src — the
 * unbiased single-bounce estimator. p̂ cancels (see reservoirPtHero.wgsl.ts
 * unbiasedness note); storing the REAL p_src is what makes a GLOSSY xv unbiased.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * ENERGY-CONSISTENCY RISKS the producer guards (and the ones it does NOT)
 * ════════════════════════════════════════════════════════════════════════════
 * GUARDED:
 *   • SPECULAR / TRANSMISSIVE visible vertex → write an EMPTY reservoir (M = 0).
 *     A near-singular prefix BSDF cannot be reused via the geometric reconnection
 *     shift (reusing a neighbour's Lo through a different wi_recon is invalid),
 *     and the diffuse-cosine target is meaningless there. Matches the GI producer
 *     writing empty for glass/metal primaries. The pixel simply does not reuse.
 *   • PRIMARY MISS / degenerate reconnection edge (xv≈xs, cos≤0) → EMPTY.
 *   • p_src ≤ 0 (the sampled direction has zero forward density) → EMPTY.
 *   • RECONNECTION-RAY ESCAPE (the bounce ray leaves the scene) → NOT empty: a FAR
 *     synthetic reconnection vertex along the escape direction with Lo = the
 *     environment radiance, mirroring the proven GI producer (risGi.wgsl.ts:252-
 *     257). Writing empty here would zero xv and the temporal pass would lose the
 *     pixel's entire history (the open-Cornell-face ~22%-of-lit-pixels dropout) —
 *     the env-escape indirect IS part of this pixel's reuse, not solely the
 *     megakernel's job. See the reconnection-trace block below.
 *
 * HONEST UNGUARDED bias (documented regime limits, NOT bugs):
 *   • MODERATELY-GLOSSY xv: the reconnection shift is geometric (holds xs fixed,
 *     re-roots the edge) but the glossy BRDF at xv is direction-sensitive; the
 *     diffuse-cosine proxy target under-weights it, so cross-pixel/temporal reuse
 *     of such a reservoir is APPROXIMATE. The PRODUCER itself is unbiased for
 *     glossy xv (it stores the real p_src); the bias is introduced by the REUSE
 *     pass (temporal feedback can drift on glossy surfaces). Prefix-1 reconnection
 *     reuse is physically exact only for a DIFFUSE visible vertex.
 *   • Lo SUFFIX TRUNCATION: the suffix walk is bounce-limited + RR-terminated, so
 *     Lo is a finite-bounce estimate of the true outgoing radiance — the SAME
 *     truncation the megakernel applies, but the reconnection-vertex suffix
 *     budget here is independent of the megakernel's bounce budget (the wiring
 *     step may want them aligned for A/B parity against the unidirectional path).
 *   • Lo is a SINGLE stochastic estimate per produced reservoir (one suffix
 *     sample). Across frames the temporal reuse averages many such Lo, but a
 *     fixed Lo per reservoir means the reconnection sample's suffix is NOT
 *     re-estimated under reuse — standard for ReSTIR-PT reconnection (the suffix
 *     is the cached, shift-invariant tail).
 *
 * ── Bind groups ─────────────────────────────────────────────────────────────
 * This pass composes the SHARED pt-webgpu modules (for traceClosest / evaluateBrdf
 * / NEE helpers), which already own @group(0..3). The ReSTIR-PT reservoir output
 * is declared in @group(4) to avoid any collision with the inherited groups; the
 * WIRING step builds a pipeline layout that includes @group(4) and (because the
 * producer statically uses tracing + NEE) groups 0/1/2. (maxBindGroups ≥ 5 — see
 * the compose-module note; the wiring agent may relocate this group if a target
 * adapter caps maxBindGroups at 4.)
 */

export const RESTIR_PT_PRODUCER_WGSL = /* wgsl */ `
// ReSTIR-PT producer output — the per-pixel reconnection reservoir (strided u32).
@group(4) @binding(0) var<storage, read_write> rpt_reservoirOut: array<u32>;
@group(4) @binding(4) var<uniform> rptParams: RestirPtParams;

// Visible-vertex classification: a sufficiently-diffuse primary hit is reusable;
// a specular / transmissive one is not (see the producer header). Mirrors the GI
// producer's "write empty for glass/metal" gate.
fn rptIsReusableVisibleVertex(roughness: f32, metallic: f32, transmission: f32) -> bool {
  // Transmissive → never (the reconnection edge is a refraction, not a reusable
  // diffuse/glossy bounce). Near-mirror metal → never. Otherwise reusable, with
  // the documented glossy-reuse approximation for the moderate-roughness middle.
  if (transmission > 0.01) { return false; }
  if (metallic > 0.5 && roughness < 0.08) { return false; }
  return roughness >= 0.08;
}

// Direct-lighting NEE at the RECONNECTION vertex xs (visible-vertex independent).
// Adds the rect-/disc-area + directional + point + environment connections with
// the SAME analytic estimators the megakernel uses, at the suffix throughput
// passed in (the suffix starts at throughput 1 at xs). The rect-area branch is
// ENERGY-CRITICAL (it is the engine's primary area-light NEE — see its inline
// note); the delta + env branches are full-weight, the area branch is the
// un-MIS-weighted area-measure connection (the producer has no BSDF-side
// emissive-on-hit to complement an MIS split — see the rect-area branch). Uses a
// per-light loop (no light-tree selection) to keep the producer's group footprint
// minimal and the Lo estimate well-defined. Returns the radiance leaving xs.
fn rptDirectAtVertex(
  rng: ptr<function, u32>,
  pos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  suffixThroughput: vec3f,
) -> vec3f {
  var contrib = vec3f(0.0);
  // Directional light (delta): full weight, no MIS.
  if (params.lightDir.w > 1e-6) {
    let lightDir = safe_normalize(params.lightDir.xyz);
    let nDotL = max(0.0, dot(normal, lightDir));
    if (nDotL > 0.0) {
      let shadowRay = Ray(pos + normal * 1e-3, lightDir);
      if (!traceAny(shadowRay, 1e-4, INFINITY)) {
        let brdf = evaluateBrdfFull(
          baseColor, roughness, metallic, normal, wo, lightDir,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          0.0, 0.0,
        );
        contrib = contrib + suffixThroughput * brdf * nDotL * params.lightDir.w;
      }
    }
  }
  // Rect/disc area lights: the dominant area-light NEE the suffix Lo needs.
  // Shape discriminator in emission.w: ≈ 0 → rect, ≈ 1 → analytic disc.
  // FULL WEIGHT (no MIS) — see inline comment in kernel.wgsl.ts rect loop for rationale.
  // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
  // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
  for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
    let rb = ri * 4u;
    let rpos = rectAreaLights[rb].xyz;
    let ru = rectAreaLights[rb + 1u].xyz;
    let rv = rectAreaLights[rb + 2u].xyz;
    let rshapeR = rectAreaLights[rb + 3u];
    let rr = rshapeR.rgb;
    let isDiscR = abs(rshapeR.w - 1.0) < 0.5;
    let xi1r = rand_f32(rng);
    let xi2r = rand_f32(rng);
    var lpos: vec3f;
    var area: f32;
    if (isDiscR) {
      let rrad = length(ru);
      let a = xi1r * 2.0 - 1.0;
      let b = xi2r * 2.0 - 1.0;
      var cr: f32; var cphi: f32;
      if (abs(a) >= abs(b)) {
        cr = a; cphi = (PI / 4.0) * (b / max(abs(a), 1e-9));
      } else {
        cr = b; cphi = (PI / 2.0) - (PI / 4.0) * (a / max(abs(b), 1e-9));
      }
      lpos = rpos + ru * (cr * cos(cphi)) + rv * (cr * sin(cphi));
      area = max(PI * rrad * rrad, 1e-6);
    } else {
      lpos = rpos + ru * (xi1r * 2.0 - 1.0) + rv * (xi2r * 2.0 - 1.0);
      area = max(4.0 * length(cross(ru, rv)), 1e-6);
    }
    let toLight = lpos - pos;
    let dist2 = max(dot(toLight, toLight), 1e-6);
    let dist = sqrt(dist2);
    let wi = toLight / dist;
    let nDotL = max(dot(normal, wi), 0.0);
    if (nDotL > 0.0) {
      let lightNormal = safe_normalize(cross(ru, rv));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight > 0.0) {
        let lightPdf = dist2 / max(cosLight * area, 1e-6);
        let shadowRay = Ray(pos + normal * 1e-3, wi);
        if (!traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
          let brdf = evaluateBrdfFull(
            baseColor, roughness, metallic, normal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            0.0, 0.0,
          );
          contrib = contrib + suffixThroughput * brdf * nDotL * rr / max(lightPdf, 1e-6);
        }
      }
    }
  }
  // Point lights (delta): full weight, no MIS.
  // H51-D: stride 3 (3 vec4 = 12 f32): position, radiance, [distance, decay, 0, 0]
  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    let base = pi * 3u;
    let lp = pointLights[base].xyz;
    let rad = pointLights[base + 1u].rgb;
    let ptExtra = pointLights[base + 2u];
    let ptMaxDist = ptExtra.x;
    let ptDecay   = ptExtra.y;
    let toPoint = lp - pos;
    let dist2 = max(dot(toPoint, toPoint), 1e-5);
    let dist = sqrt(dist2);
    if (ptMaxDist > 0.0 && dist > ptMaxDist) { continue; }
    let wi = toPoint / dist;
    let nDotL = max(0.0, dot(normal, wi));
    if (nDotL > 0.0) {
      let shadowRay = Ray(pos + normal * 1e-3, wi);
      if (!traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
        let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -ptDecay), ptDecay > 0.01);
        let brdf = evaluateBrdfFull(
          baseColor, roughness, metallic, normal, wo, wi,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          0.0, 0.0,
        );
        contrib = contrib + suffixThroughput * brdf * nDotL * rad * attenuation;
      }
    }
  }
  // Spot lights (delta): full weight, no MIS.
  // H14-B + H51-D: mirrors kernel NEE; stride 4 (4 vec4 = 16 f32).
  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    let sb = si * 4u;
    let spos = spotLights[sb].xyz;
    let saxis = spotLights[sb + 1u];
    let sradW = spotLights[sb + 2u];
    let spExtra = spotLights[sb + 3u];
    let spotDir = safe_normalize(saxis.xyz);
    let cosOuter = saxis.w;
    let cosInner = sradW.w;
    let srad = sradW.rgb;
    let spMaxDist = spExtra.x;
    let spDecay   = spExtra.y;
    let toSpot = spos - pos;
    let dist2 = max(dot(toSpot, toSpot), 1e-5);
    let dist = sqrt(dist2);
    if (spMaxDist > 0.0 && dist > spMaxDist) { continue; }
    let wi = toSpot / dist;
    let coneCos = dot(-wi, spotDir);
    if (coneCos >= cosOuter) {
      let nDotL = max(0.0, dot(normal, wi));
      if (nDotL > 0.0) {
        let shadowRay = Ray(pos + normal * 1e-3, wi);
        if (!traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
          let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
          let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -spDecay), spDecay > 0.01);
          let brdf = evaluateBrdfFull(
            baseColor, roughness, metallic, normal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            0.0, 0.0,
          );
          contrib = contrib + suffixThroughput * brdf * nDotL * softness * srad * attenuation;
        }
      }
    }
  }
  // Environment (importance-sampled if a map is present), MIS vs the BRDF pdf.
  if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {
    var envDir = vec3f(0.0, 1.0, 0.0);
    var envColor = vec3f(0.0);
    var envPdf = 0.0;
    let envSample = sampleEnvironmentImportance(rng);
    if (envSample.pdf > 0.0) {
      envDir = envSample.wi;
      envColor = envSample.value;
      envPdf = envSample.pdf;
    } else {
      let diffSample = cosineHemisphereSample(rng, normal);
      envDir = diffSample.wi;
      envColor = sampleEnvironmentColor(envDir);
      envPdf = max(environmentPdf(envDir), 1e-8);
    }
    let nDotL = max(dot(normal, envDir), 0.0);
    if (nDotL > 1e-6) {
      let shadowRay = Ray(pos + normal * 1e-3, envDir);
      if (!traceAny(shadowRay, 1e-4, INFINITY)) {
        let brdf = evaluateBrdfFull(
          baseColor, roughness, metallic, normal, wo, envDir,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          0.0, 0.0,
        );
        let brdfPdf = brdfDirectionalPdfFull(
          baseColor, roughness, metallic, 0.0, 1.0, normal, wo, envDir,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          0.0, 0.0,
        );
        let misWeight = powerHeuristic(envPdf, brdfPdf);
        contrib = contrib + suffixThroughput * brdf * nDotL * envColor * misWeight / max(envPdf, 1e-8);
      }
    }
  }
  return contrib;
}

// The SUFFIX radiance Lo leaving the reconnection vertex xs toward xv. A short
// BSDF-sampled walk ROOTED AT xs with suffix throughput initialised to 1 (the
// prefix throughput / reconnection-edge BRDF at xv are applied at RESOLVE, NOT
// here — see the producer header). Accumulates xs's own emission (as seen from
// xv) + NEE at each suffix vertex + emissive/env on onward hits, exactly as the
// megakernel does for the corresponding suffix segment, but rooted at xs with
// throughput 1. The reconnection vertex's material is passed in DIRECTLY (the
// caller already has the SceneHit at xs) so b==0 needs no re-trace; onward
// bounces decode the new hit's material in the loop.
//   - xsMatId / nsShade: the reconnection vertex's material id + front-face
//     shading normal (toward xv).
//   - reconDir: the direction xs → xv (wo at xs; the outgoing direction Lo is on).
fn rptComputeLoAtReconnection(
  rng: ptr<function, u32>,
  xs: vec3f,
  xsMatId: u32,
  nsShade: vec3f,
  reconDir: vec3f,
  suffixBounces: u32,
) -> vec3f {
  var Lo = vec3f(0.0);
  var suffixThroughput = vec3f(1.0);
  var wo = reconDir;          // wo at the current suffix vertex (toward xv at b==0)
  var pos = xs;
  var normal = nsShade;       // front-face shading normal at the current vertex
  var matId = xsMatId;
  // xs's emission seen from xv is a directly-viewed term (no prior diffuse bounce
  // MIS-accounts for it), so the first iteration adds it ungated.
  var prevAllowsAreaMis = false;

  for (var b = 0u; b < suffixBounces; b = b + 1u) {
    let mat = decodeMaterial(matId);
    let baseColor = mat.baseColor;
    let roughness = max(mat.roughness, 0.02);
    let metallic = mat.metallic;
    let emissive = mat.emissive;

    // Emission of this suffix vertex along wo (toward xv at b==0). Gated so an
    // onward diffuse bounce's NEE does not double-count the next hit's emission.
    if (!prevAllowsAreaMis) {
      Lo = Lo + suffixThroughput * emissive;
    }
    // Direct lighting (NEE) at this suffix vertex.
    Lo = Lo + rptDirectAtVertex(
      rng, pos, normal, wo, baseColor, roughness, metallic,
      mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
      mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
      suffixThroughput,
    );

    // Sample the next onward direction with the cosine (diffuse) lobe — the robust
    // default that keeps Lo well-defined for any onward surface; the reconnection
    // vertex was validated diffuse-ish by the reusable-visible-vertex gate, and the
    // onward indirect is a second-order term whose exact lobe choice changes
    // variance, not the mean.
    let cosSample = cosineHemisphereSample(rng, normal);
    let nextDir = cosSample.wi;
    let nDotNext = max(dot(normal, nextDir), 0.0);
    if (nDotNext <= 1e-5) { break; }
    // Onward-bounce throughput = the EXACT cosine-sampling MC estimator
    //   f·cos / pdf   with pdf = cos·INV_PI   (cosSample.pdf),
    // evaluated with the FULL BRDF (evaluateBrdf, whose diffuse kd uses the
    // HALF-VECTOR Fresnel). This is critical at GRAZING wo: the prior throughput
    // "(1 - fresnelSchlick(dot(n,wo)))*baseColor" applied the VIEW-ANGLE Fresnel,
    // which -> 1 as wo grazes, collapsing the diffuse transport toward 0 (a ~10%/
    // bounce energy loss that COMPOUNDED with suffix depth — the ReSTIR-PT reuse
    // ~15% deficit; the deeper a suffix bounce, the more oblique its wo). The
    // physically-correct Lambertian cosine-sample throughput has NO view-angle
    // collapse (f·cos/pdf = albedo for an ideal diffuse lobe). Verified exact
    // (ratio 1.000 ∀ wo angle) vs dense-quadrature in wsl-gpu/scripts/
    // restir-pt-onward-jsmodel.ts. evaluateBrdf also folds the small onward
    // specular response in, matching the megakernel's onward transport.
    let fOnward = evaluateBrdfFull(
      baseColor, roughness, metallic, normal, wo, nextDir,
      mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
      mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
      0.0, 0.0,
    );
    suffixThroughput = suffixThroughput * fOnward * nDotNext / max(cosSample.pdf, 1e-8);
    prevAllowsAreaMis = true; // diffuse onward bounce: next emission handled by NEE/MIS.

    let nextHit = traceClosest(Ray(pos + normal * 1e-3, nextDir), 1e-4, INFINITY);
    if (!nextHit.didHit) {
      Lo = Lo + suffixThroughput * sampleEnvironmentColor(nextDir);
      break;
    }
    pos = (pos + normal * 1e-3) + nextDir * nextHit.dist;
    matId = hitMaterialId(nextHit);
    let nextFront = dot(nextHit.normal, nextDir) < 0.0;
    normal = select(-nextHit.normal, nextHit.normal, nextFront);
    wo = -nextDir;

    // Russian roulette on the suffix throughput.
    if (b >= 1u) {
      let surv = clamp(max(suffixThroughput.r, max(suffixThroughput.g, suffixThroughput.b)), 0.05, 0.95);
      if (rand_f32(rng) > surv) { break; }
      suffixThroughput = suffixThroughput / surv;
    }
  }
  return Lo;
}

@compute @workgroup_size(8, 8, 1)
fn restirPtProduce(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let pixelIdx = gid.y * params.width + gid.x;

  var rng = pcgInit(gid.x, gid.y, params.frameSeed ^ params.frameIndex);
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  let primaryRay = generatePrimaryRay(gid.x, gid.y, jitter);

  // ── 1. Primary ray → visible vertex xv (the path prefix) ──
  let vHit = traceClosest(primaryRay, 1e-4, INFINITY);
  if (!vHit.didHit) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }
  let vMatId = hitMaterialId(vHit);
  let vMat = decodeMaterial(vMatId);
  let xv = primaryRay.origin + primaryRay.direction * vHit.dist;
  let vIsFront = dot(vHit.normal, primaryRay.direction) < 0.0;
  let nv = select(-vHit.normal, vHit.normal, vIsFront);
  let woV = -primaryRay.direction; // eye-side direction at xv
  let baseColorV = vMat.baseColor;
  let roughnessV = max(vMat.roughness, 0.02);
  let metallicV = vMat.metallic;
  let transmissionV = vMat.transmission;
  let clearcoatV = vMat.clearcoat;
  let clearcoatRoughnessV = vMat.clearcoatRoughness;
  let sheenV = vMat.sheen;
  let sheenRoughnessV = vMat.sheenRoughness;
  let sheenColorV = vMat.sheenColor;
  let iridescenceV = vMat.iridescence;
  let iridescenceIorV = vMat.iridescenceIor;
  let iridescenceThicknessMinV = vMat.iridescenceThicknessMin;
  let iridescenceThicknessMaxV = vMat.iridescenceThicknessMax;
  let anisotropyV = materialAnisotropy(vMatId, vHit.triIndex, vHit.baryVW);
  let anisotropyRotationV = materialAnisotropyRotation(vMatId, vHit.triIndex, vHit.baryVW);

  // Specular / transmissive visible vertex → not reusable; write empty.
  if (vMat.isUnlit || !rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV)) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }

  // ── 2. One BSDF-sampled bounce off xv → reconnection vertex xs ──
  // Sample the reconnection direction wi_recon from the visible-vertex BSDF.
  // The hero stack stores the REAL source pdf so a glossy xv stays unbiased.
  var tanT: vec3f;
  var tanB: vec3f;
  buildOnb(nv, &tanT, &tanB);
  if (anisotropyV > 1e-4) {
    let c = cos(anisotropyRotationV);
    let s = sin(anisotropyRotationV);
    let rotatedT = c * tanT + s * tanB;
    let rotatedB = -s * tanT + c * tanB;
    tanT = rotatedT;
    tanB = rotatedB;
  }
  let cosO = max(dot(nv, woV), 0.0);
  let f0V = mix(vec3f(0.04), baseColorV, metallicV);
  let fresV = fresnelSchlick(cosO, f0V);
  // Partition specular vs diffuse exactly as sampleNextBounceDirection's
  // non-transmissive branch, so wi_recon's pdf matches brdfDirectionalPdf.
  let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresV), metallicV)), 0.04, 0.96);
  let baseDiffProb = max(0.0, 1.0 - baseSpecProb);
  let sumProb = max(baseSpecProb + baseDiffProb, 1e-4);
  let specProb = baseSpecProb / sumProb;
  var wiRecon = vec3f(0.0);
  let xiLobe = rand_f32(&rng);
  if (xiLobe < specProb) {
    var bs: BsdfSample;
    if (anisotropyV > 1e-4) {
      bs = glossyReflectionSampleAnisotropic(&rng, woV, nv, tanT, tanB, roughnessV, anisotropyV);
    } else {
      bs = glossyReflectionSample(&rng, woV, nv, tanT, tanB, roughnessV);
    }
    wiRecon = bs.wi;
  } else {
    let bs = cosineHemisphereSample(&rng, nv);
    wiRecon = bs.wi;
  }
  let nDotRecon = dot(nv, wiRecon);
  if (nDotRecon <= 1e-5) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }
  // The REAL source directional pdf that generated wi_recon (unbiasedness-
  // critical — resolve forms f·cos·Lo/p_src; see the header). A degenerate
  // single sample (p_src ≈ 0, e.g. a near-mirror VNDF lobe sampled back-facing)
  // writes empty: a cosine "retry" is NOT taken because it would make p_src
  // conditional on the glossy sample having failed (a direction-dependent density
  // the mixture brdfDirectionalPdf does not capture) — that mismatch would BIAS
  // the f·cos·Lo/p_src estimator. p_src ≤ 0 is rare (and its f·cos contribution is
  // ~0 anyway), so dropping the single frame's sample is the correct, unbiased
  // choice; the temporal history is re-seeded the next non-degenerate frame.
  let pdfSrc = brdfDirectionalPdfFull(
    baseColorV, roughnessV, metallicV, 0.0, vMat.ior, nv, woV, wiRecon,
    0.0, clearcoatRoughnessV, 0.0, sheenRoughnessV,
    iridescenceV, iridescenceIorV, iridescenceThicknessMinV, iridescenceThicknessMaxV,
    anisotropyV, anisotropyRotationV,
  );
  if (pdfSrc <= 1e-8) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }

  let reconRay = Ray(xv + nv * 1e-3, wiRecon);
  let sHit = traceClosest(reconRay, 1e-4, INFINITY);
  var xs: vec3f;
  var ns: vec3f;
  var Lo: vec3f;
  if (!sHit.didHit) {
    // The reconnection ray ESCAPED to the environment. DO NOT write empty — that
    // would (a) drop this pixel's indirect entirely and (b) zero xv so the
    // temporal pass cannot reproject and the pixel loses ALL its history, leaving
    // it permanently black (the open-Cornell-face dropout that made ~22% of lit
    // pixels read zero). Mirror the PROVEN GI producer (risGi.wgsl.ts:252-257):
    // synthesize a FAR reconnection vertex along the escape direction and treat
    // the environment radiance as the outgoing radiance Lo leaving it. The
    // reconnection shift holds this synthetic xs fixed exactly like a real one
    // (ns = −wi_recon ⇒ cosθ_out = 1; the half-G dist² term is finite & large,
    // so the Jacobian is well-defined). sampleEnvironmentColor matches the
    // megakernel's escape term (kernel.wgsl.ts:326).
    let kReconEscapeDist: f32 = 100.0; // GI RECONNECT_MAX_DIST analogue
    xs = reconRay.origin + wiRecon * kReconEscapeDist;
    ns = -wiRecon;
    Lo = sampleEnvironmentColor(wiRecon);
  } else {
    xs = reconRay.origin + reconRay.direction * sHit.dist;
    let sIsFront = dot(sHit.normal, reconRay.direction) < 0.0;
    ns = select(-sHit.normal, sHit.normal, sIsFront);
    let xsMatId = hitMaterialId(sHit);

    // ── 3. Suffix radiance Lo leaving xs toward xv ──
    let reconDirToXv = safe_normalize(xv - xs); // wo at xs (Lo is measured here)
    // The reconnection-vertex shading normal must face xv (so NEE/emission at xs
    // use the front side seen from the prefix). ns is already front-relative to
    // the reconnection ray; re-orient toward xv for the suffix's wo.
    let nsTowardXv = select(-ns, ns, dot(ns, reconDirToXv) > 0.0);
    // Suffix bounce budget: bounded short walk (the reconnection-vertex tail). Kept
    // modest; the wiring step can align it with the megakernel bounce budget for
    // A/B parity. maxBounces is the host's path budget; the suffix is at most that.
    let suffixBounces = max(1u, min(params.maxBounces, 4u));
    Lo = rptComputeLoAtReconnection(&rng, xs, xsMatId, nsTowardXv, reconDirToXv, suffixBounces);
  }

  // ── Seed a 1-sample RIS reservoir, finalise, store ──
  var r = emptyReservoirPTHero();
  r.xv = xv; r.nv = nv;
  r.albV = baseColorV; r.roughnessV = roughnessV; r.metalV = metallicV;
  r.clearcoatV = clearcoatV;
  r.clearcoatRoughnessV = clearcoatRoughnessV;
  r.sheenV = sheenV;
  r.sheenRoughnessV = sheenRoughnessV;
  r.sheenColorV = sheenColorV;
  r.iridescenceV = iridescenceV;
  r.iridescenceIorV = iridescenceIorV;
  r.iridescenceThicknessMinV = iridescenceThicknessMinV;
  r.iridescenceThicknessMaxV = iridescenceThicknessMaxV;
  r.anisotropyV = anisotropyV;
  r.anisotropyRotationV = anisotropyRotationV;
  r.prefixVertexCount = 1u;
  // Candidate target (integrand-matching: f_bsdf·cos·Lo with the visible-vertex
  // BRDF) for the single reconnection sample. The candidate weight is p̂ / p_src.
  // For 1-sample RIS p̂ cancels in W = w_sum/p̂ = 1/p_src, so this does NOT change the
  // producer's mean — it sets the cross-frame-consistent p̂ the temporal MIS
  // resamples against (the same target finalise uses).
  let pHat = restirPtTargetForDomain(r, woV, xs, Lo);
  let wCandidate = select(0.0, pHat / pdfSrc, pdfSrc > 1e-8);
  updateReservoirPT(&r, xs, ns, Lo, pdfSrc, wCandidate, &rng);
  // GRIS finalize: W = w_sum / p̂ (NO /M — the temporal pass folds with MIS).
  finaliseReservoirPTWGris(&r, rptParams.wCap, params.cameraPos.xyz);
  // Refresh the reconnection-shift cache from the chosen base edge xv → xs.
  refreshReconnectionCachePT(&r);

  storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, r);
}
`;
