/**
 * Caustic module — the two strategy paths the main kernel dispatches when
 * `params.causticStrategy != 0`.
 *
 * Bundled here:
 *  - `perturbAroundDirection` — cone-jittered direction sampler used by the
 *    transmissive cone-search MNEE
 *  - `traceSpecularTransmissiveChain` — multi-bounce specular-transmissive
 *    chain walker (shared between the cone-search MNEE and the photon-map gather)
 *  - `pointLightReflectionCaustic` — caustic strategy mode 1, REFLECTION case:
 *    a REAL Hanika-2015 manifold-NEE point-light specular-reflection caustic. At a
 *    diffuse receiver it seed-traces the hemisphere to find a smooth metallic
 *    mirror, half-vector Newton-solves the EXACT specular vertex on that mirror
 *    (`mneeReflectionIrradiance` from mneeNewton.wgsl.ts — GPU-validated against the
 *    analytic mirror-image irradiance), visibility-tests both legs of the
 *    light→mirror→receiver connection, and weights by the receiver BRDF. This is
 *    the deterministic mirror-image caustic ordinary NEE/BSDF sampling cannot reach
 *    (it is zero-measure for them — the whole reason MNEE exists).
 *  - `manifoldNeeContribution` — caustic strategy mode 1 dispatcher: sums the
 *    REAL reflection caustic above PLUS the legacy transmissive (glass) cone-search
 *    APPROXIMATION (roughness-scaled cone perturbation of a DIRECTIONAL light
 *    through a specular-transmissive chain + a dot>0.75 alignment accept). The
 *    transmissive branch is NOT yet a true manifold solve (no half-vector
 *    constraint / Newton / change-of-variables Jacobian — Phase I.1 step 4 promotes
 *    it onto `mneeNewtonSolveChain2` once the reflection render-A/B is proven).
 *  - `photonMapContribution` — caustic strategy mode 2 (Jensen 1996 photon
 *    mapping with a tiny in-shader photon pass + Gaussian gather kernel)
 *
 * Depends on FrameParams bindings (materials, lightDir, pointLights,
 * spotLights) from `material.wgsl.ts`, evaluateBrdf + brdfDirectionalPdf and
 * `buildOnb` from `bsdf.wgsl.ts`, traceClosest/traceAny/hitMaterialId
 * from `intersection.wgsl.ts`, and `mneeReflectionIrradiance` from
 * `mneeNewton.wgsl.ts` (composed AHEAD of this module in
 * `pathTraceBruteforce.wgsl.ts` so the symbol is in scope).
 */
export const PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL = /* wgsl */ `
fn perturbAroundDirection(baseDir: vec3f, xi: vec2f, coneAngle: f32) -> vec3f {
  var t: vec3f;
  var b: vec3f;
  buildOnb(baseDir, &t, &b);
  let cosThetaMin = cos(coneAngle);
  let cosTheta = mix(cosThetaMin, 1.0, xi.x);
  let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
  let phi = 2.0 * PI * xi.y;
  let local = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  return safe_normalize(local.x * t + local.y * b + local.z * baseDir);
}

fn traceSpecularTransmissiveChain(
  startPos: vec3f,
  startNormal: vec3f,
  startDir: vec3f,
  maxChain: u32,
  exitPos: ptr<function, vec3f>,
  exitDir: ptr<function, vec3f>,
  chainAttenuation: ptr<function, vec3f>,
) -> bool {
  var ray = Ray(startPos + startNormal * 1e-3, safe_normalize(startDir));
  var att = vec3f(1.0);
  // WS4 — Beer-Lambert medium extinction along the specular chain. When a
  // front-face refraction enters a translucent medium, the NEXT segment is
  // travelled inside it; attenuate that segment by exp(-σ_t · segmentLength).
  // (Single-scatter only: the manifold/photon chain stays a specular path, so
  // we model the volume as pure extinction, not in-medium scatter.)
  // Ref: PBR4e §11.1 homogeneous transmittance.
  var chainInMedium = false;
  var chainSigmaT = vec3f(0.0);
  for (var step = 0u; step < 8u; step = step + 1u) {
    if (step >= maxChain) {
      *exitPos = ray.origin;
      *exitDir = ray.direction;
      *chainAttenuation = att;
      return true;
    }
    let hit = traceClosest(ray, 1e-4, INFINITY);
    if (!hit.didHit) {
      *exitPos = ray.origin;
      *exitDir = ray.direction;
      *chainAttenuation = att;
      return true;
    }
    // Attenuate the segment just travelled if it was inside a medium.
    if (chainInMedium && max(chainSigmaT.x, max(chainSigmaT.y, chainSigmaT.z)) > 1e-6) {
      att = att * exp(-chainSigmaT * hit.dist);
    }
    let matId = hitMaterialId(hit);
    // Decode is now canonical (decodeMaterial owns the m0/m2/m3/m22 offset
    // arithmetic + per-field clamps). baseColor is re-clamped to [0,1] below to
    // preserve caustic's historical clamp inside the mix (decodeMaterial leaves
    // baseColor unclamped). (material.wgsl.ts decodeMaterial)
    let mat = decodeMaterial(matId);
    let transmission = mat.transmission;
    if (transmission <= 1e-4) {
      return false;
    }
    let ior = mat.ior;
    let hitPos = ray.origin + ray.direction * hit.dist;
    let frontFace = dot(ray.direction, hit.normal) < 0.0;
    let surfaceNormal = select(-hit.normal, hit.normal, frontFace);
    let eta = select(ior, 1.0 / ior, frontFace);
    let refr = refract(ray.direction, surfaceNormal, eta);
    let hasRefr = dot(refr, refr) > 1e-8;
    let nextDir = select(reflect(ray.direction, surfaceNormal), safe_normalize(refr), hasRefr);
    att = att * mix(vec3f(1.0), clamp(mat.baseColor, vec3f(0.0), vec3f(1.0)), 0.2) * max(transmission, 0.05);
    if (max(att.r, max(att.g, att.b)) < 1e-4) {
      return false;
    }
    // Update medium state for the NEXT segment from this refraction event.
    if (hasRefr && frontFace) {
      let segSigmaA = select(vec3f(0.0), mat.sigmaA, mat.hasSigmaA);
      let segSigmaS = max(mat.scatteringRgb, vec3f(mat.scatteringCoeff));
      chainSigmaT = max(segSigmaA + segSigmaS, vec3f(0.0));
      chainInMedium = max(chainSigmaT.x, max(chainSigmaT.y, chainSigmaT.z)) > 1e-6;
    } else if (hasRefr && !frontFace) {
      chainInMedium = false;
      chainSigmaT = vec3f(0.0);
    }
    ray.origin = hitPos + nextDir * 1e-3;
    ray.direction = nextDir;
  }
  *exitPos = ray.origin;
  *exitDir = ray.direction;
  *chainAttenuation = att;
  return true;
}

// ── REAL MNEE: point-light specular-REFLECTION caustic (Hanika 2015) ─────────
// At a diffuse receiver the mirror caustic of a point light is a DELTA connection
// light → v(mirror) → receiver that obeys the specular reflection law at v. The
// mirror is NOT on the receiver→light ray, so ordinary NEE/BSDF sampling can never
// find it (the path is zero-measure — the whole reason MNEE exists). We:
//   (a) SEED: cast a few UNIFORM-hemisphere rays from the receiver and
//       traceClosest each; the first hit that is a SMOOTH METALLIC surface (a
//       mirror: roughness ≤ REFLECT_ROUGH_MAX and metallic ≥ REFLECT_METAL_MIN) is
//       taken as the candidate reflector plane (its hit point + geometric normal).
//       (Uniform — not cosine — because a side-wall mirror near the floor horizon
//       is almost never sampled by a cosine seed, so the caustic would never fire.)
//   (b) SOLVE: half-vector Newton-solve the EXACT specular vertex on that plane and
//       get the incident irradiance E = I·cosθ_recv/d_unfolded² via the
//       GPU-validated mneeReflectionIrradiance (mneeNewton.wgsl.ts).
//   (c) VISIBILITY: the connection is real only if BOTH legs are unobstructed —
//       receiver→v (traceAny, bounded short of the mirror) and v→light. The v→light
//       leg STEPS THROUGH the mirror's own facets (a thin mirror SOLID has a second
//       facet between v and the light; a naive shadow ray self-occludes on it —
//       this is exactly the bug the render A/B caught) and only a NON-mirror hit
//       shadows the connection. The solved vertex must also LIE on the seed surface
//       (re-trace receiver→v; reject if the closest hit is nearer than v, i.e. an
//       occluder, or v overshoots the finite mirror).
//   (d) ACCUMULATE: throughput · f_r(wo, wi=normalize(v−recv)) · E. E already folds
//       in the receiver cosine (matching the engine's point-light NEE
//       throughput·brdf·nDotL·rad/dist² convention), so this is the bare BRDF×E
//       product — a DELTA connection, no pdf division, no MIS (no other technique
//       reaches this path, so there is no second strategy to balance against).
//
// v1 scope: only the FIRST mirror found per light contributes (break on first valid
// connection). A receiver lit by two+ distinct reflectors of the same light, or a
// multi-bounce specular chain, is not yet summed — those are Phase I.1 follow-ups
// (the 2-vertex chain solver mneeNewtonSolveChain2 already exists for the latter).
const REFLECT_SEED_RAYS = 16u;       // stratified hemisphere seeds / light
const REFLECT_ROUGH_MAX = 0.08;      // a "mirror" is near-smooth
const REFLECT_METAL_MIN = 0.5;       // …and metallic (a polished reflector)
fn pointLightReflectionCaustic(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  throughput: vec3f,
) -> vec3f {
  // A specular reflector cannot itself host the receiver-side diffuse caustic, and
  // a metallic/near-mirror receiver scatters specularly (its own glossy bounce
  // already carries the reflection) — so only run the diffuse-receiver caustic on a
  // sufficiently rough, non-metallic receiver. Keeps the seed search off mirrors.
  if (metallic > 0.5 || roughness < 0.2) {
    return vec3f(0.0);
  }
  let pointCount = min(params.pointLightCount, 16u);
  if (pointCount == 0u) {
    return vec3f(0.0);
  }
  var contribution = vec3f(0.0);
  for (var li = 0u; li < 16u; li = li + 1u) {
    if (li >= pointCount) { break; }
    let lbase = li * 2u;
    let lightPos = pointLights[lbase].xyz;
    let lightI = pointLights[lbase + 1u].rgb;
    if (max(lightI.r, max(lightI.g, lightI.b)) <= 1e-6) { continue; }
    var found = false;
    for (var s = 0u; s < 16u; s = s + 1u) {
      if (s >= REFLECT_SEED_RAYS || found) { break; }
      // UNIFORM hemisphere seed around the receiver normal (cosθ = u1 uniform).
      // A cosine-weighted seed concentrates near the up-normal and almost never
      // aims at a SIDE-WALL mirror (near the floor's horizon), so the caustic never
      // fires; uniform samples the horizon enough to actually find the reflector.
      let u1 = rand_f32(rng);
      let u2 = rand_f32(rng);
      let cz = u1;
      let r = sqrt(max(0.0, 1.0 - u1 * u1));
      let phi = 2.0 * PI * u2;
      var st: vec3f;
      var sb: vec3f;
      buildOnb(normal, &st, &sb);
      let seedDir = safe_normalize((r * cos(phi)) * st + (r * sin(phi)) * sb + cz * normal);
      let seedRay = Ray(hitPos + normal * 1e-3, seedDir);
      let seedHit = traceClosest(seedRay, 1e-4, INFINITY);
      if (!seedHit.didHit) { continue; }
      let mMat = decodeMaterial(hitMaterialId(seedHit));
      // Is the seed hit a mirror (smooth + metallic)?
      if (mMat.roughness > REFLECT_ROUGH_MAX || mMat.metallic < REFLECT_METAL_MIN) {
        continue;
      }
      let mirrorP = seedRay.origin + seedRay.direction * seedHit.dist;
      let mirrorN = safe_normalize(
        select(-seedHit.normal, seedHit.normal, dot(seedRay.direction, seedHit.normal) < 0.0));
      var mTu: vec3f;
      var mTv: vec3f;
      buildOnb(mirrorN, &mTu, &mTv);
      // EXACT half-vector Newton solve on the mirror plane + the mirror-image
      // irradiance E = I·cosθ_recv/d_unfolded² (GPU-validated core).
      let e = mneeReflectionIrradiance(hitPos, normal, mirrorP, mirrorN, mTu, mTv, lightPos, lightI);
      if (max(e.r, max(e.g, e.b)) <= 1e-8) { continue; }
      // The specular vertex v for the VISIBILITY tests. mneeReflectionIrradiance
      // returns only E (not v), but its Newton solve runs on the SAME plane
      // (mirrorP, mirrorN) we pass, and a flat-plane reflection solve provably
      // converges to the mirror-IMAGE point — so v is, identically, the intersection
      // of the receiver→image segment with that plane. image = reflect of lightPos
      // in the plane. (v1 is a FLAT-mirror caustic: both E's solve and this v use
      // the geometric-normal plane; a curved reflector would need true geometry.)
      let dPlane = dot(lightPos - mirrorP, mirrorN);
      let image = lightPos - 2.0 * dPlane * mirrorN;
      let toImage = image - hitPos;
      let denom = dot(toImage, mirrorN);
      if (abs(denom) < 1e-6) { continue; }
      let tHit = dot(mirrorP - hitPos, mirrorN) / denom;
      if (tHit <= 1e-4 || tHit >= 1.0) { continue; } // image must be on the far side
      let v = hitPos + toImage * tHit;
      let wi = safe_normalize(v - hitPos);
      // leg A: receiver → v unobstructed (the mirror itself is the endpoint, so
      // bound the ray just short of v).
      let distA = length(v - hitPos);
      let rayA = Ray(hitPos + normal * 1e-3, wi);
      if (traceAny(rayA, 1e-4, max(distA - 2e-3, 1e-3))) { continue; }
      // The seed surface MUST be the surface actually hit toward v (reject if the
      // closest hit is a DIFFERENT/nearer surface — v drifted off the finite mirror).
      let chkHit = traceClosest(rayA, 1e-4, INFINITY);
      if (!chkHit.didHit || abs(chkHit.dist - distA) > 5e-3) { continue; }
      let chkMat = decodeMaterial(hitMaterialId(chkHit));
      if (chkMat.roughness > REFLECT_ROUGH_MAX || chkMat.metallic < REFLECT_METAL_MIN) { continue; }
      // leg B: v → light unobstructed, EXCLUDING the mirror itself. The reflection
      // vertex sits ON the mirror, and a thin mirror SOLID has a second facet
      // between v and the light, so a naive shadow ray self-occludes on it (every
      // connection died here before this skip). Step the ray through any mirror
      // (smooth + metallic) facets it meets, then test the remaining segment for a
      // REAL (non-mirror) occluder. v1: a single reflector, so stepping past its
      // own facets is correct; a SECOND distinct mirror between v and the light is
      // a Phase-I.1 multi-reflector follow-up.
      let toLight = lightPos - v;
      let distB = length(toLight);
      let dirB = toLight / max(distB, 1e-8);
      var legBOrigin = v + dirB * 1e-3;
      var legBRemaining = distB - 1e-3;
      var legBBlocked = false;
      for (var stepB = 0u; stepB < 4u; stepB = stepB + 1u) {
        let segRay = Ray(legBOrigin, dirB);
        let segHit = traceClosest(segRay, 1e-4, max(legBRemaining - 1e-3, 1e-4));
        if (!segHit.didHit) { break; } // clear to the light
        let segMat = decodeMaterial(hitMaterialId(segHit));
        let isMirror = segMat.roughness <= REFLECT_ROUGH_MAX && segMat.metallic >= REFLECT_METAL_MIN;
        if (!isMirror) { legBBlocked = true; break; } // a real occluder shadows the connection
        // Mirror self-facet: advance just past it and keep testing toward the light.
        let advance = segHit.dist + 1e-3;
        legBOrigin = legBOrigin + dirB * advance;
        legBRemaining = legBRemaining - advance;
        if (legBRemaining <= 1e-3) { break; }
      }
      if (legBBlocked) { continue; }
      // DELTA connection: throughput · f_r · E (E already carries cosθ_recv).
      let fr = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
      contribution = contribution + throughput * fr * e;
      found = true;
    }
  }
  return contribution;
}

fn manifoldNeeContribution(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  throughput: vec3f,
) -> vec3f {
  // REAL Hanika-2015 reflection caustic (point lights + a smooth metallic mirror).
  // Runs for ANY receiver (independent of the transmissive gate below) so a diffuse
  // floor catches a mirror caustic with no glass in the scene. Delta connection —
  // already MIS-complete on its own (no other technique reaches it).
  var total = pointLightReflectionCaustic(rng, hitPos, normal, wo, baseColor, roughness, metallic, throughput);

  // Legacy transmissive (glass) cone-search APPROXIMATION — DIRECTIONAL light only.
  // Promoting this onto the validated mneeNewtonSolveChain2 is Phase I.1 step 4
  // (gated behind a caustic render-A/B, per the validation discipline).
  if (transmission <= 1e-4 || params.lightDir.w <= 1e-6) {
    return total;
  }
  let mneeSteps = clamp(params.mneeMaxIterations, 1u, 8u);
  let maxChain = clamp(params.mneeMaxChainLength, 1u, 8u);
  let baseLightDir = safe_normalize(params.lightDir.xyz);
  let coneAngle = mix(0.01, 0.12, clamp(roughness, 0.0, 1.0));
  var transmissiveContribution = vec3f(0.0);
  for (var step = 0u; step < 8u; step = step + 1u) {
    if (step >= mneeSteps) {
      break;
    }
    let jitter = vec2f(rand_f32(rng), rand_f32(rng));
    let candidateDir = perturbAroundDirection(baseLightDir, jitter, coneAngle);
    let nDotL = max(dot(normal, candidateDir), 0.0);
    if (nDotL <= 1e-5) {
      continue;
    }
    var exitPos = vec3f(0.0);
    var exitDir = vec3f(0.0, 1.0, 0.0);
    var chainAtt = vec3f(1.0);
    if (!traceSpecularTransmissiveChain(hitPos, normal, candidateDir, maxChain, &exitPos, &exitDir, &chainAtt)) {
      continue;
    }
    let align = max(dot(exitDir, baseLightDir), 0.0);
    if (align <= 0.75) {
      continue;
    }
    let visibilityRay = Ray(exitPos + exitDir * 1e-3, baseLightDir);
    if (traceAny(visibilityRay, 1e-4, INFINITY)) {
      continue;
    }
    let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, candidateDir);
    let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, candidateDir);
    let conePdf = 1.0 / max(2.0 * PI * (1.0 - cos(coneAngle)), 1e-6);
    let samplePdf = conePdf / f32(mneeSteps);
    let misWeight = powerHeuristic(samplePdf, brdfPdf);
    let lightRadiance = vec3f(params.lightDir.w) * align;
    transmissiveContribution = transmissiveContribution +
      throughput * chainAtt * brdf * nDotL * lightRadiance * misWeight / max(samplePdf, 1e-6);
  }
  return total + transmissiveContribution;
}

fn photonMapContribution(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughput: vec3f,
) -> vec3f {
  var availableLightCount = 0u;
  if (params.lightDir.w > 1e-6) { availableLightCount = availableLightCount + 1u; }
  if (params.pointLightCount > 0u) { availableLightCount = availableLightCount + params.pointLightCount; }
  if (params.spotLightCount > 0u) { availableLightCount = availableLightCount + params.spotLightCount; }
  if (availableLightCount == 0u) { return vec3f(0.0); }
  let photonCount = u32(clamp(f32(params.mneeMaxIterations) * 2.0, 8.0, 32.0));
  let maxChain = clamp(params.mneeMaxChainLength, 1u, 8u);
  // Photon-gather radius in world units. Hardcoded at 0.35 for the current
  // calibration scene. Exposed as a named local so the photon density / cell
  // size relationship is easy to tune in one place. Future: lift to a params
  // field if hosts need scene-relative tuning.
  let gatherRadius = 0.35;
  let gatherRadius2 = gatherRadius * gatherRadius;
  var contribution = vec3f(0.0);
  for (var photonIdx = 0u; photonIdx < 32u; photonIdx = photonIdx + 1u) {
    if (photonIdx >= photonCount) {
      break;
    }
    let pick = u32(min(floor(rand_f32(rng) * f32(availableLightCount)), f32(availableLightCount - 1u)));
    var current = 0u;
    var photonOrigin = hitPos;
    var photonDir = vec3f(0.0, 1.0, 0.0);
    var photonFlux = vec3f(0.0);
    var seeded = false;
    if (params.lightDir.w > 1e-6) {
      if (current == pick) {
        photonOrigin = hitPos - safe_normalize(params.lightDir.xyz) * 24.0;
        photonDir = safe_normalize(params.lightDir.xyz);
        photonFlux = vec3f(params.lightDir.w);
        seeded = true;
      }
      current = current + 1u;
    }
    if (params.pointLightCount > 0u) {
      if (pick >= current && pick < current + params.pointLightCount) {
        let pointIdx = pick - current;
        let pointBase = pointIdx * 2u;
        photonOrigin = pointLights[pointBase].xyz;
        photonDir = uniformSphere(vec2f(rand_f32(rng), rand_f32(rng)));
        photonFlux = pointLights[pointBase + 1u].rgb;
        seeded = true;
      }
      current = current + params.pointLightCount;
    }
    if (params.spotLightCount > 0u && pick >= current && pick < current + params.spotLightCount) {
      let spotIdx = pick - current;
      let spotBase = spotIdx * 3u;
      photonOrigin = spotLights[spotBase].xyz;
      let coneXi = vec2f(rand_f32(rng), rand_f32(rng));
      let cosMin = spotLights[spotBase + 1u].w;
      let cosTheta = mix(cosMin, 1.0, coneXi.x);
      let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
      let phi = 2.0 * PI * coneXi.y;
      let local = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
      let spotAxis = safe_normalize(-spotLights[spotBase + 1u].xyz);
      var t: vec3f;
      var b: vec3f;
      buildOnb(spotAxis, &t, &b);
      photonDir = safe_normalize(local.x * t + local.y * b + local.z * spotAxis);
      photonFlux = spotLights[spotBase + 2u].rgb;
      seeded = true;
    }
    if (!seeded) {
      continue;
    }
    var ray = Ray(photonOrigin + photonDir * 1e-3, photonDir);
    var flux = photonFlux / max(f32(photonCount), 1.0);
    for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
      if (bounce >= maxChain) { break; }
      let hit = traceClosest(ray, 1e-4, INFINITY);
      if (!hit.didHit) { break; }
      let matId = hitMaterialId(hit);
      // Decode is now canonical (decodeMaterial owns the offset arithmetic +
      // per-field clamps). baseColor is re-clamped to [0,1] in the flux mix
      // below to preserve caustic's historical clamp. (material.wgsl.ts decodeMaterial)
      let mat = decodeMaterial(matId);
      let mTransmission = mat.transmission;
      let mIor = mat.ior;
      let hp = ray.origin + ray.direction * hit.dist;
      let dist2ToReceiver = dot(hp - hitPos, hp - hitPos);
      if (dist2ToReceiver <= gatherRadius2) {
        let wi = -ray.direction;
        let nDotL = max(dot(normal, wi), 0.0);
        if (nDotL > 1e-6) {
          let receiverBrdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
          let kernel = exp(-dist2ToReceiver / max(2.0 * gatherRadius2, 1e-6)) / max(PI * gatherRadius2, 1e-6);
          contribution = contribution + throughput * flux * receiverBrdf * nDotL * kernel;
        }
      }
      if (mTransmission <= 1e-4) {
        break;
      }
      let frontFace = dot(ray.direction, hit.normal) < 0.0;
      let n = select(-hit.normal, hit.normal, frontFace);
      let eta = select(mIor, 1.0 / mIor, frontFace);
      let refr = refract(ray.direction, n, eta);
      let hasRefr = dot(refr, refr) > 1e-8;
      let nextDir = select(reflect(ray.direction, n), safe_normalize(refr), hasRefr);
      flux = flux * mix(vec3f(1.0), clamp(mat.baseColor, vec3f(0.0), vec3f(1.0)), 0.2) * max(mTransmission, 0.05);
      if (max(flux.r, max(flux.g, flux.b)) < 1e-5) {
        break;
      }
      ray.origin = hp + nextDir * 1e-3;
      ray.direction = nextDir;
    }
  }
  let strategyScale = 1.0 + 0.25 * transmission;
  return contribution * strategyScale;
}
`;
