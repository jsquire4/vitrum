import { PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL } from './bsdf.wgsl.js';
import {
  composeShadePrologueWgsl,
  SHADE_PROLOGUE_EMISSIVE_COMMENT_FULL,
  SHADE_PROLOGUE_BASE_COLOR_TEX_APPLY_FULL,
} from './shadePrologue.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL } from './kernelCore.wgsl.js';

/**
 * Kernel module — primary-ray generation, motion-vector projection, Russian
 * roulette helpers, per-pixel accumulation, and the `@compute` entry point
 * that ties every other module together.
 *
 * Bundled here:
 *  - `generatePrimaryRay` — inverse-VP camera ray + sub-pixel jitter
 *  - `projectToNdc` — VP-clip projection used for motion vectors
 *  - `causticMode` — UBO accessor for the caustic-strategy selector
 *  - `RRResult` struct + `russianRoulette` — bounce-termination helper
 *  - `accumulateFrame` — output texture writes + variance-moments update
 *  - `main` — the @compute @workgroup_size(8,8,1) kernel that walks each ray
 *
 * This module is the LAST concatenated chunk because it consumes every other
 * module: traceClosest/hitMaterialId (intersection), decodeMaterial /
 * thinFilmTmmRt / fresnelSchlick / sampleMaterialSpectralMu (material),
 * sampleNextBounceDirection / cosineHemisphereSample / evaluateBrdf /
 * brdfDirectionalPdf (bsdf), sampleEnvironmentColor / sampleEnvironmentImportance /
 * environmentPdf / bsdfAreaLightConnectionContribution /
 * bsdfEnvironmentConnectionContribution (connect), and manifoldNeeContribution /
 * photonMapContribution (caustic).
 *
 * WS4 — volumetric subsurface scattering. The medium random walk (free-flight
 * distance sampling + Henyey-Greenstein phase scatter + in-medium NEE with
 * phase↔light MIS) is emitted ONLY when \`volumetricSss\` is true. It is gated
 * OFF (compile-time, not a runtime UBO branch) whenever the BDPT integrator is
 * enabled: the BDPT light subpath has no medium logic, so attenuating only the
 * eye path inside a medium would break energy conservation. With the gate off
 * the kernel falls back to the legacy per-channel Beer-Lambert absorption.
 */
export function composePathTraceKernelWgsl(opts: { readonly volumetricSss: boolean }): string {
  const sss = opts.volumetricSss;
  // Henyey-Greenstein phase helpers are top-level WGSL functions used only by
  // the volumetric walk; include them only when the walk is compiled in so the
  // BDPT-on shader carries no SSS symbols (structural gate, no dead code).
  const hgHelpers = sss ? PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL : '';
  // The transmissive-material block: full volumetric walk when SSS is on, the
  // legacy Beer-Lambert + forward-scatter-radiance fallback when it is off
  // (BDPT-on path — kept byte-for-byte from the pre-WS4 kernel).
  const transmissiveBlock = sss
    ? /* wgsl */ `
    // WS4 volumetric random walk. inMedium is set when the previous bounce
    // refracted INTO this medium (see medium-state update after the bounce
    // sample). σ_t = σ_a + σ_s; σ_a is the host-derived Beer-Lambert
    // absorption (decodeMaterial.sigmaA), σ_s the scattering coefficient.
    // Ref: PBR4e §11 "Volume Scattering"; Henyey-Greenstein 1941.
    if (inMedium) {
      let walkSigmaT = max(mediumSigmaT, vec3f(0.0));
      // Hero-channel σ_t drives the free-flight distance in spectral mode so a
      // single wavelength is tracked per path; otherwise use the max channel
      // (conservative — the densest channel sets the collision rate, the rest
      // ride along via the per-channel transmittance below).
      let heroSigmaT = select(
        max(walkSigmaT.x, max(walkSigmaT.y, walkSigmaT.z)),
        walkSigmaT.x,
        params.spectralEnabled != 0u,
      );
      if (heroSigmaT > 1e-6) {
        let xiFlight = rand_f32(&rng);
        let freeFlightDist = -log(max(1.0 - xiFlight, 1e-9)) / heroSigmaT;
        if (freeFlightDist < hit.dist) {
          // Real collision inside the medium BEFORE the surface: scatter.
          let scatterPos = ray.origin + ray.direction * freeFlightDist;
          // Per-channel single-scattering albedo σ_s/σ_t at the chosen flight
          // distance, re-weighted by the ratio of the per-channel pdf to the
          // hero-channel pdf so non-hero channels stay unbiased (spectral MIS).
          let pdfHero = heroSigmaT * exp(-heroSigmaT * freeFlightDist);
          let transmittance = exp(-walkSigmaT * freeFlightDist);
          let pdfChannel = walkSigmaT * transmittance;
          let channelW = select(vec3f(1.0), pdfChannel / max(pdfHero, 1e-9), params.spectralEnabled == 0u);
          let singleScatterAlbedo = mediumSigmaS / max(walkSigmaT, vec3f(1e-6));
          throughput = throughput * singleScatterAlbedo * channelW;
          let throughputInMedium = throughput;

          // In-medium NEE: connect to the directional light through the medium.
          // The directional light is a DELTA — phase sampling (the medium's
          // analogue of BSDF sampling) has zero probability of ever hitting it,
          // so light sampling is the only strategy that can reach it and takes
          // FULL weight 1.0 (no MIS down-weighting). This mirrors the surface
          // NEE, which also adds the directional contribution at weight 1. The
          // earlier powerHeuristic(1, phaseVal) was area-light-style MIS wrongly
          // applied to a delta light and dimmed in-medium single-scatter from the
          // sun. The estimator is throughput · L_i · phase(ω_scatter→ω_light); the
          // single-scatter albedo σ_s/σ_t is already folded into throughputInMedium.
          if (params.lightDir.w > 1e-6) {
            let lightDir = safe_normalize(params.lightDir.xyz);
            let shadowRay = Ray(scatterPos, lightDir);
            if (!traceAny(shadowRay, 1e-4, INFINITY)) {
              let cosScatter = dot(ray.direction, lightDir);
              let phaseVal = hgPhase(cosScatter, mediumG);
              radiance = radiance + throughputInMedium * vec3f(params.lightDir.w) * phaseVal;
            }
          }

          // Sample the next direction from the HG phase function and continue
          // the walk. The phase-sampled estimator is unbiased (f/pdf = 1); the
          // light it later hits is weighted by the complementary MIS term
          // powerHeuristic(phasePdf, lightPdf) inside the next-bounce emission
          // path, so it balances the NEE term added above (partition of unity).
          ray.origin = scatterPos;
          ray.direction = sampleHenyeyGreenstein(&rng, ray.direction, mediumG);

          if (bounce > 2u) {
            let rrMedium = russianRoulette(&rng, throughput);
            if (!rrMedium.survives) { break; }
            throughput = throughput * rrMedium.throughputMul;
          }
          continue; // skip the surface BSDF this bounce — we scattered in the medium.
        } else {
          // No collision before the surface: reach it and fall through to the
          // surface interaction. The HERO-channel transmittance is ALREADY
          // realized by the free-flight importance sampling — a path only reaches
          // here with probability P(t ≥ d) = exp(-heroSigmaT·hit.dist) — so the
          // estimator must divide the true per-channel transmittance by that
          // survival probability: exp(-σ_t·d)/exp(-heroSigmaT·d) =
          // exp(-(σ_t - heroSigmaT)·d). The hero channel cancels to ×1 (its
          // attenuation lives in the survival fraction); lower-σ_t channels get a
          // >1 correction (they absorb less). Multiplying by the FULL exp(-σ_t·d)
          // here (the prior code) DOUBLE-counted the transmittance → exp(-2σ_t·d),
          // over-darkening every medium by the square of its transmittance. V23.
          throughput = throughput * exp(-(walkSigmaT - vec3f(heroSigmaT)) * hit.dist);
        }
      }
    }`
    : /* wgsl */ `
    // BDPT-on fallback (volumetric walk gated off): legacy per-channel
    // Beer-Lambert absorption + a small forward-scatter radiance term.
    if (transmission > 0.0 && isTranslucent) {
      var spectralMu = vec3f(spectralAvgMu);
      if (spectralSampleCount > 0u) {
        if (params.spectralEnabled != 0u) {
          let mu = sampleMaterialSpectralMu(matId, heroLambdaTo01(heroLambda));
          spectralMu = vec3f(mu);
        } else {
          let sampledMuR = sampleMaterialSpectralMu(matId, 0.15);
          let sampledMuG = sampleMaterialSpectralMu(matId, 0.50);
          let sampledMuB = sampleMaterialSpectralMu(matId, 0.85);
          spectralMu = vec3f(sampledMuR, sampledMuG, sampledMuB);
        }
      }
      let sigmaA = select(vec3f(0.0), max(spectralMu, vec3f(0.0)), hasSpectralAttenuation);
      let sigmaS = max(scatteringRgb, vec3f(scatteringCoeff));
      let sigmaT = max(sigmaA + sigmaS, vec3f(0.0));
      if (max(sigmaT.x, max(sigmaT.y, sigmaT.z)) > 0.0) {
        throughput = throughput * exp(-sigmaT * hit.dist);
      }
      if (scatteringCoeff > 0.0) {
        let anisotropyBoost = 1.0 + 0.5 * scatteringAnisotropy;
        radiance = radiance + throughputAtVertex * sigmaS * (0.02 * scatteringCoeff * anisotropyBoost);
      }
    }`;

  // Medium-state declarations (only present when the walk is compiled in).
  const mediumStateDecls = sss
    ? /* wgsl */ `
  // WS4 volumetric path state. inMedium toggles when a refraction bounce
  // crosses the surface; the σ_t/σ_s/g triple is the medium the eye path is
  // currently traversing.
  var inMedium = false;
  var mediumSigmaT = vec3f(0.0);
  var mediumSigmaS = vec3f(0.0);
  var mediumG = 0.0;`
    : '';

  // Medium-state update after the bounce sample (only when the walk is in).
  const mediumStateUpdate = sss
    ? /* wgsl */ `
    // WS4 — update the medium the eye path is in based on this bounce's
    // surface-crossing event. Derive σ_a from decodeMaterial.sigmaA (host
    // Beer-Lambert) and σ_s from the scattering coefficient(s); the phase
    // anisotropy g is the material's scatteringAnisotropy.
    if (bs.enteredMedium) {
      // σ_a: prefer the spectral-attenuation curve when authored (hero λ in
      // spectral mode, RGB triple otherwise), else the host Beer-Lambert
      // σ_a derived from attenuationColor/attenuationDistance.
      var sigmaAWalk = select(vec3f(0.0), mat.sigmaA, mat.hasSigmaA);
      if (hasSpectralAttenuation && spectralSampleCount > 0u) {
        if (params.spectralEnabled != 0u) {
          let mu = sampleMaterialSpectralMu(matId, heroLambdaTo01(heroLambda));
          sigmaAWalk = vec3f(max(mu, 0.0));
        } else {
          let muR = sampleMaterialSpectralMu(matId, 0.15);
          let muG = sampleMaterialSpectralMu(matId, 0.50);
          let muB = sampleMaterialSpectralMu(matId, 0.85);
          sigmaAWalk = max(vec3f(muR, muG, muB), vec3f(0.0));
        }
      }
      let sigmaSWalk = max(scatteringRgb, vec3f(scatteringCoeff));
      mediumSigmaS = sigmaSWalk;
      mediumSigmaT = max(sigmaAWalk + sigmaSWalk, vec3f(0.0));
      mediumG = clamp(scatteringAnisotropy, -0.95, 0.95);
      inMedium = max(mediumSigmaT.x, max(mediumSigmaT.y, mediumSigmaT.z)) > 1e-6;
    } else if (bs.exitedMedium) {
      inMedium = false;
      mediumSigmaT = vec3f(0.0);
      mediumSigmaS = vec3f(0.0);
      mediumG = 0.0;
    }`
    : '';

  return /* wgsl */ `
${hgHelpers}
${PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL}

fn accumulateFrame(
  gid: vec3u,
  radiance: vec3f,
  firstHitValid: bool,
  firstHitPos: vec3f,
  firstHitNormal: vec3f,
  firstHitAlbedo: vec3f,
  firstHitDepth: f32,
) {
  let sampleColor = max(radiance, vec3f(0.0));

  let pixelIndex = gid.y * params.width + gid.x;
  var accum = accumBuffer[pixelIndex];
  accum = accum + vec4f(sampleColor, 1.0);
  accumBuffer[pixelIndex] = accum;
  let sampleLum = luminance(sampleColor);
  var moments = varianceMomentsBuffer[pixelIndex];
  moments.x = moments.x + sampleLum;
  moments.y = moments.y + sampleLum * sampleLum;
  moments.z = moments.z + 1.0;
  varianceMomentsBuffer[pixelIndex] = moments;

  let display = accum.xyz / max(accum.w, 1.0);
  let count = max(moments.z, 1.0);
  let mean = moments.x / count;
  let varL = max(0.0, moments.y / count - mean * mean);
  textureStore(outputTexture, vec2i(gid.xy), vec4f(display, 1.0));
  if (firstHitValid) {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(firstHitNormal, firstHitDepth));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(firstHitAlbedo, 1.0));
    let ndc = projectToNdc(firstHitPos, params.viewProj);
    let prevNdc = projectToNdc(firstHitPos, params.prevViewProj);
    let motionPx = (ndc - prevNdc) * 0.5 * vec2f(f32(params.width), f32(params.height));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(motionPx, 0.0, 1.0));
  } else {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
  }
  textureStore(varianceTexture, vec2i(gid.xy), vec4f(varL, varL, varL, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  var rng = pcgInit(gid.x, gid.y, params.frameSeed ^ params.frameIndex);
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  var ray = generatePrimaryRay(gid.x, gid.y, jitter);

  // BDPT eye-subpath scratch stack — bind this pixel for the deeply-nested
  // stack helpers (bdptEyeStackStore/Load) used by the full §10.3 connection.
  if (params.bdptEnabled != 0u) {
    bdptSetCurrentPixel(gid.y * params.width + gid.x);
  }
  // Forward scatter pdf at the previous eye vertex (the density that produced
  // the current vertex). For the primary hit E_0 the "previous vertex" is the
  // pinhole camera; its importance directional pdf is modelled as 1.0 (We for
  // an aperture-less pinhole — the one vertex without an aperture model). This
  // replaces the old hardcoded eyePdfFwd=1.0 for all SCENE-surface vertices,
  // where the real BSDF scatter pdf now flows in.
  var bdptPrevScatterPdf = 1.0;
  var bdptPrevPos = params.cameraPos.xyz;

  // Camera-visible emitters: whether the PREVIOUS bounce's BSDF sample was a
  // diffuse/glossy direction that the analytic BSDF↔light connection already
  // MIS-accounts for. The emissive-on-hit term below is added ONLY when this is
  // false — i.e. on the camera ray (init false, so a directly-viewed emitter
  // glows) and after a refraction/specular-transmission bounce (which sets
  // sampleAllowsAreaMis=false, so an emitter seen THROUGH glass glows) — the two
  // paths bsdfAreaLightConnectionContribution cannot reach. This prevents
  // double-counting the emissive hit against the analytic connection on
  // diffuse/glossy bounces. (When cameraVisibleEmitters is off the primitive
  // emissive is zero, so the gate is a no-op and the render is byte-identical.)
  var prevSampleAllowsAreaMis = false;

  var heroLambda = params.heroLambdaNm;
  var heroPdf = params.heroPdf;
  if (params.spectralEnabled != 0u) {
    let hero = sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng));
    heroLambda = hero.x;
    heroPdf = hero.y;
  }

  var radiance = vec3f(0.0);
  var throughput = vec3f(1.0);
  let bounceLimit = max(1u, min(params.maxBounces, 8u));
  var firstHitValid = false;
  var firstHitPos = vec3f(0.0);
  var firstHitNormal = vec3f(0.0, 1.0, 0.0);
  var firstHitAlbedo = vec3f(0.0);
  var firstHitDepth = 0.0;
${mediumStateDecls}

  for (var bounce = 0u; bounce < bounceLimit; bounce = bounce + 1u) {
    var hit = traceClosest(ray, 1e-4, INFINITY);
    // P2 alpha-test pass-through: a baseColor-texture alpha mask/blend hit is
    // "not there" — advance the ray past it and re-trace, WITHOUT consuming a
    // scatter bounce (capped at 8 layers/bounce). Opaque materials return false
    // on the first test, so this loop is a no-op for them (byte-identical).
    for (var aSkip = 0u; aSkip < 8u; aSkip = aSkip + 1u) {
      if (!hit.didHit || !alphaTestPassThrough(hitMaterialId(hit), hit.triIndex, hit.baryVW, &rng)) {
        break;
      }
      ray.origin = ray.origin + ray.direction * (hit.dist + 1e-4);
      hit = traceClosest(ray, 1e-4, INFINITY);
    }
    if (!hit.didHit) {
      radiance = radiance + throughput * sampleEnvironmentColor(ray.direction);
      break;
    }

${composeShadePrologueWgsl(SHADE_PROLOGUE_EMISSIVE_COMMENT_FULL, SHADE_PROLOGUE_BASE_COLOR_TEX_APPLY_FULL)}
    let throughputAtVertex = throughput;
${transmissiveBlock}
    let cosThetaO = max(0.0, dot(normal, wo));
    let f0 = mix(vec3f(0.04), baseColor, metallic);
    let fresnel = fresnelSchlick(cosThetaO, f0);

    var lightCount = 0u;
    if (params.lightDir.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    lightCount = lightCount + params.pointLightCount;
    lightCount = lightCount + params.spotLightCount;
    lightCount = lightCount + params.rectAreaLightCount;
    lightCount = lightCount + params.meshAreaLightCount;
    if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    if (lightCount > 0u) {
      // WS2 — power-weighted light selection (Conty Estévez & Kulla 2018,
      // Shirley 1996). When the full-tier light tree is built (≥2 lights), pick
      // the emitter by a power × spatial-proximity descent and divide the
      // contribution by the tree's branch-product selection pdf lt.pdf. When
      // disabled (lite tier never composes this, or <2 lights) fall back to the
      // uniform pick whose pdf is 1/lightCount (compensated by ·lightCount).
      //
      // The light tree's leaf emitterIndex is built in the SAME order this walk
      // visits lights (directional, point, spot, rect, mesh, env), so picked
      // indexes the same linear slot whichever path produced it.
      //
      // LT_DIST2_FLOOR is the selection-only proximity floor: it caps the
      // distance importance near a light and is NOT the NEE geometry-term clamp
      // (those keep their own per-branch 1e-5/1e-6 floors below).
      var picked: u32 = 0u;
      // 1 / p_select for the chosen light. Every NEE branch divides its
      // contribution by p_select (== multiplies by this reciprocal) to compensate
      // the one-of-N pick. Because the engine's emissive-BRDF hit is added
      // unweighted (line 183), the per-branch MIS uses the per-light area pdf
      // ALONE and the selection compensation stays OUTSIDE the power heuristic —
      // so it CANCELS in expectation and the converged mean is INDEPENDENT of the
      // selection pdf (uniform-vs-tree means match; only variance changes). This
      // is the property V22's unbiasedness A/B checks.
      var lightSelectInvPdf: f32 = f32(lightCount); // 1 / (1/lightCount)
      let lightTreeActive = params.lightTreeEnabled != 0u && params.lightTreeNodeCount > 0u;
      if (lightTreeActive) {
        let lt = sampleLightTree(hitPos, LT_DIST2_FLOOR, params.lightTreeNodeCount, &rng);
        if (lt.emitterIndex >= 0 && lt.pdf > 0.0 && u32(lt.emitterIndex) < lightCount) {
          picked = u32(lt.emitterIndex);
          lightSelectInvPdf = 1.0 / lt.pdf;
        } else {
          // Degenerate tree draw — fall back to the uniform pick this iteration.
          picked = u32(min(floor(rand_f32(&rng) * f32(lightCount)), f32(lightCount - 1u)));
          lightSelectInvPdf = f32(lightCount);
        }
      } else {
        picked = u32(min(floor(rand_f32(&rng) * f32(lightCount)), f32(lightCount - 1u)));
        lightSelectInvPdf = f32(lightCount);
      }
      var current = 0u;
      var directLi = vec3f(0.0);
      if (params.lightDir.w > 1e-6) {
        if (current == picked) {
          let lightDir = safe_normalize(params.lightDir.xyz);
          let shadowRay = Ray(hitPos + normal * 1e-3, lightDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY)) {
            let nDotL = max(0.0, dot(normal, lightDir));
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, lightDir);
            // Delta light (no MIS): compensate the one-of-N selection by /p_select.
            directLi = throughput * brdf * nDotL * params.lightDir.w * lightSelectInvPdf;
          }
        }
        current = current + 1u;
      }
      for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
        if (current == picked) {
          let base = pi * 2u;
          let lp = pointLights[base].xyz;
          let rad = pointLights[base + 1u].rgb;
          let toPoint = lp - hitPos;
          let dist2 = max(dot(toPoint, toPoint), 1e-5);
          let dist = sqrt(dist2);
          let wi = toPoint / dist;
          let pointShadowRay = Ray(hitPos + normal * 1e-3, wi);
          if (!traceAny(pointShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
            let nDotL = max(0.0, dot(normal, wi));
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
            // Delta light (no MIS): compensate the one-of-N selection by /p_select.
            directLi = throughput * brdf * nDotL * (rad / dist2) * lightSelectInvPdf;
          }
        }
        current = current + 1u;
      }
      for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
        if (current == picked) {
          let sb = si * 3u;
          let spos = spotLights[sb].xyz;
          let saxis = spotLights[sb + 1u];
          let srad = spotLights[sb + 2u].rgb;
          let spotDir = safe_normalize(saxis.xyz);
          let cosOuter = saxis.w;
          let toSpot = spos - hitPos;
          let dist2 = max(dot(toSpot, toSpot), 1e-5);
          let dist = sqrt(dist2);
          let wi = toSpot / dist;
          let coneCos = dot(-wi, spotDir);
          if (coneCos >= cosOuter) {
            let spotShadowRay = Ray(hitPos + normal * 1e-3, wi);
            if (!traceAny(spotShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
              let nDotL = max(0.0, dot(normal, wi));
              let softness = smoothstep(cosOuter, 1.0, coneCos);
              let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
              // Delta light (no MIS): compensate the one-of-N selection by /p_select.
              directLi = throughput * brdf * nDotL * softness * (srad / dist2) * lightSelectInvPdf;
            }
          }
        }
        current = current + 1u;
      }
      for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
        if (current == picked) {
          let rb = ri * 4u;
          let rpos = rectAreaLights[rb].xyz;
          let ru = rectAreaLights[rb + 1u].xyz;
          let rv = rectAreaLights[rb + 2u].xyz;
          let rr = rectAreaLights[rb + 3u].rgb;
          let u = rand_f32(&rng) * 2.0 - 1.0;
          let v = rand_f32(&rng) * 2.0 - 1.0;
          let lpos = rpos + ru * u + rv * v;
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = max(dot(normal, wi), 0.0);
          if (nDotL > 0.0) {
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
            let lightNormal = safe_normalize(cross(ru, rv));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let area = max(4.0 * length(cross(ru, rv)), 1e-6);
              let lightPdf = dist2 / max(cosLight * area, 1e-6);
              let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
              // MIS balances the per-light AREA-sampling pdf against the BRDF pdf
              // (the engine's emissive-BRDF hit is added unweighted at line 183,
              // so the NEE MIS uses p_area ALONE — NOT p_select·p_area). The light
              // SELECTION is compensated OUTSIDE the heuristic by ·lightSelectInvPdf,
              // which cancels in expectation and so leaves the converged mean
              // INDEPENDENT of the selection pdf (uniform-vs-tree means match);
              // only the variance changes. (Folding p_select into the heuristic
              // would make the NEE total depend on the selection pdf, since the
              // BRDF side is unweighted — that would bias tree-vs-uniform.)
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let shadowRay = Ray(hitPos + normal * 1e-3, wi);
              if (!traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
                directLi = throughput * brdf * nDotL * rr * misWeight / max(lightPdf, 1e-6) * lightSelectInvPdf;
              }
            }
          }
        }
        current = current + 1u;
      }
      for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
        if (current == picked) {
          let mb = mi * 4u;
          let a = meshAreaLights[mb].xyz;
          let b = meshAreaLights[mb + 1u].xyz;
          let c = meshAreaLights[mb + 2u].xyz;
          let mr = meshAreaLights[mb + 3u].rgb;
          let r1 = rand_f32(&rng);
          let r2 = rand_f32(&rng);
          let su = sqrt(r1);
          let uu = 1.0 - su;
          let vv = r2 * su;
          let ww = 1.0 - uu - vv;
          let lpos = a * uu + b * vv + c * ww;
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = max(dot(normal, wi), 0.0);
          if (nDotL > 0.0) {
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
            let lightNormal = safe_normalize(cross(b - a, c - a));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
              let lightPdf = dist2 / max(cosLight * area, 1e-6);
              let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
              // Selection compensated OUTSIDE the MIS (·lightSelectInvPdf) — see
              // the rect-area branch. Keeps the converged mean independent of the
              // selection pdf (tree-vs-uniform means match), variance differs.
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let shadowRay = Ray(hitPos + normal * 1e-3, wi);
              if (!traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
                directLi = throughput * brdf * nDotL * mr * misWeight / max(lightPdf, 1e-6) * lightSelectInvPdf;
              }
            }
          }
        }
        current = current + 1u;
      }
      if ((hasEnvironmentMap() || params.environmentSun.w > 1e-6) && current == picked) {
        var envDir = vec3f(0.0, 1.0, 0.0);
        var envColor = vec3f(0.0);
        var envPdf = 0.0;
        let envSample = sampleEnvironmentImportance(&rng);
        if (envSample.pdf > 0.0) {
          envDir = envSample.wi;
          envColor = envSample.value;
          envPdf = envSample.pdf;
        } else {
          let diffSample = cosineHemisphereSample(&rng, normal);
          envDir = diffSample.wi;
          envColor = sampleEnvironmentColor(envDir);
          envPdf = max(environmentPdf(envDir), 1e-8);
        }
        let nDotL = max(dot(normal, envDir), 0.0);
        if (nDotL > 1e-6) {
          let shadowRay = Ray(hitPos + normal * 1e-3, envDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY)) {
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, envDir);
            let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, envDir);
            // Selection compensated OUTSIDE the MIS (·lightSelectInvPdf) — see the
            // rect-area branch. Mean stays independent of the selection pdf.
            let misWeight = powerHeuristic(envPdf, brdfPdf);
            directLi = throughput * brdf * nDotL * envColor * misWeight / max(envPdf, 1e-8) * lightSelectInvPdf;
          }
        }
      }
      // Every branch already multiplied its contribution by lightSelectInvPdf
      // (= 1/p_select) to compensate the one-of-N pick — OUTSIDE the per-light
      // MIS power heuristic (which uses the per-light area / env pdf alone). So
      // the accumulation is a bare add; the single-sample NEE is unbiased and,
      // crucially, its expectation is INDEPENDENT of the selection pdf (the
      // p_select cancels), so the power-weighted tree and the uniform pick share
      // the same converged mean and differ only in variance.
      radiance = radiance + directLi;
    }

    if (params.bdptEnabled != 0u) {
      // Push this eye vertex (E_bounce) onto the scratch stack BEFORE connecting:
      // pdfRev = forward scatter pdf at the previous vertex that produced E_bounce
      // (camera importance 1.0 for the primary hit). pdfFwd is filled one
      // iteration later by the swapped-direction reverse density (and overridden
      // by the connection straddle terms when this vertex is E_e / E_{e-1}).
      let eyeIsSpecular = transmission > 0.5 && roughness < 0.05;
      bdptEyeStackStore(bounce, hitPos, normal, 0.0, bdptPrevScatterPdf, eyeIsSpecular);
      // Skip the primary hit (bounce 0): an explicit connection there would
      // double-count with the unidirectional NEE above (fork !state.firstRay).
      if (bounce > 0u) {
        let maxLv = min(params.bdptMaxLightBounces, 3u);
        for (var lvi = 0u; lvi < maxLv; lvi++) {
          radiance = radiance + evaluateBdptConnection(
            hitPos,
            normal,
            wo,
            throughputAtVertex,
            baseColor,
            roughness,
            metallic,
            transmission,
            ior,
            bounce,
            i32(lvi),
          );
        }
      }
    }

    let caustic = causticMode();
    if (caustic == 1u) {
      radiance = radiance + manifoldNeeContribution(
        &rng,
        hitPos,
        normal,
        wo,
        baseColor,
        roughness,
        metallic,
        transmission,
        ior,
        throughputAtVertex,
      );
    } else if (caustic == 2u) {
      radiance = radiance + photonMapContribution(
        &rng,
        hitPos,
        normal,
        wo,
        baseColor,
        roughness,
        metallic,
        transmission,
        throughputAtVertex,
      );
    }

    let bs = sampleNextBounceDirection(
      &rng,
      ray.direction,
      hitPos,
      hit.normal,
      normal,
      baseColor,
      roughness,
      metallic,
      transmission,
      ior,
      fresnel,
      thinFilmTransmitTint,
      isTranslucent,
    );
    ray.origin = bs.newRayOrigin;
    ray.direction = bs.newRayDir;
    throughput = throughput * bs.throughputMul;
    let sampledDir = bs.sampledDir;
    let sampleAllowsAreaMis = bs.sampleAllowsAreaMis;
    // Carry to the NEXT iteration's emissive-on-hit gate: if THIS bounce allows
    // area MIS (diffuse/glossy), the analytic connection below covers the light
    // it may hit, so the next hit must NOT re-add that emission. A refraction
    // bounce sets this false → an emitter seen through glass glows next iteration.
    prevSampleAllowsAreaMis = sampleAllowsAreaMis;
${mediumStateUpdate}

    if (params.bdptEnabled != 0u) {
      // The forward scatter pdf of the chosen next direction at E_bounce — fed
      // to the next iteration as E_{bounce+1}'s reverse density. (eyePdfFwd is
      // now this real value, not the old hardcoded 1.0.)
      let scatterPdfFwd = brdfDirectionalPdf(
        baseColor, roughness, metallic, transmission, ior, normal, wo, sampledDir);
      // Merged pdfFwd(E_{bounce-1}): swapped-direction reverse density at the
      // PREVIOUS eye vertex toward E_bounce, using the natural next eye direction
      // as wo (PBRT camera[d-1].pdfRev set while at camera[d]). Write into the
      // previous scratch slot (D1 — non-symmetric reverse density).
      if (bounce >= 1u) {
        let toPrev = safe_normalize(bdptPrevPos - hitPos);
        let swappedRev = brdfDirectionalPdf(
          baseColor, roughness, metallic, transmission, ior, normal, sampledDir, toPrev);
        bdptEyeStackSetFwd(bounce - 1u, swappedRev);
      }
      bdptPrevScatterPdf = scatterPdfFwd;
      bdptPrevPos = hitPos;
    }

    if (sampleAllowsAreaMis) {
      radiance = radiance + bsdfAreaLightConnectionContribution(
        hitPos,
        normal,
        wo,
        sampledDir,
        baseColor,
        roughness,
        metallic,
        transmission,
        ior,
        throughputAtVertex,
      );
      radiance = radiance + bsdfEnvironmentConnectionContribution(
        hitPos,
        normal,
        wo,
        sampledDir,
        baseColor,
        roughness,
        metallic,
        transmission,
        ior,
        throughputAtVertex,
      );
    }

    if (bounce > 2u) {
      let rr = russianRoulette(&rng, throughput);
      if (!rr.survives) { break; }
      throughput = throughput * rr.throughputMul;
    }
  }

  var outRadiance = radiance;
  if (params.spectralEnabled != 0u) {
    outRadiance = heroWavelengthToRgb(heroLambda, luminance(radiance), heroPdf);
  }

  accumulateFrame(
    gid,
    outRadiance,
    firstHitValid,
    firstHitPos,
    firstHitNormal,
    firstHitAlbedo,
    firstHitDepth,
  );
}
`;
}

/**
 * Default full-tier kernel composition — BDPT off ⇒ volumetric SSS walk
 * compiled in. The pipeline picks the BDPT-on (SSS-off) variant explicitly via
 * \`composePathTraceKernelWgsl({ volumetricSss: false })\` when BDPT is enabled.
 */
export const PT_WEBGPU_PATH_TRACE_KERNEL_WGSL = composePathTraceKernelWgsl({
  volumetricSss: true,
});
