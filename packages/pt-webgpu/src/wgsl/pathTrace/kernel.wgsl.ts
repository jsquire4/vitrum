import { PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL } from './bsdf.wgsl.js';
import {
  composeShadePrologueWgsl,
  SHADE_PROLOGUE_EMISSIVE_COMMENT_FULL,
  SHADE_PROLOGUE_BASE_COLOR_TEX_APPLY_FULL,
  SHADE_PROLOGUE_EMISSIVE_TEX_APPLY_FULL,
  SHADE_PROLOGUE_ORM_TEX_APPLY_FULL,
  SHADE_PROLOGUE_NORMAL_MAP_APPLY_FULL,
  SHADE_PROLOGUE_AO_APPLY_FULL,
  SHADE_PROLOGUE_LIGHT_MAP_APPLY_FULL,
  SHADE_PROLOGUE_BUMP_MAP_APPLY_FULL,
  SHADE_PROLOGUE_TRANSMISSION_MAP_APPLY_FULL,
  SHADE_PROLOGUE_VOLUME_THICKNESS_MAP_APPLY_FULL,
  SHADE_PROLOGUE_EXTENSION_LOBE_TEX_APPLY_FULL,
  SHADE_PROLOGUE_CLEARCOAT_NORMAL_MAP_APPLY_FULL,
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
export function composePathTraceKernelWgsl(opts: {
  readonly volumetricSss: boolean;
  /**
   * A1 — ReSTIR-PT composite mode. When true the megakernel renders ONLY the
   * DIRECT illumination at the primary visible vertex E0 (camera-visible emission
   * of E0 + analytic NEE at E0: deltas + area-light NEE + env NEE) and the ReSTIR-PT
   * resolve pass supplies ALL the indirect (everything from the first bounce off
   * E0 onward). See the estimator-split note at the composite gate. OFF (default)
   * emits the verbatim full-path kernel — byte-identical to pre-A1. */
  readonly restirPtComposite?: boolean;
}): string {
  const sss = opts.volumetricSss;
  const composite = opts.restirPtComposite === true;
  // BDPT estimator boundary: bdptOptions.experimentalMultiVertex remains a
  // research diagnostic. The connection kernel owns real Veach 10.3 MIS for an
  // explicit eye-light connection, but this megakernel still accumulates the
  // ordinary eye-path estimator at full weight (radiance += directLi,
  // emissive/env hits, and BSDF continuation). Until those regular eye-path
  // strategies are composed into the same strategy family, the extra
  // multi-vertex connections are not a promotable production estimator.
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
        let attenuationDist = min(hit.dist, mediumAttenuationLimit);
        if (freeFlightDist < attenuationDist) {
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

          // In-medium NEE: connect to every packed directional light through the
          // medium. Delta directionals and finite soft-sun cones use the same
          // light-sampled estimator as the surface NEE branch; phase sampling is
          // not paired with a directional-light MIS term in this kernel. The
          // estimator is throughput · L_i · phase(ω_scatter→ω_light); the
          // single-scatter albedo σ_s/σ_t is already folded into
          // throughputInMedium. This uses the N-directional storage buffer
          // instead of the legacy scalar params.lightDir.w mirror, preserving RGB
          // irradiance and >1 sun.
          for (var medDi = 0u; medDi < params.directionalLightCount; medDi = medDi + 1u) {
            let dBase = medDi * 2u;
            let dDirAD = directionalLights[dBase];
            let dIrrMean = directionalLights[dBase + 1u];
            if (dIrrMean.w > 1e-6) {
              let angDiamRaw = dDirAD.w;
              let dirShadowDisabled = angDiamRaw < 0.0;
              let angDiam = select(angDiamRaw, -1.0 - angDiamRaw, dirShadowDisabled);
              let lightDir = sampleDirectionalCone(&rng, dDirAD.xyz, angDiam);
              let shadowRay = Ray(scatterPos, lightDir);
              if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY)) {
                let cosScatter = dot(ray.direction, lightDir);
                let phaseVal = hgPhase(cosScatter, mediumG);
                radiance = radiance + throughputInMedium * dIrrMean.rgb * phaseVal;
              }
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
          throughput = throughput * exp(-(walkSigmaT - vec3f(heroSigmaT)) * attenuationDist);
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
        throughput = throughput * exp(-sigmaT * materialAttenuationDistance(hit.dist, mat));
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
  var mediumG = 0.0;
  var mediumAttenuationLimit = INFINITY;`
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
      mediumAttenuationLimit = materialAttenuationDistance(INFINITY, mat);
      inMedium = max(mediumSigmaT.x, max(mediumSigmaT.y, mediumSigmaT.z)) > 1e-6;
    } else if (bs.exitedMedium) {
      inMedium = false;
      mediumSigmaT = vec3f(0.0);
      mediumSigmaS = vec3f(0.0);
      mediumG = 0.0;
      mediumAttenuationLimit = INFINITY;
    }`
    : '';

  // A1 — composite mode module-scope binding: the ReSTIR-PT resolve output
  // (one vec4f / px; .rgb = reconnection indirect, .a = contributing flag). The
  // compose wrapper (composePtWebgpuCompositeTraceWgsl) relocates this
  // @group(4)@binding(3) decl onto @group(0)@binding(23) — the SAME relocated slot
  // the reuse passes' rpt_result occupies — so the megakernel reads exactly what
  // resolve wrote. Absent entirely when composite is OFF (byte-identical default).
  // OFF (default) emits the EMPTY string with no surrounding newline so the
  // composed kernel is byte-for-byte the pre-A1 string (the byte-identity pin).
  // NB: declared read_write to MATCH the shared reuse group-0 layout slot (binding
  // 23 is `storage`/read_write there — the resolve pass WRITES it). Reading a
  // read_write storage global is legal; this only sets the access qualifier.
  const rptResultBinding = composite
    ? /* wgsl */ `@group(4) @binding(3) var<storage, read_write> rpt_result_in: array<vec4f>;
`
    : '';
  // A1 — the per-bounce BSDF→light/env MIS connection adds. OFF (default) emits
  // the VERBATIM original block (byte-identical pin). In composite mode the
  // BSDF-sampled direct connections MUST BE KEPT for composited pixels: analytic
  // lights (rect-area, disc, env/sky, directional) are NOT in the TLAS/BVH, so the
  // producer's reconnection vertex xs can NEVER land on an analytic light. Dropping
  // them (the previous !rptCompositeContributed gate) caused a ~46% energy
  // under-bias verified by the A/B harness
  // (tools/radiometric-ab/ab-restir-pt.mjs, 2026-06-10).
  //
  // Mesh area lights are different: xs CAN be the emissive mesh, so the resolve's
  // Lo already includes emissive(xs). In composite mode, contributed pixels keep
  // rect/disc analytic BSDF connections but suppress only the mesh-area branch of
  // bsdfAreaLightConnectionContribution. Producer-dropped pixels and the default
  // megakernel keep the full rect/disc/mesh connection set.
  //
  // A1 composite preamble — read the resolve indirect for THIS pixel once. A pixel
  // the producer contributed to (rpt.a > 0.5) gets the E0-direct-only + composited-
  // indirect split; a producer-DROPPED pixel (specular/transmissive E0 → rpt.a == 0)
  // falls through to the FULL path so glass/mirror primaries still trace their
  // reflection/refraction. Empty when composite is OFF (byte-identical default).
  const compositePreamble = composite
    ? /* wgsl */ `    let rptCompositeIdx = gid.y * params.width + gid.x;
    let rptComposite = rpt_result_in[rptCompositeIdx];
    let rptCompositeContributed = rptComposite.a > 0.5;
`
    : '';
  // The BSDF→light/env area-MIS connection condition. Identical in both composite
  // and default mode: the BSDF-side MIS connections at E0 must run for ALL pixels
  // (composited or not) because analytic lights are never reconnection vertices,
  // so there is no double-count risk. OFF = the verbatim original.
  const sampleAllowsAreaMisCond = 'sampleAllowsAreaMis';
  const includeMeshAreaLightsExpr = composite ? '!rptCompositeContributed' : 'true';
  const bsdfAreaConnect = /* wgsl */ `    if (${sampleAllowsAreaMisCond}) {
      // H52/PTWG-MAT: BSDF-side area/env connections receive the decoded
      // extension lobe scalars so connect.wgsl can use evaluateBrdfFull and
      // brdfDirectionalPdfFull. Zero-default materials stay byte-equivalent.
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
        mat.clearcoat,
        mat.clearcoatRoughness,
        mat.sheen,
        mat.sheenRoughness,
        mat.sheenColor,
        mat.iridescence,
        mat.iridescenceIor,
        mat.iridescenceThicknessMin,
        mat.iridescenceThicknessMax,
        mat.specularColor,
        mat.specularIntensity,
        anisoStrength,
        anisoRotation,
        throughputAtVertex,
        heroLambda,
        ${includeMeshAreaLightsExpr},
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
        mat.clearcoat,
        mat.clearcoatRoughness,
        mat.sheen,
        mat.sheenRoughness,
        mat.sheenColor,
        mat.iridescence,
        mat.iridescenceIor,
        mat.iridescenceThicknessMin,
        mat.iridescenceThicknessMax,
        mat.specularColor,
        mat.specularIntensity,
        anisoStrength,
        anisoRotation,
        throughputAtVertex,
        heroLambda,
        matId,
      );
    }
`;
  // A1 — composite early-out. After E0's emission + E0's full direct (NEE + BSDF-side
  // MIS connections — BOTH halves are now included for composited pixels), composite the
  // ReSTIR-PT reconnection indirect (f_bsdf(E0)·cos·Lo·W, reconstructed by resolve)
  // into the BEAUTY accumulator and TERMINATE. The megakernel supplies E0's complete
  // direct contribution; the resolve supplies the indirect (first bounce off E0 onward).
  // Double-count-free: analytic lights (rect-area/disc/env/sky/directional) are NOT in
  // the TLAS, so the producer cannot place xs on them and rptComposite.rgb never
  // overlaps the BSDF-side MIS terms. Producer-dropped pixels skip this and continue
  // the FULL path. OFF (default) emits NOTHING (byte-identical).
  const compositeEarlyOut = composite
    ? /* wgsl */ `    if (rptCompositeContributed) {
      radiance = radiance + rptComposite.rgb;
      break;
    }
`
    : '';

  return /* wgsl */ `${rptResultBinding}
${hgHelpers}
${PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL}

fn sampleDirectionalCone(rng: ptr<function, u32>, axisIn: vec3f, angularDiameter: f32) -> vec3f {
  var sampleDir = safe_normalize(axisIn);
  if (angularDiameter > 0.0) {
    let cosHalfAngle = cos(angularDiameter * 0.5);
    let xi1 = rand_f32(rng);
    let xi2 = rand_f32(rng);
    let cosTheta = mix(cosHalfAngle, 1.0, xi1);
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let phi = 6.28318530718 * xi2;
    let tangentX = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sampleDir.x) > 0.9);
    let basisY = normalize(cross(sampleDir, tangentX));
    let basisX = cross(basisY, sampleDir);
    sampleDir = normalize(sinTheta * cos(phi) * basisX + sinTheta * sin(phi) * basisY + cosTheta * sampleDir);
  }
  return sampleDir;
}

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

  var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));
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

  // A4-progressive: per-pixel flat index for sppmPixelStats read/write.
  let pixelIndex = gid.y * params.width + gid.x;
  // PTWG-04: sppmPixelStats has one record per pixel, so the progressive update
  // may run at most once per frame. We update at the first diffuse-ish gather
  // surface along the eye path; later bounces keep tracing but do not shrink the
  // same pixel's radius/N again.
  var sppmGatherUpdated = false;

  var radiance = vec3f(0.0);
  var throughput = vec3f(1.0);
  let bounceLimit = max(1u, min(params.maxBounces, 8u));
  var firstHitValid = false;
  var firstHitPos = vec3f(0.0);
  var firstHitNormal = vec3f(0.0, 1.0, 0.0);
  var firstHitAlbedo = vec3f(0.0);
  var firstHitDepth = 0.0;
  // Item 8 — carry the LAST shaded surface's envMapIntensity into the BSDF-escape
  // env pickup (the no-hit branch below). The NEE half already applies the same
  // factor (materialEnvMapIntensity(matId) at line ~755), so scaling the escape
  // half makes both MIS halves consistent — envMapIntensity≠1 scenes no longer
  // diverge between backends. Camera-visible env (no prior hit, bounce=0 escape)
  // stays unscaled (initialized to 1.0 below). Ref: pt-webgl2 state.envMapIntensity.
  var lastEnvMapIntensity = 1.0;
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
      // A3 — environment radiance in spectral mode: upsample the RGB env to a
      // hero-λ spectral value (same approximation as emitters) so the throughput
      // (already scalar-spectral) times the env stays a single-wavelength
      // quantity. RGB mode: env color unchanged → byte-identical.
      let envRgb = sampleEnvironmentColor(ray.direction);
      let envContribution = select(envRgb, spectralEmissionAtHero(envRgb, heroLambda), params.spectralEnabled != 0u);
      // Item 8 — apply the last shaded surface's envMapIntensity to this BSDF-escape
      // env pickup, matching the scale already applied in the NEE half (~line 755).
      // lastEnvMapIntensity is 1.0 for camera-visible env (bounce=0, no prior hit).
      radiance = radiance + throughput * envContribution * lastEnvMapIntensity;
      break;
    }

${composeShadePrologueWgsl(SHADE_PROLOGUE_EMISSIVE_COMMENT_FULL, SHADE_PROLOGUE_BASE_COLOR_TEX_APPLY_FULL, SHADE_PROLOGUE_EMISSIVE_TEX_APPLY_FULL, SHADE_PROLOGUE_ORM_TEX_APPLY_FULL, SHADE_PROLOGUE_NORMAL_MAP_APPLY_FULL, SHADE_PROLOGUE_AO_APPLY_FULL, SHADE_PROLOGUE_LIGHT_MAP_APPLY_FULL, SHADE_PROLOGUE_BUMP_MAP_APPLY_FULL, SHADE_PROLOGUE_TRANSMISSION_MAP_APPLY_FULL, SHADE_PROLOGUE_VOLUME_THICKNESS_MAP_APPLY_FULL, SHADE_PROLOGUE_EXTENSION_LOBE_TEX_APPLY_FULL, SHADE_PROLOGUE_CLEARCOAT_NORMAL_MAP_APPLY_FULL)}
    // Item 8 — record this surface's envMapIntensity for the forward env escape
    // pickup on the NEXT iteration (mirrors pt-webgl2 state.envMapIntensity update).
    lastEnvMapIntensity = materialEnvMapIntensity(matId);
    // Item 7 — per-hit anisotropy (used in NEE eval/pdf + sampleNextBounceDirection).
    let anisoStrength = materialAnisotropy(matId, hit.triIndex, hit.baryVW);
    let anisoRotation = materialAnisotropyRotation(matId, hit.triIndex, hit.baryVW);
    let throughputAtVertex = throughput;
${transmissiveBlock}
    let cosThetaO = max(0.0, dot(normal, wo));
    let f0Base = materialSpecularF0(baseColor, metallic, mat.specularColor, mat.specularIntensity);
    let f0 = iridescenceModifiedF0(
      f0Base,
      mat.iridescence,
      mat.iridescenceIor,
      mat.iridescenceThicknessMin,
      mat.iridescenceThicknessMax,
      cosThetaO,
    );
    let fresnel = fresnelSchlick(cosThetaO, f0);

    var lightCount = 0u;
    // N-directional: count each directional with non-zero mean irradiance.
    // The storage-buffer record already packs mean_irradiance in [di*2+1].w;
    // directionalLightCount is the total records (only packed when meanIrr > 1e-6
    // at pack time, so every record here is a real light).
    lightCount = lightCount + params.directionalLightCount;
    lightCount = lightCount + params.pointLightCount;
    lightCount = lightCount + params.spotLightCount;
    lightCount = lightCount + params.rectAreaLightCount;
    lightCount = lightCount + params.meshAreaLightCount;
    if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    if (lightCount > 0u) {
      let sumDirectLighting = params.directLightingMode == 1u;
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
      if (sumDirectLighting) {
        lightSelectInvPdf = 1.0;
      } else if (lightTreeActive) {
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
      let directLightingScale = select(lightSelectInvPdf, 1.0, sumDirectLighting);
      var current = 0u;
      var directLi = vec3f(0.0);
      // N-directional loop: each record in directionalLights[] is 2 vec4f:
      //   [di*2+0]: towardLight.xyz, angularDiameter
      //   [di*2+1]: irradiance.rgb,  mean_irradiance
      // N-directional: replaces the single "if (params.lightDir.w > 1e-6)" path;
      // the first directional (di=0) remains equivalent for single-directional
      // scenes because the same packed direction/irradiance data drives both
      // this surface loop and the in-medium NEE loop above.
      for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
        if (sumDirectLighting || current == picked) {
          let dBase = di * 2u;
          let dDirAD = directionalLights[dBase];        // .xyz = toward-light dir, .w = angularDiameter
          let dIrrMean = directionalLights[dBase + 1u]; // .rgb = irradiance,        .w = mean irradiance
          var sampleDir = safe_normalize(dDirAD.xyz);
          // SHADOW-01 — emitter castShadow:false is sign-encoded into the
          // angularDiameter lane (packed = -1 - ad; see emitterPacking.ts).
          // Default lights pack ad >= 0 → decode is the identity.
          let angDiamRaw = dDirAD.w;
          let dirShadowDisabled = angDiamRaw < 0.0;
          // D3 soft-sun cone sampling — reuses the same cone logic as the
          // original single-directional path (angularDiameter > 0 ⟹ sample a
          // uniformly-random direction within the solid-angle cone).
          let angDiam = select(angDiamRaw, -1.0 - angDiamRaw, dirShadowDisabled);
          sampleDir = sampleDirectionalCone(&rng, sampleDir, angDiam);
          let shadowRay = Ray(hitPos + normal * 1e-3, sampleDir);
          if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY)) {
            let nDotL = max(0.0, dot(normal, sampleDir));
            // H52: evaluateBrdfFull adds clearcoat/sheen/iridescence lobes;
            // zero-default → identical to evaluateBrdf when all scalars are 0.
            let brdf = evaluateBrdfFullWithClearcoatNormal(baseColor, roughness, metallic, normal, clearcoatNormal, wo, sampleDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation);
            // A3 — spectralise the directional irradiance at the hero λ (RGB unchanged).
            let dIrrOut = select(dIrrMean.rgb, spectralEmissionAtHero(dIrrMean.rgb, heroLambda), params.spectralEnabled != 0u);
            // Delta light (no MIS): compensate the one-of-N selection by /p_select.
            directLi = directLi + throughput * brdf * nDotL * dIrrOut * directLightingScale;
          }
        }
        current = current + 1u;
      }
      for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
        if (sumDirectLighting || current == picked) {
          // H51-D: stride 3 (3 vec4 = 12 f32): position, radiance, [distance, decay, 0, 0]
          let base = pi * 3u;
          let lp = pointLights[base].xyz;
          let rad = pointLights[base + 1u].rgb;
          let ptExtra = pointLights[base + 2u];
          let ptMaxDist = ptExtra.x;  // 0 = no cutoff
          let ptDecay   = ptExtra.y;  // 0 = no falloff, 2 = physical inverse-square
          let toPoint = lp - hitPos;
          let dist2 = max(dot(toPoint, toPoint), 1e-5);
          let dist = sqrt(dist2);
          // Distance cutoff: skip if ptMaxDist > 0 and hit is beyond it.
          if (ptMaxDist > 0.0 && dist > ptMaxDist) {
            current = current + 1u;
            continue;
          }
          let wi = toPoint / dist;
          let pointShadowRay = Ray(hitPos + normal * 1e-3, wi);
          // SHADOW-01 — ptExtra.z carries the emitter castShadowDisabled flag.
          if (ptExtra.z > 0.5 || !traceAny(pointShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
            let nDotL = max(0.0, dot(normal, wi));
            let brdf = evaluateBrdfFullWithClearcoatNormal(baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation);
            // Ranged-decay falloff: pow(max(dist,1), -ptDecay). decay=0 → attenuation=1;
            // decay=2 → physical inverse-square (matches rad/dist2 at dist≥1).
            let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -ptDecay), ptDecay > 0.01);
            // Delta light (no MIS): compensate the one-of-N selection by /p_select.
            // A3 — spectralise the light radiance at the hero λ (RGB unchanged).
            let radOut = select(rad, spectralEmissionAtHero(rad, heroLambda), params.spectralEnabled != 0u);
            directLi = directLi + throughput * brdf * nDotL * radOut * attenuation * directLightingScale;
          }
        }
        current = current + 1u;
      }
      for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
        if (sumDirectLighting || current == picked) {
          // H51-D: stride 4 (4 vec4 = 16 f32): position, dir+cosOuter, radiance+cosInner, [distance, decay, 0, 0]
          let sb = si * 4u;
          let spos = spotLights[sb].xyz;
          let saxis = spotLights[sb + 1u];
          let sradW = spotLights[sb + 2u];  // .rgb = radiance, .w = cosInner
          let spExtra = spotLights[sb + 3u];
          let spotDir = safe_normalize(saxis.xyz);
          let cosOuter = saxis.w;
          let cosInner = sradW.w;  // cosInner >= cosOuter (inner cone is narrower)
          let srad = sradW.rgb;
          let spMaxDist = spExtra.x;  // 0 = no cutoff
          let spDecay   = spExtra.y;  // 0 = no falloff, 2 = physical inverse-square
          let toSpot = spos - hitPos;
          let dist2 = max(dot(toSpot, toSpot), 1e-5);
          let dist = sqrt(dist2);
          // Distance cutoff: skip if spMaxDist > 0 and hit is beyond it.
          if (spMaxDist > 0.0 && dist > spMaxDist) {
            current = current + 1u;
            continue;
          }
          let wi = toSpot / dist;
          let coneCos = dot(-wi, spotDir);
          if (coneCos >= cosOuter) {
            let spotShadowRay = Ray(hitPos + normal * 1e-3, wi);
            // SHADOW-01 — spExtra.z carries the emitter castShadowDisabled flag.
            if (spExtra.z > 0.5 || !traceAny(spotShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
              let nDotL = max(0.0, dot(normal, wi));
              // Smooth penumbra: smoothstep from cosOuter to cosInner (hard edge when equal).
              let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
              let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -spDecay), spDecay > 0.01);
              let brdf = evaluateBrdfFullWithClearcoatNormal(baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation);
              // Delta light (no MIS): compensate the one-of-N selection by /p_select.
              // A3 — spectralise the spot radiance at the hero λ (RGB unchanged).
              let sradOut = select(srad, spectralEmissionAtHero(srad, heroLambda), params.spectralEnabled != 0u);
              directLi = directLi + throughput * brdf * nDotL * softness * sradOut * attenuation * directLightingScale;
            }
          }
        }
        current = current + 1u;
      }
      // Rect/disc area light loop — both record kinds share the same 4-vec4 stride.
      // Shape discriminator: rectAreaLights[rb+3].w ≈ 0 → rect, ≈ 1 → analytic disc.
      // Rect  sampling: uniform in [-1,1]²; area = 4·|u×v|.
      // Disc  sampling: concentric-disc map; area = π·|u|² (|u| = radius).
      // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
      // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
      for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
        if (sumDirectLighting || current == picked) {
          let rb = ri * 4u;
          let rpos = rectAreaLights[rb].xyz;
          let ru = rectAreaLights[rb + 1u].xyz;
          let rv = rectAreaLights[rb + 2u].xyz;
          let rshape = rectAreaLights[rb + 3u];
          let rr = rshape.rgb;
          let isDisc = abs(rshape.w - 1.0) < 0.5;
          // Sample a point on the emitter surface.
          let xi1 = rand_f32(&rng);
          let xi2 = rand_f32(&rng);
          var lpos: vec3f;
          var area: f32;
          if (isDisc) {
            // D9.11 — Shirley & Chiu 1997 concentric-disc map via shared kernelCore helper.
            // Scale by radius (= |ru|); uniform area measure → pdf = 1/(π·r²).
            let r = length(ru);
            let disc = concentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0));
            lpos = rpos + ru * disc.x + rv * disc.y;
            area = max(PI * r * r, 1e-6);
          } else {
            let u = xi1 * 2.0 - 1.0;
            let v = xi2 * 2.0 - 1.0;
            lpos = rpos + ru * u + rv * v;
            area = max(4.0 * length(cross(ru, rv)), 1e-6);
          }
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = max(dot(normal, wi), 0.0);
          if (nDotL > 0.0) {
            let brdf = evaluateBrdfFullWithClearcoatNormal(baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation);
            let lightNormal = safe_normalize(cross(ru, rv));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let lightPdf = dist2 / max(cosLight * area, 1e-6);
              let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, ior, normal, clearcoatNormal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation);
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
              // SHADOW-01 — rectAreaLights[rb].w carries castShadowDisabled.
              if (rectAreaLights[rb].w > 0.5 || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
                // A3 — spectralise the rect/disc-area radiance at the hero λ.
                let rrOut = select(rr, spectralEmissionAtHero(rr, heroLambda), params.spectralEnabled != 0u);
                directLi = directLi + throughput * brdf * nDotL * rrOut * misWeight / max(lightPdf, 1e-6) * directLightingScale;
              }
            }
          }
        }
        current = current + 1u;
      }
      for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
        if (sumDirectLighting || current == picked) {
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
            let brdf = evaluateBrdfFullWithClearcoatNormal(baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation);
            let lightNormal = safe_normalize(cross(b - a, c - a));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
              let lightPdf = dist2 / max(cosLight * area, 1e-6);
              let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, ior, normal, clearcoatNormal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation);
              // Selection compensated OUTSIDE the MIS (·lightSelectInvPdf) — see
              // the rect-area branch. Keeps the converged mean independent of the
              // selection pdf (tree-vs-uniform means match), variance differs.
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let shadowRay = Ray(hitPos + normal * 1e-3, wi);
              // SHADOW-01 — meshAreaLights[mb+3].w carries castShadowDisabled.
              if (meshAreaLights[mb + 3u].w > 0.5 || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
                // A3 — spectralise the mesh-area radiance at the hero λ.
                let mrOut = select(mr, spectralEmissionAtHero(mr, heroLambda), params.spectralEnabled != 0u);
                directLi = directLi + throughput * brdf * nDotL * mrOut * misWeight / max(lightPdf, 1e-6) * directLightingScale;
              }
            }
          }
        }
        current = current + 1u;
      }
      if ((hasEnvironmentMap() || params.environmentSun.w > 1e-6) && (sumDirectLighting || current == picked)) {
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
            let brdf = evaluateBrdfFullWithClearcoatNormal(baseColor, roughness, metallic, normal, clearcoatNormal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation);
            let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, ior, normal, clearcoatNormal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation);
            // Selection compensated OUTSIDE the MIS (·lightSelectInvPdf) — see the
            // rect-area branch. Mean stays independent of the selection pdf.
            // A3 — spectralise the env radiance at the hero λ (RGB mode unchanged).
            // D3 / Item 8 — per-material envMapIntensity scales both MIS halves:
            // this explicit env NEE/connect path uses materialEnvMapIntensity(matId),
            // and the BSDF-escape env pickup (the no-hit branch above) uses
            // lastEnvMapIntensity (updated from matId each hit). Both halves now
            // carry the same scale → no MIS divergence when envMapIntensity≠1.
            // envMapIntensity == 1 (default) ⇒ scale == 1 ⇒ byte-identical.
            // Ref: THREE.envMapIntensity; pt-webgl2 state.envMapIntensity pattern.
            let envScale = materialEnvMapIntensity(matId);
            let envColorOut = select(envColor, spectralEmissionAtHero(envColor, heroLambda), params.spectralEnabled != 0u) * envScale;
            let misWeight = powerHeuristic(envPdf, brdfPdf);
            directLi = directLi + throughput * brdf * nDotL * envColorOut * misWeight / max(envPdf, 1e-8) * directLightingScale;
          }
        }
      }
      // Sampled mode: each branch multiplies by 1/p_select OUTSIDE the per-light
      // MIS power heuristic, so tree and uniform picks share the same converged
      // mean and differ only in variance. Summed inverse mode visits every branch
      // once and uses scale 1, matching the deterministic direct-light adjoint.
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
        let maxLv = min(params.bdptMaxLightBounces, 8u);
        // lvi=0 is the emitter endpoint: connecting E_bounce directly to it is
        // the same direct-light strategy already estimated by the per-bounce NEE
        // block above. Start at the first scattered light vertex so BDPT adds
        // only the complementary multi-vertex strategies instead of brightening
        // every indirect hit by double-counting secondary NEE.
        for (var lvi = 1u; lvi < maxLv; lvi++) {
          radiance = radiance + evaluateBdptConnection(
            hitPos,
            normal,
            clearcoatNormal,
            wo,
            throughputAtVertex,
            baseColor,
            roughness,
            metallic,
            transmission,
            ior,
            mat.clearcoat,
            mat.clearcoatRoughness,
            mat.sheen,
            mat.sheenRoughness,
            mat.sheenColor,
            mat.iridescence,
            mat.iridescenceIor,
            mat.iridescenceThicknessMin,
            mat.iridescenceThicknessMax,
            mat.specularColor,
            mat.specularIntensity,
            anisoStrength,
            anisoRotation,
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
        mat.clearcoat,
        mat.clearcoatRoughness,
        mat.sheen,
        mat.sheenRoughness,
        mat.sheenColor,
        mat.iridescence,
        mat.iridescenceIor,
        mat.iridescenceThicknessMin,
        mat.iridescenceThicknessMax,
        mat.specularColor,
        mat.specularIntensity,
        anisoStrength,
        anisoRotation,
        throughputAtVertex,
      );
    } else if (caustic == 2u) {
      // A4-progressive: pass pixelIndex so sppmGatherProgressive can read/write
      // the per-pixel (τ, R², N) stats buffer for the Hachisuka update rule.
      // Item 21: heroLambda lets the gather spectralise each photon's RGB flux.
      // Non-spectral path (spectralEnabled=0): heroLambda is unused → byte-identical.
      let sppmReceiverEligible = transmission <= 0.3 &&
        !(metallic > 0.9 && roughness < 0.15);
      if (!sppmGatherUpdated && sppmReceiverEligible) {
        radiance = radiance + photonMapContribution(
          &rng,
          pixelIndex,
          hitPos,
          normal,
          wo,
          baseColor,
          roughness,
          metallic,
          transmission,
          mat.clearcoat,
          mat.clearcoatRoughness,
          mat.sheen,
          mat.sheenRoughness,
          mat.sheenColor,
          mat.iridescence,
          mat.iridescenceIor,
          mat.iridescenceThicknessMin,
          mat.iridescenceThicknessMax,
          mat.specularColor,
          mat.specularIntensity,
          anisoStrength,
          anisoRotation,
          throughputAtVertex,
          heroLambda,
        );
        sppmGatherUpdated = true;
      }
    }

    let bs = sampleNextBounceDirectionWithClearcoatNormal(
      &rng,
      ray.direction,
      hitPos,
      hit.normal,
      normal,
      clearcoatNormal,
      baseColor,
      roughness,
      metallic,
      transmission,
      ior,
      fresnel,
      thinFilmTransmitTint,
      isTranslucent,
      mat.clearcoat,
      mat.clearcoatRoughness,
      mat.sheen,
      mat.sheenRoughness,
      mat.sheenColor,
      anisoStrength,
      anisoRotation,
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
      let scatterPdfFwd = brdfDirectionalPdfFullSampledWithClearcoatNormal(
        baseColor, roughness, metallic, transmission, ior, normal, clearcoatNormal, wo, sampledDir,
        mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
        mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
        mat.specularColor, mat.specularIntensity,
        anisoStrength, anisoRotation);
      // Merged pdfFwd(E_{bounce-1}): swapped-direction reverse density at the
      // PREVIOUS eye vertex toward E_bounce, using the natural next eye direction
      // as wo (PBRT camera[d-1].pdfRev set while at camera[d]). Write into the
      // previous scratch slot (D1 — non-symmetric reverse density).
      if (bounce >= 1u) {
        let toPrev = safe_normalize(bdptPrevPos - hitPos);
        let swappedRev = brdfDirectionalPdfFullSampledWithClearcoatNormal(
          baseColor, roughness, metallic, transmission, ior, normal, clearcoatNormal, sampledDir, toPrev,
          mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
          mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
          mat.specularColor, mat.specularIntensity,
          anisoStrength, anisoRotation);
        bdptEyeStackSetFwd(bounce - 1u, swappedRev);
      }
      bdptPrevScatterPdf = scatterPdfFwd;
      bdptPrevPos = hitPos;
    }

${compositePreamble}${bsdfAreaConnect}${compositeEarlyOut}
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
