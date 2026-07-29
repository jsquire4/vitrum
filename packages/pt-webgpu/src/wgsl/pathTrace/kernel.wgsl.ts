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
 * phase↔light MIS) is emitted when \`volumetricSss\` is true. Production
 * composers enable it for ordinary PT and BDPT; the latter now carries matching
 * eye/light medium vertices, segment densities, and transmittance.
 */
export function composePathTraceKernelWgsl(opts: {
  readonly volumetricSss: boolean;
  /**
   * Compile the BDPT s=n-1/t=1 camera-splat staging and resolve path. The
   * declaration of its atomic RGB buffer lives in bdptCameraSplat.wgsl.ts and
   * is composed only for a bdpt:true full-tier pipeline. */
  readonly bdptCameraSplat?: boolean;
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
  const cameraSplat = opts.bdptCameraSplat === true;
  // BDPT estimator boundary: explicit connections own finite-emitter terminal
  // strategies. The ordinary eye estimator retains distant/environment paths
  // plus camera-visible and delta-hit finite emission. This runtime partition
  // prevents additive double counting; bdptEnabled=0 preserves every old branch.
  // Henyey-Greenstein phase helpers are top-level WGSL functions used only by
  // the volumetric walk; include them only when the walk is compiled in so the
  // non-volumetric test composition carries no SSS symbols.
  const hgHelpers = sss ? PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL : '';
  // The transmissive-material block: full volumetric walk when SSS is on, the
  // legacy Beer-Lambert + forward-scatter-radiance fallback when it is off
  // (kept for explicit non-volumetric test compositions).
  const transmissiveBlock = sss
    ? /* wgsl */ `
    let surfaceMediumBoundary = mediumBoundaryIdentity(
      hit.triIndex, hit.instanceIndex,
    );
    let surfaceCrossesMedium = mat.isTranslucent && transmission > 0.0;
    if (surfaceCrossesMedium && !mediumBoundaryIsValid(surfaceMediumBoundary)) {
      break;
    }
    if (bdptMediumDepth == 0u && !isFrontFace && surfaceCrossesMedium) {
      bdptMediumStack[0] = bdptMediumLayer(
        matId, mat, heroLambda, surfaceMediumBoundary,
      );
      bdptMediumDepth = 1u;
    }
    var incidentIor = 1.0;
    if (bdptMediumDepth > 0u) {
      incidentIor = bdptMediumStack[bdptMediumDepth - 1u].ior;
    } else if (!isFrontFace) {
      incidentIor = max(ior, 1e-4);
    }
    var transmittedIor = max(ior, 1e-4);
    if (!isFrontFace) {
      transmittedIor = 1.0;
      if (surfaceCrossesMedium) {
        if (
          bdptMediumDepth == 0u ||
          !bdptMediumLayerMatchesBoundary(
            bdptMediumStack[bdptMediumDepth - 1u],
            matId,
            surfaceMediumBoundary,
          )
        ) {
          break;
        }
        if (bdptMediumDepth > 1u) {
          transmittedIor = bdptMediumStack[bdptMediumDepth - 2u].ior;
        }
      }
    }
    let surfaceEtaTOverI = transmittedIor / max(incidentIor, 1e-4);
    var bdptEyeSegmentForwardDensity = 1.0;
    var bdptEyeActiveMedium = BDPT_NO_MEDIUM;
    // WS4 volumetric random walk. inMedium is set when the previous bounce
    // refracted INTO this medium (see medium-state update after the bounce
    // sample). σ_t = σ_a + σ_s; σ_a is the host-derived Beer-Lambert
    // absorption (decodeMaterial.sigmaA), σ_s the scattering coefficient.
    // Ref: PBR4e §11 "Volume Scattering"; Henyey-Greenstein 1941.
    if (bdptMediumDepth > 0u) {
      let topIndex = bdptMediumDepth - 1u;
      let eyeMedium = bdptMediumStack[topIndex];
      bdptEyeActiveMedium = eyeMedium.matId;
      let walkSigmaT = max(eyeMedium.sigmaT, vec3f(0.0));
      // Hero-channel σ_t drives the free-flight distance in spectral mode so a
      // single wavelength is tracked per path; otherwise use the max channel
      // (conservative — the densest channel sets the collision rate, the rest
      // ride along via the per-channel transmittance below).
      let heroSigmaT = select(
        max(walkSigmaT.x, max(walkSigmaT.y, walkSigmaT.z)),
        walkSigmaT.x,
        params.spectralEnabled != 0u,
      );
      if (heroSigmaT > 0.0) {
        let xiFlight = rand_f32(&rng);
        let freeFlightDist = -log(max(1.0 - xiFlight, 1e-9)) / heroSigmaT;
        let attenuationDist = min(hit.dist, eyeMedium.remainingDistance);
        if (freeFlightDist < attenuationDist) {
          // Real collision inside the medium BEFORE the surface: scatter.
          let scatterPos = ray.origin + ray.direction * freeFlightDist;
          // Per-channel single-scattering albedo σ_s/σ_t at the chosen flight
          // distance, re-weighted by the ratio of the per-channel pdf to the
          // hero-channel pdf so non-hero channels stay unbiased (spectral MIS).
          let pdfHero = heroSigmaT * exp(-heroSigmaT * freeFlightDist);
          let transmittance = exp(-walkSigmaT * freeFlightDist);
          throughput =
            throughput * eyeMedium.sigmaS * transmittance / max(pdfHero, 1e-9);
          let throughputInMedium = throughput;
          var mediumLayerAtScatter = eyeMedium;
          mediumLayerAtScatter.remainingDistance = max(
            eyeMedium.remainingDistance - freeFlightDist, 0.0,
          );

          // The volume collision is a genuine diffuse SPPM receiver. Gather
          // before phase continuation and claim the same per-path update token
          // used by the later surface gather, so one eye path cannot update its
          // progressive state twice in a frame.
          var sppmGatheredThisMedium = false;
          if (sppmActive && !sppmGatherUpdated) {
            sppmUpdateVolumeProgressive(
              pixelIndex,
              scatterPos,
              -ray.direction,
              eyeMedium.matId,
              throughputInMedium,
              heroLambda,
              heroPdf,
              advancedEstimatorStateInvPdf,
            );
            sppmGatherUpdated = true;
            sppmGatheredThisMedium = true;
          }

          // Conditional direct-light family: BDPT owns finite emitters, while
          // directional/environment transport remains an ordinary eye estimator.
          var mediumLightCount = params.directionalLightCount +
            params.pointLightCount + params.spotLightCount +
            params.rectAreaLightCount + params.meshAreaLightCount;
          if (hasEnvironmentMap()) {
            mediumLightCount = mediumLightCount + 1u;
          }
          let mediumFamilyCount = select(
            mediumLightCount, distantDirectEmitterCount(),
            bdptOwnsFiniteLightFamily,
          );
          if (mediumFamilyCount > 0u) {
            let sumMediumLights = params.directLightingMode == 1u;
            var mediumSelection: DirectLightSelection;
            if (bdptOwnsFiniteLightFamily) {
              mediumSelection = sampleDistantDirectLight(sumMediumLights, &rng);
            } else {
              mediumSelection = sampleCanonicalDirectLight(
                scatterPos, mediumLightCount, sumMediumLights, &rng,
              );
            }
            let finiteBegin = params.directionalLightCount;
            let finiteEnd = finiteBegin + params.pointLightCount +
              params.spotLightCount + params.rectAreaLightCount +
              params.meshAreaLightCount;
            for (
              var mediumFlat = 0u; mediumFlat < mediumLightCount;
              mediumFlat = mediumFlat + 1u
            ) {
              let familyOwnsEmitter = !bdptOwnsFiniteLightFamily ||
                mediumFlat < finiteBegin || mediumFlat >= finiteEnd;
              if (
                familyOwnsEmitter && (sumMediumLights ||
                mediumFlat == mediumSelection.emitterIndex)
              ) {
                radiance = radiance + mediumNeeForEmitter(
                  mediumFlat,
                  select(mediumSelection.invPdf, 1.0, sumMediumLights),
                  scatterPos, ray.direction, mediumLayerAtScatter,
                  &bdptMediumStack, bdptMediumDepth,
                  throughputInMedium, heroLambda,
                  bdptOwnsFiniteLightFamily,
                  bounce,
                  bdptPrevScatterPdf * pdfHero,
                  &rng,
                );
              }
            }
          }

          // Sample the next direction from the HG phase function and continue
          // the walk. The phase-sampled estimator is unbiased (f/pdf = 1); the
          // light it later hits is weighted by the complementary MIS term
          // powerHeuristic(phasePdf, lightPdf) inside the next-bounce emission
          // path, so it balances the NEE term added above (partition of unity).
          let phaseDir = sampleHenyeyGreenstein(&rng, ray.direction, eyeMedium.g);
          let phasePdf = hgPhase(dot(ray.direction, phaseDir), eyeMedium.g);
          radiance = radiance + mediumPhaseEmitterConnection(
            scatterPos, ray.direction, phaseDir, phasePdf, mediumLayerAtScatter,
            &bdptMediumStack, bdptMediumDepth,
            throughputInMedium, heroLambda, !bdptOwnsFiniteLightFamily,
            bdptOwnsFiniteLightFamily,
            bounce,
            bdptPrevScatterPdf * pdfHero,
            &rng,
          );
          prevEventKind = 2u;
          prevDirectionalPdf = phasePdf;
          prevSampleAllowsAreaMis = prevEventKind != 0u && prevDirectionalPdf > 0.0;
          let mediumRemainingAtVertex =
            mediumLayerAtScatter.remainingDistance;
          if (
            bdptOwnsFiniteLightFamily &&
            bounce < min(params.bdptMaxEyeDepth, 8u)
          ) {
            bdptEyeStackStore(
              bounce,
              scatterPos,
              vec3f(0.0),
              0.0,
              bdptPrevScatterPdf * pdfHero,
              false,
              true,
              eyeMedium.g,
              eyeMedium.matId,
              mediumRemainingAtVertex,
              eyeMedium.matId,
              mediumRemainingAtVertex,
              eyeMedium.matId,
              mediumRemainingAtVertex,
            );
            let maxMediumLv = min(params.bdptMaxLightBounces, 8u);
            for (var mediumLvi = 0u; mediumLvi < maxMediumLv; mediumLvi++) {
              radiance = radiance + evaluateBdptConnection(
                scatterPos, vec3f(0.0), vec3f(0.0), -ray.direction,
                throughputInMedium, vec3f(1.0),
                1.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 1.0, vec3f(0.0),
                0.0, 1.0, 0.0, 0.0,
                vec3f(1.0), 1.0,
                0.0, 0.0, bsdfNoThinFilm(), bounce,
                true, eyeMedium.g, eyeMedium.matId, i32(mediumLvi), &rng,
              );
            }
            if (bounce >= 1u) {
              let prevEye = bdptEyeStackLoad(bounce - 1u);
              let currentEye = bdptEyeStackLoad(bounce);
              let toPrevEye = safe_normalize(prevEye.pos - scatterPos);
              let currentPrefixMedium = bdptSelectEndpointMedium(
                currentEye.medium, currentEye.nrm, toPrevEye,
                currentEye.mediumMatId, currentEye.mediumRemainingDistance,
                currentEye.incidentMediumMatId,
                currentEye.incidentMediumRemainingDistance,
                currentEye.transmittedMediumMatId,
                currentEye.transmittedMediumRemainingDistance,
              );
              let previousPrefixMedium = bdptSelectEndpointMedium(
                prevEye.medium, prevEye.nrm, -toPrevEye,
                prevEye.mediumMatId, prevEye.mediumRemainingDistance,
                prevEye.incidentMediumMatId,
                prevEye.incidentMediumRemainingDistance,
                prevEye.transmittedMediumMatId,
                prevEye.transmittedMediumRemainingDistance,
              );
              let reverseDistancePdf = bdptEndpointEdgeDistanceDensity(
                currentPrefixMedium,
                previousPrefixMedium,
                distance(scatterPos, prevEye.pos),
                prevEye.medium,
                heroLambda,
              );
              let reversePhasePdf = phasePdf;
              bdptEyeStackSetFwd(
                bounce - 1u, reversePhasePdf * reverseDistancePdf,
              );
            }
            bdptPrevScatterPdf = phasePdf;
            bdptPrevPos = scatterPos;
          }
          bdptMediumStack[topIndex].remainingDistance = mediumRemainingAtVertex;
          ray.origin = scatterPos;
          ray.direction = phaseDir;

          // A gathered phase event starts the same D-S* ownership prefix as a
          // gathered finite surface event. An ungathered volume event is a new
          // finite scatter and terminates any older receiver prefix.
          sppmReceiverPrefixActive =
            sppmGatheredThisMedium && phasePdf > 0.0;
          sppmOwnedDeltaDepth = 0u;
          mneeReceiverPrefixActive = false;
          mneeOwnedDeltaDepth = 0u;

          if (!bdptOwnsFiniteLightFamily && bounce > 2u) {
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
          bdptEyeSegmentForwardDensity = exp(-heroSigmaT * attenuationDist);
        }
      }
      bdptMediumStack[topIndex].remainingDistance = max(
        eyeMedium.remainingDistance - min(hit.dist, eyeMedium.remainingDistance),
        0.0,
      );
    }`
    : /* wgsl */ `
    let surfaceMediumBoundary = mediumBoundaryIdentity(
      hit.triIndex, hit.instanceIndex,
    );
    let surfaceCrossesMedium = mat.isTranslucent && transmission > 0.0;
    if (surfaceCrossesMedium && !mediumBoundaryIsValid(surfaceMediumBoundary)) {
      break;
    }
    if (bdptMediumDepth == 0u && !isFrontFace && surfaceCrossesMedium) {
      bdptMediumStack[0] = bdptMediumLayer(
        matId, mat, heroLambda, surfaceMediumBoundary,
      );
      bdptMediumDepth = 1u;
    }
    var incidentIor = 1.0;
    if (bdptMediumDepth > 0u) {
      incidentIor = bdptMediumStack[bdptMediumDepth - 1u].ior;
    } else if (!isFrontFace) {
      incidentIor = max(ior, 1e-4);
    }
    var transmittedIor = max(ior, 1e-4);
    if (!isFrontFace) {
      transmittedIor = 1.0;
      if (surfaceCrossesMedium) {
        if (
          bdptMediumDepth == 0u ||
          !bdptMediumLayerMatchesBoundary(
            bdptMediumStack[bdptMediumDepth - 1u],
            matId,
            surfaceMediumBoundary,
          )
        ) {
          break;
        }
        if (bdptMediumDepth > 1u) {
          transmittedIor = bdptMediumStack[bdptMediumDepth - 2u].ior;
        }
      }
    }
    let surfaceEtaTOverI = transmittedIor / max(incidentIor, 1e-4);
    var bdptEyeSegmentForwardDensity = 1.0;
    var bdptEyeActiveMedium = BDPT_NO_MEDIUM;
    if (bdptMediumDepth > 0u) {
      bdptEyeActiveMedium = bdptMediumStack[bdptMediumDepth - 1u].matId;
    }`;

  // Medium-state declarations (only present when the walk is compiled in).

  const mediumSideState = /* wgsl */ `
    var bdptEyeIncidentMedium = bdptNoEndpointMedium();
    if (bdptMediumDepth > 0u) {
      let incidentLayer = bdptMediumStack[bdptMediumDepth - 1u];
      bdptEyeIncidentMedium = BdptEndpointMedium(
        incidentLayer.matId, incidentLayer.remainingDistance,
      );
    }
    var bdptEyeTransmittedMedium = bdptNoEndpointMedium();
    if (surfaceCrossesMedium) {
      if (isFrontFace) {
        let enteredLayer = bdptMediumLayer(
          matId, mat, heroLambda, surfaceMediumBoundary,
        );
        bdptEyeTransmittedMedium = BdptEndpointMedium(
          enteredLayer.matId, enteredLayer.remainingDistance,
        );
      } else {
        if (bdptMediumDepth > 1u) {
          let belowLayer = bdptMediumStack[bdptMediumDepth - 2u];
          bdptEyeTransmittedMedium = BdptEndpointMedium(
            belowLayer.matId, belowLayer.remainingDistance,
          );
        }
      }
    }`;
  const mediumStateDecls = /* wgsl */ `
  var bdptMediumStack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;
  var bdptMediumDepth = 0u;`;

  // Medium-state update after the bounce sample (only when the walk is in).
  const mediumStateUpdate = /* wgsl */ `
    if (bs.enteredMedium) {
      if (bdptMediumDepth >= BDPT_MEDIUM_STACK_LIMIT) { break; }
      bdptMediumStack[bdptMediumDepth] =
        bdptMediumLayer(matId, mat, heroLambda, surfaceMediumBoundary);
      bdptMediumDepth = bdptMediumDepth + 1u;
    } else if (bs.exitedMedium) {
      if (
        bdptMediumDepth == 0u ||
        !bdptMediumLayerMatchesBoundary(
          bdptMediumStack[bdptMediumDepth - 1u],
          matId,
          surfaceMediumBoundary,
        )
      ) {
        break;
      }
      bdptMediumDepth = bdptMediumDepth - 1u;
    }`;

  const preShadeMediumAttenuation = sss
    ? ''
    : /* wgsl */ `
    if (bdptMediumDepth > 0u) {
      let topIndex = bdptMediumDepth - 1u;
      let distanceInMedium = min(
        hit.dist, bdptMediumStack[topIndex].remainingDistance,
      );
      throughput = throughput * exp(
        -bdptMediumStack[topIndex].sigmaA * max(distanceInMedium, 0.0),
      );
      bdptMediumStack[topIndex].remainingDistance = max(
        bdptMediumStack[topIndex].remainingDistance - distanceInMedium, 0.0,
      );
    } else {
      let preMatId = hitMaterialId(hit);
      let preMat = decodeMaterial(preMatId);
      let preFrontFace = hit.frontFace;
      if (!preFrontFace && preMat.isTranslucent) {
        let inferredDistance = materialAttenuationDistance(hit.dist, preMat);
        throughput = throughput * exp(
          -bdptMaterialSigmaA(preMatId, preMat, heroLambda) * inferredDistance,
        );
      }
    }`;

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
  // A1 composite state — read the resolve indirect for THIS pixel once. A pixel
  // the producer contributed to (rpt.a > 0.5) gets the E0-direct-only + composited-
  // indirect split; a producer-DROPPED pixel (specular/transmissive E0 → rpt.a == 0)
  // falls through to the FULL path so glass/mirror primaries still trace their
  // reflection/refraction. Empty when composite is OFF (byte-identical default).
  const compositeStateDecls = composite
    ? /* wgsl */ `  let rptComposite = rpt_result_in[pixelIndex];
  let rptProducerContributed = rptComposite.a > 0.5;
  let advancedPeerEnabled = params.bdptEnabled != 0u || caustic != 0u;
  // ReSTIR reuse and each advanced peer are complete estimators of the same
  // transport integral. Select one estimator for the whole frame: a per-pixel
  // coin is invalid for BDPT t=1 because one source invocation can splat into
  // any target pixel. This integer hash depends only on frame-global uniforms,
  // is identical in every invocation and is independent of the configured path
  // sampling backend. Averaging complete estimators needs no outer 1/p factor.
  var frameEstimatorHash =
    params.frameSeed ^ (params.frameIndex * 0x9e3779b9u);
  frameEstimatorHash =
    (frameEstimatorHash ^ (frameEstimatorHash >> 16u)) * 0x7feb352du;
  frameEstimatorHash =
    (frameEstimatorHash ^ (frameEstimatorHash >> 15u)) * 0x846ca68bu;
  frameEstimatorHash =
    frameEstimatorHash ^ (frameEstimatorHash >> 16u);
  let rptMixtureSelected =
    !advancedPeerEnabled || (frameEstimatorHash & 1u) == 0u;
  let advancedEstimatorSelected =
    advancedPeerEnabled && !rptMixtureSelected;
  let rptCompositeContributed = rptProducerContributed && rptMixtureSelected;
  if (advancedEstimatorSelected) {
    // SPPM's persistent denominator counts every emitted-photon frame. Its
    // gather runs on half the frames in this mixture, so compensate only the
    // newly absorbed flux; the complete peer estimator itself is not scaled.
    advancedEstimatorStateInvPdf = 2.0;
  }
`
    : '';
  const bdptOwnershipExpr = composite
    ? 'params.bdptEnabled != 0u && advancedEstimatorSelected'
    : 'params.bdptEnabled != 0u';
  const mneeActiveDecl = composite
    ? '  let mneeActive = caustic == 1u && advancedEstimatorSelected;\n'
    : '';
  const mneeOwnershipExpr = composite ? 'mneeActive' : 'caustic == 1u';
  const sppmActiveSuffix = composite ? ' && advancedEstimatorSelected' : '';
  // The BSDF→light/env area-MIS connection condition. Identical in both composite
  // and default mode: the BSDF-side MIS connections at E0 must run for ALL pixels
  // (composited or not) because analytic lights are never reconnection vertices,
  // so there is no double-count risk. OFF = the verbatim original.
  const sampleAllowsAreaMisCond = 'sampleAllowsAreaMis';
  const includeMeshAreaLightsExpr = composite ? '!rptCompositeContributed' : 'true';
  const bsdfAreaConnect = /* wgsl */ `    if (${sampleAllowsAreaMisCond}) {
      // Finite BSDF→light connections belong to the explicit BDPT family while
      // it is active. Environment connections remain in the ordinary eye family.
      if (!bdptOwnsFiniteLightFamily) {
        radiance = radiance + bsdfAreaLightConnectionContribution(
          hitPos,
          normal,
          clearcoatNormal,
          wo,
          sampledDir,
          baseColor,
          roughness,
          metallic,
          transmission,
          surfaceEtaTOverI,
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
          thinFilm,
          throughputAtVertex,
          heroLambda,
          ${includeMeshAreaLightsExpr},
          &rng,
        );
      }
      var environmentEscapeMisWeight = -1.0;
      if (bdptOwnsFiniteLightFamily) {
        let environmentGlobalIndex =
          params.directionalLightCount + params.pointLightCount +
          params.spotLightCount + params.rectAreaLightCount +
          params.meshAreaLightCount;
        let envProposalPdf = environmentNeeProposalPdf(sampledDir, normal);
        let sampledBsdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(
          baseColor, roughness, metallic, transmission, surfaceEtaTOverI,
          normal, clearcoatNormal, wo, sampledDir,
          mat.clearcoat, mat.clearcoatRoughness,
          mat.sheen, mat.sheenRoughness,
          mat.iridescence, mat.iridescenceIor,
          mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
          mat.specularColor, mat.specularIntensity,
          anisoStrength, anisoRotation, thinFilm,
        );
        let swappedPdf = bdptMarginalSurfacePdf(
          baseColor, roughness, metallic, transmission, surfaceEtaTOverI,
          normal, clearcoatNormal, sampledDir, wo,
          mat.clearcoat, mat.clearcoatRoughness,
          mat.sheen, mat.sheenRoughness,
          mat.iridescence, mat.iridescenceIor,
          mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
          mat.specularColor, mat.specularIntensity,
          anisoStrength, anisoRotation, thinFilm,
        );
        environmentEscapeMisWeight = bdptInfiniteEyeFamilyWeight(
          0u,
          true,
          false,
          sampledBsdfPdf,
          distantDirectSelectionPdf(environmentGlobalIndex) * envProposalPdf,
          bdptInfiniteRootLaunchPdf(envProposalPdf),
          sampledDir,
          hitPos,
          normal,
          false,
          bdptPrevScatterPdf * bdptEyeSegmentForwardDensity,
          swappedPdf,
          swappedPdf <= 0.0,
          bounce,
        );
      }
      radiance = radiance + bsdfEnvironmentConnectionContribution(
        hitPos,
        normal,
        clearcoatNormal,
        wo,
        sampledDir,
        baseColor,
        roughness,
        metallic,
        transmission,
        surfaceEtaTOverI,
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
        thinFilm,
        throughputAtVertex,
        heroLambda,
        matId,
        environmentEscapeMisWeight,
        &rng,
      );
    }
`;
  // A1 — composite early-out. After E0's emission + E0's full direct (NEE + BSDF-side
  // MIS connections — BOTH halves are now included for composited pixels), terminate
  // the scalar path walk. The megakernel supplies E0's complete
  // direct contribution; the resolve supplies the indirect (first bounce off E0 onward).
  // Double-count-free: analytic lights (rect-area/disc/env/sky/directional) are NOT in
  // the TLAS, so the producer cannot place xs on them and rptComposite.rgb never
  // overlaps the BSDF-side MIS terms. Producer-dropped pixels skip this and continue
  // the FULL path. OFF (default) emits NOTHING (byte-identical).
  const compositeEarlyOut = composite
    ? /* wgsl */ `    if (rptCompositeContributed) {
      break;
    }
`
    : '';
  // Resolve has already reconstructed reused spectral energy with the selected
  // reservoir's hero wavelength.  Add that RGB only AFTER the megakernel has
  // reconstructed its independent current-path scalar with the current lambda;
  // adding before that conversion would relabel history with today's wavelength.
  const compositeFinalAdd = composite
    ? /* wgsl */ `  if (rptCompositeContributed) {
    outRadiance = outRadiance + rptComposite.rgb;
  }
`
    : '';
  const cameraSplatAfterLightBuild = cameraSplat
    ? /* wgsl */ `
    bdptAccumulateCameraSplatStrategies(gid.xy, heroPdf);`
    : '';
  const bdptPrimaryCameraPdfDecl = cameraSplat
    ? /* wgsl */ `  // Forward scatter pdf at the previous eye vertex (the density
  // that produced the current vertex). For primary E_0 this is the exact
  // PerspectiveCamera::Pdf_We directional density. The same density is used by
  // the selected t=1 splat, so ordinary and camera-splat strategies enumerate
  // one coherent MIS denominator.
  var bdptPrevScatterPdf =
    bdptCameraDirectionalPdfForDirection(ray.direction);`
    : /* wgsl */ `  // Forward scatter pdf at the previous eye vertex (the density that produced
  // the current vertex). For the primary hit E_0 the "previous vertex" is the
  // pinhole camera; its importance directional pdf is modelled as 1.0 (We for
  // an aperture-less pinhole — the one vertex without an aperture model). This
  // replaces the old hardcoded eyePdfFwd=1.0 for all SCENE-surface vertices,
  // where the real BSDF scatter pdf now flows in.
  var bdptPrevScatterPdf = 1.0;`;
  const bdptEyeStackPrimaryPdfComment = cameraSplat
    ? /* wgsl */ `      // (the PerspectiveCamera::Pdf_We density for the primary hit). pdfFwd
      // is filled one iteration later by the swapped-direction reverse density
      // (and overridden by the connection straddle terms when this vertex is
      // E_e / E_{e-1}).`
    : /* wgsl */ `      // (camera importance 1.0 for the primary hit). pdfFwd is filled one
      // iteration later by the swapped-direction reverse density (and overridden
      // by the connection straddle terms when this vertex is E_e / E_{e-1}).`;
  const cameraSplatFrameHelpers = cameraSplat
    ? /* wgsl */ `fn bdptWriteCameraSampleAux(
  gid: vec3u,
  firstHitValid: bool,
  firstHitPos: vec3f,
  firstHitNormal: vec3f,
  firstHitAlbedo: vec3f,
  firstHitDepth: f32,
) {
  if (firstHitValid) {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(firstHitNormal * 0.5 + vec3f(0.5), firstHitDepth));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(firstHitAlbedo, 1.0));
    let ndc = projectToNdc(firstHitPos, params.viewProj);
    let prevNdc = projectToNdc(firstHitPos, params.prevViewProj);
    let motionPx = (ndc - prevNdc) * 0.5 * vec2f(f32(params.width), f32(params.height));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(motionPx, 0.0, 1.0));
  } else {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(0.5, 1.0, 0.5, 0.0));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
  }
}

fn bdptStageCameraSample(
  gid: vec3u,
  radiance: vec3f,
  firstHitValid: bool,
  firstHitPos: vec3f,
  firstHitNormal: vec3f,
  firstHitAlbedo: vec3f,
  firstHitDepth: f32,
) {
  bdptAtomicAddCameraRgb(
    gid.y * params.width + gid.x,
    max(radiance, vec3f(0.0)),
  );
  bdptWriteCameraSampleAux(
    gid,
    firstHitValid,
    firstHitPos,
    firstHitNormal,
    firstHitAlbedo,
    firstHitDepth,
  );
}

fn bdptResolveCameraSample(gid: vec3u, radiance: vec3f) {
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
  textureStore(varianceTexture, vec2i(gid.xy), vec4f(varL, 0.0, 0.0, 0.0));
}`
    : '';
  const frameAccumulationCall = cameraSplat
    ? /* wgsl */ `  bdptStageCameraSample(
    gid,
    outRadiance,
    firstHitValid,
    firstHitPos,
    firstHitNormal,
    firstHitAlbedo,
    firstHitDepth,
  );`
    : /* wgsl */ `  accumulateFrame(
    gid,
    outRadiance,
    firstHitValid,
    firstHitPos,
    firstHitNormal,
    firstHitAlbedo,
    firstHitDepth,
  );`;
  const cameraSplatResolveEntryPoint = cameraSplat
    ? /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn bdptResolveCameraSplats(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let pixelIndex = gid.y * params.width + gid.x;
  bdptResolveCameraSample(gid, bdptLoadCameraRgb(pixelIndex));
}
`
    : '';

  return /* wgsl */ `${rptResultBinding}
${hgHelpers}
${PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL}

fn sampleDirectionalCone(rng: ptr<function, PtRngState>, axisIn: vec3f, angularDiameter: f32) -> vec3f {
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
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(firstHitNormal * 0.5 + vec3f(0.5), firstHitDepth));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(firstHitAlbedo, 1.0));
    let ndc = projectToNdc(firstHitPos, params.viewProj);
    let prevNdc = projectToNdc(firstHitPos, params.prevViewProj);
    let motionPx = (ndc - prevNdc) * 0.5 * vec2f(f32(params.width), f32(params.height));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(motionPx, 0.0, 1.0));
  } else {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(0.5, 1.0, 0.5, 0.0));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
  }
  textureStore(varianceTexture, vec2i(gid.xy), vec4f(varL, 0.0, 0.0, 0.0));
}
${cameraSplatFrameHelpers}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  var ray = generatePrimaryRay(gid.x, gid.y, jitter);
  let primaryRayOrigin = ray.origin;
  let pixelIndex = gid.y * params.width + gid.x;
  let caustic = causticMode();
  var advancedEstimatorStateInvPdf = 1.0;
${compositeStateDecls}

  // BDPT eye-subpath scratch stack — bind this pixel for the deeply-nested
  // stack helpers (bdptEyeStackStore/Load) used by the full §10.3 connection.
  let bdptOwnsFiniteLightFamily = ${bdptOwnershipExpr};
${bdptPrimaryCameraPdfDecl}
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
  // 0=camera/delta, 1=surface finite, 2=volume phase.
  var prevEventKind = 0u;
  var prevDirectionalPdf = 0.0;

  var heroLambda = params.heroLambdaNm;
  var heroPdf = params.heroPdf;
  if (params.spectralEnabled != 0u) {
    // SPPM photons and eye gathers must evaluate the same wavelength. Use one
    // frame-global hero only for SPPM; other modes retain decorrelated pixels.
    let hero = select(
      sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng)),
      sppmFrameHeroSample(),
      caustic == 2u,
    );
    heroLambda = hero.x;
    heroPdf = hero.y;
  }
  if (bdptOwnsFiniteLightFamily) {
    bdptSetInvocationHeroLambda(heroLambda);
    bdptBuildInvocationLightSubpath(gid.xy);${cameraSplatAfterLightBuild}
  }

  // PTWG-04: sppmPixelStats has separate surface and volume records per pixel,
  // but one eye path may mutate at most one record per frame. Later bounces keep
  // tracing without shrinking another measure's radius/N.
  var sppmGatherUpdated = false;
${mneeActiveDecl}  let sppmActive = caustic == 2u${sppmActiveSuffix};
  var sppmReceiverPrefixActive = false;
  var sppmOwnedDeltaDepth = 0u;
  // Exact MNEE ownership state for an ordinary eye continuation. Once a finite
  // receiver has evaluated MNEE, one to configuredMax consecutive delta events
  // terminate at a mesh-area emitter through MNEE only. Longer and mixed
  // prefixes fall back to the ordinary eye estimator without suppression.
  var mneeReceiverPrefixActive = false;
  var mneeOwnedDeltaDepth = 0u;

  var radiance = vec3f(0.0);
  // In spectral SPPM mode the persistent tau buffer stores wavelength samples
  // reconstructed into RGB. Keep that RGB estimate outside scalar hero-path
  // radiance so it is never reconstructed a second time at today's wavelength.
  var sppmRgbRadiance = vec3f(0.0);
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
    let sppmOwnsCurrentEmission = sppmActive && sppmReceiverPrefixActive && sppmOwnedDeltaDepth > 0u;
    let alphaTraceOrigin = ray.origin;
    var alphaAdvance = 0.0;
    var hit = traceClosest(ray, 1e-4, INFINITY);
    // P2 alpha-test pass-through: a baseColor-texture alpha mask/blend hit is
    // "not there" — advance the ray past it and re-trace, WITHOUT consuming a
    // scatter bounce. Opaque materials return false on the first test. The
    // scene-derived bound covers every distinct straight-ray surface hit; a
    // hit beyond it signals corrupt/unbounded traversal and fails the path dark.
    let alphaSurfaceHitLimit = sceneSurfaceHitLimit();
    var alphaSurfaceHitCount = 0u;
    var alphaTraversalValid = true;
    loop {
      if (!hit.didHit) { break; }
      // Observe the final miss after exactly alphaSurfaceHitLimit pass-through
      // surfaces before treating a further hit as impossible/corrupt.
      if (alphaSurfaceHitCount >= alphaSurfaceHitLimit) {
        alphaTraversalValid = false;
        break;
      }
      alphaSurfaceHitCount = alphaSurfaceHitCount + 1u;
      if (!alphaTestPassThrough(
        hitMaterialId(hit), hit.triIndex, hit.baryVW, hit.instanceIndex, &rng,
      )) { break; }
      let alphaStep = hit.dist + 1e-4;
      alphaAdvance = alphaAdvance + alphaStep;
      ray.origin = ray.origin + ray.direction * alphaStep;
      hit = traceClosest(ray, 1e-4, INFINITY);
    }
    ray.origin = alphaTraceOrigin;
    if (!alphaTraversalValid) { break; }
    if (hit.didHit) { hit.dist = hit.dist + alphaAdvance; }
    let mneeOwnsCurrentEmission = ${mneeOwnershipExpr} &&
      mneeReceiverPrefixActive && mneeOwnedDeltaDepth > 0u &&
      mneeOwnedDeltaDepth <= min(params.mneeMaxChainLength, 8u) &&
      hit.didHit && hit.triIndex < params.triangleCount;
    if (!hit.didHit) {
      // A BSDF sample eligible for area/env MIS was already evaluated by
      // bsdfEnvironmentConnectionContribution at the previous vertex. Adding the
      // raw miss term as well would count the same path twice. Camera misses and
      // delta/specular-transmission escapes do not run that connection helper and
      // therefore keep the ordinary unweighted environment pickup.
      if (!prevSampleAllowsAreaMis && !sppmOwnsCurrentEmission) {
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
      }
      break;
    }

${preShadeMediumAttenuation}
${composeShadePrologueWgsl(SHADE_PROLOGUE_EMISSIVE_COMMENT_FULL, SHADE_PROLOGUE_BASE_COLOR_TEX_APPLY_FULL, SHADE_PROLOGUE_EMISSIVE_TEX_APPLY_FULL, SHADE_PROLOGUE_ORM_TEX_APPLY_FULL, SHADE_PROLOGUE_NORMAL_MAP_APPLY_FULL, SHADE_PROLOGUE_AO_APPLY_FULL, SHADE_PROLOGUE_LIGHT_MAP_APPLY_FULL, SHADE_PROLOGUE_BUMP_MAP_APPLY_FULL, SHADE_PROLOGUE_TRANSMISSION_MAP_APPLY_FULL, SHADE_PROLOGUE_VOLUME_THICKNESS_MAP_APPLY_FULL, SHADE_PROLOGUE_EXTENSION_LOBE_TEX_APPLY_FULL, SHADE_PROLOGUE_CLEARCOAT_NORMAL_MAP_APPLY_FULL, ' && !mneeOwnsCurrentEmission')}
    // Item 8 — record this surface's envMapIntensity for the forward env escape
    // pickup on the NEXT iteration (mirrors pt-webgl2 state.envMapIntensity update).
    lastEnvMapIntensity = materialEnvMapIntensity(matId);
    // Item 7 — per-hit anisotropy (used in NEE eval/pdf + sampleNextBounceDirection).
    let anisoStrength = materialAnisotropy(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
    let anisoRotation = materialAnisotropyRotation(matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex);
    let throughputAtVertex = throughput;
${transmissiveBlock}
    let cosThetaO = max(0.0, dot(normal, wo));
    let f0Base = materialSpecularF0(
      baseColor, metallic, surfaceEtaTOverI,
      mat.specularColor, mat.specularIntensity,
    );
    let f0 = iridescenceModifiedF0(
      f0Base,
      mat.iridescence,
      mat.iridescenceIor,
      mat.iridescenceThicknessMin,
      mat.iridescenceThicknessMax,
      cosThetaO,
    );
    let fresnel = materialSpecularFresnelSchlick(
      cosThetaO, f0, metallic, mat.specularIntensity,
    );

    // SPPM owns only the D-S*-L family. A pure-delta current vertex keeps that
    // ownership for its terminal light connection; a finite/glossy vertex ends
    // the family and ordinary NEE continues to own the new connection.
    let sppmOwnsCurrentDirect = sppmOwnsCurrentEmission &&
      !bsdfHasFiniteConnectionSupport(
        roughness, metallic, transmission, mat.clearcoat, mat.sheen,
      );
    var lightCount = 0u;
    // N-directional: every packed record remains in the selection domain.
    // A positive authored irradiance, however small, is still a real light;
    // zero-valued records evaluate to zero without changing flat indexing.
    lightCount = lightCount + params.directionalLightCount;
    lightCount = lightCount + params.pointLightCount;
    lightCount = lightCount + params.spotLightCount;
    lightCount = lightCount + params.rectAreaLightCount;
    lightCount = lightCount + params.meshAreaLightCount;
    if (hasEnvironmentMap()) {
      lightCount = lightCount + 1u;
    }
    let directFamilyCount = select(
      lightCount, distantDirectEmitterCount(), bdptOwnsFiniteLightFamily,
    );
    // Coherent thin film replaces the bare Fresnel inside the same finite GGX
    // interface. Direct lighting and every connection estimator therefore use
    // the ordinary BSDF family with the shared layered response.
    if (directFamilyCount > 0u && !sppmOwnsCurrentDirect) {
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
      var lightSelection: DirectLightSelection;
      if (bdptOwnsFiniteLightFamily) {
        lightSelection = sampleDistantDirectLight(sumDirectLighting, &rng);
      } else {
        lightSelection = sampleCanonicalDirectLight(
          hitPos, lightCount, sumDirectLighting, &rng,
        );
      }
      let picked = lightSelection.emitterIndex;
      let lightSelectInvPdf = lightSelection.invPdf;
      let directLightingScale = select(
        lightSelectInvPdf, 1.0, sumDirectLighting,
      );
      var current = 0u;
      var directLi = vec3f(0.0);
      // N-directional loop: each record in directionalLights[] is 2 vec4f:
      //   [di*2+0]: towardLight.xyz, angularDiameter
      //   [di*2+1]: irradiance.rgb,  mean_irradiance
      // The packed directional array is authoritative for both this surface
      // loop and the in-medium NEE loop above.
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
          let directOffsetNormal = select(-normal, normal, dot(normal, sampleDir) > 0.0);
          let shadowRay = Ray(hitPos + directOffsetNormal * 1e-3, sampleDir);
          if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY, &rng)) {
            let nDotL = abs(dot(normal, sampleDir));
            // H52: evaluateBrdfFull adds clearcoat/sheen/iridescence lobes;
            // zero-default → identical to evaluateBrdf when all scalars are 0.
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, sampleDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation, thinFilm, false);
            // A3 — spectralise the directional irradiance at the hero λ (RGB unchanged).
            let dIrrOut = select(dIrrMean.rgb, spectralEmissionAtHero(dIrrMean.rgb, heroLambda), params.spectralEnabled != 0u);
            var distantMisWeight = 1.0;
            if (bdptOwnsFiniteLightFamily) {
              let directionIsDelta = angDiam <= 0.0;
              var directionPdf = 1.0;
              if (!directionIsDelta) {
                let coneQuarterSin = sin(angDiam * 0.25);
                directionPdf =
                  1.0 / (4.0 * PI * coneQuarterSin * coneQuarterSin);
              }
              let selectionPdf = select(
                1.0 / max(lightSelectInvPdf, 1e-20),
                1.0,
                sumDirectLighting,
              );
              let swappedPdf = bdptMarginalSurfacePdf(
                baseColor, roughness, metallic, transmission,
                surfaceEtaTOverI, normal, clearcoatNormal, sampleDir, wo,
                mat.clearcoat, mat.clearcoatRoughness,
                mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor,
                mat.iridescenceThicknessMin,
                mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation, thinFilm,
              );
              distantMisWeight = bdptInfiniteEyeFamilyWeight(
                1u,
                false,
                false,
                0.0,
                selectionPdf * directionPdf,
                bdptInfiniteRootLaunchPdf(directionPdf),
                sampleDir,
                hitPos,
                normal,
                false,
                bdptPrevScatterPdf * bdptEyeSegmentForwardDensity,
                swappedPdf,
                swappedPdf <= 0.0,
                bounce,
              );
            }
            directLi = directLi + throughput * brdf * nDotL * dIrrOut *
              directLightingScale * distantMisWeight;
          }
        }
        current = current + 1u;
      }
      for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
        if (!bdptOwnsFiniteLightFamily && (sumDirectLighting || current == picked)) {
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
          let pointOffsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
          let pointShadowRay = Ray(hitPos + pointOffsetNormal * 1e-3, wi);
          // SHADOW-01 — ptExtra.z carries the emitter castShadowDisabled flag.
          if (ptExtra.z > 0.5 || !traceAny(pointShadowRay, 1e-4, max(dist - 2e-3, 1e-3), &rng)) {
            let nDotL = abs(dot(normal, wi));
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation, thinFilm, false);
            // Ranged-decay falloff: pow(max(dist,1), -ptDecay). decay=0 → attenuation=1;
            // decay=2 → physical inverse-square (matches rad/dist2 at dist≥1).
            let attenuation = pointSpotDistanceAttenuation(dist, ptMaxDist, ptDecay);
            // Delta light (no MIS): compensate the one-of-N selection by /p_select.
            // A3 — spectralise the light radiance at the hero λ (RGB unchanged).
            let radOut = select(rad, spectralEmissionAtHero(rad, heroLambda), params.spectralEnabled != 0u);
            directLi = directLi + throughput * brdf * nDotL * radOut * attenuation * directLightingScale;
          }
        }
        current = current + 1u;
      }
      for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
        if (!bdptOwnsFiniteLightFamily && (sumDirectLighting || current == picked)) {
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
            let spotOffsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
            let spotShadowRay = Ray(hitPos + spotOffsetNormal * 1e-3, wi);
            // SHADOW-01 — spExtra.z carries the emitter castShadowDisabled flag.
            if (spExtra.z > 0.5 || !traceAny(spotShadowRay, 1e-4, max(dist - 2e-3, 1e-3), &rng)) {
              let nDotL = abs(dot(normal, wi));
              // Smooth penumbra: smoothstep from cosOuter to cosInner (hard edge when equal).
              let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
              let attenuation = pointSpotDistanceAttenuation(dist, spMaxDist, spDecay);
              let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation, thinFilm, false);
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
      // Disc  sampling: concentric-disc map; area = π·|u×v|.
      // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
      // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
      for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
        if (!bdptOwnsFiniteLightFamily && (sumDirectLighting || current == picked)) {
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
            // The affine unit-disc map has area Jacobian |ru×rv|.
            let disc = concentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0));
            lpos = rpos + ru * disc.x + rv * disc.y;
            area = PI * length(cross(ru, rv));
          } else {
            let u = xi1 * 2.0 - 1.0;
            let v = xi2 * 2.0 - 1.0;
            lpos = rpos + ru * u + rv * v;
            area = 4.0 * length(cross(ru, rv));
          }
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = abs(dot(normal, wi));
          if (nDotL > 0.0 && area > 0.0) {
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation, thinFilm, false);
            let lightNormal = safe_normalize(cross(ru, rv));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let lightPdf = dist2 / (cosLight * area);
              let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation, thinFilm);
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
              let rectOffsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
              let shadowRay = Ray(hitPos + rectOffsetNormal * 1e-3, wi);
              // SHADOW-01 — rectAreaLights[rb].w carries castShadowDisabled.
              if (rectAreaLights[rb].w > 0.5 || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3), &rng)) {
                // A3 — spectralise the rect/disc-area radiance at the hero λ.
                let rrOut = select(rr, spectralEmissionAtHero(rr, heroLambda), params.spectralEnabled != 0u);
                directLi = directLi + throughput * brdf * nDotL * rrOut * misWeight / lightPdf * directLightingScale;
              }
            }
          }
        }
        current = current + 1u;
      }
      for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
        if (!bdptOwnsFiniteLightFamily && (sumDirectLighting || current == picked)) {
          let mb = meshAreaLightBase(mi);
          let a = meshAreaLights[mb].xyz;
          let b = meshAreaLights[mb + 1u].xyz;
          let c = meshAreaLights[mb + 2u].xyz;
          let r1 = rand_f32(&rng);
          let r2 = rand_f32(&rng);
          let su = sqrt(r1);
          let uu = 1.0 - su;
          let vv = r2 * su;
          let ww = 1.0 - uu - vv;
          let lpos = a * uu + b * vv + c * ww;
          let mr = sampleMeshAreaLightRadiance(
            mi, vec3f(uu, vv, ww), lpos,
          );
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = abs(dot(normal, wi));
          let area = 0.5 * length(cross(b - a, c - a));
          if (nDotL > 0.0 && area > 0.0) {
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation, thinFilm, false);
            let lightNormal = safe_normalize(cross(b - a, c - a));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let lightPdf = dist2 / (cosLight * area);
              let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation, thinFilm);
              // Selection compensated OUTSIDE the MIS (·lightSelectInvPdf) — see
              // the rect-area branch. Keeps the converged mean independent of the
              // selection pdf (tree-vs-uniform means match), variance differs.
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let meshOffsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
              let shadowRay = Ray(hitPos + meshOffsetNormal * 1e-3, wi);
              // SHADOW-01 — row 3.w carries castShadowDisabled.
              if (meshAreaLights[mb + 3u].w > 0.5 || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3), &rng)) {
                // A3 — spectralise the mesh-area radiance at the hero λ.
                let mrOut = select(mr, spectralEmissionAtHero(mr, heroLambda), params.spectralEnabled != 0u);
                directLi = directLi + throughput * brdf * nDotL * mrOut * misWeight / lightPdf * directLightingScale;
              }
            }
          }
        }
        current = current + 1u;
      }
      if (hasEnvironmentMap() && (sumDirectLighting || current == picked)) {
        var envDir = vec3f(0.0, 1.0, 0.0);
        var envColor = vec3f(0.0);
        var envPdf = 0.0;
        let envSample = sampleEnvironmentImportance(&rng);
        if (envSample.pdf > 0.0) {
          envDir = envSample.wi;
          envColor = envSample.value;
          envPdf = envSample.pdf;
        } else {
          envDir = uniformSphere(
            vec2f(rand_f32(&rng), rand_f32(&rng)),
          );
          envColor = sampleEnvironmentColor(envDir);
          envPdf = 0.25 * INV_PI;
        }
        let nDotL = abs(dot(normal, envDir));
        if (nDotL > 0.0) {
          let envOffsetNormal = select(-normal, normal, dot(normal, envDir) > 0.0);
          let shadowRay = Ray(hitPos + envOffsetNormal * 1e-3, envDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY, &rng)) {
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation, thinFilm, false);
            let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, clearcoatNormal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              anisoStrength, anisoRotation, thinFilm);
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
            var misWeight = powerHeuristic(envPdf, brdfPdf);
            if (bdptOwnsFiniteLightFamily) {
              let selectionPdf = select(
                1.0 / max(lightSelectInvPdf, 1e-20),
                1.0,
                sumDirectLighting,
              );
              let swappedPdf = bdptMarginalSurfacePdf(
                baseColor, roughness, metallic, transmission,
                surfaceEtaTOverI, normal, clearcoatNormal, envDir, wo,
                mat.clearcoat, mat.clearcoatRoughness,
                mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor,
                mat.iridescenceThicknessMin,
                mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                anisoStrength, anisoRotation, thinFilm,
              );
              misWeight = bdptInfiniteEyeFamilyWeight(
                1u,
                true,
                false,
                brdfPdf,
                selectionPdf * envPdf,
                bdptInfiniteRootLaunchPdf(envPdf),
                envDir,
                hitPos,
                normal,
                false,
                bdptPrevScatterPdf * bdptEyeSegmentForwardDensity,
                swappedPdf,
                swappedPdf <= 0.0,
                bounce,
              );
            }
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

${mediumSideState}
    if (bdptOwnsFiniteLightFamily &&
        bounce < min(params.bdptMaxEyeDepth, 8u)) {
      // Push this eye vertex (E_bounce) onto the scratch stack BEFORE connecting:
      // pdfRev = forward scatter pdf at the previous vertex that produced E_bounce
${bdptEyeStackPrimaryPdfComment}
      bdptEyeStackStore(
        bounce, hitPos, normal, 0.0,
        bdptPrevScatterPdf * bdptEyeSegmentForwardDensity,
        false,
        false,
        0.0,
        bdptEyeActiveMedium,
        bdptEyeIncidentMedium.remainingDistance,
        bdptEyeIncidentMedium.matId,
        bdptEyeIncidentMedium.remainingDistance,
        bdptEyeTransmittedMedium.matId,
        bdptEyeTransmittedMedium.remainingDistance,
      );
      let maxLv = min(params.bdptMaxLightBounces, 8u);
      // The primary eye vertex and emitter endpoint are part of this family;
      // ordinary finite NEE is suppressed above while BDPT owns the family.
      for (var lvi = 0u; lvi < maxLv; lvi++) {
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
          surfaceEtaTOverI,
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
          thinFilm,
          bounce,
          false,
          0.0,
          bdptEyeActiveMedium,
          i32(lvi),
          &rng,
        );
      }
    }

    var sppmGatheredThisVertex = false;
    var mneeSampledThisVertex = false;
    // MNEE owns its solved D-S*-L manifold family. In SPPM mode the BDPT
    // connection evaluator additionally rejects any connection whose interior
    // light prefix contains a delta event, assigning those E-…-D-S*-L paths to
    // SPPM while BDPT retains the complementary finite-prefix family.
    let mneeReceiverEligible = bsdfHasFiniteConnectionSupport(
        roughness, metallic, transmission, mat.clearcoat, mat.sheen,
      );
    if (${mneeOwnershipExpr} && mneeReceiverEligible) {
      radiance = radiance + manifoldNeeContribution(
        &rng,
        hitPos,
        normal,
        clearcoatNormal,
        wo,
        baseColor,
        roughness,
        metallic,
        transmission,
        surfaceEtaTOverI,
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
        thinFilm,
        heroLambda,
        throughputAtVertex,
      );
      mneeSampledThisVertex = true;
    } else if (sppmActive) {
      // A4-progressive: pass pixelIndex so photonMapUpdateProgressive can write
      // the per-pixel (τ, R², N) stats buffer for the Hachisuka update rule.
      // Item 21: heroLambda lets the gather spectralise each photon's RGB flux.
      // Non-spectral path (spectralEnabled=0): heroLambda is unused → byte-identical.
      let sppmReceiverEligible =
        (1.0 - metallic) * (1.0 - transmission) > 0.0;
      if (!sppmGatherUpdated && sppmReceiverEligible) {
        photonMapUpdateProgressive(
          pixelIndex,
          hitPos,
          normal,
          clearcoatNormal,
          wo,
          baseColor,
          roughness,
          metallic,
          transmission,
          surfaceEtaTOverI,
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
          thinFilm,
          throughputAtVertex,
          heroLambda,
          heroPdf,
          advancedEstimatorStateInvPdf,
        );
        sppmGatherUpdated = true;
        sppmGatheredThisVertex = true;
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
      surfaceEtaTOverI,
      false,
      fresnel,
      mat.iridescence,
      mat.iridescenceIor,
      mat.iridescenceThicknessMin,
      mat.iridescenceThicknessMax,
      mat.specularColor,
      mat.specularIntensity,
      thinFilm,
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
    prevEventKind = select(0u, 1u, sampleAllowsAreaMis);
    prevDirectionalPdf = select(0.0, bs.sampledEventPdf, sampleAllowsAreaMis);
    prevSampleAllowsAreaMis = prevEventKind != 0u && prevDirectionalPdf > 0.0;
    if (sppmActive) {
      if (sppmGatheredThisVertex) {
        sppmReceiverPrefixActive =
          !bs.sampledIsDelta && bs.sampledEventPdf > 0.0;
        sppmOwnedDeltaDepth = 0u;
      } else if (sppmReceiverPrefixActive) {
        if (bs.sampledIsDelta && bs.sampledEventPdf > 0.0) {
          sppmOwnedDeltaDepth = sppmOwnedDeltaDepth + 1u;
        } else {
          sppmReceiverPrefixActive = false;
          sppmOwnedDeltaDepth = 0u;
        }
      }
    }
    if (${mneeOwnershipExpr}) {
      let mneeDeltaEvent = bs.sampledIsDelta && bs.sampledEventPdf > 0.0;
      let mneeMaxDepth = min(params.mneeMaxChainLength, 8u);
      if (mneeSampledThisVertex) {
        mneeReceiverPrefixActive = mneeDeltaEvent && mneeMaxDepth > 0u;
        mneeOwnedDeltaDepth = select(0u, 1u, mneeReceiverPrefixActive);
      } else if (mneeReceiverPrefixActive) {
        if (mneeDeltaEvent && mneeOwnedDeltaDepth < mneeMaxDepth) {
          mneeOwnedDeltaDepth = mneeOwnedDeltaDepth + 1u;
        } else {
          mneeReceiverPrefixActive = false;
          mneeOwnedDeltaDepth = 0u;
        }
      }
    }
${mediumStateUpdate}

    if (bdptOwnsFiniteLightFamily) {
      // Delta is a property of the sampled event, not of the material class.
      // bdptMergedVertex temporarily treats an explicit connection endpoint as
      // non-delta while it recomputes the connection-straddling densities.
      bdptEyeStackSetSpec(bounce, bs.sampledIsDelta);
      // The forward scatter pdf of the chosen next direction at E_bounce — fed
      // to the next iteration as E_{bounce+1}'s reverse density. (eyePdfFwd is
      // now this real value, not the old hardcoded 1.0.)
      let scatterPdfFwd = bs.sampledEventPdf;
      // Merged pdfFwd(E_{bounce-1}): swapped-direction reverse density at the
      // PREVIOUS eye vertex toward E_bounce, using the natural next eye direction
      // as wo (PBRT camera[d-1].pdfRev set while at camera[d]). Write into the
      // previous scratch slot (D1 — non-symmetric reverse density).
      if (bounce >= 1u) {
        let prevEyeForReverse = bdptEyeStackLoad(bounce - 1u);
        let currentEyeForReverse = bdptEyeStackLoad(bounce);
        let toPrev = safe_normalize(bdptPrevPos - hitPos);
        let currentPrefixMedium = bdptSelectEndpointMedium(
          currentEyeForReverse.medium, currentEyeForReverse.nrm, toPrev,
          currentEyeForReverse.mediumMatId,
          currentEyeForReverse.mediumRemainingDistance,
          currentEyeForReverse.incidentMediumMatId,
          currentEyeForReverse.incidentMediumRemainingDistance,
          currentEyeForReverse.transmittedMediumMatId,
          currentEyeForReverse.transmittedMediumRemainingDistance,
        );
        let previousPrefixMedium = bdptSelectEndpointMedium(
          prevEyeForReverse.medium, prevEyeForReverse.nrm, -toPrev,
          prevEyeForReverse.mediumMatId,
          prevEyeForReverse.mediumRemainingDistance,
          prevEyeForReverse.incidentMediumMatId,
          prevEyeForReverse.incidentMediumRemainingDistance,
          prevEyeForReverse.transmittedMediumMatId,
          prevEyeForReverse.transmittedMediumRemainingDistance,
        );
        let reverseDistancePdf = bdptEndpointEdgeDistanceDensity(
          currentPrefixMedium, previousPrefixMedium,
          distance(hitPos, prevEyeForReverse.pos),
          prevEyeForReverse.medium,
          heroLambda,
        );
        var swappedRev = 0.0;
        if (!bs.sampledIsDelta) {
          swappedRev = bdptMarginalSurfacePdf(
            baseColor,
            roughness,
            metallic,
            transmission,
            bs.sampledEtaTOverI,
            normal,
            clearcoatNormal,
            sampledDir,
            toPrev,
            mat.clearcoat,
            mat.clearcoatRoughness,
            mat.sheen,
            mat.sheenRoughness,
            mat.iridescence,
            mat.iridescenceIor,
            mat.iridescenceThicknessMin,
            mat.iridescenceThicknessMax,
            mat.specularColor,
            mat.specularIntensity,
            anisoStrength,
            anisoRotation,
            thinFilm,
          );
        }
        bdptEyeStackSetFwd(
          bounce - 1u, swappedRev * reverseDistancePdf,
        );
      }
      bdptPrevScatterPdf = scatterPdfFwd;
      bdptPrevPos = hitPos;
    }

${bsdfAreaConnect}${compositeEarlyOut}
    // Both BDPT subpaths use fixed bounded depths. Keeping q=1 on the eye path
    // avoids asymmetric, camera-throughput-dependent RR factors that cannot be
    // reconstructed for the reverse strategies in bdptMISWeightFull.
    if (!bdptOwnsFiniteLightFamily && bounce > 2u) {
      let rr = russianRoulette(&rng, throughput);
      if (!rr.survives) { break; }
      throughput = throughput * rr.throughputMul;
    }
  }

  var outRadiance = radiance;
  if (params.spectralEnabled != 0u) {
    outRadiance = heroWavelengthToRgb(heroLambda, luminance(radiance), heroPdf);
  }
  // tau/Ne already contains surface/volume/no-receiver occurrence frequency.
  // Publish both persistent measures on every frame selected for the complete
  // SPPM estimator; scaling this cumulative value by the estimator-mixture PDF
  // would bias the outer complete-estimator mixture.
  if (sppmActive) {
    sppmRgbRadiance = sppmCurrentProgressiveEstimate(pixelIndex);
  }
  outRadiance = outRadiance + sppmRgbRadiance;
${compositeFinalAdd}

${frameAccumulationCall}
}${cameraSplatResolveEntryPoint}
`;
}

/**
 * Default full-tier kernel composition. Production full-tier variants,
 * including BDPT, compile the volumetric SSS walk; specialized test
 * compositions may still omit it explicitly.
 */
export const PT_WEBGPU_PATH_TRACE_KERNEL_WGSL = composePathTraceKernelWgsl({
  volumetricSss: true,
});
