import { describe, expect, it } from 'vitest';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../../../shaders/nrcIndependentSuffix.wgsl.ts';
import { RIS_GI_NRC_BODY, buildRisGiNrcModule } from '../../../shaders/risGiNrc.wgsl.ts';
import { composeWgsl } from '../../../pipeline/wgslComposer.ts';
import { WGSL_MODULES } from '../../../pipeline/wgslModules.ts';
import { RESTIR_GI_DIELECTRIC_SUFFIX_WGSL } from '../../../shaders/risGi.wgsl.ts';

const CFG = {
  levels: 8,
  featuresPerEntry: 2,
  oneBlobBins: 8,
  width: 64,
  outWidth: 3,
  hidden: 6,
} as const;

describe('NRC independent suffix teacher', () => {
  it('uses an independently traced target rather than DDGI/cache distillation', () => {
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain('fn nrcTraceIndependentSuffix(');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).not.toContain('sampleDDGIAtPoint(');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).not.toContain('nrcQueryRadiance(');
    expect(RIS_GI_NRC_BODY).toContain('nrcTrackTarget = nrcTraceIndependentSuffix(');
    expect(RIS_GI_NRC_BODY).not.toContain('nrcTrackTarget = ddgiLo');
  });

  it('pins bounded depth, Russian roulette, and the matching defensive mixture PDF', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('const NRC_TEACHER_MAX_VERTICES: u32 = 4u;');
    expect(src).toContain('const NRC_TEACHER_RR_START: u32 = 2u;');
    expect(src).toContain('specularMixProbability * pdfSpec +');
    expect(src).toContain('(1.0 - specularMixProbability) * pdfCos;');
    expect(src).toMatch(
      /nrcTeacherMaterialResponse\(\s+currentHit,\s+payload,\s+normal,\s+wo,\s+nextDir,\s+closureTransmission,/,
    );
    expect(src).toContain(') / proposalPdf / reflectionBranchPdf,');
    expect(src).toContain(
      'nextThroughput = nextThroughput * (1.0 / survive);',
    );
    expect(src).toContain(
      'let survive = represented_bernoulli_probability_f32(',
    );
  });

  it('covers authored light families and mapped surface radiance', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('sampleEmitterIdx(count, rand_f32(rng))');
    expect(src).toContain('emitterCdfPmf(count, lid)');
    expect(src).toContain('sampleEmitterLeAtXi(emitter, xi)');
    expect(src).toContain('nrcTeacherAnalyticNee(');
    expect(src).toContain('nrcTeacherSunNee(');
    expect(src).toContain('restir_gi_surface_emission_for_hit(hit)');
    expect(src).toContain('sampleLightMap(hit)');
    expect(src).toContain('envRadiance(nextDir)');
  });

  it('matches live suffix local-source ownership, layer, and volume semantics', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    const helperStart = src.indexOf('fn nrcTeacherLocalSourceForHit(');
    const helperEnd = src.indexOf(
      'fn nrcTraceIndependentSuffixForChannel(', helperStart,
    );
    expect(helperStart).toBeGreaterThan(0);
    expect(helperEnd).toBeGreaterThan(helperStart);
    const helper = src.slice(helperStart, helperEnd);
    expect(helper).toContain('restir_gi_surface_emission_for_hit(hit)');
    expect(helper).toContain('payload.albedo * INV_PI * sampleLightMap(hit) *');
    expect(helper).toContain('(1.0 - clamp(transmission, 0.0, 1.0));');
    expect(helper).toContain(
      '(surfaceEmission + bakedDiffuse) * payload.layerTransmission;',
    );
    expect(helper).toContain('let proxySource = applyHomogeneousVolumeSingleScatter(');
    expect(helper).toContain(
      'return select(proxySource, rawSource, explicitSegmentVolume);',
    );

    const trace = src.slice(helperEnd);
    expect(trace).toMatch(
      /let includeEmitterEmission =\s+depth == 0u \|\| arrivedWithoutNeeOwner \|\| !areaNeeOwnsHit;/,
    );
    expect(trace).toMatch(/nrcTeacherLocalSourceForHit\(\s+currentHit,/);
  });

  it('passes full colored incident radiance through directional volume', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    const materialStart = src.indexOf('fn nrcTeacherMaterialResponse(');
    const responseStart = src.indexOf(
      'fn nrcTeacherDirectionalIncidentResponse(', materialStart,
    );
    const responseEnd = src.indexOf('fn nrcTeacherShadowTint(', responseStart);
    expect(materialStart).toBeGreaterThan(0);
    expect(responseStart).toBeGreaterThan(materialStart);
    expect(responseEnd).toBeGreaterThan(responseStart);
    const material = src.slice(materialStart, responseStart);
    const response = src.slice(responseStart, responseEnd);
    expect(material).toContain('return applyMaterialLayerTransmissionToBrdf(');
    expect(material).not.toContain(
      'applyHomogeneousVolumeSingleScatterDirectional(',
    );
    expect(response).toContain(
      'let rawResponse = incidentRadiance * nrcTeacherMaterialResponse(',
    );
    expect(response).toContain(
      'let proxyResponse = applyHomogeneousVolumeSingleScatterDirectional(\n    rawResponse,',
    );
    expect(response).toContain(
      'return select(proxyResponse, rawResponse, explicitSegmentVolume);',
    );

    expect(src).toContain('Le * visibility * geometry / pdfArea,');
    expect(src).toContain(
      'l1.xyz * visibility * cone * attenuation * estimatorWeight,',
    );
    expect(src).toContain('vec3f(ubo.sunIntensity) * visibility,');
  });

  it('keeps dielectric reflection and the complementary opaque share at partial transmission', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('fn nrcTeacherExactInterfaceReflection(');
    expect(src).toContain('let authoredSpecular = sampleSpecularControls(hit);');
    expect(src).toContain('let authoredIridescence = sampleIridescenceControls(hit);');
    expect(src).toContain('let signedInterfaceResidual = authoredExact -');
    expect(src).toContain('let opaqueBody = opaqueClosure - authoredBase;');
    expect(src).toContain('return interfaceReflection.exact *');
    expect(src).toContain(
      '(opaqueBody + signedInterfaceResidual) *',
    );
    expect(src).toContain('payload.reflectionLayerTransmission +');
    expect(src).toContain('payload.layerTransmission;');
    // Non-transmissive materials retain the canonical rich-material path.
    expect(src).toContain('evalGGXReflectionWithTransmissionMix(');
    expect(src).toContain(
      'evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(',
    );
    expect(src).toContain('applyMaterialLayerTransmissionToBrdf(');
    expect(src).toContain('payload.reflectionLayerTransmission,');
    expect(src).toContain(
      'transmissionPhysicalWeight / (1.0 + transmissionPhysicalWeight);',
    );
    expect(src).toContain('let reflectionBranchPdf = 1.0 - transmissionBranchPdf;');
    expect(src).toContain('transmissionPhysicalWeight / transmissionBranchPdf');
    expect(src).toContain('/ reflectionBranchPdf,');
    expect(src).not.toContain('fn nrcTeacherExactTirReflectionCorrection(');
  });

  it('closes every complementary opaque partition after a paid bulk entry', () => {
    const rawTransmission = 0.37;
    const persistentReflection = 0.19;
    const opaqueBody = 0.61;
    const bakedIrradiance = 0.43;
    const pairedPaidExit = true;
    const closureTransmission = pairedPaidExit ? 1 : rawTransmission;
    const reflection = persistentReflection +
      (1 - closureTransmission) * opaqueBody;
    const bakedDiffuse = (1 - closureTransmission) * bakedIrradiance;

    expect(reflection).toBe(persistentReflection);
    expect(reflection).toBeGreaterThan(0);
    expect(bakedDiffuse).toBe(0);
    expect(
      persistentReflection + (1 - rawTransmission) * opaqueBody,
    ).toBeGreaterThan(reflection);

    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain(
      'let closureTransmission = transmissionPhysicalWeight;',
    );
    expect(src).toMatch(
      /nrcTeacherLocalSourceForHit\([\s\S]*?closureTransmission,[\s\S]*?includeEmitterEmission,/,
    );
    for (const helper of [
      'nrcTeacherAreaNee(',
      'nrcTeacherAnalyticNee(',
      'nrcTeacherSunNee(',
    ]) {
      const call = src.slice(src.lastIndexOf(helper));
      expect(call).toMatch(/payload,\s+closureTransmission,/);
    }
    expect(src).toMatch(
      /nrcTeacherMaterialResponse\([\s\S]*?nextDir,[\s\S]*?closureTransmission,/,
    );
  });

  it('samples exact rough anisotropic dielectric lobes in the authored tangent frame', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    const helperStart = src.indexOf(
      'fn nrcTeacherSampleDielectricLobe(',
    );
    const helperEnd = src.indexOf('// Raw bvh_beer reference.', helperStart);
    expect(helperStart).toBeGreaterThan(0);
    expect(helperEnd).toBeGreaterThan(helperStart);
    const helper = src.slice(helperStart, helperEnd);
    expect(helper).toContain('ggxSampleVndfTangentAnisotropic(');
    expect(helper).toContain('ggxDielectricTransmissionAxes(');
    expect(helper).toContain('distributionGGXAnisotropic(');
    expect(helper).toContain('dielectricInterfaceTransmissionRgb(');
    expect(helper).toContain('etaIncident / etaTarget');
    expect(src).toMatch(
      /nrcTeacherSampleDielectricLobe\([\s\S]*?payload\.anisotropyTangent,[\s\S]*?payload\.anisotropyBitangent,[\s\S]*?payload\.rough,[\s\S]*?etaIncident,[\s\S]*?etaTarget,/,
    );
    expect(src).not.toContain('safe_normalize(reflect(incident, orientedNormal))');
  });

  it('applies selected-channel thin-film and face-layer transfer inside the exact lobe', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('let film = materialThinFilmResponse(');
    expect(src).toContain('transmittanceRgb = film.transmittance;');
    expect(src).toContain('reflectanceRgb = film.reflectance;');
    expect(src).toContain('let layerTransferRgb = faceLayerTransmission(layer);');
    expect(src).toContain('out.weightRgb = transmittanceRgb * directionalBaseWeight;');
  });

  it('traces correlated RGB geometry with each represented channel IOR', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('fn nrcTraceIndependentSuffixForChannel(');
    expect(src).toContain('let materialIor = materialDispersionIorRgb(');
    expect(src).toContain(
      'let etaIncident = nrcTeacherChannel(etaIncidentRgb, channel);',
    );
    expect(src).toContain('let sharedRng = (*rng);');
    expect(src).toMatch(/nrcTraceIndependentSuffixForChannel\([\s\S]*?0u,[\s\S]*?&rngR/);
    expect(src).toMatch(/nrcTraceIndependentSuffixForChannel\([\s\S]*?1u,[\s\S]*?&rngG/);
    expect(src).toMatch(/nrcTraceIndependentSuffixForChannel\([\s\S]*?2u,[\s\S]*?&rngB/);
    expect(src).toContain('return vec3f(radianceR.r, radianceG.g, radianceB.b);');
    expect(src).not.toMatch(
      /materialDispersionIorRgb\([\s\S]{0,200}\)\.g/,
    );
  });

  it('evaluates direct lighting at the terminal opaque vertex before truncation', () => {
    const trace = NRC_INDEPENDENT_SUFFIX_WGSL.slice(
      NRC_INDEPENDENT_SUFFIX_WGSL.indexOf(
        'fn nrcTraceIndependentSuffixForChannel(',
      ),
    );
    const direct = trace.indexOf('nrcTeacherAreaNee(');
    const terminal = trace.indexOf(
      'if (depth + 1u >= NRC_TEACHER_MAX_VERTICES) { break; }',
      direct,
    );
    expect(direct).toBeGreaterThan(0);
    expect(terminal).toBeGreaterThan(direct);
  });

  it('charges Beer per geometric medium segment and pairs authored exits', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('fn nrcTeacherBeerForSegment(');
    const beerStart = src.indexOf('fn nrcTeacherMappedBeerReference(');
    const beerEnd = src.indexOf('fn nrcTeacherThicknessMapScale(', beerStart);
    const beerHelper = src.slice(beerStart, beerEnd);
    expect(beerHelper).toContain('hit.indices.w % BVH_BEER_TEX_WIDTH');
    expect(beerHelper).not.toContain('BVH_MATERIAL_TEX_WIDTH');
    expect(src).toContain('transportDistance / referenceThickness');
    expect(src).toContain(
      'let authoredTransmissionTopology = materialHasTransmission(scalar.a);',
    );
    expect(src).toContain(
      'authoredTransmissionTopology && currentBoundaryId != 0u;',
    );
    expect(src).toContain(
      'let thinSheet = authoredTransmissionTopology && !bulkMedium;',
    );
    expect(src).not.toContain('effectiveBulkThickness');
    expect(src).toContain(
      'mediumAuthoredThickness[mediumDepth] = authoredThickness;',
    );
    expect(src).toContain('var mediumThicknessMapScale: array<f32, 4>;');
    expect(src).toContain(
      'nrcTeacherThicknessMapScale(currentHit);',
    );
    expect(src).toContain(
      'referenceThickness * clamp(thicknessMapScale, 0.0, 1.0),',
    );
    expect(src).toContain(
      'mediumScatter[mediumDepth] = payload.volumeScattering;',
    );
    expect(src).toContain('var mediumTransmissionPaid: array<u32, 4>;');
    expect(src).toContain('mediumTransmissionPaid[top] != 0u');
    expect(src).toContain('transmissionPhysicalWeight = 1.0;');
    expect(src).toContain('mediumTransmissionPaid[mediumDepth] = 1u;');
    expect(src).toContain('restirGiSuffixSegmentTransfer(');
    expect(src).toContain('segmentAlbedo,');
    expect(src).toContain(
      'throughput = restirGiSuffixDiagonalTransfer(vec3f(1.0));',
    );
    expect(src).toContain('let entering = currentHit.side >= 0.0;');
    expect(src).toContain('mediumMaterialId[top] == currentBoundaryId &&');
    expect(src).toContain(
      'authoredTransmissionTopology && currentRepresentedId == 0u',
    );
    expect(src).toContain(
      'mediumInstance[top] == currentRepresentedId;',
    );
    expect(src).toContain('if (mediumDepth != 0u) { break; }');
    expect(src).not.toContain(
      'nextThroughput = nextThroughput * nrcTeacherBeerTint(currentHit)',
    );
  });

  it('seeds every alpha-aware containing bulk shell before tracing the suffix', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    const classifyStart = src.indexOf('fn nrcTeacherClassifyContainingMedia(');
    const classifyEnd = src.indexOf('// Receiver-local sources', classifyStart);
    expect(classifyStart).toBeGreaterThan(0);
    expect(classifyEnd).toBeGreaterThan(classifyStart);
    const classify = src.slice(classifyStart, classifyEnd);
    expect(classify).toContain('materialShadowClassifyContainingMedia(');
    expect(classify).toContain('initialPos,');
    expect(classify).toContain('safe_normalize(initialWo),');
    expect(classify).toContain(
      'out.materialId[seed] = classified.state.materialId[seed];',
    );
    expect(classify).toContain(
      'out.instance[seed] = classified.state.instance[seed];',
    );
    expect(classify).toContain('out.transmissionPaid[seed] = 0u;');
    expect(classify).toContain(
      'out.scatter[seed] = classified.state.scattering[seed];',
    );
    expect(classify).toContain(
      'out.mappedBeerReference[seed] = classified.state.tint[seed];',
    );
    expect(classify).toContain(
      'classified.state.thicknessMapScale[seed];',
    );
    expect(classify).not.toContain('sampleTransmissionMapForHit(');

    const trace = src.slice(src.indexOf('fn nrcTraceIndependentSuffixForChannel('));
    const vertexLoop = trace.indexOf(
      'for (var depth = 0u; depth < NRC_TEACHER_MAX_VERTICES;',
    );
    const copySeed = trace.indexOf(
      'mediumMaterialId[seed] = containingMedia.materialId[seed];',
    );
    expect(copySeed).toBeGreaterThan(0);
    expect(vertexLoop).toBeGreaterThan(copySeed);
    expect(trace).toContain('mediumDepth = containingMedia.depth;');

    const wrapper = src.slice(src.indexOf('fn nrcTraceIndependentSuffix('));
    const classifyCall = wrapper.indexOf('nrcTeacherClassifyContainingMedia(');
    const redTrace = wrapper.indexOf(
      'let radianceR = nrcTraceIndependentSuffixForChannel(',
    );
    expect(classifyCall).toBeGreaterThan(0);
    expect(redTrace).toBeGreaterThan(classifyCall);
    expect(wrapper).toContain(
      'initialHit, initialSourceFeature,\n    initialPos, initialWo, containingMedia, 0u, &rngR,',
    );
  });

  it('uses exact reciprocal Fresnel and total-internal-reflection in same-side and sheet families', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    const reflectionStart = src.indexOf(
      'fn nrcTeacherExactInterfaceReflection(',
    );
    const reflectionEnd = src.indexOf(
      'fn nrcTeacherMaterialResponse(', reflectionStart,
    );
    const reflection = src.slice(reflectionStart, reflectionEnd);
    expect(reflection).toContain('dielectricInterfaceTransmissionRgb(');
    expect(reflection).toContain('let film = materialThinFilmResponse(');
    expect(reflection).toContain('exactReflectance = film.reflectance;');
    expect(reflection).toContain('sin2Target >= vec3f(1.0),');
    expect(reflection).toContain('out.exact = exactReflectance * reflectionScale;');
    expect(reflection).toContain('out.canonicalSchlick = canonicalFresnel * reflectionScale;');
    expect(reflection).not.toContain('payload.layerTransmission');
    for (const oracle of [reflection, RESTIR_GI_DIELECTRIC_SUFFIX_WGSL]) {
      expect(oracle).toContain('dielectricInterfaceTransmissionRgb(');
      expect(oracle).toContain('materialThinFilmResponse(');
      expect(oracle).toContain('sin2Target');
    }

    expect(src).toContain('if (tir && collapseTirToReflection)');
    expect(src).toContain('sampleReflection = true;');
    expect(src).toContain(
      'slabSelectedPdf, 1.0, slabLobe.tir != 0u,',
    );
    expect(src).toContain(
      'slabLobe.kind == NRC_TEACHER_DIELECTRIC_EVENT_TRANSMISSION',
    );
    expect(src).toContain('applyNormalMapForSideForHit(');
    expect(src).toContain('slabFrontFacing = !slabFrontFacing;');
  });

  it('pins exact unpolarised dielectric energy away from the Schlick approximation', () => {
    const cosIncident = 0.5;
    const etaIncident = 1;
    const etaTarget = 1.5;
    const eta = etaIncident / etaTarget;
    const sin2Target = eta * eta * (1 - cosIncident * cosIncident);
    const cosTarget = Math.sqrt(1 - sin2Target);
    const rs = (etaIncident * cosIncident - etaTarget * cosTarget) /
      (etaIncident * cosIncident + etaTarget * cosTarget);
    const rp = (etaTarget * cosIncident - etaIncident * cosTarget) /
      (etaTarget * cosIncident + etaIncident * cosTarget);
    const exactReflectance = 0.5 * (rs * rs + rp * rp);
    const exactTransmittance = 1 - exactReflectance;
    const f0 = ((etaTarget - etaIncident) /
      (etaTarget + etaIncident)) ** 2;
    const schlick = f0 + (1 - f0) * (1 - cosIncident) ** 5;

    expect(exactReflectance).toBeCloseTo(0.08918671280221276, 14);
    expect(exactReflectance + exactTransmittance).toBeCloseTo(1, 15);
    expect(Math.abs(exactReflectance - schlick)).toBeGreaterThan(0.019);
  });

  it('keeps the signed rich-material residual finite and non-negative without final clamping', () => {
    const schlick = (cosine: number, f0: number) =>
      f0 + (1 - f0) * (1 - cosine) ** 5;
    const dielectric = (cosine: number, etaI: number, etaT: number) => {
      const sin2Target = (etaI / etaT) ** 2 * (1 - cosine * cosine);
      if (sin2Target >= 1) return 1;
      const cosTarget = Math.sqrt(1 - sin2Target);
      const rs = (etaI * cosine - etaT * cosTarget) /
        (etaI * cosine + etaT * cosTarget);
      const rp = (etaT * cosine - etaI * cosTarget) /
        (etaT * cosine + etaI * cosTarget);
      return 0.5 * (rs * rs + rp * rp);
    };
    const films = [
      { present: false, reflectance: 0, transmittance: 1 },
      { present: true, reflectance: 0.01, transmittance: 0.95 },
      { present: true, reflectance: 0.65, transmittance: 0.25 },
    ] as const;
    let cases = 0;
    for (const transmission of [0, 0.25, 1]) {
      for (const [etaI, etaT] of [[1, 1.5], [1.5, 1]] as const) {
        const microfacetCos = 0.5;
        const canonicalF0 = ((etaT - etaI) / (etaT + etaI)) ** 2;
        const canonicalFresnel = schlick(microfacetCos, canonicalF0);
        const bareExact = dielectric(microfacetCos, etaI, etaT);
        for (const roughness of [0.05, 0.5, 1]) {
          const alpha = Math.max(0.01, roughness) ** 2;
          for (const anisotropy of [0, 0.8]) {
            const aspect = Math.sqrt(1 - 0.9 * anisotropy);
            const alphaX = alpha / aspect;
            const alphaY = alpha * aspect;
            const sinTheta = Math.sqrt(1 - microfacetCos ** 2);
            const g1 = (2 * microfacetCos) /
              (microfacetCos + Math.sqrt(
                alphaX ** 2 * sinTheta ** 2 + microfacetCos ** 2,
              ));
            const distribution = 1 / (Math.PI * alphaX * alphaY);
            const rawScale = distribution * g1 * g1 /
              (4 * microfacetCos);
            for (const metallic of [0, 1]) {
              for (const specularFactor of [0, 1]) {
                const albedo = 0.35;
                const authoredF0 = canonicalF0 * specularFactor *
                  (1 - metallic) + albedo * metallic;
                const authoredFresnel = schlick(
                  microfacetCos, authoredF0,
                );
                for (const clearcoat of [0, 1]) {
                  for (const sheen of [0, 1]) {
                    const baseAttenuation =
                      (1 - 0.04 * clearcoat) * (1 - 0.2 * sheen);
                    const reflectionScale = rawScale * baseAttenuation;
                    const authoredBase = authoredFresnel * reflectionScale;
                    const diffuse = (1 - authoredFresnel) *
                      (1 - metallic) * albedo / Math.PI *
                      microfacetCos * baseAttenuation;
                    const positiveLayerLobes =
                      clearcoat * 0.01 + sheen * 0.01;
                    const opaqueClosure = authoredBase + diffuse +
                      positiveLayerLobes;
                    const opaqueBody = opaqueClosure - authoredBase;
                    for (const film of films) {
                      const tir = (etaI / etaT) ** 2 *
                        (1 - microfacetCos ** 2) >= 1;
                      const exactFresnel = tir
                        ? 1
                        : film.present ? film.reflectance : bareExact;
                      const exact = exactFresnel * reflectionScale;
                      const authoredExact = canonicalFresnel > 1e-6
                        ? exactFresnel * authoredFresnel /
                          canonicalFresnel * reflectionScale
                        : exact;
                      const signedResidual = authoredExact - exact;
                      const faceTransfer = 0.9;
                      const bodyTransfer = faceTransfer *
                        (film.present ? film.transmittance : 1);
                      const response = exact * faceTransfer +
                        (opaqueBody + signedResidual) *
                          (1 - transmission) * bodyTransfer;
                      expect(Number.isFinite(response)).toBe(true);
                      expect(response).toBeGreaterThanOrEqual(-1e-12);
                      cases += 1;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    expect(cases).toBe(1728);
    const body = NRC_INDEPENDENT_SUFFIX_WGSL.slice(
      NRC_INDEPENDENT_SUFFIX_WGSL.indexOf('fn nrcTeacherMaterialResponse('),
      NRC_INDEPENDENT_SUFFIX_WGSL.indexOf(
        'fn nrcTeacherDirectionalIncidentResponse(',
      ),
    );
    expect(body).not.toContain('max(opaqueBody');
    expect(body).not.toMatch(/return\s+max\(/);
  });

  it('matches visible unlit face-layer semantics at a teacher terminal', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('if (decodeIsUnlitMaterial(materialWord))');
    expect(src).toContain(
      'payload.albedo * payload.layerTransmission;',
    );
  });

  it('caps independent suffix traces to one collision-free candidate per record slot', () => {
    expect(RIS_GI_NRC_BODY).toContain('let nrcTeacherStride = max(1u, 1u +');
    expect(RIS_GI_NRC_BODY).toContain('let nrcTeacherSlot = pixelIdxGi / nrcTeacherStride;');
    expect(RIS_GI_NRC_BODY).toContain('if (!nrcFired && nrcTeacherEligible)');
    expect(RIS_GI_NRC_BODY).toContain('nrcWriteRecord(\n      nrcTeacherSlot,');
  });

  it('keeps glass out of a key that has no IOR/transmission coordinate', () => {
    expect(RIS_GI_NRC_BODY).toContain(
      '!materialHasTransmission(xsTransmission)',
    );
    expect(RIS_GI_NRC_BODY).not.toContain('xsTransmission <= 0.3');
    const gateStart = RIS_GI_NRC_BODY.indexOf(
      'nrcShouldTerminateIntoCache(aX, a0, nrcCfg.spreadC)',
    );
    const gateEnd = RIS_GI_NRC_BODY.indexOf(
      '} else {\n        Lo = ddgiLo;',
      gateStart,
    );
    expect(gateStart).toBeGreaterThan(0);
    expect(gateEnd).toBeGreaterThan(gateStart);
    const gatedCachePath = RIS_GI_NRC_BODY.slice(gateStart, gateEnd);
    expect(gatedCachePath).toContain('!materialHasTransmission(xsTransmission)');
    expect(gatedCachePath).toContain('Lo = select(');
    expect(gatedCachePath).toContain('nrcQueryRadiance(');
    expect(gatedCachePath).toContain('nrcTrackTarget = nrcTraceIndependentSuffix(');
    expect(gatedCachePath).toContain('var teacherRng = pcgInit(');
  });

  it('composes all teacher bindings and helpers into the production NRC shader', () => {
    const src = composeWgsl(buildRisGiNrcModule(CFG), WGSL_MODULES);
    expect(src).toContain('@group(1) @binding(2) var<storage, read> sceneLightingArena');
    expect(src).toContain('fn sceneLoadEmitter(');
    expect(src).toContain('fn sceneLoadEmitterCdf(');
    expect(src).not.toMatch(/var<storage,\s*read>\s+emitters\b/);
    expect(src).not.toMatch(/var<storage,\s*read>\s+emitterCdf\b/);
    expect(src).toContain('@group(1) @binding(13) var analytic_lights');
    expect(src).toContain('fn sampleEmitterLeAtXi(');
    expect(src).toContain('fn nrc_teacherPointSpotAttenuation(');
  });
});

describe('defensive-mixture estimator oracle', () => {
  it('does not charge thin-film transmission to the reflected closure', () => {
    const faceTransmission = 0.8;
    const filmReflectance = 0.3;
    const filmTransmittance = 0.5;
    const reflectedClosure = filmReflectance;
    const baseClosure = 0.7;
    const mixedClosure = reflectedClosure + baseClosure;
    const reflectionLayerTransmission = faceTransmission;
    const layerTransmission = faceTransmission * filmTransmittance;

    const splitResponse =
      Math.max(mixedClosure - reflectedClosure, 0) * layerTransmission +
      reflectedClosure * reflectionLayerTransmission;
    expect(reflectedClosure * reflectionLayerTransmission).toBeCloseTo(0.24, 14);
    expect(baseClosure * layerTransmission).toBeCloseTo(0.28, 14);
    expect(splitResponse).toBeCloseTo(0.52, 14);
    expect(reflectedClosure * layerTransmission).toBeCloseTo(0.12, 14);
    expect(splitResponse).not.toBeCloseTo(
      mixedClosure * layerTransmission,
      14,
    );
  });

  it('recovers Lambertian hemispherical reflectance when every sample divides by the mixture PDF', () => {
    // An abstract two-proposal analogue of the shader's cosine + VNDF mixture.
    // Proposal A is cosine-weighted, proposal B uniform-hemisphere; both have
    // full support. The target integral of rho/pi * cos(theta) is exactly rho.
    const rho = 0.73;
    const alpha = 0.5;
    const count = 200_000;
    let state = 0x12345678;
    const random = (): number => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x1_0000_0000;
    };
    let sum = 0;
    for (let i = 0; i < count; i++) {
      const chooseA = random() < alpha;
      const u = random();
      const cosTheta = chooseA ? Math.sqrt(1 - u) : u;
      const pCos = cosTheta / Math.PI;
      const pUniform = 1 / (2 * Math.PI);
      const pMix = alpha * pCos + (1 - alpha) * pUniform;
      sum += (rho / Math.PI) * cosTheta / pMix;
    }
    expect(sum / count).toBeCloseTo(rho, 2);
  });

  it('preserves reflection at full transmission and exactly recovers both material branches', () => {
    const persistentReflection = 0.17;
    const opaqueBody = 0.2;
    const transmitted = 0.81;
    for (const transmission of [0.01, 0.25, 0.7, 1]) {
      const reflection =
        persistentReflection + (1 - transmission) * opaqueBody;
      const transmissionPdf = transmission / (1 + transmission);
      const reflectionPdf = 1 - transmissionPdf;
      const reflectionEstimate = reflection / reflectionPdf;
      const transmissionEstimate =
        transmission * transmitted / transmissionPdf;
      const expectation =
        reflectionPdf * reflectionEstimate +
        transmissionPdf * transmissionEstimate;
      expect(expectation).toBeCloseTo(
        reflection + transmission * transmitted,
        14,
      );
      expect(reflectionPdf).toBeGreaterThan(0);
      if (transmission === 1) {
        expect(reflection).toBe(persistentReflection);
      }
    }
    expect(1 / (1 + 1)).toBe(0.5);
    expect(1 - 1 / (1 + 1)).toBe(0.5);
  });
});

describe('NRC bulk-medium transfer oracle', () => {
  it('applies mapped absorption and RGB scattering over the geometric segment exactly once', () => {
    const attenuationColor = [0.21, 0.47, 0.83] as const;
    const scatteringCoefficient = [0.12, 0.31, 0.56] as const;
    const attenuationDistance = 3.7;
    const authoredThickness = 2.4;
    const thicknessMapFactor = 0.35;
    const segmentLength = 1.6;

    // bvh_beer stores attenuationColor^(authoredThickness / distance).
    const packedBeer = attenuationColor.map((channel) =>
      channel ** (authoredThickness / attenuationDistance));
    // applyThicknessMapToBeerTint raises that reference tint to G. Stable
    // authored thickness remains the denominator, so the map modulates sigma_a
    // once without changing whether the material is a closed bulk medium.
    const mappedBeer = packedBeer.map((channel) =>
      channel ** thicknessMapFactor);
    const shaderTransfer = mappedBeer.map((channel, index) =>
      channel ** (segmentLength / authoredThickness) *
      Math.exp(-scatteringCoefficient[index]! * segmentLength));
    const expectedTransfer = attenuationColor.map((channel, index) =>
      channel ** (thicknessMapFactor * segmentLength / attenuationDistance) *
      Math.exp(-scatteringCoefficient[index]! * segmentLength));

    for (let channel = 0; channel < 3; channel += 1) {
      expect(shaderTransfer[channel]).toBeCloseTo(expectedTransfer[channel]!, 14);
    }
    const zeroMapAbsorption = packedBeer.map((channel) => channel ** 0)
      .map((channel) => channel ** (segmentLength / authoredThickness));
    expect(zeroMapAbsorption).toEqual([1, 1, 1]);

    const spectralCoefficient = [0.08, 0.41, 0.93] as const;
    const spectralTransfer = spectralCoefficient.map((coefficient) =>
      Math.exp(-coefficient * segmentLength * thicknessMapFactor));
    const unscaledSpectralTransfer = spectralCoefficient.map((coefficient) =>
      Math.exp(-coefficient * segmentLength));
    expect(spectralTransfer).not.toEqual(unscaledSpectralTransfer);
    expect(spectralCoefficient.map((coefficient) =>
      Math.exp(-coefficient * segmentLength * 0))).toEqual([1, 1, 1]);
  });

  it('retains the live suffix cross-channel single-scatter source', () => {
    const absorption = [0.8, 0.9, 0.7] as const;
    const sigmaS = [0.5, 0, 0] as const;
    const albedo = [1, 0, 0] as const;
    const distance = 1;
    const sigmaA = absorption.map((value) => -Math.log(value) / distance);
    const sigmaT = sigmaA.map((value, channel) => value + sigmaS[channel]!);
    const transmittance = absorption.map((value, channel) =>
      value * Math.exp(-sigmaS[channel]! * distance));
    const scatterAlbedo = sigmaT.map((value, channel) =>
      value > 0 ? sigmaS[channel]! / value : 0);
    const phaseAtRightAngle = 1 / (4 * Math.PI);
    const sourceScale = albedo.map((value, channel) =>
      value * scatterAlbedo[channel]! *
      (1 - transmittance[channel]!) * phaseAtRightAngle);

    // Column 1 of the shader matrix is the response to pure-green downstream
    // radiance. Its Rec.709 luminance drives a red in-scatter source even
    // though diagonal extinction alone could never create red.
    const redFromPureGreen = sourceScale[0]! * 0.7152;
    expect(redFromPureGreen).toBeGreaterThan(0);
    expect(0 * transmittance[0]!).toBe(0);

    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain(
      'throughput = throughput * restirGiSuffixSegmentTransfer(',
    );
    expect(src).toContain('mediumAlbedo[mediumDepth] = payload.albedo;');
    expect(src).toContain(
      'nextThroughput = nextThroughput * restirGiSuffixDiagonalTransfer(',
    );
  });

  it('seeds value-semantic NEE media, filters nonshadow shells, and pays each active segment once', () => {
    type Medium = Readonly<{
      materialId: number;
      instance: number;
      castShadow: boolean;
      transmissionPaid: boolean;
      transmission: number;
      sigmaT: number;
    }>;
    const active: readonly Medium[] = [
      {
        materialId: 11,
        instance: 2,
        castShadow: true,
        transmissionPaid: false,
        transmission: 0.53,
        sigmaT: 0.17,
      },
      {
        materialId: 29,
        instance: 7,
        castShadow: false,
        transmissionPaid: true,
        transmission: 0.31,
        sigmaT: 0.91,
      },
      {
        materialId: 41,
        instance: 7,
        castShadow: true,
        transmissionPaid: true,
        transmission: 0.67,
        sigmaT: 0.43,
      },
    ];
    const seed = active.filter((medium) => medium.castShadow);
    expect(seed.map(({ materialId, instance }) => [materialId, instance])).toEqual([
      [11, 2],
      [41, 7],
    ]);

    const finiteDistance = 0.8;
    const finiteEndpoint = Math.exp(-seed.at(-1)!.sigmaT * finiteDistance);
    expect(finiteEndpoint).toBeCloseTo(Math.exp(-0.43 * finiteDistance), 14);
    expect(finiteEndpoint).not.toBeCloseTo(
      Math.exp(-(0.43 + 0.91) * finiteDistance),
      14,
    );

    const proposal = seed.map((medium) => ({ ...medium }));
    const original = seed.map((medium) => ({ ...medium }));
    let transfer = Math.exp(-proposal.at(-1)!.sigmaT * 1.25);
    const top = proposal.at(-1)!;
    expect([top.materialId, top.instance]).toEqual([41, 7]);
    if (!top.transmissionPaid) transfer *= top.transmission;
    proposal.pop();
    expect(transfer).toBeCloseTo(Math.exp(-0.43 * 1.25), 14);
    expect(seed).toEqual(original);

    const outer = proposal.at(-1)!;
    expect([outer.materialId, outer.instance]).toEqual([11, 2]);
    transfer *= Math.exp(-outer.sigmaT * 0.5);
    if (!outer.transmissionPaid) transfer *= outer.transmission;
    expect(transfer).toBeCloseTo(
      Math.exp(-0.43 * 1.25) * Math.exp(-0.17 * 0.5) * 0.53,
      14,
    );

    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain(
      'var shadowMediumState = materialShadowEmptyMediumState();',
    );
    expect(src).toContain('if ((shadowMaterialWord & 1u) != 0u)');
    expect(src).toContain(
      'shadowMediumState.transmissionPaid[shadowDepth] =',
    );
    expect(src).toContain(
      'traceSceneAlphaTintTransmittanceTexturedWithState(',
    );
    expect(src).toContain('initialMediumState, false,');
  });

  it('keeps G out of topology and pays scalar transmission once across a matched bulk traversal', () => {
    type Boundary = Readonly<{
      entering: boolean;
      materialId: number;
      instance: number;
      authoredThickness: number;
      thicknessMapG: number;
      transmission: number;
    }>;
    type Medium = Readonly<{
      materialId: number;
      instance: number;
      entryThicknessMapG: number;
      transmissionPaid: boolean;
    }>;
    const stack: Medium[] = [];
    let scalarTransfer = 1;
    const cross = (boundary: Boundary): void => {
      const bulkMedium = boundary.authoredThickness > 0;
      if (!bulkMedium) {
        // The reciprocal two-interface sheet is one material event.
        scalarTransfer *= boundary.transmission;
        return;
      }
      if (boundary.entering) {
        scalarTransfer *= boundary.transmission;
        stack.push({
          materialId: boundary.materialId,
          instance: boundary.instance,
          entryThicknessMapG: boundary.thicknessMapG,
          transmissionPaid: true,
        });
        return;
      }
      const top = stack[stack.length - 1];
      if (
        top == null ||
        top.materialId !== boundary.materialId ||
        top.instance !== boundary.instance
      ) {
        throw new Error('unpaired bulk exit');
      }
      if (!top.transmissionPaid) scalarTransfer *= boundary.transmission;
      stack.pop();
    };

    cross({
      entering: true,
      materialId: 17,
      instance: 3,
      authoredThickness: 2.4,
      thicknessMapG: 0,
      transmission: 0.37,
    });
    expect(stack).toEqual([{
      materialId: 17,
      instance: 3,
      entryThicknessMapG: 0,
      transmissionPaid: true,
    }]);
    expect(scalarTransfer).toBeCloseTo(0.37, 14);
    expect(Math.exp(-0.93 * 1.6 * stack[0]!.entryThicknessMapG)).toBe(1);

    cross({
      entering: false,
      materialId: 17,
      instance: 3,
      authoredThickness: 2.4,
      thicknessMapG: 0.91,
      transmission: 0.08,
    });
    expect(stack).toHaveLength(0);
    expect(scalarTransfer).toBeCloseTo(0.37, 14);

    cross({
      entering: true,
      materialId: 99,
      instance: 4,
      authoredThickness: 0,
      thicknessMapG: 1,
      transmission: 0.6,
    });
    expect(stack).toHaveLength(0);
    expect(scalarTransfer).toBeCloseTo(0.37 * 0.6, 14);
  });

  it('keeps an opaque suffix origin inside glass paired and pays its first exit once', () => {
    type Medium = {
      materialId: number;
      instance: number;
      ior: number;
      beer: readonly [number, number, number];
      thicknessMapG: number;
      scatter: readonly [number, number, number];
      transmissionPaid: boolean;
    };

    // The containment ray may cross the opaque candidate first; only authored
    // transmissive bulk contributes winding. Its first glass crossing is a back
    // face, so the scan records one enclosing shell and reverses it into the
    // transport stack with an unpaid scalar.
    const containmentCrossings = [
      { transmissive: false, bulk: false, side: 1, materialId: 2, instance: 5 },
      { transmissive: true, bulk: true, side: -1, materialId: 17, instance: 3 },
    ] as const;
    const seeds: Medium[] = [];
    let winding = 0;
    let minimum = 0;
    for (const crossing of containmentCrossings) {
      if (!crossing.transmissive || !crossing.bulk) continue;
      winding += crossing.side >= 0 ? 1 : -1;
      if (winding < minimum) {
        minimum = winding;
        seeds.push({
          materialId: crossing.materialId,
          instance: crossing.instance,
          ior: 1.52,
          beer: [0.82, 0.71, 0.63],
          thicknessMapG: 0.4,
          scatter: [0.1, 0.2, 0.3],
          transmissionPaid: false,
        });
      }
    }
    const stack = seeds.reverse();
    expect(stack).toHaveLength(1);
    expect(stack[0]).toMatchObject({
      materialId: 17,
      instance: 3,
      ior: 1.52,
      thicknessMapG: 0.4,
      transmissionPaid: false,
    });

    // Shading/continuing from the opaque object does not touch containment.
    expect(stack).toHaveLength(1);
    let scalarTransfer = 1;
    const exitTransmission = 0.37;
    const exit = { materialId: 17, instance: 3 };
    const top = stack.at(-1)!;
    expect(exit).toMatchObject({
      materialId: top.materialId,
      instance: top.instance,
    });
    if (!top.transmissionPaid) scalarTransfer *= exitTransmission;
    stack.pop();
    expect(scalarTransfer).toBeCloseTo(exitTransmission, 14);
    expect(stack).toHaveLength(0);
  });
});

describe('NRC exact total-internal-reflection oracle', () => {
  const exactDielectricFresnel = (
    cosThetaI: number,
    etaIncident: number,
    etaTarget: number,
  ): number => {
    const ci = Math.min(Math.abs(cosThetaI), 1);
    const eta = etaIncident / etaTarget;
    const sin2ThetaT = eta * eta * (1 - ci * ci);
    if (sin2ThetaT >= 1) return 1;
    const ct = Math.sqrt(1 - sin2ThetaT);
    const rs = (etaIncident * ci - etaTarget * ct) /
      (etaIncident * ci + etaTarget * ct);
    const rp = (etaTarget * ci - etaIncident * ct) /
      (etaTarget * ci + etaIncident * ct);
    return 0.5 * (rs * rs + rp * rp);
  };

  it('uses unit exact Fresnel below the critical angle instead of Schlick', () => {
    const etaIncident = 1.5;
    const etaTarget = 1;
    const cosTheta = 0.5;
    const exact = exactDielectricFresnel(cosTheta, etaIncident, etaTarget);
    const f0 = ((etaIncident - etaTarget) / (etaIncident + etaTarget)) ** 2;
    const schlick = f0 + (1 - f0) * (1 - cosTheta) ** 5;
    expect(exact).toBe(1);
    expect(schlick).toBeLessThan(1);
  });

  it('recovers TIR through the same-side branch and collapses sheet choices', () => {
    const transmission = 0.73;
    const transmissionPdf = transmission / (1 + transmission);
    const reflectionPdf = 1 - transmissionPdf;
    const exactTirReflection = 1;
    const sameSideEstimate = exactTirReflection / reflectionPdf;
    const zeroTransmissionEstimate = 0;
    const expectation =
      reflectionPdf * sameSideEstimate +
      transmissionPdf * zeroTransmissionEstimate;
    expect(expectation).toBeCloseTo(exactTirReflection, 14);

    // At a reciprocal sheet's internal TIR boundary, either prior discrete
    // request maps to reflection. Their union therefore has probability one.
    for (const requestedTransmission of [false, true]) {
      const tir = true;
      const actualReflection = !requestedTransmission || tir;
      const effectiveDiscretePdf = tir ? 1 : 0.5;
      expect(actualReflection).toBe(true);
      expect(exactTirReflection / effectiveDiscretePdf).toBe(1);
    }
  });
});
