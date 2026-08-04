import { describe, expect, it } from 'vitest';
import { REFRACTIVE_CAUSTICS_WGSL } from '../refractiveCaustics.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from '../risGiGlassWalk.wgsl.js';
import {
  RESTIR_GI_DIELECTRIC_SUFFIX_WGSL,
  RIS_GI_WGSL,
} from '../risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { SHADE_WGSL } from '../shade.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';

type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v: Vec3): Vec3 {
  const length = Math.sqrt(dot(v, v));
  return [v[0] / length, v[1] / length, v[2] / length];
}

/** CPU oracle for WGSL refract(I, N, eta). Null is total internal reflection. */
function refract(i: Vec3, n: Vec3, eta: number): Vec3 | null {
  const ni = dot(n, i);
  const k = 1 - eta * eta * (1 - ni * ni);
  if (k < 0) return null;
  const c = eta * ni + Math.sqrt(k);
  return normalize([
    eta * i[0] - c * n[0],
    eta * i[1] - c * n[1],
    eta * i[2] - c * n[2],
  ]);
}

function distance(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function composeDirectWithProducerOwnedPrefix(
  localSurfaceRadiance: Vec3,
  transmittedPrefixRadiance: Vec3,
  faceLayerTransmission: Vec3,
): Vec3 {
  return [
    localSurfaceRadiance[0] * faceLayerTransmission[0] + transmittedPrefixRadiance[0],
    localSurfaceRadiance[1] * faceLayerTransmission[1] + transmittedPrefixRadiance[1],
    localSurfaceRadiance[2] * faceLayerTransmission[2] + transmittedPrefixRadiance[2],
  ];
}

function hiddenClosureExpectation(
  localSource: number,
  reflectionContinuation: number,
  transmissionContinuation: number,
  transmission: number,
): number {
  const localPdf = 1 / 3;
  const opticalPdf = (1 - localPdf) * 0.5;
  const localDraw = (localSource * localPdf) / localPdf;
  const reflectionDraw = (reflectionContinuation * opticalPdf) / opticalPdf;
  const transmissionDraw =
    (transmissionContinuation * transmission * opticalPdf) / opticalPdf;
  return localDraw + reflectionDraw + transmissionDraw;
}

function boundedReciprocalSheetExit(
  reflectance: number,
  transmittance: number,
  remainingInterfaces: number,
): number {
  let reflectedMass = 1;
  let exitMass = 0;
  for (let i = 0; i < remainingInterfaces; i += 1) {
    exitMass += reflectedMass * transmittance;
    reflectedMass *= reflectance;
  }
  return exitMass;
}

type InterfaceEvent = 'bulk' | 'thin' | 'opaque' | 'miss';

/** Mirrors the four-interface budget plus terminal-query contract. */
function escapesWithinBudget(events: readonly InterfaceEvent[]): boolean {
  let interfaces = 0;
  for (const event of events) {
    if (event === 'miss') return interfaces > 0;
    if (event === 'opaque') return false;
    const cost = event === 'thin' ? 2 : 1;
    if (interfaces + cost > 4) return false;
    interfaces += cost;
  }
  return false;
}

type NativeTerminalEvent = 'glass' | 'opaque' | 'miss';

/** Mirrors the native walk after its already-consumed primary interface. */
function nativeTerminalAccepted(events: readonly NativeTerminalEvent[]): boolean {
  let interfaces = 1;
  for (const event of events) {
    if (event === 'miss' || event === 'opaque') return true;
    if (interfaces >= 8) return false;
    interfaces += 1;
  }
  return false;
}

describe('bounded hybrid dielectric transport closure', () => {
  it('keeps the refractive camera prefix native to the full-resolution shade path', () => {
    for (const source of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(source).not.toContain('var rGlass: ReservoirGI');
      expect(source).not.toContain('GLASS_WALK_MAX_INTERFACES');
    }
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'const GLASS_WALK_MAX_INTERFACES: u32 = 8u;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('var mediumDepth: u32 = 0u;');
    expect(NATIVE_GLASS_GI_WGSL).toContain('dielectricInterfaceTransmissionRgb(');
    expect(NATIVE_GLASS_GI_WGSL).toContain('materialSpectralAttenuation(');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'ggxSampleDielectricTransmissionAnisotropyFrame(',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'refractDir = interfaceLobe.direction;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'applyNormalMapForHit(\n      walkHit,\n      walkSmoothNormal,',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'referenceThickness * clamp(mediumThicknessMapScale[top], 0.0, 1.0)',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let entering = walkHit.side >= 0.0;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'mediumInstance[top] != walkRepresentedId',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'mediumMaterialId[top] != walkBoundaryId',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'if (walkRepresentedId == 0u) { break; }',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain('GLASS_GI_MAX_ROUGHNESS');
  });

  it('does not apply the camera-prefix Fresnel/Beer throughput twice in shade', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let receiverContribution = receiver.prefixTransfer * receiverResponse;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let transmittedReceiverDirect = receiver.prefixTransfer * receiverDirect;',
    );
    expect(SHADE_WGSL).toContain(
      'let directRadiance = visiblePrimaryDirect + Lo_transmittedGI;',
    );
    expect(SHADE_WGSL).not.toMatch(/Lo_transmittedGI\s*\*\s*layerTransmission/);
  });

  it('does not apply the primary face layer twice to transmitted GI', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'faceLayerTransmission(primaryLayer)',
    );
    expect(SHADE_WGSL).toContain(
      'let directRadiance = visiblePrimaryDirect + Lo_transmittedGI;',
    );

    const prefixComplete: Vec3 = [0.8, 0.6, 0.4];
    const composed = composeDirectWithProducerOwnedPrefix(
      [0, 0, 0],
      prefixComplete,
      [0.25, 0.5, 0.75],
    );
    expect(composed).toEqual(prefixComplete);
  });

  it('does not attenuate absolute thin-film reflection by film transmission', () => {
    expect(SHADING_TERMS_WGSL).toContain(
      'return applyMaterialLayerTransmissionToBrdf(',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'physicalSpecularLog,\n    reflectionLayerTransmission,',
    );
    expect(SHADE_WGSL).toContain(
      'aggregateSurfaceDirect + Lo_indirectSpec',
    );
    expect(SHADE_WGSL).not.toContain(
      'Lo_indirectSpec * reflectionLayerTransmission',
    );
    expect(SHADE_WGSL).toContain(
      '(Lo_emit + Lo_emitterGlow + Lo_lightMap +',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'suffixLo * receiverBrdf,',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain(
      'suffixLo * receiverBrdf * receiverPayload.layerTransmission',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'receiverLocalSource * receiverPayload.layerTransmission,',
    );
  });

  it('keeps deterministic transmitted sources direct and clamps only stochastic suffix GI', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain('return transmittedEnvironment;');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let transmittedReceiverDirect = receiver.prefixTransfer * receiverDirect;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let scaledIndirect = indirect * ubo.glassMixScale;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'ubo.indirectFireflyClamp * ubo.glassMixScale',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'return transmittedReceiverDirect + clampedIndirect;',
    );
  });

  it('uses native represented WRS for the full-resolution stochastic suffix', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain('var wrs = representedWrsInit();');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let logPHat = reservoirGiLogPositive(luminance(receiverContribution));',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let logPSrc = reservoirGiLogPositive(cosTheta * INV_PI);',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('let logWeight = logPHat - logPSrc;');
    expect(NATIVE_GLASS_GI_WGSL).toContain('finaliseGIReservoirFromNativeWrs(');
    expect(NATIVE_GLASS_GI_WGSL).not.toContain('w_sum');
  });

  it('preserves continuous opaque diffuse, reflection, and transmitted shares', () => {
    expect(SHADING_TERMS_WGSL).toContain('evalGGXReflectionWithTransmissionMix(');
    expect(SHADING_TERMS_WGSL).toContain(
      '(1.0 - clamp(transmission, 0.0, 1.0));',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'if (transmission <= 0.0 && metal <= 0.0',
    );
    expect(SHADE_WGSL).toContain('envMapIntensity,\n    matColor.a,\n    metal,');
    expect(SHADE_WGSL).toContain('let Lo_transmittedGI = lo_transmittedGI(');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let localBranchThreshold = represented_bernoulli_probability_f32(',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let reflectionBranchThreshold = represented_bernoulli_probability_f32(',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let transmissionBranchPdf = 1.0 - reflectionBranchThreshold;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let interfaceLobe = restirGiSampleDielectricLobe(',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('receiverClosureMode = 1u;');

    expect(hiddenClosureExpectation(7, 5, 11, 0.4)).toBeCloseTo(
      7 + 5 + 11 * 0.4,
      12,
    );
    // At TIR the transmission technique contributes zero, while the separately
    // weighted reflection technique retains the exact unit reflection mass.
    expect(hiddenClosureExpectation(3, 13, 99, 0)).toBeCloseTo(16, 12);
  });

  it('traces RGB dispersion with channel-specific geometry, PDFs, and terminals', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'nativeGlassGiChannel(primaryIncidentIor, transportChannel)',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let etaIncident = restirGiSuffixChannel(etaIncidentRgb, channel);',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'primaryAlbedo,\n    0u,\n    &rngR,',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'primaryAlbedo,\n    1u,\n    &rngG,',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'primaryAlbedo,\n    2u,\n    &rngB,',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'return vec3f(radianceR.r, radianceG.g, radianceB.b);',
    );
  });

  it('initializes nested medium state when the camera starts inside bulk glass', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'fn classifyNativeGlassGiContainingMedia(',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'primaryIncidentIor = mediumIor[mediumDepth - 1u];',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'primaryTargetIor = mediumIor[mediumDepth - 2u];',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let segmentDistance = primaryHit.dist;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'mediumTransmissionPaid[mediumDepth - 1u] == 0u',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain(
      'if (!primaryEntering) { return result; }',
    );
  });

  it('keeps closed-volume topology stable across thickness-map texels', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let primaryThickness = materialShadowAuthoredThickness(primaryHit);',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let walkThickness = materialShadowAuthoredThickness(walkHit);',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let baseMaterialThickness = materialShadowAuthoredThickness(currentHit);',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain(
      'primaryThickness = primaryThickness *',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain(
      'walkThickness = walkThickness *',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let primaryBulkMedium = primaryBoundaryId != 0u;',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain('applyThicknessMapToBeerTint(');
  });

  it('keeps spectral thickness scaling and cross-channel in-scatter support', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'referenceThickness * clamp(mediumThicknessMapScale[top], 0.0, 1.0)',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let support = transfer * vec3f(1.0);',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'if (!restirGiSuffixTransferFinite(transfer)) { return false; }',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain(
      'nativeGlassGiChannel(prefixThroughput',
    );

    const attenuationColor = 0.36;
    const attenuationDistance = 2.5;
    const authoredThickness = 3;
    const distance = 1.75;
    const thicknessMapG = 0.4;
    const transportDistance = Math.min(
      distance,
      authoredThickness * thicknessMapG,
    );
    const expected = attenuationColor **
      (transportDistance / attenuationDistance);
    const missingG = attenuationColor ** (distance / attenuationDistance);
    expect(expected).not.toBeCloseTo(missingG, 10);
    expect(expected).toBeGreaterThan(missingG);
  });

  it('handles unlit and all direct-light families at the hidden receiver', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'if (decodeIsUnlitMaterial(receiverWord))',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('fn nativeGlassGiAnalyticNee(');
    expect(NATIVE_GLASS_GI_WGSL).toContain('fn nativeGlassGiSunNee(');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'let receiverArea = nativeGlassGiAreaEmitterNee(',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('let receiverLocalSource =');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'receiver.opaqueShare;',
    );
  });

  it('embeds one byte-identical bounded suffix in ordinary, NRC, and native shade', () => {
    const start = RIS_GI_WGSL.indexOf(
      'const RESTIR_GI_SUFFIX_MAX_INTERFACES: u32 = 8u;',
    );
    const end = RIS_GI_WGSL.indexOf(
      '// sampleCosineHemisphere is the canonical helper',
      start,
    );
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    expect(RESTIR_GI_DIELECTRIC_SUFFIX_WGSL).toBe(
      RIS_GI_WGSL.slice(start, end),
    );
    expect(RIS_GI_NRC_BODY).toContain(RESTIR_GI_DIELECTRIC_SUFFIX_WGSL);
    expect(NATIVE_GLASS_GI_WGSL).toContain(RESTIR_GI_DIELECTRIC_SUFFIX_WGSL);
    expect(
      NATIVE_GLASS_GI_WGSL.match(/fn traceRestirGiDielectricSuffix\(/g),
    ).toHaveLength(1);
  });

  it('retains reciprocal thin-sheet reflection mass within the global budget', () => {
    const reflectance = 0.2;
    const transmittance = 0.8;
    expect(boundedReciprocalSheetExit(reflectance, transmittance, 1)).toBe(0.8);
    expect(boundedReciprocalSheetExit(reflectance, transmittance, 4)).toBeCloseTo(
      transmittance * (1 + reflectance + reflectance ** 2 + reflectance ** 3),
      12,
    );
    expect(boundedReciprocalSheetExit(reflectance, transmittance, 32)).toBeCloseTo(
      1,
      12,
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain('applyNormalMapForSideForHit(');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'primaryHit, primarySmoothNormal, slabFrontFacing,',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'walkHit, walkSmoothNormal, slabFrontFacing,',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain('slabNormal = -slabNormal;');
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'if (interfaceCount >= GLASS_WALK_MAX_INTERFACES) { break; }',
    );
  });

  it('pays exact interface Fresnel in the independent caustic walk', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'dielectricInterfaceTransmissionRgb(',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'interfaceWeight * exitWeight * transmission',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'applyNormalMapForHit(hit, interfaceSmoothNormal)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'applyNormalMapForSideForHit(',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let exitRoughness = faceLayerRoughness(mappedBaseRoughness, exitLayer);',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'ggxSampleDielectricTransmissionAnisotropyFrame(',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('faceLayerTransmission(layerControls)');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let entering = hit.side >= 0.0;',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'mediumMaterialId[top] != hitBoundaryId',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'if (authoredTransmissionTopology && hitRepresentedId == 0u)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).not.toContain('materialRm.x > 0.0001');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let materialThickness = materialShadowAuthoredThickness(hit);',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).not.toContain(
      'materialThickness = materialThickness *',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'fn classifyRefractiveCausticContainingMedia(',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'if (!containingMediaClassified && authoredTransmissionTopology)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'mediumDepth = containingMedia.depth;',
    );
    const containmentStart = REFRACTIVE_CAUSTICS_WGSL.indexOf(
      'fn classifyRefractiveCausticContainingMedia(',
    );
    const containmentEnd = REFRACTIVE_CAUSTICS_WGSL.indexOf(
      'fn traceRefractiveCausticPath(', containmentStart,
    );
    const containment = REFRACTIVE_CAUSTICS_WGSL.slice(
      containmentStart, containmentEnd,
    );
    expect(containment).not.toContain('(channel * 0x85ebca6bu)');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'traceSceneFirstHitAlphaMaskTexturedCastShadow(',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'out.transmissionPaid[seed] =',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'containingMedia.state.transmissionPaid[seed];',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'pairedExitTransmissionPaid = mediumTransmissionPaid[top];',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'if (entering || pairedExitTransmissionPaid == 0u)',
    );
  });

  it('transports a caustic ray born inside a closed shell and pays its first exit once', () => {
    const authoredTransmission = 0.37;
    const attenuationColor = 0.42;
    const attenuationDistance = 3.1;
    const segmentDistance = 1.7;
    const thicknessMapG = 0.4;
    const sigmaS = 0.23;
    const authoredThickness = 2.4;
    const transportDistance = Math.min(
      segmentDistance,
      authoredThickness * thicknessMapG,
    );
    const segmentTransfer =
      attenuationColor **
        (transportDistance / attenuationDistance) *
      Math.exp(-sigmaS * transportDistance);

    // A containment seed is unpaid because no forward entry was observed.
    // The receiver-to-exit segment is transported first, then the paired exit
    // owns the scalar exactly once.
    let throughput = 1;
    throughput *= segmentTransfer;
    const transmissionPaid = false;
    if (!transmissionPaid) throughput *= authoredTransmission;
    expect(throughput).toBeCloseTo(
      segmentTransfer * authoredTransmission,
      14,
    );
    expect(throughput).not.toBeCloseTo(segmentTransfer, 14);
    expect(throughput).not.toBeCloseTo(
      segmentTransfer * authoredTransmission ** 2,
      14,
    );
  });

  it('retains unit support at a paid bulk exit whose mapped texel is zero', () => {
    const entryTransmission = 0.37;
    const exitMappedTransmission = 0;
    const paidExit = true;
    const exitSupported = exitMappedTransmission > 0 || paidExit;
    const throughput = entryTransmission * (paidExit ? 1 : exitMappedTransmission);
    expect(exitSupported).toBe(true);
    expect(throughput).toBeCloseTo(entryTransmission, 14);
    expect(exitMappedTransmission > 0 || false).toBe(false);

    const src = REFRACTIVE_CAUSTICS_WGSL;
    const pair = src.indexOf('let pairedPaidExit =');
    const mappedGate = src.indexOf(
      'if (!materialHasTransmission(transmission) && !pairedPaidExit)',
    );
    expect(pair).toBeGreaterThan(0);
    expect(mappedGate).toBeGreaterThan(pair);
    expect(src).toContain(
      'let authoredTransmissionTopology = materialHasTransmission(scalar.a);',
    );
    expect(src).toContain(
      'bulkMedium && !entering && pairedExitTransmissionPaid != 0u;',
    );
  });

  it('routes no-glass escapes and opaque blockers through the neutral direct-sun baseline', () => {
    const baseline = 0.73;
    const candidate = (sawGlass: boolean, terminatedOpaque: boolean): number =>
      !sawGlass && terminatedOpaque ? baseline : 0;
    expect(candidate(false, true) - baseline).toBe(0);

    expect(REFRACTIVE_CAUSTICS_WGSL).toMatch(
      /if \(out\.sawGlass == 0u\) \{\s*out\.eligible = 0u;/,
    );
    const opaqueGate = REFRACTIVE_CAUSTICS_WGSL.indexOf(
      'if (!authoredTransmissionTopology)',
    );
    const mappedGate = REFRACTIVE_CAUSTICS_WGSL.indexOf(
      'if (!materialHasTransmission(transmission) && !pairedPaidExit)',
    );
    expect(opaqueGate).toBeGreaterThan(0);
    expect(mappedGate).toBeGreaterThan(opaqueGate);
    expect(REFRACTIVE_CAUSTICS_WGSL.slice(opaqueGate, mappedGate)).toContain(
      'out.eligible = 0u;',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL.slice(mappedGate)).toContain(
      'out.eligible = 0u;',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'if (path.eligible == 0u)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let fallback = refractiveCausticChannel(baseline, channel) * 0.5;',
    );
  });

  it('reserves terminal queries after eight camera-GI or four caustic interfaces', () => {
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'gi <= GLASS_WALK_MAX_INTERFACES',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'if (interfaceCount >= GLASS_WALK_MAX_INTERFACES) { break; }',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('depth <= 4u');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'interfaceCount + interfaceCost > 4u',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'select(1u, 2u, thinSheet)',
    );
    expect(escapesWithinBudget(['bulk', 'bulk', 'bulk', 'bulk', 'miss'])).toBe(true);
    expect(escapesWithinBudget(['bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'miss'])).toBe(false);
    expect(escapesWithinBudget(['thin', 'thin', 'miss'])).toBe(true);
    expect(escapesWithinBudget(['thin', 'thin', 'thin', 'miss'])).toBe(false);
    expect(nativeTerminalAccepted([
      'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'opaque',
    ])).toBe(true);
    expect(nativeTerminalAccepted([
      'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'miss',
    ])).toBe(true);
    expect(nativeTerminalAccepted([
      'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass', 'glass',
      'opaque',
    ])).toBe(false);
  });

  it('uses the exact solid-angle proposal weight without a hidden firefly clamp', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'ubo.sunAngular.x <= 1.5707963268',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let jacobianWeight = omegaSearch / omegaSun;',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).not.toContain(
      'min(32.0, omegaSearch / omegaSun)',
    );
  });

  it('refracts back out of a tilted parallel slab instead of keeping the entry direction', () => {
    const incident = normalize([0, 0, -1]);
    const orientedBoundaryNormal = normalize([0.342, 0, 0.94]);
    const ior = 1.52;
    const inside = refract(incident, orientedBoundaryNormal, 1 / ior);
    expect(inside).not.toBeNull();
    const exit = refract(inside!, orientedBoundaryNormal, ior);
    expect(exit).not.toBeNull();

    // The entry solve bends materially; a straight-through exit would retain
    // this wrong direction. The reciprocal exit solve restores the incident
    // direction for a parallel slab (with a lateral position offset only).
    expect(distance(inside!, incident)).toBeGreaterThan(0.05);
    expect(distance(exit!, incident)).toBeLessThan(1e-10);
  });

  it('fails the transmitted branch on total internal reflection', () => {
    const sixtyDegreesInside = normalize([
      Math.sin(Math.PI / 3),
      0,
      -Math.cos(Math.PI / 3),
    ]);
    expect(refract(sixtyDegreesInside, [0, 0, 1], 1.52)).toBeNull();
  });
});
