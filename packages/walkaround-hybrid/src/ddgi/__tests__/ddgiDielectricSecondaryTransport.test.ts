import { describe, expect, it } from 'vitest';
import { representBernoulliProbabilityF32 } from '@vitrum/shared-samplers';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';

type V3 = [number, number, number];

interface Crossing {
  readonly boundary: number;
  readonly represented: number;
  readonly side: 'front' | 'back';
  readonly alphaAdmitted?: boolean;
  readonly castsShadow?: boolean;
}

interface SeededMedium {
  readonly boundary: number;
  readonly represented: number;
  readonly transmissionPaid: boolean;
}

/** CPU mirror of the shader's exact front-push/back-pop containment scan. */
function reconstructContainingMedia(
  crossings: readonly Crossing[],
  maxDepth = 8,
): SeededMedium[] | null {
  const temporary: Array<Pick<Crossing, 'boundary' | 'represented'>> = [];
  const innerToOuter: SeededMedium[] = [];

  for (const crossing of crossings) {
    if (crossing.castsShadow === false) continue;
    if (crossing.boundary === 0 || crossing.represented === 0) return null;
    if (crossing.side === 'front') {
      if (temporary.length >= maxDepth) return null;
      temporary.push(crossing);
      continue;
    }
    const top = temporary.at(-1);
    if (top != null) {
      if (
        top.boundary !== crossing.boundary ||
        top.represented !== crossing.represented
      ) return null;
      temporary.pop();
      continue;
    }
    if (crossing.alphaAdmitted === false || innerToOuter.length >= maxDepth) {
      return null;
    }
    innerToOuter.push({
      boundary: crossing.boundary,
      represented: crossing.represented,
      transmissionPaid: false,
    });
  }
  if (temporary.length !== 0) return null;
  return innerToOuter.reverse();
}

type MediumState = SeededMedium;

/** CPU mirror of scalar-transmission ownership for bulk entries/exits. */
function crossBulkBoundary(
  stack: MediumState[],
  crossing: Crossing,
  mappedTransmission: number,
): number | null {
  if (crossing.side === 'front') {
    stack.push({
      boundary: crossing.boundary,
      represented: crossing.represented,
      transmissionPaid: true,
    });
    return mappedTransmission;
  }
  const top = stack.at(-1);
  if (top === undefined ||
      top.boundary !== crossing.boundary ||
      top.represented !== crossing.represented) return null;
  stack.pop();
  return top.transmissionPaid ? 1 : mappedTransmission;
}

function rgbAbsorption(
  attenuationColor: number,
  attenuationDistance: number,
  authoredThickness: number,
  exitMapScale: number,
  segmentDistance: number,
): number {
  const transportDistance = authoredThickness > 0
    ? Math.min(segmentDistance, authoredThickness * exitMapScale)
    : segmentDistance;
  return Math.pow(
    attenuationColor,
    transportDistance / attenuationDistance,
  );
}

type M3 = [V3, V3, V3];

function volumeSegmentTransfer(
  absorption: V3,
  sigmaS: V3,
  albedo: V3,
  segmentDistance: number,
  anisotropy = 0,
): M3 {
  const sigmaA = absorption.map((value) =>
    -Math.log(Math.max(value, 1e-30)) / segmentDistance) as V3;
  const sigmaT = sigmaS.map((value, channel) =>
    value + sigmaA[channel]!) as V3;
  const transmittance = absorption.map((value, channel) =>
    value * Math.exp(-sigmaS[channel]! * segmentDistance)) as V3;
  const scatterAlbedo = sigmaS.map((value, channel) =>
    sigmaT[channel]! > 0 ? value / sigmaT[channel]! : 0) as V3;
  const g = Math.min(Math.max(anisotropy, -0.99), 0.99);
  const denominator = 1 + g * g;
  const phase = (1 - g * g) /
    (4 * Math.PI * denominator * Math.sqrt(denominator));
  const source = albedo.map((value, channel) =>
    Math.max(value, 0) * scatterAlbedo[channel]! *
      (1 - transmittance[channel]!) * phase) as V3;
  return [
    [transmittance[0] + source[0] * 0.2126, source[1] * 0.2126, source[2] * 0.2126],
    [source[0] * 0.7152, transmittance[1] + source[1] * 0.7152, source[2] * 0.7152],
    [source[0] * 0.0722, source[1] * 0.0722, transmittance[2] + source[2] * 0.0722],
  ];
}

function applyTransfer(transfer: M3, radiance: V3): V3 {
  return [0, 1, 2].map((row) =>
    transfer[0][row]! * radiance[0] +
    transfer[1][row]! * radiance[1] +
    transfer[2][row]! * radiance[2]) as V3;
}

function dielectricMixtureExpectation(
  scalarTransmission: number,
  reflection: number,
  transmission: number,
  localLo: number,
  reflectedLo: number,
  transmittedLo: number,
): number {
  const t = Math.min(Math.max(scalarTransmission, 0), 1);
  const pTransmission = transmission > 0
    ? representBernoulliProbabilityF32(t / (1 + t))
    : 0;
  const pReflection = 1 - pTransmission;
  const reflectionEstimator =
    ((1 - t) * localLo + reflection * reflectedLo) / pReflection;
  const transmissionEstimator = pTransmission > 0
    ? t * transmission * transmittedLo / pTransmission
    : 0;
  return pReflection * reflectionEstimator +
    pTransmission * transmissionEstimator;
}

function boundedTerminalWalk(kinds: readonly ('glass' | 'opaque' | 'env')[]): string {
  let interfaceCount = 0;
  for (let inspection = 0; inspection <= 8; inspection += 1) {
    const kind = kinds[inspection] ?? 'env';
    if (kind !== 'glass') return kind;
    if (interfaceCount >= 8) return 'closed-overflow';
    interfaceCount += 1;
  }
  return 'closed-overflow';
}

describe('DDGI dielectric secondary transport state machine', () => {
  it('reconstructs nested containing media outer-to-inner and ignores alpha-discarded shells', () => {
    const stack = reconstructContainingMedia([
      { boundary: 20, represented: 7, side: 'back' },
      { boundary: 10, represented: 3, side: 'back' },
      { boundary: 77, represented: 1, side: 'front' },
      { boundary: 77, represented: 1, side: 'back', alphaAdmitted: false },
    ]);

    expect(stack).toEqual([
      { boundary: 10, represented: 3, transmissionPaid: false },
      { boundary: 20, represented: 7, transmissionPaid: false },
    ]);
  });

  it('keeps equal component IDs in different represented ranges distinct', () => {
    expect(reconstructContainingMedia([
      { boundary: 5, represented: 2, side: 'back' },
      { boundary: 5, represented: 1, side: 'back' },
    ])).toEqual([
      { boundary: 5, represented: 1, transmissionPaid: false },
      { boundary: 5, represented: 2, transmissionPaid: false },
    ]);
  });

  it('fails closed when a winding scan is not a valid ray-to-infinity suffix', () => {
    expect(reconstructContainingMedia([
      { boundary: 1, represented: 1, side: 'front' },
    ])).toBeNull();
  });

  it('ignores a disabled transmissive shell around enabled nested topology', () => {
    expect(reconstructContainingMedia([
      { boundary: 1, represented: 1, side: 'front', castsShadow: false },
      { boundary: 2, represented: 2, side: 'front' },
      { boundary: 2, represented: 2, side: 'back' },
      { boundary: 1, represented: 1, side: 'back', castsShadow: false },
    ])).toEqual([]);
  });

  it('excludes a disabled launch-containing bulk without losing an enabled inner medium', () => {
    expect(reconstructContainingMedia([
      { boundary: 2, represented: 2, side: 'back' },
      { boundary: 1, represented: 1, side: 'back', castsShadow: false },
    ])).toEqual([
      { boundary: 2, represented: 2, transmissionPaid: false },
    ]);
  });

  it('pays scalar transmission once at an observed entry and not again at its paired exit', () => {
    const stack: MediumState[] = [];
    expect(crossBulkBoundary(
      stack,
      { boundary: 3, represented: 9, side: 'front' },
      0.4,
    )).toBeCloseTo(0.4, 12);
    expect(crossBulkBoundary(
      stack,
      { boundary: 3, represented: 9, side: 'back' },
      0.1,
    )).toBe(1);
    expect(stack).toHaveLength(0);
  });

  it('pays scalar transmission once at the first exit when the suffix starts inside', () => {
    const stack: MediumState[] = [
      { boundary: 4, represented: 2, transmissionPaid: false },
    ];
    expect(crossBulkBoundary(
      stack,
      { boundary: 4, represented: 2, side: 'back' },
      0.65,
    )).toBeCloseTo(0.65, 12);
    expect(stack).toHaveLength(0);
  });

  it('rejects a back boundary whose material-instance identity does not own the stack top', () => {
    const stack: MediumState[] = [
      { boundary: 4, represented: 1, transmissionPaid: true },
    ];
    expect(crossBulkBoundary(
      stack,
      { boundary: 4, represented: 2, side: 'back' },
      0.8,
    )).toBeNull();
    expect(stack).toHaveLength(1);
  });

  it('uses analyzed boundary identity for topology and exit-mapped thickness only as a cap', () => {
    const isBulk = (encodedBoundaryId: number): boolean => encodedBoundaryId !== 0;
    expect(isBulk(0)).toBe(false);
    expect(isBulk(1)).toBe(true);
    expect(rgbAbsorption(0.2, 0.5, 2, 0, 9)).toBe(1);
    expect(rgbAbsorption(0.2, 0.5, 2, 0.25, 9)).toBeCloseTo(0.2, 12);
    expect(rgbAbsorption(0.2, 0.5, -1, 0, 2)).toBeCloseTo(0.0016, 12);
  });

  it('applies an authored mapped cap to both absorption and scattering extinction', () => {
    const segment = 2.5;
    const g = 0.4;
    const transportDistance = segment * g;
    expect(transportDistance).toBe(1);
    const absorption = rgbAbsorption(0.5, 1, segment, g, segment);
    expect(absorption).toBeCloseTo(0.5, 12);
    const result = applyTransfer(volumeSegmentTransfer(
      [absorption, absorption, absorption],
      [0.3, 0.3, 0.3],
      [0.6, 0.6, 0.6],
      transportDistance,
    ), [1, 1, 1]);
    const wrongUncappedScatter = applyTransfer(volumeSegmentTransfer(
      [absorption, absorption, absorption],
      [0.3, 0.3, 0.3],
      [0.6, 0.6, 0.6],
      segment,
    ), [1, 1, 1]);
    expect(result[0]).not.toBeCloseTo(wrongUncappedScatter[0], 8);
  });

  it('keeps the bounded single-scatter source between pure extinction and unattenuated energy', () => {
    const pureExtinction = Math.exp(-0.8 * 3);
    const withSource = applyTransfer(volumeSegmentTransfer(
      [1, 1, 1], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9], 3,
    ), [1, 1, 1])[0];
    expect(withSource).toBeGreaterThan(pureExtinction);
    expect(withSource).toBeLessThan(1);
  });

  it('retains luminance-driven cross-channel single scattering', () => {
    const result = applyTransfer(volumeSegmentTransfer(
      [1, 1, 1], [0.8, 0.8, 0.8], [0, 1, 0], 2,
    ), [1, 0, 0]);
    expect(result[0]).toBeGreaterThan(0);
    expect(result[1]).toBeGreaterThan(0);
    expect(result[2]).toBe(0);
  });

  it('unit-envelope sampling preserves local, exact Fresnel reflection, and scalar-weighted transmission', () => {
    const t = 0.7;
    const r = 0.12;
    const opticalT = 0.88;
    const expected = (1 - t) * 2 + r * 5 + t * opticalT * 7;
    expect(dielectricMixtureExpectation(t, r, opticalT, 2, 5, 7))
      .toBeCloseTo(expected, 12);
  });

  it('total internal reflection has no transmission branch and retains exact reflected mass at t=1', () => {
    expect(dielectricMixtureExpectation(1, 1, 0, 4, 6, 9))
      .toBeCloseTo(6, 12);
  });

  it('allows the eighth interface to terminate on an opaque receiver, then closes a ninth interface', () => {
    expect(boundedTerminalWalk([
      'glass', 'glass', 'glass', 'glass',
      'glass', 'glass', 'glass', 'glass',
      'opaque',
    ])).toBe('opaque');
    expect(boundedTerminalWalk([
      'glass', 'glass', 'glass', 'glass',
      'glass', 'glass', 'glass', 'glass',
      'glass',
    ])).toBe('closed-overflow');
  });
});

describe('DDGI dielectric secondary transport WGSL composition', () => {
  const source = makeProbeUpdateRaysWGSL(64);

  it('pins component topology, exit-mapped authored caps, and synthetic full segments', () => {
    expect(source).toContain('let hasBulkTopology = boundaryId != 0u;');
    expect(source).toContain('out.authoredThickness = select(');
    expect(source).toContain('out.thicknessMapScale = select(');
    expect(source).toContain('let mappedCap = authoredThickness * clamp(');
    expect(source).toContain('min(segmentLength, mappedCap),');
    expect(source).toContain('materialOpticalHasAuthoredThickness(triIndex)');
    expect(source).toContain('fn ddgiMediumSegmentTransfer(');
    expect(source).toContain('let transmittance = absorption *');
    expect(source).toContain('ddgiHomogeneousBeerTransmittanceRgb(sigmaS, transportDistance);');
    expect(source).toContain('let sigmaA = -log(max(absorption, vec3f(1e-30))) /');
    expect(source).toContain('ddgiHenyeyGreensteinPhase(0.0, scatter.a);');
    expect(source).toContain('sourceScale * 0.2126');
    expect(source).toContain('sourceScale * 0.7152');
    expect(source).toContain('sourceScale * 0.0722');
    expect(source).toContain('var prefixTransfer = ddgiDiagonalTransfer(vec3f(1.0));');
    expect(source).not.toContain('pathLength / projectedCosine');
  });

  it('pins actual-direction fixed-origin containment and component/range LIFO ownership', () => {
    expect(source).toContain('fn ddgiClassifyContainingMedia(');
    expect(source).toContain('let event = traceSceneOpticalBoundaryEvent(');
    expect(source).toContain('var exclusiveMinT = 0.0;');
    expect(source).toContain('temporaryBoundary[top] != event.encodedBoundaryId ||');
    expect(source).toContain('event.representedPrimitiveInstanceId');
    expect(source).toContain('innerToOuter.transmissionPaid[depth] = 0u;');
    expect(source).toContain('probeOrigin,\n      dir,\n      opticalSourceFeatureInvalid(),');
    expect(source).not.toContain('containmentDirection');
    expect(source).not.toContain('ddgiBuildProbeContainingMedia');
    expect(source).not.toContain('workgroupBarrier();');
    expect(source).not.toContain('boundaryStep');
  });

  it('treats cast-shadow-disabled transmissive boundaries as absent from the exact shadow walk', () => {
    const classifierStart = source.indexOf('fn ddgiClassifyContainingMedia(');
    const classifierEnd = source.indexOf('fn ddgiMediumExtinctionForSegment(', classifierStart);
    const classifier = source.slice(classifierStart, classifierEnd);
    const walkerStart = source.indexOf('fn ddgiTraceShadowVisibility(');
    const walkerEnd = source.indexOf('fn traceSunVisibility(', walkerStart);
    const walker = source.slice(walkerStart, walkerEnd);

    expect(classifier).toContain(
      'respectCastShadow &&\n      (entry.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u\n    ) { continue; }',
    );
    expect(classifier.indexOf(') { continue; }')).toBeLessThan(
      classifier.indexOf('if (event.side > 0.0)'),
    );
    expect(walker).toContain(
      'if ((entry.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u) {',
    );
    expect(walker).not.toContain('if (hasTransmission) { return vec3f(0.0); }');
    expect(walker).toContain('exclusiveMinT = acceptedT;');
  });

  it('uses a true unbounded directional-shadow sentinel beyond the old 1e15 cap', () => {
    const farFiniteF32 = Math.fround(2e15);
    expect(Number.isFinite(farFiniteF32)).toBe(true);
    expect(farFiniteF32).toBeGreaterThan(1e15);
    expect(farFiniteF32).toBeLessThan(Number.POSITIVE_INFINITY);
    expect(source).toContain(
      'let unboundedShadowDistance = bitcast<f32>(0x7f800000u);',
    );
    expect(source).toContain(
      'ddgiTraceShadowVisibility(origin, sunDir, unboundedShadowDistance)',
    );
    expect(source).not.toContain('ddgiTraceShadowVisibility(origin, sunDir, 1e15)');
  });

  it('pins exact TIR/reflection, rough anisotropy, dispersion, thin film, and face layers', () => {
    expect(source).toContain('fn ddgiSampleVisibleGgxNormalAnisotropic(');
    expect(source).toContain('fn ddgiDistributionGgxAnisotropic(');
    expect(source).toContain('fn ddgiSampleDielectricInterface(');
    expect(source).toContain('let tir = dot(refractedRaw, refractedRaw) <= 1e-12;');
    expect(source).toContain('let tirRgb = sin2TargetRgb >= vec3f(1.0);');
    expect(source).toContain('reflectanceRgb = select(reflectanceRgb, vec3f(1.0), tirRgb);');
    expect(source).toContain('transmittanceRgb = select(transmittanceRgb, vec3f(0.0), tirRgb);');
    expect(source).toContain('let film = materialThinFilmResponse(');
    expect(source).toContain('let layerTransferRgb = ddgiFaceLayerTransmission(layer);');
    expect(source).toContain('materialDispersionIorRgb(hit.indices.w, transportIor)');
    expect(source).toContain('ddgiFaceLayerRoughness(currentMaterial.baseRoughness, slabLayer)');
  });

  it('pins bounded multi-interface terminal receiver transport and correlated RGB draws', () => {
    expect(source).toContain('depth <= DDGI_GLASS_MAX_INTERFACES;');
    expect(source).toContain('return accumulatedRadiance + ddgiTransferredChannel(');
    expect(source).toContain('ddgiEvaluateProbeSurfaceRadiance(');
    expect(source).toContain('sampleSkyColor(rayDirection), channel,');
    expect(source).toContain('if (mediumState.depth != 0u) { return accumulatedRadiance; }');
    expect(source.match(/ddgiTraceGlassChannel\(/g)).toHaveLength(4);
    expect(source).toContain('var rng = ddgiPcgHashU32(seed ^ 0x44444749u);');
    expect(source).toContain('let familyXi = ddgiGlassRandF32(&rng);');
    expect(source).toContain('let slabFamilyXi = ddgiGlassRandF32(&rng);');
    expect(source).toContain('transmissionPdf > 0.0 && familyXi < transmissionPdf;');
    expect(source).toContain('slabFamilyXi < slabTransmissionPdf;');
    expect(source).not.toContain('channel * 0x9e3779b9u');
  });

  it('pins scalar transmission ownership and reflection persistence at t=1', () => {
    expect(source).toContain('scalarTransmission / (1.0 + scalarTransmission)');
    expect(source).toContain('transmissionPdf = represented_bernoulli_probability_f32(');
    expect(source).toContain('let reflectionPdf = 1.0 - transmissionPdf;');
    expect(source).toContain('1.0 - mappedTransmission,');
    expect(source).toContain('interfaceSample.reflectionWeight /');
    expect(source).toContain('prefixTransfer * ddgiDiagonalTransfer(');
    expect(source).toContain('nextMediumState.transmissionPaid[top] = 1u;');
    expect(source).toContain('pairedPaidExit,');
    expect(source).not.toContain('radiance = mix(\n            radiance,\n            transmitted,');
  });

  it('pins exact represented continuation and every 24-bit Bernoulli threshold', () => {
    expect(source).toContain('@group(0) @binding(11) var<storage, read> sceneOpticalTriangleIdentity');
    expect(source).toContain('@group(0) @binding(12) var<storage, read> sceneOpticalInstanceBoundaryIdBasePlusOne');
    expect(source).toContain('fn ddgiTraceFirstHitAlphaMaskTexturedWithOpticalSource(');
    expect(source).toContain('traceSceneFirstHitWithOpticalSourceExclusion(');
    expect(source).toContain('let nextRay = Ray(currentPos, rayDirection);');
    expect(source).toContain('represented_bernoulli_probability_f32(0.5)');
    expect(source).toContain('let representedCoverage = represented_bernoulli_probability_f32(');
    expect(source).toContain('let representedQ = represented_bernoulli_probability_f32(q);');
    expect(source).toContain('let representedQ = represented_bernoulli_probability_f32(entry.x);');
    expect(source).not.toContain('currentPos + rayDirection *');
  });
});
