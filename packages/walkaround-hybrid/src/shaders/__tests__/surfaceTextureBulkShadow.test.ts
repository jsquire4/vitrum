import { describe, expect, it } from 'vitest';

import { SURFACE_TEXTURES_WGSL } from '../surfaceTextures.wgsl.js';

type Rgb = readonly [number, number, number];

function segmentTransfer(
  attenuationColor: Rgb,
  authoredThickness: number,
  thicknessMapScale: number,
  scattering: Rgb,
  segmentLength: number,
): Rgb {
  const transportDistance = authoredThickness > 0
    ? Math.min(
        segmentLength,
        authoredThickness * Math.min(Math.max(thicknessMapScale, 0), 1),
      )
    : segmentLength;
  const channelTransfer = (channel: 0 | 1 | 2): number =>
    Math.pow(
      attenuationColor[channel],
      transportDistance / Math.max(authoredThickness, 1),
    ) * Math.exp(-scattering[channel] * transportDistance);
  return [channelTransfer(0), channelTransfer(1), channelTransfer(2)];
}

function applyBulkBoundary(
  depth: number,
  encodedBoundaryId: number,
  entering: boolean,
): number {
  if (encodedBoundaryId === 0) return depth;
  if (entering) return depth + 1;
  if (depth === 0) throw new Error('unpaired bulk exit');
  return depth - 1;
}

describe('shared textured shadow bulk transport', () => {
  it('keeps topology stable when entry and exit sample different thickness texels', () => {
    for (const [entryScale, exitScale] of [[0, 1], [1, 0]] as const) {
      let depth = 0;
      // The map values deliberately do not participate in the state transition.
      expect(entryScale).not.toBe(exitScale);
      depth = applyBulkBoundary(depth, 7, true);
      expect(depth).toBe(1);
      depth = applyBulkBoundary(depth, 7, false);
      expect(depth).toBe(0);
    }
  });

  it('applies the mapped positive-thickness cap to absorption and scattering', () => {
    const color: Rgb = [0.25, 0.5, 0.8];
    const sigmaS: Rgb = [0.1, 0.2, 0.3];
    const segment = 2.5;
    const distance = 4;

    const zero = segmentTransfer(color, distance, 0, sigmaS, segment);
    const half = segmentTransfer(color, distance, 0.5, sigmaS, segment);
    const full = segmentTransfer(color, distance, 1, sigmaS, segment);
    for (const channel of [0, 1, 2] as const) {
      expect(zero[channel]).toBeCloseTo(
        1,
        12,
      );
      expect(full[channel]).toBeLessThan(half[channel]);
      expect(half[channel]).toBeLessThan(zero[channel]);
    }
  });

  it('wires component topology, positive caps, spectral scaling, and scatter extinction', () => {
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'fn materialShadowAuthoredThickness(hit: IntersectionResult)',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'referenceThickness * clamp(thicknessMapScale, 0.0, 1.0)',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'mediumState.thicknessMapScale[mediumState.depth] = thicknessMapScale;',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'mediumState.scattering[mediumState.depth] =',
    );
    expect(SURFACE_TEXTURES_WGSL).not.toContain(
      'fn materialShadowEffectiveThickness(',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'let bulkMedium = boundaryId != 0u;',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'mediumState.instance[mediumState.depth] = representedId;',
    );
  });

  it('copies seeded state per proposal and distinguishes paid from unpaid exits', () => {
    const scalar = 0.37;
    const face = 0.82;
    const segment = segmentTransfer(
      [0.4, 0.6, 0.8],
      3,
      0.5,
      [0.1, 0.2, 0.3],
      1.25,
    );
    const paidExit = segment.map((value) => value * face) as unknown as Rgb;
    const unpaidExit = segment.map(
      (value) => value * face * scalar,
    ) as unknown as Rgb;
    for (const channel of [0, 1, 2] as const) {
      expect(unpaidExit[channel]).toBeCloseTo(
        paidExit[channel] * scalar,
        14,
      );
      expect(paidExit[channel]).toBeGreaterThan(unpaidExit[channel]);
    }

    expect(SURFACE_TEXTURES_WGSL).toContain(
      'struct MaterialShadowMediumState {',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'var mediumState = initialMediumState;',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'if (mediumState.transmissionPaid[top] == 0u)',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'materialShadowMappedTransmission(hit)',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'mediumState.transmissionPaid[mediumState.depth] = 1u;',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'actualContainingMedia.state, blockMaterialTransmission,',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'let containingMedia = materialShadowClassifyContainingMedia(',
    );
  });

  it('integrates a finite endpoint inside the seeded top medium exactly once', () => {
    const transfer = segmentTransfer(
      [0.31, 0.52, 0.79],
      2.7,
      0.4,
      [0.14, 0.22, 0.38],
      0.65,
    );
    const replayed = transfer.map((value) => value * value);
    for (const channel of [0, 1, 2] as const) {
      expect(transfer[channel]).toBeGreaterThan(replayed[channel]!);
    }
    expect(SURFACE_TEXTURES_WGSL).toContain(
      '&mediumState, max(tMax - exclusiveMinT, 0.0),',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'tau * materialShadowEndpointTransmission(',
    );
    expect(SURFACE_TEXTURES_WGSL).toMatch(
      /materialShadowEndpointTransmission\([\s\S]*?state\.distance\[depth\],[\s\S]*?return clamp\(attenuation,/,
    );
  });

  it('resolves alpha coverage before mutating a bulk boundary stack', () => {
    const coverage = SURFACE_TEXTURES_WGSL.indexOf(
      'let coverage = materialShadowCoverageForHit(hit, word);',
      SURFACE_TEXTURES_WGSL.indexOf(
        'fn traceSceneAlphaTintTransmittanceTexturedWithState(',
      ),
    );
    const pairedExit = SURFACE_TEXTURES_WGSL.indexOf(
      'let pairedExit =', coverage,
    );
    expect(coverage).toBeGreaterThan(0);
    expect(pairedExit).toBeGreaterThan(coverage);
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'if (bulkMedium && coverage < 1.0)',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'A remaining unmatched back face is broken',
    );
  });

  it('replays outside-to-origin crossings without a disjoint-shell key ceiling', () => {
    const stack: string[] = [];
    let peakDepth = 0;
    const cross = (key: string, entering: boolean): void => {
      if (entering) {
        stack.push(key);
        peakDepth = Math.max(peakDepth, stack.length);
        return;
      }
      expect(stack.pop()).toBe(key);
    };
    for (let shell = 0; shell < 80; shell += 1) {
      cross(`disjoint-${shell}`, true);
      cross(`disjoint-${shell}`, false);
    }
    for (let shell = 0; shell < 3; shell += 1) {
      cross(`containing-${shell}`, true);
    }
    expect(stack).toEqual([
      'containing-0',
      'containing-1',
      'containing-2',
    ]);
    expect(peakDepth).toBe(3);

    const start = SURFACE_TEXTURES_WGSL.indexOf(
      'fn materialShadowClassifyContainingMedia(',
    );
    const end = SURFACE_TEXTURES_WGSL.indexOf(
      'fn traceSceneAlphaTintTransmittanceTexturedWithState(', start,
    );
    const classifier = SURFACE_TEXTURES_WGSL.slice(start, end);
    expect(classifier).toContain('let ray = Ray(origin, direction);');
    expect(classifier).toContain('temporaryBoundaryId[temporaryDepth] =');
    expect(classifier).toContain(
      'temporaryBoundaryId[top] != event.encodedBoundaryId',
    );
    expect(classifier).toContain(
      'innerToOuter.materialId[depth] = event.encodedBoundaryId;',
    );
    expect(classifier).toContain(
      'innerToOuter.instance[depth] = event.representedPrimitiveInstanceId;',
    );
    expect(classifier).toContain('innerToOuter.transmissionPaid[depth] = 0u;');
    expect(classifier).toContain(
      'if (!complete || temporaryDepth != 0u) { return out; }',
    );
    expect(classifier).toContain(
      'let source = innerToOuter.depth - 1u - destination;',
    );
    expect(classifier).not.toContain('CONTAINMENT_KEY_CAPACITY');
    expect(classifier).not.toContain('keyCount');
  });

  it('defines infinite-segment lane limits without evaluating one-to-infinity powers', () => {
    const infiniteLane = (
      tint: number,
      thicknessScale: number,
      scattering: number,
    ): number => {
      if (![tint, thicknessScale, scattering].every(Number.isFinite)) return 0;
      const absorption = thicknessScale <= 0 || tint === 1 ? 1 : 0;
      const scatter = scattering <= 0 ? 1 : 0;
      return absorption * scatter;
    };
    expect(infiniteLane(1, 1, 0)).toBe(1);
    expect(infiniteLane(0.999, 1, 0)).toBe(0);
    expect(infiniteLane(0.2, 0, 0)).toBe(1);
    expect(infiniteLane(1, 1, 1e-12)).toBe(0);
    expect(infiniteLane(Number.NaN, 1, 0)).toBe(0);
    expect(infiniteLane(1, Number.POSITIVE_INFINITY, 0)).toBe(0);

    const beerStart = SURFACE_TEXTURES_WGSL.indexOf(
      'fn materialShadowBeerForSegment(',
    );
    const beerEnd = SURFACE_TEXTURES_WGSL.indexOf(
      'fn materialShadowSmoothNormalForHit(', beerStart,
    );
    const beer = SURFACE_TEXTURES_WGSL.slice(beerStart, beerEnd);
    expect(beer).toContain('if (transportDistance > 3.402823466e38)');
    expect(beer).toContain(
      'materialBeerTransmittanceExact(sample.x, transportDistance)',
    );
    expect(beer).not.toContain('max(sample.x, 0.0) == 0.0');
    expect(beer).toContain('select(0.0, 1.0, tint.x == 1.0)');
    expect(beer).toContain('homogeneousBeerTransmittanceRgb(');
    expect(beer).toContain(
      'if (exponent > MATERIAL_SHADOW_MAX_FINITE_F32)',
    );
    expect(beer).toContain('rgbBeer = pow(tint, vec3f(exponent));');
    expect(beer.indexOf('if (transportDistance > 3.402823466e38)'))
      .toBeLessThan(beer.indexOf('rgbBeer = pow('));
  });

  it('validates finite endpoint containment while requiring an empty stack at infinity', () => {
    const start = SURFACE_TEXTURES_WGSL.indexOf(
      'fn materialShadowEndpointTransmission(',
    );
    const end = SURFACE_TEXTURES_WGSL.indexOf(
      'fn traceSceneAlphaTintTransmittanceTexturedWithState(', start,
    );
    const endpoint = SURFACE_TEXTURES_WGSL.slice(start, end);
    expect(endpoint).toContain(
      'return select(vec3f(0.0), vec3f(1.0), state.depth == 0u);',
    );
    expect(endpoint).toContain('materialShadowClassifyContainingMedia(');
    expect(endpoint).toContain(
      'classified.state.depth != state.depth',
    );
    expect(endpoint).toContain(
      'classified.state.materialId[depth] != state.materialId[depth]',
    );
    expect(endpoint).toContain('classified.state.tri[depth],');
    expect(endpoint).toContain('state.distance[depth],');
    expect(SURFACE_TEXTURES_WGSL.match(
      /materialShadowEndpointTransmission\(/g,
    )).toHaveLength(4);
  });

  it('multiplies both reciprocal thin-sheet face controls in either direction', () => {
    const front: Rgb = [0.8, 0.5, 0.9];
    const back: Rgb = [0.4, 0.7, 0.6];
    const destructiveFilm: Rgb = [0.2, 0.3, 0.1];
    const frontToBack = front.map(
      (value, channel) =>
        value * back[channel]! * destructiveFilm[channel]!,
    );
    const backToFront = back.map(
      (value, channel) =>
        value * front[channel]! * destructiveFilm[channel]!,
    );
    expect(frontToBack).toEqual(backToFront);
    for (const channel of [0, 1, 2] as const) {
      const incorrectlySquared = front[channel] * back[channel] *
        destructiveFilm[channel] * destructiveFilm[channel];
      expect(frontToBack[channel]).toBeGreaterThan(incorrectlySquared);
    }

    const start = SURFACE_TEXTURES_WGSL.indexOf(
      'fn materialShadowThinSheetTransmission(',
    );
    const end = SURFACE_TEXTURES_WGSL.indexOf(
      'fn materialShadowCoverageForHit(', start,
    );
    const helper = SURFACE_TEXTURES_WGSL.slice(start, end);
    expect(helper).toContain('let incidentFrontFacing = hit.side >= 0.0;');
    expect(helper).toContain(
      'materialShadowFaceAbsorptionForSide(hit, !incidentFrontFacing)',
    );
    expect(helper.match(/materialShadowFaceAbsorptionForSide\(/g))
      .toHaveLength(2);
    expect(helper.match(/materialShadowThinFilmTransmissionForSide\(/g))
      .toHaveLength(1);
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'materialShadowThinSheetTransmission(hit, dir) *',
    );
  });
});
