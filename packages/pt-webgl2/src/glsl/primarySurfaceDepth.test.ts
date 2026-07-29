import { describe, expect, it } from 'vitest';
import { DEFAULT_TRACE_FEATURES, type TraceFeatures } from '../featureTypes.js';
import {
  composeNeeCandidateGlsl,
  composeNeeResolveGlsl,
  composeTraceGlsl,
} from './composeTraceGlsl.js';

const MAPPED_TIERS: ReadonlyArray<{
  readonly name: string;
  readonly features: TraceFeatures;
}> = [
  {
    name: 'mapped-rich',
    features: DEFAULT_TRACE_FEATURES,
  },
  {
    name: 'mapped-pbr',
    features: {
      ...DEFAULT_TRACE_FEATURES,
      mappedPbrMaterials: true,
      mappedRichMaterials: false,
    },
  },
];

function compact(source: string): string {
  return source.replace(/\s+/g, ' ');
}

type CameraIntersection = 'alpha-skip' | 'fog-boundary' | 'surface';

function acceptedSurfaceDepths(
  intersections: readonly CameraIntersection[],
): number[] {
  const accepted: number[] = [];
  let bounceIndex = 0;
  let transmissiveTraversals = 8;

  for (const intersection of intersections) {
    const surfacePathDepth = bounceIndex;
    if (intersection === 'surface') {
      accepted.push(surfacePathDepth);
    } else {
      const rewind = Math.sign(transmissiveTraversals);
      bounceIndex -= rewind;
      transmissiveTraversals -= rewind;
    }
    // Mirrors the render loop's i++ update expression.
    bounceIndex += 1;
  }

  return accepted;
}

function packCandidatePathDepth(pathDepth: number): number {
  return (Math.min(pathDepth, 127) << 3) >>> 0;
}

function unpackCandidatePathDepth(flags: number): number {
  return (flags >>> 3) & 127;
}

describe('primary camera-surface depth', () => {
  for (const { name, features } of MAPPED_TIERS) {
    it(`${name} reaches lightMap at accepted depth zero in trace and NEE replay`, () => {
      const trace = compact(composeTraceGlsl(features));
      const candidate = compact(composeNeeCandidateGlsl(features));
      const resolve = compact(composeNeeResolveGlsl(features));

      for (const source of [trace, candidate, resolve]) {
        expect(source).toMatch(
          /material\.lightMap\s*!=\s*-\s*1\s*&&\s*pathDepth\s*==\s*0/,
        );
      }
      for (const source of [trace, candidate]) {
        expect(source).toContain('int surfacePathDepth = i;');
        expect(source).toContain(
          'state.accumulatedRoughness, surfacePathDepth, state.wavelength',
        );
      }
      expect(candidate).toContain(
        'min( uint( surfacePathDepth ), 127u ) << 3u',
      );
      expect(resolve).toContain(
        'candidate0.w, int( pathDepth ), candidate1.w, true, surf',
      );
    });
  }

  it('keeps the first accepted surface primary through pass-throughs and excludes indirect lightMap energy', () => {
    const depths = acceptedSurfaceDepths([
      'alpha-skip',
      'fog-boundary',
      'surface',
      'surface',
    ]);
    expect(depths).toEqual([0, 1]);

    const baseEmission = 0.25;
    const bakedRadiance = 1.5;
    const lightMapIntensity = 0.8;
    const emitted = depths.map(
      (depth) =>
        baseEmission +
        (depth === 0 ? bakedRadiance * lightMapIntensity : 0),
    );
    expect(emitted[0]).toBeCloseTo(1.45, 15);
    expect(emitted[1]).toBe(0.25);

    for (const depth of depths) {
      expect(
        unpackCandidatePathDepth(packCandidatePathDepth(depth)),
      ).toBe(depth);
    }
  });
});
