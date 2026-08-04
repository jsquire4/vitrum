import { describe, expect, it } from 'vitest';
import { DEFAULT_TRACE_FEATURES, type TraceFeatures } from '../featureTypes.js';
import {
  composeNeeCandidateGlsl,
  composeTraceGlsl,
} from './composeTraceGlsl.js';

interface TierEvidence {
  readonly name: string;
  readonly features: TraceFeatures;
  readonly decoderFetch: string;
  readonly decoderAssignment: string;
  readonly radianceCoverageEvidence: string;
}

const TIERS: readonly TierEvidence[] = [
  {
    name: 'basic',
    features: {
      ...DEFAULT_TRACE_FEATURES,
      basicMaterials: true,
      mappedRichMaterials: false,
    },
    decoderFetch: 'vec4 s13 = texelFetch1D( tex, i + 13u );',
    decoderAssignment: 'material.side = s13.a;',
    radianceCoverageEvidence:
      'material.side != 0.0 && surfaceHit.side != material.side',
  },
  {
    name: 'scalar-rich',
    features: {
      ...DEFAULT_TRACE_FEATURES,
      scalarRichMaterials: true,
      mappedRichMaterials: false,
    },
    decoderFetch: 'vec4 s13 = texelFetch1D( tex, i + 13u );',
    decoderAssignment: 'm.side = s13.a;',
    radianceCoverageEvidence:
      'material.side, surfaceHit.side, material.alphaTest',
  },
  {
    name: 'mapped-pbr',
    features: {
      ...DEFAULT_TRACE_FEATURES,
      mappedPbrMaterials: true,
      mappedRichMaterials: false,
    },
    decoderFetch: 'vec4 s13 = texelFetch1D( tex, i + 13u );',
    decoderAssignment: 'm.alphaTest = s13.b; m.side = s13.a;',
    radianceCoverageEvidence:
      'material.side != 0.0 && surfaceHit.side != material.side',
  },
  {
    name: 'mapped-rich',
    features: DEFAULT_TRACE_FEATURES,
    decoderFetch: 's = texelFetch1D( tex, i + 13u );',
    decoderAssignment: 'm.alphaTest = s.b; m.side = s.a;',
    radianceCoverageEvidence:
      'material.side, surfaceHit.side, alphaTest',
  },
];

describe('pt-webgl2 doubleSided compiler-tier evidence', () => {
  for (const tier of TIERS) {
    it(`${tier.name} consumes the packed side lane for radiance and visibility`, () => {
      const trace = composeTraceGlsl(tier.features);
      const visibility = composeNeeCandidateGlsl(tier.features);

      for (const source of [trace, visibility]) {
        expect(source).toContain(tier.decoderFetch);
        expect(source).toContain(tier.decoderAssignment);
      }
      expect(trace).toContain(tier.radianceCoverageEvidence);
      expect(visibility).toContain(
        'material.side != 0.0 && surfaceHit.side != material.side',
      );
      expect(visibility).not.toContain(
        'material.side != 0.0 && surfaceHit.side == material.side',
      );
      expect(trace).toContain('vec3 sampledNormal = textureSampleBarycoord');
      expect(trace).toContain('vec3 normal = length( sampledNormal ) > 1e-6');
      expect(trace).not.toContain('vec3 normal = normalize( textureSampleBarycoord');
    });
  }

  it('covers every material compiler tier exactly once', () => {
    expect(TIERS.map((tier) => tier.name)).toEqual([
      'basic',
      'scalar-rich',
      'mapped-pbr',
      'mapped-rich',
    ]);
  });
});
