import { describe, expect, it } from 'vitest';
import { REPRESENTED_PROPOSAL_BUCKET_COUNT } from '@vitrum/shared-samplers';
import { applyRepresentedEmitterProposalPmf } from './representedEmitterProposal.js';
import type { EnvTextureData, LightsTextureData } from './sceneTextures.js';
import type { MeshAreaLightsData } from './meshAreaLights.js';
import * as BdptLightSubpathNS from '../glsl/render/bdpt_light_subpath.glsl.js';
import * as DirectLightContributionNS from '../glsl/render/direct_light_contribution_function.glsl.js';
import * as LightSamplingFunctionsNS from '../glsl/shader/sampling/light_sampling_functions.glsl.js';
import * as EquirectFunctionsNS from '../glsl/shader/sampling/equirect_sampling_functions.glsl.js';

function glslChunk(namespace: object, name: string): string {
  const chunk = (namespace as Record<string, unknown>)[name];
  if (typeof chunk !== 'string') {
    throw new TypeError(`Expected GLSL module "${name}" to export a string.`);
  }
  return chunk;
}

const bdpt_light_subpath = glslChunk(BdptLightSubpathNS, 'bdpt_light_subpath');
const direct_light_contribution_function = glslChunk(
  DirectLightContributionNS,
  'direct_light_contribution_function',
);
const light_sampling_functions = glslChunk(
  LightSamplingFunctionsNS,
  'light_sampling_functions',
);
const equirect_functions = glslChunk(EquirectFunctionsNS, 'equirect_functions');

describe('represented WebGL2 global emitter proposal', () => {
  it('retains analytic, mesh, and environment support across adversarial ratios', () => {
    const lights: LightsTextureData = {
      data: new Float32Array(24),
      dim: 3,
      kind: 'rgba32f',
      lightCount: 1,
      proposalWeights: new Float64Array([2 ** -30]),
    };
    const mesh: MeshAreaLightsData = {
      data: new Float32Array(24),
      dim: 3,
      triLightCount: 1,
      totalEmissiveArea: 1,
      totalEmissivePower: 2 ** -30,
      proposalWeights: new Float64Array([2 ** -30]),
    };
    const env: EnvTextureData = {
      map: { data: new Float32Array([1, 1, 1, 0]), width: 1, height: 1 },
      marginal: null,
      conditional: { data: new Float32Array(4), width: 1, height: 1 },
      totalSum: 1,
    };

    applyRepresentedEmitterProposalPmf(lights, mesh, env, { kind: 'none' });

    const pmfs = [lights.data[23]!, mesh.data![18]!, env.conditional!.data[2]!];
    expect(pmfs[0]).toBe(1 / REPRESENTED_PROPOSAL_BUCKET_COUNT);
    expect(pmfs[1]).toBe(1 / REPRESENTED_PROPOSAL_BUCKET_COUNT);
    expect(pmfs[2]).toBe(
      (REPRESENTED_PROPOSAL_BUCKET_COUNT - 2) /
        REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
    expect(Math.fround(pmfs[0]! + pmfs[1]! + pmfs[2]!)).toBe(1);
    expect(bdpt_light_subpath).toContain('tri.bdptProposalPmf <= 0.0');
    expect(bdpt_light_subpath).not.toContain('tri.power <= 0.0');
  });

  it('assigns every represented endpoint to the next positive emitter interval', () => {
    const bucketCount = REPRESENTED_PROPOSAL_BUCKET_COUNT;
    const pmfs = [1 / bucketCount, 1 / bucketCount, (bucketCount - 2) / bucketCount];
    const select = (bucket: number): number => {
      const xi = bucket / bucketCount;
      let cumulative = 0;
      for (let i = 0; i < pmfs.length; i += 1) {
        cumulative = Math.fround(cumulative + pmfs[i]!);
        if (xi < cumulative) return i;
      }
      return pmfs.length - 1;
    };

    expect(select(0)).toBe(0);
    expect(select(1)).toBe(1);
    expect(select(2)).toBe(2);
    expect(select(bucketCount - 1)).toBe(2);
    expect(light_sampling_functions).toContain('if ( uPick < cum ) break;');
    expect(direct_light_contribution_function).toContain('if ( pick < cumulative ) break;');
    expect(bdpt_light_subpath).toContain('p > 0.0 && pick < cumulative');
  });

  it('uses first-forward-CDF-strictly-greater inversion for environment rows and columns', () => {
    // Each search stores endpoint CDF[i+1] at texel i. Equality therefore
    // advances lo to mid+1 and selects the following half-open interval.
    expect(equirect_functions.match(/if \( cdf <= xi \)/g)).toHaveLength(2);
    expect(equirect_functions.match(/lo = mid \+ 1;/g)).toHaveLength(2);
  });
});
