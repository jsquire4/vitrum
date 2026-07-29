import { describe, expect, it } from 'vitest';
import {
  composeNeeCandidateGlsl,
  composeNeeResolveGlsl,
  composeTraceGlsl,
} from '../glsl/composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';

function finiteEmitterPmf(powers: readonly number[]): number[] {
  const finite = powers.map((power) => Number.isFinite(power) ? Math.max(power, 0) : 0);
  const total = finite.reduce((sum, power) => sum + power, 0);
  return total > 0 ? finite.map((power) => power / total) : finite.map(() => 0);
}

function powerWeights(densities: readonly number[]): number[] {
  const powers = densities.map((density) => density * density);
  const denominator = powers.reduce((sum, power) => sum + power, 0);
  return denominator > 0
    ? powers.map((power) => power / denominator)
    : powers.map(() => 0);
}

describe('production general BDPT estimator', () => {
  const source = composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES, bdpt: true });
  const candidateSource = composeNeeCandidateGlsl({
    ...DEFAULT_TRACE_FEATURES,
    bdpt: true,
  });
  const resolveSource = composeNeeResolveGlsl({
    ...DEFAULT_TRACE_FEATURES,
    bdpt: true,
  });

  it('power-partitions every live emitter family, including the environment', () => {
    // Rect, point, disc, spot, directional, mesh, environment.
    const powers = [2, 1, 3, 4, 2, 5, 3];
    const total = powers.reduce((sum, power) => sum + power, 0);
    const pmf = finiteEmitterPmf(powers);

    expect(pmf.reduce((sum, probability) => sum + probability, 0)).toBeCloseTo(1, 14);
    powers.forEach((power, index) => {
      expect(power / pmf[index]!).toBeCloseTo(total, 14);
    });
    expect(finiteEmitterPmf([0, 1e-25])).toEqual([0, 1]);

    expect(source).toContain('bdptAnalyticEmitterPower( i )');
    expect(source).toContain('bdptMeshEmitterPower( i )');
    expect(source).toContain('bdptEnvironmentEmitterPower()');
    expect(source).toContain('finitePositiveLightPower');
    expect(source).not.toContain('max( tmpLight.power, 1e-20 )');
    expect(source).not.toContain('sumPower > 1e-30');
    expect(source).toContain('selectedFamily = 2');
      expect(source).toContain('BDPT_LV_POINT_EMITTER_MATID');
      expect(source).toContain('BDPT_LV_SPOT_EMITTER_MATID');
      expect(source).toContain('BDPT_LV_POINT_EMITTER_MATID, BDPT_KIND_LIGHT');
      expect(source).toContain('BDPT_LV_SPOT_EMITTER_MATID, BDPT_KIND_LIGHT');
      expect(source).not.toContain('BDPT_LV_POINT_EMITTER_MATID, BDPT_KIND_DELTA');
    expect(source).toContain('BDPT_LV_DIRECTIONAL_EMITTER_MATID');
      expect(source).toContain('BDPT_LV_ENVIRONMENT_EMITTER_MATID');
      expect(source).toContain('vec4( tri.castShadowDisabled, 1.0, 0.0, 0.0 )');
      expect(source).toContain('vec4( light.castShadowDisabled, 0.0, 0.0, 0.0 )');
  });

  it('extends real BSDF subpaths and patches predecessor reverse densities', () => {
    expect(source).toContain('ScatterRecord rec = bsdfSample');
    expect(source).toContain('bdptLoadSurfaceRecord');
    expect(source).toContain('bdptSampledSurfaceEventIsDelta');
    expect(source).toContain('bdptPredecessor0');
    expect(source).toContain('bdptPredecessor2');
    expect(source).toContain(
      'predecessor2.w = max( reverseScatterPdf, 0.0 ) * p2.w;',
    );
    expect(source.replace(/\s+/g, ' ')).toContain(
      'predecessor2 = texelFetch( lightPathTex, ivec2( vertexCol - 2, 2 ), 0 );',
    );
    expect(source).not.toContain('predecessor2 = p2;');
    expect(source).toContain(
      'v2 = vec4( newThroughput, segmentReverseDensity );',
    );
    expect(source).toContain('bdptCol == uBdptVertexCol - 2');
    expect(source).not.toContain(
      'bdptCol == uBdptVertexCol - 1 && bdptRow == 2',
    );
    expect(source).toContain('bdptEyeSegmentReverseDensity');
    expect(source).toContain('reverseScatterPdf = bsdfResult(');
    expect(source).toContain('float bdptSwappedRev = bsdfPdfResult(');
    expect(source.replace(/\s+/g, ' ')).toContain(
      'bdptSwappedRev * bdptEyeSegmentReverseDensity',
    );
    expect(source).toContain('lightVtxIdx - 1, 0');
    expect(source).toContain('mRev[ i ] = l2.w;');

    const patchTarget = (successorColumn: number) => successorColumn - 2;
    expect(patchTarget(2)).toBe(0);
    expect(patchTarget(3)).toBe(1);
    expect(patchTarget(7)).toBe(5);

    // Extending L0→L1→L2 discovers L0's reverse density, but must not replace
    // the emitter throughput with L1's throughput. That was a real c=2 bias:
    // the row-2 patch was routed to L0 while carrying all four channels of L1.
    const patchReverseDensity = (
      target: readonly [number, number, number, number],
      reverseDensity: number,
    ): readonly [number, number, number, number] => [
      target[0], target[1], target[2], reverseDensity,
    ];
    const emitterRow2 = [2.5, 2.5, 2.5, 0] as const;
    const firstScatterRow2 = [0.42, 0.31, 0.18, 1] as const;
    expect(patchReverseDensity(emitterRow2, 0.17)).toEqual([
      2.5, 2.5, 2.5, 0.17,
    ]);
    expect(patchReverseDensity(emitterRow2, 0.17)).not.toEqual([
      ...firstScatterRow2.slice(0, 3), 0.17,
    ]);
  });

  it('connects finite c=0 plus every stored c>=1 vertex and evaluates Veach MIS in log space', () => {
    expect(source).toContain('for ( int bdptLvi = 0; bdptLvi < 8; bdptLvi ++ )');
    expect(source).toContain('if ( bdptLvi >= uBdptMaxLightBounces ) break;');
    expect(source).not.toContain('lightVtxIdx != 1');
    expect(source).toContain('float logPdfs[ BDPT_MAX_MERGED ]');
    expect(source).toContain('maxPowerLog');
    expect(source).toContain('exp( 2.0 * logPdfs');
    expect(source).toContain('bool areaEndpoint = lightIsEndpoint');
    expect(source).toContain('bool pointEndpoint = lightIsEndpoint');
    expect(source).toContain('bool spotEndpoint = lightIsEndpoint');
    expect(source).toContain('if ( infiniteEndpoint ) return vec3( 0.0 );');
      expect(source).not.toContain('BDPT_CONTRIBUTION_CLAMP');
      expect(source).toContain('#if FEATURE_RUSSIAN_ROULETTE && ! FEATURE_BDPT');
      expect(source).toContain('bool twoSidedEndpoint = lv4.y > 0.5;');
      expect(source).toContain('twoSidedEndpoint ? 2.0 * PI : PI');
    expect(source).toContain('vec3 eeToPrev = normalize( eyeWo );');
    expect(source).not.toContain('vec3 eeMinusPos = camPos;');
    expect(source).toContain(
      'if ( bdptLvi + bdptEyeDepth >= bounces ) break;',
    );
    expect(source).toContain(
      'if ( lightVtxIdx + eyeDepth >= bounces ) return vec3( 0.0 );',
    );

    // c is the number of light-side scattering vertices and e+1 is the
    // number of eye-side scattering vertices. Their sum is strategy-invariant
    // for a fixed full path and must fit the ordinary accepted-bounce budget.
    const connectionFitsBudget = (c: number, e: number, bounces: number) =>
      c + e < bounces;
    expect(connectionFitsBudget(0, 0, 1)).toBe(true);
    expect(connectionFitsBudget(1, 0, 1)).toBe(false);
    expect(connectionFitsBudget(1, 1, 3)).toBe(true);
    expect(connectionFitsBudget(2, 1, 3)).toBe(false);
  });

  it('uses one power-heuristic denominator for distant s=0, s=1, and bounded s>=2', () => {
    const ownedBranchStart = candidateSource.indexOf(
      'float distantDenom = bdptDistantNeeDenom();',
    );
    const ownedBranchEnd = candidateSource.indexOf('#else', ownedBranchStart);
    const ownedBranch = candidateSource.slice(ownedBranchStart, ownedBranchEnd);

    expect(ownedBranchStart).toBeGreaterThan(-1);
    expect(ownedBranch).toContain('bdptSampleDirectionalNee');
    expect(ownedBranch).toContain('sampleEquirectProbability');
    expect(ownedBranch).not.toContain('randomLightSample');
    expect(ownedBranch).not.toContain('sampleMeshAreaLight');
    expect(candidateSource).toContain('bdptInfiniteEyeFamilyWeight(');
    expect(candidateSource).toContain('neeCrossFamilyMisWeight');
    expect(candidateSource).toContain('bdptEyePos[ BDPT_MAX_EYE_DEPTH ]');
    expect(candidateSource.indexOf('bdptEyePos[ BDPT_MAX_EYE_DEPTH ]')).toBeLessThan(
      candidateSource.lastIndexOf('bdptInfiniteEyeFamilyWeight('),
    );
    expect(candidateSource).not.toContain('writeLightSubpathVertex(');
    expect(source).toContain('bdptPendingEnvironmentMisWeight');
    expect(source).toContain('bdptInfiniteEyeFamilyWeight(');
    expect(source).toContain('validPdfs[ 0 ] = false;');
    expect(source).toContain('validPdfs[ 1 ] = false;');
    expect(source).toContain('validPdfs[ 2 ]');
    expect(source).toContain('! mSpec[ 1 ]');
    expect(source).toContain('float neeToLaunchAreaRatio =');
    expect(source).toContain('if ( terminalDelta ) return 0.0;');
    expect(source).toContain(
      'if ( selectedS == 0 && pureEyeSampledDelta ) return 1.0;',
    );
    expect(resolveSource).toContain('bdptCrossFamilyMisWeight');
    expect(resolveSource).toContain('misWeight = bdptCrossFamilyMisWeight');
    expect(source).toContain(
      'if ( prevMatId == BDPT_LV_ENVIRONMENT_EMITTER_MATID )',
    );
    expect(source).toContain('scatterThroughput *= newSurf.envMapIntensity;');

    const noHitStart = source.indexOf('if ( hitType == NO_HIT )');
    const noHitEnd = source.indexOf('uint materialIndex', noHitStart);
    const noHitBranch = source.slice(noHitStart, noHitEnd);
    expect(noHitStart).toBeGreaterThan(-1);
    expect(noHitBranch).toContain('bdptPendingEnvironmentMisWeight');

    const cases = {
      primaryEnvironment: [1, 0, 0],
      diffuseToEnvironment: [0.25, 0.5, 0],
      deltaToEnvironment: [1, 0, 0],
      diffuseToSurfaceToEnvironment: [0.2, 0.4, 0.1],
      mirrorThenDiffuseToEnvironment: [0.2, 0.4, 0],
    } as const;
    for (const densities of Object.values(cases)) {
      const weights = powerWeights(densities);
      expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 14);
      weights.forEach((weight) => expect(Number.isFinite(weight)).toBe(true));
    }
    expect(powerWeights(cases.diffuseToEnvironment)).toEqual([0.2, 0.8, 0]);
    expect(powerWeights(cases.deltaToEnvironment)).toEqual([1, 0, 0]);
    expect(powerWeights(cases.mirrorThenDiffuseToEnvironment)[2]).toBe(0);
  });

  it('does not misclassify a high-density rough-glossy arrival as delta', () => {
    const roughGlossy = { pdf: 12.5, sampledDelta: false };
    const keepUniqueForwardEmission = (firstRay: boolean, sampledDelta: boolean) =>
      firstRay || sampledDelta;

    // The retired density threshold would classify this finite lobe as unique.
    expect(roughGlossy.pdf).toBeGreaterThan(0.999);
    expect(keepUniqueForwardEmission(false, roughGlossy.sampledDelta)).toBe(false);
    expect(source).toContain('bool sampledDelta;');
    expect(source).toContain('result.sampledDelta = sampledDelta;');
    expect(source).toContain('sampleRec.sampledDelta = false;');
    expect(source).not.toContain('scatterRec.specularPdf > 0.999');
  });

  it('carries nested homogeneous-medium vertices through construction, visibility, and connection', () => {
    expect(source).toContain('BDPT_LV_MEDIUM_MATID');
    expect(source).toContain('const int MEDIUM_STACK_CAPACITY = 8;');
    expect(source).toContain('bdptPackMediumStack');
    expect(source).toContain('bdptUnpackMediumStack');
    expect(source).toContain('bvhBuildMediumStack');
    expect(source).toContain('enterMedium(');
    expect(source).toContain('leaveMedium(');
    expect(source).toContain('segmentForwardDensity');
    expect(source).toContain('segmentReverseDensity');
    expect(source).toContain('segmentRatioWeight');
    expect(source).toContain('fogFreeFlightRatioWeight');
    expect(source).toContain('fogSegmentTransmittance');
    expect(source).toContain('mediumPhasePdf');
    expect(source).toContain('sampleMediumPhase');
    expect(source).toContain('lightIsMedium');
    expect(source).toContain('eyeMedium');
  });

  it('initialises the first light segment from its translated endpoint launch ray', () => {
    const insideSphere = (
      point: readonly [number, number, number],
      center: readonly [number, number, number],
      radius: number,
    ) => Math.hypot(
      point[0] - center[0],
      point[1] - center[1],
      point[2] - center[2],
    ) < radius;
    const translatedCenter = [12, -3, 5] as const;
    const sampledEndpoint = [12.25, -3, 5] as const;

    expect(insideSphere([0, 0, 0], translatedCenter, 2)).toBe(false);
    expect(insideSphere(sampledEndpoint, translatedCenter, 2)).toBe(true);
    expect(source).toContain('vec3 endpointLaunchOrigin =');
    expect(source).toContain('p0.xyz + endpointLaunchDirection * RAY_OFFSET');
    expect(source).toContain('bvhBuildMediumStack(');
    expect(source).toContain('endpointLaunchOrigin,');
    expect(source).not.toContain('fogRay.origin = vec3( 0.0 );');
  });
});
