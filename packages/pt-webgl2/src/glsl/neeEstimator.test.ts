import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';
import {
  composeNeeCandidateGlsl,
  composeNeeResolveGlsl,
  composeTraceGlsl,
} from './composeTraceGlsl.js';
import { NEE_RESOLVE_MAIN } from './neeResolveMain.glsl.js';
import { RENDER_MAIN } from './renderMain.glsl.js';
import { BLEND_FRAG } from '../gl/blendFrag.js';
import * as DirectLightSource from './render/direct_light_contribution_function.glsl.js';
import * as SurfaceRecordSource from './render/get_surface_record_function.glsl.js';

type U32x4 = [number, number, number, number];

const directLightSource = (
  DirectLightSource as unknown as Record<string, string>
).direct_light_contribution_function!;
const surfaceRecordSource = (
  SurfaceRecordSource as unknown as Record<string, string>
).get_surface_record_function!;
const glResourcesSource = readFileSync(
  fileURLToPath(new URL('../gl/glResources.ts', import.meta.url)),
  'utf8',
);

function pcg4d(value: U32x4): U32x4 {
  const v: U32x4 = value.map((x) => x >>> 0) as U32x4;
  for (let i = 0; i < 4; i += 1) {
    v[i] = (Math.imul(v[i]!, 1_664_525) + 1_013_904_223) >>> 0;
  }
  v[0] = (v[0] + Math.imul(v[1], v[3])) >>> 0;
  v[1] = (v[1] + Math.imul(v[2], v[0])) >>> 0;
  v[2] = (v[2] + Math.imul(v[0], v[1])) >>> 0;
  v[3] = (v[3] + Math.imul(v[1], v[2])) >>> 0;
  for (let i = 0; i < 4; i += 1) v[i] = (v[i]! ^ (v[i]! >>> 16)) >>> 0;
  v[0] = (v[0] + Math.imul(v[1], v[3])) >>> 0;
  v[1] = (v[1] + Math.imul(v[2], v[0])) >>> 0;
  v[2] = (v[2] + Math.imul(v[0], v[1])) >>> 0;
  v[3] = (v[3] + Math.imul(v[1], v[2])) >>> 0;
  return v;
}

function stripCandidateDisabledBlocks(source: string): string {
  const output: string[] = [];
  const disabledDepths: number[] = [];
  let depth = 0;
  for (const line of source.split('\n')) {
    const trimmed = line.trim();
    if (/^#if(?:def|ndef)?\b/.test(trimmed)) {
      depth += 1;
      if (trimmed === '#if FEATURE_BDPT && ! NEE_CANDIDATE_PASS') {
        disabledDepths.push(depth);
      }
      if (disabledDepths.length === 0) output.push(line);
      continue;
    }
    if (trimmed === '#endif') {
      if (disabledDepths.at(-1) === depth) disabledDepths.pop();
      depth -= 1;
      if (disabledDepths.length === 0) output.push(line);
      continue;
    }
    if (disabledDepths.length === 0) output.push(line);
  }
  return output.join('\n');
}

describe('separate NEE estimator', () => {
  const main = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
  const candidate = composeNeeCandidateGlsl({
    ...DEFAULT_TRACE_FEATURES,
    bdpt: true,
  });
  const resolve = composeNeeResolveGlsl();

  it('physically omits candidate-only NEE helpers from the ordinary trace', () => {
    expect(main).not.toContain('struct DirectLightSample');
    expect(main).not.toContain('sampleDirectLight(');
    expect(main).not.toContain('prepareDirectLightSample(');
    expect(main).not.toContain('evaluatePreparedDirectLightSample(');
    expect(main).not.toContain('gNeeCandidate3');
  });

  it('keeps arbitrary light-direction BSDF evaluation out of candidate path reachability', () => {
    expect(candidate).toContain('struct DirectLightSample');
    expect(candidate).toContain('surf, geometricHitPoint');
    expect(candidate).not.toContain('vec3 evaluateBdptConnection(');
    const reachableMain = stripCandidateDisabledBlocks(
      candidate.slice(candidate.indexOf('void main() {')),
    );
    expect(reachableMain).not.toContain('bsdfResult(');
    expect(reachableMain).toContain('bsdfSample(');
  });

  it('makes the resolver path-loop-free while retaining exact BSDF and MIS evaluation', () => {
    expect(NEE_RESOLVE_MAIN).not.toMatch(/\b(?:for|while)\s*\(/);
    expect(resolve).not.toContain('pathStep');
    expect(resolve).not.toContain('RENDER_MAIN');
    expect(NEE_RESOLVE_MAIN).toContain('float bsdfPdf = bsdfResult(');
    expect(NEE_RESOLVE_MAIN).toContain('misHeuristic( lightPdf, bsdfPdf )');
    expect(NEE_RESOLVE_MAIN).toContain('deltaLight');
  });

  it('Horvitz-Thompson scaling exactly recovers the sum over all eligible vertices', () => {
    const contributions = [
      [0.25, 2, 0],
      [0, 0, 0],
      [1.5, 0.5, 3],
      [0.75, 1, 0.25],
    ] as const;
    const k = contributions.length;
    const expected = [0, 1, 2].map((channel) =>
      contributions.reduce((sum, value) => sum + value[channel]!, 0),
    );
    const averageEstimate = [0, 1, 2].map(
      (channel) =>
        contributions.reduce((sum, selected) => sum + k * selected[channel]!, 0) / k,
    );
    expect(averageEstimate).toEqual(expected);
  });

  it('counts and uniformly owns null proposals instead of conditioning the reservoir', () => {
    let probabilities = [1];
    for (let count = 2; count <= 8; count += 1) {
      probabilities = probabilities.map((p) => p * (1 - 1 / count));
      probabilities.push(1 / count);
    }
    for (const probability of probabilities) {
      expect(probability).toBeCloseTo(1 / probabilities.length, 14);
    }
    const countIncrement = RENDER_MAIN.indexOf('neeCandidateCount ++;');
    const replacement = RENDER_MAIN.indexOf('neeReservoirReplacementU(', countIncrement);
    const zeroRecord = RENDER_MAIN.indexOf('neeCandidate0 = vec4( 0.0 );', replacement);
    const proposal = RENDER_MAIN.indexOf('sampleDirectLight(', zeroRecord);
    expect(countIncrement).toBeGreaterThanOrEqual(0);
    expect(countIncrement).toBeLessThan(replacement);
    expect(replacement).toBeLessThan(zeroRecord);
    expect(zeroRecord).toBeLessThan(proposal);
  });

  it('restores PCG and Sobol state so replay proposals cannot perturb continuation RNG', () => {
    const initial: U32x4 = [23, 41, 101, 64];
    const baseline1 = pcg4d(initial);
    const baseline2 = pcg4d(baseline1);

    const saved = [...initial] as U32x4;
    let replayState = pcg4d(initial);
    replayState = pcg4d(replayState);
    expect(replayState).not.toEqual(saved);
    replayState = [...saved] as U32x4;
    expect(pcg4d(replayState)).toEqual(baseline1);
    expect(pcg4d(pcg4d(replayState))).toEqual(baseline2);

    const savePcg = RENDER_MAIN.indexOf('neeSavedWhiteNoiseSeed = WHITE_NOISE_SEED');
    const saveSobol = RENDER_MAIN.indexOf('neeSavedSobolBounceIndex = sobolBounceIndex');
    const proposal = RENDER_MAIN.indexOf('sampleDirectLight(', saveSobol);
    const restorePcg = RENDER_MAIN.indexOf('WHITE_NOISE_SEED = neeSavedWhiteNoiseSeed', proposal);
    const restoreSobol = RENDER_MAIN.indexOf('sobolBounceIndex = neeSavedSobolBounceIndex', proposal);
    expect(savePcg).toBeLessThan(proposal);
    expect(saveSobol).toBeLessThan(proposal);
    expect(restorePcg).toBeGreaterThan(proposal);
    expect(restoreSobol).toBeGreaterThan(proposal);
  });

  it('packs bounded path depth and K without corrupting validity/delta ownership', () => {
    const valid = 1;
    const delta = 1;
    const pathDepth = 32;
    const candidateCount = 64;
    const flags = valid | (delta << 2) | (pathDepth << 3) | (candidateCount << 10);
    expect(flags & 1).toBe(1);
    expect((flags >> 2) & 1).toBe(1);
    expect((flags >> 3) & 127).toBe(pathDepth);
    expect((flags >> 10) & 127).toBe(candidateCount);
    expect(NEE_RESOLVE_MAIN).toContain('candidateCount > 64u');
    expect(NEE_RESOLVE_MAIN).toContain('candidate1.w < 380.0 || candidate1.w > 780.0');
  });

  it('uses complementary power-heuristic ownership and one spectral reconstruction', () => {
    const pLight = 0.37;
    const pBsdf = 0.81;
    const lightWeight = pLight ** 2 / (pLight ** 2 + pBsdf ** 2);
    const bsdfWeight = pBsdf ** 2 / (pBsdf ** 2 + pLight ** 2);
    expect(lightWeight + bsdfWeight).toBeCloseTo(1, 15);
    const compactResolve = NEE_RESOLVE_MAIN.replace(/\s+/g, ' ');
    expect(compactResolve).toContain(
      '#if FEATURE_BDPT float misWeight = bdptCrossFamilyMisWeight; #else float misWeight = deltaLight ? 1.0 : misHeuristic( lightPdf, bsdfPdf ); #endif',
    );
    expect(NEE_RESOLVE_MAIN.match(/wavelengthToRGB\(/g)).toHaveLength(1);
    expect(NEE_RESOLVE_MAIN).toContain('neeHeroWavelengthPdf( candidate1.w )');
    expect(RENDER_MAIN).toContain('state.wavelength = uBdptSharedWavelength;');
    expect(RENDER_MAIN).toContain('state.wavelengthPdf = uBdptSharedWavelengthPdf;');
  });

  it('preserves finite-light measure so non-BDPT NEE and forward BSDF weights are complementary', () => {
    const nonBdptBranchStart = directLightSource.indexOf('#else');
    const nonBdptBranch = directLightSource.slice(
      nonBdptBranchStart,
      directLightSource.indexOf('#endif', nonBdptBranchStart),
    );
    expect(nonBdptBranch.replace(/\s+/g, ' ')).toContain(
      'lightSample.delta = lightRec.type == DIR_LIGHT_TYPE ? 1.0 : lightRec.delta;',
    );
    expect(nonBdptBranch).toContain('lightSample.delta = 0.0;');

    const powerWeight = (sampledPdf: number, competingPdf: number): number =>
      sampledPdf ** 2 / (sampledPdf ** 2 + competingPdf ** 2);
    for (const [lightPdf, bsdfPdf] of [
      [0.05, 0.8],
      [0.37, 0.81],
      [2.5, 0.04],
    ] as const) {
      const neeWeight = powerWeight(lightPdf, bsdfPdf);
      const forwardWeight = powerWeight(bsdfPdf, lightPdf);
      expect(neeWeight + forwardWeight).toBeCloseTo(1, 15);
      // The former forced-delta NEE weight produced this overweight sum.
      expect(1 + forwardWeight).toBeGreaterThan(1);
    }
  });

  it('accumulates the completed main-plus-NEE sample once without a hidden clamp', () => {
    expect(RENDER_MAIN).not.toContain('float sampleLuminance');
    expect(BLEND_FRAG).not.toContain('radianceClamp');
    expect(BLEND_FRAG).not.toContain('sampleLuminance');
    const runningMeanAt = BLEND_FRAG.indexOf('float invOpacity');
    expect(runningMeanAt).toBeGreaterThanOrEqual(0);
    expect(BLEND_FRAG).not.toContain('color2.a =');
  });

  it('does not add ordinary NEE lighting after a debug program replaces radiance', () => {
    const debugGuard = glResourcesSource.indexOf('if (this.#debugMode === 0) {');
    const candidateDraw = glResourcesSource.indexOf(
      'const candidateProgram = this.#neeCandidateProgram;',
      debugGuard,
    );
    const resolveDraw = glResourcesSource.indexOf(
      'const resolveProgram = this.#neeResolveProgram;',
      candidateDraw,
    );
    const composite = glResourcesSource.indexOf('this.#compositeBlendStep(', resolveDraw);
    expect(debugGuard).toBeGreaterThanOrEqual(0);
    expect(candidateDraw).toBeGreaterThan(debugGuard);
    expect(resolveDraw).toBeGreaterThan(candidateDraw);
    expect(composite).toBeGreaterThan(resolveDraw);
    expect(glResourcesSource.slice(debugGuard, composite)).toContain(
      'gl.blendFunc(gl.ONE, gl.ONE);',
    );
    expect(glResourcesSource).toContain(
      'this.#compositeBlendStep();',
    );
  });

  it('reconstructs an already-accepted transparent candidate without a second opacity draw', () => {
    const compactSurfaceSource = surfaceRecordSource.replace(/\s+/g, ' ');
    expect(compactSurfaceSource).toContain(
      '! stochasticOpacityAlreadyAccepted && material.transparent && ! useAlphaTest && albedo.a < rand( 3 )',
    );
    expect(compactSurfaceSource).toContain(
      'accumulatedRoughness, pathDepth, heroWavelength, false, surf',
    );
    expect(NEE_RESOLVE_MAIN.replace(/\s+/g, ' ')).toContain(
      'candidate0.w, int( pathDepth ), candidate1.w, true, surf',
    );

    const survivesOpacityReplay = (
      albedoAlpha: number,
      replayRandom: number,
      alreadyAccepted: boolean,
    ) => alreadyAccepted || albedoAlpha >= replayRandom;
    for (const replayRandom of [0, 0.2, 0.5, 0.999_999]) {
      expect(survivesOpacityReplay(0.1, replayRandom, true)).toBe(true);
    }
    expect(survivesOpacityReplay(0.1, 0.9, false)).toBe(false);
  });

  it('forms light proposals at the geometric hit and applies one light-hemisphere offset', () => {
    const geometricPoint = RENDER_MAIN.indexOf(
      'vec3 geometricHitPoint =\n\t\t\t\t\t\t\t\tray.origin + ray.direction * surfaceHit.dist;',
    );
    const continuationPoint = RENDER_MAIN.indexOf('vec3 hitPoint = stepRayOrigin(', geometricPoint);
    const proposal = RENDER_MAIN.indexOf('sampleDirectLight(', continuationPoint);
    const prepare = RENDER_MAIN.indexOf('prepareDirectLightSample(', proposal);
    const proposalEnd = RENDER_MAIN.indexOf(');', prepare) + 2;
    const proposalBlock = RENDER_MAIN.slice(proposal, proposalEnd);
    expect(geometricPoint).toBeGreaterThanOrEqual(0);
    expect(continuationPoint).toBeGreaterThan(geometricPoint);
    expect(proposal).toBeGreaterThan(continuationPoint);
    expect(prepare).toBeGreaterThan(proposal);
    expect(proposalBlock.match(/geometricHitPoint/g)).toHaveLength(2);
    expect(proposalBlock).not.toContain('hitPoint, neeLightSample');
    expect(directLightSource.replace(/\s+/g, ' ')).toContain(
      'lightRay.origin = stepRayOrigin( rayOrigin, vec3( 0.0 ), lightIsBelowSurface ? - surf.faceNormal : surf.faceNormal, 0.0 );',
    );

    const offsetOrigin = (pointY: number, normalY: number, lightDirectionY: number) => {
      const side = normalY * lightDirectionY < 0 ? -1 : 1;
      return pointY + side * normalY * 1e-4;
    };
    expect(offsetOrigin(2, 1, 1)).toBeCloseTo(2.0001, 12);
    expect(offsetOrigin(2, 1, -1)).toBeCloseTo(1.9999, 12);
  });
});
