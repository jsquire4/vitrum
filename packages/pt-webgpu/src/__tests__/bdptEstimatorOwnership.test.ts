import { describe, expect, it } from 'vitest';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { BDPT_EXPLICIT_STRATEGY_MASK_WGSL } from '@vitrum/shared-samplers';

import {
  PT_WEBGPU_BDPT_CONNECTION_WGSL,
  composePtWebgpuBdptConnectionWgsl,
} from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';

const DELTA = 1;
const nativeCameraSplatConnection =
  composePtWebgpuBdptConnectionWgsl(true);

function bdptAcceptsSppmConnection(lightKinds: readonly number[], c: number): boolean {
  for (let j = 1; j < c; j += 1) {
    if (lightKinds[j] === DELTA) return false;
  }
  return true;
}

function bdptAcceptsMneeConnection(
  lightKinds: readonly number[],
  c: number,
  maxChainLength: number,
): boolean {
  const chainLength = c - 1;
  if (chainLength < 1 || chainLength > maxChainLength) return true;
  return !lightKinds.slice(1, c).every((kind) => kind === DELTA);
}

describe('BDPT global estimator ownership', () => {
  it('reports its bounded strategy family instead of advertising full camera-splat BDPT', () => {
    expect(
      BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.bidirectionalPathTracing,
    ).toEqual({
      mode: 'bounded-explicit-connections',
      maxLightVertices: 8,
      maxEyeVertices: 8,
      pureEyeStrategy: 'partitioned-eye-estimator',
      cameraSplatStrategy: 'native',
      misDenominator: 'sampled-strategies-only',
    });
  });

  it('assigns every finite direct-light branch to BDPT without changing the off path', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'let bdptOwnsFiniteLightFamily = params.bdptEnabled != 0u;',
    );
    const finiteNeeGuards = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.match(
      /if \(!bdptOwnsFiniteLightFamily && \(sumDirectLighting \|\| current == picked\)\)/g,
    );
    expect(finiteNeeGuards).toHaveLength(4);

    const directionalStart = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'for (var di = 0u; di < params.directionalLightCount;',
    );
    const pointStart = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'for (var pi = 0u; pi < params.pointLightCount;',
      directionalStart,
    );
    const directionalBranch = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.slice(
      directionalStart,
      pointStart,
    );
    expect(directionalBranch).toContain(
      'if (sumDirectLighting || current == picked) {',
    );
    expect(directionalBranch).not.toContain('!bdptOwnsFiniteLightFamily');
  });

  it('keeps the environment eye family and gates only the finite BSDF connection', () => {
    const finiteConnection = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'radiance = radiance + bsdfAreaLightConnectionContribution(',
    );
    const environmentConnection = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'radiance = radiance + bsdfEnvironmentConnectionContribution(',
      finiteConnection,
    );
    expect(finiteConnection).toBeGreaterThan(-1);
    expect(environmentConnection).toBeGreaterThan(finiteConnection);
    expect(
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.slice(
        finiteConnection - 160,
        finiteConnection,
      ),
    ).toContain('if (!bdptOwnsFiniteLightFamily) {');
    expect(
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.slice(
        finiteConnection,
        environmentConnection,
      ),
    ).toContain('\n      }\n');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'if (!prevSampleAllowsAreaMis && !sppmOwnsCurrentEmission) {',
    );
  });

  it('samples the primary eye vertex and emitter endpoint as explicit strategies', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'if (bdptOwnsFiniteLightFamily &&',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'for (var lvi = 0u; lvi < maxLv; lvi++) {',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).not.toContain(
      'if (bounce > 0u)',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).not.toContain(
      'for (var lvi = 1u;',
    );
  });

  it('assigns SPPM and bounded all-delta MNEE prefixes to one owner', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'caustic == 1u && mneeReceiverEligible',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('let sppmActive = caustic == 2u;');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).not.toContain(
      'if (!bdptOwnsFiniteLightFamily && caustic == 1u) {',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'bsdfHasFiniteConnectionSupport(',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'fn bdptLightPrefixContainsInteriorDelta(connectionIndex: u32) -> bool',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'for (var j = 1u; j < connectionIndex; j = j + 1u)',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let sppmOwnsLightPrefix =');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let mneeOwnsLightPrefix =');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'fn bdptLightPrefixIsMneeOwned(connectionIndex: u32) -> bool',
    );

    // c=0/1 has no interior prefix. A delta source at index 0 is intentionally
    // ignored, while any delta in [1,c) makes SPPM the sole owner. The terminal
    // at c is outside this prefix rule and remains governed by finite-support
    // endpoint validation.
    expect(bdptAcceptsSppmConnection([DELTA], 0)).toBe(true);
    expect(bdptAcceptsSppmConnection([DELTA, 0], 1)).toBe(true);
    expect(bdptAcceptsSppmConnection([DELTA, DELTA, 0], 2)).toBe(false);
    expect(bdptAcceptsSppmConnection([0, 0, DELTA], 2)).toBe(true);
    expect(bdptAcceptsSppmConnection([0, 0, DELTA, 0], 3)).toBe(false);
    expect(bdptAcceptsSppmConnection([0, 0, 0, 0], 3)).toBe(true);

    expect(bdptAcceptsMneeConnection([0, DELTA, 0], 2, 1)).toBe(false);
    expect(bdptAcceptsMneeConnection([0, DELTA, DELTA, 0], 3, 2)).toBe(false);
    expect(bdptAcceptsMneeConnection([0, DELTA, DELTA, 0], 3, 1)).toBe(true);
    expect(bdptAcceptsMneeConnection([0, 0, DELTA, 0], 3, 2)).toBe(true);
  });

  it('balances infinite p0/p1/p2+ while keeping direct and root PMFs distinct', () => {
    const countFn = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.slice(
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf('fn bdptEmitterCount()'),
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf('fn bdptRandomU32('),
    );
    expect(countFn).toContain('params.directionalLightCount');
    expect(countFn).toContain('bdptHasEnvironmentEndpoint()');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'lightSelection = sampleDistantDirectLight(sumDirectLighting, &rng);',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'radiance = radiance + bsdfEnvironmentConnectionContribution(',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'distantDirectSelectionPdf(environmentGlobalIndex) * envProposalPdf,',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let discretePdf = 1.0 / f32(emitterCount);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'fn bdptInfiniteEyeFamilyWeight(',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'fn bdptInfiniteRootLaunchPdf(directionPdf: f32) -> f32',
    );
    expect(nativeCameraSplatConnection).toContain(
      BDPT_EXPLICIT_STRATEGY_MASK_WGSL.trim(),
    );
    expect(nativeCameraSplatConnection).toContain(
      'var validExplicitStrategy = bdptExplicitConnectionStrategyIsValid(',
    );
    expect(nativeCameraSplatConnection).toContain(
      'validExplicitStrategy = n > 0u && n - 1u <= maxEyeVertices;',
    );
  });

  it('evaluates bounded strategy ratios and the power heuristic in log space', () => {
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'fn bdptLogDensitySAtoArea(',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'let scaledEdge = d / edgeScale;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'logP = logP + logRev.value - logFwd.value;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'logP = logP + logFwd.value - logRev.value;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'maxPowerLog = max(maxPowerLog, 2.0 * logPdfs[k]);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'denom = denom + exp(2.0 * logPdfs[k] - maxPowerLog);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'if (edgeScale == 0.0) {\n    return BdptLogDensity(0.0, false);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain(
      'p = p * (pRev / pFwd);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain(
      'p = p * (pFwd / pRev);',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain('pk * pk');
  });

  it('uses a fixed-depth q=1 eye walk while BDPT owns finite paths', () => {
    const surfaceRoulette = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.lastIndexOf(
      'let rr = russianRoulette(&rng, throughput);',
    );
    expect(surfaceRoulette).toBeGreaterThan(-1);
    expect(
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.slice(
        surfaceRoulette - 420,
        surfaceRoulette,
      ),
    ).toContain(
      'if (!bdptOwnsFiniteLightFamily && bounce > 2u) {',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'let maxLv = min(params.bdptMaxLightBounces, 8u);',
    );
  });

  it('preserves finite BDPT contributions above the former 100-unit clamp', () => {
    const contributionStart = PT_WEBGPU_BDPT_CONNECTION_WGSL.indexOf(
      'var contribution = lightThroughput',
    );
    const contributionTail = PT_WEBGPU_BDPT_CONNECTION_WGSL.slice(
      contributionStart,
    );
    expect(contributionStart).toBeGreaterThan(-1);
    expect(contributionTail).toContain(
      'let finiteProbe = contribution - contribution;',
    );
    expect(contributionTail).toContain(
      'any(contribution < vec3f(0.0));',
    );
    expect(contributionTail).toContain('return contribution;');
    expect(contributionTail).not.toContain('BDPT_CONTRIBUTION_CLAMP');
    expect(contributionTail).not.toMatch(
      /\b(?:clamp|min|max)\s*\(\s*contribution/,
    );
  });
});
