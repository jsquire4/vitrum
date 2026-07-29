import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

type TraversalResult = 'escaped' | 'accepted-surface' | 'invalid-overflow';

/**
 * Independent executable oracle for the shader policy. `true` is an
 * alpha/pass-through surface, `false` is an accepted terminal surface, and
 * exhausting the stream observes the miss.
 */
function walkPassThroughSupport(
  surfaceHitLimit: number,
  hits: readonly boolean[],
): TraversalResult {
  let surfaceHitCount = 0;
  for (const passThrough of hits) {
    if (surfaceHitCount >= surfaceHitLimit) return 'invalid-overflow';
    surfaceHitCount += 1;
    if (!passThrough) return 'accepted-surface';
  }
  return 'escaped';
}

function directBlasSupport(triangleCount: number): number {
  return triangleCount;
}

describe('scene-derived straight-ray traversal bounds', () => {
  it('derives saturated triangle-instance and analytic-shape support', () => {
    for (const token of [
      'fn sceneTraversalSaturatingAdd(a: u32, b: u32) -> u32',
      'fn sceneTraversalSaturatingMul(a: u32, b: u32) -> u32',
      'let triangleSupport = min(params.triangleCount, arrayLength(&indices));',
      'var meshSupport = triangleSupport;',
      'arrayLength(&tlasInstanceIndices)',
      'arrayLength(&tlasBlasRoots)',
      'let analyticTotal = min(params.analyticCount, arrayLength(&analyticHeaders));',
      'shapeId >= 1u && shapeId <= 4u',
      'shapeId == 5u',
      'multiplicity = 6u;',
    ]) {
      expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(token);
    }
  });

  it.each([
    ['camera path alpha', PT_WEBGPU_PATH_TRACE_KERNEL_WGSL],
    ['SPPM photon alpha', SPPM_PHOTON_PASS_WGSL],
    ['medium visibility', PT_WEBGPU_MEDIUM_NEE_WGSL],
    ['BDPT alpha', PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL],
  ])('%s direct-BLAS support handles zero, N, and corrupt N+1 hits', (_name, source) => {
    expect(source).toContain('sceneSurfaceHitLimit()');
    expect(walkPassThroughSupport(directBlasSupport(0), [])).toBe('escaped');
    const support = directBlasSupport(4);
    expect(walkPassThroughSupport(support, [true, true, true, true])).toBe(
      'escaped',
    );
    expect(walkPassThroughSupport(
      support,
      [true, true, true, true, true],
    )).toBe('invalid-overflow');
  });

  it('orders the final miss before overflow rejection in every traversal', () => {
    expect(walkPassThroughSupport(4, [true, true, false])).toBe(
      'accepted-surface',
    );

    const mediumMiss = PT_WEBGPU_MEDIUM_NEE_WGSL.indexOf(
      'if (!hit.didHit) {',
    );
    const mediumOverflow = PT_WEBGPU_MEDIUM_NEE_WGSL.indexOf(
      'if (surfaceHitCount >= surfaceHitLimit) {',
    );
    expect(mediumMiss).toBeGreaterThanOrEqual(0);
    expect(mediumOverflow).toBeGreaterThan(mediumMiss);

    const bdptMiss = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf(
      'if (!hit.didHit) { break; }',
    );
    const bdptOverflow = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf(
      'if (alphaSurfaceHitCount >= alphaSurfaceHitLimit) {',
    );
    expect(bdptMiss).toBeGreaterThanOrEqual(0);
    expect(bdptOverflow).toBeGreaterThan(bdptMiss);

    for (const source of [
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
      SPPM_PHOTON_PASS_WGSL,
    ]) {
      const miss = source.indexOf('if (!hit.didHit) { break; }');
      const overflow = source.indexOf(
        'if (alphaSurfaceHitCount >= alphaSurfaceHitLimit) {',
      );
      expect(miss).toBeGreaterThanOrEqual(0);
      expect(overflow).toBeGreaterThan(miss);
      expect(source).toContain('alphaTraversalValid = false;');
    }
  });

  it('removes every fixed alpha/medium cap and fails closed on excess support', () => {
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).not.toContain('step < 16u');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('alphaSkip < 8u');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).not.toContain('aSkip < 8u');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('aSkip < 8u');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'alphaTraversalValid = false;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bdptWriteInvalid(col);',
    );
  });
});
