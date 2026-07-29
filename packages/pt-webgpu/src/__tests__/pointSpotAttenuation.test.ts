import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';

function pointSpotDistanceAttenuation(
  distance: number,
  cutoffDistance: number,
  decay: number,
): number {
  const safeDistance = Math.max(distance, 1e-4);
  let attenuation = 1;
  if (decay > 0.01) {
    attenuation = 1 / Math.max(safeDistance ** decay, 1e-8);
  }
  if (cutoffDistance > 0) {
    const window = Math.min(Math.max(1 - (safeDistance / cutoffDistance) ** 4, 0), 1);
    attenuation *= window;
  }
  return attenuation;
}

function pointSpotPathMeasureScale(distance: number, cutoffDistance: number, decay: number): number {
  const safeDistance = Math.max(distance, 1e-4);
  return pointSpotDistanceAttenuation(safeDistance, cutoffDistance, decay) *
    safeDistance * safeDistance;
}

describe('point and spot distance attenuation', () => {
  it('implements the public decay=0 no-falloff contract', () => {
    for (const distance of [0.25, 1, 4, 100]) {
      expect(pointSpotDistanceAttenuation(distance, 0, 0)).toBe(1);
    }
  });

  it('implements physical inverse-square decay=2 without a unit-distance clamp', () => {
    expect(pointSpotDistanceAttenuation(0.5, 0, 2)).toBeCloseTo(4, 13);
    expect(pointSpotDistanceAttenuation(2, 0, 2)).toBeCloseTo(0.25, 13);
    expect(pointSpotDistanceAttenuation(10, 0, 2)).toBeCloseTo(0.01, 13);
  });
  it('replaces physical photon spreading on the source edge exactly once', () => {
    for (const sourceEdgeDistance of [0.25, 1, 4, 100]) {
      expect(pointSpotPathMeasureScale(sourceEdgeDistance, 0, 2)).toBeCloseTo(1, 13);
    }
    expect(pointSpotPathMeasureScale(2, 0, 0)).toBeCloseTo(4, 13);
    expect(pointSpotPathMeasureScale(2, 0, 1)).toBeCloseTo(2, 13);
    expect(pointSpotPathMeasureScale(10, 10, 0)).toBe(0);
  });



  it('uses the unsquared KHR quartic window and reaches zero at the range', () => {
    const distance = 5;
    const cutoff = 10;
    const inverseSquare = 1 / (distance * distance);
    const window = 1 - (distance / cutoff) ** 4;
    expect(pointSpotDistanceAttenuation(distance, cutoff, 2)).toBeCloseTo(
      inverseSquare * window,
      13,
    );
    expect(pointSpotDistanceAttenuation(cutoff, cutoff, 2)).toBe(0);
    expect(pointSpotDistanceAttenuation(cutoff * 2, cutoff, 2)).toBe(0);
  });

  it('keeps the canonical helper in the material module', () => {
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(
      'fn pointSpotDistanceAttenuation(',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(
      'fn pointSpotPathMeasureScale(',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(
      'attenuation = attenuation * window;',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).not.toContain(
      'attenuation = attenuation * window * window;',
    );
  });

  it('routes volume and BDPT point/spot connections through the same helper', () => {
    expect(
      (PT_WEBGPU_MEDIUM_NEE_WGSL.match(/pointSpotDistanceAttenuation\(/g) ?? []).length,
    ).toBe(2);
    expect(
      (PT_WEBGPU_BDPT_CONNECTION_WGSL.match(
        /pointSpotDistanceAttenuation\(/g,
      ) ?? []).length,
    ).toBe(2);
    expect(
      (PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.match(
        /pointSpotPathMeasureScale\(/g,
      ) ?? []).length,
    ).toBe(1);
  });

  it('routes full, lite, and ReSTIR-PT point/spot lighting through the helper', () => {
    const modules: ReadonlyArray<readonly [string, number]> = [
      [PT_WEBGPU_PATH_TRACE_KERNEL_WGSL, 2],
      [PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL, 2],
      [RESTIR_PT_PRODUCER_WGSL, 2],
    ];
    for (const [module, expectedOccurrences] of modules) {
      expect(
        (module.match(/pointSpotDistanceAttenuation\(/g) ?? []).length,
      ).toBe(expectedOccurrences);
      expect(module).not.toContain(
        'select(1.0 / dist2, pow(max(dist, 1.0)',
      );
    }
  });

  it('applies authored attenuation after spreading in the unified point/spot MNEE path', () => {
    expect(
      (PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.match(
        /pointSpotPathMeasureScale\(/g,
      ) ?? []).length,
    ).toBe(2);
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'pathDistance, emitter.maxDistance, emitter.decay,',
    );
  });

  it('honors mneeMaxIterations in every constrained Newton solve', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'fn configuredMneeIterations(maxSupported: u32) -> u32',
    );
    expect(
      (PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.match(
        /configuredMneeIterations\((?:16|32)u\)/g,
      ) ?? []).length,
    ).toBe(4);
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain('DIRECTIONAL_CAUSTIC_SAMPLES');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain('mneeSteps');
  });

  it('applies the same authored attenuation to point and spot SPPM photon deposits', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'var photonUsesPointSpotAttenuation = false;',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (bounce == 0u && photonUsesPointSpotAttenuation) {');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'hit.dist, photonCutoffDistance, photonDecay,',
    );
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('photonPathDistance');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('depositedFlux');
    expect(
      (SPPM_PHOTON_PASS_WGSL.match(/photonUsesPointSpotAttenuation = true;/g) ?? []).length,
    ).toBe(2);
  });
});
