import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('pt-webgpu WGSL material contract', () => {
  it('uses the bounded rich material payload layout', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const MATERIAL_VEC4_STRIDE = 22u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const THIN_FILM_LAYER_LIMIT = 8u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const SPECTRAL_SAMPLE_COUNT = 32u;');
  });

  it('threads transmission probability into directional MIS pdf helper', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let nDotT = max(abs(wiDotN), 1e-5);');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('transProb * pdfTransApprox');
  });

  it('accounts for uniform light selection probability in direct lighting', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('radiance = radiance + directLi * f32(lightCount);');
  });

  it('contains active strategy-specific caustic paths', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn causticMode() -> u32');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn manifoldNeeContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn photonMapContribution');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (causticMode() == 1u)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('else if (causticMode() == 2u)');
  });
});
