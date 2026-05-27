import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('pt-webgpu BDPT (WG-7)', () => {
  it('full trace shader includes connection evaluator and frame params', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEnabled');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('evaluateBdptConnection');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptLightPath');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptExtendLightSubpath');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptWriteBounce0');
  });
});
