import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('pt-webgpu BDPT (WG-7)', () => {
  it('full trace shader includes connection evaluator and frame params', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEnabled');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('evaluateBdptConnection');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptLightPath');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptExtendLightSubpath');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptWriteBounce0');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('params.spotLightCount');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEmitterPower');
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/spotLights\[sb \+ 2u\]\.rgb/);
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptHasEnvironmentEmitter');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('sampleEnvironmentImportance');
  });

  it('uses the full Veach §10.3 multi-strategy sweep, not the 2-strategy approximation', () => {
    // The old 2-strategy bdptMISWeight2 must be gone; the full PBRT MISWeight
    // recurrence (ConvertDensity area ratio + β=2 power heuristic) is in.
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('bdptMISWeight2');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptMISWeightFull');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptConvertDensitySAtoArea');
    // Eye-subpath scratch stack (D2) + eye-vertex reverse density via swapped
    // brdfDirectionalPdf (D1).
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEyeStack');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEyeStackStore');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEyeStackSetFwd');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptMaxEyeDepth');
    // The light subpath stores bare solid-angle pdfs (no baked-in geometry term).
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let pdfFwd = pdfScatter;');
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/pdfFwd = pdfScatter \* max\(gTerm/);
  });
});
