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
    // BDPT light-subpath estimator coherence (2026-06-10, extended for PTWG-MAT):
    // the scatter direction sampled at prevPos is the SAME direction used to
    // extend the path, compute f·|cos|/pdf, and store pdfFwd. The old two-step
    // (cosine-hemisphere trace + discard + real-BSDF sample at newPos) is gone,
    // and scalar clearcoat/sheen now route through the shared sampled-density
    // helper rather than the base-only BRDF pdf.
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'pdfScatter = brdfDirectionalPdfFullSampled(prevBc, prevRough, prevMetal, 0.0, prevMat.ior,',
    );
    // pdfFwd at the new vertex = the scatter pdf at prevPos for the traced direction.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let pdfFwd = pdfScatter;');
    // pdfRev(prevCol) is patched to the TRUE reverse density (Item-3 fix 2026-06-10):
    // for surface vertices brdfDirectionalPdf(prevNormal, scatterDir, woAtPrev);
    // for emitter vertices (Lambertian, symmetric) pdfFwd is the correct pdfRev.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptLightPath[bdptLightPathIndex(prevCol, 2u)] = vec4f(old_r2prev.xyz, pdfRevAtPrev);');
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/pdfFwd = pdfScatter \* max\(gTerm/);
    // The §10.3 connection evaluates the REAL light-vertex BSDF (4-row light path).
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const BDPT_LIGHT_PATH_ROWS = 4u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (lvMatId >= 0.0) {');
    // A9 — light-bounce cap raised 3 → 8.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let maxLv = min(params.bdptMaxLightBounces, 8u);');
  });
});
