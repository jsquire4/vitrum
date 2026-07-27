import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('pt-webgpu invocation-local BDPT', () => {
  it('builds one bounded light prefix inside each path-trace invocation', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEnabled');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('evaluateBdptConnection');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var<private> bdptLightPath: array<vec4f, 56>;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var<private> bdptEyeStackPrivate: array<BdptEyeVtx, 8>;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'bdptBuildInvocationLightSubpath(gid.xy);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bdptBuildInvocationLightSubpath(pixel: vec2u)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('@compute @workgroup_size(1, 1, 1)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('bdptExtendLightSubpath');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('bdptSetCurrentPixel');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('@group(2) @binding(5)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('@group(2) @binding(6)');
  });

  it('keeps full Veach strategy bookkeeping and rich light-vertex material state', () => {
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('bdptMISWeight2');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptMISWeightFull');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptLogDensitySAtoArea');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEyeStackStore');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEyeStackSetFwd');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let bsPrev = sampleNextBounceDirectionWithClearcoatNormal(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('pdfScatter = bsPrev.sampledEventPdf;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let pdfFwd = pdfScatter * segmentForwardDensity;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let predecessorCol = prevCol - 1;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'swappedDirectionalPdf * reverseEdgeDensity,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const BDPT_LIGHT_PATH_ROWS = 7u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.specularColor,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.specularIntensity,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let maxLv = min(params.bdptMaxLightBounces, 8u);');
  });
});
