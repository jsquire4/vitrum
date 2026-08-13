import { describe, expect, it } from 'vitest';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';

// PT_WEBGPU_TRACE_WGSL is the bdpt:false composition. Since the BDPT estimator
// became a compose-time gate, the BDPT *call sites* live only in the bdpt:true
// composition — a bdpt:false pipeline can never reach them (params.bdptEnabled is
// constant for a pipeline's lifetime), so emitting them there was dead code that
// Mesa's NIR still fully inlined. Wiring assertions therefore target the ON
// composition, which is where this plumbing actually has to be correct.
const PT_WEBGPU_TRACE_WGSL_BDPT_ON = composePtWebgpuTraceWgsl(true);

describe('pt-webgpu invocation-local BDPT', () => {
  it('builds one bounded light prefix inside each path-trace invocation', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('bdptEnabled');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('evaluateBdptConnection');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var<private> bdptLightPath: array<vec4f, 64>;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var<private> bdptEyeStackPrivate: array<BdptEyeVtx, 8>;',
    );
    expect(PT_WEBGPU_TRACE_WGSL_BDPT_ON).toContain(
      'bdptBuildInvocationLightSubpath(gid.xy);',
    );
    // ...and the OFF composition must NOT carry that call site — that is the
    // whole point of the gate.
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'bdptBuildInvocationLightSubpath(gid.xy);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bdptBuildInvocationLightSubpath(pixel: vec2u)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('@compute @workgroup_size(1, 1, 1)');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('bdptExtendLightSubpath');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('bdptSetCurrentPixel');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'var<storage, read_write> bdptLightPath',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'var<storage, read_write> bdptEyeStack',
    );
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
    expect(PT_WEBGPU_TRACE_WGSL).toContain('const BDPT_LIGHT_PATH_ROWS = 8u;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.specularColor,');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mat.specularIntensity,');
    expect(PT_WEBGPU_TRACE_WGSL_BDPT_ON).toContain(
      'let maxLv = min(params.bdptMaxLightBounces, 8u);',
    );
  });
});
