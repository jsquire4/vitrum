/**
 * Road-to-100 reconciliation guards:
 * - RC out-of-model weighting must keep ReSTIR-GI when RC is disabled/empty.
 * - Checkerboard sparse shading must have a live resolve/prefill path, not only
 *   preset plumbing.
 */
import { describe, expect, it } from 'vitest';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { RESOLVE_WGSL } from '../resolve.wgsl.js';
import { CB_PREFILL_MODULE } from '../cbPrefill.wgsl.js';

function fnBody(src: string, name: string): string {
  const start = src.indexOf(`fn ${name}(`);
  expect(start).toBeGreaterThanOrEqual(0);
  const next = src.indexOf('\nfn ', start + 1);
  return src.slice(start, next < 0 ? undefined : next);
}

describe('RC out-of-model confidence blend', () => {
  it('gates RC confidence on actual cascade energy before blending with ReSTIR-GI', () => {
    const body = fnBody(SHADING_TERMS_WGSL, 'lo_indirect');

    expect(body).toContain('let Lo_rc = sampleCascadeC0(pos, normal);');
    expect(body).toContain('let rcHasEnergy = max(Lo_rc.r, max(Lo_rc.g, Lo_rc.b)) > 1e-6;');
    expect(body).toContain(
      'let cRc = clamp(rcParams.rcWeight, 0.0, 1.0) * (1.0 - m) * select(0.0, 1.0, rcHasEnergy);',
    );
    expect(body).toContain('let wRestirGi = 1.0 - wRc;');
    expect(body).toContain('return wRestirGi * Lo_indirect + wRc * Lo_rc;');
    expect(body).not.toContain('let cRc = clamp(rcParams.rcWeight, 0.0, 1.0) * (1.0 - m);');
  });
});

describe('checkerboard sparse-shading resolve path', () => {
  it('resolve shader passthroughs when checkerboard is off and gap-fills when it is on', () => {
    expect(RESOLVE_WGSL).toContain('checkerboardOn: u32');
    expect(RESOLVE_WGSL).toContain('if (checkerboardOn == 0u) { return true; }');
    expect(RESOLVE_WGSL).toContain('return ((px + py) & 1u) == frameParity;');
    expect(RESOLVE_WGSL).toContain('let temporal = textureLoad(t_prev_radiance, prevXY, 0);');
    expect(RESOLVE_WGSL).toMatch(/let spatial\s*=\s*\(textureLoad\(t_current_radiance,\s*xL,\s*0\)/);
    expect(RESOLVE_WGSL).toContain('radiance = mix(spatial, temporal, wTemporal);');
  });

  it('prefill writes only gap-parity pixels before real denoisers read hdrColorTexture', () => {
    const src = CB_PREFILL_MODULE.source;

    expect(src).toContain('fn cbPrefillKernel(');
    expect(src).toContain('if (((px + py) & 1u) == u_cb.frameParity) { return; }');
    expect(src).toContain('let filled = textureLoad(t_prev_radiance, prevXY, 0);');
    expect(src).toContain('textureStore(t_hdr_out, vec2<u32>(px, py), filled);');
    expect(src).not.toContain('t_hdr_in');
  });
});
