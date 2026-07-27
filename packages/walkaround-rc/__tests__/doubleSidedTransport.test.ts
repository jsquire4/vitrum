import { describe, expect, it } from 'vitest';

import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';

describe('RC double-sided transport', () => {
  it('skips one-sided opaque backfaces while retaining transmissive exits', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcMaterialSideAdmittedForHit(');
    expect(PROBE_RAY_CAST_WGSL).toContain('(mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u');
    expect(PROBE_RAY_CAST_WGSL).toContain('(mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u');
    expect(PROBE_RAY_CAST_WGSL).toContain('mat.transmission > 0.0');
    expect(PROBE_RAY_CAST_WGSL).toContain('if (!rcMaterialSideAdmittedForHit(hit))');
  });
});
