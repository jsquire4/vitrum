/**
 * Sprint 18 — Per-channel denoising (direct vs indirect) structural tests.
 *
 * Validates:
 *  - shade.wgsl now writes split direct + indirect outputs;
 *  - indirectCombine.wgsl performs the broad-sigma bilateral blur on the
 *    indirect channel and sums with direct;
 *  - pass layout includes `indirect-combine` between the denoiser chain
 *    and `temporalAccum` in every variant.
 */

import { describe, expect, it } from 'vitest';
import { SHADE_WGSL } from '../src/shaders/shade.wgsl.js';
import { INDIRECT_COMBINE_WGSL } from '../src/shaders/indirectCombine.wgsl.js';
import {
  MAX_PASS_COUNT,
  buildPassLayout,
} from '../src/pipeline/timestampQueries.js';

describe('Sprint 18 — shade.wgsl split output', () => {
  it('declares the hdrIndirectOut storage texture at frame BGL binding 12', () => {
    expect(SHADE_WGSL).toContain('@group(0) @binding(12) var hdrIndirectOut');
    expect(SHADE_WGSL).toMatch(/texture_storage_2d<rgba16float,\s*write>/);
  });

  it('splits the final radiance into directRadiance + indirectRadiance', () => {
    expect(SHADE_WGSL).toContain('let directRadiance');
    expect(SHADE_WGSL).toContain('let indirectRadiance');
    // Lo_emit + (Lo_direct + Lo_sunCaustic + Lo_skyAperture * 0.08) * ao
    expect(SHADE_WGSL).toMatch(/let directRadiance\s*=\s*Lo_emit/);
    expect(SHADE_WGSL).toMatch(/let indirectRadiance\s*=\s*Lo_indirect\s*\*\s*ao/);
  });

  it('writes split outputs at the end of shadeMain', () => {
    expect(SHADE_WGSL).toMatch(/textureStore\(hdrColorOut,\s*gid\.xy,\s*vec4f\(clampedDirect/);
    expect(SHADE_WGSL).toMatch(/textureStore\(hdrIndirectOut,\s*gid\.xy,\s*vec4f\(clampedIndirect/);
  });

});

describe('Sprint 18 — indirect-combine WGSL', () => {
  it('declares indirectCombineMain entry point with 16x16 workgroup', () => {
    expect(INDIRECT_COMBINE_WGSL).toContain('fn indirectCombineMain');
    expect(INDIRECT_COMBINE_WGSL).toContain('@workgroup_size(16, 16, 1)');
  });

  it('binds denoisedDirect, denoisedIndirect, combinedOut, and albedo', () => {
    // W5-I2 (2026-05-18): previous slot-2 `ic_gNormalDepth` was declared
    // "for BGL compat" but never read; dropped along with the host-side
    // BGL entry. Bindings 3/4 renumbered to 2/3.
    expect(INDIRECT_COMBINE_WGSL).toContain('@group(0) @binding(0) var ic_denoisedDirect');
    expect(INDIRECT_COMBINE_WGSL).toContain('@group(0) @binding(1) var ic_denoisedIndirect');
    expect(INDIRECT_COMBINE_WGSL).toContain('@group(0) @binding(2) var ic_combinedOut');
    expect(INDIRECT_COMBINE_WGSL).toContain('@group(0) @binding(3) var ic_albedo');
    expect(INDIRECT_COMBINE_WGSL).not.toContain('ic_gNormalDepth');
  });

  it('sums denoised direct + denoised indirect into the combined texture', () => {
    expect(INDIRECT_COMBINE_WGSL).toMatch(/textureStore\(ic_combinedOut,\s*gid\.xy,\s*vec4f\(direct\s*\+\s*indirect/);
  });
});

describe('Sprint 18 — pass-layout placement', () => {
  it('places indirect-combine before temporalAccum in every variant (with DDGI border slots in between)', () => {
    for (const denoiserMode of ['atrous-variance', 'atrous'] as const) {
      const layout = buildPassLayout({ denoiserMode });
      const combine = layout.labels.indexOf('indirect-combine');
      const accum   = layout.labels.indexOf('temporalAccum');
      expect(combine).toBeGreaterThanOrEqual(0);
      // Item 3 (DDGI border fill): ddgi-border-irr and ddgi-border-vis now sit
      // between indirect-combine and temporalAccum so the border is filled
      // before temporal accumulation reads the atlas.
      expect(accum).toBe(combine + 3);
    }
  });

  it('MAX_PASS_COUNT accommodates indirect-combine + every other slot', () => {
    expect(MAX_PASS_COUNT).toBeGreaterThanOrEqual(20);
  });
});
