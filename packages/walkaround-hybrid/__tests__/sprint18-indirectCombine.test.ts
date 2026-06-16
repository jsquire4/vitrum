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
import { STAINED_GLASS_SHADE_WGSL } from '../src/shaders/stainedGlassShade.wgsl.js';
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
    // directRadiance = Lo_emit + (Lo_direct + Lo_sunCaustic + Lo_skyAperture) * ao
    expect(SHADE_WGSL).toMatch(/let directRadiance\s*=\s*Lo_emit/);
    expect(SHADE_WGSL).toMatch(/let indirectRadiance\s*=\s*Lo_indirect\s*\*\s*ao/);
  });

  it('writes split outputs at the end of shadeMain', () => {
    // `pix` is the decoded pixel coordinate: gid.xy on the full-res (OFF) path,
    // or the compacted-dispatch-decoded active-parity pixel on the checkerboard
    // (ON) path. The split-output stores target `pix` in both cases.
    expect(SHADE_WGSL).toMatch(/textureStore\(hdrColorOut,\s*pix,\s*vec4f\(clampedDirect/);
    expect(SHADE_WGSL).toMatch(/textureStore\(hdrIndirectOut,\s*pix,\s*vec4f\(clampedIndirect/);
  });

});

describe('T5 — stained-glass terms extracted to lo_sg_* opt-in module', () => {
  it('shade.wgsl calls lo_sg_caustic / lo_sg_aperture (new opt-in call surface)', () => {
    // T5 — the historical local names Lo_sunCaustic / Lo_skyAperture are
    // preserved (the directRadiance summation + structural tests depend on
    // them), but they are now produced by the extracted lo_sg_* helpers.
    expect(SHADE_WGSL).toMatch(/let Lo_sunCaustic\s*=\s*lo_sg_caustic\(/);
    expect(SHADE_WGSL).toMatch(/let Lo_skyAperture\s*=\s*lo_sg_aperture\(/);
    // Both Lo_sunCaustic and Lo_skyAperture still feed directRadiance × ao.
    // (Lo_emitterGlow — camera-visible emitters, 2026-05-30 — joins Lo_emit
    // outside the AO term; it sits between Lo_emit and the AO-scaled group.)
    // H41 — Lo_analyticNEE (point/spot additive NEE) inserted between Lo_direct
    // and Lo_sunCaustic inside the AO-scaled group.
    // item 4 (2026-06-10) — Lo_sunNEE (direct sun NEE, default-ON) inserted between
    // Lo_analyticNEE and Lo_sunCaustic in the AO-scaled group.
    // B1 — Lo_indirectSpec (glossy/metal specular indirect) joins the
    // UN-demodulated, non-AO direct group alongside Lo_emit / Lo_emitterGlow
    // (specular reflections are not albedo-demodulated and not GTAO-darkened).
    // B1 tail (2026-06-10) — Lo_transmittedGI (glass refracted-GI × Fresnel-T ×
    // Beer tint) joins the UN-demodulated, non-AO direct group alongside
    // Lo_indirectSpec; glass transmission is not GTAO-darkened.
    // Phase-3D lightMap slice — Lo_lightMap is baked outgoing radiance and also
    // joins the non-AO direct group.
    expect(SHADE_WGSL).toMatch(
      /directRadiance\s*=\s*Lo_emit\s*\+\s*Lo_emitterGlow\s*\+\s*Lo_lightMap\s*\+\s*Lo_indirectSpec\s*\+\s*Lo_transmittedGI\s*\+\s*\(Lo_direct\s*\+\s*Lo_analyticNEE\s*\+\s*Lo_sunNEE\s*\+\s*Lo_sunCaustic\s*\+\s*Lo_skyAperture\)\s*\*\s*ao/,
    );
  });

  it('shade.wgsl no longer defines the stained-glass term bodies inline', () => {
    // The fn definitions moved to stainedGlassShade.wgsl.ts; shade only
    // CALLS the new helpers. (The old names must not be redefined here.)
    expect(SHADE_WGSL).not.toContain('fn lo_sun_caustic');
    expect(SHADE_WGSL).not.toContain('fn lo_sky_aperture');
    expect(SHADE_WGSL).not.toContain('fn lo_sg_caustic');
    expect(SHADE_WGSL).not.toContain('fn lo_sg_aperture');
  });

  it('stainedGlassShade.wgsl defines both lo_sg_* helpers', () => {
    expect(STAINED_GLASS_SHADE_WGSL).toContain('fn lo_sg_caustic(');
    expect(STAINED_GLASS_SHADE_WGSL).toContain('fn lo_sg_aperture(');
  });

  it('each helper is gated by its UBO flag bit and early-returns vec3f(0) when OFF', () => {
    // Flag-OFF early-return path: bitwise-AND of the flag bit against
    // ubo.stainedGlassFlags == 0u → return vec3f(0.0). One per helper.
    expect(STAINED_GLASS_SHADE_WGSL).toMatch(
      /\(ubo\.stainedGlassFlags\s*&\s*SG_FLAG_SUN_CAUSTIC\)\s*==\s*0u\s*\)\s*\{\s*return vec3f\(0\.0\);/,
    );
    expect(STAINED_GLASS_SHADE_WGSL).toMatch(
      /\(ubo\.stainedGlassFlags\s*&\s*SG_FLAG_SKY_APERTURE\)\s*==\s*0u\s*\)\s*\{\s*return vec3f\(0\.0\);/,
    );
  });

  it('flag-ON math is byte-equivalent to the original inline caustic body', () => {
    // The load-bearing caustic math: tinted visibility clamped by
    // ubo.causticVisClamp, scaled by sun intensity × Lambert × causticBoost.
    // This is the exact return expression the inline lo_sun_caustic used; if
    // it drifts, the Cornell-SG flags-on reference render no longer matches.
    expect(STAINED_GLASS_SHADE_WGSL).toContain(
      'let visClamped = min(vis, vec3f(ubo.causticVisClamp));',
    );
    expect(STAINED_GLASS_SHADE_WGSL).toContain(
      'return visClamped * ubo.sunIntensity * nDotSun * albedo * INV_PI * ubo.causticBoost;',
    );
  });

  it('flag-ON math is byte-equivalent to the original inline aperture body', () => {
    // 5-tap scalar-luminance sky probe → skyTint × skyIrradiance × albedo × INV_PI.
    expect(STAINED_GLASS_SHADE_WGSL).toContain(
      'let skyVisScalar = skyAccum / max(weightAccum, 1e-6);',
    );
    expect(STAINED_GLASS_SHADE_WGSL).toContain(
      'return skyVisAvg * skyTint * skyIrradiance * albedo * INV_PI;',
    );
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
      // Item 3 (DDGI border fill): ddgi-border-irr and ddgi-border-vis sit
      // after indirect-combine, and transparent-oit now sits before
      // temporalAccum so the camera-visible alpha composite feeds history.
      expect(accum).toBe(combine + 4);
    }
  });

  it('MAX_PASS_COUNT accommodates indirect-combine + every other slot', () => {
    expect(MAX_PASS_COUNT).toBeGreaterThanOrEqual(20);
  });
});
