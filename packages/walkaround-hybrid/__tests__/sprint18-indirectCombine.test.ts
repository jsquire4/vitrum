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
import { REFRACTIVE_CAUSTICS_WGSL } from '../src/shaders/refractiveCaustics.wgsl.js';
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
    expect(SHADE_WGSL).toMatch(/let directRadiance\s*=\s*applyHomogeneousVolumeSingleScatter\(\s*\(Lo_emit/);
    expect(SHADE_WGSL).toMatch(/let indirectRadiance\s*=\s*applyHomogeneousVolumeSingleScatter\(\s*Lo_indirect\s*\*\s*ao\s*\*\s*layerTransmission/);
  });

  it('writes split outputs at the end of shadeMain', () => {
    // `pix` is the decoded pixel coordinate: gid.xy on the full-res (OFF) path,
    // or the compacted-dispatch-decoded active-parity pixel on the checkerboard
    // (ON) path. The split-output stores target `pix` in both cases.
    expect(SHADE_WGSL).toMatch(/textureStore\(hdrColorOut,\s*pix,\s*vec4f\(clampedDirect/);
    expect(SHADE_WGSL).toMatch(/textureStore\(hdrIndirectOut,\s*pix,\s*vec4f\(clampedIndirect/);
  });

});

describe('T5 — explicit generic caustics plus stained-glass aperture', () => {
  it('shade calls the bounded refractive estimator and the flag-gated aperture', () => {
    expect(SHADE_WGSL).toMatch(/let Lo_refractiveCaustic\s*=\s*lo_refractive_caustic\(/);
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
      /directRadiance\s*=\s*applyHomogeneousVolumeSingleScatter\(\s*\(Lo_emit\s*\+\s*Lo_emitterGlow\s*\+\s*Lo_lightMap\s*\+\s*Lo_indirectSpec\s*\+\s*\(Lo_direct\s*\+\s*Lo_analyticNEE\s*\+\s*Lo_sunNEE\s*\+\s*Lo_refractiveCaustic\s*\+\s*Lo_skyAperture\)\s*\*\s*ao\)\s*\*\s*layerTransmission\s*\+\s*Lo_transmittedGI/,
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

  it('stainedGlassShade owns only aperture; generic caustics have their own module', () => {
    expect(STAINED_GLASS_SHADE_WGSL).not.toContain('fn lo_sg_caustic(');
    expect(STAINED_GLASS_SHADE_WGSL).toContain('fn lo_sg_aperture(');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('fn lo_refractive_caustic(');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('fn traceRefractiveCausticPath(');
  });

  it('aperture is flag-gated while caustic strategy uses the independent UBO gate', () => {
    expect(STAINED_GLASS_SHADE_WGSL).toMatch(
      /\(ubo\.stainedGlassFlags\s*&\s*SG_FLAG_SKY_APERTURE\)\s*==\s*0u\s*\)\s*\{\s*return vec3f\(0\.0\);/,
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('if (ubo.sunAngular.z < 0.5');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('(ubo.stainedGlassFlags & SG_FLAG_SUN_CAUSTIC) != 0u');
  });

  it('generic estimator carries bounded spectral interface traversal and signed NEE correction', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('for (var depth = 0u; depth <= 4u;');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('for (var candidate = 0u; candidate < 2u;');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('materialDispersionIorRgb(');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('materialThinFilmResponse(');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('let baselineT = traceSceneAlphaTintTransmittanceTextured(');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('var correction = (estimate - baseline)');
  });

  it('flag-ON math preserves the aperture model with finite staged sky scaling', () => {
    // 5-tap scalar-luminance sky probe, then staged sky radiance and Lambert.
    expect(STAINED_GLASS_SHADE_WGSL).toContain(
      'let skyVisScalar = skyAccum / max(weightAccum, 1e-6);',
    );
    expect(STAINED_GLASS_SHADE_WGSL).toContain(
      'let scalarSkyRadiance = walkaroundScaleEnvironmentRadiance(',
    );
    expect(STAINED_GLASS_SHADE_WGSL).toContain(
      'return skyVisAvg * scalarSkyRadiance * albedo * INV_PI;',
    );
    expect(STAINED_GLASS_SHADE_WGSL).not.toContain('skyTint * skyIrradiance');
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
