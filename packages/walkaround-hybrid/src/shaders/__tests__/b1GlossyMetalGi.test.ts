/**
 * B1 (road-to-100) — glossy/metal GI structural gate.
 *
 * Pins the shader-level contract of the B1 change so a future refactor cannot
 * silently re-introduce the metal/glass early-outs or the hardcoded
 * roughness/metalness, and so the p̂/consumption-consistency boundary stays
 * documented:
 *   1. ris / restirCastPrimary / shade decode REAL roughness/metal from the
 *      bvh_material texture (no `select(0.85, 0.05, isGlass)` / `metal = 0.0`).
 *   2. shade lo_direct / lo_analyticNEE no longer early-out on isMetal
 *      (metals get DIRECT light); glass still skips.
 *   3. risGi / risGiNrc no longer punt metals to an empty GI reservoir
 *      (glass still punts — refracted GI is out of scope).
 *   4. shade adds the glossy/metal SPECULAR-indirect term (evalGGXSpecularOnly
 *      against the ReSTIR-GI reservoir sample), routed to the un-demodulated
 *      direct channel.
 */
import { describe, it, expect } from 'vitest';
import { RIS_WGSL } from '../ris.wgsl.js';
import { SHADE_WGSL } from '../shade.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { RESTIR_CAST_PRIMARY_WGSL } from '../restirCastPrimary.wgsl.js';
import { MATERIAL_DECODE_WGSL } from '../materialDecode.wgsl.js';
import { GGX_BRDF_WGSL } from '../ggxBrdf.wgsl.js';

describe('B1 — real per-tri roughness/metalness decode', () => {
  it('materialDecode provides decodeRoughMetal + the bvh_material texel width', () => {
    expect(MATERIAL_DECODE_WGSL).toContain('fn decodeRoughMetal(packed: u32) -> vec2f');
    expect(MATERIAL_DECODE_WGSL).toContain('BVH_MATERIAL_TEX_WIDTH');
  });

  it('ris / cast / shade declare the bvh_material binding (group 1, binding 14)', () => {
    for (const src of [RIS_WGSL, RESTIR_CAST_PRIMARY_WGSL, SHADE_WGSL]) {
      expect(src).toContain('@group(1) @binding(14) var bvh_material: texture_2d<u32>;');
    }
  });

  it('ris / cast / shade no longer hardcode roughness/metalness', () => {
    for (const src of [RIS_WGSL, RESTIR_CAST_PRIMARY_WGSL, SHADE_WGSL]) {
      expect(src).toContain('decodeRoughMetal(textureLoad(bvh_material');
      // The old hardcoded forms must be gone.
      expect(src).not.toContain('select(0.85, 0.05, isGlass)');
    }
  });
});

/** Slice a single `fn name(...) { ... }` body out of a WGSL source so a
 *  negative-match stays scoped to that function (lo_indirect — the diffuse
 *  channel — legitimately KEEPS `isGlass || isMetal`, so a file-wide negative
 *  match would false-positive). Returns the text from `fn name` to the next
 *  top-level `fn ` declaration. */
function fnBody(src: string, name: string): string {
  const start = src.indexOf(`fn ${name}(`);
  expect(start).toBeGreaterThanOrEqual(0);
  const next = src.indexOf('\nfn ', start + 1);
  return src.slice(start, next < 0 ? undefined : next);
}

describe('B1 — metals receive direct light (no isMetal early-out)', () => {
  it('lo_direct early-out is glass-only', () => {
    const body = fnBody(SHADE_WGSL, 'lo_direct');
    expect(body).toContain('if (isGlass) { return vec3f(0.0); }');
    expect(body).not.toContain('isGlass || isMetal');
  });

  it('lo_analyticNEE early-out is glass-only', () => {
    const body = fnBody(SHADE_WGSL, 'lo_analyticNEE');
    expect(body).toContain('if (isGlass) { return vec3f(0.0); }');
    expect(body).not.toContain('isGlass || isMetal');
  });
});

describe('B1 — glossy/metal GI reservoir (no empty-reservoir punt for metal)', () => {
  it('risGi punts ONLY glass to an empty reservoir', () => {
    expect(RIS_GI_WGSL).not.toContain('if (isGlass || isMetal)');
    expect(RIS_GI_WGSL).toMatch(/if \(isGlass\) \{[\s\S]*?storeReservoirGI_rw\(&reservoirGiCurrent, pixelIdxGi, emptyReservoirGI\(\)\);/);
  });

  it('risGiNrc punts ONLY glass to an empty reservoir', () => {
    expect(RIS_GI_NRC_BODY).not.toContain('if (isGlass || isMetal)');
  });

  it('the GI target p̂ stays the Lambertian luminance form (unchanged for GRIS correctness)', () => {
    // p̂ = luminance(Lo) · cosθ · INV_PI — risGi candidate + finalise must keep it.
    expect(RIS_GI_WGSL).toContain('luminance(Lo) * cosTheta * INV_PI');
  });
});

describe('B1 — glossy/metal specular indirect term', () => {
  it('ggxBrdf exposes evalGGXSpecularOnly (specular lobe, conductor F0)', () => {
    expect(GGX_BRDF_WGSL).toContain('fn evalGGXSpecularOnly(');
    expect(GGX_BRDF_WGSL).toContain('mix(vec3f(0.04), albedo, metal)');
  });

  it('shade computes lo_indirectSpecular and folds it into the un-demodulated direct channel', () => {
    expect(SHADE_WGSL).toContain('fn lo_indirectSpecular(');
    expect(SHADE_WGSL).toContain('evalGGXSpecularOnly(albedo, rough, metal, normal, wo, wi)');
    expect(SHADE_WGSL).toContain('let Lo_indirectSpec = lo_indirectSpecular(');
    // It joins directRadiance (NOT the demodulated indirect channel).
    expect(SHADE_WGSL).toMatch(/directRadiance\s*=[\s\S]*?Lo_indirectSpec/);
  });

  it('specular indirect is gated off for default-diffuse surfaces (invariant)', () => {
    // metal <= 0 && rough >= SPEC_GI_ROUGH_MAX → zero (default rough 0.85).
    expect(SHADE_WGSL).toContain('if (metal <= 0.0 && rough >= SPEC_GI_ROUGH_MAX)');
  });
});
