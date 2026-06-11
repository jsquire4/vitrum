/**
 * B1 (road-to-100) — glossy/metal GI structural gate.
 * B1-ior-per-tri (2026-06-10) — per-tri IOR lane structural gate.
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
 *   5. B1-ior-per-tri: materialDecode exposes `decodeIor`; risGi glass walk
 *      uses per-tri IOR instead of a fixed 1.5 constant; shade `lo_transmittedGI`
 *      derives F0 from per-tri IOR ((ior-1)/(ior+1))² instead of the
 *      hard-coded GLASS_F0=0.04; risGi rough-glass GI perturbation applied
 *      when roughness > ROUGH_GLASS_THRESHOLD.
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

describe('B1-ior-per-tri — per-triangle IOR lane structural pins', () => {
  it('materialDecode provides decodeIor (bits[15:8] decode: 1.0 + byte/255*2.0)', () => {
    expect(MATERIAL_DECODE_WGSL).toContain('fn decodeIor(packed: u32) -> f32');
    expect(MATERIAL_DECODE_WGSL).toContain('(packed >> 8u) & 0xFFu');
    expect(MATERIAL_DECODE_WGSL).toContain('1.0 + f32(byte) / 255.0 * 2.0');
  });

  it('risGi glass walk no longer uses a hardcoded IOR_GLASS=1.5 constant', () => {
    // The fixed `const IOR_GLASS: f32 = 1.5;` must be gone (replaced by decodeIor()).
    expect(RIS_GI_WGSL).not.toContain('const IOR_GLASS: f32 = 1.5;');
    // Per-tri decode must be present.
    expect(RIS_GI_WGSL).toContain('decodeIor(glassPrimaryPacked)');
  });

  it('risGi glass walk binds bvh_material (group 1, binding 14)', () => {
    expect(RIS_GI_WGSL).toContain('@group(1) @binding(14) var bvh_material: texture_2d<u32>;');
  });

  it('shade lo_transmittedGI no longer uses hardcoded GLASS_F0 = 0.04', () => {
    // The hardcoded `const GLASS_F0: f32 = 0.04;` must be gone.
    expect(SHADE_WGSL).not.toContain('const GLASS_F0: f32 = 0.04;');
    expect(SHADE_WGSL).not.toContain('GLASS_F0 +');
  });

  it('shade lo_transmittedGI derives F0 from per-tri IOR via physical formula', () => {
    // Physical Schlick F0 = ((ior-1)/(ior+1))².
    expect(SHADE_WGSL).toContain('decodeIor(packedG)');
    expect(SHADE_WGSL).toContain('iorMinus1 / iorPlus1');
  });

  it('risGi rough-glass GI perturbation is gated on ROUGH_GLASS_THRESHOLD', () => {
    // Smooth glass (rough < threshold) keeps exact Snell direction — byte-identical.
    expect(RIS_GI_WGSL).toContain('const ROUGH_GLASS_THRESHOLD: f32 = 0.1;');
    expect(RIS_GI_WGSL).toContain('if (glassPrimaryRough > ROUGH_GLASS_THRESHOLD)');
  });
});
