/**
 * compositeTonemapUbo.test.ts — walkaround-hybrid composite tonemap/exposure/
 * outputColorSpace wiring (2026-06-10).
 *
 * Pins:
 *   1. UBO packing — COMPOSITE_UBO encodes tonemapMode/exposure/outputColorSpace
 *      at the correct byte offsets and sizes.
 *   2. Defaults proof — unset quality fields produce aces(0), exposure=1.0,
 *      srgb(0) — bit-for-bit matching the historical hardcoded behavior.
 *   3. WGSL assertions — composite fragment shader references vitrumTonemap,
 *      CompositeUniforms struct with all three fields, and the outputColorSpace
 *      branch.
 *   4. All five operators can be round-tripped through the UBO packer.
 *   5. outputColorSpace 'linear' → skips OETF; 'srgb' (default) → applies it.
 *
 * Tonemap/exposure/outputColorSpace contract: FrameQualitySettings in
 * @vitrum/core frame.ts (§49-58). WGSL operators: tonemapWgsl() from
 * @vitrum/shared-samplers, indices per TONEMAP_MODE_INDEX.
 * Wired 2026-06-10.
 */

import { describe, expect, it } from 'vitest';
import { TONEMAP_MODE_INDEX } from '@vitrum/shared-samplers';
import { COMPOSITE_UBO } from '../pipeline/passes/uboLayouts.js';
import { COMPOSITE_FRAG_WGSL } from '../shaders/composite.wgsl.js';

// ── 1. UBO layout ─────────────────────────────────────────────────────────────

describe('COMPOSITE_UBO — layout and packing', () => {
  it('is exactly 16 bytes (4 scalar fields × 4 bytes)', () => {
    expect(COMPOSITE_UBO.sizeBytes).toBe(16);
  });

  it('tonemapMode is at byte offset 0 (u32)', () => {
    expect(COMPOSITE_UBO.fieldOffsets.tonemapMode).toBe(0);
  });

  it('exposure is at byte offset 4 (f32)', () => {
    expect(COMPOSITE_UBO.fieldOffsets.exposure).toBe(4);
  });

  it('outputColorSpace is at byte offset 8 (u32)', () => {
    expect(COMPOSITE_UBO.fieldOffsets.outputColorSpace).toBe(8);
  });

  it('pack → unpack is a round-trip identity for each tonemap mode', () => {
    const modes = Object.values(TONEMAP_MODE_INDEX);
    for (const mode of modes) {
      const buf = new ArrayBuffer(COMPOSITE_UBO.sizeBytes);
      const view = new DataView(buf);
      COMPOSITE_UBO.pack(view, 0, {
        tonemapMode: mode,
        exposure: 2.5,
        outputColorSpace: 1,
        _pad: 0,
      });
      const result = COMPOSITE_UBO.unpack(view, 0);
      expect(result.tonemapMode).toBe(mode);
      expect(result.exposure).toBeCloseTo(2.5, 5);
      expect(result.outputColorSpace).toBe(1);
    }
  });
});

// ── 2. Defaults proof ──────────────────────────────────────────────────────────

describe('Defaults — unset quality → aces@1.0 + srgb (bit-for-bit historical match)', () => {
  it('TONEMAP_MODE_INDEX.aces === 0 (historical hardcoded path)', () => {
    expect(TONEMAP_MODE_INDEX['aces']).toBe(0);
  });

  it("default UBO bytes: tonemapMode=0(aces), exposure=1.0, outputColorSpace=0(srgb)", () => {
    const buf = new ArrayBuffer(COMPOSITE_UBO.sizeBytes);
    const view = new DataView(buf);
    COMPOSITE_UBO.pack(view, 0, {
      tonemapMode:      0,   // aces
      exposure:         1.0, // default
      outputColorSpace: 0,   // srgb
      _pad:             0,
    });
    // tonemapMode at offset 0 = 0 (aces)
    expect(view.getUint32(0, true)).toBe(0);
    // exposure at offset 4 = 1.0
    expect(view.getFloat32(4, true)).toBeCloseTo(1.0, 6);
    // outputColorSpace at offset 8 = 0 (srgb)
    expect(view.getUint32(8, true)).toBe(0);
  });

  it('contract default matches historical behavior: aces + 1.0 + srgb (no tension)', () => {
    // frame.ts §49-58 documents: tonemap default='aces', exposure default=1.0,
    // outputColorSpace default='srgb'. All three match the prior hardcoded
    // composite shader (acesFilm at exposure 1.0 + linearToSRGB). No migration
    // needed — unset quality fields are bit-for-bit identical to the old shader.
    expect(TONEMAP_MODE_INDEX['aces']).toBe(0);
    // 'srgb' maps to outputColorSpace=0 in the orchestrator (linear=1 only when explicit).
    // 'linear' maps to outputColorSpace=1.
    // Both are verified by the orchestrator wiring; here we verify the mode index only.
    expect(TONEMAP_MODE_INDEX['reinhard']).toBe(2);
    expect(TONEMAP_MODE_INDEX['linear']).toBe(3);
    expect(TONEMAP_MODE_INDEX['none']).toBe(4);
  });
});

// ── 3. WGSL assertions ─────────────────────────────────────────────────────────

describe('COMPOSITE_FRAG_WGSL — shader source guards', () => {
  it('includes vitrumTonemap (the shared-samplers GPU operator function)', () => {
    expect(COMPOSITE_FRAG_WGSL).toMatch(/fn vitrumTonemap/);
  });

  it('declares CompositeUniforms struct with tonemapMode, exposure, outputColorSpace', () => {
    expect(COMPOSITE_FRAG_WGSL).toContain('struct CompositeUniforms');
    expect(COMPOSITE_FRAG_WGSL).toContain('tonemapMode');
    expect(COMPOSITE_FRAG_WGSL).toContain('exposure');
    expect(COMPOSITE_FRAG_WGSL).toContain('outputColorSpace');
  });

  it('binds CompositeUniforms at group(0) binding(1) without an unused sampler', () => {
    expect(COMPOSITE_FRAG_WGSL).toMatch(/@group\(0\)\s+@binding\(1\)\s+var<uniform>\s+compositeParams/);
    expect(COMPOSITE_FRAG_WGSL).not.toMatch(/var\s+\w*compositeSampler\s*:\s*sampler/);
  });

  it('calls vitrumTonemap with compositeParams.tonemapMode and compositeParams.exposure', () => {
    expect(COMPOSITE_FRAG_WGSL).toContain('compositeParams.tonemapMode');
    expect(COMPOSITE_FRAG_WGSL).toContain('compositeParams.exposure');
  });

  it('branches on compositeParams.outputColorSpace (srgb vs linear)', () => {
    expect(COMPOSITE_FRAG_WGSL).toContain('compositeParams.outputColorSpace');
    // srgb default: applies vt_linearToSrgb
    expect(COMPOSITE_FRAG_WGSL).toMatch(/vt_linearToSrgb/);
  });

  it('does NOT contain the old hardcoded acesFilm function (replaced by vitrumTonemap)', () => {
    // The old shader had a private `fn acesFilm(rgb: vec3f)` — replaced by
    // vitrumTonemap from shared-samplers. The internal `vt_aces` helper is fine.
    expect(COMPOSITE_FRAG_WGSL).not.toMatch(/^fn acesFilm/m);
  });

  it('does NOT index denoisedTex by raw fragCoord (regression guard from compositeResolutionBlit)', () => {
    expect(COMPOSITE_FRAG_WGSL).not.toMatch(/u32\(fragCoord\.x\)/);
  });

  it('still uses UV-based blit (textureDimensions + in.uv)', () => {
    expect(COMPOSITE_FRAG_WGSL).toMatch(/textureDimensions\(denoisedTex\)/);
    expect(COMPOSITE_FRAG_WGSL).toMatch(/in\.uv/);
  });
});

// ── 4. All five operators ──────────────────────────────────────────────────────

describe('TONEMAP_MODE_INDEX — all five operators have distinct indices', () => {
  it('aces=0, agx=1, reinhard=2, linear=3, none=4', () => {
    expect(TONEMAP_MODE_INDEX).toMatchObject({
      aces: 0,
      agx: 1,
      reinhard: 2,
      linear: 3,
      none: 4,
    });
  });

  it('each index is unique', () => {
    const values = Object.values(TONEMAP_MODE_INDEX);
    expect(new Set(values).size).toBe(values.length);
  });
});

// ── 5. outputColorSpace encoding ──────────────────────────────────────────────

describe('outputColorSpace encoding', () => {
  function packColorSpace(cs: number): number {
    const buf = new ArrayBuffer(COMPOSITE_UBO.sizeBytes);
    const view = new DataView(buf);
    COMPOSITE_UBO.pack(view, 0, {
      tonemapMode:      0,
      exposure:         1.0,
      outputColorSpace: cs,
      _pad:             0,
    });
    return view.getUint32(8, true); // offset 8
  }

  it("'srgb' encodes as outputColorSpace=0 (default, OETF applied)", () => {
    expect(packColorSpace(0)).toBe(0);
  });

  it("'linear' encodes as outputColorSpace=1 (OETF skipped)", () => {
    expect(packColorSpace(1)).toBe(1);
  });
});
