/**
 * T5 — stained-glass opt-in flag packer + UBO byte-stability tests.
 *
 * `packStainedGlassFlags` maps the per-engine `stainedGlass` option booleans
 * into the u32 bitfield that lands at WalkaroundUBO offset 344 (the repurposed
 * `_tracePad0` slot). The WGSL side (`walkaroundUbo.wgsl.ts`) reads that field
 * with `SG_FLAG_SUN_CAUSTIC` (bit 0) and `SG_FLAG_SKY_APERTURE` (bit 1); these
 * tests pin (a) the packer's bit semantics, (b) that the TS + WGSL constants
 * agree, and (c) the UBO byte layout: flags at offset 344, and (since the W9
 * PPG guided-sampling landing) ppgEnabled at the former _tracePad1 slot (348)
 * with the struct grown to 368 bytes by the appended ppgMixAlpha + pad block.
 */

import { describe, expect, it } from 'vitest';
import {
  packStainedGlassFlags,
  SG_FLAG_SUN_CAUSTIC,
  SG_FLAG_SKY_APERTURE,
} from '../src/pipeline/uboUpdater.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';

describe('T5 — packStainedGlassFlags', () => {
  it('defaults to 0 (both terms OFF → generic scenes get zero stained-glass physics)', () => {
    expect(packStainedGlassFlags({})).toBe(0);
    expect(packStainedGlassFlags({ sunCaustic: false, skyAperture: false })).toBe(0);
  });

  it('sets bit 0 for sunCaustic only', () => {
    expect(packStainedGlassFlags({ sunCaustic: true })).toBe(SG_FLAG_SUN_CAUSTIC);
    expect(packStainedGlassFlags({ sunCaustic: true })).toBe(1);
  });

  it('sets bit 1 for skyAperture only', () => {
    expect(packStainedGlassFlags({ skyAperture: true })).toBe(SG_FLAG_SKY_APERTURE);
    expect(packStainedGlassFlags({ skyAperture: true })).toBe(2);
  });

  it('OR-combines both bits when both terms opt in (Cornell-SG host)', () => {
    expect(packStainedGlassFlags({ sunCaustic: true, skyAperture: true })).toBe(
      SG_FLAG_SUN_CAUSTIC | SG_FLAG_SKY_APERTURE,
    );
    expect(packStainedGlassFlags({ sunCaustic: true, skyAperture: true })).toBe(3);
  });

  it('TS bit masks are distinct single bits (no overlap)', () => {
    expect(SG_FLAG_SUN_CAUSTIC).toBe(1);
    expect(SG_FLAG_SKY_APERTURE).toBe(2);
    expect(SG_FLAG_SUN_CAUSTIC & SG_FLAG_SKY_APERTURE).toBe(0);
  });
});

describe('T5 — WGSL / TS flag-mask agreement', () => {
  it('walkaroundUbo.wgsl declares matching SG_FLAG_* constants', () => {
    // The WGSL early-return gate reads these; they MUST equal the TS packer's
    // bit values or flag-on/flag-off would disagree across the boundary.
    expect(WALKAROUND_UBO_WGSL).toContain(
      `const SG_FLAG_SUN_CAUSTIC: u32 = ${SG_FLAG_SUN_CAUSTIC}u;`,
    );
    expect(WALKAROUND_UBO_WGSL).toContain(
      `const SG_FLAG_SKY_APERTURE: u32 = ${SG_FLAG_SKY_APERTURE}u;`,
    );
  });
});

describe('T5 — UBO byte layout (offset-344 flags slot + W9 PPG tail)', () => {
  it('stainedGlassFlags occupies the offset-344 slot (was _tracePad0)', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('stainedGlassFlags:          u32,');
    // _tracePad0 was renamed to stainedGlassFlags (T5); _tracePad1 was
    // repurposed to ppgEnabled (W9 guided sampling) — neither pad name
    // survives.
    expect(WALKAROUND_UBO_WGSL).not.toContain('_tracePad0:');
    expect(WALKAROUND_UBO_WGSL).not.toContain('_tracePad1:');
  });

  it('ppgEnabled occupies offset 348 (the former _tracePad1 slot)', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('ppgEnabled:                 u32,');
    expect(WALKAROUND_UBO_WGSL).toContain('offset 348 — PPG guided-sampling gate');
  });

  it('struct now ends at 416 bytes (ReGIR grid block appended after the light-tree fields)', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('ppgMixAlpha:                f32,');
    // The PPG tail ends at 368; the ReGIR grid block (offsets 368..412) grows
    // the struct to 416 bytes (416 % 16 == 0).
    expect(WALKAROUND_UBO_WGSL).toContain('struct size 416 bytes');
  });
});
