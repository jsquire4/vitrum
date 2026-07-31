/**
 * W2-C5 — canonical MaterialEntry packing byte-layout tests.
 *
 * Locks the 16-float / 64-byte layout produced by `packMaterials()` and the
 * default values applied to missing fields. Engine-specific adapters
 * (DDGI's `threeToMaterialEntryInput`, RC's same-named local helper)
 * have their own tests; this suite covers the canonical packer in
 * isolation.
 */

import { describe, expect, it } from 'vitest';

import type { MaterialSpec } from '@vitrum/core';

import {
  MATERIAL_ATTEN_DIST_INFINITE,
  MATERIAL_DEFAULT_ROUGHNESS,
  MATERIAL_ENTRY_FLOATS,
  MATERIAL_ENTRY_STRIDE_BYTES,
  MATERIAL_FLAG_CAST_SHADOW_DISABLED,
  MATERIAL_FLAG_DOUBLE_SIDED,
  MATERIAL_FLAG_IS_GLASS,
  coreMaterialToMaterialEntry,
  packMaterials,
  toProductionEmissiveRadiance,
  type MaterialEntryInput,
} from '../materialEntry.js';

const ENTRY = MATERIAL_ENTRY_FLOATS;

/** Minimal valid MaterialSpec (only the three required fields). */
function spec(overrides: Partial<MaterialSpec> = {}): MaterialSpec {
  return { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, ...overrides };
}

function u32(out: Float32Array): Uint32Array {
  return new Uint32Array(out.buffer);
}

describe('canonical MaterialEntry packing (W2-C5)', () => {
  it('exposes 16 floats / 64 bytes per entry', () => {
    expect(MATERIAL_ENTRY_FLOATS).toBe(16);
    expect(MATERIAL_ENTRY_STRIDE_BYTES).toBe(64);
  });

  it('empty input returns a single zero-padded entry', () => {
    const out = packMaterials([]);
    expect(out.length).toBe(ENTRY);
    expect(out.every((v) => v === 0)).toBe(true);
  });

  it('maxCount pads the output up to the requested count', () => {
    const out = packMaterials([{ baseColor: [1, 0, 0] }], 4);
    expect(out.length).toBe(4 * ENTRY);
    // Slot 0 populated.
    expect(out[0]).toBe(1);
    // Slot 1..3 zero-padded.
    for (let i = ENTRY; i < 4 * ENTRY; i++) {
      expect(out[i]).toBe(0);
    }
  });

  it('rejects an input list larger than maxCount', () => {
    const inputs: MaterialEntryInput[] = [
      { baseColor: [1, 0, 0] },
      { baseColor: [0, 1, 0] },
      { baseColor: [0, 0, 1] },
    ];
    expect(() => packMaterials(inputs, 2)).toThrow(/exceed.*capacity/i);
  });

  it('applies library defaults to a fully-empty entry', () => {
    const out = packMaterials([{}]);
    // baseColor → (1,1,1)
    expect([out[0], out[1], out[2]]).toEqual([1, 1, 1]);
    // roughness → 1.0 (canonical default)
    expect(out[3]).toBe(MATERIAL_DEFAULT_ROUGHNESS);
    expect(out[3]).toBe(1.0);
    // emissive → (0,0,0)
    expect([out[4], out[5], out[6]]).toEqual([0, 0, 0]);
    // metalness → 0
    expect(out[7]).toBe(0);
    // ior → 1.5
    expect(out[8]).toBeCloseTo(1.5);
    // transmission → 0
    expect(out[9]).toBe(0);
    // attenuationDistance → MATERIAL_ATTEN_DIST_INFINITE
    expect(out[10]).toBe(MATERIAL_ATTEN_DIST_INFINITE);
    // thickness → 0
    expect(out[11]).toBe(0);
    // attenuationColor → (1,1,1)
    expect([out[12], out[13], out[14]]).toEqual([1, 1, 1]);
    // flags → 0
    expect(u32(out)[15]).toBe(0);
  });

  it('packs an opaque diffuse entry at the canonical slot offsets', () => {
    const out = packMaterials([
      {
        baseColor: [0.5, 0.6, 0.7],
        roughness: 0.4,
        metalness: 0.1,
        emissive: [0.01, 0.02, 0.03],
      },
    ]);
    expect(out[0]).toBeCloseTo(0.5);     // baseColor.r at slot 0
    expect(out[1]).toBeCloseTo(0.6);
    expect(out[2]).toBeCloseTo(0.7);
    expect(out[3]).toBeCloseTo(0.4);     // roughness at slot 3
    expect(out[4]).toBeCloseTo(0.01);    // emissive.r at slot 4
    expect(out[5]).toBeCloseTo(0.02);
    expect(out[6]).toBeCloseTo(0.03);
    expect(out[7]).toBeCloseTo(0.1);     // metalness at slot 7
    expect(out[8]).toBeCloseTo(1.5);     // ior default at slot 8
    expect(out[9]).toBe(0);              // transmission at slot 9
  });

  it('derives flags bit 0 from transmission when not explicitly set', () => {
    const out = packMaterials([
      { transmission: 0 },
      { transmission: 0.0001 },
      { transmission: 1 },
    ]);
    const U = u32(out);
    expect(U[15]).toBe(0);
    expect(U[1 * ENTRY + 15]).toBe(MATERIAL_FLAG_IS_GLASS);
    expect(U[2 * ENTRY + 15]).toBe(MATERIAL_FLAG_IS_GLASS);
  });

  it('respects an explicit flags override', () => {
    const out = packMaterials([
      { transmission: 0, flags: 0xFEEDFACE },
    ]);
    expect(u32(out)[15]).toBe(0xFEEDFACE);
  });

  it('coreMaterialToMaterialEntry preserves primitive castShadow:false in flag bit 1', () => {
    const out = packMaterials([
      coreMaterialToMaterialEntry(spec({ castShadow: false } as Partial<MaterialSpec>)),
      coreMaterialToMaterialEntry(spec({ transmission: 0.7, castShadow: false } as Partial<MaterialSpec>)),
    ]);
    const U = u32(out);
    expect(U[15]).toBe(MATERIAL_FLAG_CAST_SHADOW_DISABLED);
    expect(U[ENTRY + 15]).toBe(MATERIAL_FLAG_IS_GLASS | MATERIAL_FLAG_CAST_SHADOW_DISABLED);
  });

  it('coreMaterialToMaterialEntry packs doubleSided independently in flag bit 2', () => {
    const out = packMaterials([
      coreMaterialToMaterialEntry(spec({ doubleSided: true })),
      coreMaterialToMaterialEntry(spec({ transmission: 0.7, doubleSided: true })),
    ]);
    const U = u32(out);
    expect(U[15]).toBe(MATERIAL_FLAG_DOUBLE_SIDED);
    expect(U[ENTRY + 15]).toBe(MATERIAL_FLAG_IS_GLASS | MATERIAL_FLAG_DOUBLE_SIDED);
  });

  it('flags is written as a real u32 not the IEEE-754 bit pattern of 1.0', () => {
    const out = packMaterials([{ transmission: 1 }]);
    const U = u32(out);
    expect(U[15]).toBe(1);
    expect(U[15]).not.toBe(0x3F800000);
  });

  it('preserves +Infinity and rejects invalid attenuationDistance', () => {
    const out = packMaterials([
      { attenuationDistance: Infinity },
      { attenuationDistance: 2.5 },
    ]);
    expect(out[10]).toBe(MATERIAL_ATTEN_DIST_INFINITE);
    expect(out[ENTRY + 10]).toBeCloseTo(2.5);
    for (const attenuationDistance of [-3, 0, Number.NaN]) {
      expect(() => packMaterials([{ attenuationDistance }])).toThrow(/attenuationDistance/);
    }
  });
});

describe('coreMaterialToMaterialEntry — THREE-free MaterialSpec adapter', () => {
  it('folds emissiveIntensity into production radiance exactly once', () => {
    const folded = toProductionEmissiveRadiance(
      spec({ emissive: [0.5, 0.25, 0.1], emissiveIntensity: 4 }),
    );
    const expected = [2, 1, Math.fround(Math.fround(0.1) * 4)];
    expect(folded.emissive).toEqual(expected);
    expect(folded.emissiveIntensity).toBe(1);
    expect(coreMaterialToMaterialEntry(folded).emissive).toEqual(expected);
  });

  it('passes through baseColor / roughness / metallic (→ metalness)', () => {
    const e = coreMaterialToMaterialEntry(
      spec({ baseColor: [0.2, 0.4, 0.6], roughness: 0.3, metallic: 0.8 }),
    );
    expect(e.baseColor).toEqual([0.2, 0.4, 0.6]);
    expect(e.roughness).toBe(0.3);
    expect(e.metalness).toBe(0.8); // core `metallic` → entry `metalness`
  });

  it('pre-multiplies emissive by emissiveIntensity (matching the RC adapter)', () => {
    const e = coreMaterialToMaterialEntry(
      spec({ emissive: [0.5, 0.25, 0.1], emissiveIntensity: 4 }),
    );
    expect(e.emissive).toEqual([2.0, 1.0, Math.fround(Math.fround(0.1) * 4)]);
  });

  it('defaults emissiveIntensity to ×1 when absent (matches PBR_DEFAULTS)', () => {
    const e = coreMaterialToMaterialEntry(spec({ emissive: [0.3, 0.6, 0.9] }));
    expect(e.emissive).toEqual([
      Math.fround(0.3),
      Math.fround(0.6),
      Math.fround(0.9),
    ]);
  });

  it('omits emissive entirely when the spec has none (→ packMaterials default (0,0,0))', () => {
    const e = coreMaterialToMaterialEntry(spec());
    expect(e.emissive).toBeUndefined();
    const out = packMaterials([e]);
    expect([out[4], out[5], out[6]]).toEqual([0, 0, 0]);
  });

  it('passes through the transmission / refraction fields 1:1', () => {
    const e = coreMaterialToMaterialEntry(
      spec({
        transmission: 0.9,
        ior: 1.7,
        attenuationColor: [0.8, 0.2, 0.2],
        attenuationDistance: 3.5,
        thickness: 0.25,
      }),
    );
    expect(e.transmission).toBe(0.9);
    expect(e.ior).toBe(1.7);
    expect(e.attenuationColor).toEqual([0.8, 0.2, 0.2]);
    expect(e.attenuationDistance).toBe(3.5);
    expect(e.thickness).toBe(0.25);
  });

  it('does NOT apply RC\'s thickness→0.1 floor (faithful pass-through; default stays 0)', () => {
    // RC applies its own 0.1 floor on top of this adapter; the adapter itself
    // must not bake in RC policy. Absent thickness ⇒ undefined ⇒ packs to 0.
    const e = coreMaterialToMaterialEntry(spec({ transmission: 0.5 }));
    expect(e.thickness).toBeUndefined();
    const out = packMaterials([e]);
    expect(out[11]).toBe(0); // thickness slot
  });

  it('round-trips through packMaterials to canonical bytes with library defaults', () => {
    // A bare spec (no transmission/ior/etc.) → packMaterials applies the
    // canonical defaults exactly as for a default-constructed THREE material.
    const out = packMaterials([coreMaterialToMaterialEntry(spec())]);
    expect([out[0], out[1], out[2]]).toEqual([1, 1, 1]);     // baseColor
    // The spec carries roughness 0.5 explicitly, so it passes through (it is
    // NOT replaced by MATERIAL_DEFAULT_ROUGHNESS — the default only fires when
    // a field is undefined). Assert against a spec that omits roughness to see
    // the default kick in.
    expect(out[3]).toBe(0.5);                                  // spec roughness
    // Omit roughness entirely to exercise packMaterials' default. Building the
    // bag by destructuring-out `roughness` keeps it strictly absent (not
    // `undefined`-valued), which exactOptionalPropertyTypes requires.
    const { roughness: _omitR, ...noRoughness } = coreMaterialToMaterialEntry(spec());
    void _omitR;
    const outDefault = packMaterials([noRoughness]);
    expect(outDefault[3]).toBe(MATERIAL_DEFAULT_ROUGHNESS);    // → 1.0 default
    expect(out[7]).toBe(0);                                    // metalness
    expect(out[8]).toBeCloseTo(1.5);                           // ior default
    expect(out[9]).toBe(0);                                    // transmission default
    expect(out[10]).toBe(MATERIAL_ATTEN_DIST_INFINITE);       // attenDist default
    expect(u32(out)[15]).toBe(0);                             // flags: not glass
  });

  it('a transmissive spec derives the glass flag via packMaterials', () => {
    const out = packMaterials([coreMaterialToMaterialEntry(spec({ transmission: 0.6 }))]);
    expect(u32(out)[15]).toBe(MATERIAL_FLAG_IS_GLASS);
  });
});
