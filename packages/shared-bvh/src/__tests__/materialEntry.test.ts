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

import {
  MATERIAL_ATTEN_DIST_INFINITE,
  MATERIAL_DEFAULT_ROUGHNESS,
  MATERIAL_ENTRY_FLOATS,
  MATERIAL_ENTRY_STRIDE_BYTES,
  MATERIAL_FLAG_IS_GLASS,
  packMaterials,
  type MaterialEntryInput,
} from '../materialEntry.js';

const ENTRY = MATERIAL_ENTRY_FLOATS;

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

  it('maxCount truncates an oversize input list', () => {
    const inputs: MaterialEntryInput[] = [
      { baseColor: [1, 0, 0] },
      { baseColor: [0, 1, 0] },
      { baseColor: [0, 0, 1] },
    ];
    const out = packMaterials(inputs, 2);
    expect(out.length).toBe(2 * ENTRY);
    expect(out[0]).toBe(1);
    expect(out[ENTRY + 1]).toBe(1);
    // Slot 2's baseColor.b would land at `2*ENTRY + 2` if present; truncated.
    expect(out[2 * ENTRY - 1]).toBe(0); // last slot of slot 1, not slot 2
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

  it('flags is written as a real u32 not the IEEE-754 bit pattern of 1.0', () => {
    const out = packMaterials([{ transmission: 1 }]);
    const U = u32(out);
    expect(U[15]).toBe(1);
    expect(U[15]).not.toBe(0x3F800000);
  });

  it('non-finite / negative attenuationDistance is replaced with the sentinel', () => {
    const out = packMaterials([
      { attenuationDistance: Infinity },
      { attenuationDistance: -3 },
      { attenuationDistance: 0 },
      { attenuationDistance: Number.NaN },
      { attenuationDistance: 2.5 },
    ]);
    expect(out[10]).toBe(MATERIAL_ATTEN_DIST_INFINITE);                 // Inf
    expect(out[1 * ENTRY + 10]).toBe(MATERIAL_ATTEN_DIST_INFINITE);     // -3
    expect(out[2 * ENTRY + 10]).toBe(MATERIAL_ATTEN_DIST_INFINITE);     // 0
    expect(out[3 * ENTRY + 10]).toBe(MATERIAL_ATTEN_DIST_INFINITE);     // NaN
    expect(out[4 * ENTRY + 10]).toBeCloseTo(2.5);                        // finite ok
  });
});
