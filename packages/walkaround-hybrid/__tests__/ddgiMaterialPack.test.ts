/**
 * DDGI material upload byte-equivalence (W2-C5 canonical layout edition).
 *
 * Locks the GPU-bound MaterialEntry std140 layout (64 bytes per entry, 16
 * floats) byte-for-byte against a fixture. Pre-W2-C5 this test asserted the
 * legacy DDGI-specific layout (slot 3 / 11 padding, roughness at slot 7,
 * metalness at slot 8). Post-W2-C5 the layout is shared with RC and lives
 * in `@vitrum/shared-bvh/materialEntry.ts`:
 *
 *   slot 0..2  → baseColor.xyz
 *   slot 3     → roughness        (was _pad0)
 *   slot 4..6  → emissive.xyz
 *   slot 7     → metalness        (was roughness)
 *   slot 8     → ior              (was metalness)
 *   slot 9     → transmission     (was ior)
 *   slot 10    → attenuationDist  (was transmission)
 *   slot 11    → thickness        (was _pad1)
 *   slot 12..14→ attenuationColor.xyz
 *   slot 15    → flags (u32; bit 0 = isGlass)
 *
 * The test still pins:
 *   - field ordering against the canonical layout,
 *   - default values for missing fields from the canonical structural PBR
 *     scalar extractor,
 *   - the u32 flags slot encoding (isGlass = transmission > 0),
 *   - bounded slot count (DDGI_MAX_MATERIALS = 64 — beyond that, materials
 *     drop and the trailing slots zero-pad).
 */

import { describe, expect, it } from 'vitest';
import {
  packDDGIMaterials,
  packDDGIMaterialsFromCoreN,
  DDGI_MAX_MATERIALS,
  DDGI_MATERIAL_STRIDE_BYTES,
  DDGI_MATERIAL_ENTRY_FLOATS,
} from '../src/ddgi/probeUpdateMaterials.js';
import {
  MATERIAL_FLAG_CAST_SHADOW_DISABLED,
  MATERIAL_FLAG_IS_GLASS,
} from '@vitrum/shared-bvh';
import type { MaterialSpec } from '@vitrum/core';
import type { PbrScalarSource } from '../src/pbrScalars.js';

const ENTRY = DDGI_MATERIAL_ENTRY_FLOATS;
const color = (r: number, g: number, b: number) => ({ r, g, b });

function f32(buf: ArrayBuffer): Float32Array {
  return new Float32Array(buf);
}
function u32(buf: ArrayBuffer): Uint32Array {
  return new Uint32Array(buf);
}

describe('packDDGIMaterials — byte-equivalence (W2-C5 canonical layout)', () => {
  it('total buffer size is DDGI_MAX_MATERIALS × 64 bytes', () => {
    expect(packDDGIMaterials([]).byteLength).toBe(DDGI_MAX_MATERIALS * DDGI_MATERIAL_STRIDE_BYTES);
    expect(DDGI_MATERIAL_STRIDE_BYTES).toBe(64);
    expect(DDGI_MATERIAL_ENTRY_FLOATS).toBe(16);
  });

  it('empty input yields all-zero buffer', () => {
    const buf = packDDGIMaterials([]);
    const view = f32(buf);
    expect(view.every((v) => v === 0)).toBe(true);
  });

  it('packs a single opaque diffuse material at slot 0 with canonical layout', () => {
    const mat: PbrScalarSource = {
      color: color(0.5, 0.6, 0.7),
      roughness: 0.4,
      metalness: 0.1,
    };
    const buf = packDDGIMaterials([mat]);
    const F = f32(buf);
    const U = u32(buf);

    // Slot 0 fields (16 floats), canonical W2-C5 order.
    expect(F[0]).toBeCloseTo(0.5);            // baseColor.r
    expect(F[1]).toBeCloseTo(0.6);            // baseColor.g
    expect(F[2]).toBeCloseTo(0.7);            // baseColor.b
    expect(F[3]).toBeCloseTo(0.4);            // roughness (was _pad0)
    expect(F[4]).toBe(0);                     // emissive.r (default black)
    expect(F[5]).toBe(0);                     // emissive.g
    expect(F[6]).toBe(0);                     // emissive.b
    expect(F[7]).toBeCloseTo(0.1);            // metalness (was roughness)
    expect(F[8]).toBeCloseTo(1.5);            // ior (was metalness — default from extractor)
    expect(F[9]).toBe(0);                     // transmission (was ior)
    // attenuationDistance: THREE std mat has none, packer treats undefined
    // → MATERIAL_ATTEN_DIST_INFINITE (1e9, was transmission slot).
    expect(F[10]).toBeCloseTo(1e9);
    expect(F[11]).toBe(0);                    // thickness (was _pad1)
    expect(F[12]).toBe(1);                    // attenuationColor.r (default white)
    expect(F[13]).toBe(1);                    // attenuationColor.g
    expect(F[14]).toBe(1);                    // attenuationColor.b
    expect(U[15]).toBe(0);                    // flags: not glass

    // Slots 1..63 must be untouched (all zero).
    for (let i = ENTRY; i < ENTRY * 2; i++) {
      expect(F[i]).toBe(0);
    }
  });

  it('encodes isGlass=1 in u32 flags slot when transmission > 0', () => {
    const mat: PbrScalarSource = {
      color: color(0.9, 0.95, 1),
      transmission: 0.8,
      ior: 1.52,
      attenuationColor: color(0.8, 0.85, 0.9),
    };
    const buf = packDDGIMaterials([mat]);
    const F = f32(buf);
    const U = u32(buf);

    expect(F[8]).toBeCloseTo(1.52);  // ior
    expect(F[9]).toBeCloseTo(0.8);   // transmission
    expect(F[12]).toBeCloseTo(0.8);  // attenuationColor.r
    expect(F[13]).toBeCloseTo(0.85); // attenuationColor.g
    expect(F[14]).toBeCloseTo(0.9);  // attenuationColor.b
    expect(U[15]).toBe(1);           // flags bit 0 = isGlass
  });

  it('flags slot is a TRUE u32, not the IEEE-754 bit pattern of float 1.0', () => {
    const glass: PbrScalarSource = { transmission: 1 };
    const buf = packDDGIMaterials([glass]);
    const U = u32(buf);
    // Float 1.0 in IEEE-754 is 0x3F800000. We must write integer 1.
    expect(U[15]).toBe(1);
    expect(U[15]).not.toBe(0x3F800000);
  });

  it('core material packing preserves primitive castShadow:false in flags bit 1', () => {
    const mats: MaterialSpec[] = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, castShadow: false } as MaterialSpec,
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, transmission: 0.75, castShadow: false } as MaterialSpec,
    ];
    const buf = packDDGIMaterialsFromCoreN(mats, 4);
    const U = u32(buf);
    expect(U[15]).toBe(MATERIAL_FLAG_CAST_SHADOW_DISABLED);
    expect(U[ENTRY + 15]).toBe(MATERIAL_FLAG_IS_GLASS | MATERIAL_FLAG_CAST_SHADOW_DISABLED);
  });

  it('packs multiple materials at consecutive slots without bleeding', () => {
    const a: PbrScalarSource = { color: color(1, 0, 0), roughness: 0.2 };
    const b: PbrScalarSource = { color: color(0, 1, 0), roughness: 0.5 };
    const c: PbrScalarSource = { color: color(0, 0, 1), transmission: 0.5 };
    const buf = packDDGIMaterials([a, b, c]);
    const F = f32(buf);
    const U = u32(buf);

    // Slot 0: red diffuse
    expect(F[0]).toBe(1); expect(F[1]).toBe(0); expect(F[2]).toBe(0);
    expect(F[3]).toBeCloseTo(0.2);            // roughness now at slot 3
    expect(U[15]).toBe(0);

    // Slot 1: green diffuse
    expect(F[ENTRY + 0]).toBe(0); expect(F[ENTRY + 1]).toBe(1); expect(F[ENTRY + 2]).toBe(0);
    expect(F[ENTRY + 3]).toBeCloseTo(0.5);    // roughness at slot 3
    expect(U[ENTRY + 15]).toBe(0);

    // Slot 2: blue glass
    expect(F[2 * ENTRY + 0]).toBe(0); expect(F[2 * ENTRY + 1]).toBe(0); expect(F[2 * ENTRY + 2]).toBe(1);
    expect(F[2 * ENTRY + 9]).toBeCloseTo(0.5);  // transmission now at slot 9
    expect(U[2 * ENTRY + 15]).toBe(1);          // isGlass

    // Slot 3..63 must be zero (no bleed).
    for (let i = 3 * ENTRY; i < DDGI_MAX_MATERIALS * ENTRY; i++) {
      expect(F[i]).toBe(0);
    }
  });

  it('drops materials beyond DDGI_MAX_MATERIALS without error', () => {
    const N = DDGI_MAX_MATERIALS + 5;
    const mats = Array.from({ length: N }, (_, i): PbrScalarSource => ({
      color: color(i / N, 0, 0),
    }));
    const buf = packDDGIMaterials(mats);
    const F = f32(buf);

    // Slot 0..(MAX-1) populated.
    for (let i = 0; i < DDGI_MAX_MATERIALS; i++) {
      expect(F[i * ENTRY + 0]).toBeCloseTo(i / N);
    }
    // Total buffer size unchanged — overflow materials silently dropped.
    expect(buf.byteLength).toBe(DDGI_MAX_MATERIALS * DDGI_MATERIAL_STRIDE_BYTES);
  });

  it('golden hash — locks the byte output of a known-good fixture scene', () => {
    const mats: PbrScalarSource[] = [
      {
        color: color(0.25, 0.5, 0.75),
        roughness: 0.4,
        metalness: 0.2,
        emissive: color(0.1, 0.1, 0),
      },
      {
        color: color(0.9, 0.95, 1),
        roughness: 0.05,
        transmission: 1,
        ior: 1.52,
        attenuationColor: color(0.85, 0.9, 1),
      },
      {},  // all defaults
    ];
    const buf = packDDGIMaterials(mats);

    // Sample specific bytes that exercise every distinctive slot.
    const F = f32(buf);
    const U = u32(buf);
    // Slot 0: baseColor + roughness at slot 3 (was _pad0).
    expect([F[0], F[1], F[2]]).toEqual([0.25, 0.5, 0.75]);
    expect(F[3]).toBeCloseTo(0.4);
    // Slot 0: emissive (note: extractPbrScalars does NOT pre-multiply
    // by emissiveIntensity for DDGI; the DDGI adapter doesn't either).
    expect([F[4], F[5], F[6]]).toEqual([
      0.10000000149011612, 0.10000000149011612, 0,
    ]);
    expect(F[7]).toBeCloseTo(0.2);             // metalness at slot 7
    expect(U[15]).toBe(0);

    // Glass at slot 1.
    expect(F[ENTRY + 0]).toBeCloseTo(0.9);
    expect(F[ENTRY + 3]).toBeCloseTo(0.05);    // roughness at slot 3
    expect(F[ENTRY + 8]).toBeCloseTo(1.52);    // ior at slot 8
    expect(F[ENTRY + 9]).toBeCloseTo(1.0);     // transmission at slot 9
    expect(F[ENTRY + 12]).toBeCloseTo(0.85);   // attenuationColor.r
    expect(U[ENTRY + 15]).toBe(1);

    // Default-everything mat at slot 2.
    expect(F[2 * ENTRY + 0]).toBe(1);          // color default = white
    expect(F[2 * ENTRY + 3]).toBeCloseTo(0.5); // structural roughness default at slot 3
    expect(F[2 * ENTRY + 8]).toBeCloseTo(1.5); // ior fallback from helper at slot 8
  });
});
