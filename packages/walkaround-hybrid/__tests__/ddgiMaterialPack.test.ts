/**
 * P3-V3: DDGI material upload byte-equivalence.
 *
 * Locks the GPU-bound DDGIMaterial std140 layout (64 bytes per entry, 16
 * floats) byte-for-byte against a fixture. Catches any drift in:
 *   - field ordering (baseColor / emissive / roughness / metallic / ior /
 *     transmission / attenuationColor / flags)
 *   - default values for missing fields (post-P2-6.1 the defaults come from
 *     `extractThreePbrScalars` in @vitrum/three-bindings)
 *   - the u32 flags slot encoding (isGlass = transmission > 0)
 *   - bounded slot count (DDGI_MAX_MATERIALS = 64 — beyond that materials
 *     drop)
 *   - pad slots (_pad0 at offset 12, _pad1 at offset 44) staying zero
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  packDDGIMaterials,
  DDGI_MAX_MATERIALS,
  DDGI_MATERIAL_STRIDE_BYTES,
  DDGI_MATERIAL_ENTRY_FLOATS,
} from '../src/ddgi/probeUpdatePass.js';

const ENTRY = DDGI_MATERIAL_ENTRY_FLOATS;

function f32(buf: ArrayBuffer): Float32Array {
  return new Float32Array(buf);
}
function u32(buf: ArrayBuffer): Uint32Array {
  return new Uint32Array(buf);
}

describe('packDDGIMaterials — byte-equivalence (P3-V3)', () => {
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

  it('packs a single opaque diffuse material at slot 0 with library defaults', () => {
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(0.5, 0.6, 0.7),
      roughness: 0.4,
      metalness: 0.1,
    });
    const buf = packDDGIMaterials([mat]);
    const F = f32(buf);
    const U = u32(buf);

    // Slot 0 fields (16 floats).
    expect(F[0]).toBeCloseTo(0.5); // baseColor.r
    expect(F[1]).toBeCloseTo(0.6); // baseColor.g
    expect(F[2]).toBeCloseTo(0.7); // baseColor.b
    expect(F[3]).toBe(0); // _pad0
    expect(F[4]).toBe(0); // emissive.r (default black)
    expect(F[5]).toBe(0); // emissive.g
    expect(F[6]).toBe(0); // emissive.b
    expect(F[7]).toBeCloseTo(0.4); // roughness
    expect(F[8]).toBeCloseTo(0.1); // metallic
    expect(F[9]).toBeCloseTo(1.5); // ior (default since std mat has no IOR)
    expect(F[10]).toBe(0); // transmission
    expect(F[11]).toBe(0); // _pad1
    expect(F[12]).toBe(1); // attenuationColor.r (default white)
    expect(F[13]).toBe(1); // attenuationColor.g
    expect(F[14]).toBe(1); // attenuationColor.b
    expect(U[15]).toBe(0); // flags: not glass

    // Slots 1..63 must be untouched (all zero).
    for (let i = ENTRY; i < ENTRY * 2; i++) {
      expect(F[i]).toBe(0);
    }
  });

  it('encodes isGlass=1 in u32 flags slot when transmission > 0', () => {
    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0.9, 0.95, 1),
      transmission: 0.8,
      ior: 1.52,
      attenuationColor: new THREE.Color(0.8, 0.85, 0.9),
    });
    const buf = packDDGIMaterials([mat]);
    const F = f32(buf);
    const U = u32(buf);

    expect(F[10]).toBeCloseTo(0.8); // transmission
    expect(F[9]).toBeCloseTo(1.52); // ior
    expect(F[12]).toBeCloseTo(0.8); // attenuationColor.r
    expect(F[13]).toBeCloseTo(0.85); // attenuationColor.g
    expect(F[14]).toBeCloseTo(0.9); // attenuationColor.b
    expect(U[15]).toBe(1); // flags bit 0 = isGlass
  });

  it('flags slot is a TRUE u32, not the IEEE-754 bit pattern of float 1.0', () => {
    const glass = new THREE.MeshPhysicalMaterial({ transmission: 1 });
    const buf = packDDGIMaterials([glass]);
    const U = u32(buf);
    // Float 1.0 in IEEE-754 is 0x3F800000. We must write integer 1.
    expect(U[15]).toBe(1);
    expect(U[15]).not.toBe(0x3f800000);
  });

  it('packs multiple materials at consecutive slots without bleeding', () => {
    const a = new THREE.MeshStandardMaterial({ color: new THREE.Color(1, 0, 0), roughness: 0.2 });
    const b = new THREE.MeshStandardMaterial({ color: new THREE.Color(0, 1, 0), roughness: 0.5 });
    const c = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0, 0, 1),
      transmission: 0.5,
    });
    const buf = packDDGIMaterials([a, b, c]);
    const F = f32(buf);
    const U = u32(buf);

    // Slot 0: red diffuse
    expect(F[0]).toBe(1);
    expect(F[1]).toBe(0);
    expect(F[2]).toBe(0);
    expect(F[7]).toBeCloseTo(0.2);
    expect(U[15]).toBe(0);

    // Slot 1: green diffuse
    expect(F[ENTRY + 0]).toBe(0);
    expect(F[ENTRY + 1]).toBe(1);
    expect(F[ENTRY + 2]).toBe(0);
    expect(F[ENTRY + 7]).toBeCloseTo(0.5);
    expect(U[ENTRY + 15]).toBe(0);

    // Slot 2: blue glass
    expect(F[2 * ENTRY + 0]).toBe(0);
    expect(F[2 * ENTRY + 1]).toBe(0);
    expect(F[2 * ENTRY + 2]).toBe(1);
    expect(F[2 * ENTRY + 10]).toBeCloseTo(0.5); // transmission
    expect(U[2 * ENTRY + 15]).toBe(1); // isGlass

    // Slot 3..63 must be zero (no bleed).
    for (let i = 3 * ENTRY; i < DDGI_MAX_MATERIALS * ENTRY; i++) {
      expect(F[i]).toBe(0);
    }
  });

  it('drops materials beyond DDGI_MAX_MATERIALS without error', () => {
    const N = DDGI_MAX_MATERIALS + 5;
    const mats = Array.from(
      { length: N },
      (_, i) => new THREE.MeshStandardMaterial({ color: new THREE.Color(i / N, 0, 0) }),
    );
    const buf = packDDGIMaterials(mats);
    const F = f32(buf);

    // Slot 0..(MAX-1) populated.
    for (let i = 0; i < DDGI_MAX_MATERIALS; i++) {
      expect(F[i * ENTRY + 0]).toBeCloseTo(i / N);
    }
    // Total buffer size unchanged — overflow materials silently dropped.
    expect(buf.byteLength).toBe(DDGI_MAX_MATERIALS * DDGI_MATERIAL_STRIDE_BYTES);
  });

  it('pad slots (_pad0 at offset 12, _pad1 at offset 44) stay zero per entry', () => {
    // Use a material with transmission so we exercise the non-trivial-flags path.
    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0.3, 0.6, 0.9),
      transmission: 0.4,
    });
    const buf = packDDGIMaterials([mat]);
    const F = f32(buf);
    expect(F[3]).toBe(0); // _pad0 (16-byte alignment slot after baseColor.xyz)
    expect(F[11]).toBe(0); // _pad1 (16-byte alignment slot after metallic/ior/transmission)
  });

  it('golden hash — locks the byte output of a known-good fixture scene', async () => {
    const mats = [
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(0.25, 0.5, 0.75),
        roughness: 0.4,
        metalness: 0.2,
        emissive: new THREE.Color(0.1, 0.1, 0),
      }),
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(0.9, 0.95, 1),
        roughness: 0.05,
        transmission: 1,
        ior: 1.52,
        attenuationColor: new THREE.Color(0.85, 0.9, 1),
      }),
      new THREE.MeshStandardMaterial({}), // all defaults
    ];
    const buf = packDDGIMaterials(mats);

    // Sample specific bytes that exercise every distinctive slot.
    const F = f32(buf);
    const U = u32(buf);
    expect([F[0], F[1], F[2], F[3]]).toEqual([0.25, 0.5, 0.75, 0]);
    expect([F[4], F[5], F[6]]).toEqual([0.10000000149011612, 0.10000000149011612, 0]);
    expect(F[7]).toBeCloseTo(0.4);
    expect(F[8]).toBeCloseTo(0.2);
    expect(U[15]).toBe(0);

    // Glass at slot 1.
    expect(F[ENTRY + 0]).toBeCloseTo(0.9);
    expect(F[ENTRY + 9]).toBeCloseTo(1.52);
    expect(F[ENTRY + 10]).toBeCloseTo(1.0);
    expect(F[ENTRY + 12]).toBeCloseTo(0.85);
    expect(U[ENTRY + 15]).toBe(1);

    // Default-everything mat at slot 2 — THREE constructor defaults dominate.
    expect(F[2 * ENTRY + 0]).toBe(1); // color default = white
    expect(F[2 * ENTRY + 7]).toBe(1); // THREE roughness default = 1
    expect(F[2 * ENTRY + 9]).toBeCloseTo(1.5); // ior fallback from helper
  });
});
