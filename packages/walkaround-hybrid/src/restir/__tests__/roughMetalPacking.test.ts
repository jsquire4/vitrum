/**
 * B1 (road-to-100) — per-triangle roughness+metalness lane packing.
 *
 * Pins the bit layout (bits[31:24]=rough×255, bits[23:16]=metal×255), the
 * DIFFUSE-DEFAULT INVARIANT (no authored roughness → 0.85, glass → 0.05,
 * metal → 0), the WGSL decode round-trip, and parity between the structural
 * (`packBVHRoughMetal`) and core (`packBVHRoughMetalFromCore`) packers.
 */
import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  packBVHRoughMetal,
  packBVHRoughMetalFromCore,
  type PbrMaterialLike,
} from '../packingHelpers.js';

/** Mirror of the WGSL `decodeRoughMetal` (materialDecode.wgsl) — bits[31:24]
 *  rough, bits[23:16] metal, /255. */
function decodeRoughMetal(packed: number): { rough: number; metal: number } {
  const rough = ((packed >>> 24) & 0xff) / 255;
  const metal = ((packed >>> 16) & 0xff) / 255;
  return { rough, metal };
}

const ROUGH_DEFAULT = 0.85;
const ROUGH_GLASS = 0.05;

describe('packBVHRoughMetal — bit layout + decode round-trip', () => {
  it('packs authored roughness/metalness into the high two bytes', () => {
    const mats: PbrMaterialLike[] = [{ roughness: 0.2, metalness: 1.0 }];
    const buf = packBVHRoughMetal(new Uint32Array([0]), mats, 1);
    const { rough, metal } = decodeRoughMetal(buf[0]!);
    expect(rough).toBeCloseTo(0.2, 2);
    expect(metal).toBeCloseTo(1.0, 2);
    // low 16 bits reserved (zero).
    expect(buf[0]! & 0xffff).toBe(0);
  });

  it('DIFFUSE-DEFAULT INVARIANT: unspecified roughness → 0.85, metal → 0', () => {
    const buf = packBVHRoughMetal(new Uint32Array([0]), [{}], 1);
    const { rough, metal } = decodeRoughMetal(buf[0]!);
    expect(rough).toBeCloseTo(ROUGH_DEFAULT, 2);
    expect(metal).toBe(0);
  });

  it('glass (transmission > 0.5) with no authored roughness defaults to 0.05', () => {
    const buf = packBVHRoughMetal(new Uint32Array([0]), [{ transmission: 1.0 }], 1);
    const { rough } = decodeRoughMetal(buf[0]!);
    expect(rough).toBeCloseTo(ROUGH_GLASS, 2);
  });

  it('authored roughness on glass overrides the glass default', () => {
    const buf = packBVHRoughMetal(new Uint32Array([0]), [{ transmission: 1.0, roughness: 0.5 }], 1);
    const { rough } = decodeRoughMetal(buf[0]!);
    expect(rough).toBeCloseTo(0.5, 2);
  });

  it('clamps roughness/metalness to [0,1]', () => {
    const buf = packBVHRoughMetal(new Uint32Array([0]), [{ roughness: 4, metalness: -1 }], 1);
    const { rough, metal } = decodeRoughMetal(buf[0]!);
    expect(rough).toBeCloseTo(1.0, 2);
    expect(metal).toBe(0);
  });

  it('missing material slot falls back to (0.85, 0)', () => {
    // triMaterialId references slot 0 but the materials array is empty.
    const buf = packBVHRoughMetal(new Uint32Array([0]), [], 1);
    const { rough, metal } = decodeRoughMetal(buf[0]!);
    expect(rough).toBeCloseTo(ROUGH_DEFAULT, 2);
    expect(metal).toBe(0);
  });

  it('indexes per-triangle via triMaterialId', () => {
    const mats: PbrMaterialLike[] = [
      { roughness: 0.1, metalness: 1.0 },
      { roughness: 0.9, metalness: 0.0 },
    ];
    // tris [0,1,1] → materials [0,1,1]
    const buf = packBVHRoughMetal(new Uint32Array([0, 1, 1]), mats, 3);
    expect(decodeRoughMetal(buf[0]!).metal).toBeCloseTo(1.0, 2);
    expect(decodeRoughMetal(buf[1]!).rough).toBeCloseTo(0.9, 2);
    expect(decodeRoughMetal(buf[2]!).rough).toBeCloseTo(0.9, 2);
  });
});

describe('packBVHRoughMetalFromCore — parity with the structural packer', () => {
  it('core MaterialSpec (roughness/metallic) packs identically to PbrMaterialLike', () => {
    const coreMats: MaterialSpec[] = [
      { baseColor: [1, 1, 1], roughness: 0.3, metallic: 1.0 },
      { baseColor: [1, 1, 1], roughness: 0.7, metallic: 0.0, transmission: 0 },
    ];
    const pbrMats: PbrMaterialLike[] = [
      { roughness: 0.3, metalness: 1.0 },
      { roughness: 0.7, metalness: 0.0, transmission: 0 },
    ];
    const triIds = new Uint32Array([0, 1]);
    const core = packBVHRoughMetalFromCore(triIds, coreMats, 2);
    const pbr = packBVHRoughMetal(triIds, pbrMats, 2);
    expect(Array.from(core)).toEqual(Array.from(pbr));
  });

  it('core diffuse-default invariant matches (unspecified → 0.85; glass → 0.05)', () => {
    // `roughness`/`metallic` are technically required on MaterialSpec, but a
    // host may hand us a partial spec; the packer's `?? default` path is what
    // delivers the diffuse-default invariant, so the test exercises the partial
    // (undefined-roughness) shape deliberately.
    const coreMats = [
      { baseColor: [1, 1, 1] },                       // no roughness → 0.85
      { baseColor: [1, 1, 1], transmission: 1.0 },    // glass → 0.05
    ] as unknown as MaterialSpec[];
    const buf = packBVHRoughMetalFromCore(new Uint32Array([0, 1]), coreMats, 2);
    expect(decodeRoughMetal(buf[0]!).rough).toBeCloseTo(ROUGH_DEFAULT, 2);
    expect(decodeRoughMetal(buf[1]!).rough).toBeCloseTo(ROUGH_GLASS, 2);
  });
});
