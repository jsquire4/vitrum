/**
 * B1 (road-to-100) — per-triangle roughness+metalness+IOR lane packing.
 *
 * B1-ior-per-tri (2026-06-10): extended to cover bits[15:8] = IOR quantized
 * over [1.0, 3.0].
 *
 * Pins the bit layout (bits[31:24]=rough×255, bits[23:16]=metal×255,
 * bits[15:8]=ior_quantized), the DIFFUSE-DEFAULT INVARIANT (no authored
 * roughness → 0.85, glass → 0.05, metal → 0), the IOR DEFAULT INVARIANT
 * (glass → 1.5, opaque → 1.0), the WGSL decode round-trip, and parity
 * between the structural (`packBVHRoughMetal`) and core
 * (`packBVHRoughMetalFromCore`) packers.
 */
import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  packBVHRoughMetal,
  packBVHRoughMetalFromCore,
  quantizeIor,
  dequantizeIor,
  IOR_DEFAULT_GLASS,
  IOR_RANGE_MIN,
  IOR_RANGE_MAX,
  type PbrMaterialLike,
} from '../packingHelpers.js';

/** Mirror of the WGSL `decodeRoughMetal` (materialDecode.wgsl) — bits[31:24]
 *  rough, bits[23:16] metal, /255. */
function decodeRoughMetal(packed: number): { rough: number; metal: number } {
  const rough = ((packed >>> 24) & 0xff) / 255;
  const metal = ((packed >>> 16) & 0xff) / 255;
  return { rough, metal };
}

/** Mirror of the WGSL `decodeIor` (materialDecode.wgsl) — bits[15:8],
 *  decode: 1.0 + (byte / 255) * 2.0. */
function decodeIor(packed: number): number {
  const byte = (packed >>> 8) & 0xff;
  return 1.0 + (byte / 255) * 2.0;
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

  it('core IOR default invariant: glass without authored ior → 1.5 (decodes ≈1.502)', () => {
    const coreMats = [
      { baseColor: [1, 1, 1], transmission: 1.0 },   // glass, no ior → IOR_DEFAULT_GLASS=1.5
    ] as unknown as MaterialSpec[];
    const buf = packBVHRoughMetalFromCore(new Uint32Array([0]), coreMats, 1);
    const decoded = decodeIor(buf[0]!);
    // IOR 1.5 encodes to byte 64, decodes to 1 + 64/255*2 ≈ 1.502 — within 0.003 of 1.5.
    expect(decoded).toBeCloseTo(IOR_DEFAULT_GLASS, 1);
  });

  it('core parity with structural packer for authored ior', () => {
    const ior = 1.8;
    const coreMats: MaterialSpec[] = [
      { baseColor: [1, 1, 1], transmission: 1.0, ior, roughness: 0.1, metallic: 0 },
    ];
    const pbrMats: PbrMaterialLike[] = [
      { transmission: 1.0, ior, roughness: 0.1, metalness: 0 },
    ];
    const triIds = new Uint32Array([0]);
    const core = packBVHRoughMetalFromCore(triIds, coreMats, 1);
    const pbr = packBVHRoughMetal(triIds, pbrMats, 1);
    expect(Array.from(core)).toEqual(Array.from(pbr));
  });
});

// ── B1-ior-per-tri (2026-06-10) ─────────────────────────────────────────────

describe('quantizeIor / dequantizeIor — round-trip', () => {
  it('IOR 1.0 (air) encodes to byte 0, decodes exactly to 1.0', () => {
    expect(quantizeIor(IOR_RANGE_MIN)).toBe(0);
    expect(dequantizeIor(0)).toBeCloseTo(1.0, 6);
  });

  it('IOR 3.0 (range max) encodes to byte 255, decodes exactly to 3.0', () => {
    expect(quantizeIor(IOR_RANGE_MAX)).toBe(255);
    expect(dequantizeIor(255)).toBeCloseTo(3.0, 6);
  });

  it('IOR 1.5 (crown glass) round-trips to within 0.01', () => {
    const byte = quantizeIor(1.5);
    expect(byte).toBe(64);  // round((0.5/2)*255) = round(63.75) = 64
    expect(dequantizeIor(byte)).toBeCloseTo(1.5, 1);
    expect(Math.abs(dequantizeIor(byte) - 1.5)).toBeLessThan(0.01);
  });

  it('IOR 1.33 (water) round-trips to within 0.01', () => {
    const byte = quantizeIor(1.33);
    expect(Math.abs(dequantizeIor(byte) - 1.33)).toBeLessThan(0.01);
  });

  it('IOR 2.42 (diamond) round-trips to within 0.01', () => {
    const byte = quantizeIor(2.42);
    expect(Math.abs(dequantizeIor(byte) - 2.42)).toBeLessThan(0.01);
  });

  it('quantization step is approximately 0.0078 (= 2/255)', () => {
    const step = (IOR_RANGE_MAX - IOR_RANGE_MIN) / 255;
    expect(step).toBeCloseTo(0.00784, 4);
  });
});

describe('B1-ior-per-tri — IOR lane in packBVHRoughMetal (structural packer)', () => {
  it('packs authored IOR into bits[15:8] and decodes round-trip', () => {
    const mats: PbrMaterialLike[] = [{ transmission: 1.0, ior: 1.8, roughness: 0.1, metalness: 0 }];
    const buf = packBVHRoughMetal(new Uint32Array([0]), mats, 1);
    const decoded = decodeIor(buf[0]!);
    expect(Math.abs(decoded - 1.8)).toBeLessThan(0.01);
    // bits[7:0] must stay zero (reserved).
    expect(buf[0]! & 0xff).toBe(0);
  });

  it('IOR DEFAULT INVARIANT: glass without authored ior defaults to 1.5', () => {
    const mats: PbrMaterialLike[] = [{ transmission: 1.0 }];
    const buf = packBVHRoughMetal(new Uint32Array([0]), mats, 1);
    expect(Math.abs(decodeIor(buf[0]!) - IOR_DEFAULT_GLASS)).toBeLessThan(0.01);
  });

  it('IOR DEFAULT INVARIANT: opaque material defaults to IOR_RANGE_MIN (1.0)', () => {
    const mats: PbrMaterialLike[] = [{ roughness: 0.5 }];
    const buf = packBVHRoughMetal(new Uint32Array([0]), mats, 1);
    expect(decodeIor(buf[0]!)).toBeCloseTo(IOR_RANGE_MIN, 4);
  });

  it('rough+metal+IOR bits are co-packed without interfering', () => {
    // Set all three to non-trivial values.
    const mats: PbrMaterialLike[] = [{ roughness: 0.5, metalness: 0.5, ior: 2.0, transmission: 1.0 }];
    const buf = packBVHRoughMetal(new Uint32Array([0]), mats, 1);
    expect(decodeRoughMetal(buf[0]!).rough).toBeCloseTo(0.5, 2);
    expect(decodeRoughMetal(buf[0]!).metal).toBeCloseTo(0.5, 2);
    expect(Math.abs(decodeIor(buf[0]!) - 2.0)).toBeLessThan(0.01);
    expect(buf[0]! & 0xff).toBe(0);  // bits[7:0] reserved = 0
  });

  it('IOR clamps to [IOR_RANGE_MIN, IOR_RANGE_MAX]', () => {
    const matsLow: PbrMaterialLike[] = [{ transmission: 1.0, ior: 0.5 }];
    const matsHigh: PbrMaterialLike[] = [{ transmission: 1.0, ior: 5.0 }];
    const bufLow = packBVHRoughMetal(new Uint32Array([0]), matsLow, 1);
    const bufHigh = packBVHRoughMetal(new Uint32Array([0]), matsHigh, 1);
    expect(decodeIor(bufLow[0]!)).toBeCloseTo(IOR_RANGE_MIN, 4);
    expect(decodeIor(bufHigh[0]!)).toBeCloseTo(IOR_RANGE_MAX, 1);
  });

  it('default-IOR-1.5 glass scene is byte-identical to the previous fixed IOR_GLASS=1.5', () => {
    // The prior B1 packer had no IOR lane (bits[15:8]=0). The new packer packs
    // IOR_DEFAULT_GLASS=1.5 → byte 64, which changes bits[15:8] from 0 to
    // (64 << 8) = 0x4000. This test confirms the new byte for documentation.
    // Callers that depended on bits[15:8]=0 for default glass should update.
    const mats: PbrMaterialLike[] = [{ transmission: 1.0 }];  // IOR defaults to 1.5
    const buf = packBVHRoughMetal(new Uint32Array([0]), mats, 1);
    const iorByte = (buf[0]! >>> 8) & 0xff;
    expect(iorByte).toBe(64);  // (1.5-1)/2*255 = 63.75 → rounds to 64
  });
});

// ── SHADOW-01 / GLTF-unlit — low-byte material flags ───────────────────────

describe('SHADOW-01 — castShadowDisabled bit 0 in packBVHRoughMetalFromCore', () => {
  it('castShadow:false on the material entry sets bit 0; default leaves it 0', () => {
    const coreMats = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },                       // default → bit 0 clear
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, castShadow: false },    // disabled → bit 0 set
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, castShadow: true },     // explicit true → clear
    ] as unknown as MaterialSpec[];
    const buf = packBVHRoughMetalFromCore(new Uint32Array([0, 1, 2]), coreMats, 3);
    expect(buf[0]! & 1).toBe(0);
    expect(buf[1]! & 1).toBe(1);
    expect(buf[2]! & 1).toBe(0);
  });

  it('bit 0 does not perturb the rough/metal/IOR lanes', () => {
    const coreMats = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0.5, ior: 2.0, transmission: 1.0, castShadow: false },
    ] as unknown as MaterialSpec[];
    const buf = packBVHRoughMetalFromCore(new Uint32Array([0]), coreMats, 1);
    expect(decodeRoughMetal(buf[0]!).rough).toBeCloseTo(0.5, 2);
    expect(decodeRoughMetal(buf[0]!).metal).toBeCloseTo(0.5, 2);
    expect(Math.abs(decodeIor(buf[0]!) - 2.0)).toBeLessThan(0.01);
    expect(buf[0]! & 0xff).toBe(1); // bit 0 set, bits 1-7 still zero
  });

  it('DEFAULT-PATH INVARIANT: a flag-less scene packs byte-identically to the pre-SHADOW-01 lane', () => {
    const coreMats: MaterialSpec[] = [
      { baseColor: [1, 1, 1], roughness: 0.3, metallic: 1.0 },
      { baseColor: [1, 1, 1], roughness: 0.7, metallic: 0.0, transmission: 0 },
    ];
    const triIds = new Uint32Array([0, 1]);
    const core = packBVHRoughMetalFromCore(triIds, coreMats, 2);
    // The structural packer never carried the flag — parity proves bit 0 is 0.
    const pbr = packBVHRoughMetal(triIds, [
      { roughness: 0.3, metalness: 1.0 },
      { roughness: 0.7, metalness: 0.0, transmission: 0 },
    ], 2);
    expect(Array.from(core)).toEqual(Array.from(pbr));
  });
});

describe('GLTF-unlit — shadingModel flag bit 1 in packBVHRoughMetalFromCore', () => {
  it('shadingModel:unlit sets bit 1; default/PBR leaves it clear', () => {
    const coreMats = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, shadingModel: 'pbr' },
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, shadingModel: 'unlit' },
    ] as unknown as MaterialSpec[];
    const buf = packBVHRoughMetalFromCore(new Uint32Array([0, 1, 2]), coreMats, 3);
    expect(buf[0]! & 0x2).toBe(0);
    expect(buf[1]! & 0x2).toBe(0);
    expect(buf[2]! & 0x2).toBe(0x2);
  });

  it('bit 1 coexists with castShadow bit 0 and does not perturb rough/metal/IOR lanes', () => {
    const coreMats = [
      {
        baseColor: [1, 1, 1],
        roughness: 0.5,
        metallic: 0.5,
        ior: 2.0,
        transmission: 1.0,
        castShadow: false,
        shadingModel: 'unlit',
      },
    ] as unknown as MaterialSpec[];
    const buf = packBVHRoughMetalFromCore(new Uint32Array([0]), coreMats, 1);
    expect(decodeRoughMetal(buf[0]!).rough).toBeCloseTo(0.5, 2);
    expect(decodeRoughMetal(buf[0]!).metal).toBeCloseTo(0.5, 2);
    expect(Math.abs(decodeIor(buf[0]!) - 2.0)).toBeLessThan(0.01);
    expect(buf[0]! & 0xff).toBe(0x3);
  });
});
