import { describe, expect, it } from 'vitest';
import type { Scene, SpectralCurve, SurfaceAbsorptionLayer, ThinFilmStack } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import {
  materialToPackedVec4s,
  MATERIAL_FLOAT_STRIDE,
  THIN_FILM_RGB_LUT_BINS,
} from '../scene/materialPacking.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('material emissive Float32 publication', () => {
  const baseMaterial = {
    baseColor: [1, 1, 1] as [number, number, number],
    roughness: 0.5,
    metallic: 0,
  };

  it('uses staged f32 operands rather than a binary64 intermediate product', () => {
    const authored = 1.0000002;
    const intensity = 1.0000002;
    const directBinary64Product = Math.fround(authored * intensity);
    const stagedProduct = Math.fround(
      Math.fround(authored) * Math.fround(intensity),
    );
    expect(stagedProduct).not.toBe(directBinary64Product);

    const packed = materialToPackedVec4s({
      ...baseMaterial,
      emissive: [authored, authored, authored],
      emissiveIntensity: intensity,
    });
    expect(packed.slice(4, 7)).toEqual([
      stagedProduct,
      stagedProduct,
      stagedProduct,
    ]);
  });

  it('rejects overflow and total positive collapse while retaining zero and partial-lane underflow', () => {
    const maxFloat32 = Math.fround(3.4028234663852886e38);
    const minFloat32 = Math.fround(1.401298464324817e-45);

    expect(() => materialToPackedVec4s({
      ...baseMaterial,
      emissive: [maxFloat32, 0, 0],
      emissiveIntensity: 2,
    })).toThrow(/color.*intensity.*finite.*Float32/);
    expect(() => materialToPackedVec4s({
      ...baseMaterial,
      emissive: [minFloat32, 0, 0],
      emissiveIntensity: 0.5,
    })).toThrow(/color.*intensity.*underflow.*Float32/);

    expect(materialToPackedVec4s({
      ...baseMaterial,
      emissive: [minFloat32, 1, 0],
      emissiveIntensity: 0.5,
    }).slice(4, 7)).toEqual([0, 0.5, 0]);
    expect(materialToPackedVec4s({
      ...baseMaterial,
      emissive: [maxFloat32, 0, 0],
      emissiveIntensity: 0,
    }).slice(4, 7)).toEqual([0, 0, 0]);
  });
});

describe('buildPackedScene material payload packing', () => {
  it('packs layered/spectral/thin-film summaries', () => {
    const spectralValues = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const thinFilm: ThinFilmStack = { layers: [{ ior: 2.1, thicknessNm: 70 }, { ior: 1.45, thicknessNm: 110 }] };
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'tri',
        positions: new Float32Array([0,0,0, 1,0,0, 0,1,0]),
        normals: new Float32Array([0,0,1, 0,0,1, 0,0,1]),
        material: {
          baseColor: [0.4, 0.5, 0.6], roughness: 0.35, metallic: 0.1, transmission: 1, ior: 1.52,
          scatteringCoefficient: 0.8, scatteringAnisotropy: 0.4, scatteringCoefficientRGB: [0.2, 0.3, 0.4],
          frontLayer: { transmission: [0.9, 0.8, 0.7], roughness: 0.15 } satisfies SurfaceAbsorptionLayer,
          backLayer: { transmission: [0.7, 0.8, 0.9], roughness: 0.45 } satisfies SurfaceAbsorptionLayer,
          thinFilmStack: thinFilm,
          spectralAttenuation: { wavelengthStart: 380, wavelengthEnd: 700, values: spectralValues } satisfies SpectralCurve,
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = buildPackedScene(scene);
    expect(packed.materials.length).toBe(
      MATERIAL_FLOAT_STRIDE + THIN_FILM_RGB_LUT_BINS * 16,
    );
    expect(packed.materials[28 * 4 + 2]).toBe(29);
    expect(packed.materials[10]).toBeCloseTo(0.8);
    expect(packed.materials[24]).toBeCloseTo(1);
    expect(packed.materials[28]).toBeCloseTo(2.1);
    expect(packed.materials[52]).toBeCloseTo(0.1);
    expect(packed.materials[87]).toBeCloseTo(32);
  });

  it('clamps unsafe material values', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'tri-clamped',
        positions: new Float32Array([0,0,0, 1,0,0, 0,1,0]),
        normals: new Float32Array([0,0,1, 0,0,1, 0,0,1]),
        material: {
          baseColor: [0.5, 0.6, 0.7], roughness: 0.3, metallic: 0.05, transmission: 1,
          frontLayer: { transmission: [-0.2, 0.5, 1.4], roughness: 1.6 },
          backLayer: { transmission: [2, 0.25, -5], roughness: -2 },
          thinFilmStack: { layers: [{ ior: -3, thicknessNm: -40 }] },
          spectralAttenuation: { wavelengthStart: 380, wavelengthEnd: 780, values: new Float32Array([0, 0.2, 0.3, 0.4]) },
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = buildPackedScene(scene);
    expect(packed.materials[16]).toBeCloseTo(0);
    expect(packed.materials[18]).toBeCloseTo(1);
    expect(packed.materials[23]).toBeCloseTo(0);
    expect(packed.materials[29]).toBeGreaterThanOrEqual(0);
    expect(packed.materials[52]).toBeGreaterThanOrEqual(0);
  });

  it('fails closed instead of silently clamping invalid spectral samples', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'tri-invalid-spectrum',
        positions: new Float32Array([0,0,0, 1,0,0, 0,1,0]),
        normals: new Float32Array([0,0,1, 0,0,1, 0,0,1]),
        material: {
          baseColor: [0.5, 0.6, 0.7], roughness: 0.3, metallic: 0,
          spectralAttenuation: {
            wavelengthStart: 380,
            wavelengthEnd: 780,
            values: new Float32Array([-1, 0.2, 0.3, 0.4]),
          },
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(() => buildPackedScene(scene)).toThrow(
      /sampleSpectralCurve\.values\[0\] must be finite and non-negative/,
    );
  });
});

// ── PTWG-05 — spectral attenuation and Cauchy dispersion use separate lanes ──
describe('PTWG-05 spectral attenuation / dispersion packing', () => {
  const SPECTRAL_SUMMARY_OFFSET = 21 * 4; // vec4 #21: avgMu, dispersionAbbe, maxMu, sampleCount

  it('does not populate dispersionAbbe from spectral attenuation min-mu', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.7, 0.7, 0.7],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 780,
        values: new Float32Array([0.25, 0.5, 0.75]),
      },
    });

    expect(packed[SPECTRAL_SUMMARY_OFFSET + 0]).toBeGreaterThan(0);
    expect(packed[SPECTRAL_SUMMARY_OFFSET + 1]).toBe(0);
    expect(packed[SPECTRAL_SUMMARY_OFFSET + 2]).toBeGreaterThan(0);
    expect(packed[SPECTRAL_SUMMARY_OFFSET + 3]).toBe(32);
  });

  it('packs authored dispersionAbbe independently of spectral attenuation', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.7, 0.7, 0.7],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
      dispersionAbbeNumber: 42,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 780,
        values: new Float32Array([0.25, 0.5, 0.75]),
      },
    });

    expect(packed[SPECTRAL_SUMMARY_OFFSET + 1]).toBe(42);
    expect(packed[SPECTRAL_SUMMARY_OFFSET + 3]).toBe(32);
  });
});

// ── H52 — Disney extension lobe packing (clearcoat / sheen / iridescence) ────
// Vec4 layout: #23 = clearcoat/sheen scalars, #24 = sheenColor.rgb + iridescence,
//              #25 = iridescenceIor + thicknessMin + thicknessMax + pad.
// Float offsets: 92, 96, 100 respectively.
describe('H52 Disney extension lobe packing', () => {
  const CC_OFFSET   = 23 * 4; // 92
  const SH_OFFSET   = 24 * 4; // 96
  const IRID_OFFSET = 25 * 4; // 100

  it('packs clearcoat and clearcoatRoughness in vec4 #23 slots x,y', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      clearcoat: 0.75, clearcoatRoughness: 0.3,
    } as never);
    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    expect(packed[CC_OFFSET + 0]).toBeCloseTo(0.75);     // clearcoat
    expect(packed[CC_OFFSET + 1]).toBeCloseTo(0.3);      // clearcoatRoughness
    expect(packed[CC_OFFSET + 2]).toBeCloseTo(0);        // sheen = 0 default
    expect(packed[CC_OFFSET + 3]).toBeCloseTo(0);        // sheenRoughness = 0 default
  });

  it('packs sheen and sheenRoughness in vec4 #23 slots z,w', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      sheen: 0.6, sheenRoughness: 0.4,
    } as never);
    expect(packed[CC_OFFSET + 2]).toBeCloseTo(0.6);      // sheen
    expect(packed[CC_OFFSET + 3]).toBeCloseTo(0.4);      // sheenRoughness
  });

  it('packs sheenColor.rgb and iridescence in vec4 #24', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      sheenColor: [0.9, 0.5, 0.1], iridescence: 0.8,
    } as never);
    expect(packed[SH_OFFSET + 0]).toBeCloseTo(0.9);      // sheenColor.r
    expect(packed[SH_OFFSET + 1]).toBeCloseTo(0.5);      // sheenColor.g
    expect(packed[SH_OFFSET + 2]).toBeCloseTo(0.1);      // sheenColor.b
    expect(packed[SH_OFFSET + 3]).toBeCloseTo(0.8);      // iridescence
  });

  it('packs iridescenceIor, thicknessMin, thicknessMax, pad in vec4 #25', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      iridescence: 1, iridescenceIor: 2.0,
      iridescenceThicknessRange: [50, 600],
    } as never);
    expect(packed[IRID_OFFSET + 0]).toBeCloseTo(2.0);    // iridescenceIor
    expect(packed[IRID_OFFSET + 1]).toBeCloseTo(50);     // thicknessMin
    expect(packed[IRID_OFFSET + 2]).toBeCloseTo(600);    // thicknessMax
    expect(packed[IRID_OFFSET + 3]).toBe(0);             // pad
  });

  it('defaults all extension scalars to 0 when fields absent (zero-default invariant)', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0,
    } as never);
    // clearcoat, clearcoatRoughness, sheen, sheenRoughness all 0
    expect(packed[CC_OFFSET + 0]).toBe(0);
    expect(packed[CC_OFFSET + 1]).toBe(0);
    expect(packed[CC_OFFSET + 2]).toBe(0);
    expect(packed[CC_OFFSET + 3]).toBe(0);
    // sheenColor [0,0,0], iridescence 0
    expect(packed[SH_OFFSET + 0]).toBe(0);
    expect(packed[SH_OFFSET + 1]).toBe(0);
    expect(packed[SH_OFFSET + 2]).toBe(0);
    expect(packed[SH_OFFSET + 3]).toBe(0);
    // iridescenceIor defaults to 1.3 (safe even when iridescence=0 — WGSL never reads it)
    expect(packed[IRID_OFFSET + 0]).toBeCloseTo(1.3);
    expect(packed[IRID_OFFSET + 1]).toBeCloseTo(100);    // thicknessMin default
    expect(packed[IRID_OFFSET + 2]).toBeCloseTo(400);    // thicknessMax default
    expect(packed[IRID_OFFSET + 3]).toBe(0);             // pad
  });

  it('clamps extension scalars to [0,1] and iridescenceIor to ≥1', () => {
    const packed = materialToPackedVec4s({
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      clearcoat: 2.5, clearcoatRoughness: -0.1,
      sheen: -1, sheenRoughness: 3,
      sheenColor: [5, -2, 0.5], iridescence: 1.5,
      iridescenceIor: 0.5, // must clamp to ≥1
      iridescenceThicknessRange: [-10, 800],
    } as never);
    expect(packed[CC_OFFSET + 0]).toBeCloseTo(1);        // clearcoat clamped from 2.5
    expect(packed[CC_OFFSET + 1]).toBeCloseTo(0);        // clearcoatRoughness clamped from -0.1
    expect(packed[CC_OFFSET + 2]).toBeCloseTo(0);        // sheen clamped from -1
    expect(packed[CC_OFFSET + 3]).toBeCloseTo(1);        // sheenRoughness clamped from 3
    expect(packed[SH_OFFSET + 0]).toBeCloseTo(1);        // sheenColor.r clamped from 5
    expect(packed[SH_OFFSET + 1]).toBeCloseTo(0);        // sheenColor.g clamped from -2
    expect(packed[SH_OFFSET + 2]).toBeCloseTo(0.5);      // sheenColor.b in range
    expect(packed[SH_OFFSET + 3]).toBeCloseTo(1);        // iridescence clamped from 1.5
    expect(packed[IRID_OFFSET + 0]).toBeGreaterThanOrEqual(1); // iridescenceIor ≥ 1
    expect(packed[IRID_OFFSET + 1]).toBeCloseTo(0);      // thicknessMin clamped from -10
    expect(packed[IRID_OFFSET + 2]).toBeCloseTo(800);    // thicknessMax
  });
});

// ── A3 — baseColor Jakob-Hanika spectral coefficient packing (vec4 #26) ──────
describe('A3 spectral reflectance coefficient packing', () => {
  const SPEC_OFFSET = 26 * 4; // 104

  it('packs Jakob-Hanika coeffs c0,c1,c2 + flag=1 in vec4 #26', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.6, 0.2, 0.1], roughness: 0.5, metallic: 0,
    } as never);
    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    // The three coeffs are finite raw-nm sigmoid-polynomial coefficients.
    expect(Number.isFinite(packed[SPEC_OFFSET + 0]!)).toBe(true);
    expect(Number.isFinite(packed[SPEC_OFFSET + 1]!)).toBe(true);
    expect(Number.isFinite(packed[SPEC_OFFSET + 2]!)).toBe(true);
    expect(packed[SPEC_OFFSET + 3]).toBe(1); // hasSpectralReflectance flag
  });

  it('packs MaterialSpec.shadingModel=unlit as bit1 in vec4 #26 .w', () => {
    const pbr = materialToPackedVec4s({
      baseColor: [0.2, 0.4, 0.8], roughness: 0.5, metallic: 0,
    } as never);
    const unlit = materialToPackedVec4s({
      baseColor: [0.2, 0.4, 0.8], roughness: 0.5, metallic: 0, shadingModel: 'unlit',
    } as never);
    expect(pbr[SPEC_OFFSET + 3]).toBe(1);
    expect(unlit[SPEC_OFFSET + 3]).toBe(3);
  });

  it('packs MaterialSpec.doubleSided as bit2 and defaults it to false', () => {
    const oneSided = materialToPackedVec4s({
      baseColor: [0.2, 0.4, 0.8], roughness: 0.5, metallic: 0,
    });
    const twoSided = materialToPackedVec4s({
      baseColor: [0.2, 0.4, 0.8], roughness: 0.5, metallic: 0, doubleSided: true,
    });
    const twoSidedUnlit = materialToPackedVec4s({
      baseColor: [0.2, 0.4, 0.8], roughness: 0.5, metallic: 0,
      doubleSided: true, shadingModel: 'unlit',
    });
    expect(oneSided[SPEC_OFFSET + 3]).toBe(1);
    expect(twoSided[SPEC_OFFSET + 3]).toBe(5);
    expect(twoSidedUnlit[SPEC_OFFSET + 3]).toBe(7);
  });

  it('round-trips a neutral grey to a flat spectrum (≈ albedo at all λ)', () => {
    // A neutral grey albedo must upsample to a near-flat reflectance whose
    // CMF integral reproduces the grey — the flat-spectrum invariant the GPU
    // hero-λ transport relies on. We verify the coefficients reproduce ~0.5.
    const grey = 0.5;
    const packed = materialToPackedVec4s({
      baseColor: [grey, grey, grey], roughness: 0.5, metallic: 0,
    } as never);
    const c0 = packed[SPEC_OFFSET + 0]!;
    const c1 = packed[SPEC_OFFSET + 1]!;
    const c2 = packed[SPEC_OFFSET + 2]!;
    const evalS = (lam: number): number => {
      const x = c0 + c1 * lam + c2 * lam * lam;
      return 0.5 + x / (2 * Math.sqrt(1 + x * x));
    };
    // Sample across the visible band; a neutral grey should be near-flat.
    for (const lam of [420, 500, 580, 660, 720]) {
      expect(evalS(lam)).toBeGreaterThan(0.35);
      expect(evalS(lam)).toBeLessThan(0.65);
    }
  });

  it('pure black packs the solver shortcut (S≈0)', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0, 0, 0], roughness: 0.5, metallic: 0,
    } as never);
    const c0 = packed[SPEC_OFFSET + 0]!;
    const x = c0 + packed[SPEC_OFFSET + 1]! * 550 + packed[SPEC_OFFSET + 2]! * 550 * 550;
    const s = 0.5 + x / (2 * Math.sqrt(1 + x * x));
    expect(s).toBeLessThan(0.05);
  });
});

// ── SPEC-01 — KHR_materials_specular scalar packing (vec4 #27) ───────────────
describe('SPEC-01 specular scalar packing', () => {
  const SPECULAR_OFFSET = 27 * 4; // 108

  it('packs specularColor.rgb and specularIntensity after the spectral flags', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.8, 0.7, 0.6],
      roughness: 0.5,
      metallic: 0,
      specularColor: [0.2, 0.4, 0.9],
      specularIntensity: 0.35,
    });
    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    expect(packed[SPECULAR_OFFSET + 0]).toBeCloseTo(0.2);
    expect(packed[SPECULAR_OFFSET + 1]).toBeCloseTo(0.4);
    expect(packed[SPECULAR_OFFSET + 2]).toBeCloseTo(0.9);
    expect(packed[SPECULAR_OFFSET + 3]).toBeCloseTo(0.35);
  });

  it('preserves KHR specularColor factors above one', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.8, 0.7, 0.6],
      roughness: 0.5,
      metallic: 0,
      specularColor: [1.25, 4, 0],
    });
    expect(packed.slice(SPECULAR_OFFSET, SPECULAR_OFFSET + 3))
      .toEqual([1.25, 4, 0]);
  });

  it('defaults to the KHR_materials_specular no-op values', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.8, 0.7, 0.6],
      roughness: 0.5,
      metallic: 0,
    });
    expect(packed[SPECULAR_OFFSET + 0]).toBe(1);
    expect(packed[SPECULAR_OFFSET + 1]).toBe(1);
    expect(packed[SPECULAR_OFFSET + 2]).toBe(1);
    expect(packed[SPECULAR_OFFSET + 3]).toBe(1);
  });
});

// ── Material stride consistency gate (TS vs WGSL lockstep) ───────────────────
// MATERIAL_VEC4_STRIDE is a constant that exists in three places:
//   1. TypeScript: materialPacking.ts (MATERIAL_VEC4_STRIDE = 29, exported as
//      MATERIAL_FLOAT_STRIDE = 116)
//   2. WGSL: material.wgsl.ts (const MATERIAL_VEC4_STRIDE = 29u;)
//   3. WGSL: emissive-only adjointPass.wgsl.ts (checked by
//      emissivePathReplayOracle.test.ts)
// If they diverge, every material read in the GPU kernel/replay is silently misaligned.
// This test checks both sources agree, and that the TS float-stride is exactly
// 4× the WGSL vec4-stride.
describe('material stride consistency (TS vs WGSL lockstep)', () => {
  it('MATERIAL_FLOAT_STRIDE equals 29 * 4 = 116 (VOL-THICKNESS bumped 28→29)', () => {
    expect(MATERIAL_FLOAT_STRIDE).toBe(116);
  });

  it('WGSL MATERIAL_VEC4_STRIDE constant matches TS stride / 4', () => {
    // Parse the integer from the WGSL constant declaration.
    const match = PT_WEBGPU_TRACE_WGSL.match(/const MATERIAL_VEC4_STRIDE\s*=\s*(\d+)u;/);
    expect(match).not.toBeNull();
    const wgslStride = parseInt(match![1]!, 10);
    expect(wgslStride * 4).toBe(MATERIAL_FLOAT_STRIDE);
  });
});

describe('VOL-THICKNESS KHR_materials_volume scalar packing', () => {
  const VOLUME_THICKNESS_OFFSET = 28 * 4;

  it('packs thickness plus a presence flag in vec4 #28', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.7, 0.7, 0.7],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
      thickness: 0.42,
    });
    expect(packed[VOLUME_THICKNESS_OFFSET]).toBeCloseTo(0.42);
    expect(packed[VOLUME_THICKNESS_OFFSET + 1]).toBe(1);
  });

  it('marks presence when only thicknessMap is authored, preserving glTF default factor 0', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.7, 0.7, 0.7],
      roughness: 0.5,
      metallic: 0,
      thicknessMap: { handle: { id: 'thickness' } },
    } as never);
    expect(packed[VOLUME_THICKNESS_OFFSET]).toBe(0);
    expect(packed[VOLUME_THICKNESS_OFFSET + 1]).toBe(1);
  });

  it('leaves the clamp absent for legacy materials', () => {
    const packed = materialToPackedVec4s({
      baseColor: [0.7, 0.7, 0.7],
      roughness: 0.5,
      metallic: 0,
    });
    expect(packed[VOLUME_THICKNESS_OFFSET]).toBe(0);
    expect(packed[VOLUME_THICKNESS_OFFSET + 1]).toBe(0);
  });
});

// ── SHADOW-01 (2026-06-11) — primitive castShadow lane (vec4 #25 .w) ─────────
describe('SHADOW-01 primitive castShadow material lane', () => {
  const IRID_OFFSET = 25 * 4; // 100

  it('castShadow:false packs 1.0 into vec4 #25 .w; default packs 0.0 (byte-identical pad)', () => {
    const mat = { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 } as never;
    expect(materialToPackedVec4s(mat)[IRID_OFFSET + 3]).toBe(0);
    expect(materialToPackedVec4s(mat, {})[IRID_OFFSET + 3]).toBe(0);
    expect(materialToPackedVec4s(mat, { castShadow: true })[IRID_OFFSET + 3]).toBe(0);
    expect(materialToPackedVec4s(mat, { castShadow: undefined })[IRID_OFFSET + 3]).toBe(0);
    expect(materialToPackedVec4s(mat, { castShadow: false })[IRID_OFFSET + 3]).toBe(1);
  });

  it('DEFAULT-PATH INVARIANT: context-less packing equals empty-context packing byte-for-byte', () => {
    const mat = {
      baseColor: [0.3, 0.6, 0.9], roughness: 0.2, metallic: 0.8,
      iridescence: 1, iridescenceIor: 2.0, iridescenceThicknessRange: [50, 600],
    } as never;
    expect(materialToPackedVec4s(mat)).toEqual(materialToPackedVec4s(mat, { castShadow: true }));
  });
});
