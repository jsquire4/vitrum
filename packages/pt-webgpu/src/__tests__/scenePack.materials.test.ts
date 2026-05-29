import { describe, expect, it } from 'vitest';
import type { Scene, SpectralCurve, SurfaceAbsorptionLayer, ThinFilmStack } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

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
    expect(packed.materials.length).toBe(92); // WS4: MATERIAL_FLOAT_STRIDE 88 → 92 (σ_a vec4)
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
          spectralAttenuation: { wavelengthStart: 380, wavelengthEnd: 780, values: new Float32Array([-1, 0.2, -0.3, 0.4]) },
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
});
