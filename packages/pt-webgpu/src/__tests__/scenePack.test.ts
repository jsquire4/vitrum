import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'tri',
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
        ]),
        normals: new Float32Array([
          0, 0, 1,
          0, 0, 1,
          0, 0, 1,
        ]),
        material: {
          baseColor: [0.25, 0.5, 0.75],
          roughness: 0.4,
          metallic: 0.2,
          emissive: [0.2, 0.1, 0],
          emissiveIntensity: 2,
        },
      },
    ],
    emitters: [
      {
        kind: 'directional',
        id: 'sun',
        direction: [0, -1, 0],
        color: [1, 0.8, 0.6],
        intensity: 3,
      },
    ],
    environment: { kind: 'none' },
  };
}

describe('buildPackedScene', () => {
  it('packs one triangle and one material payload', () => {
    const packed = buildPackedScene(makeScene());
    expect(packed.triangleCount).toBe(1);
    expect(Array.from(packed.indices)).toEqual([0, 1, 2, 0]);
    expect(packed.normals.length).toBe(12);
    expect(packed.bvhNodes.length).toBeGreaterThanOrEqual(8);
    expect(packed.bvhNodes.length % 8).toBe(0);
    // Twenty vec4s per material.
    expect(packed.materials.length).toBe(88);
    // base rgb + roughness
    expect(packed.materials[0]).toBeCloseTo(0.25);
    expect(packed.materials[1]).toBeCloseTo(0.5);
    expect(packed.materials[2]).toBeCloseTo(0.75);
    expect(packed.materials[3]).toBeCloseTo(0.4);
    // emissive rgb scaled by intensity + metallic
    expect(packed.materials[4]).toBeCloseTo(0.4);
    expect(packed.materials[5]).toBeCloseTo(0.2);
    expect(packed.materials[6]).toBeCloseTo(0.0);
    expect(packed.materials[7]).toBeCloseTo(0.2);
    // transmission + ior + scattering payload
    expect(packed.materials[8]).toBeCloseTo(0.0);
    expect(packed.materials[9]).toBeCloseTo(1.5);
    expect(packed.materials[10]).toBeCloseTo(0.0);
    expect(packed.materials[11]).toBeCloseTo(0.0);
    expect(packed.hasPointLight).toBe(false);
    expect(packed.environmentSunStrength).toBe(0);
  });

  it('supports rect-area emitters and packs HDRI importance data when pixel payload is provided', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      emitters: [
        ...base.emitters,
        {
          kind: 'rect-area',
          id: 'rect-a',
          position: [0, 1, 0],
          uAxis: [0.5, 0, 0],
          vAxis: [0, 0.5, 0],
          color: [1, 1, 1],
          intensity: 10,
        },
      ],
      environment: {
        kind: 'hdri',
        hdri: {
          width: 2,
          height: 2,
          data: new Float32Array([
            4, 1, 1,
            1, 4, 1,
            1, 1, 4,
            2, 2, 2,
          ]),
        },
      },
    };

    const packed = buildPackedScene(scene);
    expect(packed.hasRectAreaLight).toBe(true);
    expect(packed.rectAreaPosition).toEqual([0, 1, 0]);
    expect(packed.rectAreaRadiance[0]).toBeCloseTo(10);
    expect(packed.warnings.some((w) => w.includes('rect-area'))).toBe(false);
    expect(packed.hasEnvironmentMap).toBe(true);
    expect(packed.environmentMapWidth).toBe(2);
    expect(packed.environmentMapHeight).toBe(2);
    expect(packed.environmentMapTexels.length).toBe(16);
    expect(packed.environmentMapCdf.length).toBe(5);
    expect(packed.warnings.some((w) => w.includes('HDRI environment'))).toBe(false);
  });

  it('packs first point light', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      emitters: [
        ...base.emitters,
        {
          kind: 'point',
          id: 'point-a',
          position: [2, 3, 4],
          color: [0.5, 1, 0.25],
          intensity: 8,
        },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.hasPointLight).toBe(true);
    expect(packed.pointLightPosition).toEqual([2, 3, 4]);
    expect(packed.pointLightRadiance[0]).toBeCloseTo(4);
    expect(packed.pointLightRadiance[1]).toBeCloseTo(8);
    expect(packed.pointLightRadiance[2]).toBeCloseTo(2);
  });

  it('packs multiple point lights with counts (bounded arrays)', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      emitters: [
        ...base.emitters,
        {
          kind: 'point',
          id: 'p1',
          position: [1, 0, 0],
          color: [1, 0, 0],
          intensity: 1,
        },
        {
          kind: 'point',
          id: 'p2',
          position: [0, 2, 0],
          color: [0, 1, 0],
          intensity: 2,
        },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.pointLightCount).toBe(2);
    expect(packed.pointLightsData[0]).toBeCloseTo(1);
    expect(packed.pointLightsData[4]).toBeCloseTo(1);
    expect(packed.pointLightsData[8]).toBeCloseTo(0);
    expect(packed.pointLightsData[12]).toBeCloseTo(0);
    expect(packed.pointLightsData[13]).toBeCloseTo(2);
  });

  it('packs first spot light', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      emitters: [
        ...base.emitters,
        {
          kind: 'spot',
          id: 'spot-a',
          position: [5, 6, 7],
          direction: [0, -2, 0],
          angle: 0.5,
          color: [1, 0.5, 0.25],
          intensity: 4,
        },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.hasSpotLight).toBe(true);
    expect(packed.spotLightPosition).toEqual([5, 6, 7]);
    expect(packed.spotLightDirection[1]).toBeCloseTo(-1);
    expect(packed.spotLightRadiance[0]).toBeCloseTo(4);
    expect(packed.spotLightRadiance[1]).toBeCloseTo(2);
    expect(packed.spotLightRadiance[2]).toBeCloseTo(1);
  });

  it('packs first mesh-area emitter triangle sample', () => {
    const scene = makeScene();
    const meshAreaScene: Scene = {
      ...scene,
      emitters: [
        ...scene.emitters,
        {
          kind: 'mesh-area',
          id: 'mesh-light',
          meshId: 'tri',
          color: [0.5, 0.25, 1],
          intensity: 6,
        },
      ],
    };
    const packed = buildPackedScene(meshAreaScene);
    expect(packed.hasMeshAreaLight).toBe(true);
    expect(packed.meshAreaTriA).toEqual([0, 0, 0]);
    expect(packed.meshAreaTriB).toEqual([1, 0, 0]);
    expect(packed.meshAreaTriC).toEqual([0, 1, 0]);
    expect(packed.meshAreaRadiance[0]).toBeCloseTo(3);
    expect(packed.meshAreaRadiance[1]).toBeCloseTo(1.5);
    expect(packed.meshAreaRadiance[2]).toBeCloseTo(6);
  });

  it('packs layered, spectral, scattering, and thin-film payload summaries', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'tri',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.4, 0.5, 0.6],
            roughness: 0.35,
            metallic: 0.1,
            transmission: 1,
            ior: 1.52,
            scatteringCoefficient: 0.8,
            scatteringAnisotropy: 0.4,
            scatteringCoefficientRGB: [0.2, 0.3, 0.4],
            frontLayer: { transmission: [0.9, 0.8, 0.7], roughness: 0.15 },
            backLayer: { transmission: [0.7, 0.8, 0.9], roughness: 0.45 },
            thinFilmStack: {
              layers: [
                { ior: 2.1, thicknessNm: 70 },
                { ior: 1.45, thicknessNm: 110 },
              ],
            },
            spectralAttenuation: {
              wavelengthStart: 380,
              wavelengthEnd: 700,
              values: new Float32Array([0.1, 0.2, 0.3, 0.4]),
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = buildPackedScene(scene);
    expect(packed.materials.length).toBe(88);
    // scattering payload
    expect(packed.materials[10]).toBeCloseTo(0.8);
    expect(packed.materials[11]).toBeCloseTo(0.4);
    expect(packed.materials[12]).toBeCloseTo(0.2);
    expect(packed.materials[13]).toBeCloseTo(0.3);
    expect(packed.materials[14]).toBeCloseTo(0.4);
    // front/back layers
    expect(packed.materials[16]).toBeCloseTo(0.9);
    expect(packed.materials[17]).toBeCloseTo(0.8);
    expect(packed.materials[18]).toBeCloseTo(0.7);
    expect(packed.materials[19]).toBeCloseTo(0.15);
    expect(packed.materials[20]).toBeCloseTo(0.7);
    expect(packed.materials[21]).toBeCloseTo(0.8);
    expect(packed.materials[22]).toBeCloseTo(0.9);
    expect(packed.materials[23]).toBeCloseTo(0.45);
    // thin film + spectral summaries
    expect(packed.materials[24]).toBeCloseTo(1);
    expect(packed.materials[25]).toBeCloseTo(2);
    expect(packed.materials[26]).toBeCloseTo(1);
    expect(packed.materials[27]).toBeCloseTo(0);
    // thin-film bounded layer payload begins at index 28 (ior, thicknessNm, extinction)×layers.
    expect(packed.materials[28]).toBeCloseTo(2.1);
    expect(packed.materials[29]).toBeCloseTo(70);
    expect(packed.materials[30]).toBeCloseTo(0);
    expect(packed.materials[31]).toBeCloseTo(1.45);
    expect(packed.materials[32]).toBeCloseTo(110);
    expect(packed.materials[33]).toBeCloseTo(0);
    // spectral fixed-grid payload begins at index 52.
    expect(packed.materials[52]).toBeCloseTo(0.1);
    expect(packed.materials[53]).toBeGreaterThanOrEqual(0.1);
    expect(packed.materials[83]).toBeCloseTo(0.4, 1);
    // spectral metadata at indices 84..87.
    expect(packed.materials[84]).toBeGreaterThan(0);
    expect(packed.materials[85]).toBeCloseTo(0.1);
    expect(packed.materials[86]).toBeCloseTo(0.4);
    expect(packed.materials[87]).toBeCloseTo(32);
  });

  it('clamps mixed-material payload inputs to safe ranges', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'tri-clamped',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [0.5, 0.6, 0.7],
            roughness: 0.3,
            metallic: 0.05,
            transmission: 1,
            frontLayer: { transmission: [-0.2, 0.5, 1.4], roughness: 1.6 },
            backLayer: { transmission: [2, 0.25, -5], roughness: -2 },
            thinFilmStack: {
              layers: [{ ior: -3, thicknessNm: -40 }],
            },
            spectralAttenuation: {
              wavelengthStart: 380,
              wavelengthEnd: 780,
              values: new Float32Array([-1, 0.2, -0.3, 0.4]),
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene);
    expect(packed.materials[16]).toBeCloseTo(0);
    expect(packed.materials[17]).toBeCloseTo(0.5);
    expect(packed.materials[18]).toBeCloseTo(1);
    expect(packed.materials[19]).toBeCloseTo(1);
    expect(packed.materials[20]).toBeCloseTo(1);
    expect(packed.materials[22]).toBeCloseTo(0);
    expect(packed.materials[23]).toBeCloseTo(0);
    expect(packed.materials[28]).toBeGreaterThanOrEqual(1);
    expect(packed.materials[29]).toBeGreaterThanOrEqual(0);
    expect(packed.materials[52]).toBeGreaterThanOrEqual(0);
    expect(packed.materials[85]).toBeGreaterThanOrEqual(0);
  });

  it('derives procedural sky environment params', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      environment: {
        kind: 'procedural-sky',
        sunDirection: [0, 2, 0],
        turbidity: 2,
        rayleigh: 1.5,
        mieCoefficient: 0.02,
        mieDirectionalG: 0.8,
        intensity: 3,
      },
    };
    const packed = buildPackedScene(scene);
    expect(packed.environmentSunStrength).toBeCloseTo(3);
    expect(packed.environmentSunDirection[1]).toBeCloseTo(1);
    expect(packed.environmentTint[2]).toBeCloseTo(3);
  });

  it('warns and falls back when HDRI payload is opaque', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      environment: {
        kind: 'hdri',
        hdri: { mock: true },
      },
    };
    const packed = buildPackedScene(scene);
    expect(packed.hasEnvironmentMap).toBe(false);
    expect(packed.warnings.some((w) => w.includes('HDRI environment'))).toBe(true);
  });

  it('packs analytic primitives for shader intersections', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      primitives: [
        ...base.primitives,
        {
          kind: 'analytic',
          id: 'a-sphere',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 0.5]),
          material: {
            baseColor: [0.8, 0.2, 0.1],
            roughness: 0.5,
            metallic: 0.1,
          },
        },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.analyticCount).toBe(1);
    expect(packed.analyticHeaders.length).toBe(4);
    expect(packed.analyticParams.length).toBe(8);
    expect(packed.analyticLocalToWorld.length).toBe(16);
    expect(packed.analyticWorldToLocal.length).toBe(16);
  });
});
