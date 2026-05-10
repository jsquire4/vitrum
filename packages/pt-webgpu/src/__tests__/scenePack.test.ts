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
    // Three vec4s per material.
    expect(packed.materials.length).toBe(12);
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
    // transmission + ior
    expect(packed.materials[8]).toBeCloseTo(0.0);
    expect(packed.materials[9]).toBeCloseTo(1.5);
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
