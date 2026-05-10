import { describe, expect, it } from 'vitest';
import { Mesh } from 'three';
import { vitrumSceneToThree } from '../vitrumSceneToThree.js';

describe('vitrumSceneToThree RFE userData stamping', () => {
  it('stamps frontLayer/backLayer metadata for fork packing', () => {
    const scene = vitrumSceneToThree({
      primitives: [
        {
          kind: 'mesh',
          id: 'pane',
          positions: new Float32Array([
            -1, -1, 0,
            1, -1, 0,
            0, 1, 0,
          ]),
          normals: new Float32Array([
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
          ]),
          indices: new Uint32Array([0, 1, 2]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 0.2,
            metallic: 0,
            transmission: 1,
            ior: 1.52,
            frontLayer: {
              transmission: [0.2, 0.3, 0.4],
              roughness: 0.6,
            },
            backLayer: {
              transmission: [0.9, 0.9, 0.9],
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    } as never);

    const mesh = scene.children.find((x) => x instanceof Mesh) as Mesh | undefined;
    expect(mesh).toBeDefined();
    const material = mesh?.material as { userData?: Record<string, unknown> } | undefined;
    const front = material?.userData?.['vitrumFrontLayer'] as { transmission?: number[]; roughness?: number };
    const back = material?.userData?.['vitrumBackLayer'] as { transmission?: number[] };

    expect(front?.transmission).toEqual([0.2, 0.3, 0.4]);
    expect(front?.roughness).toBeCloseTo(0.6);
    expect(back?.transmission).toEqual([0.9, 0.9, 0.9]);
  });

  it('stamps mixed spectral/scattering/layer/thin-film metadata together', () => {
    const scene = vitrumSceneToThree({
      primitives: [
        {
          kind: 'mesh',
          id: 'mixed-pane',
          positions: new Float32Array([
            -1, -1, 0,
            1, -1, 0,
            0, 1, 0,
          ]),
          normals: new Float32Array([
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
          ]),
          indices: new Uint32Array([0, 1, 2]),
          material: {
            baseColor: [0.7, 0.8, 0.9],
            roughness: 0.12,
            metallic: 0,
            transmission: 1,
            ior: 1.5,
            dispersionAbbeNumber: 36,
            scatteringCoefficient: 0.25,
            scatteringCoefficientRGB: [0.21, 0.22, 0.23],
            scatteringAnisotropy: 0.35,
            spectralAttenuation: {
              samples: [
                [380, 0.15],
                [520, 0.35],
                [780, 0.7],
              ],
            },
            thinFilmStack: {
              layers: [
                { ior: 1.33, thicknessNm: 90 },
                { ior: 1.5, thicknessNm: 180 },
              ],
            },
            frontLayer: { transmission: [0.4, 0.5, 0.6], roughness: 0.22 },
            backLayer: { transmission: [0.9, 0.85, 0.8], roughness: 0.1 },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    } as never);

    const mesh = scene.children.find((x) => x instanceof Mesh) as Mesh | undefined;
    expect(mesh).toBeDefined();
    const userData = (mesh?.material as { userData?: Record<string, unknown> })?.userData ?? {};

    expect(userData['vitrumDispersionAbbeNumber']).toBe(36);
    expect(userData['vitrumScatteringCoefficient']).toBe(0.25);
    expect(userData['vitrumScatteringCoefficientRGB']).toEqual([0.21, 0.22, 0.23]);
    expect(userData['vitrumScatteringAnisotropy']).toBe(0.35);
    expect(userData['vitrumSpectralAttenuation']).toEqual({
      samples: [
        [380, 0.15],
        [520, 0.35],
        [780, 0.7],
      ],
    });
    expect(userData['vitrumThinFilmStack']).toEqual({
      layers: [
        { ior: 1.33, thicknessNm: 90 },
        { ior: 1.5, thicknessNm: 180 },
      ],
    });
    expect(userData['vitrumFrontLayer']).toEqual({ transmission: [0.4, 0.5, 0.6], roughness: 0.22 });
    expect(userData['vitrumBackLayer']).toEqual({ transmission: [0.9, 0.85, 0.8], roughness: 0.1 });
  });
});
