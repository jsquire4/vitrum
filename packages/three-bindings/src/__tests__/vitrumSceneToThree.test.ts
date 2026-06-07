import { describe, expect, it } from 'vitest';
import { DirectionalLight, Mesh, SpotLight, Texture, Vector3 } from 'three';
import { vitrumSceneToThree } from '../vitrumSceneToThree.js';
import { resolveEnvironment } from '../environment.js';

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

describe('vitrumSceneToThree light target transforms', () => {
  it('parents directional and spot targets with local offsets that preserve world-space direction', () => {
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [
        {
          kind: 'directional',
          id: 'sun',
          color: [1, 1, 1],
          intensity: 1,
          direction: [0, 1, 0],
        },
        {
          kind: 'spot',
          id: 'spot',
          color: [1, 1, 1],
          intensity: 1,
          position: [4, 5, 6],
          direction: [0, 0, 1],
          angle: Math.PI / 5,
        },
      ],
      environment: { kind: 'none' },
    });
    const sun = scene.children.find((x) => x instanceof DirectionalLight) as DirectionalLight | undefined;
    const spot = scene.children.find((x) => x instanceof SpotLight) as SpotLight | undefined;
    expect(sun).toBeDefined();
    expect(spot).toBeDefined();
    if (sun == null || spot == null) return;

    // Compose world matrices BEFORE reading positions — lights are added with
    // `position` set but matrixWorld still identity until an update pass runs
    // (consumers like the fork call updateMatrixWorld themselves).
    scene.updateMatrixWorld(true);

    const sunPos = new Vector3().setFromMatrixPosition(sun.matrixWorld);
    const sunTarget = new Vector3().setFromMatrixPosition(sun.target.matrixWorld);
    expect(sunPos.sub(sunTarget).normalize().y).toBeCloseTo(1);

    const spotPos = new Vector3().setFromMatrixPosition(spot.matrixWorld);
    const spotTarget = new Vector3().setFromMatrixPosition(spot.target.matrixWorld);
    expect(spotPos.sub(spotTarget).normalize().z).toBeCloseTo(1);
  });

  it('writes light castShadow and directional angular diameter metadata', () => {
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [
        {
          kind: 'directional',
          id: 'sun',
          color: [1, 1, 1],
          intensity: 1,
          direction: [0, 1, 0],
          castShadow: false,
          angularDiameter: 0.0093,
        },
        {
          kind: 'spot',
          id: 'spot',
          color: [1, 1, 1],
          intensity: 1,
          position: [4, 5, 6],
          direction: [0, 0, 1],
          angle: Math.PI / 5,
          castShadow: true,
        },
      ],
      environment: { kind: 'none' },
    });
    const sun = scene.children.find((x) => x instanceof DirectionalLight) as DirectionalLight | undefined;
    const spot = scene.children.find((x) => x instanceof SpotLight) as SpotLight | undefined;
    expect(sun?.castShadow).toBe(false);
    expect(sun?.userData['vitrumLightAngularDiameter']).toBeCloseTo(0.0093);
    expect(spot?.castShadow).toBe(true);
  });
});

describe('vitrumSceneToThree HDRI environment intensity / rotation (Fix 2)', () => {
  it('writes environment.intensity to BOTH environmentIntensity and backgroundIntensity', () => {
    const hdri = new Texture();
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [],
      environment: { kind: 'hdri', hdri, intensity: 2.5 },
    });
    expect(scene.environment).toBe(hdri);
    expect(scene.environmentIntensity).toBeCloseTo(2.5);
    expect(scene.backgroundIntensity).toBeCloseTo(2.5);
  });

  it('writes environment.rotationY onto environmentRotation.y and backgroundRotation.y', () => {
    const hdri = new Texture();
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [],
      environment: { kind: 'hdri', hdri, intensity: 1, rotationY: Math.PI / 6 },
    });
    expect(scene.environmentRotation.y).toBeCloseTo(Math.PI / 6);
    expect(scene.backgroundRotation.y).toBeCloseTo(Math.PI / 6);
  });

  it('defaults intensity=1 and rotationY=0 when omitted on HDRI env', () => {
    const hdri = new Texture();
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [],
      environment: { kind: 'hdri', hdri },
    });
    expect(scene.environmentIntensity).toBe(1);
    expect(scene.backgroundIntensity).toBe(1);
    expect(scene.environmentRotation.y).toBe(0);
    expect(scene.backgroundRotation.y).toBe(0);
  });

  it('round-trips environment intensity + rotationY through resolveEnvironment → vitrumSceneToThree', () => {
    const hdri = new Texture();
    // Step 1: pretend a host scene has env + intensity 0.4 + rotation -π/4
    const env0 = { kind: 'hdri' as const, hdri, intensity: 0.4, rotationY: -Math.PI / 4 };
    // Step 2: vitrum → THREE
    const threeScene = vitrumSceneToThree({ primitives: [], emitters: [], environment: env0 });
    expect(threeScene.environmentIntensity).toBeCloseTo(0.4);
    expect(threeScene.environmentRotation.y).toBeCloseTo(-Math.PI / 4);
    // Step 3: THREE → vitrum (resolveEnvironment)
    const env1 = resolveEnvironment(threeScene);
    expect(env1.kind).toBe('hdri');
    if (env1.kind !== 'hdri') return;
    expect(env1.intensity).toBeCloseTo(0.4);
    expect(env1.rotationY).toBeCloseTo(-Math.PI / 4);
  });
});
