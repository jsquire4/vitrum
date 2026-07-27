import type { MaterialSpec, Scene } from '@vitrum/core';
import { describe, expect, it } from 'vitest';
import {
  deriveSceneTraceFeatures,
  materialUsesBasicWebGl2Shader,
  materialUsesMappedPbrWebGl2Shader,
  materialUsesMappedRichWebGl2Shader,
  materialUsesScalarRichWebGl2Shader,
  validateWebGl2SceneMaterials,
} from './sceneTraceFeatures.js';

const BASIC_MATERIAL: MaterialSpec = {
  baseColor: [0.7, 0.4, 0.2],
  roughness: 0.6,
  metallic: 0.1,
};

function sceneWith(
  material: MaterialSpec,
  emitters: Scene['emitters'] = [],
  environment: Scene['environment'] = { kind: 'none' },
): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'triangle',
        positions: new Float32Array([-1, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint32Array([0, 1, 2]),
        material,
      },
    ],
    emitters,
    environment,
  };
}

describe('deriveSceneTraceFeatures', () => {
  it('selects the compact tier only for the authored opaque base-PBR subset', () => {
    expect(materialUsesBasicWebGl2Shader(BASIC_MATERIAL)).toBe(true);
    expect(
      materialUsesBasicWebGl2Shader({
        ...BASIC_MATERIAL,
        emissive: [0.1, 0.2, 0.3],
        emissiveIntensity: 2,
        shadingModel: 'unlit',
      }),
    ).toBe(true);
  });

  it('selects the scalar-rich tier for texture-free full transport materials', () => {
    const material: MaterialSpec = {
      ...BASIC_MATERIAL,
      alphaMode: 'blend',
      opacity: 0.75,
      transmission: 0.9,
      ior: 1.52,
      attenuationColor: [0.8, 0.9, 1],
      attenuationDistance: 2,
      thickness: 0.4,
      clearcoat: 0.3,
      sheen: 0.2,
      iridescence: 0.5,
      iridescenceIor: 1.3,
      iridescenceThicknessRange: [120, 400],
      specularIntensity: 0.8,
      specularColor: [1, 0.9, 0.8],
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0.1, 0.2, 0.3]),
      },
      dispersionAbbeNumber: 45,
      scatteringCoefficientRGB: [0.02, 0.03, 0.04],
      scatteringAnisotropy: 0.3,
      frontLayer: { transmission: [0.9, 0.8, 0.7], roughness: 0.2 },
      backLayer: { transmission: [0.7, 0.8, 0.9] },
      thinFilmStack: {
        incidentIor: 1,
        angleDependent: true,
        layers: [{ ior: 1.4, extinctionCoefficient: 0.01, thicknessNm: 180 }],
      },
      anisotropy: 0.4,
      anisotropyRotation: 0.7,
    };

    expect(materialUsesBasicWebGl2Shader(material)).toBe(false);
    expect(materialUsesScalarRichWebGl2Shader(material)).toBe(true);
    expect(deriveSceneTraceFeatures(sceneWith(material))).toMatchObject({
      basicMaterials: false,
      scalarRichMaterials: true,
      mappedPbrMaterials: false,
      mappedRichMaterials: false,
      fog: true,
    });
  });

  it('accepts double-sided materials and routes them through a side-aware shader tier', () => {
    const material: MaterialSpec = {
      ...BASIC_MATERIAL,
      doubleSided: true,
    };

    // The compact tier has no material-side lane, so authored double-sided
    // behavior must promote to scalar-rich rather than being dropped.
    expect(materialUsesBasicWebGl2Shader(material)).toBe(false);
    expect(materialUsesScalarRichWebGl2Shader(material)).toBe(true);
    expect(() => validateWebGl2SceneMaterials(sceneWith(material))).not.toThrow();
    expect(deriveSceneTraceFeatures(sceneWith(material))).toMatchObject({
      basicMaterials: false,
      scalarRichMaterials: true,
      mappedPbrMaterials: false,
      mappedRichMaterials: false,
    });
  });

  it('selects mapped PBR only when every authored field has mapped-tier behavior', () => {
    const texture = { handle: new Uint8Array([255, 128, 64, 255]) };
    const material: MaterialSpec = {
      ...BASIC_MATERIAL,
      alphaMode: 'mask',
      alphaCutoff: 0.4,
      opacity: 0.9,
      doubleSided: true,
      ior: 1.45,
      baseColorMap: texture,
      normalMap: texture,
      normalScale: 0.7,
      roughnessMap: texture,
      metallicMap: texture,
      emissiveMap: texture,
      alphaMap: texture,
      aoMap: texture,
      aoMapIntensity: 0.6,
      bumpMap: texture,
      bumpScale: 0.3,
      displacementMap: texture,
      displacementScale: 0.05,
      displacementBias: -0.01,
      displacementSubdivisions: 1,
      lightMap: texture,
      lightMapIntensity: 0.8,
      envMapIntensity: 0.75,
    };

    expect(materialUsesBasicWebGl2Shader(material)).toBe(false);
    expect(materialUsesScalarRichWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedPbrWebGl2Shader(material)).toBe(true);
    expect(deriveSceneTraceFeatures(sceneWith(material))).toMatchObject({
      basicMaterials: false,
      scalarRichMaterials: false,
      mappedPbrMaterials: true,
      mappedRichMaterials: false,
    });
  });

  it('selects mapped-rich for mixed maps, transmission, Disney, thin-film, and volume', () => {
    const texture = { handle: new Uint8Array([128, 128, 255, 255]) };
    const material: MaterialSpec = {
      ...BASIC_MATERIAL,
      baseColorMap: texture,
      normalMap: texture,
      transmission: 0.82,
      transmissionMap: texture,
      thickness: 0.35,
      thicknessMap: texture,
      attenuationColor: [0.72, 0.88, 0.95],
      attenuationDistance: 1.7,
      scatteringCoefficientRGB: [0.03, 0.02, 0.01],
      scatteringAnisotropy: 0.25,
      clearcoat: 0.55,
      clearcoatMap: texture,
      clearcoatNormalMap: texture,
      sheen: 0.2,
      sheenColorMap: texture,
      iridescence: 0.4,
      iridescenceMap: texture,
      specularIntensityMap: texture,
      anisotropy: 0.35,
      anisotropyMap: texture,
      frontLayer: {
        transmission: [0.9, 0.8, 0.7],
        roughness: 0.15,
        normalMap: texture,
      },
      thinFilmStack: {
        incidentIor: 1,
        angleDependent: true,
        layers: [{ ior: 1.38, thicknessNm: 115 }],
      },
    };

    expect(materialUsesMappedPbrWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedRichWebGl2Shader(material)).toBe(true);
    expect(() => validateWebGl2SceneMaterials(sceneWith(material))).not.toThrow();
    expect(deriveSceneTraceFeatures(sceneWith(material))).toMatchObject({
      basicMaterials: false,
      scalarRichMaterials: false,
      mappedPbrMaterials: false,
      mappedRichMaterials: true,
    });
  });

  it('accepts multiple participating-medium boundaries for the bounded stack', () => {
    const volume: MaterialSpec = {
      ...BASIC_MATERIAL,
      transmission: 1,
      scatteringCoefficient: 0.1,
    };
    const first = sceneWith(volume).primitives[0]!;
    const scene: Scene = {
      primitives: [first, { ...first, id: 'second-volume' }],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(() => validateWebGl2SceneMaterials(scene, 'setScene')).not.toThrow();
    expect(deriveSceneTraceFeatures(scene)).toEqual(
      expect.objectContaining({
        fog: true,
        scalarRichMaterials: true,
      }),
    );
  });

  it.each([
    ['extensions', { extensions: { vendor: true } }],
    ['unknown runtime key', { futureMaterialField: true }],
    [
      'unknown nested spectral field',
      {
        spectralAttenuation: {
          wavelengthStart: 380,
          wavelengthEnd: 700,
          values: new Float32Array([0.1, 0.2, 0.3]),
          interpolation: 'cubic',
        },
      },
    ],
  ])('fails closed for %s', (_label, authoredField) => {
    const material = { ...BASIC_MATERIAL, ...authoredField } as unknown as MaterialSpec;
    expect(materialUsesBasicWebGl2Shader(material)).toBe(false);
    expect(materialUsesScalarRichWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedPbrWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedRichWebGl2Shader(material)).toBe(false);
    expect(deriveSceneTraceFeatures(sceneWith(material))).toMatchObject({
      basicMaterials: false,
      scalarRichMaterials: false,
      mappedPbrMaterials: false,
      mappedRichMaterials: false,
    });
    expect(() => validateWebGl2SceneMaterials(sceneWith(material))).toThrow(
      /\[vitrum\/pt-webgl2\] setScene: primitive "triangle" material/,
    );
  });

  it('forces the full shader for symbol-keyed runtime extensions', () => {
    const extension = Symbol('extension');
    const material = { ...BASIC_MATERIAL, [extension]: true } as MaterialSpec;
    expect(materialUsesBasicWebGl2Shader(material)).toBe(false);
    expect(materialUsesScalarRichWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedPbrWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedRichWebGl2Shader(material)).toBe(false);
    expect(() => validateWebGl2SceneMaterials(sceneWith(material))).toThrow(
      /unsupported symbol key/,
    );
  });

  it('does not ignore non-enumerable runtime fields', () => {
    const material = { ...BASIC_MATERIAL };
    Object.defineProperty(material, 'futureMaterialField', { value: true });
    expect(materialUsesBasicWebGl2Shader(material)).toBe(false);
    expect(materialUsesScalarRichWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedPbrWebGl2Shader(material)).toBe(false);
    expect(materialUsesMappedRichWebGl2Shader(material)).toBe(false);
    expect(() => validateWebGl2SceneMaterials(sceneWith(material))).toThrow(/futureMaterialField/);
  });

  it('derives analytic, mesh, and environment families independently', () => {
    const features = deriveSceneTraceFeatures(
      sceneWith(
        BASIC_MATERIAL,
        [
          {
            kind: 'point',
            id: 'point',
            position: [0, 1, 0],
            color: [1, 1, 1],
            intensity: 1,
          },
          {
            kind: 'mesh-area',
            id: 'mesh-light',
            meshId: 'triangle',
            color: [1, 1, 1],
            intensity: 1,
          },
        ],
        {
          kind: 'procedural-sky',
          sunDirection: [0, 1, 0],
          turbidity: 2,
          rayleigh: 1,
          mieCoefficient: 0.005,
          mieDirectionalG: 0.8,
          intensity: 1,
        },
      ),
    );
    expect(features).toEqual({
      basicMaterials: true,
      scalarRichMaterials: false,
      mappedPbrMaterials: false,
      mappedRichMaterials: false,
      analyticLights: true,
      meshLights: true,
      environmentLight: true,
      fog: false,
    });
  });

  it('keeps a null scene on the conservative full-program superset', () => {
    expect(deriveSceneTraceFeatures(null)).toEqual({
      basicMaterials: false,
      scalarRichMaterials: false,
      mappedPbrMaterials: false,
      mappedRichMaterials: true,
      analyticLights: true,
      meshLights: true,
      environmentLight: true,
      fog: false,
    });
  });
});
