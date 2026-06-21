import { describe, expect, it } from 'vitest';
import type { MaterialSpec, Scene } from '@vitrum/core';
import { pickBackend, recommendBackendForSceneMaterials } from '../createEngineScale.js';

function sceneWithMaterial(material: MaterialSpec): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material,
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('createEngine backend selection', () => {
  it('uses a glTF recommended backend when prefer is auto', () => {
    expect(pickBackend('auto', true, 12, false, 'pt-webgpu')).toBe('pt-webgpu');
    expect(pickBackend('auto', true, 1_000_000, false, 'walkaround-hybrid')).toBe('walkaround-hybrid');
    expect(pickBackend('auto', true, 12, false, 'pt-webgl2')).toBe('pt-webgl2');
  });

  it('keeps explicit preference stronger than the glTF recommendation', () => {
    expect(pickBackend('quality', true, 12, false, 'walkaround-hybrid')).toBe('pt-webgl2');
    expect(pickBackend('realtime', true, 1_000_000, false, 'pt-webgpu')).toBe('walkaround-hybrid');
  });

  it('falls back to pt-webgl2 for WebGPU recommendations on WebGL-only hosts', () => {
    expect(pickBackend('auto', false, 12, false, 'pt-webgpu')).toBe('pt-webgl2');
    expect(pickBackend('auto', false, 12, false, 'walkaround-hybrid')).toBe('pt-webgl2');
  });

  it('recommends a PT backend for plain Scene material fields unsupported by walkaround', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.35,
      metallic: 0,
      thinFilmStack: {
        layers: [{ ior: 1.45, thicknessNm: 120 }],
      },
    }), true);

    expect(recommendation).toEqual({
      backend: 'pt-webgpu',
      fields: ['thinFilmStack'],
    });
    expect(pickBackend('auto', true, 12, false, undefined, recommendation?.backend)).toBe('pt-webgpu');
  });

  it.each([
    ['spectralAttenuation', {
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0.1, 0.2, 0.3]),
      },
    }],
    ['dispersionAbbeNumber', { dispersionAbbeNumber: 55 }],
    ['scatteringCoefficient', { scatteringCoefficient: 0.12 }],
    ['scatteringAnisotropy', { scatteringAnisotropy: 0.35 }],
    ['scatteringCoefficientRGB', { scatteringCoefficientRGB: [0.1, 0.2, 0.3] }],
    ['frontLayer', { frontLayer: { transmission: [0.85, 0.9, 1] } }],
    ['backLayer', { backLayer: { transmission: [1, 0.9, 0.85] } }],
    ['thinFilmStack', {
      thinFilmStack: {
        layers: [{ ior: 1.45, thicknessNm: 120 }],
      },
    }],
  ] as const)('recommends PT for plain Scene %s when auto-routing', (_field, materialPatch) => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      ...materialPatch,
    }), true);

    expect(recommendation).toEqual({
      backend: 'pt-webgpu',
      fields: [_field],
    });
    expect(pickBackend('auto', true, 12, false, undefined, recommendation?.backend)).toBe('pt-webgpu');
  });

  it('falls back to pt-webgl2 for plain Scene PT-only material fields on WebGL hosts', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      thinFilmStack: {
        layers: [{ ior: 1.45, thicknessNm: 120 }],
      },
    }), false);

    expect(recommendation).toEqual({
      backend: 'pt-webgl2',
      fields: ['thinFilmStack'],
    });
    expect(pickBackend('auto', false, 12, false, undefined, recommendation?.backend)).toBe('pt-webgl2');
  });

  it('sorts multiple PT-only material fields for deterministic warnings', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      frontLayer: { transmission: [0.85, 0.9, 1] },
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0.1, 0.2, 0.3]),
      },
      thinFilmStack: {
        layers: [{ ior: 1.45, thicknessNm: 120 }],
      },
    }), true);

    expect(recommendation).toEqual({
      backend: 'pt-webgpu',
      fields: ['frontLayer', 'spectralAttenuation', 'thinFilmStack'],
    });
  });

  it('does not reroute for material fields walkaround already consumes approximately', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      displacementScale: 0.1,
    }), true);

    expect(recommendation).toBeNull();
    expect(pickBackend('auto', true, 12, false, undefined, recommendation?.backend)).toBe('walkaround-hybrid');
  });

  it('keeps explicit realtime stronger than a plain Scene material recommendation', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.35,
      metallic: 0,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0.1, 0.2, 0.3]),
      },
    }), true);

    expect(recommendation?.backend).toBe('pt-webgpu');
    expect(pickBackend('realtime', true, 12, false, undefined, recommendation?.backend)).toBe('walkaround-hybrid');
  });
});
