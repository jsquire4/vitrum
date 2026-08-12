import { describe, expect, it } from 'vitest';
import type { MaterialSpec, Scene } from '@vitrum/core';
import { pickBackend, recommendBackendForSceneMaterials, shouldAttemptProgressiveViewer } from '../createEngineScale.js';

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

  it('keeps walkaround for approximate plain Scene volume-scattering fields', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.35,
      metallic: 0,
      scatteringCoefficient: 0.12,
    }), true);

    expect(recommendation).toBeNull();
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
    ['thinFilmStack', {
      thinFilmStack: {
        layers: [{ ior: 1.45, thicknessNm: 120 }],
      },
    }],
  ] as const)('keeps walkaround for bounded approximate plain Scene %s support', (_field, materialPatch) => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      ...materialPatch,
    }), true);

    expect(recommendation).toBeNull();
    expect(pickBackend('auto', true, 12, false, undefined, recommendation?.backend)).toBe('pt-webgpu');
  });

  it('does not create a material recommendation for approximate face-layer fields on WebGL hosts', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      frontLayer: { transmission: [0.85, 0.9, 1], roughness: 0.25 },
    }), false);

    expect(recommendation).toBeNull();
    expect(pickBackend('auto', false, 12, false, undefined, recommendation?.backend)).toBe('pt-webgl2');
  });

  it('does not reroute multiple fields that walkaround implements approximately', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.4,
      metallic: 0,
      frontLayer: { transmission: [0.85, 0.9, 1], roughness: 0.25 },
      scatteringCoefficient: 0.12,
    }), true);

    expect(recommendation).toBeNull();
  });

  it('does not reroute for material fields walkaround already consumes approximately', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      displacementScale: 0.1,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0.1, 0.2, 0.3]),
      },
      dispersionAbbeNumber: 55,
      thinFilmStack: {
        layers: [{ ior: 1.45, thicknessNm: 120 }],
      },
    }), true);

    expect(recommendation).toBeNull();
    expect(pickBackend('auto', true, 12, false, undefined, recommendation?.backend)).toBe('pt-webgpu');
  });

  it('keeps explicit realtime for an approximately supported plain Scene material', () => {
    const recommendation = recommendBackendForSceneMaterials(sceneWithMaterial({
      baseColor: [1, 1, 1],
      roughness: 0.35,
      metallic: 0,
      scatteringCoefficient: 0.12,
    }), true);

    expect(recommendation).toBeNull();
    expect(pickBackend('realtime', true, 12, false, undefined, recommendation?.backend)).toBe('walkaround-hybrid');
  });

  it('attempts the progressive viewer for auto on WebGPU unless walkaround cannot accept the scene', () => {
    expect(shouldAttemptProgressiveViewer('auto', true)).toBe(true);
    expect(shouldAttemptProgressiveViewer('auto', true, 'walkaround-hybrid')).toBe(true);
    expect(shouldAttemptProgressiveViewer('auto', true, 'pt-webgpu')).toBe(true);
    expect(shouldAttemptProgressiveViewer('auto', true, 'pt-webgl2')).toBe(false);
    expect(shouldAttemptProgressiveViewer('auto', true, undefined, 'pt-webgpu')).toBe(false);
    expect(shouldAttemptProgressiveViewer('auto', false)).toBe(false);
    expect(shouldAttemptProgressiveViewer('realtime', true)).toBe(false);
    expect(shouldAttemptProgressiveViewer('quality-webgpu', true)).toBe(false);
  });
});
