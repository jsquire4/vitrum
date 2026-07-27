/**
 * Tests for the material-field consumption warning (Wave 2, §3 item 1).
 *
 * Pins:
 *   (a) A scene supplying baseColorMap + normalMap + frontLayer.normalMap does
 *       not warn because layer-local normal maps now have walkaround paths.
 *   (b) A scene using only consumed fields (baseColor, roughness, metallic,
 *       emissive, emissiveIntensity, shadingModel, transmission, attenuationColor,
 *       attenuationDistance, thickness, ior, extensions) produces no warning.
 */

import { describe, expect, it } from 'vitest';
import type { TextureRef } from '@vitrum/core';
import {
  collectUnconsumedMaterialFields,
  CONSUMED_MATERIAL_FIELDS,
} from '../src/restir/consumedMaterialFields.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** A minimal TextureRef stub (the exact texture payload doesn't matter here). */
function stubTextureRef(): TextureRef {
  // TextureRef is an opaque handle; the warning check only tests whether the
  // field is defined and non-null, not its internal shape.
  return { width: 1, height: 1, data: new Uint8Array(4) } as unknown as TextureRef;
}

type PrimLike = { kind: string; material?: Record<string, unknown> };

function primitivesWithMaterial(material: Record<string, unknown>): PrimLike[] {
  return [{ kind: 'mesh', material }];
}

// ---------------------------------------------------------------------------
// Unit tests: collectUnconsumedMaterialFields
// ---------------------------------------------------------------------------

describe('collectUnconsumedMaterialFields', () => {
  it('returns [] for a material using only consumed scalar fields', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      baseColorMap: { handle: stubTextureRef() },
      roughness: 0.5,
      metallic: 0,
      emissive: [1, 0, 0],
      emissiveIntensity: 2,
      lightMap: { handle: stubTextureRef() },
      lightMapIntensity: 1.5,
      shadingModel: 'unlit',
      normalMap: { handle: stubTextureRef() },
      normalScale: 0.5,
      bumpMap: { handle: stubTextureRef() },
      bumpScale: 0.25,
      transmission: 0.9,
      attenuationColor: [0.8, 0.9, 1],
      attenuationDistance: 1,
      thickness: 0.1,
      ior: 1.5,
      extensions: { surfaceTextureId: 0 },
    };
    expect(collectUnconsumedMaterialFields(primitivesWithMaterial(mat))).toEqual([]);
  });

  it('does not name baseColorMap when a TextureRef is supplied', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle: stubTextureRef() },
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    expect(result).not.toContain('baseColorMap');
  });

  it('does not name normalMap when supplied', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      normalMap: stubTextureRef(),
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    expect(result).not.toContain('normalMap');
  });

  it('does not name iridescence when supplied', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      iridescence: 1.0,
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    expect(result).not.toContain('iridescence');
  });

  it('(pin a) treats frontLayer.normalMap as consumed with baseColorMap + normalMap + iridescence', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle: stubTextureRef() },
      normalMap: stubTextureRef(),
      iridescence: 1.0,
      frontLayer: { transmission: [1, 0.5, 0.25], normalMap: { handle: stubTextureRef() } },
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    expect(result).toEqual([]);
  });

  it('does not warn for consumed frontLayer/backLayer transmission and roughness', () => {
    const prims: PrimLike[] = [
      { kind: 'mesh', material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, frontLayer: { transmission: [1, 1, 1] } } },
      { kind: 'mesh', material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, backLayer: { transmission: [0.5, 0.5, 0.5], roughness: 0.25 } } },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual([]);
  });

  it('consumes optical-stack fields on analytic primitives', () => {
    const prims: PrimLike[] = [
      { kind: 'analytic', material: { baseColor: [1, 1, 1], roughness: 0, metallic: 0, thinFilmStack: { layers: [] } } },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual([]);
  });

  it('does not count undefined or null fields as unconsumed', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      normalMap: undefined,  // explicitly undefined — should not warn
      clearcoatMap: null,    // null — should not warn
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    expect(result).not.toContain('normalMap');
    expect(result).not.toContain('clearcoatMap');
  });

  it('consumes optical-stack fields on skinned and instanced primitives', () => {
    const kinds = ['skinned-mesh', 'instanced-mesh'] as const;
    for (const kind of kinds) {
      const prims: PrimLike[] = [
        { kind, material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, thinFilmStack: { layers: [] } } },
      ];
      expect(collectUnconsumedMaterialFields(prims)).toEqual([]);
    }
  });
});

// ---------------------------------------------------------------------------
// Integration smoke: CONSUMED_MATERIAL_FIELDS completeness
// ---------------------------------------------------------------------------

describe('CONSUMED_MATERIAL_FIELDS', () => {
  it('contains the minimal expected consumed scalars', () => {
    for (const field of [
      'baseColor', 'roughness', 'metallic',
      'emissive', 'emissiveIntensity',
      'shadingModel', 'transmission', 'attenuationColor', 'attenuationDistance',
      'thickness', 'ior', 'extensions', 'clearcoat', 'clearcoatRoughness',
      'sheen', 'sheenColor', 'sheenRoughness',
      'specularColorMap', 'specularIntensityMap',
      'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap', 'clearcoatNormalScale',
      'sheenColorMap', 'sheenRoughnessMap',
      'anisotropy', 'anisotropyRotation', 'anisotropyMap',
      'frontLayer', 'backLayer',
      'spectralAttenuation', 'dispersionAbbeNumber', 'thinFilmStack',
      'scatteringCoefficient', 'scatteringAnisotropy', 'scatteringCoefficientRGB',
      'iridescence', 'iridescenceIor', 'iridescenceThicknessRange',
      'iridescenceMap', 'iridescenceThicknessMap',
      'baseColorMap', 'roughnessMap', 'metallicMap',
      'aoMap', 'aoMapIntensity', 'alphaMap', 'emissiveMap', 'transmissionMap',
      'thicknessMap',
      'displacementMap', 'displacementScale', 'displacementBias', 'displacementSubdivisions',
      'normalMap', 'normalScale', 'bumpMap', 'bumpScale', 'lightMap', 'lightMapIntensity',
    ]) {
      expect(CONSUMED_MATERIAL_FIELDS.has(field)).toBe(true);
    }
  });
});
