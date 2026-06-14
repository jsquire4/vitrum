/**
 * Tests for the material-field consumption warning (Wave 2, §3 item 1).
 *
 * Pins:
 *   (a) A scene supplying baseColorMap + normalMap + clearcoat warns only for
 *       the fields without a walkaround texture path.
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
      shadingModel: 'unlit',
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

  it('names normalMap when supplied', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      normalMap: stubTextureRef(),
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    expect(result).toContain('normalMap');
  });

  it('names clearcoat when supplied', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      clearcoat: 1.0,
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    expect(result).toContain('clearcoat');
  });

  it('(pin a) names normalMap + clearcoat while baseColorMap is consumed', () => {
    const mat: Record<string, unknown> = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle: stubTextureRef() },
      normalMap: stubTextureRef(),
      clearcoat: 1.0,
    };
    const result = collectUnconsumedMaterialFields(primitivesWithMaterial(mat));
    // Result is sorted alphabetically.
    expect(result).toEqual(['clearcoat', 'normalMap']);
  });

  it('dedupes unconsumed fields across multiple primitives', () => {
    const prims: PrimLike[] = [
      { kind: 'mesh', material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, clearcoat: 1 } },
      { kind: 'mesh', material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, clearcoat: 0.5 } },
    ];
    const result = collectUnconsumedMaterialFields(prims);
    // clearcoat should appear exactly once.
    expect(result.filter((f) => f === 'clearcoat')).toHaveLength(1);
  });

  it('skips non-mesh primitives', () => {
    const prims: PrimLike[] = [
      { kind: 'analytic', material: { baseColor: [1, 1, 1], roughness: 0, metallic: 0, clearcoat: 1 } },
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

  it('includes skinned-mesh and instanced-mesh primitives', () => {
    const kinds = ['skinned-mesh', 'instanced-mesh'] as const;
    for (const kind of kinds) {
      const prims: PrimLike[] = [
        { kind, material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, sheen: 0.5 } },
      ];
      expect(collectUnconsumedMaterialFields(prims)).toContain('sheen');
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
      'thickness', 'ior', 'extensions', 'baseColorMap', 'roughnessMap', 'metallicMap',
    ]) {
      expect(CONSUMED_MATERIAL_FIELDS.has(field)).toBe(true);
    }
  });

  it('does NOT include unsupported texture-map fields', () => {
    const textureMaps = [
      'normalMap',
      'transmissionMap', 'emissiveMap', 'thicknessMap', 'alphaMap', 'aoMap',
      'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap',
      'sheenColorMap', 'sheenRoughnessMap', 'iridescenceMap',
      'iridescenceThicknessMap', 'anisotropyMap', 'specularColorMap',
      'specularIntensityMap', 'bumpMap', 'displacementMap', 'lightMap',
    ];
    for (const field of textureMaps) {
      expect(CONSUMED_MATERIAL_FIELDS.has(field)).toBe(false);
    }
  });
});
