/**
 * H33 unit tests — materialSig Beer-Lambert fields
 *
 * Verifies that attenuationColor / attenuationDistance / thickness are
 * included in the dedup signature so transmissive materials differing only
 * in Beer tint/depth are not collapsed into one LUT slot.
 */

import { describe, expect, it } from 'vitest';
import type { MaterialSpec, Scene } from '@vitrum/core';
import { SceneBvh } from '../sceneBvh.js';
import { materialSig, mergeWorldSpaceFromCore } from '../worldSpaceMerge.js';

const BASE: MaterialSpec = {
  baseColor: [0.8, 0.8, 0.8],
  roughness: 0.1,
  metallic: 0,
  transmission: 1,
  ior: 1.5,
};

describe('materialSig — Beer-Lambert fields (H33)', () => {
  it('two materials differing only in attenuationColor produce different signatures', () => {
    const a: MaterialSpec = { ...BASE, attenuationColor: [1, 0, 0] };
    const b: MaterialSpec = { ...BASE, attenuationColor: [0, 0, 1] };
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('two materials differing only in attenuationDistance produce different signatures', () => {
    const a: MaterialSpec = { ...BASE, attenuationColor: [0.5, 0.5, 0.5], attenuationDistance: 1.0 };
    const b: MaterialSpec = { ...BASE, attenuationColor: [0.5, 0.5, 0.5], attenuationDistance: 10.0 };
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('two materials differing only in thickness produce different signatures', () => {
    const a: MaterialSpec = { ...BASE, thickness: 0.5 };
    const b: MaterialSpec = { ...BASE, thickness: 2.0 };
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('two identical transmissive materials (all Beer fields equal) produce the same signature', () => {
    const a: MaterialSpec = {
      ...BASE,
      attenuationColor: [0.9, 0.1, 0.2],
      attenuationDistance: 3.14,
      thickness: 0.25,
    };
    const b: MaterialSpec = {
      ...BASE,
      attenuationColor: [0.9, 0.1, 0.2],
      attenuationDistance: 3.14,
      thickness: 0.25,
    };
    expect(materialSig(a)).toBe(materialSig(b));
  });

  it('Infinity attenuationDistance normalises to "Inf" (stable token, no JSON-null trap)', () => {
    const infDist: MaterialSpec = { ...BASE, attenuationColor: [1, 1, 1], attenuationDistance: Infinity };
    const absent: MaterialSpec = { ...BASE };
    // Both should map to the same "Inf" token
    const sigInf = materialSig(infDist);
    const sigAbsent = materialSig(absent);
    // Absent defaults to "Inf" as well → same sig
    expect(sigInf).toBe(sigAbsent);
    // Finite distance must be different
    const sigFinite: MaterialSpec = { ...BASE, attenuationColor: [1, 1, 1], attenuationDistance: 5.0 };
    expect(materialSig(sigFinite)).not.toBe(sigInf);
  });

  it('attenuationColor absent defaults to 1,1,1 token (same as explicit [1,1,1])', () => {
    const implicit: MaterialSpec = { ...BASE };
    const explicit: MaterialSpec = { ...BASE, attenuationColor: [1, 1, 1] };
    expect(materialSig(implicit)).toBe(materialSig(explicit));
  });

  it('includes the folded mesh-emitter shadow flag in the material signature', () => {
    const caster: MaterialSpec = { ...BASE, emissive: [1, 1, 1], emissiveIntensity: 2 };
    const shadowless = {
      ...caster,
      meshEmitterCastShadowDisabled: true,
    } as MaterialSpec & { meshEmitterCastShadowDisabled: true };
    expect(materialSig(caster)).not.toBe(materialSig(shadowless));
  });

  it('includes alpha/coverage controls that affect atlas traversal metadata', () => {
    const opaque: MaterialSpec = { ...BASE, alphaMode: 'opaque', opacity: 1, alphaCutoff: 0.5 };
    const masked: MaterialSpec = { ...BASE, alphaMode: 'mask', opacity: 0.75, alphaCutoff: 0.9 };
    expect(materialSig(opaque)).not.toBe(materialSig(masked));
  });

  it('includes every packed texture-map handle identity', () => {
    const handleA = { name: 'alpha-a' };
    const handleB = { name: 'alpha-b' };
    const a: MaterialSpec = { ...BASE, alphaMap: { handle: handleA } };
    const b: MaterialSpec = { ...BASE, alphaMap: { handle: handleB } };
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('includes texture UV channel, transform, wrap, and filter metadata', () => {
    const handle = { name: 'roughness' };
    const a: MaterialSpec = {
      ...BASE,
      roughnessMap: {
        handle,
        texCoord: 0,
        transform: { offset: [0, 0], scale: [1, 1], rotation: 0 },
        wrapS: 'repeat',
        wrapT: 'repeat',
      },
    };
    const b: MaterialSpec = {
      ...BASE,
      roughnessMap: {
        handle,
        texCoord: 1,
        transform: { offset: [0.25, 0], scale: [0.5, 1], rotation: 0.5 },
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
        magFilter: 'nearest',
        minFilter: 'linear',
        mipFilter: 'nearest',
      },
    };
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('does not round away Float32-visible texture transform differences', () => {
    const handle = { name: 'normal' };
    const a: MaterialSpec = {
      ...BASE,
      normalMap: {
        handle,
        transform: { offset: [0.00001, 0], scale: [1, 1], rotation: 0.00001 },
      },
    };
    const b: MaterialSpec = {
      ...BASE,
      normalMap: {
        handle,
        transform: { offset: [0.00002, 0], scale: [1, 1], rotation: 0.00002 },
      },
    };
    expect(Math.fround(0.00001)).not.toBe(Math.fround(0.00002));
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('does not round away Float32-visible atlas scalar metadata differences', () => {
    const a: MaterialSpec = { ...BASE, normalScale: 1.00001, envMapIntensity: 0.50001 };
    const b: MaterialSpec = { ...BASE, normalScale: 1.00002, envMapIntensity: 0.50002 };
    expect(Math.fround(a.normalScale!)).not.toBe(Math.fround(b.normalScale!));
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('signs bare texture-object values the same way the walkaround atlas compatibility shim consumes them', () => {
    const handleA = { name: 'bare-alpha-a', width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) };
    const handleB = { name: 'bare-alpha-b', width: 1, height: 1, data: new Uint8Array([0, 255, 0, 255]) };
    const a = { ...BASE, alphaMap: handleA } as unknown as MaterialSpec;
    const b = { ...BASE, alphaMap: handleB } as unknown as MaterialSpec;
    expect(materialSig(a)).not.toBe(materialSig(b));
    expect(materialSig(a)).not.toBe(materialSig(BASE));
  });

  it('keeps unsupported and non-finite texture texCoord values distinct in the dedup key', () => {
    const handle = { name: 'uv-edge' };
    const unsupported: MaterialSpec = { ...BASE, lightMap: { handle, texCoord: 2 } };
    const nanTexCoord: MaterialSpec = { ...BASE, lightMap: { handle, texCoord: NaN } };
    const defaultUv: MaterialSpec = { ...BASE, lightMap: { handle, texCoord: 0 } };
    expect(materialSig(unsupported)).not.toBe(materialSig(defaultUv));
    expect(materialSig(nanTexCoord)).not.toBe(materialSig(defaultUv));
    expect(materialSig(nanTexCoord)).not.toBe(materialSig(unsupported));
  });

  it('includes extension lobe scalar controls packed by renderer material payloads', () => {
    const lowClearcoat: MaterialSpec = { ...BASE, clearcoat: 0.1, clearcoatRoughness: 0.2 };
    const highClearcoat: MaterialSpec = { ...BASE, clearcoat: 0.8, clearcoatRoughness: 0.2 };
    expect(materialSig(lowClearcoat)).not.toBe(materialSig(highClearcoat));
  });
});

describe('mergeWorldSpaceFromCore material slots', () => {
  const TRI_POS = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const TRI_NORM = new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]);

  function sceneWithMixedCastShadow(): Scene {
    return {
      primitives: [
        {
          kind: 'mesh',
          id: 'caster',
          positions: TRI_POS,
          normals: TRI_NORM,
          material: BASE,
          castShadow: true,
        },
        {
          kind: 'mesh',
          id: 'non-caster',
          positions: TRI_POS,
          normals: TRI_NORM,
          material: BASE,
          castShadow: false,
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
  }

  function sceneWithMaterials(a: MaterialSpec, b: MaterialSpec): Scene {
    return {
      primitives: [
        {
          kind: 'mesh',
          id: 'mat-a',
          positions: TRI_POS,
          normals: TRI_NORM,
          material: a,
        },
        {
          kind: 'mesh',
          id: 'mat-b',
          positions: TRI_POS,
          normals: TRI_NORM,
          material: b,
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
  }

  it('keeps historical material dedup unless cast-shadow splitting is requested', () => {
    const merged = mergeWorldSpaceFromCore(sceneWithMixedCastShadow());
    expect(merged.materials.length).toBe(1);
    expect(Array.from(merged.mergedTriMaterialId)).toEqual([0, 0]);
  });

  it('can split otherwise-identical materials by primitive castShadow', () => {
    const merged = mergeWorldSpaceFromCore(sceneWithMixedCastShadow(), {
      splitMaterialsByCastShadow: true,
    });
    expect(merged.materials.length).toBe(2);
    expect(Array.from(merged.mergedTriMaterialId)).toEqual([0, 1]);
    expect((merged.materials[0] as MaterialSpec & { castShadow?: boolean }).castShadow).toBe(true);
    expect((merged.materials[1] as MaterialSpec & { castShadow?: boolean }).castShadow).toBe(false);
  });

  it('SceneBvh preserves primitive castShadow slots for standalone DDGI material packing', () => {
    const bvh = new SceneBvh();
    bvh.updateFromCore(sceneWithMixedCastShadow());
    expect(bvh.buffers?.materials.length).toBe(2);
    expect(Array.from(bvh.buffers?.triMaterialId ?? [])).toEqual([0, 1]);
    expect((bvh.buffers?.materials[0] as MaterialSpec & { castShadow?: boolean }).castShadow).toBe(true);
    expect((bvh.buffers?.materials[1] as MaterialSpec & { castShadow?: boolean }).castShadow).toBe(false);
  });

  it('does not collapse materials that differ only by atlas-consumed texture metadata', () => {
    const handle = { name: 'shared-lightmap' };
    const scene = sceneWithMaterials(
      {
        ...BASE,
        lightMap: { handle, texCoord: 0 },
      },
      {
        ...BASE,
        lightMap: {
          handle,
          texCoord: 1,
          transform: { offset: [0.5, 0], scale: [0.5, 0.5], rotation: 0.25 },
          wrapS: 'clamp-to-edge',
          wrapT: 'mirrored-repeat',
        },
      },
    );
    const merged = mergeWorldSpaceFromCore(scene);
    expect(merged.materials.length).toBe(2);
    expect(Array.from(merged.mergedTriMaterialId)).toEqual([0, 1]);
  });

  it('does not collapse materials that differ only by non-base packed map fields', () => {
    const scene = sceneWithMaterials(
      { ...BASE, clearcoatMap: { handle: { name: 'coat-a' } } },
      { ...BASE, clearcoatMap: { handle: { name: 'coat-b' } } },
    );
    const merged = mergeWorldSpaceFromCore(scene);
    expect(merged.materials.length).toBe(2);
    expect(Array.from(merged.mergedTriMaterialId)).toEqual([0, 1]);
  });
});
