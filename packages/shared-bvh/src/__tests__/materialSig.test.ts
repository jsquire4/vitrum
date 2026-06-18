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
});
