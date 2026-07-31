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

function f32FromBits(bits: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, bits >>> 0, false);
  return view.getFloat32(0, false);
}

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

  it('distinguishes adjacent float32 attenuation distances exactly', () => {
    const lower = f32FromBits(0x3f80_0000);
    const upper = f32FromBits(0x3f80_0001);
    expect(upper).toBeGreaterThan(lower);
    expect(materialSig({ ...BASE, attenuationDistance: lower }))
      .not.toBe(materialSig({ ...BASE, attenuationDistance: upper }));
  });

  it('uses exact float32 tokens inside nested layer and spectral records', () => {
    const lower = f32FromBits(0x3f00_0000);
    const upper = f32FromBits(0x3f00_0001);
    const layerA: MaterialSpec = {
      ...BASE,
      frontLayer: { transmission: [lower, 1, 1] },
    };
    const layerB: MaterialSpec = {
      ...BASE,
      frontLayer: { transmission: [upper, 1, 1] },
    };
    const spectralA: MaterialSpec = {
      ...BASE,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([lower, 1, 2]),
      },
    };
    const spectralB: MaterialSpec = {
      ...BASE,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([upper, 1, 2]),
      },
    };
    expect(materialSig(layerA)).not.toBe(materialSig(layerB));
    expect(materialSig(spectralA)).not.toBe(materialSig(spectralB));
  });

  it('preserves signed-zero bit identity for packed numeric fields', () => {
    const positiveZero: MaterialSpec = { ...BASE, anisotropyRotation: 0 };
    const negativeZero: MaterialSpec = { ...BASE, anisotropyRotation: -0 };
    expect(Object.is(positiveZero.anisotropyRotation, negativeZero.anisotropyRotation)).toBe(false);
    expect(materialSig(positiveZero)).not.toBe(materialSig(negativeZero));
  });

  it('canonicalises NaN and keeps every non-finite class distinct from defaults', () => {
    const withRotation = (anisotropyRotation: number): MaterialSpec => ({
      ...BASE,
      anisotropyRotation,
    });
    expect(materialSig(withRotation(Number.NaN)))
      .toBe(materialSig(withRotation(Number.NaN)));
    expect(new Set([
      materialSig(BASE),
      materialSig(withRotation(Number.NaN)),
      materialSig(withRotation(Number.POSITIVE_INFINITY)),
      materialSig(withRotation(Number.NEGATIVE_INFINITY)),
    ]).size).toBe(4);
  });

  it('attenuationColor absent defaults to 1,1,1 token (same as explicit [1,1,1])', () => {
    const implicit: MaterialSpec = { ...BASE };
    const explicit: MaterialSpec = { ...BASE, attenuationColor: [1, 1, 1] };
    expect(materialSig(implicit)).toBe(materialSig(explicit));
  });

  it('makes omitted optional fields equivalent to their explicit contract defaults', () => {
    const explicitDefaults: MaterialSpec = {
      ...BASE,
      emissive: [0, 0, 0],
      emissiveIntensity: 1,
      shadingModel: 'pbr',
      alphaMode: 'opaque',
      alphaCutoff: 0.5,
      opacity: 1,
      doubleSided: false,
      attenuationColor: [1, 1, 1],
      attenuationDistance: Number.POSITIVE_INFINITY,
      thickness: 0,
      normalScale: 1,
      clearcoatNormalScale: 1,
      aoMapIntensity: 1,
      bumpScale: 1,
      lightMapIntensity: 1,
      envMapIntensity: 1,
      specularColor: [1, 1, 1],
      specularIntensity: 1,
      clearcoat: 0,
      clearcoatRoughness: 0,
      sheen: 0,
      sheenColor: [0, 0, 0],
      sheenRoughness: 0,
      anisotropy: 0,
      anisotropyRotation: 0,
      iridescence: 0,
      iridescenceIor: 1.3,
      iridescenceThicknessRange: [100, 400],
      displacementScale: 1,
      displacementBias: 0,
      displacementSubdivisions: 0,
      scatteringCoefficient: 0,
      scatteringAnisotropy: 0,
      scatteringCoefficientRGB: [0, 0, 0],
      dispersionAbbeNumber: 0,
    };
    expect(materialSig(BASE)).toBe(materialSig(explicitDefaults));
  });

  it('includes doubleSided because it is a canonical MaterialEntry flag', () => {
    expect(materialSig(BASE)).not.toBe(materialSig({ ...BASE, doubleSided: true }));
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

  it('includes displacement subdivision level because it changes generated geometry', () => {
    const vertexOnly: MaterialSpec = { ...BASE, displacementSubdivisions: 0 };
    const diced: MaterialSpec = { ...BASE, displacementSubdivisions: 2 };
    expect(materialSig(vertexOnly)).not.toBe(materialSig(diced));
  });

  it('includes every packed texture-map handle identity', () => {
    const handleA = { name: 'alpha-a' };
    const handleB = { name: 'alpha-b' };
    const a: MaterialSpec = { ...BASE, alphaMap: { handle: handleA } };
    const b: MaterialSpec = { ...BASE, alphaMap: { handle: handleB } };
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('includes face-layer normal-map handle and sampler identity', () => {
    const sharedTransmission = [0.8, 0.9, 1] as const;
    const a: MaterialSpec = {
      ...BASE,
      frontLayer: {
        transmission: sharedTransmission,
        normalMap: {
          handle: { name: 'front-a' },
          wrapS: 'repeat',
        },
      },
    };
    const b: MaterialSpec = {
      ...BASE,
      frontLayer: {
        transmission: sharedTransmission,
        normalMap: {
          handle: { name: 'front-b' },
          wrapS: 'clamp-to-edge',
        },
      },
    };
    expect(materialSig(a)).not.toBe(materialSig(b));
  });

  it('includes the behavior-affecting material extension lanes', () => {
    expect(materialSig(BASE)).not.toBe(materialSig({
      ...BASE,
      extensions: { skipEmitter: true },
    }));
    expect(materialSig(BASE)).not.toBe(materialSig({
      ...BASE,
      extensions: { surfaceTextureId: 7 },
    }));
    // Invalid/absent surface ids have the same effective smooth-surface lane.
    expect(materialSig(BASE)).toBe(materialSig({
      ...BASE,
      extensions: { surfaceTextureId: 99 },
    }));
  });

  it('keeps primitive opaque handles type-safe, delimiter-safe, and symbol-identity-safe', () => {
    const withHandle = (handle: unknown): MaterialSpec => ({
      ...BASE,
      alphaMap: { handle },
    });
    expect(materialSig(withHandle(1))).not.toBe(materialSig(withHandle('1')));
    expect(materialSig(withHandle(true))).not.toBe(materialSig(withHandle('true')));
    expect(materialSig(withHandle('a;uv0|maps=x')))
      .not.toBe(materialSig(withHandle('a')));
    const symbolA = Symbol('same-description');
    const symbolB = Symbol('same-description');
    expect(materialSig(withHandle(symbolA))).toBe(materialSig(withHandle(symbolA)));
    expect(materialSig(withHandle(symbolA))).not.toBe(materialSig(withHandle(symbolB)));
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
          uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
          uv1: new Float32Array([0, 0, 1, 0, 0, 1]),
          material: a,
        },
        {
          kind: 'mesh',
          id: 'mat-b',
          positions: TRI_POS,
          normals: TRI_NORM,
          uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
          uv1: new Float32Array([0, 0, 1, 0, 0, 1]),
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
    expect(bvh.buffers?.coreMaterials).toBe(bvh.buffers?.materials);
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
