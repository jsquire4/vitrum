// gltfTextureSweep.test.ts — GLTF-06 per-extension texture-map fixture sweep.
//
// Proves that EVERY texture map the adapter imports (base PBR + all KHR
// material extensions routed through resolveTextureRef) preserves:
//   - the decoded handle,
//   - the effective UV set (`TextureRef.texCoord`), including the
//     KHR_texture_transform-level `texCoord` OVERRIDE (757477d4 fix), and
//   - the KHR_texture_transform offset / scale / rotation fields,
// end-to-end through gltfToScene() into the core MaterialSpec.
//
// The imported map list is enumerated from materials.ts/textures.ts call
// sites; if a new resolveTextureRef consumer is added, extend SWEEP_MAPS.
//
// Also pins KHR_materials_volume.thicknessTexture now that core carries it as
// the reserved `thicknessMap` material field.

import { describe, it, expect } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import { loadGltfAsset } from './index.js';
import type { GltfJson, GltfTextureInfo } from './gltfTypes.js';
import { gltfTextureColorSpaceForField, type GltfMaterialTextureField } from './texturePipeline.js';
import type { MaterialSpec, MeshPrimitive, TextureRef } from '@vitrum/core';

// ── Fixture helpers ──────────────────────────────────────────────────────────

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

const TRIANGLE_POSITIONS = [0, 0, 0, 1, 0, 0, 0, 1, 0];
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47];

/** Per-ordinal distinct transform so a wrong-slot mapping cannot pass. */
function expectedTransform(i: number): { offset: [number, number]; scale: [number, number]; rotation: number } {
  return {
    offset: [0.01 * (i + 1), 0.02 * (i + 1)],
    scale: [1 + 0.1 * (i + 1), 2 + 0.1 * (i + 1)],
    rotation: 0.001 * (i + 1),
  };
}

function expectedWrap(i: number): { wrapS: 'repeat' | 'clamp-to-edge' | 'mirrored-repeat'; wrapT: 'repeat' | 'clamp-to-edge' | 'mirrored-repeat' } {
  const modes = ['repeat', 'clamp-to-edge', 'mirrored-repeat'] as const;
  return {
    wrapS: modes[i % modes.length]!,
    wrapT: modes[(i + 2) % modes.length]!,
  };
}

function samplerForOrdinal(i: number): { wrapS?: number; wrapT?: number } {
  const { wrapS, wrapT } = expectedWrap(i);
  const code = (mode: 'repeat' | 'clamp-to-edge' | 'mirrored-repeat'): number | undefined =>
    mode === 'repeat' ? undefined : mode === 'clamp-to-edge' ? 33071 : 33648;
  const sampler: { wrapS?: number; wrapT?: number } = {};
  const wrapSCode = code(wrapS);
  const wrapTCode = code(wrapT);
  if (wrapSCode !== undefined) sampler.wrapS = wrapSCode;
  if (wrapTCode !== undefined) sampler.wrapT = wrapTCode;
  return sampler;
}

/** Texture info for ordinal `i`: info-level texCoord 0, transform-level
 *  texCoord OVERRIDE 1 + distinct offset/scale/rotation. */
function texInfo(i: number): GltfTextureInfo {
  const t = expectedTransform(i);
  return {
    index: i,
    texCoord: 0,
    extensions: {
      KHR_texture_transform: {
        texCoord: 1, // override must win (757477d4)
        offset: t.offset,
        scale: t.scale,
        rotation: t.rotation,
      },
    },
  };
}

/** [core MaterialSpec map field, texture ordinal in the fixture] */
const SWEEP_MAPS: ReadonlyArray<readonly [keyof MaterialSpec, number]> = [
  ['baseColorMap', 0],
  ['roughnessMap', 1],            // combined metallic-roughness …
  ['metallicMap', 1],             // … maps to BOTH (WEBGL2-04 closure)
  ['normalMap', 2],
  ['aoMap', 3],
  ['emissiveMap', 4],
  ['transmissionMap', 5],         // KHR_materials_transmission
  ['specularIntensityMap', 6],    // KHR_materials_specular
  ['specularColorMap', 7],
  ['sheenColorMap', 8],           // KHR_materials_sheen
  ['sheenRoughnessMap', 9],
  ['clearcoatMap', 10],           // KHR_materials_clearcoat
  ['clearcoatRoughnessMap', 11],
  ['clearcoatNormalMap', 12],
  ['iridescenceMap', 13],         // KHR_materials_iridescence
  ['iridescenceThicknessMap', 14],
  ['anisotropyMap', 15],          // KHR_materials_anisotropy
];

const TEXTURE_COUNT = 16;

const SRGB_SWEEP_FIELDS = new Set<keyof MaterialSpec>([
  'baseColorMap',
  'emissiveMap',
  'specularColorMap',
  'sheenColorMap',
]);

function makeSweepGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const posBuf = f32Buffer(TRIANGLE_POSITIONS);
  const imageBytes = new Uint8Array(PNG_MAGIC);
  const total = new Uint8Array(posBuf.byteLength + imageBytes.length);
  total.set(new Uint8Array(posBuf), 0);
  total.set(imageBytes, posBuf.byteLength);

  const gltf: GltfJson = {
    asset: { version: '2.0' },
    scenes: [{ nodes: [0] }],
    scene: 0,
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    materials: [{
      pbrMetallicRoughness: {
        baseColorTexture: texInfo(0),
        metallicRoughnessTexture: texInfo(1),
      },
      normalTexture: { ...texInfo(2), scale: 0.5 },
      occlusionTexture: { ...texInfo(3), strength: 0.75 },
      emissiveFactor: [1, 1, 1],
      emissiveTexture: texInfo(4),
      extensions: {
        KHR_materials_transmission: {
          transmissionFactor: 0.9,
          transmissionTexture: texInfo(5),
        },
        KHR_materials_specular: {
          specularFactor: 0.5,
          specularTexture: texInfo(6),
          specularColorFactor: [0.9, 0.8, 0.7],
          specularColorTexture: texInfo(7),
        },
        KHR_materials_sheen: {
          sheenColorFactor: [0.5, 0.3, 0.1],
          sheenColorTexture: texInfo(8),
          sheenRoughnessFactor: 0.4,
          sheenRoughnessTexture: texInfo(9),
        },
        KHR_materials_clearcoat: {
          clearcoatFactor: 0.8,
          clearcoatTexture: texInfo(10),
          clearcoatRoughnessFactor: 0.1,
          clearcoatRoughnessTexture: texInfo(11),
          clearcoatNormalTexture: { ...texInfo(12), scale: 0.25 },
        },
        KHR_materials_iridescence: {
          iridescenceFactor: 0.7,
          iridescenceTexture: texInfo(13),
          iridescenceIor: 2.0,
          iridescenceThicknessMinimum: 200,
          iridescenceThicknessMaximum: 800,
          iridescenceThicknessTexture: texInfo(14),
        },
        KHR_materials_anisotropy: {
          anisotropyStrength: 0.6,
          anisotropyRotation: 1.0,
          anisotropyTexture: texInfo(15),
        },
      },
    }],
    textures: Array.from({ length: TEXTURE_COUNT }, (_, i) => ({ source: 0, sampler: i })),
    samplers: Array.from({ length: TEXTURE_COUNT }, (_, i) => samplerForOrdinal(i)),
    images: [{ bufferView: 1, mimeType: 'image/png' }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: posBuf.byteLength },
      { buffer: 0, byteOffset: posBuf.byteLength, byteLength: imageBytes.length },
    ],
    buffers: [{ byteLength: total.byteLength }],
  };
  return { gltf, buffers: new Map([[0, total.buffer]]) };
}

// ── The sweep ────────────────────────────────────────────────────────────────

describe('KHR extension texture sweep (GLTF-06)', () => {
  async function importSweepMaterial(): Promise<{ mat: MaterialSpec; handle: object }> {
    const handle = { kind: 'decoded-texture' };
    const { gltf, buffers } = makeSweepGltf();
    const { scene } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => handle,
    });
    return { mat: (scene.primitives[0] as MeshPrimitive).material, handle };
  }

  it.each(SWEEP_MAPS.map(([field, ordinal]) => ({ field, ordinal })))(
    '$field preserves handle + texCoord override + KHR_texture_transform',
    async ({ field, ordinal }) => {
      const { mat, handle } = await importSweepMaterial();
      const ref = mat[field] as TextureRef | undefined;
      expect(ref, `${String(field)} missing from imported material`).toBeDefined();
      expect(ref!.handle).toBe(handle);
      // Transform-level texCoord override (1) must win over info-level (0).
      expect(ref!.texCoord).toBe(1);
      const t = expectedTransform(ordinal);
      expect(ref!.transform?.offset).toEqual(t.offset);
      expect(ref!.transform?.scale).toEqual(t.scale);
      expect(ref!.transform?.rotation).toBeCloseTo(t.rotation, 10);
      const wrap = expectedWrap(ordinal);
      expect(ref!.wrapS ?? 'repeat').toBe(wrap.wrapS);
      expect(ref!.wrapT ?? 'repeat').toBe(wrap.wrapT);
    },
  );

  it('combined metallic-roughness texture is the SAME ref on roughnessMap and metallicMap', async () => {
    const { mat } = await importSweepMaterial();
    expect(mat.roughnessMap).toBeDefined();
    expect(mat.metallicMap).toBe(mat.roughnessMap);
  });

  it.each(SWEEP_MAPS.map(([field]) => ({ field })))(
    '$field has an explicit glTF texture color-space policy',
    ({ field }) => {
      const expected = SRGB_SWEEP_FIELDS.has(field) ? 'srgb' : 'linear';
      expect(gltfTextureColorSpaceForField(field as GltfMaterialTextureField)).toBe(expected);
    },
  );

  it('loadGltfAsset textureDecodeReport covers every imported sweep texture field', async () => {
    const handle = { kind: 'decoded-texture' };
    const { gltf, buffers } = makeSweepGltf();
    const result = await loadGltfAsset(gltf, {
      buffers,
      decodeImage: async () => handle,
    });

    expect(result.textureDecodeReport.mapCount).toBe(SWEEP_MAPS.length);
    for (const [field, ordinal] of SWEEP_MAPS) {
      const entry = result.textureDecodeReport.entries.find((candidate) =>
        candidate.materialField === field,
      );
      expect(entry, `${String(field)} missing from textureDecodeReport`).toBeDefined();
      const transform = expectedTransform(ordinal);
      const wrap = expectedWrap(ordinal);
      expect(entry).toMatchObject({
        primitiveId: 'gltf-prim-0',
        primitiveKind: 'mesh',
        primitiveIndex: 0,
        materialField: field,
        path: `scene.primitives[0].material.${String(field)}`,
        texCoord: 1,
        hasTransform: true,
        wrapS: wrap.wrapS,
        wrapT: wrap.wrapT,
        colorSpace: SRGB_SWEEP_FIELDS.has(field) ? 'srgb' : 'linear',
        handleKind: 'opaque',
        backendReadiness: {
          ptWebgl2: 'opaque',
          ptWebgpu: 'opaque',
          walkaroundHybrid:
            field === 'baseColorMap' || field === 'normalMap' || field === 'roughnessMap' || field === 'metallicMap' || field === 'aoMap' || field === 'alphaMap' || field === 'emissiveMap' || field === 'transmissionMap'
              ? 'opaque'
              : 'ignored',
        },
      });
      expect(transform.rotation).toBeGreaterThan(0);
    }
  });

  it('scalar companions still map (normalScale, aoMapIntensity, clearcoatNormalScale)', async () => {
    const { mat } = await importSweepMaterial();
    expect(mat.normalScale).toBeCloseTo(0.5);
    expect(mat.aoMapIntensity).toBeCloseTo(0.75);
    expect(mat.clearcoatNormalScale).toBeCloseTo(0.25);
  });

  it('KHR_materials_volume.thicknessTexture maps to thicknessMap', async () => {
    const { gltf, buffers } = makeSweepGltf();
    const mats = gltf.materials!;
    mats[0] = {
      ...mats[0]!,
      extensions: {
        ...mats[0]!.extensions,
        KHR_materials_volume: {
          thicknessFactor: 0.5,
          thicknessTexture: { index: 0 },
          attenuationDistance: 2.0,
        },
      },
    };
    const { scene, warnings } = await gltfToScene(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
    });
    const mat = (scene.primitives[0] as MeshPrimitive).material;
    expect(mat.thickness).toBeCloseTo(0.5);
    expect(mat.thicknessMap?.handle).toEqual({ kind: 'decoded-texture' });
    expect(warnings.some(w => w.includes('thicknessTexture'))).toBe(false);
  });
});
