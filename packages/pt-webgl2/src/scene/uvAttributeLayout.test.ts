import { describe, expect, it } from 'vitest';
import type {
  InstancedMeshPrimitive,
  MaterialSpec,
  MeshPrimitive,
  Scene,
  SkinnedMeshPrimitive,
} from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '@vitrum/shared-bvh';
import { packAttributesArray } from './attributesTextureArray.js';
import { packMaterialsTexture } from './materialsTexture.js';
import { buildSceneGeometryTextureData } from './uploadSceneTextures.js';
import {
  MATERIAL_MAP_FIELD_ORDER,
  MATERIAL_UV_SELECTOR_TEXEL_OFFSET,
} from '../glsl/shader/structs/materialStride.js';
import {
  ATTR_LAYER_UV,
  ATTR_LAYER_UV1,
  buildUvAttributeLayout,
} from './uvAttributeLayout.js';

const IDENTITY = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

function material(overrides: Partial<MaterialSpec> = {}): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 0.5,
    metallic: 0,
    ...overrides,
  };
}

const POSITIONS = new Float32Array([
  0, 0, 0,
  1, 0, 0,
  0, 1, 0,
]);
const NORMALS = new Float32Array([
  0, 0, 1,
  0, 0, 1,
  0, 0, 1,
]);
const INDICES = new Uint32Array([0, 1, 2]);

function meshWithUvSets(mat: MaterialSpec): MeshPrimitive {
  const uv0 = new Float32Array([0, 0, 1, 0, 0, 1]);
  const uv1 = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  const uv2 = new Float32Array([0, 0, 1, 0, 0, 1]);
  return {
    kind: 'mesh',
    id: 'mesh',
    positions: POSITIONS,
    normals: NORMALS,
    uvs: uv0,
    uv1,
    uvSets: [uv0, uv1, uv2],
    indices: INDICES,
    material: mat,
  };
}

describe('pt-webgl2 scalable UV attribute layout', () => {
  it('packs UV0/UV1/UV2 in stable/dense layers and barycentrically interpolates UV2', () => {
    const h0 = {};
    const h1 = {};
    const h2 = {};
    const mat = material({
      baseColorMap: { handle: h0, texCoord: 0 },
      roughnessMap: { handle: h1, texCoord: 1 },
      emissiveMap: { handle: h2, texCoord: 2 },
    });
    const scene: Scene = {
      primitives: [meshWithUvSets(mat)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
    const layout = buildUvAttributeLayout(scene, merged, merged.materials);
    const attributes = packAttributesArray({
      ...merged,
      uv1: layout.mergedByTexCoord.get(1)!,
      extraUvLayers: layout.extraUvLayers,
    });

    expect(layout.layerByTexCoord.get(0)).toBe(ATTR_LAYER_UV);
    expect(layout.layerByTexCoord.get(1)).toBe(ATTR_LAYER_UV1);
    expect(layout.layerByTexCoord.get(2)).toBe(5);
    expect(attributes.layers).toBe(6);

    const floatsPerLayer = attributes.dim * attributes.dim * 4;
    const uv2Base = 5 * floatsPerLayer;
    const weights = [0.2, 0.3, 0.5] as const;
    const u = weights[0] * attributes.data[uv2Base]! +
      weights[1] * attributes.data[uv2Base + 4]! +
      weights[2] * attributes.data[uv2Base + 8]!;
    const v = weights[0] * attributes.data[uv2Base + 1]! +
      weights[1] * attributes.data[uv2Base + 5]! +
      weights[2] * attributes.data[uv2Base + 9]!;
    expect(u).toBeCloseTo(0.3, 6);
    expect(v).toBeCloseTo(0.5, 6);

    const atlas = new Map<unknown, number>([[h0, 0], [h1, 1], [h2, 2]]);
    const packed = packMaterialsTexture([mat], atlas, {
      uvLayerByTexCoord: layout.layerByTexCoord,
    });
    const selector = (mapIndex: number): number =>
      packed.data[MATERIAL_UV_SELECTOR_TEXEL_OFFSET * 4 + mapIndex]!;
    expect(selector(0)).toBe(2); // baseColorMap -> UV0
    expect(selector(2)).toBe(4); // roughnessMap -> UV1
    expect(selector(4)).toBe(5); // emissiveMap -> UV2
  });

  it('makes a sparse high texCoord cost one extra layer rather than max-id layers', () => {
    const texCoord = 10_000;
    const handle = {};
    const highUv = new Float32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
    const primitive = meshWithUvSets(material({
      baseColorMap: { handle, texCoord },
    }));
    const sparse = primitive.uvSets?.slice() ?? [];
    sparse[2] = undefined;
    sparse[texCoord] = highUv;
    const scene: Scene = {
      primitives: [{ ...primitive, uvSets: sparse }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
    const layout = buildUvAttributeLayout(scene, merged, merged.materials);
    expect(layout.layerCount).toBe(6);
    expect(layout.layerByTexCoord.get(texCoord)).toBe(5);
    expect(layout.mergedByTexCoord.get(texCoord)).toEqual(highUv);
  });

  it('densely packs authored UV ids at and above the native array-index ceiling', () => {
    const nativeCeilingIndex = 0xffff_fffe;
    const ordinaryPropertyIndex = 0x1_0000_0001;
    const handle = {};
    const nativeUv = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const ordinaryUv = new Float32Array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
    const primitive = meshWithUvSets(material({
      baseColorMap: { handle, texCoord: ordinaryPropertyIndex },
    }));
    const sparse: Array<Float32Array | undefined> = [];
    sparse[0] = primitive.uvs;
    sparse[1] = primitive.uv1;
    sparse[nativeCeilingIndex] = nativeUv;
    sparse[ordinaryPropertyIndex] = ordinaryUv;
    const scene: Scene = {
      primitives: [{ ...primitive, uvSets: sparse }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
    const layout = buildUvAttributeLayout(scene, merged, merged.materials);

    expect(layout.layerCount).toBe(7);
    expect(layout.layerByTexCoord.get(nativeCeilingIndex)).toBe(5);
    expect(layout.layerByTexCoord.get(ordinaryPropertyIndex)).toBe(6);
    expect(layout.mergedByTexCoord.get(nativeCeilingIndex)).toEqual(nativeUv);
    expect(layout.mergedByTexCoord.get(ordinaryPropertyIndex)).toEqual(ordinaryUv);
  });

  it('aligns arbitrary UV streams for mesh, instanced-mesh, and skinned-mesh ranges', () => {
    const uv2 = new Float32Array([0.2, 0.2, 0.4, 0.4, 0.6, 0.6]);
    const base = {
      positions: POSITIONS,
      normals: NORMALS,
      indices: INDICES,
      uvSets: [undefined, undefined, uv2] as const,
      material: material(),
    };
    const mesh: MeshPrimitive = { kind: 'mesh', id: 'm', ...base };
    const instanced: InstancedMeshPrimitive = {
      kind: 'instanced-mesh', id: 'i', ...base, instances: [IDENTITY, IDENTITY],
    };
    const skinned: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 's',
      ...base,
      skinIndices: new Uint32Array(12),
      skinWeights: new Float32Array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
      ]),
      bones: new Float32Array(IDENTITY),
      boneInverses: new Float32Array(IDENTITY),
    };
    const scene: Scene = {
      primitives: [mesh, instanced, skinned],
      emitters: [],
      environment: { kind: 'none' },
    };
    const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
    const layout = buildUvAttributeLayout(scene, merged, merged.materials);
    const stream = layout.mergedByTexCoord.get(2)!;
    expect(merged.meshVertexRanges).toHaveLength(4);
    for (const range of merged.meshVertexRanges) {
      expect(Array.from(stream.slice(range.vertexStart * 2, range.vertexStart * 2 + 6)))
        .toEqual(Array.from(uv2));
    }
  });

  it('uses merge provenance when an earlier primitive is intentionally filtered', () => {
    const discardedUv = new Float32Array([9, 9, 9, 9, 9, 9]);
    const retainedUv = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const discarded: MeshPrimitive = {
      ...meshWithUvSets(material()),
      id: 'discarded',
      uvSets: [undefined, undefined, discardedUv],
      tangents: new Float32Array([
        0, 1, 0, 1,
        0, 1, 0, 1,
        0, 1, 0, 1,
      ]),
      colors: new Float32Array([
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
      ]),
      material: material({ baseColor: [0.25, 0.25, 0.25] }),
    };
    const retained: MeshPrimitive = {
      ...meshWithUvSets(material()),
      id: 'retained',
      uvSets: [undefined, undefined, retainedUv],
      tangents: new Float32Array([
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
      ]),
      colors: new Float32Array([
        0, 1, 0, 1,
        0, 1, 0, 1,
        0, 1, 0, 1,
      ]),
      material: material({ baseColor: [0.75, 0.75, 0.75] }),
    };
    const scene: Scene = {
      primitives: [discarded, retained],
      emitters: [],
      environment: { kind: 'none' },
    };
    const merged = mergeWorldSpaceFromCore(scene, {
      positionStride: 4,
      filter: (primitive) => primitive.id !== 'discarded',
    });
    expect(merged.meshVertexRanges).toHaveLength(1);
    expect(merged.meshVertexRanges[0]?.sourcePrimitiveId).toBe('retained');
    const layout = buildUvAttributeLayout(scene, merged, merged.materials);
    expect(layout.mergedByTexCoord.get(2)).toEqual(retainedUv);

    const retainedScene: Scene = { ...scene, primitives: [retained] };
    const geometry = buildSceneGeometryTextureData(retainedScene);
    const floatsPerLayer = geometry.attrData.dim * geometry.attrData.dim * 4;
    expect(Array.from(geometry.attrData.data.slice(floatsPerLayer, floatsPerLayer + 4)))
      .toEqual([1, 0, 0, 1]);
    expect(Array.from(geometry.attrData.data.slice(3 * floatsPerLayer, 3 * floatsPerLayer + 4)))
      .toEqual([0, 1, 0, 1]);
    expect([...geometry.vertexColorMaterialIds]).toEqual([0]);
  });

  it('packs selectors for all 21 mapped-rich slots plus both layer normals', () => {
    const handle = {};
    const record = { ...material() } as Record<string, unknown>;
    const layout = new Map<number, number>([[0, 2], [1, 4], [2, 5]]);
    for (let mapIndex = 0; mapIndex < MATERIAL_MAP_FIELD_ORDER.length; mapIndex += 1) {
      record[MATERIAL_MAP_FIELD_ORDER[mapIndex]!] = {
        handle,
        texCoord: mapIndex % 3,
      };
    }
    record.frontLayer = {
      transmission: [1, 1, 1], normalMap: { handle, texCoord: 2 },
    };
    record.backLayer = {
      transmission: [1, 1, 1], normalMap: { handle, texCoord: 1 },
    };
    const packed = packMaterialsTexture(
      [record as unknown as MaterialSpec],
      new Map<unknown, number>([[handle, 0]]),
      { uvLayerByTexCoord: layout },
    ).data;
    for (let mapIndex = 0; mapIndex < MATERIAL_MAP_FIELD_ORDER.length; mapIndex += 1) {
      const expectedLayer = [2, 4, 5][mapIndex % 3]!;
      expect(packed[MATERIAL_UV_SELECTOR_TEXEL_OFFSET * 4 + mapIndex]).toBe(expectedLayer);
    }
    // Layer-normal payload texel 129 stores dense attribute layers, not authored ids.
    expect(packed[129 * 4]).toBe(5);
    expect(packed[129 * 4 + 1]).toBe(4);
  });
});
