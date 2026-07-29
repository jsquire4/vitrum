import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import type { UploadedSceneTextures } from './sceneTextures.js';
import {
  materialTextureMapPatchFields,
  tryFastPathGeometryMutation,
  tryFastPathMaterialMutation,
} from './mutateSceneTextures.js';
import { ATTR_LAYER_COLOR } from './attributesTextureArray.js';
import { MATERIAL_PIXELS } from './materialsTexture.js';
import { buildSceneGeometryTextureData } from './uploadSceneTextures.js';

function fakeGl(): WebGL2RenderingContext {
  const gl = {
    RGBA32F: 0x8814,
    RGBA: 0x1908,
    FLOAT: 0x1406,
    TEXTURE_2D: 0x0de1,
    TEXTURE_2D_ARRAY: 0x8c1a,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    NEAREST: 0x2600,
    CLAMP_TO_EDGE: 0x812f,
    MAX_TEXTURE_SIZE: 0x0d33,
    isContextLost: vi.fn(() => false),
    getParameter: vi.fn(() => 8192),
    createTexture: vi.fn(() => ({})),
    bindTexture: vi.fn(),
    texParameteri: vi.fn(),
    texImage2D: vi.fn(),
    texSubImage2D: vi.fn(),
    texImage3D: vi.fn(),
    texStorage3D: vi.fn(),
    texSubImage3D: vi.fn(),
    deleteTexture: vi.fn(),
  };
  return gl as unknown as WebGL2RenderingContext;
}

function material(overrides: Partial<MaterialSpec>): MaterialSpec {
  return { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0, ...overrides };
}

function panelPrimitive(mat: MaterialSpec): MeshPrimitive {
  return {
    kind: 'mesh',
    id: 'panel',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
    normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: mat,
  };
}

function sceneWithPrimitive(primitive: MeshPrimitive): Scene {
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}

function fakeMerged(materials: readonly MaterialSpec[]): WorldSpaceMergeResult {
  const positions = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]);
  const mergedIndices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return {
    bvhNodes: new Float32Array(),
    positions,
    positionStrideFloats: 4,
    indices: mergedIndices,
    bvhIndexStride: 3,
    triMaterialId: new Uint32Array([0, 0]),
    bvhTriToMergedTri: new Uint32Array([0, 1]),
    normals: new Float32Array(positions.length),
    tangents: new Float32Array(positions.length),
    colors: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    uvs: new Float32Array(8),
    mergedIndices,
    mergedTriMaterialId: new Uint32Array([0, 0]),
    materials: [...materials],
    boundingBox: { min: [0, 0, 0], max: [1, 0, 1] },
    meshVertexRanges: [{ name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 2 }],
    warnings: [],
    vertexCount: 4,
    triangleCount: 2,
  };
}

function fakeCurrent(overrides: Partial<UploadedSceneTextures> = {}): UploadedSceneTextures {
  const tex = {} as WebGLTexture;
  return {
    bvhBounds: tex,
    bvhContents: tex,
    bvhPosition: tex,
    bvhIndex: tex,
    materialIndex: tex,
    materials: tex,
    attributesArray: tex,
    lights: tex,
    lightCount: 0,
    meshLights: tex,
    meshLightCount: 2,
    totalEmissiveArea: 1,
    totalEmissivePower: 1,
    envMap: null,
    envMarginal: null,
    envConditional: null,
    envTotalSum: 0,
    envWidth: 0,
    envHeight: 0,
    textures2DArray: null,
    materialAtlasDim: 0,
    materialAtlasLayerCount: 0,
    materialAtlasLayerCapacity: 0,
    materialHdrTextures2DArray: null,
    materialHdrAtlasDim: 0,
    materialHdrAtlasLayerCount: 0,
    materialHdrAtlasLayerCapacity: 0,
    materialLayerMap: { ldr: null, hdr: null },
    vertexColorMaterialIds: new Set(),
    triangleCount: 2,
    destroy: vi.fn(),
    ...overrides,
  };
}

describe('tryFastPathMaterialMutation', () => {
  it('classifies nested layered normal maps as texture-map patch fields', () => {
    expect(
      materialTextureMapPatchFields({
        material: {
          roughness: 0.5,
          baseColorMap: { handle: { id: 'base' } },
          frontLayer: { normalMap: undefined, normalScale: 0.25 },
          backLayer: undefined,
        },
      } as never),
    ).toEqual(['backLayer.normalMap', 'baseColorMap', 'frontLayer.normalMap']);
  });

  it('subuploads only material rows for scalar-only material mutations', () => {
    const gl = fakeGl();
    const previous = material({ roughness: 1 });
    const next = material({ roughness: 0.25 });

    const swap = tryFastPathMaterialMutation(
      gl,
      fakeCurrent({
        meshLights: null,
        meshLightCount: 0,
        totalEmissiveArea: 0,
        totalEmissivePower: 0,
      }),
      fakeMerged([previous, material({ baseColor: [0.1, 0.1, 0.1] })]),
      sceneWithPrimitive(panelPrimitive(next)),
      'panel',
      { material: next },
    );

    expect(swap).not.toBeNull();
    expect(swap?.geoPack?.materials[0]).toEqual(expect.objectContaining({ roughness: 0.25 }));
    expect((gl.texImage2D as unknown as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
    const subImageCalls = (gl.texSubImage2D as unknown as ReturnType<typeof vi.fn>).mock.calls;
    expect(subImageCalls.length).toBeGreaterThan(1);
    for (const call of subImageCalls) {
      expect(call[5]).toBe(1);
    }
  });

  it('re-packs mesh-area light data after an emissive material mutation', () => {
    const gl = fakeGl();
    const previous = material({ emissive: [0.1, 0, 0], emissiveIntensity: 1 });
    const next = material({ emissive: [2, 0, 0], emissiveIntensity: 3 });

    const swap = tryFastPathMaterialMutation(
      gl,
      fakeCurrent(),
      fakeMerged([previous]),
      sceneWithPrimitive(panelPrimitive(next)),
      'panel',
      { material: next },
    );

    expect(swap).not.toBeNull();
    expect(swap?.geoPack?.materials[0]).toEqual(
      expect.objectContaining({
        emissive: [2, 0, 0],
        emissiveIntensity: 3,
      }),
    );
    expect(swap?.textures.meshLightCount).toBe(2);
    expect(swap?.textures.totalEmissiveArea).toBeCloseTo(1, 6);

    const subImageCalls = (gl.texSubImage2D as unknown as ReturnType<typeof vi.fn>).mock.calls;
    const calls = (gl.texImage2D as unknown as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls).toHaveLength(0);
    const meshLightCall = subImageCalls.find((call) => {
      const data = call[8];
      return data instanceof Float32Array && data[4] === 6;
    });
    expect(meshLightCall).toBeDefined();
    const meshLightData = meshLightCall?.[8] as Float32Array;
    expect(meshLightData[4]).toBeCloseTo(6, 6);
    expect(meshLightData[5]).toBeCloseTo(0, 6);
    expect(meshLightData[6]).toBeCloseTo(0, 6);
  });

  it('defers an atlas-changing patch to the staged full-scene transaction', () => {
    const gl = fakeGl();
    const oldAtlas = { id: 'old-atlas' } as unknown as WebGLTexture;
    const oldNormalHandle = { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) };
    const previous = material({
      frontLayer: {
        transmission: [1, 1, 1],
        roughness: 0.25,
        normalMap: { handle: oldNormalHandle },
        normalScale: 0.75,
      },
    });
    const next = material({
      frontLayer: {
        transmission: [1, 1, 1],
        roughness: 0.25,
        normalScale: 0.75,
      },
    });

    const swap = tryFastPathMaterialMutation(
      gl,
      fakeCurrent({
        textures2DArray: oldAtlas,
        materialAtlasDim: 1,
        materialAtlasLayerCount: 1,
        materialAtlasLayerCapacity: 1,
        materialLayerMap: {
          ldr: { srgb: new Map(), linear: new Map([[oldNormalHandle, 0]]) },
          hdr: null,
        },
        meshLights: null,
        meshLightCount: 0,
        totalEmissiveArea: 0,
        totalEmissivePower: 0,
      }),
      fakeMerged([previous]),
      sceneWithPrimitive(panelPrimitive(next)),
      'panel',
      { material: { frontLayer: { normalMap: undefined } } } as never,
    );

    expect(swap).toBeNull();
    expect(gl.texSubImage2D).not.toHaveBeenCalled();
    expect(gl.texSubImage3D).not.toHaveBeenCalled();
    expect((gl.texImage3D as unknown as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
  });

});

describe('tryFastPathGeometryMutation — arbitrary UV sets', () => {
  function uv2Scene(values: Float32Array): Scene {
    const primitive = panelPrimitive(material({}));
    return sceneWithPrimitive({
      ...primitive,
      uvSets: [primitive.uvs, undefined, values],
    });
  }

  it('subuploads a changed UV2 stream when the dense layout remains stable', () => {
    const before = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
    const after = new Float32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    const currentBuild = buildSceneGeometryTextureData(uv2Scene(before));
    const gl = fakeGl();
    const current = fakeCurrent({
      meshLights: null,
      meshLightCount: 0,
      totalEmissiveArea: 0,
      totalEmissivePower: 0,
      uvLayerByTexCoord: currentBuild.uvLayerByTexCoord,
      attributeLayerCount: currentBuild.attrData.layers,
    });

    const swap = tryFastPathGeometryMutation(gl, current, currentBuild.merged, uv2Scene(after), {
      uvSets: [undefined, undefined, after],
    });
    expect(swap).not.toBeNull();
    const calls = (gl.texSubImage3D as unknown as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0]?.[7]).toBe(6);
    const data = calls[0]?.[10] as Float32Array;
    const floatsPerLayer = currentBuild.attrData.dim * currentBuild.attrData.dim * 4;
    const uv2Base = 5 * floatsPerLayer;
    expect(Array.from(data.slice(uv2Base, uv2Base + 14).filter((_, i) => i % 4 < 2))).toEqual(
      Array.from(after),
    );
  });

  it('declines in-place mutation when adding a new UV id changes array storage', () => {
    const uv2 = new Float32Array(8);
    const uv3 = new Float32Array(8).fill(0.75);
    const currentBuild = buildSceneGeometryTextureData(uv2Scene(uv2));
    const current = fakeCurrent({
      meshLights: null,
      meshLightCount: 0,
      totalEmissiveArea: 0,
      totalEmissivePower: 0,
      uvLayerByTexCoord: currentBuild.uvLayerByTexCoord,
      attributeLayerCount: currentBuild.attrData.layers,
    });
    const next = uv2Scene(uv2);
    const primitive = next.primitives[0] as MeshPrimitive;
    const nextScene = sceneWithPrimitive({
      ...primitive,
      uvSets: [primitive.uvs, undefined, uv2, uv3],
    });
    expect(
      tryFastPathGeometryMutation(fakeGl(), current, currentBuild.merged, nextScene, {
        uvSets: [undefined, undefined, uv2, uv3],
      }),
    ).toBeNull();
  });
});

describe('tryFastPathGeometryMutation — vertex color selection', () => {
  it('refreshes the selected color stream and material vertex-color flag together', () => {
    const disabled = {
      ...panelPrimitive(material({})),
      colorSets: [
        new Float32Array([
          1, 0, 0, 1,
          1, 0, 0, 1,
          1, 0, 0, 1,
          1, 0, 0, 1,
        ]),
        new Float32Array([
          0, 1, 0, 1,
          0, 1, 0, 1,
          0, 1, 0, 1,
          0, 1, 0, 1,
        ]),
      ],
      vertexColorSet: null,
    } satisfies MeshPrimitive;
    const enabled = { ...disabled, vertexColorSet: 1 } satisfies MeshPrimitive;
    const currentBuild = buildSceneGeometryTextureData(sceneWithPrimitive(disabled));
    expect(currentBuild.vertexColorMaterialIds.size).toBe(0);

    const gl = fakeGl();
    const current = fakeCurrent({
      meshLights: null,
      meshLightCount: 0,
      totalEmissiveArea: 0,
      totalEmissivePower: 0,
      vertexColorMaterialIds: currentBuild.vertexColorMaterialIds,
      uvLayerByTexCoord: currentBuild.uvLayerByTexCoord,
      attributeLayerCount: currentBuild.attrData.layers,
    });
    const swap = tryFastPathGeometryMutation(
      gl,
      current,
      currentBuild.merged,
      sceneWithPrimitive(enabled),
      { vertexColorSet: 1 },
    );

    expect(swap).not.toBeNull();
    expect(swap?.textures.vertexColorMaterialIds).toEqual(new Set([0]));

    const attributeCall = (
      gl.texSubImage3D as unknown as ReturnType<typeof vi.fn>
    ).mock.calls.find((call) => call[7] === currentBuild.attrData.layers);
    expect(attributeCall).toBeDefined();
    const attributes = attributeCall?.[10] as Float32Array;
    const floatsPerLayer = currentBuild.attrData.dim * currentBuild.attrData.dim * 4;
    const colorBase = ATTR_LAYER_COLOR * floatsPerLayer;
    expect(Array.from(attributes.slice(colorBase, colorBase + 16))).toEqual([
      0, 1, 0, 1,
      0, 1, 0, 1,
      0, 1, 0, 1,
      0, 1, 0, 1,
    ]);

    const materialCall = (
      gl.texSubImage2D as unknown as ReturnType<typeof vi.fn>
    ).mock.calls.find((call) => {
      const payload = call[8];
      return payload instanceof Float32Array && payload.length >= MATERIAL_PIXELS * 4;
    });
    expect(materialCall).toBeDefined();
    const materialPayload = materialCall?.[8] as Float32Array;
    expect(materialPayload[14 * 4 + 2]).toBe(1);
  });
});
