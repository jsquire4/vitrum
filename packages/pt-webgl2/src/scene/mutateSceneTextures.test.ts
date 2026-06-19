import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import type { UploadedSceneTextures } from './sceneTextures.js';
import { tryFastPathMaterialMutation } from './mutateSceneTextures.js';

function fakeGl(): WebGL2RenderingContext {
  const gl = {
    RGBA32F: 0x8814,
    RGBA: 0x1908,
    FLOAT: 0x1406,
    TEXTURE_2D: 0x0DE1,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    NEAREST: 0x2600,
    CLAMP_TO_EDGE: 0x812F,
    MAX_TEXTURE_SIZE: 0x0D33,
    isContextLost: vi.fn(() => false),
    getParameter: vi.fn(() => 8192),
    createTexture: vi.fn(() => ({}) as WebGLTexture),
    bindTexture: vi.fn(),
    texParameteri: vi.fn(),
    texImage2D: vi.fn(),
    texSubImage2D: vi.fn(),
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
    positions: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 0, 1,
      0, 0, 1,
    ]),
    normals: new Float32Array([
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
    ]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: mat,
  };
}

function sceneWithPrimitive(primitive: MeshPrimitive): Scene {
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}

function fakeMerged(materials: readonly MaterialSpec[]): WorldSpaceMergeResult {
  const positions = new Float32Array([
    0, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 1, 0,
    0, 0, 1, 0,
  ]);
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
    colors: new Float32Array([
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
    ]),
    uvs: new Float32Array(8),
    mergedIndices,
    mergedTriMaterialId: new Uint32Array([0, 0]),
    materials: [...materials],
    boundingBox: { min: [0, 0, 0], max: [1, 0, 1] },
    meshVertexRanges: [
      { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 2 },
    ],
    vertexCount: 4,
    triangleCount: 2,
  };
}

function fakeCurrent(): UploadedSceneTextures {
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
    materialLayerMap: null,
    vertexColorMaterialIds: new Set(),
    triangleCount: 2,
    destroy: vi.fn(),
  };
}

describe('tryFastPathMaterialMutation', () => {
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
    expect(swap?.geoPack?.materials[0]).toEqual(expect.objectContaining({
      emissive: [2, 0, 0],
      emissiveIntensity: 3,
    }));
    expect(swap?.textures.meshLightCount).toBe(2);
    expect(swap?.textures.totalEmissiveArea).toBeCloseTo(1, 6);

    const subImageCalls = (gl.texSubImage2D as unknown as ReturnType<typeof vi.fn>).mock.calls;
    expect(subImageCalls).toHaveLength(1);
    const calls = (gl.texImage2D as unknown as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls).toHaveLength(1);
    const meshLightData = calls[0]?.[8] as Float32Array;
    expect(meshLightData[4]).toBeCloseTo(6, 6);
    expect(meshLightData[5]).toBeCloseTo(0, 6);
    expect(meshLightData[6]).toBeCloseTo(0, 6);
  });
});
