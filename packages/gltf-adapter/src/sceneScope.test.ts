import { describe, expect, it, vi } from 'vitest';
import { loadGltfAsset } from './assetLoader.js';
import { collectGltfSceneReachability } from './sceneScope.js';
import type { GltfJson, GltfMaterial } from './gltfTypes.js';

function triangleAsset(material: GltfMaterial): {
  readonly gltf: GltfJson;
  readonly geometry: ArrayBuffer;
} {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const uvs = new Float32Array([
    0, 0,
    1, 0,
    0, 1,
  ]);
  const geometry = new Uint8Array(positions.byteLength + uvs.byteLength);
  geometry.set(new Uint8Array(positions.buffer), 0);
  geometry.set(new Uint8Array(uvs.buffer), positions.byteLength);
  return {
    geometry: geometry.buffer,
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, TEXCOORD_0: 1 },
          material: 0,
        }],
      }],
      accessors: [{
        bufferView: 0,
        componentType: 5126,
        count: 3,
        type: 'VEC3',
      }, {
        bufferView: 1,
        componentType: 5126,
        count: 3,
        type: 'VEC2',
      }],
      bufferViews: [{
        buffer: 0,
        byteOffset: 0,
        byteLength: positions.byteLength,
      }, {
        buffer: 0,
        byteOffset: positions.byteLength,
        byteLength: uvs.byteLength,
      }],
      buffers: [{ byteLength: geometry.byteLength }],
      materials: [material],
    },
  };
}

describe('material texture scene reachability', () => {
  it('enumerates every supported base and extension texture slot without traversing metadata', () => {
    const material = {
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
        metallicRoughnessTexture: { index: 1 },
      },
      normalTexture: { index: 2 },
      occlusionTexture: { index: 3 },
      emissiveTexture: { index: 4 },
      extensions: {
        KHR_materials_transmission: {
          transmissionTexture: { index: 5 },
        },
        KHR_materials_volume: {
          thicknessTexture: { index: 6 },
        },
        KHR_materials_specular: {
          specularTexture: { index: 7 },
          specularColorTexture: { index: 8 },
        },
        KHR_materials_sheen: {
          sheenColorTexture: { index: 9 },
          sheenRoughnessTexture: { index: 10 },
        },
        KHR_materials_clearcoat: {
          clearcoatTexture: { index: 11 },
          clearcoatRoughnessTexture: { index: 12 },
          clearcoatNormalTexture: { index: 13 },
        },
        KHR_materials_iridescence: {
          iridescenceTexture: { index: 14 },
          iridescenceThicknessTexture: { index: 15 },
        },
        KHR_materials_anisotropy: {
          anisotropyTexture: { index: 16 },
        },
        KHR_materials_pbrSpecularGlossiness: {
          diffuseTexture: { index: 17 },
          specularGlossinessTexture: { index: 18 },
        },
        KHR_materials_ior: {
          ior: 1.5,
          index: 20,
        },
        VENDOR_material_metadata: {
          nested: { index: 21 },
        },
      },
      extras: {
        index: 19,
        nested: [{ index: 22 }],
      },
    } as unknown as GltfMaterial;
    const { gltf } = triangleAsset(material);
    gltf.textures = Array.from({ length: 23 }, (_, source) => ({ source }));
    gltf.images = Array.from(
      { length: 23 },
      (_, index) => ({ uri: `image-${index}.png` }),
    );

    const reachability = collectGltfSceneReachability(gltf, 0);

    expect([...reachability.textureIndices]).toEqual(
      Array.from({ length: 19 }, (_, index) => index),
    );
    expect([...reachability.imageIndices]).toEqual(
      Array.from({ length: 19 }, (_, index) => index),
    );
    expect(reachability.textureIndices.has(19)).toBe(false);
    expect(reachability.textureIndices.has(20)).toBe(false);
    expect(reachability.textureIndices.has(21)).toBe(false);
    expect(reachability.textureIndices.has(22)).toBe(false);
  });

  it('does not fetch or decode an image referenced only by extras metadata', async () => {
    const material = {
      extras: {
        preview: { index: 0 },
      },
      extensions: {
        VENDOR_material_metadata: {
          index: 0,
        },
      },
    } as unknown as GltfMaterial;
    const { gltf, geometry } = triangleAsset(material);
    gltf.textures = [{ source: 0 }];
    gltf.images = [{ uri: 'irrelevant.png', mimeType: 'image/png' }];
    const fetch = vi.fn(async () => {
      throw new Error('irrelevant image must not be fetched');
    });
    const decodeImage = vi.fn(async () => ({ kind: 'decoded-image' }));

    await loadGltfAsset(gltf, {
      baseUri: 'https://example.test/scene.gltf',
      buffers: new Map([[0, geometry]]),
      fetch,
      decodeImage,
    });

    expect(fetch).not.toHaveBeenCalled();
    expect(decodeImage).not.toHaveBeenCalled();
  });

  it('still fetches and decodes a canonical material texture reference', async () => {
    const material: GltfMaterial = {
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
      },
    };
    const { gltf, geometry } = triangleAsset(material);
    gltf.textures = [{ source: 0 }];
    gltf.images = [{ uri: 'base-color.png', mimeType: 'image/png' }];
    const fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      arrayBuffer: async () => new Uint8Array([1, 2, 3, 4]).buffer,
    }));
    const handle = { kind: 'decoded-image' };
    const decodeImage = vi.fn(async () => handle);

    const asset = await loadGltfAsset(gltf, {
      baseUri: 'https://example.test/scene.gltf',
      buffers: new Map([[0, geometry]]),
      fetch,
      decodeImage,
    });

    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledWith(
      'https://example.test/base-color.png',
      undefined,
    );
    expect(decodeImage).toHaveBeenCalledOnce();
    expect(asset.scene.primitives[0]?.material.baseColorMap).toMatchObject({
      handle,
    });
  });
});
