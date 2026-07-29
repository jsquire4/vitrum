import { describe, expect, it, vi } from 'vitest';
import {
  analyzeGltfAsset,
  createGltfSceneController,
  evaluateGltfBackendCompatibility,
  gltfToScene,
  type GltfJson,
  type GltfTextureSourceExtension,
} from './index.js';
import type { MeshPrimitive, TextureRef } from '@vitrum/core';
import {
  effectiveGltfTextureSourceExtensions,
  hasBuiltinWebpTextureSourceDecoder,
} from './textures.js';

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function i16Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 2);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setInt16(i * 2, v, true));
  return buf;
}

function concat(buffers: readonly ArrayBuffer[]): ArrayBuffer {
  const total = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const buffer of buffers) {
    out.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }
  return out.buffer;
}

function minimalMaterialGltf(material: NonNullable<GltfJson['materials']>[number]): {
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
} {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const uvs = f32Buffer([0, 0, 1, 0, 0, 1]);
  const buffer = concat([positions, uvs]);
  return {
    buffers: new Map([[0, buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
      materials: [material],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 0, byteOffset: positions.byteLength, byteLength: uvs.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    },
  };
}

function textureSourceGltf(extension: GltfTextureSourceExtension): {
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
  fallbackBytes: number[];
  extensionBytes: number[];
} {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const uvs = f32Buffer([0, 0, 1, 0, 0, 1]);
  const fallbackBytes = [0x89, 0x50, 0x4e, 0x47];
  const extensionBytes = [0xab, 0xcd, 0xef, 0x01];
  const fallback = new Uint8Array(fallbackBytes).buffer;
  const alternate = new Uint8Array(extensionBytes).buffer;
  const buffer = concat([positions, uvs, fallback, alternate]);
  const fallbackOffset = positions.byteLength + uvs.byteLength;
  const alternateOffset = fallbackOffset + fallback.byteLength;
  return {
    fallbackBytes,
    extensionBytes,
    buffers: new Map([[0, buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
      textures: [{
        source: 0,
        extensions: {
          [extension]: { source: 1 },
        },
      }],
      images: [
        { bufferView: 2, mimeType: 'image/png' },
        {
          bufferView: 3,
          mimeType: extension === 'KHR_texture_basisu'
            ? 'image/ktx2'
            : extension === 'MSFT_texture_dds'
              ? 'image/vnd-ms.dds'
              : 'image/webp',
        },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 0, byteOffset: positions.byteLength, byteLength: uvs.byteLength },
        { buffer: 0, byteOffset: fallbackOffset, byteLength: fallback.byteLength },
        { buffer: 0, byteOffset: alternateOffset, byteLength: alternate.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    },
  };
}

describe('glTF common extension policy', () => {
  it('classifies every unimplemented optional extension, including vendor prefixes', () => {
    const { gltf } = minimalMaterialGltf({
      extensions: {
        ADOBE_materials_clearcoat_tint: { tintFactor: [1, 0.5, 0.25] },
      },
    });
    gltf.extensionsUsed = ['ADOBE_materials_clearcoat_tint', 'CESIUM_primitive_outline'];

    const report = analyzeGltfAsset(gltf);

    expect(report.extensions.unsupportedOptional).toEqual([
      'ADOBE_materials_clearcoat_tint',
      'CESIUM_primitive_outline',
    ]);
    expect(report.materials.unsupportedKnownExtensions).toContain(
      'ADOBE_materials_clearcoat_tint',
    );
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgpu').issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          category: 'extension',
          name: 'ADOBE_materials_clearcoat_tint',
          support: 'unsupported',
        }),
        expect.objectContaining({
          category: 'extension',
          name: 'CESIUM_primitive_outline',
          support: 'unsupported',
        }),
      ]),
    );
  });

  it('reports a selected EXT_lights_image_based environment instead of silently dropping it', async () => {
    const { gltf, buffers } = minimalMaterialGltf({});
    gltf.extensionsUsed = ['EXT_lights_image_based'];
    gltf.extensions = {
      EXT_lights_image_based: {
        lights: [{
          intensity: 2,
          irradianceCoefficients: Array.from({ length: 9 }, () => [0, 0, 0]),
          specularImageSize: 16,
          specularImages: [[0, 1, 2, 3, 4, 5]],
        }],
      },
    };
    gltf.scenes![0]!.extensions = {
      EXT_lights_image_based: { light: 0 },
    };

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.unsupportedOptional).toContain('EXT_lights_image_based');

    const result = await gltfToScene(gltf, { buffers });
    expect(result.scene.environment).toEqual({ kind: 'none' });
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      severity: 'warning',
      code: 'unsupported-image-based-light',
      path: 'scenes[0].extensions.EXT_lights_image_based',
    }));
  });

  it('accepts required KHR_mesh_quantization when normalized accessors unpack to scene floats', async () => {
    const positions = i16Buffer([
      -32767, -32767, 0,
      32767, -32767, 0,
      -32767, 32767, 0,
    ]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      extensionsUsed: ['KHR_mesh_quantization'],
      extensionsRequired: ['KHR_mesh_quantization'],
      accessors: [{
        bufferView: 0,
        componentType: 5122,
        normalized: true,
        count: 3,
        type: 'VEC3',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
      buffers: [{ byteLength: positions.byteLength }],
    };

    const { scene, warnings } = await gltfToScene(gltf, { buffers: new Map([[0, positions]]) });
    const primitive = scene.primitives[0] as MeshPrimitive;

    expect(warnings.some((warning) => warning.includes('KHR_mesh_quantization'))).toBe(false);
    expect(primitive.positions[0]).toBeCloseTo(-1);
    expect(primitive.positions[1]).toBeCloseTo(-1);
    expect(primitive.positions[3]).toBeCloseTo(1);
    expect(primitive.positions[4]).toBeCloseTo(-1);
    expect(primitive.positions[6]).toBeCloseTo(-1);
    expect(primitive.positions[7]).toBeCloseTo(1);

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.supported).toContain('KHR_mesh_quantization');
    expect(report.extensions.unsupportedRequired).not.toContain('KHR_mesh_quantization');
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgpu').issues).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          category: 'extension',
          name: 'KHR_mesh_quantization',
        }),
      ]),
    );
  });

  it('accepts non-normalized integer POSITION data permitted by KHR_mesh_quantization', async () => {
    const positions = i16Buffer([
      0, 0, 0,
      10, 0, 0,
      0, 10, 0,
    ]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      extensionsUsed: ['KHR_mesh_quantization'],
      extensionsRequired: ['KHR_mesh_quantization'],
      accessors: [{
        bufferView: 0,
        componentType: 5122,
        count: 3,
        type: 'VEC3',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
      buffers: [{ byteLength: positions.byteLength }],
    };

    const { scene, diagnostics } = await gltfToScene(gltf, {
      buffers: new Map([[0, positions]]),
    });
    const primitive = scene.primitives[0] as MeshPrimitive;

    expect(Array.from(primitive.positions)).toEqual([
      0, 0, 0,
      10, 0, 0,
      0, 10, 0,
    ]);
    expect(diagnostics).not.toContainEqual(expect.objectContaining({ severity: 'error' }));
  });

  it('imports KHR_materials_dispersion as MaterialSpec.dispersionAbbeNumber', async () => {
    const { gltf, buffers } = minimalMaterialGltf({
      name: 'dispersive glass',
      extensions: {
        KHR_materials_ior: { ior: 1.5 },
        KHR_materials_dispersion: { dispersion: 0.4 },
      },
    });
    gltf.extensionsUsed = ['KHR_materials_dispersion'];
    const { scene, warnings } = await gltfToScene(gltf, { buffers });
    const material = (scene.primitives[0] as MeshPrimitive).material;

    expect(material.ior).toBeCloseTo(1.5);
    expect(material.dispersionAbbeNumber).toBeCloseTo(50);
    expect(warnings.some((w) => w.includes('KHR_materials_dispersion'))).toBe(false);

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.supported).toContain('KHR_materials_dispersion');
    expect(report.extensions.unsupportedOptional).not.toContain('KHR_materials_dispersion');
    expect(report.materials.materialFields).toContain('dispersionAbbeNumber');
  });

  it('imports unlit, converts archived spec-gloss factors, and preserves raw spec-gloss data', async () => {
    const { gltf, buffers } = minimalMaterialGltf({
      name: 'legacy material',
      extensions: {
        KHR_materials_unlit: {},
        KHR_materials_pbrSpecularGlossiness: {
          diffuseFactor: [1, 0, 0, 1],
          specularFactor: [1, 1, 1],
          glossinessFactor: 0.5,
        },
      },
    });
    gltf.extensionsUsed = ['KHR_materials_unlit', 'KHR_materials_pbrSpecularGlossiness'];
    gltf.extensionsRequired = ['KHR_materials_unlit', 'KHR_materials_pbrSpecularGlossiness'];

    const { scene, warnings, diagnostics } = await gltfToScene(gltf, { buffers });
    const material = (scene.primitives[0] as MeshPrimitive).material;

    expect(warnings.some((w) => w.includes('KHR_materials_unlit'))).toBe(false);
    expect(warnings.some((w) => w.includes('KHR_materials_pbrSpecularGlossiness') && w.includes('converted approximately'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'spec-gloss-approximation',
      path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness',
      extensionName: 'KHR_materials_pbrSpecularGlossiness',
      materialIndex: 0,
    }));
    expect(material.shadingModel).toBe('unlit');
    expect(material.baseColor).toEqual([1, 0, 0]);
    expect(material.roughness).toBeCloseTo(0.5);
    expect(material.metallic).toBe(0);
    expect(material.specularColor).toEqual([1, 1, 1]);
    expect(material.extensions?.KHR_materials_pbrSpecularGlossiness).toBeDefined();

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.supported).toContain('KHR_materials_unlit');
    expect(report.extensions.unsupportedOptional).not.toContain('KHR_materials_unlit');
    expect(report.extensions.supported).toContain('KHR_materials_pbrSpecularGlossiness');
    expect(report.extensions.unsupportedOptional).not.toContain('KHR_materials_pbrSpecularGlossiness');
    expect(report.materials.unsupportedKnownExtensions).toEqual([]);
    expect(report.materials.materialFields).toEqual(
      expect.arrayContaining(['shadingModel', 'baseColor', 'roughness', 'metallic', 'specularColor']),
    );
    expect(report.materials.specularGlossinessMaterialCount).toBe(1);
    expect(report.materials.specularGlossinessTextureCount).toBe(0);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgl2').issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          category: 'material',
          name: 'KHR_materials_pbrSpecularGlossiness',
          support: 'approximate',
        }),
      ]),
    );
  });

  it('reports spec-gloss texture alpha as an approximate compatibility issue', async () => {
    const { gltf, buffers } = minimalMaterialGltf({
      name: 'legacy textured material',
      extensions: {
        KHR_materials_pbrSpecularGlossiness: {
          diffuseFactor: [0.8, 0.7, 0.6, 1],
          specularFactor: [0.2, 0.3, 0.4],
          glossinessFactor: 0.75,
          specularGlossinessTexture: { index: 0 },
        },
      },
    });
    gltf.extensionsUsed = ['KHR_materials_pbrSpecularGlossiness'];
    gltf.extensionsRequired = ['KHR_materials_pbrSpecularGlossiness'];
    gltf.textures = [{ source: 0 }];
    gltf.images = [{ uri: 'spec-gloss.png', mimeType: 'image/png' }];
    const decodedHandle = { kind: 'decoded-spec-gloss' };

    const { scene, warnings, diagnostics } = await gltfToScene(gltf, {
      buffers,
      imageBytes: {
        0: { bytes: new Uint8Array([0x89, 0x50, 0x4e, 0x47]), mimeType: 'image/png' },
      },
      decodeImage: vi.fn(async () => decodedHandle),
    });
    const material = (scene.primitives[0] as MeshPrimitive).material;

    expect((material.specularColorMap as TextureRef).handle).toBe(decodedHandle);
    expect(material.roughness).toBeCloseTo(0.25);
    expect(material.roughnessMap).toBeUndefined();
    expect(warnings.some((w) => w.includes('glossiness-in-alpha'))).toBe(true);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'spec-gloss-texture-alpha-approximation',
      path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
      extensionName: 'KHR_materials_pbrSpecularGlossiness',
      materialIndex: 0,
    }));

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.specularGlossinessTextureCount).toBe(1);
    expect(report.materials.issuePaths.specGlossGlossinessAlpha).toEqual([
      'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
    ]);
    const issue = evaluateGltfBackendCompatibility(report, 'pt-webgl2').issues.find(
      (candidate) => candidate.name === 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha',
    );
    expect(issue).toMatchObject({
      category: 'material',
      support: 'approximate',
      path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
    });
    expect(issue?.message).toContain('raw import uses scalar glossinessFactor');
    expect(issue?.message).toContain('bake glossiness-in-alpha into a CPU-linear roughnessMap');
  });

  it('selects KHR_materials_variants mappings by name or index and otherwise falls back to base material', async () => {
    const { gltf, buffers } = minimalMaterialGltf({
      name: 'base red',
      pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] },
    });
    gltf.extensionsUsed = ['KHR_materials_variants'];
    gltf.extensionsRequired = ['KHR_materials_variants'];
    gltf.extensions = {
      KHR_materials_variants: {
        variants: [{ name: 'blue' }],
      },
    };
    gltf.materials!.push({
      name: 'variant blue',
      pbrMetallicRoughness: { baseColorFactor: [0, 0, 1, 1] },
    });
    gltf.meshes![0]!.primitives[0]!.extensions = {
      KHR_materials_variants: {
        mappings: [{ material: 1, variants: [0] }],
      },
    };

    const base = await gltfToScene(gltf, { buffers });
    const byName = await gltfToScene(gltf, { buffers, materialVariant: 'blue' });
    const byIndex = await gltfToScene(gltf, { buffers, materialVariant: 0 });
    const missing = await gltfToScene(gltf, { buffers, materialVariant: 'green' });

    expect(((base.scene.primitives[0] as MeshPrimitive).material.baseColor)).toEqual([1, 0, 0]);
    expect(((byName.scene.primitives[0] as MeshPrimitive).material.baseColor)).toEqual([0, 0, 1]);
    expect(((byIndex.scene.primitives[0] as MeshPrimitive).material.baseColor)).toEqual([0, 0, 1]);
    expect(((missing.scene.primitives[0] as MeshPrimitive).material.baseColor)).toEqual([1, 0, 0]);
    expect(missing.warnings.some((w) => w.includes('materialVariant "green"'))).toBe(true);

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.supported).toContain('KHR_materials_variants');
    expect(report.extensions.unsupportedRequired).not.toContain('KHR_materials_variants');
    expect(report.materials.unsupportedKnownExtensions).not.toContain('KHR_materials_variants');
  });

  it('reports broken KHR_materials_variants mappings before a selected variant silently falls back', async () => {
    const { gltf, buffers } = minimalMaterialGltf({
      name: 'base red',
      pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] },
    });
    gltf.extensionsUsed = ['KHR_materials_variants'];
    gltf.extensionsRequired = ['KHR_materials_variants'];
    gltf.extensions = {
      KHR_materials_variants: {
        variants: [{ name: 'blue' }],
      },
    };
    gltf.meshes![0]!.primitives[0]!.extensions = {
      KHR_materials_variants: {
        mappings: [
          { material: 99, variants: [0] },
          { material: 0, variants: [7] },
          { material: 0 } as unknown as { material: number; variants: number[] },
        ],
      },
    };

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.variantMappingIssues).toEqual([
      expect.objectContaining({
        kind: 'missing-material',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[0].material',
        materialIndex: 99,
      }),
      expect.objectContaining({
        kind: 'missing-variant',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[1].variants[0]',
        variantIndex: 7,
      }),
      expect.objectContaining({
        kind: 'missing-variant-list',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[2].variants',
      }),
    ]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'KHR_materials_variants.mapping.missing-material',
        support: 'unsupported',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[0].material',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'KHR_materials_variants.mapping.missing-variant',
        support: 'unsupported',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[1].variants[0]',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'KHR_materials_variants.mapping.missing-variant-list',
        support: 'unsupported',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[2].variants',
      }),
    ]));

    const selected = await gltfToScene(gltf, { buffers, materialVariant: 'blue' });
    expect(((selected.scene.primitives[0] as MeshPrimitive).material.baseColor)).toEqual([1, 0, 0]);
    expect(selected.warnings.some((w) => w.includes('references missing material 99'))).toBe(true);
    expect(selected.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'material-variant-material-missing',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[0].material',
      }),
      expect.objectContaining({
        code: 'material-variant-mapping-malformed',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[2].variants',
      }),
    ]));
  });

  it.each([
    ['missing', undefined],
    ['NaN', Number.NaN],
  ])('fails a selected variant mapping with a %s material index back to the base material', async (
    _label,
    invalidMaterial,
  ) => {
    const { gltf, buffers } = minimalMaterialGltf({
      name: 'base red',
      pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] },
    });
    gltf.extensionsUsed = ['KHR_materials_variants'];
    gltf.extensions = {
      KHR_materials_variants: { variants: [{ name: 'bad' }] },
    };
    gltf.meshes![0]!.primitives[0]!.extensions = {
      KHR_materials_variants: {
        mappings: [{
          material: invalidMaterial,
          variants: [0],
        } as unknown as { material: number; variants: number[] }],
      },
    };

    const selected = await gltfToScene(gltf, {
      buffers,
      materialVariant: 'bad',
    });

    expect((selected.scene.primitives[0] as MeshPrimitive).material.baseColor)
      .toEqual([1, 0, 0]);
    expect(selected.diagnostics).toContainEqual(expect.objectContaining({
      code: 'material-variant-material-missing',
      path:
        'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[0].material',
    }));
  });

  it('reports malformed root KHR_materials_variants lists without throwing', async () => {
    const { gltf, buffers } = minimalMaterialGltf({
      name: 'base red',
      pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] },
    });
    gltf.extensionsUsed = ['KHR_materials_variants'];
    gltf.extensionsRequired = ['KHR_materials_variants'];
    gltf.extensions = {};
    (gltf.extensions as Record<string, unknown>).KHR_materials_variants = {
      variants: { name: 'blue' },
    };
    gltf.meshes![0]!.primitives[0]!.extensions = {
      KHR_materials_variants: {
        mappings: [{ material: 0, variants: [0] }],
      },
    };

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.variantMappingIssues).toEqual([
      expect.objectContaining({
        kind: 'malformed-root-variant-list',
        path: 'extensions.KHR_materials_variants.variants',
      }),
    ]);
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'KHR_materials_variants.variants.malformed-list',
        support: 'unsupported',
        path: 'extensions.KHR_materials_variants.variants',
      }),
    ]));

    const selected = await gltfToScene(gltf, { buffers, materialVariant: 'blue' });
    expect(((selected.scene.primitives[0] as MeshPrimitive).material.baseColor)).toEqual([1, 0, 0]);
    expect(selected.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'material-variant-list-malformed',
        path: 'extensions.KHR_materials_variants.variants',
      }),
    ]));
  });

  it('selects required KHR_texture_basisu through the built-in codec path', async () => {
    const { gltf, buffers, extensionBytes } = textureSourceGltf('KHR_texture_basisu');
    gltf.extensionsRequired = ['KHR_texture_basisu'];
    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.textureSourceUses).toEqual([
      {
        extension: 'KHR_texture_basisu',
        textureIndex: 0,
        sourceImageIndex: 1,
        path: 'textures[0].extensions.KHR_texture_basisu',
        selected: true,
        required: true,
        hasBaseSource: true,
        requiresHook: false,
        mimeType: 'image/ktx2',
      },
    ]);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgl2').issues).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          category: 'extension',
          name: 'KHR_texture_basisu',
          support: 'requires-hook',
        }),
      ]),
    );
    const builtinScene = await gltfToScene(gltf, { buffers });
    const builtinHandle = (
      builtinScene.scene.primitives[0] as MeshPrimitive
    ).material.baseColorMap as TextureRef;
    expect(builtinHandle.handle).toMatchObject({
      kind: 'raw-image',
      mimeType: 'image/ktx2',
    });

    const decodeImage = vi.fn(async (bytes: Uint8Array, mimeType: string) => {
      expect(Array.from(bytes)).toEqual(extensionBytes);
      expect(mimeType).toBe('image/ktx2');
      return { decoded: mimeType };
    });
    const { scene } = await gltfToScene(gltf, {
      buffers,
      decodeImage,
    });

    expect(decodeImage).toHaveBeenCalledTimes(1);
    const material = (scene.primitives[0] as MeshPrimitive).material;
    expect((material.baseColorMap as TextureRef).handle).toEqual({ decoded: 'image/ktx2' });
  });

  it('selects optional WebP sources automatically when this host has an adapter-owned decoder', async () => {
    const { gltf, buffers, extensionBytes } = textureSourceGltf('EXT_texture_webp');
    gltf.extensionsUsed = ['EXT_texture_webp'];
    const seen: number[][] = [];
    const decodeImage = vi.fn(async (bytes: Uint8Array, mimeType: string) => {
      seen.push(Array.from(bytes));
      return { decoded: mimeType, bytes: Array.from(bytes) };
    });

    await gltfToScene(gltf, { buffers, decodeImage });
    await gltfToScene(gltf, {
      buffers,
      decodeImage,
      textureSourceExtensions: ['EXT_texture_webp'],
    });

    expect(hasBuiltinWebpTextureSourceDecoder()).toBe(true);
    expect(effectiveGltfTextureSourceExtensions(undefined)).toContain('EXT_texture_webp');
    expect(seen).toEqual([extensionBytes, extensionBytes]);
    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.supported).toContain('EXT_texture_webp');
    expect(report.extensions.requiresHook).not.toContain('EXT_texture_webp');
    expect(report.extensions.unsupportedOptional).not.toContain('EXT_texture_webp');
    expect(report.extensions.textureSourceUses).toEqual([
      {
        extension: 'EXT_texture_webp',
        textureIndex: 0,
        sourceImageIndex: 1,
        path: 'textures[0].extensions.EXT_texture_webp',
        selected: true,
        required: false,
        hasBaseSource: true,
        requiresHook: false,
        mimeType: 'image/webp',
      },
    ]);
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).not.toContainEqual(expect.objectContaining({
      category: 'extension',
      name: 'EXT_texture_webp',
    }));

    const selectedReport = analyzeGltfAsset(gltf, { textureSourceExtensions: ['EXT_texture_webp'] });
    expect(selectedReport.extensions.requiresHook).not.toContain('EXT_texture_webp');
    expect(selectedReport.extensions.textureSourceUses).toEqual([
      {
        extension: 'EXT_texture_webp',
        textureIndex: 0,
        sourceImageIndex: 1,
        path: 'textures[0].extensions.EXT_texture_webp',
        selected: true,
        required: false,
        hasBaseSource: true,
        requiresHook: false,
        mimeType: 'image/webp',
      },
    ]);
    expect(evaluateGltfBackendCompatibility(selectedReport, 'pt-webgl2').issues.some(
      (issue) => issue.category === 'extension' && issue.name === 'EXT_texture_webp',
    )).toBe(false);
  });

  it('recognizes built-in WebP support when no base source fallback exists', () => {
    const { gltf } = textureSourceGltf('EXT_texture_webp');
    gltf.extensionsUsed = ['EXT_texture_webp'];
    delete gltf.textures![0]!.source;

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.requiresHook).not.toContain('EXT_texture_webp');
    expect(report.extensions.unsupportedOptional).not.toContain('EXT_texture_webp');
    expect(report.extensions.textureSourceUses).toEqual([
      expect.objectContaining({
        extension: 'EXT_texture_webp',
        textureIndex: 0,
        sourceImageIndex: 1,
        path: 'textures[0].extensions.EXT_texture_webp',
        required: false,
        hasBaseSource: false,
        requiresHook: false,
      }),
    ]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some(
      (issue) => issue.category === 'extension' && issue.name === 'EXT_texture_webp',
    )).toBe(false);
  });

  it('treats authored mipmapped nearest sampler policy as exact on pt-webgl2', () => {
    const { gltf } = minimalMaterialGltf({
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
      },
    });
    gltf.textures = [{ source: 0, sampler: 0 }];
    gltf.images = [{ uri: 'albedo.png', mimeType: 'image/png' }];
    gltf.samplers = [{ magFilter: 9728, minFilter: 9984 }];

    const report = analyzeGltfAsset(gltf);

    expect(report.materials.samplerPolicies).toEqual([
      {
        materialField: 'baseColorMap',
        textureIndex: 0,
        samplerIndex: 0,
        materialPath: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        path: 'samplers[0].minFilter',
        magFilter: 'nearest',
        minFilter: 'nearest',
        mipFilter: 'nearest',
        usesMipmaps: true,
      },
    ]);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgl2').issues.some(
      (issue) => issue.name === 'baseColorMap.samplerPolicy',
    )).toBe(false);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgpu').issues.some(
      (issue) => issue.name === 'baseColorMap.samplerPolicy',
    )).toBe(false);
  });

  it('treats linear mipmapped sampler policy as exact on pt-webgpu full and pt-webgl2', () => {
    const { gltf } = minimalMaterialGltf({
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
      },
    });
    gltf.textures = [{ source: 0, sampler: 0 }];
    gltf.images = [{ uri: 'albedo.png', mimeType: 'image/png' }];
    gltf.samplers = [{ magFilter: 9729, minFilter: 9987 }];

    const report = analyzeGltfAsset(gltf);

    expect(report.materials.samplerPolicies).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        path: 'samplers[0].minFilter',
        magFilter: 'linear',
        minFilter: 'linear',
        mipFilter: 'linear',
        usesMipmaps: true,
      }),
    ]);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgpu').issues.some(
      (issue) => issue.name === 'baseColorMap.samplerPolicy',
    )).toBe(false);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgl2').issues.some(
      (issue) => issue.name === 'baseColorMap.samplerPolicy',
    )).toBe(false);
    expect(evaluateGltfBackendCompatibility(report, 'walkaround-hybrid').issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          category: 'material',
          name: 'baseColorMap.samplerPolicy',
          support: 'approximate',
          path: 'samplers[0].minFilter',
        }),
      ]),
    );
  });

  it('treats pt-webgpu linear mag/min sampler policies with none or nearest mip filters as native', () => {
    const { gltf } = minimalMaterialGltf({
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
        metallicRoughnessTexture: { index: 1 },
      },
    });
    gltf.textures = [{ source: 0, sampler: 0 }, { source: 1, sampler: 1 }];
    gltf.images = [
      { uri: 'albedo.png', mimeType: 'image/png' },
      { uri: 'orm.png', mimeType: 'image/png' },
    ];
    gltf.samplers = [
      { magFilter: 9729, minFilter: 9729 },
      { magFilter: 9729, minFilter: 9985 },
    ];

    const report = analyzeGltfAsset(gltf);
    const ptWebgpuIssues = evaluateGltfBackendCompatibility(report, 'pt-webgpu').issues;

    expect(report.materials.samplerPolicies).toEqual(expect.arrayContaining([
      expect.objectContaining({
        materialField: 'baseColorMap',
        minFilter: 'linear',
        mipFilter: 'none',
      }),
      expect.objectContaining({
        materialField: 'roughnessMap',
        minFilter: 'linear',
        mipFilter: 'nearest',
      }),
    ]));
    expect(ptWebgpuIssues.some((issue) => issue.name.endsWith('.samplerPolicy'))).toBe(false);
  });

  it('treats nearest non-mip sampler policy as exact on pt-webgl2, pt-webgpu full, and walkaround', () => {
    const { gltf } = minimalMaterialGltf({
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0 },
      },
    });
    gltf.textures = [{ source: 0, sampler: 0 }];
    gltf.images = [{ uri: 'albedo.png', mimeType: 'image/png' }];
    gltf.samplers = [{ magFilter: 9728, minFilter: 9728 }];

    const report = analyzeGltfAsset(gltf);

    expect(evaluateGltfBackendCompatibility(report, 'pt-webgl2').issues.some(
      (issue) => issue.name === 'baseColorMap.samplerPolicy',
    )).toBe(false);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgpu').issues.some(
      (issue) => issue.name === 'baseColorMap.samplerPolicy',
    )).toBe(false);
    expect(evaluateGltfBackendCompatibility(report, 'walkaround-hybrid').issues.some(
      (issue) => issue.name === 'baseColorMap.samplerPolicy',
    )).toBe(false);
  });

  it('applies inherited KHR_node_visibility to meshes/lights, never cameras, and restores via STEP pointer', async () => {
    const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const times = f32Buffer([0, 1]);
    const visibility = new Uint8Array([0, 255]).buffer;
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [
        {
          children: [1, 2],
          extensions: { KHR_node_visibility: { visible: false } },
        },
        {
          mesh: 0,
          extensions: { KHR_lights_punctual: { light: 0 } },
        },
        { camera: 0 },
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      cameras: [{
        type: 'perspective',
        perspective: { yfov: 1, znear: 0.1, zfar: 100 },
      }],
      extensions: {
        KHR_lights_punctual: {
          lights: [{ type: 'point', color: [1, 1, 1], intensity: 1, range: 10 }],
        },
      },
      extensionsUsed: ['KHR_node_visibility', 'KHR_animation_pointer', 'KHR_lights_punctual'],
      extensionsRequired: ['KHR_node_visibility', 'KHR_animation_pointer'],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 2, componentType: 5121, count: 2, type: 'SCALAR' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 1, byteOffset: 0, byteLength: times.byteLength },
        { buffer: 2, byteOffset: 0, byteLength: visibility.byteLength },
      ],
      buffers: [
        { byteLength: positions.byteLength },
        { byteLength: times.byteLength },
        { byteLength: visibility.byteLength },
      ],
      animations: [{
        name: 'show-visual-subtree',
        samplers: [{ input: 1, output: 2, interpolation: 'STEP' }],
        channels: [{
          sampler: 0,
          target: {
            path: 'pointer',
            extensions: {
              KHR_animation_pointer: {
                pointer: '/nodes/0/extensions/KHR_node_visibility/visible',
              },
            },
          },
        }],
      }],
    };
    const imported = await gltfToScene(gltf, {
      buffers: new Map([[0, positions], [1, times], [2, visibility]]),
    });

    expect(imported.scene.primitives).toHaveLength(0);
    expect(imported.scene.emitters).toHaveLength(0);
    expect(imported.cameras).toHaveLength(1);
    expect(imported.nodeVisibilityPrimitives).toHaveLength(1);
    expect(imported.nodeVisibilityEmitters).toHaveLength(1);
    expect(analyzeGltfAsset(gltf).extensions.supported).toEqual(expect.arrayContaining([
      'KHR_node_visibility',
      'KHR_animation_pointer',
    ]));

    const setScene = vi.fn();
    const controller = createGltfSceneController({ ...imported, gltf }, {
      engine: { setScene },
      setSceneOnAttach: false,
    });
    const shown = controller.applyAnimation('show-visual-subtree', 1);

    expect(shown.scene.primitives).toHaveLength(1);
    expect(shown.scene.emitters).toHaveLength(1);
    expect(shown.cameras).toHaveLength(1);
    expect(shown.usedSetScene).toBe(true);
    expect(setScene).toHaveBeenCalledWith(shown.scene);

    const hidden = controller.applyAnimation('show-visual-subtree', 0);
    expect(hidden.scene.primitives).toHaveLength(0);
    expect(hidden.scene.emitters).toHaveLength(0);
    expect(hidden.cameras).toHaveLength(1);
  });
});
