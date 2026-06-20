// gltfKhronosSweep.test.ts - analyze-only fixture sweep for Road GATE-GLTF.
//
// These are small, vendored-in-test JSON fixtures shaped after common Khronos
// sample categories. They intentionally call only analyzeGltfAsset() plus the
// compatibility planner: no fetch, no image decode, no buffer decode, no engine.

import { describe, expect, it } from 'vitest';
import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  evaluateGltfBackendProfileCompatibility,
  rankGltfBackends,
  type GltfBackendCompatibility,
  type GltfFeatureReport,
  type GltfJson,
} from './index.js';
import { collectGltfSceneReachability } from './sceneScope.js';

function expectSourcePaths(compatibility: GltfBackendCompatibility): void {
  expect(compatibility.issues.every((issue) => issue.path.length > 0)).toBe(true);
}

function minimalTriangle(): GltfJson {
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/Box' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
    bufferViews: [{ buffer: 0, byteLength: 36 }],
    buffers: [{ byteLength: 36 }],
  };
}

function texturedPbr(): GltfJson {
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/DamagedHelmet-like' },
    extensionsUsed: ['KHR_texture_transform'],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: {
          POSITION: 0,
          NORMAL: 1,
          TANGENT: 2,
          TEXCOORD_0: 3,
          TEXCOORD_1: 4,
        },
        material: 0,
      }],
    }],
    materials: [{
      pbrMetallicRoughness: {
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: {
          index: 0,
          texCoord: 0,
          extensions: {
            KHR_texture_transform: {
              texCoord: 1,
              offset: [0.25, 0.5],
              scale: [2, 2],
              rotation: 0.125,
            },
          },
        },
        metallicFactor: 1,
        roughnessFactor: 0.5,
        metallicRoughnessTexture: { index: 1, texCoord: 1 },
      },
      normalTexture: { index: 2, scale: 0.75 },
      occlusionTexture: { index: 3, strength: 0.5 },
      emissiveFactor: [0.1, 0.2, 0.3],
      emissiveTexture: { index: 4 },
    }],
    textures: [
      { source: 0, sampler: 0 },
      { source: 1, sampler: 0 },
      { source: 2, sampler: 0 },
      { source: 3, sampler: 0 },
      { source: 4, sampler: 0 },
    ],
    samplers: [{ magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 33648 }],
    images: [
      { uri: 'baseColor.png', mimeType: 'image/png' },
      { uri: 'metallicRoughness.png', mimeType: 'image/png' },
      { uri: 'normal.png', mimeType: 'image/png' },
      { uri: 'occlusion.png', mimeType: 'image/png' },
      { uri: 'emissive.png', mimeType: 'image/png' },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC4' },
      { bufferView: 3, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 4, componentType: 5126, count: 3, type: 'VEC2' },
    ],
    bufferViews: [{ buffer: 0, byteLength: 256 }],
    buffers: [{ uri: 'mesh.bin', byteLength: 256 }],
  };
}

function uv2MaterialTexture(): GltfJson {
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/MultiUV-like' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: {
          POSITION: 0,
          TEXCOORD_0: 1,
          TEXCOORD_1: 2,
          TEXCOORD_2: 3,
        },
        material: 0,
      }],
    }],
    materials: [{
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0, texCoord: 2 },
      },
    }],
    textures: [{ source: 0 }],
    images: [{ uri: 'uv2-baseColor.png', mimeType: 'image/png' }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 3, componentType: 5126, count: 3, type: 'VEC2' },
    ],
    bufferViews: Array.from({ length: 4 }, (_, index) => ({
      buffer: 0,
      byteOffset: index * 32,
      byteLength: 32,
    })),
    buffers: [{ byteLength: 128 }],
  };
}

function extensionGlass(): GltfJson {
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/TransmissionTest-like' },
    extensionsUsed: [
      'KHR_lights_punctual',
      'KHR_materials_transmission',
      'KHR_materials_ior',
      'KHR_materials_volume',
      'KHR_materials_specular',
      'KHR_materials_sheen',
      'KHR_materials_clearcoat',
      'KHR_materials_iridescence',
      'KHR_materials_anisotropy',
      'KHR_materials_dispersion',
      'KHR_materials_emissive_strength',
      'KHR_materials_pbrSpecularGlossiness',
    ],
    extensions: {
      KHR_lights_punctual: {
        lights: [{ type: 'point', intensity: 100 }],
      },
    },
    cameras: [{ type: 'perspective' }],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, camera: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, material: 0 }] }],
    materials: [{
      doubleSided: true,
      alphaMode: 'BLEND',
      pbrMetallicRoughness: {
        baseColorFactor: [0.8, 0.9, 1, 0.5],
        baseColorTexture: { index: 0 },
      },
      extensions: {
        KHR_materials_transmission: { transmissionFactor: 0.9, transmissionTexture: { index: 1 } },
        KHR_materials_ior: { ior: 1.45 },
        KHR_materials_volume: {
          thicknessFactor: 0.2,
          thicknessTexture: { index: 2 },
          attenuationDistance: 5,
          attenuationColor: [0.8, 0.9, 1],
        },
        KHR_materials_specular: {
          specularFactor: 0.8,
          specularTexture: { index: 3 },
          specularColorFactor: [1, 0.95, 0.9],
          specularColorTexture: { index: 4 },
        },
        KHR_materials_sheen: {
          sheenColorFactor: [0.5, 0.4, 0.3],
          sheenColorTexture: { index: 5 },
          sheenRoughnessFactor: 0.25,
          sheenRoughnessTexture: { index: 6 },
        },
        KHR_materials_clearcoat: {
          clearcoatFactor: 0.6,
          clearcoatTexture: { index: 7 },
          clearcoatRoughnessFactor: 0.1,
          clearcoatRoughnessTexture: { index: 8 },
          clearcoatNormalTexture: { index: 9, scale: 0.5 },
        },
        KHR_materials_iridescence: {
          iridescenceFactor: 0.7,
          iridescenceTexture: { index: 10 },
          iridescenceIor: 1.8,
          iridescenceThicknessMinimum: 100,
          iridescenceThicknessMaximum: 400,
          iridescenceThicknessTexture: { index: 11 },
        },
        KHR_materials_anisotropy: {
          anisotropyStrength: 0.4,
          anisotropyRotation: 0.2,
          anisotropyTexture: { index: 12 },
        },
        KHR_materials_dispersion: { dispersion: 20 },
        KHR_materials_emissive_strength: { emissiveStrength: 2 },
        KHR_materials_pbrSpecularGlossiness: {
          diffuseFactor: [1, 1, 1, 0.8],
          specularFactor: [0.8, 0.7, 0.6],
          glossinessFactor: 0.4,
          specularGlossinessTexture: { index: 13 },
        },
      },
    }],
    textures: Array.from({ length: 14 }, (_, index) => ({ source: index })),
    images: Array.from({ length: 14 }, (_, index) => ({ uri: `tex-${index}.png`, mimeType: 'image/png' })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
    ],
    bufferViews: [{ buffer: 0, byteLength: 128 }],
    buffers: [{ uri: 'mesh.bin', byteLength: 128 }],
  };
}

function skinMorphAnimation(): GltfJson {
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/SimpleSkin-MorphStress-like' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, skin: 0, weights: [0.25] }, { name: 'joint' }],
    skins: [{ joints: [1], inverseBindMatrices: 9 }],
    meshes: [{
      weights: [0.1],
      primitives: [{
        attributes: {
          POSITION: 0,
          NORMAL: 1,
          TANGENT: 2,
          TEXCOORD_0: 3,
          TEXCOORD_1: 4,
          COLOR_0: 5,
          JOINTS_0: 6,
          WEIGHTS_0: 7,
        },
        targets: [{ POSITION: 10, NORMAL: 11, TANGENT: 12 }],
      }],
    }],
    animations: [{
      samplers: [
        { input: 13, output: 14, interpolation: 'LINEAR' },
        { input: 13, output: 15, interpolation: 'CUBICSPLINE' },
        { input: 13, output: 16, interpolation: 'LINEAR' },
      ],
      channels: [
        { sampler: 0, target: { node: 0, path: 'translation' } },
        { sampler: 2, target: { node: 1, path: 'rotation' } },
        { sampler: 1, target: { node: 0, path: 'weights' } },
      ],
    }],
    accessors: Array.from({ length: 17 }, (_, index) => ({
      bufferView: index,
      componentType: 5126,
      count: 3,
      type: index === 9 ? 'MAT4' : index === 3 || index === 4 ? 'VEC2' : index === 16 ? 'VEC4' : 'VEC3',
    })),
    bufferViews: Array.from({ length: 17 }, (_, index) => ({
      buffer: 0,
      byteOffset: index * 16,
      byteLength: index === 16 ? 48 : 16,
    })),
    buffers: [{ byteLength: 304 }],
  };
}

function compressedAndAlternateSources(): GltfJson {
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/CompressionHooks-like' },
    extensionsUsed: ['KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'EXT_texture_webp'],
    extensionsRequired: ['KHR_draco_mesh_compression'],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0 },
        material: 0,
        extensions: {
          KHR_draco_mesh_compression: {
            bufferView: 1,
            attributes: { POSITION: 0 },
          },
        },
      }],
    }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    textures: [{
      extensions: {
        EXT_texture_webp: { source: 0 },
      },
    }],
    images: [{ uri: 'baseColor.webp', mimeType: 'image/webp' }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
    bufferViews: [
      { buffer: 0, byteLength: 36 },
      {
        buffer: 0,
        byteOffset: 36,
        byteLength: 64,
        extensions: {
          EXT_meshopt_compression: {
            buffer: 0,
            byteOffset: 36,
            byteLength: 64,
            byteStride: 12,
            count: 3,
            mode: 'ATTRIBUTES',
            filter: 'NONE',
          },
        },
      },
    ],
    buffers: [{ uri: 'compressed.bin', byteLength: 100 }],
  };
}

function meshoptFallbackBufferSample(
  opts: { fallbackStub?: boolean; extensionName?: 'EXT_meshopt_compression' | 'KHR_meshopt_compression' } = {},
): GltfJson {
  const extensionName = opts.extensionName ?? 'EXT_meshopt_compression';
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/MeshoptFallback-like' },
    extensionsUsed: [extensionName],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0 },
        indices: 1,
      }],
    }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    bufferViews: [
      {
        buffer: 0,
        byteOffset: 0,
        byteLength: 36,
        byteStride: 12,
        extensions: {
          [extensionName]: {
            buffer: 1,
            byteOffset: 0,
            byteLength: 16,
            byteStride: 12,
            count: 3,
            mode: 'ATTRIBUTES',
            filter: 'NONE',
          },
        },
      },
      {
        buffer: 0,
        byteOffset: 36,
        byteLength: 6,
        extensions: {
          [extensionName]: {
            buffer: 1,
            byteOffset: 16,
            byteLength: 16,
            byteStride: 2,
            count: 3,
            mode: 'TRIANGLES',
            filter: 'NONE',
          },
        },
      },
    ],
    buffers: [
      opts.fallbackStub
        ? { byteLength: 0, extensions: { [extensionName]: { fallback: true } } }
        : { uri: 'fallback.bin', byteLength: 42 },
      { uri: 'meshopt.bin', byteLength: 32 },
    ],
  };
}

function optionalDracoSample(opts: { withFallback?: boolean; required?: boolean } = {}): GltfJson {
  return {
    asset: { version: '2.0', generator: 'KhronosSampleModels/DracoFallback-like' },
    extensionsUsed: ['KHR_draco_mesh_compression'],
    ...(opts.required ? { extensionsRequired: ['KHR_draco_mesh_compression'] } : {}),
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0, NORMAL: 1 },
        indices: 2,
        extensions: {
          KHR_draco_mesh_compression: {
            bufferView: 3,
            attributes: { POSITION: 10, NORMAL: 11 },
          },
        },
      }],
    }],
    accessors: [
      {
        ...(opts.withFallback ? { bufferView: 0 } : {}),
        componentType: 5126,
        count: 3,
        type: 'VEC3',
      },
      {
        ...(opts.withFallback ? { bufferView: 1 } : {}),
        componentType: 5126,
        count: 3,
        type: 'VEC3',
      },
      {
        ...(opts.withFallback ? { bufferView: 2 } : {}),
        componentType: 5123,
        count: 3,
        type: 'SCALAR',
      },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 36 },
      { buffer: 0, byteOffset: 72, byteLength: 6 },
      { buffer: 0, byteOffset: 78, byteLength: 64 },
    ],
    buffers: [{ uri: 'draco-fallback.bin', byteLength: 142 }],
  };
}

function reportFor(gltf: GltfJson): GltfFeatureReport {
  const report = analyzeGltfAsset(gltf);
  expect(report.assetVersion).toBe('2.0');
  return report;
}

describe('GATE-GLTF analyze-only Khronos-style sweep', () => {
  it('covers representative fixtures without loading buffers, images, or engines', () => {
    const fixtures: ReadonlyArray<readonly [string, GltfJson]> = [
      ['Box', minimalTriangle()],
      ['DamagedHelmet-like textured PBR', texturedPbr()],
      ['MultiUV-like unsupported UV2 material texture', uv2MaterialTexture()],
      ['TransmissionTest-like material extensions', extensionGlass()],
      ['SimpleSkin/Morph animation', skinMorphAnimation()],
      ['Compression hook fixture', compressedAndAlternateSources()],
      ['Meshopt fallback fixture', meshoptFallbackBufferSample()],
    ];

    for (const [name, gltf] of fixtures) {
      const report = reportFor(gltf);
      const ranked = rankGltfBackends(report, 'fidelity');
      expect(ranked.length, name).toBeGreaterThan(0);
      for (const compatibility of ranked) {
        expectSourcePaths(compatibility);
      }
    }
  });

  it('classifies scalar triangle assets as fully compatible with the fidelity default', () => {
    const report = reportFor(minimalTriangle());
    const ranked = rankGltfBackends(report, 'fidelity');
    const selected = ranked[0]!;

    expect(report.primitives).toMatchObject({
      total: 1,
      expectedPrimitiveKinds: ['mesh'],
      hasSkins: false,
      hasMorphTargets: false,
    });
    expect(selected).toMatchObject({
      backend: 'pt-webgl2',
      unsupportedCount: 0,
      requiresHookCount: 0,
      isCompatible: true,
    });
  });

  it('keeps textured PBR feature inventory source-path complete without decode work', () => {
    const report = reportFor(texturedPbr());
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.resources.externalBufferCount).toBe(1);
    expect(report.resources.externalImageCount).toBe(5);
    expect(report.primitives).toMatchObject({
      hasTangents: true,
      hasUv1: true,
    });
    expect(report.materials.materialFields).toEqual(expect.arrayContaining([
      'baseColor',
      'baseColorMap',
      'metallic',
      'metallicMap',
      'roughness',
      'roughnessMap',
      'normalMap',
      'normalScale',
      'aoMap',
      'aoMapIntensity',
      'emissive',
      'emissiveMap',
    ]));
    expect(report.materials.textureTransformCount).toBe(1);
    expect(report.materials.samplerPolicies.length).toBeGreaterThan(0);
    expect(webgl2.unsupportedCount).toBe(0);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap.samplerPolicy',
        support: 'approximate',
        path: 'samplers[0].minFilter',
      }),
    ]));
  });

  it('accepts TEXCOORD_2 material textures when the primitive can remap them into uv1', () => {
    const report = reportFor(uv2MaterialTexture());

    expect(report.primitives.attributeSemantics).toContain('TEXCOORD_2');
    expect(report.materials.uvSets).toEqual([2]);
    expect(report.materials.unrepresentableUvSets).toEqual([]);
    for (const profile of ['pt-webgl2', 'pt-webgpu', 'pt-webgpu-lite', 'walkaround-hybrid'] as const) {
      const compatibility = evaluateGltfBackendProfileCompatibility(report, profile);
      expect(compatibility.issues.some((issue) => issue.name === 'TEXCOORD_2')).toBe(false);
    }
  });

  it('reports material-extension glass caveats with concrete paths', () => {
    const report = reportFor(extensionGlass());
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.extensions.supported).toEqual(expect.arrayContaining([
      'KHR_materials_transmission',
      'KHR_materials_volume',
      'KHR_materials_specular',
      'KHR_materials_clearcoat',
      'KHR_materials_sheen',
      'KHR_materials_iridescence',
      'KHR_materials_anisotropy',
      'KHR_lights_punctual',
    ]));
    expect(report.sceneGraph).toMatchObject({ cameras: 1, punctualLights: 1 });
    expect(report.materials).toMatchObject({
      doubleSidedCount: 1,
      volumeThicknessTextureCount: 1,
      specularGlossinessMaterialCount: 1,
      specularGlossinessTextureCount: 1,
    });
    expect(report.materials.materialFields).toEqual(expect.arrayContaining([
      'transmission',
      'transmissionMap',
      'ior',
      'thickness',
      'thicknessMap',
      'attenuationDistance',
      'attenuationColor',
      'specularIntensity',
      'specularIntensityMap',
      'specularColor',
      'specularColorMap',
      'clearcoat',
      'clearcoatMap',
      'sheen',
      'sheenColorMap',
      'iridescence',
      'iridescenceMap',
      'anisotropy',
      'anisotropyMap',
      'dispersionAbbeNumber',
      'emissiveIntensity',
    ]));
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'scene',
        name: 'cameras',
        support: 'approximate',
        path: 'cameras[0]',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'doubleSided',
        support: 'approximate',
        path: 'materials[0].doubleSided',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha',
        support: 'approximate',
        path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
      }),
    ]));
  });

  it('scores KHR_materials_emissive_strength as supported scalar emissiveIntensity on every backend profile', () => {
    const report = reportFor(extensionGlass());

    expect(report.materials.materialFields).toContain('emissiveIntensity');
    for (const profile of ['pt-webgl2', 'pt-webgpu', 'pt-webgpu-lite', 'walkaround-hybrid'] as const) {
      const compatibility = evaluateGltfBackendProfileCompatibility(report, profile);
      expect(compatibility.issues.some((issue) =>
        issue.category === 'material' &&
        issue.name === 'emissiveIntensity',
      ), profile).toBe(false);
    }
  });

  it('detects skin, morph tangent, vertex color, uv1, and animation paths before import', () => {
    const report = reportFor(skinMorphAnimation());
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    const walkaround = evaluateGltfBackendCompatibility(report, 'walkaround-hybrid');

    expect(report.primitives).toMatchObject({
      expectedPrimitiveKinds: ['mesh', 'skinned-mesh'],
      hasSkins: true,
      hasMorphTargets: true,
      hasMorphTargetTangents: true,
      hasVertexColors: true,
      hasUv1: true,
    });
    expect(report.animations).toMatchObject({
      count: 1,
      channelCount: 3,
      paths: ['rotation', 'translation', 'weights'],
      interpolations: ['CUBICSPLINE', 'LINEAR'],
      malformedChannels: [],
      targetNodeCount: 2,
    });
    expect(webgl2.unsupportedCount).toBe(0);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'primitive',
        name: 'morphTargetTangents',
        support: 'approximate',
        path: 'meshes[0].primitives[0].targets[0].TANGENT',
      }),
    ]));
    expect(walkaround.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'primitive',
        name: 'vertexColors',
        support: 'approximate',
        path: 'meshes[0].primitives[0].attributes.COLOR_0',
      }),
    ]));
  });

  it('reports secondary vertex color sets as unsupported ignored data', () => {
    const gltf = minimalTriangle();
    gltf.meshes![0]!.primitives[0]!.attributes.COLOR_1 = 1;
    gltf.accessors!.push({ bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' });
    gltf.bufferViews!.push({ buffer: 0, byteOffset: 36, byteLength: 36 });
    gltf.buffers![0]!.byteLength = 72;

    const report = reportFor(gltf);
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.ignoredVertexColorSets).toEqual(['COLOR_1']);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'primitive',
        name: 'COLOR_1',
        support: 'unsupported',
        path: 'meshes[0].primitives[0].attributes.COLOR_1',
      }),
    ]));
    expect(webgl2.unsupportedCount).toBeGreaterThanOrEqual(1);
  });

  it('keeps Draco and no-base alternate texture-source assets in requires-hook space', () => {
    const report = reportFor(compressedAndAlternateSources());
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives).toMatchObject({
      usesDraco: true,
      usesMeshopt: true,
    });
    expect(report.extensions.requiresHook).toEqual([
      'EXT_texture_webp',
      'KHR_draco_mesh_compression',
    ]);
    expect(report.extensions.textureSourceUses).toEqual([
      expect.objectContaining({
        extension: 'EXT_texture_webp',
        textureIndex: 0,
        sourceImageIndex: 0,
        path: 'textures[0].extensions.EXT_texture_webp',
        required: false,
        hasBaseSource: false,
        requiresHook: true,
        mimeType: 'image/webp',
      }),
    ]);
    expect(report.materials.textureReferenceIssues).toEqual([
      expect.objectContaining({
        kind: 'disabled-texture-source-extension',
        materialField: 'baseColorMap',
        textureIndex: 0,
        path: 'textures[0].extensions.EXT_texture_webp',
        textureSourceExtensions: ['EXT_texture_webp'],
      }),
    ]);
    expect(webgl2.requiresHookCount).toBe(3);
    expect(webgl2.unsupportedCount).toBe(0);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'KHR_draco_mesh_compression',
        support: 'requires-hook',
        path: 'extensionsRequired[0]',
      }),
      expect.objectContaining({
        category: 'extension',
        name: 'EXT_texture_webp',
        support: 'requires-hook',
        path: 'textures[0].extensions.EXT_texture_webp',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap.textureRef.disabled-texture-source-extension',
        support: 'requires-hook',
        path: 'textures[0].extensions.EXT_texture_webp',
      }),
    ]));
  });

  it('treats optional meshopt assets with real fallback buffers as hook-free compatible', () => {
    const report = reportFor(meshoptFallbackBufferSample());
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.usesMeshopt).toBe(true);
    expect(report.extensions.requiresHook).toEqual([]);
    expect(webgl2.requiresHookCount).toBe(0);
    expect(webgl2.unsupportedCount).toBe(0);
    expect(webgl2.issues.some((issue) =>
      issue.category === 'extension' &&
      issue.name === 'EXT_meshopt_compression' &&
      issue.support === 'requires-hook',
    )).toBe(false);
  });

  it('keeps optional meshopt fallback-stub assets in requires-hook space', () => {
    const report = reportFor(meshoptFallbackBufferSample({ fallbackStub: true }));
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.usesMeshopt).toBe(true);
    expect(report.extensions.requiresHook).toEqual(['EXT_meshopt_compression']);
    expect(webgl2.requiresHookCount).toBe(1);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'EXT_meshopt_compression',
        support: 'requires-hook',
        path: 'bufferViews[0].extensions.EXT_meshopt_compression',
      }),
    ]));
  });

  it('classifies Khronos KHR_meshopt_compression samples with the same hook policy', () => {
    const report = reportFor(meshoptFallbackBufferSample({
      fallbackStub: true,
      extensionName: 'KHR_meshopt_compression',
    }));
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.usesMeshopt).toBe(true);
    expect(report.extensions.supported).toContain('KHR_meshopt_compression');
    expect(report.extensions.requiresHook).toEqual(['KHR_meshopt_compression']);
    expect(report.extensions.unsupportedRequired).not.toContain('KHR_meshopt_compression');
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'KHR_meshopt_compression',
        support: 'requires-hook',
        path: 'bufferViews[0].extensions.KHR_meshopt_compression',
      }),
    ]));
  });

  it('keeps selected-scene bufferView meshopt and mesh-quantization extensions in scoped reports', () => {
    const gltf = meshoptFallbackBufferSample({
      fallbackStub: true,
      extensionName: 'KHR_meshopt_compression',
    });
    gltf.extensionsUsed = ['KHR_mesh_quantization', 'KHR_meshopt_compression'];
    gltf.extensionsRequired = ['KHR_mesh_quantization', 'KHR_meshopt_compression'];
    gltf.accessors![0] = {
      ...gltf.accessors![0]!,
      componentType: 5122,
      normalized: true,
    };

    const report = analyzeGltfAsset(gltf, { sceneIndex: 0 });
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    const reachability = collectGltfSceneReachability(gltf, 0);

    expect(report.extensions.used).toEqual(['KHR_mesh_quantization', 'KHR_meshopt_compression']);
    expect(report.extensions.required).toEqual(['KHR_mesh_quantization', 'KHR_meshopt_compression']);
    expect(report.extensions.supported).toEqual(expect.arrayContaining([
      'KHR_mesh_quantization',
      'KHR_meshopt_compression',
    ]));
    expect(report.extensions.requiresHook).toEqual(['KHR_meshopt_compression']);
    expect(report.primitives.usesMeshopt).toBe(true);
    expect(report.extensions.sourcePaths.KHR_meshopt_compression).toEqual(expect.arrayContaining([
      'bufferViews[0].extensions.KHR_meshopt_compression',
      'extensionsUsed[1]',
      'extensionsRequired[1]',
    ]));
    expect([...reachability.bufferIndices].sort()).toEqual([0, 1]);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'KHR_meshopt_compression',
        support: 'requires-hook',
        path: 'bufferViews[0].extensions.KHR_meshopt_compression',
      }),
    ]));
    expect(webgl2.issues).not.toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'KHR_mesh_quantization',
        support: 'unsupported',
      }),
    ]));
  });

  it('treats optional Draco assets with real fallback accessors as hook-free compatible', () => {
    const report = reportFor(optionalDracoSample({ withFallback: true }));
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.usesDraco).toBe(true);
    expect(report.extensions.requiresHook).toEqual([]);
    expect(webgl2.requiresHookCount).toBe(0);
    expect(webgl2.unsupportedCount).toBe(0);
    expect(webgl2.issues.some((issue) =>
      issue.category === 'extension' &&
      issue.name === 'KHR_draco_mesh_compression' &&
      issue.support === 'requires-hook',
    )).toBe(false);
  });

  it('keeps optional Draco assets without fallback accessors in requires-hook space', () => {
    const report = reportFor(optionalDracoSample());
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.usesDraco).toBe(true);
    expect(report.extensions.requiresHook).toEqual(['KHR_draco_mesh_compression']);
    expect(webgl2.requiresHookCount).toBe(1);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'KHR_draco_mesh_compression',
        support: 'requires-hook',
      }),
    ]));
  });

  it('keeps required Draco assets in requires-hook space even when fallback accessors exist', () => {
    const report = reportFor(optionalDracoSample({ withFallback: true, required: true }));
    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.usesDraco).toBe(true);
    expect(report.extensions.requiresHook).toEqual(['KHR_draco_mesh_compression']);
    expect(webgl2.requiresHookCount).toBe(1);
    expect(webgl2.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'KHR_draco_mesh_compression',
        support: 'requires-hook',
        path: 'extensionsRequired[0]',
      }),
    ]));
  });

  it('keeps pt-webgpu full and lite profile differences visible in the sweep', () => {
    const report = reportFor(texturedPbr());
    const full = evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu');
    const lite = evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu-lite');

    expect(full.unsupportedCount).toBe(0);
    expect(lite.unsupportedCount).toBeGreaterThan(full.unsupportedCount);
    expect(lite.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap',
        support: 'unsupported',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'normalMap',
        support: 'unsupported',
        path: 'materials[0].normalTexture',
      }),
    ]));
  });
});
