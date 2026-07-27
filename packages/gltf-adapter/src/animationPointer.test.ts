import type { MaterialSpec } from '@vitrum/core';
import { describe, expect, it } from 'vitest';
import { GltfComponentType, type GltfJson } from './gltfTypes.js';
import {
  applyGltfMaterialAnimationPointerValue,
  gltfAnimationPointerInterpolationError,
  gltfAnimationPointerOutputAccessorError,
  gltfAnimationPointerSampleValueError,
  gltfAnimationPointerTargetDefinitionError,
  gltfAnimationPointerTargetIdentity,
  gltfAnimationTargetsConflict,
  gltfNativeAnimationTargetIdentity,
  resolveGltfAnimationPointer,
  supportedGltfAnimationPointers,
} from './animationPointer.js';
import { buildTextureDecodeReport } from './textureDecodeReport.js';
import { attachGltfTextureRefSource, gltfTextureRefSource } from './textures.js';

const ASSET: GltfJson = {
  asset: { version: '2.0' },
  nodes: [{
    mesh: 0,
    extensions: { KHR_node_visibility: { visible: true } },
  }],
  meshes: [{
    primitives: [{
      attributes: { POSITION: 0 },
      targets: [{ POSITION: 1 }, { POSITION: 2 }],
    }],
    weights: [0, 0],
  }],
  materials: [{
    pbrMetallicRoughness: {
      baseColorFactor: [1, 1, 1, 1],
      baseColorTexture: {
        index: 0,
        extensions: {
          KHR_texture_transform: { offset: [0, 0], rotation: 0, scale: [1, 1] },
        },
      },
    },
    normalTexture: { index: 0, scale: 1 },
    emissiveFactor: [0, 0, 0],
    extensions: {
      KHR_materials_iridescence: {
        iridescenceFactor: 1,
        iridescenceThicknessMinimum: 100,
        iridescenceThicknessMaximum: 400,
      },
    },
  }],
  cameras: [{
    type: 'perspective',
    perspective: { yfov: 1, znear: 0.1, zfar: 100, aspectRatio: 1.5 },
  }],
  extensions: {
    KHR_lights_punctual: {
      lights: [{
        type: 'spot',
        color: [1, 1, 1],
        intensity: 1,
        range: 10,
        spot: { innerConeAngle: 0, outerConeAngle: 0.5 },
      }],
    },
  },
};

function target(pointer: string) {
  const resolved = resolveGltfAnimationPointer(pointer);
  expect(resolved, pointer).toBeDefined();
  return resolved!;
}

describe('ratified KHR_animation_pointer object-model support', () => {
  it('resolves native-equivalent node channels, weight elements, cameras, lights, and visibility', () => {
    expect(target('/nodes/0/rotation')).toMatchObject({ kind: 'node', path: 'rotation', components: 4 });
    expect(target('/nodes/0/weights/1')).toMatchObject({ kind: 'node-weight', weightIndex: 1 });
    expect(target('/nodes/0/extensions/KHR_node_visibility/visible')).toMatchObject({
      kind: 'node-visibility', valueType: 'boolean',
    });
    expect(target('/cameras/0/perspective/yfov')).toMatchObject({ kind: 'camera', field: 'yfov' });
    expect(target('/extensions/KHR_lights_punctual/lights/0/spot/outerConeAngle')).toMatchObject({
      kind: 'punctual-light', field: 'spotOuterConeAngle',
    });
  });

  it('lists and resolves texture-transform offset/rotation/scale for every imported texture slot', () => {
    const templates = supportedGltfAnimationPointers().filter((pointer) =>
      pointer.includes('/extensions/KHR_texture_transform/')
    );
    expect(templates.length).toBeGreaterThan(20);
    expect(templates.some((pointer) => pointer.endsWith('/texCoord'))).toBe(false);
    for (const template of templates) {
      const resolved = resolveGltfAnimationPointer(template.replace('{index}', '0'));
      expect(resolved, template).toMatchObject({ kind: 'material-texture-transform', materialIndex: 0 });
    }
    expect(resolveGltfAnimationPointer(
      '/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/texCoord',
    )).toBeUndefined();
  });

  it('preserves texture source identity and decode-report paths through texture-transform animation', () => {
    const source = {
      path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
      imageSourcePath: 'images[0].uri',
      textureIndex: 0,
      imageIndex: 0,
      imageUri: 'albedo.png',
    } as const;
    const baseColorMap = attachGltfTextureRefSource({
      handle: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) },
      texCoord: 0,
      transform: { offset: [0, 0] },
    }, source);
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap,
    };
    const pointerTarget = target(
      '/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset',
    );
    expect(pointerTarget.kind).toBe('material-texture-transform');
    if (pointerTarget.kind !== 'material-texture-transform') throw new Error('unexpected pointer target');

    const animated = applyGltfMaterialAnimationPointerValue(
      material,
      pointerTarget,
      new Float32Array([0.25, 0.5]),
    );
    expect(animated.baseColorMap?.transform?.offset).toEqual([0.25, 0.5]);
    expect(gltfTextureRefSource(animated.baseColorMap!)).toBe(source);

    const report = buildTextureDecodeReport({
      primitives: [{
        kind: 'mesh',
        id: 'animated-texture',
        positions: new Float32Array(9),
        normals: new Float32Array(9),
        material: animated,
      }],
      emitters: [],
      environment: { kind: 'none' },
    });
    expect(report.entries[0]).toMatchObject({
      path: source.path,
      imageSourcePath: source.imageSourcePath,
      textureIndex: source.textureIndex,
      imageIndex: source.imageIndex,
      imageUri: source.imageUri,
      hasTransform: true,
    });
  });

  it('rejects malformed RFC 6901 escapes and non-canonical array indices', () => {
    expect(resolveGltfAnimationPointer('/nodes/00/translation')).toBeUndefined();
    expect(resolveGltfAnimationPointer('/nodes/0~2/translation')).toBeUndefined();
    expect(resolveGltfAnimationPointer('/nodes/0~/translation')).toBeUndefined();
    expect(resolveGltfAnimationPointer(
      '/materials/00/pbrMetallicRoughness/baseColorFactor',
    )).toBeUndefined();
    expect(resolveGltfAnimationPointer(
      '/materials/0/pbrMetallicRoughness/baseColorFactor~2',
    )).toBeUndefined();
  });

  it('requires a real property/default-bearing enclosing object and validates weight bounds', () => {
    expect(gltfAnimationPointerTargetDefinitionError(ASSET, target('/nodes/0/weights/1'))).toBeUndefined();
    expect(gltfAnimationPointerTargetDefinitionError(ASSET, target('/nodes/0/weights/2'))).toContain('out of bounds');
    expect(gltfAnimationPointerTargetDefinitionError(ASSET, target('/cameras/0/perspective/zfar'))).toBeUndefined();
    const noZfar: GltfJson = { ...ASSET, cameras: [{ type: 'perspective', perspective: { yfov: 1, znear: 0.1 } }] };
    expect(gltfAnimationPointerTargetDefinitionError(noZfar, target('/cameras/0/perspective/zfar'))).toContain('not defined');
    const noTransform: GltfJson = {
      ...ASSET,
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    };
    expect(gltfAnimationPointerTargetDefinitionError(
      noTransform,
      target('/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset'),
    )).toContain('KHR_texture_transform');
  });

  it('enforces accessor shape/type and normative nonzero UBYTE boolean conversion', () => {
    const visibility = target('/nodes/0/extensions/KHR_node_visibility/visible');
    expect(gltfAnimationPointerOutputAccessorError(ASSET, visibility, {
      componentType: GltfComponentType.UNSIGNED_BYTE,
      count: 2,
      type: 'SCALAR',
    })).toBeUndefined();
    expect(gltfAnimationPointerOutputAccessorError(ASSET, visibility, {
      componentType: GltfComponentType.UNSIGNED_BYTE,
      normalized: true,
      count: 2,
      type: 'SCALAR',
    })).toContain('non-normalized');
    expect(gltfAnimationPointerSampleValueError(ASSET, visibility, new Float32Array([255]))).toBeUndefined();
    expect(gltfAnimationPointerInterpolationError(visibility, 'LINEAR')).toContain('STEP');

    const intensity = target('/extensions/KHR_lights_punctual/lights/0/intensity');
    expect(gltfAnimationPointerOutputAccessorError(ASSET, intensity, {
      componentType: GltfComponentType.UNSIGNED_SHORT,
      count: 2,
      type: 'SCALAR',
    })).toBeUndefined();
    expect(gltfAnimationPointerOutputAccessorError(ASSET, intensity, {
      componentType: GltfComponentType.SHORT,
      normalized: true,
      count: 2,
      type: 'SCALAR',
    })).toBeUndefined();
    expect(gltfAnimationPointerOutputAccessorError(ASSET, intensity, {
      componentType: GltfComponentType.UNSIGNED_INT,
      normalized: true,
      count: 2,
      type: 'SCALAR',
    })).toContain('normalized');
  });

  it('uses exact schema domains and fails closed for unconstrained cubic overshoot', () => {
    expect(gltfAnimationPointerSampleValueError(
      ASSET,
      target('/materials/0/emissiveFactor'),
      new Float32Array([1, 0.5, 1.01]),
    )).toContain('[0, 1]');
    expect(gltfAnimationPointerSampleValueError(
      ASSET,
      target('/materials/0/alphaCutoff'),
      new Float32Array([2]),
    )).toBeUndefined();
    expect(gltfAnimationPointerSampleValueError(
      ASSET,
      target('/materials/0/normalTexture/scale'),
      new Float32Array([-2]),
    )).toBeUndefined();
    expect(gltfAnimationPointerInterpolationError(
      target('/extensions/KHR_lights_punctual/lights/0/intensity'),
      'CUBICSPLINE',
    )).toContain('bounded');
    expect(gltfAnimationPointerInterpolationError(
      target('/materials/0/normalTexture/scale'),
      'CUBICSPLINE',
    )).toBeUndefined();
  });

  it('detects native/pointer aliases and whole-array/element conflicts', () => {
    expect(gltfAnimationTargetsConflict(
      gltfNativeAnimationTargetIdentity(0, 'rotation'),
      gltfAnimationPointerTargetIdentity(target('/nodes/0/rotation')),
    )).toBe(true);
    expect(gltfAnimationTargetsConflict(
      gltfNativeAnimationTargetIdentity(0, 'weights'),
      gltfAnimationPointerTargetIdentity(target('/nodes/0/weights/1')),
    )).toBe(true);
    expect(gltfAnimationTargetsConflict(
      gltfAnimationPointerTargetIdentity(target('/nodes/0/weights/0')),
      gltfAnimationPointerTargetIdentity(target('/nodes/0/weights/1')),
    )).toBe(false);
  });
});
