import type { MaterialSpec, TextureRef } from '@vitrum/core';
import { GltfComponentType, type GltfJson, type GltfMaterial } from './gltfTypes.js';
import {
  applyGltfMaterialPointerValue,
  resolveGltfMaterialAnimationPointer,
  supportedGltfMaterialAnimationPointers,
  type GltfMaterialPointerTarget,
} from './materialPointerAnimation.js';
import { attachGltfTextureRefSource, gltfTextureRefSource } from './textures.js';

export type GltfAnimationPointerValueType = 'float' | 'boolean' | 'float-array';
export type GltfAnimationPointerComponents = 1 | 2 | 3 | 4 | 'dynamic';

interface GltfAnimationPointerTargetBase {
  readonly pointer: string;
  readonly components: GltfAnimationPointerComponents;
  readonly valueType: GltfAnimationPointerValueType;
}

export interface GltfMaterialPropertyAnimationPointerTarget extends GltfAnimationPointerTargetBase {
  readonly kind: 'material-property';
  readonly materialIndex: number;
  readonly target: GltfMaterialPointerTarget;
}

export type GltfMaterialTextureRefField =
  | 'baseColorMap'
  | 'normalMap'
  | 'roughnessMap'
  | 'metallicMap'
  | 'transmissionMap'
  | 'thicknessMap'
  | 'emissiveMap'
  | 'aoMap'
  | 'clearcoatMap'
  | 'clearcoatRoughnessMap'
  | 'clearcoatNormalMap'
  | 'sheenColorMap'
  | 'sheenRoughnessMap'
  | 'iridescenceMap'
  | 'iridescenceThicknessMap'
  | 'anisotropyMap'
  | 'specularColorMap'
  | 'specularIntensityMap';

export type GltfTextureTransformAnimationField = 'offset' | 'rotation' | 'scale';

export interface GltfMaterialTextureTransformAnimationPointerTarget extends GltfAnimationPointerTargetBase {
  readonly kind: 'material-texture-transform';
  readonly materialIndex: number;
  readonly texturePath: string;
  readonly materialFields: readonly GltfMaterialTextureRefField[];
  readonly field: GltfTextureTransformAnimationField;
}

export interface GltfNodeAnimationPointerTarget extends GltfAnimationPointerTargetBase {
  readonly kind: 'node';
  readonly nodeIndex: number;
  readonly path: 'translation' | 'rotation' | 'scale' | 'weights';
}

export interface GltfNodeWeightElementAnimationPointerTarget extends GltfAnimationPointerTargetBase {
  readonly kind: 'node-weight';
  readonly nodeIndex: number;
  readonly weightIndex: number;
}

export interface GltfNodeVisibilityAnimationPointerTarget extends GltfAnimationPointerTargetBase {
  readonly kind: 'node-visibility';
  readonly nodeIndex: number;
  readonly field: 'visible';
}

export type GltfPerspectiveCameraAnimationField = 'yfov' | 'znear' | 'zfar' | 'aspectRatio';
export type GltfOrthographicCameraAnimationField = 'xmag' | 'ymag' | 'znear' | 'zfar';

export interface GltfCameraAnimationPointerTarget extends GltfAnimationPointerTargetBase {
  readonly kind: 'camera';
  readonly cameraIndex: number;
  readonly cameraType: 'perspective' | 'orthographic';
  readonly field: GltfPerspectiveCameraAnimationField | GltfOrthographicCameraAnimationField;
}

export type GltfPunctualLightAnimationField =
  | 'color'
  | 'intensity'
  | 'range'
  | 'spotInnerConeAngle'
  | 'spotOuterConeAngle';

export interface GltfPunctualLightAnimationPointerTarget extends GltfAnimationPointerTargetBase {
  readonly kind: 'punctual-light';
  readonly lightIndex: number;
  readonly field: GltfPunctualLightAnimationField;
}

export type GltfAnimationPointerTarget =
  | GltfMaterialPropertyAnimationPointerTarget
  | GltfMaterialTextureTransformAnimationPointerTarget
  | GltfNodeAnimationPointerTarget
  | GltfNodeWeightElementAnimationPointerTarget
  | GltfNodeVisibilityAnimationPointerTarget
  | GltfCameraAnimationPointerTarget
  | GltfPunctualLightAnimationPointerTarget;

export interface GltfAnimationPointerReachability {
  readonly nodeIndices: ReadonlySet<number>;
  readonly materialIndices: ReadonlySet<number>;
  readonly cameraIndices: ReadonlySet<number>;
  readonly punctualLightIndices: ReadonlySet<number>;
}

export interface GltfAnimationTargetIdentity {
  readonly key: string;
  readonly arrayRoot?: string;
  readonly isWholeArray?: boolean;
}

interface MaterialTextureSlotSpec {
  readonly materialFields: readonly GltfMaterialTextureRefField[];
}

const MATERIAL_TEXTURE_SLOT_SPECS: Readonly<Record<string, MaterialTextureSlotSpec>> = Object.freeze({
  'pbrMetallicRoughness/baseColorTexture': { materialFields: ['baseColorMap'] },
  'pbrMetallicRoughness/metallicRoughnessTexture': { materialFields: ['roughnessMap', 'metallicMap'] },
  normalTexture: { materialFields: ['normalMap'] },
  occlusionTexture: { materialFields: ['aoMap'] },
  emissiveTexture: { materialFields: ['emissiveMap'] },
  'extensions/KHR_materials_transmission/transmissionTexture': { materialFields: ['transmissionMap'] },
  'extensions/KHR_materials_volume/thicknessTexture': { materialFields: ['thicknessMap'] },
  'extensions/KHR_materials_specular/specularTexture': { materialFields: ['specularIntensityMap'] },
  'extensions/KHR_materials_specular/specularColorTexture': { materialFields: ['specularColorMap'] },
  'extensions/KHR_materials_sheen/sheenColorTexture': { materialFields: ['sheenColorMap'] },
  'extensions/KHR_materials_sheen/sheenRoughnessTexture': { materialFields: ['sheenRoughnessMap'] },
  'extensions/KHR_materials_clearcoat/clearcoatTexture': { materialFields: ['clearcoatMap'] },
  'extensions/KHR_materials_clearcoat/clearcoatRoughnessTexture': { materialFields: ['clearcoatRoughnessMap'] },
  'extensions/KHR_materials_clearcoat/clearcoatNormalTexture': { materialFields: ['clearcoatNormalMap'] },
  'extensions/KHR_materials_iridescence/iridescenceTexture': { materialFields: ['iridescenceMap'] },
  'extensions/KHR_materials_iridescence/iridescenceThicknessTexture': { materialFields: ['iridescenceThicknessMap'] },
  'extensions/KHR_materials_anisotropy/anisotropyTexture': { materialFields: ['anisotropyMap'] },
  'extensions/KHR_materials_pbrSpecularGlossiness/diffuseTexture': { materialFields: ['baseColorMap'] },
  'extensions/KHR_materials_pbrSpecularGlossiness/specularGlossinessTexture': {
    materialFields: ['specularColorMap', 'roughnessMap'],
  },
});

/**
 * Enumerate only schema-defined texture references supported by the adapter.
 * This deliberately does not recurse through material extras or unknown
 * extensions, where an unrelated metadata object may also own an `index`.
 */
export function collectGltfMaterialTextureIndices(
  material: GltfMaterial | undefined,
): readonly number[] {
  if (material === undefined) return [];
  const indices = new Set<number>();
  for (const path of Object.keys(MATERIAL_TEXTURE_SLOT_SPECS)) {
    const info = valueAtObjectPath(material, path);
    if (!isRecord(info) || !hasOwn(info, 'index')) continue;
    const index = info.index;
    if (typeof index === 'number' && Number.isInteger(index) && index >= 0) {
      indices.add(index);
    }
  }
  return [...indices].sort((a, b) => a - b);
}

const PERSPECTIVE_CAMERA_FIELDS: ReadonlySet<string> = new Set(['yfov', 'znear', 'zfar', 'aspectRatio']);
const ORTHOGRAPHIC_CAMERA_FIELDS: ReadonlySet<string> = new Set(['xmag', 'ymag', 'znear', 'zfar']);

export function resolveGltfAnimationPointer(pointer: string | undefined): GltfAnimationPointerTarget | undefined {
  if (typeof pointer !== 'string') return undefined;

  const materialTarget = resolveGltfMaterialAnimationPointer(pointer);
  if (materialTarget !== undefined) {
    return {
      kind: 'material-property',
      pointer,
      materialIndex: materialTarget.materialIndex,
      components: materialTarget.components,
      valueType: 'float',
      target: materialTarget,
    };
  }

  const segments = decodePointerSegments(pointer);
  if (segments === undefined) return undefined;

  if (segments[0] === 'materials' && segments.length >= 6) {
    const materialIndex = parsePointerIndex(segments[1]);
    const field = segments[segments.length - 1];
    const extensionName = segments[segments.length - 2];
    const extensionsToken = segments[segments.length - 3];
    if (
      materialIndex !== undefined &&
      extensionsToken === 'extensions' &&
      extensionName === 'KHR_texture_transform' &&
      isTextureTransformField(field)
    ) {
      const texturePath = segments.slice(2, -3).join('/');
      const slot = MATERIAL_TEXTURE_SLOT_SPECS[texturePath];
      if (slot !== undefined) {
        return {
          kind: 'material-texture-transform',
          pointer,
          materialIndex,
          texturePath,
          materialFields: slot.materialFields,
          field,
          components: field === 'offset' || field === 'scale' ? 2 : 1,
          valueType: 'float',
        };
      }
    }
  }

  if (segments[0] === 'nodes') {
    const nodeIndex = parsePointerIndex(segments[1]);
    if (nodeIndex === undefined) return undefined;
    if (segments.length === 3) {
      const path = segments[2];
      if (path === 'translation' || path === 'scale') {
        return { kind: 'node', pointer, nodeIndex, path, components: 3, valueType: 'float' };
      }
      if (path === 'rotation') {
        return { kind: 'node', pointer, nodeIndex, path, components: 4, valueType: 'float' };
      }
      if (path === 'weights') {
        return { kind: 'node', pointer, nodeIndex, path, components: 'dynamic', valueType: 'float-array' };
      }
    }
    if (segments.length === 4 && segments[2] === 'weights') {
      const weightIndex = parsePointerIndex(segments[3]);
      if (weightIndex !== undefined) {
        return {
          kind: 'node-weight',
          pointer,
          nodeIndex,
          weightIndex,
          components: 1,
          valueType: 'float',
        };
      }
    }
    if (
      segments.length === 5 &&
      segments[2] === 'extensions' &&
      segments[3] === 'KHR_node_visibility' &&
      segments[4] === 'visible'
    ) {
      return {
        kind: 'node-visibility',
        pointer,
        nodeIndex,
        field: 'visible',
        components: 1,
        valueType: 'boolean',
      };
    }
  }

  if (segments[0] === 'cameras' && segments.length === 4) {
    const cameraIndex = parsePointerIndex(segments[1]);
    const cameraType = segments[2];
    const field = segments[3];
    if (cameraIndex === undefined) return undefined;
    if (cameraType === 'perspective' && PERSPECTIVE_CAMERA_FIELDS.has(field ?? '')) {
      return {
        kind: 'camera', pointer, cameraIndex, cameraType,
        field: field as GltfPerspectiveCameraAnimationField,
        components: 1, valueType: 'float',
      };
    }
    if (cameraType === 'orthographic' && ORTHOGRAPHIC_CAMERA_FIELDS.has(field ?? '')) {
      return {
        kind: 'camera', pointer, cameraIndex, cameraType,
        field: field as GltfOrthographicCameraAnimationField,
        components: 1, valueType: 'float',
      };
    }
  }

  if (
    segments[0] === 'extensions' &&
    segments[1] === 'KHR_lights_punctual' &&
    segments[2] === 'lights'
  ) {
    const lightIndex = parsePointerIndex(segments[3]);
    if (lightIndex === undefined) return undefined;
    const field = segments[4];
    if (segments.length === 5 && field === 'color') {
      return { kind: 'punctual-light', pointer, lightIndex, field, components: 3, valueType: 'float' };
    }
    if (segments.length === 5 && (field === 'intensity' || field === 'range')) {
      return { kind: 'punctual-light', pointer, lightIndex, field, components: 1, valueType: 'float' };
    }
    if (segments.length === 6 && field === 'spot') {
      const coneField = segments[5];
      if (coneField === 'innerConeAngle' || coneField === 'outerConeAngle') {
        return {
          kind: 'punctual-light',
          pointer,
          lightIndex,
          field: coneField === 'innerConeAngle' ? 'spotInnerConeAngle' : 'spotOuterConeAngle',
          components: 1,
          valueType: 'float',
        };
      }
    }
  }

  return undefined;
}

export function supportedGltfAnimationPointers(): readonly string[] {
  const pointers = [
    ...supportedGltfMaterialAnimationPointers(),
    '/nodes/{index}/translation',
    '/nodes/{index}/rotation',
    '/nodes/{index}/scale',
    '/nodes/{index}/weights',
    '/nodes/{index}/weights/{element}',
    '/nodes/{index}/extensions/KHR_node_visibility/visible',
    '/cameras/{index}/perspective/yfov',
    '/cameras/{index}/perspective/znear',
    '/cameras/{index}/perspective/zfar',
    '/cameras/{index}/perspective/aspectRatio',
    '/cameras/{index}/orthographic/xmag',
    '/cameras/{index}/orthographic/ymag',
    '/cameras/{index}/orthographic/znear',
    '/cameras/{index}/orthographic/zfar',
    '/extensions/KHR_lights_punctual/lights/{index}/color',
    '/extensions/KHR_lights_punctual/lights/{index}/intensity',
    '/extensions/KHR_lights_punctual/lights/{index}/range',
    '/extensions/KHR_lights_punctual/lights/{index}/spot/innerConeAngle',
    '/extensions/KHR_lights_punctual/lights/{index}/spot/outerConeAngle',
  ];
  for (const path of Object.keys(MATERIAL_TEXTURE_SLOT_SPECS)) {
    pointers.push(`/materials/{index}/${path}/extensions/KHR_texture_transform/offset`);
    pointers.push(`/materials/{index}/${path}/extensions/KHR_texture_transform/rotation`);
    pointers.push(`/materials/{index}/${path}/extensions/KHR_texture_transform/scale`);
  }
  return pointers;
}

export function isGltfAnimationPointerTargetReachable(
  target: GltfAnimationPointerTarget,
  reachability: GltfAnimationPointerReachability,
): boolean {
  switch (target.kind) {
    case 'material-property':
    case 'material-texture-transform':
      return reachability.materialIndices.has(target.materialIndex);
    case 'node':
    case 'node-weight':
    case 'node-visibility':
      return reachability.nodeIndices.has(target.nodeIndex);
    case 'camera':
      return reachability.cameraIndices.has(target.cameraIndex);
    case 'punctual-light':
      return reachability.punctualLightIndices.has(target.lightIndex);
  }
}

export function gltfAnimationPointerTargetComponentCount(
  gltf: GltfJson,
  target: GltfAnimationPointerTarget,
): number | undefined {
  if (target.kind === 'node' && target.path === 'weights') {
    return nodeMorphTargetCount(gltf, target.nodeIndex);
  }
  return target.components === 'dynamic' ? undefined : target.components;
}

export function gltfNativeAnimationTargetIdentity(
  nodeIndex: number,
  path: string,
): GltfAnimationTargetIdentity {
  const key = `node:${nodeIndex}:${path}`;
  return path === 'weights' ? { key, arrayRoot: key, isWholeArray: true } : { key };
}

export function gltfAnimationPointerTargetIdentity(
  target: GltfAnimationPointerTarget,
): GltfAnimationTargetIdentity {
  switch (target.kind) {
    case 'node':
      return gltfNativeAnimationTargetIdentity(target.nodeIndex, target.path);
    case 'node-weight': {
      const arrayRoot = `node:${target.nodeIndex}:weights`;
      return { key: `${arrayRoot}:${target.weightIndex}`, arrayRoot, isWholeArray: false };
    }
    case 'node-visibility':
      return { key: `node:${target.nodeIndex}:visibility` };
    case 'material-property':
      return { key: `material:${target.materialIndex}:property:${target.target.field}` };
    case 'material-texture-transform':
      return { key: `material:${target.materialIndex}:texture:${target.texturePath}:${target.field}` };
    case 'camera':
      return { key: `camera:${target.cameraIndex}:${target.cameraType}:${target.field}` };
    case 'punctual-light':
      return { key: `light:${target.lightIndex}:${target.field}` };
  }
}

export function gltfAnimationTargetsConflict(
  existing: GltfAnimationTargetIdentity,
  candidate: GltfAnimationTargetIdentity,
): boolean {
  return existing.key === candidate.key || (
    existing.arrayRoot !== undefined &&
    existing.arrayRoot === candidate.arrayRoot &&
    (existing.isWholeArray === true || candidate.isWholeArray === true)
  );
}

export function gltfAnimationPointerOutputAccessorError(
  gltf: GltfJson,
  target: GltfAnimationPointerTarget,
  accessor: NonNullable<GltfJson['accessors']>[number] | undefined,
): string | undefined {
  if (!accessor) return 'the output accessor does not exist';
  const components = gltfAnimationPointerTargetComponentCount(gltf, target);
  if (components === undefined) return 'the target component count is undefined';
  const expectedType = components === 1
    ? 'SCALAR'
    : components === 2
      ? 'VEC2'
      : components === 3
        ? 'VEC3'
        : components === 4
          ? 'VEC4'
          : target.valueType === 'float-array'
            ? 'SCALAR'
            : undefined;
  const requiredAccessorType = target.valueType === 'float-array' ? 'SCALAR' : expectedType;
  if (requiredAccessorType === undefined || accessor.type !== requiredAccessorType) {
    return `expected ${requiredAccessorType ?? `${components}-component`} but found ${accessor.type}`;
  }
  if (target.valueType === 'boolean') {
    if (accessor.componentType !== GltfComponentType.UNSIGNED_BYTE || accessor.normalized === true) {
      return 'boolean targets require a non-normalized UNSIGNED_BYTE SCALAR accessor';
    }
  } else {
    // KHR_animation_pointer float* values may come from FLOAT, normalized
    // byte/short integers, or non-normalized integers. UNSIGNED_INT and FLOAT
    // cannot be normalized under the core accessor model.
    const normalizedTypes = new Set<number>([
      GltfComponentType.BYTE,
      GltfComponentType.UNSIGNED_BYTE,
      GltfComponentType.SHORT,
      GltfComponentType.UNSIGNED_SHORT,
    ]);
    if (accessor.normalized === true && !normalizedTypes.has(accessor.componentType)) {
      return 'normalized floating targets require BYTE, UNSIGNED_BYTE, SHORT, or UNSIGNED_SHORT components';
    }
  }
  return undefined;
}

/**
 * Returns a reason when the pointer names a known mutable path that is not
 * actually defined by this asset. KHR_animation_pointer requires either an
 * explicit property or a spec default whose enclosing object is present.
 */
export function gltfAnimationPointerTargetDefinitionError(
  gltf: GltfJson,
  target: GltfAnimationPointerTarget,
): string | undefined {
  switch (target.kind) {
    case 'node': {
      const node = gltf.nodes?.[target.nodeIndex];
      if (!node) return `node ${target.nodeIndex} does not exist`;
      if (target.path === 'weights' && nodeMorphTargetCount(gltf, target.nodeIndex) === undefined) {
        return `node ${target.nodeIndex} has no consistently defined morph targets`;
      }
      return undefined;
    }
    case 'node-weight': {
      const count = nodeMorphTargetCount(gltf, target.nodeIndex);
      if (count === undefined) return `node ${target.nodeIndex} has no consistently defined morph targets`;
      if (target.weightIndex >= count) {
        return `node ${target.nodeIndex} has ${count} morph weights, so element ${target.weightIndex} is out of bounds`;
      }
      return undefined;
    }
    case 'node-visibility': {
      const node = gltf.nodes?.[target.nodeIndex];
      if (!node) return `node ${target.nodeIndex} does not exist`;
      if (!isRecord(node.extensions?.KHR_node_visibility)) {
        return `nodes[${target.nodeIndex}].extensions.KHR_node_visibility is not defined`;
      }
      return undefined;
    }
    case 'camera': {
      const camera = gltf.cameras?.[target.cameraIndex];
      if (!camera || camera.type !== target.cameraType) {
        return `camera ${target.cameraIndex} is not a ${target.cameraType} camera`;
      }
      const projection = target.cameraType === 'perspective' ? camera.perspective : camera.orthographic;
      if (!isRecord(projection) || !hasOwn(projection, target.field)) {
        return `cameras[${target.cameraIndex}].${target.cameraType}.${target.field} is not defined`;
      }
      return undefined;
    }
    case 'punctual-light': {
      const light = punctualLightAt(gltf, target.lightIndex);
      if (!light) return `punctual light ${target.lightIndex} does not exist`;
      if (target.field === 'range' && !hasOwn(light, 'range')) {
        return `extensions.KHR_lights_punctual.lights[${target.lightIndex}].range is not defined`;
      }
      if (target.field === 'spotInnerConeAngle' || target.field === 'spotOuterConeAngle') {
        if (light.type !== 'spot' || !isRecord(light.spot)) {
          return `punctual light ${target.lightIndex} has no spot object`;
        }
      }
      return undefined;
    }
    case 'material-property': {
      const material = gltf.materials?.[target.materialIndex];
      if (!material) return `material ${target.materialIndex} does not exist`;
      return materialPropertyEnclosingObjectDefined(material, target.target.field)
        ? undefined
        : `the enclosing object for materials[${target.materialIndex}] field ${target.target.field} is not defined`;
    }
    case 'material-texture-transform': {
      const material = gltf.materials?.[target.materialIndex];
      if (!material) return `material ${target.materialIndex} does not exist`;
      const textureInfo = valueAtObjectPath(material, target.texturePath);
      if (!isRecord(textureInfo)) {
        return `materials[${target.materialIndex}].${target.texturePath.replaceAll('/', '.')} is not defined`;
      }
      const extensions = textureInfo.extensions;
      if (!isRecord(extensions) || !isRecord(extensions.KHR_texture_transform)) {
        return `the KHR_texture_transform object at materials[${target.materialIndex}].${target.texturePath.replaceAll('/', '.')} is not defined`;
      }
      return undefined;
    }
  }
}

export function gltfAnimationPointerValuesError(
  gltf: GltfJson,
  target: GltfAnimationPointerTarget,
  values: Float32Array,
  keyframeCount: number,
  interpolation: 'LINEAR' | 'STEP' | 'CUBICSPLINE',
): string | undefined {
  const components = gltfAnimationPointerTargetComponentCount(gltf, target);
  if (components === undefined || components <= 0) return 'the target component count is undefined';
  const cubic = interpolation === 'CUBICSPLINE';
  const stride = components * (cubic ? 3 : 1);
  const valueOffset = cubic ? components : 0;
  for (let keyframe = 0; keyframe < keyframeCount; keyframe += 1) {
    const offset = keyframe * stride + valueOffset;
    const value = values.subarray(offset, offset + components);
    for (const component of value) {
      if (!Number.isFinite(component)) return `keyframe ${keyframe} contains a non-finite value`;
    }
    const constraintError = pointerValueConstraintError(gltf, target, value);
    if (constraintError !== undefined) return `keyframe ${keyframe} ${constraintError}`;
  }
  return undefined;
}

/**
 * Validate one already-interpolated pointer value. Runtime callers use this
 * in addition to authored-key validation because blending and cubic curves can
 * otherwise manufacture values outside the property's normative schema.
 */
export function gltfAnimationPointerSampleValueError(
  gltf: GltfJson,
  target: GltfAnimationPointerTarget,
  value: Float32Array,
): string | undefined {
  const components = gltfAnimationPointerTargetComponentCount(gltf, target);
  if (components === undefined || components <= 0 || value.length !== components) {
    return `expected ${String(components)} components but received ${value.length}`;
  }
  for (const component of value) {
    if (!Number.isFinite(component)) return 'contains a non-finite value';
  }
  return pointerValueConstraintError(gltf, target, value);
}

/**
 * Returns a reason when interpolation itself cannot preserve the target
 * property's domain. Integer/bool STEP is normative. For bounded scalar/vector
 * properties, arbitrary CUBICSPLINE tangents can overshoot between valid keys;
 * the adapter fails closed rather than clamp sampled values.
 */
export function gltfAnimationPointerInterpolationError(
  target: GltfAnimationPointerTarget,
  interpolation: 'LINEAR' | 'STEP' | 'CUBICSPLINE',
): string | undefined {
  if (target.valueType === 'boolean' && interpolation !== 'STEP') {
    return `${target.valueType} properties require STEP interpolation`;
  }
  if (interpolation === 'CUBICSPLINE' && pointerHasBoundedDomain(target)) {
    return 'CUBICSPLINE cannot guarantee this bounded property remains inside its normative domain between keys';
  }
  return undefined;
}

function pointerValueConstraintError(
  gltf: GltfJson,
  target: GltfAnimationPointerTarget,
  value: Float32Array,
): string | undefined {
  if (target.kind === 'node' && target.path === 'rotation') {
    const magnitude = Math.hypot(value[0]!, value[1]!, value[2]!, value[3]!);
    if (Math.abs(magnitude - 1) > 1e-3) return 'must be a unit quaternion';
    return undefined;
  }
  if (target.kind === 'camera') {
    const scalar = value[0]!;
    if (target.field === 'yfov' && !(scalar > 0 && scalar < Math.PI)) return 'must be > 0 and < π';
    if (target.field === 'znear' && target.cameraType === 'orthographic' && scalar < 0) return 'must be >= 0';
    if (target.field === 'znear' && target.cameraType === 'perspective' && scalar <= 0) return 'must be > 0';
    if ((target.field === 'zfar' || target.field === 'aspectRatio' || target.field === 'xmag' || target.field === 'ymag') && scalar <= 0) {
      return 'must be > 0';
    }
    return undefined;
  }
  if (target.kind === 'punctual-light') {
    if (target.field === 'color' && !allInRange(value, 0, 1)) return 'must contain components in [0, 1]';
    const scalar = value[0]!;
    if (target.field === 'intensity' && scalar < 0) return 'must be >= 0';
    if (target.field === 'range' && scalar <= 0) return 'must be > 0';
    if (target.field === 'spotInnerConeAngle' && !(scalar >= 0 && scalar < Math.PI / 2)) {
      return 'must be >= 0 and < π/2';
    }
    if (target.field === 'spotOuterConeAngle' && !(scalar > 0 && scalar <= Math.PI / 2)) {
      return 'must be > 0 and <= π/2';
    }
    return undefined;
  }
  if (target.kind === 'node-visibility') {
    // KHR_animation_pointer defines UBYTE 0 as false and every nonzero value
    // as true, so the full 0..255 accessor domain is valid here.
    return undefined;
  }
  if (target.kind !== 'material-property') return undefined;
  const field = target.target.field;
  const scalar = value[0]!;
  if (
    field === 'baseColorFactor' || field === 'metallicFactor' || field === 'roughnessFactor' ||
    field === 'emissiveFactor' || field === 'aoMapIntensity' || field === 'transmissionFactor' ||
    field === 'attenuationColor' || field === 'specularFactor' || field === 'specularColorFactor' ||
    field === 'clearcoatFactor' || field === 'clearcoatRoughnessFactor' || field === 'sheenColorFactor' ||
    field === 'sheenRoughnessFactor' || field === 'iridescenceFactor' || field === 'anisotropyStrength'
  ) {
    if (!allInRange(value, 0, 1)) return 'must contain components in [0, 1]';
  }
  if (
    (field === 'alphaCutoff' || field === 'emissiveStrength' ||
      field === 'thicknessFactor' ||
      field === 'iridescenceThicknessMinimum' || field === 'iridescenceThicknessMaximum' ||
      field === 'dispersion') && scalar < 0
  ) return 'must be >= 0';
  if (field === 'attenuationDistance' && scalar <= 0) return 'must be > 0';
  if ((field === 'ior' || field === 'iridescenceIor') && scalar < 1) return 'must be >= 1';
  return undefined;
}

function pointerHasBoundedDomain(target: GltfAnimationPointerTarget): boolean {
  if (target.kind === 'camera' || target.kind === 'punctual-light' || target.kind === 'node-visibility') {
    return true;
  }
  if (target.kind !== 'material-property') return false;
  return target.target.field !== 'normalScale' &&
    target.target.field !== 'clearcoatNormalScale' &&
    target.target.field !== 'anisotropyRotation';
}

function nodeMorphTargetCount(gltf: GltfJson, nodeIndex: number): number | undefined {
  const node = gltf.nodes?.[nodeIndex];
  if (!node || node.mesh === undefined) return undefined;
  const mesh = gltf.meshes?.[node.mesh];
  if (!mesh || mesh.primitives.length === 0) return undefined;
  const counts = mesh.primitives.map((primitive) => primitive.targets?.length ?? 0);
  const count = counts[0] ?? 0;
  return count > 0 && counts.every((candidate) => candidate === count) ? count : undefined;
}

function materialPropertyEnclosingObjectDefined(
  material: GltfMaterial,
  field: GltfMaterialPointerTarget['field'],
): boolean {
  if (field === 'baseColorFactor' || field === 'metallicFactor' || field === 'roughnessFactor') {
    return isRecord(material.pbrMetallicRoughness);
  }
  if (field === 'normalScale') return isRecord(material.normalTexture);
  if (field === 'aoMapIntensity') return isRecord(material.occlusionTexture);
  if (field === 'emissiveFactor' || field === 'alphaCutoff') return true;
  const extensionByField: Partial<Record<GltfMaterialPointerTarget['field'], string>> = {
    emissiveStrength: 'KHR_materials_emissive_strength',
    transmissionFactor: 'KHR_materials_transmission',
    thicknessFactor: 'KHR_materials_volume',
    attenuationColor: 'KHR_materials_volume',
    attenuationDistance: 'KHR_materials_volume',
    ior: 'KHR_materials_ior',
    specularFactor: 'KHR_materials_specular',
    specularColorFactor: 'KHR_materials_specular',
    clearcoatFactor: 'KHR_materials_clearcoat',
    clearcoatRoughnessFactor: 'KHR_materials_clearcoat',
    clearcoatNormalScale: 'KHR_materials_clearcoat',
    sheenColorFactor: 'KHR_materials_sheen',
    sheenRoughnessFactor: 'KHR_materials_sheen',
    iridescenceFactor: 'KHR_materials_iridescence',
    iridescenceIor: 'KHR_materials_iridescence',
    iridescenceThicknessMinimum: 'KHR_materials_iridescence',
    iridescenceThicknessMaximum: 'KHR_materials_iridescence',
    anisotropyStrength: 'KHR_materials_anisotropy',
    anisotropyRotation: 'KHR_materials_anisotropy',
    dispersion: 'KHR_materials_dispersion',
  };
  const extensionName = extensionByField[field];
  const extension = extensionName === undefined ? undefined : material.extensions?.[extensionName];
  if (!isRecord(extension)) return false;
  return field !== 'clearcoatNormalScale' || isRecord(extension.clearcoatNormalTexture);
}

function punctualLightAt(gltf: GltfJson, lightIndex: number): Record<string, unknown> | undefined {
  const extension = gltf.extensions?.KHR_lights_punctual;
  if (!isRecord(extension) || !isUnknownArray(extension.lights)) return undefined;
  const light = extension.lights[lightIndex];
  return isRecord(light) ? light : undefined;
}

function valueAtObjectPath(value: unknown, path: string): unknown {
  let current = value;
  for (const segment of path.split('/')) {
    if (!isRecord(current)) return undefined;
    current = current[segment];
  }
  return current;
}

function allInRange(values: Float32Array, min: number, max: number): boolean {
  for (const value of values) if (value < min || value > max) return false;
  return true;
}

function hasOwn(value: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function isUnknownArray(value: unknown): value is readonly unknown[] {
  return Array.isArray(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function applyGltfMaterialAnimationPointerValue(
  material: MaterialSpec,
  target: GltfMaterialPropertyAnimationPointerTarget | GltfMaterialTextureTransformAnimationPointerTarget,
  value: Float32Array,
): MaterialSpec {
  return target.kind === 'material-property'
    ? applyGltfMaterialPointerValue(material, target.target, value)
    : applyGltfMaterialTextureTransformPointerValue(material, target, value);
}

export function applyGltfMaterialTextureTransformPointerValue(
  material: MaterialSpec,
  target: GltfMaterialTextureTransformAnimationPointerTarget,
  value: Float32Array,
): MaterialSpec {
  const result: Record<string, unknown> = { ...material };
  for (const materialField of target.materialFields) {
    const ref = material[materialField];
    if (!isTextureRef(ref)) continue;
    result[materialField] = applyTextureTransformPointerValue(ref, target.field, value);
  }
  return result as unknown as MaterialSpec;
}

function applyTextureTransformPointerValue(
  ref: TextureRef,
  field: GltfTextureTransformAnimationField,
  value: Float32Array,
): TextureRef {
  const transform = { ...(ref.transform ?? {}) };
  if (field === 'offset') {
    transform.offset = [value[0]!, value[1]!];
  } else if (field === 'scale') {
    transform.scale = [value[0]!, value[1]!];
  } else {
    transform.rotation = value[0]!;
  }
  return attachGltfTextureRefSource(
    { ...ref, transform },
    gltfTextureRefSource(ref),
  );
}

function isTextureRef(value: unknown): value is TextureRef {
  return value !== null && typeof value === 'object' && 'handle' in value;
}

function isTextureTransformField(value: string | undefined): value is GltfTextureTransformAnimationField {
  return value === 'offset' || value === 'rotation' || value === 'scale';
}

function decodePointerSegments(pointer: string): string[] | undefined {
  if (!pointer.startsWith('/')) return undefined;
  const decoded: string[] = [];
  for (const segment of pointer.slice(1).split('/')) {
    if (/~(?:[^01]|$)/.test(segment)) return undefined;
    decoded.push(segment.replace(/~1/g, '/').replace(/~0/g, '~'));
  }
  return decoded;
}

function parsePointerIndex(value: string | undefined): number | undefined {
  if (value === undefined || !/^(0|[1-9]\d*)$/.test(value)) return undefined;
  const index = Number(value);
  return Number.isSafeInteger(index) && index >= 0 ? index : undefined;
}
