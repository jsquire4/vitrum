/**
 * mesh.ts — THREE.Mesh → @vitrum/core MeshPrimitive converter.
 *
 * Validates material type, extracts geometry attributes, and delegates to the
 * material converter. Throws for unsupported mesh types and unsupported
 * materials — the caller (`sceneFromThreeJS`) should not silently skip meshes.
 */

import * as THREE from 'three';
import {
  asMat4,
  type InstancedMeshPrimitive,
  type MaterialSpec,
  type MeshPrimitive,
  type SkinnedMeshPrimitive,
  type Mat4,
  type SceneEmitter,
} from '@vitrum/core';
import { convertMaterial, convertBasicMaterial } from './material.js';
import { luminance } from './math.js';

/**
 * Detect emissive meshes that should be treated as area-light emitters.
 * This legacy single-mesh helper checks `material[0]` for material arrays;
 * `sceneFromThreeJS` uses the material-aware helper below for grouped
 * multi-material meshes. Callers strip the emissive contribution from the
 * corresponding MeshPrimitive material so emission is not double-counted.
 */
export function emissiveMeshAreaEmitter(mesh: THREE.Mesh): SceneEmitter | null {
  const rawMat = Array.isArray(mesh.material) ? mesh.material[0] ?? null : mesh.material;
  return emissiveMaterialAreaEmitter(rawMat, mesh.uuid);
}

/** Detect a sampled mesh-area light from the source THREE material. */
export function emissiveMaterialAreaEmitter(
  rawMat: THREE.Material | null,
  meshId: MeshPrimitive['id'],
): SceneEmitter | null {
  if (rawMat == null) return null;
  const asStd = rawMat as THREE.MeshStandardMaterial & { emissiveIntensity?: number };
  if (asStd.emissive == null) return null;
  const ei = asStd.emissiveIntensity ?? 1;
  const em = asStd.emissive;
  if (luminance(em.r, em.g, em.b, ei) < 1e-7) return null;
  return {
    kind: 'mesh-area',
    id: `mesh-emissive-${String(meshId)}`,
    meshId,
    color: [em.r, em.g, em.b],
    intensity: ei,
    castShadow: true,
  };
}

/** Returns a copy of `prim` with the emissive contribution zeroed so the
 *  same surface is not double-counted as both a path-traced emissive
 *  surface and a sampled area-light emitter. */
type MaterialBearingPrimitive = MeshPrimitive | InstancedMeshPrimitive | SkinnedMeshPrimitive;

export function stripEmissive<T extends MaterialBearingPrimitive>(prim: T): T {
  return {
    ...prim,
    material: { ...prim.material, emissive: [0, 0, 0], emissiveIntensity: 0 },
  } as T;
}

// ────────────────────────────────────────────────────────────────────────────
// Attribute extractors
// ────────────────────────────────────────────────────────────────────────────

interface FloatAttribute {
  readonly array: Float32Array;
  readonly itemSize: number;
}

/** Exact identity check for a 16-element column-major matrix. */
function isIdentityMat16(m: ArrayLike<number>): boolean {
  return (
    m[0] === 1 && m[5] === 1 && m[10] === 1 && m[15] === 1 &&
    m[1] === 0 && m[2] === 0 && m[3] === 0 &&
    m[4] === 0 && m[6] === 0 && m[7] === 0 &&
    m[8] === 0 && m[9] === 0 && m[11] === 0 &&
    m[12] === 0 && m[13] === 0 && m[14] === 0
  );
}

function extractFloatAttribute(
  geo: THREE.BufferGeometry,
  name: string,
): FloatAttribute | undefined {
  const attr = geo.getAttribute(name);
  if (attr == null) return undefined;
  const arr = attr.array;
  return {
    array: arr instanceof Float32Array ? arr : new Float32Array(arr),
    itemSize: attr.itemSize,
  };
}

function extractIndex(
  geo: THREE.BufferGeometry,
): Uint32Array | Uint16Array | undefined {
  const idx = geo.index;
  if (idx == null) return undefined;
  const arr = idx.array;
  if (arr instanceof Uint32Array || arr instanceof Uint16Array) return arr;
  return new Uint32Array(arr);
}

function requireFloatAttribute(
  geo: THREE.BufferGeometry,
  name: 'position' | 'normal',
  label: string,
  meshTypeName: string,
): FloatAttribute {
  const attr = extractFloatAttribute(geo, name);
  if (attr == null) {
    if (name === 'normal') {
      throw new Error(
        `${meshTypeName} "${label}" has no normal attribute. Compute normals before calling sceneFromThreeJS.`,
      );
    }
    throw new Error(`${meshTypeName} "${label}" has no position attribute.`);
  }
  return attr;
}

// ────────────────────────────────────────────────────────────────────────────
// Material narrowing — shared by all three mesh converters
// ────────────────────────────────────────────────────────────────────────────

/**
 * Validate + narrow the first material of a (possibly array) THREE material,
 * dispatching to `convertBasicMaterial` (unlit MeshBasicMaterial → flat
 * emissive) vs `convertMaterial` (MeshStandard/MeshPhysical). Throws on
 * unsupported/missing material types. `convertMaterial` handles MeshPhysical
 * internally, so MeshPhysical is accepted via the same `isStd`/`!isBasic` path
 * without a separate branch.
 *
 * @param meshTypeName THREE class name for the error prefix.
 * @param errorSubject optional subject inserted before "material" in the
 *   unsupported-type error (SkinnedMesh historically said "SkinnedMesh material").
 */
function convertFirstMaterial(
  rawMatOrArray: THREE.Material | THREE.Material[] | null,
  label: string,
  meshTypeName: MaterialConversionContext['meshTypeName'],
  errorSubject = '',
  options: MeshConversionOptions = {},
  context: Partial<MaterialConversionContext> = {},
) {
  const rawMat = Array.isArray(rawMatOrArray) ? rawMatOrArray[0] : rawMatOrArray;
  if (rawMat != null && options.materialConverter != null) {
    const converted = options.materialConverter(rawMat, {
      label,
      meshTypeName,
      ...context,
    });
    if (converted != null) return converted;
  }
  const isStd = (rawMat as THREE.MeshStandardMaterial | null)?.isMeshStandardMaterial === true;
  const isPhys = (rawMat as THREE.MeshPhysicalMaterial | null)?.isMeshPhysicalMaterial === true;
  // MeshBasicMaterial is the third accepted type. It renders unlit in three.js;
  // we synthesize a flat-emissive vitrum material so it appears as a self-lit
  // flat color regardless of scene lighting. Used by app-side overlay meshes
  // (panel mount preview, debug overlays, grid layers).
  const isBasic = (rawMat as THREE.MeshBasicMaterial | null)?.isMeshBasicMaterial === true;
  if (rawMat == null || (!isStd && !isPhys && !isBasic)) {
    const typeName = rawMat != null ? (rawMat as object).constructor.name : 'null';
    const subject = errorSubject !== '' ? `${errorSubject} ` : '';
    throw new Error(
      `Unsupported THREE type at "${label}": ${subject}material ${typeName}. Supported types are listed in the backend's EngineCapabilities.`,
    );
  }
  return isBasic
    ? convertBasicMaterial(rawMat as THREE.MeshBasicMaterial)
    : convertMaterial(rawMat as THREE.MeshStandardMaterial);
}

function convertMaterialAt(
  materials: readonly THREE.Material[],
  materialIndex: number,
  label: string,
  groupIndex: number,
  meshTypeName: MaterialConversionContext['meshTypeName'] = 'Mesh',
  options: MeshConversionOptions = {},
) {
  const rawMat = materials[materialIndex] ?? null;
  return convertFirstMaterial(
    rawMat,
    `${label} group ${groupIndex}`,
    meshTypeName,
    '',
    options,
    { groupIndex, materialIndex },
  );
}

export interface MaterialConversionContext {
  /** Human-readable object/group label used in adapter diagnostics. */
  readonly label: string;
  /** THREE renderable kind currently being converted. */
  readonly meshTypeName: 'Mesh' | 'InstancedMesh' | 'SkinnedMesh';
  /** Geometry group index when converting grouped multi-material geometry. */
  readonly groupIndex?: number;
  /** Material-array slot selected for the current primitive/group. */
  readonly materialIndex?: number;
}

export type ThreeMaterialConverter = (
  material: THREE.Material,
  context: MaterialConversionContext,
) => MaterialSpec | null | undefined;

export interface MeshConversionOptions {
  readonly materialConverter?: ThreeMaterialConverter;
}

interface InternalMeshConversionOptions extends MeshConversionOptions {
  readonly suppressMultiMaterialWarning?: boolean;
}

// ────────────────────────────────────────────────────────────────────────────
// Mesh converter
// ────────────────────────────────────────────────────────────────────────────

interface MeshAttributeSet {
  readonly positions: FloatAttribute;
  readonly normals: FloatAttribute;
  readonly uvs?: FloatAttribute;
  readonly uv1?: FloatAttribute;
  readonly tangents?: FloatAttribute;
  readonly colors?: FloatAttribute;
  readonly indices?: Uint32Array | Uint16Array;
}

function extractMeshAttributeSet(
  geo: THREE.BufferGeometry,
  label: string,
  meshTypeName: string,
): MeshAttributeSet {
  const uvs = extractFloatAttribute(geo, 'uv');
  const uv1 = extractFloatAttribute(geo, 'uv1') ?? extractFloatAttribute(geo, 'uv2');
  const tangents = extractFloatAttribute(geo, 'tangent');
  const colors = extractFloatAttribute(geo, 'color');
  const indices = extractIndex(geo);

  return {
    positions: requireFloatAttribute(geo, 'position', label, meshTypeName),
    normals: requireFloatAttribute(geo, 'normal', label, meshTypeName),
    ...(uvs != null ? { uvs } : {}),
    // THREE r152+ names the 2nd UV set 'uv1'; older geometry used 'uv2'.
    ...(uv1 != null ? { uv1 } : {}),
    ...(tangents != null ? { tangents } : {}),
    ...(colors != null ? { colors } : {}),
    ...(indices != null ? { indices } : {}),
  };
}

function buildMeshPrimitive(
  id: string,
  attrs: MeshAttributeSet,
  transform: Mat4,
  material: MeshPrimitive['material'],
): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: attrs.positions.array,
    normals: attrs.normals.array,
    transform,
    material,
    ...(attrs.uvs != null ? { uvs: attrs.uvs.array } : {}),
    ...(attrs.uv1 != null ? { uv1: attrs.uv1.array } : {}),
    ...(attrs.tangents != null ? { tangents: attrs.tangents.array } : {}),
    ...(attrs.colors != null ? { colors: attrs.colors.array } : {}),
    ...(attrs.indices != null ? { indices: attrs.indices } : {}),
  };
}

export function convertMesh(obj: THREE.Mesh, options: MeshConversionOptions = {}): MeshPrimitive {
  const geo = obj.geometry;
  const label = obj.name || obj.uuid;

  const attrs = extractMeshAttributeSet(geo, label, 'Mesh');
  const transform = new Float32Array(obj.matrixWorld.elements) as Mat4;

  // Multi-material meshes: this legacy single-primitive helper falls back to
  // material[0]. sceneFromThreeJS uses convertMeshToPrimitives for group-aware
  // expansion without this warning.
  if (Array.isArray(obj.material) && obj.material.length > 1) {
    console.warn(
      `@vitrum/three-bindings: convertMesh() received a multi-material Mesh at "${label}" (${obj.material.length} materials). ` +
      `This legacy single-primitive helper uses material[0]; use sceneFromThreeJS() for grouped multi-material expansion.`,
    );
  }

  const material = convertFirstMaterial(obj.material, label, 'Mesh', '', options);

  return buildMeshPrimitive(obj.uuid, attrs, transform, material);
}

function validateGroupRange(
  group: THREE.BufferGeometry['groups'][number],
  groupIndex: number,
  label: string,
  limit: number,
  rangeKind: 'index' | 'vertex',
): void {
  if (!Number.isInteger(group.start) || !Number.isInteger(group.count) || group.start < 0 || group.count < 0) {
    throw new Error(
      `Mesh "${label}" group ${groupIndex} has invalid ${rangeKind} range start=${group.start}, count=${group.count}.`,
    );
  }
  if (group.start + group.count > limit) {
    throw new Error(
      `Mesh "${label}" group ${groupIndex} ${rangeKind} range [${group.start}, ${group.start + group.count}) exceeds ${rangeKind} count ${limit}.`,
    );
  }
}

function materialIndexForGroup(
  group: THREE.BufferGeometry['groups'][number],
  groupIndex: number,
  materialCount: number,
  label: string,
): number {
  const materialIndex = group.materialIndex ?? 0;
  if (!Number.isInteger(materialIndex) || materialIndex < 0 || materialIndex >= materialCount) {
    throw new Error(
      `Mesh "${label}" group ${groupIndex} references material index ${materialIndex}, but the mesh has ${materialCount} materials.`,
    );
  }
  return materialIndex;
}

function vertexIndicesForGroup(
  group: THREE.BufferGeometry['groups'][number],
  groupIndex: number,
  label: string,
  attrs: MeshAttributeSet,
): Uint32Array {
  const vertexCount = Math.floor(attrs.positions.array.length / attrs.positions.itemSize);
  if (attrs.indices != null) {
    validateGroupRange(group, groupIndex, label, attrs.indices.length, 'index');
    const out = new Uint32Array(group.count);
    for (let i = 0; i < group.count; i += 1) {
      const vertexIndex = attrs.indices[group.start + i]!;
      if (vertexIndex >= vertexCount) {
        throw new Error(
          `Mesh "${label}" group ${groupIndex} references vertex ${vertexIndex}, but the position attribute has ${vertexCount} vertices.`,
        );
      }
      out[i] = vertexIndex;
    }
    return out;
  }

  validateGroupRange(group, groupIndex, label, vertexCount, 'vertex');
  const out = new Uint32Array(group.count);
  for (let i = 0; i < group.count; i += 1) out[i] = group.start + i;
  return out;
}

function copyAttributeForVertices(attr: FloatAttribute, vertexIndices: Uint32Array): Float32Array {
  const out = new Float32Array(vertexIndices.length * attr.itemSize);
  for (let outVertex = 0; outVertex < vertexIndices.length; outVertex += 1) {
    const srcOffset = vertexIndices[outVertex]! * attr.itemSize;
    const dstOffset = outVertex * attr.itemSize;
    for (let component = 0; component < attr.itemSize; component += 1) {
      out[dstOffset + component] = attr.array[srcOffset + component] ?? 0;
    }
  }
  return out;
}

function copyFloat32ComponentsForVertices(
  src: Float32Array,
  vertexIndices: Uint32Array,
  itemSize: number,
): Float32Array {
  const out = new Float32Array(vertexIndices.length * itemSize);
  for (let outVertex = 0; outVertex < vertexIndices.length; outVertex += 1) {
    const srcOffset = vertexIndices[outVertex]! * itemSize;
    const dstOffset = outVertex * itemSize;
    for (let component = 0; component < itemSize; component += 1) {
      out[dstOffset + component] = src[srcOffset + component] ?? 0;
    }
  }
  return out;
}

function copyUint32ComponentsForVertices(
  src: Uint32Array,
  vertexIndices: Uint32Array,
  itemSize: number,
): Uint32Array {
  const out = new Uint32Array(vertexIndices.length * itemSize);
  for (let outVertex = 0; outVertex < vertexIndices.length; outVertex += 1) {
    const srcOffset = vertexIndices[outVertex]! * itemSize;
    const dstOffset = outVertex * itemSize;
    for (let component = 0; component < itemSize; component += 1) {
      out[dstOffset + component] = src[srcOffset + component] ?? 0;
    }
  }
  return out;
}

function validateSkinAttributeShapes(
  label: string,
  positions: FloatAttribute,
  normals: FloatAttribute,
  skinIndexAttr: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
  skinWeightAttr: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
): number {
  if (positions.itemSize !== 3) {
    throw new Error(`SkinnedMesh "${label}" position attribute itemSize ${positions.itemSize}; expected 3.`);
  }
  if (normals.itemSize !== 3) {
    throw new Error(`SkinnedMesh "${label}" normal attribute itemSize ${normals.itemSize}; expected 3.`);
  }
  const vertexCount = positions.array.length / 3;
  if (!Number.isInteger(vertexCount)) {
    throw new Error(`SkinnedMesh "${label}" position attribute length ${positions.array.length} is not divisible by 3.`);
  }
  if (skinIndexAttr.itemSize !== 4) {
    throw new Error(`SkinnedMesh "${label}" skinIndex attribute itemSize ${skinIndexAttr.itemSize}; expected 4.`);
  }
  if (skinWeightAttr.itemSize !== 4) {
    throw new Error(`SkinnedMesh "${label}" skinWeight attribute itemSize ${skinWeightAttr.itemSize}; expected 4.`);
  }
  if (skinIndexAttr.array.length !== vertexCount * 4) {
    throw new Error(
      `SkinnedMesh "${label}" skinIndex length ${skinIndexAttr.array.length}; expected ${vertexCount * 4}.`,
    );
  }
  if (skinWeightAttr.array.length !== vertexCount * 4) {
    throw new Error(
      `SkinnedMesh "${label}" skinWeight length ${skinWeightAttr.array.length}; expected ${vertexCount * 4}.`,
    );
  }
  return vertexCount;
}

function validateSkinPayloadValues(
  label: string,
  skinIndexArray: ArrayLike<number>,
  skinWeightArray: ArrayLike<number>,
  boneCount: number,
): void {
  for (let i = 0; i < skinIndexArray.length; i += 1) {
    const idx = skinIndexArray[i]!;
    if (!Number.isInteger(idx) || idx < 0 || idx >= boneCount) {
      throw new Error(
        `SkinnedMesh "${label}" skinIndex[${i}] references bone ${idx}; skeleton has ${boneCount} bones.`,
      );
    }
  }
  for (let i = 0; i < skinWeightArray.length; i += 1) {
    const weight = skinWeightArray[i]!;
    if (!Number.isFinite(weight) || weight < 0) {
      throw new Error(`SkinnedMesh "${label}" skinWeight[${i}] is invalid (${weight}).`);
    }
  }
}

function sliceMeshAttributesForGroup(
  attrs: MeshAttributeSet,
  vertexIndices: Uint32Array,
): MeshAttributeSet {
  return {
    positions: {
      array: copyAttributeForVertices(attrs.positions, vertexIndices),
      itemSize: attrs.positions.itemSize,
    },
    normals: {
      array: copyAttributeForVertices(attrs.normals, vertexIndices),
      itemSize: attrs.normals.itemSize,
    },
    ...(attrs.uvs != null
      ? {
          uvs: {
            array: copyAttributeForVertices(attrs.uvs, vertexIndices),
            itemSize: attrs.uvs.itemSize,
          },
        }
      : {}),
    ...(attrs.uv1 != null
      ? {
          uv1: {
            array: copyAttributeForVertices(attrs.uv1, vertexIndices),
            itemSize: attrs.uv1.itemSize,
          },
        }
      : {}),
    ...(attrs.tangents != null
      ? {
          tangents: {
            array: copyAttributeForVertices(attrs.tangents, vertexIndices),
            itemSize: attrs.tangents.itemSize,
          },
        }
      : {}),
    ...(attrs.colors != null
      ? {
          colors: {
            array: copyAttributeForVertices(attrs.colors, vertexIndices),
            itemSize: attrs.colors.itemSize,
          },
        }
      : {}),
  };
}

/**
 * Convert a normal THREE.Mesh into one or more vitrum MeshPrimitives.
 *
 * When THREE geometry groups and a material array are present, each group is
 * materialized as a compact triangle-list primitive so backends do not need a
 * new draw-range field in the core contract. convertMesh remains the legacy
 * single-primitive helper for callers/tests that rely on that shape.
 */
export function convertMeshToPrimitives(obj: THREE.Mesh, options: MeshConversionOptions = {}): MeshPrimitive[] {
  const materials = obj.material;
  if (!Array.isArray(materials) || obj.geometry.groups.length === 0) {
    return [convertMesh(obj, options)];
  }

  const geo = obj.geometry;
  const label = obj.name || obj.uuid;
  const attrs = extractMeshAttributeSet(geo, label, 'Mesh');
  const transform = new Float32Array(obj.matrixWorld.elements) as Mat4;

  return geo.groups.map((group, groupIndex): MeshPrimitive => {
    const materialIndex = materialIndexForGroup(group, groupIndex, materials.length, label);
    const material = convertMaterialAt(materials, materialIndex, label, groupIndex, 'Mesh', options);
    const vertexIndices = vertexIndicesForGroup(group, groupIndex, label, attrs);
    const groupAttrs = sliceMeshAttributesForGroup(attrs, vertexIndices);
    return buildMeshPrimitive(
      `${obj.uuid}:group:${groupIndex}:material:${materialIndex}`,
      groupAttrs,
      transform,
      material,
    );
  });
}

// ────────────────────────────────────────────────────────────────────────────
// InstancedMesh converter — multi-mesh TLAS production path
// ────────────────────────────────────────────────────────────────────────────

export function convertInstancedMesh(
  obj: THREE.InstancedMesh,
  options: MeshConversionOptions = {},
): InstancedMeshPrimitive {
  const geo = obj.geometry;
  const label = obj.name || obj.uuid;

  const attrs = extractMeshAttributeSet(geo, label, 'InstancedMesh');

  if (Array.isArray(obj.material) && obj.material.length > 1) {
    console.warn(
      `@vitrum/three-bindings: convertInstancedMesh() received a multi-material InstancedMesh at "${label}" (${obj.material.length} materials). ` +
        `This legacy single-primitive helper uses material[0]; use sceneFromThreeJS() for grouped multi-material expansion.`,
    );
  }

  const material = convertFirstMaterial(obj.material, label, 'InstancedMesh', '', options);
  const instances = extractInstancedTransforms(obj);

  return buildInstancedMeshPrimitive(obj.uuid, attrs, material, instances);
}

function extractInstancedTransforms(obj: THREE.InstancedMesh): Mat4[] {
  const instances: Mat4[] = [];
  const tmp = new THREE.Matrix4();
  for (let i = 0; i < obj.count; i += 1) {
    obj.getMatrixAt(i, tmp);
    tmp.premultiply(obj.matrixWorld);
    instances.push(asMat4(new Float32Array(tmp.elements)));
  }
  return instances;
}

function buildInstancedMeshPrimitive(
  id: string,
  attrs: MeshAttributeSet,
  material: InstancedMeshPrimitive['material'],
  instances: Mat4[],
): InstancedMeshPrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: attrs.positions.array,
    normals: attrs.normals.array,
    material,
    instances,
    ...(attrs.uvs != null ? { uvs: attrs.uvs.array } : {}),
    ...(attrs.uv1 != null ? { uv1: attrs.uv1.array } : {}),
    ...(attrs.tangents != null ? { tangents: attrs.tangents.array } : {}),
    ...(attrs.colors != null ? { colors: attrs.colors.array } : {}),
    ...(attrs.indices != null ? { indices: attrs.indices } : {}),
  };
}

/**
 * Convert a THREE.InstancedMesh into one or more vitrum InstancedMeshPrimitives.
 *
 * Grouped multi-material instanced geometry mirrors the normal mesh expansion:
 * each material group becomes a compact triangle-list primitive, all sharing
 * the source instance transforms. Direct `convertInstancedMesh()` remains the
 * legacy single-primitive helper for callers that rely on that exact shape.
 */
export function convertInstancedMeshToPrimitives(
  obj: THREE.InstancedMesh,
  options: MeshConversionOptions = {},
): InstancedMeshPrimitive[] {
  const materials = obj.material;
  if (!Array.isArray(materials) || obj.geometry.groups.length === 0) {
    return [convertInstancedMesh(obj, options)];
  }

  const geo = obj.geometry;
  const label = obj.name || obj.uuid;
  const attrs = extractMeshAttributeSet(geo, label, 'InstancedMesh');
  const instances = extractInstancedTransforms(obj);

  return geo.groups.map((group, groupIndex): InstancedMeshPrimitive => {
    const materialIndex = materialIndexForGroup(group, groupIndex, materials.length, label);
    const material = convertMaterialAt(materials, materialIndex, label, groupIndex, 'InstancedMesh', options);
    const vertexIndices = vertexIndicesForGroup(group, groupIndex, label, attrs);
    const groupAttrs = sliceMeshAttributesForGroup(attrs, vertexIndices);
    return buildInstancedMeshPrimitive(
      `${obj.uuid}:group:${groupIndex}:material:${materialIndex}`,
      groupAttrs,
      material,
      instances,
    );
  });
}

// ────────────────────────────────────────────────────────────────────────────
// SkinnedMesh converter — C1 (2026-05-19)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Convert a THREE.SkinnedMesh into a SkinnedMeshPrimitive.
 *
 * Captures the rest-pose geometry + the current pose's bone matrices.
 * Per-frame pose changes are pushed via `HybridEngine.updatePrimitive`
 * (host responsibility — call `obj.skeleton.update()` first, then re-extract
 * `bones` and submit). See `SkinnedMeshPrimitive` JSDoc in @vitrum/core for
 * the deformation formula.
 */
export function convertSkinnedMesh(
  obj: THREE.SkinnedMesh,
  options: MeshConversionOptions = {},
): SkinnedMeshPrimitive {
  return convertSkinnedMeshInternal(obj, options);
}

function convertSkinnedMeshInternal(
  obj: THREE.SkinnedMesh,
  options: InternalMeshConversionOptions = {},
): SkinnedMeshPrimitive {
  const geo = obj.geometry;
  const label = obj.name || obj.uuid;

  const attrs = extractMeshAttributeSet(geo, label, 'SkinnedMesh');
  const positions = attrs.positions.array;
  const normals = attrs.normals.array;

  // SkinnedMesh requires skinIndex + skinWeight attributes per glTF 2.0.
  const skinIndexAttr = geo.getAttribute('skinIndex');
  if (skinIndexAttr == null) {
    throw new Error(`SkinnedMesh "${label}" has no skinIndex attribute.`);
  }
  const skinWeightAttr = geo.getAttribute('skinWeight');
  if (skinWeightAttr == null) {
    throw new Error(`SkinnedMesh "${label}" has no skinWeight attribute.`);
  }
  validateSkinAttributeShapes(label, attrs.positions, attrs.normals, skinIndexAttr, skinWeightAttr);
  // Widen skinIndex (typically Uint16Array) to Uint32Array for the contract.
  // The narrower-typed array is upcasted without loss; downstream skinning
  // solvers index a Float32Array bones buffer with these values.
  const skinIndices = new Uint32Array(skinIndexAttr.array);
  // skinWeight may be Float32Array already; coerce if not.
  const skinWeights = skinWeightAttr.array instanceof Float32Array
    ? skinWeightAttr.array
    : new Float32Array(skinWeightAttr.array);

  // Skeleton: flatten per-bone matrices into single Float32Arrays.
  const skel = obj.skeleton;
  if (skel == null) {
    throw new Error(`SkinnedMesh "${label}" has no skeleton.`);
  }
  const boneCount = skel.bones.length;
  if (boneCount === 0) {
    throw new Error(`SkinnedMesh "${label}" has an empty skeleton (0 bones).`);
  }
  if (skel.boneInverses.length !== boneCount) {
    throw new Error(
      `SkinnedMesh "${label}" skeleton has ${boneCount} bones but ${skel.boneInverses.length} inverse-bind matrices.`,
    );
  }
  validateSkinPayloadValues(label, skinIndexAttr.array, skinWeightAttr.array, boneCount);
  // Ensure the bone matrices reflect the current scene hierarchy.
  for (const bone of skel.bones) {
    bone.updateMatrixWorld(true);
  }
  const bones = new Float32Array(boneCount * 16);
  const boneInverses = new Float32Array(boneCount * 16);
  for (let i = 0; i < boneCount; i++) {
    const boneMat = skel.bones[i]!.matrixWorld.elements;     // column-major 16 f32s
    const invMat  = skel.boneInverses[i]!.elements;
    for (let k = 0; k < 16; k++) {
      bones[i * 16 + k] = boneMat[k]!;
      boneInverses[i * 16 + k] = invMat[k]!;
    }
  }

  // Mesh bind matrices: identity for a mesh bound at the origin, but the
  // pair must be emitted when EITHER is non-identity. In THREE's 'attached'
  // bindMode, `bindMatrixInverse = inverse(matrixWorld)` even while
  // `bindMatrix` stays identity — that inverse is exactly the term that
  // cancels the node transform back out of the world-space bone matrices,
  // making the solveSkin output MESH-LOCAL so consumers apply `transform`
  // exactly once. (Checking only bindMatrix here used to drop the pair,
  // so solveSkin emitted world-space positions and every consumer that
  // applied `transform` on top double-transformed any skinned mesh whose
  // node carries a non-identity world matrix.)
  const bmIdentity = isIdentityMat16(obj.bindMatrix.elements);
  const bmiIdentity = isIdentityMat16(obj.bindMatrixInverse.elements);
  const emitBind = !(bmIdentity && bmiIdentity);
  const bindMatrix = emitBind ? new Float32Array(obj.bindMatrix.elements) : undefined;
  const bindMatrixInverse = emitBind
    ? new Float32Array(obj.bindMatrixInverse.elements)
    : undefined;

  const transform = new Float32Array(obj.matrixWorld.elements) as Mat4;

  // ── Morph targets ────────────────────────────────────────────────────────
  // glTF default (three.js `morphTargetsRelative === true`) → already deltas.
  // Absolute mode → subtract `positions` to obtain a delta.
  let morphTargets: Float32Array[] | undefined;
  let morphTargetNormals: Float32Array[] | undefined;
  let morphWeights: Float32Array | undefined;
  const posMorphs = geo.morphAttributes.position;
  if (posMorphs != null && posMorphs.length > 0) {
    const relative = geo.morphTargetsRelative;
    morphTargets = [];
    for (const attr of posMorphs) {
      const arr = attr.array instanceof Float32Array
        ? new Float32Array(attr.array)
        : new Float32Array(attr.array);
      if (!relative) {
        // Absolute → delta: arr[i] - positions[i]
        for (let i = 0; i < arr.length; i++) arr[i] = arr[i]! - positions[i]!;
      }
      morphTargets.push(arr);
    }
    const normMorphs = geo.morphAttributes.normal;
    if (normMorphs != null && normMorphs.length === posMorphs.length) {
      morphTargetNormals = [];
      for (const attr of normMorphs) {
        const arr = attr.array instanceof Float32Array
          ? new Float32Array(attr.array)
          : new Float32Array(attr.array);
        if (!relative) {
          for (let i = 0; i < arr.length; i++) arr[i] = arr[i]! - normals[i]!;
        }
        morphTargetNormals.push(arr);
      }
    }
    const influences = obj.morphTargetInfluences;
    morphWeights = influences != null
      ? new Float32Array(influences)
      : new Float32Array(posMorphs.length);   // default all zero
  }

  // Multi-material handling mirrors convertMesh.
  if (
    Array.isArray(obj.material) &&
    obj.material.length > 1 &&
    options.suppressMultiMaterialWarning !== true
  ) {
    console.warn(
      `@vitrum/three-bindings: convertSkinnedMesh() received a multi-material SkinnedMesh at "${label}" (${obj.material.length} materials). ` +
      `This legacy single-primitive helper uses material[0]; use sceneFromThreeJS() for grouped multi-material expansion.`,
    );
  }
  const material = convertFirstMaterial(obj.material, label, 'SkinnedMesh', 'SkinnedMesh', options);

  return {
    kind: 'skinned-mesh',
    id: obj.uuid,
    positions,
    normals,
    skinIndices,
    skinWeights,
    bones,
    boneInverses,
    transform,
    material,
    ...(attrs.uvs != null ? { uvs: attrs.uvs.array } : {}),
    ...(attrs.uv1 != null ? { uv1: attrs.uv1.array } : {}),
    ...(attrs.tangents != null ? { tangents: attrs.tangents.array } : {}),
    ...(attrs.colors != null ? { colors: attrs.colors.array } : {}),
    ...(attrs.indices != null ? { indices: attrs.indices } : {}),
    ...(morphTargets != null ? { morphTargets } : {}),
    ...(morphTargetNormals != null ? { morphTargetNormals } : {}),
    ...(morphWeights != null ? { morphWeights } : {}),
    ...(bindMatrix != null ? { bindMatrix } : {}),
    ...(bindMatrixInverse != null ? { bindMatrixInverse } : {}),
  };
}

/**
 * Convert a THREE.SkinnedMesh into one or more vitrum SkinnedMeshPrimitives.
 *
 * Grouped multi-material skinned geometry mirrors the static Mesh expansion:
 * each group becomes a compact skinned primitive with sliced rest geometry,
 * skin indices/weights, and morph deltas, while sharing the same skeleton,
 * bind matrices, morph weights, and world transform.
 */
export function convertSkinnedMeshToPrimitives(
  obj: THREE.SkinnedMesh,
  options: MeshConversionOptions = {},
): SkinnedMeshPrimitive[] {
  const materials = obj.material;
  if (!Array.isArray(materials) || obj.geometry.groups.length === 0) {
    return [convertSkinnedMesh(obj, options)];
  }

  const geo = obj.geometry;
  const label = obj.name || obj.uuid;
  const attrs = extractMeshAttributeSet(geo, label, 'SkinnedMesh');
  const base = convertSkinnedMeshInternal(obj, {
    ...options,
    suppressMultiMaterialWarning: true,
  });

  return geo.groups.map((group, groupIndex): SkinnedMeshPrimitive => {
    const materialIndex = materialIndexForGroup(group, groupIndex, materials.length, label);
    const material = convertMaterialAt(materials, materialIndex, label, groupIndex, 'SkinnedMesh', options);
    const vertexIndices = vertexIndicesForGroup(group, groupIndex, label, attrs);
    const groupAttrs = sliceMeshAttributesForGroup(attrs, vertexIndices);
    const {
      id: _id,
      positions: _positions,
      normals: _normals,
      uvs: _uvs,
      uv1: _uv1,
      tangents: _tangents,
      colors: _colors,
      indices: _indices,
      skinIndices: _skinIndices,
      skinWeights: _skinWeights,
      morphTargets: _morphTargets,
      morphTargetNormals: _morphTargetNormals,
      material: _material,
      ...shared
    } = base;
    return {
      ...shared,
      id: `${obj.uuid}:group:${groupIndex}:material:${materialIndex}`,
      positions: groupAttrs.positions.array,
      normals: groupAttrs.normals.array,
      material,
      skinIndices: copyUint32ComponentsForVertices(base.skinIndices, vertexIndices, 4),
      skinWeights: copyFloat32ComponentsForVertices(base.skinWeights, vertexIndices, 4),
      ...(groupAttrs.uvs != null ? { uvs: groupAttrs.uvs.array } : {}),
      ...(groupAttrs.uv1 != null ? { uv1: groupAttrs.uv1.array } : {}),
      ...(groupAttrs.tangents != null ? { tangents: groupAttrs.tangents.array } : {}),
      ...(groupAttrs.colors != null ? { colors: groupAttrs.colors.array } : {}),
      ...(base.morphTargets != null
        ? { morphTargets: base.morphTargets.map((target) => copyFloat32ComponentsForVertices(target, vertexIndices, 3)) }
        : {}),
      ...(base.morphTargetNormals != null
        ? { morphTargetNormals: base.morphTargetNormals.map((target) => copyFloat32ComponentsForVertices(target, vertexIndices, 3)) }
        : {}),
    };
  });
}
