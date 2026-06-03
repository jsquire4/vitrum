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
  type MeshPrimitive,
  type SkinnedMeshPrimitive,
  type Mat4,
  type SceneEmitter,
} from '@vitrum/core';
import { convertMaterial, convertBasicMaterial } from './material.js';
import { luminance } from './math.js';

/**
 * Detect emissive meshes that should be treated as area-light emitters.
 * Returns a SceneEmitter when the mesh's material has non-zero emissive
 * luminance; null otherwise. Callers strip the emissive contribution from
 * the corresponding MeshPrimitive material so emission is not double-counted.
 */
export function emissiveMeshAreaEmitter(mesh: THREE.Mesh): SceneEmitter | null {
  const rawMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  if (rawMat == null) return null;
  const asStd = rawMat as THREE.MeshStandardMaterial & { emissiveIntensity?: number };
  if (asStd.emissive == null) return null;
  const ei = asStd.emissiveIntensity ?? 1;
  const em = asStd.emissive;
  if (luminance(em.r, em.g, em.b, ei) < 1e-7) return null;
  return {
    kind: 'mesh-area',
    id: `mesh-emissive-${mesh.uuid}`,
    meshId: mesh.uuid,
    color: [em.r, em.g, em.b],
    intensity: ei,
    castShadow: true,
  };
}

/** Returns a copy of `prim` with the emissive contribution zeroed so the
 *  same surface is not double-counted as both a path-traced emissive
 *  surface and a sampled area-light emitter. */
export function stripEmissive(prim: MeshPrimitive): MeshPrimitive {
  return {
    ...prim,
    material: { ...prim.material, emissive: [0, 0, 0], emissiveIntensity: 0 },
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Attribute extractors
// ────────────────────────────────────────────────────────────────────────────

function extractAttribute(
  geo: THREE.BufferGeometry,
  name: string,
): Float32Array | undefined {
  const attr = geo.getAttribute(name);
  if (attr == null) return undefined;
  const arr = attr.array;
  if (arr instanceof Float32Array) return arr;
  return new Float32Array(arr);
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

/**
 * Extract a required vertex attribute (position/normal), throwing a converter
 * error labelled with the mesh type when it is missing. `label` is the
 * caller-facing mesh identifier; `meshTypeName` is the THREE class name used in
 * the error prefix (e.g. "Mesh", "InstancedMesh", "SkinnedMesh").
 */
function requireAttribute(
  geo: THREE.BufferGeometry,
  name: 'position' | 'normal',
  label: string,
  meshTypeName: string,
): Float32Array {
  const arr = extractAttribute(geo, name);
  if (arr == null) {
    if (name === 'normal') {
      throw new Error(
        `${meshTypeName} "${label}" has no normal attribute. Compute normals before calling sceneFromThreeJS.`,
      );
    }
    throw new Error(`${meshTypeName} "${label}" has no position attribute.`);
  }
  return arr;
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
  meshTypeName: string,
  errorSubject = '',
) {
  const rawMat = Array.isArray(rawMatOrArray) ? rawMatOrArray[0] : rawMatOrArray;
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

// ────────────────────────────────────────────────────────────────────────────
// Mesh converter
// ────────────────────────────────────────────────────────────────────────────

export function convertMesh(obj: THREE.Mesh): MeshPrimitive {
  const geo = obj.geometry;
  const label = obj.name || obj.uuid;

  const positions = requireAttribute(geo, 'position', label, 'Mesh');
  const normals = requireAttribute(geo, 'normal', label, 'Mesh');

  const uvs = extractAttribute(geo, 'uv');
  // THREE r152+ names the 2nd UV set 'uv1'; older geometry used 'uv2'.
  const uv1 = extractAttribute(geo, 'uv1') ?? extractAttribute(geo, 'uv2');
  const tangents = extractAttribute(geo, 'tangent');
  const colors = extractAttribute(geo, 'color');
  const indices = extractIndex(geo);

  const transform = new Float32Array(obj.matrixWorld.elements) as Mat4;

  // Multi-material meshes: warn and fall back to first material.
  // Geometry-group splitting (each group gets its own material) is a future enhancement.
  if (Array.isArray(obj.material) && obj.material.length > 1) {
    console.warn(
      `@vitrum/three-bindings: unsupported multi-material mesh at "${label}" (${obj.material.length} materials). ` +
      `Only the first material will be used. Supported types are listed in the backend's EngineCapabilities.`,
    );
  }

  const material = convertFirstMaterial(obj.material, label, 'Mesh');

  return {
    kind: 'mesh',
    id: obj.uuid,
    positions,
    normals,
    transform,
    material,
    ...(uvs != null ? { uvs } : {}),
    ...(uv1 != null ? { uv1 } : {}),
    ...(tangents != null ? { tangents } : {}),
    ...(colors != null ? { colors } : {}),
    ...(indices != null ? { indices } : {}),
  };
}

// ────────────────────────────────────────────────────────────────────────────
// InstancedMesh converter — multi-mesh TLAS production path
// ────────────────────────────────────────────────────────────────────────────

export function convertInstancedMesh(obj: THREE.InstancedMesh): InstancedMeshPrimitive {
  const geo = obj.geometry;
  const label = obj.name || obj.uuid;

  const positions = requireAttribute(geo, 'position', label, 'InstancedMesh');
  const normals = requireAttribute(geo, 'normal', label, 'InstancedMesh');

  const uvs = extractAttribute(geo, 'uv');
  const tangents = extractAttribute(geo, 'tangent');
  const indices = extractIndex(geo);

  if (Array.isArray(obj.material) && obj.material.length > 1) {
    console.warn(
      `@vitrum/three-bindings: unsupported multi-material InstancedMesh at "${label}". ` +
        `Only the first material will be used.`,
    );
  }

  const material = convertFirstMaterial(obj.material, label, 'InstancedMesh');

  const instances: Mat4[] = [];
  const tmp = new THREE.Matrix4();
  for (let i = 0; i < obj.count; i += 1) {
    obj.getMatrixAt(i, tmp);
    tmp.premultiply(obj.matrixWorld);
    instances.push(asMat4(new Float32Array(tmp.elements)));
  }

  return {
    kind: 'instanced-mesh',
    id: obj.uuid,
    positions,
    normals,
    material,
    instances,
    ...(uvs != null ? { uvs } : {}),
    ...(tangents != null ? { tangents } : {}),
    ...(indices != null ? { indices } : {}),
  };
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
export function convertSkinnedMesh(obj: THREE.SkinnedMesh): SkinnedMeshPrimitive {
  const geo = obj.geometry;
  const label = obj.name || obj.uuid;

  const positions = requireAttribute(geo, 'position', label, 'SkinnedMesh');
  const normals = requireAttribute(geo, 'normal', label, 'SkinnedMesh');

  // SkinnedMesh requires skinIndex + skinWeight attributes per glTF 2.0.
  const skinIndexAttr = geo.getAttribute('skinIndex');
  if (skinIndexAttr == null) {
    throw new Error(`SkinnedMesh "${label}" has no skinIndex attribute.`);
  }
  const skinWeightAttr = geo.getAttribute('skinWeight');
  if (skinWeightAttr == null) {
    throw new Error(`SkinnedMesh "${label}" has no skinWeight attribute.`);
  }
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

  // Mesh bind matrix: identity for glTF-typical use, but non-identity when
  // the host called `mesh.bind(skeleton, customBindMatrix)` or bound after
  // positioning the mesh. Three.js stores both bindMatrix and its inverse.
  // We only emit them when bindMatrix is not identity (saves bytes + makes
  // the common case clearly identity-defaulted).
  const bm = obj.bindMatrix.elements;
  const isIdentityBind =
    bm[0] === 1 && bm[5] === 1 && bm[10] === 1 && bm[15] === 1 &&
    bm[1] === 0 && bm[2] === 0 && bm[3] === 0 &&
    bm[4] === 0 && bm[6] === 0 && bm[7] === 0 &&
    bm[8] === 0 && bm[9] === 0 && bm[11] === 0 &&
    bm[12] === 0 && bm[13] === 0 && bm[14] === 0;
  const bindMatrix = isIdentityBind ? undefined : new Float32Array(obj.bindMatrix.elements);
  const bindMatrixInverse = isIdentityBind
    ? undefined
    : new Float32Array(obj.bindMatrixInverse.elements);

  const uvs = extractAttribute(geo, 'uv');
  const tangents = extractAttribute(geo, 'tangent');
  const indices = extractIndex(geo);
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
  if (Array.isArray(obj.material) && obj.material.length > 1) {
    console.warn(
      `@vitrum/three-bindings: unsupported multi-material SkinnedMesh at "${label}" (${obj.material.length} materials). ` +
      `Only the first material will be used.`,
    );
  }
  const material = convertFirstMaterial(obj.material, label, 'SkinnedMesh', 'SkinnedMesh');

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
    ...(uvs != null ? { uvs } : {}),
    ...(tangents != null ? { tangents } : {}),
    ...(indices != null ? { indices } : {}),
    ...(morphTargets != null ? { morphTargets } : {}),
    ...(morphTargetNormals != null ? { morphTargetNormals } : {}),
    ...(morphWeights != null ? { morphWeights } : {}),
    ...(bindMatrix != null ? { bindMatrix } : {}),
    ...(bindMatrixInverse != null ? { bindMatrixInverse } : {}),
  };
}
