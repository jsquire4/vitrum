/**
 * mesh.ts — THREE.Mesh → @vitrum/core MeshPrimitive converter.
 *
 * Validates material type, extracts geometry attributes, and delegates to the
 * material converter. Throws for unsupported mesh types and unsupported
 * materials — the caller (`sceneFromThreeJS`) should not silently skip meshes.
 */

import type * as THREE from 'three';
import type { MeshPrimitive, Mat4 } from '@vitrum/core';
import { convertMaterial } from './material.js';

// ────────────────────────────────────────────────────────────────────────────
// Attribute extractors
// ────────────────────────────────────────────────────────────────────────────

export function extractAttribute(
  geo: THREE.BufferGeometry,
  name: string,
): Float32Array | undefined {
  const attr = geo.getAttribute(name);
  if (attr == null) return undefined;
  const arr = attr.array;
  if (arr instanceof Float32Array) return arr;
  return new Float32Array(arr);
}

export function extractIndex(
  geo: THREE.BufferGeometry,
): Uint32Array | Uint16Array | undefined {
  const idx = geo.index;
  if (idx == null) return undefined;
  const arr = idx.array;
  if (arr instanceof Uint32Array || arr instanceof Uint16Array) return arr;
  return new Uint32Array(arr);
}

// ────────────────────────────────────────────────────────────────────────────
// Mesh converter
// ────────────────────────────────────────────────────────────────────────────

export function convertMesh(obj: THREE.Mesh): MeshPrimitive {
  const geo = obj.geometry as THREE.BufferGeometry;
  const label = obj.name || obj.uuid;

  const positions = extractAttribute(geo, 'position');
  if (positions == null) {
    throw new Error(`Mesh "${label}" has no position attribute.`);
  }

  const normals = extractAttribute(geo, 'normal');
  if (normals == null) {
    throw new Error(
      `Mesh "${label}" has no normal attribute. Compute normals before calling sceneFromThreeJS.`,
    );
  }

  const uvs = extractAttribute(geo, 'uv');
  const tangents = extractAttribute(geo, 'tangent');
  const indices = extractIndex(geo);

  const transform = new Float32Array(obj.matrixWorld.elements) as Mat4;

  // Multi-material meshes: warn and fall back to first material.
  // Sprint 2 will add geometry-group splitting so each group gets its material.
  if (Array.isArray(obj.material) && obj.material.length > 1) {
    console.warn(
      `@vitrum/three-bindings: unsupported multi-material mesh at "${label}" (${obj.material.length} materials). ` +
      `Only the first material will be used. Supported types are added per Phase 6 sprint.`,
    );
  }

  const rawMat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
  if (
    rawMat == null ||
    (!(rawMat as THREE.MeshStandardMaterial).isMeshStandardMaterial &&
      !(rawMat as THREE.MeshPhysicalMaterial).isMeshPhysicalMaterial)
  ) {
    const typeName = rawMat != null ? (rawMat as object).constructor.name : 'null';
    throw new Error(
      `Unsupported THREE type at "${label}": material ${typeName}. Supported types are added per Phase 6 sprint.`,
    );
  }

  const material = convertMaterial(rawMat as THREE.MeshStandardMaterial);

  return {
    kind: 'mesh',
    id: obj.uuid,
    positions,
    normals,
    transform,
    material,
    ...(uvs != null ? { uvs } : {}),
    ...(tangents != null ? { tangents } : {}),
    ...(indices != null ? { indices } : {}),
  };
}
