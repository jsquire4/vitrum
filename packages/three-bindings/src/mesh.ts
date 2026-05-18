/**
 * mesh.ts — THREE.Mesh → @vitrum/core MeshPrimitive converter.
 *
 * Validates material type, extracts geometry attributes, and delegates to the
 * material converter. Throws for unsupported mesh types and unsupported
 * materials — the caller (`sceneFromThreeJS`) should not silently skip meshes.
 */

import type * as THREE from 'three';
import type { MeshPrimitive, Mat4, SceneEmitter } from '@vitrum/core';
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

// extractAttribute / extractIndex are file-local. W7-G7 (fe2be20)
// un-exported them from three-bindings/src/index.ts; the matching
// `export` keyword stayed on the source declarations until 2026-05-18
// dead-code sweep confirmed zero non-self consumers.
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
  // Geometry-group splitting (each group gets its own material) is a future enhancement.
  if (Array.isArray(obj.material) && obj.material.length > 1) {
    console.warn(
      `@vitrum/three-bindings: unsupported multi-material mesh at "${label}" (${obj.material.length} materials). ` +
      `Only the first material will be used. Supported types are added per Phase 6 sprint.`,
    );
  }

  const rawMat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
  const isStd  = (rawMat as THREE.MeshStandardMaterial | null)?.isMeshStandardMaterial === true;
  const isPhys = (rawMat as THREE.MeshPhysicalMaterial | null)?.isMeshPhysicalMaterial === true;
  // MeshBasicMaterial is the third accepted type. It renders unlit in three.js;
  // we synthesize a flat-emissive vitrum material so it appears as a self-lit
  // flat color regardless of scene lighting. Used by app-side overlay meshes
  // (panel mount preview, debug overlays, grid layers).
  const isBasic = (rawMat as THREE.MeshBasicMaterial | null)?.isMeshBasicMaterial === true;
  if (rawMat == null || (!isStd && !isPhys && !isBasic)) {
    const typeName = rawMat != null ? (rawMat as object).constructor.name : 'null';
    throw new Error(
      `Unsupported THREE type at "${label}": material ${typeName}. Supported types are added per Phase 6 sprint.`,
    );
  }

  const material = isBasic
    ? convertBasicMaterial(rawMat as THREE.MeshBasicMaterial)
    : convertMaterial(rawMat as THREE.MeshStandardMaterial);

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
