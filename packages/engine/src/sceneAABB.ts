// Compute world-space axis-aligned bounding box for a vitrum Scene.
//
// Used by createEngine() to derive scale-sensitive defaults (Möller-Trumbore
// epsilon, camera-move-reset threshold, emitter distance² floor, GTAO depth
// threshold, etc.) from the scene's diagonal length D.
//
// Mesh + InstancedMesh primitives contribute their transformed vertex AABB.
// Analytic primitives contribute their fallback mesh's AABB if present, or
// a point at their transform origin (best-effort — analytic shapes are
// expected to be rare in user scenes).

import type { Scene, ScenePrimitive, Mat4, Vec3 } from '@vitrum/core';

export interface SceneAABB {
  readonly min: Vec3;
  readonly max: Vec3;
  readonly center: Vec3;
  readonly extent: Vec3;
  /** Diagonal length: sqrt(extent.x² + extent.y² + extent.z²). */
  readonly diagonal: number;
  /** Number of triangles (3-vertex faces) summed across all primitives. */
  readonly triangleCount: number;
}

const EMPTY_MIN: Vec3 = [Infinity, Infinity, Infinity];
const EMPTY_MAX: Vec3 = [-Infinity, -Infinity, -Infinity];

/**
 * Identity-fallback diagonal. Used when the scene has no mesh primitives
 * (e.g. only environment + analytic shapes with no fallbackMesh) so callers
 * still get sane scale defaults rather than NaN.
 */
const FALLBACK_DIAGONAL = 1.0;

export function computeSceneAABB(scene: Scene): SceneAABB {
  let minX = EMPTY_MIN[0];
  let minY = EMPTY_MIN[1];
  let minZ = EMPTY_MIN[2];
  let maxX = EMPTY_MAX[0];
  let maxY = EMPTY_MAX[1];
  let maxZ = EMPTY_MAX[2];
  let triangleCount = 0;
  let foundAny = false;

  for (const prim of scene.primitives) {
    const contribution = primitiveBounds(prim);
    if (contribution == null) continue;
    foundAny = true;
    if (contribution.min[0] < minX) minX = contribution.min[0];
    if (contribution.min[1] < minY) minY = contribution.min[1];
    if (contribution.min[2] < minZ) minZ = contribution.min[2];
    if (contribution.max[0] > maxX) maxX = contribution.max[0];
    if (contribution.max[1] > maxY) maxY = contribution.max[1];
    if (contribution.max[2] > maxZ) maxZ = contribution.max[2];
    triangleCount += contribution.triangles;
  }

  if (!foundAny) {
    return {
      min: [-0.5, -0.5, -0.5],
      max: [0.5, 0.5, 0.5],
      center: [0, 0, 0],
      extent: [1, 1, 1],
      diagonal: FALLBACK_DIAGONAL,
      triangleCount: 0,
    };
  }

  const ex = maxX - minX;
  const ey = maxY - minY;
  const ez = maxZ - minZ;
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center: [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5],
    extent: [ex, ey, ez],
    diagonal: Math.max(FALLBACK_DIAGONAL, Math.sqrt(ex * ex + ey * ey + ez * ez)),
    triangleCount,
  };
}

interface BoundsContribution {
  readonly min: Vec3;
  readonly max: Vec3;
  readonly triangles: number;
}

function primitiveBounds(prim: ScenePrimitive): BoundsContribution | null {
  if (prim.kind === 'mesh') {
    const local = vertexAabb(prim.positions);
    if (local == null) return null;
    const transformed = transformAabb(local, prim.transform);
    const triCount = triangleCountFor(prim.positions, prim.indices);
    return { ...transformed, triangles: triCount };
  }
  if (prim.kind === 'instanced-mesh') {
    const local = vertexAabb(prim.positions);
    if (local == null) return null;
    const triPerInstance = triangleCountFor(prim.positions, prim.indices);
    let aMinX = Infinity,
      aMinY = Infinity,
      aMinZ = Infinity;
    let aMaxX = -Infinity,
      aMaxY = -Infinity,
      aMaxZ = -Infinity;
    for (const m of prim.instances) {
      const t = transformAabb(local, m);
      if (t.min[0] < aMinX) aMinX = t.min[0];
      if (t.min[1] < aMinY) aMinY = t.min[1];
      if (t.min[2] < aMinZ) aMinZ = t.min[2];
      if (t.max[0] > aMaxX) aMaxX = t.max[0];
      if (t.max[1] > aMaxY) aMaxY = t.max[1];
      if (t.max[2] > aMaxZ) aMaxZ = t.max[2];
    }
    return {
      min: [aMinX, aMinY, aMinZ],
      max: [aMaxX, aMaxY, aMaxZ],
      triangles: triPerInstance * prim.instances.length,
    };
  }
  if (prim.kind === 'analytic') {
    if (prim.fallbackMesh != null) {
      const local = vertexAabb(prim.fallbackMesh.positions);
      if (local != null) {
        const transformed = transformAabb(local, prim.transform);
        const triCount = triangleCountFor(prim.fallbackMesh.positions, prim.fallbackMesh.indices);
        return { ...transformed, triangles: triCount };
      }
    }
    if (prim.transform != null) {
      const tx = prim.transform[12]!;
      const ty = prim.transform[13]!;
      const tz = prim.transform[14]!;
      return {
        min: [tx, ty, tz],
        max: [tx, ty, tz],
        triangles: 0,
      };
    }
    return null;
  }
  return null;
}

function vertexAabb(positions: Float32Array): { min: Vec3; max: Vec3 } | null {
  if (positions.length < 3) return null;
  let minX = positions[0]!;
  let minY = positions[1]!;
  let minZ = positions[2]!;
  let maxX = minX;
  let maxY = minY;
  let maxZ = minZ;
  for (let i = 3; i < positions.length; i += 3) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    const z = positions[i + 2]!;
    if (x < minX) minX = x;
    else if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    else if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    else if (z > maxZ) maxZ = z;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

// Transform a local-space AABB by a column-major Mat4 by transforming all 8
// corners and taking the resulting min/max. Fast, conservative, exact for
// affine transforms; ignores perspective rows (acceptable since vitrum scene
// transforms are affine).
function transformAabb(
  local: { min: Vec3; max: Vec3 },
  m: Mat4 | undefined,
): { min: Vec3; max: Vec3 } {
  if (m == null) return { min: local.min, max: local.max };

  const corners: ReadonlyArray<Vec3> = [
    [local.min[0], local.min[1], local.min[2]],
    [local.min[0], local.min[1], local.max[2]],
    [local.min[0], local.max[1], local.min[2]],
    [local.min[0], local.max[1], local.max[2]],
    [local.max[0], local.min[1], local.min[2]],
    [local.max[0], local.min[1], local.max[2]],
    [local.max[0], local.max[1], local.min[2]],
    [local.max[0], local.max[1], local.max[2]],
  ];
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (const c of corners) {
    const x = c[0],
      y = c[1],
      z = c[2];
    const tx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
    const ty = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
    const tz = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
    if (tx < minX) minX = tx;
    if (tx > maxX) maxX = tx;
    if (ty < minY) minY = ty;
    if (ty > maxY) maxY = ty;
    if (tz < minZ) minZ = tz;
    if (tz > maxZ) maxZ = tz;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

function triangleCountFor(
  positions: Float32Array,
  indices: Uint32Array | Uint16Array | undefined,
): number {
  if (indices != null) return Math.floor(indices.length / 3);
  return Math.floor(positions.length / 9);
}
