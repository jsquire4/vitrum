/**
 * CPU ray-cast primitive picking (T3.G — plan/trust-remediation-plan-2026-06-10 #30).
 *
 * Unprojects a screen pixel to a world-space ray using the last frame's camera,
 * then does a closest-hit traverse of the retained core `Scene` on the CPU.
 * No GPU readback, no primitive-ID attachment — exact for triangle meshes
 * (Möller–Trumbore); analytic shapes use a world bounding sphere (approximate —
 * picking the silhouette, not the facet geometry), unless a `fallbackMesh` is
 * present (triangles are then exact).
 *
 * Complexity: O(triangles). Acceptable for a debug-surface call that is NOT
 * per-frame. For scenes with many thousands of triangles a full BVH traversal
 * would be O(log N), but that complexity is not warranted for an interactive
 * inspector pick that fires at most a few times per second.
 *
 * Mat4s follow the `@vitrum/core` three.js column-major convention:
 * element (row, col) = m[col*4 + row].
 */
import type { Mat4, Scene, ScenePrimitive, Vec3 } from '@vitrum/core';
import {
  invertMat4,
  mat4Mul,
  mat4MulVec4,
  v3Sub,
  v3Cross,
  v3Dot,
  v3Normalize,
  type V3,
} from './mathUtils.js';

export interface PickCamera {
  /** Column-major world→view matrix (three.js convention), length-16. */
  readonly viewMatrix: Float32Array;
  /** Column-major projection matrix, length-16. */
  readonly projMatrix: Float32Array;
  /** World-space camera position. */
  readonly cameraPosition: Vec3;
}

/** Transform an affine point (w = 1, no perspective divide). */
function transformPoint(m: Float32Array | undefined, p: V3): V3 {
  if (m == null) return p;
  const r = mat4MulVec4(m, p[0], p[1], p[2], 1);
  return [r[0], r[1], r[2]];
}

// ── ray construction ─────────────────────────────────────────────────────────
interface Ray { readonly o: V3; readonly d: V3; }

/**
 * Unproject pixel (px, py) into a world ray. Origin = camera; direction = toward
 * the far-plane unprojection of the pixel. Using NDC z = +1 (the far plane in
 * BOTH the OpenGL [-1,1] and WebGPU [0,1] depth conventions) keeps this robust
 * to which convention the host's projMatrix uses.
 */
function screenToWorldRay(cam: PickCamera, px: number, py: number, width: number, height: number): Ray | null {
  const ndcX = (px / width) * 2 - 1;
  const ndcY = 1 - (py / height) * 2; // screen-down → NDC-up
  const vp = mat4Mul(cam.projMatrix, cam.viewMatrix);
  const inv = invertMat4(vp as unknown as Mat4);
  if (inv == null) return null;
  const far = mat4MulVec4(inv, ndcX, ndcY, 1, 1);
  if (far[3] === 0) return null;
  const farW: V3 = [far[0] / far[3], far[1] / far[3], far[2] / far[3]];
  const o: V3 = [cam.cameraPosition[0], cam.cameraPosition[1], cam.cameraPosition[2]];
  const d = v3Normalize(v3Sub(farW, o));
  if (d[0] === 0 && d[1] === 0 && d[2] === 0) return null;
  return { o, d };
}

// ── Möller–Trumbore ──────────────────────────────────────────────────────────
const MT_EPS = 1e-7;
/** Ray–triangle hit distance, or null. Two-sided (picks back-facing too). */
function rayTriangle(ray: Ray, v0: V3, v1: V3, v2: V3): number | null {
  const e1 = v3Sub(v1, v0);
  const e2 = v3Sub(v2, v0);
  const p = v3Cross(ray.d, e2);
  const det = v3Dot(e1, p);
  if (det > -MT_EPS && det < MT_EPS) return null; // parallel
  const invDet = 1 / det;
  const tvec = v3Sub(ray.o, v0);
  const u = v3Dot(tvec, p) * invDet;
  if (u < 0 || u > 1) return null;
  const q = v3Cross(tvec, e1);
  const v = v3Dot(ray.d, q) * invDet;
  if (v < 0 || u + v > 1) return null;
  const t = v3Dot(e2, q) * invDet;
  return t > MT_EPS ? t : null;
}

/** Ray–sphere nearest positive hit, or null. */
function raySphere(ray: Ray, center: V3, radius: number): number | null {
  const oc = v3Sub(ray.o, center);
  const b = v3Dot(oc, ray.d);
  const c = v3Dot(oc, oc) - radius * radius;
  const disc = b * b - c;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const t0 = -b - sq;
  if (t0 > MT_EPS) return t0;
  const t1 = -b + sq;
  return t1 > MT_EPS ? t1 : null;
}

// ── per-primitive intersection ───────────────────────────────────────────────
function vertexCount(positions: Float32Array): number {
  return Math.floor(positions.length / 3);
}

/** Read + world-transform vertex `i`'s position from a flat xyz array. */
function vert(positions: Float32Array, i: number, transform: Mat4 | undefined): V3 | null {
  const x = positions[i * 3];
  const y = positions[i * 3 + 1];
  const z = positions[i * 3 + 2];
  if (typeof x !== 'number' || typeof y !== 'number' || typeof z !== 'number') return null;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
  return transformPoint(transform, [x, y, z]);
}

function validTriangleIndices(
  positions: Float32Array,
  i0: number,
  i1: number,
  i2: number,
): boolean {
  const count = vertexCount(positions);
  return Number.isInteger(i0) && Number.isInteger(i1) && Number.isInteger(i2) &&
    i0 >= 0 && i0 < count &&
    i1 >= 0 && i1 < count &&
    i2 >= 0 && i2 < count;
}

function intersectTriangleSoup(
  positions: Float32Array,
  indices: Uint32Array | Uint16Array | undefined,
  transform: Mat4 | undefined,
  ray: Ray,
): number | null {
  let best: number | null = null;
  const triCount = indices != null ? Math.floor(indices.length / 3) : Math.floor(vertexCount(positions) / 3);
  for (let t = 0; t < triCount; t++) {
    const i0 = indices != null ? indices[t * 3]! : t * 3;
    const i1 = indices != null ? indices[t * 3 + 1]! : t * 3 + 1;
    const i2 = indices != null ? indices[t * 3 + 2]! : t * 3 + 2;
    if (!validTriangleIndices(positions, i0, i1, i2)) continue;
    const v0 = vert(positions, i0, transform);
    const v1 = vert(positions, i1, transform);
    const v2 = vert(positions, i2, transform);
    if (v0 == null || v1 == null || v2 == null) continue;
    const hit = rayTriangle(ray, v0, v1, v2);
    if (hit != null && (best == null || hit < best)) best = hit;
  }
  return best;
}

/** World bounding sphere for an analytic shape (approximate pick). */
function analyticBoundingSphere(shape: string, params: Float32Array, transform: Mat4 | undefined): { center: V3; radius: number } | null {
  const p = (i: number): number => params[i] ?? 0;
  let cLocal: V3;
  let r: number;
  switch (shape) {
    case 'sphere': // [cx,cy,cz,radius]
      cLocal = [p(0), p(1), p(2)]; r = p(3); break;
    case 'box': // [cx,cy,cz,hx,hy,hz]
      cLocal = [p(0), p(1), p(2)]; r = Math.hypot(p(3), p(4), p(5)); break;
    case 'cylinder': // [cx,cy,cz,radius,halfHeight]
      cLocal = [p(0), p(1), p(2)]; r = Math.hypot(p(3), p(4)); break;
    case 'capsule': { // [ax,ay,az,bx,by,bz,radius]
      cLocal = [(p(0) + p(3)) / 2, (p(1) + p(4)) / 2, (p(2) + p(5)) / 2];
      r = Math.hypot(p(3) - p(0), p(4) - p(1), p(5) - p(2)) / 2 + p(6);
      break;
    }
    default:
      return null; // unknown shape (e.g. h-channel-came) → not pickable analytically
  }
  // Approximate uniform scale from the transform's first column length.
  let scale = 1;
  if (transform != null) scale = Math.hypot(transform[0] ?? 0, transform[1] ?? 0, transform[2] ?? 0) || 1;
  return { center: transformPoint(transform, cLocal), radius: r * scale };
}

function intersectPrimitive(prim: ScenePrimitive, ray: Ray): number | null {
  switch (prim.kind) {
    case 'mesh':
    case 'skinned-mesh': // rest-pose positions; deformation ignored (approximate, debug-only)
      return intersectTriangleSoup(prim.positions, prim.indices, prim.transform, ray);
    case 'instanced-mesh': {
      let best: number | null = null;
      for (const inst of prim.instances) {
        const hit = intersectTriangleSoup(prim.positions, prim.indices, inst, ray);
        if (hit != null && (best == null || hit < best)) best = hit;
      }
      return best;
    }
    case 'analytic': {
      if (prim.fallbackMesh != null) {
        return intersectTriangleSoup(prim.fallbackMesh.positions, prim.fallbackMesh.indices, prim.transform, ray);
      }
      const bs = analyticBoundingSphere(prim.shape, prim.params, prim.transform);
      return bs != null ? raySphere(ray, bs.center, bs.radius) : null;
    }
    default:
      return null;
  }
}

/**
 * Closest-hit CPU pick: returns the `id` of the nearest primitive under pixel
 * (px, py), or null on a miss, degenerate camera, or zero-size viewport.
 *
 * Used by all three backends' `debug.pickPrimitive` implementations.
 */
export function pickPrimitiveCpu(
  scene: Scene,
  camera: PickCamera,
  px: number,
  py: number,
  width: number,
  height: number,
): string | null {
  if (width <= 0 || height <= 0) return null;
  const ray = screenToWorldRay(camera, px, py, width, height);
  if (ray == null) return null;
  let bestT = Infinity;
  let bestId: string | null = null;
  for (const prim of scene.primitives) {
    const t = intersectPrimitive(prim, ray);
    if (t != null && t < bestT) {
      bestT = t;
      bestId = prim.id;
    }
  }
  return bestId;
}
