/**
 * CPU ray-cast primitive picking for {@link HybridEngine}'s debug surface
 * (`EngineDebugSurface.pickPrimitive`, items_to_fix.md T3.G).
 *
 * Option (a) of the T3.G fix: unproject the screen pixel into a world-space ray
 * using the last frame's camera, then closest-hit against the retained core
 * `Scene` on the CPU. Cheap, no GPU readback, no primitive-ID attachment. Exact
 * for triangle meshes (Möller–Trumbore); analytic shapes use a world bounding
 * sphere (approximate — picking a gemstone's silhouette, not its facets), unless
 * an `analytic.fallbackMesh` is present, in which case its triangles are exact.
 *
 * Mat4s follow the `@vitrum/core` three.js convention: column-major, element
 * (row, col) = m[col*4 + row]. `invertMat4` is the single-homed inverse from
 * `@vitrum/shared-bvh`.
 */
import type { Scene, Mat4, Vec3, ScenePrimitive } from '@vitrum/core';
import { invertMat4 } from '@vitrum/shared-bvh';

export interface PickCamera {
  /** Column-major world→view matrix (three.js convention), length-16. */
  readonly viewMatrix: Float32Array;
  /** Column-major projection matrix, length-16. */
  readonly projMatrix: Float32Array;
  /** World-space camera position. */
  readonly cameraPosition: Vec3;
}

type V3 = readonly [number, number, number];

// ── column-major mat4 helpers (element (row,col) = m[col*4+row]) ──────────────
function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += (a[k * 4 + row] ?? 0) * (b[col * 4 + k] ?? 0);
      o[col * 4 + row] = s;
    }
  }
  return o;
}

/** m · (x,y,z,w) → [x',y',z',w']. */
function mat4MulVec4(m: Float32Array, x: number, y: number, z: number, w: number): [number, number, number, number] {
  const g = (i: number): number => m[i] ?? 0;
  return [
    g(0) * x + g(4) * y + g(8) * z + g(12) * w,
    g(1) * x + g(5) * y + g(9) * z + g(13) * w,
    g(2) * x + g(6) * y + g(10) * z + g(14) * w,
    g(3) * x + g(7) * y + g(11) * z + g(15) * w,
  ];
}

/** Transform an affine point (w = 1, no perspective divide). */
function transformPoint(m: Float32Array | undefined, p: V3): V3 {
  if (m == null) return p;
  const r = mat4MulVec4(m, p[0], p[1], p[2], 1);
  return [r[0], r[1], r[2]];
}

// ── vec3 helpers ─────────────────────────────────────────────────────────────
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function normalize(a: V3): V3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : a;
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
  const d = normalize(sub(farW, o));
  if (d[0] === 0 && d[1] === 0 && d[2] === 0) return null;
  return { o, d };
}

// ── Möller–Trumbore ──────────────────────────────────────────────────────────
const EPS = 1e-7;
/** Ray–triangle hit distance, or null. Two-sided (picks back-facing too). */
function rayTriangle(ray: Ray, v0: V3, v1: V3, v2: V3): number | null {
  const e1 = sub(v1, v0);
  const e2 = sub(v2, v0);
  const p = cross(ray.d, e2);
  const det = dot(e1, p);
  if (det > -EPS && det < EPS) return null; // parallel
  const invDet = 1 / det;
  const tvec = sub(ray.o, v0);
  const u = dot(tvec, p) * invDet;
  if (u < 0 || u > 1) return null;
  const q = cross(tvec, e1);
  const v = dot(ray.d, q) * invDet;
  if (v < 0 || u + v > 1) return null;
  const t = dot(e2, q) * invDet;
  return t > EPS ? t : null;
}

/** Ray–sphere nearest positive hit, or null. */
function raySphere(ray: Ray, center: V3, radius: number): number | null {
  const oc = sub(ray.o, center);
  const b = dot(oc, ray.d);
  const c = dot(oc, oc) - radius * radius;
  const disc = b * b - c;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const t0 = -b - sq;
  if (t0 > EPS) return t0;
  const t1 = -b + sq;
  return t1 > EPS ? t1 : null;
}

// ── per-primitive intersection ───────────────────────────────────────────────
/** Read + world-transform vertex `i`'s position from a flat xyz array. */
function vert(positions: Float32Array, i: number, transform: Mat4 | undefined): V3 {
  return transformPoint(transform, [positions[i * 3] ?? 0, positions[i * 3 + 1] ?? 0, positions[i * 3 + 2] ?? 0]);
}

function intersectTriangleSoup(
  positions: Float32Array,
  indices: Uint32Array | Uint16Array | undefined,
  transform: Mat4 | undefined,
  ray: Ray,
): number | null {
  let best: number | null = null;
  const triCount = indices != null ? (indices.length / 3) | 0 : (positions.length / 9) | 0;
  for (let t = 0; t < triCount; t++) {
    const i0 = (indices != null ? indices[t * 3] : t * 3) ?? 0;
    const i1 = (indices != null ? indices[t * 3 + 1] : t * 3 + 1) ?? 0;
    const i2 = (indices != null ? indices[t * 3 + 2] : t * 3 + 2) ?? 0;
    const hit = rayTriangle(ray, vert(positions, i0, transform), vert(positions, i1, transform), vert(positions, i2, transform));
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
      return null; // unknown shape (e.g. h-channel-came) → not pickable here
  }
  // Approximate uniform scale from the transform's first column length.
  let scale = 1;
  if (transform != null) scale = Math.hypot(transform[0] ?? 0, transform[1] ?? 0, transform[2] ?? 0) || 1;
  return { center: transformPoint(transform, cLocal), radius: r * scale };
}

function intersectPrimitive(p: ScenePrimitive, ray: Ray): number | null {
  switch (p.kind) {
    case 'mesh':
    case 'skinned-mesh': // rest-pose positions; deformation ignored (approximate, debug-only)
      return intersectTriangleSoup(p.positions, p.indices, p.transform, ray);
    case 'instanced-mesh': {
      let best: number | null = null;
      for (const inst of p.instances) {
        const hit = intersectTriangleSoup(p.positions, p.indices, inst, ray);
        if (hit != null && (best == null || hit < best)) best = hit;
      }
      return best;
    }
    case 'analytic': {
      if (p.fallbackMesh != null) {
        return intersectTriangleSoup(p.fallbackMesh.positions, p.fallbackMesh.indices, p.transform, ray);
      }
      const bs = analyticBoundingSphere(p.shape, p.params, p.transform);
      return bs != null ? raySphere(ray, bs.center, bs.radius) : null;
    }
    default:
      return null;
  }
}

/**
 * Closest-hit pick: returns the `id` of the nearest primitive under pixel
 * (px, py), or null if the ray misses everything (or the camera is degenerate).
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
