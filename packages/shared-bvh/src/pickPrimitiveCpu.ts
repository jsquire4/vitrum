/**
 * CPU ray-cast primitive picking (T3.G — plan/trust-remediation-plan-2026-06-10 #30).
 *
 * Unprojects a screen pixel to a world-space ray using the last frame's camera,
 * then does a closest-hit traverse of the retained core `Scene` on the CPU.
 * No GPU readback, no primitive-ID attachment — exact for triangle meshes
 * (Möller–Trumbore) and for every declared analytic shape. Analytic rays are
 * transformed to shape-local space, preserving exact intersections under
 * arbitrary invertible affine transforms. When a `fallbackMesh` is present,
 * its triangles remain the authoritative pick geometry.
 *
 * Complexity: O(triangles). Acceptable for a debug-surface call that is NOT
 * per-frame. For scenes with many thousands of triangles a full BVH traversal
 * would be O(log N), but that complexity is not warranted for an interactive
 * inspector pick that fires at most a few times per second.
 *
 * Mat4s follow the `@vitrum/core` three.js column-major convention:
 * element (row, col) = m[col*4 + row].
 */
import {
  decodeAnalyticParams,
  solveSkin,
  type AnalyticPrimitive,
  type Mat4,
  type Scene,
  type ScenePrimitive,
  type Vec3,
} from '@vitrum/core';
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
interface Ray {
  readonly o: V3;
  readonly d: V3;
}

/**
 * Unproject pixel (px, py) into a world ray. The unprojected near-plane point is
 * the origin for both perspective and orthographic projections, so picking
 * honors near clipping and orthographic pixels receive distinct origins. The
 * division-free homogeneous direction also supports an infinite far plane.
 */
function screenToWorldRay(
  cam: PickCamera,
  px: number,
  py: number,
  width: number,
  height: number,
): Ray | null {
  const ndcX = (px / width) * 2 - 1;
  const ndcY = 1 - (py / height) * 2; // screen-down → NDC-up
  const vp = mat4Mul(cam.projMatrix, cam.viewMatrix);
  const inv = invertMat4(vp as unknown as Mat4);
  if (inv == null) return null;
  const farRaw = mat4MulVec4(inv, ndcX, ndcY, 1, 1);
  const nearRaw = mat4MulVec4(inv, ndcX, ndcY, -1, 1);
  const farScale = Math.max(...farRaw.map(Math.abs));
  const nearScale = Math.max(...nearRaw.map(Math.abs));
  if (
    !(farScale > 0) || !Number.isFinite(farScale) ||
    !(nearScale > 0) || !Number.isFinite(nearScale)
  ) {
    return null;
  }
  const far = farRaw.map((value) => value / farScale);
  const near = nearRaw.map((value) => value / nearScale);
  if (near[3] === 0) return null;
  const o: V3 = [
    near[0]! / near[3]!,
    near[1]! / near[3]!,
    near[2]! / near[3]!,
  ];
  if (!o.every(Number.isFinite)) return null;
  const orientation =
    far[3] === 0 ? 1 : Math.sign(far[3]! * near[3]!);
  const d = v3Normalize([
    (far[0]! * near[3]! - near[0]! * far[3]!) * orientation,
    (far[1]! * near[3]! - near[1]! * far[3]!) * orientation,
    (far[2]! * near[3]! - near[2]! * far[3]!) * orientation,
  ]);
  if (d[0] === 0 && d[1] === 0 && d[2] === 0) return null;
  return { o, d };
}

// ── Möller–Trumbore ──────────────────────────────────────────────────────────
const MT_ANGULAR_EPSILON = 1e-7;
const MT_BARYCENTRIC_EPSILON = 1e-6;

function maxAbs3(value: V3): number {
  return Math.max(Math.abs(value[0]), Math.abs(value[1]), Math.abs(value[2]));
}

/**
 * Roots of a·x² + 2b·x + c = 0. Inputs are spatially equilibrated by each
 * caller, and the q-form avoids losing the far root to cancellation.
 */
function quadraticRootsHalfB(a: number, b: number, c: number): readonly [number, number] | null {
  if (!(a > 0) || !Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) {
    return null;
  }
  const discriminant = b * b - a * c;
  if (!(discriminant >= 0) || !Number.isFinite(discriminant)) return null;
  const squareRoot = Math.sqrt(discriminant);
  if (squareRoot === 0) {
    const root = -b / a;
    return Number.isFinite(root) ? [root, root] : null;
  }
  const q = -(b + (b >= 0 ? squareRoot : -squareRoot));
  if (q === 0 || !Number.isFinite(q)) return null;
  const root0 = q / a;
  const root1 = c / q;
  if (!Number.isFinite(root0) || !Number.isFinite(root1)) return null;
  return root0 <= root1 ? [root0, root1] : [root1, root0];
}

function scaledRayParameter(
  scaledParameter: number,
  spatialScale: number,
  directionScale: number,
): number | null {
  const t = scaledParameter * spatialScale / directionScale;
  return t > 0 && Number.isFinite(t) ? t : null;
}

/** Ray–triangle hit distance, or null. Two-sided (picks back-facing too). */
function rayTriangle(ray: Ray, v0: V3, v1: V3, v2: V3): number | null {
  const e1Raw = v3Sub(v1, v0);
  const e2Raw = v3Sub(v2, v0);
  const edgeScale = Math.max(
    Math.abs(e1Raw[0]),
    Math.abs(e1Raw[1]),
    Math.abs(e1Raw[2]),
    Math.abs(e2Raw[0]),
    Math.abs(e2Raw[1]),
    Math.abs(e2Raw[2]),
  );
  const directionScale = Math.max(
    Math.abs(ray.d[0]),
    Math.abs(ray.d[1]),
    Math.abs(ray.d[2]),
  );
  if (
    !(edgeScale > 0) ||
    !Number.isFinite(edgeScale) ||
    !(directionScale > 0) ||
    !Number.isFinite(directionScale)
  ) {
    return null;
  }
  const e1: V3 = [
    e1Raw[0] / edgeScale,
    e1Raw[1] / edgeScale,
    e1Raw[2] / edgeScale,
  ];
  const e2: V3 = [
    e2Raw[0] / edgeScale,
    e2Raw[1] / edgeScale,
    e2Raw[2] / edgeScale,
  ];
  const direction: V3 = [
    ray.d[0] / directionScale,
    ray.d[1] / directionScale,
    ray.d[2] / directionScale,
  ];
  const aoRaw = v3Sub(ray.o, v0);
  const ao: V3 = [
    aoRaw[0] / edgeScale,
    aoRaw[1] / edgeScale,
    aoRaw[2] / edgeScale,
  ];
  const n = v3Cross(e1, e2);
  const p = v3Cross(direction, e2);
  const det = v3Dot(e1, p);
  const normalLength = Math.hypot(n[0], n[1], n[2]);
  const directionLength = Math.hypot(
    direction[0],
    direction[1],
    direction[2],
  );
  if (
    !(normalLength > 0) ||
    !Number.isFinite(normalLength) ||
    !(directionLength > 0) ||
    !Number.isFinite(directionLength) ||
    Math.abs(det) / (normalLength * directionLength) <= MT_ANGULAR_EPSILON
  ) {
    return null;
  }
  const invDet = 1 / det;
  const u = v3Dot(ao, p) * invDet;
  const q = v3Cross(ao, e1);
  const v = v3Dot(direction, q) * invDet;
  const w = 1 - u - v;
  if (
    !Number.isFinite(u) ||
    !Number.isFinite(v) ||
    !Number.isFinite(w) ||
    u < -MT_BARYCENTRIC_EPSILON ||
    v < -MT_BARYCENTRIC_EPSILON ||
    w < -MT_BARYCENTRIC_EPSILON
  ) {
    return null;
  }
  const t = (v3Dot(e2, q) * invDet * edgeScale) / directionScale;
  if (!Number.isFinite(t)) return null;
  return t > 0 ? t : null;
}

/** Ray–sphere nearest positive hit, or null. Supports non-unit directions. */
function raySphere(ray: Ray, center: V3, radius: number): number | null {
  const ocRaw = v3Sub(ray.o, center);
  const spatialScale = Math.max(maxAbs3(ocRaw), Math.abs(radius));
  const directionScale = maxAbs3(ray.d);
  if (
    !(spatialScale > 0)
    || !Number.isFinite(spatialScale)
    || !(directionScale > 0)
    || !Number.isFinite(directionScale)
  ) {
    return null;
  }
  const oc: V3 = [
    ocRaw[0] / spatialScale,
    ocRaw[1] / spatialScale,
    ocRaw[2] / spatialScale,
  ];
  const direction: V3 = [
    ray.d[0] / directionScale,
    ray.d[1] / directionScale,
    ray.d[2] / directionScale,
  ];
  const scaledRadius = radius / spatialScale;
  const roots = quadraticRootsHalfB(
    v3Dot(direction, direction),
    v3Dot(oc, direction),
    v3Dot(oc, oc) - scaledRadius * scaledRadius,
  );
  if (roots == null) return null;
  for (const root of roots) {
    const t = scaledRayParameter(root, spatialScale, directionScale);
    if (t != null) return t;
  }
  return null;
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
  return (
    Number.isInteger(i0) &&
    Number.isInteger(i1) &&
    Number.isInteger(i2) &&
    i0 >= 0 &&
    i0 < count &&
    i1 >= 0 &&
    i1 < count &&
    i2 >= 0 &&
    i2 < count
  );
}

function intersectTriangleSoup(
  positions: Float32Array,
  indices: Uint32Array | Uint16Array | undefined,
  transform: Mat4 | undefined,
  ray: Ray,
): number | null {
  let best: number | null = null;
  const triCount =
    indices != null ? Math.floor(indices.length / 3) : Math.floor(vertexCount(positions) / 3);
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

function rayAabb(ray: Ray, min: V3, max: V3): number | null {
  const interval = rayAabbInterval(ray, min, max);
  if (interval == null) return null;
  const [near, far] = interval;
  if (near > 0) return near;
  return far > 0 ? far : null;
}

function rayAabbInterval(ray: Ray, min: V3, max: V3): readonly [number, number] | null {
  let near = -Infinity;
  let far = Infinity;
  for (let axis = 0; axis < 3; axis += 1) {
    const origin = ray.o[axis]!;
    const direction = ray.d[axis]!;
    const lo = min[axis]!;
    const hi = max[axis]!;
    if (direction === 0) {
      if (origin < lo || origin > hi) return null;
      continue;
    }
    let t0 = (lo - origin) / direction;
    let t1 = (hi - origin) / direction;
    if (t0 > t1) [t0, t1] = [t1, t0];
    near = Math.max(near, t0);
    far = Math.min(far, t1);
    if (near > far) return null;
  }
  return [near, far];
}

/**
 * First positive boundary of a union of axis-aligned boxes.
 *
 * Merging the one-dimensional ray intervals is essential when boxes overlap:
 * taking the minimum individual-box hit can return an internal seam while the
 * ray remains inside another member of the union.
 */
function rayAabbUnion(ray: Ray, boxes: ReadonlyArray<readonly [V3, V3]>): number | null {
  const intervals = boxes
    .map(([min, max]) => rayAabbInterval(ray, min, max))
    .filter((interval): interval is readonly [number, number] => interval != null)
    .sort((a, b) => a[0] - b[0]);
  if (intervals.length === 0) return null;

  let mergedNear = intervals[0]![0];
  let mergedFar = intervals[0]![1];
  for (let i = 1; i < intervals.length; i += 1) {
    const [near, far] = intervals[i]!;
    const boundaryTolerance =
      Number.EPSILON * 16 * Math.max(Math.abs(near), Math.abs(mergedFar));
    if (near <= mergedFar || near - mergedFar <= boundaryTolerance) {
      mergedFar = Math.max(mergedFar, far);
      continue;
    }
    if (mergedFar > 0) {
      return mergedNear > 0 ? mergedNear : mergedFar;
    }
    mergedNear = near;
    mergedFar = far;
  }
  if (mergedFar <= 0) return null;
  return mergedNear > 0 ? mergedNear : mergedFar;
}

function keepNearest(current: number | null, candidate: number | null): number | null {
  if (candidate == null) return current;
  return current == null || candidate < current ? candidate : current;
}

function rayCylinderY(ray: Ray, center: V3, radius: number, halfHeight: number): number | null {
  const offsetRaw: V3 = [
    ray.o[0] - center[0],
    ray.o[1] - center[1],
    ray.o[2] - center[2],
  ];
  const spatialScale = Math.max(
    maxAbs3(offsetRaw),
    Math.abs(radius),
    Math.abs(halfHeight),
  );
  const directionScale = maxAbs3(ray.d);
  if (
    !(spatialScale > 0)
    || !Number.isFinite(spatialScale)
    || !(directionScale > 0)
    || !Number.isFinite(directionScale)
  ) {
    return null;
  }
  const ox = offsetRaw[0] / spatialScale;
  const oy = offsetRaw[1] / spatialScale;
  const oz = offsetRaw[2] / spatialScale;
  const dx = ray.d[0] / directionScale;
  const dy = ray.d[1] / directionScale;
  const dz = ray.d[2] / directionScale;
  const scaledRadius = radius / spatialScale;
  const scaledHalfHeight = halfHeight / spatialScale;
  let best: number | null = null;

  const sideA = dx * dx + dz * dz;
  const sideB = ox * dx + oz * dz;
  const sideC = ox * ox + oz * oz - scaledRadius * scaledRadius;
  const sideRoots = quadraticRootsHalfB(sideA, sideB, sideC);
  if (sideRoots != null) {
    for (const scaledT of sideRoots) {
      const y = oy + scaledT * dy;
      const t = scaledRayParameter(scaledT, spatialScale, directionScale);
      if (t != null && y >= -scaledHalfHeight && y <= scaledHalfHeight) {
        best = keepNearest(best, t);
      }
    }
  }

  if (dy !== 0) {
    for (const capY of [-scaledHalfHeight, scaledHalfHeight]) {
      const scaledT = (capY - oy) / dy;
      const t = scaledRayParameter(scaledT, spatialScale, directionScale);
      if (t == null) continue;
      const x = ox + scaledT * dx;
      const z = oz + scaledT * dz;
      if (x * x + z * z <= scaledRadius * scaledRadius) {
        best = keepNearest(best, t);
      }
    }
  }
  return best;
}

function rayCapsule(ray: Ray, endpointA: V3, endpointB: V3, radius: number): number | null {
  const segmentRaw = v3Sub(endpointB, endpointA);
  const offsetRaw = v3Sub(ray.o, endpointA);
  const spatialScale = Math.max(
    maxAbs3(segmentRaw),
    maxAbs3(offsetRaw),
    Math.abs(radius),
  );
  const directionScale = maxAbs3(ray.d);
  if (
    !(spatialScale > 0)
    || !Number.isFinite(spatialScale)
    || !(directionScale > 0)
    || !Number.isFinite(directionScale)
  ) {
    return null;
  }
  const segment: V3 = [
    segmentRaw[0] / spatialScale,
    segmentRaw[1] / spatialScale,
    segmentRaw[2] / spatialScale,
  ];
  const length = Math.hypot(segment[0], segment[1], segment[2]);
  if (length === 0) return raySphere(ray, endpointA, radius);
  const axis: V3 = [segment[0] / length, segment[1] / length, segment[2] / length];
  const offset: V3 = [
    offsetRaw[0] / spatialScale,
    offsetRaw[1] / spatialScale,
    offsetRaw[2] / spatialScale,
  ];
  const direction: V3 = [
    ray.d[0] / directionScale,
    ray.d[1] / directionScale,
    ray.d[2] / directionScale,
  ];
  const scaledRadius = radius / spatialScale;
  const directionAxis = v3Dot(direction, axis);
  const offsetAxis = v3Dot(offset, axis);
  const directionPerp: V3 = [
    direction[0] - axis[0] * directionAxis,
    direction[1] - axis[1] * directionAxis,
    direction[2] - axis[2] * directionAxis,
  ];
  const offsetPerp: V3 = [
    offset[0] - axis[0] * offsetAxis,
    offset[1] - axis[1] * offsetAxis,
    offset[2] - axis[2] * offsetAxis,
  ];

  let best: number | null = null;
  const sideA = v3Dot(directionPerp, directionPerp);
  const sideB = v3Dot(offsetPerp, directionPerp);
  const sideC = v3Dot(offsetPerp, offsetPerp) - scaledRadius * scaledRadius;
  const sideRoots = quadraticRootsHalfB(sideA, sideB, sideC);
  if (sideRoots != null) {
    for (const scaledT of sideRoots) {
      const along = offsetAxis + scaledT * directionAxis;
      const t = scaledRayParameter(scaledT, spatialScale, directionScale);
      if (t != null && along >= 0 && along <= length) {
        best = keepNearest(best, t);
      }
    }
  }

  const directionLengthSquared = v3Dot(direction, direction);
  const capRoots = (centerOffset: V3): readonly [number, number] | null => {
    return quadraticRootsHalfB(
      directionLengthSquared,
      v3Dot(centerOffset, direction),
      v3Dot(centerOffset, centerOffset) - scaledRadius * scaledRadius,
    );
  };
  const rootsA = capRoots(offset);
  if (rootsA != null) {
    for (const scaledT of rootsA) {
      const t = scaledRayParameter(scaledT, spatialScale, directionScale);
      if (t == null) continue;
      const point = [
        offset[0] + direction[0] * scaledT,
        offset[1] + direction[1] * scaledT,
        offset[2] + direction[2] * scaledT,
      ] as V3;
      if (v3Dot(point, axis) <= 0) {
        best = keepNearest(best, t);
      }
    }
  }
  const offsetFromB = v3Sub(offset, segment);
  const rootsB = capRoots(offsetFromB);
  if (rootsB != null) {
    for (const scaledT of rootsB) {
      const t = scaledRayParameter(scaledT, spatialScale, directionScale);
      if (t == null) continue;
      const point = [
        offsetFromB[0] + direction[0] * scaledT,
        offsetFromB[1] + direction[1] * scaledT,
        offsetFromB[2] + direction[2] * scaledT,
      ] as V3;
      if (v3Dot(point, axis) >= 0) {
        best = keepNearest(best, t);
      }
    }
  }
  return best;
}

function transformRayToLocal(ray: Ray, transform: Mat4 | undefined): Ray | null {
  if (transform == null) return ray;
  const inverse = invertMat4(transform);
  if (inverse == null) return null;
  const origin = mat4MulVec4(inverse, ray.o[0], ray.o[1], ray.o[2], 1);
  if (origin[3] === 0) return null;
  const inverseW = 1 / origin[3];
  const direction = mat4MulVec4(inverse, ray.d[0], ray.d[1], ray.d[2], 0);
  const localDirection: V3 = [direction[0], direction[1], direction[2]];
  const localDirectionScale = Math.max(
    Math.abs(localDirection[0]),
    Math.abs(localDirection[1]),
    Math.abs(localDirection[2]),
  );
  if (!(localDirectionScale > 0) || !Number.isFinite(localDirectionScale)) return null;
  return {
    o: [origin[0] * inverseW, origin[1] * inverseW, origin[2] * inverseW],
    d: localDirection,
  };
}

function intersectAnalytic(prim: AnalyticPrimitive, worldRay: Ray): number | null {
  const ray = transformRayToLocal(worldRay, prim.transform);
  if (ray == null) return null;
  switch (prim.shape) {
    case 'sphere': {
      const [cx, cy, cz, radius] = decodeAnalyticParams('sphere', prim.params);
      return raySphere(ray, [cx, cy, cz], radius);
    }
    case 'box': {
      const [cx, cy, cz, hx, hy, hz] = decodeAnalyticParams('box', prim.params);
      return rayAabb(ray, [cx - hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz + hz]);
    }
    case 'cylinder': {
      const [cx, cy, cz, radius, halfHeight] = decodeAnalyticParams('cylinder', prim.params);
      return rayCylinderY(ray, [cx, cy, cz], radius, halfHeight);
    }
    case 'capsule': {
      const [ax, ay, az, bx, by, bz, radius] = decodeAnalyticParams('capsule', prim.params);
      return rayCapsule(ray, [ax, ay, az], [bx, by, bz], radius);
    }
    case 'h-channel-came': {
      const [length, railWidth, blockHeight, webThickness] = decodeAnalyticParams(
        'h-channel-came',
        prim.params,
      );
      const hx = length * 0.5;
      const hy = blockHeight * 0.5;
      const hz = railWidth * 0.5;
      const halfWeb = webThickness * 0.5;
      return rayAabbUnion(ray, [
        [
          [-hx, hy - halfWeb, -hz],
          [hx, hy, hz],
        ],
        [
          [-hx, -hy, -hz],
          [hx, -hy + halfWeb, hz],
        ],
        [
          [-hx, -hy + halfWeb, -halfWeb],
          [hx, hy - halfWeb, halfWeb],
        ],
      ]);
    }
  }
}

function intersectPrimitive(prim: ScenePrimitive, ray: Ray): number | null {
  switch (prim.kind) {
    case 'mesh':
      return intersectTriangleSoup(prim.positions, prim.indices, prim.transform, ray);
    case 'skinned-mesh': {
      // Debug picking must follow the geometry currently visible on screen.
      // `positions` is the rest-pose stream; the current bone matrices and
      // morph weights live on the primitive and are resolved by the canonical
      // core solver into the same mesh-local space before `transform`.
      const posed = solveSkin(prim);
      return intersectTriangleSoup(posed.positions, prim.indices, prim.transform, ray);
    }
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
        return intersectTriangleSoup(
          prim.fallbackMesh.positions,
          prim.fallbackMesh.indices,
          prim.transform,
          ray,
        );
      }
      return intersectAnalytic(prim, ray);
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
