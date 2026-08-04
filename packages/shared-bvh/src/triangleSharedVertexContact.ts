type V3 = readonly [number, number, number];
type V2 = readonly [number, number];

/** Internal exact-topology helper; intentionally not exported from package root. */
export function coplanarTrianglesContactBeyondSharedVertex(
  a: readonly [V3, V3, V3],
  b: readonly [V3, V3, V3],
  shared: V3,
): boolean {
  const normal = cross(sub(a[1], a[0]), sub(a[2], a[0]));
  const drop = dominantAxis(normal);
  const a2 = a.map((point) => project2(point, drop));
  const b2 = b.map((point) => project2(point, drop));
  const shared2 = project2(shared, drop);
  for (const point of a2) {
    if (same2(point, shared2)) continue;
    if (pointInTriangle2(point, b2)) return true;
  }
  for (const point of b2) {
    if (same2(point, shared2)) continue;
    if (pointInTriangle2(point, a2)) return true;
  }
  for (let edgeA = 0; edgeA < 3; edgeA += 1) {
    for (let edgeB = 0; edgeB < 3; edgeB += 1) {
      if (segmentsIntersectBeyondSharedPoint2(
        a2[edgeA]!, a2[(edgeA + 1) % 3]!,
        b2[edgeB]!, b2[(edgeB + 1) % 3]!,
        shared2,
      )) return true;
    }
  }
  return false;
}

/** Internal exact-topology helper; intentionally not exported from package root. */
export function nonCoplanarTrianglesContactBeyondSharedVertex(
  a: readonly [V3, V3, V3],
  b: readonly [V3, V3, V3],
  shared: V3,
  tolerance: number,
): boolean {
  const normalA = cross(sub(a[1], a[0]), sub(a[2], a[0]));
  const normalB = cross(sub(b[1], b[0]), sub(b[2], b[0]));
  const direction = normalize(cross(normalA, normalB));
  const ia = trianglePlaneLineInterval(a, b[0], normalB, shared, direction, tolerance);
  const ib = trianglePlaneLineInterval(b, a[0], normalA, shared, direction, tolerance);
  if (ia == null || ib == null) return true;
  const overlap: readonly [number, number] = [
    Math.max(ia[0], ib[0]),
    Math.min(ia[1], ib[1]),
  ];
  // The shared vertex is parameter zero. A second isolated intersection has a
  // zero-width interval away from zero and is still invalid self-contact.
  return overlap[0] < -tolerance || overlap[1] > tolerance;
}

function trianglePlaneLineInterval(
  triangle: readonly [V3, V3, V3],
  planePoint: V3,
  planeNormal: V3,
  lineOrigin: V3,
  lineDirection: V3,
  tolerance: number,
): readonly [number, number] | null {
  const unitNormal = normalize(planeNormal);
  const distances = triangle.map((point) => {
    const distance = dot(sub(point, planePoint), unitNormal);
    return Math.abs(distance) <= tolerance ? 0 : distance;
  });
  const points: V3[] = [];
  for (let vertex = 0; vertex < 3; vertex += 1) {
    if (distances[vertex] === 0) points.push(triangle[vertex]!);
    const next = (vertex + 1) % 3;
    const da = distances[vertex]!;
    const db = distances[next]!;
    if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
      const t = da / (da - db);
      points.push(add(triangle[vertex]!, scale(sub(triangle[next]!, triangle[vertex]!), t)));
    }
  }
  if (points.length === 0) return null;
  const denominator = lengthSquared(lineDirection);
  const values = points.map((point) => dot(sub(point, lineOrigin), lineDirection) / denominator);
  return [Math.min(...values), Math.max(...values)];
}

function dominantAxis(normal: V3): 0 | 1 | 2 {
  const x = Math.abs(normal[0]);
  const y = Math.abs(normal[1]);
  const z = Math.abs(normal[2]);
  return x >= y && x >= z ? 0 : y >= z ? 1 : 2;
}

function project2(point: V3, drop: 0 | 1 | 2): V2 {
  return drop === 0 ? [point[1], point[2]] :
    drop === 1 ? [point[0], point[2]] : [point[0], point[1]];
}

function orient2(a: V2, b: V2, c: V2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function same2(a: V2, b: V2): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

function pointInTriangle2(point: V2, triangle: readonly V2[]): boolean {
  const o0 = orient2(triangle[0]!, triangle[1]!, point);
  const o1 = orient2(triangle[1]!, triangle[2]!, point);
  const o2 = orient2(triangle[2]!, triangle[0]!, point);
  return (o0 >= 0 && o1 >= 0 && o2 >= 0) ||
    (o0 <= 0 && o1 <= 0 && o2 <= 0);
}

function between(value: number, a: number, b: number): boolean {
  return value >= Math.min(a, b) && value <= Math.max(a, b);
}

function pointOnSegment2(point: V2, a: V2, b: V2): boolean {
  return orient2(a, b, point) === 0 &&
    between(point[0], a[0], b[0]) && between(point[1], a[1], b[1]);
}

function segmentsIntersectBeyondSharedPoint2(
  a0: V2,
  a1: V2,
  b0: V2,
  b1: V2,
  shared: V2,
): boolean {
  const o0 = orient2(a0, a1, b0);
  const o1 = orient2(a0, a1, b1);
  const o2 = orient2(b0, b1, a0);
  const o3 = orient2(b0, b1, a1);
  if (
    ((o0 > 0 && o1 < 0) || (o0 < 0 && o1 > 0)) &&
    ((o2 > 0 && o3 < 0) || (o2 < 0 && o3 > 0))
  ) return true;
  for (const point of [a0, a1]) {
    if (!same2(point, shared) && pointOnSegment2(point, b0, b1)) return true;
  }
  for (const point of [b0, b1]) {
    if (!same2(point, shared) && pointOnSegment2(point, a0, a1)) return true;
  }
  return false;
}

function add(a: V3, b: V3): V3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: V3, b: V3): V3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(value: V3, scalar: number): V3 {
  return [value[0] * scalar, value[1] * scalar, value[2] * scalar];
}

function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: V3, b: V3): V3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function lengthSquared(value: V3): number {
  return dot(value, value);
}

function normalize(value: V3): V3 {
  return scale(value, 1 / Math.sqrt(lengthSquared(value)));
}
