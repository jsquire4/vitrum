/** CPU/f32 oracle for the shared optical watertight WGSL contract. */
export type OpticalV3 = readonly [number, number, number];

export interface OpticalWatertightCpuHit {
  readonly hit: boolean;
  readonly t: number;
  readonly point: OpticalV3;
  readonly bary: OpticalV3;
  readonly side: -1 | 0 | 1;
  readonly zeroEdgeMask: number;
}

const f = Math.fround;

function sub3(a: OpticalV3, b: OpticalV3): OpticalV3 {
  return [f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2])];
}

function edgeFunction(a: readonly [number, number], b: readonly [number, number]): number {
  if (a[0] === b[0] && a[1] === b[1]) return 0;
  const ordered = a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const first = ordered ? a : b;
  const second = ordered ? b : a;
  const canonical = f(f(first[0] * second[1]) - f(first[1] * second[0]));
  return ordered ? canonical : f(-canonical);
}

function rayParameter(origin: OpticalV3, direction: OpticalV3, point: OpticalV3): number {
  const delta = sub3(point, origin);
  const deltaScale = Math.max(Math.abs(delta[0]), Math.abs(delta[1]), Math.abs(delta[2]));
  const directionScale = Math.max(
    Math.abs(direction[0]), Math.abs(direction[1]), Math.abs(direction[2]),
  );
  if (!(directionScale > 0) || !Number.isFinite(directionScale) || !Number.isFinite(deltaScale)) {
    return -1;
  }
  if (!(deltaScale > 0)) return 0;
  const d: OpticalV3 = [
    f(direction[0] / directionScale),
    f(direction[1] / directionScale),
    f(direction[2] / directionScale),
  ];
  const q: OpticalV3 = [
    f(delta[0] / deltaScale),
    f(delta[1] / deltaScale),
    f(delta[2] / deltaScale),
  ];
  const numerator = f(f(f(q[0] * d[0]) + f(q[1] * d[1])) + f(q[2] * d[2]));
  const denominator = f(f(f(d[0] * d[0]) + f(d[1] * d[1])) + f(d[2] * d[2]));
  return f(f(numerator / denominator) * f(deltaScale / directionScale));
}

function lexBefore(a: OpticalV3, b: OpticalV3): boolean {
  return a[0] < b[0] || (a[0] === b[0] && (
    a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])
  ));
}

function edgeRayParameter(
  origin: OpticalV3,
  direction: OpticalV3,
  firstInput: OpticalV3,
  secondInput: OpticalV3,
  firstProjectedInput: readonly [number, number],
  secondProjectedInput: readonly [number, number],
): { readonly t: number; readonly point: OpticalV3 } {
  const ordered = lexBefore(firstInput, secondInput);
  const first = ordered ? firstInput : secondInput;
  const second = ordered ? secondInput : firstInput;
  const fp = ordered ? firstProjectedInput : secondProjectedInput;
  const sp = ordered ? secondProjectedInput : firstProjectedInput;
  const delta: readonly [number, number] = [f(sp[0] - fp[0]), f(sp[1] - fp[1])];
  let interpolation: number;
  if (Math.abs(delta[0]) >= Math.abs(delta[1])) {
    if (delta[0] === 0) return { t: -1, point: [0, 0, 0] };
    interpolation = f(-fp[0] / delta[0]);
  } else {
    if (delta[1] === 0) return { t: -1, point: [0, 0, 0] };
    interpolation = f(-fp[1] / delta[1]);
  }
  const point: OpticalV3 = [
    f(first[0] + f(interpolation * f(second[0] - first[0]))),
    f(first[1] + f(interpolation * f(second[1] - first[1]))),
    f(first[2] + f(interpolation * f(second[2] - first[2]))),
  ];
  return { t: rayParameter(origin, direction, point), point };
}

const MISS: OpticalWatertightCpuHit = {
  hit: false,
  t: Infinity,
  point: [0, 0, 0],
  bary: [0, 0, 0],
  side: 0,
  zeroEdgeMask: 0,
};

export function intersectOpticalTriangleWatertightF32(
  origin: OpticalV3,
  direction: OpticalV3,
  a: OpticalV3,
  b: OpticalV3,
  c: OpticalV3,
  exclusiveMinT = 0,
): OpticalWatertightCpuHit {
  const directionScale = Math.max(
    Math.abs(direction[0]), Math.abs(direction[1]), Math.abs(direction[2]),
  );
  if (!(directionScale > 0) || !Number.isFinite(directionScale) || Number.isNaN(exclusiveMinT)) {
    return MISS;
  }
  const d: OpticalV3 = [
    f(direction[0] / directionScale),
    f(direction[1] / directionScale),
    f(direction[2] / directionScale),
  ];
  let kz = Math.abs(d[1]) > Math.abs(d[0]) ? 1 : 0;
  if (Math.abs(d[2]) > Math.abs(d[kz]!)) kz = 2;
  let kx = (kz + 1) % 3;
  let ky = (kx + 1) % 3;
  if (d[kz]! < 0) [kx, ky] = [ky, kx];
  const sx = f(d[kx]! / d[kz]!);
  const sy = f(d[ky]! / d[kz]!);
  const sz = f(1 / d[kz]!);
  const pa = sub3(a, origin);
  const pb = sub3(b, origin);
  const pc = sub3(c, origin);
  const project = (point: OpticalV3): readonly [number, number] => [
    f(point[kx]! - f(sx * point[kz]!)),
    f(point[ky]! - f(sy * point[kz]!)),
  ];
  const ap = project(pa);
  const bp = project(pb);
  const cp = project(pc);
  const e0 = edgeFunction(bp, cp);
  const e1 = edgeFunction(cp, ap);
  const e2 = edgeFunction(ap, bp);
  const determinant = f(f(e0 + e1) + e2);
  if (determinant === 0 || !Number.isFinite(determinant)) return MISS;
  const positive = determinant > 0;
  const n0 = positive ? e0 : f(-e0);
  const n1 = positive ? e1 : f(-e1);
  const n2 = positive ? e2 : f(-e2);
  if (n0 < 0 || n1 < 0 || n2 < 0) return MISS;
  const az = f(pa[kz]! * sz);
  const bz = f(pb[kz]! * sz);
  const cz = f(pc[kz]! * sz);
  const scaledT = f(f(f(e0 * az) + f(e1 * bz)) + f(e2 * cz));
  let t = f(f(scaledT / determinant) / directionScale);
  let point: OpticalV3 = [0, 0, 0];
  const zeroCount = Number(n0 === 0) + Number(n1 === 0) + Number(n2 === 0);
  if (zeroCount >= 2) {
    if (n0 !== 0) { point = a; t = rayParameter(origin, direction, point); }
    if (n1 !== 0) { point = b; t = rayParameter(origin, direction, point); }
    if (n2 !== 0) { point = c; t = rayParameter(origin, direction, point); }
  } else if (n0 === 0) {
    ({ t, point } = edgeRayParameter(origin, direction, b, c, bp, cp));
  } else if (n1 === 0) {
    ({ t, point } = edgeRayParameter(origin, direction, c, a, cp, ap));
  } else if (n2 === 0) {
    ({ t, point } = edgeRayParameter(origin, direction, a, b, ap, bp));
  }
  if (!(t > Math.max(exclusiveMinT, 0)) || !Number.isFinite(t)) return MISS;
  const e1Raw = sub3(b, a);
  const e2Raw = sub3(c, a);
  const edgeScale = Math.max(
    Math.abs(e1Raw[0]), Math.abs(e1Raw[1]), Math.abs(e1Raw[2]),
    Math.abs(e2Raw[0]), Math.abs(e2Raw[1]), Math.abs(e2Raw[2]),
  );
  if (!(edgeScale > 0) || !Number.isFinite(edgeScale)) return MISS;
  const scaledEdge1: OpticalV3 = [
    f(e1Raw[0] / edgeScale), f(e1Raw[1] / edgeScale), f(e1Raw[2] / edgeScale),
  ];
  const scaledEdge2: OpticalV3 = [
    f(e2Raw[0] / edgeScale), f(e2Raw[1] / edgeScale), f(e2Raw[2] / edgeScale),
  ];
  const normalRaw: OpticalV3 = [
    f(f(scaledEdge1[1] * scaledEdge2[2]) - f(scaledEdge1[2] * scaledEdge2[1])),
    f(f(scaledEdge1[2] * scaledEdge2[0]) - f(scaledEdge1[0] * scaledEdge2[2])),
    f(f(scaledEdge1[0] * scaledEdge2[1]) - f(scaledEdge1[1] * scaledEdge2[0])),
  ];
  const normalScale = Math.max(
    Math.abs(normalRaw[0]), Math.abs(normalRaw[1]), Math.abs(normalRaw[2]),
  );
  if (!(normalScale > 0) || !Number.isFinite(normalScale)) return MISS;
  const normalDirection: OpticalV3 = [
    f(normalRaw[0] / normalScale),
    f(normalRaw[1] / normalScale),
    f(normalRaw[2] / normalScale),
  ];
  const normalLengthSquared = f(f(
    f(f(normalDirection[0] * normalDirection[0]) +
      f(normalDirection[1] * normalDirection[1])),
  ) + f(normalDirection[2] * normalDirection[2]));
  const normalLength = f(Math.sqrt(normalLengthSquared));
  if (!(normalLength > 0) || !Number.isFinite(normalLength)) return MISS;
  const normal: OpticalV3 = [
    f(normalDirection[0] / normalLength),
    f(normalDirection[1] / normalLength),
    f(normalDirection[2] / normalLength),
  ];
  const sideDet = f(-f(f(f(d[0] * normal[0]) + f(d[1] * normal[1])) + f(d[2] * normal[2])));
  if (sideDet === 0 || !Number.isFinite(sideDet)) return MISS;
  const bary: OpticalV3 = [f(e0 / determinant), f(e1 / determinant), f(e2 / determinant)];
  if (zeroCount === 0) {
    point = [
      f(a[0] + f(f(bary[1] * f(b[0] - a[0])) + f(bary[2] * f(c[0] - a[0])))),
      f(a[1] + f(f(bary[1] * f(b[1] - a[1])) + f(bary[2] * f(c[1] - a[1])))),
      f(a[2] + f(f(bary[1] * f(b[2] - a[2])) + f(bary[2] * f(c[2] - a[2])))),
    ];
  }
  return {
    hit: true,
    t,
    point,
    bary,
    side: sideDet > 0 ? 1 : -1,
    zeroEdgeMask: (n0 === 0 ? 1 : 0) | (n1 === 0 ? 2 : 0) | (n2 === 0 ? 4 : 0),
  };
}

export interface OpticalBoundaryEventCandidate {
  readonly t: number;
  readonly encodedBoundaryId: number;
  readonly side: -1 | 1;
}

export type GroupedOpticalBoundaryEvent =
  | { readonly kind: 'none' }
  | { readonly kind: 'invalid-input' }
  | { readonly kind: 'crossing'; readonly t: number; readonly encodedBoundaryId: number; readonly side: -1 | 1 }
  | { readonly kind: 'tangent'; readonly t: number; readonly encodedBoundaryId: number }
  | { readonly kind: 'invalid-tie'; readonly t: number };

function f32Bits(value: number): number {
  const values = new Float32Array([value]);
  return new Uint32Array(values.buffer)[0]!;
}

/** Exact-f32-t aggregation contract shared by all optical traversal lanes. */
export function groupOpticalBoundaryEventCandidates(
  candidates: readonly OpticalBoundaryEventCandidate[],
): GroupedOpticalBoundaryEvent {
  if (candidates.length === 0) return { kind: 'none' };
  const represented = candidates.map((candidate) => ({
    ...candidate,
    t: f(candidate.t),
  }));
  if (represented.some((candidate) => (
    !(candidate.t > 0) || !Number.isFinite(candidate.t) ||
    !Number.isInteger(candidate.encodedBoundaryId) || candidate.encodedBoundaryId <= 0 ||
    (candidate.side !== -1 && candidate.side !== 1)
  ))) {
    return { kind: 'invalid-input' };
  }
  let minimum = represented[0]!.t;
  for (const candidate of represented) minimum = Math.min(minimum, candidate.t);
  const minimumBits = f32Bits(minimum);
  const tied = represented.filter((candidate) => f32Bits(candidate.t) === minimumBits);
  const boundary = tied[0]!.encodedBoundaryId;
  if (boundary === 0 || tied.some((candidate) => candidate.encodedBoundaryId !== boundary)) {
    return { kind: 'invalid-tie', t: minimum };
  }
  let front = 0;
  let back = 0;
  for (const candidate of tied) {
    if (candidate.side > 0) front += 1;
    else back += 1;
  }
  if (front > 0 && back > 0) {
    return front === back
      ? { kind: 'tangent', t: minimum, encodedBoundaryId: boundary }
      : { kind: 'invalid-tie', t: minimum };
  }
  return {
    kind: 'crossing',
    t: minimum,
    encodedBoundaryId: boundary,
    side: front > 0 ? 1 : -1,
  };
}

export interface OpticalSourceFeatureCandidate {
  readonly encodedBoundaryId: number;
  /**
   * Encoded (`ordinal + 1`) represented primitive-instance/range identity;
   * zero is invalid. TLAS traversal can encode its packed instance slot;
   * flattened/direct traversal must encode the
   * source primitive range rather than a constant zero. This remains required
   * for thin sheets, whose encoded boundary ID is intentionally zero.
   */
  readonly representedPrimitiveInstanceId: number;
  /** Global packed triangle ordinal, not a material or source-primitive ID. */
  readonly triangleIndex: number;
  /** The three represented world-space f32 vertices in index order. */
  readonly vertices: readonly [OpticalV3, OpticalV3, OpticalV3];
}

export type OpticalSourceFeature =
  | {
    readonly kind: 'face';
    readonly encodedBoundaryId: number;
    readonly representedPrimitiveInstanceId: number;
    readonly triangleIndex: number;
  }
  | {
    readonly kind: 'edge';
    readonly encodedBoundaryId: number;
    readonly representedPrimitiveInstanceId: number;
    readonly first: OpticalV3;
    readonly second: OpticalV3;
  }
  | {
    readonly kind: 'vertex';
    readonly encodedBoundaryId: number;
    readonly representedPrimitiveInstanceId: number;
    readonly point: OpticalV3;
  };

function samePoint(a: OpticalV3, b: OpticalV3): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function canonicalEdge(
  a: OpticalV3,
  b: OpticalV3,
): readonly [OpticalV3, OpticalV3] {
  return lexBefore(a, b) ? [a, b] : [b, a];
}

/**
 * Classifies the exact source feature from the watertight edge-zero mask.
 * No barycentric epsilon is permitted here: a near-edge interior hit is a
 * face hit and must not suppress an adjacent represented face.
 */
export function createOpticalSourceFeature(
  candidate: OpticalSourceFeatureCandidate,
  zeroEdgeMask: number,
): OpticalSourceFeature {
  if (
    !Number.isInteger(candidate.representedPrimitiveInstanceId) ||
    candidate.representedPrimitiveInstanceId <= 0 ||
    candidate.representedPrimitiveInstanceId > 0xffff_ffff
  ) {
    throw new RangeError(
      'Optical source feature requires an encoded represented primitive-instance ID.',
    );
  }
  const common = {
    encodedBoundaryId: candidate.encodedBoundaryId,
    representedPrimitiveInstanceId: candidate.representedPrimitiveInstanceId,
  } as const;
  const [a, b, c] = candidate.vertices;
  switch (zeroEdgeMask) {
    case 0:
      return { kind: 'face', ...common, triangleIndex: candidate.triangleIndex };
    case 1: {
      const [first, second] = canonicalEdge(b, c);
      return { kind: 'edge', ...common, first, second };
    }
    case 2: {
      const [first, second] = canonicalEdge(c, a);
      return { kind: 'edge', ...common, first, second };
    }
    case 4: {
      const [first, second] = canonicalEdge(a, b);
      return { kind: 'edge', ...common, first, second };
    }
    case 3:
      return { kind: 'vertex', ...common, point: c };
    case 5:
      return { kind: 'vertex', ...common, point: b };
    case 6:
      return { kind: 'vertex', ...common, point: a };
    default:
      throw new RangeError(
        `Optical source feature requires a valid exact zero-edge mask; got ${zeroEdgeMask}.`,
      );
  }
}

function candidateContainsPoint(
  candidate: OpticalSourceFeatureCandidate,
  point: OpticalV3,
): boolean {
  return candidate.vertices.some((vertex) => samePoint(vertex, point));
}

/**
 * Returns true only for the represented source feature itself. Edge and
 * vertex matching deliberately uses represented world-space coordinates, so
 * duplicated-index UV seams and f32-welded instance transforms cannot cause
 * an immediate self-hit. A nearby, geometrically distinct face remains live.
 */
export function opticalSourceFeatureSuppressesCandidate(
  feature: OpticalSourceFeature,
  candidate: OpticalSourceFeatureCandidate,
): boolean {
  if (
    candidate.encodedBoundaryId !== feature.encodedBoundaryId ||
    candidate.representedPrimitiveInstanceId !== feature.representedPrimitiveInstanceId
  ) {
    return false;
  }
  if (feature.kind === 'face') {
    return candidate.triangleIndex === feature.triangleIndex;
  }
  if (feature.kind === 'vertex') {
    return candidateContainsPoint(candidate, feature.point);
  }
  return candidateContainsPoint(candidate, feature.first) &&
    candidateContainsPoint(candidate, feature.second);
}
