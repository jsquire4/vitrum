/**
 * Watertight, inclusive triangle intersection for optical-boundary replay.
 *
 * Ordinary radiance traversal intentionally keeps the existing tolerant
 * Moller-Trumbore kernel. Optical medium replay has a stricter requirement:
 * an exact ray through a triangulation diagonal or vertex must produce
 * bit-identical candidates that can be grouped into one represented crossing.
 * This is the dominant-axis/sheared-edge formulation from Woop, Benthin &
 * Wald. Edge/vertex candidates remain inclusive: traversal must see a tied
 * hit from a distinct boundary ID and fail closed instead of hiding it behind
 * an ownership rule.
 *
 * `exclusiveMinT` is a ray-parameter lower bound. Keeping the ray origin fixed
 * and increasing this value avoids an origin bias jumping over a nearby nested
 * optical boundary.
 *
 * @see CREDITS.md (Woop, Benthin & Wald 2013)
 */
export const OPTICAL_WATERTIGHT_TRIANGLE_WGSL = /* wgsl */ `
struct OpticalWatertightHit {
  hit: bool,
  t: f32,
  // The represented f32 hit point. Edge and vertex hits are reconstructed
  // solely from their canonical shared feature, so neighboring triangles do
  // not reintroduce their unrelated third vertex through barycentric ratios.
  position: vec3f,
  bary: vec3f,
  side: f32,
  normal: vec3f,
  // Bit i is set only when exact projected edge function ei is zero. This is
  // the sole source-feature classifier; tolerant Moller barycentrics must not
  // be used to infer face/edge/vertex ownership.
  zeroEdgeMask: u32,
};

fn opticalWatertightMiss() -> OpticalWatertightHit {
  var result: OpticalWatertightHit;
  result.hit = false;
  result.t = 3.402823e38;
  result.position = vec3f(0.0);
  result.bary = vec3f(0.0);
  result.side = 0.0;
  result.normal = vec3f(0.0);
  result.zeroEdgeMask = 0u;
  return result;
}

// Evaluate an edge in a canonical endpoint order, then restore its requested
// direction. Reversing an edge is therefore exactly a sign-bit change instead
// of a second, potentially differently-rounded multiply/subtract sequence.
fn opticalProjectedEdgeFunction(a: vec2f, b: vec2f) -> f32 {
  if (all(a == b)) { return 0.0; }
  let ordered = a.x < b.x || (a.x == b.x && a.y < b.y);
  let first = select(b, a, ordered);
  let second = select(a, b, ordered);
  let canonical = first.x * second.y - first.y * second.x;
  return select(-canonical, canonical, ordered);
}

fn opticalRayParameterAtPoint(
  origin: vec3f,
  direction: vec3f,
  point: vec3f,
) -> f32 {
  let delta = point - origin;
  let deltaScale = max(abs(delta.x), max(abs(delta.y), abs(delta.z)));
  let directionScale = max(
    abs(direction.x), max(abs(direction.y), abs(direction.z)),
  );
  if (
    !(directionScale > 0.0) || directionScale > 3.402823e38 ||
    deltaScale > 3.402823e38
  ) {
    return -1.0;
  }
  if (!(deltaScale > 0.0)) { return 0.0; }
  let scaledDirection = direction / directionScale;
  return (
    dot(delta / deltaScale, scaledDirection) /
      dot(scaledDirection, scaledDirection)
  ) * (deltaScale / directionScale);
}

fn opticalPointLexicographicallyBefore(a: vec3f, b: vec3f) -> bool {
  return a.x < b.x ||
    (a.x == b.x && (
      a.y < b.y || (a.y == b.y && a.z < b.z)
    ));
}

struct OpticalProjectedEdgeHit {
  t: f32,
  point: vec3f,
};

// A shared-edge hit uses the exact same canonically ordered endpoints in both
// incident triangles. Computing its ray parameter from that edge, rather than
// from each face's determinant, makes the tied f32 t bit-identical.
fn opticalRayParameterAtProjectedEdge(
  origin: vec3f,
  direction: vec3f,
  firstInput: vec3f,
  secondInput: vec3f,
  firstProjectedInput: vec2f,
  secondProjectedInput: vec2f,
) -> OpticalProjectedEdgeHit {
  var result: OpticalProjectedEdgeHit;
  result.t = -1.0;
  result.point = vec3f(0.0);
  let ordered = opticalPointLexicographicallyBefore(firstInput, secondInput);
  let first = select(secondInput, firstInput, ordered);
  let second = select(firstInput, secondInput, ordered);
  let firstProjected = select(secondProjectedInput, firstProjectedInput, ordered);
  let secondProjected = select(firstProjectedInput, secondProjectedInput, ordered);
  let projectedDelta = secondProjected - firstProjected;
  var interpolation = 0.0;
  if (abs(projectedDelta.x) >= abs(projectedDelta.y)) {
    if (projectedDelta.x == 0.0) { return result; }
    interpolation = -firstProjected.x / projectedDelta.x;
  } else {
    if (projectedDelta.y == 0.0) { return result; }
    interpolation = -firstProjected.y / projectedDelta.y;
  }
  result.point = first + interpolation * (second - first);
  result.t = opticalRayParameterAtPoint(origin, direction, result.point);
  return result;
}

fn opticalWatertightTriangleIntersect(
  origin: vec3f,
  direction: vec3f,
  a: vec3f,
  b: vec3f,
  c: vec3f,
  exclusiveMinT: f32,
) -> OpticalWatertightHit {
  var result = opticalWatertightMiss();

  let directionScale = max(
    abs(direction.x), max(abs(direction.y), abs(direction.z)),
  );
  if (
    !(directionScale > 0.0) || directionScale > 3.402823e38 ||
    exclusiveMinT != exclusiveMinT
  ) {
    return result;
  }
  let d = direction / directionScale;

  // Permute the dominant direction onto Z. Swapping X/Y for a negative
  // dominant direction gives every ray one canonical projected orientation.
  var kz = 0u;
  if (abs(d.y) > abs(d.x)) { kz = 1u; }
  if (abs(d.z) > abs(d[kz])) { kz = 2u; }
  var kx = (kz + 1u) % 3u;
  var ky = (kx + 1u) % 3u;
  if (d[kz] < 0.0) {
    let swap = kx;
    kx = ky;
    ky = swap;
  }

  let sx = d[kx] / d[kz];
  let sy = d[ky] / d[kz];
  let sz = 1.0 / d[kz];
  let pa = a - origin;
  let pb = b - origin;
  let pc = c - origin;
  let ap = vec2f(pa[kx] - sx * pa[kz], pa[ky] - sy * pa[kz]);
  let bp = vec2f(pb[kx] - sx * pb[kz], pb[ky] - sy * pb[kz]);
  let cp = vec2f(pc[kx] - sx * pc[kz], pc[ky] - sy * pc[kz]);
  if (
    any(ap != ap) || any(bp != bp) || any(cp != cp) ||
    any(abs(ap) > vec2f(3.402823e38)) ||
    any(abs(bp) > vec2f(3.402823e38)) ||
    any(abs(cp) > vec2f(3.402823e38))
  ) {
    return result;
  }

  // Edge functions at the projected ray origin. Canonical endpoint ordering
  // makes the shared-edge value bitwise antisymmetric between neighbors.
  let e0 = opticalProjectedEdgeFunction(bp, cp);
  let e1 = opticalProjectedEdgeFunction(cp, ap);
  let e2 = opticalProjectedEdgeFunction(ap, bp);
  let determinant = e0 + e1 + e2;
  if (!(determinant != 0.0) || determinant != determinant) { return result; }

  let positive = determinant > 0.0;
  let n0 = select(-e0, e0, positive);
  let n1 = select(-e1, e1, positive);
  let n2 = select(-e2, e2, positive);
  if (n0 < 0.0 || n1 < 0.0 || n2 < 0.0) { return result; }

  let az = pa[kz] * sz;
  let bz = pb[kz] * sz;
  let cz = pc[kz] * sz;
  let scaledT = e0 * az + e1 * bz + e2 * cz;
  var t = (scaledT / determinant) / directionScale;
  var canonicalPoint = vec3f(0.0);
  let zeroCount = select(0u, 1u, n0 == 0.0) +
    select(0u, 1u, n1 == 0.0) +
    select(0u, 1u, n2 == 0.0);
  if (zeroCount >= 2u) {
    // Two zero barycentric edge functions identify one shared vertex. Every
    // incident triangle computes t directly from that identical f32 vertex.
    if (n0 != 0.0) {
      canonicalPoint = a;
      t = opticalRayParameterAtPoint(origin, direction, canonicalPoint);
    }
    if (n1 != 0.0) {
      canonicalPoint = b;
      t = opticalRayParameterAtPoint(origin, direction, canonicalPoint);
    }
    if (n2 != 0.0) {
      canonicalPoint = c;
      t = opticalRayParameterAtPoint(origin, direction, canonicalPoint);
    }
  } else if (n0 == 0.0) {
    let edgeHit = opticalRayParameterAtProjectedEdge(
      origin, direction, b, c, bp, cp,
    );
    t = edgeHit.t;
    canonicalPoint = edgeHit.point;
  } else if (n1 == 0.0) {
    let edgeHit = opticalRayParameterAtProjectedEdge(
      origin, direction, c, a, cp, ap,
    );
    t = edgeHit.t;
    canonicalPoint = edgeHit.point;
  } else if (n2 == 0.0) {
    let edgeHit = opticalRayParameterAtProjectedEdge(
      origin, direction, a, b, ap, bp,
    );
    t = edgeHit.t;
    canonicalPoint = edgeHit.point;
  }
  if (
    !(t > max(exclusiveMinT, 0.0)) ||
    t > 3.402823e38
  ) {
    return result;
  }

  let edge0Raw = b - a;
  let edge1Raw = c - a;
  let edgeScale = max(
    max(abs(edge0Raw.x), max(abs(edge0Raw.y), abs(edge0Raw.z))),
    max(abs(edge1Raw.x), max(abs(edge1Raw.y), abs(edge1Raw.z))),
  );
  if (!(edgeScale > 0.0) || edgeScale > 3.402823e38) { return result; }
  let authoredNormalRaw = cross(edge0Raw / edgeScale, edge1Raw / edgeScale);
  let normalScale = max(
    abs(authoredNormalRaw.x),
    max(abs(authoredNormalRaw.y), abs(authoredNormalRaw.z)),
  );
  if (!(normalScale > 0.0) || normalScale > 3.402823e38) { return result; }
  let authoredNormalDirection = authoredNormalRaw / normalScale;
  let authoredNormal = authoredNormalDirection / length(authoredNormalDirection);
  let sideDeterminant = -dot(d, authoredNormal);
  if (sideDeterminant == 0.0 || sideDeterminant != sideDeterminant) {
    return result;
  }

  let bary = vec3f(e0, e1, e2) / determinant;
  if (zeroCount == 0u) {
    canonicalPoint = a + bary.y * (b - a) + bary.z * (c - a);
  }
  result.hit = true;
  result.t = t;
  result.position = canonicalPoint;
  result.bary = bary;
  result.side = select(-1.0, 1.0, sideDeterminant > 0.0);
  result.normal = result.side * authoredNormal;
  result.zeroEdgeMask = select(0u, 1u, n0 == 0.0) |
    select(0u, 2u, n1 == 0.0) |
    select(0u, 4u, n2 == 0.0);
  return result;
}

// Reconstruct the accepted point from the represented triangle itself, never
// from origin + direction * t. The latter is an independently rounded ray
// point and can land on the far side of an immediately adjacent f32 surface.
// Exact vertex hits return the authored vertex bit-for-bit; edge hits use one
// canonical endpoint interpolation; face hits use a single anchored affine
// expression. This point is also the value transformed by TLAS traversal.
fn opticalCanonicalHitPoint(
  hit: OpticalWatertightHit,
  a: vec3f,
  b: vec3f,
  c: vec3f,
) -> vec3f {
  return hit.position;
}

const OPTICAL_BOUNDARY_EVENT_NONE: u32 = 0u;
const OPTICAL_BOUNDARY_EVENT_CROSSING: u32 = 1u;
const OPTICAL_BOUNDARY_EVENT_TANGENT: u32 = 2u;
const OPTICAL_BOUNDARY_EVENT_INVALID: u32 = 3u;

struct OpticalBoundaryEventAccumulator {
  hasCandidate: u32,
  invalidInput: u32,
  invalidTie: u32,
  t: f32,
  encodedBoundaryId: u32,
  frontCount: u32,
  backCount: u32,
};

struct OpticalBoundaryEvent {
  status: u32,
  t: f32,
  encodedBoundaryId: u32,
  side: f32,
};

fn opticalBoundaryEventAccumulatorInit() -> OpticalBoundaryEventAccumulator {
  var accumulator: OpticalBoundaryEventAccumulator;
  accumulator.hasCandidate = 0u;
  accumulator.invalidInput = 0u;
  accumulator.invalidTie = 0u;
  accumulator.t = 3.402823e38;
  accumulator.encodedBoundaryId = 0u;
  accumulator.frontCount = 0u;
  accumulator.backCount = 0u;
  return accumulator;
}

// Feed every inclusive optical triangle candidate to this accumulator. Only
// exact f32-equal nearest t values are grouped; there is intentionally no
// distance epsilon. A nearer candidate resets tie state, while malformed
// inputs poison the whole query so callers fail closed.
fn opticalBoundaryEventAccumulate(
  accumulator: ptr<function, OpticalBoundaryEventAccumulator>,
  candidateT: f32,
  encodedBoundaryId: u32,
  side: f32,
) {
  if (
    !(candidateT > 0.0) || candidateT > 3.402823e38 ||
    encodedBoundaryId == 0u || (side != -1.0 && side != 1.0)
  ) {
    (*accumulator).invalidInput = 1u;
    return;
  }
  if ((*accumulator).hasCandidate == 0u || candidateT < (*accumulator).t) {
    (*accumulator).hasCandidate = 1u;
    (*accumulator).invalidTie = 0u;
    (*accumulator).t = candidateT;
    (*accumulator).encodedBoundaryId = encodedBoundaryId;
    (*accumulator).frontCount = select(0u, 1u, side > 0.0);
    (*accumulator).backCount = select(0u, 1u, side < 0.0);
    return;
  }
  if (candidateT != (*accumulator).t) { return; }
  if (encodedBoundaryId != (*accumulator).encodedBoundaryId) {
    (*accumulator).invalidTie = 1u;
    return;
  }
  (*accumulator).frontCount += select(0u, 1u, side > 0.0);
  (*accumulator).backCount += select(0u, 1u, side < 0.0);
}

fn opticalBoundaryEventFinalize(
  accumulator: OpticalBoundaryEventAccumulator,
) -> OpticalBoundaryEvent {
  var event: OpticalBoundaryEvent;
  event.status = OPTICAL_BOUNDARY_EVENT_NONE;
  event.t = accumulator.t;
  event.encodedBoundaryId = accumulator.encodedBoundaryId;
  event.side = 0.0;
  if (accumulator.invalidInput != 0u || accumulator.invalidTie != 0u) {
    event.status = OPTICAL_BOUNDARY_EVENT_INVALID;
    return event;
  }
  if (accumulator.hasCandidate == 0u) { return event; }
  if (accumulator.frontCount > 0u && accumulator.backCount > 0u) {
    event.status = select(
      OPTICAL_BOUNDARY_EVENT_INVALID,
      OPTICAL_BOUNDARY_EVENT_TANGENT,
      accumulator.frontCount == accumulator.backCount,
    );
    return event;
  }
  event.status = OPTICAL_BOUNDARY_EVENT_CROSSING;
  event.side = select(-1.0, 1.0, accumulator.frontCount > 0u);
  return event;
}

const OPTICAL_SOURCE_FEATURE_INVALID: u32 = 0u;
const OPTICAL_SOURCE_FEATURE_FACE: u32 = 1u;
const OPTICAL_SOURCE_FEATURE_EDGE: u32 = 2u;
const OPTICAL_SOURCE_FEATURE_VERTEX: u32 = 3u;

struct OpticalSourceFeature {
  kind: u32,
  encodedBoundaryId: u32,
  representedPrimitiveInstanceId: u32,
  triangleIndex: u32,
  first: vec3f,
  second: vec3f,
};

fn opticalSourceFeatureInvalid() -> OpticalSourceFeature {
  var feature: OpticalSourceFeature;
  feature.kind = OPTICAL_SOURCE_FEATURE_INVALID;
  feature.encodedBoundaryId = 0u;
  feature.representedPrimitiveInstanceId = 0u;
  feature.triangleIndex = 0u;
  feature.first = vec3f(0.0);
  feature.second = vec3f(0.0);
  return feature;
}

fn opticalCanonicalSourceEdge(
  feature: ptr<function, OpticalSourceFeature>,
  a: vec3f,
  b: vec3f,
) {
  let ordered = opticalPointLexicographicallyBefore(a, b);
  (*feature).first = select(b, a, ordered);
  (*feature).second = select(a, b, ordered);
}

// zeroEdgeMask must come from opticalWatertightTriangleIntersect. In
// particular, a tolerant Moller barycentric near an edge must never be passed
// here as though it were an exact represented edge hit.
fn opticalCreateSourceFeature(
  encodedBoundaryId: u32,
  representedPrimitiveInstanceId: u32,
  triangleIndex: u32,
  zeroEdgeMask: u32,
  a: vec3f,
  b: vec3f,
  c: vec3f,
) -> OpticalSourceFeature {
  var feature = opticalSourceFeatureInvalid();
  if (representedPrimitiveInstanceId == 0u) { return feature; }
  feature.encodedBoundaryId = encodedBoundaryId;
  feature.representedPrimitiveInstanceId = representedPrimitiveInstanceId;
  feature.triangleIndex = triangleIndex;
  switch zeroEdgeMask {
    case 0u: {
      feature.kind = OPTICAL_SOURCE_FEATURE_FACE;
    }
    case 1u: {
      feature.kind = OPTICAL_SOURCE_FEATURE_EDGE;
      opticalCanonicalSourceEdge(&feature, b, c);
    }
    case 2u: {
      feature.kind = OPTICAL_SOURCE_FEATURE_EDGE;
      opticalCanonicalSourceEdge(&feature, c, a);
    }
    case 4u: {
      feature.kind = OPTICAL_SOURCE_FEATURE_EDGE;
      opticalCanonicalSourceEdge(&feature, a, b);
    }
    case 3u: {
      feature.kind = OPTICAL_SOURCE_FEATURE_VERTEX;
      feature.first = c;
    }
    case 5u: {
      feature.kind = OPTICAL_SOURCE_FEATURE_VERTEX;
      feature.first = b;
    }
    case 6u: {
      feature.kind = OPTICAL_SOURCE_FEATURE_VERTEX;
      feature.first = a;
    }
    default: {}
  }
  return feature;
}

fn opticalTriangleContainsExactPoint(
  point: vec3f,
  a: vec3f,
  b: vec3f,
  c: vec3f,
) -> bool {
  return all(point == a) || all(point == b) || all(point == c);
}

// Boundary identity alone is insufficient here: thin sheets intentionally
// use encodedBoundaryId zero. representedPrimitiveInstanceId is encoded as
// ordinal+1 (zero invalid) and scopes their exact coordinate fan to one packed
// instance or flattened primitive range.
fn opticalSourceFeatureSuppressesTriangle(
  feature: OpticalSourceFeature,
  encodedBoundaryId: u32,
  representedPrimitiveInstanceId: u32,
  triangleIndex: u32,
  a: vec3f,
  b: vec3f,
  c: vec3f,
) -> bool {
  if (
    feature.kind == OPTICAL_SOURCE_FEATURE_INVALID ||
    feature.encodedBoundaryId != encodedBoundaryId ||
    feature.representedPrimitiveInstanceId != representedPrimitiveInstanceId
  ) {
    return false;
  }
  if (feature.kind == OPTICAL_SOURCE_FEATURE_FACE) {
    return feature.triangleIndex == triangleIndex;
  }
  if (feature.kind == OPTICAL_SOURCE_FEATURE_VERTEX) {
    return opticalTriangleContainsExactPoint(feature.first, a, b, c);
  }
  if (feature.kind == OPTICAL_SOURCE_FEATURE_EDGE) {
    return opticalTriangleContainsExactPoint(feature.first, a, b, c) &&
      opticalTriangleContainsExactPoint(feature.second, a, b, c);
  }
  return false;
}
`;
