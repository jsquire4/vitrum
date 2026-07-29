/**
 * Scale-aware numeric primitives and the bounded 1–8 vertex manifold-NEE
 * solver used by the pt-webgpu caustic pipeline.
 *
 * The production estimator proposes every chain length, facet, and
 * reflection/transmission event explicitly, then solves the coupled
 * specular constraints with an O(N) block-tridiagonal Newton iteration.
 *
 * Ref: Hanika, Droske, Fascione, "Manifold Next Event Estimation," EGSR 2015;
 *      Jakob & Marschner, "Manifold Exploration," SIGGRAPH 2012.
 */

export const MNEE_BOUNDED_CHAIN_CORE_WGSL = /* wgsl */ `
const MNEE_F32_EPSILON = 1.1920928955078125e-7;
const MNEE_SQRT_F32_EPSILON = 3.4526698300124393e-4;
const MNEE_MAX_RELATIVE_LENGTH_FLOOR = 0.01;

fn mneePointCoordinateScale(point: vec3f) -> f32 {
  return max(abs(point.x), max(abs(point.y), abs(point.z)));
}

fn mneeLengthFloorFromScales(scales: vec2f) -> f32 {
  return max(
    max(params.triIntersectEpsilon, 0.0),
    max(
      scales.x * (4.0 * MNEE_F32_EPSILON),
      max(scales.y * (4.0 * MNEE_F32_EPSILON), bitcast<f32>(0x00800000u)),
    ),
  );
}

fn mneeFdStepFromScales(scales: vec2f) -> f32 {
  return max(
    mneeLengthFloorFromScales(scales),
    scales.y * MNEE_SQRT_F32_EPSILON,
  );
}

// Half-vector residuals are dimensionless. Convert the local representable
// length floor to an angular floor instead of imposing a metre-scale constant.
fn mneeResidualToleranceFromScales(scales: vec2f) -> f32 {
  let lengthFloor = mneeLengthFloorFromScales(scales);
  return max(
    16.0 * MNEE_F32_EPSILON,
    min(
      lengthFloor / max(scales.y, lengthFloor),
      MNEE_MAX_RELATIVE_LENGTH_FLOOR,
    ),
  );
}

fn mneeScalesRepresentable(scales: vec2f) -> bool {
  let lengthFloor = mneeLengthFloorFromScales(scales);
  return scales.x == scales.x && scales.y == scales.y &&
    abs(scales.x) < INFINITY && abs(scales.y) < INFINITY &&
    scales.y > bitcast<f32>(0x00800000u) &&
    lengthFloor < scales.y * MNEE_MAX_RELATIVE_LENGTH_FLOOR;
}

fn mneeSafeSignedDenominator(value: f32, magnitudeFloor: f32) -> f32 {
  let fallback = select(-magnitudeFloor, magnitudeFloor, value >= 0.0);
  return select(fallback, value, abs(value) > magnitudeFloor);
}

fn mnee_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < bitcast<f32>(0x00800000u)) { return vec3f(0.0); }
  return v / l;
}

`;

/** Maximum Newton iterations accepted by the bounded chain solver. */
export const MNEE_CHAIN_MAX_ITERS = 32;
/** Publicly supported upper bound for `causticOptions.mneeMaxChainLength`. */
export const MNEE_CHAIN_MAX_VERTICES = 8;

/** Fixed-capacity, coupled 1–8 vertex specular-manifold solver. */
export const MNEE_BOUNDED_CHAIN_WGSL = /* wgsl */ `
fn mneeMat2Invertible(m: mat2x2f) -> bool {
  let determinant = m[0][0] * m[1][1] - m[1][0] * m[0][1];
  let determinantFloor = max(
    length(m[0]) * length(m[1]) * (32.0 * MNEE_F32_EPSILON),
    bitcast<f32>(0x00800000u),
  );
  return abs(determinant) > determinantFloor &&
    abs(determinant) < INFINITY &&
    determinantFloor < INFINITY;
}

// 2×2 inverse (column-major mat2x2f: m[0]=(m00,m10), m[1]=(m01,m11)).
fn mnee_inv2x2(m: mat2x2f) -> mat2x2f {
  let det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
  let detFloor = max(
    length(m[0]) * length(m[1]) * (32.0 * MNEE_F32_EPSILON),
    bitcast<f32>(0x00800000u),
  );
  let inv = 1.0 / mneeSafeSignedDenominator(det, detFloor);
  return mat2x2f(vec2f(m[1][1], -m[0][1]) * inv, vec2f(-m[1][0], m[0][0]) * inv);
}

// Fixed-capacity N-vertex manifold chain used by the production 1..8 vertex
// estimator. Each residual depends only on its previous/current/next vertex, so
// the Newton Jacobian is block tridiagonal with 2x2 blocks. The block Thomas
// solve below is O(N), preserves all inter-vertex coupling, and avoids a fake
// dense 16x16 inverse. Facet planes and event eta pairs are supplied by the
// probability-known proposal in caustic.wgsl.ts.
struct MneeBoundedChainGeometry {
  count: u32,
  // 0 = finite point/area endpoint, 1 = directional endpoint.
  sourceMode: u32,
  // Direction from the first manifold vertex toward a directional source.
  sourceDirection: vec3f,
  planeP: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  normal: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  tangentU: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  tangentV: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
}

fn mneeBoundedChainScales(
  geometry: MneeBoundedChainGeometry,
  lightP: vec3f,
  recv: vec3f,
) -> vec2f {
  var coordinateScale = max(
    mneePointCoordinateScale(lightP), mneePointCoordinateScale(recv),
  );
  var localSpan = length(recv - lightP);
  var previous = lightP;
  for (var index = 0u; index < ${MNEE_CHAIN_MAX_VERTICES}u; index = index + 1u) {
    if (index >= geometry.count) { break; }
    let point = geometry.planeP[index];
    coordinateScale = max(coordinateScale, mneePointCoordinateScale(point));
    localSpan = max(
      localSpan,
      max(length(point - previous), length(recv - point)),
    );
    previous = point;
  }
  return vec2f(coordinateScale, localSpan);
}

struct MneeBoundedChainMedia {
  etaI: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>,
  etaT: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>,
}

struct MneeBoundedChainResult {
  vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  residual: f32,
  iters: u32,
  valid: u32,
}

fn mneeBoundedChainResidualAt(
  vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  vertexIndex: u32,
  lightP: vec3f,
  recv: vec3f,
) -> vec2f {
  let v = vertices[vertexIndex];
  var previous = lightP;
  if (vertexIndex > 0u) { previous = vertices[vertexIndex - 1u]; }
  var next = recv;
  if (vertexIndex + 1u < geometry.count) { next = vertices[vertexIndex + 1u]; }
  var wi = mnee_safe_normalize(previous - v);
  if (vertexIndex == 0u && geometry.sourceMode == 1u) {
    wi = mnee_safe_normalize(geometry.sourceDirection);
  }
  let wo = mnee_safe_normalize(next - v);
  let h = mnee_safe_normalize(
    media.etaI[vertexIndex] * wi + media.etaT[vertexIndex] * wo,
  );
  let n = geometry.normal[vertexIndex];
  let hTan = h - dot(h, n) * n;
  return vec2f(
    dot(hTan, geometry.tangentU[vertexIndex]),
    dot(hTan, geometry.tangentV[vertexIndex]),
  );
}

fn mneeBoundedChainResidualMax(
  vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  lightP: vec3f,
  recv: vec3f,
) -> f32 {
  var residualMax = 0.0;
  for (var i = 0u; i < ${MNEE_CHAIN_MAX_VERTICES}u; i = i + 1u) {
    if (i >= geometry.count) { break; }
    residualMax = max(
      residualMax,
      length(mneeBoundedChainResidualAt(vertices, geometry, media, i, lightP, recv)),
    );
  }
  return residualMax;
}

fn mneeNewtonSolveChainBounded(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  lightP: vec3f,
  recv: vec3f,
  seedVertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  maxIter: u32,
) -> MneeBoundedChainResult {
  var out: MneeBoundedChainResult;
  out.vertices = seedVertices;
  out.residual = 1e20;
  out.iters = 0u;
  out.valid = 0u;
  if (geometry.count == 0u || geometry.count > ${MNEE_CHAIN_MAX_VERTICES}u) {
    return out;
  }

  var coordA: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>;
  var coordB: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>;
  for (var initIndex = 0u; initIndex < ${MNEE_CHAIN_MAX_VERTICES}u; initIndex = initIndex + 1u) {
    if (initIndex >= geometry.count) { break; }
    let offset = seedVertices[initIndex] - geometry.planeP[initIndex];
    coordA[initIndex] = dot(offset, geometry.tangentU[initIndex]);
    coordB[initIndex] = dot(offset, geometry.tangentV[initIndex]);
  }

  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return out; }
  let eps = mneeFdStepFromScales(solverScales);
  let residualTolerance = mneeResidualToleranceFromScales(solverScales);
  for (var it = 0u; it < maxIter; it = it + 1u) {
    var vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>;
    for (var vi = 0u; vi < ${MNEE_CHAIN_MAX_VERTICES}u; vi = vi + 1u) {
      if (vi >= geometry.count) { break; }
      vertices[vi] = geometry.planeP[vi] +
        coordA[vi] * geometry.tangentU[vi] +
        coordB[vi] * geometry.tangentV[vi];
    }
    let residualMax = mneeBoundedChainResidualMax(
      vertices, geometry, media, lightP, recv,
    );
    out.vertices = vertices;
    out.residual = residualMax;
    out.iters = it;
    if (residualMax < residualTolerance) {
      out.valid = 1u;
      return out;
    }

    var residual: array<vec2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var lower: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var diagonal: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var upper: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    for (var ji = 0u; ji < ${MNEE_CHAIN_MAX_VERTICES}u; ji = ji + 1u) {
      if (ji >= geometry.count) { break; }
      let r0 = mneeBoundedChainResidualAt(vertices, geometry, media, ji, lightP, recv);
      residual[ji] = r0;

      var perturbed = vertices;
      perturbed[ji] = vertices[ji] + eps * geometry.tangentU[ji];
      let ownA = (
        mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
      ) / eps;
      perturbed[ji] = vertices[ji] + eps * geometry.tangentV[ji];
      let ownB = (
        mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
      ) / eps;
      diagonal[ji] = mat2x2f(ownA, ownB);

      lower[ji] = mat2x2f(vec2f(0.0), vec2f(0.0));
      if (ji > 0u) {
        perturbed = vertices;
        perturbed[ji - 1u] = vertices[ji - 1u] + eps * geometry.tangentU[ji - 1u];
        let prevA = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        perturbed[ji - 1u] = vertices[ji - 1u] + eps * geometry.tangentV[ji - 1u];
        let prevB = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        lower[ji] = mat2x2f(prevA, prevB);
      }

      upper[ji] = mat2x2f(vec2f(0.0), vec2f(0.0));
      if (ji + 1u < geometry.count) {
        perturbed = vertices;
        perturbed[ji + 1u] = vertices[ji + 1u] + eps * geometry.tangentU[ji + 1u];
        let nextA = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        perturbed[ji + 1u] = vertices[ji + 1u] + eps * geometry.tangentV[ji + 1u];
        let nextB = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        upper[ji] = mat2x2f(nextA, nextB);
      }
    }

    // Block Thomas elimination for J delta = -r.
    var cPrime: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var dPrime: array<vec2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var singular = false;
    for (var fi = 0u; fi < ${MNEE_CHAIN_MAX_VERTICES}u; fi = fi + 1u) {
      if (fi >= geometry.count) { break; }
      var denom = diagonal[fi];
      var rhs = -residual[fi];
      if (fi > 0u) {
        denom = denom - lower[fi] * cPrime[fi - 1u];
        rhs = rhs - lower[fi] * dPrime[fi - 1u];
      }
      if (!mneeMat2Invertible(denom)) {
        singular = true;
        break;
      }
      let invDenom = mnee_inv2x2(denom);
      if (fi + 1u < geometry.count) {
        cPrime[fi] = invDenom * upper[fi];
      } else {
        cPrime[fi] = mat2x2f(vec2f(0.0), vec2f(0.0));
      }
      dPrime[fi] = invDenom * rhs;
    }
    if (singular) { return out; }

    var delta: array<vec2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var reverseIndex = geometry.count;
    loop {
      if (reverseIndex == 0u) { break; }
      reverseIndex = reverseIndex - 1u;
      delta[reverseIndex] = dPrime[reverseIndex];
      if (reverseIndex + 1u < geometry.count) {
        delta[reverseIndex] = delta[reverseIndex] -
          cPrime[reverseIndex] * delta[reverseIndex + 1u];
      }
    }

    // Globalize the coupled solve with a residual-decreasing line search.
    var scale = 1.0;
    var accepted = false;
    for (var bt = 0u; bt < 10u; bt = bt + 1u) {
      var trialVertices = vertices;
      for (var ti = 0u; ti < ${MNEE_CHAIN_MAX_VERTICES}u; ti = ti + 1u) {
        if (ti >= geometry.count) { break; }
        let trialA = coordA[ti] + scale * delta[ti].x;
        let trialB = coordB[ti] + scale * delta[ti].y;
        trialVertices[ti] = geometry.planeP[ti] +
          trialA * geometry.tangentU[ti] +
          trialB * geometry.tangentV[ti];
      }
      let trialResidual = mneeBoundedChainResidualMax(
        trialVertices, geometry, media, lightP, recv,
      );
      if (trialResidual < residualMax) {
        for (var ai = 0u; ai < ${MNEE_CHAIN_MAX_VERTICES}u; ai = ai + 1u) {
          if (ai >= geometry.count) { break; }
          coordA[ai] = coordA[ai] + scale * delta[ai].x;
          coordB[ai] = coordB[ai] + scale * delta[ai].y;
        }
        accepted = true;
        break;
      }
      scale = scale * 0.5;
    }
    if (!accepted) { return out; }
  }

  var finalVertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>;
  for (var finalIndex = 0u; finalIndex < ${MNEE_CHAIN_MAX_VERTICES}u; finalIndex = finalIndex + 1u) {
    if (finalIndex >= geometry.count) { break; }
    finalVertices[finalIndex] = geometry.planeP[finalIndex] +
      coordA[finalIndex] * geometry.tangentU[finalIndex] +
      coordB[finalIndex] * geometry.tangentV[finalIndex];
  }
  out.vertices = finalVertices;
  out.residual = mneeBoundedChainResidualMax(
    finalVertices, geometry, media, lightP, recv,
  );
  out.iters = maxIter;
  if (out.residual <= residualTolerance) { out.valid = 1u; }
  return out;
}

fn mneeBoundedChainFocusingDet(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  solved: MneeBoundedChainResult,
  lightP: vec3f,
  recv: vec3f,
  recvTu: vec3f,
  recvTv: vec3f,
  maxIter: u32,
) -> f32 {
  if (solved.valid == 0u || geometry.count == 0u) { return 0.0; }
  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let eps = mneeFdStepFromScales(solverScales);
  let baseDirection = mnee_safe_normalize(solved.vertices[0] - lightP);
  let solveU = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTu * eps, solved.vertices, maxIter,
  );
  let solveV = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTv * eps, solved.vertices, maxIter,
  );
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let derivativeU = (
    mnee_safe_normalize(solveU.vertices[0] - lightP) - baseDirection
  ) / eps;
  let derivativeV = (
    mnee_safe_normalize(solveV.vertices[0] - lightP) - baseDirection
  ) / eps;
  let determinant = length(cross(derivativeU, derivativeV));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
}

// Directional-source irradiance transport. Perturb one square metre of receiver
// tangent area, re-solve the whole chain, and measure the projected source-side
// footprint at vertex zero. This Jacobian already contains receiver cosine.
fn mneeBoundedChainDirectionalFocusingDet(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  solved: MneeBoundedChainResult,
  lightP: vec3f,
  recv: vec3f,
  recvTu: vec3f,
  recvTv: vec3f,
  maxIter: u32,
) -> f32 {
  if (solved.valid == 0u || geometry.count == 0u || geometry.sourceMode != 1u) {
    return 0.0;
  }
  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let eps = mneeFdStepFromScales(solverScales);
  let solveU = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTu * eps, solved.vertices, maxIter,
  );
  let solveV = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTv * eps, solved.vertices, maxIter,
  );
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let derivativeU = (solveU.vertices[0] - solved.vertices[0]) / eps;
  let derivativeV = (solveV.vertices[0] - solved.vertices[0]) / eps;
  let determinant = abs(dot(
    cross(derivativeU, derivativeV),
    mnee_safe_normalize(geometry.sourceDirection),
  ));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
}

// Area endpoint change of variables: dA_light -> dOmega_receiver through the
// complete solved chain. The light axes are orthonormal physical metre axes, so
// the returned determinant is per unit emitter area.
fn mneeBoundedChainAreaPdfDet(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  solved: MneeBoundedChainResult,
  lightP: vec3f,
  recv: vec3f,
  lightU: vec3f,
  lightV: vec3f,
  maxIter: u32,
) -> f32 {
  if (solved.valid == 0u || geometry.count == 0u || geometry.sourceMode != 0u) {
    return 0.0;
  }
  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let eps = mneeFdStepFromScales(solverScales);
  let lightAreaScale = length(cross(lightU, lightV));
  if (!(lightAreaScale > mneeLengthFloorFromScales(solverScales) *
      mneeLengthFloorFromScales(solverScales)) ||
      !(lightAreaScale < INFINITY)) { return 0.0; }
  let tu = mnee_safe_normalize(lightU);
  let lightN = mnee_safe_normalize(cross(lightU, lightV));
  let tv = mnee_safe_normalize(cross(lightN, tu));
  let solveU = mneeNewtonSolveChainBounded(
    geometry, media, lightP + tu * eps, recv, solved.vertices, maxIter,
  );
  let solveV = mneeNewtonSolveChainBounded(
    geometry, media, lightP + tv * eps, recv, solved.vertices, maxIter,
  );
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let last = geometry.count - 1u;
  let baseDirection = mnee_safe_normalize(solved.vertices[last] - recv);
  let derivativeU = (
    mnee_safe_normalize(solveU.vertices[last] - recv) - baseDirection
  ) / eps;
  let derivativeV = (
    mnee_safe_normalize(solveV.vertices[last] - recv) - baseDirection
  ) / eps;
  let determinant = length(cross(derivativeU, derivativeV));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
}
`;
