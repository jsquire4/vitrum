/**
 * Intersection CORE — the byte-identical shared prefix of both the full-tier
 * (`intersection.wgsl.ts`) and lite-tier (`intersectionLite.wgsl.ts`)
 * intersection modules.
 *
 * Bundled here (verbatim, shared by both tiers):
 *  - AABB intersection (`intersectAabb`, `intersectAabbDetailed`)
 *  - `SceneHit` struct + `SHAPE_*` discriminants
 *  - Analytic-shape local-frame intersectors (`intersectSphereLocal`,
 *    `intersectCylinderLocal`, `intersectCapsuleLocal`, `intersectHChannelLocal`)
 *  - World↔local transform helpers (`transformPointCols`,
 *    `transformDirectionCols`, `transformNormalFromWorldToLocalCols`)
 *  - The mesh BVH traversal kernel (`traceMeshBvh`)
 *
 * The full tier appends `traceAnalyticShapes` + the TLAS wrappers; the lite
 * tier appends its mesh-only `traceClosest`/`traceAny`/`hitMaterialId`. Both
 * compositions remain byte-identical to the pre-extraction monolithic strings.
 *
 * No leading/trailing newline is added here: each tier interpolates this const
 * directly where the shared body used to be inlined.
 */
import { BVH_INTERSECT_STACK_DEPTH } from '@vitrum/shared-bvh';

export const PT_WEBGPU_INTERSECTION_CORE_WGSL = /* wgsl */ `fn intersectAabb(ray: Ray, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32) -> bool {
  let invDir = safeInvDir(ray.direction);
  let t1 = (bmin - ray.origin) * invDir;
  let t2 = (bmax - ray.origin) * invDir;
  let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  let tFar = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
  return !(tNear > tFar || tFar < tMin || tNear > tMax);
}

struct SceneHit {
  didHit: bool,
  dist: f32,
  // Exact represented-space hit carried through TLAS local-to-world transform.
  // Continuation starts here instead of reconstructing origin + direction*t,
  // whose independent rounding can overstep an adjacent representable surface.
  position: vec3f,
  triIndex: u32,
  normal: vec3f,
  // Geometric-winding orientation at the accepted hit. Unlike the interpolated
  // shading normal above, this is stable under authored vertex normals and is
  // parity-corrected when a BLAS hit crosses a mirrored TLAS transform.
  frontFace: bool,
  // Barycentric weights (v, w) of the hit on its triangle (u = 1 - v - w),
  // computed in BLAS-local space alongside the shading normal. The kernel
  // interpolates per-vertex UVs with these for texture sampling. Space-invariant
  // (no transform needed when propagated through the TLAS instance frame). Zero
  // for analytic-shape hits and non-shading (any-hit) traversals. (P2)
  baryVW: vec2f,
  // Actual TLAS instance for mesh hits, or INVALID_TLAS_INSTANCE_INDEX for the
  // merged-BLAS/lite/analytic paths. Full-tier normal maps use this to transform
  // their per-hit tangent basis into the same world frame as hit.normal.
  instanceIndex: u32,
  // Exact projected-edge classification from the watertight predicate that
  // accepted this triangle. Analytic hits use zero. Transmission continuation
  // uses this payload to exclude only the just-crossed represented feature.
  zeroEdgeMask: u32,
};

const SHAPE_SPHERE = 1u;
const SHAPE_BOX = 2u;
const SHAPE_CAPSULE = 3u;
const SHAPE_CYLINDER = 4u;
const SHAPE_H_CHANNEL_CAME = 5u;

fn transformPointCols(c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, p: vec3f) -> vec3f {
  let raw = c0 * p.x + c1 * p.y + c2 * p.z + c3;
  let scale = max(max(abs(raw.x), abs(raw.y)), max(abs(raw.z), abs(raw.w)));
  if (!(scale > 0.0) || scale > 3.402823e38) { return vec3f(0.0); }
  let r = raw / scale;
  if (r.w == 0.0) { return vec3f(0.0); }
  let point = r.xyz / r.w;
  if (!all(point == point) || any(abs(point) > vec3f(3.402823e38))) {
    return vec3f(0.0);
  }
  return point;
}

fn transformDirectionCols(c0: vec4f, c1: vec4f, c2: vec4f, d: vec3f) -> vec3f {
  return safe_normalize((c0 * d.x + c1 * d.y + c2 * d.z).xyz);
}

fn transformNormalFromWorldToLocalCols(w2l0: vec4f, w2l1: vec4f, w2l2: vec4f, nLocal: vec3f) -> vec3f {
  // Column-major local-to-world transforms require transpose(worldToLocal)
  // for normals. Each output component is a dot with one W2L column; using
  // rows would incorrectly apply W2L and only pass diagonal-scale tests.
  return safe_normalize(vec3f(
    dot(w2l0.xyz, nLocal),
    dot(w2l1.xyz, nLocal),
    dot(w2l2.xyz, nLocal),
  ));
}

fn transformLinearOrientationSign(c0: vec4f, c1: vec4f, c2: vec4f) -> f32 {
  let scale0 = max(abs(c0.x), max(abs(c0.y), abs(c0.z)));
  let scale1 = max(abs(c1.x), max(abs(c1.y), abs(c1.z)));
  let scale2 = max(abs(c2.x), max(abs(c2.y), abs(c2.z)));
  if (
    !(scale0 > 0.0) || scale0 > 3.402823e38 ||
    !(scale1 > 0.0) || scale1 > 3.402823e38 ||
    !(scale2 > 0.0) || scale2 > 3.402823e38
  ) {
    return 1.0;
  }
  let determinant = dot(
    cross(c0.xyz / scale0, c1.xyz / scale1),
    c2.xyz / scale2,
  );
  return select(-1.0, 1.0, determinant >= 0.0);
}

fn intersectAabbDetailed(ray: Ray, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32, nOut: ptr<function, vec3f>) -> f32 {
  let invDir = safeInvDir(ray.direction);
  let t0 = (bmin - ray.origin) * invDir;
  let t1 = (bmax - ray.origin) * invDir;
  let tsm = min(t0, t1);
  let tbg = max(t0, t1);
  let tNear = max(max(tsm.x, tsm.y), tsm.z);
  let tFar = min(min(tbg.x, tbg.y), tbg.z);
  if (tNear > tFar || tFar < tMin || tNear > tMax) {
    return INFINITY;
  }
  var tHit = tNear;
  var fromFar = false;
  if (tHit < tMin) {
    tHit = tFar;
    fromFar = true;
  }
  var n = vec3f(0.0);
  if (!fromFar) {
    if (tHit == tsm.x) {
      n = vec3f(select(1.0, -1.0, ray.direction.x > 0.0), 0.0, 0.0);
    } else if (tHit == tsm.y) {
      n = vec3f(0.0, select(1.0, -1.0, ray.direction.y > 0.0), 0.0);
    } else {
      n = vec3f(0.0, 0.0, select(1.0, -1.0, ray.direction.z > 0.0));
    }
  } else {
    if (tHit == tbg.x) {
      n = vec3f(select(-1.0, 1.0, ray.direction.x > 0.0), 0.0, 0.0);
    } else if (tHit == tbg.y) {
      n = vec3f(0.0, select(-1.0, 1.0, ray.direction.y > 0.0), 0.0);
    } else {
      n = vec3f(0.0, 0.0, select(-1.0, 1.0, ray.direction.z > 0.0));
    }
  }
  *nOut = n;
  return tHit;
}

fn intersectSphereLocal(ray: Ray, center: vec3f, radius: f32, nOut: ptr<function, vec3f>) -> f32 {
  let oc = ray.origin - center;
  let rayLengthSquared = dot(ray.direction, ray.direction);
  let spatialScale = max(
    max(abs(oc.x), max(abs(oc.y), abs(oc.z))),
    abs(radius),
  );
  if (
    !(radius > 0.0) ||
    !(rayLengthSquared > 0.0) || rayLengthSquared > 3.402823e38 ||
    !(spatialScale > 0.0) ||
    spatialScale > 3.402823e38
  ) {
    return INFINITY;
  }
  let scaledOc = oc / spatialScale;
  let scaledRadius = radius / spatialScale;
  let b = dot(scaledOc, ray.direction);
  let c = dot(scaledOc, scaledOc) - scaledRadius * scaledRadius;
  let disc = b * b - rayLengthSquared * c;
  if (disc < 0.0) { return INFINITY; }
  let s = sqrt(disc);
  let t0 = ((-b - s) / rayLengthSquared) * spatialScale;
  let t1 = ((-b + s) / rayLengthSquared) * spatialScale;
  var t = t0;
  if (!(t > 0.0)) { t = t1; }
  if (!(t > 0.0)) { return INFINITY; }
  let p = ray.origin + ray.direction * t;
  *nOut = safe_normalize(p - center);
  return t;
}

fn intersectCylinderLocal(ray: Ray, center: vec3f, radius: f32, halfHeight: f32, nOut: ptr<function, vec3f>) -> f32 {
  let ro = ray.origin - center;
  let spatialScale = max(
    max(abs(ro.x), max(abs(ro.y), abs(ro.z))),
    max(abs(radius), abs(halfHeight)),
  );
  if (
    !(radius > 0.0) || !(halfHeight > 0.0) ||
    !(spatialScale > 0.0) || spatialScale > 3.402823e38
  ) {
    return INFINITY;
  }
  let scaledRo = ro / spatialScale;
  let scaledRadius = radius / spatialScale;
  let scaledHalfHeight = halfHeight / spatialScale;
  let rd = ray.direction;
  let a = rd.x * rd.x + rd.z * rd.z;
  let b = 2.0 * (scaledRo.x * rd.x + scaledRo.z * rd.z);
  let c =
    scaledRo.x * scaledRo.x +
    scaledRo.z * scaledRo.z -
    scaledRadius * scaledRadius;
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  let disc = b * b - 4.0 * a * c;
  if (disc >= 0.0 && a > 0.0) {
    let s = sqrt(disc);
    let t0Scaled = (-b - s) / (2.0 * a);
    let t1Scaled = (-b + s) / (2.0 * a);
    let t0 = t0Scaled * spatialScale;
    let t1 = t1Scaled * spatialScale;
    if (t0 > 0.0) {
      let y = scaledRo.y + rd.y * t0Scaled;
      if (abs(y) <= scaledHalfHeight) {
        bestT = t0;
        bestN = safe_normalize(vec3f(
          scaledRo.x + rd.x * t0Scaled,
          0.0,
          scaledRo.z + rd.z * t0Scaled,
        ));
      }
    }
    if (t1 > 0.0 && t1 < bestT) {
      let y = scaledRo.y + rd.y * t1Scaled;
      if (abs(y) <= scaledHalfHeight) {
        bestT = t1;
        bestN = safe_normalize(vec3f(
          scaledRo.x + rd.x * t1Scaled,
          0.0,
          scaledRo.z + rd.z * t1Scaled,
        ));
      }
    }
  }
  if (rd.y != 0.0) {
    let topTScaled = (scaledHalfHeight - scaledRo.y) / rd.y;
    let topT = topTScaled * spatialScale;
    if (topT > 0.0 && topT < bestT) {
      let p = scaledRo + rd * topTScaled;
      if (p.x * p.x + p.z * p.z <= scaledRadius * scaledRadius) {
        bestT = topT;
        bestN = vec3f(0.0, 1.0, 0.0);
      }
    }
    let bottomTScaled = (-scaledHalfHeight - scaledRo.y) / rd.y;
    let bottomT = bottomTScaled * spatialScale;
    if (bottomT > 0.0 && bottomT < bestT) {
      let p = scaledRo + rd * bottomTScaled;
      if (p.x * p.x + p.z * p.z <= scaledRadius * scaledRadius) {
        bestT = bottomT;
        bestN = vec3f(0.0, -1.0, 0.0);
      }
    }
  }
  *nOut = bestN;
  return bestT;
}

fn intersectCapsuleLocal(ray: Ray, pa: vec3f, pb: vec3f, radius: f32, nOut: ptr<function, vec3f>) -> f32 {
  let rawBa = pb - pa;
  let rawOa = ray.origin - pa;
  let spatialScale = max(
    max(
      max(abs(rawBa.x), max(abs(rawBa.y), abs(rawBa.z))),
      max(abs(rawOa.x), max(abs(rawOa.y), abs(rawOa.z))),
    ),
    abs(radius),
  );
  if (
    !(radius > 0.0) ||
    !(spatialScale > 0.0) || spatialScale > 3.402823e38
  ) {
    return INFINITY;
  }
  let ba = rawBa / spatialScale;
  let oa = rawOa / spatialScale;
  let scaledRadius = radius / spatialScale;
  let baba = dot(ba, ba);
  if (!(baba > 0.0)) {
    return intersectSphereLocal(ray, pa, radius, nOut);
  }
  let bard = dot(ba, ray.direction);
  let baoa = dot(ba, oa);
  let rdrd = dot(ray.direction, ray.direction);
  if (!(rdrd > 0.0) || rdrd > 3.402823e38) {
    return INFINITY;
  }
  let rdoa = dot(ray.direction, oa);
  let oaoa = dot(oa, oa);
  let a = baba * rdrd - bard * bard;
  let b = baba * rdoa - baoa * bard;
  let c = baba * oaoa - baoa * baoa - scaledRadius * scaledRadius * baba;
  let h = b * b - a * c;
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  if (h >= 0.0 && a > 0.0) {
    let sqrtH = sqrt(h);
    let tBodyNearScaled = (-b - sqrtH) / a;
    let tBodyNear = tBodyNearScaled * spatialScale;
    let yBodyNear = baoa + tBodyNearScaled * bard;
    if (tBodyNear > 0.0 && yBodyNear > 0.0 && yBodyNear < baba) {
      let p = oa + ray.direction * tBodyNearScaled - ba * (yBodyNear / baba);
      bestT = tBodyNear;
      bestN = safe_normalize(p);
    }
    let tBodyFarScaled = (-b + sqrtH) / a;
    let tBodyFar = tBodyFarScaled * spatialScale;
    let yBodyFar = baoa + tBodyFarScaled * bard;
    if (tBodyFar > 0.0 && tBodyFar < bestT && yBodyFar > 0.0 && yBodyFar < baba) {
      let p = oa + ray.direction * tBodyFarScaled - ba * (yBodyFar / baba);
      bestT = tBodyFar;
      bestN = safe_normalize(p);
    }
  }
  let ocA = oa;
  let bA = dot(ocA, ray.direction);
  let cA = dot(ocA, ocA) - scaledRadius * scaledRadius;
  let hA = bA * bA - rdrd * cA;
  if (hA >= 0.0) {
    let sqrtHA = sqrt(hA);
    let tANearScaled = (-bA - sqrtHA) / rdrd;
    let tANear = tANearScaled * spatialScale;
    let capPointA = ocA + ray.direction * tANearScaled;
    if (tANear > 0.0 && tANear < bestT && dot(capPointA, ba) <= 0.0) {
      bestT = tANear;
      bestN = safe_normalize(capPointA);
    }
    let tAFarScaled = (-bA + sqrtHA) / rdrd;
    let tAFar = tAFarScaled * spatialScale;
    let capPointAFar = ocA + ray.direction * tAFarScaled;
    if (tAFar > 0.0 && tAFar < bestT && dot(capPointAFar, ba) <= 0.0) {
      bestT = tAFar;
      bestN = safe_normalize(capPointAFar);
    }
  }
  let ocB = oa - ba;
  let bB = dot(ocB, ray.direction);
  let cB = dot(ocB, ocB) - scaledRadius * scaledRadius;
  let hB = bB * bB - rdrd * cB;
  if (hB >= 0.0) {
    let sqrtHB = sqrt(hB);
    let tBNearScaled = (-bB - sqrtHB) / rdrd;
    let tBNear = tBNearScaled * spatialScale;
    let capPointB = ocB + ray.direction * tBNearScaled;
    if (tBNear > 0.0 && tBNear < bestT && dot(capPointB, ba) >= 0.0) {
      bestT = tBNear;
      bestN = safe_normalize(capPointB);
    }
    let tBFarScaled = (-bB + sqrtHB) / rdrd;
    let tBFar = tBFarScaled * spatialScale;
    let capPointBFar = ocB + ray.direction * tBFarScaled;
    if (tBFar > 0.0 && tBFar < bestT && dot(capPointBFar, ba) >= 0.0) {
      bestT = tBFar;
      bestN = safe_normalize(capPointBFar);
    }
  }
  *nOut = bestN;
  return bestT;
}

fn hChannelContainsLocal(p: vec3f, hx: f32, hy: f32, hz: f32, t: f32) -> bool {
  let inTop = all(p >= vec3f(-hx, hy - t, -hz)) &&
    all(p <= vec3f(hx, hy, hz));
  let inBottom = all(p >= vec3f(-hx, -hy, -hz)) &&
    all(p <= vec3f(hx, -hy + t, hz));
  let inWeb = all(p >= vec3f(-hx, -hy + t, -t)) &&
    all(p <= vec3f(hx, hy - t, t));
  return inTop || inBottom || inWeb;
}

fn intersectHChannelLocal(ray: Ray, lengthX: f32, railWidth: f32, blockHeight: f32, webThickness: f32, nOut: ptr<function, vec3f>) -> f32 {
  if (
    !(lengthX > 0.0) || !(railWidth > 0.0) ||
    !(blockHeight > 0.0) || !(webThickness > 0.0) ||
    webThickness >= min(railWidth, blockHeight)
  ) {
    return INFINITY;
  }
  let hx = lengthX * 0.5;
  let hy = blockHeight * 0.5;
  let hz = railWidth * 0.5;
  let t = webThickness * 0.5;
  let featureScale = min(min(hx, hy), min(hz, t));
  let boundaryProbe = featureScale * 1e-5;
  var advancedT = 0.0;
  var scanRay = ray;
  for (var boundary = 0u; boundary < 6u; boundary = boundary + 1u) {
    var bestT = INFINITY;
    var bestN = vec3f(0.0, 1.0, 0.0);
    var candidateN: vec3f;
    let railTop = intersectAabbDetailed(
      scanRay, vec3f(-hx, hy - t, -hz), vec3f(hx, hy, hz),
      0.0, INFINITY, &candidateN,
    );
    if (railTop < bestT) {
      bestT = railTop;
      bestN = candidateN;
    }
    let railBottom = intersectAabbDetailed(
      scanRay, vec3f(-hx, -hy, -hz), vec3f(hx, -hy + t, hz),
      0.0, INFINITY, &candidateN,
    );
    if (railBottom < bestT) {
      bestT = railBottom;
      bestN = candidateN;
    }
    let web = intersectAabbDetailed(
      scanRay, vec3f(-hx, -hy + t, -t), vec3f(hx, hy - t, t),
      0.0, INFINITY, &candidateN,
    );
    if (web < bestT) {
      bestT = web;
      bestN = candidateN;
    }
    if (bestT >= INFINITY) {
      return INFINITY;
    }
    let absoluteT = advancedT + bestT;
    let beforePoint = ray.origin + ray.direction * max(0.0, absoluteT - boundaryProbe);
    let afterPoint = ray.origin + ray.direction * (absoluteT + boundaryProbe);
    let insideBefore = hChannelContainsLocal(beforePoint, hx, hy, hz, t);
    let insideAfter = hChannelContainsLocal(afterPoint, hx, hy, hz, t);
    if (insideBefore != insideAfter) {
      *nOut = bestN;
      return absoluteT;
    }
    advancedT = absoluteT + 2.0 * boundaryProbe;
    scanRay.origin = ray.origin + ray.direction * advancedT;
  }
  return INFINITY;
}

// SHADOW-01 — primitive castShadow. Material vec4 #25 .w (the former pad lane)
// carries 1.0 when the SOURCE PRIMITIVE set castShadow:false (0.0 default —
// see scene/materialPacking.ts vec4 #25). Any-hit (occlusion) traversals skip
// such triangles so NEE shadow rays / visibility connections pass through;
// closest-hit (camera / radiance) traversals never call this, keeping the
// geometry camera-visible. Both tiers compose the material module that
// declares \`materials\` / \`triMaterialIds\` / MATERIAL_VEC4_STRIDE before this
// module, so the symbols resolve in every composition.
fn materialShadowCastDisabled(matId: u32) -> bool {
  let materialBase = materialRecordBase(matId);
  let vecIndex = materialRecordIndex(materialBase, 25u);
  if (vecIndex == MATERIAL_INVALID_INDEX) { return false; }
  let disabled = materials[vecIndex].w;
  return materialRecordFiniteF32(disabled) && disabled > 0.5;
}

fn triShadowCastDisabled(triIdx: u32) -> bool {
  if (triIdx >= arrayLength(&triMaterialIds)) { return false; }
  return materialShadowCastDisabled(triMaterialIds[triIdx].x);
}

// Mesh BVH traversal — closest: shrinking ray interval (hit.dist) for slab tests
// and full SceneHit on triangles; false uses fixed tMaxBound and returns
// true on first triangle hit in (tMin, tMaxBound).
fn traceMeshBvh(
  ray: Ray,
  tMin: f32,
  tMaxBound: f32,
  closest: bool,
  hit: ptr<function, SceneHit>,
  rootNode: u32,
  captureShadingDetails: bool,
) -> bool {
  (*hit).didHit = false;
  (*hit).dist = tMaxBound;
  (*hit).position = vec3f(0.0);
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).frontFace = false;
  (*hit).baryVW = vec2f(0.0);
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  (*hit).zeroEdgeMask = 0u;
  if (params.bvhNodeCount == 0u || arrayLength(&bvhNodes) == 0u || rootNode >= min(params.bvhNodeCount, arrayLength(&bvhNodes))) {
    return false;
  }

  var stack: array<u32, ${BVH_INTERSECT_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = rootNode;
  stackPtr = stackPtr + 1u;

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(params.bvhNodeCount, arrayLength(&bvhNodes))) {
      continue;
    }
    let node = bvhNodes[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    let farBound = select(tMaxBound, (*hit).dist, closest);
    if (!intersectAabb(ray, bmin, bmax, tMin, farBound)) {
      continue;
    }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let t = start + i;
        if (t >= min(params.triangleCount, arrayLength(&indices))) {
          continue;
        }
        let tri = indices[t];
        if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
          continue;
        }
        let a = positions[tri.x].xyz;
        let b = positions[tri.y].xyz;
        let c = positions[tri.z].xyz;
        let triHit = opticalWatertightTriangleIntersect(
          ray.origin,
          ray.direction,
          a,
          b,
          c,
          tMin,
        );
        let hitT = triHit.t;
        // LIVE upper bound — in closest mode compare against the RUNNING
        // nearest (*hit).dist (initialized to tMaxBound above), re-read every
        // iteration. A per-leaf snapshot here made the accept last-writer-wins
        // WITHIN a leaf: a farther triangle tested later overwrote a nearer
        // accepted hit, so thin slabs (walls) shaded from their buried far
        // face — NEE always occluded + bounce rays trapped → black geometry
        // (G-P0.3 capture found this via the face-on Cornell back wall).
        if (hitT > tMin && hitT < select(tMaxBound, (*hit).dist, closest)) {
          if (opticalTraversalSuppressesTriangle(t, tri, a, b, c)) {
            continue;
          }
          if (!closest) {
            // SHADOW-01 — any-hit mode is exclusively occlusion (shadow /
            // visibility) queries: skip castShadow:false geometry.
            if (triShadowCastDisabled(t)) {
              continue;
            }
            return true;
          }
          var shadeNormal = triHit.normal * triHit.side;
          let frontFace = triHit.side > 0.0;
          // Carry canonical (v,w) weights into texture and smooth-normal
          // interpolation. Re-solving this payload from the hit point loses
          // precision independently of the intersection that accepted it.
          let shadeBaryVW = vec2f(triHit.bary.y, triHit.bary.z);
          if (captureShadingDetails) {
            if (tri.x < arrayLength(&normals) && tri.y < arrayLength(&normals) && tri.z < arrayLength(&normals)) {
              let na = normals[tri.x].xyz;
              let nb = normals[tri.y].xyz;
              let nc = normals[tri.z].xyz;
              shadeNormal = safe_normalize(
                na * triHit.bary.x +
                nb * triHit.bary.y +
                nc * triHit.bary.z
              );
            }
          }
          (*hit).didHit = true;
          (*hit).dist = hitT;
          (*hit).position = opticalCanonicalHitPoint(triHit, a, b, c);
          (*hit).triIndex = t;
          (*hit).normal = shadeNormal;
          (*hit).frontFace = frontFace;
          (*hit).baryVW = shadeBaryVW;
          (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
          (*hit).zeroEdgeMask = triHit.zeroEdgeMask;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      // rightChildOrTriOffset is a RELATIVE offset (node units) from the current
      // node index; left child is always nodeIdx + 1. This matches the canonical
      // relative-offset encoding used by shared-bvh/normalizeBvhInteriorOffsets
      // and walkaround-hybrid/common.wgsl. Invariant: 1 <= offset < totalNodes.
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u <= ${BVH_INTERSECT_STACK_DEPTH}u) {
        stack[stackPtr] = rightChild;
        stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild;
        stackPtr = stackPtr + 1u;
      } else {
        // Canonical builds are depth-capped below this budget. Corrupt or
        // externally supplied layouts fail closed for occlusion rather than
        // silently leaking light; closest-hit preserves the best complete hit.
        return select(true, (*hit).didHit, closest);
      }
    }
  }
  return select(false, (*hit).didHit, closest);
}

// Optical-medium boundary identity. The payload is encoded as id + 1 in the
// triangle index .w lane so zero remains an unambiguous non-boundary sentinel.
// Full-tier TLAS identities include the represented instance; lite geometry is
// already expanded into one merged world-space triangle stream.
const MEDIUM_BOUNDARY_KIND_TLAS: u32 = 0u;
const MEDIUM_BOUNDARY_KIND_MERGED: u32 = 1u;
const MEDIUM_BOUNDARY_KIND_INVALID: u32 = 0xffffffffu;
const OPTICAL_MEDIUM_STACK_LIMIT: u32 = 8u;

fn mediumBoundaryIdentity(triIndex: u32, instanceIndex: u32) -> vec3u {
  if (triIndex >= min(params.triangleCount, arrayLength(&indices))) {
    return vec3u(MEDIUM_BOUNDARY_KIND_INVALID);
  }
  let encodedBoundary = opticalEncodedBoundaryId(triIndex, instanceIndex);
  if (encodedBoundary == 0u) {
    return vec3u(MEDIUM_BOUNDARY_KIND_INVALID);
  }
  if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX) {
    return vec3u(MEDIUM_BOUNDARY_KIND_TLAS, instanceIndex, encodedBoundary);
  }
  return vec3u(MEDIUM_BOUNDARY_KIND_MERGED, 0u, encodedBoundary);
}

fn mediumBoundaryIsValid(boundary: vec3u) -> bool {
  return boundary.x != MEDIUM_BOUNDARY_KIND_INVALID && boundary.z != 0u;
}

fn mediumBoundaryMatches(
  kind: u32,
  index: u32,
  component: u32,
  boundary: vec3u,
) -> bool {
  return mediumBoundaryIsValid(boundary) &&
    kind == boundary.x && index == boundary.y && component == boundary.z;
}

struct OpticalLocalBoundaryHit {
  didHit: bool,
  valid: bool,
  ambiguous: bool,
  tangent: bool,
  dist: f32,
  triIndex: u32,
  baryVW: vec2f,
  frontFace: bool,
  frontFaceCount: u32,
  backFaceCount: u32,
};

fn opticalResetLocalBoundaryHit(
  hit: ptr<function, OpticalLocalBoundaryHit>,
  tMax: f32,
) {
  (*hit).didHit = false;
  (*hit).valid = true;
  (*hit).ambiguous = false;
  (*hit).tangent = false;
  (*hit).dist = tMax;
  (*hit).triIndex = 0u;
  (*hit).baryVW = vec2f(0.0);
  (*hit).frontFace = false;
  (*hit).frontFaceCount = 0u;
  (*hit).backFaceCount = 0u;
}

// Traverse one represented BLAS with the canonical watertight
// triangle predicate. Only positively encoded authored bulk boundaries
// participate; opaque geometry and transmissive sheets are intentionally
// invisible to containment reconstruction.
fn traceOpticalMeshBvhLocal(
  ray: Ray,
  exclusiveMinT: f32,
  tMax: f32,
  rootNode: u32,
  hit: ptr<function, OpticalLocalBoundaryHit>,
) {
  opticalResetLocalBoundaryHit(hit, tMax);
  let nodeLimit = min(params.bvhNodeCount, arrayLength(&bvhNodes));
  if (
    params.bvhNodeCount == 0u || arrayLength(&bvhNodes) == 0u ||
    rootNode >= nodeLimit
  ) {
    (*hit).valid = false;
    return;
  }

  var stack: array<u32, ${BVH_INTERSECT_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = rootNode;
  stackPtr = stackPtr + 1u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= nodeLimit) {
      (*hit).valid = false;
      return;
    }
    let node = bvhNodes[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!intersectAabb(ray, bmin, bmax, exclusiveMinT, (*hit).dist)) {
      continue;
    }
    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let triIndex = start + i;
        if (triIndex >= min(params.triangleCount, arrayLength(&indices))) {
          (*hit).valid = false;
          return;
        }
        let tri = indices[triIndex];
        if (tri.w == 0u) { continue; }
        if (
          triIndex >= arrayLength(&triMaterialIds) ||
          !decodeMaterial(triMaterialIds[triIndex].x).isBulkMedium
        ) {
          // A stamped boundary whose live material is no longer bulk means the
          // host published geometry/material state non-atomically. Fail dark.
          (*hit).valid = false;
          return;
        }
        if (
          tri.x >= arrayLength(&positions) ||
          tri.y >= arrayLength(&positions) ||
          tri.z >= arrayLength(&positions)
        ) {
          (*hit).valid = false;
          return;
        }
        let candidate = opticalWatertightTriangleIntersect(
          ray.origin,
          ray.direction,
          positions[tri.x].xyz,
          positions[tri.y].xyz,
          positions[tri.z].xyz,
          exclusiveMinT,
        );
        if (!candidate.hit || !(candidate.t < tMax)) { continue; }
        if (!(*hit).didHit || candidate.t < (*hit).dist) {
          (*hit).didHit = true;
          (*hit).ambiguous = false;
          (*hit).tangent = false;
          (*hit).dist = candidate.t;
          (*hit).triIndex = triIndex;
          (*hit).baryVW = candidate.bary.yz;
          (*hit).frontFace = candidate.side > 0.0;
          (*hit).frontFaceCount = select(0u, 1u, candidate.side > 0.0);
          (*hit).backFaceCount = select(0u, 1u, candidate.side < 0.0);
        } else if (
          bitcast<u32>(candidate.t) == bitcast<u32>((*hit).dist)
        ) {
          let sameBoundary =
            tri.w == indices[(*hit).triIndex].w &&
            triMaterialIds[triIndex].x == triMaterialIds[(*hit).triIndex].x;
          if (!sameBoundary) {
            (*hit).ambiguous = true;
          } else {
            (*hit).frontFaceCount = (*hit).frontFaceCount +
              select(0u, 1u, candidate.side > 0.0);
            (*hit).backFaceCount = (*hit).backFaceCount +
              select(0u, 1u, candidate.side < 0.0);
          }
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (
        leftChild >= nodeLimit || rightChild >= nodeLimit ||
        stackPtr + 2u > ${BVH_INTERSECT_STACK_DEPTH}u
      ) {
        (*hit).valid = false;
        return;
      }
      stack[stackPtr] = rightChild;
      stackPtr = stackPtr + 1u;
      stack[stackPtr] = leftChild;
      stackPtr = stackPtr + 1u;
    }
  }
  if ((*hit).didHit && !(*hit).ambiguous) {
    let hasFront = (*hit).frontFaceCount > 0u;
    let hasBack = (*hit).backFaceCount > 0u;
    if (hasFront && hasBack) {
      if ((*hit).frontFaceCount == (*hit).backFaceCount) {
        // A balanced mixed-sign fan is a tangent touch: it has no winding
        // transition and therefore makes no medium-stack change.
        (*hit).tangent = true;
      } else {
        // An unbalanced mixed-sign exact-t event cannot be reduced to one
        // canonical crossing without depending on triangle enumeration order.
        (*hit).ambiguous = true;
      }
    } else {
      // Any number of duplicate/coincident faces with one uniform sign is one
      // represented boundary crossing, not repeated pushes or pops.
      (*hit).frontFace = hasFront;
    }
  }
}

struct OpticalBoundaryHit {
  didHit: bool,
  valid: bool,
  ambiguous: bool,
  tangent: bool,
  dist: f32,
  triIndex: u32,
  instanceIndex: u32,
  baryVW: vec2f,
  matId: u32,
  boundary: vec3u,
  frontFace: bool,
};

fn opticalResetBoundaryHit(
  hit: ptr<function, OpticalBoundaryHit>,
  tMax: f32,
) {
  (*hit).didHit = false;
  (*hit).valid = true;
  (*hit).ambiguous = false;
  (*hit).tangent = false;
  (*hit).dist = tMax;
  (*hit).triIndex = 0u;
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  (*hit).baryVW = vec2f(0.0);
  (*hit).matId = 0u;
  (*hit).boundary = vec3u(MEDIUM_BOUNDARY_KIND_INVALID);
  (*hit).frontFace = false;
}

fn opticalBoundaryHitSameEvent(
  aBoundary: vec3u,
  aFrontFace: bool,
  bBoundary: vec3u,
  bFrontFace: bool,
) -> bool {
  return all(aBoundary == bBoundary) && aFrontFace == bFrontFace;
}

struct OpticalContainment {
  valid: bool,
  depth: u32,
  matIds: array<u32, OPTICAL_MEDIUM_STACK_LIMIT>,
  boundaries: array<vec3u, OPTICAL_MEDIUM_STACK_LIMIT>,
  triIndices: array<u32, OPTICAL_MEDIUM_STACK_LIMIT>,
  instanceIndices: array<u32, OPTICAL_MEDIUM_STACK_LIMIT>,
  baryVWs: array<vec2f, OPTICAL_MEDIUM_STACK_LIMIT>,
};

fn opticalHasRepresentedBoundaries() -> bool {
  let triangleLimit = min(params.triangleCount, arrayLength(&indices));
  for (var triIndex = 0u; triIndex < triangleLimit; triIndex = triIndex + 1u) {
    if (indices[triIndex].w != 0u) { return true; }
  }
  return false;
}

// Reconstruct the exact ordered medium state on the open ray immediately after
// a launch point. Starting from the fixed launch origin avoids a fabricated
// outside anchor and defines the intended side when the origin lies exactly on
// a boundary. Front/back pairs encountered outward cancel through a temporary
// LIFO. Unmatched back faces identify origin-containing components in
// inner-to-outer order; reversing them yields the live outer-to-inner stack.
// There is deliberately no crossing-count ceiling: only live nesting is
// bounded, while any number of disjoint closed components may be crossed.
fn opticalContainmentAlongRay(
  origin: vec3f,
  directionRaw: vec3f,
) -> OpticalContainment {
  var result: OpticalContainment;
  result.valid = true;
  result.depth = 0u;
  for (var i = 0u; i < OPTICAL_MEDIUM_STACK_LIMIT; i = i + 1u) {
    result.matIds[i] = 0u;
    result.boundaries[i] = vec3u(MEDIUM_BOUNDARY_KIND_INVALID);
    result.triIndices[i] = 0u;
    result.instanceIndices[i] = INVALID_TLAS_INSTANCE_INDEX;
    result.baryVWs[i] = vec2f(0.0);
  }
  if (!opticalHasRepresentedBoundaries()) { return result; }
  let directionScale = max(
    abs(directionRaw.x), max(abs(directionRaw.y), abs(directionRaw.z)),
  );
  if (
    !all(origin == origin) ||
    any(abs(origin) > vec3f(3.402823e38)) ||
    !(directionScale > 0.0) || directionScale > 3.402823e38
  ) {
    result.valid = false;
    return result;
  }
  let ray = Ray(origin, safe_normalize(directionRaw));
  var enteredDepth = 0u;
  var enteredMatIds: array<u32, OPTICAL_MEDIUM_STACK_LIMIT>;
  var enteredBoundaries: array<vec3u, OPTICAL_MEDIUM_STACK_LIMIT>;
  var containedDepth = 0u;
  var containedMatIds: array<u32, OPTICAL_MEDIUM_STACK_LIMIT>;
  var containedBoundaries: array<vec3u, OPTICAL_MEDIUM_STACK_LIMIT>;
  var containedTriIndices: array<u32, OPTICAL_MEDIUM_STACK_LIMIT>;
  var containedInstanceIndices: array<u32, OPTICAL_MEDIUM_STACK_LIMIT>;
  var containedBaryVWs: array<vec2f, OPTICAL_MEDIUM_STACK_LIMIT>;
  var cursor = 0.0;
  loop {
    let event = traceOpticalBoundaryClosest(ray, cursor, INFINITY);
    if (!event.valid || event.ambiguous) {
      result.valid = false;
      return result;
    }
    if (!event.didHit) {
      if (enteredDepth != 0u) {
        result.valid = false;
        return result;
      }
      result.depth = containedDepth;
      for (var i = 0u; i < containedDepth; i = i + 1u) {
        let reverseIndex = containedDepth - 1u - i;
        result.matIds[i] = containedMatIds[reverseIndex];
        result.boundaries[i] = containedBoundaries[reverseIndex];
        result.triIndices[i] = containedTriIndices[reverseIndex];
        result.instanceIndices[i] = containedInstanceIndices[reverseIndex];
        result.baryVWs[i] = containedBaryVWs[reverseIndex];
      }
      return result;
    }
    if (!(event.dist > cursor)) {
      result.valid = false;
      return result;
    }
    if (event.tangent) {
      cursor = event.dist;
      continue;
    }
    if (event.frontFace) {
      if (enteredDepth >= OPTICAL_MEDIUM_STACK_LIMIT) {
        result.valid = false;
        return result;
      }
      enteredMatIds[enteredDepth] = event.matId;
      enteredBoundaries[enteredDepth] = event.boundary;
      enteredDepth = enteredDepth + 1u;
    } else {
      if (enteredDepth > 0u) {
        if (
          !all(enteredBoundaries[enteredDepth - 1u] == event.boundary) ||
          enteredMatIds[enteredDepth - 1u] != event.matId
        ) {
          result.valid = false;
          return result;
        }
        enteredDepth = enteredDepth - 1u;
      } else {
        if (containedDepth >= OPTICAL_MEDIUM_STACK_LIMIT) {
          result.valid = false;
          return result;
        }
        containedMatIds[containedDepth] = event.matId;
        containedBoundaries[containedDepth] = event.boundary;
        containedTriIndices[containedDepth] = event.triIndex;
        containedInstanceIndices[containedDepth] = event.instanceIndex;
        containedBaryVWs[containedDepth] = event.baryVW;
        containedDepth = containedDepth + 1u;
      }
    }
    cursor = event.dist;
  }
  return result;
}
`;
