/**
 * Intersection module — analytic-shape intersectors + BVH traversal.
 *
 * Bundled here:
 *  - `SceneHit` struct + `SHAPE_*` discriminants
 *  - AABB intersection (`intersectAabb`, `intersectAabbDetailed`)
 *  - Analytic-shape local-frame intersectors:
 *      - `intersectSphereLocal`
 *      - `intersectCylinderLocal`
 *      - `intersectCapsuleLocal`
 *      - `intersectHChannelLocal` (window-came H-channel)
 *  - World↔local transform helpers (`transformPointCols`,
 *    `transformDirectionCols`, `transformNormalFromWorldToLocalCols`)
 *  - BVH traversal wrappers (`traceMeshBvh`, `traceAnalyticShapes`,
 *    `traceClosest`, `traceAny`)
 *  - `hitMaterialId` — bridges SceneHit → material payload index for the
 *    main kernel and caustic dispatch.
 *
 * Depends on the FrameParams bindings and material constants declared in
 * `material.wgsl.ts`; the Möller-Trumbore `intersectTriangle` lives in
 * `common.wgsl.ts` and is referenced through the shared global scope.
 */
export const PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL = /* wgsl */ `
fn intersectAabb(ray: Ray, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32) -> bool {
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
  triIndex: u32,
  normal: vec3f,
};

const SHAPE_SPHERE = 1u;
const SHAPE_BOX = 2u;
const SHAPE_CAPSULE = 3u;
const SHAPE_CYLINDER = 4u;
const SHAPE_H_CHANNEL_CAME = 5u;

fn transformPointCols(c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, p: vec3f) -> vec3f {
  let r = c0 * p.x + c1 * p.y + c2 * p.z + c3;
  return r.xyz / max(abs(r.w), 1e-8);
}

fn transformDirectionCols(c0: vec4f, c1: vec4f, c2: vec4f, d: vec3f) -> vec3f {
  return safe_normalize((c0 * d.x + c1 * d.y + c2 * d.z).xyz);
}

fn transformNormalFromWorldToLocalCols(w2l0: vec4f, w2l1: vec4f, w2l2: vec4f, nLocal: vec3f) -> vec3f {
  return safe_normalize(vec3f(
    dot(vec3f(w2l0.x, w2l1.x, w2l2.x), nLocal),
    dot(vec3f(w2l0.y, w2l1.y, w2l2.y), nLocal),
    dot(vec3f(w2l0.z, w2l1.z, w2l2.z), nLocal),
  ));
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
  let eps = 1e-4;
  if (!fromFar) {
    if (abs(tHit - tsm.x) < eps) {
      n = vec3f(select(1.0, -1.0, ray.direction.x > 0.0), 0.0, 0.0);
    } else if (abs(tHit - tsm.y) < eps) {
      n = vec3f(0.0, select(1.0, -1.0, ray.direction.y > 0.0), 0.0);
    } else {
      n = vec3f(0.0, 0.0, select(1.0, -1.0, ray.direction.z > 0.0));
    }
  } else {
    if (abs(tHit - tbg.x) < eps) {
      n = vec3f(select(-1.0, 1.0, ray.direction.x > 0.0), 0.0, 0.0);
    } else if (abs(tHit - tbg.y) < eps) {
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
  let a = dot(ray.direction, ray.direction);
  let b = 2.0 * dot(oc, ray.direction);
  let c = dot(oc, oc) - radius * radius;
  let disc = b * b - 4.0 * a * c;
  if (disc < 0.0) { return INFINITY; }
  let s = sqrt(disc);
  let t0 = (-b - s) / (2.0 * a);
  let t1 = (-b + s) / (2.0 * a);
  var t = t0;
  if (t < 1e-5) { t = t1; }
  if (t < 1e-5) { return INFINITY; }
  let p = ray.origin + ray.direction * t;
  *nOut = safe_normalize(p - center);
  return t;
}

fn intersectCylinderLocal(ray: Ray, center: vec3f, radius: f32, halfHeight: f32, nOut: ptr<function, vec3f>) -> f32 {
  let ro = ray.origin - center;
  let rd = ray.direction;
  let a = rd.x * rd.x + rd.z * rd.z;
  let b = 2.0 * (ro.x * rd.x + ro.z * rd.z);
  let c = ro.x * ro.x + ro.z * ro.z - radius * radius;
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  let disc = b * b - 4.0 * a * c;
  if (disc >= 0.0 && abs(a) > 1e-8) {
    let s = sqrt(disc);
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);
    if (t0 > 1e-5) {
      let y = ro.y + rd.y * t0;
      if (abs(y) <= halfHeight) {
        bestT = t0;
        bestN = safe_normalize(vec3f(ro.x + rd.x * t0, 0.0, ro.z + rd.z * t0));
      }
    }
    if (t1 > 1e-5 && t1 < bestT) {
      let y = ro.y + rd.y * t1;
      if (abs(y) <= halfHeight) {
        bestT = t1;
        bestN = safe_normalize(vec3f(ro.x + rd.x * t1, 0.0, ro.z + rd.z * t1));
      }
    }
  }
  if (abs(rd.y) > 1e-8) {
    let topT = (halfHeight - ro.y) / rd.y;
    if (topT > 1e-5 && topT < bestT) {
      let p = ro + rd * topT;
      if (p.x * p.x + p.z * p.z <= radius * radius) {
        bestT = topT;
        bestN = vec3f(0.0, 1.0, 0.0);
      }
    }
    let bottomT = (-halfHeight - ro.y) / rd.y;
    if (bottomT > 1e-5 && bottomT < bestT) {
      let p = ro + rd * bottomT;
      if (p.x * p.x + p.z * p.z <= radius * radius) {
        bestT = bottomT;
        bestN = vec3f(0.0, -1.0, 0.0);
      }
    }
  }
  *nOut = bestN;
  return bestT;
}

fn intersectCapsuleLocal(ray: Ray, pa: vec3f, pb: vec3f, radius: f32, nOut: ptr<function, vec3f>) -> f32 {
  let ba = pb - pa;
  let oa = ray.origin - pa;
  let baba = dot(ba, ba);
  let bard = dot(ba, ray.direction);
  let baoa = dot(ba, oa);
  let rdoa = dot(ray.direction, oa);
  let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  let b = baba * rdoa - baoa * bard;
  let c = baba * oaoa - baoa * baoa - radius * radius * baba;
  let h = b * b - a * c;
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  if (h >= 0.0 && abs(a) > 1e-8) {
    let t = (-b - sqrt(h)) / a;
    let y = baoa + t * bard;
    if (t > 1e-5 && y > 0.0 && y < baba) {
      let p = oa + ray.direction * t - ba * (y / baba);
      bestT = t;
      bestN = safe_normalize(p);
    }
  }
  let ocA = ray.origin - pa;
  let bA = dot(ocA, ray.direction);
  let cA = dot(ocA, ocA) - radius * radius;
  let hA = bA * bA - cA;
  if (hA > 0.0) {
    let tA = -bA - sqrt(hA);
    if (tA > 1e-5 && tA < bestT) {
      bestT = tA;
      bestN = safe_normalize((ray.origin + ray.direction * tA) - pa);
    }
  }
  let ocB = ray.origin - pb;
  let bB = dot(ocB, ray.direction);
  let cB = dot(ocB, ocB) - radius * radius;
  let hB = bB * bB - cB;
  if (hB > 0.0) {
    let tB = -bB - sqrt(hB);
    if (tB > 1e-5 && tB < bestT) {
      bestT = tB;
      bestN = safe_normalize((ray.origin + ray.direction * tB) - pb);
    }
  }
  *nOut = bestN;
  return bestT;
}

fn intersectHChannelLocal(ray: Ray, lengthX: f32, railWidth: f32, blockHeight: f32, webThickness: f32, nOut: ptr<function, vec3f>) -> f32 {
  let hx = max(lengthX * 0.5, 1e-4);
  let hy = max(blockHeight * 0.5, 1e-4);
  let hz = max(railWidth * 0.5, 1e-4);
  let t = max(min(webThickness * 0.5, hy), 1e-4);
  var bestT = INFINITY;
  var bestN = vec3f(0.0, 1.0, 0.0);
  var n: vec3f;
  let railTop = intersectAabbDetailed(ray, vec3f(-hx, hy - t, -hz), vec3f(hx, hy, hz), 1e-4, INFINITY, &n);
  if (railTop < bestT) {
    bestT = railTop;
    bestN = n;
  }
  let railBottom = intersectAabbDetailed(ray, vec3f(-hx, -hy, -hz), vec3f(hx, -hy + t, hz), 1e-4, INFINITY, &n);
  if (railBottom < bestT) {
    bestT = railBottom;
    bestN = n;
  }
  let web = intersectAabbDetailed(ray, vec3f(-hx, -hy + t, -t), vec3f(hx, hy - t, t), 1e-4, INFINITY, &n);
  if (web < bestT) {
    bestT = web;
    bestN = n;
  }
  *nOut = bestN;
  return bestT;
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
) -> bool {
  if (params.bvhNodeCount == 0u || arrayLength(&bvhNodes) == 0u || rootNode >= min(params.bvhNodeCount, arrayLength(&bvhNodes))) {
    return false;
  }
  if (closest) {
    (*hit).didHit = false;
    (*hit).dist = tMaxBound;
    (*hit).triIndex = 0u;
    (*hit).normal = vec3f(0.0, 1.0, 0.0);
  }

  var stack: array<u32, 64>;
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
      let triFar = select(tMaxBound, (*hit).dist, closest);
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
        let hitT = intersectTriangle(ray.origin, ray.direction, a, b, c);
        if (hitT > tMin && hitT < triFar) {
          if (!closest) {
            return true;
          }
          let p = ray.origin + ray.direction * hitT;
          let ab = b - a;
          let ac = c - a;
          let ap = p - a;
          let d00 = dot(ab, ab);
          let d01 = dot(ab, ac);
          let d11 = dot(ac, ac);
          let d20 = dot(ap, ab);
          let d21 = dot(ap, ac);
          let denom = max(d00 * d11 - d01 * d01, 1e-8);
          let v = clamp((d11 * d20 - d01 * d21) / denom, 0.0, 1.0);
          let w = clamp((d00 * d21 - d01 * d20) / denom, 0.0, 1.0);
          let u = max(0.0, 1.0 - v - w);
          var shadeNormal = safe_normalize(cross(ab, ac));
          if (tri.x < arrayLength(&normals) && tri.y < arrayLength(&normals) && tri.z < arrayLength(&normals)) {
            let na = normals[tri.x].xyz;
            let nb = normals[tri.y].xyz;
            let nc = normals[tri.z].xyz;
            shadeNormal = safe_normalize(na * u + nb * v + nc * w);
          }
          (*hit).didHit = true;
          (*hit).dist = hitT;
          (*hit).triIndex = t;
          (*hit).normal = shadeNormal;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      // rightChildOrTriOffset is a RELATIVE offset (node units) from the current
      // node index; left child is always nodeIdx + 1. This matches the canonical
      // relative-offset encoding used by shared-bvh/normalizeBvhInteriorOffsets
      // and walkaround-hybrid/common.wgsl. Invariant: 1 <= offset < totalNodes.
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < 64u) {
        stack[stackPtr] = rightChild;
        stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild;
        stackPtr = stackPtr + 1u;
      } else {
        // Stack overflow: bail out with current best-hit (already
        // written into *hit if closest, else simply 'no hit found yet')
        // rather than silently dropping both children.  At depth 64 a
        // balanced BVH spans 2^64 triangles so this branch is
        // unreachable for any real scene; the guard exists for invariant
        // clarity and to surface degenerate inputs deterministically.
        return (*hit).didHit;
      }
    }
  }
  return false;
}

fn traceAnalyticShapes(
  ray: Ray,
  tMin: f32,
  tMaxBound: f32,
  closest: bool,
  hit: ptr<function, SceneHit>,
) -> bool {
  let analyticTotal = min(params.analyticCount, arrayLength(&analyticHeaders));
  for (var ai = 0u; ai < analyticTotal; ai = ai + 1u) {
    let header = analyticHeaders[ai];
    let shapeId = u32(max(header.x, 0.0));
    let paramOffset = u32(max(header.z, 0.0));
    let matBase = ai * 4u;
    if (matBase + 3u >= arrayLength(&analyticWorldToLocal) || matBase + 3u >= arrayLength(&analyticLocalToWorld)) {
      continue;
    }
    let w2l0 = analyticWorldToLocal[matBase];
    let w2l1 = analyticWorldToLocal[matBase + 1u];
    let w2l2 = analyticWorldToLocal[matBase + 2u];
    let w2l3 = analyticWorldToLocal[matBase + 3u];
    let l2w0 = analyticLocalToWorld[matBase];
    let l2w1 = analyticLocalToWorld[matBase + 1u];
    let l2w2 = analyticLocalToWorld[matBase + 2u];
    let l2w3 = analyticLocalToWorld[matBase + 3u];
    var localRay: Ray;
    localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
    localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
    var localN = vec3f(0.0, 1.0, 0.0);
    var localT = INFINITY;
    let p0 = select(vec4f(0.0), analyticParams[paramOffset], paramOffset < arrayLength(&analyticParams));
    let p1 = select(vec4f(0.0), analyticParams[paramOffset + 1u], paramOffset + 1u < arrayLength(&analyticParams));
    if (shapeId == SHAPE_SPHERE) {
      localT = intersectSphereLocal(localRay, p0.xyz, max(p0.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_BOX) {
      localT = intersectAabbDetailed(localRay, p0.xyz - p1.xyz, p0.xyz + p1.xyz, 1e-4, INFINITY, &localN);
    } else if (shapeId == SHAPE_CAPSULE) {
      localT = intersectCapsuleLocal(localRay, p0.xyz, p1.xyz, max(p1.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_CYLINDER) {
      localT = intersectCylinderLocal(localRay, p0.xyz, max(p0.w, 1e-4), max(p1.x, 1e-4), &localN);
    } else if (shapeId == SHAPE_H_CHANNEL_CAME) {
      localT = intersectHChannelLocal(localRay, p0.x, p0.y, p0.z, p0.w, &localN);
    }
    if (localT <= tMin || localT >= INFINITY) {
      continue;
    }
    let localHitPos = localRay.origin + localRay.direction * localT;
    let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
    let worldT = dot(worldHitPos - ray.origin, ray.direction);
    let bound = select(tMaxBound, (*hit).dist, closest);
    if (worldT > tMin && worldT < bound) {
      if (!closest) {
        return true;
      }
      (*hit).didHit = true;
      (*hit).dist = worldT;
      (*hit).triIndex = params.triangleCount + ai;
      (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localN);
    }
  }
  return false;
}

fn traceTlasClosest(ray: Ray, tMin: f32, tMax: f32, hit: ptr<function, SceneHit>) -> bool {
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceMeshBvh(ray, tMin, tMax, true, hit, 0u);
  }
  (*hit).didHit = false;
  (*hit).dist = tMax;
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(params.tlasNodeCount, arrayLength(&tlasNodes))) { continue; }
    let node = tlasNodes[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!intersectAabb(ray, bmin, bmax, tMin, (*hit).dist)) { continue; }
    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= arrayLength(&tlasInstanceIndices)) { continue; }
        let instIdx = tlasInstanceIndices[permIdx];
        let m = instIdx * 4u;
        if (m + 3u >= arrayLength(&tlasInstanceTransforms)) { continue; }
        let w2l0 = tlasInstanceTransforms[m];
        let w2l1 = tlasInstanceTransforms[m + 1u];
        let w2l2 = tlasInstanceTransforms[m + 2u];
        let w2l3 = tlasInstanceTransforms[m + 3u];
        var localRay: Ray;
        localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
        localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
        var localHit: SceneHit;
        let blasRoot = select(0u, tlasBlasRoots[instIdx], instIdx < arrayLength(&tlasBlasRoots));
        _ = traceMeshBvh(localRay, tMin, (*hit).dist, true, &localHit, blasRoot);
        if (localHit.didHit && localHit.dist > tMin && localHit.dist < (*hit).dist) {
          // TLAS instances currently use identity worldToLocal in host packing.
          // Keep distance/normal in local space until per-instance L2W lands.
          (*hit).didHit = true;
          (*hit).dist = localHit.dist;
          (*hit).triIndex = localHit.triIndex;
          (*hit).normal = localHit.normal;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < 64u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        return (*hit).didHit;
      }
    }
  }
  return (*hit).didHit;
}

fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceTlasClosest(ray, tMin, tMax, &hit);
  _ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);
  return hit;
}

fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
  var hit: SceneHit;
  if (traceTlasClosest(ray, tMin, tMax, &hit)) {
    return true;
  }
  if (traceAnalyticShapes(ray, tMin, tMax, false, &hit)) {
    return true;
  }
  return false;
}

fn hitMaterialId(hit: SceneHit) -> u32 {
  if (hit.triIndex < params.triangleCount) {
    return select(0u, triMaterialIds[hit.triIndex], hit.triIndex < arrayLength(&triMaterialIds));
  }
  let analyticIndex = hit.triIndex - params.triangleCount;
  if (analyticIndex < arrayLength(&analyticHeaders)) {
    return u32(max(analyticHeaders[analyticIndex].y, 0.0));
  }
  return 0u;
}
`;
