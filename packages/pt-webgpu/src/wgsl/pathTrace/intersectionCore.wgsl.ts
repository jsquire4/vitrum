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
  triIndex: u32,
  normal: vec3f,
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

// SHADOW-01 — primitive castShadow. Material vec4 #25 .w (the former pad lane)
// carries 1.0 when the SOURCE PRIMITIVE set castShadow:false (0.0 default —
// see scene/materialPacking.ts vec4 #25). Any-hit (occlusion) traversals skip
// such triangles so NEE shadow rays / visibility connections pass through;
// closest-hit (camera / radiance) traversals never call this, keeping the
// geometry camera-visible. Both tiers compose the material module that
// declares \`materials\` / \`triMaterialIds\` / MATERIAL_VEC4_STRIDE before this
// module, so the symbols resolve in every composition.
fn triShadowCastDisabled(triIdx: u32) -> bool {
  if (triIdx >= arrayLength(&triMaterialIds)) { return false; }
  let matId = triMaterialIds[triIdx];
  let vecIndex = matId * MATERIAL_VEC4_STRIDE + 25u;
  if (vecIndex >= arrayLength(&materials)) { return false; }
  return materials[vecIndex].w > 0.5;
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
  if (params.bvhNodeCount == 0u || arrayLength(&bvhNodes) == 0u || rootNode >= min(params.bvhNodeCount, arrayLength(&bvhNodes))) {
    return false;
  }
  if (closest) {
    (*hit).didHit = false;
    (*hit).dist = tMaxBound;
    (*hit).triIndex = 0u;
    (*hit).normal = vec3f(0.0, 1.0, 0.0);
    (*hit).baryVW = vec2f(0.0);
    (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
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
        // LIVE upper bound — in closest mode compare against the RUNNING
        // nearest (*hit).dist (initialized to tMaxBound above), re-read every
        // iteration. A per-leaf snapshot here made the accept last-writer-wins
        // WITHIN a leaf: a farther triangle tested later overwrote a nearer
        // accepted hit, so thin slabs (walls) shaded from their buried far
        // face — NEE always occluded + bounce rays trapped → black geometry
        // (G-P0.3 capture found this via the face-on Cornell back wall).
        if (hitT > tMin && hitT < select(tMaxBound, (*hit).dist, closest)) {
          if (!closest) {
            // SHADOW-01 — any-hit mode is exclusively occlusion (shadow /
            // visibility) queries: skip castShadow:false geometry.
            if (triShadowCastDisabled(t)) {
              continue;
            }
            return true;
          }
          var shadeNormal = vec3f(0.0, 1.0, 0.0);
          // baryVW (v,w) of the hit — captured with the shading normal so the
          // kernel can interpolate per-vertex UVs. Defaults to 0 (→ vertex-0 UV)
          // for any-hit traversals that skip shading details. (P2)
          var shadeBaryVW = vec2f(0.0);
          if (captureShadingDetails) {
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
            shadeBaryVW = vec2f(v, w);
            shadeNormal = safe_normalize(cross(ab, ac));
            if (tri.x < arrayLength(&normals) && tri.y < arrayLength(&normals) && tri.z < arrayLength(&normals)) {
              let na = normals[tri.x].xyz;
              let nb = normals[tri.y].xyz;
              let nc = normals[tri.z].xyz;
              shadeNormal = safe_normalize(na * u + nb * v + nc * w);
            }
          }
          (*hit).didHit = true;
          (*hit).dist = hitT;
          (*hit).triIndex = t;
          (*hit).normal = shadeNormal;
          (*hit).baryVW = shadeBaryVW;
          (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
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
  return select(false, (*hit).didHit, closest);
}`;
