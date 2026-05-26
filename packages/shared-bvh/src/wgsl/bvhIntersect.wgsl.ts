/**
 * Canonical WGSL module for Möller–Trumbore triangle intersection + BVH
 * ordered-traversal helpers.
 *
 * Hoist target — sweep-20260518/moller-trumbore-canonical. Replaces three
 * drifted in-tree copies:
 *   - DDGI    (probeUpdateRays.wgsl.ts:222-308) — hardcoded triEps = 1e-5
 *   - ReSTIR  (common.wgsl.ts:580-722)          — triEps parameter
 *   - RC      (probeRayCast.wgsl.ts:124-274)    — triEps parameter
 *
 * Single source of truth for:
 *   - The `IntersectionResult` struct (superset of all three consumers'
 *     fields, so any caller can read what it needs and ignore the rest).
 *   - The Williams-2005 IEEE-safe `safeInvDir` helper (deduplicated).
 *   - `intersectTriangle(origin, dir, a, b, c, triEps) -> IntersectionResult`
 *     — the algorithm itself; storage-agnostic (takes three world-space
 *     vertex positions).
 *   - `bvhIntersectFirstHit(bvh_index, bvh_position, bvh, ray, triEps) ->
 *     IntersectionResult` — full ordered-stack closest-hit traversal for
 *     `array<vec4u>` / `array<vec4f>` storage (ReSTIR's pre-canonical form).
 *   - `bvhIntersectAny(bvh_index, bvh_position, bvh, origin, dir, tMax,
 *     triEps, skipGlass: bool) -> bool` — shadow-ray any-hit traversal
 *     with caller-chosen glass-skip behaviour.
 *   - `bvhIntersectFirstHitV3(bvh_index, bvh_position, bvh, ray, triEps) ->
 *     IntersectionResult` — same algorithm but for `array<vec3u>` /
 *     `array<vec3f>` storage (DDGI / RC's pre-canonical three-mesh-bvh
 *     upstream form). DDGI and RC keep their existing host-side packers
 *     (12-byte stride per element) and bind the canonical V3 entry point.
 *
 * Storage forms:
 *   - vec4u / vec4f variant (ReSTIR):
 *       bvh_index    is `array<vec4u>`: .xyz = vertex indices,
 *                                       .w = packed RGB888 + (trans4|texType4)
 *                                            material color (ReSTIR), or
 *                                            zero (DDGI/RC zero-fill).
 *       bvh_position is `array<vec4f>`: .xyz = world-space position,
 *                                       .w = 16:16 unorm UV (ReSTIR), or zero.
 *   - vec3u / vec3f variant (DDGI / RC):
 *       bvh_index    is `array<vec3u>`: three vertex indices (no .w payload).
 *       bvh_position is `array<vec3f>`: world-space position (no .w payload).
 *
 *   `bvh` is `array<BVHNode>` in both cases — 32-byte flat layout
 *   (boundsMin 3×f32, boundsMax 3×f32, rightChildOrTriOffset u32,
 *   splitAxisOrTriCount u32) matching three-mesh-bvh's `BYTES_PER_NODE`.
 *
 * Pre-canonical DDGI used a nested-struct BVHNode (boundsBox→min/max + two
 * separately-named offset fields); migrating its WGSL to the canonical flat
 * struct is a rename-only change (`node.bounds.min[0]` → `node.boundsMin[0]`,
 * `node.rightChildOrTriangleOffset` → `node.rightChildOrTriOffset`,
 * `node.splitAxisOrTriangleCount` → `node.splitAxisOrTriCount`). RC's
 * upstream-shape struct undergoes the same flattening.
 *
 * @see CREDITS.md (Möller & Trumbore 1997; three-mesh-bvh; Williams 2005)
 */

export const BVH_INTERSECT_WGSL = /* wgsl */ `

// ─── BVH traversal constants ─────────────────────────────────────────────────
// Stack depth 60 supports balanced BVHs up to 2^60 triangles — unreachable
// for any real scene. The two pre-canonical copies used 60 (DDGI/RC) and 64
// (ReSTIR); 60 is sufficient and matches three-mesh-bvh upstream.
const BVH_INTERSECT_STACK_DEPTH: u32 = 60u;
const BVH_INTERSECT_INFINITY: f32 = 1e20;
const BVH_LEAFNODE_FLAG: u32 = 0xFFFF0000u;

// ─── BVH structs ─────────────────────────────────────────────────────────────
// BVHNode: 32 bytes -- exactly matches three-mesh-bvh's raw node layout.
// From three-mesh-bvh/src/core/Constants.js: BYTES_PER_NODE = 6*4 + 4 + 4 = 32.
// Binary layout:
//   bytes  0-11:  boundsMin[0..2]  (3 × f32, NO padding between min+max)
//   bytes 12-23:  boundsMax[0..2]  (3 × f32)
//   bytes 24-27:  rightChildOrTriOffset (u32)
//   bytes 28-31:  splitAxisOrTriCount   (u32, leaf flag 0xFFFF0000 | count)
// We use array<f32,3> (align 4, stride 12) NOT vec3f (align 16) so the
// boundsMin/boundsMax fields pack without padding -- matches three-mesh-bvh.
struct BVHNode {
  boundsMin: array<f32, 3>,
  boundsMax: array<f32, 3>,
  rightChildOrTriOffset: u32,
  splitAxisOrTriCount: u32,
};

struct Ray {
  origin:    vec3f,
  direction: vec3f,
};

// Superset of the three pre-canonical result structs:
//   DDGI / RC  IntersectionResult { didHit, indices: vec4u, normal,
//                                   barycoord, side, dist }
//   ReSTIR     HitResult          { didHit, dist, triIndex, bary, normal,
//                                   matColorPacked, uv }
//
// Field mapping:
//   indices.xyz   : vertex indices into bvh_position (DDGI / RC)
//   indices.w     : triIndex (== triOffset+i; ReSTIR pre-canonical triIndex)
//   normal        : geometric face normal (sign-correct via side)
//   barycoord     : (w, u, v) on the hit triangle (ReSTIR pre-canonical bary)
//   side          : sign(det) — +1 for front-face, -1 for back-face hit
//   dist          : ray-parameter t at the hit
//   matColorPacked: copy of bvh_index[triIdx].w (ReSTIR only; zero for DDGI/RC)
//   uv            : interpolated 16:16 unorm UV (ReSTIR only; vec2f(0) elsewhere)
struct IntersectionResult {
  didHit:         bool,
  indices:        vec4u,
  normal:         vec3f,
  barycoord:      vec3f,
  side:           f32,
  dist:           f32,
  matColorPacked: u32,
  uv:             vec2f,
};

// ─── Williams 2005 §4 IEEE-safe inverse-direction helper ─────────────────────
// Prevents NaN from 0 * ±Inf in slab tests when a ray direction component
// is exactly zero. For the zero/near-zero case the inv-component is set
// to a large signed sentinel (±1e30) so the slab test still picks up
// whether the ray's origin is inside the AABB on the parallel axis:
//
//   - origin inside the X slab → t0/t1 ~= ±1e30, contributes ~unbounded
//     range, and the other two axes determine entry/exit (correct).
//   - origin outside the X slab → t0 and t1 are both far negative or both
//     far positive, so tNear pushes past tFar and the slab test rejects.
//
// Earlier revision used sign(d.x) * 1e30, but WGSL sign(0) == 0 so an
// exact-zero direction yielded 0, collapsing the X slab's contribution
// to t0 == t1 == 0 regardless of origin position — a false positive for
// rays whose origin sat outside the AABB on the parallel axis. The
// select(-1e30, 1e30, d.x >= 0.0) form picks a definite sign even when
// d.x is exactly zero (treated as positive, matching IEEE 754 +0 >= 0).
fn safeInvDir(d: vec3f) -> vec3f {
  return vec3f(
    select(1.0 / d.x, select(-1e30, 1e30, d.x >= 0.0), abs(d.x) < 1e-30),
    select(1.0 / d.y, select(-1e30, 1e30, d.y >= 0.0), abs(d.y) < 1e-30),
    select(1.0 / d.z, select(-1e30, 1e30, d.z >= 0.0), abs(d.z) < 1e-30),
  );
}

// ─── Möller–Trumbore triangle intersection ───────────────────────────────────
// Möller & Trumbore 1997 — single-sided ray/triangle intersection with the
// edge-cross factoring that avoids the explicit normal computation.
// triEps is the coplanarity floor: rays whose determinant magnitude falls
// below it are treated as parallel to the triangle (returns no-hit).
// Negative-bary tolerance also uses triEps so a hit grazing an edge by
// less than the floor is accepted instead of rejected — matching the
// three-mesh-bvh upstream behaviour and the pre-canonical DDGI / RC code.
fn intersectTriangle(
  origin: vec3f, dir: vec3f,
  a: vec3f, b: vec3f, c: vec3f,
  triEps: f32,
) -> IntersectionResult {
  var result: IntersectionResult;
  result.didHit = false;
  result.dist = BVH_INTERSECT_INFINITY;

  let e1 = b - a;
  let e2 = c - a;
  let n  = cross(e1, e2);
  let det = -dot(dir, n);
  if (abs(det) < triEps) { return result; }

  let invDet = 1.0 / det;
  let AO  = origin - a;
  let DAO = cross(AO, dir);

  let u = dot(e2, DAO) * invDet;
  let v = -dot(e1, DAO) * invDet;
  let t = dot(AO, n)   * invDet;
  let w = 1.0 - u - v;

  if (u < -triEps || v < -triEps || w < -triEps || t < triEps) {
    return result;
  }

  result.didHit    = true;
  result.dist      = t;
  result.barycoord = vec3f(w, u, v);
  result.side      = sign(det);
  result.normal    = result.side * normalize(n);
  return result;
}

// ─── BVH ordered closest-hit traversal — vec4 storage (ReSTIR) ───────────────
// Wald 2007 / PBR4e §7.3.3 — stack-based DFS with near-first ordering by
// ray-direction sign on the split axis. three-mesh-bvh node layout:
//   rightChildOrTriOffset is a RELATIVE offset (in node units) from the
//   current node; the left child is always nodeIdx+1.
// On a leaf (splitAxisOrTriCount & 0xFFFF0000u == BVH_LEAFNODE_FLAG):
//   triCount  = splitAxisOrTriCount & 0x0000FFFFu
//   triOffset = rightChildOrTriOffset
fn bvhIntersectFirstHit(
  bvh_index:    ptr<storage, array<vec4u>,   read>,
  bvh_position: ptr<storage, array<vec4f>,   read>,
  bvh:          ptr<storage, array<BVHNode>, read>,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  return bvhIntersectFirstHitAtRoot(bvh_index, bvh_position, bvh, ray, triEps, 0u);
}

fn bvhIntersectFirstHitAtRoot(
  bvh_index:    ptr<storage, array<vec4u>,   read>,
  bvh_position: ptr<storage, array<vec4f>,   read>,
  bvh:          ptr<storage, array<BVHNode>, read>,
  ray: Ray,
  triEps: f32,
  rootNode: u32,
) -> IntersectionResult {
  var best: IntersectionResult;
  best.didHit = false;
  best.dist   = BVH_INTERSECT_INFINITY;
  best.matColorPacked = 0u;
  best.uv     = vec2f(0.0);

  var stack: array<u32, 60>;  // BVH_INTERSECT_STACK_DEPTH
  var pointer: i32 = 0;
  stack[0] = rootNode;

  let invDir = safeInvDir(ray.direction);

  loop {
    if (pointer < 0 || pointer >= i32(BVH_INTERSECT_STACK_DEPTH)) { break; }
    let currNodeIdx = stack[pointer];
    let node        = (*bvh)[currNodeIdx];
    pointer = pointer - 1;

    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    let t0 = (bmin - ray.origin) * invDir;
    let t1 = (bmax - ray.origin) * invDir;
    let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tFar  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (tNear > tFar || tFar < 0.0 || tNear > best.dist) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    let isLeaf = (splitOrCount & 0xFFFF0000u) == BVH_LEAFNODE_FLAG;

    if (isLeaf) {
      let triCount  = splitOrCount & 0x0000FFFFu;
      let triOffset = node.rightChildOrTriOffset;
      for (var i = 0u; i < triCount; i = i + 1u) {
        let triIdx   = triOffset + i;
        let idxEntry = (*bvh_index)[triIdx];
        let idx      = idxEntry.xyz;
        let pa4 = (*bvh_position)[idx.x];
        let pb4 = (*bvh_position)[idx.y];
        let pc4 = (*bvh_position)[idx.z];
        let tri = intersectTriangle(
          ray.origin, ray.direction,
          pa4.xyz, pb4.xyz, pc4.xyz,
          triEps,
        );
        if (tri.didHit && tri.dist < best.dist) {
          best = tri;
          best.indices        = vec4u(idx, triIdx);
          best.matColorPacked = idxEntry.w;
          // Interpolate UV from 16:16 unorm packed in each position's .w word.
          // For DDGI/RC who zero-fill .w, unpack2x16unorm returns vec2f(0) and
          // the consumers ignore best.uv.
          let uvA = unpack2x16unorm(bitcast<u32>(pa4.w));
          let uvB = unpack2x16unorm(bitcast<u32>(pb4.w));
          let uvC = unpack2x16unorm(bitcast<u32>(pc4.w));
          best.uv = best.barycoord.x * uvA + best.barycoord.y * uvB + best.barycoord.z * uvC;
        }
      }
    } else {
      // Interior node — ordered traversal (Wald 2007 / PBR4e §7.3.3).
      // splitAxisOrTriCount low 2 bits encode the split axis (0=X,1=Y,2=Z).
      // Push far child first so near child is popped (and tested) first.
      let leftIdx   = currNodeIdx + 1u;
      let rightIdx  = currNodeIdx + node.rightChildOrTriOffset;
      let axis      = splitOrCount & 0x3u;
      let leftToRight = ray.direction[axis] >= 0.0;
      let nearChild = select(rightIdx, leftIdx, leftToRight);
      let farChild  = select(leftIdx, rightIdx, leftToRight);

      // Bail out cleanly with current best-hit if pushing both children
      // would overflow. Without this guard the unconditional push wrote
      // past the end of stack[BVH_INTERSECT_STACK_DEPTH] (WGSL clamps the
      // index, corrupting stack[BVH_INTERSECT_STACK_DEPTH-1]) before the
      // loop-top check could fire. At depth 60 a balanced BVH spans 2^60
      // triangles, so this branch is unreachable for any real scene.
      if (pointer + 2 >= i32(BVH_INTERSECT_STACK_DEPTH)) {
        return best;
      }
      pointer = pointer + 1;
      stack[pointer] = farChild;
      pointer = pointer + 1;
      stack[pointer] = nearChild;
    }
  }

  return best;
}

// ─── BVH any-hit (shadow ray) traversal — vec4 storage (ReSTIR) ──────────────
// Returns true on the first triangle hit in (1e-4, tMax). The 1e-4 floor
// matches the pre-canonical ReSTIR bvhIntersectAny — it avoids self-
// intersection on shadow rays cast from a hit point.
//
// skipGlass selects the glass filter behaviour:
//   true  → treat triangles whose (bvh_index[i].w >> 4 & 0xF) exceeds 4 as
//           transparent (light passes through). Matches ReSTIR's pre-canonical
//           filter (common.wgsl.ts:545  if (trans4 > 4u) continue), which
//           lets sun rays cast through glass to produce caustic-relevant
//           direct lighting; pHat evaluation in shade.wgsl then handles the
//           glass tint via the per-channel tinted-visibility helper in
//           surfaceTextures.wgsl.
//   false → treat all hits as occluders (no glass-aware filter).
fn bvhIntersectAny(
  bvh_index:    ptr<storage, array<vec4u>,   read>,
  bvh_position: ptr<storage, array<vec4f>,   read>,
  bvh:          ptr<storage, array<BVHNode>, read>,
  origin: vec3f,
  dir:    vec3f,
  tMax:   f32,
  triEps: f32,
  skipGlass: bool,
) -> bool {
  return bvhIntersectAnyAtRoot(bvh_index, bvh_position, bvh, origin, dir, tMax, triEps, skipGlass, 0u);
}

fn bvhIntersectAnyAtRoot(
  bvh_index:    ptr<storage, array<vec4u>,   read>,
  bvh_position: ptr<storage, array<vec4f>,   read>,
  bvh:          ptr<storage, array<BVHNode>, read>,
  origin: vec3f,
  dir:    vec3f,
  tMax:   f32,
  triEps: f32,
  skipGlass: bool,
  rootNode: u32,
) -> bool {
  var stack: array<u32, 60>;  // BVH_INTERSECT_STACK_DEPTH
  var stackPtr = 0u;
  stack[stackPtr] = rootNode; stackPtr = stackPtr + 1u;

  let invDir = safeInvDir(dir);

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    let node = (*bvh)[nodeIdx];

    let nMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let nMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    let t1 = (nMin - origin) * invDir;
    let t2 = (nMax - origin) * invDir;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tNear > tFar || tFar < 0.0 || tNear > tMax) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & 0xFFFF0000u) == BVH_LEAFNODE_FLAG) {
      let count  = splitOrCount & 0x0000FFFFu;
      let offset = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let triIdx = offset + i;
        let idxEntry = (*bvh_index)[triIdx];
        let idx = idxEntry.xyz;
        if (skipGlass) {
          // Transmission is a 4-bit unorm in bits [7:4] of idxEntry.w;
          // glass has transmission > ~0.3 → packed > 4.
          let trans4 = (idxEntry.w >> 4u) & 0xFu;
          if (trans4 > 4u) { continue; }
        }
        let a = (*bvh_position)[idx.x].xyz;
        let b = (*bvh_position)[idx.y].xyz;
        let c = (*bvh_position)[idx.z].xyz;
        let tri = intersectTriangle(origin, dir, a, b, c, triEps);
        if (tri.didHit && tri.dist > 1e-4 && tri.dist < tMax) { return true; }
      }
    } else {
      // Ordered traversal (Wald 2007 / PBR4e §7.3.3).
      let rightChild  = nodeIdx + node.rightChildOrTriOffset;
      let axis        = splitOrCount & 0x3u;
      let leftToRight = dir[axis] >= 0.0;
      let nearChild   = select(rightChild,   nodeIdx + 1u, leftToRight);
      let farChild    = select(nodeIdx + 1u, rightChild,   leftToRight);
      // Stack-overflow guard: bail out with not-yet-occluded (false) rather
      // than silently dropping the far subtree.
      if (stackPtr + 1u >= 60u) {  // BVH_INTERSECT_STACK_DEPTH
        return false;
      }
      stack[stackPtr] = farChild;  stackPtr = stackPtr + 1u;
      stack[stackPtr] = nearChild; stackPtr = stackPtr + 1u;
    }
  }
  return false;
}

// ─── BVH ordered closest-hit traversal — vec3 storage (DDGI / RC) ────────────
// Identical algorithm to bvhIntersectFirstHit, but reads bvh_index and
// bvh_position as array<vec3u> / array<vec3f> (DDGI / RC three-mesh-bvh
// upstream storage form). The pre-canonical DDGI and RC consumers used
// this layout; this entry point preserves their behaviour byte-for-byte
// while sharing the algorithm with the vec4-storage variant. The returned
// IntersectionResult has matColorPacked = 0u and uv = vec2f(0) — neither
// is meaningful for stride-3 storage (no .w payload).
fn bvhIntersectFirstHitV3(
  bvh_index:    ptr<storage, array<vec3u>,   read>,
  bvh_position: ptr<storage, array<vec3f>,   read>,
  bvh:          ptr<storage, array<BVHNode>, read>,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  var best: IntersectionResult;
  best.didHit = false;
  best.dist   = BVH_INTERSECT_INFINITY;
  best.matColorPacked = 0u;
  best.uv     = vec2f(0.0);

  var stack: array<u32, 60>;  // BVH_INTERSECT_STACK_DEPTH
  var pointer: i32 = 0;
  stack[0] = 0u;

  let invDir = safeInvDir(ray.direction);

  loop {
    if (pointer < 0 || pointer >= i32(BVH_INTERSECT_STACK_DEPTH)) { break; }
    let currNodeIdx = stack[pointer];
    let node        = (*bvh)[currNodeIdx];
    pointer = pointer - 1;

    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    let t0 = (bmin - ray.origin) * invDir;
    let t1 = (bmax - ray.origin) * invDir;
    let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tFar  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (tNear > tFar || tFar < 0.0 || tNear > best.dist) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    let isLeaf = (splitOrCount & 0xFFFF0000u) == BVH_LEAFNODE_FLAG;

    if (isLeaf) {
      let triCount  = splitOrCount & 0x0000FFFFu;
      let triOffset = node.rightChildOrTriOffset;
      for (var i = 0u; i < triCount; i = i + 1u) {
        let triIdx = triOffset + i;
        let idx    = (*bvh_index)[triIdx];
        let a = (*bvh_position)[idx.x];
        let b = (*bvh_position)[idx.y];
        let c = (*bvh_position)[idx.z];
        let tri = intersectTriangle(
          ray.origin, ray.direction,
          a, b, c,
          triEps,
        );
        if (tri.didHit && tri.dist < best.dist) {
          best = tri;
          best.indices = vec4u(idx, triIdx);
        }
      }
    } else {
      let leftIdx   = currNodeIdx + 1u;
      let rightIdx  = currNodeIdx + node.rightChildOrTriOffset;
      let axis      = splitOrCount & 0x3u;
      let leftToRight = ray.direction[axis] >= 0.0;
      let nearChild = select(rightIdx, leftIdx, leftToRight);
      let farChild  = select(leftIdx, rightIdx, leftToRight);
      if (pointer + 2 >= i32(BVH_INTERSECT_STACK_DEPTH)) {
        return best;
      }
      pointer = pointer + 1;
      stack[pointer] = farChild;
      pointer = pointer + 1;
      stack[pointer] = nearChild;
    }
  }

  return best;
}

`;
