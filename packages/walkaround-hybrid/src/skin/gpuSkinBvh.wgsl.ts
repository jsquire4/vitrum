/**
 * GPU linear-blend skinning into the merged ReSTIR `bvhPositions` buffer
 * (stride-4, preserving the packed UV lane in `.w`), PLUS an optional skinned
 * NORMAL output.
 *
 * Positions are skinned by the blended bone matrices (LBS). Normals are
 * skinned by the INVERSE-TRANSPOSE of the blended skin matrix's linear part
 * (PBR4e §3.10) — correct for non-uniformly-scaled / sheared bones, and equal
 * to the plain transform for rigid (rotation-only) bones. This mirrors the CPU
 * reference `solveSkin` (three-bindings/skinSolver.ts), which uses
 * `mat3InverseTranspose` on the same blended linear part.
 *
 * Architectural note (merged-BVH normal consumption): the merged ReSTIR BVH
 * still derives the GEOMETRIC face normal of a ray hit via `cross(e1, e2)` of
 * the (already skinned) triangle vertex positions (shared-bvh/bvhIntersect.wgsl
 * `result.normal = side * normalize(n)`). WS1 (2026-05-29) added a per-vertex
 * SMOOTH shading normal on top: the primary passes barycentric-blend a
 * per-vertex `bvh_normal` buffer. THIS kernel's skinned normals ARE that
 * buffer's live skinned values — `skinnedNormals` (binding 7) is the SHARED
 * merged `bvh_normal` SSBO, written at the same `baseVertex + vi` slot as the
 * position, so the smooth-normal blend sees the deformed normal each frame.
 * (Pre-WS1 this wrote a per-mesh `skinnedNormals[vi]` buffer that the refit
 * dropped — computed-but-unconsumed; now consumed.)
 */

/**
 * Position + normal LBS variant. Accumulates skinned positions and the blended
 * skin matrix's upper-3×3, takes
 * its inverse-transpose, applies it to the rest normal, and writes the
 * normalized result into `skinnedNormals` (binding 7, stride-4 with `.w`
 * preserving packed UV1). When `applyWorld != 0`, the world matrix's upper-3×3
 * inverse-transpose is composed in (matching how positions are world-transformed).
 *
 * Indexing (WS1): positions write at `baseVertex + vi` into the SHARED merged
 * `bvhPositions` buffer; skinned normals write at the SAME `baseVertex + vi`
 * slot into the SHARED merged `bvh_normal` buffer (bound at binding 7), so the
 * smooth-shading-normal blend reads the deformed normal directly.
 */
export const GPU_SKIN_BVH_WITH_NORMALS_WGSL = /* wgsl */ `
struct SkinBvhUniforms {
  vertexCount: u32,
  baseVertex: u32,
  applyWorld: u32,
  _pad: u32,
  matrixWorld: mat4x4f,
};

@group(0) @binding(0) var<uniform> skinParams: SkinBvhUniforms;
@group(0) @binding(1) var<storage, read> restPositions: array<vec4f>;
@group(0) @binding(2) var<storage, read> restNormals: array<vec4f>;
@group(0) @binding(3) var<storage, read> skinIndices: array<vec4u>;
@group(0) @binding(4) var<storage, read> skinWeights: array<vec4f>;
@group(0) @binding(5) var<storage, read> boneMatrices: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> bvhPositions: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> skinnedNormals: array<vec4f>;

fn mat4MulVec4N(m: array<vec4f, 4>, v: vec4f) -> vec4f {
  return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
}

// Inverse-transpose of a column-major 3×3 (PBR4e §3.10). Columns c0,c1,c2.
// Mirrors @vitrum/core skinSolver.mat3InverseTranspose: a rank-2 transform
// keeps the finite cofactor orientation (its scale is removed when the normal
// is normalized), while rank < 2 has no defined normal direction and returns
// zero. Never substitute the raw matrix: that is finite but geometrically
// incorrect for a collapsed bone transform.
fn mat3InverseTranspose(c0: vec3f, c1: vec3f, c2: vec3f) -> mat3x3f {
  // Rank tests must be scale-invariant. Without this normalization, a valid
  // full-rank transform such as uniform scale 1e-11 has det 1e-33 and
  // cofactors around 1e-22, so absolute thresholds incorrectly classify it
  // below rank 2 and erase the normal.
  let componentScale = max(max(abs(c0), abs(c1)), abs(c2));
  let matrixScale = max(componentScale.x, max(componentScale.y, componentScale.z));
  if (matrixScale <= 0.0) {
    return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  }
  let n0 = c0 / matrixScale;
  let n1 = c1 / matrixScale;
  let n2 = c2 / matrixScale;
  let cofactor0 = cross(n1, n2);
  let cofactor1 = cross(n2, n0);
  let cofactor2 = cross(n0, n1);
  // det via scalar triple product of the columns.
  let det = dot(n0, cofactor0);
  if (abs(det) < 1e-20) {
    let cofactorMagnitude = sqrt(
      dot(cofactor0, cofactor0) +
      dot(cofactor1, cofactor1) +
      dot(cofactor2, cofactor2)
    );
    if (cofactorMagnitude < 1e-20) {
      return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    }
    return mat3x3f(cofactor0, cofactor1, cofactor2);
  }
  let invDetAndScale = (1.0 / det) / matrixScale;
  // For M = [c0 c1 c2], (M^-1)^T columns are the cofactor columns / det:
  //   col0 = (c1 × c2)/det, col1 = (c2 × c0)/det, col2 = (c0 × c1)/det.
  // Dividing once more by matrixScale restores the inverse of the authored
  // matrix after the scale-invariant rank classification above.
  return mat3x3f(
    cofactor0 * invDetAndScale,
    cofactor1 * invDetAndScale,
    cofactor2 * invDetAndScale,
  );
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let vi = gid.x;
  if (vi >= skinParams.vertexCount) {
    return;
  }
  let outIdx = skinParams.baseVertex + vi;
  let uvPack = bvhPositions[outIdx].w;
  let uv1Pack = skinnedNormals[outIdx].w;
  let rp = restPositions[vi].xyz;
  let rn = restNormals[vi].xyz;
  let idx = skinIndices[vi];
  let w = skinWeights[vi];

  // H17 — seed sp at zero (mirrors solveSkin's accumulate-from-zero convention).
  // The old seed vec4f(0,0,0,1) caused sp.w = 1 + Σwᵢ ≈ 2 for normalized
  // weights, which doubled the translation column when matrixWorld was applied
  // in merged mode (applyWorld=1). seeding at (0,0,0,0) gives sp.w = Σwᵢ ≈ 1,
  // and we clamp it to exactly 1.0 before the matrixWorld multiply for
  // robustness with non-normalized weights.
  var sp = vec4f(0.0);
  // Blended skin upper-3×3 columns (the linear part applied to directions).
  var col0 = vec3f(0.0);
  var col1 = vec3f(0.0);
  var col2 = vec3f(0.0);
  for (var k = 0u; k < 4u; k++) {
    let wi = w[k];
    if (wi <= 0.0) {
      continue;
    }
    let bi = idx[k];
    let base = bi * 4u;
    let m0 = boneMatrices[base + 0u];
    let m1 = boneMatrices[base + 1u];
    let m2 = boneMatrices[base + 2u];
    let m3 = boneMatrices[base + 3u];
    let p4 = mat4MulVec4N(array<vec4f, 4>(m0, m1, m2, m3), vec4f(rp, 1.0));
    sp = sp + wi * p4;
    // Columns 0..2 carry the upper-3×3 (column-major: m0/m1/m2 are columns).
    col0 = col0 + wi * m0.xyz;
    col1 = col1 + wi * m1.xyz;
    col2 = col2 + wi * m2.xyz;
  }

  var outPos = sp.xyz;
  // Normal transform via inverse-transpose of the blended skin linear part.
  var nt = mat3InverseTranspose(col0, col1, col2);
  var outN = nt * rn;

  if (skinParams.applyWorld != 0u) {
    // H17 — force sp.w = 1.0 so the homogeneous divide in matrixWorld * sp
    // is exactly a point transform regardless of whether the blended weights
    // sum to slightly less than 1 (unnormalized skin weights).
    sp.w = 1.0;
    outPos = (skinParams.matrixWorld * sp).xyz;
    // Compose the world matrix's upper-3×3 inverse-transpose onto the normal.
    let w0 = skinParams.matrixWorld[0].xyz;
    let w1 = skinParams.matrixWorld[1].xyz;
    let w2 = skinParams.matrixWorld[2].xyz;
    let wnt = mat3InverseTranspose(w0, w1, w2);
    outN = wnt * outN;
  }

  let nlen = length(outN);
  let safeN = select(vec3f(0.0), outN / nlen, nlen > 1e-12);

  bvhPositions[outIdx] = vec4f(outPos, uvPack);
  // WS1 (2026-05-29) — write the skinned normal into the SHARED merged
  // bvh_normal buffer at the SAME world-space slot as the position
  // (baseVertex + vi), so the smooth-shading-normal blend in shade/ris/
  // risGi/risGiNrc reads the up-to-date skinned normal. (Previously this wrote
  // a mesh-local per-vertex slot — index vi — into a per-mesh buffer that
  // applyGpuSkinnedRefit dropped — the skinned normals were computed but never
  // consumed. Spelled without the bracket form here so the gpuSkinNormals guard
  // asserting no mesh-local indexed write stays comment-proof.)
  skinnedNormals[outIdx] = vec4f(safeN, uv1Pack);
}
`;
