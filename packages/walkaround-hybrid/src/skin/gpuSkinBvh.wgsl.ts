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
 * derives the surface normal of a ray hit GEOMETRICALLY — `cross(e1, e2)` of
 * the (already skinned) triangle vertex positions (see
 * shared-bvh/bvhIntersect.wgsl `result.normal = side * normalize(n)`). It does
 * NOT read a per-vertex normal buffer during traversal. So the skinned per-
 * vertex normals this kernel writes are NOT consumed by the merged-BVH ray
 * normal today; they exist for parity with the CPU path and for future smooth-
 * normal / emitter-normal consumers. The kernel writes them only when a
 * `skinnedNormals` output buffer is bound (binding 7); when absent the host
 * compiles the position-only variant.
 */

/** Position-only LBS variant (no skinned-normal output). */
export const GPU_SKIN_BVH_WGSL = /* wgsl */ `
struct SkinBvhUniforms {
  vertexCount: u32,
  baseVertex: u32,
  /** 1 = merged world BVH (apply matrixWorld); 0 = TLAS local BLAS slice. */
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

fn mat4MulVec4(m: array<vec4f, 4>, v: vec4f) -> vec4f {
  return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let vi = gid.x;
  if (vi >= skinParams.vertexCount) {
    return;
  }
  let outIdx = skinParams.baseVertex + vi;
  let uvPack = bvhPositions[outIdx].w;
  let rp = restPositions[vi].xyz;
  let idx = skinIndices[vi];
  let w = skinWeights[vi];
  var sp = vec4f(0.0, 0.0, 0.0, 1.0);
  for (var k = 0u; k < 4u; k++) {
    let wi = w[k];
    if (wi <= 0.0) {
      continue;
    }
    let bi = idx[k];
    let base = bi * 4u;
    let p4 = mat4MulVec4(
      array<vec4f, 4>(
        boneMatrices[base + 0u],
        boneMatrices[base + 1u],
        boneMatrices[base + 2u],
        boneMatrices[base + 3u],
      ),
      vec4f(rp, 1.0),
    );
    sp = sp + wi * p4;
  }
  var outPos = sp.xyz;
  if (skinParams.applyWorld != 0u) {
    outPos = (skinParams.matrixWorld * sp).xyz;
  }
  bvhPositions[outIdx] = vec4f(outPos, uvPack);
}
`;

/**
 * Position + normal LBS variant. Identical position math to the position-only
 * kernel; additionally accumulates the blended skin matrix's upper-3×3, takes
 * its inverse-transpose, applies it to the rest normal, and writes the
 * normalized result into `skinnedNormals` (binding 7, stride-4 with `.w`
 * left 0). When `applyWorld != 0`, the world matrix's upper-3×3 inverse-
 * transpose is composed in (matching how positions are world-transformed).
 *
 * Indexing: positions write at `baseVertex + vi` into the SHARED merged
 * `bvhPositions` buffer; skinned normals write at `vi` (mesh-local) into a
 * PER-MESH `skinnedNormals` buffer the subsystem owns — so the normal output
 * needs no knowledge of the merged-buffer layout.
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
// Falls back to the input on a (near-)singular matrix so the result stays
// finite. Returns the inverse-transpose as a mat3x3f (columns).
fn mat3InverseTranspose(c0: vec3f, c1: vec3f, c2: vec3f) -> mat3x3f {
  // det via scalar triple product of the columns.
  let det = dot(c0, cross(c1, c2));
  if (abs(det) < 1e-20) {
    return mat3x3f(c0, c1, c2);
  }
  let invDet = 1.0 / det;
  // For M = [c0 c1 c2], (M^-1)^T columns are the cofactor columns / det:
  //   col0 = (c1 × c2)/det, col1 = (c2 × c0)/det, col2 = (c0 × c1)/det.
  return mat3x3f(
    cross(c1, c2) * invDet,
    cross(c2, c0) * invDet,
    cross(c0, c1) * invDet,
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
  let rp = restPositions[vi].xyz;
  let rn = restNormals[vi].xyz;
  let idx = skinIndices[vi];
  let w = skinWeights[vi];

  var sp = vec4f(0.0, 0.0, 0.0, 1.0);
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
    outPos = (skinParams.matrixWorld * sp).xyz;
    // Compose the world matrix's upper-3×3 inverse-transpose onto the normal.
    let w0 = skinParams.matrixWorld[0].xyz;
    let w1 = skinParams.matrixWorld[1].xyz;
    let w2 = skinParams.matrixWorld[2].xyz;
    let wnt = mat3InverseTranspose(w0, w1, w2);
    outN = wnt * outN;
  }

  let nlen = length(outN);
  let safeN = select(vec3f(0.0, 1.0, 0.0), outN / nlen, nlen > 1e-12);

  bvhPositions[outIdx] = vec4f(outPos, uvPack);
  // Normals are mesh-local (per-mesh buffer), indexed by vi, not outIdx.
  skinnedNormals[vi] = vec4f(safeN, 0.0);
}
`;
