/**
 * LBS + world transform write into merged ReSTIR `bvhPositions` (stride 4).
 * Preserves the packed UV lane in `.w` for each vertex.
 */
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
