/** Linear blend skinning compute (PR-7) — one invocation per vertex. */
export const GPU_SKIN_LBS_WGSL = /* wgsl */ `
struct SkinUniforms {
  vertexCount: u32,
  boneCount: u32,
  _pad: vec2u,
};

@group(0) @binding(0) var<uniform> skinParams: SkinUniforms;
@group(0) @binding(1) var<storage, read> restPositions: array<vec4f>;
@group(0) @binding(2) var<storage, read> restNormals: array<vec4f>;
@group(0) @binding(3) var<storage, read> skinIndices: array<vec4u>;
@group(0) @binding(4) var<storage, read> skinWeights: array<vec4f>;
@group(0) @binding(5) var<storage, read> boneMatrices: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> outPositions: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> outNormals: array<vec4f>;

fn mat4MulVec4(m: array<vec4f, 4>, v: vec4f) -> vec4f {
  return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let vi = gid.x;
  if (vi >= skinParams.vertexCount) {
    return;
  }
  let rp = restPositions[vi].xyz;
  let rn = restNormals[vi].xyz;
  let idx = skinIndices[vi];
  let w = skinWeights[vi];
  var sp = vec3f(0.0);
  var sn = vec3f(0.0);
  for (var k = 0u; k < 4u; k++) {
    let wi = w[k];
    if (wi <= 0.0) {
      continue;
    }
    let bi = idx[k];
    let base = bi * 4u;
    let p4 = mat4MulVec4(array<vec4f, 4>(
      boneMatrices[base + 0u],
      boneMatrices[base + 1u],
      boneMatrices[base + 2u],
      boneMatrices[base + 3u],
    ), vec4f(rp, 1.0));
    let n4 = mat4MulVec4(array<vec4f, 4>(
      boneMatrices[base + 0u],
      boneMatrices[base + 1u],
      boneMatrices[base + 2u],
      boneMatrices[base + 3u],
    ), vec4f(rn, 0.0));
    sp = sp + wi * p4.xyz;
    sn = sn + wi * n4.xyz;
  }
  outPositions[vi] = vec4f(sp, 1.0);
  outNormals[vi] = vec4f(normalize(sn + vec3f(1e-8)), 0.0);
}
`;
