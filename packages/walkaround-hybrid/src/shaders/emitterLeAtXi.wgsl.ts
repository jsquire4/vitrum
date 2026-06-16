import type { WgslModule } from '../pipeline/wgslComposer.js';

export const EMITTER_LE_AT_XI_WGSL = /* wgsl */ `
fn emitterSampleBarycentricFromXi(xi: vec2f) -> vec3f {
  let s = sqrt(clamp(xi.x, 0.0, 1.0));
  return vec3f(1.0 - s, s * clamp(xi.y, 0.0, 1.0), s * (1.0 - clamp(xi.y, 0.0, 1.0)));
}

fn sampleEmitterLeAtXi(e: EmitterTri, xi: vec2f) -> vec3f {
  // sourceTriIndex is packed in the active render buffers' bvh_index/material
  // atlas triangle space. Merged mode uses the BVH-reordered triangle directly;
  // TLAS mode maps the world-expanded emitter triangle back to the local BLAS
  // atlas triangle before upload. Negative/unmapped values keep average Le.
  let sourceTri = i32(round(e.sourceTriIndex));
  if (sourceTri < 0) {
    return e.Le;
  }
  let triIndex = u32(sourceTri);
  if (triIndex >= arrayLength(&bvh_index)) {
    return e.Le;
  }
  let tri = bvh_index[triIndex].xyz;
  if (tri.x >= arrayLength(&bvh_position) || tri.y >= arrayLength(&bvh_position) || tri.z >= arrayLength(&bvh_position) ||
      tri.x >= arrayLength(&bvh_normal) || tri.y >= arrayLength(&bvh_normal) || tri.z >= arrayLength(&bvh_normal)) {
    return e.Le;
  }
  let bary = emitterSampleBarycentricFromXi(xi);
  let uv0a = materialAtlasPackedUvFromVec4(bvh_position[tri.x]);
  let uv0b = materialAtlasPackedUvFromVec4(bvh_position[tri.y]);
  let uv0c = materialAtlasPackedUvFromVec4(bvh_position[tri.z]);
  let uv1a = materialAtlasPackedUvFromVec4(bvh_normal[tri.x]);
  let uv1b = materialAtlasPackedUvFromVec4(bvh_normal[tri.y]);
  let uv1c = materialAtlasPackedUvFromVec4(bvh_normal[tri.z]);
  let uv0 = bary.x * uv0a + bary.y * uv0b + bary.z * uv0c;
  let uv1 = bary.x * uv1a + bary.y * uv1b + bary.z * uv1c;
  return sampleEmissiveMap(triIndex, uv0, uv1, e.Le);
}
`;

export const EMITTER_LE_AT_XI_MODULE: WgslModule = {
  name: 'emitterLeAtXi',
  source: EMITTER_LE_AT_XI_WGSL,
  requires: ['common', 'materialAtlas'],
};
