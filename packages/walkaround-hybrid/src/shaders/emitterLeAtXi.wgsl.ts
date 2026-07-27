import type { WgslModule } from '../pipeline/wgslComposer.js';

export const EMITTER_LE_AT_XI_WGSL = /* wgsl */ `
fn emitterSampleBarycentricFromXi(xi: vec2f) -> vec3f {
  let s = sqrt(clamp(xi.x, 0.0, 1.0));
  return vec3f(1.0 - s, s * clamp(xi.y, 0.0, 1.0), s * (1.0 - clamp(xi.y, 0.0, 1.0)));
}

fn emitterSubdivWeightAt(i: u32, j: u32, level: u32) -> vec3f {
  let invLevel = 1.0 / f32(max(level, 1u));
  let u = f32(i) * invLevel;
  let v = f32(j) * invLevel;
  return vec3f(1.0 - u - v, u, v);
}

fn emitterParentBarycentricFromXi(e: EmitterTri, xi: vec2f) -> vec3f {
  let localBary = emitterSampleBarycentricFromXi(xi);
  let level = min(16u, max(1u, u32(round(max(e.sourceSubdivLevel, 1.0)))));
  if (level <= 1u) {
    return localBary;
  }

  let ordinal = u32(round(max(e.sourceSubdivOrdinal, 0.0)));
  var cursor = 0u;
  for (var i = 0u; i < level; i = i + 1u) {
    for (var j = 0u; j < level - i; j = j + 1u) {
      let a = emitterSubdivWeightAt(i, j, level);
      let b = emitterSubdivWeightAt(i + 1u, j, level);
      let c = emitterSubdivWeightAt(i, j + 1u, level);
      if (cursor == ordinal) {
        return localBary.x * a + localBary.y * b + localBary.z * c;
      }
      cursor = cursor + 1u;

      if (i + j < level - 1u) {
        let d = emitterSubdivWeightAt(i + 1u, j + 1u, level);
        if (cursor == ordinal) {
          return localBary.x * b + localBary.y * d + localBary.z * c;
        }
        cursor = cursor + 1u;
      }
    }
  }

  return localBary;
}

fn sampleEmitterLeAtXi(e: EmitterTri, xi: vec2f) -> vec3f {
  // sourceTriIndex is packed in the active render buffers' bvh_index/material
  // atlas triangle space. Merged mode uses the BVH-reordered triangle directly;
  // TLAS mode maps the world-expanded emitter triangle back to the local BLAS
  // atlas triangle before upload. sourceSubdivLevel/sourceSubdivOrdinal map
  // micro-emitter samples back to parent-triangle UV barycentrics. Encodings:
  // -1 = average-Le fallback; -(tri + 2) = mirrored TLAS instance, source tri
  // with reversed barycentrics.
  let encodedSourceTri = i32(round(e.sourceTriIndex));
  if (encodedSourceTri == -1) {
    return e.Le;
  }
  let mirroredSourceTri = encodedSourceTri < -1;
  let sourceTri = select(encodedSourceTri, -encodedSourceTri - 2, mirroredSourceTri);
  let triIndex = u32(sourceTri);
  if (triIndex >= bvhIndexCount()) {
    return e.Le;
  }
  let tri = bvhLoadIndex(triIndex).xyz;
  if (tri.x >= bvhPositionCount() || tri.y >= bvhPositionCount() || tri.z >= bvhPositionCount() ||
      tri.x >= sceneBvhNormalCount() || tri.y >= sceneBvhNormalCount() || tri.z >= sceneBvhNormalCount()) {
    return e.Le;
  }
  var bary = emitterParentBarycentricFromXi(e, xi);
  if (mirroredSourceTri) {
    bary = vec3f(bary.z, bary.y, bary.x);
  }
  let uv0a = materialAtlasPackedUvFromVec4(bvhLoadPosition(tri.x));
  let uv0b = materialAtlasPackedUvFromVec4(bvhLoadPosition(tri.y));
  let uv0c = materialAtlasPackedUvFromVec4(bvhLoadPosition(tri.z));
  let uv1a = materialAtlasPackedUvFromVec4(sceneLoadBvhNormal(tri.x));
  let uv1b = materialAtlasPackedUvFromVec4(sceneLoadBvhNormal(tri.y));
  let uv1c = materialAtlasPackedUvFromVec4(sceneLoadBvhNormal(tri.z));
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
