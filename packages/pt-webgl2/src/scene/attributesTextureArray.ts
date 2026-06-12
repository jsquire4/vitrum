// attributesTextureArray.ts — the 5-layer vertex-attribute sampler2DArray payload
// (plan/three-removal/03-scene-bvh-packers.md §7).
//
// Port of the fork's `AttributesTextureArray.js` (verified layer order
// `AttributesTextureArray.js:5-33`): a `sampler2DArray`, RGBA32F, NEAREST, with
//   layer 0 = normal, layer 1 = tangent, layer 2 = uv0, layer 3 = color,
//   layer 4 = uv1 (second UV channel — TextureRef.texCoord 1).
// The fork GLSL reads `texelFetch1D(attributesArray, layer, index)`
// (`texture_sample_functions.glsl.js:5-21`) — one near-square `dim×dim` slab per
// layer holding `vertexCount` texels row-major, vec3 attributes promoted to RGBA
// with `.a = 0`.
//
// ── Source of the per-vertex attributes ─────────────────────────────────────
//
// `mergeWorldSpaceFromCore` (the THREE-free merged world-space stream) emits
// `positions` + `normals` (both stride `positionStrideFloats`) but — verified by
// reading `WorldSpaceMergeResult` (`shared-bvh/src/worldSpaceMerge.ts:93-157`) —
// it does NOT emit `uvs` OR vertex `colors`. So:
//   • layer 0 (normal)  ← merged.normals          (the merged world-space normal).
//   • layer 1 (tangent) ← DERIVED here from positions + uvs (standard per-triangle
//                         accumulate → orthonormalize-against-normal) when no
//                         authored tangent stream is supplied by the caller.
//   • layer 2 (uv)      ← merged UVs (uv0) in `.xy` when present; (0,0,0,0) otherwise.
//   • layer 3 (color)   ← per-vertex colors when present; (1,1,1,1) otherwise.
//   • layer 4 (uv1)     ← per-vertex uv1 passed from the scene primitives (via the
//                         caller-built mergedUv1 array, stride 2). Falls back to
//                         uv0 (merged.uvs) per vertex when the source primitive
//                         carried no uv1 — so the layer is always valid.
//
// The merged result is the PREFERRED source (it is the same world-space stream the
// BVH textures index, so vertex indices line up 1:1). The plan's §7 fallback to
// `packSceneFromCore` for normals/uvs is unnecessary for normals (the merge ships
// them) and would not help for uvs at the merged-stream granularity — documented
// here rather than wired, to keep the attribute layers index-aligned with the BVH
// `position`/`index` textures.

import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { squareDim } from './bvhTextureAdapter.js';
import type { LayeredTexelGrid } from './sceneTextures.js';

/** Layer assignment in the 5-layer attribute array (fork-verified order + uv1 extension). */
export const ATTR_LAYER_NORMAL = 0;
export const ATTR_LAYER_TANGENT = 1;
export const ATTR_LAYER_UV = 2;
export const ATTR_LAYER_COLOR = 3;
export const ATTR_LAYER_UV1 = 4;
export const ATTR_LAYER_COUNT = 5;

/**
 * A `WorldSpaceMergeResult` that additionally carries an optional per-vertex
 * `colors` array and an optional `uv1` array (stride 2, same vertex order as
 * `merged.uvs`). When `uv1` is absent, layer 4 (ATTR_UV1) falls back to `uvs`
 * (uv0) per vertex. The caller is responsible for building `uv1` from the source
 * primitives aligned with the merged vertex order (see `buildMergedUv1` in
 * `uploadSceneTextures.ts`).
 */
interface MergeWithOptionalAttrs extends WorldSpaceMergeResult {
  readonly colors?: Float32Array;
  /** Per-vertex tangent stream, xyzw stride 4, same vertex order as merged.positions. */
  readonly tangents?: Float32Array;
  /** Per-vertex uv1, stride 2 (same vertex order as `merged.uvs`). Optional;
   *  when absent layer 4 (ATTR_UV1) copies uv0 per vertex. */
  readonly uv1?: Float32Array;
}

/** A flat readonly accessor returning component `c` (0..3) of vertex `v` from a
 *  stride-`stride` array, defaulting to `fallback` when the source is absent. */
function vComp(
  src: Float32Array | undefined,
  v: number,
  c: number,
  stride: number,
  fallback: number,
): number {
  if (src === undefined) return fallback;
  const value = src[v * stride + c];
  return value === undefined ? fallback : value;
}

/**
 * Pack the merged world-space stream into the 5-layer RGBA32F attribute array
 * (normal / tangent / uv0 / color / uv1). Each layer is a `dim×dim` slab, `dim =
 * ceil(sqrt(vertexCount))`; the returned `data` is `dim*dim*4*5` floats (5 layers
 * × dim×dim texels × RGBA).
 *
 * Tangents use authored per-vertex xyzw data when `merged.tangents` is present.
 * Missing/malformed tangents are derived from positions + uvs by the standard
 * accumulate-per-triangle method (Lengyel) and orthonormalized against the
 * per-vertex normal (Gram-Schmidt); when the UV gradient is degenerate the tangent
 * falls back to an arbitrary orthonormal basis built from the normal alone.
 *
 * Layer 4 (uv1): filled from `merged.uv1` when present; falls back to `merged.uvs`
 * (uv0) per vertex when absent, so the layer is always valid.
 */
export function packAttributesArray(merged: MergeWithOptionalAttrs): LayeredTexelGrid {
  const stride = merged.positionStrideFloats;
  const uvStride = 2; // merged.uvs is stride-2 (2D texture coords), NOT the position stride
  const vertexCount = merged.vertexCount;
  const positions = merged.positions;
  const normals = merged.normals;
  const uvs = merged.uvs;
  const uv1 = merged.uv1; // optional — stride 2, same vertex order as uvs
  const colors = merged.colors;
  const colorStride = colors === undefined
    ? 4
    : Math.max(3, Math.min(4, Math.floor(colors.length / Math.max(1, vertexCount))));
  const authoredTangents = merged.tangents;

  // BVH-reordered triangle indices are stride-3 (`indices` / `bvhIndexStride`),
  // referencing the merged (un-reordered) vertices — exactly the vertices the
  // attribute layers describe. Tangent accumulation walks these triangles.
  const indices = merged.indices;
  const triCount = merged.triangleCount;

  const dim = squareDim(vertexCount);
  const texelsPerLayer = dim * dim;
  const floatsPerLayer = texelsPerLayer * 4;
  const data = new Float32Array(floatsPerLayer * ATTR_LAYER_COUNT);

  const normalBase = ATTR_LAYER_NORMAL * floatsPerLayer;
  const tangentBase = ATTR_LAYER_TANGENT * floatsPerLayer;
  const uvBase = ATTR_LAYER_UV * floatsPerLayer;
  const colorBase = ATTR_LAYER_COLOR * floatsPerLayer;
  const uv1Base = ATTR_LAYER_UV1 * floatsPerLayer;

  // ── layer 0: normal (vec3 → RGBA, .a = 0); layer 2: uv0 (uv0 in .xy);
  //    layer 3: color (default white); layer 4: uv1 (uv1 in .xy, fallback uv0) ─
  for (let v = 0; v < vertexCount; v += 1) {
    const o = v * 4;
    data[normalBase + o] = vComp(normals, v, 0, stride, 0);
    data[normalBase + o + 1] = vComp(normals, v, 1, stride, 1);
    data[normalBase + o + 2] = vComp(normals, v, 2, stride, 0);
    data[normalBase + o + 3] = 0;

    data[uvBase + o] = vComp(uvs, v, 0, uvStride, 0);
    data[uvBase + o + 1] = vComp(uvs, v, 1, uvStride, 0);
    data[uvBase + o + 2] = 0;
    data[uvBase + o + 3] = 0;

    data[colorBase + o] = vComp(colors, v, 0, colorStride, 1);
    data[colorBase + o + 1] = vComp(colors, v, 1, colorStride, 1);
    data[colorBase + o + 2] = vComp(colors, v, 2, colorStride, 1);
    data[colorBase + o + 3] = colors === undefined || colorStride < 4 ? 1 : vComp(colors, v, 3, colorStride, 1);

    // layer 4: uv1 — use the caller-supplied uv1 stream; fall back to uv0 per vertex
    // when absent so the layer is always fully populated and the GLSL select is safe.
    if (uv1 !== undefined) {
      data[uv1Base + o] = vComp(uv1, v, 0, uvStride, 0);
      data[uv1Base + o + 1] = vComp(uv1, v, 1, uvStride, 0);
    } else {
      data[uv1Base + o] = vComp(uvs, v, 0, uvStride, 0);
      data[uv1Base + o + 1] = vComp(uvs, v, 1, uvStride, 0);
    }
    data[uv1Base + o + 2] = 0;
    data[uv1Base + o + 3] = 0;
  }

  // ── layer 1: tangent — derive per-vertex tangents ─────────────────────────
  // Accumulate the Lengyel UV-gradient tangent over each triangle, then per
  // vertex orthonormalize against the normal. Degenerate UVs (no merge UVs) leave
  // the accumulator ~zero, and we fall back to an orthonormal basis tangent.
  const tanAccum = new Float32Array(vertexCount * 3);
  const bitanAccum = new Float32Array(vertexCount * 3);
  const hasUvs = uvs !== undefined;
  if (hasUvs && triCount > 0) {
    for (let t = 0; t < triCount; t += 1) {
      const i0 = indices[t * 3] ?? 0;
      const i1 = indices[t * 3 + 1] ?? 0;
      const i2 = indices[t * 3 + 2] ?? 0;

      const p0x = positions[i0 * stride] ?? 0;
      const p0y = positions[i0 * stride + 1] ?? 0;
      const p0z = positions[i0 * stride + 2] ?? 0;
      const e1x = (positions[i1 * stride] ?? 0) - p0x;
      const e1y = (positions[i1 * stride + 1] ?? 0) - p0y;
      const e1z = (positions[i1 * stride + 2] ?? 0) - p0z;
      const e2x = (positions[i2 * stride] ?? 0) - p0x;
      const e2y = (positions[i2 * stride + 1] ?? 0) - p0y;
      const e2z = (positions[i2 * stride + 2] ?? 0) - p0z;

      const u0 = uvs[i0 * uvStride] ?? 0;
      const w0 = uvs[i0 * uvStride + 1] ?? 0;
      const du1 = (uvs[i1 * uvStride] ?? 0) - u0;
      const dw1 = (uvs[i1 * uvStride + 1] ?? 0) - w0;
      const du2 = (uvs[i2 * uvStride] ?? 0) - u0;
      const dw2 = (uvs[i2 * uvStride + 1] ?? 0) - w0;

      const denom = du1 * dw2 - du2 * dw1;
      if (Math.abs(denom) < 1e-12) continue;
      const r = 1 / denom;
      const tx = (dw2 * e1x - dw1 * e2x) * r;
      const ty = (dw2 * e1y - dw1 * e2y) * r;
      const tz = (dw2 * e1z - dw1 * e2z) * r;
      const bx = (du1 * e2x - du2 * e1x) * r;
      const by = (du1 * e2y - du2 * e1y) * r;
      const bz = (du1 * e2z - du2 * e1z) * r;

      for (const vi of [i0, i1, i2]) {
        tanAccum[vi * 3] = (tanAccum[vi * 3] ?? 0) + tx;
        tanAccum[vi * 3 + 1] = (tanAccum[vi * 3 + 1] ?? 0) + ty;
        tanAccum[vi * 3 + 2] = (tanAccum[vi * 3 + 2] ?? 0) + tz;
        bitanAccum[vi * 3] = (bitanAccum[vi * 3] ?? 0) + bx;
        bitanAccum[vi * 3 + 1] = (bitanAccum[vi * 3 + 1] ?? 0) + by;
        bitanAccum[vi * 3 + 2] = (bitanAccum[vi * 3 + 2] ?? 0) + bz;
      }
    }
  }

  for (let v = 0; v < vertexCount; v += 1) {
    const nx = vComp(normals, v, 0, stride, 0);
    const ny = vComp(normals, v, 1, stride, 1);
    const nz = vComp(normals, v, 2, stride, 0);

    let tx = vComp(authoredTangents, v, 0, 4, 0);
    let ty = vComp(authoredTangents, v, 1, 4, 0);
    let tz = vComp(authoredTangents, v, 2, 4, 0);
    let handedness = vComp(authoredTangents, v, 3, 4, 1) < 0 ? -1 : 1;
    if (Math.sqrt(tx * tx + ty * ty + tz * tz) < 1e-8) {
      tx = tanAccum[v * 3] ?? 0;
      ty = tanAccum[v * 3 + 1] ?? 0;
      tz = tanAccum[v * 3 + 2] ?? 0;
      const bx = bitanAccum[v * 3] ?? 0;
      const by = bitanAccum[v * 3 + 1] ?? 0;
      const bz = bitanAccum[v * 3 + 2] ?? 0;
      const cx = ny * tz - nz * ty;
      const cy = nz * tx - nx * tz;
      const cz = nx * ty - ny * tx;
      const bitangentDot = cx * bx + cy * by + cz * bz;
      handedness = bitangentDot < 0 ? -1 : 1;
    }

    // Gram-Schmidt: t' = normalize(t - n·(n·t))
    const ndt = nx * tx + ny * ty + nz * tz;
    tx -= nx * ndt;
    ty -= ny * ndt;
    tz -= nz * ndt;
    let len = Math.sqrt(tx * tx + ty * ty + tz * tz);

    if (len < 1e-8) {
      // No usable UV gradient → arbitrary orthonormal basis tangent from the
      // normal (Frisvad's branchless basis). Guarantees |t|=1, t·n≈0.
      const sign = nz >= 0 ? 1 : -1;
      const a = -1 / (sign + nz);
      const b = nx * ny * a;
      tx = 1 + sign * nx * nx * a;
      ty = sign * b;
      tz = -sign * nx;
      len = Math.sqrt(tx * tx + ty * ty + tz * tz) || 1;
    }

    const o = v * 4;
    data[tangentBase + o] = tx / len;
    data[tangentBase + o + 1] = ty / len;
    data[tangentBase + o + 2] = tz / len;
    data[tangentBase + o + 3] = handedness;
  }

  return { data, dim, layers: ATTR_LAYER_COUNT };
}
