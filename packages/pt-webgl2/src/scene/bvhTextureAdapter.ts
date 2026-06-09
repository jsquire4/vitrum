import type { ScenePackResult, WorldSpaceMergeResult } from '@vitrum/shared-bvh';

/**
 * BVH texture-packing adapter — the inverse of three-mesh-bvh's
 * `MeshBVHUniformStruct.bvhToTextures`, operating on `@vitrum/shared-bvh`'s flat
 * arrays instead of a THREE `MeshBVH`. This is what lets us drop `three-mesh-bvh`
 * (and its transitive THREE) while keeping the fork's `BVHShaderGLSL` traversal.
 *
 * shared-bvh's 32-byte node is byte-identical to three-mesh-bvh's (verified:
 * buildArrayBvh.ts:13-27 + bvhIntersect.wgsl.ts:180-181), so this is a re-strider,
 * NOT a node-format translation:
 *   f32[0..2] boundsMin   f32[3..5] boundsMax
 *   u32[6] = interior: RELATIVE right-child offset | leaf: absolute tri offset
 *   u32[7] = interior: split axis (0/1/2)          | leaf: 0xFFFF0000 | triCount
 *
 * The fork GLSL reads 4 data textures (bvh_struct_definitions.glsl:16-24):
 *   usampler2D index;   sampler2D position;   sampler2D bvhBounds;   usampler2D bvhContents;
 * all fetched row-major via uTexelFetch1D(uv.x = i % width, uv.y = i / width), NEAREST.
 */

export const BVH_LEAF_FLAG = 0xffff0000;

/** ceil(sqrt(n)) — the square dimension that holds `n` texels row-major. */
export function squareDim(texelCount: number): number {
  return Math.max(1, Math.ceil(Math.sqrt(Math.max(1, texelCount))));
}

export interface BvhTextureData {
  /** RGBA32F, 2 texels/node: texel 2i = boundsMin.xyz (a=0), texel 2i+1 = boundsMax.xyz (a=0). */
  readonly bounds: Float32Array;
  readonly boundsDim: number;
  /** RGBA32UI, 1 texel/node: .x = u32[7] (split|0xFFFF0000|count), .y = u32[6] (rel-right|tri-offset). */
  readonly contents: Uint32Array;
  readonly contentsDim: number;
  /** RGBA32F, 1 texel/vertex: .xyz = position, .w = 1.0 (three-mesh-bvh texelFetch1D(...).rgb semantics). */
  readonly position: Float32Array;
  readonly positionDim: number;
  /** RGBA32UI, 1 texel/tri: .xyz = the 3 GLOBAL vertex indices (.w unused). */
  readonly index: Uint32Array;
  readonly indexDim: number;
  /** R32UI as RGBA32UI .x, 1 texel/tri: per-triangle material id. */
  readonly materialIndex: Uint32Array;
  readonly materialIndexDim: number;
  readonly nodeCount: number;
  readonly vertexCount: number;
  readonly triangleCount: number;
}

export type BvhTexturePackSource = ScenePackResult | WorldSpaceMergeResult;

function isWorldSpaceMergeResult(pack: BvhTexturePackSource): pack is WorldSpaceMergeResult {
  return 'positionStrideFloats' in pack;
}

function assertScenePackHasSingleBlas(pack: ScenePackResult): void {
  if (pack.triangleCount === 0) return;
  if (pack.primitiveTlasBindings.length !== 1) {
    throw new Error(
      'pt-webgl2: packBvhTextureData requires a single-root BVH. ' +
        'Use mergeWorldSpaceFromCore(scene, { positionStride: 4 }) for multi-primitive scenes.',
    );
  }
}

/**
 * PURE: re-stride shared-bvh's flat arrays into the 4 BVH data-texture payloads.
 * No GL — directly unit-testable (CPU-traverse the output vs a brute-force oracle).
 */
export function packBvhTextureData(pack: BvhTexturePackSource): BvhTextureData {
  const merged = isWorldSpaceMergeResult(pack);
  if (!merged) assertScenePackHasSingleBlas(pack);

  const nodeF32 = pack.bvhNodes;
  const nodeU32 = new Uint32Array(nodeF32.buffer, nodeF32.byteOffset, nodeF32.length);
  const nodeCount = nodeF32.length / 8;
  const positionStride = merged ? pack.positionStrideFloats : 4;
  const indexStride = merged ? pack.bvhIndexStride : 4;
  const vertexCount = merged ? pack.vertexCount : pack.positions.length / 4;
  const triangleCount = pack.triangleCount;
  const triMaterialIds = merged ? pack.triMaterialId : pack.triMaterialIds;

  // bvhBounds — RGBA32F, 2 texels/node
  const boundsDim = squareDim(nodeCount * 2);
  const bounds = new Float32Array(boundsDim * boundsDim * 4);
  for (let i = 0; i < nodeCount; i += 1) {
    const n = i * 8;
    const bMin = 2 * i * 4;
    const bMax = (2 * i + 1) * 4;
    bounds[bMin] = nodeF32[n]!; bounds[bMin + 1] = nodeF32[n + 1]!; bounds[bMin + 2] = nodeF32[n + 2]!;
    bounds[bMax] = nodeF32[n + 3]!; bounds[bMax + 1] = nodeF32[n + 4]!; bounds[bMax + 2] = nodeF32[n + 5]!;
  }

  // bvhContents — RGBA32UI, 1 texel/node. .x = u32[7], .y = u32[6] (shared-bvh's leaf word
  // 0xFFFF0000|count is already exactly what the GLSL leaf test reads; relative right offset
  // works unchanged because the GLSL does currNode + boundsInfo.y).
  const contentsDim = squareDim(nodeCount);
  const contents = new Uint32Array(contentsDim * contentsDim * 4);
  for (let i = 0; i < nodeCount; i += 1) {
    const n = i * 8;
    contents[i * 4] = nodeU32[n + 7]!;
    contents[i * 4 + 1] = nodeU32[n + 6]!;
  }

  // position — RGBA32F, 1 texel/vertex, .w forced to 1.0 (shared-bvh ships .w=0)
  const positionDim = squareDim(vertexCount);
  const position = new Float32Array(positionDim * positionDim * 4);
  for (let v = 0; v < vertexCount; v += 1) {
    const src = v * positionStride;
    const dst = v * 4;
    position[dst] = pack.positions[src]!;
    position[dst + 1] = pack.positions[src + 1]!;
    position[dst + 2] = pack.positions[src + 2]!;
    position[dst + 3] = 1.0;
  }

  // index — RGBA32UI, 1 texel/tri, .xyz = global vertex indices
  const indexDim = squareDim(triangleCount);
  const index = new Uint32Array(indexDim * indexDim * 4);
  for (let t = 0; t < triangleCount; t += 1) {
    const src = t * indexStride;
    const dst = t * 4;
    index[dst] = pack.indices[src]!;
    index[dst + 1] = pack.indices[src + 1]!;
    index[dst + 2] = pack.indices[src + 2]!;
  }

  // materialIndex — RGBA32UI .x, PER-VERTEX. The fork GLSL reads
  // `uTexelFetch1D(materialIndexAttribute, surfaceHit.faceIndices.x).r` — indexed by a
  // VERTEX index, not a triangle index (it mirrors three-mesh-bvh's per-vertex
  // UIntVertexAttributeTexture). So assign each vertex its triangle's material id.
  const materialIndexDim = squareDim(vertexCount);
  const materialIndex = new Uint32Array(materialIndexDim * materialIndexDim * 4);
  for (let t = 0; t < triangleCount; t += 1) {
    const m = triMaterialIds[t]!;
    const src = t * indexStride;
    materialIndex[pack.indices[src]! * 4] = m;
    materialIndex[pack.indices[src + 1]! * 4] = m;
    materialIndex[pack.indices[src + 2]! * 4] = m;
  }

  return {
    bounds, boundsDim, contents, contentsDim, position, positionDim,
    index, indexDim, materialIndex, materialIndexDim,
    nodeCount, vertexCount, triangleCount,
  };
}

export interface BvhTextures {
  readonly bounds: WebGLTexture;
  readonly contents: WebGLTexture;
  readonly position: WebGLTexture;
  readonly index: WebGLTexture;
  readonly materialIndex: WebGLTexture;
  destroy(): void;
}

/** Upload the packed data to GL data textures (RGBA32F / RGBA32UI, NEAREST, ClampToEdge). */
export function uploadBvhTextures(gl: WebGL2RenderingContext, d: BvhTextureData): BvhTextures {
  const fTex = (dim: number, data: Float32Array): WebGLTexture =>
    makeTex(gl, dim, gl.RGBA32F, gl.RGBA, gl.FLOAT, data);
  const uTex = (dim: number, data: Uint32Array): WebGLTexture =>
    makeTex(gl, dim, gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data);
  const bounds = fTex(d.boundsDim, d.bounds);
  const contents = uTex(d.contentsDim, d.contents);
  const position = fTex(d.positionDim, d.position);
  const index = uTex(d.indexDim, d.index);
  const materialIndex = uTex(d.materialIndexDim, d.materialIndex);
  return {
    bounds, contents, position, index, materialIndex,
    destroy() { for (const t of [bounds, contents, position, index, materialIndex]) gl.deleteTexture(t); },
  };
}

function makeTex(
  gl: WebGL2RenderingContext,
  dim: number,
  internalFormat: number,
  format: number,
  type: number,
  data: ArrayBufferView,
): WebGLTexture {
  const tex = gl.createTexture();
  if (tex == null) throw new Error('pt-webgl2: failed to create BVH texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, dim, dim, 0, format, type, data as ArrayBufferView);
  return tex;
}
