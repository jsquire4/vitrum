/**
 * splicePack.ts — BLAS-splice helpers extracted from scenePack.ts (D11.1).
 *
 * Contains the low-level BVH-node and index-buffer rebase kernels that are
 * shared by both the same-size splice (splicePrimitiveBlasIntoPack) and the
 * resize splice (spliceResizedPrimitiveBlasIntoPack). All symbols are also
 * re-exported from scenePack.ts for backward-compatibility.
 */

import { isLeafSplit } from './buildArrayBvh.js';

/**
 * Copy one 8-word BVH node from `src` (at `srcWordBase`) into `dst` (at
 * `dstWordBase`), adding `leafTriDelta` to word[6] iff the node is a LEAF.
 *
 * Word[6] of a leaf is a GLOBAL triangle offset; of an interior node it is a
 * RELATIVE child offset (which must NOT shift when the subtree moves rigidly).
 * Shared by both BLAS-splice paths — an off-by-one here silently corrupts BVH
 * traversal, so it lives in exactly one place.
 */
export function rebaseLeafTriOffset(
  dst: Uint32Array,
  dstWordBase: number,
  src: ArrayLike<number>,
  srcWordBase: number,
  leafTriDelta: number,
): void {
  const splitOrCount = src[srcWordBase + 7] ?? 0;
  const isLeaf = isLeafSplit(splitOrCount);
  dst[dstWordBase] = src[srcWordBase] ?? 0;
  dst[dstWordBase + 1] = src[srcWordBase + 1] ?? 0;
  dst[dstWordBase + 2] = src[srcWordBase + 2] ?? 0;
  dst[dstWordBase + 3] = src[srcWordBase + 3] ?? 0;
  dst[dstWordBase + 4] = src[srcWordBase + 4] ?? 0;
  dst[dstWordBase + 5] = src[srcWordBase + 5] ?? 0;
  dst[dstWordBase + 6] = isLeaf ? (src[srcWordBase + 6] ?? 0) + leafTriDelta : (src[srcWordBase + 6] ?? 0);
  dst[dstWordBase + 7] = splitOrCount;
}

/**
 * Copy `triCount` stride-4 (vec4u) index triangles from `src` (starting at
 * triangle `srcTri`) into `dst` (starting at triangle `dstTri`), shifting each
 * of the three GLOBAL vertex refs (.x.y.z) by `vertexDelta` and zeroing the .w
 * padding lane. Also copies the parallel per-triangle material id.
 *
 * This is the downstream-rebase inner loop of the resize splice — the one place
 * a wrong stride or delta corrupts which vertices a triangle references.
 */
export function copyVec4Strided(
  dstIndices: Uint32Array,
  dstTriMaterialIds: Uint32Array,
  srcIndices: Uint32Array,
  srcTriMaterialIds: Uint32Array,
  srcTri: number,
  dstTri: number,
  vertexDelta: number,
): void {
  for (let k = 0; k < 3; k += 1) {
    dstIndices[dstTri * 4 + k] = (srcIndices[srcTri * 4 + k] ?? 0) + vertexDelta;
  }
  dstIndices[dstTri * 4 + 3] = 0;
  dstTriMaterialIds[dstTri] = srcTriMaterialIds[srcTri] ?? 0;
}

/**
 * Copy a packed slice's local `indexWords` (vec4u-strided) into `dst` starting at
 * word `dstWordBase`, adding `vertexStart` to each of the three global vertex refs
 * (`.x.y.z`) while leaving the `.w` padding lane (`i % 4 === 3`) verbatim (D12-4).
 *
 * Shared by the same-size splice and the resize splice — both rebase the changed
 * primitive's new index words to its (unchanged) `vertexStart`.
 */
export function rebaseIndexWords(
  dst: Uint32Array,
  dstWordBase: number,
  indexWords: ArrayLike<number>,
  vertexStart: number,
): void {
  for (let i = 0; i < indexWords.length; i += 1) {
    const localIdx = indexWords[i] ?? 0;
    dst[dstWordBase + i] = i % 4 === 3 ? localIdx : localIdx + vertexStart;
  }
}

