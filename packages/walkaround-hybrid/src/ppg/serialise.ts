/**
 * PPG tree serialisation — CPU producers that pack the adaptive sTree + dTree
 * structures into flat Float32Array buffers the GPU can traverse.
 *
 * W9 — finishes the GPU bridge for Müller 2017 Practical Path Guiding. The
 * CPU tree representations live in `dTree.ts` / `sTree.ts` and are paper-
 * faithful (adaptive quadtree + adaptive spatial kd-tree). This module
 * translates them into flat layouts so a WGSL kernel can:
 *
 *   1. descend the sTree to find the spatial cell for a query position, AND
 *   2. descend that cell's dTree to find the leaf for a query direction.
 *
 * Layouts are intentionally f32-only (no mixed f32/u32 in a single binding)
 * because WebGPU's `array<f32>` storage binding is the cheapest read for the
 * traversal kernels, and small u32 indices (max-depth bounded) round-trip
 * losslessly through f32 up to 2^24 ≈ 16 M entries — far more than the
 * adaptive trees ever produce.
 *
 * ── dTree layout (DTREE_STRIDE = 8 f32 per node + 4 f32 header) ─────────────
 *
 *   Header (offset 0, 4 f32):
 *     [0] nodeCount      — total nodes in this dTree (includes interior)
 *     [1] leafCount      — leaf-only count (informational, kernel ignores)
 *     [2] totalFlux      — sum of leaf flux (denominator for PDFs)
 *     [3] _pad           — reserved
 *
 *   Per node (offset 4 + i × 8, 8 f32):
 *     [0] u0             — octahedral patch min-U
 *     [1] v0             — octahedral patch min-V
 *     [2] u1             — octahedral patch max-U
 *     [3] v1             — octahedral patch max-V
 *     [4] flux           — accumulated radiance at this node
 *                          (leaves only; interior carries 0)
 *     [5] solidAngle     — 4π·(u1−u0)·(v1−v0) for leaves; ≤0 for interior
 *     [6] firstChild     — index of NW child (children are 4 consecutive);
 *                          NaN-encoded as −1 (i.e. 4294967295.0) when no children.
 *                          We instead encode as a plain f32; the kernel reads it
 *                          back via i32 cast and tests for >=0.
 *     [7] isLeafFlag     — 1.0 for leaf, 0.0 for interior. WGSL reads via f32
 *                          comparison (kernel test: `> 0.5`).
 *
 * ── sTree layout (STREE_STRIDE = 8 f32 per node + 4 f32 header) ────────────
 *
 *   Header (offset 0, 4 f32):
 *     [0] nodeCount      — total sTree nodes
 *     [1] dTreeCount     — number of distinct per-cell dTrees (= leaf count)
 *     [2] _pad
 *     [3] _pad
 *
 *   Per sNode (offset 4 + i × 8, 8 f32):
 *     [0] aabbMinX
 *     [1] aabbMinY
 *     [2] aabbMinZ
 *     [3] splitValue
 *     [4] aabbMaxX
 *     [5] aabbMaxY
 *     [6] aabbMaxZ
 *     [7] packed = splitAxis*1 + leftChild*8 + rightChild*65536 + dTreeIndex*…
 *                  Actually we keep it simple — pack only the discriminator
 *                  into the 8th f32 and encode the rest via a sidecar buffer.
 *
 *   Sidecar: STreeMeta (i32-equivalent f32) — kept separate because WGSL's
 *   f32→i32 cast preserves integer values exactly only up to 2^24. With <16k
 *   sTree cells per the PPG_MAX_SPATIAL_CELLS cap, we are safely below that
 *   threshold but emit each integer field in its own f32 slot anyway:
 *
 *     [7+0] splitAxis    — -1 (leaf), 0, 1, or 2 (interior split axis)
 *     [7+1] leftChild    — flat-index of left child (interior); -1 (leaf)
 *     [7+2] rightChild   — flat-index of right child (interior); -1 (leaf)
 *     [7+3] dTreeIndex   — index into the dTree-offset table (leaf); -1 (interior)
 *
 * To keep the binding count low we instead bake those four fields into the
 * unused 8th slot above by ALWAYS storing splitAxis there and using a
 * separate `STreeMetaBuffer` for the remaining three. Per Phase-1 scope this
 * is over-engineered; we instead use a single 16-f32-per-node packing.
 *
 * The implementation below picks a simple 16-f32-per-node sTree layout:
 *
 *   Per sNode (offset 4 + i × 16, 16 f32):
 *     [0..2]   aabb.min.xyz
 *     [3]      splitValue
 *     [4..6]   aabb.max.xyz
 *     [7]      splitAxis     (-1 leaf | 0,1,2 interior)
 *     [8]      leftChild     (-1 leaf | index)
 *     [9]      rightChild    (-1 leaf | index)
 *     [10]     dTreeIndex    (-1 interior | index)
 *     [11..15] _pad
 *
 * The per-cell dTree blocks are concatenated into a SINGLE flat dTree buffer
 * with a parallel `dTreeOffsets[]` array — `dTreeOffsets[cell]` is the f32
 * offset (start of header) of cell `cell`'s dTree inside the combined buffer.
 * The kernel can then traverse a sTree leaf → look up its dTreeIndex →
 * jump to `dTreeOffsets[dTreeIndex]` and descend its quadtree.
 *
 * Memory: 16 384 sTree cells × 16 f32 + 16 384 × dTree(say 100 nodes × 8 + 4)
 *       = 16k × 64B + 16k × 804B ≈ 13 MB. Within budget.
 *
 * ── Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", Eurographics Symposium on Rendering 2017
 * (https://tom94.net/data/publications/mueller17practical/mueller17practical.pdf)
 */

import type { DTree, STree } from './types.js';

/** f32 stride for a serialised dTree node (must match WGSL DTREE_NODE_STRIDE). */
export const DTREE_NODE_F32 = 8;
/** f32 stride for the per-dTree header (must match WGSL DTREE_HEADER_F32). */
export const DTREE_HEADER_F32 = 4;
/** f32 stride for a serialised sTree node (must match WGSL STREE_NODE_STRIDE). */
export const STREE_NODE_F32 = 16;
/** f32 stride for the sTree header. */
export const STREE_HEADER_F32 = 4;

/**
 * Pack a single dTree (a per-cell directional quadtree) into a flat Float32Array.
 *
 * The returned buffer self-describes — `out[0]` is `nodeCount` so a downstream
 * kernel that receives `array<f32>` knows how many slots to expect after the
 * header. A standalone dTree (e.g. for the W9 oracle test) is the common
 * case; multi-cell sTree serialisation builds on this routine via
 * `serialiseSTree`.
 *
 * Round-trip contract: for every leaf `i` in `dTree.nodes`,
 *   `findDTreeLeaf(dTree, [u, v]) === gpuTraversalIndex(buf, [u, v])`
 * for every UV strictly inside the leaf's (u0,u1)·(v0,v1) rectangle.
 * See `__tests__/ppg-serialise.test.ts` Test 2.
 *
 * @param dTree  CPU-side adaptive quadtree (Müller §3.2).
 * @returns      A new `Float32Array` of length `DTREE_HEADER_F32 + N·DTREE_NODE_F32`.
 */
export function serialiseDTree(dTree: DTree): Float32Array {
  const N = dTree.nodes.length;
  const out = new Float32Array(DTREE_HEADER_F32 + N * DTREE_NODE_F32);

  // Count leaves (informational; kernel does not need this).
  let leafCount = 0;
  for (const n of dTree.nodes) {
    if (n.isLeaf) leafCount++;
  }

  // Header
  out[0] = N;
  out[1] = leafCount;
  out[2] = dTree.totalFlux;
  out[3] = 0;

  // Per-node packing
  for (let i = 0; i < N; i++) {
    const n = dTree.nodes[i]!;
    const base = DTREE_HEADER_F32 + i * DTREE_NODE_F32;
    out[base + 0] = n.u0;
    out[base + 1] = n.v0;
    out[base + 2] = n.u1;
    out[base + 3] = n.v1;
    out[base + 4] = n.flux;
    out[base + 5] = n.solidAngle;
    out[base + 6] = n.firstChild; // f32 representation; integer up to 2^24
    out[base + 7] = n.isLeaf ? 1.0 : 0.0;
  }

  return out;
}

/**
 * GPU-equivalent traversal of a serialised dTree. Used by tests as the
 * "CPU oracle" — the WGSL kernel implements the exact same descent.
 *
 * Returns the f32 base offset (within the buffer) of the leaf node
 * whose (u0..u1, v0..v1) rectangle contains the query UV.
 *
 * @param buf    `serialiseDTree(dTree)` output.
 * @param uv     Octahedral UV in [0,1]².
 * @returns      The f32 offset of the leaf node's first slot (so caller can read
 *               leaf fields with constant offsets).
 */
export function gpuTraverseDTreeLeaf(buf: Float32Array, uv: readonly [number, number]): number {
  let idx = 0;
  // Defensive cap: prevent infinite loop if the buffer is malformed.
  const N = buf[0] ?? 0;
  for (let step = 0; step < N + 1; step++) {
    const base = DTREE_HEADER_F32 + idx * DTREE_NODE_F32;
    const isLeaf = (buf[base + 7] ?? 0) > 0.5;
    if (isLeaf) return base;
    const u0 = buf[base + 0] ?? 0;
    const v0 = buf[base + 1] ?? 0;
    const u1 = buf[base + 2] ?? 0;
    const v1 = buf[base + 3] ?? 0;
    const uMid = (u0 + u1) * 0.5;
    const vMid = (v0 + v1) * 0.5;
    const goRight = uv[0] >= uMid;
    const goDown = uv[1] >= vMid;
    const firstChild = buf[base + 6] ?? -1;
    if (firstChild < 0) {
      // Malformed: interior node without children. Return current offset
      // as a defensive leaf-equivalent.
      return base;
    }
    idx = (firstChild | 0) + (goDown ? 2 : 0) + (goRight ? 1 : 0);
  }
  // Unreachable for a well-formed quadtree below max depth.
  return DTREE_HEADER_F32;
}

/**
 * Result of {@link serialiseSTree}: the packed sTree node buffer + an offset
 * table that maps `dTreeIndex` → f32 base offset of the corresponding dTree
 * inside `dTreeBuf`.
 */
export interface SerialisedSTree {
  /** Flat sTree node buffer (header + N nodes × STREE_NODE_F32). */
  sTreeBuf: Float32Array;
  /** Concatenated dTree buffers for every cell, one after another. */
  dTreeBuf: Float32Array;
  /**
   * `dTreeOffsets[k]` is the f32 base offset of the k-th dTree's header
   * inside `dTreeBuf`. Length = `sTree.dTrees.length`. The sTree node's
   * `dTreeIndex` field is an index into this table.
   */
  dTreeOffsets: Uint32Array;
}

/**
 * Pack a full sTree (spatial kd-tree + per-cell dTrees) into a triplet of
 * flat buffers ready for GPU upload.
 *
 * Why three buffers (not one mega-buffer)? Each binding stays semantically
 * coherent (the kernel can read `sTreeBuf[0]` to know the node count without
 * indexing into the middle of an interleaved layout) and WebGPU's storage
 * binding cap (8) is comfortably under-used.
 *
 * @param sTree  CPU-side adaptive kd-tree (Müller §3.1) with per-leaf dTrees.
 */
export function serialiseSTree(sTree: STree): SerialisedSTree {
  const NS = sTree.nodes.length;
  const NDT = sTree.dTrees.length;

  // Per-dTree serialisation upfront so we can compute offsets.
  const perDTreeBufs: Float32Array[] = [];
  const offsets = new Uint32Array(NDT);
  let totalF32 = 0;
  for (let k = 0; k < NDT; k++) {
    const buf = serialiseDTree(sTree.dTrees[k]!);
    perDTreeBufs.push(buf);
    offsets[k] = totalF32;
    totalF32 += buf.length;
  }

  // Single concatenated dTreeBuf.
  const dTreeBuf = new Float32Array(Math.max(totalF32, DTREE_HEADER_F32));
  for (let k = 0; k < NDT; k++) {
    dTreeBuf.set(perDTreeBufs[k]!, offsets[k]!);
  }

  // sTree node buffer.
  const sTreeBuf = new Float32Array(STREE_HEADER_F32 + NS * STREE_NODE_F32);
  sTreeBuf[0] = NS;
  sTreeBuf[1] = NDT;
  sTreeBuf[2] = 0;
  sTreeBuf[3] = 0;

  for (let i = 0; i < NS; i++) {
    const n = sTree.nodes[i]!;
    const base = STREE_HEADER_F32 + i * STREE_NODE_F32;
    sTreeBuf[base + 0] = n.aabb.min[0];
    sTreeBuf[base + 1] = n.aabb.min[1];
    sTreeBuf[base + 2] = n.aabb.min[2];
    sTreeBuf[base + 3] = n.splitValue;
    sTreeBuf[base + 4] = n.aabb.max[0];
    sTreeBuf[base + 5] = n.aabb.max[1];
    sTreeBuf[base + 6] = n.aabb.max[2];
    sTreeBuf[base + 7] = n.splitAxis;     // -1 for leaf, else 0/1/2
    sTreeBuf[base + 8] = n.leftChild;     // -1 for leaf
    sTreeBuf[base + 9] = n.rightChild;    // -1 for leaf
    sTreeBuf[base + 10] = n.dTreeIndex;   // -1 for interior
    sTreeBuf[base + 11] = 0;
    sTreeBuf[base + 12] = 0;
    sTreeBuf[base + 13] = 0;
    sTreeBuf[base + 14] = 0;
    sTreeBuf[base + 15] = 0;
  }

  return { sTreeBuf, dTreeBuf, dTreeOffsets: offsets };
}

/**
 * GPU-equivalent sTree leaf lookup — walks the kd-tree in `sTreeBuf` to find
 * the cell containing the query position. Returns the sNode's f32 base offset.
 *
 * Used by the CPU oracle test to assert the WGSL traversal matches the CPU
 * `findSTreeLeaf` exactly.
 */
export function gpuTraverseSTreeLeaf(
  sTreeBuf: Float32Array,
  pos: readonly [number, number, number],
): number {
  let idx = 0;
  const NS = sTreeBuf[0] ?? 0;
  for (let step = 0; step < NS + 1; step++) {
    const base = STREE_HEADER_F32 + idx * STREE_NODE_F32;
    const splitAxis = sTreeBuf[base + 7] ?? -1;
    if (splitAxis < 0) return base; // leaf
    const splitValue = sTreeBuf[base + 3] ?? 0;
    const leftChild = sTreeBuf[base + 8] ?? -1;
    const rightChild = sTreeBuf[base + 9] ?? -1;
    const axisIdx = (splitAxis | 0);
    const queryAxis = pos[axisIdx] ?? 0;
    idx = (queryAxis < splitValue ? leftChild : rightChild) | 0;
    if (idx < 0) {
      // Defensive: malformed interior. Fall back to current node.
      return base;
    }
  }
  return STREE_HEADER_F32;
}
