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
 * ── sTree layout (STREE_NODE_F32 = 16 f32 per node + 4 f32 header) ──────────
 *
 *   Header (offset 0, 4 f32):
 *     [0] nodeCount      — total sTree nodes
 *     [1] dTreeCount     — number of distinct per-cell dTrees (= leaf count)
 *     [2] _pad
 *     [3] _pad
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
 * All integer fields (splitAxis/leftChild/rightChild/dTreeIndex) round-trip
 * losslessly through f32 — with the PPG_MAX_SPATIAL_CELLS ≈ 16k cap we are far
 * below the 2^24 exact-integer limit — so a single f32-only binding suffices.
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

import type { DTree, DTreeNode, STree, STreeNode } from './types.js';

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
 * Overflow clamp (Item A — host-config safety): when `maxNodes` is supplied
 * and the dTree has MORE nodes than the GPU buffer can hold per cell
 * (`allocatePPGResources({ maxDTreeNodesPerCell: < 341 })`), only the first
 * `maxNodes` nodes are serialised. Crucially, this must NOT leave a dangling
 * `firstChild` pointer inside the served region — the WGSL traversal
 * (`ppgPdf.wgsl.ts`) descends via `firstChild` with no `>= nodeCount` bound
 * and the per-cell dTree blocks are concatenated contiguously, so a child
 * index past the served prefix would read into the NEXT cell's data. We
 * therefore PROMOTE any served interior node whose four children fall outside
 * `[0, maxNodes)` to a leaf (clear its `firstChild` to −1 and set the leaf
 * flag), so the served tree terminates safely within its own block. This
 * matches the GPU UPDATE path, which already clamps the per-cell node range to
 * `maxDTreeNodesPerCell` (PPGCoordinator `_mergeFluxAndRefine`). The DEFAULT
 * config (`maxDTreeNodesPerCell = 341 = full depth-4 quadtree`) never triggers
 * the clamp because `refineDTree`'s `PPG_DTREE_MAX_DEPTH = 4` bounds every
 * dTree to ≤ 341 nodes — so `maxNodes` omitted (or ≥ node count) is exactly
 * the historical behaviour.
 *
 * @param dTree     CPU-side adaptive quadtree (Müller §3.2).
 * @param maxNodes  Optional hard cap on the number of nodes serialised
 *                  (= GPU `maxDTreeNodesPerCell`). Omitted ⇒ no clamp.
 * @returns         A new `Float32Array` of length
 *                  `DTREE_HEADER_F32 + min(N, maxNodes)·DTREE_NODE_F32`.
 */
export function serialiseDTree(dTree: DTree, maxNodes?: number): Float32Array {
  const N = maxNodes !== undefined
    ? Math.min(dTree.nodes.length, Math.max(0, Math.floor(maxNodes)))
    : dTree.nodes.length;
  const out = new Float32Array(DTREE_HEADER_F32 + N * DTREE_NODE_F32);

  // Count leaves AMONG THE SERVED PREFIX (informational; kernel does not need
  // this). A node promoted to a leaf by the clamp (see below) counts as a leaf.
  let leafCount = 0;

  // Per-node packing (clamped to the served prefix).
  for (let i = 0; i < N; i++) {
    const n = dTree.nodes[i]!;
    const base = DTREE_HEADER_F32 + i * DTREE_NODE_F32;
    // Overflow-safe child handling: if any of this node's four consecutive
    // children would land outside the served `[0, N)` region, the truncated
    // buffer cannot represent the subtree — serve this node AS A LEAF so the
    // contiguous-block GPU traversal terminates inside its own dTree.
    const childrenOutOfRange =
      !n.isLeaf && (n.firstChild < 0 || n.firstChild + 3 >= N);
    const servedAsLeaf = n.isLeaf || childrenOutOfRange;
    if (servedAsLeaf) leafCount++;
    out[base + 0] = n.u0;
    out[base + 1] = n.v0;
    out[base + 2] = n.u1;
    out[base + 3] = n.v1;
    out[base + 4] = n.flux;
    out[base + 5] = n.solidAngle;
    // firstChild: −1 when served as a leaf (clamped or genuine), else the
    // (in-range) child index. f32 representation; integer up to 2^24.
    out[base + 6] = servedAsLeaf ? -1 : n.firstChild;
    out[base + 7] = servedAsLeaf ? 1.0 : 0.0;
  }

  // Header
  out[0] = N;
  out[1] = leafCount;
  out[2] = dTree.totalFlux;
  out[3] = 0;

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
 * Overflow clamp (Item A): pass `maxDTreeNodesPerCell` (the GPU
 * `allocatePPGResources` cap) so each per-cell dTree is clamped to at most that
 * many nodes — keeping the concatenated `dTreeBuf` within the GPU buffer
 * allocation even when a host sets a sub-341 cap. Omitted ⇒ no clamp (default
 * 341-node-per-cell config never overflows; see {@link serialiseDTree}).
 *
 * @param sTree                 CPU-side adaptive kd-tree (Müller §3.1) with
 *                              per-leaf dTrees.
 * @param maxDTreeNodesPerCell  Optional per-cell node cap (= GPU buffer slot
 *                              stride). Omitted ⇒ no clamp.
 */
export function serialiseSTree(sTree: STree, maxDTreeNodesPerCell?: number): SerialisedSTree {
  const NS = sTree.nodes.length;
  const NDT = sTree.dTrees.length;

  // Per-dTree serialisation upfront so we can compute offsets.
  const perDTreeBufs: Float32Array[] = [];
  const offsets = new Uint32Array(NDT);
  let totalF32 = 0;
  for (let k = 0; k < NDT; k++) {
    const buf = serialiseDTree(sTree.dTrees[k]!, maxDTreeNodesPerCell);
    perDTreeBufs.push(buf);
    offsets[k] = totalF32;
    totalF32 += buf.length;
  }

  // Single concatenated dTreeBuf.
  const dTreeBuf = new Float32Array(Math.max(totalF32, DTREE_HEADER_F32));
  for (let k = 0; k < NDT; k++) {
    dTreeBuf.set(perDTreeBufs[k]!, offsets[k]);
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
 * Reconstruct a CPU-side `DTree` from a flat `Float32Array` produced by
 * {@link serialiseDTree}.
 *
 * Used by {@link deserialiseSTree} to restore the per-cell dTrees when
 * importing a PPG snapshot. The inverse faithfully recovers every field that
 * was packed by `serialiseDTree`; `depth` is not stored in the flat layout so
 * it is recovered from the tree topology (root=0, child=parent+1 by BFS level).
 *
 * `depth` is informational for the CPU; the GPU traversal never reads it. We
 * derive it with a single BFS pass over the recovered nodes array.
 */
export function deserialiseDTree(buf: Float32Array): DTree {
  const nodeCount = Math.floor(buf[0] ?? 0);
  const totalFlux = buf[2] ?? 0;
  const nodes: DTreeNode[] = [];
  for (let i = 0; i < nodeCount; i++) {
    const base = DTREE_HEADER_F32 + i * DTREE_NODE_F32;
    const isLeafFlag = (buf[base + 7] ?? 0) > 0.5;
    const firstChildRaw = buf[base + 6] ?? -1;
    nodes.push({
      isLeaf: isLeafFlag,
      u0: buf[base + 0] ?? 0,
      v0: buf[base + 1] ?? 0,
      u1: buf[base + 2] ?? 1,
      v1: buf[base + 3] ?? 1,
      flux: buf[base + 4] ?? 0,
      solidAngle: buf[base + 5] ?? 0,
      // firstChild stored as f32; -1 sentinel for leaves (serialiseDTree
      // writes −1.0 for leaves / clamped interior).
      firstChild: isLeafFlag ? -1 : Math.round(firstChildRaw),
      depth: 0, // filled in the BFS pass below
    });
  }
  // BFS pass to recover depths (root = 0; children = parent.depth + 1).
  if (nodes.length > 0) {
    nodes[0]!.depth = 0;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i]!;
      if (!n.isLeaf && n.firstChild >= 0) {
        const childDepth = n.depth + 1;
        for (let ci = 0; ci < 4; ci++) {
          const cidx = n.firstChild + ci;
          if (cidx < nodes.length) nodes[cidx]!.depth = childDepth;
        }
      }
    }
  }
  return { nodes, totalFlux };
}

/**
 * Reconstruct a CPU-side `STree` from the flat buffers produced by
 * {@link serialiseSTree}.
 *
 * Round-trip contract: `deserialiseSTree(serialiseSTree(sTree))` produces an
 * `STree` whose `serialiseSTree` output is byte-identical to the original (the
 * serialised form is the canonical wire format; f32 precision is the only loss
 * and matches the GPU's view of the tree). See the round-trip test in
 * `giStateSnapshot.test.ts`.
 *
 * `sceneBounds` must be supplied separately (it is not stored inside the node
 * buffers — the buffers only carry per-node AABBs) and is preserved verbatim
 * in the returned `STree`.
 *
 * @param s             Output of {@link serialiseSTree} (GPU-ready buffers).
 * @param sceneBounds   World-space AABB of the whole scene (from the snapshot).
 */
export function deserialiseSTree(
  s: SerialisedSTree,
  sceneBounds: { min: readonly [number, number, number]; max: readonly [number, number, number] },
): STree {
  const { sTreeBuf, dTreeBuf, dTreeOffsets } = s;
  const NS = Math.floor(sTreeBuf[0] ?? 0);
  const NDT = Math.floor(sTreeBuf[1] ?? 0);

  // Recover sTree nodes.
  const nodes: STreeNode[] = [];
  for (let i = 0; i < NS; i++) {
    const base = STREE_HEADER_F32 + i * STREE_NODE_F32;
    const splitAxisRaw = Math.round(sTreeBuf[base + 7] ?? -1);
    const splitAxis: 0 | 1 | 2 | -1 =
      splitAxisRaw === 0 ? 0 : splitAxisRaw === 1 ? 1 : splitAxisRaw === 2 ? 2 : -1;
    nodes.push({
      aabb: {
        min: [sTreeBuf[base + 0] ?? 0, sTreeBuf[base + 1] ?? 0, sTreeBuf[base + 2] ?? 0],
        max: [sTreeBuf[base + 4] ?? 0, sTreeBuf[base + 5] ?? 0, sTreeBuf[base + 6] ?? 0],
      },
      splitAxis,
      splitValue: sTreeBuf[base + 3] ?? 0,
      leftChild: Math.round(sTreeBuf[base + 8] ?? -1),
      rightChild: Math.round(sTreeBuf[base + 9] ?? -1),
      dTreeIndex: Math.round(sTreeBuf[base + 10] ?? -1),
      sampleCount: 0, // accumulators are volatile; start fresh on restore
    });
  }

  // Recover per-cell dTrees using their start offsets.
  const dTrees: DTree[] = [];
  for (let k = 0; k < NDT; k++) {
    const off = dTreeOffsets[k] ?? 0;
    const nodeCount = Math.floor(dTreeBuf[off] ?? 0);
    const slice = dTreeBuf.subarray(off, off + DTREE_HEADER_F32 + nodeCount * DTREE_NODE_F32);
    dTrees.push(deserialiseDTree(new Float32Array(slice)));
  }

  return {
    nodes,
    dTrees,
    sceneBounds: {
      min: [sceneBounds.min[0], sceneBounds.min[1], sceneBounds.min[2]],
      max: [sceneBounds.max[0], sceneBounds.max[1], sceneBounds.max[2]],
    },
  };
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
