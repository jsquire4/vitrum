/**
 * packBvhNodesForDebug — repack the engine's internal 32-byte BVH node
 * layout into the 8-float-per-node public-contract layout used by
 * `engine.debug.bvhNodes()` and consumed by `@vitrum/dev`'s overlays.
 *
 * Internal layout (32 bytes / node, 8 × u32):
 *   f32[0..2] bounds.min xyz
 *   f32[3..5] bounds.max xyz
 *   u32[6]    rightChildOffsetOrTriOffset
 *   u32[7]    splitAxisOrTriCount  (leaf when high 16 == 0xFFFF)
 *
 * Public-debug layout (8 × f32):
 *   [minX, minY, minZ, maxX, maxY, maxZ, depth, 0]
 *
 * `depth` is computed via iterative DFS over every tree in the concatenated
 * BLAS forest. Left child is always idx+1 (depth-first encoding); the right
 * child is `idx + u32[6]` because slot 6 stores a relative node offset.
 * Leaves are detected by `(split >>> 16) === 0xFFFF` — the `& 0xFFFF0000`
 * variant of this check looks correct but actually returns -65536
 * (signed int32) and fails the `=== 0xFFFF0000` (number literal stored
 * as double) comparison; the unsigned-upper-16 form is the safe one.
 *
 * Extracted from `HybridEngine.bvhNodes()` 2026-05-19 so the depth-pass
 * can be unit-tested without standing up a full engine — the prior in-
 * line implementation shipped with a load-bearing leaf-flag-check bug
 * that no existing test covered.
 */

const NODE_STRIDE_U32 = 8;
const NODE_STRIDE_BYTES = 32;

export function packBvhNodesForDebug(buf: ArrayBuffer): Float32Array {
  if (buf.byteLength % NODE_STRIDE_BYTES !== 0) {
    throw new RangeError(
      '[walkaround-hybrid] debug BVH node buffer is not 32-byte aligned.',
    );
  }
  const src    = new Float32Array(buf);
  const srcU32 = new Uint32Array(buf);
  const nodeCount = buf.byteLength / NODE_STRIDE_BYTES;
  const depths = new Uint32Array(nodeCount);

  if (nodeCount > 0) {
    // bvhNodes is a concatenated BLAS forest in TLAS mode. Every individual
    // depth-first tree occupies one contiguous interval, so the first node
    // after a completed interval is the next root.
    const state = new Uint8Array(nodeCount); // 0=unseen, 1=scheduled, 2=visited
    let root = 0;
    while (root < nodeCount) {
      const stack: Array<readonly [node: number, depth: number]> = [[root, 0]];
      state[root] = 1;
      let treeNodeCount = 0;
      let maxTreeNode = root;

      const schedule = (node: number, depth: number): void => {
        if (node < 0 || node >= nodeCount) {
          throw new Error(
            `[walkaround-hybrid] debug BVH child ${node} is outside ` +
              `[0, ${nodeCount - 1}].`,
          );
        }
        if (state[node] !== 0) {
          throw new Error(
            `[walkaround-hybrid] debug BVH node ${node} is reachable more ` +
              'than once (cycle, shared child, or overlapping BLAS roots).',
          );
        }
        state[node] = 1;
        maxTreeNode = Math.max(maxTreeNode, node);
        stack.push([node, depth]);
      };

      while (stack.length > 0) {
        const [idx, depth] = stack.pop()!;
        state[idx] = 2;
        treeNodeCount += 1;
        depths[idx] = depth;
        const base = idx * NODE_STRIDE_U32;
        const splitWord = srcU32[base + 7]!;
        const isLeaf = (splitWord >>> 16) === 0xffff;
        if (isLeaf) continue;
        if (splitWord > 2) {
          throw new Error(
            `[walkaround-hybrid] debug BVH interior node ${idx} has invalid ` +
              `split axis ${splitWord}.`,
          );
        }
        const rightOffset = srcU32[base + 6]!;
        const leftChild = idx + 1;
        const rightChild = idx + rightOffset;
        if (rightOffset <= 1) {
          throw new Error(
            `[walkaround-hybrid] debug BVH interior node ${idx} has invalid ` +
              `relative right-child offset ${rightOffset}.`,
          );
        }
        // Push right first so left (idx+1) pops first — preserves a
        // left-then-right walk for deterministic dev-overlay colors.
        schedule(rightChild, depth + 1);
        schedule(leftChild, depth + 1);
      }

      if (treeNodeCount !== maxTreeNode - root + 1) {
        throw new Error(
          `[walkaround-hybrid] debug BVH rooted at ${root} is not a ` +
            'contiguous depth-first tree.',
        );
      }
      root = maxTreeNode + 1;
    }
  }

  const out = new Float32Array(nodeCount * 8);
  for (let i = 0; i < nodeCount; i++) {
    const so = i * NODE_STRIDE_U32;
    const oo = i * 8;
    out[oo + 0] = src[so + 0]!;
    out[oo + 1] = src[so + 1]!;
    out[oo + 2] = src[so + 2]!;
    out[oo + 3] = src[so + 3]!;
    out[oo + 4] = src[so + 4]!;
    out[oo + 5] = src[so + 5]!;
    out[oo + 6] = depths[i]!;
    out[oo + 7] = 0;
  }
  return out;
}
