/**
 * packBvhNodesForDebug — repack the engine's internal 32-byte BVH node
 * layout into the 8-float-per-node public-contract layout used by
 * `engine.debug.bvhNodes()` and consumed by `@vitrum/dev`'s overlays.
 *
 * Internal layout (32 bytes / node, 8 × u32):
 *   f32[0..2] bounds.min xyz
 *   f32[3..5] bounds.max xyz
 *   u32[6]    rightChildOrTriOffset
 *   u32[7]    splitAxisOrTriCount  (leaf when high 16 == 0xFFFF)
 *
 * Public-debug layout (8 × f32):
 *   [minX, minY, minZ, maxX, maxY, maxZ, depth, 0]
 *
 * `depth` is computed via iterative DFS from the root. Left child is
 * always idx+1 (depth-first encoding); right child is u32[6]. Leaves
 * are detected by `(split >>> 16) === 0xFFFF` — the `& 0xFFFF0000`
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
  const src    = new Float32Array(buf);
  const srcU32 = new Uint32Array(buf);
  const nodeCount = buf.byteLength / NODE_STRIDE_BYTES;
  const depths = new Uint32Array(nodeCount);

  if (nodeCount > 0) {
    // Worst-case linear chain → stack sized at nodeCount.
    const stack  = new Int32Array(nodeCount);
    const stackD = new Uint32Array(nodeCount);
    let sp = 0;
    stack[sp]  = 0;
    stackD[sp] = 0;
    sp++;

    while (sp > 0) {
      sp--;
      const idx = stack[sp]!;
      const d   = stackD[sp]!;
      if (idx < 0 || idx >= nodeCount) continue;
      depths[idx] = d;
      const splitWord = srcU32[idx * NODE_STRIDE_U32 + 7]!;
      const isLeaf = (splitWord >>> 16) === 0xffff;
      if (isLeaf) continue;
      const rightChild = srcU32[idx * NODE_STRIDE_U32 + 6]!;
      // Push right first so left (idx+1) pops first — preserves a
      // left-then-right walk for deterministic dev-overlay colors.
      stack[sp]  = rightChild;
      stackD[sp] = d + 1;
      sp++;
      stack[sp]  = idx + 1;
      stackD[sp] = d + 1;
      sp++;
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
