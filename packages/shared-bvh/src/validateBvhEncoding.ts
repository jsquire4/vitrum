import { BVH_NODE_FLOATS } from './strides.js';

/**
 * Verify that a packed BVH uses the relative-right-child encoding consumed by
 * vitrum WGSL traversal kernels.
 */
export function validateBvhEncoding(
  nodeBytes: Float32Array | Uint32Array,
  totalNodes: number,
): void {
  const uint32PerNode = BVH_NODE_FLOATS;
  const leafNodeFlag = 0xffff;
  const u32 =
    nodeBytes instanceof Uint32Array
      ? nodeBytes
      : new Uint32Array(nodeBytes.buffer, nodeBytes.byteOffset, nodeBytes.length);

  for (let i = 0; i < totalNodes; i += 1) {
    const base = i * uint32PerNode;
    const splitOrCount = u32[base + 7] ?? 0;
    const isLeaf = (splitOrCount >>> 16) === leafNodeFlag;
    if (isLeaf) continue;
    const offset = u32[base + 6] ?? 0;
    if (offset < 1 || offset >= totalNodes) {
      throw new Error(
        `[@vitrum/shared-bvh] validateBvhEncoding: interior node ${i} has invalid ` +
          `relative right-child offset ${offset} (must be in [1, ${totalNodes - 1}]). ` +
          `Check that the BVH was built with relative-offset encoding.`,
      );
    }
  }
}
