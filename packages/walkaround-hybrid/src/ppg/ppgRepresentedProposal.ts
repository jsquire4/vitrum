import {
  REPRESENTED_PROPOSAL_BUCKET_COUNT,
  buildRepresentedDistributionF32,
} from '@vitrum/shared-samplers';
import {
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  type OwnedSerialisedSTree,
  type SerialisedSTree,
} from './serialise.js';

/**
 * Build the immutable GPU query view of a canonical PPG snapshot.
 *
 * The persistent/snapshot dTree ABI keeps lane 5 as leaf solid angle and lane
 * 6 as `firstChild` (`-1` for leaves). The query view overlays only otherwise
 * unused lanes:
 *
 * - interior lane 5: exact represented bucket count for the subtree;
 * - leaf lane 6: exact represented bucket count for the leaf.
 *
 * Counts are integers in [0, 2^24], so every value round-trips through f32
 * exactly. The update kernel checks the leaf flag before reading lane 6 and
 * never reads interior solid angle, making the overlay compatible with its
 * topology traversal. Snapshot bytes remain unchanged and can be rebuilt into
 * this view at every upload/import.
 */
export function buildPpgRepresentedQueryView(
  source: SerialisedSTree,
): OwnedSerialisedSTree {
  if (!(source.sTreeBuf instanceof Float32Array)) {
    throw new TypeError('PPG represented query view requires a Float32Array sTree buffer.');
  }
  if (!(source.dTreeBuf instanceof Float32Array)) {
    throw new TypeError('PPG represented query view requires a Float32Array dTree buffer.');
  }
  if (!(source.dTreeOffsets instanceof Uint32Array)) {
    throw new TypeError('PPG represented query view requires a Uint32Array dTree-offset buffer.');
  }

  const sTreeBuf = new Float32Array(source.sTreeBuf);
  const dTreeBuf = new Float32Array(source.dTreeBuf);
  const dTreeOffsets = new Uint32Array(source.dTreeOffsets);

  for (let treeIndex = 0; treeIndex < dTreeOffsets.length; treeIndex += 1) {
    overlayDTreeBuckets(dTreeBuf, dTreeOffsets[treeIndex]!, treeIndex);
  }

  return { sTreeBuf, dTreeBuf, dTreeOffsets };
}
function requireExactF32Index(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 2 ** 24) {
    throw new RangeError(`${label} must be an exact non-negative f32 integer.`);
  }
  return value;
}

function overlayDTreeBuckets(
  buffer: Float32Array,
  offset: number,
  treeIndex: number,
): void {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset + DTREE_HEADER_F32 > buffer.length) {
    throw new RangeError(`PPG dTree ${treeIndex} has an invalid query-view offset.`);
  }
  const nodeCount = requireExactF32Index(
    buffer[offset]!,
    `PPG dTree ${treeIndex} node count`,
  );
  if (nodeCount < 1) {
    throw new RangeError(`PPG dTree ${treeIndex} must contain a root node.`);
  }
  const blockEnd = offset + DTREE_HEADER_F32 + nodeCount * DTREE_NODE_F32;
  if (!Number.isSafeInteger(blockEnd) || blockEnd > buffer.length) {
    throw new RangeError(`PPG dTree ${treeIndex} query-view block is truncated.`);
  }

  const visited = new Uint8Array(nodeCount);
  const leafIndices: number[] = [];
  const stack = [0];
  while (stack.length > 0) {
    const nodeIndex = stack.pop()!;
    if (nodeIndex < 0 || nodeIndex >= nodeCount || visited[nodeIndex] !== 0) {
      throw new RangeError(
        `PPG dTree ${treeIndex} query-view topology is cyclic, shared, or out of range.`,
      );
    }
    visited[nodeIndex] = 1;
    const base = offset + DTREE_HEADER_F32 + nodeIndex * DTREE_NODE_F32;
    const leafFlag = buffer[base + 7]!;
    const flux = buffer[base + 4]!;
    if (!Number.isFinite(flux) || flux < 0) {
      throw new RangeError(`PPG dTree ${treeIndex} node ${nodeIndex} has invalid flux.`);
    }
    if (leafFlag === 1) {
      leafIndices.push(nodeIndex);
      continue;
    }
    if (leafFlag !== 0) {
      throw new RangeError(`PPG dTree ${treeIndex} node ${nodeIndex} has an invalid leaf flag.`);
    }
    const firstChild = requireExactF32Index(
      buffer[base + 6]!,
      `PPG dTree ${treeIndex} node ${nodeIndex} first child`,
    );
    if (firstChild <= nodeIndex || firstChild + 3 >= nodeCount) {
      throw new RangeError(`PPG dTree ${treeIndex} node ${nodeIndex} has invalid children.`);
    }
    for (let child = 3; child >= 0; child -= 1) stack.push(firstChild + child);
  }
  if (visited.some((state) => state === 0)) {
    throw new RangeError(`PPG dTree ${treeIndex} query-view topology has unreachable nodes.`);
  }

  // Node-index order is stable across CPU serialization, snapshot restore, and
  // GPU upload. It therefore provides the deterministic Hamilton tie-break.
  leafIndices.sort((a, b) => a - b);
  const leafWeights = leafIndices.map((nodeIndex) =>
    buffer[offset + DTREE_HEADER_F32 + nodeIndex * DTREE_NODE_F32 + 4]!
  );
  const represented = buildRepresentedDistributionF32(leafWeights);
  const nodeBuckets = new Uint32Array(nodeCount);
  for (let ordinal = 0; ordinal < leafIndices.length; ordinal += 1) {
    nodeBuckets[leafIndices[ordinal]!] = represented.bucketCounts[ordinal]!;
  }

  // Serialized children always follow their parent. Reverse index order gives
  // a bottom-up subtree sum without another traversal or floating arithmetic.
  for (let nodeIndex = nodeCount - 1; nodeIndex >= 0; nodeIndex -= 1) {
    const base = offset + DTREE_HEADER_F32 + nodeIndex * DTREE_NODE_F32;
    if (buffer[base + 7] === 1) {
      // Lane 6 is unused by a leaf in the query kernels.
      buffer[base + 6] = nodeBuckets[nodeIndex]!;
      continue;
    }
    const firstChild = buffer[base + 6]!;
    let subtreeBuckets = 0;
    for (let child = 0; child < 4; child += 1) {
      subtreeBuckets += nodeBuckets[firstChild + child]!;
    }
    if (subtreeBuckets > REPRESENTED_PROPOSAL_BUCKET_COUNT) {
      throw new RangeError(`PPG dTree ${treeIndex} represented bucket sum overflowed.`);
    }
    nodeBuckets[nodeIndex] = subtreeBuckets;
    // Lane 5 is unused by an interior node in the query kernels.
    buffer[base + 5] = subtreeBuckets;
  }

  const expectedRootBuckets = leafWeights.some((weight) => weight > 0)
    ? REPRESENTED_PROPOSAL_BUCKET_COUNT
    : 0;
  if (nodeBuckets[0] !== expectedRootBuckets) {
    throw new Error(
      `PPG dTree ${treeIndex} represented ${nodeBuckets[0]} root buckets; ` +
        `expected ${expectedRootBuckets}.`,
    );
  }
}
