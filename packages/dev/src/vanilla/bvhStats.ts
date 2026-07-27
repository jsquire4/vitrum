// bvhStats.ts — BVH depth statistics + histogram renderer.
// Shared by the vanilla BVH overlay and react/BVHVisualizer.

import { BVH_NODE_FLOATS } from '@vitrum/shared-bvh';
import { makeDiv, makeMuted } from './domUtils.js';

export interface BvhStats {
  readonly nodeCount: number;
  readonly maxDepth: number;
  readonly avgDepth: number;
  readonly histogram: ReadonlyArray<number>;
}

export function computeBvhStats(nodes: Float32Array): BvhStats {
  const packedNodeCount = Math.floor(nodes.length / BVH_NODE_FLOATS);
  if (packedNodeCount === 0) {
    return { nodeCount: 0, maxDepth: 0, avgDepth: 0, histogram: [] };
  }

  const words = new Uint32Array(
    nodes.buffer,
    nodes.byteOffset,
    packedNodeCount * BVH_NODE_FLOATS,
  );
  const visited = new Uint8Array(packedNodeCount);
  const pending: Array<readonly [nodeIndex: number, depth: number]> = [[0, 0]];
  let maxDepth = 0;
  let sumDepth = 0;
  let nodeCount = 0;
  const histogram: number[] = [];

  while (pending.length > 0) {
    const [nodeIndex, depth] = pending.pop()!;
    if (nodeIndex < 0 || nodeIndex >= packedNodeCount || visited[nodeIndex] !== 0) {
      continue;
    }

    visited[nodeIndex] = 1;
    nodeCount++;
    maxDepth = Math.max(maxDepth, depth);
    sumDepth += depth;
    while (histogram.length <= depth) histogram.push(0);
    histogram[depth] = (histogram[depth] ?? 0) + 1;

    const base = nodeIndex * BVH_NODE_FLOATS;
    const splitOrCount = words[base + 7] ?? 0;
    const isLeaf = (splitOrCount >>> 16) === 0xffff;
    if (isLeaf) continue;

    // Binary BVH layout: left child immediately follows its parent; slot 6
    // stores the right child's relative node offset. It is not a depth field.
    const rightOffset = words[base + 6] ?? 0;
    pending.push([nodeIndex + rightOffset, depth + 1]);
    pending.push([nodeIndex + 1, depth + 1]);
  }

  return { nodeCount, maxDepth, avgDepth: sumDepth / nodeCount, histogram };
}

export function renderBvhBars(target: HTMLElement, stats: BvhStats): void {
  target.replaceChildren();
  if (stats.histogram.length === 0) {
    target.append(makeMuted('empty'));
    return;
  }

  const maxCount = Math.max(...stats.histogram, 1);
  for (let depth = 0; depth < stats.histogram.length; depth++) {
    const count = stats.histogram[depth] ?? 0;
    const bar = makeDiv({
      width: '10px',
      minWidth: '4px',
      height: `${Math.max(2, Math.round((count / maxCount) * 36))}px`,
      background: `hsl(${(depth * 30) % 360}, 80%, 60%)`,
      opacity: count > 0 ? '1' : '0.25',
    });
    bar.title = `depth ${depth}: ${count}`;
    target.append(bar);
  }
}
