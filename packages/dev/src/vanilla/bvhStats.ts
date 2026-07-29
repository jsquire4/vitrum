// bvhStats.ts — BVH depth statistics + histogram renderer.
// Shared by the vanilla BVH overlay and react/BVHVisualizer.

import { makeDiv, makeMuted } from './domUtils.js';

/** `EngineDebugSurface.bvhNodes()` public layout, not the internal BVH words. */
const DEBUG_BVH_NODE_FLOATS = 8;
const DEBUG_BVH_DEPTH_SLOT = 6;

export interface BvhStats {
  readonly nodeCount: number;
  readonly maxDepth: number;
  readonly avgDepth: number;
  readonly histogram: ReadonlyArray<number>;
}

export function computeBvhStats(nodes: Float32Array): BvhStats {
  if (nodes.length % DEBUG_BVH_NODE_FLOATS !== 0) {
    throw new RangeError(
      'computeBvhStats: debug BVH table must contain exactly 8 floats per node.',
    );
  }
  const packedNodeCount = nodes.length / DEBUG_BVH_NODE_FLOATS;
  if (packedNodeCount === 0) {
    return { nodeCount: 0, maxDepth: 0, avgDepth: 0, histogram: [] };
  }

  let maxDepth = 0;
  let sumDepth = 0;
  const histogram: number[] = [];

  for (let nodeIndex = 0; nodeIndex < packedNodeCount; nodeIndex += 1) {
    const depth =
      nodes[nodeIndex * DEBUG_BVH_NODE_FLOATS + DEBUG_BVH_DEPTH_SLOT]!;
    if (
      !Number.isSafeInteger(depth) ||
      depth < 0 ||
      depth >= packedNodeCount
    ) {
      throw new RangeError(
        `computeBvhStats: node ${nodeIndex} has invalid public debug depth ${depth}.`,
      );
    }
    maxDepth = Math.max(maxDepth, depth);
    sumDepth += depth;
    while (histogram.length <= depth) histogram.push(0);
    histogram[depth] = (histogram[depth] ?? 0) + 1;
  }

  return {
    nodeCount: packedNodeCount,
    maxDepth,
    avgDepth: sumDepth / packedNodeCount,
    histogram,
  };
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
