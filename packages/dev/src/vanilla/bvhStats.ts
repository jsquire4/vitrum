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
  const nodeCount = Math.floor(nodes.length / BVH_NODE_FLOATS);
  if (nodeCount === 0) {
    return { nodeCount: 0, maxDepth: 0, avgDepth: 0, histogram: [] };
  }

  let maxDepth = 0;
  let sumDepth = 0;
  const histogram: number[] = [];
  for (let i = 0; i < nodeCount; i++) {
    const depth = Math.max(0, Math.floor(nodes[i * BVH_NODE_FLOATS + 6] ?? 0));
    maxDepth = Math.max(maxDepth, depth);
    sumDepth += depth;
    while (histogram.length <= depth) histogram.push(0);
    histogram[depth] = (histogram[depth] ?? 0) + 1;
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
