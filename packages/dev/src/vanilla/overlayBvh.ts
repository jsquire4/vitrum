// overlayBvh.ts — BVH structure panel overlay.

import type { DebuggableEngine } from '../types.js';
import { makePanel, makeTitle, makeRow, makeDiv } from './domUtils.js';
import { safeDebugCall } from './debugUtils.js';
import {
  computeBvhStats,
  renderBvhBars,
  type BvhStats,
} from './bvhStats.js';

export function addBvhDiagnostics(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
): void {
  const panel = makePanel({
    bottom: '8px',
    right: '8px',
    minWidth: '230px',
  });
  panel.append(makeTitle('BVH structure'));

  const apiRow = makeRow('api', '-');
  const nodesRow = makeRow('nodes', '-');
  const depthRow = makeRow('max depth', '-');
  const avgRow = makeRow('avg depth', '-');
  const histogram = makeDiv({
    display: 'flex',
    alignItems: 'end',
    gap: '1px',
    height: '42px',
    marginTop: '6px',
    borderTop: '1px solid rgba(255,255,255,0.16)',
    paddingTop: '4px',
  });
  panel.append(apiRow.el, nodesRow.el, depthRow.el, avgRow.el, histogram);
  add(panel);

  const render = (): void => {
    const result = safeDebugCall<BvhStats>(
      typeof engine.debug?.bvhNodes === 'function'
        ? () => {
            const nodes = engine.debug?.bvhNodes?.();
            return nodes == null ? null : computeBvhStats(nodes);
          }
        : undefined,
    );
    apiRow.setValue(result.status === 'unsupported' ? 'unavailable' : result.status);
    if (result.status !== 'ready' || result.value == null) {
      nodesRow.setValue(result.status === 'ready' ? 'not built' : '-');
      depthRow.setValue('-');
      avgRow.setValue('-');
      histogram.replaceChildren(document.createTextNode('no node table'));
      return;
    }

    const stats = result.value;
    nodesRow.setValue(String(stats.nodeCount));
    depthRow.setValue(String(stats.maxDepth));
    avgRow.setValue(stats.avgDepth.toFixed(2));
    renderBvhBars(histogram, stats);
  };

  render();
  const interval = setInterval(render, 500);
  cleanupFns.push(() => clearInterval(interval));
}
