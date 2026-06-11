// overlayFrameTime.ts — FrameTimeHUD overlay.

import type { FrameStats } from '../types.js';
import type { DebuggableEngine } from '../types.js';
import { NumberRing } from './numberRing.js';
import { makePanel, makeRow } from './domUtils.js';

export function addFrameTimeHud(
  engine: DebuggableEngine,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
  frameTimeAvgWindow: number,
): void {
  const ring = new NumberRing(frameTimeAvgWindow);
  const div = makePanel({
    top: '8px',
    right: '8px',
    minWidth: '140px',
    pointerEvents: 'none',
  });

  const frameRow = makeRow('frame', '- ms');
  const avgRow = makeRow('avg', '- ms');
  const fpsRow = makeRow('fps', '-');
  div.append(frameRow.el, avgRow.el, fpsRow.el);
  add(div);

  const update = (stats: FrameStats): void => {
    ring.push(stats.frameTimeMs);
    const avg = ring.mean();
    frameRow.setValue(`${stats.frameTimeMs.toFixed(2)} ms`);
    avgRow.setValue(`${avg.toFixed(2)} ms`);
    fpsRow.setValue(avg > 0 ? (1000 / avg).toFixed(1) : '-');
  };

  if (typeof engine.onFrame === 'function') {
    const unsub = engine.onFrame(update);
    cleanupFns.push(unsub);
    return;
  }

  let lastTime: number | null = null;
  let rafId: number | null = null;
  const tick = (now: number): void => {
    if (lastTime !== null) {
      update({ frameTimeMs: now - lastTime });
    }
    lastTime = now;
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
  cleanupFns.push(() => {
    if (rafId !== null) cancelAnimationFrame(rafId);
  });
}
