// overlayFrameTime.ts — FrameTimeHUD overlay.

import type { FrameStats } from '../types.js';
import type { DebuggableEngine } from '../types.js';
import { NumberRing, observeFrameTime } from './numberRing.js';
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

  // D2-6 — `observeFrameTime` pushes each sample into `ring` before invoking
  // this render callback, so it reads the already-updated mean.
  const render = (stats: FrameStats): void => {
    const avg = ring.mean();
    frameRow.setValue(`${stats.frameTimeMs.toFixed(2)} ms`);
    avgRow.setValue(`${avg.toFixed(2)} ms`);
    fpsRow.setValue(avg > 0 ? (1000 / avg).toFixed(1) : '-');
  };

  cleanupFns.push(observeFrameTime(engine, ring, render));
}
