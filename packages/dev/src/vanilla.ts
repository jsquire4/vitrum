// attachDebugOverlays - vanilla (no-React) debug overlay for non-React hosts.
//
// Creates DOM nodes directly on top of the provided canvas container element.
// Returns a dispose() handle for cleanup.
//
// Implementation is split into vanilla/ sub-modules; this file assembles them
// and exports the public surface.

import type { Scene } from '@vitrum/core';
import type { DebuggableEngine } from './types.js';
import { createFrameMonitor } from './vanilla/frameMonitor.js';
import { addFrameTimeHud } from './vanilla/overlayFrameTime.js';
import { addDenoiserDiagnostics } from './vanilla/overlayDenoiser.js';
import { addDdgiDiagnostics } from './vanilla/overlayDdgi.js';
import { addBvhDiagnostics } from './vanilla/overlayBvh.js';
import { addGiSignalDiagnostics } from './vanilla/overlayGiSignal.js';
import { addMaterialInspectorFallback } from './vanilla/overlayMaterialInspector.js';

// Re-export sub-module symbols so existing callers that import directly from
// this module (if any) remain working.
export type { FrameMonitor } from './vanilla/frameMonitor.js';
export { createFrameMonitor } from './vanilla/frameMonitor.js';
export { NumberRing } from './vanilla/numberRing.js';
export type { BvhStats } from './vanilla/bvhStats.js';
export { computeBvhStats, renderBvhBars } from './vanilla/bvhStats.js';

type OverlayId =
  | 'frameTime'
  | 'denoiserToggle'
  | 'ddgiAtlas'
  | 'bvhVisualizer'
  | 'giSignalSplit'
  | 'materialInspector';

const DIAGNOSTIC_OVERLAYS = new Set<OverlayId>([
  'ddgiAtlas',
  'giSignalSplit',
]);

export interface AttachDebugOverlaysOptions {
  /**
   * Which overlays to activate. Defaults to frame time + denoiser diagnostics.
   * Each optional debug overlay renders a truthful disabled/not-ready state
   * when its backend hook is absent.
   */
  overlays?: ReadonlyArray<OverlayId>;
  /**
   * Moving average window size for FrameTimeHUD. Default: 60.
   */
  frameTimeAvgWindow?: number;
  /**
   * Scene snapshot for MaterialInspector. Pass the same object passed to
   * engine.setScene() so picked primitive IDs can be resolved locally.
   */
  scene?: Scene;
  /**
   * Render canvas used for material click-picking. If omitted, the vanilla
   * overlay uses `container` when it is a canvas, or the first canvas inside it.
   */
  canvas?: HTMLCanvasElement | null;
}

export interface DebugOverlaysHandle {
  /** Remove all overlay DOM nodes and unsubscribe from all engine events. */
  dispose(): void;
}

/**
 * Attach debug overlays on top of a canvas element.
 *
 * The `container` element must have `position: relative` (or any non-static
 * position) so that the absolutely-positioned overlay nodes land inside the
 * canvas area. This mirrors how the React components work.
 *
 * @example
 * ```ts
 * const handle = attachDebugOverlays(engine, canvasContainer);
 * // later:
 * handle.dispose();
 * ```
 */
export function attachDebugOverlays(
  engine: DebuggableEngine,
  container: HTMLElement,
  opts: AttachDebugOverlaysOptions = {},
): DebugOverlaysHandle {
  const {
    overlays = ['frameTime', 'denoiserToggle'],
    frameTimeAvgWindow = 60,
    scene,
    canvas,
  } = opts;

  const overlaySet = new Set<OverlayId>(overlays);
  const cleanupFns: Array<() => void> = [];
  const nodes: HTMLElement[] = [];

  const add = (el: HTMLElement): void => {
    container.appendChild(el);
    nodes.push(el);
  };

  const frameMonitor = needsFrameMonitor(overlaySet)
    ? createFrameMonitor(engine)
    : null;
  if (frameMonitor != null) cleanupFns.push(() => frameMonitor.dispose());

  if (overlaySet.has('frameTime')) {
    addFrameTimeHud(engine, add, cleanupFns, frameTimeAvgWindow);
  }

  if (overlaySet.has('denoiserToggle')) {
    addDenoiserDiagnostics(engine, add, cleanupFns);
  }

  if (overlaySet.has('ddgiAtlas')) {
    addDdgiDiagnostics(engine, add, cleanupFns);
  }

  if (overlaySet.has('bvhVisualizer')) {
    addBvhDiagnostics(engine, add, cleanupFns);
  }

  if (overlaySet.has('giSignalSplit')) {
    addGiSignalDiagnostics(engine, add, cleanupFns, frameMonitor);
  }

  if (overlaySet.has('materialInspector')) {
    addMaterialInspectorFallback(engine, container, canvas ?? null, scene, add, cleanupFns);
  }

  return {
    dispose(): void {
      for (const fn of cleanupFns) fn();
      for (const el of nodes) el.parentNode?.removeChild(el);
      nodes.length = 0;
      cleanupFns.length = 0;
    },
  };
}

function needsFrameMonitor(overlays: ReadonlySet<OverlayId>): boolean {
  for (const key of DIAGNOSTIC_OVERLAYS) {
    if (overlays.has(key)) return true;
  }
  return false;
}
