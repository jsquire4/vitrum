// attachDebugOverlays — vanilla (no-React) debug overlay for non-React hosts.
//
// Creates DOM nodes directly on top of the provided canvas container element.
// Returns a dispose() handle for cleanup.
//
// Implemented overlays (vanilla):
//   - FrameTimeHUD: real implementation using engine.onFrame or rAF fallback.
//   - DenoiserABToggle: keyboard 'D' + button badge; engine.debug.setDenoiserEnabled
//     when available.
//
// Stub overlays (console.warn + static badge):
//   - DDGIAtlasViewer, BVHVisualizer, GISignalSplit, MaterialInspector —
//     all require engine.debug APIs not yet implemented in HybridEngine.
//     Each emits one console.warn on attach and renders a static badge.
//     TODO T3.G followup: implement when HybridEngine surfaces engine.debug.

import type { Scene } from '@vitrum/core';
import type { DebuggableEngine, FrameStats } from './types.js';
import { RingBuffer } from './react/FrameTimeHUD.js';

// ────────────────────────────────────────────────────────────────────────────
// Options
// ────────────────────────────────────────────────────────────────────────────

export interface AttachDebugOverlaysOptions {
  /**
   * Which overlays to activate. Defaults to all implemented overlays
   * (frameTime + denoiserToggle). Stub overlays can be included; they render
   * placeholder badges.
   */
  overlays?: ReadonlyArray<
    | 'frameTime'
    | 'denoiserToggle'
    | 'ddgiAtlas'
    | 'bvhVisualizer'
    | 'giSignalSplit'
    | 'materialInspector'
  >;
  /**
   * Moving average window size for FrameTimeHUD. Default: 60.
   */
  frameTimeAvgWindow?: number;
  /**
   * Scene snapshot for MaterialInspector (optional; only used if
   * 'materialInspector' is in overlays). Same object passed to engine.setScene().
   */
  scene?: Scene;
}

export interface DebugOverlaysHandle {
  /** Remove all overlay DOM nodes and unsubscribe from all engine events. */
  dispose(): void;
}

// ────────────────────────────────────────────────────────────────────────────
// Main entry point
// ────────────────────────────────────────────────────────────────────────────

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
  } = opts;

  const cleanupFns: Array<() => void> = [];
  const nodes: HTMLElement[] = [];

  const add = (el: HTMLElement): void => {
    container.appendChild(el);
    nodes.push(el);
  };

  // ── Frame Time HUD ───────────────────────────────────────────────────────
  if (overlays.includes('frameTime')) {
    const ring = new RingBuffer(frameTimeAvgWindow);
    const div = makeDiv({
      position: 'absolute', top: '8px', right: '8px',
      background: 'rgba(0,0,0,0.65)', color: '#e0e0e0',
      fontFamily: 'monospace', fontSize: '11px',
      padding: '6px 8px', borderRadius: '4px',
      userSelect: 'none', pointerEvents: 'none',
      lineHeight: '1.6', minWidth: '140px', zIndex: '9999',
    });

    const frameRow = makeRow('frame', '— ms');
    const avgRow = makeRow('avg', '— ms');
    const fpsRow = makeRow('fps', '—');
    div.append(frameRow.el, avgRow.el, fpsRow.el);
    add(div);

    const update = (stats: FrameStats): void => {
      ring.push(stats.frameTimeMs);
      const avg = ring.mean();
      frameRow.setValue(`${stats.frameTimeMs.toFixed(2)} ms`);
      avgRow.setValue(`${avg.toFixed(2)} ms`);
      fpsRow.setValue(avg > 0 ? (1000 / avg).toFixed(1) : '—');
    };

    // W3-D8: source-of-truth is engine.capabilities.frameTelemetry. When
    // true, engine.onFrame is guaranteed present per the EngineCapabilities
    // invariant; otherwise fall back to rAF wall-clock measurement.
    if (engine.capabilities.frameTelemetry) {
      const unsub = engine.onFrame!(update);
      cleanupFns.push(unsub);
    } else {
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
  }

  // ── Denoiser A/B Toggle ──────────────────────────────────────────────────
  if (overlays.includes('denoiserToggle')) {
    // W3-D8: gate the top-level engine.debug presence on the capability
    // flag, then check the specific method on the debug surface (individual
    // EngineDebugSurface fields remain optional per the contract).
    const hasDebug =
      engine.capabilities.debugSurface && !!engine.debug?.setDenoiserEnabled;
    let enabled = engine.debug?.isDenoiserEnabled?.() ?? true;

    const badge = makeDiv({
      position: 'absolute', top: '8px', left: '8px',
      background: 'rgba(0,0,0,0.65)', color: '#e0e0e0',
      fontFamily: 'monospace', fontSize: '11px',
      padding: '4px 8px', borderRadius: '4px',
      userSelect: 'none', cursor: 'pointer', zIndex: '9998',
      borderLeft: `3px solid ${enabled ? '#7dfa7d' : '#fa7d7d'}`,
    });

    const updateBadge = (): void => {
      badge.style.borderLeft = `3px solid ${enabled ? '#7dfa7d' : '#fa7d7d'}`;
      badge.textContent = `denoiser ${enabled ? '■ on' : '□ off'} [D]`;
    };
    updateBadge();
    add(badge);

    const toggle = (): void => {
      enabled = !enabled;
      if (hasDebug) {
        engine.debug!.setDenoiserEnabled!(enabled);
      } else {
        // eslint-disable-next-line no-console
        console.warn(
          '[attachDebugOverlays] denoiserToggle: engine.debug.setDenoiserEnabled() ' +
          'not implemented — T3.G followup.'
        );
      }
      updateBadge();
    };

    badge.addEventListener('click', toggle);
    const keyHandler = (e: KeyboardEvent): void => {
      if (e.key.toLowerCase() === 'd' && !e.ctrlKey && !e.metaKey && !e.altKey) toggle();
    };
    window.addEventListener('keydown', keyHandler);
    cleanupFns.push(() => {
      badge.removeEventListener('click', toggle);
      window.removeEventListener('keydown', keyHandler);
    });
  }

  // ── Stub overlays ────────────────────────────────────────────────────────
  const stubOverlays: Array<[string, string, object]> = [
    ['ddgiAtlas', 'DDGI Atlas', { position: 'absolute', top: '8px', left: '8px' }],
    ['bvhVisualizer', 'BVH Viz [B]', { position: 'absolute', bottom: '8px', right: '8px' }],
    ['giSignalSplit', 'GI Split', { position: 'absolute', bottom: '8px', left: '8px' }],
    ['materialInspector', 'Mat. Inspector', { position: 'absolute', top: '48px', right: '8px' }],
  ];

  for (const [key, label, extraStyle] of stubOverlays) {
    if (!overlays.includes(key as AttachDebugOverlaysOptions['overlays'] extends ReadonlyArray<infer T> ? T : never)) continue;
    // eslint-disable-next-line no-console
    console.warn(
      `[attachDebugOverlays] "${key}" is a stub — requires engine.debug API (T3.G followup). ` +
      'Rendering placeholder badge.'
    );
    const badge = makeDiv({
      background: 'rgba(0,0,0,0.65)', color: '#ffb347',
      fontFamily: 'monospace', fontSize: '11px',
      padding: '4px 8px', borderRadius: '4px',
      userSelect: 'none', pointerEvents: 'none', zIndex: '9997',
      ...extraStyle,
    });
    badge.textContent = `${label} [stub — T3.G followup]`;
    add(badge);
  }

  // Suppress unused `scene` warning — it's kept for future materialInspector wiring.
  void scene;

  return {
    dispose(): void {
      for (const fn of cleanupFns) fn();
      for (const el of nodes) el.parentNode?.removeChild(el);
      nodes.length = 0;
      cleanupFns.length = 0;
    },
  };
}

// ────────────────────────────────────────────────────────────────────────────
// DOM helpers (internal)
// ────────────────────────────────────────────────────────────────────────────

function makeDiv(style: Record<string, string>): HTMLDivElement {
  const el = document.createElement('div');
  Object.assign(el.style, style);
  return el;
}

function makeRow(
  label: string,
  initialValue: string,
): { el: HTMLDivElement; setValue(v: string): void } {
  const el = document.createElement('div');
  const labelSpan = document.createElement('span');
  labelSpan.style.cssText = 'color:#888;margin-right:4px';
  labelSpan.textContent = label;
  const valueSpan = document.createElement('span');
  valueSpan.textContent = initialValue;
  el.append(labelSpan, valueSpan);
  return {
    el,
    setValue(v: string): void {
      valueSpan.textContent = v;
    },
  };
}
