// BVHVisualizer — diagnostic panel showing BVH structure (depth histogram,
// node count, depth statistics) via `engine.debug.bvhNodes()`.
//
// A3 (2026-05-19) — wired to read the Float32Array node table and render
// a depth-histogram bar chart + summary stats on a 2D canvas. Does NOT
// project AABBs onto the WebGPU canvas — that would require view+proj
// matrices the debug surface doesn't expose. Hosts wanting an AABB
// overlay can render their own pass with the same `bvhNodes()` output
// + the host's camera matrices.

import React, { type FC, useCallback, useEffect, useRef, useState } from 'react';
import type { DebuggableEngine } from '../types.js';
import type { BvhStats } from '../vanilla/bvhStats.js';
import { computeBvhStats } from '../vanilla/bvhStats.js';
import { useKeyToggle } from './hooks.js';

export interface BVHVisualizerProps {
  /** The engine to inspect. Must implement engine.debug.bvhNodes (T3.G followup). */
  engine: DebuggableEngine;
  /**
   * Keyboard key that toggles visibility. Default: 'b'.
   * Set to null to disable keyboard toggle.
   */
  toggleKey?: string | null;
  /** Initial visibility. Default: false (hidden until toggled). */
  initiallyVisible?: boolean;
  /** CSS class name applied to the toggle badge. */
  className?: string;
}

const BADGE_STYLE: React.CSSProperties = {
  position: 'absolute',
  bottom: 8,
  right: 8,
  background: 'rgba(0,0,0,0.65)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '4px 8px',
  borderRadius: 4,
  userSelect: 'none',
  zIndex: 9998,
  cursor: 'pointer',
};

const WARN_STYLE: React.CSSProperties = {
  position: 'absolute',
  bottom: 36,
  right: 8,
  background: 'rgba(0,0,0,0.75)',
  color: '#ffb347',
  fontFamily: 'monospace',
  fontSize: 10,
  padding: '4px 8px',
  borderRadius: 4,
  maxWidth: 280,
  lineHeight: 1.5,
  zIndex: 9997,
};

const PANEL_STYLE: React.CSSProperties = {
  position: 'absolute',
  bottom: 44,
  right: 8,
  background: 'rgba(0,0,0,0.75)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '8px 10px',
  borderRadius: 4,
  zIndex: 9998,
  minWidth: 240,
};

function renderHistogram(canvas: HTMLCanvasElement, stats: BvhStats): void {
  const ctx = canvas.getContext('2d');
  if (ctx == null) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (stats.histogram.length === 0) return;
  const maxCount = Math.max(...stats.histogram);
  if (maxCount === 0) return;
  // Per-bar width sized to fit. Bar `i` is at horizontal slot `i`.
  const barW = Math.max(1, Math.floor(W / stats.histogram.length));
  for (let d = 0; d < stats.histogram.length; d++) {
    const count = stats.histogram[d] ?? 0;
    const barH = Math.round((count / maxCount) * (H - 4));
    const x = d * barW;
    const y = H - barH;
    // hsl(depth * 30, ...) per the W8-follow-up's depth-color convention
    ctx.fillStyle = `hsl(${(d * 30) % 360}, 80%, 60%)`;
    ctx.fillRect(x, y, Math.max(1, barW - 1), barH);
  }
}

export const BVHVisualizer: FC<BVHVisualizerProps> = ({
  engine,
  toggleKey = 'b',
  initiallyVisible = false,
  className,
}) => {
  const [visible, setVisible] = useState(initiallyVisible);
  const [stats, setStats] = useState<BvhStats | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Keyboard toggle
  const toggleVisible = useCallback(() => setVisible((v) => !v), []);
  useKeyToggle(toggleKey ?? null, toggleVisible);

  const hasDebug = typeof engine.debug?.bvhNodes === 'function';

  // A3 (2026-05-19) — re-poll BVH stats every 500 ms while visible.
  // bvhNodes() returns a host-side Float32Array (already CPU-readable),
  // so no GPU readback needed — the cost is one pass over ~N nodes.
  useEffect(() => {
    if (!visible || !hasDebug) return;
    const tick = (): void => {
      try {
        const nodes = engine.debug?.bvhNodes?.();
        if (nodes == null) return;
        const s = computeBvhStats(nodes);
        setStats(s);
        const canvas = canvasRef.current;
        if (canvas != null) renderHistogram(canvas, s);
      } catch {
        // A malformed producer table fails closed without leaving an uncaught
        // interval error in the host React tree.
        setStats(null);
      }
    };
    tick();
    const interval = setInterval(tick, 500);
    return () => { clearInterval(interval); };
  }, [engine, visible, hasDebug]);

  const label = toggleKey !== null
    ? `BVH [${toggleKey.toUpperCase()}] ${visible ? '■ on' : '□ off'}`
    : `BVH ${visible ? '■ on' : '□ off'}`;

  return (
    <>
      {visible && !hasDebug && (
        <div style={WARN_STYLE}>
          BVHVisualizer: requires <code>engine.debug.bvhNodes()</code>.
        </div>
      )}
      {visible && hasDebug && (
        <div style={PANEL_STYLE} role="region" aria-label="BVH Visualizer">
          <div style={{ fontWeight: 'bold', marginBottom: 4 }}>BVH structure</div>
          {stats != null ? (
            <>
              <div>nodes: {stats.nodeCount}</div>
              <div>max depth: {stats.maxDepth}</div>
              <div>avg depth: {stats.avgDepth.toFixed(2)}</div>
              <div style={{ fontSize: 10, color: '#aaa', marginTop: 6, marginBottom: 2 }}>
                Nodes per depth (hsl-coded):
              </div>
              <canvas ref={canvasRef} width={220} height={48} style={{ display: 'block' }} />
            </>
          ) : (
            <div style={{ color: '#aaa' }}>Waiting for BVH build…</div>
          )}
        </div>
      )}
      <div
        className={className}
        style={BADGE_STYLE}
        role="button"
        tabIndex={0}
        aria-pressed={visible}
        aria-label="Toggle BVH visualizer"
        onClick={() => setVisible((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') setVisible((v) => !v);
        }}
      >
        {label}
      </div>
    </>
  );
};
