// FrameTimeHUD — live frame-time display with 60-frame moving average
// and optional pass breakdown.
//
// Implementation mode: REAL — uses engine.onFrame (T3.E) when present;
// falls back to rAF-delta timing when onFrame is absent so it works today.
//
// Renders a fixed overlay in the top-right corner of the parent container.
// The parent must have position: relative (or any non-static position) so the
// absolute positioning lands inside the canvas area.

import React, { useEffect, useRef, useState, type FC, type CSSProperties } from 'react';
import type { DebuggableEngine, FrameStats } from '../types.js';

// ────────────────────────────────────────────────────────────────────────────
// Moving-average ring buffer (pure TS, easy to unit-test)
// ────────────────────────────────────────────────────────────────────────────

/** Fixed-capacity ring buffer that maintains a running sum for O(1) mean. */
export class RingBuffer {
  private readonly buf: Float64Array;
  private head = 0;
  private count = 0;
  private sum = 0;

  constructor(readonly capacity: number) {
    this.buf = new Float64Array(capacity);
  }

  push(value: number): void {
    const old = this.buf[this.head] ?? 0;
    this.sum -= old;
    this.buf[this.head] = value;
    this.sum += value;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  mean(): number {
    return this.count === 0 ? 0 : this.sum / this.count;
  }

  get filled(): number {
    return this.count;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Props
// ────────────────────────────────────────────────────────────────────────────

export interface FrameTimeHUDProps {
  /** The engine to watch. Must be a vitrum Engine (DebuggableEngine). */
  engine: DebuggableEngine;

  /**
   * Number of frames in the moving average window.
   * Default: 60.
   */
  averageWindow?: number;

  /**
   * Whether to show the per-pass timing breakdown (requires engine.onFrame
   * to supply passTimings). Default: true when data is available.
   */
  showPassBreakdown?: boolean;

  /** CSS class name applied to the root overlay div. */
  className?: string;

  /** Inline style overrides. Position/top/right are set by the component. */
  style?: CSSProperties;
}

// ────────────────────────────────────────────────────────────────────────────
// Styles (inline — no CSS file dependency so the component is drop-in)
// ────────────────────────────────────────────────────────────────────────────

const BASE_STYLE: CSSProperties = {
  position: 'absolute',
  top: 8,
  right: 8,
  background: 'rgba(0,0,0,0.65)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '6px 8px',
  borderRadius: 4,
  userSelect: 'none',
  pointerEvents: 'none',
  lineHeight: 1.6,
  minWidth: 140,
  zIndex: 9999,
};

const LABEL_STYLE: CSSProperties = { color: '#888', marginRight: 4 };
const VALUE_STYLE: CSSProperties = { color: '#fff' };
const PASS_STYLE: CSSProperties = { color: '#adf', marginLeft: 8 };

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export const FrameTimeHUD: FC<FrameTimeHUDProps> = ({
  engine,
  averageWindow = 60,
  showPassBreakdown = true,
  className,
  style,
}) => {
  const [latest, setLatest] = useState<FrameStats | null>(null);
  const [avgMs, setAvgMs] = useState(0);
  const ringRef = useRef(new RingBuffer(averageWindow));
  const rafRef = useRef<number | null>(null);
  const lastRafTimeRef = useRef<number | null>(null);

  useEffect(() => {
    // Recreate ring when capacity changes.
    ringRef.current = new RingBuffer(averageWindow);
  }, [averageWindow]);

  useEffect(() => {
    const ring = ringRef.current;

    // ── Path A: engine.onFrame present (T3.E) ───────────────────────────
    if (typeof engine.onFrame === 'function') {
      const unsubscribe = engine.onFrame((stats) => {
        ring.push(stats.frameTimeMs);
        setLatest(stats);
        setAvgMs(ring.mean());
      });
      return () => {
        unsubscribe();
      };
    }

    // ── Path B: rAF fallback (T3.E not yet landed) ───────────────────────
    // Measure wall-clock frame delta; build a synthetic FrameStats.
    function tick(now: number): void {
      if (lastRafTimeRef.current !== null) {
        const dt = now - lastRafTimeRef.current;
        ring.push(dt);
        const synthetic: FrameStats = { frameTimeMs: dt };
        setLatest(synthetic);
        setAvgMs(ring.mean());
      }
      lastRafTimeRef.current = now;
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      lastRafTimeRef.current = null;
    };
  }, [engine]);

  const fps = avgMs > 0 ? (1000 / avgMs).toFixed(1) : '—';
  const frameMs = latest?.frameTimeMs.toFixed(2) ?? '—';
  const avgMsStr = avgMs > 0 ? avgMs.toFixed(2) : '—';
  const gpuMs = latest?.gpuTimeMs !== undefined ? latest.gpuTimeMs.toFixed(2) : null;
  const passTimes =
    showPassBreakdown && latest?.passTimings ? Object.entries(latest.passTimings) : null;

  return (
    <div
      className={className}
      style={{ ...BASE_STYLE, ...style }}
      role="status"
      aria-live="polite"
      aria-label="Frame time HUD"
    >
      <div>
        <span style={LABEL_STYLE}>frame</span>
        <span style={VALUE_STYLE}>{frameMs} ms</span>
      </div>
      <div>
        <span style={LABEL_STYLE}>avg</span>
        <span style={VALUE_STYLE}>{avgMsStr} ms</span>
      </div>
      <div>
        <span style={LABEL_STYLE}>fps</span>
        <span style={VALUE_STYLE}>{fps}</span>
      </div>
      {gpuMs !== null && (
        <div>
          <span style={LABEL_STYLE}>gpu</span>
          <span style={VALUE_STYLE}>{gpuMs} ms</span>
        </div>
      )}
      {passTimes !== null && passTimes.length > 0 && (
        <div style={{ marginTop: 4, borderTop: '1px solid #444', paddingTop: 4 }}>
          {passTimes.map(([name, ms]) => (
            <div key={name}>
              <span style={LABEL_STYLE}>{name}</span>
              <span style={PASS_STYLE}>{ms.toFixed(2)} ms</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
