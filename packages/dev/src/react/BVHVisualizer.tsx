// BVHVisualizer — top-down wireframe overview of the scene BVH AABBs.
//
// W12 implementation: subscribes to `engine.debug?.bvhNodes?.()` (returns a
// flat Float32Array — 8 lanes per node, `[minX,minY,minZ,maxX,maxY,maxZ,
// depth,pad]` per the EngineDebugSurface contract) and rasterizes each
// node's XZ projection as an SVG <rect>. Node colour is HSL-mapped by
// position in the array (proxy for depth; HybridEngine currently fills the
// depth lane with zeros — see HybridEngine.ts:1268 "TODO parent traversal").
//
// SVG over <canvas> because:
//   - thousands of strokes are well within SVG's sweet spot
//   - resolution-independent (the overlay scales with the host canvas)
//   - debuggable in DevTools without GPU profiler overhead
//
// Keyboard toggle: default 'b'. Click the badge to toggle in absence of
// keyboard focus.

import React, {
  type CSSProperties,
  type FC,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { DebuggableEngine } from '../types.js';

export interface BVHVisualizerProps {
  /** The engine to inspect. */
  engine: DebuggableEngine;
  /** Keyboard key that toggles visibility. Default: 'b'. */
  toggleKey?: string | null;
  /** Initial visibility. Default: false. */
  initiallyVisible?: boolean;
  /**
   * Maximum nodes to render. Very deep BVHs (~100k nodes) can stutter SVG
   * paint; we draw the first `nodeLimit` lanes and append a "+N more" note.
   * Default: 4096.
   */
  nodeLimit?: number;
  /** Refresh rate (Hz) for node-array re-poll. Default: 1 (BVH rarely changes). */
  refreshRate?: number;
  /** CSS class name applied to the toggle badge. */
  className?: string;
}

const BADGE_STYLE: CSSProperties = {
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

const OVERLAY_PANEL_STYLE: CSSProperties = {
  position: 'absolute',
  bottom: 40,
  right: 8,
  background: 'rgba(0,0,0,0.85)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '6px 8px',
  borderRadius: 4,
  zIndex: 9998,
  width: 320,
};

const WARN_STYLE: CSSProperties = {
  color: '#ffb347',
  fontStyle: 'italic',
  fontSize: 10,
};

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

interface NodeRect {
  x: number;
  y: number;
  w: number;
  h: number;
  hue: number;
}

interface ProjectedNodes {
  rects: NodeRect[];
  totalCount: number;
  truncated: boolean;
  sceneMin: [number, number];
  sceneMax: [number, number];
}

/**
 * Project the BVH nodes onto the XZ plane (top-down). Computes the global
 * AABB across all nodes for sizing, then emits per-node rect in the
 * scene-space units the SVG viewBox is set to.
 *
 * BVH node layout (per EngineDebugSurface contract):
 *   [minX, minY, minZ, maxX, maxY, maxZ, depth, pad]
 */
function projectNodesTopDown(
  nodes: Float32Array,
  nodeLimit: number,
): ProjectedNodes | null {
  const STRIDE = 8;
  const total = Math.floor(nodes.length / STRIDE);
  if (total === 0) return null;
  const drawn = Math.min(total, nodeLimit);

  // First pass: scene AABB across the entire node list (not just drawn —
  // we want the projection origin centred on the whole BVH).
  let sxMin = Infinity, szMin = Infinity;
  let sxMax = -Infinity, szMax = -Infinity;
  for (let i = 0; i < total; i++) {
    const o = i * STRIDE;
    const minX = nodes[o    ] ?? 0;
    const minZ = nodes[o + 2] ?? 0;
    const maxX = nodes[o + 3] ?? 0;
    const maxZ = nodes[o + 5] ?? 0;
    if (minX < sxMin) sxMin = minX;
    if (minZ < szMin) szMin = minZ;
    if (maxX > sxMax) sxMax = maxX;
    if (maxZ > szMax) szMax = maxZ;
  }

  // Second pass: emit `drawn` rect projections.
  const rects: NodeRect[] = [];
  for (let i = 0; i < drawn; i++) {
    const o = i * STRIDE;
    const minX = nodes[o    ] ?? 0;
    const minZ = nodes[o + 2] ?? 0;
    const maxX = nodes[o + 3] ?? 0;
    const maxZ = nodes[o + 5] ?? 0;
    // HSL hue cycles through the node array — a passable proxy for depth
    // when the depth lane is zero (HybridEngine TODO at L1268).
    const hue = ((i / Math.max(1, drawn)) * 360) % 360;
    rects.push({
      x: minX,
      y: minZ,
      w: Math.max(0, maxX - minX),
      h: Math.max(0, maxZ - minZ),
      hue,
    });
  }

  return {
    rects,
    totalCount: total,
    truncated: total > drawn,
    sceneMin: [sxMin, szMin],
    sceneMax: [sxMax, szMax],
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export const BVHVisualizer: FC<BVHVisualizerProps> = ({
  engine,
  toggleKey = 'b',
  initiallyVisible = false,
  nodeLimit = 4096,
  refreshRate = 1,
  className,
}) => {
  const [visible, setVisible] = useState(initiallyVisible);
  const [tick, setTick] = useState(0);
  const tickIntervalRef = useRef<number | null>(null);

  // Keyboard toggle
  useEffect(() => {
    if (toggleKey === null) return;
    const key = toggleKey.toLowerCase();
    const handler = (e: KeyboardEvent): void => {
      if (e.key.toLowerCase() === key && !e.ctrlKey && !e.metaKey && !e.altKey) {
        setVisible((v) => !v);
      }
    };
    window.addEventListener('keydown', handler);
    return () => {
      window.removeEventListener('keydown', handler);
    };
  }, [toggleKey]);

  // Poll the BVH node array at `refreshRate` while visible.
  useEffect(() => {
    if (!visible) return;
    if (typeof window === 'undefined') return;
    const intervalMs = 1000 / Math.max(0.1, refreshRate);
    tickIntervalRef.current = window.setInterval(() => {
      setTick((t) => t + 1);
    }, intervalMs);
    return () => {
      if (tickIntervalRef.current !== null) {
        window.clearInterval(tickIntervalRef.current);
        tickIntervalRef.current = null;
      }
    };
  }, [visible, refreshRate]);

  const hasDebug = typeof engine.debug?.bvhNodes === 'function';

  // Re-evaluate the projection when `tick` advances OR when the engine
  // identity changes. Memo so SVG isn't rebuilt on every parent re-render.
  const projected = useMemo<ProjectedNodes | null>(() => {
    if (!hasDebug || !visible) return null;
    const nodes = engine.debug?.bvhNodes?.() ?? null;
    if (nodes == null) return null;
    return projectNodesTopDown(nodes, nodeLimit);
    // tick is in deps explicitly — we want re-poll on each interval tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine, hasDebug, visible, nodeLimit, tick]);

  const label = toggleKey !== null
    ? `BVH [${toggleKey.toUpperCase()}] ${visible ? '■ on' : '□ off'}`
    : `BVH ${visible ? '■ on' : '□ off'}`;

  // Compute SVG viewBox + per-rect viewport mapping. SVG handles
  // resolution-independence — we just hand it scene-space coordinates.
  const viewBoxStr = (() => {
    if (projected == null) return '0 0 1 1';
    const [xMin, zMin] = projected.sceneMin;
    const [xMax, zMax] = projected.sceneMax;
    const w = Math.max(1e-6, xMax - xMin);
    const h = Math.max(1e-6, zMax - zMin);
    // Add 5% padding so edge nodes aren't clipped by the SVG bounds.
    const padX = w * 0.05;
    const padZ = h * 0.05;
    return `${xMin - padX} ${zMin - padZ} ${w + 2 * padX} ${h + 2 * padZ}`;
  })();

  return (
    <>
      {visible && (
        <div style={OVERLAY_PANEL_STYLE} role="region" aria-label="BVH visualizer">
          <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
            BVH — top-down (XZ)
          </div>
          {!hasDebug && (
            <div style={WARN_STYLE}>
              Engine does not implement <code>debug.bvhNodes()</code>.
            </div>
          )}
          {hasDebug && projected == null && (
            <div style={{ color: '#888', fontSize: 10 }}>waiting for BVH…</div>
          )}
          {hasDebug && projected !== null && (
            <>
              <svg
                viewBox={viewBoxStr}
                preserveAspectRatio="xMidYMid meet"
                style={{
                  display: 'block',
                  width: '100%',
                  height: 200,
                  background: '#0a0a14',
                  border: '1px solid #333',
                }}
                aria-label="BVH AABB wireframe"
              >
                {/* Stroke-width as a fraction of viewBox extent so it stays
                    visually consistent across scene scales. */}
                <g
                  fill="none"
                  strokeWidth={
                    Math.max(
                      projected.sceneMax[0] - projected.sceneMin[0],
                      projected.sceneMax[1] - projected.sceneMin[1],
                    ) * 0.0015
                  }
                  vectorEffect="non-scaling-stroke"
                >
                  {projected.rects.map((r, i) => (
                    <rect
                      key={i}
                      x={r.x}
                      y={r.y}
                      width={r.w}
                      height={r.h}
                      stroke={`hsl(${r.hue}, 80%, 60%)`}
                      strokeOpacity={0.6}
                    />
                  ))}
                </g>
              </svg>
              <div style={{ color: '#888', fontSize: 10, marginTop: 4 }}>
                {projected.totalCount} nodes
                {projected.truncated && ` (showing first ${nodeLimit})`}
              </div>
            </>
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
