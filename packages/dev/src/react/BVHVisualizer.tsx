// BVHVisualizer — overlay BVH bounding boxes color-coded by depth.
//
// Implementation mode: INTERFACE STUB (approach (b))
//
// Rationale: Drawing BVH bounding boxes on top of the WebGPU canvas requires:
//   (a) Reading the BVH node array back from the GPU (or maintaining a CPU-side
//       mirror that the engine already builds during BVH construction).
//   (b) A separate WebGL2/Canvas2D/SVG overlay pass that rasterizes the AABBs
//       color-coded by depth onto the screen.
//   (c) Projecting 3D AABB corners through the current view+proj matrix.
//
// None of these are available without engine.debug.bvhNodes() (see types.ts).
// The BVH node array exists in HybridEngine but is not surfaced.
//
// TODO T3.G followup: wire this once HybridEngine implements engine.debug.
//   1. engine.debug.bvhNodes() → Float32Array of [min, max, depth] per node.
//   2. Create a <canvas> overlay (same size as WebGPU canvas) in front of it.
//   3. Per-frame: project each node's 8 corners through viewProj; draw box in
//      depth-mapped color (hsl(depth * 30, 80%, 60%)).
//   4. Toggle key (default 'B') shows/hides the overlay.

import React, { type FC, useEffect, useRef, useState } from 'react';
import type { DebuggableEngine } from '../types.js';

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

export const BVHVisualizer: FC<BVHVisualizerProps> = ({
  engine,
  toggleKey = 'b',
  initiallyVisible = false,
  className,
}) => {
  const [visible, setVisible] = useState(initiallyVisible);
  const warnedRef = useRef(false);

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

  const hasDebug = typeof engine.debug?.bvhNodes === 'function';

  if (visible && !hasDebug && !warnedRef.current) {
    warnedRef.current = true;
     
    console.warn(
      '[BVHVisualizer] engine.debug.bvhNodes() is not implemented. ' +
      'BVHVisualizer requires the T3.G followup: HybridEngine must expose ' +
      'engine.debug.bvhNodes() returning a Float32Array of node AABBs. ' +
      'See packages/dev/src/types.ts:EngineDebugSurface for the interface.'
    );
  }

  const label = toggleKey !== null
    ? `BVH [${toggleKey.toUpperCase()}] ${visible ? '■ on' : '□ off'}`
    : `BVH ${visible ? '■ on' : '□ off'}`;

  return (
    <>
      {visible && !hasDebug && (
        <div style={WARN_STYLE}>
          BVHVisualizer: requires engine.debug API (T3.G followup).
          <br />
          Implement <code>engine.debug.bvhNodes()</code> in HybridEngine.
        </div>
      )}
      {/* Future: <canvas> overlay drawn here when hasDebug is true. */}
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
