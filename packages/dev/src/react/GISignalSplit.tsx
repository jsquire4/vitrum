// GISignalSplit — 2×2 split-screen view of direct / indirect / AO / total.
//
// A3 (2026-05-19) — wired to read engine.debug.giSignalTextures() and
// blit each channel through the shared `startGpuTextureBlit` helper.
// Readback runs throttled (~10 Hz) so the GPU→CPU fence stays off the
// render path. Each channel paints into its own 2D-canvas quadrant.

import React, { type FC, useEffect, useRef, useState } from 'react';
import type { DebuggableEngine } from '../types.js';
import { startGpuTextureBlit } from './gpuTextureBlit.js';
import { useDebugDevice } from './hooks.js';

export interface GISignalSplitProps {
  /** The engine to inspect. Must implement engine.debug.giSignalTextures (T3.G followup). */
  engine: DebuggableEngine;
  /** Whether the split view is active. Default: false. */
  active?: boolean;
  /** Callback when the user toggles the split view. */
  onToggle?: (active: boolean) => void;
  /** CSS class name applied to the container. */
  className?: string;
}

const BUTTON_STYLE: React.CSSProperties = {
  position: 'absolute',
  bottom: 8,
  left: 8,
  background: 'rgba(0,0,0,0.65)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '4px 8px',
  borderRadius: 4,
  userSelect: 'none',
  cursor: 'pointer',
  zIndex: 9998,
};

const OVERLAY_STYLE: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gridTemplateRows: '1fr 1fr',
  pointerEvents: 'none',
  zIndex: 9997,
};

const QUADRANT_STYLE: React.CSSProperties = {
  position: 'relative',
  border: '1px solid rgba(255,255,255,0.15)',
  background: 'rgba(0,0,0,0.4)',
  overflow: 'hidden',
};

const CHANNEL_KEYS = ['direct', 'indirect', 'ao', 'total'] as const;
type ChannelKey = (typeof CHANNEL_KEYS)[number];
const CHANNEL_LABELS: Record<ChannelKey, string> = {
  direct: 'direct',
  indirect: 'indirect',
  ao: 'AO',
  total: 'total',
};

const QUADRANT_CANVAS_STYLE: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  width: '100%',
  height: '100%',
  imageRendering: 'pixelated',
};

const QUADRANT_LABEL_STYLE: React.CSSProperties = {
  position: 'absolute',
  top: 4,
  left: 4,
  color: '#ffb347',
  fontFamily: 'monospace',
  fontSize: 11,
  textShadow: '0 0 4px rgba(0,0,0,0.8)',
  zIndex: 2,
};

export const GISignalSplit: FC<GISignalSplitProps> = ({
  engine,
  active: activeProp,
  onToggle,
  className,
}) => {
  const [internalActive, setInternalActive] = useState(false);
  const isControlled = activeProp !== undefined;
  const active = isControlled ? activeProp : internalActive;

  const directRef = useRef<HTMLCanvasElement>(null);
  const indirectRef = useRef<HTMLCanvasElement>(null);
  const aoRef = useRef<HTMLCanvasElement>(null);
  const totalRef = useRef<HTMLCanvasElement>(null);
  const refs: Record<ChannelKey, React.RefObject<HTMLCanvasElement>> = {
    direct: directRef, indirect: indirectRef, ao: aoRef, total: totalRef,
  };

  const hasDebug = typeof engine.debug?.giSignalTextures === 'function';
  const hasDevice = typeof engine.debug?.device === 'function';
  const debugDevice = useDebugDevice(engine);
  const channelTextures = hasDebug ? engine.debug?.giSignalTextures?.() ?? null : null;

  // A3 (2026-05-19) — start one readback loop per channel when active.
  // Each useEffect's cleanup tears down its own readback; the 4 readbacks
  // share the engine queue but each has its own staging buffer.
  useEffect(() => {
    if (!active || !hasDebug || !hasDevice) return;
    if (debugDevice == null || channelTextures == null) return;

    const teardowns: Array<() => void> = [];
    for (const key of CHANNEL_KEYS) {
      const tex = channelTextures[key];
      const canvas = refs[key].current;
      if (tex == null || canvas == null) continue;
      teardowns.push(startGpuTextureBlit(canvas, debugDevice, tex, {
        throttleMs: 100,
        label: `gi-${key}`,
      }));
    }
    return () => {
      for (const t of teardowns) t();
    };
    // The refs object is stable; channel textures are included so readback
    // loops restart if the engine swaps debug texture handles.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, hasDebug, hasDevice, debugDevice, channelTextures]);

  const toggle = (): void => {
    const next = !active;
    if (!isControlled) setInternalActive(next);
    onToggle?.(next);
  };

  return (
    <>
      {active && (
        <div style={OVERLAY_STYLE} className={className}>
          {CHANNEL_KEYS.map((key) => (
            <div key={key} style={QUADRANT_STYLE}>
              <canvas ref={refs[key]} style={QUADRANT_CANVAS_STYLE} />
              <div style={QUADRANT_LABEL_STYLE}>
                {hasDebug ? CHANNEL_LABELS[key] : `${CHANNEL_LABELS[key]} — requires engine.debug`}
              </div>
            </div>
          ))}
        </div>
      )}
      <div
        style={BUTTON_STYLE}
        role="button"
        tabIndex={0}
        aria-pressed={active}
        aria-label="Toggle GI signal split view"
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') toggle();
        }}
      >
        GI split {active ? '■ on' : '□ off'}
      </div>
    </>
  );
};
