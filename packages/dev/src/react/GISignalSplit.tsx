// GISignalSplit — 2×2 split-screen view of direct / indirect / AO / total.
//
// Implementation mode: INTERFACE STUB (approach (b))
//
// Rationale: A split-screen view of GI signal channels requires:
//   (a) The engine to expose separate direct-light, indirect-light, AO, and
//       composited textures as GPUTextures. In HybridEngine these live in
//       separate render passes (DDGI receiver → ReSTIR-DI → SVGF) but are
//       not independently surfaced.
//   (b) A 4-up blit pass that reads four textures and renders them into
//       quadrants of a 2D canvas overlay.
//
// The interface is declared in types.ts (EngineDebugSurface.giSignalTextures).
//
// TODO T3.G followup: wire this once HybridEngine implements engine.debug.
//   1. HybridEngine: expose engine.debug.giSignalTextures() with direct,
//      indirect, ao, total GPUTextures (or null per-channel).
//   2. GISignalSplit: allocate a <canvas> overlay covering the render canvas.
//   3. Each frame: blit the 4 textures into the 4 quadrants via copyTextureToBuffer
//      (WebGPU) or framebuffer readback (WebGL2).
//   4. Draw channel labels ("direct", "indirect", "AO", "total") in each quad.

import React, { type FC, useState } from 'react';
import type { DebuggableEngine } from '../types.js';

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
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  border: '1px solid rgba(255,255,255,0.15)',
  background: 'rgba(0,0,0,0.4)',
  color: '#ffb347',
  fontFamily: 'monospace',
  fontSize: 12,
  fontStyle: 'italic',
};

const LABELS = ['direct', 'indirect', 'AO', 'total'] as const;

export const GISignalSplit: FC<GISignalSplitProps> = ({
  engine,
  active: activeProp,
  onToggle,
  className,
}) => {
  const [internalActive, setInternalActive] = useState(false);
  const isControlled = activeProp !== undefined;
  const active = isControlled ? activeProp : internalActive;

  const hasDebug = typeof engine.debug?.giSignalTextures === 'function';

  const toggle = (): void => {
    const next = !active;
    if (!isControlled) setInternalActive(next);
    onToggle?.(next);

    if (next && !hasDebug) {
      console.warn(
        '[GISignalSplit] engine.debug.giSignalTextures() is not implemented. ' +
          'GISignalSplit requires the T3.G followup: HybridEngine must expose ' +
          'engine.debug.giSignalTextures() returning {direct, indirect, ao, total}. ' +
          'See packages/dev/src/types.ts:EngineDebugSurface for the interface.',
      );
    }
  };

  return (
    <>
      {active && (
        <div style={OVERLAY_STYLE} className={className}>
          {LABELS.map((label) => (
            <div key={label} style={QUADRANT_STYLE}>
              {hasDebug
                ? `[${label}]` // Future: replaced with blitted texture canvas
                : `${label} — requires engine.debug (T3.G followup)`}
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
