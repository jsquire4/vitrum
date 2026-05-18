// DDGIAtlasViewer — pop-out panel showing irradiance + visibility atlases live.
//
// Implementation mode: INTERFACE STUB (approach (b))
//
// Rationale: Displaying a live GPUTexture as a canvas requires either:
//   (a) a WebGPU readback (GPUBuffer → ArrayBuffer → ImageData) every frame, or
//   (b) rendering the atlas into a dedicated 2D canvas via a blit pass.
// Both require deep pipeline access (the GPUDevice, the atlas GPUTexture handle,
// and a blit renderpass) that HybridEngine does not expose today.
//
// The engine.debug.atlasTexture() and .visibilityAtlasTexture() APIs are
// declared in types.ts (EngineDebugSurface) — the component will fully work
// once HybridEngine implements them.
//
// TODO T3.G followup: wire this once HybridEngine implements engine.debug.
//   1. Allocate a small 2D canvas overlay.
//   2. Per-frame: engine.debug.atlasTexture() → GPUTexture.
//   3. GPUCommandEncoder: copyTextureToBuffer → readback → draw to canvas.
//   4. Probe-click: map click coords to probe grid cell, emit highlight.

import React, { type FC } from 'react';
import type { DebuggableEngine } from '../types.js';

export interface DDGIAtlasViewerProps {
  /** The engine to inspect. Must implement engine.debug (T3.G followup). */
  engine: DebuggableEngine;
  /** Whether the panel is visible. Default: true. */
  visible?: boolean;
  /** CSS class name applied to the root panel div. */
  className?: string;
}

const PANEL_STYLE: React.CSSProperties = {
  position: 'absolute',
  top: 8,
  left: 8,
  background: 'rgba(0,0,0,0.75)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '8px 10px',
  borderRadius: 4,
  userSelect: 'none',
  zIndex: 9998,
  minWidth: 220,
};

const WARN_STYLE: React.CSSProperties = {
  color: '#ffb347',
  fontStyle: 'italic',
};

export const DDGIAtlasViewer: FC<DDGIAtlasViewerProps> = ({
  engine,
  visible = true,
  className,
}) => {
  if (!visible) return null;

  const hasDebug = typeof engine.debug?.atlasTexture === 'function';

  if (!hasDebug) {
    // Warn once on every mount (not every render).
     
    console.warn(
      '[DDGIAtlasViewer] engine.debug.atlasTexture() is not implemented. ' +
      'DDGIAtlasViewer requires the T3.G followup: HybridEngine must expose ' +
      'engine.debug with atlasTexture() and visibilityAtlasTexture(). ' +
      'See packages/dev/src/types.ts:EngineDebugSurface for the interface.'
    );
  }

  return (
    <div className={className} style={PANEL_STYLE} role="region" aria-label="DDGI Atlas Viewer">
      <div style={{ fontWeight: 'bold', marginBottom: 4 }}>DDGI Atlas Viewer</div>
      {hasDebug ? (
        // Future: render atlas canvas here once HybridEngine wires engine.debug.
        <div style={WARN_STYLE}>Atlas canvas not yet rendered (T3.G followup).</div>
      ) : (
        <div style={WARN_STYLE}>
          Requires engine.debug API — coming in T3.G followup.
          <br />
          HybridEngine must implement <code>engine.debug.atlasTexture()</code>.
        </div>
      )}
      <div style={{ marginTop: 6, color: '#666', fontSize: 10 }}>
        Irradiance &amp; visibility atlases will render here.
        <br />
        Click a probe to highlight it in 3D.
      </div>
    </div>
  );
};
