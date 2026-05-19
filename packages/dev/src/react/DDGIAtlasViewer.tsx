// DDGIAtlasViewer — pop-out panel showing irradiance + visibility atlases live.
//
// A3 (2026-05-19) — wired to engine.debug.{device,atlasTexture,
// visibilityAtlasTexture}() via the shared `startGpuTextureBlit` helper.
// Readback runs throttled (~10 Hz) to keep GPU→CPU fences off the main
// render path. Engine surfaces fall back to a placeholder badge.

import React, { type FC, useEffect, useRef } from 'react';
import type { DebuggableEngine } from '../types.js';
import { startGpuTextureBlit } from './gpuTextureBlit.js';

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
  const irrCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const visCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const hasAtlas = typeof engine.debug?.atlasTexture === 'function';
  const hasVisibilityAtlas = typeof engine.debug?.visibilityAtlasTexture === 'function';
  const hasDevice = typeof engine.debug?.device === 'function';

  // A3 (2026-05-19) — wire the canvas-blit readback. Re-runs when the
  // atlas/visibility texture handles change identity (the engine swaps
  // them on setScene + on ping-pong resolve). The throttled tick paints
  // the atlases at ~10 Hz so the GPU→CPU fence stays off the render path.
  useEffect(() => {
    if (!visible || !hasAtlas || !hasDevice) return;
    const device = engine.debug?.device?.();
    const atlas = engine.debug?.atlasTexture?.();
    if (device == null || atlas == null) return;
    const canvas = irrCanvasRef.current;
    if (canvas == null) return;
    return startGpuTextureBlit(canvas, device, atlas, {
      throttleMs: 100,
      label: 'ddgi-irr-atlas',
    });
  }, [engine, visible, hasAtlas, hasDevice]);

  useEffect(() => {
    if (!visible || !hasVisibilityAtlas || !hasDevice) return;
    const device = engine.debug?.device?.();
    const atlas = engine.debug?.visibilityAtlasTexture?.();
    if (device == null || atlas == null) return;
    const canvas = visCanvasRef.current;
    if (canvas == null) return;
    return startGpuTextureBlit(canvas, device, atlas, {
      throttleMs: 100,
      label: 'ddgi-vis-atlas',
    });
  }, [engine, visible, hasVisibilityAtlas, hasDevice]);

  if (!visible) return null;

  if (!hasAtlas) {
    return (
      <div className={className} style={PANEL_STYLE} role="region" aria-label="DDGI Atlas Viewer">
        <div style={{ fontWeight: 'bold', marginBottom: 4 }}>DDGI Atlas Viewer</div>
        <div style={WARN_STYLE}>
          Requires <code>engine.debug.atlasTexture()</code> — backend hasn't
          wired the debug surface.
        </div>
      </div>
    );
  }

  return (
    <div className={className} style={PANEL_STYLE} role="region" aria-label="DDGI Atlas Viewer">
      <div style={{ fontWeight: 'bold', marginBottom: 4 }}>DDGI Atlas Viewer</div>
      <div style={{ fontSize: 10, color: '#aaa', marginBottom: 2 }}>Irradiance</div>
      <canvas ref={irrCanvasRef} style={{ display: 'block', maxWidth: 200, imageRendering: 'pixelated' }} />
      {hasVisibilityAtlas ? (
        <>
          <div style={{ fontSize: 10, color: '#aaa', marginTop: 6, marginBottom: 2 }}>Visibility</div>
          <canvas ref={visCanvasRef} style={{ display: 'block', maxWidth: 200, imageRendering: 'pixelated' }} />
        </>
      ) : null}
      <div style={{ marginTop: 6, color: '#666', fontSize: 10 }}>
        Live readback @ ~10 Hz. Reinhard-tonemapped.
      </div>
    </div>
  );
};
