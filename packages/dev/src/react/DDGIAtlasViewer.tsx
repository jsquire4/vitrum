// DDGIAtlasViewer — pop-out panel showing irradiance + visibility atlases live.
//
// A3 (2026-05-19) — wired to engine.debug.{device,atlasTexture,
// visibilityAtlasTexture}() via the shared `startGpuTextureBlit` helper.
// Readback runs throttled (~10 Hz) to keep GPU→CPU fences off the main
// render path. Engine surfaces fall back to a placeholder badge.

import React, { type FC, useEffect, useRef } from 'react';
import type { DebuggableEngine } from '../types.js';
import { startGpuTextureBlit } from './gpuTextureBlit.js';
import { useDebugDevice } from './hooks.js';

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
  const debugDevice = useDebugDevice(engine);
  const irradianceAtlas = hasAtlas ? engine.debug?.atlasTexture?.() ?? null : null;
  const visibilityAtlas = hasVisibilityAtlas ? engine.debug?.visibilityAtlasTexture?.() ?? null : null;

  // A3 (2026-05-19) — wire the canvas-blit readback. Re-runs when the
  // atlas/visibility texture handles change identity (the engine swaps
  // them on setScene + on ping-pong resolve). The throttled tick paints
  // the atlases at ~10 Hz so the GPU→CPU fence stays off the render path.
  useEffect(() => {
    if (!visible || !hasAtlas || !hasDevice) return;
    if (debugDevice == null || irradianceAtlas == null) return;
    const canvas = irrCanvasRef.current;
    if (canvas == null) return;
    return startGpuTextureBlit(canvas, debugDevice, irradianceAtlas, {
      throttleMs: 100,
      label: 'ddgi-irr-atlas',
      decodeMode: 'ddgi-irradiance',
    });
  }, [visible, hasAtlas, hasDevice, debugDevice, irradianceAtlas]);

  useEffect(() => {
    if (!visible || !hasVisibilityAtlas || !hasDevice) return;
    if (debugDevice == null || visibilityAtlas == null) return;
    const canvas = visCanvasRef.current;
    if (canvas == null) return;
    return startGpuTextureBlit(canvas, debugDevice, visibilityAtlas, {
      throttleMs: 100,
      label: 'ddgi-vis-atlas',
      decodeMode: 'ddgi-visibility',
    });
  }, [visible, hasVisibilityAtlas, hasDevice, debugDevice, visibilityAtlas]);

  if (!visible) return null;

  if (!hasAtlas) {
    return (
      <div className={className} style={PANEL_STYLE} role="region" aria-label="DDGI Atlas Viewer">
        <div style={{ fontWeight: 'bold', marginBottom: 4 }}>DDGI Atlas Viewer</div>
        <div style={WARN_STYLE}>
          Requires <code>engine.debug.atlasTexture()</code> — backend hasn&apos;t
          wired the debug surface.
        </div>
      </div>
    );
  }

  if (!hasDevice || debugDevice == null || irradianceAtlas == null) {
    const reason = !hasDevice
      ? 'GPU debug-device access is unavailable on this backend.'
      : debugDevice == null
        ? 'The GPU debug device is not currently available.'
        : 'The DDGI irradiance atlas is disabled or has not been initialized.';
    return (
      <div className={className} style={PANEL_STYLE} role="region" aria-label="DDGI Atlas Viewer">
        <div style={{ fontWeight: 'bold', marginBottom: 4 }}>DDGI Atlas Viewer</div>
        <div style={WARN_STYLE}>{reason}</div>
      </div>
    );
  }

  return (
    <div className={className} style={PANEL_STYLE} role="region" aria-label="DDGI Atlas Viewer">
      <div style={{ fontWeight: 'bold', marginBottom: 4 }}>DDGI Atlas Viewer</div>
      <div style={{ fontSize: 10, color: '#aaa', marginBottom: 2 }}>Irradiance</div>
      <canvas ref={irrCanvasRef} style={{ display: 'block', maxWidth: 200, imageRendering: 'pixelated' }} />
      {hasVisibilityAtlas && visibilityAtlas != null ? (
        <>
          <div style={{ fontSize: 10, color: '#aaa', marginTop: 6, marginBottom: 2 }}>Visibility</div>
          <canvas ref={visCanvasRef} style={{ display: 'block', maxWidth: 200, imageRendering: 'pixelated' }} />
        </>
      ) : (
        <div style={{ ...WARN_STYLE, marginTop: 6 }}>
          Visibility atlas unavailable or not initialized.
        </div>
      )}
      <div style={{ marginTop: 6, color: '#666', fontSize: 10 }}>
        Live readback @ ~10 Hz. Reinhard-tonemapped.
      </div>
    </div>
  );
};
