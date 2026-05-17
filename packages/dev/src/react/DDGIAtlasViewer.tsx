// DDGIAtlasViewer — live DDGI irradiance + visibility atlas viewer.
//
// W12 implementation: subscribes to a requestAnimationFrame tick and calls
// `engine.debug?.ddgiAtlasReadback?.()` (added in W12; HybridEngine implements
// it via copyTextureToBuffer + binary16 unpack). The returned RGB Float32
// arrays are tonemapped with sRGB-ish gamma and written into two `<canvas>`
// elements stacked vertically inside the panel.
//
// Readback rate: capped to `readbackRate` per second (default 4 — twelve
// times less work than full-rate). Atlases evolve slowly, so this is more
// than enough for a "live" view and keeps the readback off the critical
// path. Each call allocates and destroys a staging GPUBuffer; see
// `HybridEngine._readbackRgba16fToRgbFloat32`.
//
// The component is fully self-contained: pass `engine` and it renders.
// When `engine.debug.ddgiAtlasReadback` is absent (non-HybridEngine backends),
// the panel renders an "unsupported" notice instead of warning every render.

import React, {
  type CSSProperties,
  type FC,
  useEffect,
  useRef,
  useState,
} from 'react';
import type { DebuggableEngine } from '../types.js';

export interface DDGIAtlasViewerProps {
  /** The engine to inspect. */
  engine: DebuggableEngine;
  /** Whether the panel is visible. Default: true. */
  visible?: boolean;
  /** Maximum readback frequency in Hz. Default: 4 (every ~250 ms). */
  readbackRate?: number;
  /**
   * Maximum width of either atlas canvas in CSS px. The atlas is scaled
   * to fit (no upscaling above its natural pixel size). Default: 256.
   */
  maxCanvasWidth?: number;
  /** Display gamma applied to readback HDR values. Default: 2.2. */
  gamma?: number;
  /** CSS class name applied to the root panel div. */
  className?: string;
}

const PANEL_STYLE: CSSProperties = {
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
  lineHeight: 1.5,
};

const HEADER_STYLE: CSSProperties = { fontWeight: 'bold', marginBottom: 6 };
const SUBLABEL_STYLE: CSSProperties = { color: '#888', fontSize: 10, marginTop: 4 };
const WARN_STYLE: CSSProperties = { color: '#ffb347', fontStyle: 'italic' };
const CANVAS_STYLE: CSSProperties = {
  display: 'block',
  imageRendering: 'pixelated',
  marginTop: 4,
  background: '#000',
};

// ────────────────────────────────────────────────────────────────────────────
// Tonemap helper — RGB Float32 atlas → ImageData (sRGB-ish display).
// ────────────────────────────────────────────────────────────────────────────

function tonemapRgbToImageData(
  rgb: Float32Array,
  width: number,
  height: number,
  gamma: number,
): ImageData {
  const out = new Uint8ClampedArray(width * height * 4);
  const invG = 1 / gamma;
  for (let i = 0; i < width * height; i++) {
    const r = rgb[i * 3    ] ?? 0;
    const g = rgb[i * 3 + 1] ?? 0;
    const b = rgb[i * 3 + 2] ?? 0;
    // Clamp before gamma to avoid NaN on negative HDR values that sometimes
    // appear in firefly pixels pre-clamping.
    const cr = Math.max(0, Math.min(1, Math.pow(Math.max(0, r), invG)));
    const cg = Math.max(0, Math.min(1, Math.pow(Math.max(0, g), invG)));
    const cb = Math.max(0, Math.min(1, Math.pow(Math.max(0, b), invG)));
    out[i * 4    ] = Math.round(cr * 255);
    out[i * 4 + 1] = Math.round(cg * 255);
    out[i * 4 + 2] = Math.round(cb * 255);
    out[i * 4 + 3] = 255;
  }
  return new ImageData(out, width, height);
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export const DDGIAtlasViewer: FC<DDGIAtlasViewerProps> = ({
  engine,
  visible = true,
  readbackRate = 4,
  maxCanvasWidth = 256,
  gamma = 2.2,
  className,
}) => {
  const irrCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const visCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [dims, setDims] = useState<{ width: number; height: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const hasDebug = typeof engine.debug?.ddgiAtlasReadback === 'function';

  useEffect(() => {
    if (!visible || !hasDebug) return;

    let rafId: number | null = null;
    let cancelled = false;
    let inFlight = false;
    let lastReadbackTime = 0;
    const minIntervalMs = 1000 / Math.max(0.1, readbackRate);

    const draw = (rgb: Float32Array, w: number, h: number, canvas: HTMLCanvasElement | null): void => {
      if (canvas == null) return;
      // Size the canvas backing store to the atlas pixel dims; CSS sizing
      // handles fit-to-panel via maxWidth on the inline style.
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (ctx == null) return;
      const imageData = tonemapRgbToImageData(rgb, w, h, gamma);
      ctx.putImageData(imageData, 0, 0);
    };

    const tick = (): void => {
      if (cancelled) return;
      const now = performance.now();
      if (!inFlight && now - lastReadbackTime >= minIntervalMs) {
        inFlight = true;
        lastReadbackTime = now;
        Promise.resolve()
          .then(() => engine.debug?.ddgiAtlasReadback?.() ?? null)
          .then((result) => {
            if (cancelled) return;
            if (result == null) {
              // Atlas not yet allocated — keep retrying silently.
              return;
            }
            setDims({ width: result.width, height: result.height });
            draw(result.irradianceData, result.width, result.height, irrCanvasRef.current);
            draw(result.visibilityData, result.width, result.height, visCanvasRef.current);
          })
          .catch((e: unknown) => {
            if (cancelled) return;
            setError(e instanceof Error ? e.message : String(e));
          })
          .finally(() => {
            inFlight = false;
          });
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, [engine, visible, hasDebug, readbackRate, gamma]);

  if (!visible) return null;

  return (
    <div className={className} style={PANEL_STYLE} role="region" aria-label="DDGI Atlas Viewer">
      <div style={HEADER_STYLE}>DDGI Atlas Viewer</div>
      {!hasDebug && (
        <div style={WARN_STYLE}>
          Engine does not implement <code>debug.ddgiAtlasReadback()</code>.
          <br />
          (Non-HybridEngine backends only — HybridEngine ships this.)
        </div>
      )}
      {hasDebug && error !== null && (
        <div style={WARN_STYLE}>readback error: {error}</div>
      )}
      {hasDebug && dims !== null && (
        <>
          <div style={SUBLABEL_STYLE}>irradiance — {dims.width}×{dims.height}</div>
          <canvas
            ref={irrCanvasRef}
            style={{ ...CANVAS_STYLE, maxWidth: maxCanvasWidth, width: '100%' }}
            aria-label="DDGI irradiance atlas"
          />
          <div style={SUBLABEL_STYLE}>visibility — {dims.width}×{dims.height}</div>
          <canvas
            ref={visCanvasRef}
            style={{ ...CANVAS_STYLE, maxWidth: maxCanvasWidth, width: '100%' }}
            aria-label="DDGI visibility atlas"
          />
        </>
      )}
      {hasDebug && dims === null && error === null && (
        <div style={{ color: '#888', fontSize: 10 }}>waiting for DDGI…</div>
      )}
    </div>
  );
};
