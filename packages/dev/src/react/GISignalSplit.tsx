// GISignalSplit — side-by-side direct / indirect / AO viewer.
//
// W12 implementation: subscribes to a requestAnimationFrame tick and calls
// `engine.debug?.giSignalReadback?.()` (added in W12; HybridEngine implements
// it via copyTextureToBuffer + binary16 unpack against the FrameResources
// hdrColorTexture / hdrIndirectTexture / aoFullTexture).
//
// Each channel is tonemapped with `pow(x, 1/2.2)` clamped to [0,1] (per the
// task brief) and drawn into a dedicated <canvas>. The three canvases live
// in a one-row 3-column grid; the canvases scale to fit the overlay panel.
//
// Readback rate: capped to `readbackRate` Hz (default 4). The shade pass
// runs every frame so the channel textures are always fresh; we deliberately
// don't try to render every frame.
//
// The "total" channel from `giSignalTextures()` is intentionally null in
// HybridEngine (the final image lives in the swap-chain, not a persistent
// texture). We display three channels, not four.

import React, {
  type CSSProperties,
  type FC,
  useEffect,
  useRef,
  useState,
} from 'react';
import type { DebuggableEngine } from '../types.js';

export interface GISignalSplitProps {
  /** The engine to inspect. */
  engine: DebuggableEngine;
  /** Whether the split view is active. Default: false. */
  active?: boolean;
  /** Callback when the user toggles the split view. */
  onToggle?: (active: boolean) => void;
  /** Maximum readback frequency in Hz. Default: 4. */
  readbackRate?: number;
  /** Display gamma applied to tonemap. Default: 2.2. */
  gamma?: number;
  /** CSS class name applied to the container. */
  className?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// Styles
// ────────────────────────────────────────────────────────────────────────────

const BUTTON_STYLE: CSSProperties = {
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

const PANEL_STYLE: CSSProperties = {
  position: 'absolute',
  bottom: 40,
  left: 8,
  background: 'rgba(0,0,0,0.85)',
  color: '#e0e0e0',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '8px 10px',
  borderRadius: 4,
  zIndex: 9997,
  display: 'grid',
  gridTemplateColumns: 'repeat(3, 1fr)',
  gap: 6,
  width: 'min(720px, 80vw)',
};

const QUAD_STYLE: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
};

const QUAD_LABEL_STYLE: CSSProperties = {
  color: '#888',
  fontSize: 10,
  marginBottom: 2,
};

const CANVAS_STYLE: CSSProperties = {
  display: 'block',
  imageRendering: 'pixelated',
  background: '#000',
  border: '1px solid #333',
  maxWidth: '100%',
  width: '100%',
  height: 'auto',
};

const WARN_STYLE: CSSProperties = {
  color: '#ffb347',
  fontStyle: 'italic',
  fontSize: 10,
  gridColumn: '1 / -1',
};

// ────────────────────────────────────────────────────────────────────────────
// Tonemap (RGB float32 → ImageData) — same family as DDGIAtlasViewer but
// kept local to avoid coupling component internals.
// ────────────────────────────────────────────────────────────────────────────

function tonemapToImageData(
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

function paintCanvas(
  canvas: HTMLCanvasElement | null,
  rgb: Float32Array | null,
  width: number,
  height: number,
  gamma: number,
): void {
  if (canvas == null) return;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (ctx == null) return;
  if (rgb == null) {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, width, height);
    return;
  }
  ctx.putImageData(tonemapToImageData(rgb, width, height, gamma), 0, 0);
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

const CHANNELS = ['direct', 'indirect', 'ao'] as const;
type ChannelKey = (typeof CHANNELS)[number];

export const GISignalSplit: FC<GISignalSplitProps> = ({
  engine,
  active: activeProp,
  onToggle,
  readbackRate = 4,
  gamma = 2.2,
  className,
}) => {
  const [internalActive, setInternalActive] = useState(false);
  const isControlled = activeProp !== undefined;
  const active = isControlled ? activeProp : internalActive;

  const directRef = useRef<HTMLCanvasElement | null>(null);
  const indirectRef = useRef<HTMLCanvasElement | null>(null);
  const aoRef = useRef<HTMLCanvasElement | null>(null);

  const [dims, setDims] = useState<{ width: number; height: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [channelPresence, setChannelPresence] = useState<Record<ChannelKey, boolean>>({
    direct: false,
    indirect: false,
    ao: false,
  });

  const hasDebug = typeof engine.debug?.giSignalReadback === 'function';

  const toggle = (): void => {
    const next = !active;
    if (!isControlled) setInternalActive(next);
    onToggle?.(next);
  };

  useEffect(() => {
    if (!active || !hasDebug) return;
    let rafId: number | null = null;
    let cancelled = false;
    let inFlight = false;
    let lastReadbackTime = 0;
    const minIntervalMs = 1000 / Math.max(0.1, readbackRate);

    const tick = (): void => {
      if (cancelled) return;
      const now = performance.now();
      if (!inFlight && now - lastReadbackTime >= minIntervalMs) {
        inFlight = true;
        lastReadbackTime = now;
        Promise.resolve()
          .then(() => engine.debug?.giSignalReadback?.() ?? null)
          .then((result) => {
            if (cancelled) return;
            if (result == null) return;
            setDims({ width: result.width, height: result.height });
            setChannelPresence({
              direct:   result.direct   != null,
              indirect: result.indirect != null,
              ao:       result.ao       != null,
            });
            paintCanvas(directRef.current,   result.direct,   result.width, result.height, gamma);
            paintCanvas(indirectRef.current, result.indirect, result.width, result.height, gamma);
            paintCanvas(aoRef.current,       result.ao,       result.width, result.height, gamma);
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
  }, [engine, active, hasDebug, readbackRate, gamma]);

  const channelCanvasRef = (k: ChannelKey): React.RefObject<HTMLCanvasElement> => {
    if (k === 'direct')   return directRef as React.RefObject<HTMLCanvasElement>;
    if (k === 'indirect') return indirectRef as React.RefObject<HTMLCanvasElement>;
    return aoRef as React.RefObject<HTMLCanvasElement>;
  };

  return (
    <>
      {active && (
        <div style={PANEL_STYLE} className={className} role="region" aria-label="GI signal split">
          {!hasDebug && (
            <div style={WARN_STYLE}>
              Engine does not implement <code>debug.giSignalReadback()</code>.
            </div>
          )}
          {hasDebug && error !== null && (
            <div style={WARN_STYLE}>readback error: {error}</div>
          )}
          {hasDebug && dims === null && error === null && (
            <div style={{ ...WARN_STYLE, color: '#888' }}>waiting for first frame…</div>
          )}
          {hasDebug && dims !== null && CHANNELS.map((k) => (
            <div key={k} style={QUAD_STYLE}>
              <div style={QUAD_LABEL_STYLE}>
                {k}{channelPresence[k] ? '' : ' (n/a)'}
              </div>
              <canvas
                ref={channelCanvasRef(k)}
                style={CANVAS_STYLE}
                aria-label={`GI ${k} radiance`}
              />
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
