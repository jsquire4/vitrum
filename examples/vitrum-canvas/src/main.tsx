/**
 * VitrumCanvas React example — drop-in component for React app hosts.
 *
 * <VitrumCanvas> owns the engine lifecycle: it creates the engine on mount,
 * disposes on unmount, and re-creates when scene / prefer changes. The
 * quality prop propagates live (no remount) via the internal getter closure.
 *
 * Capture protocol: waits for a 'pt-spp' progress event at targetSpp, with an
 * onFrame fallback for a real-time backend that does not accumulate samples.
 */

import * as React from 'react';
import * as ReactDOM from 'react-dom/client';
// VitrumCanvas is the named export from the /react subpath of @vitrum/engine.
import { VitrumCanvas } from '@vitrum/engine/react';
import type { CameraLike } from '@vitrum/engine';
import type { FrameStats, ProgressStats } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';
import {
  CORNELL_CAMERA_POSITION,
  createAxisAlignedView,
  createPerspectiveProjection,
  writePerspectiveProjection,
} from '../../shared/exampleHost.js';

// ── URL params ─────────────────────────────────────────────────────────────────
const urlParams = new URLSearchParams(location.search);
const targetSpp = Number(urlParams.get('vitrumSpp')) || 128;

// ── Static camera ─────────────────────────────────────────────────────────────
const viewElems = createAxisAlignedView(CORNELL_CAMERA_POSITION);
const projElems = createPerspectiveProjection();
let mountedCanvas: HTMLCanvasElement | null = null;

// Stable ref object — satisfies CameraLike structurally, no THREE import needed.
const staticCamera: CameraLike = {
  updateMatrixWorld() {
    writePerspectiveProjection(
      projElems,
      mountedCanvas?.width ?? 1,
      mountedCanvas?.height ?? 1,
    );
  },
  matrixWorldInverse: { elements: viewElems },
  projectionMatrix:   { elements: projElems },
  position: {
    x: CORNELL_CAMERA_POSITION[0],
    y: CORNELL_CAMERA_POSITION[1],
    z: CORNELL_CAMERA_POSITION[2],
  },
};

// ── Scene (stable reference — no re-render on scene object identity change) ───
const scene = createCornellScene();

// ── App component ─────────────────────────────────────────────────────────────
function App(): React.JSX.Element {
  const captureSignalledRef = React.useRef(false);
  const captureModeRef = React.useRef<'unknown' | 'accumulating'>('unknown');
  const renderedRealtimeFramesRef = React.useRef(0);
  const setCanvasRef = React.useCallback((canvas: HTMLCanvasElement | null): void => {
    mountedCanvas = canvas;
  }, []);

  const handleProgress = React.useCallback((progress: ProgressStats): void => {
    if (captureSignalledRef.current) return;
    // 'pt-spp' events carry accumulated sample count in progress.current.
    if (progress.kind === 'pt-spp' && progress.current >= targetSpp) {
      captureModeRef.current = 'accumulating';
      (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
      captureSignalledRef.current = true;
    } else if (progress.kind === 'pt-spp') {
      captureModeRef.current = 'accumulating';
    }
  }, []);

  const handleFrame = React.useCallback((stats: FrameStats): void => {
    if (
      captureSignalledRef.current ||
      captureModeRef.current === 'accumulating' ||
      (stats.spp ?? 0) <= 0
    ) {
      return;
    }
    renderedRealtimeFramesRef.current += 1;
    if (renderedRealtimeFramesRef.current >= targetSpp) {
      (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
      captureSignalledRef.current = true;
    }
  }, []);
  const handleAttachError = React.useCallback((error: unknown): void => {
    console.error('[VitrumCanvas example] attach failed:', error);
    (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
      error instanceof Error ? error.stack ?? error.message : error,
    );
  }, []);

  return (
    <VitrumCanvas
      ref={setCanvasRef}
      scene={scene}
      camera={staticCamera}
      quality={{ samplesTarget: targetSpp }}
      onFrame={handleFrame}
      onProgress={handleProgress}
      onAttachError={handleAttachError}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  );
}

// ── Mount ─────────────────────────────────────────────────────────────────────
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement,
);
root.render(<App />);
