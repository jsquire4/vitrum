/**
 * VitrumCanvas React example — drop-in component for React app hosts.
 *
 * <VitrumCanvas> owns the engine lifecycle: it creates the engine on mount,
 * disposes on unmount, and re-creates when scene / prefer changes. The
 * quality prop propagates live (no remount) via the internal getter closure.
 *
 * Capture protocol: sets globalThis.VITRUM_CAPTURE_READY after onProgress
 * reports a 'pt-spp' event with current >= targetSpp.
 */

import * as React from 'react';
import * as ReactDOM from 'react-dom/client';
// VitrumCanvas is the named export from the /react subpath of @vitrum/engine.
// eslint-disable-next-line import/no-unresolved
import { VitrumCanvas } from '@vitrum/engine/react';
import type { CameraLike } from '@vitrum/engine';
import type { ProgressStats } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';

// ── URL params ─────────────────────────────────────────────────────────────────
const urlParams = new URLSearchParams(location.search);
const targetSpp = Number(urlParams.get('vitrumSpp')) || 128;

// ── Static camera ─────────────────────────────────────────────────────────────
const viewElems = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0,-1,-4, 1,
]);
const winAspect = window.innerWidth / Math.max(window.innerHeight, 1);
const fovY = Math.PI / 3;
const near  = 0.1;
const far   = 100;
const ff = 1 / Math.tan(fovY / 2);
const projElems = new Float32Array([
  ff / winAspect, 0,  0,                                0,
  0,              ff, 0,                                0,
  0,              0,  (far + near) / (near - far),    -1,
  0,              0,  (2 * far * near) / (near - far),  0,
]);

// Stable ref object — satisfies CameraLike structurally, no THREE import needed.
const staticCamera: CameraLike = {
  updateMatrixWorld() { /* static */ },
  matrixWorldInverse: { elements: viewElems },
  projectionMatrix:   { elements: projElems },
  position: { x: 0, y: -1, z: -4 },
};

// ── Scene (stable reference — no re-render on scene object identity change) ───
const scene = createCornellScene();

// ── App component ─────────────────────────────────────────────────────────────
function App(): React.JSX.Element {
  const captureSignalledRef = React.useRef(false);

  const handleProgress = React.useCallback((progress: ProgressStats): void => {
    if (captureSignalledRef.current) return;
    // 'pt-spp' events carry accumulated sample count in progress.current.
    if (progress.kind === 'pt-spp' && progress.current >= targetSpp) {
      (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
      captureSignalledRef.current = true;
    }
  }, []);

  return (
    <VitrumCanvas
      scene={scene}
      camera={staticCamera}
      onProgress={handleProgress}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  );
}

// ── Mount ─────────────────────────────────────────────────────────────────────
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement,
);
root.render(<App />);
