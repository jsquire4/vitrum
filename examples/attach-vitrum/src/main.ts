/**
 * attachVitrum example — lifecycle helper that wires RAF loop, ResizeObserver,
 * and camera matrices. The host only supplies a CameraLike and an optional
 * onFrame callback.
 *
 * Capture protocol: waits for targetSpp accumulated PT samples, or the same
 * number of rendered frames when the auto-selected backend is real-time.
 */

import { attachVitrum, type CameraLike } from '@vitrum/engine';
import type { FrameStats, ProgressStats } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';
import {
  CORNELL_CAMERA_POSITION,
  createAxisAlignedView,
  createPerspectiveProjection,
  writePerspectiveProjection,
} from '../../shared/exampleHost.js';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;

// ── Camera — static view, satisfies CameraLike structurally ──────────────────
const canvas = document.getElementById('vitrum-canvas') as HTMLCanvasElement;

// Column-major 4×4 matrices stored as Float32Arrays.
// View: camera at (0, 1, 4), centred vertically on the Cornell box.
const viewElems = createAxisAlignedView(CORNELL_CAMERA_POSITION);
const projElems = createPerspectiveProjection();

const camera: CameraLike = {
  updateMatrixWorld() {
    // attachVitrum owns CSS-to-physical resize. Refresh the projection from the
    // resulting backing store before every frame so browser resizes cannot
    // leave a stale aspect ratio.
    writePerspectiveProjection(projElems, canvas.width, canvas.height);
  },
  matrixWorldInverse: { elements: viewElems },
  projectionMatrix:   { elements: projElems },
  position: {
    x: CORNELL_CAMERA_POSITION[0],
    y: CORNELL_CAMERA_POSITION[1],
    z: CORNELL_CAMERA_POSITION[2],
  },
};

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = createCornellScene();

// ── Attach ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  let captureSignalled = false;
  let renderedRealtimeFrames = 0;
  let accumulates = true;

  const handle = await attachVitrum({
    canvas,
    scene,
    camera,
    quality: { samplesTarget: targetSpp },
    onFrame(stats: FrameStats) {
      const spp = stats.spp ?? 0;
      (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
        stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;

      if (
        !accumulates &&
        spp > 0 &&
        !captureSignalled &&
        ++renderedRealtimeFrames >= targetSpp
      ) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    },
    onProgress(progress: ProgressStats) {
      if (
        progress.kind === 'pt-spp' &&
        !captureSignalled &&
        progress.current >= targetSpp
      ) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    },
  });
  accumulates = handle.engine.capabilities.accumulates;

  // Expose handle so DevTools / E2E tests can call handle.dispose().
  (globalThis as Record<string, unknown>).VITRUM_HANDLE = handle;
}

main().catch((err: unknown) => {
  console.error('[attachVitrum example] fatal:', err);
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
    err instanceof Error ? err.stack ?? err.message : err,
  );
});
