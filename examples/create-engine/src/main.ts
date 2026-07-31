/**
 * createEngine example — explicitly constructs the auto-selected backend, then
 * hands that engine to attachVitrum for backend-correct presentation and cadence.
 *
 * Capture protocol: sets globalThis.VITRUM_CAPTURE_READY = true after the
 * an accumulating engine reaches targetSpp, or after the equivalent number of
 * real-time frames for a resample-every-frame backend.
 * Reads vitrumSpp from the URL for capture quality control.
 */

import { attachVitrum, createEngine, type CameraLike } from '@vitrum/engine';
import type { FrameStats, ProgressStats } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';
import {
  CORNELL_CAMERA_POSITION,
  createAxisAlignedView,
  createPerspectiveProjection,
  syncCanvasToDisplaySize,
  writePerspectiveProjection,
} from '../../shared/exampleHost.js';

// ── URL params (capture protocol) ────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = createCornellScene();

// ── Canvas ────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const initialViewport = syncCanvasToDisplaySize(canvas);

// ── Camera (static Cornell-box view) ──────────────────────────────────────────
// Camera positioned at Z=4, looking toward Z=-1 (the scene), Y-up.
const viewMatrix = createAxisAlignedView(CORNELL_CAMERA_POSITION);

// Perspective projection: ~60° vertical FOV, near=0.1, far=100.
const projMatrix = createPerspectiveProjection(initialViewport.width, initialViewport.height);
const camera: CameraLike = {
  updateMatrixWorld() {
    writePerspectiveProjection(projMatrix, canvas.width, canvas.height);
  },
  matrixWorldInverse: { elements: viewMatrix },
  projectionMatrix: { elements: projMatrix },
  position: {
    x: CORNELL_CAMERA_POSITION[0],
    y: CORNELL_CAMERA_POSITION[1],
    z: CORNELL_CAMERA_POSITION[2],
  },
};

// ── Engine ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  const engine = await createEngine({ canvas, scene });
  let captureSignalled = false;
  let renderedRealtimeFrames = 0;
  let accumulates = true;

  // createEngine owns backend selection; attachVitrum owns the backend-specific
  // presentation path, resize propagation, and RAF cadence for the already-built
  // engine. A generic manual loop cannot correctly present every backend.
  const handle = await attachVitrum({
    canvas,
    scene,
    camera,
    engine,
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
  (globalThis as Record<string, unknown>).VITRUM_HANDLE = handle;
}

main().catch((err: unknown) => {
  console.error('[createEngine example] fatal:', err);
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
    err instanceof Error ? err.stack ?? err.message : err,
  );
});
