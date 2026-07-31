/**
 * Progressive handoff example — createProgressiveEngine().
 *
 * Runs the walkaround-hybrid realtime GI engine while the camera is moving.
 * When the camera settles (no movement for ~6 frames), the coordinator hands
 * off display to the pt-webgpu converged path tracer. The realtime seed is
 * transferred to the converged engine so the first converged frame inherits
 * the walkaround's GI solution (no cold-start black pop).
 *
 * Requires WebGPU. Will log an error on WebGL2-only browsers.
 *
 * Capture protocol: sets VITRUM_CAPTURE_READY once converged phase begins and
 * targetSpp samples are accumulated.
 *
 * API sharp edges observed while writing this example:
 * - HandoffPhase does NOT include 'converged'. The terminal phase is 'converging'
 *   (converged engine is active and accumulating). Checking for 'converged' always
 *   returns false — the correct check is phase === 'converging'.
 * - HandoffFrameResult has no 'convergedOutput' field. The SPP count for the
 *   converged image is in result.output.samplesAccumulated (result.output is the
 *   active engine's FrameOutput regardless of phase).
 * - The 'prerolling' phase is when the converged engine accumulates BEHIND the
 *   still-displayed real-time image. result.behindOutput holds the prerolling
 *   frame for hosts that want alpha cross-fade blending.
 */

import {
  attachVitrum,
  createProgressiveEngine,
  progressiveHandleAsEngine,
  type CameraLike,
} from '@vitrum/engine';
import type { FrameStats, ProgressStats } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';
import {
  CORNELL_CAMERA_POSITION,
  createAxisAlignedView,
  createPerspectiveProjection,
  syncCanvasToDisplaySize,
  writePerspectiveProjection,
} from '../../shared/exampleHost.js';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;

const canvas      = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const phaseLabel  = document.getElementById('phase') as HTMLDivElement;
const scene       = createCornellScene();
const initialViewport = syncCanvasToDisplaySize(canvas);

// ── Static camera matrices ────────────────────────────────────────────────────
const viewMatrix = createAxisAlignedView(CORNELL_CAMERA_POSITION);
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

// ── App ───────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  let progressiveHandle: Awaited<ReturnType<typeof createProgressiveEngine>>;
  try {
    progressiveHandle = await createProgressiveEngine({
      canvas,
      scene,
      convergedOptions: { maxSamplesPerPixel: targetSpp },
    });
  } catch (err) {
    console.error('[progressive] createProgressiveEngine failed:', err);
    phaseLabel.textContent = 'WebGPU unavailable';
    (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
      err instanceof Error ? err.stack ?? err.message : err,
    );
    return;
  }
  const engine = progressiveHandleAsEngine(progressiveHandle);
  let captureSignalled = false;

  // The coordinator intentionally leaves canvas presentation to its host.
  // Adapting it to Engine lets attachVitrum own cadence, resize propagation,
  // swapchain presentation, and the offscreen blit after the PT handoff.
  const lifecycleHandle = await attachVitrum({
    canvas,
    scene,
    camera,
    engine,
    quality: { samplesTarget: targetSpp },
    onFrame(stats: FrameStats) {
      phaseLabel.textContent = `phase: ${progressiveHandle.coordinator.phase}`;
      const spp = stats.spp ?? 0;
      (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
        stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;
    },
    onProgress(progress: ProgressStats) {
      if (
        progress.kind === 'pt-spp' &&
        progressiveHandle.coordinator.phase === 'converging' &&
        !captureSignalled &&
        progress.current >= targetSpp
      ) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    },
  });
  (globalThis as Record<string, unknown>).VITRUM_HANDLE = lifecycleHandle;
}

main().catch((err: unknown) => {
  console.error('[progressive example] fatal:', err);
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
    err instanceof Error ? err.stack ?? err.message : err,
  );
});
