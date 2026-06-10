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

import { createProgressiveEngine } from '@vitrum/engine';
import { asMat4 } from '@vitrum/core';
import type { HandoffFrameResult } from '@vitrum/engine';
import { createCornellScene } from '@vitrum-examples/cornell-scene';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;

// ── Static camera matrices ────────────────────────────────────────────────────
const viewMatrix = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0,-1,-4, 1,
]));

const aspect = window.innerWidth / Math.max(window.innerHeight, 1);
const fovY   = Math.PI / 3;
const near   = 0.1;
const far    = 100;
const f = 1 / Math.tan(fovY / 2);
const projMatrix = asMat4(new Float32Array([
  f / aspect, 0, 0,                               0,
  0,          f, 0,                               0,
  0,          0, (far + near) / (near - far),   -1,
  0,          0, (2 * far * near) / (near - far), 0,
]));

// ── App ───────────────────────────────────────────────────────────────────────
const canvas      = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const phaseLabel  = document.getElementById('phase') as HTMLDivElement;
const scene       = createCornellScene();

async function main(): Promise<void> {
  let handle: Awaited<ReturnType<typeof createProgressiveEngine>>;
  try {
    handle = await createProgressiveEngine({ canvas, scene });
  } catch (err) {
    console.error('[progressive] createProgressiveEngine failed:', err);
    phaseLabel.textContent = 'WebGPU unavailable';
    return;
  }

  let frameIndex       = 0;
  let captureSignalled = false;

  function tick(): void {
    const width  = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    canvas.width  = width;
    canvas.height = height;

    const result: HandoffFrameResult = handle.coordinator.frame({
      viewMatrix,
      projMatrix,
      cameraPosition: [0, -1, -4],
      viewport:  { width, height, devicePixelRatio: 1 },
      frameIndex,
      frameSeed: (frameIndex * 1664525 + 1013904223) >>> 0,
    });

    phaseLabel.textContent = `phase: ${result.phase}`;
    frameIndex++;

    // 'converging' = converged engine is the active renderer (result.output
    //  is the converged engine's FrameOutput, not the real-time engine's).
    if (result.phase === 'converging') {
      const spp = result.output.samplesAccumulated;
      if (!captureSignalled && spp >= targetSpp) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

main().catch((err: unknown) => {
  console.error('[progressive example] fatal:', err);
});
