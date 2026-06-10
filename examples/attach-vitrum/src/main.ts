/**
 * attachVitrum example — lifecycle helper that wires RAF loop, ResizeObserver,
 * and camera matrices. The host only supplies a CameraLike and an optional
 * onFrame callback.
 *
 * Capture protocol: sets globalThis.VITRUM_CAPTURE_READY after targetSpp frames.
 *
 * API sharp edges observed while writing this example:
 * - CameraLike is NOT re-exported from '@vitrum/engine' (it is internal to
 *   lifecycle/vanilla.ts). Hosts must either define the structural interface
 *   inline (as done here) or satisfy the type implicitly (TypeScript structural
 *   typing means any object with the four required fields is accepted).
 */

import { attachVitrum } from '@vitrum/engine';
import type { FrameStats } from '@vitrum/core';

// CameraLike is an internal type in @vitrum/engine (not re-exported from the
// public index). Define the structural interface inline — any object with these
// four members satisfies attachVitrum's camera parameter.
interface CameraLike {
  updateMatrixWorld(): void;
  readonly matrixWorldInverse: { readonly elements: ArrayLike<number> };
  readonly projectionMatrix: { readonly elements: ArrayLike<number> };
  readonly position: { readonly x: number; readonly y: number; readonly z: number };
}
import { createCornellScene } from '@vitrum-examples/cornell-scene';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;

// ── Camera — static view, satisfies CameraLike structurally ──────────────────
// Column-major 4×4 matrices stored as Float32Arrays.
// View: camera at (0, -1, -4) looking at origin (Z-forward, Y-up).
const viewElems = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0,-1,-4, 1,
]);

const aspect = window.innerWidth / Math.max(window.innerHeight, 1);
const fovY   = Math.PI / 3;
const near   = 0.1;
const far    = 100;
const f = 1 / Math.tan(fovY / 2);
const projElems = new Float32Array([
  f / aspect, 0, 0,                               0,
  0,          f, 0,                               0,
  0,          0, (far + near) / (near - far),   -1,
  0,          0, (2 * far * near) / (near - far), 0,
]);

const camera: CameraLike = {
  updateMatrixWorld() { /* static camera — no-op */ },
  matrixWorldInverse: { elements: viewElems },
  projectionMatrix:   { elements: projElems },
  position: { x: 0, y: -1, z: -4 },
};

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = createCornellScene();
const canvas = document.getElementById('vitrum-canvas') as HTMLCanvasElement;

// ── Attach ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  let captureSignalled = false;

  const handle = await attachVitrum({
    canvas,
    scene,
    camera,
    onFrame(stats: FrameStats) {
      const spp = stats.spp ?? 0;
      (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
        stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;

      if (!captureSignalled && spp >= targetSpp) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    },
  });

  // Expose handle so DevTools / E2E tests can call handle.dispose().
  (globalThis as Record<string, unknown>).VITRUM_HANDLE = handle;
}

main().catch((err: unknown) => {
  console.error('[attachVitrum example] fatal:', err);
});
