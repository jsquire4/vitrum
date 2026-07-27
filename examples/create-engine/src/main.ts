/**
 * createEngine example — auto-picks backend (walkaround-hybrid when WebGPU
 * is available, pt-webgl2 otherwise) and runs a render loop.
 *
 * Capture protocol: sets globalThis.VITRUM_CAPTURE_READY = true after the
 * engine reports samplesAccumulated >= targetSpp.
 * Reads vitrumSpp and vitrumBounces from URL params for quality control.
 */

import { createEngine } from '@vitrum/engine';
import type { FrameStats } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';

// ── URL params (capture protocol) ────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp = Number(params.get('vitrumSpp')) || 128;

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = createCornellScene();

// ── Canvas ────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('vitrum-canvas') as HTMLCanvasElement;

// ── Camera (static Cornell-box view) ──────────────────────────────────────────
// Camera positioned at Z=4, looking toward Z=-1 (the scene), Y-up.
const viewMatrix = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0,-1,-4, 1,  // translate: x=0, y=-1 (center box vertically), z=-4
]));

// Perspective projection: ~60° vertical FOV, near=0.1, far=100.
const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
const fovY   = Math.PI / 3;
const near   = 0.1;
const far    = 100;
const f = 1 / Math.tan(fovY / 2);
const projMatrix = asMat4(new Float32Array([
  f / aspect, 0, 0,                              0,
  0,          f, 0,                              0,
  0,          0, (far + near) / (near - far),  -1,
  0,          0, (2 * far * near) / (near - far), 0,
]));

// ── Engine ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  const engine = await createEngine({ canvas, scene });

  let frameIndex       = 0;
  let captureSignalled = false;
  let lastSpp          = 0;

  // Track samples and timing via the engine's onFrame subscription.
  engine.onFrame?.((stats: FrameStats) => {
    lastSpp = stats.spp ?? lastSpp;
    (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
      stats.frameTimeMs > 0 && lastSpp > 0 ? stats.frameTimeMs / lastSpp : 0;
  });

  function tick(): void {
    const width  = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    canvas.width  = width;
    canvas.height = height;

    const output = engine.renderFrame({
      viewMatrix,
      projMatrix,
      cameraPosition: [0, 1, 4],
      viewport:  { width, height, devicePixelRatio: 1 },
      frameIndex,
      frameSeed: (frameIndex * 1664525 + 1013904223) >>> 0,
    });

    if (output.kind === 'rendered') {
      frameIndex++;
      const spp = output.samplesAccumulated;
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
  console.error('[createEngine example] fatal:', err);
});
