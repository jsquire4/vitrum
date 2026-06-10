/**
 * pt-webgl2-direct — host acquires a WebGL2 context and drives createPTEngine_WebGL2.
 *
 * Demonstrates the backend-direct usage pattern: the host owns the
 * WebGL2RenderingContext lifecycle and passes it to the factory. The engine
 * allocates GL resources against it but never destroys the context.
 *
 * The factory signature:
 *   createPTEngine_WebGL2(opts: PTEngineWebGL2Options): Promise<Engine & PTEngineWebGL2Surface>
 *
 * Required opts: device (WebGL2RenderingContext).
 * Optional: maxBounces, maxSamplesPerPixel, spectral, bdpt.
 *
 * Capture protocol: sets VITRUM_CAPTURE_READY after targetSpp samples.
 *
 * API sharp edges observed while writing this example:
 * - The engine renders to the canvas's WebGL2 context directly. The host must
 *   NOT call gl.clear() or bind framebuffers between renderFrame() calls —
 *   the engine owns the full GL state per frame.
 * - FrameInput.viewport must match the actual canvas pixel dimensions or
 *   rendering will be clipped / stretched. Use canvas.width / canvas.height
 *   (backing store size), not clientWidth / clientHeight (CSS size).
 * - There is no built-in ResizeObserver; the host must resize the canvas
 *   and call renderFrame with the updated viewport. The engine does not cache
 *   the previous viewport size.
 * - FrameInput.frameSeed is required for correct MC sampling. Omitting it
 *   produces correlated samples (the same sample per frame).
 * - FrameStats (frameTimeMs, spp) are available via engine.onFrame, NOT in
 *   the FrameOutput returned by renderFrame. renderFrame returns
 *   samplesAccumulated and isConverged only.
 */

import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl2';
import type { FrameStats } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp  = Number(params.get('vitrumSpp'))    || 128;
const maxBounces = Number(params.get('vitrumBounces')) || 8;

// ── Canvas + WebGL2 context ───────────────────────────────────────────────────
const canvas   = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const sppLabel = document.getElementById('spp') as HTMLDivElement;

// HOST owns the WebGL2RenderingContext — pass it to the factory, never destroy it.
// The guard throws at module scope; the non-null assertion satisfies TypeScript
// for uses inside async function closures where control-flow narrowing lapse.
const glOrNull = canvas.getContext('webgl2');
if (glOrNull == null) {
  sppLabel.textContent = 'WebGL2 unavailable';
  throw new Error('[pt-webgl2-direct] WebGL2 is not available in this browser.');
}
const gl: WebGL2RenderingContext = glOrNull;

// ── Camera (static Cornell-box view) ──────────────────────────────────────────
const viewMatrix = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0,-1,-4, 1,
]));

const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
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

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = createCornellScene();

// ── Engine ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  const engine = await createPTEngine_WebGL2({
    device: gl,
    maxBounces,
    maxSamplesPerPixel: targetSpp,
  });

  await engine.setScene(scene);

  let frameIndex       = 0;
  let captureSignalled = false;

  // FrameStats (frameTimeMs, spp) come from the engine.onFrame subscription,
  // not from the renderFrame() return value.
  engine.onFrame?.((stats: FrameStats) => {
    const spp = stats.spp ?? 0;
    sppLabel.textContent = `spp: ${spp}`;
    (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
      stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;
  });

  function tick(): void {
    // Sync backing store to CSS size (host responsibility in direct mode).
    const width  = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width  = width;
      canvas.height = height;
    }

    const output = engine.renderFrame({
      viewMatrix,
      projMatrix,
      cameraPosition: [0, -1, -4],
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
  console.error('[pt-webgl2-direct] fatal:', err);
  sppLabel.textContent = `error: ${String(err)}`;
});
