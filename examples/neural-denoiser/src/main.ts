/**
 * neural-denoiser — W10 demo: walkaround-hybrid with selectable denoiser.
 *
 * Drives `@vitrum/walkaround-hybrid`'s `HybridEngine` with one of three
 * denoiser modes selectable via the `?denoiser=…` URL param:
 *
 *   ?denoiser=atrous-variance  — Welford temporal accumulator + à-trous
 *                                (default; current production mode).
 *   ?denoiser=svgf-real        — real Schied 2017 SVGF (T2.H1).
 *   ?denoiser=neural           — U-Net neural denoiser (T2.H2 / W10).
 *                                Requires `neuralWeights`; this example
 *                                synthesises deterministic-random He-init
 *                                weights via `buildRandomWeightsForSpec`
 *                                — the output will NOT be visually clean
 *                                until real trained weights are supplied
 *                                (see package README for the format).
 *
 * No swap-chain dependency on a specific denoiser: HybridEngine owns the
 * post-shade denoise chain; the host just selects the mode.
 *
 * Cornell scene is intentionally noisy (1 sample/frame; no extra accumulation
 * tricks) so the denoiser has something to denoise.
 */

import { asBackendTexture, asBackendTextureFormat, type FrameInput } from '@vitrum/core';
import { buildCornellBoxThreeScene, mat4FromThree, resizeCanvasToDisplaySize } from '@vitrum-examples/shared';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import {
  buildRandomWeightsForSpec,
  createWalkaroundEngine_Hybrid,
  HYBRID_WEBGPU_REQUIRED_FEATURES,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  WALKAROUND_DENOISER_UNET_SPEC,
  type HybridEngineOptions,
  type ModelWeights,
} from '@vitrum/walkaround-hybrid';
import * as THREE from 'three';

type DenoiserMode = NonNullable<HybridEngineOptions['denoiser']>;

function parseDenoiserMode(raw: string | null): DenoiserMode {
  if (raw === 'atrous' || raw === 'atrous-variance' || raw === 'svgf-real' || raw === 'neural') {
    return raw;
  }
  return 'atrous-variance';
}

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#c');
  const statusEl = document.querySelector<HTMLDivElement>('#status');
  const panelTitle = document.querySelector<HTMLHeadingElement>('#panel-title');
  if (!canvas || !statusEl) throw new Error('missing #c or #status');

  const params = new URLSearchParams(window.location.search);
  const denoiser = parseDenoiserMode(params.get('denoiser'));

  // Highlight the active control pill.
  const pills = Array.from(document.querySelectorAll<HTMLAnchorElement>('.controls a[data-mode]'));
  for (const a of pills) {
    if (a.dataset['mode'] === denoiser) {
      a.style.background = '#2a2a30';
      a.style.color = '#fff';
    }
  }
  if (panelTitle) panelTitle.textContent = `walkaround-hybrid — denoiser='${denoiser}'`;

  const setStatus = (msg: string): void => { statusEl.textContent = msg; };

  if (!navigator.gpu) {
    setStatus('WebGPU not available in this browser — neural-denoiser demo requires WebGPU.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    setStatus('No WebGPU adapter found.');
    return;
  }

  const tsAvailable = adapter.features.has('timestamp-query');
  const requiredFeatures: GPUFeatureName[] = [
    ...HYBRID_WEBGPU_REQUIRED_FEATURES.filter((f) => adapter.features.has(f)),
    ...(tsAvailable ? (['timestamp-query'] as GPUFeatureName[]) : []),
  ];
  const device = await adapter.requestDevice({
    requiredLimits: HYBRID_WEBGPU_REQUIRED_LIMITS,
    ...(requiredFeatures.length > 0 ? { requiredFeatures } : {}),
  });
  const format = navigator.gpu.getPreferredCanvasFormat();
  const ctx = canvas.getContext('webgpu');
  if (!ctx) {
    setStatus('canvas.getContext("webgpu") failed.');
    return;
  }
  function configureCtx(): void {
    if (!canvas || !ctx) return;
    resizeCanvasToDisplaySize(canvas);
    ctx.configure({ device, format, alphaMode: 'premultiplied' });
  }
  configureCtx();

  // Camera + scene.
  const camera = new THREE.PerspectiveCamera(40, canvas.width / Math.max(canvas.height, 1), 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  const threeScene = buildCornellBoxThreeScene();
  const vitrumScene = sceneFromThreeJS(threeScene);

  // Build neural weights when requested. Random He-init weights — denoising
  // output will be visually noisy until a real trained .vitrum-model is
  // supplied via fetch+loadWeightsFromArrayBuffer (see package README).
  let neuralWeights: ModelWeights | undefined;
  if (denoiser === 'neural') {
    setStatus('Synthesising random U-Net weights for pipeline smoke-test…');
    neuralWeights = buildRandomWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC);
    console.info(
      `[neural-denoiser-demo] generated ${neuralWeights.layers.length} layers of random ` +
      `He-init weights (param count: ${WALKAROUND_DENOISER_UNET_SPEC.paramCount}). ` +
      `Output will not be visually clean — load a real .vitrum-model for that.`,
    );
  }

  // Primary light dir (normalised).
  const rawDir: [number, number, number] = [0.35, 1, 0.2];
  const len = Math.hypot(rawDir[0], rawDir[1], rawDir[2]);
  const primaryLightDir: [number, number, number] = [rawDir[0] / len, rawDir[1] / len, rawDir[2] / len];

  setStatus(`Creating HybridEngine (denoiser='${denoiser}')…`);
  const engineOpts: HybridEngineOptions = {
    device,
    width: canvas.width,
    height: canvas.height,
    threeScene,
    primaryLightDir,
    // Cornell box uses one rect-area light — no directional sun.
    primaryLightIntensity: 0,
    skyTint: [0.55, 0.72, 1.0],
    skyIrradiance: 0,
    isSceneReady: () => true,
    denoiser,
    ...(neuralWeights ? { neuralWeights } : {}),
  };

  let hybrid: Awaited<ReturnType<typeof createWalkaroundEngine_Hybrid>>;
  try {
    hybrid = await createWalkaroundEngine_Hybrid(engineOpts);
  } catch (err) {
    setStatus(`Engine construction failed: ${String(err)}`);
    console.error('[neural-denoiser-demo] engine construction failed', err);
    return;
  }
  hybrid.setScene(vitrumScene);

  // Wait for engine 'ready' (the WGSL pipeline compilation is async).
  const readyDeadline = performance.now() + 30_000;
  while (hybrid.state !== 'ready' && hybrid.state !== 'disposed' && performance.now() < readyDeadline) {
    await new Promise<void>((r) => setTimeout(r, 40));
  }
  if (hybrid.state !== 'ready') {
    setStatus(`Engine never reached 'ready' (state=${hybrid.state}).`);
    return;
  }
  setStatus(`Rendering (denoiser='${denoiser}') — drag to orbit not wired in this demo.`);

  // RAF loop.
  let frame = 0;
  let lastW = canvas.width, lastH = canvas.height;
  function loop(): void {
    if (!canvas || !ctx) return;
    if (canvas.clientWidth !== lastW / window.devicePixelRatio ||
        canvas.clientHeight !== lastH / window.devicePixelRatio) {
      configureCtx();
      lastW = canvas.width;
      lastH = canvas.height;
    }
    camera.aspect = canvas.width / Math.max(canvas.height, 1);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    const view = ctx.getCurrentTexture().createView();
    const input: FrameInput = {
      viewMatrix: mat4FromThree(camera.matrixWorldInverse),
      projMatrix: mat4FromThree(camera.projectionMatrix),
      cameraPosition: [camera.position.x, camera.position.y, camera.position.z],
      viewport: {
        width: canvas.width,
        height: canvas.height,
        devicePixelRatio: window.devicePixelRatio,
      },
      frameIndex: frame,
      frameSeed: (frame * 1664525 + 1013904223) >>> 0,
      swapChainView: asBackendTexture<'webgpu', GPUTextureView>(view),
      swapChainFormat: asBackendTextureFormat<'webgpu', GPUTextureFormat>(format),
      quality: { bounces: 4 },
    };
    hybrid.renderFrame(input);
    frame++;
    if (frame % 30 === 0) {
      setStatus(`denoiser='${denoiser}' · frame ${frame}`);
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  window.addEventListener('resize', configureCtx);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
