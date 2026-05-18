/**
 * hero-product-viz — progressive PT product visualizer with material editor.
 *
 * Demonstrates:
 *   - attachVitrum({ prefer:'quality' }) for progressive path tracing
 *   - engine.onProgress() for SPP progress bar
 *   - engine.setScene() + engine.reset() for material-slider updates
 *   - sceneFromThreeJS() for a procedural glass/pedestal scene
 *   - High-res PNG save via an offscreen canvas + fresh engine
 *
 * Host code: <200 LoC (excluding this header comment).
 * No external assets required — scene is 100% procedural.
 */

import * as THREE from 'three';
import { attachVitrum, createEngine, type AttachVitrumHandle } from '@vitrum/engine';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import type { ProgressStats } from '@vitrum/core';
import { parsePositiveInt } from '@vitrum-examples/shared';

// ── Capture protocol globals (consumed by tools/benchmark-runner/capture-adapter-playwright.mjs) ──
//
// When the page is opened with `?vitrumScenario=hero-product-viz&vitrumAutoStart=1`, the
// example skips the interactive flow and renders to convergence, then sets the
// VITRUM_CAPTURE_READY sentinel that the Playwright adapter waits on.
declare global {
  // eslint-disable-next-line no-var
  var VITRUM_CAPTURE_READY: boolean | undefined;
  // eslint-disable-next-line no-var
  var VITRUM_MS_PER_SAMPLE: number | undefined;
  // eslint-disable-next-line no-var
  var VITRUM_CAPTURE_TELEMETRY: Record<string, unknown> | undefined;
  // eslint-disable-next-line no-var
  var VITRUM_CAPTURE_CANVAS_SELECTOR: string | undefined;
}

// ── DOM refs ──────────────────────────────────────────────────────────────────

const canvas      = document.querySelector<HTMLCanvasElement>('#c')!;
const statusEl    = document.querySelector<HTMLDivElement>('#status')!;
const sppLabel    = document.querySelector<HTMLDivElement>('#spp-label')!;
const sppBar      = document.querySelector<HTMLDivElement>('#spp-bar')!;
const btnSave     = document.querySelector<HTMLButtonElement>('#btn-save')!;

// ── Capture-mode config from URL params ──────────────────────────────────────
//
// Capture mode is active when `vitrumScenario` is present. It overrides
// interactive defaults: fixed canvas size, deterministic SPP target, no slider
// wiring (the page never receives user input during capture).
const captureParams = new URLSearchParams(window.location.search);
const captureMode = captureParams.has('vitrumScenario');
// vitrumAutoStart is honoured for protocol parity with cornell-box; hero-product-viz
// already boots on module load, so it's a no-op here aside from documenting intent.
void captureParams.get('vitrumAutoStart');
const captureWidth  = parsePositiveInt(captureParams.get('vitrumWidth'),  1280);
const captureHeight = parsePositiveInt(captureParams.get('vitrumHeight'),  720);
const captureSpp    = parsePositiveInt(captureParams.get('vitrumSpp'),    512);
const captureBounces = parsePositiveInt(captureParams.get('vitrumBounces'), 8);

globalThis.VITRUM_CAPTURE_READY = false;
globalThis.VITRUM_MS_PER_SAMPLE = undefined;
globalThis.VITRUM_CAPTURE_TELEMETRY = undefined;
globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#c';

if (captureMode) {
  // Lock canvas to requested capture size — Playwright will screenshot the
  // locator, so on-screen pixels must match the requested image dimensions.
  canvas.style.width  = `${captureWidth}px`;
  canvas.style.height = `${captureHeight}px`;
  canvas.width  = captureWidth;
  canvas.height = captureHeight;
}

// ── Material params ───────────────────────────────────────────────────────────

const mat = { roughness: 0.05, metallic: 0.0, transmission: 0.95, ior: 1.5 };

// ── Procedural glass + pedestal scene (~30 lines of setup) ───────────────────

function buildScene(): THREE.Scene {
  const scene = new THREE.Scene();

  // Pedestal / shelf — neutral matte base
  const pedestalMat = new THREE.MeshPhysicalMaterial({
    color: 0xf0ede8, roughness: 0.4, metalness: 0.0,
  });
  // Curved shelf: cylinder used as a wide low drum
  const shelf = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.2, 0.15, 48), pedestalMat);
  shelf.position.set(0, -0.075, 0);
  scene.add(shelf);

  // Glass primary: tall sphere with material-editor params
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: 0xd4eaf8,
    roughness:    mat.roughness,
    metalness:    mat.metallic,
    transmission: mat.transmission,
    ior:          mat.ior,
    thickness:    0.4,
    attenuationColor: new THREE.Color(0.85, 0.95, 1.0),
    attenuationDistance: 1.2,
  });
  const glassSphere = new THREE.Mesh(new THREE.SphereGeometry(0.45, 40, 40), glassMat);
  glassSphere.position.set(0, 0.53, 0);
  scene.add(glassSphere);

  // Secondary glass: smaller icosahedron, fixed clear glass
  const clearMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.02, metalness: 0.0,
    transmission: 0.98, ior: 1.52,
    thickness: 0.2,
  });
  const glassIco = new THREE.Mesh(new THREE.IcosahedronGeometry(0.22, 2), clearMat);
  glassIco.position.set(0.75, 0.28, 0.4);
  scene.add(glassIco);

  // Ground plane (infinite-ish diffuse floor)
  const floorMat = new THREE.MeshPhysicalMaterial({ color: 0x2a2a2a, roughness: 0.9 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.15;
  scene.add(floor);

  // Lighting: two area lights for studio look
  const keyLight = new THREE.RectAreaLight(0xfff5e8, 18, 2.5, 2.5);
  keyLight.position.set(-2, 3, 2.5);
  keyLight.lookAt(0, 0.5, 0);
  scene.add(keyLight);

  const fillLight = new THREE.RectAreaLight(0xe8f0ff, 6, 2.0, 2.0);
  fillLight.position.set(2.5, 2, -1.5);
  fillLight.lookAt(0, 0.5, 0);
  scene.add(fillLight);

  return scene;
}

// ── Camera ────────────────────────────────────────────────────────────────────

const camera = new THREE.PerspectiveCamera(
  38,
  captureMode ? captureWidth / captureHeight : 1,
  0.05,
  50,
);
camera.position.set(0, 1.2, 3.0);
camera.lookAt(0, 0.5, 0);

// ── Engine lifecycle ──────────────────────────────────────────────────────────

const SPP_TARGET = captureMode ? captureSpp : 512;
const BOUNCES    = captureMode ? captureBounces : 8;

let engineHandle: AttachVitrumHandle | null = null;
let currentVitrumScene = sceneFromThreeJS(buildScene());
const initTimeMs = performance.now();

function finalizeCapture(samplesAccumulated: number): void {
  if (globalThis.VITRUM_CAPTURE_READY === true) return;
  const elapsed = performance.now() - initTimeMs;
  globalThis.VITRUM_MS_PER_SAMPLE = elapsed / Math.max(samplesAccumulated, 1);
  globalThis.VITRUM_CAPTURE_TELEMETRY = {
    scenarioId: 'hero-product-viz',
    samplesAccumulated,
    msPerSample: globalThis.VITRUM_MS_PER_SAMPLE,
    captureCanvasSelector: '#c',
  };
  globalThis.VITRUM_CAPTURE_READY = true;
  console.info(
    `[vitrum-capture] hero-product-viz ready: ${samplesAccumulated} SPP in ${elapsed.toFixed(0)} ms`,
  );
}

const onProgress = (p: ProgressStats): void => {
  if (p.kind !== 'pt-spp') return;
  const pct = Math.round(p.fraction * 100);
  sppLabel.textContent = `SPP: ${Math.round(p.current)} / ${Math.round(p.target)}`;
  sppBar.style.width = `${pct}%`;
  // Capture-mode: signal ready when convergence reached.
  if (captureMode && p.fraction >= 1) {
    finalizeCapture(p.current);
  }
};

async function init(): Promise<void> {
  statusEl.textContent = 'Starting path tracer…';
  try {
    engineHandle = await attachVitrum({
      canvas,
      scene: currentVitrumScene,
      camera,
      prefer: 'quality',
      quality: {
        samplesTarget: SPP_TARGET,
        bounces: BOUNCES,
        resolutionFactor: 1,
        filteredGlossyFactor: 0.5,
      },
      onProgress,
    });
    statusEl.textContent = captureMode
      ? `Capture mode — rendering ${SPP_TARGET} SPP at ${captureWidth}x${captureHeight}…`
      : 'Rendering… adjust sliders to change material';
  } catch (err) {
    statusEl.textContent = `Engine error: ${String(err)}`;
    if (captureMode) {
      // Surface the failure so capture-suite's telemetry tail picks it up.
      globalThis.VITRUM_CAPTURE_TELEMETRY = { error: String(err) };
      globalThis.VITRUM_CAPTURE_READY = true;
    }
  }
}

// ── Slider wiring ─────────────────────────────────────────────────────────────

let rebuildTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleRebuild(): void {
  // Capture mode locks materials to their defaults — slider input is ignored so
  // a stray pointer event from headless Chromium can't reset accumulation
  // mid-capture.
  if (captureMode) return;
  if (rebuildTimer !== null) clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(() => {
    rebuildTimer = null;
    currentVitrumScene = sceneFromThreeJS(buildScene());
    engineHandle?.engine.setScene(currentVitrumScene);
    engineHandle?.engine.reset();
    sppBar.style.width = '0%';
    sppLabel.textContent = `SPP: 0 / ${SPP_TARGET}`;
    statusEl.textContent = 'Re-rendering with new material…';
  }, 80);
}

function bindSlider(id: string, key: keyof typeof mat, decimals = 2): void {
  const slider = document.querySelector<HTMLInputElement>(`#${id}`)!;
  const valEl  = document.querySelector<HTMLSpanElement>(`#v-${id}`)!;
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    mat[key] = v;
    valEl.textContent = v.toFixed(decimals);
    scheduleRebuild();
  });
}

bindSlider('roughness',    'roughness');
bindSlider('metallic',     'metallic');
bindSlider('transmission', 'transmission');
bindSlider('ior',          'ior');

// ── High-res save ─────────────────────────────────────────────────────────────
//
// Strategy: create a temporary offscreen canvas at 4K (clamped to
// screen.width*4), spin up a fresh pt-webgl engine, run the RAF loop until
// convergence, then call canvas.toBlob(). The main-canvas engine is paused
// during this to avoid GPU contention.
//
// NOTE: 4K renders at 512 SPP typically take 20–60 seconds depending on GPU.
// Be patient — the browser may appear to freeze while the GPU is busy.

btnSave.addEventListener('click', () => { void saveHighRes(); });

async function saveHighRes(): Promise<void> {
  btnSave.disabled = true;
  statusEl.textContent = 'Preparing 4K render… (this may take 30s+)';

  // Pause the live engine to free GPU bandwidth.
  engineHandle?.engine.pause();

  const MAX_DIM = Math.min(screen.width * 4, 3840);
  const aspect  = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
  const saveW   = MAX_DIM;
  const saveH   = Math.round(MAX_DIM / aspect);

  const offscreen = document.createElement('canvas');
  offscreen.width  = saveW;
  offscreen.height = saveH;

  let saveEngine: import('@vitrum/core').Engine | null = null;
  try {
    saveEngine = await createEngine({
      canvas: offscreen,
      scene:  currentVitrumScene,
      prefer: 'quality',
      advanced: {
        maxSamplesPerPixel: SPP_TARGET,
        maxBounces: 8,
      },
    });

    const saveCam = new THREE.PerspectiveCamera(38, saveW / saveH, 0.05, 50);
    saveCam.position.copy(camera.position);
    saveCam.quaternion.copy(camera.quaternion);
    saveCam.updateMatrixWorld();

    // Run frames until convergence.
    await new Promise<void>((resolve) => {
      let frame = 0;
      function tick(): void {
        if (!saveEngine) { resolve(); return; }
        saveCam.updateMatrixWorld();
        const view = new Float32Array(saveCam.matrixWorldInverse.elements);
        const proj = new Float32Array(saveCam.projectionMatrix.elements);
        const out = saveEngine.renderFrame({
          viewMatrix:      view,
          projMatrix:      proj,
          cameraPosition:  [saveCam.position.x, saveCam.position.y, saveCam.position.z],
          viewport:        { width: saveW, height: saveH, devicePixelRatio: 1 },
          frameIndex:      frame,
          frameSeed:       (frame * 1664525 + 1013904223) >>> 0,
          quality:         { samplesTarget: SPP_TARGET, bounces: 8, resolutionFactor: 1, filteredGlossyFactor: 0.5 },
        });
        frame++;
        const spp = Math.round(out.samplesAccumulated);
        statusEl.textContent = `Saving… SPP ${spp} / ${SPP_TARGET} at ${saveW}×${saveH}`;
        if (out.isConverged) { resolve(); return; }
        requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });

    // toBlob requires preserveDrawingBuffer on the WebGL context, which the
    // factory doesn't set. We fall back to toDataURL (synchronous, same output).
    const dataUrl = offscreen.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `vitrum-product-${saveW}x${saveH}.png`;
    a.click();

    statusEl.textContent = `Saved ${saveW}×${saveH} PNG`;
  } catch (err) {
    statusEl.textContent = `Save failed: ${String(err)}`;
  } finally {
    try { saveEngine?.dispose(); } catch { /* ignore */ }
    offscreen.remove();
    engineHandle?.engine.resume();
    btnSave.disabled = false;
  }
}

// ── Resize ────────────────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  camera.aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
  camera.updateProjectionMatrix();
});

// ── Boot ──────────────────────────────────────────────────────────────────────

void init();
