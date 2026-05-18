/**
 * hero-viewer — drop-in glTF viewer using @vitrum/engine.
 *
 * Demonstrates:
 *   - attachVitrum() for lifecycle management + RAF loop
 *   - loadGltfScene() for drag-drop .glb / .gltf loading
 *   - OrbitControls for camera navigation
 *   - Engine preference toggle (realtime GI vs quality PT)
 *
 * Host code: <200 LoC (excluding this header comment).
 * No external assets required — drag your own .glb to render.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { attachVitrum, type AttachVitrumHandle, type EnginePreference } from '@vitrum/engine';
import { loadGltfScene, sceneFromThreeJS } from '@vitrum/three-bindings';

// ── Capture protocol globals (consumed by tools/benchmark-runner/capture-adapter-playwright.mjs) ──
//
// In capture mode (`?vitrumScenario=hero-viewer&vitrumAutoStart=1`) the viewer
// skips drag-drop and instead boots a built-in procedural fallback scene so
// that headless Playwright runs do not require an external glTF asset.
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

function parsePositiveInt(raw: string | null, dflt: number): number {
  if (!raw) return dflt;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : dflt;
}
const captureParams = new URLSearchParams(window.location.search);
const captureMode = captureParams.has('vitrumScenario');
const captureWidth  = parsePositiveInt(captureParams.get('vitrumWidth'),  1280);
const captureHeight = parsePositiveInt(captureParams.get('vitrumHeight'),  720);
const captureSpp    = parsePositiveInt(captureParams.get('vitrumSpp'),    256);
const capturePrefer: EnginePreference =
  captureParams.get('vitrumPrefer') === 'quality' ? 'quality' : 'realtime';

globalThis.VITRUM_CAPTURE_READY = false;
globalThis.VITRUM_MS_PER_SAMPLE = undefined;
globalThis.VITRUM_CAPTURE_TELEMETRY = undefined;
globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#c';

// ── DOM refs ──────────────────────────────────────────────────────────────────

const canvas     = document.querySelector<HTMLCanvasElement>('#c')!;
const overlay    = document.querySelector<HTMLDivElement>('#overlay')!;
const dropZone   = document.querySelector<HTMLDivElement>('#drop-zone')!;
const fileInput  = document.querySelector<HTMLInputElement>('#file-input')!;
const fileInput2 = document.querySelector<HTMLInputElement>('#file-input-2')!;
const controls2  = document.querySelector<HTMLDivElement>('#controls')!;
const statusEl   = document.querySelector<HTMLDivElement>('#status')!;
const btnRealtime = document.querySelector<HTMLButtonElement>('#btn-realtime')!;
const btnQuality  = document.querySelector<HTMLButtonElement>('#btn-quality')!;

// ── State ─────────────────────────────────────────────────────────────────────

let handle: AttachVitrumHandle | null = null;
let prefer: EnginePreference = 'realtime';

// ── Three.js camera + orbit controls ─────────────────────────────────────────

const camera = new THREE.PerspectiveCamera(
  45,
  captureMode
    ? captureWidth / captureHeight
    : window.innerWidth / window.innerHeight,
  0.01,
  1000,
);
camera.position.set(0, 1, 3);

if (captureMode) {
  // Lock canvas to capture dimensions so the Playwright locator screenshot
  // matches the requested image size.
  canvas.style.width  = `${captureWidth}px`;
  canvas.style.height = `${captureHeight}px`;
  canvas.width  = captureWidth;
  canvas.height = captureHeight;
}

const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.dampingFactor = 0.05;

// OrbitControls drives an RAF loop internally for damping; we just need to
// call .update() each frame. attachVitrum provides its own RAF loop; we hook
// into it via the onFrame callback.
function onFrame(): void {
  orbit.update();
}

// ── Engine + scene lifecycle ──────────────────────────────────────────────────

function setStatus(msg: string): void {
  statusEl.textContent = msg;
}

/** Fit the orbit camera so the loaded scene fills the view. */
function fitCameraToScene(scene: import('@vitrum/core').Scene): void {
  // Compute AABB from triangle vertex data.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const prim of scene.primitives) {
    if (prim.kind !== 'mesh') continue;
    const pos = prim.positions;
    for (let i = 0; i < pos.length; i += 3) {
      const x = pos[i]!, y = pos[i + 1]!, z = pos[i + 2]!;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
  }
  if (!isFinite(minX)) return; // empty scene — leave camera as-is

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const diag = Math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2 + (maxZ - minZ) ** 2);
  const dist = diag * 1.2;

  camera.position.set(cx, cy + diag * 0.3, cz + dist);
  camera.lookAt(cx, cy, cz);
  orbit.target.set(cx, cy, cz);
  orbit.update();

  camera.near = diag * 0.001;
  camera.far  = diag * 10;
  camera.updateProjectionMatrix();
}

// ── Engine preference toggle ──────────────────────────────────────────────────

async function switchPrefer(p: EnginePreference): Promise<void> {
  prefer = p;
  btnRealtime.classList.toggle('active', p === 'realtime');
  btnQuality.classList.toggle('active', p === 'quality');

  // If a model is already loaded, recreate the engine with the new preference.
  if (handle) {
    const oldHandle = handle;
    handle = null;
    const currentScene = await getCurrentScene(oldHandle);
    oldHandle.dispose();

    if (currentScene) {
      setStatus(`Switching to ${prefer} engine…`);
      try {
        handle = await attachVitrum({ canvas, scene: currentScene, camera, prefer, onFrame });
        setStatus(`Engine: ${prefer === 'realtime' ? 'Realtime GI' : 'Path Trace'}`);
      } catch (err) {
        setStatus(`Engine switch failed: ${String(err)}`);
      }
    }
  }
}

// The engine handle doesn't expose the scene directly; we re-use the last
// loaded vitrum Scene stored here for engine switches.
let lastVitrumScene: import('@vitrum/core').Scene | null = null;

async function loadFileTracked(file: File): Promise<void> {
  const result = await loadGltfScene(file).catch((err) => {
    setStatus(`Load error: ${String(err)}`);
    return null;
  });
  if (!result) return;

  lastVitrumScene = result.scene;

  if (handle) { handle.dispose(); handle = null; }
  fitCameraToScene(result.scene);
  setStatus(`Starting engine (${prefer})…`);
  try {
    handle = await attachVitrum({ canvas, scene: result.scene, camera, prefer, onFrame });
    overlay.classList.add('hidden');
    controls2.style.display = 'flex';
    setStatus(`${file.name} — ${prefer === 'realtime' ? 'Realtime GI' : 'Path Trace'}`);
  } catch (err) {
    setStatus(`Engine error: ${String(err)}`);
  }
}

async function getCurrentScene(_h: AttachVitrumHandle): Promise<import('@vitrum/core').Scene | null> {
  // We store the last scene above rather than extracting from the engine handle.
  return lastVitrumScene;
}

btnRealtime.addEventListener('click', () => { void switchPrefer('realtime'); });
btnQuality.addEventListener('click',  () => { void switchPrefer('quality'); });

// ── Drag-drop and file-open handlers ─────────────────────────────────────────

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('over');
  const file = e.dataTransfer?.files?.[0];
  if (file) void loadFileTracked(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) void loadFileTracked(file);
});
fileInput2.addEventListener('change', () => {
  const file = fileInput2.files?.[0];
  if (file) void loadFileTracked(file);
});

// Allow dropping on the canvas once a model is loaded.
canvas.addEventListener('dragover', (e) => e.preventDefault());
canvas.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file) void loadFileTracked(file);
});

// ── Window resize ─────────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  if (captureMode) return; // dimensions are locked in capture mode
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// ── Capture-mode bootstrap (procedural fallback scene) ────────────────────────
//
// When `?vitrumScenario=hero-viewer` is present we skip the interactive
// drag-drop UI and render a deterministic built-in scene: a chrome sphere on
// a checkered floor under two area lights. This is what gets A/B-diffed across
// session branches at merge time.
function buildFallbackHeroScene(): THREE.Scene {
  const scene = new THREE.Scene();

  // Center chrome sphere
  const sphereMat = new THREE.MeshPhysicalMaterial({
    color: 0xeeeeee,
    roughness: 0.15,
    metalness: 1.0,
  });
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.6, 48, 32), sphereMat);
  sphere.position.set(0, 0.6, 0);
  scene.add(sphere);

  // Diffuse pedestal — kept matte so the chrome highlight stays distinct.
  const baseMat = new THREE.MeshPhysicalMaterial({
    color: 0x707070, roughness: 0.85, metalness: 0,
  });
  const base = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.0, 0.05, 48), baseMat);
  base.position.set(0, -0.025, 0);
  scene.add(base);

  // Ground
  const floorMat = new THREE.MeshPhysicalMaterial({ color: 0x222222, roughness: 0.95 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.06;
  scene.add(floor);

  // Two area lights for a clear directional response on the sphere.
  const keyLight = new THREE.RectAreaLight(0xfff4e8, 14, 2.0, 2.0);
  keyLight.position.set(-1.8, 2.5, 1.8);
  keyLight.lookAt(0, 0.6, 0);
  scene.add(keyLight);

  const fillLight = new THREE.RectAreaLight(0xdfeaff, 6, 1.6, 1.6);
  fillLight.position.set(2.0, 1.8, -1.2);
  fillLight.lookAt(0, 0.6, 0);
  scene.add(fillLight);

  return scene;
}

async function runCaptureModeBootstrap(): Promise<void> {
  const initStart = performance.now();
  try {
    const threeScene = buildFallbackHeroScene();
    const vitrumScene = sceneFromThreeJS(threeScene);

    // Fixed camera framing for determinism — no orbit damping in capture mode.
    camera.position.set(2.4, 1.6, 2.4);
    camera.lookAt(0, 0.5, 0);
    orbit.target.set(0, 0.5, 0);
    orbit.enabled = false;
    orbit.update();

    overlay.classList.add('hidden');

    handle = await attachVitrum({
      canvas,
      scene: vitrumScene,
      camera,
      prefer: capturePrefer,
      // No onFrame in capture mode — OrbitControls is disabled so damping is irrelevant.
      // Pass quality settings only for the path-trace path; realtime ignores quality.
      quality: {
        samplesTarget: captureSpp,
        bounces: 8,
        resolutionFactor: 1,
        filteredGlossyFactor: 0.5,
      },
      onProgress: (p) => {
        if (p.kind !== 'pt-spp') return;
        setStatus(`Capture SPP ${Math.round(p.current)}/${Math.round(p.target)}`);
        if (p.fraction >= 1) finalizeCapture(p.current, initStart);
      },
    });
    setStatus(`Capture mode — ${capturePrefer} engine, ${captureWidth}x${captureHeight}`);

    // For realtime: there's no SPP convergence signal — fall back to a settle
    // timer matching the Playwright adapter's default settleMs window.
    if (capturePrefer === 'realtime') {
      window.setTimeout(() => finalizeCapture(0, initStart), 4000);
    }
  } catch (err) {
    setStatus(`Capture init failed: ${String(err)}`);
    globalThis.VITRUM_CAPTURE_TELEMETRY = { error: String(err) };
    globalThis.VITRUM_CAPTURE_READY = true;
  }
}

function finalizeCapture(samplesAccumulated: number, initStart: number): void {
  if (globalThis.VITRUM_CAPTURE_READY === true) return;
  const elapsed = performance.now() - initStart;
  globalThis.VITRUM_MS_PER_SAMPLE = elapsed / Math.max(samplesAccumulated, 1);
  globalThis.VITRUM_CAPTURE_TELEMETRY = {
    scenarioId: `hero-viewer-${capturePrefer}`,
    samplesAccumulated,
    msPerSample: globalThis.VITRUM_MS_PER_SAMPLE,
    captureCanvasSelector: '#c',
    enginePrefer: capturePrefer,
  };
  globalThis.VITRUM_CAPTURE_READY = true;
  console.info(
    `[vitrum-capture] hero-viewer ready: ${samplesAccumulated} SPP in ${elapsed.toFixed(0)} ms (prefer=${capturePrefer})`,
  );
}

if (captureMode) {
  void runCaptureModeBootstrap();
}
