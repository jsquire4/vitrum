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
import { loadGltfScene } from '@vitrum/three-bindings';

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

const initialW = Math.max(canvas.clientWidth, 1);
const initialH = Math.max(canvas.clientHeight, 1);
const camera = new THREE.PerspectiveCamera(45, initialW / initialH, 0.01, 1000);
camera.position.set(0, 1, 3);

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
  const w = Math.max(canvas.clientWidth, 1);
  const h = Math.max(canvas.clientHeight, 1);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});
