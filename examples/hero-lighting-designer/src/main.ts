/**
 * hero-lighting-designer — interactive 3-light GI designer.
 *
 * Demonstrates:
 *   - attachVitrum() with prefer:'realtime' for live GI updates
 *   - sceneFromThreeJS() for procedural scene construction
 *   - engine.setScene() to push light changes through (via handle.engine)
 *   - engine.onFrame() telemetry for FPS HUD
 *
 * Host code: <200 LoC. Scene: procedural room (~50 lines).
 * No external assets required.
 */

import * as THREE from 'three';
import { attachVitrum, type AttachVitrumHandle } from '@vitrum/engine';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import type { FrameStats } from '@vitrum/core';

// ── DOM refs ──────────────────────────────────────────────────────────────────

const canvas   = document.querySelector<HTMLCanvasElement>('#c')!;
const hudEl    = document.querySelector<HTMLDivElement>('#hud')!;
const statusEl = document.querySelector<HTMLDivElement>('#status')!;

// ── Light state ───────────────────────────────────────────────────────────────

interface LightParams {
  r: number; g: number; b: number;
  intensity: number;
  x: number; y: number; z: number;
}

const lights: LightParams[] = [
  { r: 255, g: 100, b: 68,  intensity: 8, x: -2, y: 2.5, z:  1 },
  { r: 68,  g: 136, b: 255, intensity: 6, x:  2, y: 2.5, z: -1 },
  { r: 255, g: 255, b: 170, intensity: 3, x:  0, y: 3.5, z:  3 },
];

// ── Procedural architectural interior ────────────────────────────────────────
// ~50 lines of scene setup. A room: floor + ceiling + 4 walls + simple
// furniture (two boxes representing a table and a shelf unit).

function buildRoom(): THREE.Scene {
  const scene = new THREE.Scene();

  const wallMat = new THREE.MeshPhysicalMaterial({ color: 0xddddd0, roughness: 0.85, metalness: 0 });
  const floorMat = new THREE.MeshPhysicalMaterial({ color: 0x8b7355, roughness: 0.7,  metalness: 0 });
  const ceilMat  = new THREE.MeshPhysicalMaterial({ color: 0xf5f5f5, roughness: 0.9,  metalness: 0 });
  const woodMat  = new THREE.MeshPhysicalMaterial({ color: 0x6b4226, roughness: 0.6,  metalness: 0 });
  const metalMat = new THREE.MeshPhysicalMaterial({ color: 0xaaaaaa, roughness: 0.3,  metalness: 0.8 });

  const add = (geo: THREE.BufferGeometry, mat: THREE.Material, x: number, y: number, z: number, rx = 0, ry = 0, rz = 0) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.rotation.set(rx, ry, rz);
    scene.add(mesh);
  };

  const W = 8, H = 4.5, D = 10;
  const t = 0.1;

  // Floor, ceiling, walls
  add(new THREE.BoxGeometry(W, t, D),  floorMat, 0, 0,     0);
  add(new THREE.BoxGeometry(W, t, D),  ceilMat,  0, H,     0);
  add(new THREE.BoxGeometry(t, H, D),  wallMat, -W / 2, H / 2, 0);   // left wall
  add(new THREE.BoxGeometry(t, H, D),  wallMat,  W / 2, H / 2, 0);   // right wall
  add(new THREE.BoxGeometry(W, H, t),  wallMat,  0, H / 2, -D / 2);  // back wall
  // Front wall omitted so the camera can look in.

  // Furniture: table
  add(new THREE.BoxGeometry(2.5, 0.08, 1.2), woodMat, -1.5, 0.85, 1);
  add(new THREE.CylinderGeometry(0.04, 0.04, 0.85), woodMat, -2.5, 0.42, 1.5);
  add(new THREE.CylinderGeometry(0.04, 0.04, 0.85), woodMat, -0.5, 0.42, 1.5);
  add(new THREE.CylinderGeometry(0.04, 0.04, 0.85), woodMat, -2.5, 0.42, 0.5);
  add(new THREE.CylinderGeometry(0.04, 0.04, 0.85), woodMat, -0.5, 0.42, 0.5);

  // Furniture: shelf unit against back wall
  add(new THREE.BoxGeometry(2, 0.06, 0.4), woodMat, 2, 2.0, -4.7);
  add(new THREE.BoxGeometry(2, 0.06, 0.4), woodMat, 2, 1.2, -4.7);
  add(new THREE.BoxGeometry(0.06, 2, 0.4), woodMat, 1.0, 1.6, -4.7);
  add(new THREE.BoxGeometry(0.06, 2, 0.4), woodMat, 3.0, 1.6, -4.7);

  // Small decorative object on table (metallic sphere)
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.18, 24, 24), metalMat);
  sphere.position.set(-1.5, 1.0, 1.0);
  scene.add(sphere);

  // Add the 3 point lights (driven by the sliders)
  for (const lp of lights) {
    const pl = new THREE.PointLight(
      new THREE.Color(lp.r / 255, lp.g / 255, lp.b / 255),
      lp.intensity,
    );
    pl.position.set(lp.x, lp.y, lp.z);
    scene.add(pl);
  }

  return scene;
}

// ── Three camera ──────────────────────────────────────────────────────────────

const initialW = Math.max(canvas.clientWidth, 1);
const initialH = Math.max(canvas.clientHeight, 1);
const camera = new THREE.PerspectiveCamera(50, initialW / initialH, 0.05, 100);
camera.position.set(0, 1.6, 4.5);
camera.lookAt(0, 1.2, 0);

// ── Engine + slider wiring ────────────────────────────────────────────────────

let threeScene = buildRoom();
let vitrumScene = sceneFromThreeJS(threeScene);
let engineHandle: AttachVitrumHandle | null = null;

async function init(): Promise<void> {
  statusEl.textContent = 'Starting engine…';
  try {
    engineHandle = await attachVitrum({
      canvas,
      scene: vitrumScene,
      camera,
      prefer: 'realtime',
      onFrame: (stats: FrameStats) => {
        const fps = stats.frameTimeMs > 0 ? Math.round(1000 / stats.frameTimeMs) : 0;
        hudEl.textContent = `FPS: ${fps}`;
      },
    });
    statusEl.textContent = 'Ready — adjust the sliders to change lighting';
  } catch (err) {
    statusEl.textContent = `Engine error: ${String(err)}`;
  }
}

/** Debounced rebuild: batch slider changes into one setScene call. */
let debounceTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleSceneUpdate(): void {
  if (debounceTimer !== null) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    // Rebuild the THREE scene with updated light params.
    threeScene = buildRoom();
    vitrumScene = sceneFromThreeJS(threeScene);
    engineHandle?.engine.setScene(vitrumScene);
    // Reset temporal accumulator so GI re-converges with new lighting.
    engineHandle?.engine.reset();
  }, 50);
}

// ── Slider helpers ────────────────────────────────────────────────────────────

function bindSlider(id: string, onChange: (v: number) => void): void {
  const slider = document.querySelector<HTMLInputElement>(`#${id}`)!;
  const valEl  = document.querySelector<HTMLSpanElement>(`#v-${id}`)!;
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    valEl.textContent = Number.isInteger(v) ? String(v) : v.toFixed(1);
    onChange(v);
    scheduleSceneUpdate();
  });
}

function bindLightSliders(idx: number): void {
  const lp = lights[idx]!;
  const swatchEl = document.querySelector<HTMLSpanElement>(`#swatch-${idx}`)!;

  const updateSwatch = () => {
    swatchEl.style.background = `rgb(${lp.r},${lp.g},${lp.b})`;
  };

  bindSlider(`l${idx}-r`, (v) => { lp.r = v; updateSwatch(); });
  bindSlider(`l${idx}-g`, (v) => { lp.g = v; updateSwatch(); });
  bindSlider(`l${idx}-b`, (v) => { lp.b = v; updateSwatch(); });
  bindSlider(`l${idx}-i`, (v) => { lp.intensity = v; });
  bindSlider(`l${idx}-x`, (v) => { lp.x = v; });
  bindSlider(`l${idx}-y`, (v) => { lp.y = v; });
  bindSlider(`l${idx}-z`, (v) => { lp.z = v; });
}

bindLightSliders(0);
bindLightSliders(1);
bindLightSliders(2);

// ── Resize ────────────────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  const w = Math.max(canvas.clientWidth, 1);
  const h = Math.max(canvas.clientHeight, 1);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});

// ── Boot ──────────────────────────────────────────────────────────────────────

void init();
