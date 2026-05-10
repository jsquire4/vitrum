/**
 * Minimal Cornell box in three.js → @vitrum/core Scene → pt-webgl path tracer.
 *
 * Query: `?hdri=venice_sunset` (see `OUTDOOR_HDRI_PRESETS` in `@vitrum/pt-webgl` for ids)
 * loads a Poly Haven 1k HDR for `scene.environment` / background.
 */

import type { FrameInput, Mat4 } from '@vitrum/core';
import { buildCornellBoxThreeScene } from '@vitrum-examples/shared';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  createPTEngine_WebGL2,
  findHdriPresetById,
  loadHdriEquirect,
  PT_PREVIEW_OPTIONS,
} from '@vitrum/pt-webgl';
import { sceneFromThreeJS } from '@vitrum/three-bindings';

function mat4FromThree(m: THREE.Matrix4): Mat4 {
  return new Float32Array(m.elements);
}

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#c');
  const statusEl = document.querySelector<HTMLDivElement>('#status');
  if (!canvas || !statusEl) throw new Error('missing #c or #status');

  const c = canvas;
  const status = statusEl;

  const renderer = new THREE.WebGLRenderer({ canvas: c, antialias: false, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x111111, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  const threeScene = buildCornellBoxThreeScene();

  const hdriParam =
    typeof globalThis.location !== 'undefined'
      ? new URLSearchParams(globalThis.location.search).get('hdri')
      : null;
  if (hdriParam) {
    const preset = findHdriPresetById(hdriParam);
    if (preset) {
      try {
        const envMap = await loadHdriEquirect(preset.url);
        threeScene.environment = envMap;
        threeScene.background = envMap;
      } catch (err) {
        console.warn('HDRI load failed', preset.id, err);
      }
    }
  }

  const vitrumScene = sceneFromThreeJS(threeScene);

  const engine = await createPTEngine_WebGL2({ device: renderer });
  engine.setScene(vitrumScene);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.15;
  controls.target.set(-0.05, -0.15, 0);
  controls.update();

  let frameSeq = 0;
  controls.addEventListener('change', () => {
    engine.reset();
    frameSeq = 0;
  });

  const samplesTarget = 48;

  function resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }
  resize();
  window.addEventListener('resize', resize);

  function loop(): void {
    controls.update();
    camera.updateMatrixWorld();
    const input: FrameInput = {
      viewMatrix: mat4FromThree(camera.matrixWorldInverse),
      projMatrix: mat4FromThree(camera.projectionMatrix),
      cameraPosition: [camera.position.x, camera.position.y, camera.position.z],
      viewport: {
        width: c.width,
        height: c.height,
        devicePixelRatio: window.devicePixelRatio,
      },
      frameIndex: frameSeq,
      frameSeed: (frameSeq * 9973 + 12345) >>> 0,
      quality: {
        ...PT_PREVIEW_OPTIONS,
        samplesTarget,
      },
    };

    const out = engine.renderFrame(input);
    frameSeq++;
    status.textContent = `SPP: ${out.samplesAccumulated} / ${samplesTarget}${out.isConverged ? ' — converged' : ''}`;
    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
