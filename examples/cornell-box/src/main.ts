/**
 * Minimal Cornell box in three.js → @vitrum/core Scene → pt-webgl path tracer.
 */

import type { FrameInput, Mat4, Vec3 } from '@vitrum/core';
import * as THREE from 'three';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
import { sceneFromThreeJS } from '@vitrum/three-bindings';

function mat4FromThree(m: THREE.Matrix4): Mat4 {
  return new Float32Array(m.elements);
}

function buildCornellScene(): THREE.Scene {
  const scene = new THREE.Scene();
  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });

  const mk = (geo: THREE.BufferGeometry, mat: THREE.MeshPhysicalMaterial, pos: Vec3, scale: Vec3) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.scale.set(scale[0], scale[1], scale[2]);
    scene.add(mesh);
  };

  const t = 0.02;
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, -1, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(2, t, 2), white, [0, 1, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(t, 2, 2), green, [1, 0, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(t, 2, 2), red, [-1, 0, 0], [1, 1, 1]);
  mk(new THREE.BoxGeometry(2, 2, t), white, [0, 0, -1], [1, 1, 1]);

  mk(new THREE.BoxGeometry(0.6, 0.6, 0.6), white, [-0.35, -0.65, 0.2], [1, 1, 1]);
  mk(new THREE.BoxGeometry(0.6, 1.2, 0.6), white, [0.3, -0.35, -0.3], [1, 1, 1]);

  const light = new THREE.RectAreaLight(0xffffff, 12, 1.0, 1.0);
  light.position.set(0, 0.98, 0);
  light.rotation.x = -Math.PI / 2;
  scene.add(light);

  return scene;
}

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#c');
  const statusEl = document.querySelector<HTMLDivElement>('#status');
  if (!canvas || !statusEl) throw new Error('missing #c or #status');

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x111111, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  const threeScene = buildCornellScene();
  const vitrumScene = sceneFromThreeJS(threeScene);

  const engine = await createPTEngine_WebGL2({ device: renderer });
  engine.setScene(vitrumScene);

  let frame = 0;
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
    camera.updateMatrixWorld();
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
      frameSeed: (frame * 9973 + 12345) >>> 0,
      quality: {
        samplesTarget,
        bounces: 8,
        resolutionFactor: 1,
        filteredGlossyFactor: 0.5,
      },
    };

    const out = engine.renderFrame(input);
    frame++;
    statusEl.textContent = `SPP: ${out.samplesAccumulated} / ${samplesTarget}${out.isConverged ? ' — converged' : ''}`;
    if (!out.isConverged) {
      requestAnimationFrame(loop);
    }
  }

  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
