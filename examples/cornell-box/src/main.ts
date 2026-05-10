/**
 * Minimal Cornell box in three.js → @vitrum/core Scene → pt-webgl path tracer.
 */

import type { FrameInput, Mat4, Vec3 } from '@vitrum/core';
import * as THREE from 'three';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
import { sceneFromThreeJS } from '@vitrum/three-bindings';

declare global {
  // Optional capture harness hooks read by tools/benchmark-runner/capture-adapter-playwright.mjs.
  // eslint-disable-next-line no-var
  var VITRUM_CAPTURE_READY: boolean | undefined;
  // eslint-disable-next-line no-var
  var VITRUM_MS_PER_SAMPLE: number | undefined;
}

function mat4FromThree(m: THREE.Matrix4): Mat4 {
  return new Float32Array(m.elements);
}

interface CaptureConfig {
  readonly scenarioId: string;
  readonly seed: number;
  readonly width: number;
  readonly height: number;
  readonly bounces: number;
  readonly samplesTarget: number;
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly isCapture: boolean;
}

function parsePositiveInt(value: string | null, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function parseCaptureConfig(): CaptureConfig {
  const params = new URLSearchParams(window.location.search);
  const caustic = params.get('vitrumCaustic');
  const causticStrategy =
    caustic === 'manifold-nee' || caustic === 'photon-map' ? caustic : 'none';
  return {
    scenarioId: params.get('vitrumScenario') ?? 'cornell-box',
    seed: parsePositiveInt(params.get('vitrumSeed'), 12345),
    width: parsePositiveInt(params.get('vitrumWidth'), window.innerWidth || 1280),
    height: parsePositiveInt(params.get('vitrumHeight'), window.innerHeight || 720),
    bounces: parsePositiveInt(params.get('vitrumBounces'), 8),
    samplesTarget: parsePositiveInt(params.get('vitrumSpp'), 48),
    causticStrategy,
    isCapture: params.has('vitrumScenario'),
  };
}

function applyScenarioMaterialTweaks(
  material: THREE.MeshPhysicalMaterial,
  config: CaptureConfig,
): void {
  if (config.scenarioId.includes('spectral') || config.scenarioId.includes('thinfilm')) {
    material.transmission = 0.75;
    material.ior = 1.52;
    material.thickness = 0.4;
    material.attenuationDistance = 1.5;
    material.attenuationColor.setRGB(0.72, 0.9, 1.0);
    material.userData['vitrumSpectralAttenuation'] = {
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: new Float32Array([
        0.08, 0.1, 0.12, 0.15, 0.2, 0.28, 0.36, 0.44,
        0.52, 0.58, 0.64, 0.68, 0.7, 0.68, 0.62, 0.54,
        0.46, 0.38, 0.31, 0.25, 0.2, 0.16, 0.13, 0.11,
        0.1, 0.09, 0.085, 0.08, 0.078, 0.076, 0.074, 0.072,
      ]),
    };
    material.userData['vitrumThinFilmStack'] = {
      incidentIor: 1.0,
      angleDependent: true,
      layers: [
        { ior: 2.1, thicknessNm: 72, extinctionCoefficient: 0.015 },
        { ior: 1.46, thicknessNm: 118, extinctionCoefficient: 0.0 },
      ],
    };
  }

  if (config.scenarioId.includes('layered')) {
    material.userData['vitrumFrontLayer'] = { transmission: [0.95, 0.8, 0.65], roughness: 0.18 };
    material.userData['vitrumBackLayer'] = { transmission: [0.65, 0.8, 0.95], roughness: 0.28 };
  }

  if (config.scenarioId.includes('sss')) {
    material.userData['vitrumScatteringCoefficient'] = 0.18;
    material.userData['vitrumScatteringCoefficientRGB'] = [0.16, 0.2, 0.24];
    material.userData['vitrumScatteringAnisotropy'] = 0.35;
  }
}

function buildCornellScene(config: CaptureConfig): THREE.Scene {
  const scene = new THREE.Scene();
  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0xddeeff,
    roughness: 0.08,
    metalness: 0,
    transmission: 0.5,
    ior: 1.5,
    thickness: 0.25,
  });
  applyScenarioMaterialTweaks(glass, config);

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
  mk(new THREE.BoxGeometry(0.6, 1.2, 0.6), glass, [0.3, -0.35, -0.3], [1, 1, 1]);

  const light = new THREE.RectAreaLight(0xffffff, 12, 1.0, 1.0);
  light.position.set(0, 0.98, 0);
  light.rotation.x = -Math.PI / 2;
  scene.add(light);

  if (config.scenarioId.includes('parity') || config.scenarioId.includes('caustic')) {
    const point = new THREE.PointLight(0x99bbff, 2.5);
    point.position.set(-0.7, 0.2, 0.65);
    scene.add(point);
  }

  return scene;
}

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#c');
  const statusEl = document.querySelector<HTMLDivElement>('#status');
  if (!canvas || !statusEl) throw new Error('missing #c or #status');
  const config = parseCaptureConfig();
  globalThis.VITRUM_CAPTURE_READY = false;
  globalThis.VITRUM_MS_PER_SAMPLE = undefined;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
  renderer.setPixelRatio(config.isCapture ? 1 : Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x111111, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  const threeScene = buildCornellScene(config);
  const vitrumScene = sceneFromThreeJS(threeScene);

  const engine = await createPTEngine_WebGL2({
    device: renderer,
    maxBounces: config.bounces,
    maxSamplesPerPixel: config.samplesTarget,
    causticStrategy: config.causticStrategy,
  });
  engine.setScene(vitrumScene);

  let frame = 0;
  const startMs = performance.now();
  const samplesTarget = config.samplesTarget;

  function resize(): void {
    const w = config.isCapture ? config.width : window.innerWidth;
    const h = config.isCapture ? config.height : window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (config.isCapture) {
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    }
    renderer.setSize(w, h, false);
  }
  resize();
  if (!config.isCapture) window.addEventListener('resize', resize);

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
      frameSeed: (frame * 9973 + config.seed) >>> 0,
      quality: {
        samplesTarget,
        bounces: config.bounces,
        resolutionFactor: 1,
        filteredGlossyFactor: 0.5,
      },
    };

    const out = engine.renderFrame(input);
    frame++;
    statusEl.textContent = `${config.scenarioId} (${config.causticStrategy}) SPP: ${out.samplesAccumulated} / ${samplesTarget}${out.isConverged ? ' — converged' : ''}`;
    if (!out.isConverged) {
      requestAnimationFrame(loop);
    } else {
      globalThis.VITRUM_MS_PER_SAMPLE = (performance.now() - startMs) / Math.max(out.samplesAccumulated, 1);
      globalThis.VITRUM_CAPTURE_READY = true;
    }
  }

  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
