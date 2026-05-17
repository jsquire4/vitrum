/**
 * Minimal Cornell box in three.js → @vitrum/core Scene → pt-webgl path tracer.
 */

import type { FrameInput, Mat4, Vec3 } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import type {
  PTEngineWebGL2,
  PTEngineWebGL2FrameOutput,
  PTEngineWebGL2QualityMode,
} from '@vitrum/pt-webgl';
import * as THREE from 'three';
import { createPTEngine_WebGL2, readAccumulationRgbFloat } from '@vitrum/pt-webgl';
import { sceneFromThreeJS, VITRUM_USER_DATA_KEYS as K } from '@vitrum/three-bindings';
import {
  HDR_LUMINANCE_BILATERAL_DEFAULT_SIGMA_LUMINANCE,
  ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS as SVGF_DEFAULT_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_FRAME_COUNT_INPUT_GUARD_MAX as SVGF_FRAME_COUNT_INPUT_GUARD_MAX,
  ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS as SVGF_MAX_ATROUS_ITERATIONS,
} from '@vitrum/shared-denoisers';
import { BilateralPreviewCanvas, writeTonemappedRgbToCanvas, type DenoiseDisplayMode } from './denoiseDisplay.js';

declare global {
  // Optional capture harness hooks read by tools/benchmark-runner/capture-adapter-playwright.mjs.
  // eslint-disable-next-line no-var
  var VITRUM_CAPTURE_READY: boolean | undefined;
  // eslint-disable-next-line no-var
  var VITRUM_MS_PER_SAMPLE: number | undefined;
  // eslint-disable-next-line no-var
  var VITRUM_CAPTURE_TELEMETRY: Record<string, unknown> | undefined;
  /** Preferred screenshot target when capture runs after VITRUM_CAPTURE_READY (see adapter env override). */
  // eslint-disable-next-line no-var
  var VITRUM_CAPTURE_CANVAS_SELECTOR: string | undefined;
}

function mat4FromThree(m: THREE.Matrix4): Mat4 {
  return asMat4(new Float32Array(m.elements));
}

interface CaptureConfig {
  readonly scenarioId: string;
  readonly seed: number;
  readonly width: number;
  readonly height: number;
  readonly bounces: number;
  readonly samplesTarget: number;
  /** Optional override for engine samplesPerFrame (URL: vitrumSpf). When set,
   * the capture-mode default of 1 sample/frame is replaced — useful for
   * background-tab capture where rAF cadence is the bottleneck.  */
  readonly samplesPerFrame: number | null;
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly isCapture: boolean;
  readonly autoStart: boolean;
  readonly qualityMode: PTEngineWebGL2QualityMode;
  /** Interactive resize cap (capture mode uses explicit width/height). */
  readonly maxInteractiveWidth: number;
  readonly maxInteractiveHeight: number;
  /** Upper bound on render pixels (w×h) for VRAM / scheduler guardrail. */
  readonly maxMegapixels: number;
  readonly denoiseDisplay: DenoiseDisplayMode;
  readonly oidnModelUrl: string | null;
  /** σ for WebGPU luminance bilateral (`vitrumDisplay=wgsl`). */
  readonly wgslSigma: number;
  /** SVGF variance uniform `frameCount`; temporal Welford branch uses SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT from `@vitrum/shared-denoisers`. Aliases: `vitrumSvgfFrames`. Capped at SVGF_FRAME_COUNT_INPUT_GUARD_MAX. Without real `welfordMeanM2`, temporal variance sees zeros (demo only). */
  readonly svgfFrameCount: number;
  /** SVGF à-trous dispatch count (step widths 1…2^n). Clamped 1–SVGF_MAX_ATROUS_ITERATIONS in library and URL parser. Query: `vitrumSvgfAtrous`. */
  readonly svgfAtrousIterations: number;
  /** When false (`vitrumWebGpuShared=0`), each WebGPU denoise pass uses a throwaway device. */
  readonly webGpuReuseSharedDevice: boolean;
  readonly pixelAdaptiveSampling: boolean;
  readonly pixelAdaptiveCadence: number;
}

/** Parse a numeric URL param with shared validation rules. */
function parseNumber(
  value: string | null,
  fallback: number,
  opts: { integer?: boolean; min?: number } = {},
): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  const min = opts.min ?? -Infinity;
  if (n < min) return fallback;
  return opts.integer ? Math.floor(n) : n;
}

function parsePositiveMegapixels(value: string | null, fallback: number): number {
  // Positive (strictly greater than 0).
  const n = parseNumber(value, fallback);
  return n > 0 ? n : fallback;
}

function resolveScenarioId(raw: string): string {
  const aliases: Record<string, string> = {
    cornell: 'cornell-box',
    glass: 'cornell-glass',
    caustic: 'cornell-caustic',
    spectral: 'cornell-spectral',
    layered: 'cornell-layered',
    sss: 'cornell-sss',
    parity: 'cornell-parity',
  };
  return aliases[raw] ?? raw;
}

/** Shrinks [w,h] so w×h ≤ maxMegapixels·1e6 (aspect preserved). */
function clampMegapixels(w: number, h: number, maxMegapixels: number): readonly [number, number] {
  const maxPx = maxMegapixels * 1_000_000;
  const px = w * h;
  if (px <= maxPx || px <= 0) return [w, h] as const;
  const s = Math.sqrt(maxPx / px);
  return [Math.max(1, Math.floor(w * s)), Math.max(1, Math.floor(h * s))] as const;
}

function parsePositiveFloat(value: string | null, fallback: number): number {
  const n = parseNumber(value, fallback);
  return n > 0 ? n : fallback;
}

function parsePositiveInt(value: string | null, fallback: number): number {
  const n = parseNumber(value, fallback, { integer: true });
  return n > 0 ? n : fallback;
}

function parseNonNegativeInt(value: string | null, fallback: number): number {
  return parseNumber(value, fallback, { integer: true, min: 0 });
}

function clampInt(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

/** À-trous iterations for SVGF (query `vitrumSvgfAtrous`); clamps to SVGF_MAX_ATROUS_ITERATIONS. */
function parseSvgfAtrousIterations(value: string | null, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return clampInt(Math.floor(parsed), 1, SVGF_MAX_ATROUS_ITERATIONS);
}

const DEFAULT_MANUAL_SPP = 1024;
const DEFAULT_CAPTURE_SPP = 256;
const SMOKE_SPP_THRESHOLD = 16;
const DEFAULT_INTERACTIVE_MAX_W = 1920;
const DEFAULT_INTERACTIVE_MAX_H = 1080;

type InteractivePresetId =
  | 'draft'
  | 'preview'
  | 'quality'
  | 'hero'
  | 'studio'
  | 'final4k'
  | 'overnight';

interface InteractivePreset {
  readonly maxW: number;
  readonly maxH: number;
  readonly spp: number;
  readonly bounces: number;
  readonly qualityMode: PTEngineWebGL2QualityMode;
}

const INTERACTIVE_PRESETS: Record<InteractivePresetId, InteractivePreset> = {
  draft: { maxW: 960, maxH: 540, spp: 128, bounces: 4, qualityMode: 'interactive' },
  preview: { maxW: 1280, maxH: 720, spp: 512, bounces: 6, qualityMode: 'interactive' },
  quality: { maxW: 1920, maxH: 1080, spp: 1024, bounces: 8, qualityMode: 'interactive' },
  hero: { maxW: 2560, maxH: 1440, spp: 512, bounces: 10, qualityMode: 'final' },
  studio: { maxW: 2560, maxH: 1440, spp: 2048, bounces: 10, qualityMode: 'final' },
  final4k: { maxW: 3840, maxH: 2160, spp: 8192, bounces: 12, qualityMode: 'final' },
  overnight: { maxW: 3840, maxH: 2160, spp: 16384, bounces: 14, qualityMode: 'final' },
};

function parseQualityMode(value: string | null, isCapture: boolean): PTEngineWebGL2QualityMode {
  if (value === 'interactive' || value === 'final' || value === 'capture' || value === 'safe') return value;
  return isCapture ? 'capture' : 'interactive';
}

/** Defaults when vitrumBounces is omitted: diffuse box converges quickly; glass/caustics need depth. */
function defaultBouncesForScenario(scenarioId: string): number {
  if (scenarioId.includes('caustic') || scenarioId.includes('parity')) return 12;
  if (
    scenarioId.includes('spectral') ||
    scenarioId.includes('thinfilm') ||
    scenarioId.includes('dispersion')
  ) {
    return 10;
  }

  return 6;
}

function fitWithin(width: number, height: number, maxWidth: number, maxHeight: number): readonly [number, number] {
  const scale = Math.min(1, maxWidth / Math.max(width, 1), maxHeight / Math.max(height, 1));
  return [Math.max(1, Math.floor(width * scale)), Math.max(1, Math.floor(height * scale))] as const;
}

function getWebGLRendererLabel(renderer: THREE.WebGLRenderer): string {
  const gl = renderer.getContext();
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  if (debugInfo != null) {
    const unmasked = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    if (typeof unmasked === 'string' && unmasked.length > 0) return unmasked;
  }
  const fallback = gl.getParameter(gl.RENDERER);
  return typeof fallback === 'string' ? fallback : 'unknown WebGL renderer';
}

function parseDenoiseDisplay(params: URLSearchParams): DenoiseDisplayMode {
  const v = params.get('vitrumDisplay');
  if (v === 'bilateral' || v === 'oidn' || v === 'wgsl' || v === 'svgf') return v;
  return 'raw';
}

function parseCaptureConfig(): CaptureConfig {
  const params = new URLSearchParams(window.location.search);
  const caustic = params.get('vitrumCaustic');
  const causticStrategy =
    caustic === 'manifold-nee' || caustic === 'photon-map' ? caustic : 'none';
  const isCapture = params.has('vitrumScenario');
  const scenarioRaw = params.get('vitrumScenario');
  const scenarioId = resolveScenarioId(
    scenarioRaw != null && scenarioRaw.trim().length > 0 ? scenarioRaw.trim() : 'cornell-box',
  );
  const presetParam = params.get('vitrumPreset');
  const preset: InteractivePreset | undefined =
    !isCapture && presetParam != null && presetParam in INTERACTIVE_PRESETS
      ? INTERACTIVE_PRESETS[presetParam as InteractivePresetId]
      : undefined;

  const qualityMode = params.has('vitrumQuality')
    ? parseQualityMode(params.get('vitrumQuality'), isCapture)
    : isCapture
      ? 'capture'
      : (preset?.qualityMode ?? 'interactive');

  return {
    scenarioId,
    seed: parsePositiveInt(params.get('vitrumSeed'), 12345),
    width: parsePositiveInt(params.get('vitrumWidth'), window.innerWidth || 1280),
    height: parsePositiveInt(params.get('vitrumHeight'), window.innerHeight || 720),
    bounces: parsePositiveInt(
      params.get('vitrumBounces'),
      preset?.bounces ?? defaultBouncesForScenario(scenarioId),
    ),
    samplesTarget: parsePositiveInt(
      params.get('vitrumSpp'),
      isCapture ? DEFAULT_CAPTURE_SPP : (preset?.spp ?? DEFAULT_MANUAL_SPP),
    ),
    samplesPerFrame: params.has('vitrumSpf')
      ? Math.max(1, Math.min(128, parsePositiveInt(params.get('vitrumSpf'), 1)))
      : null,
    causticStrategy,
    isCapture,
    autoStart: params.get('vitrumAutoStart') === '1',
    qualityMode,
    maxInteractiveWidth: preset?.maxW ?? DEFAULT_INTERACTIVE_MAX_W,
    maxInteractiveHeight: preset?.maxH ?? DEFAULT_INTERACTIVE_MAX_H,
    maxMegapixels: parsePositiveMegapixels(params.get('vitrumMaxMp'), 24),
    denoiseDisplay: parseDenoiseDisplay(params),
    oidnModelUrl: params.get('vitrumOidnModel'),
    wgslSigma: parsePositiveFloat(params.get('vitrumWgslSigma'), HDR_LUMINANCE_BILATERAL_DEFAULT_SIGMA_LUMINANCE),
    svgfFrameCount: Math.min(
      parseNonNegativeInt(params.get('vitrumSvgfFrameCount') ?? params.get('vitrumSvgfFrames'), 0),
      SVGF_FRAME_COUNT_INPUT_GUARD_MAX,
    ),
    svgfAtrousIterations: parseSvgfAtrousIterations(params.get('vitrumSvgfAtrous'), SVGF_DEFAULT_ATROUS_ITERATIONS),
    webGpuReuseSharedDevice: params.get('vitrumWebGpuShared') !== '0',
    // Opt-in: additive Σ/count + tile-variance repeats are still experimental in WebGL2.
    pixelAdaptiveSampling: !isCapture && params.get('vitrumAdaptive') === '1',
    pixelAdaptiveCadence: parsePositiveInt(params.get('vitrumAdaptiveCadence'), 4),
  };
}

function applyScenarioMaterialTweaks(
  material: THREE.MeshPhysicalMaterial,
  config: CaptureConfig,
): void {
  if (config.scenarioId.includes('caustic')) {
    material.transmission = 0.5;
    material.ior = 1.5;
    material.thickness = 0.25;
  }

  if (config.scenarioId.includes('spectral') || config.scenarioId.includes('thinfilm')) {
    material.transmission = 0.75;
    material.ior = 1.52;
    material.thickness = 0.4;
    material.attenuationDistance = 1.5;
    material.attenuationColor.setRGB(0.72, 0.9, 1.0);
    material.userData[K.SPECTRAL_ATTEN] = {
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: new Float32Array([
        0.08, 0.1, 0.12, 0.15, 0.2, 0.28, 0.36, 0.44,
        0.52, 0.58, 0.64, 0.68, 0.7, 0.68, 0.62, 0.54,
        0.46, 0.38, 0.31, 0.25, 0.2, 0.16, 0.13, 0.11,
        0.1, 0.09, 0.085, 0.08, 0.078, 0.076, 0.074, 0.072,
      ]),
    };
    material.userData[K.THIN_FILM_STACK] = {
      incidentIor: 1.0,
      angleDependent: true,
      layers: [
        { ior: 2.1, thicknessNm: 72, extinctionCoefficient: 0.015 },
        { ior: 1.46, thicknessNm: 118, extinctionCoefficient: 0.0 },
      ],
    };
  }

  if (config.scenarioId.includes('layered')) {
    material.userData[K.FRONT_LAYER] = { transmission: [0.95, 0.8, 0.65], roughness: 0.18 };
    material.userData[K.BACK_LAYER] = { transmission: [0.65, 0.8, 0.95], roughness: 0.28 };
  }

  if (config.scenarioId.includes('sss')) {
    material.userData[K.SCATTERING_COEFF] = 0.18;
    material.userData[K.SCATTERING_RGB] = [0.16, 0.2, 0.24];
    material.userData[K.SCATTERING_ANISO] = 0.35;
  }
}

function buildCornellScene(config: CaptureConfig): THREE.Scene {
  const scene = new THREE.Scene();
  const white = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, roughness: 1, metalness: 0 });
  const red = new THREE.MeshPhysicalMaterial({ color: 0xab3a2f, roughness: 1, metalness: 0 });
  const green = new THREE.MeshPhysicalMaterial({ color: 0x2d7a3e, roughness: 1, metalness: 0 });
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0x9aa5ad,
    roughness: 0.75,
    metalness: 0,
    transmission: 0,
    ior: 1.5,
    thickness: 0.25,
  });
  applyScenarioMaterialTweaks(glass, config);
  if (config.scenarioId.includes('glass')) {
    glass.transmission = 0.93;
    glass.roughness = 0.06;
    glass.metalness = 0;
    glass.thickness = 0.35;
    glass.ior = 1.52;
  }

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
  const startButton = document.querySelector<HTMLButtonElement>('#start');
  if (!canvas || !statusEl) throw new Error('missing #c or #status');
  const config = parseCaptureConfig();
  const spectralRendering =
    config.scenarioId.includes('spectral') ||
    config.scenarioId.includes('thinfilm') ||
    config.scenarioId.includes('dispersion');
  let lastStatusText = '';
  const setStatus = (message: string): void => {
    if (message === lastStatusText) return;
    lastStatusText = message;
    statusEl.textContent = message;
  };
  globalThis.VITRUM_CAPTURE_READY = false;
  globalThis.VITRUM_MS_PER_SAMPLE = undefined;
  globalThis.VITRUM_CAPTURE_TELEMETRY = undefined;
  globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#c';

  if (!config.autoStart) {
    setStatus('Ready. Press Start WebGL render.');
    if (startButton) startButton.hidden = false;
    canvas.style.cursor = 'pointer';
    await new Promise<void>((resolve) => {
      const start = () => {
        if (startButton) startButton.hidden = true;
        canvas.style.cursor = 'default';
        resolve();
      };
      canvas.addEventListener('click', start, { once: true });
      statusEl.addEventListener('click', start, { once: true });
      startButton?.addEventListener('click', start, { once: true });
    });
  } else if (startButton) {
    startButton.hidden = true;
  }

  setStatus('Creating WebGL renderer...');
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: config.isCapture,
  });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x111111, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const rendererLabel = getWebGLRendererLabel(renderer);
  console.info(`[vitrum-capture] WebGL renderer: ${rendererLabel}`);

  setStatus('Creating camera and scene...');
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  const threeScene = buildCornellScene(config);
  const vitrumScene = sceneFromThreeJS(threeScene);

  setStatus('Creating path-tracing engine...');
  const engine = (await createPTEngine_WebGL2({
    device: renderer,
    maxBounces: config.bounces,
    maxSamplesPerPixel: config.samplesTarget,
    causticStrategy: config.causticStrategy,
    extensions: {
      'vitrum.ptWebgl.spectralRendering': spectralRendering,
      'vitrum.ptWebgl.qualityMode': config.qualityMode,
      'vitrum.ptWebgl.radianceClamp': 0,
      'vitrum.ptWebgl.pixelAdaptiveSampling': config.pixelAdaptiveSampling,
      'vitrum.ptWebgl.pixelAdaptiveCadence': config.pixelAdaptiveCadence,
      'vitrum.ptWebgl.additiveAccumulation': config.pixelAdaptiveSampling,
      // Capture-mode default is 1 sample/frame for telemetry granularity. That
      // ties wall-clock convergence to rAF cadence, which collapses under
      // background-tab throttling. Allow URL override via vitrumSpf so capture
      // sessions can converge in one rAF tick.
      ...(config.samplesPerFrame != null
        ? {
            'vitrum.ptWebgl.samplesPerFrame': config.samplesPerFrame,
            'vitrum.ptWebgl.maxSamplesPerFrame': config.samplesPerFrame,
          }
        : {}),
    },
  })) as PTEngineWebGL2;
  setStatus('Uploading scene to path tracer...');
  engine.setScene(vitrumScene);

  let frame = 0;
  const startMs = performance.now();
  const samplesTarget = config.samplesTarget;
  let renderWidth = 1;
  let renderHeight = 1;

  function resize(): void {
    const displayW = config.isCapture ? config.width : window.innerWidth;
    const displayH = config.isCapture ? config.height : window.innerHeight;
    const [w0, h0] = config.isCapture
      ? [displayW, displayH]
      : fitWithin(displayW, displayH, config.maxInteractiveWidth, config.maxInteractiveHeight);
    const [w, h] = clampMegapixels(w0, h0, config.maxMegapixels);
    renderWidth = w;
    renderHeight = h;
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

  const denoiseCanvas = document.querySelector<HTMLCanvasElement>('#denoise');
  let bilateral: BilateralPreviewCanvas | null = null;
  if (denoiseCanvas != null && config.denoiseDisplay === 'bilateral') {
    bilateral = new BilateralPreviewCanvas(denoiseCanvas);
  }
  let oidnStarted = false;
  let wgslStarted = false;
  let svgfStarted = false;
  let oidnDisplaySucceeded = false;
  let wgslDisplaySucceeded = false;
  let svgfDisplaySucceeded = false;
  let lastDivideByAlpha = false;
  /** Last converged frame output — async denoisers finalize capture using this. */
  let convergedFrame: PTEngineWebGL2FrameOutput | null = null;

  function updateCaptureCanvasHint(): void {
    if (config.denoiseDisplay === 'bilateral' && bilateral != null && denoiseCanvas != null) {
      globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#denoise';
      return;
    }
    if (config.denoiseDisplay === 'oidn' && oidnDisplaySucceeded && denoiseCanvas != null) {
      globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#denoise';
      return;
    }
    if (config.denoiseDisplay === 'wgsl' && wgslDisplaySucceeded && denoiseCanvas != null) {
      globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#denoise';
      return;
    }
    if (config.denoiseDisplay === 'svgf' && svgfDisplaySucceeded && denoiseCanvas != null) {
      globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#denoise';
      return;
    }
    globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#c';
  }

  function finalizeVitrumCapture(reason: string, out: PTEngineWebGL2FrameOutput): void {
    if (globalThis.VITRUM_CAPTURE_READY === true) return;
    globalThis.VITRUM_MS_PER_SAMPLE = (performance.now() - startMs) / Math.max(out.samplesAccumulated, 1);
    globalThis.VITRUM_CAPTURE_TELEMETRY = {
      ...out.telemetry,
      msPerSample: globalThis.VITRUM_MS_PER_SAMPLE,
      samplesAccumulated: out.samplesAccumulated,
      denoiseDisplay: config.denoiseDisplay,
      captureCanvasSelector: globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR ?? '#c',
      captureFinalizeReason: reason,
    };
    globalThis.VITRUM_CAPTURE_READY = true;
    console.info(
      `[vitrum-capture] ready (${reason}) ${config.scenarioId} in ${globalThis.VITRUM_MS_PER_SAMPLE.toFixed(2)} ms/sample`,
    );
  }

  function loop(): void {
    if (frame === 0) {
      setStatus('Rendering first sample...');
    }
    camera.updateMatrixWorld();
    const input: FrameInput = {
      viewMatrix: mat4FromThree(camera.matrixWorldInverse),
      projMatrix: mat4FromThree(camera.projectionMatrix),
      cameraPosition: [camera.position.x, camera.position.y, camera.position.z],
      viewport: {
        width: renderWidth,
        height: renderHeight,
        devicePixelRatio: 1,
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

    const out = engine.renderFrame(input) as PTEngineWebGL2FrameOutput;
    frame++;
    const displayedSpp = Number.isInteger(out.samplesAccumulated)
      ? String(out.samplesAccumulated)
      : out.samplesAccumulated.toFixed(2);
    const telemetry = out.telemetry;
    lastDivideByAlpha = telemetry?.additiveAccumulation ?? false;
    if (out.isConverged) {
      convergedFrame = out;
    }

    if (config.denoiseDisplay === 'bilateral' && bilateral != null && denoiseCanvas != null) {
      canvas.style.visibility = 'hidden';
      denoiseCanvas.style.display = 'block';
      bilateral.render(engine.getAccumulationRenderTarget().texture as THREE.Texture, renderWidth, renderHeight);
    } else if (config.denoiseDisplay !== 'oidn' && config.denoiseDisplay !== 'wgsl' && config.denoiseDisplay !== 'svgf') {
      canvas.style.visibility = 'visible';
      if (denoiseCanvas != null) denoiseCanvas.style.display = 'none';
    }

    /**
     * Run a denoise mode end-to-end: read accum RGB, call the supplied
     * denoise async fn, swap canvas → denoise canvas, finalize capture.
     * Resets the `started` flag on failure so the user can retry.
     */
    const triggerDenoise = (
      mode: 'oidn' | 'wgsl' | 'svgf',
      successLabel: string,
      startedRef: { value: boolean },
      succeededRef: { value: boolean },
      warnText: string,
      denoise: (rgb: Float32Array) => Promise<Float32Array>,
    ): void => {
      if (denoiseCanvas == null || !out.isConverged || startedRef.value || convergedFrame == null) {
        return;
      }
      startedRef.value = true;
      const capFrame = convergedFrame;
      void (async () => {
        try {
          const rt = engine.getAccumulationRenderTarget();
          const rgb = readAccumulationRgbFloat(renderer, rt, renderWidth, renderHeight, lastDivideByAlpha);
          const dod = await denoise(rgb);
          canvas.style.visibility = 'hidden';
          denoiseCanvas.style.display = 'block';
          writeTonemappedRgbToCanvas(denoiseCanvas, dod, renderWidth, renderHeight);
          succeededRef.value = true;
          updateCaptureCanvasHint();
          finalizeVitrumCapture(successLabel, capFrame);
        } catch (err) {
          console.warn(warnText, err);
          canvas.style.visibility = 'visible';
          startedRef.value = false;
          succeededRef.value = false;
          updateCaptureCanvasHint();
          finalizeVitrumCapture(`${mode}-failed`, capFrame);
        }
      })();
    };

    const oidnStartedRef = { value: oidnStarted };
    const oidnSucceededRef = { value: oidnDisplaySucceeded };
    if (
      config.denoiseDisplay === 'oidn' &&
      config.oidnModelUrl != null &&
      config.oidnModelUrl.length > 0
    ) {
      triggerDenoise(
        'oidn',
        'oidn',
        oidnStartedRef,
        oidnSucceededRef,
        '[vitrum-cornell] OIDN failed — install onnxruntime-web and supply vitrumOidnModel URL.',
        async (rgb) => {
          const { denoiseFinal } = await import('@vitrum/shared-denoisers');
          return denoiseFinal(
            { color: rgb, width: renderWidth, height: renderHeight },
            { modelUrl: config.oidnModelUrl! },
          );
        },
      );
    }
    oidnStarted = oidnStartedRef.value;
    oidnDisplaySucceeded = oidnSucceededRef.value;

    const wgslStartedRef = { value: wgslStarted };
    const wgslSucceededRef = { value: wgslDisplaySucceeded };
    if (config.denoiseDisplay === 'wgsl') {
      triggerDenoise(
        'wgsl',
        'wgsl-hdr-bilateral',
        wgslStartedRef,
        wgslSucceededRef,
        '[vitrum-cornell] WebGPU HDR bilateral failed — raw canvas unchanged.',
        async (rgb) => {
          const { runHdrLuminanceBilateralWebGPU } = await import('@vitrum/shared-denoisers');
          return runHdrLuminanceBilateralWebGPU({
            rgb,
            width: renderWidth,
            height: renderHeight,
            sigmaLuminance: config.wgslSigma,
            reuseSharedWebGpuDevice: config.webGpuReuseSharedDevice,
          });
        },
      );
    }
    wgslStarted = wgslStartedRef.value;
    wgslDisplaySucceeded = wgslSucceededRef.value;

    const svgfStartedRef = { value: svgfStarted };
    const svgfSucceededRef = { value: svgfDisplaySucceeded };
    if (config.denoiseDisplay === 'svgf') {
      triggerDenoise(
        'svgf',
        'svgf',
        svgfStartedRef,
        svgfSucceededRef,
        '[vitrum-cornell] SVGF WebGPU failed — raw canvas unchanged.',
        async (rgb) => {
          const { runAtrousVarianceWebGPU } = await import('@vitrum/shared-denoisers');
          return runAtrousVarianceWebGPU({
            rgb,
            width: renderWidth,
            height: renderHeight,
            frameCount: config.svgfFrameCount,
            atrousIterations: config.svgfAtrousIterations,
            reuseSharedWebGpuDevice: config.webGpuReuseSharedDevice,
          });
        },
      );
    }
    svgfStarted = svgfStartedRef.value;
    svgfDisplaySucceeded = svgfSucceededRef.value;

    const perfLabel = telemetry == null
      ? rendererLabel
      : [
        rendererLabel,
        `${telemetry.renderWidth}x${telemetry.renderHeight}`,
        `${telemetry.samplesPerFrame}spf`,
        `${telemetry.tileSize}x${telemetry.tileSize} tiles`,
        telemetry.sppPerSecond == null ? null : `${telemetry.sppPerSecond.toFixed(1)} spp/s`,
        telemetry.estimatedRenderTargetBytes == null
          ? null
          : `~${(telemetry.estimatedRenderTargetBytes / (1024 * 1024)).toFixed(0)} MiB RT`,
        telemetry.additiveAccumulation ? 'sum/count HDR' : null,
        telemetry.pixelAdaptiveSampling ? 'pixel-adaptive sum/count (experimental)' : null,
        telemetry.guardrail,
      ].filter(Boolean).join(' — ');
    const completionLabel = samplesTarget <= SMOKE_SPP_THRESHOLD
      ? ' — smoke complete (grainy by design)'
      : ' — converged';
    setStatus(`${config.scenarioId} (${config.causticStrategy}, ${config.qualityMode}) SPP: ${displayedSpp} / ${samplesTarget}${out.isConverged ? completionLabel : ''} — ${perfLabel}`);
    if (!out.isConverged) {
      requestAnimationFrame(loop);
      return;
    }

    updateCaptureCanvasHint();

    const awaitsPostDenoise =
      (config.denoiseDisplay === 'oidn' &&
        config.oidnModelUrl != null &&
        config.oidnModelUrl.length > 0) ||
      config.denoiseDisplay === 'wgsl' ||
      config.denoiseDisplay === 'svgf';

    if (!awaitsPostDenoise) {
      finalizeVitrumCapture('path-traced', out);
    }
  }

  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
