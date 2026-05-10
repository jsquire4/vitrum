import {
  PerspectiveCamera,
} from 'three';
import type {
  WebGLRenderer,
  WebGLRenderTarget,
  Scene as ThreeScene,
  Material as TMaterial,
  Mesh as TMesh,
  Object3D,
} from 'three';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { WebGLPathTracer } from 'three-gpu-pathtracer';
import type {
  Engine,
  EngineCapabilities,
  EngineFactory,
  EngineOptions,
  EngineState,
} from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import type { Scene, ScenePrimitive, SceneEmitter } from '@vitrum/core';
import { applyFrameToPerspectiveCamera } from './frameCamera.js';
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import { driveForkMaterialUniforms } from './forkUniformBridge.js';
import {
  MAX_TILE_GRID,
  TileVariancePass,
  computeAdaptiveTileRepeatFactors,
} from './adaptiveTileWeights.js';

// ────────────────────────────────────────────────────────────────────────────
// Device-tier threshold for analytic came (Sprint 5)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Minimum value of gl.MAX_FRAGMENT_UNIFORM_VECTORS required to enable the
 * analytic H-channel came intersection path in the fork shader.
 *
 * The came UBO packs up to 500 segments × 16 floats + 200 nodes × 4 floats
 * = 8800 floats, but the shader itself only declares a small fixed number of
 * uniform vector slots for the came struct arrays.  Low-end GPUs that report
 * MAX_FRAGMENT_UNIFORM_VECTORS < 256 (the WebGL minimum guarantee is 16, the
 * common low-end desktop/mobile floor is 256) lack the budget for the came
 * UBO binding.  On those devices the analytic path is disabled and the BVH
 * mesh came geometry is used as fallback (Decision 6 in the roadmap).
 */
const MIN_UNIFORM_VECTORS_FOR_CAME = 256;

export {
  MAX_TILE_GRID,
  TileVariancePass,
  computeAdaptiveTileRepeatFactors,
} from './adaptiveTileWeights.js';
export { readAccumulationRgbFloat } from './readbackHdr.js';
export { vitrumSceneToThree } from '@vitrum/three-bindings';
export { applyFrameToPerspectiveCamera } from './frameCamera.js';
export { packCameUBO } from './cameUniformUploader.js';
export type { CameSegment, CameNode, CameUploadOptions, CamePackedUBO } from './cameUniformUploader.js';

export interface PTEngineWebGL2Options extends EngineOptions {
  readonly device: WebGLRenderer;
}

const DEFAULT_MAX_BOUNCES = 12;
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const DEFAULT_TILE_SIZE = 3;

export type PTEngineWebGL2QualityMode = 'interactive' | 'final' | 'capture' | 'safe';

export interface PTEngineWebGL2Telemetry {
  readonly qualityMode: PTEngineWebGL2QualityMode;
  readonly renderer: string;
  readonly requestedWidth: number;
  readonly requestedHeight: number;
  readonly renderWidth: number;
  readonly renderHeight: number;
  readonly samplesPerFrame: number;
  readonly tileSize: number;
  readonly batchMs: number;
  readonly msPerSample: number | null;
  readonly sppDelta: number;
  readonly sppPerSecond: number | null;
  readonly estimatedRenderTargetBytes: number;
  readonly renderTargetBudgetBytes: number;
  readonly guardrail: string | null;
  readonly additiveAccumulation: boolean;
  readonly pixelAdaptiveSampling: boolean;
}

export type PTEngineWebGL2FrameOutput = FrameOutput & {
  readonly telemetry?: PTEngineWebGL2Telemetry | undefined;
};

interface DeviceLimits {
  readonly maxTextureSize: number;
  readonly maxRenderbufferSize: number;
  readonly renderer: string;
}

interface RenderSizePlan {
  readonly width: number;
  readonly height: number;
  readonly estimatedBytes: number;
  readonly guardrail: string | null;
}

interface SchedulerOptions {
  readonly qualityMode: PTEngineWebGL2QualityMode;
  readonly adaptive: boolean;
  readonly targetBatchMs: number;
  readonly minSamplesPerFrame: number;
  readonly maxSamplesPerFrame: number;
  readonly initialSamplesPerFrame: number;
  readonly initialTileSize: number;
  readonly maxTileSize: number;
  readonly renderTargetBudgetBytes: number;
}

const BYTES_PER_RGBA16F_PIXEL = 8;
const ESTIMATED_RENDER_TARGET_COUNT = 4;
const DEFAULT_RENDER_TARGET_OVERHEAD_BYTES = 64 * 1024 * 1024;

function extensionNumber(
  extensions: Readonly<Record<string, unknown>> | undefined,
  key: string,
  fallback: number,
): number {
  const value = extensions?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function extensionBoolean(
  extensions: Readonly<Record<string, unknown>> | undefined,
  key: string,
  fallback: boolean,
): boolean {
  const value = extensions?.[key];
  return typeof value === 'boolean' ? value : fallback;
}

function extensionQualityMode(
  extensions: Readonly<Record<string, unknown>> | undefined,
): PTEngineWebGL2QualityMode {
  const value = extensions?.['vitrum.ptWebgl.qualityMode'];
  return value === 'interactive' || value === 'final' || value === 'capture' || value === 'safe'
    ? value
    : 'capture';
}

function clampInt(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.floor(value)));
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function defaultSchedulerOptions(
  extensions: Readonly<Record<string, unknown>> | undefined,
): SchedulerOptions {
  const qualityMode = extensionQualityMode(extensions);
  const modeDefaults: Record<PTEngineWebGL2QualityMode, Omit<SchedulerOptions, 'qualityMode'>> = {
    interactive: {
      adaptive: true,
      targetBatchMs: 40,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 48,
      initialSamplesPerFrame: 6,
      initialTileSize: 1,
      maxTileSize: 4,
      renderTargetBudgetBytes: 1024 * 1024 * 1024,
    },
    final: {
      adaptive: true,
      targetBatchMs: 120,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 96,
      initialSamplesPerFrame: 12,
      initialTileSize: 1,
      maxTileSize: 4,
      renderTargetBudgetBytes: 2 * 1024 * 1024 * 1024,
    },
    capture: {
      adaptive: false,
      targetBatchMs: 0,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 1,
      initialSamplesPerFrame: 1,
      initialTileSize: DEFAULT_TILE_SIZE,
      maxTileSize: DEFAULT_TILE_SIZE,
      renderTargetBudgetBytes: 512 * 1024 * 1024,
    },
    safe: {
      adaptive: true,
      targetBatchMs: 8,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 4,
      initialSamplesPerFrame: 1,
      initialTileSize: DEFAULT_TILE_SIZE,
      maxTileSize: 4,
      renderTargetBudgetBytes: 256 * 1024 * 1024,
    },
  };
  const base = modeDefaults[qualityMode];
  const requestedSamplesPerFrame = extensionNumber(
    extensions,
    'vitrum.ptWebgl.samplesPerFrame',
    base.initialSamplesPerFrame,
  );
  const requestedTileSize = extensionNumber(extensions, 'vitrum.ptWebgl.tileSize', base.initialTileSize);
  const maxSamplesPerFrame = clampInt(
    extensionNumber(extensions, 'vitrum.ptWebgl.maxSamplesPerFrame', base.maxSamplesPerFrame),
    base.minSamplesPerFrame,
    128,
  );
  return {
    qualityMode,
    adaptive: extensionBoolean(extensions, 'vitrum.ptWebgl.adaptiveScheduler', base.adaptive),
    targetBatchMs: Math.max(0, extensionNumber(extensions, 'vitrum.ptWebgl.targetBatchMs', base.targetBatchMs)),
    minSamplesPerFrame: base.minSamplesPerFrame,
    maxSamplesPerFrame,
    initialSamplesPerFrame: clampInt(requestedSamplesPerFrame, base.minSamplesPerFrame, maxSamplesPerFrame),
    initialTileSize: clampInt(requestedTileSize, 1, base.maxTileSize),
    maxTileSize: clampInt(extensionNumber(extensions, 'vitrum.ptWebgl.maxTileSize', base.maxTileSize), 1, 8),
    renderTargetBudgetBytes: Math.max(
      64 * 1024 * 1024,
      extensionNumber(extensions, 'vitrum.ptWebgl.renderTargetBudgetBytes', base.renderTargetBudgetBytes),
    ),
  };
}

/**
 * Internal state-setter token. The factory constructs a `StateSlot`, passes it
 * to `PTEngineWebGL2`, then calls `slot.set('ready')`. External callers cannot
 * obtain the slot; only the factory transitions `initializing` → `ready`.
 */
interface StateSlot {
  readonly get: () => EngineState;
  readonly set: (s: EngineState) => void;
}

function makeStateSlot(initial: EngineState = 'initializing'): StateSlot {
  let s: EngineState = initial;
  return {
    get: () => s,
    set: (v) => {
      s = v;
    },
  };
}

function disposeObject3DTree(obj: Object3D): void {
  obj.traverse((o) => {
    const mesh = o as TMesh;
    if (mesh.isMesh === true) {
      mesh.geometry?.dispose();
      const m = mesh.material as TMaterial | TMaterial[] | undefined;
      if (Array.isArray(m)) m.forEach((x) => {
        x.dispose?.();
      });
      else m?.dispose?.();
    }
  });
}

interface PTEngineWebGL2Init {
  readonly renderer: WebGLRenderer;
  readonly pathTracer: WebGLPathTracer;
  readonly camera: PerspectiveCamera;
  /** True when the GL context reports MAX_FRAGMENT_UNIFORM_VECTORS >= 256.
   *  Controls whether 'h-channel-came' is included in supportedAnalyticShapes. */
  readonly supportsAnalyticCame: boolean;
}

export class PTEngineWebGL2 implements Engine {
  readonly #slot: StateSlot;

  readonly #renderer: WebGLRenderer;
  readonly #pathTracer: WebGLPathTracer;
  readonly #camera: PerspectiveCamera;
  readonly #supportsAnalyticCame: boolean;

  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;
  readonly #spectralRendering: boolean;
  readonly #radianceClamp: number;
  readonly #limits: DeviceLimits;
  readonly #schedulerOptions: SchedulerOptions;

  #vitrumScene: Scene | null = null;
  #threeSceneRoot: ThreeScene | null = null;
  #cameraSignature = '';
  #samplesPerFrame: number;
  #tileSize: number;
  #lastRenderWidth = 0;
  #lastRenderHeight = 0;
  #contextLost = false;
  #lastTelemetry: PTEngineWebGL2Telemetry | undefined;
  #additiveAccumulation = false;
  #pixelAdaptiveSampling = false;
  #pixelAdaptiveCadence = 4;
  #tileVariancePass: TileVariancePass | null = null;
  #tileFactorsScratch: Uint8Array = new Uint8Array(MAX_TILE_GRID * MAX_TILE_GRID);

  constructor(opts: PTEngineWebGL2Options, gpu: PTEngineWebGL2Init, slot: StateSlot) {
    this.#slot = slot;
    this.#maxBouncesLimit = opts.maxBounces ?? DEFAULT_MAX_BOUNCES;
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    // RFE-05: strategy is forwarded to fork uniforms and mirrored in `capabilities.causticStrategy`.
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    this.#mneeMaxIterations = Math.max(1, opts.mneeMaxIterations ?? 8);
    this.#mneeMaxChainLength = Math.max(1, opts.mneeMaxChainLength ?? 3);
    this.#spectralRendering = opts.extensions?.['vitrum.ptWebgl.spectralRendering'] === true;
    const requestedRadianceClamp = opts.extensions?.['vitrum.ptWebgl.radianceClamp'];
    this.#radianceClamp = typeof requestedRadianceClamp === 'number' && Number.isFinite(requestedRadianceClamp)
      ? Math.max(0, requestedRadianceClamp)
      : 0;
    this.#schedulerOptions = defaultSchedulerOptions(opts.extensions);
    this.#samplesPerFrame = this.#schedulerOptions.initialSamplesPerFrame;
    this.#tileSize = this.#schedulerOptions.initialTileSize;
    this.#renderer = gpu.renderer;
    this.#pathTracer = gpu.pathTracer;
    this.#camera = gpu.camera;
    this.#supportsAnalyticCame = gpu.supportsAnalyticCame;
    this.#limits = this.#detectDeviceLimits();
    const adaptiveRequested = opts.extensions?.['vitrum.ptWebgl.pixelAdaptiveSampling'] === true;
    this.#additiveAccumulation =
      opts.extensions?.['vitrum.ptWebgl.additiveAccumulation'] === true || adaptiveRequested;
    this.#pixelAdaptiveSampling = adaptiveRequested;
    this.#pixelAdaptiveCadence = Math.max(
      1,
      extensionNumber(opts.extensions, 'vitrum.ptWebgl.pixelAdaptiveCadence', 4),
    );
    this.#tileFactorsScratch.fill(1);
    if (this.#pixelAdaptiveSampling) {
      this.#tileVariancePass = new TileVariancePass(MAX_TILE_GRID);
    }
    gpu.pathTracer.configureAdditiveAccumulation(this.#additiveAccumulation, this.#additiveAccumulation);
    this.#renderer.domElement?.addEventListener?.('webglcontextlost', () => {
      this.#contextLost = true;
      this.#samplesPerFrame = 1;
      this.#tileSize = Math.max(this.#tileSize, DEFAULT_TILE_SIZE);
    });
    this.#pathTracer.renderDelay = 0;
    this.#pathTracer.minSamples = 0;
    this.#pathTracer.fadeDuration = 0;
  }

  get state(): EngineState {
    return this.#slot.get();
  }

  get capabilities(): EngineCapabilities {
    // Analytic H-channel came is enabled only when the GL context has
    // sufficient fragment uniform vectors (Sprint 5 device-tier fallback).
    // On low-end GPUs (MAX_FRAGMENT_UNIFORM_VECTORS < 256) the came UBO
    // is too expensive; mesh-came geometry in the BVH is used as fallback.
    const analyticShapes = new Set<string>();
    if (this.#supportsAnalyticCame) {
      analyticShapes.add('h-channel-came');
    }

    return {
      supportsIncrementalScene: false,
      supportsMotionBlur: false,
      supportsAuxBuffers: false,
      accumulates: true,
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      supportedAnalyticShapes: analyticShapes,
      supportedEmitterKinds: new Set<string>([
        'directional',
        'rect-area',
        'disc-area',
        'point',
        'spot',
        'mesh-area',
      ]),
      causticStrategy: this.#causticStrategy,
    };
  }

  #assertLive(method: string): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error(`${method}: engine is disposed`);
    }
    if (this.#vitrumScene == null) {
      throw new Error(`${method}: call setScene() before ${method}`);
    }
  }

  #detectDeviceLimits(): DeviceLimits {
    const gl = this.#renderer.getContext();
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    const rendererFromDebug = debugInfo == null
      ? null
      : gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    const rendererFallback = gl.getParameter(gl.RENDERER);
    const maxTextureSize = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE));
    const maxRenderbufferSize = Number(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE));
    return {
      maxTextureSize: Number.isFinite(maxTextureSize) && maxTextureSize > 0 ? maxTextureSize : 4096,
      maxRenderbufferSize: Number.isFinite(maxRenderbufferSize) && maxRenderbufferSize > 0
        ? maxRenderbufferSize
        : 4096,
      renderer: typeof rendererFromDebug === 'string' && rendererFromDebug.length > 0
        ? rendererFromDebug
        : typeof rendererFallback === 'string'
          ? rendererFallback
          : 'unknown WebGL renderer',
    };
  }

  #estimateRenderTargetBytes(width: number, height: number): number {
    return (
      width *
      height *
      BYTES_PER_RGBA16F_PIXEL *
      ESTIMATED_RENDER_TARGET_COUNT +
      DEFAULT_RENDER_TARGET_OVERHEAD_BYTES
    );
  }

  #planRenderSize(width: number, height: number): RenderSizePlan {
    const requestedWidth = Math.max(1, Math.floor(width));
    const requestedHeight = Math.max(1, Math.floor(height));
    const maxDimension = Math.max(1, Math.min(this.#limits.maxTextureSize, this.#limits.maxRenderbufferSize));
    let scale = Math.min(1, maxDimension / requestedWidth, maxDimension / requestedHeight);
    let guardrail: string | null = scale < 1
      ? `capped to WebGL max render dimension ${maxDimension}`
      : null;
    let plannedWidth = Math.max(1, Math.floor(requestedWidth * scale));
    let plannedHeight = Math.max(1, Math.floor(requestedHeight * scale));
    let estimatedBytes = this.#estimateRenderTargetBytes(plannedWidth, plannedHeight);
    if (estimatedBytes > this.#schedulerOptions.renderTargetBudgetBytes) {
      const targetBytes = Math.max(
        1,
        this.#schedulerOptions.renderTargetBudgetBytes - DEFAULT_RENDER_TARGET_OVERHEAD_BYTES,
      );
      const pixelBytes = Math.max(1, plannedWidth * plannedHeight * BYTES_PER_RGBA16F_PIXEL * ESTIMATED_RENDER_TARGET_COUNT);
      const memoryScale = Math.min(1, Math.sqrt(targetBytes / pixelBytes));
      scale *= memoryScale;
      plannedWidth = Math.max(1, Math.floor(requestedWidth * scale));
      plannedHeight = Math.max(1, Math.floor(requestedHeight * scale));
      estimatedBytes = this.#estimateRenderTargetBytes(plannedWidth, plannedHeight);
      guardrail = guardrail == null
        ? `downscaled to fit ${Math.round(this.#schedulerOptions.renderTargetBudgetBytes / 1024 / 1024)} MiB render-target budget`
        : `${guardrail}; downscaled to fit render-target budget`;
    }
    return {
      width: plannedWidth,
      height: plannedHeight,
      estimatedBytes,
      guardrail,
    };
  }

  #updateScheduler(batchMs: number): void {
    if (!this.#schedulerOptions.adaptive || this.#schedulerOptions.targetBatchMs <= 0) return;
    if (this.#contextLost) {
      this.#samplesPerFrame = 1;
      this.#tileSize = Math.min(this.#schedulerOptions.maxTileSize, Math.max(this.#tileSize, DEFAULT_TILE_SIZE));
      return;
    }
    const target = this.#schedulerOptions.targetBatchMs;
    if (batchMs > target * 1.35) {
      this.#samplesPerFrame = Math.max(
        this.#schedulerOptions.minSamplesPerFrame,
        Math.floor(this.#samplesPerFrame * 0.5),
      );
      if (batchMs > target * 2 && this.#tileSize < this.#schedulerOptions.maxTileSize) {
        this.#tileSize += 1;
      }
      return;
    }
    if (batchMs < target * 0.55 && this.#samplesPerFrame < this.#schedulerOptions.maxSamplesPerFrame) {
      this.#samplesPerFrame = Math.min(
        this.#schedulerOptions.maxSamplesPerFrame,
        Math.max(this.#samplesPerFrame + 1, Math.ceil(this.#samplesPerFrame * 1.2)),
      );
      if (batchMs < target * 0.25 && this.#tileSize > this.#schedulerOptions.initialTileSize) {
        this.#tileSize -= 1;
      }
    }
  }

  setScene(scene: Scene): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('setScene: engine is disposed');
    }
    if (this.#threeSceneRoot != null) {
      disposeObject3DTree(this.#threeSceneRoot);
    }
    this.#vitrumScene = scene;
    this.#cameraSignature = '';
    const threeScene = vitrumSceneToThree(scene);
    this.#threeSceneRoot = threeScene;
    this.#pathTracer.setScene(
      threeScene as unknown as Parameters<WebGLPathTracer['setScene']>[0],
      this.#camera as unknown as Parameters<WebGLPathTracer['setScene']>[1],
    );
    driveForkMaterialUniforms(this.#pathTracer, threeScene, {
      strategy: this.#causticStrategy,
      mneeMaxIterations: this.#mneeMaxIterations,
      mneeMaxChainLength: this.#mneeMaxChainLength,
      spectralRendering: this.#spectralRendering,
      radianceClamp: this.#radianceClamp,
    });
  }

  updatePrimitive(_id: string, _patch: Partial<ScenePrimitive>): void {
    this.#assertLive('updatePrimitive');
    throw new Error('Not implemented: updatePrimitive (pt-webgl requires full setScene)');
  }

  updateEmitter(_id: string, _patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    throw new Error('Not implemented: updateEmitter (pt-webgl requires full setScene)');
  }

  #makeCameraSignature(input: FrameInput): string {
    const viewport = input.viewport;
    return [
      ...input.viewMatrix,
      ...input.projMatrix,
      ...input.cameraPosition,
      viewport.width,
      viewport.height,
      viewport.devicePixelRatio,
    ].join(',');
  }

  renderFrame(input: FrameInput): PTEngineWebGL2FrameOutput {
    this.#assertLive('renderFrame');
    if (this.#slot.get() === 'paused') {
      const spp = this.#pathTracer.samples;
      const cap = this.#maxSamplesLimit;
      return {
        primaryRadiance: this.#pathTracer.target.texture,
        samplesAccumulated: spp,
        isConverged: spp >= cap,
        telemetry: this.#lastTelemetry,
      };
    }

    const cameraSignature = this.#makeCameraSignature(input);
    if (cameraSignature !== this.#cameraSignature) {
      applyFrameToPerspectiveCamera(this.#camera, input);
      this.#pathTracer.setCamera(
        this.#camera as unknown as Parameters<WebGLPathTracer['setCamera']>[0],
      );
      this.#cameraSignature = cameraSignature;
    }

    const q = input.quality ?? {};
    const b = Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit);
    const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
    this.#pathTracer.bounces = b;
    this.#pathTracer.transmissiveBounces = Math.min(b, 12);
    this.#pathTracer.filterGlossyFactor = q.filteredGlossyFactor ?? 0;

    const factor = q.resolutionFactor ?? 1;
    const requestedWidth = Math.max(1, Math.floor(input.viewport.width * factor));
    const requestedHeight = Math.max(1, Math.floor(input.viewport.height * factor));
    const sizePlan = this.#planRenderSize(requestedWidth, requestedHeight);
    const w = sizePlan.width;
    const h = sizePlan.height;
    if (this.#tileSize <= 1 || w * h <= 640 * 360) {
      this.#pathTracer.tiles.set(1, 1);
    } else {
      this.#pathTracer.tiles.set(this.#tileSize, this.#tileSize);
    }
    if (w !== this.#lastRenderWidth || h !== this.#lastRenderHeight) {
      this.#renderer.setSize(w, h, false);
      this.#lastRenderWidth = w;
      this.#lastRenderHeight = h;
    }

    let spp = this.#pathTracer.samples;
    const sppBefore = spp;
    const batchStart = nowMs();
    const samplesThisFrame = Math.min(this.#samplesPerFrame, Math.max(0, Math.ceil(targetSpp - spp)));

    const tilesX = Math.max(1, Math.floor(this.#pathTracer.tiles.x));
    const tilesY = Math.max(1, Math.floor(this.#pathTracer.tiles.y));
    const tileCount = tilesX * tilesY;

    if (this.#pixelAdaptiveSampling && this.#additiveAccumulation && this.#tileVariancePass != null) {
      if (sppBefore >= 2 && sppBefore % this.#pixelAdaptiveCadence === 0) {
        computeAdaptiveTileRepeatFactors(
          this.#tileVariancePass,
          this.#renderer,
          this.#pathTracer.target.texture as unknown as import('three').Texture,
          w,
          h,
          tilesX,
          tilesY,
          this.#tileFactorsScratch,
        );
        this.#pathTracer.tileRepeatFactors = this.#tileFactorsScratch.subarray(0, tileCount);
      }
    } else if (!this.#pixelAdaptiveSampling) {
      this.#pathTracer.tileRepeatFactors = null;
    }

    for (let i = 0; i < samplesThisFrame && spp < targetSpp; i += 1) {
      this.#pathTracer.renderSample();
      spp = this.#pathTracer.samples;
    }
    const batchMs = Math.max(0, nowMs() - batchStart);
    const sppDelta = Math.max(0, spp - sppBefore);
    this.#updateScheduler(batchMs);
    const msPerSample = sppDelta > 0 ? batchMs / sppDelta : null;
    this.#lastTelemetry = {
      qualityMode: this.#schedulerOptions.qualityMode,
      renderer: this.#limits.renderer,
      requestedWidth,
      requestedHeight,
      renderWidth: w,
      renderHeight: h,
      samplesPerFrame: samplesThisFrame,
      tileSize: this.#tileSize <= 1 || w * h <= 640 * 360 ? 1 : this.#tileSize,
      batchMs,
      msPerSample,
      sppDelta,
      sppPerSecond: sppDelta > 0 && batchMs > 0 ? (sppDelta * 1000) / batchMs : null,
      estimatedRenderTargetBytes: sizePlan.estimatedBytes,
      renderTargetBudgetBytes: this.#schedulerOptions.renderTargetBudgetBytes,
      guardrail: this.#contextLost
        ? 'webgl context loss observed; scheduler reduced workload'
        : sizePlan.guardrail,
      additiveAccumulation: this.#additiveAccumulation,
      pixelAdaptiveSampling: this.#pixelAdaptiveSampling,
    };
    return {
      primaryRadiance: this.#pathTracer.target.texture,
      samplesAccumulated: spp,
      isConverged: spp >= targetSpp,
      telemetry: this.#lastTelemetry,
    };
  }

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#pathTracer.reset();
  }

  /**
   * HDR accumulation buffer backing `primaryRadiance` / fork canvas preview.
   * Texels are running averages unless additive accumulation is enabled (then RGB sum / alpha count).
   */
  getAccumulationRenderTarget(): WebGLRenderTarget {
    this.#assertLive('getAccumulationRenderTarget');
    return this.#pathTracer.target as unknown as WebGLRenderTarget;
  }

  pause(): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('pause: engine is disposed');
    }
    this.#slot.set('paused');
  }

  resume(): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('resume: engine is disposed');
    }
    this.#slot.set('ready');
  }

  dispose(): void {
    if (this.#slot.get() === 'disposed') return;
    if (this.#tileVariancePass != null) {
      this.#tileVariancePass.dispose();
      this.#tileVariancePass = null;
    }
    if (this.#threeSceneRoot != null) {
      disposeObject3DTree(this.#threeSceneRoot);
      this.#threeSceneRoot = null;
    }
    this.#pathTracer.dispose();
    this.#slot.set('disposed');
  }
}

export * from './constants.js';
export * from './sunGeometry.js';
export { bakeSkyEquirect, clearSkyEquirectCache } from './iblBaker.js';
export { debounceMsForEditRate, PT_DEBOUNCE_MS_NORMAL, PT_DEBOUNCE_MS_BURST } from './debounce.js';
export { computeLightingState } from './lightingState.js';
export type { LightingState, LightingStateInputs } from './lightingState.js';
export { skyParamsFor, worldSunPosition, SUN_LIGHT_DISTANCE } from './skyParams.js';
export type { SkyParams } from './skyParams.js';
export {
  COLOR_TEMP_HEX,
  SUN_INTENSITY,
  getSunIntensity,
  pointIntensityFromLumens,
  rectAreaIntensityFromLumens,
} from './lightingIntensityTable.js';

export const createPTEngine_WebGL2: EngineFactory<PTEngineWebGL2Options> = async (
  opts: PTEngineWebGL2Options,
): Promise<Engine> => {
  if (opts.device == null || typeof (opts.device as { getContext?: unknown }).getContext !== 'function') {
    throw new TypeError(
      'createPTEngine_WebGL2: device must be a THREE.WebGLRenderer instance (got null/undefined or an object without a getContext() method)',
    );
  }
  const glContext = (opts.device as { getContext(): unknown }).getContext();
  if (!(glContext instanceof WebGL2RenderingContext)) {
    throw new TypeError(
      'createPTEngine_WebGL2: device.getContext() must return a WebGL2RenderingContext; WebGL1 is not supported',
    );
  }

  const maxBounces = opts.maxBounces;
  if (maxBounces !== undefined && maxBounces < 1) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxBounces structural cap must be >= 1 (got ${maxBounces})`,
    );
  }

  const maxSpp = opts.maxSamplesPerPixel;
  if (maxSpp !== undefined && maxSpp < 1) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxSamplesPerPixel structural cap must be >= 1 (got ${maxSpp})`,
    );
  }

  const renderer = opts.device as WebGLRenderer;
  RectAreaLightUniformsLib.init();

  const pathTracer = new WebGLPathTracer(
    renderer as unknown as ConstructorParameters<typeof WebGLPathTracer>[0],
  );
  pathTracer.renderDelay = 0;
  pathTracer.minSamples = 1;
  pathTracer.dynamicLowRes = false;
  pathTracer.multipleImportanceSampling = true;
  pathTracer.tiles.set(1, 1);

  const camera = new PerspectiveCamera();

  // Sprint 5 device-tier fallback: query MAX_FRAGMENT_UNIFORM_VECTORS once at
  // engine creation.  This parameter is fixed for a GL context's lifetime so
  // there's no need to query it per-frame.  The result is baked into the
  // engine's capabilities and does not change.
  const maxFragUniforms = glContext.getParameter(glContext.MAX_FRAGMENT_UNIFORM_VECTORS) as number;
  const supportsAnalyticCame = maxFragUniforms >= MIN_UNIFORM_VECTORS_FOR_CAME;

  const slot = makeStateSlot();
  const engine = new PTEngineWebGL2(
    opts,
    { renderer, pathTracer, camera, supportsAnalyticCame },
    slot,
  );
  slot.set('ready');
  return engine;
}
