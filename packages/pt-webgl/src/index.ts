import {
  PerspectiveCamera,
  WebGLRenderer,
  Scene as ThreeScene,
  type Material as TMaterial,
  type Mesh as TMesh,
  type Object3D,
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

export { vitrumSceneToThree } from '@vitrum/three-bindings';
export { applyFrameToPerspectiveCamera } from './frameCamera.js';
export { packCameUBO } from './cameUniformUploader.js';
export type { CameSegment, CameNode, CameUploadOptions, CamePackedUBO } from './cameUniformUploader.js';

export interface PTEngineWebGL2Options extends EngineOptions {
  readonly device: WebGLRenderer;
}

const DEFAULT_MAX_BOUNCES = 12;
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;

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
      if (Array.isArray(m)) m.forEach((x) => x.dispose?.());
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

class PTEngineWebGL2 implements Engine {
  readonly #slot: StateSlot;

  readonly #renderer: WebGLRenderer;
  readonly #pathTracer: WebGLPathTracer;
  readonly #camera: PerspectiveCamera;
  readonly #supportsAnalyticCame: boolean;

  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;

  #vitrumScene: Scene | null = null;
  #threeSceneRoot: ThreeScene | null = null;

  constructor(opts: PTEngineWebGL2Options, gpu: PTEngineWebGL2Init, slot: StateSlot) {
    this.#slot = slot;
    this.#maxBouncesLimit = opts.maxBounces ?? DEFAULT_MAX_BOUNCES;
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#renderer = gpu.renderer;
    this.#pathTracer = gpu.pathTracer;
    this.#camera = gpu.camera;
    this.#supportsAnalyticCame = gpu.supportsAnalyticCame;
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

  setScene(scene: Scene): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('setScene: engine is disposed');
    }
    if (this.#threeSceneRoot != null) {
      disposeObject3DTree(this.#threeSceneRoot);
    }
    this.#vitrumScene = scene;
    const threeScene = vitrumSceneToThree(scene);
    this.#threeSceneRoot = threeScene;
    this.#pathTracer.setScene(threeScene, this.#camera);
  }

  updatePrimitive(_id: string, _patch: Partial<ScenePrimitive>): void {
    this.#assertLive('updatePrimitive');
    throw new Error('Not implemented: updatePrimitive (pt-webgl requires full setScene)');
  }

  updateEmitter(_id: string, _patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    throw new Error('Not implemented: updateEmitter (pt-webgl requires full setScene)');
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');
    if (this.#slot.get() === 'paused') {
      const spp = Math.round(this.#pathTracer.samples);
      const cap = this.#maxSamplesLimit;
      return {
        primaryRadiance: this.#pathTracer.target.texture,
        samplesAccumulated: spp,
        isConverged: spp >= cap,
      };
    }

    applyFrameToPerspectiveCamera(this.#camera, input);
    this.#pathTracer.setCamera(this.#camera);

    const q = input.quality ?? {};
    const b = Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit);
    const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
    this.#pathTracer.bounces = b;
    this.#pathTracer.transmissiveBounces = Math.min(b, 12);
    this.#pathTracer.filterGlossyFactor = q.filteredGlossyFactor ?? 0;

    const factor = q.resolutionFactor ?? 1;
    const w = Math.max(1, Math.floor(input.viewport.width * factor));
    const h = Math.max(1, Math.floor(input.viewport.height * factor));
    this.#renderer.setSize(w, h, false);

    this.#pathTracer.renderSample();
    const spp = Math.round(this.#pathTracer.samples);
    return {
      primaryRadiance: this.#pathTracer.target.texture,
      samplesAccumulated: spp,
      isConverged: spp >= targetSpp,
    };
  }

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#pathTracer.reset();
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

  const pathTracer = new WebGLPathTracer(renderer);
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
