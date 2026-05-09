import type {
  Engine,
  EngineCapabilities,
  EngineOptions,
  EngineState,
} from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import type { Scene, ScenePrimitive, SceneEmitter } from '@vitrum/core';

export interface PTEngineWebGL2Options extends EngineOptions {
  // WebGL2 required — three-gpu-pathtracer's BVH texture lookup uses `textureGather`
  readonly device: WebGLRenderingContext | WebGL2RenderingContext;
}

/** Default structural caps for the pt-webgl backend. These match the
 *  three-gpu-pathtracer fork's shader-compile-time limits. Override via
 *  `PTEngineWebGL2Options.maxBounces` / `.maxSamplesPerPixel` at engine
 *  creation if your use case needs different allocation bounds. */
const DEFAULT_MAX_BOUNCES = 12;
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;

class PTEngineWebGL2 implements Engine {
  #state: EngineState = 'initializing';

  /** Structural cap: per-path bounce limit chosen at engine creation.
   *  Exposed via `capabilities.maxBounces`. Per-frame
   *  `FrameInput.quality.bounces` is clamped to this value during Sprint 1
   *  render wiring. */
  readonly #maxBouncesLimit: number;

  /** Structural cap: samples-per-pixel ceiling chosen at engine creation.
   *  Exposed via `capabilities.maxSamplesPerPixel`. Per-frame
   *  `FrameInput.quality.samplesTarget` is clamped to this value during
   *  Sprint 1 render wiring. */
  readonly #maxSamplesLimit: number;

  /** Denoiser pipeline wired at engine creation. Changing this requires
   *  shader recompilation — the host must dispose and recreate the engine. */
  readonly #denoiser: EngineOptions['denoiser'];

  /** Backend-specific creation-time extensions. */
  readonly #extensions: Readonly<Record<string, unknown>> | undefined;

  constructor(opts: PTEngineWebGL2Options) {
    this.#maxBouncesLimit = opts.maxBounces ?? DEFAULT_MAX_BOUNCES;
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#denoiser = opts.denoiser;
    this.#extensions = opts.extensions;
  }

  get state(): EngineState {
    return this.#state;
  }

  get capabilities(): EngineCapabilities {
    return {
      supportsIncrementalScene: false,
      supportsMotionBlur: false,
      supportsAuxBuffers: false,
      accumulates: true,
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      supportedAnalyticShapes: new Set<string>(),
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

  /** Called by the factory immediately after construction. */
  _markReady(): void {
    this.#state = 'ready';
  }

  #assertLive(method: string): void {
    if (this.#state === 'disposed') {
      throw new Error(`${method}: engine is disposed`);
    }
  }

  setScene(_scene: Scene): void {
    this.#assertLive('setScene');
    throw new Error('Not implemented: setScene');
  }

  updatePrimitive(_id: string, _patch: Partial<ScenePrimitive>): void {
    this.#assertLive('updatePrimitive');
    throw new Error('Not implemented: updatePrimitive');
  }

  updateEmitter(_id: string, _patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    throw new Error('Not implemented: updateEmitter');
  }

  renderFrame(_input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');
    throw new Error('Not implemented: renderFrame');
  }

  reset(): void {
    this.#assertLive('reset');
    // No accumulator state yet; Sprint 1 will reset sample counter here
  }

  pause(): void {
    this.#assertLive('pause');
    this.#state = 'paused';
  }

  resume(): void {
    this.#assertLive('resume');
    this.#state = 'ready';
  }

  dispose(): void {
    this.#state = 'disposed';
  }
}

export async function createPTEngine_WebGL2(
  opts: PTEngineWebGL2Options,
): Promise<Engine> {
  if (!(opts.device instanceof WebGL2RenderingContext)) {
    throw new TypeError(
      'createPTEngine_WebGL2: device must be a WebGL2RenderingContext; WebGL1 is not supported',
    );
  }

  // Structural cap validations — these govern buffer allocation, not per-frame
  // behavior. Per-frame quality (samplesTarget, bounces, resolutionFactor)
  // is validated at renderFrame time in Sprint 1.
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

  const engine = new PTEngineWebGL2(opts);

  // Sprint 1 async GPU init (BVH upload, shader compile) goes here
  engine._markReady();

  return engine;
}
