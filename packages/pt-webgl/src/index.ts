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

const CAPABILITIES: EngineCapabilities = {
  supportsIncrementalScene: false,
  supportsMotionBlur: false,
  supportsAuxBuffers: false,
  accumulates: true,
  maxSamplesPerPixel: 4096,
  maxBounces: 12,
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

class PTEngineWebGL2 implements Engine {
  #state: EngineState = 'initializing';

  /** Stored for Sprint 1 render wiring. */
  readonly samplesPerPixel: number;
  readonly maxBounces: number;
  readonly denoiser: EngineOptions['denoiser'];
  readonly resolutionFactor: number | undefined;
  readonly extensions: Readonly<Record<string, unknown>> | undefined;

  constructor(opts: PTEngineWebGL2Options) {
    this.samplesPerPixel = opts.samplesPerPixel ?? 1;
    this.maxBounces = opts.maxBounces ?? 5;
    this.denoiser = opts.denoiser;
    this.resolutionFactor = opts.resolutionFactor;
    this.extensions = opts.extensions;
  }

  get state(): EngineState {
    return this.#state;
  }

  get capabilities(): EngineCapabilities {
    return CAPABILITIES;
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

  const spp = opts.samplesPerPixel;
  if (spp !== undefined && spp < 1) {
    throw new RangeError(
      `createPTEngine_WebGL2: samplesPerPixel must be >= 1 (got ${spp})`,
    );
  }

  const bounces = opts.maxBounces;
  if (bounces !== undefined && bounces < 0) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxBounces must be >= 0 (got ${bounces})`,
    );
  }

  const rf = opts.resolutionFactor;
  if (rf !== undefined && (rf <= 0 || rf > 1)) {
    throw new RangeError(
      `createPTEngine_WebGL2: resolutionFactor must be in (0, 1] (got ${rf})`,
    );
  }

  const engine = new PTEngineWebGL2(opts);

  // Sprint 1 async GPU init (BVH upload, shader compile) goes here
  engine._markReady();

  return engine;
}
