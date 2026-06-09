import type {
  Engine,
  EngineCapabilities,
  EngineFactory,
  EngineState,
  FrameInput,
  FrameOutput,
  FrameStats,
  ProgressStats,
  Scene,
} from '@vitrum/core';
import { partitionSceneBySupport } from '@vitrum/core';
import { packSceneFromCore, type ScenePackResult } from '@vitrum/shared-bvh';
import { buildCapabilities } from './capabilities.js';
import { makeStateSlot, type StateSlot } from './state.js';
import type { PTEngineWebGL2Options, WebGl2TraceTier } from './options.js';

const DEFAULT_MAX_SPP = 4096;
const DEFAULT_MAX_BOUNCES = 32;

/**
 * Slice 0 surface — the converged display texture once the GL pipeline lands.
 * (No `getDenoisedFrame` yet; OIDN wires in a later slice.)
 */
export interface PTEngineWebGL2Surface {
  /** @internal Slice 0: the retained ScenePackResult (for tests/inspection). */
  readonly _debugGeoPack: ScenePackResult | null;
}

/**
 * THREE-free WebGL2 path-tracing backend. Slice 0 = the contract spine + scene
 * ingestion (shared-bvh pack); the GL render pipeline (GlResources/GlProgram + the
 * GLSL kernels) is the next increment and gated on a real WebGL2 capture host, so
 * `renderFrame` returns a legal `FrameSkipped` until then.
 */
class PTEngineWebGL2 implements Engine, PTEngineWebGL2Surface {
  readonly #slot: StateSlot;
  readonly #gl: WebGL2RenderingContext;
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: EngineCapabilities['causticStrategy'];
  readonly #traceTier: WebGl2TraceTier;
  readonly #supportsAuxBuffers: boolean;

  #scene: Scene | null = null;
  #geoPack: ScenePackResult | null = null;
  #samplesAccumulated = 0;
  #onFrameSubs = new Set<(s: FrameStats) => void>();
  #onProgressSubs = new Set<(p: ProgressStats) => void>();

  constructor(opts: PTEngineWebGL2Options, slot: StateSlot, traceTier: WebGl2TraceTier) {
    this.#slot = slot;
    this.#gl = opts.device;
    this.#maxBouncesLimit = Math.max(1, opts.maxBounces ?? DEFAULT_MAX_BOUNCES);
    this.#maxSamplesLimit = Math.max(1, opts.maxSamplesPerPixel ?? DEFAULT_MAX_SPP);
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    this.#traceTier = traceTier;
    this.#supportsAuxBuffers = traceTier === 'full';
  }

  get state(): EngineState {
    return this.#slot.get();
  }

  get capabilities(): EngineCapabilities {
    return buildCapabilities(
      this.#causticStrategy,
      this.#maxBouncesLimit,
      this.#maxSamplesLimit,
      this.#supportsAuxBuffers,
    );
  }

  get _debugGeoPack(): ScenePackResult | null {
    return this.#geoPack;
  }

  setScene(scene: Scene): void {
    this.#guardLive('setScene');
    const { supported, warnings } = partitionSceneBySupport(scene, this.capabilities);
    for (const w of warnings) console.warn(`[vitrum/pt-webgl2] ${w}`);
    this.#geoPack = packSceneFromCore(supported, { tlas: false, resolveMaterialId: () => 0 });
    this.#scene = supported;
    // TODO(Slice 1+): upload BVH/material/light textures via uploadSceneTextures (WS3),
    //   then GlResources.ensureProgram + the accumulation draw (WS2/WS5).
    this.reset();
  }

  getScene(): Scene | null {
    return this.#scene;
  }

  renderFrame(_input: FrameInput): FrameOutput {
    this.#guardLive('renderFrame');
    // Slice 0: the GL pipeline is not wired yet — return a legal skip.
    return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
  }

  reset(): void {
    this.#samplesAccumulated = 0;
    // TODO(Slice 0+): GlResources.clearAccum() once the accumulation FBO exists.
  }

  pause(): void {
    this.#guardLive('pause');
    this.#slot.set('paused');
  }

  resume(): void {
    this.#guardLive('resume');
    this.#slot.set('ready');
  }

  dispose(): void {
    if (this.#slot.get() === 'disposed') return;
    // TODO(Slice 0+): GlResources.dispose() + scene-texture destroy() once they exist.
    this.#scene = null;
    this.#geoPack = null;
    this.#onFrameSubs.clear();
    this.#onProgressSubs.clear();
    this.#slot.set('disposed');
  }

  onFrame(cb: (s: FrameStats) => void): () => void {
    this.#onFrameSubs.add(cb);
    return () => this.#onFrameSubs.delete(cb);
  }

  onProgress(cb: (p: ProgressStats) => void): () => void {
    this.#onProgressSubs.add(cb);
    return () => this.#onProgressSubs.delete(cb);
  }

  #guardLive(method: string): void {
    if (this.#slot.get() === 'disposed') throw new Error(`${method}: engine is disposed`);
  }
}

/**
 * Minimal WebGL2 trace-tier gate (Slice 0). Full implementation in traceTier.ts (WS5):
 * gate on EXT_color_buffer_float / EXT_float_blend / MAX_DRAW_BUFFERS / texunits.
 */
function resolveWebGl2TraceTier(gl: WebGL2RenderingContext, force?: WebGl2TraceTier): WebGl2TraceTier {
  if (force) return force;
  const floatColor = gl.getExtension('EXT_color_buffer_float') != null;
  if (!floatColor) {
    throw new Error('pt-webgl2: EXT_color_buffer_float is required (RGBA32F render targets)');
  }
  const drawBuffers = (gl.getParameter(gl.MAX_DRAW_BUFFERS) as number) ?? 1;
  return drawBuffers >= 3 ? 'full' : 'lite';
}

export const createPTEngine_WebGL2: EngineFactory<
  PTEngineWebGL2Options,
  Engine & PTEngineWebGL2Surface
> = async (opts: PTEngineWebGL2Options): Promise<Engine & PTEngineWebGL2Surface> => {
  const gl = opts.device;
  if (gl == null || typeof gl.createFramebuffer !== 'function') {
    throw new TypeError('createPTEngine_WebGL2: device must be a WebGL2RenderingContext');
  }
  if (opts.maxBounces !== undefined && opts.maxBounces < 1) {
    throw new RangeError(`createPTEngine_WebGL2: maxBounces must be >= 1 (got ${opts.maxBounces})`);
  }
  if (opts.maxSamplesPerPixel !== undefined && opts.maxSamplesPerPixel < 1) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxSamplesPerPixel must be >= 1 (got ${opts.maxSamplesPerPixel})`,
    );
  }
  const traceTier = resolveWebGl2TraceTier(gl, opts.traceTier);
  const slot = makeStateSlot();
  const engine = new PTEngineWebGL2(opts, slot, traceTier);
  slot.set('ready');
  return engine;
};

export type { PTEngineWebGL2Options } from './options.js';
export { PT_WEBGL2_SUPPORT } from './capabilities.js';
export { packBvhTextureData, uploadBvhTextures } from './scene/bvhTextureAdapter.js';
