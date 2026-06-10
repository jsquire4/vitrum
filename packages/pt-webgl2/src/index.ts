import type {
  Engine,
  EngineCapabilities,
  EngineFactory,
  EngineState,
  FrameInput,
  FrameOutput,
  FrameRendered,
  FrameStats,
  ProgressStats,
  Scene,
  SceneEmitter,
  SceneEnvironment,
  ScenePrimitive,
} from '@vitrum/core';
import {
  asBackendTexture,
  patchEmitterInScene,
  patchPrimitiveInScene,
} from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { buildCapabilities } from './capabilities.js';
import { makeStateSlot, type StateSlot } from './state.js';
import type { PTEngineWebGL2Options } from './options.js';
import { resolveWebGl2TraceTier, type WebGl2TraceTier } from './traceTier.js';
import { GlResources } from './gl/glResources.js';
import { probeGlCaps } from './gl/glCaps.js';
import { buildSceneTextures } from './scene/uploadSceneTextures.js';
import type { UploadedSceneTextures } from './scene/sceneTextures.js';
import { invertMat4 } from './mat4.js';
import type { FrameUniforms } from './gl/glResources.js';
import { DEFAULT_TRACE_FEATURES, type AccumRegime, type TraceFeatures } from './featureTypes.js';

const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

const DEFAULT_MAX_SPP = 4096;
const DEFAULT_MAX_BOUNCES = 32;
const DEFAULT_SPP_TARGET = 16;

export interface PTEngineWebGL2Surface {
  /** @internal The retained single-root merged BVH pack (for tests/inspection). */
  readonly _debugGeoPack: WorldSpaceMergeResult | null;
  /** @internal The retained scene texture summary (for tests/inspection). */
  readonly _debugSceneTex: { envMap: boolean; envTotalSum: number; envWidth: number; envHeight: number; lightCount: number } | null;
}

/**
 * THREE-free WebGL2 path-tracing backend. The framework (GL resources, the packers,
 * the ported GLSL kernels) is fully wired here; pixel-level fidelity is validated
 * against the fork on a real-GPU WebGL2 capture host (plan 06 — the one external gate).
 */
class PTEngineWebGL2 implements Engine, PTEngineWebGL2Surface {
  readonly #slot: StateSlot;
  readonly #gl: WebGL2RenderingContext;
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: EngineCapabilities['causticStrategy'];
  readonly #spectralEnabled: boolean;
  readonly #bdpt: boolean;
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;
  readonly #backgroundAlpha: number;
  readonly #traceTier: WebGl2TraceTier;
  readonly #supportsAuxBuffers: boolean;
  readonly #regime: AccumRegime;
  readonly #gpu: GlResources;

  #scene: Scene | null = null;
  #geoPack: WorldSpaceMergeResult | null = null;
  #sceneTextures: UploadedSceneTextures | null = null;
  #samplesAccumulated = 0;
  #onFrameSubs = new Set<(s: FrameStats) => void>();
  #onProgressSubs = new Set<(p: ProgressStats) => void>();

  constructor(opts: PTEngineWebGL2Options, slot: StateSlot, traceTier: WebGl2TraceTier) {
    this.#slot = slot;
    this.#gl = opts.device;
    this.#maxBouncesLimit = Math.max(1, opts.maxBounces ?? DEFAULT_MAX_BOUNCES);
    this.#maxSamplesLimit = Math.max(1, opts.maxSamplesPerPixel ?? DEFAULT_MAX_SPP);
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    this.#spectralEnabled = opts.spectral ?? false;
    this.#bdpt = opts.bdpt ?? false;
    if (this.#bdpt) {
      // H5 (2026-06-09): the BDPT GLSL kernels compile, but the host does NOT yet
      // drive the light-subpath passes (uBdptLightSubpathPass/VertexCol/MaxLightBounces
      // are never set → the light subpath is never generated, so connections add
      // nothing). The frame renders unidirectionally. Surfaced honestly rather than
      // silently presenting unidirectional output as bidirectional. Full orchestration
      // is tracked in items_to_fix §H5. (The unbound-sampler crash is fixed separately.)
      console.warn(
        '[pt-webgl2] bdpt: true — BDPT light-subpath passes are not yet host-driven; ' +
          'rendering is currently unidirectional. See items_to_fix §H5.',
      );
    }
    this.#mneeMaxIterations = opts.causticOptions?.mneeMaxIterations ?? 8;
    this.#mneeMaxChainLength = opts.causticOptions?.mneeMaxChainLength ?? 3;
    this.#backgroundAlpha = Math.min(1, Math.max(0, opts.backgroundAlpha ?? 1));
    this.#traceTier = traceTier;
    this.#supportsAuxBuffers = traceTier === 'full';
    // Additive HDR accumulation needs EXT_float_blend; otherwise the alpha-composite
    // ping-pong regime is the unbiased fallback (plan 02 §3). A transparent
    // background (backgroundAlpha < 1) ALSO forces alpha-composite — the 'normal'
    // SRC_ALPHA running-average blend cannot composite partial background coverage
    // (mirrors the fork's `needsAlphaComposite = bgAlpha !== 1 || !floatBlend`).
    this.#regime =
      probeGlCaps(opts.device).floatBlend && this.#backgroundAlpha === 1
        ? 'normal'
        : 'alpha-composite';
    this.#gpu = new GlResources(opts.device, this.#supportsAuxBuffers);
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

  get _debugGeoPack(): WorldSpaceMergeResult | null {
    return this.#geoPack;
  }

  /** Debug-only: presence/summary of the uploaded scene textures (env + lights). */
  get _debugSceneTex(): { envMap: boolean; envTotalSum: number; envWidth: number; envHeight: number; lightCount: number } | null {
    const t = this.#sceneTextures;
    if (t == null) return null;
    return {
      envMap: t.envMap != null,
      envTotalSum: t.envTotalSum,
      envWidth: t.envWidth,
      envHeight: t.envHeight,
      lightCount: t.lightCount,
    };
  }

  setScene(scene: Scene): void {
    this.#guardLive('setScene');
    // H7 FIX (2026-06-09): partition ONCE. setScene used to call
    // partitionSceneBySupport here AND buildSceneTextures re-partitioned the
    // already-filtered scene internally (uploadSceneTextures.ts) — redundant work.
    // buildSceneTextures now returns `supported`, so the filter runs a single time.
    const built = buildSceneTextures(this.#gl, scene, this.capabilities);
    for (const w of built.warnings) console.warn(`[vitrum/pt-webgl2] ${w}`);
    this.#sceneTextures?.destroy();
    this.#sceneTextures = built.textures;
    this.#geoPack = built.merged;
    this.#scene = built.supported;
    this.reset();
  }

  getScene(): Scene | null {
    return this.#scene;
  }

  addPrimitive(primitive: ScenePrimitive): void {
    this.#guardLive('addPrimitive');
    if (this.#scene == null) {
      throw new Error('addPrimitive: call setScene() before addPrimitive()');
    }
    if (this.#scene.primitives.some((p) => String(p.id) === String(primitive.id))) {
      throw new Error(`addPrimitive: primitive "${primitive.id}" already exists in current scene`);
    }
    this.setScene({
      ...this.#scene,
      primitives: [...this.#scene.primitives, primitive],
    });
  }

  removePrimitive(id: ScenePrimitive['id']): void {
    this.#guardLive('removePrimitive');
    if (this.#scene == null) {
      throw new Error('removePrimitive: call setScene() before removePrimitive()');
    }
    let matched = false;
    const primitives = this.#scene.primitives.filter((p) => {
      const keep = String(p.id) !== String(id);
      if (!keep) matched = true;
      return keep;
    });
    if (!matched) {
      throw new Error(`removePrimitive: primitive "${id}" not found in current scene`);
    }
    this.setScene({
      ...this.#scene,
      primitives,
    });
  }

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    this.#guardLive('updatePrimitive');
    if (this.#scene == null) {
      throw new Error('updatePrimitive: call setScene() before updatePrimitive()');
    }
    this.setScene(patchPrimitiveInScene(this.#scene, id, patch));
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#guardLive('updateEmitter');
    if (this.#scene == null) {
      throw new Error('updateEmitter: call setScene() before updateEmitter()');
    }
    this.setScene(patchEmitterInScene(this.#scene, id, patch));
  }

  updateEnvironment(env: SceneEnvironment | null): void {
    this.#guardLive('updateEnvironment');
    if (this.#scene == null) {
      throw new Error('updateEnvironment: call setScene() before updateEnvironment()');
    }
    this.setScene({
      ...this.#scene,
      environment: env ?? { kind: 'none' },
    });
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#guardLive('renderFrame');
    if (this.#sceneTextures == null || this.#geoPack == null) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }

    const q = input.quality ?? {};
    const activeBounces = Math.max(1, Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit));
    const targetSpp = Math.min(q.samplesTarget ?? DEFAULT_SPP_TARGET, this.#maxSamplesLimit);
    const res = q.resolutionFactor ?? 1;
    const w = Math.max(1, Math.floor(input.viewport.width * res));
    const h = Math.max(1, Math.floor(input.viewport.height * res));

    // Paused → return the current accumulation without drawing.
    if (this.#slot.get() === 'paused') {
      const tex = this.#gpu.resultTexture();
      if (tex == null) return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      return this.#frameRendered(tex, this.#samplesAccumulated, this.#samplesAccumulated >= targetSpp, targetSpp);
    }

    if (this.#gpu.ensureAccumResources(w, h)) this.#samplesAccumulated = 0;
    this.#gpu.ensureProgram(this.#traceFeatures());

    // Converged → fast-out without drawing (this is how accumulation terminates).
    if (this.#samplesAccumulated >= targetSpp) {
      const tex = this.#gpu.resultTexture();
      if (tex == null) return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      return this.#frameRendered(tex, this.#samplesAccumulated, true, targetSpp);
    }

    const frameUniforms = this.#frameUniforms(input, activeBounces, w, h);
    // H7 FIX (2026-06-09): real per-frame time for the onFrame telemetry (was
    // hardcoded 0). This is the CPU-side cost of building + SUBMITTING the accum
    // step (uniform packing + the GL draw call); it is NOT GPU execution time
    // (that would need EXT_disjoint_timer_query) — but it is a real, non-zero
    // monotonic frame-cost signal instead of a constant 0. The no-draw paused/
    // converged fast-outs honestly report 0 (they enqueue no work).
    const t0 = performance.now();
    this.#gpu.drawAccumStep(this.#sceneTextures, this.#regime, input.frameSeed, frameUniforms);
    this.#samplesAccumulated = Math.min(this.#samplesAccumulated + 1, this.#maxSamplesLimit);

    const tex = this.#gpu.resultTexture();
    if (tex == null) return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    const frameTimeMs = performance.now() - t0;
    return this.#frameRendered(tex, this.#samplesAccumulated, this.#samplesAccumulated >= targetSpp, targetSpp, frameTimeMs);
  }

  reset(): void {
    this.#samplesAccumulated = 0;
    this.#gpu.clearAccum();
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
    this.#gpu.dispose();
    this.#sceneTextures?.destroy();
    this.#sceneTextures = null;
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

  // ── internals ──────────────────────────────────────────────────────────────

  #traceFeatures(): TraceFeatures {
    return { ...DEFAULT_TRACE_FEATURES, bdpt: this.#bdpt };
  }

  #frameUniforms(input: FrameInput, bounces: number, w: number, h: number): FrameUniforms {
    const cameraWorldMatrix = invertMat4(input.viewMatrix);
    const invProjectionMatrix = invertMat4(input.projMatrix);
    if (cameraWorldMatrix == null || invProjectionMatrix == null) {
      throw new Error('renderFrame: singular view/projection matrix');
    }
    const caustic =
      this.#causticStrategy === 'manifold-nee' ? 1 : this.#causticStrategy === 'photon-map' ? 2 : 0;
    // H6 FIX (2026-06-09): honour the HDRI environment's `intensity` contract field
    // (was hardcoded to 1, so `environment.intensity` was silently ignored).
    // Mirrors pt-webgpu (environmentPacking.ts:54: `env.intensity ?? 1`).
    const env = this.#scene?.environment;
    const envIntensity = env != null && env.kind === 'hdri' ? env.intensity ?? 1 : 1;
    return {
      resolution: [w, h],
      bounces,
      transmissiveBounces: bounces,
      filterGlossyFactor: input.quality?.filteredGlossyFactor ?? 0,
      radianceClamp: 0,
      cameraWorldMatrix,
      invProjectionMatrix,
      environmentIntensity: this.#sceneTextures?.envMap != null ? envIntensity : 0,
      // environment.rotationY is NOT yet honoured here — and it is ignored by EVERY
      // current backend (no rotationY→matrix convention is wired in pt-webgpu either,
      // verified 2026-06-09). That is a cross-backend contract gap (a convention must
      // be chosen + applied consistently), tracked separately, not a pt-webgl2 fix.
      environmentRotation: IDENTITY_MAT4,
      spectralEnabled: this.#spectralEnabled,
      causticStrategy: caustic,
      mneeMaxIterations: this.#mneeMaxIterations,
      mneeMaxChainLength: this.#mneeMaxChainLength,
      backgroundAlpha: this.#backgroundAlpha,
    };
  }

  #frameRendered(tex: WebGLTexture, samples: number, isConverged: boolean, target: number, frameTimeMs = 0): FrameRendered {
    const nd = this.#supportsAuxBuffers ? this.#gpu.normalDepthTex : null;
    const al = this.#supportsAuxBuffers ? this.#gpu.albedoTex : null;
    const out: FrameRendered = {
      kind: 'rendered',
      primaryRadiance: asBackendTexture<'pt-webgl2', WebGLTexture>(tex),
      ...(nd != null ? { normalDepth: asBackendTexture<'pt-webgl2', WebGLTexture>(nd) } : {}),
      ...(al != null ? { albedo: asBackendTexture<'pt-webgl2', WebGLTexture>(al) } : {}),
      samplesAccumulated: samples,
      isConverged,
    };
    const fraction = target > 0 ? Math.min(samples / target, 1) : 1;
    for (const cb of this.#onProgressSubs) cb({ kind: 'pt-spp', current: samples, target, fraction });
    for (const cb of this.#onFrameSubs) cb({ frameTimeMs, spp: samples });
    return out;
  }

  #guardLive(method: string): void {
    if (this.#slot.get() === 'disposed') throw new Error(`${method}: engine is disposed`);
  }
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
