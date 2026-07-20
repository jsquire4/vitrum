import type {
  CapturedFrame,
  CaptureFrameOptions,
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineError,
  EngineFactory,
  EngineState,
  EngineWarning,
  FrameInput,
  FrameOutput,
  FrameRendered,
  FrameStats,
  InverseSession,
  InverseSessionOptions,
  MaterialSpec,
  ProgressStats,
  Scene,
  SceneEmitter,
  SceneEnvironment,
  ScenePrimitive,
} from '@vitrum/core';
import {
  asBackendTexture,
  BACKEND_PROMISE_LEDGER,
  MATERIAL_SPEC_FIELDS,
  patchEmitterInScene,
  patchPrimitiveInScene,
} from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { pickPrimitiveCpu, type PickCamera } from '@vitrum/shared-bvh';
import { buildCapabilities } from './capabilities.js';
import { makeStateSlot, type StateSlot } from './state.js';
import type { PTEngineWebGL2Options } from './options.js';
import { resolveWebGl2TraceTier, type WebGl2TraceTier } from './traceTier.js';
import { GlResources } from './gl/glResources.js';
import { probeGlCaps } from './gl/glCaps.js';
import { buildSceneTextures } from './scene/uploadSceneTextures.js';
import type { UploadedSceneTextures } from './scene/sceneTextures.js';
import {
  fastPathEnvironmentMutation,
  materialTextureMapPatchFields,
  tryFastPathEmitterMutation,
  tryFastPathGeometryMutation,
  tryFastPathMaterialMutation,
  tryFastPathPrimitiveListMutation,
  type WebGl2MutationSwap,
} from './scene/mutateSceneTextures.js';
import {
  buildPrimitiveListFallbackWarning,
  buildPrimitiveMutationFallbackWarning,
} from './scene/mutationFallbackWarnings.js';
import { packFrameUniforms } from './gl/frameUniformsPacker.js';
import {
  resolveBdptMaxLightBounces,
  validateAndResolveWebgl2Options,
} from './options.validate.js';
import type { FrameUniforms } from './gl/glResources.js';
import { DEFAULT_TRACE_FEATURES, type AccumRegime, type TraceFeatures } from './featureTypes.js';
import {
  OIDNFinalDispatcher,
  type DenoisedFrame,
} from './denoise/oidnFinalDispatcher.js';
import { WebGl2FiniteDifferenceInverseSession } from './inverse/finiteDifferenceSession.js';

interface UnsupportedMaterialFieldUse {
  readonly primitiveId: string;
  readonly fields: readonly string[];
}

function collectFieldUnion(uses: readonly UnsupportedMaterialFieldUse[]): string[] {
  const fields = new Set<string>();
  for (const use of uses) {
    for (const field of use.fields) fields.add(field);
  }
  return Array.from(fields).sort();
}

function collectUnsupportedMaterialFieldUses(scene: Scene): UnsupportedMaterialFieldUse[] {
  const uses: UnsupportedMaterialFieldUse[] = [];
  for (const primitive of scene.primitives) {
    const material = (primitive as { readonly material?: Partial<MaterialSpec> }).material;
    if (material == null) continue;
    const fields = new Set<string>();
    for (const field of UNSUPPORTED_MATERIAL_FIELDS) {
      if (material[field] != null) fields.add(field);
    }
    if (fields.size > 0) {
      uses.push({ primitiveId: primitive.id, fields: Array.from(fields).sort() });
    }
  }
  return uses;
}

// CAP-01 — the remaining material fields this backend silently drops, derived
// from the ledger's per-field support matrix so warning + capability rows can
// never drift. `extensions` is the contract-sanctioned host-discretionary escape
// hatch (no warning).
const UNSUPPORTED_MATERIAL_FIELDS: readonly (keyof MaterialSpec)[] = MATERIAL_SPEC_FIELDS.filter(
  (field) =>
    BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.materials[field] === 'unsupported' &&
    field !== 'extensions',
);

const DEFAULT_MAX_SPP = 4096;
const DEFAULT_MAX_BOUNCES = 32;
const DEFAULT_SPP_TARGET = 16;

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export interface PTEngineWebGL2Surface {
  /** @internal The retained single-root merged BVH pack (for tests/inspection). */
  readonly _debugGeoPack: WorldSpaceMergeResult | null;
  /** @internal The retained scene texture summary (for tests/inspection). */
  readonly _debugSceneTex: {
    envMap: boolean;
    envTotalSum: number;
    envWidth: number;
    envHeight: number;
    lightCount: number;
    meshLightCount: number;
    totalEmissiveArea: number;
    totalEmissivePower: number;
  } | null;
  /** Latest completed OIDN final-pass CPU result for `denoiser: 'oidn-final'`. */
  getLatestDenoised(): DenoisedFrame | null;
}

/**
 * THREE-free WebGL2 path-tracing backend. The framework (GL resources, the packers,
 * the ported GLSL kernels) is fully wired here; pixel-level fidelity is validated
 * against the fork on a real-GPU WebGL2 capture host (plan 06 — the one external gate).
 */
class PTEngineWebGL2 implements Engine, PTEngineWebGL2Surface {
  // ── Device + lifecycle ────────────────────────────────────────────────────
  readonly #slot: StateSlot;
  readonly #gl: WebGL2RenderingContext;
  readonly #gpu: GlResources;

  // ── Rendering config ──────────────────────────────────────────────────────
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: EngineCapabilities['causticStrategy'];
  readonly #spectralEnabled: boolean;
  readonly #bdpt: boolean;
  readonly #randomType: TraceFeatures['randomType'];
  readonly #bdptMaxLightBounces: number;
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;
  readonly #materialLodDepth: number;
  readonly #backgroundAlpha: number;
  readonly #backgroundBlur: number;
  readonly #regime: AccumRegime;
  // eslint-disable-next-line no-unused-private-class-members -- reserved for lite-tier branching (road-to-100 B12)
  readonly #traceTier: WebGl2TraceTier;
  readonly #supportsAuxBuffers: boolean;
  readonly #postDenoiser: OIDNFinalDispatcher | null;

  // ── Camera + optics ───────────────────────────────────────────────────────
  readonly #cameraType: 0 | 1 | 2;
  readonly #dof: PTEngineWebGL2Options['dof'];

  // ── Context-loss state ─────────────────────────────────────────────────────
  // `#contextLost` is set to true on `webglcontextlost` (before the user-space
  // event fires) and cleared if the host disposes and recreates. Per the core
  // contract the HOST recreates the engine on device loss — this engine does NOT
  // auto-restore. The event listeners are registered against the canvas element
  // (which the WebGL2RenderingContext always exposes via `.canvas`).
  #contextLost = false;
  readonly #onContextLost: (e: Event) => void;
  readonly #onContextRestored: () => void;

  // ── Scene state ───────────────────────────────────────────────────────────
  #scene: Scene | null = null;
  #geoPack: WorldSpaceMergeResult | null = null;
  #sceneTextures: UploadedSceneTextures | null = null;
  #samplesAccumulated = 0;
  #requestedSize: { width: number; height: number } | null = null;
  #resolutionFactor = 1;
  /** Last-frame input retained for the debug click-to-pick surface (T3.G #30). */
  #lastFrameInput: FrameInput | null = null;
  #inInverseRender = false;

  // ── Subscriptions ─────────────────────────────────────────────────────────
  #onFrameSubs = new Set<(s: FrameStats) => void>();
  #onProgressSubs = new Set<(p: ProgressStats) => void>();
  #onErrorSubs = new Set<(e: EngineError) => void>();
  #onWarningSubs = new Set<(w: EngineWarning) => void>();
  #fallbackMutationWarnings = new Set<string>();

  constructor(opts: PTEngineWebGL2Options, slot: StateSlot, traceTier: WebGl2TraceTier) {
    this.#slot = slot;
    this.#gl = opts.device;
    if (opts.onWarning != null) {
      this.#onWarningSubs.add(opts.onWarning);
    }
    this.#maxBouncesLimit = Math.max(1, opts.maxBounces ?? DEFAULT_MAX_BOUNCES);
    this.#maxSamplesLimit = Math.max(1, opts.maxSamplesPerPixel ?? DEFAULT_MAX_SPP);
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    this.#spectralEnabled = opts.spectral ?? false;
    this.#bdpt = opts.bdpt ?? false;
    this.#randomType = opts.sampling === 'sobol' ? 1 : 0;
    this.#bdptMaxLightBounces = resolveBdptMaxLightBounces(opts.bdptOptions?.maxLightBounces);
    // A5 (2026-06-10): the BDPT light-subpath passes are now host-driven (GlResources
    // .#buildBdptLightSubpath builds the ping-pong light-path texture per sample and
    // binds it as uBdptLightPathTex; the eye pass connects to it). The old inert-warn
    // was removed. When bdpt:false the BDPT path is never touched (byte-identical
    // unidirectional render); see capabilities.experimentalFeatures (pt-webgl2-bdpt).
    this.#mneeMaxIterations = opts.causticOptions?.mneeMaxIterations ?? 8;
    this.#mneeMaxChainLength = opts.causticOptions?.mneeMaxChainLength ?? 3;
    this.#materialLodDepth = Math.max(0, Math.floor(opts.materialLodDepth ?? 0));
    this.#backgroundAlpha = Math.min(1, Math.max(0, opts.backgroundAlpha ?? 1));
    this.#backgroundBlur = Math.max(0, opts.backgroundBlur ?? 0);
    // Flag-plumbing audit (2026-06-10): cameraType + dof are now real options.
    this.#cameraType =
      opts.cameraType === 'orthographic' ? 1 : opts.cameraType === 'equirectangular' ? 2 : 0;
    // Item 22 — DOF × equirect guard: force DOF off for equirectangular (the factory
    // already warns; here we make the engine actually ignore it so FEATURE_DOF=0
    // even if the host passes dof). Orthographic + dof is left intact (coherent model).
    this.#dof = opts.cameraType === 'equirectangular' ? undefined : opts.dof;
    this.#traceTier = traceTier;
    this.#supportsAuxBuffers = traceTier === 'full';
    this.#regime = PTEngineWebGL2.#resolveRegime(opts.device, this.#backgroundAlpha);
    this.#gpu = new GlResources(opts.device, this.#supportsAuxBuffers);
    if (opts.denoiser === 'oidn-final') {
      const modelUrl = opts.oidn?.modelUrl;
      const eps = opts.oidn?.executionProviders?.filter(
        (p) => p === 'webnn' || p === 'webgpu' || p === 'wasm',
      );
      if (typeof modelUrl !== 'string' || modelUrl.length === 0) {
        throw new Error(
          "createPTEngine_WebGL2: denoiser: 'oidn-final' is not turnkey - it " +
            'requires TWO host-provided assets that vitrum does not ship: ' +
            '(1) oidn: { modelUrl } - a non-empty URL to an OIDN ONNX model ' +
            '(use oidn_rt_hdr_alb_nrm.onnx when supplying albedo + normal aux); ' +
            "and (2) the 'onnxruntime-web' optional peer dependency installed in " +
            'the host. Omit the `denoiser` option to render without a final denoise.',
        );
      }
      const dispatcherOpts =
        eps !== undefined && eps.length > 0
          ? { modelUrl, executionProviders: eps }
          : { modelUrl };
      this.#postDenoiser = new OIDNFinalDispatcher(
        dispatcherOpts,
        opts.oidnBridgeLoader,
        opts.oidnReadbackFn,
        {
          onError: (err) => {
            this.#emitError({
              kind: 'denoiser',
              message: `[vitrum/pt-webgl2] OIDN final denoiser failed: ${errorMessage(err)}`,
              fatal: false,
              raw: err,
            });
          },
        },
      );
    } else {
      this.#postDenoiser = null;
    }

    // ── Context-loss listeners ────────────────────────────────────────────────
    // Per the WebGL spec, calling `preventDefault()` on the `webglcontextlost`
    // event signals that the page wants to handle restore — required before
    // `webglcontextrestored` will ever fire. We set `#contextLost` to block all
    // rendering + resource-creation paths. Per the core contract the HOST disposes
    // and recreates the engine when the context is lost; this engine does NOT
    // attempt to auto-rebuild. The `webglcontextrestored` handler only warns the
    // host to do so (do not attempt auto-restore from inside the engine).
    this.#onContextLost = (e: Event): void => {
      e.preventDefault(); // Required so 'webglcontextrestored' can fire later.
      this.#contextLost = true;
      const msg = '[vitrum/pt-webgl2] WebGL context lost. Rendering is suspended. ' +
        'Per the core contract, the host should call engine.dispose() and ' +
        'create a fresh engine with the recovered context.';
      console.warn(msg);
      // Route through onError so the host can react programmatically (item 28).
      for (const cb of this.#onErrorSubs) {
        try { cb({ kind: 'context-lost', message: msg, fatal: true, raw: e }); } catch { /* subscriber errors must not stop context-loss notification — ignore */ }
      }
    };
    this.#onContextRestored = (): void => {
      // Do NOT attempt auto-restore (the GPU resource state is stale). Warn the
      // host to dispose + recreate as specified by the core Engine contract.
      this.#warn({
        code: 'pt-webgl2.context-restored-recreate-required',
        backend: 'pt-webgl2',
        phase: 'lifecycle',
        method: 'webglcontextrestored',
        message:
          '[vitrum/pt-webgl2] WebGL context restored. ' +
          'Call engine.dispose() and create a fresh engine with the recovered context — ' +
          'this engine cannot resume after a context loss.',
      });
    };
    // `gl.canvas` is the HTMLCanvasElement or OffscreenCanvas the context was
    // created from. addEventListener is present on both; the cast to EventTarget
    // handles the union.
    const canvas = opts.device.canvas as EventTarget;
    if (canvas != null && typeof canvas.addEventListener === 'function') {
      canvas.addEventListener('webglcontextlost', this.#onContextLost);
      canvas.addEventListener('webglcontextrestored', this.#onContextRestored);
    }
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
      { bdpt: this.#bdpt, spectral: this.#spectralEnabled, oidn: this.#postDenoiser != null },
    );
  }

  get _debugGeoPack(): WorldSpaceMergeResult | null {
    return this.#geoPack;
  }

  /** Debug-only: presence/summary of the uploaded scene textures (env + lights). */
  get _debugSceneTex(): {
    envMap: boolean;
    envTotalSum: number;
    envWidth: number;
    envHeight: number;
    lightCount: number;
    meshLightCount: number;
    totalEmissiveArea: number;
    totalEmissivePower: number;
  } | null {
    const t = this.#sceneTextures;
    if (t == null) return null;
    return {
      envMap: t.envMap != null,
      envTotalSum: t.envTotalSum,
      envWidth: t.envWidth,
      envHeight: t.envHeight,
      lightCount: t.lightCount,
      meshLightCount: t.meshLightCount,
      totalEmissiveArea: t.totalEmissiveArea,
      totalEmissivePower: t.totalEmissivePower,
    };
  }

  setScene(scene: Scene): void {
    this.#guardLive('setScene');
    // CAP-01 — warn on remaining unsupported material fields (matrix-driven).
    // Once per setScene.
    const unsupportedMaterialUses = collectUnsupportedMaterialFieldUses(scene);
    const unsupportedMaterialFields = collectFieldUnion(unsupportedMaterialUses);
    if (unsupportedMaterialFields.length > 0) {
      this.#warn({
        code: 'pt-webgl2.unsupported-material-fields',
        backend: 'pt-webgl2',
        phase: 'setScene',
        method: 'setScene',
        message:
          `[vitrum/pt-webgl2] setScene: material fields are supplied ` +
          `but not rendered by this backend: ${unsupportedMaterialFields.join(', ')}.`,
        details: {
          fields: unsupportedMaterialFields,
          primitiveIds: unsupportedMaterialUses.map((use) => use.primitiveId),
          primitiveFields: unsupportedMaterialUses,
        },
      });
    }
    // SHADOW-01 — receiveShadow is @reserved on all shipping backends (a
    // "receiver ignores occlusion" toggle is non-physical for a GI path
    // tracer). Structured signal when a scene sets it to false.
    const receiveShadowIds = scene.primitives
      .filter((p) => (p as { receiveShadow?: boolean }).receiveShadow === false)
      .map((p) => p.id);
    if (receiveShadowIds.length > 0) {
      this.#warn({
        code: 'pt-webgl2.reserved-receive-shadow',
        backend: 'pt-webgl2',
        phase: 'setScene',
        method: 'setScene',
        message:
          `[vitrum/pt-webgl2] setScene: receiveShadow:false is reserved and not ` +
          `consumed by any backend (non-physical for GI); primitives: ${receiveShadowIds.join(', ')}.`,
        details: { primitiveIds: receiveShadowIds },
      });
    }
    // H7 FIX (2026-06-09): partition ONCE. setScene used to call
    // partitionSceneBySupport here AND buildSceneTextures re-partitioned the
    // already-filtered scene internally (uploadSceneTextures.ts) — redundant work.
    // buildSceneTextures now returns `supported`, so the filter runs a single time.
    const built = buildSceneTextures(this.#gl, scene, this.capabilities);
    for (const w of built.warnings) {
      this.#warn({
        code: 'pt-webgl2.scene-upload-warning',
        backend: 'pt-webgl2',
        phase: 'setScene',
        method: 'setScene',
        message: `[vitrum/pt-webgl2] ${w}`,
        details: { warning: w },
      });
    }
    for (const warning of built.structuredWarnings) {
      this.#warn(warning);
    }
    this.#sceneTextures?.destroy();
    this.#sceneTextures = built.textures;
    this.#geoPack = built.merged;
    this.#scene = built.supported;
    this.reset();
  }

  getScene(): Scene | null {
    return this.#scene;
  }

  createInverseSession(opts: InverseSessionOptions): InverseSession {
    this.#guardLive('createInverseSession');
    if (this.#scene == null) {
      throw new Error('createInverseSession: call setScene() before opening an inverse session.');
    }
    if (this.#lastFrameInput == null) {
      throw new Error(
        'createInverseSession: call renderFrame() at least once before opening an ' +
          'inverse session - the session re-renders the most-recent camera view.',
      );
    }
    return new WebGl2FiniteDifferenceInverseSession({
      getScene: () => this.#scene!,
      renderAndReadback: (width, height, samples) =>
        this.#renderAndReadbackForInverse(width, height, samples),
      patchMaterial: (primitiveId, patch) => {
        this.updatePrimitive(primitiveId, { material: patch } as Partial<ScenePrimitive>);
      },
      patchEmitter: (emitterId, patch) => {
        this.updateEmitter(emitterId, patch);
      },
    }, opts);
  }

  // ── Debug introspection (T3.G #30) ────────────────────────────────────────
  // CPU click-to-pick using the retained scene + last-frame camera matrices.
  // Returns null before the first renderFrame (no camera) or on a miss.
  readonly debug: EngineDebugSurface = {
    pickPrimitive: (x: number, y: number): string | null => {
      const scene = this.#scene;
      const last = this.#lastFrameInput;
      if (scene == null || last == null) return null;
      const w = last.viewport.width;
      const h = last.viewport.height;
      const cam: PickCamera = {
        viewMatrix: last.viewMatrix,
        projMatrix: last.projMatrix,
        cameraPosition: last.cameraPosition,
      };
      return pickPrimitiveCpu(scene, cam, x, y, w, h);
    },
  };

  addPrimitive(primitive: ScenePrimitive): void {
    this.#guardLive('addPrimitive');
    if (this.#scene == null) {
      throw new Error('addPrimitive: call setScene() before addPrimitive()');
    }
    if (this.#scene.primitives.some((p) => String(p.id) === String(primitive.id))) {
      throw new Error(`addPrimitive: primitive "${primitive.id}" already exists in current scene`);
    }
    const nextScene = {
      ...this.#scene,
      primitives: [...this.#scene.primitives, primitive],
    };
    const fast = tryFastPathPrimitiveListMutation(this.#gl, this.#sceneTextures, this.#geoPack, nextScene, {
      method: 'addPrimitive',
      primitiveId: String(primitive.id),
    });
    if (fast != null) {
      if (fast.mutationFallback != null) {
        this.#warnPrimitiveListFallback(
          'addPrimitive',
          String(primitive.id),
          'primitive-list-texture-refresh',
          fast.mutationFallback.textureRefreshMode,
        );
      }
      this.#commitMutationSwap(nextScene, fast);
      return;
    }
    this.#warnPrimitiveListFallback('addPrimitive', String(primitive.id), 'primitive-list-scene-repack');
    this.setScene(nextScene);
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
    const nextScene = {
      ...this.#scene,
      primitives,
    };
    const fast = tryFastPathPrimitiveListMutation(this.#gl, this.#sceneTextures, this.#geoPack, nextScene, {
      method: 'removePrimitive',
      primitiveId: String(id),
    });
    if (fast != null) {
      if (fast.mutationFallback != null) {
        this.#warnPrimitiveListFallback(
          'removePrimitive',
          String(id),
          'primitive-list-texture-refresh',
          fast.mutationFallback.textureRefreshMode,
        );
      }
      this.#commitMutationSwap(nextScene, fast);
      return;
    }
    this.#warnPrimitiveListFallback('removePrimitive', String(id), 'primitive-list-scene-repack');
    this.setScene(nextScene);
  }

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    this.#guardLive('updatePrimitive');
    if (this.#scene == null) {
      throw new Error('updatePrimitive: call setScene() before updatePrimitive()');
    }
    const nextScene = patchPrimitiveInScene(this.#scene, id, patch);
    if ((patch as { receiveShadow?: boolean }).receiveShadow === false) {
      this.#warn({
        code: 'pt-webgl2.reserved-receive-shadow',
        backend: 'pt-webgl2',
        phase: 'mutation',
        method: 'updatePrimitive',
        message:
          `[vitrum/pt-webgl2] updatePrimitive("${id}"): receiveShadow:false is reserved ` +
          'and not consumed by any backend (non-physical for GI).',
        details: { primitiveIds: [id] },
      });
    }
    const fast = tryFastPathMaterialMutation(
      this.#gl,
      this.#sceneTextures,
      this.#geoPack,
      nextScene,
      id,
      patch,
    );
    if (fast != null) {
      this.#commitMutationSwap(nextScene, fast);
      return;
    }
    const geometryFast = tryFastPathGeometryMutation(
      this.#gl,
      this.#sceneTextures,
      this.#geoPack,
      nextScene,
      patch,
    );
    if (geometryFast != null) {
      if (geometryFast.mutationFallback != null) {
        this.#warnPrimitiveMutationFallback(id, patch, geometryFast.mutationFallback);
      }
      this.#commitMutationSwap(nextScene, geometryFast);
      return;
    }
    this.#warnPrimitiveMutationFallback(id, patch);
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#guardLive('updateEmitter');
    if (this.#scene == null) {
      throw new Error('updateEmitter: call setScene() before updateEmitter()');
    }
    const nextScene = patchEmitterInScene(this.#scene, id, patch);
    const fast = tryFastPathEmitterMutation(this.#gl, this.#sceneTextures, this.#geoPack, nextScene, id);
    if (fast != null) {
      this.#commitMutationSwap(nextScene, fast);
      return;
    }
    this.setScene(nextScene);
  }

  updateEnvironment(env: SceneEnvironment | null): void {
    this.#guardLive('updateEnvironment');
    if (this.#scene == null) {
      throw new Error('updateEnvironment: call setScene() before updateEnvironment()');
    }
    const nextScene: Scene = {
      ...this.#scene,
      environment: env ?? { kind: 'none' },
    };
    const fast = fastPathEnvironmentMutation(this.#gl, this.#sceneTextures, nextScene);
    if (fast != null) {
      this.#commitMutationSwap(nextScene, fast);
      return;
    }
    this.setScene(nextScene);
  }

  setSize(width: number, height: number): void {
    this.#guardLive('setSize');
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    const next = {
      width: Math.max(1, Math.floor(width)),
      height: Math.max(1, Math.floor(height)),
    };
    if (this.#requestedSize?.width === next.width && this.#requestedSize.height === next.height) return;
    this.#requestedSize = next;
    if (this.#gpu.accumDims.width > 0 && this.#gpu.accumDims.height > 0) {
      this.#gpu.ensureAccumResources(
        Math.max(1, Math.floor(next.width * this.#resolutionFactor)),
        Math.max(1, Math.floor(next.height * this.#resolutionFactor)),
      );
    }
    this.reset();
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#guardLive('renderFrame');
    // Retain camera for the debug click-to-pick surface (T3.G #30).
    this.#lastFrameInput = input;
    // Context-loss guard: the GL device is no longer usable. Return a safe skipped
    // output (allowed by the contract — the host owns the device-lost recovery path).
    if (this.#contextLost) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }
    if (this.#sceneTextures == null || this.#geoPack == null) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }

    const q = input.quality ?? {};
    const activeBounces = Math.max(1, Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit));
    const targetSpp = Math.min(q.samplesTarget ?? DEFAULT_SPP_TARGET, this.#maxSamplesLimit);
    const requestedResolutionFactor = q.resolutionFactor ?? 1;
    const res = Number.isFinite(requestedResolutionFactor) && requestedResolutionFactor > 0
      ? requestedResolutionFactor
      : 1;
    this.#resolutionFactor = res;
    const baseWidth = this.#requestedSize?.width ?? input.viewport.width;
    const baseHeight = this.#requestedSize?.height ?? input.viewport.height;
    const w = Math.max(1, Math.floor(baseWidth * res));
    const h = Math.max(1, Math.floor(baseHeight * res));

    // Paused → return the current accumulation without drawing.
    if (this.#slot.get() === 'paused' && !this.#inInverseRender) {
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
    const isConverged = this.#samplesAccumulated >= targetSpp;
    if (this.#postDenoiser != null && isConverged && this.#samplesAccumulated > 0) {
      const readback = this.#gpu.readOidnInputsRgba32f();
      if (readback != null) this.#postDenoiser.kickIfReady(readback);
    }
    const frameTimeMs = performance.now() - t0;
    return this.#frameRendered(tex, this.#samplesAccumulated, isConverged, targetSpp, frameTimeMs);
  }

  /**
   * Capture the engine's rendered output as a host-side CPU Float32 RGBA image,
   * row-major, top-left origin (GL's bottom-left is flipped here).
   *
   * `colorSpace:'linear'` (default) reads the RGBA32F accumulation FBO — the
   * running mean of all accumulated path-trace samples (linear-light HDR,
   * scene radiance units). Requires EXT_color_buffer_float (enforced at engine
   * creation; always present when the engine is alive).
   *
   * `colorSpace:'output'` reads the RGBA32F present FBO — the tonemapped output
   * written by the present pass (tonemap + optional OETF). The present FBO is
   * DELIBERATELY RGBA32F (not RGBA8): the present texture is also the public
   * `primaryRadiance`, which hosts and the GPU validation harnesses read with
   * FLOAT readPixels — a UNORM8 target makes that read fail silently
   * (all-zeros). See createPresentTexture.
   *
   * Returns `null` before the first frame (FBO not yet allocated).
   * Synchronous (WebGL readPixels is always synchronous — no async stall). Wraps
   * in a resolved Promise to match the cross-backend contract.
   */
  // eslint-disable-next-line @typescript-eslint/require-await -- WebGL readPixels is synchronous; async wraps the result to match the cross-backend contract
  async captureFrame(opts?: CaptureFrameOptions): Promise<CapturedFrame | null> {
    const colorSpace = opts?.colorSpace ?? 'linear';
    const rgba = this.#gpu.readPixelsRgba32f(colorSpace === 'output' ? 'output' : 'linear');
    if (rgba == null) return null;
    return {
      width: this.#gpu.accumDims.width,
      height: this.#gpu.accumDims.height,
      rgba,
    };
  }

  async #renderAndReadbackForInverse(
    width: number,
    height: number,
    samples: number,
  ): Promise<{ rgba: Float32Array; channels: 4 }> {
    const last = this.#lastFrameInput!;
    const frozenSeedBase = 0x5eed5eed;
    const previousRequestedSize = this.#requestedSize;
    const previousResolutionFactor = this.#resolutionFactor;
    this.#inInverseRender = true;
    this.#requestedSize = { width, height };
    this.#resolutionFactor = 1;
    try {
      this.reset();
      for (let s = 0; s < samples; s += 1) {
        this.renderFrame({
          ...last,
          frameIndex: 0,
          frameSeed: (frozenSeedBase + s) >>> 0,
          viewport: { width, height, devicePixelRatio: 1 },
          quality: {
            ...(last.quality ?? {}),
            samplesTarget: samples,
            resolutionFactor: 1,
          },
        });
      }
      const frame = await this.captureFrame({ colorSpace: 'linear' });
      return {
        rgba: frame?.rgba ?? new Float32Array(width * height * 4),
        channels: 4,
      };
    } finally {
      this.#inInverseRender = false;
      this.#requestedSize = previousRequestedSize;
      this.#resolutionFactor = previousResolutionFactor;
      this.reset();
    }
  }

  reset(): void {
    this.#samplesAccumulated = 0;
    this.#gpu.clearAccum();
    this.#postDenoiser?.invalidate();
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
    // Remove context-loss listeners before tearing down resources — the listeners
    // are no longer needed and must not fire after dispose().
    const canvas = this.#gl.canvas as EventTarget;
    if (canvas != null && typeof canvas.removeEventListener === 'function') {
      canvas.removeEventListener('webglcontextlost', this.#onContextLost);
      canvas.removeEventListener('webglcontextrestored', this.#onContextRestored);
    }
    this.#gpu.dispose();
    this.#postDenoiser?.dispose();
    this.#sceneTextures?.destroy();
    this.#sceneTextures = null;
    this.#scene = null;
    this.#geoPack = null;
    this.#lastFrameInput = null;
    this.#onFrameSubs.clear();
    this.#onProgressSubs.clear();
    this.#onErrorSubs.clear();
    this.#onWarningSubs.clear();
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

  onError(cb: (error: EngineError) => void): () => void {
    this.#onErrorSubs.add(cb);
    return () => this.#onErrorSubs.delete(cb);
  }

  onWarning(cb: (warning: EngineWarning) => void): () => void {
    this.#onWarningSubs.add(cb);
    return () => this.#onWarningSubs.delete(cb);
  }

  getLatestDenoised(): DenoisedFrame | null {
    return this.#postDenoiser?.getLatestDenoised() ?? null;
  }

  #emitError(error: EngineError): void {
    for (const cb of this.#onErrorSubs) {
      try {
        cb(error);
      } catch {
        // Error subscribers must not break engine lifecycle.
      }
    }
  }

  #emitWarning(warning: EngineWarning): void {
    for (const cb of this.#onWarningSubs) {
      try {
        cb(warning);
      } catch {
        // Warning callbacks must not break rendering.
      }
    }
  }

  #warn(warning: EngineWarning, ...consoleArgs: readonly unknown[]): void {
    console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
    this.#emitWarning(warning);
  }

  // ── internals ──────────────────────────────────────────────────────────────

  #warnPrimitiveListFallback(
    method: 'addPrimitive' | 'removePrimitive',
    primitiveId: string,
    fallbackReason: 'primitive-list-texture-refresh' | 'primitive-list-scene-repack',
    textureRefreshMode?: string,
  ): void {
    const key = `${method}:${primitiveId}:${fallbackReason}`;
    if (this.#fallbackMutationWarnings.has(key)) return;
    this.#fallbackMutationWarnings.add(key);
    this.#warn(
      buildPrimitiveListFallbackWarning(method, primitiveId, fallbackReason, textureRefreshMode),
    );
  }

  #warnPrimitiveMutationFallback(
    id: string,
    patch: Partial<ScenePrimitive>,
    mutationFallback?: {
      readonly fallbackReason: string;
      readonly nativePatchMissing: string;
      readonly textureRefreshMode?: string;
    },
  ): void {
    const { key, warning } = buildPrimitiveMutationFallbackWarning(
      id,
      patch,
      materialTextureMapPatchFields(patch),
      mutationFallback,
    );
    if (this.#fallbackMutationWarnings.has(key)) return;
    this.#fallbackMutationWarnings.add(key);
    this.#warn(warning);
  }

  #commitMutationSwap(scene: Scene, swap: WebGl2MutationSwap): void {
    for (const warning of swap.structuredWarnings ?? []) {
      this.#warn(warning);
    }
    this.#sceneTextures = swap.textures;
    if (swap.geoPack != null) this.#geoPack = swap.geoPack;
    this.#scene = swap.scene ?? scene;
    for (const tex of swap.deleteOldTextures) {
      if (tex != null) this.#gl.deleteTexture(tex);
    }
    this.reset();
  }

  #traceFeatures(): TraceFeatures {
    return {
      ...DEFAULT_TRACE_FEATURES,
      bdpt: this.#bdpt,
      cameraType: this.#cameraType,
      dof: this.#dof != null,
      randomType: this.#randomType,
    };
  }

  #frameUniforms(input: FrameInput, bounces: number, w: number, h: number): FrameUniforms {
    return packFrameUniforms(input, bounces, w, h, {
      causticStrategy: this.#causticStrategy,
      scene: this.#scene,
      hasEnvMap: this.#sceneTextures?.envMap != null,
      materialLodDepth: this.#materialLodDepth,
      backgroundBlur: this.#backgroundBlur,
      spectralEnabled: this.#spectralEnabled,
      mneeMaxIterations: this.#mneeMaxIterations,
      mneeMaxChainLength: this.#mneeMaxChainLength,
      backgroundAlpha: this.#backgroundAlpha,
      bdpt: this.#bdpt,
      bdptMaxLightBounces: this.#bdptMaxLightBounces,
      dof: this.#dof,
    });
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
    const denoiserState = this.#postDenoiser != null
      ? this.#postDenoiser.getState()
      : { status: 'disabled' as const, reason: null };
    for (const cb of this.#onFrameSubs) cb({ frameTimeMs, spp: samples, denoiserState });
    return out;
  }

  /**
   * D10.13: Determine the accumulation regime from the device capabilities and
   * the configured background alpha.
   *
   * Running-average HDR accumulation (`'normal'`) requires `EXT_float_blend`; otherwise the
   * alpha-composite ping-pong regime is the unbiased fallback (plan 02 §3). A
   * transparent background (`backgroundAlpha < 1`) ALSO forces `'alpha-composite'` —
   * the `SRC_ALPHA` running-average blend cannot composite partial background coverage
   * (mirrors the fork's `needsAlphaComposite = bgAlpha !== 1 || !floatBlend`).
   */
  static #resolveRegime(device: WebGL2RenderingContext, backgroundAlpha: number): AccumRegime {
    return probeGlCaps(device).floatBlend && backgroundAlpha === 1
      ? 'normal'
      : 'alpha-composite';
  }

  #guardLive(method: string): void {
    if (this.#slot.get() === 'disposed') throw new Error(`${method}: engine is disposed`);
  }
}

export const createPTEngine_WebGL2: EngineFactory<
  PTEngineWebGL2Options,
  Engine & PTEngineWebGL2Surface
  // eslint-disable-next-line @typescript-eslint/require-await -- factory signature is async to match EngineFactory<…> contract; no async setup needed for WebGL2
> = async (opts: PTEngineWebGL2Options): Promise<Engine & PTEngineWebGL2Surface> => {
  const effectiveOpts = validateAndResolveWebgl2Options(opts);
  const traceTier = resolveWebGl2TraceTier(opts.device, opts.traceTier);
  const slot = makeStateSlot();
  const engine = new PTEngineWebGL2(effectiveOpts, slot, traceTier);
  slot.set('ready');
  return engine;
};

export type { PTEngineWebGL2Options } from './options.js';
export type {
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
} from './denoise/oidnFinalDispatcher.js';
export { PT_WEBGL2_SUPPORT } from './capabilities.js';
export { packBvhTextureData, uploadBvhTextures } from './scene/bvhTextureAdapter.js';
