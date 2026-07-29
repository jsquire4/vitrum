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
  MATERIAL_SPEC_FIELDS,
  deriveCameraPositionFromViewMatrix,
  patchEmitterInScene,
  patchPrimitiveInScene,
  validateScene as validateCoreScene,
} from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { pickPrimitiveCpu, type PickCamera } from '@vitrum/shared-bvh';
import { buildCapabilities } from './capabilities.js';
import { PT_WEBGL2_MATERIAL_SUPPORT } from './supportManifest.js';
import { makeStateSlot, type StateSlot } from './state.js';
import type { PTEngineWebGL2Options } from './options.js';
import { resolveWebGl2TraceTier, type WebGl2TraceTier } from './traceTier.js';
import { GlResources } from './gl/glResources.js';
import { retireTexturesIndependently } from './gl/resourceRetirement.js';
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
import { resolveBdptMaxLightBounces, validateAndResolveWebgl2Options } from './options.validate.js';
export { validateWebgl2AdvancedOptions } from './options.validate.js';
import type { FrameUniforms } from './gl/glResources.js';
import { DEFAULT_TRACE_FEATURES, type TraceFeatures } from './featureTypes.js';
import { OIDNFinalDispatcher, type DenoisedFrame } from './denoise/oidnFinalDispatcher.js';
import { WebGl2FiniteDifferenceInverseSession } from './inverse/finiteDifferenceSession.js';
import { resolveWebGl2FrameQuality, withResolvedWebGl2FrameQuality } from './frameQuality.js';
import { validateWebGl2FrameInput, validateWebGl2PixelSize } from './frameValidation.js';
import { WEBGL2_DEFAULT_BOUNCES } from './limits.js';
import { DEFAULT_RENDER_TARGET_BUDGET_BYTES } from './gl/renderTargetBudget.js';
import {
  deriveSceneTraceFeatures,
  validateWebGl2SceneMaterials,
} from './scene/sceneTraceFeatures.js';

interface UnsupportedMaterialFieldUse {
  readonly primitiveId: string;
  readonly fields: readonly string[];
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

function assertNoUnsupportedMaterialFields(scene: Scene, method: string): void {
  const uses = collectUnsupportedMaterialFieldUses(scene);
  if (uses.length === 0) return;
  const detail = uses
    .map((use) => `primitive ${JSON.stringify(use.primitiveId)}: ${use.fields.join(', ')}`)
    .join('; ');
  throw new RangeError(
    `[vitrum/pt-webgl2] ${method}: material field(s) unsupported by this backend (${detail}).`,
  );
}

// The renderer's rejection gate and live capability record consume the same
// backend-local executable material manifest. `extensions` is the
// contract-sanctioned host-discretionary escape hatch (no warning).
const UNSUPPORTED_MATERIAL_FIELDS: readonly (keyof MaterialSpec)[] = MATERIAL_SPEC_FIELDS.filter(
  (field) =>
    PT_WEBGL2_MATERIAL_SUPPORT[field] === 'unsupported' &&
    field !== 'extensions',
);

const DEFAULT_MAX_SPP = 4096;
const DEFAULT_SPP_TARGET = 16;

interface AccumulationSignature {
  readonly viewMatrix: readonly number[];
  readonly projMatrix: readonly number[];
  readonly bounces: number;
  readonly filteredGlossyFactor: number;
}

interface PresentationSignature {
  readonly tonemapMode: number;
  readonly exposure: number;
  readonly outputColorSpace: number;
}

function sameNumberArray(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

interface BdptSceneBounds {
  readonly center: readonly [number, number, number];
  readonly radius: number;
}

function computeBdptSceneBounds(pack: WorldSpaceMergeResult): BdptSceneBounds {
  const positions = pack.positions;
  const stride = pack.positionStrideFloats;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i + 2 < positions.length; i += stride) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    const z = positions[i + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  if (!Number.isFinite(minX)) return { center: [0, 0, 0], radius: 1 };
  const center: readonly [number, number, number] = [
    (minX + maxX) * 0.5,
    (minY + maxY) * 0.5,
    (minZ + maxZ) * 0.5,
  ];
  let radiusSquared = 0;
  for (let i = 0; i + 2 < positions.length; i += stride) {
    const dx = positions[i]! - center[0];
    const dy = positions[i + 1]! - center[1];
    const dz = positions[i + 2]! - center[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (Number.isFinite(d2)) radiusSquared = Math.max(radiusSquared, d2);
  }
  return { center, radius: Math.max(Math.sqrt(radiusSquared) * 1.001, 1e-3) };
}

function sameAccumulationSignature(
  a: AccumulationSignature | null,
  b: AccumulationSignature,
): boolean {
  return (
    a != null &&
    a.bounces === b.bounces &&
    a.filteredGlossyFactor === b.filteredGlossyFactor &&
    sameNumberArray(a.viewMatrix, b.viewMatrix) &&
    sameNumberArray(a.projMatrix, b.projMatrix)
  );
}

function samePresentationSignature(
  a: PresentationSignature | null,
  b: PresentationSignature,
): boolean {
  return (
    a != null &&
    a.tonemapMode === b.tonemapMode &&
    a.exposure === b.exposure &&
    a.outputColorSpace === b.outputColorSpace
  );
}

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
  readonly #spectralEnabled: boolean;
  readonly #bdpt: boolean;
  readonly #randomType: TraceFeatures['randomType'];
  readonly #bdptMaxLightBounces: number;
  readonly #materialLodDepth: number;
  readonly #backgroundAlpha: number;
  readonly #backgroundBlur: number;
  readonly #supportsAuxBuffers: boolean;
  readonly #postDenoiser: OIDNFinalDispatcher | null;
  #pendingDenoised: DenoisedFrame | null = null;

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
  #bdptSceneBounds: BdptSceneBounds = { center: [0, 0, 0], radius: 1 };
  #samplesAccumulated = 0;
  #resolutionFactor = 1;
  #accumulationSignature: AccumulationSignature | null = null;
  #presentationSignature: PresentationSignature | null = null;
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
    this.#maxBouncesLimit = opts.maxBounces ?? WEBGL2_DEFAULT_BOUNCES;
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SPP;
    this.#spectralEnabled = opts.spectral ?? false;
    this.#bdpt = opts.bdpt === true || opts.causticStrategy === 'bdpt';
    this.#randomType = opts.sampling === 'sobol' ? 1 : 0;
    this.#bdptMaxLightBounces = resolveBdptMaxLightBounces(opts.bdptOptions?.maxLightBounces);
    // A5 (2026-06-10): the BDPT light-subpath passes are now host-driven (GlResources
    // .#buildBdptLightSubpath builds the ping-pong light-path texture per sample and
    // binds it as uBdptLightPathTex; the eye pass connects to it). The old inert-warn
    // was removed. When bdpt:false the BDPT path is never touched (byte-identical
    // unidirectional render); see capabilities.activeFeatures (pt-webgl2-bdpt).
    this.#materialLodDepth = opts.materialLodDepth ?? 0;
    this.#backgroundAlpha = opts.backgroundAlpha ?? 1;
    this.#backgroundBlur = opts.backgroundBlur ?? 0;
    // Flag-plumbing audit (2026-06-10): cameraType + dof are now real options.
    this.#cameraType =
      opts.cameraType === 'orthographic' ? 1 : opts.cameraType === 'equirectangular' ? 2 : 0;
    // The factory rejects the incoherent equirectangular+DOF combination.
    // Orthographic + DOF remains a coherent tilt-shift-style projection.
    this.#dof = opts.dof;
    this.#supportsAuxBuffers = traceTier === 'full';
    this.#gpu = new GlResources(
      opts.device,
      this.#supportsAuxBuffers,
      opts.maxRenderTargetBytes ?? DEFAULT_RENDER_TARGET_BUDGET_BYTES,
    );
    if (opts.denoiser === 'oidn-final') {
      const modelUrl = opts.oidn?.modelUrl as string;
      const eps = opts.oidn?.executionProviders?.filter(
        (p) => p === 'webnn' || p === 'webgpu' || p === 'wasm',
      );
      const dispatcherOpts =
        eps !== undefined && eps.length > 0 ? { modelUrl, executionProviders: eps } : { modelUrl };
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
          onComplete: (frame) => {
            if (this.#contextLost || this.#slot.get() === 'disposed') return;
            // The WebGL context is host-owned. Never mutate shared GL state from
            // an inference promise callback; consume this at the next renderFrame.
            this.#pendingDenoised = frame;
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
      this.#slot.set('error');
      const msg =
        '[vitrum/pt-webgl2] WebGL context lost. Rendering is suspended. ' +
        'Per the core contract, the host should call engine.dispose() and ' +
        'create a fresh engine with the recovered context.';
      console.warn(msg);
      // Route through onError so the host can react programmatically (item 28).
      this.#emitError({ kind: 'context-lost', message: msg, fatal: true, raw: e });
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
      this.#postDenoiser == null ? 'none' : 'oidn-final',
      this.#maxBouncesLimit,
      this.#maxSamplesLimit,
      {
        bdpt: this.#bdpt,
        spectral: this.#spectralEnabled,
        sampling: this.#randomType === 1 ? 'sobol' : 'pcg',
      },
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
    validateCoreScene(scene);
    validateWebGl2SceneMaterials(scene, 'setScene');
    assertNoUnsupportedMaterialFields(scene, 'setScene');
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
    const previousTextures = this.#sceneTextures;
    const nextBdptSceneBounds = computeBdptSceneBounds(built.merged);
    this.#sceneTextures = built.textures;
    this.#geoPack = built.merged;
    this.#bdptSceneBounds = nextBdptSceneBounds;
    this.#scene = built.supported;
    if (previousTextures != null) {
      try {
        previousTextures.destroy();
      } catch (error) {
        this.#warn({
          code: 'pt-webgl2.texture-retirement-failed',
          backend: 'pt-webgl2',
          phase: 'lifecycle',
          method: 'setScene',
          message:
            '[vitrum/pt-webgl2] Replacement scene was published, but one or more old textures failed to retire.',
          details: { error: errorMessage(error) },
        });
      }
    }
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
    return new WebGl2FiniteDifferenceInverseSession(
      {
        getScene: () => this.#scene!,
        renderAndReadback: (width, height, samples) =>
          this.#renderAndReadbackForInverse(width, height, samples),
        patchMaterial: (primitiveId, patch) => {
          this.updatePrimitive(primitiveId, { material: patch } as Partial<ScenePrimitive>);
        },
        patchEmitter: (emitterId, patch) => {
          this.updateEmitter(emitterId, patch);
        },
      },
      opts,
    );
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
        cameraPosition: deriveCameraPositionFromViewMatrix(
          last.viewMatrix,
          'PTEngineWebGL2.debug.pickPrimitive.viewMatrix',
        ),
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
    validateCoreScene(nextScene);
    validateWebGl2SceneMaterials(nextScene, 'addPrimitive');
    assertNoUnsupportedMaterialFields(nextScene, 'addPrimitive');
    const fast = tryFastPathPrimitiveListMutation(
      this.#gl,
      this.#sceneTextures,
      this.#geoPack,
      nextScene,
      {
        method: 'addPrimitive',
        primitiveId: String(primitive.id),
      },
    );
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
    this.#warnPrimitiveListFallback(
      'addPrimitive',
      String(primitive.id),
      'primitive-list-scene-repack',
    );
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
    validateCoreScene(nextScene);
    const fast = tryFastPathPrimitiveListMutation(
      this.#gl,
      this.#sceneTextures,
      this.#geoPack,
      nextScene,
      {
        method: 'removePrimitive',
        primitiveId: String(id),
      },
    );
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
    validateWebGl2SceneMaterials(nextScene, 'updatePrimitive');
    assertNoUnsupportedMaterialFields(nextScene, 'updatePrimitive');
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
    const fast = tryFastPathEmitterMutation(
      this.#gl,
      this.#sceneTextures,
      this.#geoPack,
      nextScene,
      id,
    );
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
    validateCoreScene(nextScene);
    const fast = fastPathEnvironmentMutation(this.#gl, this.#sceneTextures, nextScene);
    if (fast != null) {
      this.#commitMutationSwap(nextScene, fast);
      return;
    }
    this.setScene(nextScene);
  }

  /**
   * Atomically replace either or both runtime lighting domains.
   *
   * Unlike a host-side sequence of `updateEmitter()` / `updateEnvironment()`,
   * this validates and uploads one complete candidate scene before publishing
   * any part of it. `setScene()` builds replacement resources before swapping
   * the retained scene, so a validation or upload failure preserves the prior
   * lighting and GL resources.
   */
  updateLighting(opts: Readonly<Record<string, unknown>>): void {
    this.#guardLive('updateLighting');
    if (opts == null || typeof opts !== 'object' || Array.isArray(opts)) {
      throw new TypeError('updateLighting: options must be an object');
    }

    const knownKeys = new Set<PropertyKey>(['emitters', 'environment']);
    for (const key of Reflect.ownKeys(opts)) {
      if (!knownKeys.has(key)) {
        throw new RangeError(
          `updateLighting: unknown option key ${typeof key === 'symbol' ? String(key) : JSON.stringify(key)}`,
        );
      }
    }
    const hasEmitters =
      Object.prototype.hasOwnProperty.call(opts, 'emitters') &&
      opts.emitters !== undefined;
    const hasEnvironment =
      Object.prototype.hasOwnProperty.call(opts, 'environment') &&
      opts.environment !== undefined;

    if (hasEmitters && !Array.isArray(opts.emitters)) {
      throw new TypeError('updateLighting: emitters must be an array when supplied');
    }
    if (
      hasEnvironment &&
      opts.environment !== null &&
      (typeof opts.environment !== 'object' || Array.isArray(opts.environment))
    ) {
      throw new TypeError(
        'updateLighting: environment must be a scene environment object or null',
      );
    }

    if (!hasEmitters && !hasEnvironment) {
      return;
    }
    if (this.#scene == null) {
      throw new Error('updateLighting: call setScene() before updateLighting()');
    }

    const nextScene: Scene = {
      ...this.#scene,
      ...(hasEmitters
        ? { emitters: [...(opts.emitters as readonly SceneEmitter[])] }
        : {}),
      ...(hasEnvironment
        ? {
            environment:
              (opts.environment as SceneEnvironment | null) ??
              ({ kind: 'none' } as const),
          }
        : {}),
    };

    validateCoreScene(nextScene);
    this.setScene(nextScene);
  }

  setSize(width: number, height: number): void {
    this.#guardLive('setSize');
    validateWebGl2PixelSize('setSize', width, height);
    const targetWidth = Math.max(1, Math.floor(width * this.#resolutionFactor));
    const targetHeight = Math.max(1, Math.floor(height * this.#resolutionFactor));
    if (
      this.#gpu.accumDims.width === targetWidth &&
      this.#gpu.accumDims.height === targetHeight
    ) return;
    if (this.#gpu.accumDims.width > 0 && this.#gpu.accumDims.height > 0) {
      this.#gpu.ensureAccumResources(targetWidth, targetHeight);
    }
    this.reset();
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#guardLive('renderFrame');
    validateWebGl2FrameInput(input);
    // Context-loss guard: the GL device is no longer usable. Return a safe skipped
    // output (allowed by the contract — the host owns the device-lost recovery path).
    if (this.#contextLost) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }
    if (this.#sceneTextures == null || this.#geoPack == null) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }

    const quality = resolveWebGl2FrameQuality(
      input.quality,
      this.#maxBouncesLimit,
      this.#maxSamplesLimit,
      DEFAULT_SPP_TARGET,
    );
    const effectiveInput: FrameInput = {
      ...input,
      quality: withResolvedWebGl2FrameQuality(quality),
    };
    const activeBounces = quality.bounces;
    const targetSpp = quality.samplesTarget;
    const res = quality.resolutionFactor;
    this.#resolutionFactor = res;
    const requestedBaseWidth = input.viewport.width;
    const requestedBaseHeight = input.viewport.height;
    const baseWidth =
      Number.isFinite(requestedBaseWidth) && requestedBaseWidth > 0 ? requestedBaseWidth : 1;
    const baseHeight =
      Number.isFinite(requestedBaseHeight) && requestedBaseHeight > 0 ? requestedBaseHeight : 1;
    const w = Math.max(1, Math.floor(baseWidth * res));
    const h = Math.max(1, Math.floor(baseHeight * res));
    this.#gpu.validateAccumRequest(w, h);

    // Cold ANGLE links for the production trace graph are polled through
    // KHR_parallel_shader_compile. Until every pass is ready, return without
    // allocating frame-sized targets or mutating accumulation state.
    if (!this.#gpu.ensureProgram(this.#traceFeatures())) {
      return {
        kind: 'skipped',
        samplesAccumulated: 0,
        isConverged: false,
      };
    }
    // Compile-pending calls must not replace the last presented/debug camera.
    this.#lastFrameInput = input;
    if (this.#gpu.ensureAccumResources(w, h)) {
      this.#samplesAccumulated = 0;
      this.#pendingDenoised = null;
      this.#postDenoiser?.invalidate();
    }

    const accumulationSignature: AccumulationSignature = {
      viewMatrix: Array.from(input.viewMatrix),
      projMatrix: Array.from(input.projMatrix),
      bounces: activeBounces,
      filteredGlossyFactor: quality.filteredGlossyFactor,
    };
    const accumulationChanged =
      this.#accumulationSignature != null &&
      !sameAccumulationSignature(this.#accumulationSignature, accumulationSignature);
    if (accumulationChanged && this.#samplesAccumulated > 0) {
      this.reset();
    }
    this.#accumulationSignature = accumulationSignature;

    // Advance the shared spectral+BDPT wavelength with the actual accumulated
    // sample even when a host repeats frameSeed. Computing this after any
    // invalidation makes reset + the same seed reproducible.
    const frameUniforms = this.#frameUniforms(
      effectiveInput,
      activeBounces,
      w,
      h,
      this.#samplesAccumulated,
    );
    const presentationSignature: PresentationSignature = {
      tonemapMode: frameUniforms.tonemapMode,
      exposure: frameUniforms.exposure,
      outputColorSpace: frameUniforms.outputColorSpace,
    };
    const presentationChanged =
      this.#presentationSignature != null &&
      !samePresentationSignature(this.#presentationSignature, presentationSignature);
    this.#presentationSignature = presentationSignature;
    if (
      this.#postDenoiser != null &&
      this.#samplesAccumulated > 0 &&
      this.#samplesAccumulated < targetSpp &&
      (this.#postDenoiser.isInFlight() || this.#postDenoiser.getLatestDenoised() != null)
    ) {
      this.#pendingDenoised = null;
      this.#postDenoiser.invalidate();
      this.#gpu.clearDenoisedResult();
    }

    this.#presentPendingDenoised(presentationSignature);

    // Paused → return the current accumulation without drawing.
    if (this.#slot.get() === 'paused' && !this.#inInverseRender) {
      if (this.#samplesAccumulated === 0) {
        return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      }
      if (presentationChanged) {
        this.#gpu.presentAccumulation(
          presentationSignature.tonemapMode,
          presentationSignature.exposure,
          presentationSignature.outputColorSpace,
        );
      }
      const tex = this.#gpu.resultTexture();
      if (tex == null) return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      return this.#frameRendered(
        tex,
        this.#samplesAccumulated,
        this.#samplesAccumulated >= targetSpp,
        targetSpp,
      );
    }

    // Converged → fast-out without drawing (this is how accumulation terminates).
    if (this.#samplesAccumulated >= targetSpp) {
      if (presentationChanged) {
        this.#gpu.presentAccumulation(
          presentationSignature.tonemapMode,
          presentationSignature.exposure,
          presentationSignature.outputColorSpace,
        );
      }
      const tex = this.#gpu.resultTexture();
      if (tex == null) return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      return this.#frameRendered(tex, this.#samplesAccumulated, true, targetSpp);
    }

    // H7 FIX (2026-06-09): real per-frame time for the onFrame telemetry (was
    // hardcoded 0). This is the CPU-side cost of building + SUBMITTING the accum
    // step (uniform packing + the GL draw call); it is NOT GPU execution time
    // (that would need EXT_disjoint_timer_query) — but it is a real, non-zero
    // monotonic frame-cost signal instead of a constant 0. The no-draw paused/
    // converged fast-outs honestly report 0 (they enqueue no work).
    const t0 = performance.now();
    this.#gpu.drawAccumStep(this.#sceneTextures, input.frameSeed, frameUniforms);
    this.#samplesAccumulated = Math.min(this.#samplesAccumulated + 1, this.#maxSamplesLimit);

    const tex = this.#gpu.resultTexture();
    if (tex == null) return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    const isConverged = this.#samplesAccumulated >= targetSpp;
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
   * (all-zeros). The presentation target therefore remains RGBA32F.
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
    const previousResolutionFactor = this.#resolutionFactor;
    this.#inInverseRender = true;
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
      this.#resolutionFactor = previousResolutionFactor;
      this.reset();
    }
  }

  reset(): void {
    this.#samplesAccumulated = 0;
    this.#pendingDenoised = null;
    this.#gpu.clearAccum();
    this.#postDenoiser?.invalidate();
  }

  pause(): void {
    this.#guardLive('pause');
    this.#slot.set('paused');
  }

  resume(): void {
    this.#guardLive('resume');
    if (this.#contextLost || this.#slot.get() === 'error') {
      throw new Error(
        'resume: engine cannot resume after fatal WebGL context loss; dispose and recreate it',
      );
    }
    this.#slot.set('ready');
  }

  dispose(): void {
    if (this.#slot.get() === 'disposed') return;
    const errors: unknown[] = [];
    const attempt = (cleanup: () => void): void => {
      try {
        cleanup();
      } catch (error) {
        errors.push(error);
      }
    };
    // Remove context-loss listeners before tearing down resources — the listeners
    // are no longer needed and must not fire after dispose().
    const canvas = this.#gl.canvas as EventTarget;
    if (canvas != null && typeof canvas.removeEventListener === 'function') {
      attempt(() => canvas.removeEventListener('webglcontextlost', this.#onContextLost));
      attempt(() => canvas.removeEventListener('webglcontextrestored', this.#onContextRestored));
    }
    attempt(() => this.#gpu.dispose());
    attempt(() => this.#postDenoiser?.dispose());
    attempt(() => this.#sceneTextures?.destroy());
    this.#sceneTextures = null;
    this.#scene = null;
    this.#geoPack = null;
    this.#lastFrameInput = null;
    this.#pendingDenoised = null;
    this.#onFrameSubs.clear();
    this.#onProgressSubs.clear();
    this.#onErrorSubs.clear();
    this.#onWarningSubs.clear();
    this.#slot.set('disposed');
    if (errors.length === 1) throw errors[0];
    if (errors.length > 1) {
      throw new AggregateError(
        errors,
        'pt-webgl2: one or more resource cleanups failed during dispose()',
      );
    }
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
    try {
      console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
    } catch {
      // A hostile/replaced console must never break publication or lifecycle.
    }
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
    if (swap.geoPack != null) {
      this.#geoPack = swap.geoPack;
      this.#bdptSceneBounds = computeBdptSceneBounds(swap.geoPack);
    }
    this.#scene = swap.scene ?? scene;
    try {
      retireTexturesIndependently(
        this.#gl,
        swap.deleteOldTextures,
        'pt-webgl2: one or more superseded mutation textures failed to retire',
      );
    } catch (error) {
      this.#warn({
        code: 'pt-webgl2.texture-retirement-failed',
        backend: 'pt-webgl2',
        phase: 'lifecycle',
        method: 'incremental-mutation',
        message:
          '[vitrum/pt-webgl2] Incremental scene mutation was published, but one or more old textures failed to retire.',
        details: { error: errorMessage(error) },
      });
    }
    this.reset();
  }

  #traceFeatures(): TraceFeatures {
    const sceneFeatures = deriveSceneTraceFeatures(this.#scene);
    return {
      ...DEFAULT_TRACE_FEATURES,
      bdpt: this.#bdpt,
      cameraType: this.#cameraType,
      dof: this.#dof != null,
      randomType: this.#randomType,
      ...sceneFeatures,
    };
  }

  #frameUniforms(
    input: FrameInput,
    bounces: number,
    w: number,
    h: number,
    accumulatedSample: number,
  ): FrameUniforms {
    return packFrameUniforms(
      input,
      bounces,
      w,
      h,
      {
        scene: this.#scene,
        hasEnvMap: this.#sceneTextures?.envMap != null,
        materialLodDepth: this.#materialLodDepth,
        backgroundBlur: this.#backgroundBlur,
        spectralEnabled: this.#spectralEnabled,
        backgroundAlpha: this.#backgroundAlpha,
        bdpt: this.#bdpt,
        bdptMaxLightBounces: this.#bdptMaxLightBounces,
        bdptSceneCenter: this.#bdptSceneBounds.center,
        bdptSceneRadius: this.#bdptSceneBounds.radius,
        dof: this.#dof,
      },
      accumulatedSample,
    );
  }

  #kickPostDenoiserIfReady(samples: number): void {
    const denoiser = this.#postDenoiser;
    if (
      denoiser == null ||
      samples <= 0 ||
      denoiser.isInFlight() ||
      denoiser.getLatestDenoised() != null
    )
      return;
    const readback = this.#gpu.readOidnInputsRgba32f();
    if (readback != null) denoiser.kickIfReady(readback);
  }

  #presentPendingDenoised(presentation: PresentationSignature): void {
    const pending = this.#pendingDenoised;
    if (pending == null) return;
    this.#pendingDenoised = null;
    try {
      this.#gpu.presentDenoisedResult(
        pending,
        presentation.tonemapMode,
        presentation.exposure,
        presentation.outputColorSpace,
      );
    } catch (err) {
      // Core publishes before invoking onComplete. Retire that accepted result
      // so the converged fast-out can start a clean retry on this same call.
      this.#gpu.clearDenoisedResult();
      this.#postDenoiser?.invalidate();
      this.#emitError({
        kind: 'denoiser',
        message: `[vitrum/pt-webgl2] Could not present OIDN result: ${errorMessage(err)}`,
        fatal: false,
        raw: err,
      });
    }
  }

  #frameRendered(
    tex: WebGLTexture,
    samples: number,
    isConverged: boolean,
    target: number,
    frameTimeMs = 0,
  ): FrameRendered {
    if (isConverged) this.#kickPostDenoiserIfReady(samples);
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
    for (const cb of this.#onProgressSubs) {
      try {
        cb({ kind: 'pt-spp', current: samples, target, fraction });
      } catch {
        // Telemetry subscribers are host code and must not break rendering.
      }
    }
    const denoiserState =
      this.#postDenoiser != null
        ? this.#postDenoiser.getState()
        : { status: 'disabled' as const, reason: null };
    for (const cb of this.#onFrameSubs) {
      try {
        cb({ frameTimeMs, spp: samples, denoiserState });
      } catch {
        // Telemetry subscribers are host code and must not break rendering.
      }
    }
    return out;
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
  const traceTier = resolveWebGl2TraceTier(effectiveOpts.device, effectiveOpts.traceTier);
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
export { WEBGL2_DEFAULT_BOUNCES, WEBGL2_MAX_BOUNCES } from './limits.js';
export {
  PT_WEBGL2_TEXTURE_SNAPSHOT_BUDGET_BYTES,
  PT_WEBGL2_TEXTURE_SOURCE_KIND,
  createPtWebgl2TextureSource,
  isPtWebgl2TextureSource,
  type PtWebgl2RawTextureSourceInput,
  type PtWebgl2TextureColorSpace,
  type PtWebgl2TextureCpuMirror,
  type PtWebgl2TextureDataType,
  type PtWebgl2TextureSource,
  type PtWebgl2TextureSourceOptions,
} from './materialTextureSource.js';
export {
  AUX_ALLOCATION_BYTES_PER_PIXEL,
  AUX_RENDER_TARGET_BYTES_PER_PIXEL,
  BASE_ALLOCATION_BYTES_PER_PIXEL,
  BASE_RENDER_TARGET_BYTES_PER_PIXEL,
  BLEND_RENDER_TARGET_BYTES_PER_PIXEL,
  DEFAULT_RENDER_TARGET_BUDGET_BYTES,
  DENOISED_RENDER_TARGET_BYTES_PER_PIXEL,
  estimateWebGl2AllocationBytes,
  estimateWebGl2DenoisedTargetBytes,
  estimateWebGl2ResidentBytes,
  estimateWebGl2RenderTargetBytes,
} from './gl/renderTargetBudget.js';
export { packBvhTextureData, uploadBvhTextures } from './scene/bvhTextureAdapter.js';
