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
  TEXTURE_MAP_FIELDS,
  tryFastPathEmitterMutation,
  tryFastPathGeometryMutation,
  tryFastPathMaterialMutation,
  tryFastPathPrimitiveListMutation,
  type WebGl2MutationSwap,
} from './scene/mutateSceneTextures.js';
import { invertMat4, makeRotationYMat4 } from './mat4.js';
import { CAUCHY_CROWN_GLASS, TONEMAP_MODE_INDEX } from '@vitrum/shared-samplers';
import type { FrameUniforms } from './gl/glResources.js';
import { DEFAULT_TRACE_FEATURES, type AccumRegime, type TraceFeatures } from './featureTypes.js';
import {
  OIDNFinalDispatcher,
  type DenoisedFrame,
} from './denoise/oidnFinalDispatcher.js';

const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function primitivePatchFields(patch: Partial<ScenePrimitive>): string[] {
  return Object.keys(patch)
    .filter((key) => key !== 'id' && key !== 'kind')
    .sort();
}

function materialPatchFields(patch: Partial<ScenePrimitive>): string[] {
  if (patch.material == null) return [];
  return Object.keys(patch.material as unknown as Record<string, unknown>).sort();
}

function materialTextureMapPatchFields(patch: Partial<ScenePrimitive>): string[] {
  return materialPatchFields(patch).filter((field) => TEXTURE_MAP_FIELDS.has(field));
}

const GEOMETRY_REBUILD_PATCH_FIELDS = new Set([
  'transform',
  'positions',
  'normals',
  'indices',
  'uvs',
  'uv1',
  'tangents',
  'colors',
  'instances',
  'shape',
  'params',
  'morphTargets',
  'morphTargetNormals',
  'morphTargetTangents',
  'morphWeights',
]);

const FULL_SCENE_REPACK_PATCH_FIELDS = new Set([
  'shape',
  'params',
]);

const ANIMATION_REBUILD_PATCH_FIELDS = new Set([
  'bones',
  'boneInverses',
  'skinIndices',
  'skinWeights',
  'morphTargets',
  'morphTargetNormals',
  'morphTargetTangents',
  'morphWeights',
]);

function primitiveFallbackReason(fields: readonly string[]): {
  readonly fallbackReason: string;
  readonly nativePatchMissing: string;
  readonly animationFields?: readonly string[];
  readonly fullUploadFields?: readonly string[];
} | null {
  const animationFields = fields.filter((field) => ANIMATION_REBUILD_PATCH_FIELDS.has(field));
  if (animationFields.length > 0) {
    return {
      fallbackReason: 'animation-geometry-rebuild',
      nativePatchMissing: 'targeted-skinned-or-morph-geometry-update',
      animationFields,
    };
  }
  if (fields.includes('colors')) {
    return {
      fallbackReason: 'geometry-material-texture-rebuild',
      nativePatchMissing: 'targeted-vertex-color-attribute-material-flag-update',
    };
  }
  if (fields.some((field) => GEOMETRY_REBUILD_PATCH_FIELDS.has(field))) {
    const fullUploadFields = fields.filter((field) => FULL_SCENE_REPACK_PATCH_FIELDS.has(field));
    if (fullUploadFields.length > 0) {
      return {
        fallbackReason: 'primitive-scene-texture-repack',
        nativePatchMissing: 'targeted-primitive-layout-or-analytic-update',
        fullUploadFields,
      };
    }
    return {
      fallbackReason: 'geometry-bvh-texture-rebuild',
      nativePatchMissing: 'targeted-geometry-bvh-refit',
    };
  }
  return null;
}

// A5 — light-subpath ping-pong width (one column per light bounce). MUST match the
// `BDPT_MAX_LIGHT_BOUNCES=3` layout the GLSL light-subpath/connection kernels assume
// (bdpt_light_subpath.glsl.js header; the connection sweep caps the merged path at
// BDPT_MAX_MERGED=19 with n = c + e + 3, BDPT_MAX_EYE_DEPTH=8).
const BDPT_SAFE_DEFAULT_LIGHT_BOUNCES = 1;
const BDPT_MAX_LIGHT_BOUNCES = 3;

// H2 follow-on — scene-global spectral coefficients (the GLSL declares u_jakobCoeffs +
// iorCauchyA/B/C as global uniforms, not per-material).
//   • iorCauchy: Crown Glass three-term Cauchy IOR (n(λ)). Uploaded only when
//     spectral is on so any material carrying `dispersionAbbeNumber` (→ per-material
//     dispersionStrength in the materials texture) actually disperses. All-zero =
//     the GLSL `cauchyEnabled` fast-path (no dispersion), which is the non-spectral
//     and the no-dispersion default → byte-identical when spectral:false.
//   • jakobCoeffs: stays the flat (0,0,0) ⇒ S≡½ no-op. u_jakobCoeffs is a SINGLE
//     global reflectance the GLSL uses only for representative MEDIUM albedo
//     (volume single-scatter / SSS); baseColor reflectance is per-material via
//     the materials texture, solved once at scene build.
const SPECTRAL_IOR_CAUCHY: readonly [number, number, number] = [
  CAUCHY_CROWN_GLASS.A,
  CAUCHY_CROWN_GLASS.B,
  CAUCHY_CROWN_GLASS.C,
];
const FLAT_JAKOB_COEFFS: readonly [number, number, number] = [0, 0, 0];
const NO_IOR_CAUCHY: readonly [number, number, number] = [0, 0, 0];

const UNSUPPORTED_DISPLACEMENT_MATERIAL_FIELDS = [
  'displacementMap',
  'displacementScale',
  'displacementBias',
] as const satisfies readonly (keyof MaterialSpec)[];

interface UnsupportedMaterialFieldUse {
  readonly primitiveId: string;
  readonly fields: readonly string[];
}

function collectPrimitiveMaterialFieldUses(
  scene: Scene,
  unsupportedFields: readonly (keyof MaterialSpec)[],
): UnsupportedMaterialFieldUse[] {
  const uses: UnsupportedMaterialFieldUse[] = [];
  const fields = new Set<string>();
  for (const primitive of scene.primitives) {
    const material = (primitive as { readonly material?: Partial<MaterialSpec> }).material;
    if (material == null) continue;
    fields.clear();
    for (const field of unsupportedFields) {
      if (material[field] != null) fields.add(field);
    }
    if (fields.size > 0) {
      uses.push({ primitiveId: primitive.id, fields: Array.from(fields).sort() });
    }
  }
  return uses;
}

function collectFieldUnion(uses: readonly UnsupportedMaterialFieldUse[]): string[] {
  const fields = new Set<string>();
  for (const use of uses) {
    for (const field of use.fields) fields.add(field);
  }
  return Array.from(fields).sort();
}

function collectUnsupportedDisplacementFieldUses(scene: Scene): UnsupportedMaterialFieldUse[] {
  return collectPrimitiveMaterialFieldUses(scene, UNSUPPORTED_DISPLACEMENT_MATERIAL_FIELDS);
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
// never drift. Displacement keeps its own dedicated warning above; `extensions`
// is the contract-sanctioned host-discretionary escape hatch (no warning).
const UNSUPPORTED_MATERIAL_FIELDS: readonly (keyof MaterialSpec)[] = MATERIAL_SPEC_FIELDS.filter(
  (field) =>
    BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.materials[field] === 'unsupported' &&
    !(UNSUPPORTED_DISPLACEMENT_MATERIAL_FIELDS as readonly string[]).includes(field) &&
    field !== 'extensions',
);

const DEFAULT_MAX_SPP = 4096;
const DEFAULT_MAX_BOUNCES = 32;
const DEFAULT_SPP_TARGET = 16;

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function emitWebgl2Warning(
  opts: Pick<PTEngineWebGL2Options, 'onWarning'>,
  warning: EngineWarning,
  ...consoleArgs: readonly unknown[]
): void {
  console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
  try {
    opts.onWarning?.(warning);
  } catch {
    // Host warning callbacks must not break engine construction.
  }
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
    const unsupportedDisplacementUses = collectUnsupportedDisplacementFieldUses(scene);
    const unsupportedDisplacementFields = collectFieldUnion(unsupportedDisplacementUses);
    if (unsupportedDisplacementFields.length > 0) {
      this.#warn({
        code: 'pt-webgl2.unsupported-displacement-material',
        backend: 'pt-webgl2',
        phase: 'setScene',
        method: 'setScene',
        message:
          `[vitrum/pt-webgl2] setScene: displacement material fields are supplied ` +
          `but not rendered by this backend: ${unsupportedDisplacementFields.join(', ')}.`,
        details: {
          fields: unsupportedDisplacementFields,
          primitiveIds: unsupportedDisplacementUses.map((use) => use.primitiveId),
          primitiveFields: unsupportedDisplacementUses,
        },
      });
    }
    // CAP-01 — warn on remaining unsupported material fields (matrix-driven).
    // Once per setScene, mirroring the displacement warning above.
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
    if (
      this.#bdpt &&
      built.textures.lightCount === 0 &&
      (built.textures.meshLightCount > 0 || built.textures.envMap != null || built.textures.envTotalSum > 0)
    ) {
      this.#warn({
        code: 'pt-webgl2.bdpt-analytic-light-subpaths-only',
        backend: 'pt-webgl2',
        phase: 'setScene',
        method: 'setScene',
        message:
          '[vitrum/pt-webgl2] bdpt:true currently builds light subpaths only from analytic ' +
          'Scene.emitters. This scene has mesh-area and/or environment lighting but no ' +
          'analytic light records, so BDPT connections fall back to the unidirectional ' +
          'mesh/environment NEE and BSDF paths for those sources.',
        details: {
          analyticLightCount: built.textures.lightCount,
          meshLightCount: built.textures.meshLightCount,
          hasEnvironmentMap: built.textures.envMap != null,
          envTotalSum: built.textures.envTotalSum,
        },
      });
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
    const fast = tryFastPathPrimitiveListMutation(this.#gl, this.#sceneTextures, nextScene, {
      method: 'addPrimitive',
    });
    if (fast != null) {
      this.#warnPrimitiveListFallback('addPrimitive', String(primitive.id), 'primitive-list-texture-refresh');
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
    const fast = tryFastPathPrimitiveListMutation(this.#gl, this.#sceneTextures, nextScene, {
      method: 'removePrimitive',
    });
    if (fast != null) {
      this.#warnPrimitiveListFallback('removePrimitive', String(id), 'primitive-list-texture-refresh');
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
      nextScene,
      patch,
    );
    if (geometryFast != null) {
      this.#warnPrimitiveMutationFallback(id, patch);
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
  ): void {
    const key = `${method}:${primitiveId}:${fallbackReason}`;
    if (this.#fallbackMutationWarnings.has(key)) return;
    this.#fallbackMutationWarnings.add(key);
    const refreshed = fallbackReason === 'primitive-list-texture-refresh';
    this.#warn({
      code: 'pt-webgl2.primitive-list-fallback-rebuild',
      backend: 'pt-webgl2',
      phase: 'mutation',
      method,
      message:
        `[vitrum/pt-webgl2] ${method}("${primitiveId}") rebuilds the backend ` +
        (refreshed ? 'geometry/material/atlas/BVH texture pack' : 'scene-texture/BVH pack') +
        '. This is supported, but it is not a targeted ' +
        'native geometry patch.',
      details: {
        primitiveId,
        operation: method,
        fallbackReason,
        nativePatchMissing: 'targeted-primitive-list-splice',
      },
    });
  }

  #warnPrimitiveMutationFallback(id: string, patch: Partial<ScenePrimitive>): void {
    const fields = primitivePatchFields(patch);
    const materialFields = materialPatchFields(patch);
    const materialTextureFields = materialTextureMapPatchFields(patch);
    const patchFallback = primitiveFallbackReason(fields);
    const signature = fields.length > 0 ? fields.join(',') : '<none>';
    const key = `updatePrimitive:${id}:${signature}`;
    if (this.#fallbackMutationWarnings.has(key)) return;
    this.#fallbackMutationWarnings.add(key);
    const details: Record<string, unknown> = { primitiveId: id, fields };
    if (materialFields.length > 0) details.materialFields = materialFields;
    if (materialTextureFields.length > 0) {
      details.materialTextureFields = materialTextureFields;
      details.fallbackReason = 'texture-map-material-patch';
      details.nativePatchMissing = 'targeted-material-atlas-texture-update';
    } else if (patchFallback != null) {
      details.fallbackReason = patchFallback.fallbackReason;
      details.nativePatchMissing = patchFallback.nativePatchMissing;
      if (patchFallback.animationFields !== undefined) {
        details.animationFields = patchFallback.animationFields;
      }
      if (patchFallback.fullUploadFields !== undefined) {
        details.fullUploadFields = patchFallback.fullUploadFields;
      }
    }
    this.#warn({
      code: 'pt-webgl2.primitive-mutation-fallback-rebuild',
      backend: 'pt-webgl2',
      phase: 'mutation',
      method: 'updatePrimitive',
      message:
        `[vitrum/pt-webgl2] updatePrimitive("${id}") fields [${signature}] ` +
        'are supported by rebuilding the backend scene-texture/BVH pack rather ' +
        'than by a targeted native patch.',
      details,
    });
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
    };
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
      materialLodDepth: this.#materialLodDepth,
      radianceClamp: 0,
      cameraWorldMatrix,
      invProjectionMatrix,
      environmentIntensity: this.#sceneTextures?.envMap != null ? envIntensity : 0,
      // H6 FIX (2026-06-09): honour HdriEnvironment.rotationY (CCW env dome rotation
      // around +Y, radians).  Convention: a world-space direction `d` looks up the
      // UNROTATED map at `RY(−rotationY) * d`, so the uniform matrix is
      // makeRotationYMat4(−rotationY).  The GLSL then evaluates:
      //   envRotation3x3 = mat3(environmentRotation)   → RY(−rotationY)
      //   lookupDir      = envRotation3x3 * worldDir   → RY(−rotationY) * d ✓
      // rotationY = 0 → identity → byte-identical to the pre-H6 IDENTITY_MAT4 path.
      environmentRotation: (env?.kind === 'hdri' && env.rotationY != null && env.rotationY !== 0)
        ? makeRotationYMat4(-(env.rotationY))
        : IDENTITY_MAT4,
      backgroundBlur: this.#backgroundBlur,
      spectralEnabled: this.#spectralEnabled,
      causticStrategy: caustic,
      mneeMaxIterations: this.#mneeMaxIterations,
      mneeMaxChainLength: this.#mneeMaxChainLength,
      backgroundAlpha: this.#backgroundAlpha,
      // A5 — BDPT host-driver inputs (no-op when bdpt:false).
      bdpt: this.#bdpt,
      bdptMaxLightBounces: this.#bdptMaxLightBounces,
      // H2 follow-on — global spectral coefficients. Cauchy IOR only when spectral is
      // on (else the no-dispersion fast path → byte-identical); Jakob stays flat.
      iorCauchy: this.#spectralEnabled ? SPECTRAL_IOR_CAUCHY : NO_IOR_CAUCHY,
      jakobCoeffs: FLAT_JAKOB_COEFFS,
      dof:
        this.#dof != null
          ? {
              focusDistance: this.#dof.focusDistance,
              bokehSize: this.#dof.bokehSize,
              apertureBlades: this.#dof.apertureBlades ?? 0,
              apertureRotation: this.#dof.apertureRotation ?? 0,
              anamorphicRatio: this.#dof.anamorphicRatio ?? 1,
            }
          : null,
      // ── Tonemap / present-pass dials (2026-06-10) ─────────────────────────
      // Matches the contract (FrameQualitySettings) and the walkaround-hybrid
      // orchestrator wiring (HybridEngineFrameOrchestrator.ts:764).
      // Default: aces(0) @ 1.0 @ srgb(0) — same as walkaround and the contract.
      //
      // CONTRACT-DEFAULT TENSION: pt-webgl2 previously returned raw linear HDR
      // (no present pass).  Adding the present pass with default aces+srgb
      // changes the default visual output.  Hosts that relied on the raw HDR
      // should pass quality.tonemap='none' + quality.outputColorSpace='linear'.
      tonemapMode:      TONEMAP_MODE_INDEX[input.quality?.tonemap ?? 'aces'],
      exposure:         input.quality?.exposure ?? 1.0,
      outputColorSpace: input.quality?.outputColorSpace === 'linear' ? 1 : 0,
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
  if (
    opts.materialLodDepth !== undefined &&
    (!Number.isFinite(opts.materialLodDepth) || opts.materialLodDepth < 0)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: materialLodDepth must be a finite number >= 0 (got ${opts.materialLodDepth})`,
    );
  }
  const bdptMaxLightBounces = opts.bdptOptions?.maxLightBounces;
  if (
    bdptMaxLightBounces !== undefined &&
    (!Number.isFinite(bdptMaxLightBounces) || bdptMaxLightBounces < 1)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: bdptOptions.maxLightBounces must be a finite number >= 1 (got ${bdptMaxLightBounces})`,
    );
  }
  if (bdptMaxLightBounces !== undefined && bdptMaxLightBounces > BDPT_MAX_LIGHT_BOUNCES) {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.bdpt-max-light-bounces-clamped',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `clamping to supported WebGL2 BDPT light-subpath limit ${BDPT_MAX_LIGHT_BOUNCES}.`,
      details: { requested: bdptMaxLightBounces, clampedTo: BDPT_MAX_LIGHT_BOUNCES },
    });
  }
  if (
    bdptMaxLightBounces !== undefined &&
    bdptMaxLightBounces <= BDPT_MAX_LIGHT_BOUNCES &&
    !Number.isInteger(bdptMaxLightBounces)
  ) {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.bdpt-max-light-bounces-rounded',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `rounding down to integer ${resolveBdptMaxLightBounces(bdptMaxLightBounces)}.`,
      details: {
        requested: bdptMaxLightBounces,
        roundedTo: resolveBdptMaxLightBounces(bdptMaxLightBounces),
      },
    });
  }
  const resolvedBdptMaxLightBounces = resolveBdptMaxLightBounces(bdptMaxLightBounces);
  if (opts.bdpt === true && resolvedBdptMaxLightBounces > BDPT_SAFE_DEFAULT_LIGHT_BOUNCES) {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.bdpt-multivertex-research-mode',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] bdptOptions.maxLightBounces=${resolvedBdptMaxLightBounces} enables the ` +
        'multi-vertex BDPT research path. Current radiometric promotion evidence is endpoint-only, ' +
        'so this WebGL2 path is explicitly opt-in; omit bdptOptions.maxLightBounces for the ' +
        'endpoint-only safe default.',
      details: {
        requested: bdptMaxLightBounces,
        resolved: resolvedBdptMaxLightBounces,
        safeDefault: BDPT_SAFE_DEFAULT_LIGHT_BOUNCES,
      },
    });
  }
  if (
    opts.backgroundBlur !== undefined &&
    (!Number.isFinite(opts.backgroundBlur) || opts.backgroundBlur < 0)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: backgroundBlur must be a finite number >= 0 (got ${opts.backgroundBlur})`,
    );
  }
  // Unsupported realtime denoisers still degrade to no-denoise with a clear
  // warning. `oidn-final` is handled by the engine constructor because it is a
  // real asynchronous final-pass path that requires host-provided model config.
  if (opts.denoiser != null && opts.denoiser !== 'none' && opts.denoiser !== 'oidn-final') {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.unsupported-denoiser',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] denoiser="${opts.denoiser}" requested, but pt-webgl2 only supports ` +
        '`none` and the asynchronous final-pass `oidn-final` denoiser. ' +
        'Degrading to no-denoise (denoiserState will report "disabled").',
      details: { requested: opts.denoiser },
    });
  }
  // Item 22 — DOF × equirectangular regime guard (trust-remediation-plan §22).
  // Thin-lens DOF applied to equirectangular projection is physically undefined:
  // the blur direction has no meaning per sphere region (the GLSL DOF block
  // translates ray.origin by an aperture sample in camera space, but the equirect
  // ray directions span the full sphere — there is no consistent focal plane).
  // Silently generating blurry/incorrect output is worse than ignoring the option,
  // so we force DOF off for equirect and warn once so the host is not surprised.
  //
  // Orthographic + DOF is left as-is: the GLSL produces tilt-shift-style focus
  // (focalPoint = fixed point on the -Z frustum; new ray.direction = focalPoint -
  // shifted_origin) which is physically coherent — parallel projections plus an
  // aperture offset is the standard orthographic-camera DOF model.
  if (opts.cameraType === 'equirectangular' && opts.dof != null) {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.equirectangular-dof-ignored',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        '[vitrum/pt-webgl2] dof is ignored when cameraType is "equirectangular". ' +
        'Thin-lens depth of field is physically undefined for full-sphere equirectangular ' +
        'projection (blur direction has no meaning per sphere region). ' +
        'The engine will render without DOF. Remove the dof option to suppress this warning.',
      details: { cameraType: opts.cameraType },
    });
  }
  const traceTier = resolveWebGl2TraceTier(gl, opts.traceTier);
  const slot = makeStateSlot();
  const engine = new PTEngineWebGL2(opts, slot, traceTier);
  slot.set('ready');
  return engine;
};

function resolveBdptMaxLightBounces(value: number | undefined): number {
  if (value === undefined) return BDPT_SAFE_DEFAULT_LIGHT_BOUNCES;
  return Math.max(1, Math.min(BDPT_MAX_LIGHT_BOUNCES, Math.floor(value)));
}

export type { PTEngineWebGL2Options } from './options.js';
export type {
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
} from './denoise/oidnFinalDispatcher.js';
export { PT_WEBGL2_SUPPORT } from './capabilities.js';
export { packBvhTextureData, uploadBvhTextures } from './scene/bvhTextureAdapter.js';
