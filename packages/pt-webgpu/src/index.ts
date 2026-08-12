import type {
  BackendTexture,
  CapturedFrame,
  CaptureFrameOptions,
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineError,
  EngineFactory,
  EngineOptions,
  EngineState,
  EngineWarning,
  FrameInput,
  FrameOutput,
  FrameStats,
  GpuMemoryBreakdown,
  InverseSession,
  InverseSessionOptions,
  MaterialSpec,
  ProgressStats,
  Scene,
  SceneEmitter,
  ScenePrimitive,
  ScenePrimitivePatch,
} from '@vitrum/core';
import {
  asBackendTexture,
  canonicalizeFrameCamera,
  narrowToBackendTexture,
  resolveFrameCameraPosition,
  validateSceneEmitters,
  validateSceneEnvironment,
  validateScene as validateCoreScene,
} from '@vitrum/core';
import {
  PtWebgpuInverseSession,
  type InverseEngineHooks,
  type AdjointGradientRequest,
} from './inverse/inverseSession.js';
import { materialIndexForPrimitive } from './scene/incrementalPatch.js';
import {
  readOidnInputsFromTextures,
  readRgba16fTextureToF32,
} from './denoise/rgba16fReadback.js';
import { summarizeScene, type SceneSummary } from './scene/flattenScene.js';
import { assertLiteSceneSupported } from './scene/liteSceneWarnings.js';
import { ptWebgpuCapabilities } from './capabilities.js';
import {
  validatePtWebgpuOptions,
  validatePtWebgpuFrameInput,
  validatePtWebgpuPixelSize,
  resolveBdptMaxLightBounces,
  assertPtWebgpuBdptFrameCameraSupported,
  PT_WEBGPU_DEFAULT_BOUNCES,
} from './ptWebgpuValidation.js';
export { validatePtWebgpuAdvancedOptions } from './ptWebgpuValidation.js';
import {
  assertOneEdgeReconnectionSceneSupported,
} from './oneEdgeReconnectionSceneValidation.js';
export {
  OneEdgeReconnectionDomainError,
  assertOneEdgeReconnectionSceneSupported,
} from './oneEdgeReconnectionSceneValidation.js';
import {
  buildPackedScene,
  scenePackResultFromPacked,
  uploadPackedScene,
  uploadedSceneGpuResources,
  PT_WEBGPU_ANALYTIC_SHAPES,
  type UploadedSceneBuffers,
} from './scene/uploadSceneBuffers.js';
import {
  assertSpectralSceneSupported,
  assertThinFilmSceneSupported,
  assertVolumeSceneSupported,
} from './spectralSceneValidation.js';
import { createMaterialInputSnapshotContext } from './scene/materialTextures.js';
import {
  packLiteLightTexture,
  packLiteEnvTexture,
  packLiteEnvCdfTexture,
} from './scene/litePackedTextures.js';
import {
  assertOpticalMediumTopology,
  type ScenePackResult,
  pickPrimitiveCpu,
  type PickCamera,
} from '@vitrum/shared-bvh';
import {
  FRAME_PARAMS_BUFFER_ALLOC_BYTES,
  packFrameParams,
} from './frameParamsPacker.js';
import { SceneMutationRouter } from './sceneMutationRouter.js';
import { type PtWebgpuTraceTier } from './traceTier.js';
import {
  collectUnsupportedMaterialFieldsForTraceTier,
} from './supportDetails.js';
import { ptWebgpuSupportManifest } from './supportManifest.js';
import { GpuResources, type LiteTextureReplacement, type PtWebgpuBvhTraversalMode } from './gpuResources.js';
import {
  OIDNFinalDispatcher,
  type DenoisedFrame,
} from './denoise/oidnFinalDispatcher.js';
import { AdjointPass } from './adjointPass.js';
import {
  PT_WEBGPU_COMMON_WGSL,
  type PtWebgpuSamplingMode,
} from './wgsl/common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  OCTAHEDRAL_CORE_WGSL,
  TONEMAP_MODE_INDEX,
} from '@vitrum/shared-samplers';
import {
  SPPM_PHOTON_COUNT,
  SPPM_PHOTON_CELLS_BYTES,
  SPPM_CELL_COUNTERS_BYTES,
  SPPM_STATS_BYTES,
  SPPM_PIXEL_STATS_BYTES_PER_PIXEL,
  sppmSceneBoundsFromCenterRadius,
  sppmSceneBoundsFromPackedPositions,
} from './sppmParams.js';

export {
  ThinFilmNumericError,
  type ThinFilmNumericFailureReason,
} from './math/thinFilm.js';

export { PT_WEBGPU_COMMON_WGSL, HAMMERSLEY_WGSL, OCTAHEDRAL_CORE_WGSL };
export type { PtWebgpuBvhTraversalMode } from './gpuResources.js';
export {
  createPtWebgpuTextureSource,
  isPtWebgpuTextureSource,
  PT_WEBGPU_TEXTURE_SOURCE_KIND,
  type PtWebgpuTextureColorSpace,
  type PtWebgpuTextureCpuMirror,
  type PtWebgpuTextureCpuMirrorDataType,
  type PtWebgpuTextureCpuMirrorInput,
  type PtWebgpuTextureSource,
  type PtWebgpuTextureSourceOptions,
} from './materialTextureSource.js';
export {
  PT_WEBGPU_REQUIRED_LIMITS,
  PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  // Full-tier device-limit floors — re-exported so `@vitrum/engine`'s
  // progressive-engine facade can compute the cross-backend limit UNION
  // (max of the hybrid + pt-webgpu full floors) for a shared device.
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_BDPT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  mergeAdapterRequiredLimits,
  ptWebgpuRequiredLimitsForAdapter,
} from './webgpuLimits.js';
export {
  resolvePtWebgpuTraceTier,
  selectPtWebgpuTraceTier,
  type PtWebgpuTraceTier,
} from './traceTier.js';
export { PT_WEBGPU_TRACE_LITE_WGSL } from './wgsl/pathTraceBruteforceLite.wgsl.js';
export { summarizeScene };
export type { SceneSummary };
// TLAS build is implemented in `@vitrum/shared-bvh`; pt-webgpu re-exports it
// under the `buildSceneTlas` name (the prior tlasBridge.ts pass-through wrapper
// was removed once its only other export, `refitSceneTlas`, was found dead).
export { buildTlas as buildSceneTlas, type TlasInstance, type TlasData } from '@vitrum/shared-bvh';

/** Runtime lighting vocabulary accepted by {@link Engine.updateLighting} on pt-webgpu. */
export interface PTEngineWebGPULightingOptions {
  /** Atomically replace the complete scene-emitter array. */
  readonly emitters?: readonly SceneEmitter[];
  /** Atomically replace the environment; `null` is normalized to `{kind:'none'}`. */
  readonly environment?: Scene['environment'] | null;
}

interface UnsupportedMaterialFieldUse {
  readonly primitiveId: string;
  readonly fields: readonly string[];
}

function displacementWarningDetails(warning: string): Readonly<Record<string, unknown>> {
  const details: Record<string, unknown> = { warning };
  const match = / displacementMap at (.+?)(?: handle | requests | has | displacementSubdivisions| triangle )/.exec(warning);
  if (match?.[1] !== undefined) details.sourcePath = match[1];
  return details;
}

function collectFieldUnion(uses: readonly UnsupportedMaterialFieldUse[]): string[] {
  const fields = new Set<string>();
  for (const use of uses) {
    for (const field of use.fields) fields.add(field);
  }
  return Array.from(fields).sort();
}

function collectUnsupportedMaterialFieldUses(
  scene: Scene,
  traceTier: PtWebgpuTraceTier,
): UnsupportedMaterialFieldUse[] {
  const uses: UnsupportedMaterialFieldUse[] = [];
  for (const primitive of scene.primitives) {
    const material = (primitive as { readonly material?: Partial<MaterialSpec> }).material;
    if (material == null) continue;
    const fields = collectUnsupportedMaterialFieldsForTraceTier(material, traceTier);
    if (fields.length > 0) {
      uses.push({ primitiveId: primitive.id, fields });
    }
  }
  return uses;
}

export interface PTEngineWebGPUOptions extends EngineOptions {
  /**
   * Denoiser pipeline for this converged backend. `auto` resolves to
   * `oidn-final` only when host OIDN assets are configured, otherwise to
   * `none`. Realtime-only and backend-specific denoisers are rejected at
   * construction instead of being silently replaced with another estimator.
   */
  readonly denoiser?: 'none' | 'auto' | 'oidn-final';
  /** pt-webgpu implements MNEE and SPPM; the realtime hybrid-only
   *  `refractive-trace` strategy is intentionally not accepted here. */
  readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';
  /**
   * The host-owned `GPUDevice`. The engine allocates GPU resources against it
   * but does NOT own its lifecycle (host-owns-lifecycle): the host creates the
   * device and disposes it. The cross-backend analogue is pt-webgl's
   * `device: THREE.WebGLRenderer`.
   */
  readonly device: GPUDevice;
  /**
   * Override adapter-based tier selection. Omit to auto-pick: `full` when the device
   * supports at least 28 storage buffers per shader stage and 5 storage textures
   * per shader stage (split full layout: TLAS, HDRI, area lights, caustics, etc.);
   * `lite` on 8/4 adapters.
   */
  readonly traceTier?: PtWebgpuTraceTier;
  /**
   * Enable Jakob-Hanika spectral hero-wavelength rendering (dispersion, thin-film,
   * spectral SSS). Off by default. (Graduated from the former
   * `extensions['vitrum.ptWebgpu.spectralHeroWavelength']`.)
   */
  readonly spectral?: boolean;
  /**
   * Enable bounded bidirectional path tracing with full Veach-style connection
   * MIS over the allocated explicit strategies. Full tier only; off by default.
   * Every finite, directional, and environment source can seed an invocation-local
   * light path. Ordinary eye direct/escape owns the non-connectable infinite
   * endpoint strategy; BDPT owns connections to its real scattering vertices.
   */
  readonly bdpt?: boolean;
  /**
   * Enable the opaque one-edge GRIS reconnection pre-passes (Lin 2022).
   * Full tier only. **OFF by default.**
   *
   * When OFF the default megakernel render is byte-identical (no reuse buffers or
   * pipelines are created). When ON (A1), a producer→temporal→SPATIAL→resolve
   * sequence runs BEFORE the megakernel each frame; resolve writes the
   * reconnection-INDIRECT estimate to `rpt_result` (also exposed for debug via
   * `getRestirPtResultBuffer()`), and the megakernel runs in COMPOSITE mode:
   * E0-direct-only (camera-visible emission + NEE at the primary vertex) + the
   * resolve indirect, composited into the BEAUTY accumulator. The estimator split
   * is exact + double-count-free (megakernel = E0's own emission+direct; resolve =
   * everything from the first bounce off E0 onward; see the kernel composite gate).
   * The spatial pass adds GRIS pairwise-MIS reuse over 5 disc neighbours (variance
   * reduction). The compile-time naga gate is
   * `wsl-gpu/scripts/restir-pt-compile-gate.ts`; the unbiasedness + equal-spp
   * variance A/Bs are executable promotion proofs under
   * `tools/radiometric-ab/ab-restir-pt.mjs`.
   */
  readonly oneEdgeReconnectionReuse?: boolean;
  /** Reconnection tuning; read only when one-edge reuse is enabled. */
  readonly oneEdgeReconnectionReuseOptions?: {
    /** Temporal M-clamp (history confidence cap, 1..4095). Default 20. */
    readonly mClamp?: number;
  };
  /**
   * @deprecated Compatibility alias for {@link oneEdgeReconnectionReuse}.
   * It selects exactly the same opaque, one-edge strategy; it does not enable
   * multi-vertex random replay or transmissive reconnection.
   */
  readonly restirPtReuse?: boolean;
  /** @deprecated Compatibility alias for {@link oneEdgeReconnectionReuseOptions}. */
  readonly restirPtReuseOptions?: {
    readonly mClamp?: number;
  };
  /**
   * Mesh BVH traversal backend. Default `'binary'` uses the canonical binary BVH.
   * `'cwbvh-closest'` opts the full-tier megakernel into the
   * uploaded compressed-wide BVH forest for closest-hit and any-hit mesh
   * traversal while preserving `castShadow:false` shadow predicate parity.
   */
  readonly bvhTraversal?: PtWebgpuBvhTraversalMode;
  /** BDPT tuning — read only when bdpt is true. */
  readonly bdptOptions?: {
    /**
     * Number of invocation-private light-subpath vertices. Integer 1..8; default 2.
     * One stores only the sampled source endpoint, two adds one scattering
     * vertex, and higher values extend the same bounded MIS strategy family.
     */
    readonly maxLightBounces?: number;
  };
  /**
   * Power-weighted light-tree importance sampling for direct-light NEE (Conty
   * Estévez & Kulla 2018). Full tier only, and only active when the scene has ≥2
   * selectable lights. ON by default; set `false` to force the unbiased uniform
   * emitter pick instead (same converged mean, higher variance) — used to A/B the
   * variance-reduction win. The lite tier always uses the uniform pick.
   */
  readonly lightTreeImportanceSampling?: boolean;
  /**
   * Primary path-sampling sequence. Default 'pcg' preserves the historical
   * stream. 'sobol' is a stable binding-free sequence across the megakernel and
   * SPPM, ReSTIR-PT, and BDPT auxiliary pipelines: dimensions 0 through 3 use
   * per-pixel, per-frame-block Owen-scrambled Sobol samples with an 8x8 ranked
   * rotation; dimension 4 onward uses a deterministic independent PCG
   * continuation. The full 32-bit frame index is represented as a 16-bit sample
   * index plus a bijective 16-bit block key, so accumulation does not repeat at
   * the 65,536-sample boundary.
   */
  readonly sampling?: 'pcg' | 'sobol';
  /**
   * Camera-visible emitters: emissive meshes glow when seen DIRECTLY by the
   * camera and THROUGH refractive surfaces (the stained-glass case), not only via
   * NEE on receiving surfaces. ON by default. When a scene represents an emissive
   * mesh as a mesh-area light and keeps the primitive material non-emissive, this
   * re-attaches that emitter radiance onto the primitive's material at pack time
   * so the path tracer's emissive-on-hit term fires. The existing BSDF↔light MIS
   * (`bsdfAreaLightConnectionContribution`) already covers diffuse/glossy bounces;
   * the emissive-on-hit term is gated to the camera + refraction paths the
   * analytic connection cannot reach, so there is NO double-count. Set `false` to
   * keep emitters camera-invisible (NEE-only, the pre-2026-05-30 behaviour).
   */
  readonly cameraVisibleEmitters?: boolean;
  /**
   * Intel Open Image Denoise final-pass config. REQUIRED when
   * `denoiser: 'oidn-final'`. (Graduated from the former
   * `extensions['vitrum.ptWebgpu.oidnModelUrl' | 'oidnExecutionProviders']`.)
   *
   * **Host-provided assets (`oidn-final` is NOT turnkey — vitrum ships neither):**
   *  1. `modelUrl` — the host fetches/bundles an OIDN ONNX model
   *     (e.g. `oidn_rt_hdr_alb_nrm.onnx`).
   *  2. The `onnxruntime-web` peer dependency must be installed by the host
   *     (it is intentionally an optional peer dep so the 5–20 MB runtime isn't
   *     forced on hosts that don't denoise). A missing model throws at engine
   *     construction; a missing `onnxruntime-web` throws at the first converged
   *     frame with an actionable "install onnxruntime-web" message.
   */
  readonly oidn?: {
    /** URL of the ONNX OIDN model (e.g. `oidn_rt_hdr_alb_nrm.onnx`). */
    readonly modelUrl: string;
    /** ONNX Runtime execution-provider preference order. */
    readonly executionProviders?: readonly ('webnn' | 'webgpu' | 'wasm')[];
  };
  /**
   * Test-only: inject a mock OIDN bridge (mirrors pt-webgl `oidnBridgeLoader`).
   */
  readonly oidnBridgeLoader?: import('./denoise/oidnFinalDispatcher.js').OIDNBridgeLoader;
  /**
   * Test-only: replace GPU texture readback (defaults to real `copyTextureToBuffer`).
   */
  readonly oidnReadbackFn?: import('./denoise/oidnFinalDispatcher.js').OidnReadbackFn;
}

export type {
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
} from './denoise/oidnFinalDispatcher.js';

const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const WORKGROUP_SIZE = 8;
const SPPM_WORKGROUP_COUNT = Math.ceil(SPPM_PHOTON_COUNT / 64);

// D8.14 — Tier capability sets are derived from the backend-local executable
// manifests in `supportManifest.ts`. The capabilities builder and all scene
// validation paths consume those same selected sets.

interface StateSlot {
  readonly get: () => EngineState;
  readonly set: (s: EngineState) => void;
}

interface PresentationSignature {
  readonly tonemapMode: number;
  readonly exposure: number;
  readonly outputColorSpace: number;
}

function presentationSignatureFor(input: FrameInput): PresentationSignature {
  const quality = input.quality ?? {};
  return {
    tonemapMode: TONEMAP_MODE_INDEX[quality.tonemap ?? 'aces'],
    exposure: quality.exposure ?? 1,
    outputColorSpace: quality.outputColorSpace === 'linear' ? 1 : 0,
  };
}

function samePresentationSignature(a: PresentationSignature, b: PresentationSignature): boolean {
  return a.tonemapMode === b.tonemapMode &&
    a.exposure === b.exposure &&
    a.outputColorSpace === b.outputColorSpace;
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

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

class PTEngineWebGPU implements Engine {
  readonly #slot: StateSlot;
  readonly backendProfileId: 'pt-webgpu' | 'pt-webgpu-lite';
  readonly #device: GPUDevice;
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;
  readonly #traceTier: PtWebgpuTraceTier;
  readonly #bvhTraversal: PtWebgpuBvhTraversalMode;

  #scene: Scene | null = null;
  #sceneBuffers: UploadedSceneBuffers | null = null;
  #geoPack: ScenePackResult | null = null;
  #samplesAccumulated = 0;
  #activeBounces = 1;
  /** WS5 — the most-recent host FrameInput, captured so an inverse-rendering
   *  session can re-render the SAME view with a frozen seed between optimizer
   *  steps. NOT overwritten by the session's own synthetic inverse renders. */
  #lastFrameInput: FrameInput | null = null;
  /** WS5 — true while an inverse session is driving synthetic renders, so the
   *  cached host camera (`#lastFrameInput`) isn't clobbered by them. */
  #inInverseRender = false;
  /** WS5 Phase-1 — lazily-created adjoint pass (D8.10 extraction). Engine-owned;
   *  freed in dispose. Created on first computeAdjointGradient call. */
  #adjointPass: AdjointPass | null = null;

  /**
   * The cohesive GPU-resource-lifecycle cluster (T14-followup extraction): accum
   * + aux textures and views, accum / varianceMoments / params buffers, the
   * compute pipeline(s), the group-0 bind-group layout, the cached per-frame bind
   * groups, and the accum dims. The engine owns exactly one instance.
   */
  readonly #gpu: GpuResources;
  #onFrameSubs = new Set<(stats: FrameStats) => void>();
  #onProgressSubs = new Set<(progress: ProgressStats) => void>();
  #onErrorSubs = new Set<(error: EngineError) => void>();
  #onWarningSubs = new Set<(warning: EngineWarning) => void>();
  // ── GPU error surface (item 28, trust-remediation-plan-2026-06-10) ────────
  // Dedup-throttle: only report one error per distinct message per N frames to
  // avoid spamming the host when a shader validation error fires every frame.
  #errorThrottleMap = new Map<string, number>(); // message → last-reported frameCount
  #errorFrameCount = 0;
  static readonly #ERROR_THROTTLE_FRAMES = 32;
  static readonly #ERROR_THROTTLE_MAX_MESSAGES = 256;
  readonly #onUncapturedError: (event: Event) => void;
  // device.lost is a Promise<GPUDeviceLostInfo> — we store no reference to it but
  // we attach a .then handler that routes into #emitError.  The handler is
  // removed implicitly when the Promise settles (Promises don't hold listener refs).
  readonly #postDenoiser: OIDNFinalDispatcher | null;
  readonly #denoiser: 'none' | 'oidn-final';
  #pendingDenoised: DenoisedFrame | null = null;
  #presentationSignature: PresentationSignature | null = null;
  readonly #spectralEnabled: boolean;
  readonly #lightTreeImportanceSampling: boolean;
  readonly #sampling: PtWebgpuSamplingMode;
  readonly #cameraVisibleEmitters: boolean;
  readonly #bdpt: boolean;
  readonly #bdptMaxLightBounces: number;
  /** ReSTIR-PT reuse — compile-time + full-tier; OFF by default. */
  readonly #restirPtReuse: boolean;
  readonly #restirPtMClamp: number;
  readonly #debugEnabled: boolean;

  // ── SPPM state (A4-progressive, photon-map strategy) ─────────────────────
  /** Cached initial radius r₀ = diagonal/100 from the scene AABB.
   *  Recomputed on every setScene. Used as the initial per-pixel linear-R seed
   *  (r₀²) in sppmUpdateProgressiveKind; the per-measure radius then shrinks
   *  progressively via the Hachisuka update rule (A4-progressive). */
  #sppmR0 = 0.017; // 1.7 cm — a safe pre-setScene default (1 m Cornell box)
  /** Half-diagonal of the scene used for the directional-light disk emitter. */
  #sppmSceneExtent = 10.0; // world units; refreshed on setScene

  /** Center of the scene AABB. Infinite-source launch disks are centered here,
   *  so photon coverage is independent of camera position. */
  #sppmSceneCenter: [number, number, number] = [0, 0, 0];
  /**
   * Scene-mutation fast-path dispatch (Task 4.3, Theme A). The engine stays the
   * thin owner of state; the router operates on that state through the
   * {@link MutationHost} seam wired in the constructor. Initialized there once
   * the device + gpu resources exist.
   */
  readonly #mutationRouter: SceneMutationRouter;

  static readonly #SUPPORTED_ANALYTIC_SHAPES = new Set(
    PT_WEBGPU_ANALYTIC_SHAPES.slice(1),
    );
    static readonly #NO_ANALYTIC_SHAPES = new Set<string>();

  constructor(opts: PTEngineWebGPUOptions, slot: StateSlot, traceTier: PtWebgpuTraceTier) {
    this.#slot = slot;
    this.#device = opts.device;
    this.#debugEnabled = opts.debug === true;
    if (this.#debugEnabled) {
      Object.defineProperty(this, 'debug', {
        value: this.#debugSurface,
        enumerable: true,
        configurable: false,
        writable: false,
      });
    }
    if (opts.onWarning != null) {
      this.#onWarningSubs.add(opts.onWarning);
    }
    this.#spectralEnabled = opts.spectral === true;
    this.#lightTreeImportanceSampling = opts.lightTreeImportanceSampling !== false;
    this.#sampling = opts.sampling === 'sobol' ? 'sobol' : 'pcg';
    this.#cameraVisibleEmitters = opts.cameraVisibleEmitters !== false;
    this.#traceTier = traceTier;
    this.backendProfileId = traceTier === 'lite' ? 'pt-webgpu-lite' : 'pt-webgpu';
    this.#bvhTraversal = opts.bvhTraversal === 'cwbvh-closest'
      ? 'cwbvh-closest'
      : 'binary';
    this.#maxBouncesLimit = opts.maxBounces ?? PT_WEBGPU_DEFAULT_BOUNCES;
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    const causticOpts = opts.causticOptions ?? {};
    this.#mneeMaxIterations = causticOpts.mneeMaxIterations ?? 8;
    this.#mneeMaxChainLength = causticOpts.mneeMaxChainLength ?? 3;
    this.#bdpt = opts.bdpt === true;
    // All integer depths 1..8 select the same exclusive finite-emitter
    // strategy family; 1 is endpoint-only and 2 adds the first extension.
    this.#bdptMaxLightBounces = resolveBdptMaxLightBounces(opts.bdptOptions?.maxLightBounces);
    // ReSTIR-PT reuse: compile-time opt-in, full-tier only. GpuResources
    // gates the full-tier requirement internally; mirror the resolved value here so
    // the capability + renderFrame sequencing agree with what GpuResources will run.
    this.#restirPtReuse =
      (opts.oneEdgeReconnectionReuse ?? opts.restirPtReuse) === true &&
      traceTier === 'full';
    const rptOpts =
      opts.oneEdgeReconnectionReuseOptions ?? opts.restirPtReuseOptions ?? {};
    this.#restirPtMClamp =
      rptOpts.mClamp ?? 20;
    this.#gpu = new GpuResources(
      opts.device,
      traceTier,
      this.#bdpt,
      this.#restirPtReuse,
      (warning) => this.#warn(warning),
      this.#sampling,
      this.#bvhTraversal,
    );
    this.#denoiser = opts.denoiser === 'oidn-final' ? 'oidn-final' : 'none';
    if (this.#denoiser === 'oidn-final') {
      const modelUrl = opts.oidn?.modelUrl;
      const eps = opts.oidn?.executionProviders?.filter(
        (p) => p === 'webnn' || p === 'webgpu' || p === 'wasm',
      );
      if (typeof modelUrl !== 'string' || modelUrl.length === 0) {
        throw new Error(
          "createPTEngine_WebGPU: denoiser: 'oidn-final' is not turnkey — it " +
            'requires TWO host-provided assets that vitrum does not ship: ' +
            '(1) oidn: { modelUrl } — a non-empty URL to an OIDN ONNX model ' +
            '(use oidn_rt_hdr_alb_nrm.onnx when supplying albedo + normal aux); ' +
            "and (2) the 'onnxruntime-web' optional peer dependency installed in " +
            'the host (a missing runtime otherwise throws at the first converged ' +
            'frame). Omit the `denoiser` option to render without a final denoise.',
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
              message: `[vitrum/pt-webgpu] OIDN final denoiser failed: ${errorMessage(err)}`,
              fatal: false,
              raw: err,
            });
          },
          onComplete: (frame) => {
            const state = this.#slot.get();
            if (state === 'disposed' || state === 'error') return;
            // Async inference completion queues CPU data only. Host-owned GPU
            // state is mutated exclusively by the next explicit renderFrame.
            this.#pendingDenoised = frame;
          },
        },
      );
    } else {
      this.#postDenoiser = null;
    }
    // Wire the scene-mutation router against the engine's own state seam. The
    // host closures read/write the engine's private fields; the router holds no
    // state of its own (Task 4.3, Theme A — god-class decomposition).
    this.#mutationRouter = new SceneMutationRouter({
      device: this.#device,
      assertLive: (method) => this.#assertLive(method),
      getScene: () => this.#scene,
      validatePrimitiveCandidate: (scene, primitiveId) => {
        assertOpticalMediumTopology(scene, {
          backend: this.backendProfileId,
          method: 'updatePrimitive',
          maxNestedMedia: 8,
          analyticGeometry: 'generated-triangle',
          transformArithmetic: this.#traceTier === 'lite'
            ? 'merged-world-f64-to-f32'
            : 'tlas-shader-f32',
        });
        const primitive = scene.primitives.find(
          (candidate) => candidate.id === primitiveId,
        );
        if (primitive == null) {
          throw new Error(
            `pt-webgpu primitive validation: "${primitiveId}" is absent from the candidate scene.`,
          );
        }
        const scopedScene: Scene = {
          primitives: [primitive],
          emitters: [],
          environment: { kind: 'none' },
        };
        if (this.#restirPtReuse) {
          assertOneEdgeReconnectionSceneSupported(scopedScene);
        }
        if (this.#traceTier === 'lite') assertLiteSceneSupported(scopedScene);
        if (this.#spectralEnabled) assertSpectralSceneSupported(scopedScene);
        assertThinFilmSceneSupported(scopedScene);
        assertVolumeSceneSupported(scopedScene);
      },
      validateEmitterCandidate: (scene, emitterId) => {
        if (this.#traceTier !== 'lite') return;
        const emitter = scene.emitters.find(
          (candidate) => candidate.id === emitterId,
        );
        if (emitter != null) {
          assertLiteSceneSupported({
            primitives: [],
            emitters: [emitter],
            environment: { kind: 'none' },
          });
        }
      },
      validateEnvironmentCandidate: (environment) => {
        validateSceneEnvironment(environment);
      },
      validateEmittersCandidate: (emitters, primitives) => {
        validateSceneEmitters(emitters, primitives);
        if (this.#traceTier === 'lite') {
          assertLiteSceneSupported({
            primitives: [],
            emitters,
            environment: { kind: 'none' },
          });
        }
      },
      setSceneState: (scene) => {
        this.#scene = scene;
      },
      getSceneBuffers: () => this.#sceneBuffers,
      getGeoPack: () => this.#geoPack,
      setGeoPack: (pack) => {
        this.#geoPack = pack;
      },
      invalidateBindGroups: () => this.#gpu.invalidateBindGroups(),
      supportedAnalyticShapes: () => this.#supportedAnalyticShapes(),
      cameraVisibleEmitters: () => this.#cameraVisibleEmitters,
      spectralEnabled: () => this.#spectralEnabled,
      stageLiteTextures: (sceneBuffers) =>
        this.#stageLiteTextures(
          sceneBuffers,
          uploadedSceneGpuResources(sceneBuffers),
        ),
      isLiteTier: () => this.#traceTier === 'lite',
      requiresMneeFacetTableRepack: () =>
        this.#causticStrategy === 'manifold-nee' && this.#traceTier === 'full',
      warn: (warning, ...args) => this.#warn(warning, ...args),
      repackScene: (scene, opts) => this.#repackScene(scene, opts),
      refreshSceneGeometryStats: () => {
        const previous = {
          r0: this.#sppmR0,
          extent: this.#sppmSceneExtent,
          center: this.#sppmSceneCenter,
          buffersReady: this.#gpu.sppm.sppmBuffersReady,
        };
        if (this.#causticStrategy === 'photon-map' && this.#traceTier === 'full') {
          this.#computeSppmSceneStats();
        }
        return () => {
          this.#sppmR0 = previous.r0;
          this.#sppmSceneExtent = previous.extent;
          this.#sppmSceneCenter = previous.center;
          this.#gpu.sppm.sppmBuffersReady = previous.buffersReady;
        };
      },
      setScene: (scene) => this.setScene(scene),
      reset: () => this.reset(),
    });

    // ── GPU error wiring (item 28) ─────────────────────────────────────────
    // Listen for validation/internal errors on the device.  The spec fires these
    // through the `uncapturederror` event on GPUDevice; we throttle per distinct
    // message to avoid per-frame spam when a shader has a persistent bug.
    // The listener is attached at construction and removed at dispose (host-owns-
    // lifecycle: we listen to a device we do not own, so we MUST clean up).
    this.#onUncapturedError = (event: Event): void => {
      if (this.#slot.get() === 'disposed') return;
      const gpuEvent = event as { error?: { message?: string } };
      const rawError = gpuEvent.error;
      // eslint-disable-next-line @typescript-eslint/no-base-to-string -- fallback stringification of GPUUncapturedErrorEvent; acceptable for diagnostic messages
      const message = rawError?.message ?? String(event);
      // Detect kind from the error constructor name (spec mandates GPUValidationError
      // and GPUInternalError as the only two concrete subtypes).
      const kind: EngineError['kind'] = rawError != null && rawError.constructor?.name === 'GPUInternalError'
        ? 'gpu-internal'
        : 'gpu-validation';
      // Throttle: only report the first occurrence per message per N frames.
      const lastFrame = this.#errorThrottleMap.get(message) ?? -Infinity;
      if (this.#errorFrameCount - lastFrame >= PTEngineWebGPU.#ERROR_THROTTLE_FRAMES) {
        if (
          !this.#errorThrottleMap.has(message) &&
          this.#errorThrottleMap.size >= PTEngineWebGPU.#ERROR_THROTTLE_MAX_MESSAGES
        ) {
          const oldest = this.#errorThrottleMap.keys().next();
          if (!oldest.done) this.#errorThrottleMap.delete(oldest.value);
        }
        this.#errorThrottleMap.set(message, this.#errorFrameCount);
        this.#emitError({ kind, message, fatal: false, raw: rawError });
      }
    };
    opts.device.addEventListener('uncapturederror', this.#onUncapturedError);

    // device.lost: settle → transition engine to 'error' state and emit fatal.
    // We do NOT own the device so we only report + block further rendering.
    opts.device.lost.then((info: { reason?: string; message?: string }) => {
      if (this.#slot.get() === 'disposed') return;
      this.#slot.set('error');
      this.#emitError({
        kind: 'device-lost',
        message: info.message ?? `GPUDevice lost (reason: ${info.reason ?? 'unknown'})`,
        fatal: true,
        raw: info,
      });
    }).catch(() => {
      // device.lost should not reject per spec, but guard defensively.
    });
  }

  get state(): EngineState {
    return this.#slot.get();
  }

  get capabilities(): EngineCapabilities {
    return ptWebgpuCapabilities({
      traceTier: this.#traceTier,
      maxSamplesLimit: this.#maxSamplesLimit,
      maxBouncesLimit: this.#maxBouncesLimit,
      bdpt: this.#bdpt,
      restirPtReuse: this.#restirPtReuse,
      sampling: this.#sampling,
      bvhTraversal: this.#bvhTraversal,
      causticStrategy: this.#causticStrategy,
      spectral: this.#spectralEnabled,
      denoiser: this.#denoiser,
      debug: this.#debugEnabled,
    });
  }

  /** Supported analytic shapes (id > 0) — passed to the pure incrementalPatch
   *  resolvers so they reproduce buildPackedScene's material/analytic ordering. */
  #supportedAnalyticShapes(): ReadonlySet<string> {
    return this.#traceTier === 'lite'
      ? PTEngineWebGPU.#NO_ANALYTIC_SHAPES
      : PTEngineWebGPU.#SUPPORTED_ANALYTIC_SHAPES;
  }

  // ── Debug introspection (T3.G followup + #30 pickPrimitive) ──────────────
  // Atlas / BVH / denoiser-toggle hooks are walkaround-hybrid concepts that
  // don't apply to a brute-force compute path tracer. GPU-memory estimate
  // and CPU click-to-pick (pickPrimitive) are both wired.
  readonly #debugSurface: EngineDebugSurface = {
    // T3.G #30 — CPU ray-cast pick using the retained scene + last-frame camera.
    // Returns null before the first renderFrame (no camera) or on a miss.
    pickPrimitive: (x: number, y: number): string | null => {
      const scene = this.#scene;
      const last = this.#lastFrameInput;
      if (scene == null || last == null) return null;
      const w = last.viewport.width;
      const h = last.viewport.height;
      const cam: PickCamera = {
        viewMatrix: last.viewMatrix,
        projMatrix: last.projMatrix,
        cameraPosition: resolveFrameCameraPosition(
          last,
          'PTEngineWebGPU.debug.pickPrimitive',
        ),
      };
      return pickPrimitiveCpu(scene, cam, x, y, w, h);
    },

    estimatedGpuMemoryBytes: (): GpuMemoryBreakdown | null => {
      const gpu = this.#gpu;
      const W = gpu.accumWidth;
      const H = gpu.accumHeight;
      // Pre-init / pre-renderFrame: no accum textures yet.
      if (W <= 0 || H <= 0 || gpu.accumTexture == null) return null;

      // Per-texel bytes inferred from the actual formats at allocation time.
      // Color/G-buffer targets are rgba16float; scalar variance is r32float.
      const RGBA16F = 8;
      const R32F = 4;
      const texPixels = W * H;
      const accumBytes        = texPixels * RGBA16F;
      const normalDepthBytes  = texPixels * RGBA16F;
      const albedoBytes       = texPixels * RGBA16F;
      const varianceBytes     = texPixels * R32F;
      const motionBytes       = gpu.motionVectorsTexture != null
        ? texPixels * RGBA16F : 0;
      const presentTextureBytes = gpu.present.presentTexture != null
        ? texPixels * RGBA16F : 0;
      const denoisedLinearTextureBytes = gpu.present.denoisedLinearTexture != null
        ? texPixels * 16 : 0; // rgba32float
      const accumBufBytes     = gpu.accumBufferByteSize;
      const varMomentBufBytes = gpu.varianceMomentsBuffer != null
        ? gpu.accumBufferByteSize : 0;
      const bdptCameraSplatBufBytes = gpu.bdptCameraSplatBuffer != null
        ? gpu.accumBufferByteSize : 0;
      const paramsBufBytes    = gpu.paramsBuffer != null ? FRAME_PARAMS_BUFFER_ALLOC_BYTES : 0;
      const reservoirStorageBytes =
        (gpu.reservoir.rptReservoirCur != null ? gpu.reservoir.rptReservoirByteSize : 0) +
        (gpu.reservoir.rptReservoirPrev != null ? gpu.reservoir.rptReservoirByteSize : 0) +
        (gpu.reservoir.rptResultBuffer != null ? gpu.reservoir.rptResultByteSize : 0);
      const reservoirUniformBytes = gpu.reservoir.rptParamsBuffer != null
        ? GpuResources.RESTIR_PT_PARAMS_BYTES : 0;
      const sppmStorageBytes =
        (gpu.sppm.sppmPhotonCellsBuffer != null
          ? (gpu.sppm.sppmBuffersReady ? SPPM_PHOTON_CELLS_BYTES : 64)
          : 0) +
        (gpu.sppm.sppmCellCountersBuffer != null
          ? (gpu.sppm.sppmBuffersReady ? SPPM_CELL_COUNTERS_BYTES : 64)
          : 0) +
        (gpu.sppm.sppmPixelStatsBuffer != null
          ? (gpu.sppm.sppmPixelStatsWidth > 0 && gpu.sppm.sppmPixelStatsHeight > 0
              ? Math.max(
                  64,
                  gpu.sppm.sppmPixelStatsWidth *
                    gpu.sppm.sppmPixelStatsHeight *
                    SPPM_PIXEL_STATS_BYTES_PER_PIXEL,
                )
              : 64)
          : 0);
      const sppmUniformBytes = gpu.sppm.sppmStatsBuffer != null
        ? (gpu.sppm.sppmBuffersReady ? SPPM_STATS_BYTES : 64)
        : 0;
      const presentationStorageBytes = gpu.present.seedBlitVarPlaceholder != null
        ? gpu.present.seedBlitVarPlaceholderByteSize : 0;
      const presentationUniformBytes =
        (gpu.present.presentParamsBuffer != null ? GpuResources.PRESENT_PARAMS_BYTES : 0) +
        (gpu.present.seedBlitParamsBuffer != null ? 32 : 0);
      // Scene buffers (BVH/materials/indices/TLAS/emitters/light-tree/UVs) + the
      // material texture arrays, summed live off the current handles (was
      // previously omitted, so `total` under-reported by the whole scene).
      const scene = this.#sceneBuffers != null
        ? this.#sceneBuffers.gpuMemoryBytes()
        : { bufferBytes: 0, textureBytesByFormat: {} as Readonly<Record<string, number>> };
      const sceneTexBytes = Object.values(scene.textureBytesByFormat).reduce((a, b) => a + b, 0);
      const sceneBytes = scene.bufferBytes + sceneTexBytes;

      const commonTexBytes = accumBytes + normalDepthBytes + albedoBytes
        + varianceBytes + motionBytes;
      const presentationTexBytes = presentTextureBytes + denoisedLinearTextureBytes;
      const commonBufBytes = accumBufBytes + varMomentBufBytes
        + bdptCameraSplatBufBytes + paramsBufBytes;
      const reservoirBytes = reservoirStorageBytes + reservoirUniformBytes;
      const sppmBytes = sppmStorageBytes + sppmUniformBytes;
      const presentationBytes =
        presentationTexBytes + presentationStorageBytes + presentationUniformBytes;
      const total =
        commonTexBytes + commonBufBytes + sceneBytes +
        reservoirBytes + sppmBytes + presentationBytes;

      // byTextureFormat: common render targets plus the scene's material arrays
      // by their actual format. Keeps the secondary tables summing to `total`
      // (the documented invariant).
      const textureFormats: Record<string, number> = {
        rgba16float: commonTexBytes - varianceBytes + presentTextureBytes,
        r32float: varianceBytes,
      };
      if (denoisedLinearTextureBytes > 0) {
        textureFormats['rgba32float'] = denoisedLinearTextureBytes;
      }
      for (const [fmt, bytes] of Object.entries(scene.textureBytesByFormat)) {
        textureFormats[fmt] = (textureFormats[fmt] ?? 0) + bytes;
      }

      return Object.freeze({
        total,
        byCategory: Object.freeze({
          common: commonTexBytes + commonBufBytes,
          scene: sceneBytes,
          reservoir: reservoirBytes,
          sppm: sppmBytes,
          presentation: presentationBytes,
        }),
        byTextureFormat: Object.freeze(textureFormats),
        byBufferUsage: Object.freeze({
          storage:
            accumBufBytes + varMomentBufBytes +
            bdptCameraSplatBufBytes + scene.bufferBytes +
            reservoirStorageBytes + sppmStorageBytes + presentationStorageBytes,
          uniform:
            paramsBufBytes + reservoirUniformBytes +
            sppmUniformBytes + presentationUniformBytes,
        }),
      });
    },
  };

  #assertUsable(method: string): void {
    const state = this.#slot.get();
    if (state === 'disposed') {
      throw new Error(`${method}: engine is disposed`);
    }
    if (state === 'error') {
      throw new Error(`${method}: engine is in a fatal error state`);
    }
  }

  #assertLive(method: string): void {
    this.#assertUsable(method);
    if (this.#scene == null) {
      throw new Error(`${method}: call setScene() before ${method}`);
    }
  }

  #emitFrameStats(stats: FrameStats): void {
    for (const cb of this.#onFrameSubs) {
      try {
        cb(stats);
      } catch {
        // Telemetry callbacks must not break rendering.
      }
    }
  }

  #emitProgress(progress: ProgressStats): void {
    for (const cb of this.#onProgressSubs) {
      try {
        cb(progress);
      } catch {
        // Telemetry callbacks must not break rendering.
      }
    }
  }

  #emitError(error: EngineError): void {
    for (const cb of this.#onErrorSubs) {
      try {
        cb(error);
      } catch {
        // Error callbacks must not break rendering.
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

  /**
   * Build the `kind:'rendered'` FrameOutput: primary radiance plus the four
   * conditionally-present aux textures (normalDepth / albedo / variance /
   * motionVectors), each wrapped in `asBackendTexture`. Written verbatim at
   * three return sites (paused fast-out, already-converged fast-out, and the
   * post-dispatch path) before T14; centralized here. The post-dispatch site
   * passes `accumTexturePost` as `primary` (re-read after dispatch) while the
   * fast-outs pass the pre-read `#gpu.accumTexture` — that distinction is
   * preserved by the caller supplying the `primary` texture explicitly.
   */
  #frameOutput(
    primary: GPUTexture,
    samplesAccumulated: number,
    isConverged: boolean,
  ): FrameOutput {
    // Prefer the tonemapped presentTexture as primaryRadiance (written by the
    // present pass each renderFrame).  Fall back to the raw linear accumTexture
    // on the fast-outs (before the first render, paused, converged) where the
    // present pass does not dispatch — presentTexture already carries the last
    // frame's tonemapped output and is a valid displayable surface.
    const displayTex = this.#gpu.present.presentTexture ?? primary;
    return {
      kind: 'rendered',
      primaryRadiance: asBackendTexture<'webgpu', GPUTexture>(displayTex),
      ...(this.#gpu.normalDepthTexture != null
        ? { normalDepth: asBackendTexture<'webgpu', GPUTexture>(this.#gpu.normalDepthTexture) }
        : {}),
      ...(this.#gpu.albedoTexture != null
        ? { albedo: asBackendTexture<'webgpu', GPUTexture>(this.#gpu.albedoTexture) }
        : {}),
      ...(this.#traceTier === 'full' && this.#gpu.varianceTexture != null
        ? { variance: asBackendTexture<'webgpu', GPUTexture>(this.#gpu.varianceTexture) }
        : {}),
      ...(this.#gpu.motionVectorsTexture != null
        ? { motionVectors: asBackendTexture<'webgpu', GPUTexture>(this.#gpu.motionVectorsTexture) }
        : {}),
      samplesAccumulated,
      isConverged,
    };
  }

  /**
   * Presentation SOURCE for this offscreen backend (V1-1 / R2). pt-webgpu
   * declares `presentationMode: 'offscreen-texture'` — it renders to an internal
   * texture and never presents to the host canvas itself. A canvas-owning host
   * (e.g. `attachVitrum`) blits the returned texture to the canvas each frame.
   *
   * Returns the backend's own {@link GPUDevice} (which owns the texture) plus the
   * displayable surface — the same tonemapped `presentTexture` `#frameOutput`
   * hands back as `primaryRadiance`. `device` is typed `unknown` in the core
   * contract so the backend-agnostic core carries no WebGPU types; the host casts
   * at the boundary. Null before any GPU resources exist (pre-first-frame /
   * disposed) so the host skips the blit until there is something to present.
   */
  getPresentationSource(): { device: unknown; texture: BackendTexture } | null {
    const displayTex = this.#gpu.present.presentTexture ?? this.#gpu.accumTexture;
    if (displayTex == null) return null;
    return {
      device: this.#device,
      texture: asBackendTexture<'webgpu', GPUTexture>(displayTex),
    };
  }

  /**
   * Emit the per-frame telemetry pair (onFrame stats + onProgress) that all
   * three renderFrame return paths share. `spp` is 0 for the no-op fast-outs
   * and 1 for an actual dispatch; `fraction` is computed from
   * `current/target` with the same clamp the call sites used (the converged
   * fast-out previously hard-coded `fraction: 1`, which equals the clamped
   * ratio once `current >= target`).
   */
  #emitFrameTelemetry(
    frameStartMs: number,
    spp: number,
    current: number,
    target: number,
  ): void {
    const hasFrameSubs = this.#onFrameSubs.size > 0;
    const hasProgressSubs = this.#onProgressSubs.size > 0;
    if (!hasFrameSubs && !hasProgressSubs) return;

    if (hasFrameSubs) {
      const frameEndMs = globalThis.performance?.now?.() ?? Date.now();
      const mem = this.#debugEnabled
        ? this.#debugSurface.estimatedGpuMemoryBytes?.() ?? null
        : null;
      const denoiserState = this.#postDenoiser != null
        ? this.#postDenoiser.getState()
        : { status: 'disabled' as const, reason: null };
      this.#emitFrameStats({
        frameTimeMs: Math.max(0, frameEndMs - frameStartMs),
        spp,
        ...(mem != null ? { gpuMemoryBytes: mem, estimatedGpuMemoryBytes: mem.total } : {}),
        denoiserState,
      });
    }
    if (hasProgressSubs) {
      this.#emitProgress({
        kind: 'pt-spp',
        current,
        target,
        fraction: target > 0 ? Math.max(0, Math.min(1, current / target)) : 1,
      });
    }
  }

  /**
   * Pack the per-frame uniform buffer. Thin delegate to the extracted pure
   * {@link packFrameParams} (Task 4.3, Theme A) — the engine owns its state
   * (#sceneBuffers + config); the packer operates on a snapshot of it. Callers
   * must have already validated that #sceneBuffers is non-null (renderFrame's
   * preconditions handle this). Byte layout is pinned to pathTraceBruteforce.wgsl.
   */
  #buildParamsBuffer(input: FrameInput, width: number, height: number): ArrayBuffer {
    return packFrameParams(
      {
        activeBounces: this.#activeBounces,
        mneeMaxIterations: this.#mneeMaxIterations,
        mneeMaxChainLength: this.#mneeMaxChainLength,
        causticStrategy: this.#causticStrategy,
        spectralEnabled: this.#spectralEnabled,
        traceTier: this.#traceTier,
        bdpt: this.#bdpt,
        bdptMaxLightBounces: this.#bdptMaxLightBounces,
        lightTreeImportanceSampling: this.#lightTreeImportanceSampling,
        directLightingMode: this.#inInverseRender ? 'summed-expectation' : 'sampled-selection',
      },
      this.#sceneBuffers!,
      input,
      width,
      height,
    );
  }

  setScene(scene: Scene): void {
    this.#assertUsable('setScene');
    if (this.#traceTier === 'lite') {
      assertLiteSceneSupported(scene);
    }
    this.#repackScene(scene, { warnOnEmpty: true });
  }

  /**
   * D8.15 — Recompute the SPPM scale-aware initial radius (#sppmR0) and scene
   * half-diagonal (#sppmSceneExtent) from the current packed-scene root bounds.
   *
   * The uploaded scene's `sceneCenter`/`sceneRadius` are derived from the TLAS
   * root when present, so transform-only refits update this state even while
   * local BLAS positions remain byte-identical. The packed-position scan is
   * retained only as a pre-upload/legacy fallback.
   *
   * Called only from `#repackScene` when `causticStrategy === 'photon-map'` on
   * the full tier — the guard is preserved by the caller (behavior-preserving).
   */
  #computeSppmSceneStats(): void {
    const sceneBuffers = this.#sceneBuffers;
    const bounds = sceneBuffers != null
      ? sppmSceneBoundsFromCenterRadius(
          sceneBuffers.sceneCenter,
          sceneBuffers.sceneRadius,
        )
      : sppmSceneBoundsFromPackedPositions(this.#geoPack?.positions ?? []);
    if (bounds != null) {
      this.#sppmR0 = bounds.initialRadius;
      this.#sppmSceneExtent = bounds.extent;
      this.#sppmSceneCenter = [...bounds.center];
    }
    // A4-progressive: invalidate SPPM buffers so they rebuild with the new
    // scene-extent stats (r₀ recomputed above).  The per-pixel stats buffer
    // is cleared in reset() (called below) — static-eye-point invariant holds
    // because a setScene always resets accumulation.
    this.#gpu.sppm.sppmBuffersReady = false;
  }

  /**
   * Full scene repack: rebuild the packed BLAS/TLAS/material/analytic/light
   * arrays from `scene`, destroy the stale scene buffers, upload the fresh set,
   * re-init the BDPT light path, invalidate cached bind groups, and reset
   * accumulation. This is the shared spine of `setScene` and the
   * `addPrimitive` / `removePrimitive` whole-primitive mutators — the latter two
   * hand a mutated copy of the live scene so the dense material / analytic /
   * triMaterialId packing is reproduced by the SAME packing logic (no fragile
   * per-array index remap; see class header on the add/remove design choice).
   */
  #repackScene(scene: Scene, opts: { readonly warnOnEmpty: boolean }): void {
    validateCoreScene(scene);
    // The core contract requires immutable/fresh MaterialSpec and TextureRef
    // objects. After that boundary, capture their downstream renderer view once
    // so MNEE admission/table construction, texture descriptors, role metadata,
    // and atlas staging cannot independently re-observe the host objects.
    const materialInputSnapshotContext = createMaterialInputSnapshotContext();
    assertOpticalMediumTopology(scene, {
      backend: this.backendProfileId,
      method: 'setScene',
      maxNestedMedia: 8,
      analyticGeometry: 'generated-triangle',
      transformArithmetic: this.#traceTier === 'lite'
        ? 'merged-world-f64-to-f32'
        : 'tlas-shader-f32',
    });
    if (this.#restirPtReuse) {
      assertOneEdgeReconnectionSceneSupported(scene);
    }
    // CAP-01 — reject material fields the selected tier cannot reproduce (matrix-
    // driven: every 'unsupported' row in the ledger's pt-webgpu material support
    assertThinFilmSceneSupported(scene);
    assertVolumeSceneSupported(scene);
    if (this.#spectralEnabled) assertSpectralSceneSupported(scene);
    // matrix). Once per setScene/repack.
    const unsupportedMaterialUses = collectUnsupportedMaterialFieldUses(scene, this.#traceTier);
    const unsupportedMaterialFields = collectFieldUnion(unsupportedMaterialUses);
    if (unsupportedMaterialFields.length > 0) {
      throw new TypeError(
        `[vitrum/pt-webgpu] setScene: material fields are supplied but not rendered ` +
          `by the selected ${this.#traceTier} tier: ${unsupportedMaterialFields.join(', ')} ` +
          `(primitive fields: ${unsupportedMaterialUses.map((use) =>
            `${use.primitiveId}=[${use.fields.join(', ')}]`).join('; ')}).`,
      );
    }
    const packed = buildPackedScene(scene, {
      cameraVisibleEmitters: this.#cameraVisibleEmitters,
      spectralEnabled: this.#spectralEnabled,
      geometryMode: this.#traceTier === 'lite' ? 'merged' : 'tlas',
      includeMneeFacetCandidates:
        this.#causticStrategy === 'manifold-nee' && this.#traceTier === 'full',
      mneeFacetCandidateStorageLimitBytes: Math.min(
        Number(this.#device.limits.maxBufferSize),
        Number(this.#device.limits.maxStorageBufferBindingSize),
      ),
      materialInputSnapshotContext,
      warningPhase: 'setScene',
      warningMethod: 'setScene',
    });
    const preUploadWarnings = new Set(
      packed.warnings.filter((warning) => warning.includes('displacementMap')),
    );
    for (const warning of preUploadWarnings) {
      this.#warn({
        code: 'pt-webgpu.scene-pack-warning',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        message: `[vitrum/pt-webgpu] ${warning}`,
        details: displacementWarningDetails(warning),
      });
    }
    // V2-2: atomic setScene. Upload into a LOCAL first; only after it succeeds do
    // we swap it in and destroy the old buffers. Previously the old buffers were
    // destroyed BEFORE uploadPackedScene ran — a mid-upload throw (uploadPackedScene
    // does ~30 sequential buffer creates) left the engine sceneless AND leaked the
    // buffers the failed upload had already created. uploadPackedScene now cleans up
    // its own partial allocations on throw (see uploadSceneBuffers.ts); this swap
    // keeps the previously-valid scene installed if the new upload fails.
    const previous = {
      scene: this.#scene,
      sceneBuffers: this.#sceneBuffers,
      geoPack: this.#geoPack,
      sppmSceneCenter: this.#sppmSceneCenter,
      sppmR0: this.#sppmR0,
      sppmSceneExtent: this.#sppmSceneExtent,
      sppmBuffersReady: this.#gpu.sppm.sppmBuffersReady,
    };
    const transactionPreservedResources: object[] = [
      ...(previous.sceneBuffers != null ? uploadedSceneGpuResources(previous.sceneBuffers) : []),
      ...[
        this.#gpu.liteEnvTexture,
        this.#gpu.liteEnvCdfTexture,
        this.#gpu.liteLightTexture,
      ].filter((resource): resource is GPUTexture => resource != null),
    ];

    let uploadedScene: UploadedSceneBuffers | null = null;
    let liteReplacement: LiteTextureReplacement | null = null;
    try {
      uploadedScene = uploadPackedScene(this.#device, packed, transactionPreservedResources);
      const candidateOwnedResources = [
        ...uploadedSceneGpuResources(uploadedScene),
      ];
      liteReplacement = this.#stageLiteTextures(uploadedScene, [
        ...transactionPreservedResources,
        ...candidateOwnedResources,
      ]);
    } catch (error) {
      liteReplacement?.rollback();
      try { uploadedScene?.destroy(); } catch { /* preserve allocation failure */ }
      throw error;
    }

    // Publication is reversible until reset's GPU uploads/submission succeed.
    // Old resources remain live throughout this block.
    this.#geoPack = scenePackResultFromPacked(packed);
    this.#sceneBuffers = uploadedScene;
    this.#scene = scene;
    liteReplacement?.commit();
    this.#gpu.invalidateBindGroups();
    try {
      // A4 — recompute SPPM scale-aware initial radius from the scene AABB.
      if (this.#causticStrategy === 'photon-map' && this.#traceTier === 'full') {
        this.#computeSppmSceneStats();
      }
      this.reset();
    } catch (error) {
      this.#scene = previous.scene;
      this.#sceneBuffers = previous.sceneBuffers;
      this.#geoPack = previous.geoPack;
      this.#sppmSceneCenter = previous.sppmSceneCenter;
      this.#sppmR0 = previous.sppmR0;
      this.#sppmSceneExtent = previous.sppmSceneExtent;
      this.#gpu.sppm.sppmBuffersReady = previous.sppmBuffersReady;
      liteReplacement?.rollback();
      this.#gpu.invalidateBindGroups();
      try { uploadedScene.destroy(); } catch { /* preserve reset failure */ }
      throw error;
    }

    // Irreversible retirement happens only after every candidate and reset
    // upload has succeeded.  Destruction failures cannot invalidate the newly
    // published resource set, so teardown is deliberately best-effort.
    liteReplacement?.finalize();
    try { previous.sceneBuffers?.destroy(); } catch { /* new scene is live */ }
    if (opts.warnOnEmpty) {
      const sceneSummary = summarizeScene(scene);
      if (sceneSummary.primitiveCount === 0) {
        this.#warn({
          code: 'pt-webgpu.empty-scene',
          backend: 'pt-webgpu',
          phase: 'setScene',
          method: 'setScene',
          message: '[vitrum/pt-webgpu] Empty scene provided; rendering sky-only fallback.',
        });
      }
    }
    for (const warning of uploadedScene.structuredWarnings) {
      this.#warn(warning);
    }
    const structuredScenePackWarnings = new Set(
      uploadedScene.structuredWarnings
        .map((warning) => warning.details?.warning)
        .filter((warning): warning is string => typeof warning === 'string'),
    );
    for (const warning of uploadedScene.warnings) {
      if (preUploadWarnings.has(warning)) continue;
      if (structuredScenePackWarnings.has(warning)) continue;
      this.#warn({
        code: 'pt-webgpu.scene-pack-warning',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        message: `[vitrum/pt-webgpu] ${warning}`,
        details: displacementWarningDetails(warning),
      });
    }
  }

  /** Build a complete lite texture candidate without publishing it. */
  #stageLiteTextures(
    sceneBuffers: UploadedSceneBuffers | null,
    additionalForbiddenResources: readonly object[] = [],
  ): LiteTextureReplacement | null {
    if (this.#traceTier !== 'lite' || sceneBuffers == null) return null;
    const lightTex = packLiteLightTexture(
      sceneBuffers.directionalLightsData,
      sceneBuffers.pointLightsData,
      sceneBuffers.spotLightsData,
      sceneBuffers.rectAreaLightsData,
    );
    const envTex = packLiteEnvTexture(
      sceneBuffers.environmentMapTexels,
      sceneBuffers.environmentMapWidth,
      sceneBuffers.environmentMapHeight,
      sceneBuffers.hasEnvironmentMap,
    );
    const cdfTex = packLiteEnvCdfTexture(
      sceneBuffers.environmentMapCdf,
      sceneBuffers.environmentMapWidth,
      sceneBuffers.environmentMapHeight,
      sceneBuffers.hasEnvironmentMap,
    );
    return this.#gpu.stageLiteTextureReplacement(
      lightTex,
      envTex,
      cdfTex,
      additionalForbiddenResources,
    );
  }

  // ── Scene mutation (Task 4.3, Theme A) ───────────────────────────────────
  // The fast-path dispatch for add/remove/updatePrimitive/updateEmitter/
  // updateEnvironment lives in `SceneMutationRouter`. The engine stays the thin
  // owner of state (#scene / #sceneBuffers / #geoPack / device / pipelines); the
  // router operates on that state through the `MutationHost` seam below.
  addPrimitive(primitive: ScenePrimitive): void {
    this.#mutationRouter.addPrimitive(primitive);
  }

  removePrimitive(id: ScenePrimitive['id']): void {
    this.#mutationRouter.removePrimitive(id);
  }

  updatePrimitive(id: string, patch: ScenePrimitivePatch): void {
    this.#mutationRouter.updatePrimitive(id, patch);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#mutationRouter.updateEmitter(id, patch);
  }

  updateEnvironment(env: Scene['environment'] | null): void {
    this.#mutationRouter.updateEnvironment(env);
  }

  updateLighting(opts: Readonly<Record<string, unknown>>): void {
    this.#mutationRouter.updateLighting(opts);
  }

  setSize(width: number, height: number): void {
    this.#assertUsable('setSize');
    validatePtWebgpuPixelSize('setSize', width, height);
    if (!this.#gpu.ensureAccumResources(width, height)) return;
    // ensureAccumResources publishes a completely zeroed replacement cohort.
    // ReSTIR/SPPM size-dependent resources are rebuilt on the next frame, so no
    // second GPU clear is necessary here. Reset only the engine-owned history.
    this.#samplesAccumulated = 0;
    this.#pendingDenoised = null;
    this.#postDenoiser?.invalidate();
    this.#presentationSignature = null;
  }

  /**
   * D8.4 — Ensure all per-frame GPU resources are allocated and return the set
   * of readiness flags that the pass-encoding step consumes.
   *
   * Covers (in order, PRESERVING the documented invariant):
   *   1. ALL SPPM buffer allocation (BEFORE buildBindGroups — Item-1 fix invariant).
   *   2. Params UBO packing + queue.writeBuffer.
   *   3. buildBindGroups (scene bind groups).
   *   4. ReSTIR-PT reservoir ensure / pipeline build / writeReservoirParams /
   *      buildReservoirBindGroups.
   *
   * Returns `{ bindGroup, sppmReady, restirPtReady }`.
   * Callers must have already verified `#gpu.isReadyToRender()` and that
   * `#sceneBuffers` is non-null (renderFrame's preconditions cover this).
   */
  #ensurePerFrameResources(
    input: FrameInput,
    width: number,
    height: number,
  ): {
    bindGroup: GPUBindGroup;
    sppmReady: boolean;
    restirPtReady: boolean;
  } {
    const gpu = this.#gpu;

    // Per-subsystem preparation, in the SAME order as the former monolithic body:
    //   1. SPPM hash-grid / per-pixel-stats buffers,
    //   2. FrameParams UBO pack + write (after SPPM readiness is established),
    //   3. build the scene bind groups,
    //   4. ReSTIR-PT reuse reservoirs/pipelines/params.
    const sppmReady = this.#ensureSppmPerFrame(gpu, width, height);
    this.#ensureParamsPerFrame(gpu, input, width, height);
    const bindGroup = gpu.buildBindGroups(this.#sceneBuffers!);
    const restirPtReady = this.#ensureRestirPtPerFrame(gpu, width, height);
    return { bindGroup, sppmReady, restirPtReady };
  }

  /**
   * Per-frame FrameParams UBO pack + write. BDPT path state is invocation-local
   * WGSL memory and therefore has no host resource readiness dependency.
   */
  #ensureParamsPerFrame(
    gpu: GpuResources,
    input: FrameInput,
    width: number,
    height: number,
  ): void {
    // Requested SPPM readiness is established before this pack or throws.
    const paramsArrayBuffer = this.#buildParamsBuffer(input, width, height);
    this.#device.queue.writeBuffer(gpu.paramsBuffer!, 0, paramsArrayBuffer);
  }

  /**
   * Per-frame SPPM hash-grid + per-pixel-stats buffer allocation. MUST run before
   * `buildBindGroups` so group-3 is built with the correct (real or placeholder)
   * buffer handles. Returns whether the SPPM photon pass is ready to dispatch.
   * Extracted from `#ensurePerFrameResources` (T3-B) — behaviour identical.
   */
  #ensureSppmPerFrame(gpu: GpuResources, width: number, height: number): boolean {
    // A4 + Item-1 fix (2026-06-10): ALL SPPM buffer allocation happens HERE,
    // BEFORE buildBindGroups, so that group-3 is always built with the correct
    // (real or placeholder) buffer handles.  The previous ordering had
    // ensureSppmBuffers(true) AFTER buildBindGroups, which caused
    // invalidateGroup3BindGroup() to null pathTraceBindGroup3 AFTER group-0 was
    // already cached — buildBindGroups then returned the cached group-0 without
    // rebuilding group-3, leaving pathTraceBindGroup3 null on every subsequent
    // frame (photon pass and megakernel both silently skipped group-3 → photons
    // were never written, gather read nothing, lum=0 gpuErrs=1 on full tier).
    //
    // Per-pixel (τ, linear R, N) persists across iterations and resets with temporal
    // accumulation. The hash-grid bucket heads are cleared before every photon
    // pass, then this iteration's unique photon records are published. Gather applies the progressive
    // update rule (N'=N+αM, R'²=R²·N'/(N+M), τ'=(τ+Φ_M)·ratio) instead of the
    // frozen-radius density estimate.
    const sppmActive =
      this.#causticStrategy === 'photon-map' && this.#traceTier === 'full';
    let sppmReady = false;
    if (sppmActive) {
      const sppmBuffersOk = gpu.ensureSppmBuffers(true);
      if (!sppmBuffersOk) {
        throw new Error(
          '@vitrum/pt-webgpu: causticStrategy="photon-map" requested, but the SPPM photon grid could not be allocated within the device storage-buffer limits.',
        );
      }
      // A4-progressive: ensure per-pixel stats buffer at the current render dims.
      // ensureSppmPixelStatsBuffer is idempotent on a cache hit (same W×H); on
      // a first allocation or dim change it GPU-clears (τ/R/N → 0) and
      // invalidates group-3.  A 64-byte placeholder is already created by
      // ensureSppmBuffers(false) above when SPPM is off, so the else branch below
      // doesn't need to call it separately.
      const sppmPixelStatsOk = gpu.ensureSppmPixelStatsBuffer(width, height);
      if (!sppmPixelStatsOk || gpu.sppm.sppmPixelStatsBuffer == null) {
        throw new Error(
          '@vitrum/pt-webgpu: causticStrategy="photon-map" requested, but the per-pixel SPPM state buffer is unavailable at the current viewport size.',
        );
      }
      if (gpu.sppm.sppmPhotonPipeline == null) {
        throw new Error('@vitrum/pt-webgpu: causticStrategy="photon-map" requested, but the SPPM photon pipeline is unavailable.');
      }
      // `frameAccumulated` counts iterations including the pending photon pass,
      // so WGSL computes Ne = frameAccumulated × photonCount exactly once.
      // emitted NOW (pre-megakernel), so the completed count INCLUDING this frame
      // is #samplesAccumulated + 1 (incremented at the end of renderFrame).
      // `currentRadius` is still r₀ (the initial search radius); the per-pixel
      // radius stored in sppmPixelStats shrinks progressively — the UBO field is
      // kept for the photon-insertion pass which still needs the global grid cell
      // size (tied to r₀, not the shrinking per-pixel radius).
      gpu.writeSppmStats(
          this.#sppmR0,
          this.#sppmR0,
          this.#samplesAccumulated + 1,
          SPPM_PHOTON_COUNT,
          this.#sppmSceneExtent,
          this.#sppmSceneCenter,
        );
      sppmReady = true;
    } else if (this.#traceTier === 'full') {
      // Ensure placeholder SPPM buffers exist so group-3 bindings 6/7/8/9 are
      // satisfied (the gather is guarded by causticMode() == 2u, so the
      // placeholders are never accessed).
      gpu.ensureSppmBuffers(false);
    }
    return sppmReady;
  }

  /**
   * Per-frame ReSTIR-PT reuse setup (OFF by default). When ON:
   * (re)allocate the two reservoir buffers + result + params, build the reuse
   * pipelines + bind groups, and write RestirPtParams. All gated inside
   * GpuResources, so this is a no-op + allocates nothing when the flag is off
   * (default render untouched). Returns whether the reuse sequence is ready.
   * Extracted from `#ensurePerFrameResources` (T3-B) — behaviour identical.
   */
  #ensureRestirPtPerFrame(gpu: GpuResources, width: number, height: number): boolean {
    let restirPtReady = false;
    if (this.#restirPtReuse) {
      const reservoirBuffersReady = gpu.ensureReservoirBuffers(width, height);
      if (!reservoirBuffersReady) {
        throw new Error(
          '[vitrum/pt-webgpu] ReSTIR-PT was requested but its buffers were not made ready.',
        );
      }
      gpu.ensureReservoirPipelines();
      gpu.writeReservoirParams(this.#restirPtMClamp);
      if (this.#sceneBuffers == null) {
        throw new Error(
          '[vitrum/pt-webgpu] ReSTIR-PT was requested before scene buffers were ready.',
        );
      }
      gpu.buildReservoirBindGroups(this.#sceneBuffers);
      restirPtReady =
        gpu.reservoir.rptReservoirCur != null &&
        gpu.reservoir.rptReservoirPrev != null &&
        gpu.reservoir.rptResultBuffer != null &&
        gpu.reservoir.rptParamsBuffer != null &&
        gpu.reservoir.rptProducerPipeline != null &&
        gpu.reservoir.rptTemporalPipeline != null &&
        gpu.reservoir.rptSpatialPipeline != null &&
        gpu.reservoir.rptResolvePipeline != null &&
        gpu.reservoir.rptCompositePipeline != null &&
        gpu.reservoir.rptProducerGroup0 != null &&
        gpu.reservoir.rptTemporalGroup0 != null &&
        gpu.reservoir.rptSpatialGroup0 != null &&
        gpu.reservoir.rptResolveGroup0 != null &&
        gpu.pathTraceBindGroup1 != null &&
        gpu.pathTraceBindGroup2 != null &&
        gpu.pathTraceBindGroup3 != null;
      if (!restirPtReady) {
        throw new Error(
          '[vitrum/pt-webgpu] ReSTIR-PT requested-mode setup ended in an incomplete state.',
        );
      }
    }
    return restirPtReady;
  }

  /**
   * D8.4 — Encode the conditional compute passes onto `encoder`:
   *   1. SPPM photon-emission pass (before the megakernel, when `sppmReady`).
   *   2. ReSTIR-PT reuse sequence: producer → temporal → spatial → resolve
   *      (when `restirPtReady`).
   *   3. Megakernel (composite when ReSTIR-PT active, otherwise standard).
   *
   * Call order is IDENTICAL to the prior inline body. `encoder` must be a fresh
   * command encoder; the present pass is dispatched separately by the caller.
   */
  #encodePathTracePasses(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    sppmReady: boolean,
    restirPtReady: boolean,
    width: number,
    height: number,
  ): void {
    const gpu = this.#gpu;

    // ── A4-progressive SPPM photon-emission pass (before the megakernel) ────────
    // The photon pass binds the SAME groups 0/1/2/3 as the megakernel.
    // Group 3 carries the SPPM hash-grid buffers at bindings 6/7/8 (+ per-pixel
    // stats at binding 9, read/written by the megakernel's SPPM helpers)
    // alongside the light-tree / material textures (0-5), no group-4 needed.
    //
    // Bucket heads are per-frame state. Clear them before publishing any photon
    // records so the gather can traverse only this iteration's bounded list.
    if (sppmReady && gpu.sppm.sppmPhotonPipeline != null && gpu.sppm.sppmCellCountersBuffer != null) {
      encoder.clearBuffer(gpu.sppm.sppmCellCountersBuffer);
      const photonPass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.sppm.photonPass' });
      photonPass.setPipeline(gpu.sppm.sppmPhotonPipeline);
      photonPass.setBindGroup(0, bindGroup);
      photonPass.setBindGroup(1, gpu.pathTraceBindGroup1);
      photonPass.setBindGroup(2, gpu.pathTraceBindGroup2);
      if (gpu.pathTraceBindGroup3 != null) {
        photonPass.setBindGroup(3, gpu.pathTraceBindGroup3);
      }
      photonPass.dispatchWorkgroups(SPPM_WORKGROUP_COUNT, 1, 1);
      photonPass.end();
    }

    // ── ReSTIR-PT reuse sequence: producer → temporal → spatial → resolve
    //    before the megakernel. Resolve writes both the diagnostics buffer and
    //    the composite contribution consumed by the beauty path. The passes
    //    share the megakernel's group-1/2/3 scene bind groups (same explicit
    //    layouts) + their own group-0 (the reuse-extended one). ──
    if (restirPtReady) {
      this.#encodeRestirPtReusePasses(encoder, gpu, width, height);
    }
    // C35 — t=1 light-subpath-to-camera splats target arbitrary pixels, so the
    // BDPT megakernel atomically stages both its ordinary eye sample and all
    // projected splats into a per-frame RGB buffer. Clear it after the optional
    // producer passes and before either standard/composite megakernel writes it.
    if (this.#bdpt) {
      if (
        gpu.bdptCameraSplatBuffer == null ||
        gpu.bdptCameraSplatResolvePipeline == null
      ) {
        throw new Error(
          '[vitrum/pt-webgpu] bdpt:true camera-splat resources are incomplete before dispatch.',
        );
      }
      encoder.clearBuffer(gpu.bdptCameraSplatBuffer);
    }
    // A1 — when ReSTIR-PT reuse + its composite megakernel are ready, dispatch the
    // COMPOSITE megakernel (E0-direct-only + adds the resolve indirect from
    // rpt_result) bound to the reuse-extended group 0 (which carries rpt_result at
    // the relocated binding 23). Otherwise the default full-path megakernel. The
    // composite path composites the ReSTIR-PT indirect into the BEAUTY accumulator;
    // the default path is byte-identical to pre-A1.
    const useComposite =
      restirPtReady && gpu.reservoir.rptCompositePipeline != null && gpu.reservoir.rptProducerGroup0 != null;
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.pathTrace.pass' });
    if (useComposite) {
      pass.setPipeline(gpu.reservoir.rptCompositePipeline!);
      pass.setBindGroup(0, gpu.reservoir.rptProducerGroup0);
      pass.setBindGroup(1, gpu.pathTraceBindGroup1);
      pass.setBindGroup(2, gpu.pathTraceBindGroup2);
      pass.setBindGroup(3, gpu.pathTraceBindGroup3);
    } else {
      pass.setPipeline(gpu.computePipeline!);
      pass.setBindGroup(0, bindGroup);
      if (this.#traceTier === 'full' && gpu.pathTraceBindGroup1 != null && gpu.pathTraceBindGroup2 != null) {
        pass.setBindGroup(1, gpu.pathTraceBindGroup1);
        pass.setBindGroup(2, gpu.pathTraceBindGroup2);
        if (gpu.pathTraceBindGroup3 != null) {
          // A4 — group 3 now includes SPPM bindings 6/7/8 in addition to the
          // light-tree / material textures (0-5). The gather is guarded by
          // causticMode() == 2u so off-path renders are unaffected.
          pass.setBindGroup(3, gpu.pathTraceBindGroup3);
        }
      }
    }
    pass.dispatchWorkgroups(
      Math.ceil(width / WORKGROUP_SIZE),
      Math.ceil(height / WORKGROUP_SIZE),
      1,
    );
    pass.end();

    if (this.#bdpt) {
      const resolvePass = encoder.beginComputePass({
        label: 'vitrum.pt-webgpu.bdpt.cameraSplatResolve.pass',
      });
      resolvePass.setPipeline(gpu.bdptCameraSplatResolvePipeline!);
      // The resolver uses the ordinary group-0 layout even when the preceding
      // trace used ReSTIR's extended composite group. Both layouts bind the same
      // atomic splat buffer at binding 14.
      resolvePass.setBindGroup(0, bindGroup);
      resolvePass.dispatchWorkgroups(
        Math.ceil(width / WORKGROUP_SIZE),
        Math.ceil(height / WORKGROUP_SIZE),
        1,
      );
      resolvePass.end();
    }
  }

  #handleDenoiserPresentationFailure(error: unknown): void {
    this.#pendingDenoised = null;
    this.#gpu.clearDenoisedResult();
    // The core publishes an accepted result before invoking onComplete. Retire
    // that CPU result too so a converged cadence can start a clean retry.
    this.#postDenoiser?.invalidate();
    this.#emitError({
      kind: 'denoiser',
      message: `[vitrum/pt-webgpu] Could not present OIDN result: ${errorMessage(error)}`,
      fatal: false,
      raw: error,
    });
  }

  /** Consume accepted OIDN output only on the host's explicit render cadence. */
  #preparePostDenoiserPresentation(
    targetSpp: number,
    presentation: PresentationSignature,
    presentationChanged: boolean,
    rePresentCurrent: boolean,
  ): void {
    const denoiser = this.#postDenoiser;

    // A larger target resumes accumulation. Every result from the old terminal
    // sample count is stale, including in-flight work and a retained GPU source.
    if (
      denoiser != null &&
      this.#samplesAccumulated > 0 &&
      this.#samplesAccumulated < targetSpp &&
      (
        this.#pendingDenoised != null ||
        denoiser.isInFlight() ||
        denoiser.getLatestDenoised() != null ||
        this.#gpu.hasDenoisedResult()
      )
    ) {
      this.#pendingDenoised = null;
      denoiser.invalidate();
      this.#gpu.clearDenoisedResult();
    }

    const pending = this.#pendingDenoised;
    if (pending != null) {
      this.#pendingDenoised = null;
      try {
        this.#gpu.presentDenoisedResult(
          pending,
          presentation.tonemapMode,
          presentation.exposure,
          presentation.outputColorSpace,
        );
      } catch (error) {
        this.#handleDenoiserPresentationFailure(error);
      }
      return;
    }

    // Paused and already-converged frames do not execute the ordinary present
    // pass below. Re-run it over whichever retained linear source is current
    // (OIDN when available, otherwise the accumulator) so exposure, tonemap,
    // and output-colour-space changes take effect without another path sample.
    if (presentationChanged && rePresentCurrent) {
      try {
        this.#gpu.presentCurrentResult(
          presentation.tonemapMode,
          presentation.exposure,
          presentation.outputColorSpace,
        );
      } catch (error) {
        this.#handleDenoiserPresentationFailure(error);
      }
    }
  }

  #kickPostDenoiserIfReady(samples: number, width: number, height: number): void {
    const denoiser = this.#postDenoiser;
    const gpu = this.#gpu;
    if (
      denoiser == null ||
      samples <= 0 ||
      denoiser.isInFlight() ||
      denoiser.getLatestDenoised() != null ||
      gpu.accumTexture == null ||
      gpu.albedoTexture == null ||
      gpu.normalDepthTexture == null
    ) return;
    denoiser.kickIfReady(
      this.#device,
      { color: gpu.accumTexture, albedo: gpu.albedoTexture, normalDepth: gpu.normalDepthTexture },
      width,
      height,
    );
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');
    validatePtWebgpuFrameInput(input);
    input = canonicalizeFrameCamera(input, 'PTEngineWebGPU.renderFrame');
    if (this.#bdpt) {
      assertPtWebgpuBdptFrameCameraSupported(input);
    }
    const gpu = this.#gpu;
    if (
      this.#causticStrategy === 'photon-map' &&
      this.#traceTier === 'full' &&
      gpu.sppmWouldExceedCeiling()
    ) {
      throw new Error(
        'vitrum/pt-webgpu: causticStrategy="photon-map" requested, but the SPPM photon grid could not be allocated within the device storage-buffer limits.',
      );
    }
    // Advance the error-throttle frame counter so per-frame GPU errors don't
    // spam the host on every call (see #onUncapturedError throttle logic).
    this.#errorFrameCount++;
    if (!this.#inInverseRender) {
      this.#lastFrameInput = input;
    }
    const frameStartMs = globalThis.performance?.now?.() ?? Date.now();

    const q = input.quality ?? {};
    const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
    // Track the effective regime before the paused-frame fast-out. Inverse
    // sessions re-render the most recently supplied FrameInput even when that
    // input arrived while paused, so replay preflight must inspect this same
    // bounce count rather than the last count that happened to dispatch.
    this.#activeBounces = Math.min(
      q.bounces ?? this.#maxBouncesLimit,
      this.#maxBouncesLimit,
    );
    const presentation = presentationSignatureFor(input);
    const presentationChanged = this.#presentationSignature != null &&
      !samePresentationSignature(this.#presentationSignature, presentation);
    this.#presentationSignature = presentation;

    if (this.#slot.get() === 'paused' && !this.#inInverseRender) {
      const accumTexture = gpu.accumTexture;
      if (accumTexture == null) {
        return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      }
      this.#preparePostDenoiserPresentation(
        targetSpp, presentation, presentationChanged, true,
      );
      const isConverged = this.#samplesAccumulated >= targetSpp;
      if (isConverged) {
        this.#kickPostDenoiserIfReady(this.#samplesAccumulated, gpu.accumWidth, gpu.accumHeight);
      }
      const output = this.#frameOutput(
        accumTexture,
        this.#samplesAccumulated,
        isConverged,
      );
      this.#emitFrameTelemetry(frameStartMs, 0, this.#samplesAccumulated, targetSpp);
      return output;
    }

    const resolution = q.resolutionFactor ?? 1;
    const width = Math.max(1, Math.floor(input.viewport.width * resolution));
    const height = Math.max(1, Math.floor(input.viewport.height * resolution));

    // A recreate resets the sample accumulator (the prior inline
    // #ensureAccumResources set #samplesAccumulated = 0 here; that one piece of
    // engine state is reported back rather than mutated inside GpuResources).
    if (gpu.ensureAccumResources(width, height)) {
      this.#samplesAccumulated = 0;
      this.#pendingDenoised = null;
      this.#postDenoiser?.invalidate();
    }
    gpu.ensurePipeline();
    // D8.5 — The 12-way GPU-resource null-guard is codified in isReadyToRender()
    // (gpuResources.ts); the 13th condition (scene buffers) lives on the engine
    // because GpuResources has no visibility into #sceneBuffers.  Both must pass
    // before dispatching — identical semantics to the prior open-coded check.
    if (!gpu.isReadyToRender() || this.#sceneBuffers == null) {
      throw new Error('renderFrame: failed to initialize WebGPU pipeline resources');
    }

    const accumTexture = gpu.accumTexture;
    const alreadyConverged =
      accumTexture != null && this.#samplesAccumulated >= targetSpp;
    this.#preparePostDenoiserPresentation(
      targetSpp, presentation, presentationChanged, alreadyConverged,
    );
    if (alreadyConverged) {
      const output = this.#frameOutput(accumTexture, this.#samplesAccumulated, true);
      this.#emitFrameTelemetry(frameStartMs, 0, this.#samplesAccumulated, targetSpp);
      this.#kickPostDenoiserIfReady(this.#samplesAccumulated, width, height);
      return output;
    }

    const { bindGroup, sppmReady, restirPtReady } =
      this.#ensurePerFrameResources(input, width, height);

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.pathTrace.encoder' });

    this.#encodePathTracePasses(encoder, bindGroup, sppmReady, restirPtReady, width, height);

    // ── Present pass: tonemap / exposure / outputColorSpace ───────────────────
    // Reads the just-written accumTexture (running-mean linear HDR) and writes
    // the tonemapped + OETF-encoded result to presentTexture. Both textures are
    // created in ensureAccumResources and have TEXTURE_BINDING | STORAGE_BINDING
    // usages. The present pass always runs (aces@1.0@srgb by default); hosts
    // that want raw linear output can set tonemap:'none' + outputColorSpace:'linear'.
    // Adjoint/OIDN readbacks always use accumTexture (not presentTexture).
    if (gpu.present.presentTexture != null) {
      gpu.ensurePresentPipeline();
      gpu.writePresentParams(
        presentation.tonemapMode,
        presentation.exposure,
        presentation.outputColorSpace,
      );
      gpu.dispatchPresentPass(encoder, width, height);
    }

    this.#device.queue.submit([encoder.finish()]);

    this.#samplesAccumulated = Math.min(this.#samplesAccumulated + 1, this.#maxSamplesLimit);
    const accumTexturePost = gpu.accumTexture;
    if (accumTexturePost == null) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }
    const isConverged = this.#samplesAccumulated >= targetSpp;
    if (isConverged) {
      this.#kickPostDenoiserIfReady(this.#samplesAccumulated, width, height);
    }

    const output = this.#frameOutput(accumTexturePost, this.#samplesAccumulated, isConverged);
    this.#emitFrameTelemetry(frameStartMs, 1, this.#samplesAccumulated, targetSpp);
    return output;
  }

  /**
   * Encode the ReSTIR-PT reuse sequence (producer → temporal → spatial → resolve)
   * onto the supplied command encoder. Each pass binds the reuse-extended group 0
   * plus the megakernel's group-1/2/3 scene bind groups (built against the SAME
   * explicit layouts the reuse pipeline layout reuses — so they set cleanly on the
   * reuse pipelines). Preconditions are checked by the caller (`restirPtReady`).
   *
   * Ordering: producer fills `Cur` from a fresh primary+1-bounce trace; temporal
   * fuses the reprojected `Prev` history into `Cur` in place; spatial folds 5 disc
   * neighbours of `Cur` (full GBH) into `Spatial`; resolve reconstructs the
   * reconnection-indirect from `Spatial` into `rpt_result`. Four sequential compute
   * passes on one encoder — each `pass.end()` is an execution barrier, so the
   * producer's writes to `Cur` are visible to the temporal pass and so on. The
   * COMPOSITE megakernel (dispatched after) reads `rpt_result` and composites the
   * indirect into the beauty accumulator.
   */
  #encodeRestirPtReusePasses(
    encoder: GPUCommandEncoder,
    gpu: GpuResources,
    width: number,
    height: number,
  ): void {
    const g1 = gpu.pathTraceBindGroup1!;
    const g2 = gpu.pathTraceBindGroup2!;
    const g3 = gpu.pathTraceBindGroup3!;
    const wgX = Math.ceil(width / WORKGROUP_SIZE);
    const wgY = Math.ceil(height / WORKGROUP_SIZE);
    const dispatch = (
      label: string,
      pipeline: GPUComputePipeline,
      group0: GPUBindGroup,
    ): void => {
      const p = encoder.beginComputePass({ label });
      p.setPipeline(pipeline);
      p.setBindGroup(0, group0);
      // Groups 1/2/3 carry the scene/TLAS/light/material bindings the producer's
      // and temporal's trace statically use (resolve binds them too — its layout
      // declares them; extra unused groups are legal). Bound on every reuse pass
      // so the shared 4-group pipeline layout validates.
      p.setBindGroup(1, g1);
      p.setBindGroup(2, g2);
      p.setBindGroup(3, g3);
      p.dispatchWorkgroups(wgX, wgY, 1);
      p.end();
    };
    dispatch('vitrum.pt-webgpu.restirPt.produce', gpu.reservoir.rptProducerPipeline!, gpu.reservoir.rptProducerGroup0!);
    dispatch('vitrum.pt-webgpu.restirPt.temporal', gpu.reservoir.rptTemporalPipeline!, gpu.reservoir.rptTemporalGroup0!);
    dispatch('vitrum.pt-webgpu.restirPt.spatial', gpu.reservoir.rptSpatialPipeline!, gpu.reservoir.rptSpatialGroup0!);
    dispatch('vitrum.pt-webgpu.restirPt.resolve', gpu.reservoir.rptResolvePipeline!, gpu.reservoir.rptResolveGroup0!);
  }

  /**
   * Capture the engine's rendered output as a host-side CPU Float32 RGBA image.
   *
   * `colorSpace:'linear'` (default) reads `accumTexture` — the rgba16float
   * texture written by `accumulateFrame` that stores the per-pixel Welford
   * running mean (accum.xyz / accum.w).  This is linear-light HDR radiance in
   * scene units, identical to what OIDN / the present pass consumes.
   *
   * `colorSpace:'output'` reads `presentTexture` — the rgba16float texture
   * written by the present pass (tonemap + OETF).
   *
   * Returns `null` before the first frame (accumTexture not yet allocated).
   * Pipeline stall: submits a GPU→CPU copyTextureToBuffer + mapAsync; use
   * for debugging/export, not per-frame readback.
   */
  async captureFrame(opts?: CaptureFrameOptions): Promise<CapturedFrame | null> {
    this.#assertUsable('captureFrame');
    const colorSpace = opts?.colorSpace ?? 'linear';
    const gpu = this.#gpu;
    const texture = colorSpace === 'output' ? gpu.present.presentTexture : gpu.accumTexture;
    if (texture == null) return null;
    const width = gpu.accumWidth;
    const height = gpu.accumHeight;
    if (width <= 0 || height <= 0) return null;
    const rgba = await readRgba16fTextureToF32(this.#device, texture, width, height);
    if (rgba == null) return null;
    return { width, height, rgba };
  }

  /**
   * ReSTIR-PT reconnection-indirect result buffer (one vec4f /
   * full-res pixel: .rgb = reconnection indirect HDR, .a = contributing flag), or
   * `null` when the engine was not built with `restirPtReuse: true` or no frame
   * has rendered yet. This is a SEPARATE debug output. When the composite pipeline
   * is active (`rptCompositePipeline` + `rptProducerGroup0` are both non-null), the
   * composite megakernel folds the ReSTIR-PT resolve into the beauty accumulator;
   * the beauty image already contains the composited result. Hosts read this buffer
   * back via `copyBufferToBuffer` → a MAP_READ staging buffer (the buffer carries
   * `COPY_SRC`).
   *
   * Available only when `capabilities.activeFeatures` has
   * `'pt-webgpu-one-edge-gris-reconnection'`.
   */
  getRestirPtResultBuffer(): GPUBuffer | null {
    this.#assertUsable('getRestirPtResultBuffer');
    return this.#restirPtReuse ? this.#gpu.reservoir.rptResultBuffer : null;
  }

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    if (this.#slot.get() === 'error') {
      throw new Error('reset: engine is in a fatal error state');
    }
    // Submit all GPU clears before publishing the host-side reset. A synchronous
    // encoder/submit failure therefore leaves the observable reset state intact.
    this.#gpu.clearTemporalBuffers();
    this.#pendingDenoised = null;
    this.#gpu.clearDenoisedResult();
    this.#postDenoiser?.invalidate();
    this.#presentationSignature = null;
    this.#samplesAccumulated = 0;
  }

  /**
   * Progressive walkaround→PT handoff (P8): seed the accumulator with an initial
   * image (typically the real-time engine's last frame) so a freshly-still camera
   * shows a plausible picture immediately instead of a 1-sample blizzard — WITHOUT
   * biasing the converged result.
   *
   * `seed` is injected as a DECAYING PRIOR of virtual weight `opts.weight`: the
   * accum buffers become `accumBuffer = (seedRGB·W, W)` (+ matching variance
   * moments). After `M` real samples land, the displayed mean is
   * `μ + W/(W+M)·(seedRGB − μ)`, so the seed's influence W/(W+M) → 0 and the
   * CONVERGED mean is exactly the no-seed result `μ` for ANY seed (see
   * seedBlit.wgsl.ts for the derivation).
   *
   * Crucially this does NOT advance `#samplesAccumulated`: `weight` is a
   * virtual-sample prior, NOT a real SPP — keeping them separate is precisely
   * what makes the converged-mean math hold (and what stops convergence /
   * telemetry from over-reporting).
   *
   * `opts.width`/`opts.height` are the accum (destination) dims — typically the
   * converged engine's render size for this view. `seed` may be any size (it is
   * bilinearly resampled). The accum buffers are (re)allocated to these dims and
   * CLEARED before the seed is written, so the seed is the sole prior regardless
   * of any prior accumulation. Call this BEFORE the first `renderFrame` of a
   * still cohort (the host's handoff coordinator does: `reset()` →
   * `seedAccumulator()` → accumulate).
   *
   * Available only when `capabilities.supportsAccumulatorSeed === true`; hosts
   * MUST typeof-check before calling.
   */
  seedAccumulator(
    seed: BackendTexture,
    opts: { weight: number; width: number; height: number },
  ): void {
    this.#assertLive('seedAccumulator');
    const width = Math.max(1, Math.floor(opts.width));
    const height = Math.max(1, Math.floor(opts.height));
    const seedTex = narrowToBackendTexture<'webgpu', GPUTexture>(seed);
    if (seedTex == null) {
      throw new Error('seedAccumulator: seed texture is null/undefined.');
    }
    // Ensure the accum buffers exist at the seed dims. A (re)allocation clears
    // them; a cache-hit at the same dims does NOT, so clear explicitly in that
    // case — the seed must land on a zeroed accumulator to be the sole prior.
    // Either way `#samplesAccumulated` is reset to 0: the prior is the new
    // starting point, with NO real samples yet (the prior weight W is virtual).
    this.#gpu.ensureAccumResources(width, height);
    // A seed starts a new temporal cohort even when its dimensions forced fresh
    // accumulation buffers. Clear every other retained history (ReSTIR-PT and
    // SPPM included) before publishing the prior.
    this.#gpu.clearTemporalBuffers();
    this.#samplesAccumulated = 0;
    // Write the decaying prior. `weight` (virtual samples) is deliberately NOT
    // added to `#samplesAccumulated` — see the method doc.
    this.#gpu.seedAccumBuffer(seedTex, opts.weight, width, height);
    this.#pendingDenoised = null;
    this.#gpu.clearDenoisedResult();
    this.#postDenoiser?.invalidate();
    this.#presentationSignature = null;
  }

  /**
   * OIDN denoised RGB for the current converged cohort (WG-1), or null while
   * inference is in flight / engine was not built with `denoiser: 'oidn-final'`.
   */
  getDenoisedFrame(): DenoisedFrame | null {
    return this.#postDenoiser?.getLatestDenoised() ?? null;
  }

  /** Read back the retained canonical core {@link Scene} (the UNFILTERED scene
   *  passed to the most recent {@link setScene} call), or null before the first
   *  `setScene` — and also after {@link dispose}, which drops `#scene`.
   *  The scene is stored as-is by {@link #repackScene}; no capability filtering is
   *  applied to the returned object. Implements the optional `Engine.getScene`
   *  contract — see its JSDoc for the no-defensive-copy / frozen-by-contract
   *  semantics. */
  getScene(): Scene | null {
    return this.#scene;
  }

  // ── Inverse rendering ─────────────────────────────────────────────────────
  //
  // Explicit finite difference supports the full optimizable parameter set.
  // The separate path-replay method is certified only for one-bounce RGB,
  // opaque triangle visibility and unlit material emissive RGB. Both use the
  // same frozen seed sequence; the host owns the session cadence.
  createInverseSession(opts: InverseSessionOptions): InverseSession {
    this.#assertLive('createInverseSession');
    if (this.#lastFrameInput == null) {
      throw new Error(
        'createInverseSession: call renderFrame() at least once before opening an ' +
          'inverse session — the session re-renders the most-recent camera view.',
      );
    }
    const hooks: InverseEngineHooks = {
      getScene: () => this.#scene!,
      renderAndReadback: (width, height, samples) =>
        this.#renderAndReadbackForInverse(width, height, samples),
      patchMaterial: (primitiveId: string, patch: Partial<MaterialSpec>) => {
        this.updatePrimitive(primitiveId, { material: patch });
      },
      patchEmitter: (emitterId: string, patch: Partial<SceneEmitter>) => {
        this.updateEmitter(emitterId, patch);
      },
      getPathReplayRenderContext: () => ({
        bounces: this.#activeBounces,
        spectral: this.#spectralEnabled,
        bdpt: this.#bdpt && this.#traceTier === 'full',
        restirPtReuse: this.#restirPtReuse,
        causticStrategy: this.#traceTier === 'lite' ? 'none' : this.#causticStrategy,
        cameraVisibleEmitters: this.#cameraVisibleEmitters,
      }),
      getMaterialSupportDetails: () =>
        ptWebgpuSupportManifest(this.#traceTier).materials,
      getEmitterSupportDetails: () =>
        ptWebgpuSupportManifest(this.#traceTier).emitters,
      ...(this.#traceTier === 'full'
        ? {
            computeAdjointGradient: (req: AdjointGradientRequest) =>
              this.#computeAdjointGradient(req),
          }
        : {}),
    };
    return new PtWebgpuInverseSession(hooks, opts);
  }

  /**
   * Certified one-bounce emissive path replay. AdjointPass owns the exact
   * triangle visibility replay, analytic emissive derivative, transient-buffer
   * lifecycle, dispatch, and readback. Every other field/transport/geometry
   * domain is rejected before this private hook is reached.
   */
  async #computeAdjointGradient(req: AdjointGradientRequest): Promise<Float32Array> {
    this.#assertLive('computeAdjointGradient');
    const sb = this.#sceneBuffers;
    const last = this.#lastFrameInput;
    if (sb == null || last == null || this.#scene == null) {
      throw new Error('computeAdjointGradient: no scene/camera (render at least one frame first).');
    }
    if (this.#adjointPass == null) {
      this.#adjointPass = new AdjointPass(this.#device, this.#sampling);
    }
    return this.#adjointPass.computeGradient(
      req,
      sb,
      last,
      this.#scene,
      materialIndexForPrimitive,
      this.#cameraVisibleEmitters,
    );
  }

  /**
   * Render `samples` accumulated SPP at exactly `width × height` with a FROZEN
   * RNG seed (so two renders that differ only in a perturbed material/emitter
   * parameter share their random choices — the finite-difference + path-replay
   * requirement), then read the accum texture back as interleaved RGB float.
   *
   * Resets accumulation first, builds a synthetic FrameInput from the cached
   * camera (viewport forced to the target dims, resolutionFactor 1, a fixed
   * seed/index), and submits `samples` single-SPP dispatches via the normal
   * renderFrame path. Reuses `readOidnInputsFromTextures` for the GPU→CPU copy.
   */
  async #renderAndReadbackForInverse(
    width: number,
    height: number,
    samples: number,
  ): Promise<{ rgb: Float32Array; channels: 3 | 4 }> {
    const last = this.#lastFrameInput!;
    // FROZEN seed SEQUENCE — the per-sample seed sequence is identical every
    // inverse render (baseline and every ±ε probe), so the ONLY thing that
    // changes between two renders is the perturbed scene parameter (path
    // replay's frozen-RNG discipline). We still vary the seed PER SAMPLE
    // (FROZEN_SEED_BASE + s) so the `samples` average reduces Monte-Carlo
    // variance rather than re-tracing one identical path; the sequence itself
    // is deterministic and reproduced bit-for-bit on the probe render.
    const FROZEN_SEED_BASE = 0x5eed5eed;
    const FROZEN_INDEX = 0;
    this.#inInverseRender = true;
    try {
      this.reset();
      for (let s = 0; s < samples; s++) {
        const frame: FrameInput = {
          ...last,
          frameIndex: FROZEN_INDEX,
          frameSeed: (FROZEN_SEED_BASE + s) >>> 0,
          viewport: { width, height, devicePixelRatio: 1 },
          quality: {
            ...(last.quality ?? {}),
            samplesTarget: samples,
            resolutionFactor: 1,
          },
        };
        this.renderFrame(frame);
      }
    } finally {
      this.#inInverseRender = false;
    }
    const accum = this.#gpu.accumTexture;
    if (accum == null) {
      return { rgb: new Float32Array(width * height * 3), channels: 3 };
    }
    const result = await readOidnInputsFromTextures(
      this.#device,
      { color: accum },
      width,
      height,
    );
    return { rgb: result.color, channels: 3 };
  }

  pause(): void {
    this.#assertUsable('pause');
    this.#slot.set('paused');
  }

  resume(): void {
    this.#assertUsable('resume');
    this.#slot.set('ready');
  }

  dispose(): void {
    if (this.#slot.get() === 'disposed') return;
    // Remove GPU error listener before tearing down resources so the handler
    // never fires after dispose (the device is no longer ours to observe).
    this.#device.removeEventListener('uncapturederror', this.#onUncapturedError);
    this.#pendingDenoised = null;
    this.#presentationSignature = null;
    this.#postDenoiser?.dispose();
    // Tears down accum textures + buffers, the cached bind groups, and destroys
    // + nulls the params buffer / pipeline / layout (same order as the prior
    // inline dispose body). Scene buffers stay engine-owned, destroyed below.
    this.#gpu.dispose();
    this.#sceneBuffers?.destroy();
    this.#sceneBuffers = null;
    this.#adjointPass?.dispose();
    this.#adjointPass = null;
    this.#scene = null;
    this.#geoPack = null;
    this.#onFrameSubs.clear();
    this.#onProgressSubs.clear();
    this.#onErrorSubs.clear();
    this.#onWarningSubs.clear();
    this.#errorThrottleMap.clear();
    this.#slot.set('disposed');
  }

  onFrame(cb: (stats: FrameStats) => void): () => void {
    this.#onFrameSubs.add(cb);
    return () => {
      this.#onFrameSubs.delete(cb);
    };
  }

  onProgress(cb: (progress: ProgressStats) => void): () => void {
    this.#onProgressSubs.add(cb);
    return () => {
      this.#onProgressSubs.delete(cb);
    };
  }

  onError(cb: (error: EngineError) => void): () => void {
    this.#onErrorSubs.add(cb);
    return () => {
      this.#onErrorSubs.delete(cb);
    };
  }

  onWarning(cb: (warning: EngineWarning) => void): () => void {
    this.#onWarningSubs.add(cb);
    return () => {
      this.#onWarningSubs.delete(cb);
    };
  }
}

/**
 * The WebGPU path-tracer backend's STABLE, public-facing surface BEYOND the
 * host-agnostic {@link Engine} contract. {@link createPTEngine_WebGPU} returns
 * `Promise<Engine & PTEngineWebGPUSurface>` so a host that deliberately picks
 * this backend by name gets the extra methods typed — without `createEngine`
 * (the erased facade) having to leak backend specifics into the universal
 * contract.
 *
 * Deliberately minimal: only the converged-denoise read-back and resolved profile
 * are backend-specific public API. Debug GPU-buffer accessors remain on the
 * universal engine contract for diagnostics; this surface adds no raw handles.
 */
export interface PTEngineWebGPUSurface {
  /** Resolved adapter profile; profile selection is independent of active features. */
  readonly backendProfileId: 'pt-webgpu' | 'pt-webgpu-lite';
  /** The most recently completed OIDN-denoised RGB image for the current
   *  converged cohort, or null when the engine was not built with
   *  `denoiser: 'oidn-final'` / inference is still in flight. */
  getDenoisedFrame(): DenoisedFrame | null;
}

export const createPTEngine_WebGPU: EngineFactory<
  PTEngineWebGPUOptions,
  Engine & PTEngineWebGPUSurface
> = async (
  opts: PTEngineWebGPUOptions,
// eslint-disable-next-line @typescript-eslint/require-await -- factory signature is async to match EngineFactory<…> contract; no async setup needed in this code path
): Promise<Engine & PTEngineWebGPUSurface> => {
  const { traceTier, effectiveOpts } = validatePtWebgpuOptions(opts);
  const slot = makeStateSlot();
  const engine = new PTEngineWebGPU(effectiveOpts, slot, traceTier);
  slot.set('ready');
  return engine;
};
