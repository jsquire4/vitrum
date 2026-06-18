import type {
  BackendTexture,
  BackendSupportMode,
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
} from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, MATERIAL_SPEC_FIELDS, asBackendTexture, narrowToBackendTexture } from '@vitrum/core';
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
import {
  buildPackedScene,
  scenePackResultFromPacked,
  uploadPackedScene,
  PT_WEBGPU_ANALYTIC_SHAPES,
  PT_WEBGPU_SUPPORT,
  type UploadedSceneBuffers,
} from './scene/uploadSceneBuffers.js';
import {
  packLiteLightTexture,
  packLiteEnvTexture,
  packLiteEnvCdfTexture,
} from './scene/litePackedTextures.js';
import { type ScenePackResult, pickPrimitiveCpu, type PickCamera } from '@vitrum/shared-bvh';
import { FrameParamsSlot } from './scene/frameParamsLayout.js';
import { packFrameParams } from './frameParamsPacker.js';
import { SceneMutationRouter } from './sceneMutationRouter.js';
import { resolvePtWebgpuTraceTier, type PtWebgpuTraceTier } from './traceTier.js';
import { GpuResources } from './gpuResources.js';
import {
  OIDNFinalDispatcher,
  type DenoisedFrame,
} from './denoise/oidnFinalDispatcher.js';
import {
  BdptLightPathBufferWebGPU,
  createBdptLightPathPlaceholder,
} from './bdpt/bdptLightPathBufferWebGPU.js';
import { AdjointPass } from './adjointPass.js';
import { PT_WEBGPU_COMMON_WGSL } from './wgsl/common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  OCTAHEDRAL_CORE_WGSL,
  TONEMAP_MODE_INDEX,
} from '@vitrum/shared-samplers';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from './webgpuLimits.js';
import {
  sppmInitialRadius,
} from './sppmParams.js';

export { PT_WEBGPU_COMMON_WGSL, HAMMERSLEY_WGSL, OCTAHEDRAL_CORE_WGSL };
export {
  PT_WEBGPU_REQUIRED_LIMITS,
  PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  // Full-tier device-limit floors — re-exported so `@vitrum/engine`'s
  // progressive-engine facade can compute the cross-backend limit UNION
  // (max of the hybrid + pt-webgpu full floors) for a shared device.
  PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
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
export {
  BdptLightPathBufferWebGPU,
  type BdptLightPathBufferWebGPUOptions,
} from './bdpt/bdptLightPathBufferWebGPU.js';

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

function collectUnsupportedLayerNormalFields(
  fields: Set<string>,
  prefix: 'frontLayer' | 'backLayer',
  layer: MaterialSpec['frontLayer'] | MaterialSpec['backLayer'] | undefined,
): void {
  if (layer?.normalMap != null) fields.add(`${prefix}.normalMap`);
  if (layer?.normalScale != null) fields.add(`${prefix}.normalScale`);
}

function collectUnsupportedMaterialFieldUses(
  scene: Scene,
  traceTier: PtWebgpuTraceTier,
): UnsupportedMaterialFieldUse[] {
  const unsupportedFields = traceTier === 'lite'
    ? PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS
    : UNSUPPORTED_MATERIAL_FIELDS;
  const uses: UnsupportedMaterialFieldUse[] = [];
  for (const primitive of scene.primitives) {
    const material = (primitive as { readonly material?: Partial<MaterialSpec> }).material;
    if (material == null) continue;
    const fields = new Set<string>();
    for (const field of unsupportedFields) {
      if (material[field] != null) fields.add(field);
    }
    if (traceTier === 'lite') {
      collectUnsupportedLayerNormalFields(fields, 'frontLayer', material.frontLayer);
      collectUnsupportedLayerNormalFields(fields, 'backLayer', material.backLayer);
    }
    if (fields.size > 0) {
      uses.push({ primitiveId: primitive.id, fields: Array.from(fields).sort() });
    }
  }
  return uses;
}

// CAP-01 — the remaining material fields this backend silently drops, derived
// from the ledger's per-field support matrix so warning + capability rows can
// never drift. Displacement keeps its own dedicated warning; `extensions` is
// the contract-sanctioned host-discretionary escape hatch (no warning).
const UNSUPPORTED_MATERIAL_FIELDS: readonly (keyof MaterialSpec)[] = MATERIAL_SPEC_FIELDS.filter(
  (field) =>
    BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials[field] === 'unsupported' &&
    !(UNSUPPORTED_DISPLACEMENT_MATERIAL_FIELDS as readonly string[]).includes(field) &&
    field !== 'extensions',
);

const PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS = [
  // The lite trace shader composes no full-tier group-3 material texture
  // bindings. These fields are therefore unrendered on lite even when the full
  // pt-webgpu tier supports them.
  'baseColorMap',
  'normalMap',
  'normalScale',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'aoMapIntensity',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'clearcoatNormalScale',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'bumpScale',
  'lightMap',
  'lightMapIntensity',
  // Lite also omits the full-tier alpha-test, per-material environment scale,
  // and anisotropic-BSDF routes.
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'envMapIntensity',
  'anisotropy',
  'anisotropyRotation',
] as const satisfies readonly (keyof MaterialSpec)[];

const PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS = Object.freeze([
  ...new Set([
    ...UNSUPPORTED_MATERIAL_FIELDS,
    ...PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS,
  ]),
]);

const PT_WEBGPU_LITE_MATERIALS = Object.freeze({
  ...BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials,
  ...Object.fromEntries(
    PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS.map((field) =>
      [field, 'unsupported' as BackendSupportMode],
    ),
  ),
});

function liteCanBakeVertexColors(primitive: ScenePrimitive): boolean {
  const colors = (primitive as { readonly colors?: Float32Array }).colors;
  const positions = (primitive as { readonly positions?: Float32Array }).positions;
  if (colors == null || colors.length === 0) return true;
  if (positions == null || positions.length === 0) return false;
  const vertexCount = Math.floor(positions.length / 3);
  const stride = colors.length >= vertexCount * 4
    ? 4
    : colors.length >= vertexCount * 3
      ? 3
      : 0;
  if (vertexCount === 0 || stride === 0) return false;
  const r = colors[0] ?? 1;
  const g = colors[1] ?? 1;
  const b = colors[2] ?? 1;
  const eps = 1e-6;
  for (let i = 0; i < vertexCount; i += 1) {
    const o = i * stride;
    if (
      Math.abs((colors[o] ?? 1) - r) > eps ||
      Math.abs((colors[o + 1] ?? 1) - g) > eps ||
      Math.abs((colors[o + 2] ?? 1) - b) > eps
    ) {
      return false;
    }
    if (stride === 4 && Math.abs((colors[o + 3] ?? 1) - 1) > eps) {
      return false;
    }
  }
  return true;
}

function collectLiteUnsupportedVertexColorPrimitiveIds(scene: Scene): string[] {
  const ids: string[] = [];
  for (const primitive of scene.primitives) {
    const colors = (primitive as { readonly colors?: Float32Array }).colors;
    if (colors != null && colors.length > 0 && !liteCanBakeVertexColors(primitive)) ids.push(primitive.id);
  }
  return ids.sort();
}

export interface PTEngineWebGPUOptions extends EngineOptions {
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
   * Enable the bidirectional path tracer (full Veach §10.3 connection MIS). Full
   * tier only. Off by default. (Graduated from `extensions['vitrum.ptWebgpu.bdpt']`.)
   */
  readonly bdpt?: boolean;
  /**
   * Enable the EXPERIMENTAL ReSTIR-PT reservoir/reuse pre-passes (GRIS hero-stack
   * temporal reconnection reuse — Lin 2022). Full tier only. **OFF by default.**
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
   * variance A/Bs are V28 queue entries (see road-to-100 A1).
   */
  readonly restirPtReuse?: boolean;
  /** ReSTIR-PT reuse tuning — read only when {@link restirPtReuse} is `true`. */
  readonly restirPtReuseOptions?: {
    /** Temporal M-clamp (history confidence cap). Default 20. */
    readonly mClamp?: number;
    /** GRIS W-cap (temporal-feedback gain bound, the V19 grison guard). Default 10. */
    readonly wCap?: number;
  };
  /** BDPT tuning — read only when {@link bdpt} is `true`. */
  readonly bdptOptions?: {
    /**
     * Max light-subpath bounces, clamped 1–8 with a construction warning. Default 1
     * keeps `bdpt:true` endpoint-only and radiometrically neutral against the
     * unidirectional estimator; values >1 opt into the current multi-vertex research
     * path and emit a warning until full all-strategy weighting lands.
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

const EXPERIMENTAL_MAX_BOUNCES = 8;
const BDPT_MAX_LIGHT_BOUNCES = 8;
const BDPT_SAFE_DEFAULT_LIGHT_BOUNCES = 1;
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const WORKGROUP_SIZE = 8;
/** A4 — SPPM photons emitted per frame: 65536 (= 1024 workgroups × 64 lanes). */
const SPPM_PHOTON_COUNT = 65536;
const SPPM_WORKGROUP_COUNT = Math.ceil(SPPM_PHOTON_COUNT / 64);

// D8.14 — Lite-tier capability sets (derived from PT_WEBGPU_SUPPORT diff).
// Sourced from shared constants rather than inline literals so any change to the
// full-tier support set is the single update point.
/** Emitter kinds supported by the lite tier via texture packing (B12). */
const PT_WEBGPU_LITE_SUPPORTED_EMITTER_KINDS: ReadonlySet<import('@vitrum/core').SceneEmitter['kind']> =
  new Set(['directional', 'point', 'spot', 'rect-area', 'disc-area']);
/** Emitter kinds in the full-tier set but NOT in the lite tier (no NEE path in lite kernel). */
const PT_WEBGPU_LITE_UNSUPPORTED_EMITTER_KINDS: ReadonlySet<import('@vitrum/core').SceneEmitter['kind']> =
  new Set(
    [...PT_WEBGPU_SUPPORT.supportedEmitterKinds].filter(
      (k) => !PT_WEBGPU_LITE_SUPPORTED_EMITTER_KINDS.has(k),
    ),
  );

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

function emitPteWarning(
  opts: Pick<PTEngineWebGPUOptions, 'onWarning'>,
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

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function resolveBdptMaxLightBounces(requested: number | undefined): number {
  if (requested === undefined) return BDPT_SAFE_DEFAULT_LIGHT_BOUNCES;
  return Math.min(BDPT_MAX_LIGHT_BOUNCES, Math.floor(requested));
}

class PTEngineWebGPU implements Engine {
  readonly #slot: StateSlot;
  readonly #device: GPUDevice;
  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;
  readonly #traceTier: PtWebgpuTraceTier;

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
  readonly #onUncapturedError: (event: Event) => void;
  // device.lost is a Promise<GPUDeviceLostInfo> — we store no reference to it but
  // we attach a .then handler that routes into #emitError.  The handler is
  // removed implicitly when the Promise settles (Promises don't hold listener refs).
  readonly #postDenoiser: OIDNFinalDispatcher | null;
  readonly #spectralEnabled: boolean;
  readonly #lightTreeImportanceSampling: boolean;
  readonly #cameraVisibleEmitters: boolean;
  readonly #bdpt: boolean;
  readonly #bdptMaxLightBounces: number;
  #bdptLightPath: BdptLightPathBufferWebGPU | null = null;
  #bdptExternalBuffer: GPUBuffer | null = null;
  #bdptPlaceholderBuffer: GPUBuffer | null = null;
  /** EXPERIMENTAL ReSTIR-PT reuse — compile-time + full-tier; OFF by default. */
  readonly #restirPtReuse: boolean;
  readonly #restirPtMClamp: number;
  readonly #restirPtWCap: number;

  // ── SPPM state (A4-progressive, photon-map strategy) ─────────────────────
  /** Cached initial radius r₀ = max(diagonal/100, 1e-3) from the scene AABB.
   *  Recomputed on every setScene.  Used as the INITIAL per-pixel R² seed
   *  (r₀²) in sppmGatherProgressive; the per-pixel radius then shrinks
   *  progressively via the Hachisuka update rule (A4-progressive). */
  #sppmR0 = 0.017; // 1.7 cm — a safe pre-setScene default (1 m Cornell box)
  /** Half-diagonal of the scene used for the directional-light disk emitter. */
  #sppmSceneExtent = 10.0; // world units; refreshed on setScene

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

  constructor(opts: PTEngineWebGPUOptions, slot: StateSlot, traceTier: PtWebgpuTraceTier) {
    this.#slot = slot;
    this.#device = opts.device;
    if (opts.onWarning != null) {
      this.#onWarningSubs.add(opts.onWarning);
    }
    this.#spectralEnabled = opts.spectral === true;
    this.#lightTreeImportanceSampling = opts.lightTreeImportanceSampling !== false;
    this.#cameraVisibleEmitters = opts.cameraVisibleEmitters !== false;
    this.#traceTier = traceTier;
    this.#maxBouncesLimit = Math.max(1, Math.min(opts.maxBounces ?? 3, EXPERIMENTAL_MAX_BOUNCES));
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    const causticOpts = opts.causticOptions ?? {};
    const mneeIter =
      typeof causticOpts.mneeMaxIterations === 'number' ? causticOpts.mneeMaxIterations : 8;
    const mneeChain =
      typeof causticOpts.mneeMaxChainLength === 'number' ? causticOpts.mneeMaxChainLength : 3;
    this.#mneeMaxIterations = Math.max(1, mneeIter);
    this.#mneeMaxChainLength = Math.max(1, mneeChain);
    // Warn on unknown causticOptions keys — the contract requires backends to
    // surface unrecognised keys rather than silently ignoring them.
    if (opts.causticOptions != null) {
      const knownCausticKeys = new Set(['mneeMaxIterations', 'mneeMaxChainLength']);
      const unknownCausticKeys = Object.keys(opts.causticOptions).filter(
        (k) => !knownCausticKeys.has(k),
      );
      if (unknownCausticKeys.length > 0) {
        this.#warn({
          code: 'pt-webgpu.unknown-caustic-options',
          backend: 'pt-webgpu',
          phase: 'construction',
          method: 'createPTEngine_WebGPU',
          message:
            `[vitrum/pt-webgpu] Unknown causticOptions keys will be ignored: ${unknownCausticKeys.map((k) => JSON.stringify(k)).join(', ')}. ` +
            'The recognised keys are: mneeMaxIterations, mneeMaxChainLength.',
          details: { keys: unknownCausticKeys },
        });
      }
    }
    this.#bdpt = opts.bdpt === true;
    // A9 — light-subpath bounce cap raised 3 → 8 (matches the eye cap; the merged
    // pdf array BDPT_MAX_MERGED=19 = c≤8 + e≤8 + 3 headroom). Default is endpoint-only
    // until the multi-vertex connection estimator is weighted against the regular
    // eye-path strategy; hosts may opt up to 8 for research captures.
    this.#bdptMaxLightBounces = resolveBdptMaxLightBounces(opts.bdptOptions?.maxLightBounces);
    // EXPERIMENTAL ReSTIR-PT reuse: compile-time opt-in, full-tier only. GpuResources
    // gates the full-tier requirement internally; mirror the resolved value here so
    // the capability + renderFrame sequencing agree with what GpuResources will run.
    this.#restirPtReuse = opts.restirPtReuse === true && traceTier === 'full';
    const rptOpts = opts.restirPtReuseOptions ?? {};
    this.#restirPtMClamp =
      typeof rptOpts.mClamp === 'number' && rptOpts.mClamp >= 1 ? Math.floor(rptOpts.mClamp) : 20;
    this.#restirPtWCap = typeof rptOpts.wCap === 'number' && rptOpts.wCap > 0 ? rptOpts.wCap : 10;
    this.#gpu = new GpuResources(
      opts.device,
      traceTier,
      this.#bdpt,
      this.#restirPtReuse,
      (warning) => this.#warn(warning),
    );
    if (opts.denoiser === 'oidn-final') {
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
      syncLiteTextures: (sceneBuffers) => this.#syncLiteTextures(sceneBuffers),
      isLiteTier: () => this.#traceTier === 'lite',
      warn: (warning, ...args) => this.#warn(warning, ...args),
      repackScene: (scene, opts) => this.#repackScene(scene, opts),
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
    return {
      // Material / transform / positions primitive patches upload in place.
      // `topology: true` — every COUNT-changing patch updatePrimitive can legally
      // receive is absorbed without a full setScene:
      //   • instanced-mesh instance-count change → TLAS-only rebuild, BLAS reused
      //     verbatim (slice-1);
      //   • mesh/skinned-mesh vertex/index-count change → rebuild ONLY the
      //     changed primitive's BLAS, splice into the concat buffers, rebase
      //     downstream offsets + TLAS roots, realloc the 10 geometry buffers
      //     (slice-2).
      // `id`/`kind` morphs throw (contract violation); whole-primitive add/remove
      // is `setScene`, not a patch — both correctly outside the `topology` flag.
      supportsIncrementalScene: true,
      incrementalPatchSupport: this.#traceTier === 'lite'
        ? {
            transform: false,
            positions: false,
            material: true,
            emitter: true,
            topology: false,
          }
        : {
            transform: true,
            positions: true,
            material: true,
            emitter: true,
            topology: true,
          },
      // Explicit whole-primitive add/remove API (addPrimitive / removePrimitive)
      // is implemented via a full buildPackedScene repack of the mutated scene.
      supportsAddRemovePrimitive: true,
      supportsAuxBuffers: true,
      accumulates: true,
      // Progressive walkaround→PT handoff (P8): this engine can seed its accum
      // buffers with an initial image as a decaying prior (seedAccumulator).
      supportsAccumulatorSeed: true,
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      // H12 — lite-tier capabilities reflect what the lite kernel ACTUALLY binds:
      //   • No analytic shapes (group-1 is not bound on the lite layout; the
      //     analytic geometry/params/localToWorld/worldToLocal buffers are absent).
      //   • Mesh/skinned/instanced primitives are statically supported by baking
      //     all mesh-like primitives into one world-space BLAS at setScene time.
      //   • Emitters: directional + point + spot + rect-area + disc-area
      //     (B12 — texture-packed). Mesh-area remains unsupported.
      //   • Environments: none + procedural-sky + hdri (B12 — texture-packed).
      //   • No pt-webgpu-bdpt in experimentalFeatures even when bdpt:true was
      //     passed at construction (BDPT requires the full-tier group-2 layout).
      //
      // B12 (Wave B) — lite-tier fidelity cliff, SHIPPED.
      // The lite tier targets adapters reporting maxStorageBuffersPerShaderStage
      // as low as 8 (PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE). The lite
      // group-0 layout already consumes 7 storage buffers (bindings 2,3,4,5,6,7,8
      // = accum, positions, indices, triMaterialIds, materials, bvhNodes,
      // normals), leaving exactly ONE free storage-buffer slot under the 8 cap.
      //
      // B12 resolution: light data and HDRI env packed as sampled texture_2d<f32>
      // (bindings 12-14 in group-0, type = 'texture' not 'buffer' — counted from
      // maxSampledTexturesPerShaderStage ≥ 16, NOT the storage-buffer budget).
      //   • liteLightTex (binding 14): 1×N RGBA32F, directional (2 vec4/light)
      //     + point (3 vec4/light) + spot (4 vec4/light) + rect/disc-area
      //     (4 vec4/light) packed contiguously.
      //   • liteEnvTex (binding 12): W×H RGBA32F, .rgb = HDR radiance, .a = pdf/sr.
      //   • liteEnvCdfTex (binding 13): W×H RGBA32F, .r = marginal/conditional CDF
      //     value at pixel i+1 (2D layout to avoid 8192-width limit).
      // Budget arithmetic post-B12: 7 storage buffers (unchanged) + 3 sampled
      // textures (new, drawn from a separate ≥16 budget). The budget arithmetic is
      // PINNED by the liteTierBindingBudget test in webgpuLimits.test.ts.
      //
      // For the full tier the capability is derived from PT_WEBGPU_SUPPORT so
      // the declared set and the ingestion/packer behavior stay in sync.
      supportedAnalyticShapes: this.#traceTier === 'lite'
        ? new Set<import('@vitrum/core').AnalyticShape>()
        : new Set(PT_WEBGPU_SUPPORT.supportedAnalyticShapes),
      supportedEmitterKinds: this.#traceTier === 'lite'
        // B12 — point/spot/rect/disc-area now supported via lite texture packing (liteLightTex).
        ? new Set(PT_WEBGPU_LITE_SUPPORTED_EMITTER_KINDS)
        : new Set(PT_WEBGPU_SUPPORT.supportedEmitterKinds),
      supportedPrimitiveKinds: this.#traceTier === 'lite'
        ? new Set<import('@vitrum/core').ScenePrimitive['kind']>(['mesh', 'skinned-mesh', 'instanced-mesh'])
        : new Set(PT_WEBGPU_SUPPORT.supportedPrimitiveKinds),
      supportedEnvironmentKinds: this.#traceTier === 'lite'
        // B12 — HDRI env now supported via lite texture packing (liteEnvTex + liteEnvCdfTex).
        ? new Set<import('@vitrum/core').Scene['environment']['kind']>(['none', 'procedural-sky', 'hdri'])
        : new Set(PT_WEBGPU_SUPPORT.supportedEnvironmentKinds),
      presentationMode: 'offscreen-texture',
      // H12 — lite-tier supportDetails must reflect what group-0 ACTUALLY binds,
      // not the full-tier ledger. Group-0 lite omits group-1 (analytic, env, lights)
      // and group-2 (TLAS, BDPT). Mesh-area emitters, analytic primitives, and
      // analytic shapes remain unsupported (no NEE path for those in the lite kernel).
      // Static instanced meshes are native because the lite packer bakes each
      // instance into its single root-0 BLAS. Geometry, transform, and topology
      // patches are accepted through a fallback merged-BLAS repack rather than
      // the full-tier BLAS/TLAS-native fast paths.
      // B12 — point/spot/rect/disc-area upgraded to 'native' (texture-packed NEE).
      // B12 — hdri upgraded to 'native' (liteEnvTex + liteEnvCdfTex importance sampling).
      supportDetails:
        this.#traceTier === 'lite'
          ? {
              primitives: {
                mesh: 'native',
                'skinned-mesh': 'native',
                'instanced-mesh': 'native',
                analytic: 'unsupported',
              },
              emitters: {
                directional: 'native',
                point: 'native',
                spot: 'native',
                'rect-area': 'native',
                'disc-area': 'native',
                'mesh-area': 'unsupported',
              },
              environments: {
                none: 'native',
                'procedural-sky': 'approximate',
                hdri: 'native',
              },
              analyticShapes: {
                sphere: 'unsupported',
                box: 'unsupported',
                capsule: 'unsupported',
                cylinder: 'unsupported',
                'h-channel-came': 'unsupported',
              },
              materials: PT_WEBGPU_LITE_MATERIALS,
              // SHADOW-01 — same rows as the full tier: primitive castShadow is
              // enforced in the SHARED traceMeshBvh any-hit path; the lite NEE
              // loops gate directional/point/spot/rect emitter flags through
              // the lite light texture records.
              shadows: BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.shadows,
              denoisers: BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.denoisers,
              mutations: {
                ...BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.mutations,
                transform: 'fallback-rebuild',
                positions: 'fallback-rebuild',
                material: 'native',
                topology: 'fallback-rebuild',
              },
            }
          : BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails,
      experimentalFeatures: new Set([
        ...(this.#traceTier === 'lite' ? (['pt-webgpu-lite-tier'] as const) : []),
        ...(this.#postDenoiser instanceof OIDNFinalDispatcher
          ? (['pt-webgpu-oidn-final'] as const)
          : []),
        // BDPT requires the full-tier group-2 layout; suppress from lite even
        // when bdpt:true was passed at construction (the engine silently ignores
        // the flag for lite — the shader does not have the bdptEnabled UBO slot
        // and the BDPT sub-path pipeline is not created on the lite layout).
        ...(this.#bdpt && this.#traceTier !== 'lite' ? (['pt-webgpu-bdpt'] as const) : []),
        ...(this.#restirPtReuse ? (['pt-webgpu-restir-pt-reuse'] as const) : []),
        ...(this.#traceTier !== 'lite' && this.#causticStrategy === 'photon-map'
          ? (['pt-webgpu-photon-map-sppm'] as const)
          : []),
      ]),
      causticStrategy: this.#traceTier === 'lite' ? 'none' : this.#causticStrategy,
      // W3-D8 — this engine exposes `debug.estimatedGpuMemoryBytes()`.
      debugSurface: true,
    };
  }

  /** Supported analytic shapes (id > 0) — passed to the pure incrementalPatch
   *  resolvers so they reproduce buildPackedScene's material/analytic ordering. */
  #supportedAnalyticShapes(): ReadonlySet<string> {
    return PTEngineWebGPU.#SUPPORTED_ANALYTIC_SHAPES;
  }

  // ── Debug introspection (T3.G followup + #30 pickPrimitive) ──────────────
  // Atlas / BVH / denoiser-toggle hooks are walkaround-hybrid concepts that
  // don't apply to a brute-force compute path tracer. GPU-memory estimate
  // and CPU click-to-pick (pickPrimitive) are both wired.
  readonly debug: EngineDebugSurface = {
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
        cameraPosition: last.cameraPosition,
      };
      return pickPrimitiveCpu(scene, cam, x, y, w, h);
    },

    estimatedGpuMemoryBytes: (): GpuMemoryBreakdown | null => {
      const gpu = this.#gpu;
      const W = gpu.accumWidth;
      const H = gpu.accumHeight;
      // Pre-init / pre-renderFrame: no accum textures yet.
      if (W <= 0 || H <= 0 || gpu.accumTexture == null) return null;

      // Per-texel bytes inferred from the actual format at allocation time
      // (see GpuResources.ensureAccumResources: accum / normalDepth / albedo /
      // variance / motionVectors are all rgba16float = 8 bytes/texel).
      const RGBA16F = 8;
      const texPixels = W * H;
      const accumBytes        = texPixels * RGBA16F;
      const normalDepthBytes  = texPixels * RGBA16F;
      const albedoBytes       = texPixels * RGBA16F;
      const varianceBytes     = texPixels * RGBA16F;
      const motionBytes       = texPixels * RGBA16F;
      const accumBufBytes     = gpu.accumBufferByteSize;
      const varMomentBufBytes = gpu.varianceMomentsBuffer != null
        ? gpu.accumBufferByteSize : 0;
      const paramsBufBytes    = gpu.paramsBuffer != null ? 512 : 0;
      // Scene buffers (BVH/materials/indices/TLAS/emitters/light-tree/UVs) + the
      // two material texture arrays, summed live off the current handles (was
      // previously omitted, so `total` under-reported by the whole scene).
      const scene = this.#sceneBuffers != null
        ? this.#sceneBuffers.gpuMemoryBytes()
        : { bufferBytes: 0, textureBytesByFormat: {} as Readonly<Record<string, number>> };
      const sceneTexBytes = Object.values(scene.textureBytesByFormat).reduce((a, b) => a + b, 0);
      const sceneBytes = scene.bufferBytes + sceneTexBytes;

      const commonTexBytes = accumBytes + normalDepthBytes + albedoBytes
        + varianceBytes + motionBytes;
      const commonBufBytes = accumBufBytes + varMomentBufBytes + paramsBufBytes;
      const total = commonTexBytes + commonBufBytes + sceneBytes;

      // byTextureFormat: common render targets (rgba16float) + the scene's
      // material arrays by their actual format. Keeps the secondary tables
      // summing to `total` (the documented invariant).
      const textureFormats: Record<string, number> = { rgba16float: commonTexBytes };
      for (const [fmt, bytes] of Object.entries(scene.textureBytesByFormat)) {
        textureFormats[fmt] = (textureFormats[fmt] ?? 0) + bytes;
      }

      return Object.freeze({
        total,
        byCategory: Object.freeze({
          common: commonTexBytes + commonBufBytes,
          scene: sceneBytes,
        }),
        byTextureFormat: Object.freeze(textureFormats),
        byBufferUsage: Object.freeze({
          storage: accumBufBytes + varMomentBufBytes + scene.bufferBytes,
          uniform: paramsBufBytes,
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
    const displayTex = this.#gpu.presentTexture ?? primary;
    return {
      kind: 'rendered',
      primaryRadiance: asBackendTexture<'webgpu', GPUTexture>(displayTex),
      ...(this.#gpu.normalDepthTexture != null
        ? { normalDepth: asBackendTexture<'webgpu', GPUTexture>(this.#gpu.normalDepthTexture) }
        : {}),
      ...(this.#gpu.albedoTexture != null
        ? { albedo: asBackendTexture<'webgpu', GPUTexture>(this.#gpu.albedoTexture) }
        : {}),
      ...(this.#gpu.varianceTexture != null
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
      const mem = this.debug.estimatedGpuMemoryBytes?.() ?? null;
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
      // Warn when the scene contains content the lite tier cannot handle.
      // B12 — point/spot/rect-area emitters and HDRI environments are now
      // supported via texture packing (liteLightTex, liteEnvTex, liteEnvCdfTex).
      // Remaining unsupported: analytic primitives (group-1 absent) and mesh-area
      // emitters (no NEE path in lite kernel). Static
      // mesh/skinned/instanced primitives, including non-identity transforms, are
      // baked into the lite tier's single world-space BLAS at pack time.
      const analyticPrimitives = scene.primitives.filter((p) => p.kind === 'analytic');
      if (analyticPrimitives.length > 0) {
        const primitiveIds = analyticPrimitives.map((p) => p.id);
        this.#warn({
          code: 'pt-webgpu.lite-analytic-primitive',
          backend: 'pt-webgpu',
          phase: 'setScene',
          method: 'setScene',
          message:
            `[vitrum/pt-webgpu] Lite tier: scene contains ${analyticPrimitives.length} analytic primitive(s) — ` +
            'analytic shape rendering requires the full tier (group-1 bindings are absent on lite). ' +
            'These will be silently ignored.',
          details: {
            count: analyticPrimitives.length,
            primitiveIds,
            primitiveKinds: ['analytic'],
            requiredTier: 'full',
          },
        });
      }
      // B12 — only mesh-area is unsupported on lite; point/spot/rect/disc-area
      // are handled via texture-packed NEE (liteLightTex).
      // D8.14: kind-set sourced from PT_WEBGPU_LITE_UNSUPPORTED_EMITTER_KINDS
      // (derived from PT_WEBGPU_SUPPORT diff) rather than inline literals.
      const unsupportedEmitters = scene.emitters.filter(
        (e) => PT_WEBGPU_LITE_UNSUPPORTED_EMITTER_KINDS.has(e.kind),
      );
      if (unsupportedEmitters.length > 0) {
        const kinds = [...new Set(unsupportedEmitters.map((e) => e.kind))].join(', ');
        const emitterIds = unsupportedEmitters.map((e) => e.id);
        const emitterKinds = unsupportedEmitters.map((e) => e.kind);
        this.#warn({
          code: 'pt-webgpu.lite-unsupported-emitters',
          backend: 'pt-webgpu',
          phase: 'setScene',
          method: 'setScene',
          message:
            `[vitrum/pt-webgpu] Lite tier: scene contains emitters of kind(s) [${kinds}] — ` +
            'mesh-area emitters are not supported on the lite tier (no NEE path in lite kernel). ' +
            'These will be silently ignored.',
          details: {
            kinds,
            count: unsupportedEmitters.length,
            emitterIds,
            emitterKinds,
            requiredTier: 'full',
          },
        });
      }
      // B12 follow-up — HDRI environments and all directional emitters are now
      // supported via texture packing; no first-directional truncation warning.
      const vertexColorPrimitiveIds = collectLiteUnsupportedVertexColorPrimitiveIds(scene);
      if (vertexColorPrimitiveIds.length > 0) {
        this.#warn({
          code: 'pt-webgpu.lite-unsupported-vertex-colors',
          backend: 'pt-webgpu',
          phase: 'setScene',
          method: 'setScene',
          message:
            `[vitrum/pt-webgpu] Lite tier: scene contains ${vertexColorPrimitiveIds.length} primitive(s) with ` +
            'non-constant or alpha-bearing vertex colors (COLOR_0). Constant RGB vertex colors are baked into ' +
            'the lite material base color, but arbitrary COLOR_0 still needs the full-tier vertex-color binding.',
          details: {
            primitiveIds: vertexColorPrimitiveIds,
            bakedWhen: 'constant-rgb-alpha-one',
            requiredTier: 'full',
          },
        });
      }
    }
    this.#repackScene(scene, { warnOnEmpty: true });
  }

  /**
   * D8.15 — Recompute the SPPM scale-aware initial radius (#sppmR0) and scene
   * half-diagonal (#sppmSceneExtent) from the current geometry pack's positions.
   *
   * Performs an inline AABB walk over `#geoPack.positions` (stride-4 Float32Array:
   * xyz + packed-uv-in-w per vertex).  On success writes `#sppmR0` and
   * `#sppmSceneExtent`; also invalidates `#gpu.sppmBuffersReady` so the buffers
   * rebuild with the new scene-extent stats on the next frame.
   *
   * Called only from `#repackScene` when `causticStrategy === 'photon-map'` on
   * the full tier — the guard is preserved by the caller (behavior-preserving).
   */
  #computeSppmSceneStats(): void {
    const pos = this.#geoPack?.positions;
    if (pos != null && pos.length >= 4) {
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let i = 0; i + 2 < pos.length; i += 4) {
        const x = pos[i]!, y = pos[i + 1]!, z = pos[i + 2]!;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }
      if (Number.isFinite(minX)) {
        this.#sppmR0 = sppmInitialRadius(
          [minX, minY, minZ] as const,
          [maxX, maxY, maxZ] as const,
        );
        const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
        this.#sppmSceneExtent = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;
      }
    }
    // A4-progressive: invalidate SPPM buffers so they rebuild with the new
    // scene-extent stats (r₀ recomputed above).  The per-pixel stats buffer
    // is cleared in reset() (called below) — static-eye-point invariant holds
    // because a setScene always resets accumulation.
    this.#gpu.sppmBuffersReady = false;
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
    const unsupportedDisplacementUses = collectUnsupportedDisplacementFieldUses(scene);
    const unsupportedDisplacementFields = collectFieldUnion(unsupportedDisplacementUses);
    if (unsupportedDisplacementFields.length > 0) {
      this.#warn({
        code: 'pt-webgpu.unsupported-displacement-material',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        message:
          `[vitrum/pt-webgpu] setScene: displacement material fields are supplied ` +
          `but not rendered by this backend: ${unsupportedDisplacementFields.join(', ')}.`,
        details: {
          fields: unsupportedDisplacementFields,
          primitiveIds: unsupportedDisplacementUses.map((use) => use.primitiveId),
          primitiveFields: unsupportedDisplacementUses,
        },
      });
    }
    // CAP-01 — warn on the remaining silently-dropped material fields (matrix-
    // driven: every 'unsupported' row in the ledger's pt-webgpu material support
    // matrix except displacement, which has its own warning above). Once per
    // setScene/repack, mirroring the displacement warning.
    const unsupportedMaterialUses = collectUnsupportedMaterialFieldUses(scene, this.#traceTier);
    const unsupportedMaterialFields = collectFieldUnion(unsupportedMaterialUses);
    if (unsupportedMaterialFields.length > 0) {
      this.#warn({
        code: 'pt-webgpu.unsupported-material-fields',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        message:
          `[vitrum/pt-webgpu] setScene: material fields are supplied ` +
          `but not rendered by this backend: ${unsupportedMaterialFields.join(', ')}.`,
        details: {
          fields: unsupportedMaterialFields,
          primitiveIds: unsupportedMaterialUses.map((use) => use.primitiveId),
          primitiveFields: unsupportedMaterialUses,
        },
      });
    }
    // SHADOW-01 — `receiveShadow` is @reserved on all shipping backends (a
    // "receiver ignores occlusion" toggle is non-physical for a GI path
    // tracer). Surface a structured signal when a scene sets it to false so
    // hosts aren't left guessing why nothing changed.
    const receiveShadowIds = scene.primitives
      .filter((p) => (p as { receiveShadow?: boolean }).receiveShadow === false)
      .map((p) => p.id);
    if (receiveShadowIds.length > 0) {
      this.#warn({
        code: 'pt-webgpu.reserved-receive-shadow',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        message:
          `[vitrum/pt-webgpu] setScene: receiveShadow:false is reserved and not ` +
          `consumed by any backend (non-physical for GI); primitives: ${receiveShadowIds.join(', ')}.`,
        details: { primitiveIds: receiveShadowIds },
      });
    }
    const packed = buildPackedScene(scene, {
      cameraVisibleEmitters: this.#cameraVisibleEmitters,
      geometryMode: this.#traceTier === 'lite' ? 'merged' : 'tlas',
      onWarning: (warning) => this.#warn(warning),
      warningPhase: 'setScene',
      warningMethod: 'setScene',
    });
    this.#geoPack = scenePackResultFromPacked(packed);
    this.#sceneBuffers?.destroy();
    this.#sceneBuffers = uploadPackedScene(this.#device, packed);
    this.#syncLiteTextures(this.#sceneBuffers);
    this.#bdptLightPath?.dispose();
    this.#bdptLightPath = null;
    if (this.#bdpt && this.#traceTier === 'full') {
      this.#bdptLightPath = new BdptLightPathBufferWebGPU(this.#device, {
        maxLightBounces: this.#bdptMaxLightBounces,
      });
    }
    this.#gpu.invalidateBindGroups();
    this.#scene = scene;
    // A4 — recompute SPPM scale-aware initial radius from the scene AABB.
    // Guard preserved: only runs when causticStrategy === 'photon-map' on full tier.
    if (this.#causticStrategy === 'photon-map' && this.#traceTier === 'full') {
      this.#computeSppmSceneStats();
    }
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
    for (const warning of packed.warnings) {
      this.#warn({
        code: 'pt-webgpu.scene-pack-warning',
        backend: 'pt-webgpu',
        phase: 'setScene',
        method: 'setScene',
        message: `[vitrum/pt-webgpu] ${warning}`,
        details: { warning },
      });
    }
    this.reset();
  }

  #syncLiteTextures(sceneBuffers: UploadedSceneBuffers | null): void {
    if (this.#traceTier !== 'lite' || sceneBuffers == null) return;
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
    this.#gpu.uploadLiteTextures(lightTex, envTex, cdfTex);
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

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    this.#mutationRouter.updatePrimitive(id, patch);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#mutationRouter.updateEmitter(id, patch);
  }

  updateEnvironment(env: Scene['environment'] | null): void {
    this.#mutationRouter.updateEnvironment(env);
  }

  /**
   * D8.4 — Ensure all per-frame GPU resources are allocated and return the set
   * of readiness flags that the pass-encoding step consumes.
   *
   * Covers (in order, PRESERVING the documented invariant):
   *   1. BDPT eye-stack sizing.
   *   2. Params UBO packing + queue.writeBuffer.
   *   3. ALL SPPM buffer allocation (BEFORE buildBindGroups — Item-1 fix invariant).
   *   4. buildBindGroups (scene bind groups).
   *   5. ReSTIR-PT reservoir ensure / pipeline build / writeReservoirParams /
   *      buildReservoirBindGroups.
   *
   * Returns `{ bindGroup, sppmReady, restirPtReady, bdptStackReady }`.
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
    bdptStackReady: boolean;
  } {
    const gpu = this.#gpu;

    // BDPT eye-subpath scratch stack (D2). Sized per-pixel × the active bounce
    // depth; refuses to grow beyond the safety ceiling (returns false → BDPT
    // connections skipped this frame, unidirectional path unaffected). A 32-byte
    // placeholder is allocated when BDPT is off so the auto layout stays valid.
    const bdptActive = this.#bdpt && this.#traceTier === 'full';
    const bdptStackReady = gpu.ensureBdptEyeStack(
      width,
      height,
      this.#activeBounces,
      bdptActive,
    );

    const paramsArrayBuffer = this.#buildParamsBuffer(input, width, height);
    if (bdptActive && !bdptStackReady) {
      // Over-budget: disable BDPT in the UBO so the shader takes the
      // unidirectional path (and never touches the placeholder eye stack).
      const u32 = new Uint32Array(paramsArrayBuffer);
      u32[FrameParamsSlot.bdptEnabled] = 0;
    }
    this.#device.queue.writeBuffer(gpu.paramsBuffer!, 0, paramsArrayBuffer);

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
    // A4-progressive (2026-06-10): true Hachisuka SPPM — per-pixel (τ, R², N)
    // buffer at @group(3) @binding(9), reset on accumulator reset.  The hash
    // grid (bindings 6/7) still re-deposits fresh photons every frame (counters
    // are NOT cleared per frame — photons accumulate in the ring buffer across
    // frames for stable coverage), but the gather now applies the progressive
    // update rule (N'=N+αM, R'²=R²·N'/(N+M), τ'=(τ+Φ_M)·ratio) instead of the
    // frozen-radius density estimate.
    const sppmActive =
      this.#causticStrategy === 'photon-map' && this.#traceTier === 'full';
    let sppmReady = false;
    if (sppmActive) {
      const sppmBuffersOk = gpu.ensureSppmBuffers(true);
      if (!sppmBuffersOk) {
        // Ceiling exceeded — fall back silently (manifold-nee semantics).
        gpu.ensureSppmBuffers(false); // ensure placeholder satisfies the layout
      }
      // A4-progressive: ensure per-pixel stats buffer at the current render dims.
      // ensureSppmPixelStatsBuffer is idempotent on a cache hit (same W×H); on
      // a first allocation or dim change it GPU-clears (τ/R²/N → 0) and
      // invalidates group-3.  A 64-byte placeholder is already created by
      // ensureSppmBuffers(false) above when SPPM is off, so the else branch below
      // doesn't need to call it separately.
      const sppmPixelStatsOk = gpu.ensureSppmPixelStatsBuffer(width, height);
      // A4-progressive: `frameAccumulated` counts completed frames so WGSL can
      // compute Ne = frameAccumulated × photonCount.  This frame's photons are
      // emitted NOW (pre-megakernel), so the completed count INCLUDING this frame
      // is #samplesAccumulated + 1 (incremented at the end of renderFrame).
      // `currentRadius` is still r₀ (the initial search radius); the per-pixel
      // radius stored in sppmPixelStats shrinks progressively — the UBO field is
      // kept for the photon-insertion pass which still needs the global grid cell
      // size (tied to r₀, not the shrinking per-pixel radius).
      if (sppmBuffersOk && sppmPixelStatsOk) {
        gpu.writeSppmStats(
          this.#sppmR0,
          this.#sppmR0,
          this.#samplesAccumulated + 1,
          SPPM_PHOTON_COUNT,
          this.#sppmSceneExtent,
        );
      }
      sppmReady =
        sppmBuffersOk &&
        sppmPixelStatsOk &&
        gpu.sppmPhotonPipeline != null &&
        gpu.sppmPixelStatsBuffer != null;
    } else if (this.#traceTier === 'full') {
      // Ensure placeholder SPPM buffers exist so group-3 bindings 6/7/8/9 are
      // satisfied (the gather is guarded by causticMode() == 2u, so the
      // placeholders are never accessed).
      gpu.ensureSppmBuffers(false);
    }

    const bindGroup = gpu.buildBindGroups(this.#sceneBuffers!, () => this.#bdptLightPathBuffer());

    // EXPERIMENTAL ReSTIR-PT reuse (OFF by default). When ON: (re)allocate the
    // reservoir ping-pong + result + params, build the reuse pipelines + bind
    // groups, and write RestirPtParams. All gated inside GpuResources, so this is
    // a no-op + allocates nothing when the flag is off (default render untouched).
    let restirPtReady = false;
    if (this.#restirPtReuse) {
      const reservoirBuffersReady = gpu.ensureReservoirBuffers(width, height);
      if (reservoirBuffersReady) {
        gpu.ensureReservoirPipelines();
        gpu.writeReservoirParams(width, height, this.#restirPtMClamp, this.#restirPtWCap);
        gpu.buildReservoirBindGroups(this.#sceneBuffers!);
        restirPtReady =
          gpu.rptProducerPipeline != null &&
          gpu.rptTemporalPipeline != null &&
          gpu.rptSpatialPipeline != null &&
          gpu.rptResolvePipeline != null &&
          gpu.rptProducerGroup0 != null &&
          gpu.pathTraceBindGroup1 != null &&
          gpu.pathTraceBindGroup2 != null &&
          gpu.pathTraceBindGroup3 != null;
      }
    }

    return { bindGroup, sppmReady, restirPtReady, bdptStackReady };
  }

  /**
   * D8.4 — Encode the four conditional compute passes onto `encoder`:
   *   1. SPPM photon-emission pass (before the megakernel, when `sppmReady`).
   *   2. ReSTIR-PT reuse sequence: producer → temporal → spatial → resolve
   *      (when `restirPtReady`).
   *   3. BDPT light-subpath pass (when wired and `bdptStackReady`).
   *   4. Megakernel (composite when ReSTIR-PT active, otherwise standard).
   *
   * Call order is IDENTICAL to the prior inline body. `encoder` must be a fresh
   * command encoder; the present pass is dispatched separately by the caller.
   */
  #encodePathTracePasses(
    encoder: GPUCommandEncoder,
    bindGroup: GPUBindGroup,
    sppmReady: boolean,
    restirPtReady: boolean,
    bdptStackReady: boolean,
    width: number,
    height: number,
  ): void {
    const gpu = this.#gpu;

    // ── A4-progressive SPPM photon-emission pass (before the megakernel) ────────
    // The photon pass binds the SAME groups 0/1/2/3 as the megakernel.
    // Group 3 carries the SPPM hash-grid buffers at bindings 6/7/8 (+ per-pixel
    // stats at binding 9, read/written by the megakernel's sppmGatherProgressive)
    // alongside the light-tree / material textures (0-5), no group-4 needed.
    //
    // Counters are NOT cleared per frame — photons accumulate in the ring buffer
    // (streaming window of last SPPM_CELL_CAPACITY photons per cell), giving
    // stable coverage.  The progressive radius/τ shrink is handled per-pixel in
    // the megakernel's sppmGatherProgressive, not in the emission pass.
    if (sppmReady && gpu.sppmPhotonPipeline != null) {
      const photonPass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.sppm.photonPass' });
      photonPass.setPipeline(gpu.sppmPhotonPipeline);
      photonPass.setBindGroup(0, bindGroup);
      photonPass.setBindGroup(1, gpu.pathTraceBindGroup1);
      photonPass.setBindGroup(2, gpu.pathTraceBindGroup2);
      if (gpu.pathTraceBindGroup3 != null) {
        photonPass.setBindGroup(3, gpu.pathTraceBindGroup3);
      }
      photonPass.dispatchWorkgroups(SPPM_WORKGROUP_COUNT, 1, 1);
      photonPass.end();
    }

    // ── ReSTIR-PT reuse sequence: producer → temporal → resolve (before the
    //    megakernel; the result lands in the SEPARATE rpt_result debug buffer,
    //    NOT the beauty image, this increment). The passes share the megakernel's
    //    group-1/2/3 scene bind groups (same explicit layouts) + their own
    //    group-0 (the reuse-extended one). ──
    if (restirPtReady) {
      this.#encodeRestirPtReusePasses(encoder, gpu, width, height);
    }
    if (
      gpu.bdptSubpathPipeline != null &&
      this.#bdpt &&
      bdptStackReady &&
      this.#bdptExternalBuffer == null &&
      this.#bdptLightPath != null &&
      this.#traceTier === 'full' &&
      gpu.pathTraceBindGroup1 != null &&
      gpu.pathTraceBindGroup2 != null
    ) {
      const bdptPass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.bdptLightSubpath.pass' });
      bdptPass.setPipeline(gpu.bdptSubpathPipeline);
      bdptPass.setBindGroup(0, bindGroup);
      bdptPass.setBindGroup(1, gpu.pathTraceBindGroup1);
      bdptPass.setBindGroup(2, gpu.pathTraceBindGroup2);
      // Group 3 (WS2 light tree) is part of the SHARED pipeline layout, so it
      // must be bound on the BDPT pipeline too even though the light-subpath pass
      // does not sample the tree (an unbound group fails layout validation).
      if (gpu.pathTraceBindGroup3 != null) {
        bdptPass.setBindGroup(3, gpu.pathTraceBindGroup3);
      }
      // ONE workgroup: bdptExtendLightSubpath now builds the whole light subpath
      // sequentially in a single invocation (was one workgroup per column with a
      // cross-workgroup read of column-1 — a spec-undefined-ordering data race).
      bdptPass.dispatchWorkgroups(1, 1, 1);
      bdptPass.end();
    }

    // A1 — when ReSTIR-PT reuse + its composite megakernel are ready, dispatch the
    // COMPOSITE megakernel (E0-direct-only + adds the resolve indirect from
    // rpt_result) bound to the reuse-extended group 0 (which carries rpt_result at
    // the relocated binding 23). Otherwise the default full-path megakernel. The
    // composite path composites the ReSTIR-PT indirect into the BEAUTY accumulator;
    // the default path is byte-identical to pre-A1.
    const useComposite =
      restirPtReady && gpu.rptCompositePipeline != null && gpu.rptProducerGroup0 != null;
    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.pathTrace.pass' });
    if (useComposite) {
      pass.setPipeline(gpu.rptCompositePipeline!);
      pass.setBindGroup(0, gpu.rptProducerGroup0);
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
  }

  renderFrame(input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');
    // Advance the error-throttle frame counter so per-frame GPU errors don't
    // spam the host on every call (see #onUncapturedError throttle logic).
    this.#errorFrameCount++;
    if (!this.#inInverseRender) {
      this.#lastFrameInput = input;
    }
    const frameStartMs = globalThis.performance?.now?.() ?? Date.now();

    const gpu = this.#gpu;

    if (this.#slot.get() === 'paused' && !this.#inInverseRender) {
      const pq = input.quality ?? {};
      const targetSppPaused = Math.min(pq.samplesTarget ?? 16, this.#maxSamplesLimit);
      const accumTexture = gpu.accumTexture;
      if (accumTexture == null) {
        return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      }
      const output = this.#frameOutput(
        accumTexture,
        this.#samplesAccumulated,
        this.#samplesAccumulated >= targetSppPaused,
      );
      this.#emitFrameTelemetry(frameStartMs, 0, this.#samplesAccumulated, targetSppPaused);
      return output;
    }

    const q = input.quality ?? {};
    this.#activeBounces = Math.max(1, Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit));
    const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
    const resolution = q.resolutionFactor ?? 1;
    const width = Math.max(1, Math.floor(input.viewport.width * resolution));
    const height = Math.max(1, Math.floor(input.viewport.height * resolution));

    // A recreate resets the sample accumulator (the prior inline
    // #ensureAccumResources set #samplesAccumulated = 0 here; that one piece of
    // engine state is reported back rather than mutated inside GpuResources).
    if (gpu.ensureAccumResources(width, height)) {
      this.#samplesAccumulated = 0;
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
    if (accumTexture != null && this.#samplesAccumulated >= targetSpp) {
      const output = this.#frameOutput(accumTexture, this.#samplesAccumulated, true);
      this.#emitFrameTelemetry(frameStartMs, 0, this.#samplesAccumulated, targetSpp);
      return output;
    }

    const { bindGroup, sppmReady, restirPtReady, bdptStackReady } =
      this.#ensurePerFrameResources(input, width, height);

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.pathTrace.encoder' });

    this.#encodePathTracePasses(encoder, bindGroup, sppmReady, restirPtReady, bdptStackReady, width, height);

    // ── Present pass: tonemap / exposure / outputColorSpace ───────────────────
    // Reads the just-written accumTexture (running-mean linear HDR) and writes
    // the tonemapped + OETF-encoded result to presentTexture. Both textures are
    // created in ensureAccumResources and have TEXTURE_BINDING | STORAGE_BINDING
    // usages. The present pass always runs (aces@1.0@srgb by default); hosts
    // that want raw linear output can set tonemap:'none' + outputColorSpace:'linear'.
    // Adjoint/OIDN readbacks always use accumTexture (not presentTexture).
    if (gpu.presentTexture != null) {
      const q = input.quality ?? {};
      const tonemapMode = TONEMAP_MODE_INDEX[q.tonemap ?? 'aces'];
      const exposure = q.exposure ?? 1.0;
      const outputColorSpace = q.outputColorSpace === 'linear' ? 1 : 0;
      gpu.ensurePresentPipeline();
      gpu.writePresentParams(tonemapMode, exposure, outputColorSpace);
      gpu.dispatchPresentPass(encoder, width, height);
    }

    this.#device.queue.submit([encoder.finish()]);

    // ReSTIR-PT ping-pong: this frame's resolved reservoir (`Cur`) becomes next
    // frame's history (`Prev`). No-op when reuse is off. Done after submit so the
    // temporal pass this frame read the prior `Prev` before it is overwritten.
    if (restirPtReady) {
      gpu.swapReservoirs();
    }

    this.#samplesAccumulated = Math.min(this.#samplesAccumulated + 1, this.#maxSamplesLimit);
    const accumTexturePost = gpu.accumTexture;
    if (accumTexturePost == null) {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    }
    const isConverged = this.#samplesAccumulated >= targetSpp;
    if (
      this.#postDenoiser != null &&
      isConverged &&
      this.#samplesAccumulated > 0 &&
      gpu.accumTexture != null &&
      gpu.albedoTexture != null &&
      gpu.normalDepthTexture != null
    ) {
      this.#postDenoiser.kickIfReady(
        this.#device,
        {
          color: gpu.accumTexture,
          albedo: gpu.albedoTexture,
          normalDepth: gpu.normalDepthTexture,
        },
        width,
        height,
      );
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
    dispatch('vitrum.pt-webgpu.restirPt.produce', gpu.rptProducerPipeline!, gpu.rptProducerGroup0!);
    dispatch('vitrum.pt-webgpu.restirPt.temporal', gpu.rptTemporalPipeline!, gpu.rptTemporalGroup0!);
    dispatch('vitrum.pt-webgpu.restirPt.spatial', gpu.rptSpatialPipeline!, gpu.rptSpatialGroup0!);
    dispatch('vitrum.pt-webgpu.restirPt.resolve', gpu.rptResolvePipeline!, gpu.rptResolveGroup0!);
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
    const texture = colorSpace === 'output' ? gpu.presentTexture : gpu.accumTexture;
    if (texture == null) return null;
    const width = gpu.accumWidth;
    const height = gpu.accumHeight;
    if (width <= 0 || height <= 0) return null;
    const rgba = await readRgba16fTextureToF32(this.#device, texture, width, height);
    if (rgba == null) return null;
    return { width, height, rgba };
  }

  /**
   * EXPERIMENTAL — the ReSTIR-PT reconnection-indirect result buffer (one vec4f /
   * full-res pixel: .rgb = reconnection indirect HDR, .a = contributing flag), or
   * `null` when the engine was not built with `restirPtReuse: true` or no frame
   * has rendered yet. This is a SEPARATE debug output. When the composite pipeline
   * is active (`rptCompositePipeline` + `rptProducerGroup0` are both non-null), the
   * composite megakernel folds the ReSTIR-PT resolve into the beauty accumulator;
   * the beauty image already contains the composited result. Hosts read this buffer
   * back via `copyBufferToBuffer` → a MAP_READ staging buffer (the buffer carries
   * `COPY_SRC`).
   *
   * Available only when `capabilities.experimentalFeatures` has
   * `'pt-webgpu-restir-pt-reuse'`.
   */
  getRestirPtResultBuffer(): GPUBuffer | null {
    this.#assertUsable('getRestirPtResultBuffer');
    return this.#restirPtReuse ? this.#gpu.rptResultBuffer : null;
  }

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    if (this.#slot.get() === 'error') {
      throw new Error('reset: engine is in a fatal error state');
    }
    this.#postDenoiser?.invalidate();
    this.#samplesAccumulated = 0;
    this.#gpu.clearAccumBuffer();
    // Item 2e — clear ReSTIR-PT reservoir history so stale temporal samples from
    // the previous scene do not bleed into the new one. No-op if not allocated.
    this.#gpu.clearReservoirBuffers();
    // A4-progressive — reset per-pixel SPPM statistics (τ/R²/N → 0) so the
    // progressive estimate restarts from scratch when the view resets. SPPM's
    // convergence requires a static eye point; resetting on camera-move or
    // setScene maintains that invariant. No-op if the buffer is not allocated.
    this.#gpu.clearSppmPixelStats();
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
    if (!this.#gpu.ensureAccumResources(width, height)) {
      this.#gpu.clearAccumBuffer();
    }
    this.#samplesAccumulated = 0;
    this.#postDenoiser?.invalidate();
    // Write the decaying prior. `weight` (virtual samples) is deliberately NOT
    // added to `#samplesAccumulated` — see the method doc.
    this.#gpu.seedAccumBuffer(seedTex, opts.weight, width, height);
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

  // ── Inverse rendering (differentiable RT) — WS5 ──────────────────────────
  //
  // Phase 0 (finite-difference) + the validated Phase-1 BSDF adjoint oracle.
  // The session owns the optimization loop; the host owns the cadence (it calls
  // session.step() at its own pace). The session re-renders the SAME view as
  // the most-recent renderFrame with a FROZEN RNG seed so perturbations differ
  // only in the perturbed parameter (path replay's frozen-RNG discipline).
  //
  // Ref: Vicini 2021 (Path Replay Backpropagation); Nimier-David 2020
  //      (Radiative Backpropagation).
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
        this.updatePrimitive(primitiveId, { material: patch } as Partial<ScenePrimitive>);
      },
      patchEmitter: (emitterId: string, patch: Partial<SceneEmitter>) => {
        this.updateEmitter(emitterId, patch);
      },
      getPathReplayRenderContext: () => ({
        bounces: this.#activeBounces,
        spectral: this.#spectralEnabled,
      }),
      computeAdjointGradient: (req) => this.#computeAdjointGradient(req),
    };
    return new PtWebgpuInverseSession(hooks, opts);
  }

  /**
   * WS5 Phase-1 path-replay adjoint pass (the `computeAdjointGradient` hook). One
   * dispatch of `PT_WEBGPU_ADJOINT_PASS_WGSL` over the live scene buffers: per
   * pixel it re-traces the frozen-seed primary ray (brute-force closest-hit) and
   * accumulates `∂loss/∂θ` for the optimized material params through the
   * GPU-validated partials + fixed-point `adjointScatter`:
   *  - baseColor / roughness — single-bounce directional + point + spot +
   *    stochastic area-measure rect/disc/mesh-area direct-light NEE
   *    (the BRDF partials `dBrdf_dBaseColor` / `dBrdf_dRoughness`);
   *  - emissive — the camera-DIRECT emission at the primary hit (NOT a NEE term):
   *    `∂loss/∂emissive_c = dLoss_dR_c · emissiveIntensity` (dContribution_dEmissive
   *    with throughput = 1). The packed material folds intensity into emissive.rgb,
   *    so the fixed emissiveIntensity rides in the descriptor's `.w` (bitcast f32).
   * Returns the flat gradient. Replaces the session's N-render FD probe loop with
   * one baseline render + this pass.
   *
   * The per-hit shading partials + the chain-rule accumulation are GPU-validated
   * (adjoint-validate.ts / adjoint-fd-validate.ts); the assembled pass is A/B'd
   * against the FD gradient the session also computes — baseColor/roughness via
   * `v24-inverse-fit.mjs`, emissive via `v24-emissive-fit.mjs` (sign-match +
   * convergence on lavapipe, V24).
   */
  /**
   * WS5 Phase-1 path-replay adjoint pass — thin delegate to AdjointPass (D8.10).
   * AdjointPass owns pipeline compilation, transient buffer lifecycle (H14-D
   * adjointCreated/adjointDestroyed accounting), UBO packing, dispatch, and
   * readback. The engine lazily creates and retains one AdjointPass instance.
   */
  async #computeAdjointGradient(req: AdjointGradientRequest): Promise<Float32Array> {
    this.#assertLive('computeAdjointGradient');
    const sb = this.#sceneBuffers;
    const last = this.#lastFrameInput;
    if (sb == null || last == null || this.#scene == null) {
      throw new Error('computeAdjointGradient: no scene/camera (render at least one frame first).');
    }
    if (this.#adjointPass == null) {
      this.#adjointPass = new AdjointPass(this.#device);
    }
    return this.#adjointPass.computeGradient(
      req,
      sb,
      last,
      this.#scene,
      this.#supportedAnalyticShapes(),
      materialIndexForPrimitive,
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

  /** WG-7 — supply host-owned light-path buffer (or null to use internal CPU fill). */
  bdptAdvanceFrame(lightPathBuffer: GPUBuffer | null): void {
    this.#assertUsable('bdptAdvanceFrame');
    if (!this.#bdpt) return;
    this.#bdptExternalBuffer = lightPathBuffer;
    // H9: instead of nulling group 2 (which broke rendering because
    // buildBindGroups returns early when group 0 is still cached, leaving
    // group 2 null for the rest of the frame), reconstruct ONLY group 2
    // against the new light-path buffer while groups 0/1/3 remain valid.
    // A pointer-equality fast-out inside rebuildGroup2Only skips the
    // createBindGroup call when the same buffer is re-supplied unchanged.
    // If scene buffers are not yet uploaded (pre-setScene), rebuildGroup2Only
    // is a no-op (bindGroupLayout2 is null); the regular buildBindGroups
    // path in the next renderFrame will build all groups correctly.
    if (this.#sceneBuffers != null) {
      const buf = lightPathBuffer ?? this.#bdptLightPathBuffer();
      this.#gpu.rebuildGroup2Only(this.#sceneBuffers, buf);
    }
  }

  #bdptLightPathBuffer(): GPUBuffer {
    if (this.#bdptExternalBuffer) return this.#bdptExternalBuffer;
    if (this.#bdptLightPath) return this.#bdptLightPath.buffer;
    if (!this.#bdptPlaceholderBuffer && typeof this.#device.createBuffer === 'function') {
      this.#bdptPlaceholderBuffer = createBdptLightPathPlaceholder(this.#device);
    }
    return this.#bdptPlaceholderBuffer as GPUBuffer;
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
    this.#postDenoiser?.dispose();
    this.#bdptLightPath?.dispose();
    this.#bdptLightPath = null;
    this.#bdptPlaceholderBuffer?.destroy();
    this.#bdptPlaceholderBuffer = null;
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
 * Deliberately minimal: only the converged-denoise read-back is stable host
 * API. The off-default / experimental seams (`getRestirPtResultBuffer`,
 * `bdptAdvanceFrame`) are NOT promised here — ReSTIR-PT reuse + the BDPT
 * light-subpath wiring are inert-by-default research paths, and exposing their
 * raw `GPUBuffer` plumbing in a stable surface would over-promise. The
 * concrete engine class stays unexported, so those remain internal.
 */
export interface PTEngineWebGPUSurface {
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
  if (opts.device == null || typeof (opts.device).createCommandEncoder !== 'function') {
    throw new TypeError(
      'createPTEngine_WebGPU: device must be a GPUDevice instance',
    );
  }
  const maxBounces = opts.maxBounces;
  if (maxBounces !== undefined && maxBounces < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxBounces structural cap must be >= 1 (got ${maxBounces})`,
    );
  }
  const maxSpp = opts.maxSamplesPerPixel;
  if (maxSpp !== undefined && maxSpp < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxSamplesPerPixel structural cap must be >= 1 (got ${maxSpp})`,
    );
  }
  if (maxBounces !== undefined && maxBounces > EXPERIMENTAL_MAX_BOUNCES) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.max-bounces-clamped',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message: `[vitrum/pt-webgpu] maxBounces=${maxBounces} requested, clamping to experimental limit ${EXPERIMENTAL_MAX_BOUNCES}.`,
      details: { requested: maxBounces, clampedTo: EXPERIMENTAL_MAX_BOUNCES },
    });
  }
  const bdptMaxLightBounces = opts.bdptOptions?.maxLightBounces;
  if (
    bdptMaxLightBounces !== undefined &&
    (!Number.isFinite(bdptMaxLightBounces) || bdptMaxLightBounces < 1)
  ) {
    throw new RangeError(
      `createPTEngine_WebGPU: bdptOptions.maxLightBounces must be a finite number >= 1 (got ${bdptMaxLightBounces})`,
    );
  }
  if (bdptMaxLightBounces !== undefined && bdptMaxLightBounces > BDPT_MAX_LIGHT_BOUNCES) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.bdpt-max-light-bounces-clamped',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `clamping to supported BDPT light-subpath limit ${BDPT_MAX_LIGHT_BOUNCES}.`,
      details: { requested: bdptMaxLightBounces, clampedTo: BDPT_MAX_LIGHT_BOUNCES },
    });
  }
  if (
    bdptMaxLightBounces !== undefined &&
    bdptMaxLightBounces <= BDPT_MAX_LIGHT_BOUNCES &&
    !Number.isInteger(bdptMaxLightBounces)
  ) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.bdpt-max-light-bounces-rounded',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `rounding down to integer ${resolveBdptMaxLightBounces(bdptMaxLightBounces)}.`,
      details: {
        requested: bdptMaxLightBounces,
        roundedTo: resolveBdptMaxLightBounces(bdptMaxLightBounces),
      },
    });
  }
  const resolvedBdptMaxLightBounces = resolveBdptMaxLightBounces(bdptMaxLightBounces);
  if (opts.bdpt === true && resolvedBdptMaxLightBounces > BDPT_SAFE_DEFAULT_LIGHT_BOUNCES) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.bdpt-multivertex-research-mode',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] bdptOptions.maxLightBounces=${resolvedBdptMaxLightBounces} enables the ` +
        'multi-vertex BDPT research path. Current radiometric evidence shows this path is not yet ' +
        'weighted against the regular eye-path strategy, so it is explicitly opt-in; omit ' +
        'bdptOptions.maxLightBounces for the endpoint-only radiometrically neutral default.',
      details: {
        requested: bdptMaxLightBounces,
        resolved: resolvedBdptMaxLightBounces,
        safeDefault: BDPT_SAFE_DEFAULT_LIGHT_BOUNCES,
      },
    });
  }
  if (
    opts.denoiser != null &&
    opts.denoiser !== 'none' &&
    opts.denoiser !== 'oidn-final'
  ) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.unsupported-denoiser',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] denoiser="${opts.denoiser}" requested, but only 'none' and 'oidn-final' are wired. ` +
        "pt-webgpu is a converged progressive path tracer; 'svgf-real' is a real-time 1-spp filter and is unsupported here — " +
        "use 'oidn-final' for converged denoising. Degrading to no-denoise.",
      details: { requested: opts.denoiser },
    });
  }
  // H51-C: warn once listing any extensions keys the host supplied that are
  // either (a) graduated legacy keys that no longer do anything, or (b) truly
  // unknown keys. In both cases the key is silently ignored at runtime; the
  // warn ensures the host is aware that migration is needed.
  //
  // pt-webgpu graduated its formerly-experimental extensions in 2025:
  //   vitrum.ptWebgpu.spectralHeroWavelength.*  →  opts.spectral (boolean)
  //   vitrum.ptWebgpu.bdpt.*                    →  opts.bdpt (boolean) + opts.bdptOptions
  //   vitrum.ptWebgpu.oidn.*                    →  opts.denoiser:'oidn-final' + opts.oidn
  if (opts.extensions != null) {
    const GRADUATED_KEY_MIGRATION: Record<string, string> = {
      'vitrum.ptWebgpu.spectralHeroWavelength':
        "opts.spectral (boolean) — set spectral: true to enable hero-wavelength spectral transport",
      'vitrum.ptWebgpu.bdpt':
        "opts.bdpt (boolean) + opts.bdptOptions — set bdpt: true to enable bidirectional path tracing",
      'vitrum.ptWebgpu.oidn':
        "opts.denoiser: 'oidn-final' + opts.oidn: { modelUrl } — pass denoiser:'oidn-final' with an OIDN model URL",
    };
    const allKeys = Object.keys(opts.extensions);
    for (const [prefix, migration] of Object.entries(GRADUATED_KEY_MIGRATION)) {
      const matchingKeys = allKeys.filter((k) => k.startsWith(prefix));
      if (matchingKeys.length > 0) {
        emitPteWarning(opts, {
          code: 'pt-webgpu.graduated-extension-key',
          backend: 'pt-webgpu',
          phase: 'construction',
          method: 'createPTEngine_WebGPU',
          message:
            `[vitrum/pt-webgpu] Extension key(s) ${matchingKeys.map((k) => JSON.stringify(k)).join(', ')} ` +
            `are no longer consumed — this key graduated to a first-class option. ` +
            `Replace with: ${migration}.`,
          details: { keys: matchingKeys, migration },
        });
      }
    }
    const unknownKeys = allKeys.filter(
      (k) => !Object.keys(GRADUATED_KEY_MIGRATION).some((prefix) => k.startsWith(prefix)),
    );
    if (unknownKeys.length > 0) {
      emitPteWarning(opts, {
        code: 'pt-webgpu.unknown-extension-key',
        backend: 'pt-webgpu',
        phase: 'construction',
        method: 'createPTEngine_WebGPU',
        message:
          `[vitrum/pt-webgpu] Unknown extensions keys will be ignored: ${unknownKeys.map((k) => JSON.stringify(k)).join(', ')}. ` +
          "pt-webgpu's stable extensions (spectral, bdpt, oidn, restirPtReuse) are now first-class " +
          'named options. Check the PTEngineWebGPUOptions interface for the current option set.',
        details: { keys: unknownKeys },
      });
    }
  }

  const traceTier = resolvePtWebgpuTraceTier(opts.device, opts.traceTier);
  if (opts.restirPtReuse === true) {
    assertRestirPtReuseSupported(opts.device, traceTier);
  }
  if (traceTier === 'full') {
    console.info(
      '[vitrum/pt-webgpu] Full trace tier: TLAS, analytic shapes, HDRI, area lights, motion/variance aux, caustics.',
    );
  } else {
    emitPteWarning(opts, {
      code: 'pt-webgpu.lite-tier',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        '[vitrum/pt-webgpu] Lite trace tier (software-adapter fallback): merged mesh-like BVH, directional/point/spot/rect-area/disc-area emitters, HDRI and procedural-sky environments. ' +
        'Disabled on lite: analytic shapes, TLAS, mesh-area emitters, caustics, BDPT, and motion/variance aux buffers. ' +
        `On a discrete GPU host, request a device with maxStorageBuffersPerShaderStage >= ${PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE} and maxStorageTexturesPerShaderStage >= 5, or pass traceTier: "full" after verifying limits.`,
      details: { traceTier },
    });
  }
  const slot = makeStateSlot();
  const engine = new PTEngineWebGPU(opts, slot, traceTier);
  slot.set('ready');
  return engine;
};

function assertRestirPtReuseSupported(device: GPUDevice, traceTier: PtWebgpuTraceTier): void {
  if (traceTier !== 'full') {
    throw new Error(
      'createPTEngine_WebGPU: restirPtReuse requires traceTier "full"; the selected lite tier cannot bind the ReSTIR-PT reuse reservoirs.',
    );
  }
  const maxBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (maxBuffers < PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
    throw new Error(
      `createPTEngine_WebGPU: restirPtReuse requires maxStorageBuffersPerShaderStage >= ` +
      `${PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE}; device exposes ${maxBuffers}. ` +
      'Request the ReSTIR-PT reuse limit floor when acquiring the GPUDevice.',
    );
  }
}
