import type {
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineFactory,
  EngineOptions,
  EngineState,
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
import { asBackendTexture, asMat4 } from '@vitrum/core';
import {
  PtWebgpuInverseSession,
  type InverseEngineHooks,
  type AdjointGradientRequest,
} from './inverse/inverseSession.js';
import { materialIndexForPrimitive } from './scene/incrementalPatch.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import {
  PT_WEBGPU_ADJOINT_PASS_WGSL,
  ADJOINT_PARAMS_UBO_BYTES,
  ADJOINT_FIELD_ROUGHNESS,
} from './wgsl/pathTrace/adjointPass.wgsl.js';
import { ADJOINT_GRAD_FP } from './wgsl/pathTrace/pathTraceAdjoint.wgsl.js';
import { readOidnInputsFromTextures } from './denoise/rgba16fReadback.js';
import { summarizeScene, type SceneSummary } from './scene/flattenScene.js';
import {
  buildPackedScene,
  scenePackResultFromPacked,
  uploadPackedScene,
  PT_WEBGPU_ANALYTIC_SHAPES,
  PT_WEBGPU_SUPPORT,
  type UploadedSceneBuffers,
} from './scene/uploadSceneBuffers.js';
import { type ScenePackResult } from '@vitrum/shared-bvh';
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
import { PT_WEBGPU_COMMON_WGSL } from './wgsl/common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  OCTAHEDRAL_CORE_WGSL,
} from '@vitrum/shared-samplers';

export { PT_WEBGPU_COMMON_WGSL, HAMMERSLEY_WGSL, OCTAHEDRAL_CORE_WGSL };
export {
  PT_WEBGPU_REQUIRED_LIMITS,
  PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
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
   * supports ≥10 storage buffers per bind group and ≥5 storage textures (split
   * full layout: TLAS, HDRI, area lights, caustics, etc.); `lite` on 8/4 adapters.
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
  /** BDPT tuning — read only when {@link bdpt} is `true`. */
  readonly bdptOptions?: {
    /** Max light-subpath bounces, clamped 1–3. Default 3. */
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
   * NEE on receiving surfaces. ON by default. `sceneFromThreeJS` converts an
   * emissive mesh into a mesh-area light emitter and zeroes the primitive's
   * emissive (so it is sampled, not double-counted); this re-attaches that
   * emitter radiance onto the primitive's material at pack time so the path
   * tracer's emissive-on-hit term fires. The existing BSDF↔light MIS
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
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const WORKGROUP_SIZE = 8;

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
  /** WS5 Phase-1 — lazily-built path-replay adjoint compute pipeline (the
   *  computeAdjointGradient hook). Engine-owned; freed in dispose. */
  #adjointPipeline: GPUComputePipeline | null = null;

  /**
   * The cohesive GPU-resource-lifecycle cluster (T14-followup extraction): accum
   * + aux textures and views, accum / varianceMoments / params buffers, the
   * compute pipeline(s), the group-0 bind-group layout, the cached per-frame bind
   * groups, and the accum dims. The engine owns exactly one instance.
   */
  readonly #gpu: GpuResources;
  #onFrameSubs = new Set<(stats: FrameStats) => void>();
  #onProgressSubs = new Set<(progress: ProgressStats) => void>();
  readonly #postDenoiser: OIDNFinalDispatcher | null;
  readonly #spectralEnabled: boolean;
  readonly #lightTreeImportanceSampling: boolean;
  readonly #cameraVisibleEmitters: boolean;
  readonly #bdpt: boolean;
  readonly #bdptMaxLightBounces: number;
  #bdptLightPath: BdptLightPathBufferWebGPU | null = null;
  #bdptExternalBuffer: GPUBuffer | null = null;
  #bdptPlaceholderBuffer: GPUBuffer | null = null;

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
    this.#spectralEnabled = opts.spectral === true;
    this.#lightTreeImportanceSampling = opts.lightTreeImportanceSampling !== false;
    this.#cameraVisibleEmitters = opts.cameraVisibleEmitters !== false;
    this.#traceTier = traceTier;
    this.#maxBouncesLimit = Math.max(1, Math.min(opts.maxBounces ?? 3, EXPERIMENTAL_MAX_BOUNCES));
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    const causticOpts = opts.causticOptions ?? {};
    const mneeIter = typeof causticOpts.mneeMaxIterations === 'number' ? causticOpts.mneeMaxIterations : 8;
    const mneeChain = typeof causticOpts.mneeMaxChainLength === 'number' ? causticOpts.mneeMaxChainLength : 3;
    this.#mneeMaxIterations = Math.max(1, mneeIter);
    this.#mneeMaxChainLength = Math.max(1, mneeChain);
    this.#bdpt = opts.bdpt === true;
    const requestedBdptBounces = opts.bdptOptions?.maxLightBounces;
    this.#bdptMaxLightBounces =
      typeof requestedBdptBounces === 'number' && requestedBdptBounces >= 1
        ? Math.min(3, Math.floor(requestedBdptBounces))
        : 3;
    this.#gpu = new GpuResources(opts.device, traceTier, this.#bdpt);
    if (opts.denoiser === 'oidn-final') {
      const modelUrl = opts.oidn?.modelUrl;
      const eps = opts.oidn?.executionProviders?.filter(
        (p) => p === 'webnn' || p === 'webgpu' || p === 'wasm',
      );
      if (typeof modelUrl !== 'string' || modelUrl.length === 0) {
        throw new Error(
          "createPTEngine_WebGPU: denoiser: 'oidn-final' requires " +
            'oidn: { modelUrl } (a non-empty string). ' +
            'Use oidn_rt_hdr_alb_nrm.onnx when supplying albedo + normal aux.',
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
      repackScene: (scene, opts) => this.#repackScene(scene, opts),
      setScene: (scene) => this.setScene(scene),
      reset: () => this.reset(),
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
      incrementalPatchSupport: {
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
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      // Advertised support is derived from the SAME `PT_WEBGPU_SUPPORT` sets the
      // scene packer partitions against (uploadSceneBuffers.ts), so the declared
      // capability and the ingestion behavior can no longer drift. Copy into
      // fresh Sets so a host mutating the returned capability object can't
      // corrupt the packer's source of truth.
      supportedAnalyticShapes: new Set(PT_WEBGPU_SUPPORT.supportedAnalyticShapes),
      supportedEmitterKinds: new Set(PT_WEBGPU_SUPPORT.supportedEmitterKinds),
      supportedPrimitiveKinds: new Set(PT_WEBGPU_SUPPORT.supportedPrimitiveKinds),
      supportedEnvironmentKinds: new Set(PT_WEBGPU_SUPPORT.supportedEnvironmentKinds),
      presentationMode: 'offscreen-texture',
      experimentalFeatures: new Set([
        'experimental-backend',
        ...(this.#traceTier === 'lite' ? (['pt-webgpu-lite-tier'] as const) : []),
        ...(this.#postDenoiser instanceof OIDNFinalDispatcher
          ? (['pt-webgpu-oidn-final'] as const)
          : []),
        ...(this.#bdpt ? (['pt-webgpu-bdpt'] as const) : []),
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

  // ── Debug introspection (T3.G followup) ────────────────────────────────
  // Experimental backend exposes only the GPU-memory estimate for now — atlas
  // / BVH / pick / denoiser-toggle hooks are walkaround-hybrid concepts
  // that don't apply to a brute-force compute path tracer.
  readonly debug: EngineDebugSurface = {
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
      // Scene buffers (BVH, materials, indices, etc.) are owned by
      // UploadedSceneBuffers; size accounting there would require touching
      // the uploader — left for a follow-up so this commit stays focused.

      const commonTexBytes = accumBytes + normalDepthBytes + albedoBytes
        + varianceBytes + motionBytes;
      const commonBufBytes = accumBufBytes + varMomentBufBytes + paramsBufBytes;
      const total = commonTexBytes + commonBufBytes;

      return Object.freeze({
        total,
        byCategory: Object.freeze({
          common: total,
        }),
        byTextureFormat: Object.freeze({
          rgba16float: commonTexBytes,
        }),
        byBufferUsage: Object.freeze({
          storage: accumBufBytes + varMomentBufBytes,
          uniform: paramsBufBytes,
        }),
      });
    },
  };

  #assertLive(method: string): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error(`${method}: engine is disposed`);
    }
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
    return {
      kind: 'rendered',
      primaryRadiance: asBackendTexture<'webgpu', GPUTexture>(primary),
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
    const frameEndMs = globalThis.performance?.now?.() ?? Date.now();
    const mem = this.debug.estimatedGpuMemoryBytes?.() ?? null;
    this.#emitFrameStats({
      frameTimeMs: Math.max(0, frameEndMs - frameStartMs),
      spp,
      ...(mem != null ? { gpuMemoryBytes: mem, estimatedGpuMemoryBytes: mem.total } : {}),
    });
    this.#emitProgress({
      kind: 'pt-spp',
      current,
      target,
      fraction: target > 0 ? Math.max(0, Math.min(1, current / target)) : 1,
    });
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
    if (this.#slot.get() === 'disposed') {
      throw new Error('setScene: engine is disposed');
    }
    this.#repackScene(scene, { warnOnEmpty: true });
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
    const packed = buildPackedScene(scene, {
      cameraVisibleEmitters: this.#cameraVisibleEmitters,
    });
    this.#geoPack = scenePackResultFromPacked(packed);
    this.#sceneBuffers?.destroy();
    this.#sceneBuffers = uploadPackedScene(this.#device, packed);
    this.#bdptLightPath?.dispose();
    this.#bdptLightPath = null;
    if (this.#bdpt && this.#traceTier === 'full') {
      this.#bdptLightPath = new BdptLightPathBufferWebGPU(this.#device, {
        maxLightBounces: this.#bdptMaxLightBounces,
      });
    }
    this.#gpu.invalidateBindGroups();
    this.#scene = scene;
    if (opts.warnOnEmpty) {
      const sceneSummary = summarizeScene(scene);
      if (sceneSummary.primitiveCount === 0) {
        console.warn('[vitrum/pt-webgpu] Empty scene provided; rendering sky-only fallback.');
      }
    }
    for (const warning of packed.warnings) {
      console.warn(`[vitrum/pt-webgpu] ${warning}`);
    }
    this.reset();
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

  renderFrame(input: FrameInput): FrameOutput {
    this.#assertLive('renderFrame');
    if (!this.#inInverseRender) {
      this.#lastFrameInput = input;
    }
    const frameStartMs = globalThis.performance?.now?.() ?? Date.now();

    const gpu = this.#gpu;

    if (this.#slot.get() === 'paused') {
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
    if (
      gpu.accumView == null ||
      gpu.normalDepthView == null ||
      gpu.albedoView == null ||
      gpu.varianceView == null ||
      (this.#traceTier === 'full' && gpu.motionVectorsView == null) ||
      gpu.accumBuffer == null ||
      (this.#traceTier === 'full' && gpu.varianceMomentsBuffer == null) ||
      gpu.paramsBuffer == null ||
      gpu.computePipeline == null ||
      gpu.bindGroupLayout == null ||
      this.#sceneBuffers == null
    ) {
      throw new Error('renderFrame: failed to initialize WebGPU pipeline resources');
    }

    const accumTexture = gpu.accumTexture;
    if (accumTexture != null && this.#samplesAccumulated >= targetSpp) {
      const output = this.#frameOutput(accumTexture, this.#samplesAccumulated, true);
      this.#emitFrameTelemetry(frameStartMs, 0, this.#samplesAccumulated, targetSpp);
      return output;
    }

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
    this.#device.queue.writeBuffer(gpu.paramsBuffer, 0, paramsArrayBuffer);

    const bindGroup = gpu.buildBindGroups(this.#sceneBuffers, () => this.#bdptLightPathBuffer());

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.pt-webgpu.pathTrace.encoder' });
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

    const pass = encoder.beginComputePass({ label: 'vitrum.pt-webgpu.pathTrace.pass' });
    pass.setPipeline(gpu.computePipeline);
    pass.setBindGroup(0, bindGroup);
    if (this.#traceTier === 'full' && gpu.pathTraceBindGroup1 != null && gpu.pathTraceBindGroup2 != null) {
      pass.setBindGroup(1, gpu.pathTraceBindGroup1);
      pass.setBindGroup(2, gpu.pathTraceBindGroup2);
      if (gpu.pathTraceBindGroup3 != null) {
        pass.setBindGroup(3, gpu.pathTraceBindGroup3);
      }
    }
    pass.dispatchWorkgroups(
      Math.ceil(width / WORKGROUP_SIZE),
      Math.ceil(height / WORKGROUP_SIZE),
      1,
    );
    pass.end();
    this.#device.queue.submit([encoder.finish()]);

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

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#postDenoiser?.invalidate();
    this.#samplesAccumulated = 0;
    this.#gpu.clearAccumBuffer();
  }

  /**
   * OIDN denoised RGB for the current converged cohort (WG-1), or null while
   * inference is in flight / engine was not built with `denoiser: 'oidn-final'`.
   */
  getDenoisedFrame(): DenoisedFrame | null {
    return this.#postDenoiser?.getLatestDenoised() ?? null;
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
      computeAdjointGradient: (req) => this.#computeAdjointGradient(req),
    };
    return new PtWebgpuInverseSession(hooks, opts);
  }

  /**
   * WS5 Phase-1 path-replay adjoint pass (the `computeAdjointGradient` hook). One
   * dispatch of `PT_WEBGPU_ADJOINT_PASS_WGSL` over the live scene buffers: per
   * pixel it re-traces the frozen-seed primary ray (brute-force closest-hit),
   * re-derives the single-bounce point-light direct lighting, and accumulates
   * `∂loss/∂θ` for the optimized material params through the GPU-validated BRDF
   * partials + fixed-point `adjointScatter`. Returns the flat gradient. Replaces
   * the session's N-render FD probe loop with one baseline render + this pass.
   *
   * The per-hit shading partials + the chain-rule accumulation are GPU-validated
   * (adjoint-validate.ts / adjoint-fd-validate.ts); the assembled pass is A/B'd
   * against the FD gradient the session also computes (V24).
   */
  async #computeAdjointGradient(req: AdjointGradientRequest): Promise<Float32Array> {
    const device = this.#device;
    const sb = this.#sceneBuffers;
    const last = this.#lastFrameInput;
    if (sb == null || last == null || this.#scene == null) {
      throw new Error('computeAdjointGradient: no scene/camera (render at least one frame first).');
    }
    const { width, height, channels, params, gradientLength, dLoss_dRendered } = req;

    if (this.#adjointPipeline == null) {
      const module = device.createShaderModule({ code: PT_WEBGPU_ADJOINT_PASS_WGSL });
      this.#adjointPipeline = device.createComputePipeline({
        label: 'vitrum.pt-webgpu.adjointPass',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    const pipeline = this.#adjointPipeline;

    // AdjointParams UBO: invViewProj(mat4) + cameraPos(vec4) + 2×uvec4 of counts.
    const vp = multiplyMat4(last.projMatrix, last.viewMatrix);
    const invVp = invertMat4(asMat4(vp));
    if (invVp == null) {
      throw new Error('computeAdjointGradient: camera viewProj is not invertible.');
    }
    const ubo = new ArrayBuffer(ADJOINT_PARAMS_UBO_BYTES);
    const uboF = new Float32Array(ubo);
    const uboU = new Uint32Array(ubo);
    uboF.set(invVp, 0); // mat4x4f at byte 0 (16 floats)
    uboF[16] = last.cameraPosition[0];
    uboF[17] = last.cameraPosition[1];
    uboF[18] = last.cameraPosition[2];
    uboF[19] = 1;
    uboU[20] = width >>> 0;
    uboU[21] = height >>> 0;
    uboU[22] = sb.triangleCount >>> 0;
    uboU[23] = sb.pointLightCount >>> 0;
    uboU[24] = params.length >>> 0;
    uboU[25] = channels >>> 0;

    // adjointParamDescs: per param {matId, fieldCode, gradOffset, _}.
    const descs = new Uint32Array(Math.max(params.length, 1) * 4);
    for (let i = 0; i < params.length; i++) {
      const p = params[i]!;
      const matId = materialIndexForPrimitive(this.#scene, p.id, this.#supportedAnalyticShapes());
      if (matId == null) {
        throw new Error(`computeAdjointGradient: no material index for primitive "${p.id}".`);
      }
      descs[i * 4 + 0] = matId >>> 0;
      descs[i * 4 + 1] = p.field === 'roughness' ? ADJOINT_FIELD_ROUGHNESS : 0;
      descs[i * 4 + 2] = p.offset >>> 0;
    }

    const U = (globalThis as { GPUBufferUsage: typeof GPUBufferUsage }).GPUBufferUsage;
    const mk = (size: number, usage: number, data?: ArrayBufferView): GPUBuffer => {
      const b = device.createBuffer({ size: Math.max(size, 16), usage });
      if (data) device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
      return b;
    };
    const paramsBuf = mk(ADJOINT_PARAMS_UBO_BYTES, U.UNIFORM | U.COPY_DST, uboF);
    const dLossBuf = mk(dLoss_dRendered.byteLength, U.STORAGE | U.COPY_DST, dLoss_dRendered);
    const gradBuf = mk(gradientLength * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST, new Int32Array(gradientLength));
    const descBuf = mk(descs.byteLength, U.STORAGE | U.COPY_DST, descs);
    const stage = mk(gradientLength * 4, U.MAP_READ | U.COPY_DST);

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: sb.positionsBuffer } },
        { binding: 2, resource: { buffer: sb.indicesBuffer } },
        { binding: 3, resource: { buffer: sb.triMaterialIdsBuffer } },
        { binding: 4, resource: { buffer: sb.materialsBuffer } },
        { binding: 5, resource: { buffer: sb.normalsBuffer } },
        { binding: 6, resource: { buffer: sb.pointLightsBuffer } },
        { binding: 7, resource: { buffer: dLossBuf } },
        { binding: 8, resource: { buffer: gradBuf } },
        { binding: 9, resource: { buffer: descBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    enc.copyBufferToBuffer(gradBuf, 0, stage, 0, gradientLength * 4);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync((globalThis as { GPUMapMode: typeof GPUMapMode }).GPUMapMode.READ);
    const raw = new Int32Array(stage.getMappedRange().slice(0));
    stage.unmap();

    const grad = new Float32Array(gradientLength);
    for (let i = 0; i < gradientLength; i++) grad[i] = raw[i]! / ADJOINT_GRAD_FP;

    // Per-step transient buffers — free them (no leak; the pipeline is cached).
    for (const b of [paramsBuf, dLossBuf, gradBuf, descBuf, stage]) b.destroy();
    return grad;
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
    if (!this.#bdpt) return;
    this.#bdptExternalBuffer = lightPathBuffer;
    // Drop only group 2 (the BDPT light-path group), matching the prior inline
    // behavior: group 0 stays cached, so the build branch in renderFrame won't
    // fire and group 2 remains null until a full scene/accum invalidation.
    this.#gpu.pathTraceBindGroup2 = null;
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
    this.#adjointPipeline = null; // GPUComputePipeline has no destroy(); drop the ref
    this.#scene = null;
    this.#geoPack = null;
    this.#onFrameSubs.clear();
    this.#onProgressSubs.clear();
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
}

export const createPTEngine_WebGPU: EngineFactory<PTEngineWebGPUOptions> = async (
  opts: PTEngineWebGPUOptions,
): Promise<Engine> => {
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
    console.warn(
      `[vitrum/pt-webgpu] maxBounces=${maxBounces} requested, clamping to experimental limit ${EXPERIMENTAL_MAX_BOUNCES}.`,
    );
  }
  if (
    opts.denoiser != null &&
    opts.denoiser !== 'none' &&
    opts.denoiser !== 'oidn-final'
  ) {
    console.warn(
      `[vitrum/pt-webgpu] denoiser="${opts.denoiser}" requested, but only 'none' and 'oidn-final' are wired. ` +
        "pt-webgpu is a converged progressive path tracer; 'svgf-real' is a real-time 1-spp filter and is unsupported here — " +
        "use 'oidn-final' for converged denoising. Degrading to no-denoise.",
    );
  }
  const traceTier = resolvePtWebgpuTraceTier(opts.device, opts.traceTier);
  if (traceTier === 'full') {
    console.info(
      '[vitrum/pt-webgpu] Full trace tier: TLAS, analytic shapes, HDRI, area lights, motion/variance aux, caustics.',
    );
  } else {
    console.warn(
      '[vitrum/pt-webgpu] Lite trace tier (software-adapter fallback): merged-mesh BVH, directional + procedural sky only. ' +
        'Analytic shapes, TLAS, HDRI, area lights, and caustics are disabled. ' +
        'On a discrete GPU host, request a device with ≥10 storage buffers per group and ≥5 storage textures, or pass traceTier: "full" after verifying limits.',
    );
  }
  const slot = makeStateSlot();
  const engine = new PTEngineWebGPU(opts, slot, traceTier);
  slot.set('ready');
  return engine;
};
