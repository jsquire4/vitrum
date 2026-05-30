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
} from './inverse/inverseSession.js';
import { readOidnInputsFromTextures } from './denoise/rgba16fReadback.js';
import { summarizeScene, type SceneSummary } from './scene/flattenScene.js';
import {
  applyEmitterCountMutation,
  applyEnvironmentMutation,
  rebuildLightTreeForScene,
  buildPackedScene,
  rebuildTlasForSceneTransforms,
  scenePackResultFromPacked,
  uploadPackedScene,
  uploadScenePackGeometry,
  uploadScenePackGeometryRealloc,
  uploadScenePackBlasOnly,
  uploadScenePackTlasOnly,
  uploadScenePackTlasRealloc,
  PT_WEBGPU_ANALYTIC_SHAPES,
  PT_WEBGPU_SUPPORT,
  type UploadedSceneBuffers,
} from './scene/uploadSceneBuffers.js';
import {
  analyticIndexForPrimitive,
  canFastPathGeometryPatch,
  canFastPathInstancedTopologyPatch,
  canFastPathMaterialPatch,
  canFastPathTopologyResizePatch,
  canFastPathTransformPatch,
  canReuseTlasBufferLengths,
  materialIndexForPrimitive,
} from './scene/incrementalPatch.js';
import {
  fingerprintTlasBuffers,
  rebuildPrimitiveBlas,
  rebuildTlasReuseBlas,
  type ScenePackResult,
} from '@vitrum/shared-bvh';
import { patchEmitterInScene, patchPrimitiveInScene } from './scene/patchScene.js';
import { X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL } from '@vitrum/shared-samplers';
import { FrameParamsSlot } from './scene/frameParamsLayout.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import { MATERIAL_FLOAT_STRIDE, materialToPackedVec4s } from './scene/materialPacking.js';
import { defaultDirectionalIrradiance, defaultDirectionalLight, packEmitterArrays } from './scene/emitterPacking.js';
import { environmentParams } from './scene/environmentPacking.js';
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
const IDENTITY_MAT4 = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

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
   * Pack the per-frame uniform buffer (512 bytes, vec4-aligned). Layout is
   * pinned and pathTraceBruteforce.wgsl's `params` struct reads from these
   * exact offsets. Callers must have already validated that #sceneBuffers
   * is non-null (renderFrame's preconditions handle this).
   *
   * Layout (384 bytes used out of a 512-byte buffer; trailing bytes are zero):
   *
   * Authoritative field layout is auto-generated in scene/frameParamsLayout.generated.ts (FrameParamsSlot).
   *
   * Per-light data lives in dedicated storage buffers at bind slots 20..23;
   * see `packEmitterArrays` for the layout (8 f32 / point light, 12 / spot,
   * 16 / rect-area, 16 / mesh-area).
   */
  #buildParamsBuffer(input: FrameInput, width: number, height: number): ArrayBuffer {
    const sb = this.#sceneBuffers!;
    const vp = multiplyMat4(input.projMatrix, input.viewMatrix);
    const invVp = invertMat4(asMat4(vp));
    if (invVp == null) {
      throw new Error('renderFrame: non-invertible view-projection matrix');
    }

    const paramsArrayBuffer = new ArrayBuffer(512);
    const paramsU32 = new Uint32Array(paramsArrayBuffer);
    const paramsF32 = new Float32Array(paramsArrayBuffer);
    paramsU32[FrameParamsSlot.width] = width;
    paramsU32[FrameParamsSlot.height] = height;
    paramsU32[FrameParamsSlot.frameIndex] = input.frameIndex >>> 0;
    paramsU32[FrameParamsSlot.frameSeed] = input.frameSeed >>> 0;
    paramsU32[FrameParamsSlot.triangleCount] = sb.triangleCount >>> 0;
    paramsU32[FrameParamsSlot.maxBounces] = this.#activeBounces >>> 0;
    paramsU32[FrameParamsSlot.bvhNodeCount] = sb.bvhNodeCount >>> 0;
    paramsU32[FrameParamsSlot.analyticCount] = sb.analyticCount >>> 0;
    paramsU32[FrameParamsSlot.pointLightCount] = sb.pointLightCount >>> 0;
    paramsU32[FrameParamsSlot.spotLightCount] = sb.spotLightCount >>> 0;
    paramsU32[FrameParamsSlot.rectAreaLightCount] = sb.rectAreaLightCount >>> 0;
    paramsU32[FrameParamsSlot.meshAreaLightCount] = sb.meshAreaLightCount >>> 0;
    paramsU32[FrameParamsSlot.mneeMaxIterations] = this.#mneeMaxIterations >>> 0;
    paramsU32[FrameParamsSlot.mneeMaxChainLength] = this.#mneeMaxChainLength >>> 0;
    paramsU32[FrameParamsSlot.hasEnvironmentMap] = sb.hasEnvironmentMap ? 1 : 0;
    paramsU32[FrameParamsSlot.causticStrategy] =
      this.#causticStrategy === 'manifold-nee'
        ? 1
        : this.#causticStrategy === 'photon-map'
          ? 2
          : 0;
    paramsU32[FrameParamsSlot.environmentMapWidth] = sb.environmentMapWidth >>> 0;
    paramsU32[FrameParamsSlot.environmentMapHeight] = sb.environmentMapHeight >>> 0;
    paramsF32[FrameParamsSlot.triIntersectEpsilon] = 1e-5; // triIntersectEpsilon: default metre-scale (D12)
    paramsU32[FrameParamsSlot.tlasNodeCount] = sb.tlasNodeCount >>> 0;
    paramsU32[FrameParamsSlot.spectralEnabled] = this.#spectralEnabled ? 1 : 0;
    paramsU32[FrameParamsSlot.heroStrategy] = 0;
    paramsF32[FrameParamsSlot.heroLambdaNm] = 550.0;
    paramsF32[FrameParamsSlot.heroPdf] = 1.0;
    paramsF32[FrameParamsSlot.cmfIntegralX] = X_CMF_INTEGRAL;
    paramsF32[FrameParamsSlot.cmfIntegralY] = Y_CMF_INTEGRAL;
    paramsF32[FrameParamsSlot.cmfIntegralZ] = Z_CMF_INTEGRAL;
    const bdptActive = this.#bdpt && this.#traceTier === 'full';
    paramsU32[FrameParamsSlot.bdptEnabled] = bdptActive ? 1 : 0;
    paramsU32[FrameParamsSlot.bdptMaxLightBounces] = this.#bdptMaxLightBounces >>> 0;
    // Eye-subpath scratch depth = the active per-pixel bounce limit (<= 8).
    paramsU32[FrameParamsSlot.bdptMaxEyeDepth] = this.#activeBounces >>> 0;
    // WS2 — power-weighted light selection. FULL tier only: the lite kernel keeps
    // the uniform pick and never composes the light-tree WGSL / group(3) binding.
    const lightTreeOn = this.#traceTier === 'full' && sb.lightTreeEnabled && this.#lightTreeImportanceSampling;
    paramsU32[FrameParamsSlot.lightTreeEnabled] = lightTreeOn ? 1 : 0;
    paramsU32[FrameParamsSlot.lightTreeNodeCount] = lightTreeOn ? sb.lightTreeNodeCount >>> 0 : 0;
    paramsF32[FrameParamsSlot.cameraPos] = input.cameraPosition[0];
    paramsF32[FrameParamsSlot.cameraPos + 1] = input.cameraPosition[1];
    paramsF32[FrameParamsSlot.cameraPos + 2] = input.cameraPosition[2];
    paramsF32[FrameParamsSlot.cameraPos + 3] = 1;
    paramsF32[FrameParamsSlot.lightDir] = sb.directionalLight[0];
    paramsF32[FrameParamsSlot.lightDir + 1] = sb.directionalLight[1];
    paramsF32[FrameParamsSlot.lightDir + 2] = sb.directionalLight[2];
    paramsF32[FrameParamsSlot.lightDir + 3] =
      (sb.directionalIrradiance[0] +
        sb.directionalIrradiance[1] +
        sb.directionalIrradiance[2]) /
      3;
    paramsF32[FrameParamsSlot.environmentTint] = sb.environmentTint[0];
    paramsF32[FrameParamsSlot.environmentTint + 1] = sb.environmentTint[1];
    paramsF32[FrameParamsSlot.environmentTint + 2] = sb.environmentTint[2];
    paramsF32[FrameParamsSlot.environmentTint + 3] = 0;
    paramsF32[FrameParamsSlot.environmentSun] = sb.environmentSunDirection[0];
    paramsF32[FrameParamsSlot.environmentSun + 1] = sb.environmentSunDirection[1];
    paramsF32[FrameParamsSlot.environmentSun + 2] = sb.environmentSunDirection[2];
    paramsF32[FrameParamsSlot.environmentSun + 3] = sb.environmentSunStrength;
    paramsF32.set(invVp, FrameParamsSlot.invViewProj);
    paramsF32.set(vp, FrameParamsSlot.viewProj);
    const prevVp = multiplyMat4(
      input.prevProjMatrix ?? input.projMatrix,
      input.prevViewMatrix ?? input.viewMatrix,
    );
    paramsF32.set(prevVp, FrameParamsSlot.prevViewProj);
    return paramsArrayBuffer;
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

  /**
   * Add one whole primitive to the live scene (contract: {@link Engine.addPrimitive}).
   *
   * Implementation: this is a TAIL-INSERTION — the new primitive is appended to
   * the scene's primitive list and the whole scene is re-packed via
   * {@link #repackScene}. We deliberately do NOT take the slice-2 incremental
   * BLAS-splice path here: a whole-primitive add also appends a NEW dense
   * material slot (and, for analytic primitives, header / param entries), and a
   * tail insertion needs no downstream offset remap — but the clean way to keep
   * the materials / analytic / per-triangle `triMaterialId` arrays in their
   * exact dense order is to reuse `buildPackedScene`, which is the same
   * already-correct path `setScene` runs. A bespoke incremental append would
   * have to duplicate that material/analytic packing for marginal buffer-reuse
   * gains; the full repack is correct-by-construction and reuses the
   * destroy-closure-off-the-struct realloc machinery in `uploadPackedScene`.
   */
  addPrimitive(primitive: ScenePrimitive): void {
    this.#assertLive('addPrimitive');
    const currentScene = this.#scene!;
    if (currentScene.primitives.some((p) => p.id === primitive.id)) {
      throw new Error(
        `addPrimitive: a primitive with id "${primitive.id}" already exists; ` +
          'use updatePrimitive to mutate an existing primitive.',
      );
    }
    const nextScene: Scene = {
      ...currentScene,
      primitives: [...currentScene.primitives, primitive],
    };
    this.#repackScene(nextScene, { warnOnEmpty: false });
  }

  /**
   * Remove one whole primitive from the live scene by id (contract:
   * {@link Engine.removePrimitive}).
   *
   * Implementation: drop the primitive from the scene's primitive list and
   * re-pack via {@link #repackScene}. A remove is the case that genuinely needs
   * the dense index remap the slice-2 splice does NOT cover — every downstream
   * primitive's material slot shifts by one, the per-triangle `triMaterialId`
   * baked into each downstream BLAS leaf would have to be re-resolved, and any
   * analytic header `materialId` / `paramsOffset` would have to be compacted. A
   * full `buildPackedScene` reproduces all of that correctly by construction,
   * so we repack rather than hand-roll a multi-array compaction.
   */
  removePrimitive(id: ScenePrimitive['id']): void {
    this.#assertLive('removePrimitive');
    const currentScene = this.#scene!;
    const nextPrimitives = currentScene.primitives.filter((p) => p.id !== id);
    if (nextPrimitives.length === currentScene.primitives.length) {
      throw new Error(
        `removePrimitive: no primitive with id "${id}" in the live scene.`,
      );
    }
    const nextScene: Scene = {
      ...currentScene,
      primitives: nextPrimitives,
    };
    this.#repackScene(nextScene, { warnOnEmpty: false });
  }

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    this.#assertLive('updatePrimitive');
    // #assertLive already throws when #scene is null; the non-null assertion
    // captures that invariant for the type checker.
    const currentScene = this.#scene!;
    const currentPrimitive = currentScene.primitives.find((p) => p.id === id) ?? null;
    const nextScene = patchPrimitiveInScene(currentScene, id, patch);
    if (
      currentPrimitive != null &&
      this.#geoPack != null &&
      this.#sceneBuffers != null &&
      canFastPathGeometryPatch(currentPrimitive, patch)
    ) {
      const rebuilt = rebuildPrimitiveBlas(nextScene, id, this.#geoPack, {
        tlas: true,
        resolveMaterialId: (pid) =>
          materialIndexForPrimitive(nextScene, pid, this.#supportedAnalyticShapes()) ?? 0,
      });
      if (rebuilt.ok) {
        const sb = this.#sceneBuffers;
        const prevTlasFp = fingerprintTlasBuffers({
          tlasNodes: sb.tlasNodes,
          tlasInstanceIndices: sb.tlasInstanceIndices,
          tlasBlasRoots: sb.tlasBlasRoots,
          tlasInstanceWorldToLocal: sb.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: sb.tlasInstanceLocalToWorld,
        });
        const nextTlasFp = fingerprintTlasBuffers({
          tlasNodes: rebuilt.pack.tlasNodes,
          tlasInstanceIndices: rebuilt.pack.tlasInstanceIndices,
          tlasBlasRoots: rebuilt.pack.tlasBlasRoots,
          tlasInstanceWorldToLocal: rebuilt.pack.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: rebuilt.pack.tlasInstanceLocalToWorld,
        });
        if (prevTlasFp === nextTlasFp) {
          uploadScenePackBlasOnly(this.#device, sb, rebuilt.pack);
        } else {
          uploadScenePackGeometry(this.#device, sb, rebuilt.pack);
        }
        this.#geoPack = rebuilt.pack;
        this.#gpu.invalidateBindGroups();
        this.#scene = nextScene;
        for (const warning of rebuilt.pack.warnings) {
          console.warn(`[vitrum/pt-webgpu] ${warning}`);
        }
        this.reset();
        return;
      }
    }
    if (
      currentPrimitive != null &&
      this.#geoPack != null &&
      this.#sceneBuffers != null &&
      canFastPathTopologyResizePatch(currentPrimitive, patch)
    ) {
      // Slice-2: a (skinned-)mesh's vertex/index COUNT changed. Rebuild ONLY this
      // primitive's BLAS and splice it into the packed scene — growing/shrinking
      // the concat buffers and rebasing every downstream primitive's offsets +
      // the TLAS BLAS roots. The concat buffers change SIZE, so the (5) BLAS +
      // (5) TLAS GPU buffers must be reallocated (no in-place writeBuffer).
      const rebuilt = rebuildPrimitiveBlas(nextScene, id, this.#geoPack, {
        tlas: true,
        resolveMaterialId: (pid) =>
          materialIndexForPrimitive(nextScene, pid, this.#supportedAnalyticShapes()) ?? 0,
      });
      if (rebuilt.ok) {
        uploadScenePackGeometryRealloc(this.#device, this.#sceneBuffers, rebuilt.pack);
        this.#geoPack = rebuilt.pack;
        this.#gpu.invalidateBindGroups();
        this.#scene = nextScene;
        for (const warning of rebuilt.pack.warnings) {
          console.warn(`[vitrum/pt-webgpu] ${warning}`);
        }
        this.reset();
        return;
      }
      // rebuildPrimitiveBlas rejected (primitive missing / not mesh-like) — fall
      // through to the full setScene rebuild below.
    }
    if (
      currentPrimitive != null &&
      currentPrimitive.kind === 'analytic' &&
      this.#sceneBuffers != null &&
      canFastPathTransformPatch(currentPrimitive, patch)
    ) {
      const analyticIndex = analyticIndexForPrimitive(
        nextScene,
        id,
        this.#supportedAnalyticShapes(),
      );
      if (analyticIndex != null) {
        const nextPrimitive = nextScene.primitives.find((p) => p.id === id);
        if (nextPrimitive != null && nextPrimitive.kind === 'analytic') {
          const localToWorld = asMat4(nextPrimitive.transform ?? IDENTITY_MAT4);
          const maybeWorldToLocal = invertMat4(localToWorld);
          const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
          if (maybeWorldToLocal == null) {
            console.warn(
              `[vitrum/pt-webgpu] Primitive "${nextPrimitive.id}" has non-invertible analytic transform; using identity fallback.`,
            );
          }
          const byteOffset = analyticIndex * 16 * Float32Array.BYTES_PER_ELEMENT;
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.analyticLocalToWorldBuffer,
            byteOffset,
            localToWorld.buffer,
            localToWorld.byteOffset,
            localToWorld.byteLength,
          );
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.analyticWorldToLocalBuffer,
            byteOffset,
            worldToLocal.buffer,
            worldToLocal.byteOffset,
            worldToLocal.byteLength,
          );
          this.#sceneBuffers.analyticLocalToWorld.set(localToWorld, analyticIndex * 16);
          this.#sceneBuffers.analyticWorldToLocal.set(worldToLocal, analyticIndex * 16);
          this.#scene = nextScene;
          this.reset();
          return;
        }
      }
    }
    if (
      currentPrimitive != null &&
      this.#geoPack != null &&
      this.#sceneBuffers != null &&
      canFastPathInstancedTopologyPatch(currentPrimitive, patch)
    ) {
      // Slice-1: instanced-mesh instance COUNT changed. BLAS geometry is shared
      // across instances and byte-identical, so we rebuild only the TLAS
      // (reusing the previous pack's BLAS arrays verbatim — no per-triangle
      // buildArrayBvh) and reallocate only the 5 TLAS GPU buffers.
      const rebuilt = rebuildTlasReuseBlas(nextScene, this.#geoPack);
      if (rebuilt.ok) {
        uploadScenePackTlasRealloc(this.#device, this.#sceneBuffers, {
          tlasNodes: rebuilt.pack.tlasNodes,
          tlasInstanceIndices: rebuilt.pack.tlasInstanceIndices,
          tlasBlasRoots: rebuilt.pack.tlasBlasRoots,
          tlasInstanceWorldToLocal: rebuilt.pack.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: rebuilt.pack.tlasInstanceLocalToWorld,
          tlasNodeCount: rebuilt.pack.tlasNodeCount,
          primitiveTlasBindings: rebuilt.pack.primitiveTlasBindings,
        });
        this.#geoPack = rebuilt.pack;
        this.#gpu.invalidateBindGroups();
        this.#scene = nextScene;
        for (const warning of rebuilt.pack.warnings) {
          console.warn(`[vitrum/pt-webgpu] ${warning}`);
        }
        this.reset();
        return;
      }
      // rebuildTlasReuseBlas rejected (e.g. concurrent geometry change) — fall
      // through to the full setScene rebuild below.
    }
    if (
      currentPrimitive != null &&
      this.#sceneBuffers != null &&
      canFastPathTransformPatch(currentPrimitive, patch)
    ) {
      const tlas = rebuildTlasForSceneTransforms(
        nextScene,
        this.#sceneBuffers.primitiveTlasBindings,
        {
          tlasNodes: this.#sceneBuffers.tlasNodes,
          tlasInstanceIndices: this.#sceneBuffers.tlasInstanceIndices,
          tlasBlasRoots: this.#sceneBuffers.tlasBlasRoots,
          tlasInstanceWorldToLocal: this.#sceneBuffers.tlasInstanceWorldToLocal,
        },
      );
      if (tlas.ok && canReuseTlasBufferLengths(this.#sceneBuffers, tlas)) {
        uploadScenePackTlasOnly(this.#device, this.#sceneBuffers, {
          tlasNodes: tlas.tlasNodes,
          tlasInstanceIndices: tlas.tlasInstanceIndices,
          tlasBlasRoots: tlas.tlasBlasRoots,
          tlasInstanceWorldToLocal: tlas.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: tlas.tlasInstanceLocalToWorld,
          tlasNodeCount: Math.floor(tlas.tlasNodes.length / 8),
          primitiveTlasBindings: this.#sceneBuffers.primitiveTlasBindings,
        });
        this.#scene = nextScene;
        for (const warning of tlas.warnings) {
          console.warn(`[vitrum/pt-webgpu] ${warning}`);
        }
        this.reset();
        return;
      }
    }
    if (canFastPathMaterialPatch(patch) && this.#sceneBuffers != null) {
      const materialIndex = materialIndexForPrimitive(
        nextScene,
        id,
        this.#supportedAnalyticShapes(),
      );
      const primitive = nextScene.primitives.find((p) => p.id === id);
      if (materialIndex != null && primitive != null) {
        const packed = materialToPackedVec4s(primitive.material);
        if (packed.length === MATERIAL_FLOAT_STRIDE) {
          const materialData = new Float32Array(packed);
          const floatOffset = materialIndex * MATERIAL_FLOAT_STRIDE;
          const byteOffset = floatOffset * Float32Array.BYTES_PER_ELEMENT;
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.materialsBuffer,
            byteOffset,
            materialData.buffer,
            materialData.byteOffset,
            materialData.byteLength,
          );
          this.#sceneBuffers.materials.set(materialData, floatOffset);
          this.#scene = nextScene;
          this.reset();
          return;
        }
      }
    }
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    const currentScene = this.#scene!;
    const nextScene = patchEmitterInScene(currentScene, id, patch);
    if (this.#sceneBuffers != null) {
      const packed = packEmitterArrays(nextScene);
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.pointLightsBuffer,
        0,
        packed.pointLightsData.buffer,
        packed.pointLightsData.byteOffset,
        packed.pointLightsData.byteLength,
      );
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.spotLightsBuffer,
        0,
        packed.spotLightsData.buffer,
        packed.spotLightsData.byteOffset,
        packed.spotLightsData.byteLength,
      );
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.rectAreaLightsBuffer,
        0,
        packed.rectAreaLightsData.buffer,
        packed.rectAreaLightsData.byteOffset,
        packed.rectAreaLightsData.byteLength,
      );
      this.#device.queue.writeBuffer(
        this.#sceneBuffers.meshAreaLightsBuffer,
        0,
        packed.meshAreaLightsData.buffer,
        packed.meshAreaLightsData.byteOffset,
        packed.meshAreaLightsData.byteLength,
      );
      this.#sceneBuffers.pointLightsData.set(packed.pointLightsData);
      this.#sceneBuffers.spotLightsData.set(packed.spotLightsData);
      this.#sceneBuffers.rectAreaLightsData.set(packed.rectAreaLightsData);
      this.#sceneBuffers.meshAreaLightsData.set(packed.meshAreaLightsData);
      applyEmitterCountMutation(this.#sceneBuffers, {
        pointLightCount: packed.pointLightCount,
        spotLightCount: packed.spotLightCount,
        rectAreaLightCount: packed.rectAreaLightCount,
        meshAreaLightCount: packed.meshAreaLightCount,
        directionalLight: defaultDirectionalLight(nextScene),
        directionalIrradiance: defaultDirectionalIrradiance(nextScene),
      });
      // WS2 — the light tree's powers/positions depend on the emitters, so
      // rebuild + re-upload it (reallocating + invalidating bind groups if the
      // node count changed). Without this the GPU selection would importance-
      // sample the OLD light set after an incremental emitter patch.
      if (rebuildLightTreeForScene(this.#device, this.#sceneBuffers, nextScene)) {
        this.#gpu.invalidateBindGroups();
      }
      this.#scene = nextScene;
      for (const warning of packed.warnings) {
        console.warn(`[vitrum/pt-webgpu] ${warning}`);
      }
      this.reset();
      return;
    }
    this.setScene(nextScene);
  }

  updateEnvironment(env: Scene['environment'] | null): void {
    this.#assertLive('updateEnvironment');
    const currentScene = this.#scene!;
    const nextScene: Scene = {
      ...currentScene,
      environment: env ?? { kind: 'none' },
    };
    if (this.#sceneBuffers != null) {
      const packed = environmentParams(nextScene);
      const texelLenMatches = packed.hdriTexels.length === this.#sceneBuffers.environmentMapTexels.length;
      const cdfLenMatches = packed.hdriCdf.length === this.#sceneBuffers.environmentMapCdf.length;
      if (texelLenMatches && cdfLenMatches) {
        if (packed.hdriTexels.byteLength > 0) {
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.environmentMapTexelsBuffer,
            0,
            packed.hdriTexels.buffer,
            packed.hdriTexels.byteOffset,
            packed.hdriTexels.byteLength,
          );
        }
        if (packed.hdriCdf.byteLength > 0) {
          this.#device.queue.writeBuffer(
            this.#sceneBuffers.environmentMapCdfBuffer,
            0,
            packed.hdriCdf.buffer,
            packed.hdriCdf.byteOffset,
            packed.hdriCdf.byteLength,
          );
        }
        applyEnvironmentMutation(this.#sceneBuffers, {
          environmentTint: packed.tint,
          environmentSunDirection: packed.sunDirection,
          environmentSunStrength: packed.sunStrength,
          environmentMapWidth: packed.hdriWidth,
          environmentMapHeight: packed.hdriHeight,
          hasEnvironmentMap: packed.hasHdri,
        });
        this.#sceneBuffers.environmentMapTexels.set(packed.hdriTexels);
        this.#sceneBuffers.environmentMapCdf.set(packed.hdriCdf);
        // WS2 — the env counts as a selectable light in the NEE walk, so an env
        // change can flip the light-tree gate / leaf count. Rebuild + re-upload
        // (reallocating + invalidating bind groups if the node count changed).
        if (rebuildLightTreeForScene(this.#device, this.#sceneBuffers, nextScene)) {
          this.#gpu.invalidateBindGroups();
        }
        this.#scene = nextScene;
        for (const warning of packed.warnings) {
          console.warn(`[vitrum/pt-webgpu] ${warning}`);
        }
        this.reset();
        return;
      }
    }
    this.setScene(nextScene);
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
    };
    return new PtWebgpuInverseSession(hooks, opts);
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
