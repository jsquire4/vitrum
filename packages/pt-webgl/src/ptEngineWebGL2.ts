import {
  BufferAttribute,
  Matrix4,
  PerspectiveCamera,
} from 'three';
import type {
  WebGLRenderer,
  WebGLRenderTarget,
  Scene as ThreeScene,
  Material as TMaterial,
  Mesh as TMesh,
  Object3D,
} from 'three';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { WebGLPathTracer } from 'three-gpu-pathtracer';

import type {
  Engine,
  EngineCapabilities,
  EngineFactory,
  EngineOptions,
  EngineState,
  FrameStats,
  ProgressStats,
} from '@vitrum/core';
import { asBackendTexture } from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import type { Scene, ScenePrimitive, SceneEmitter, SceneEnvironment, MeshPrimitive } from '@vitrum/core';
import { applyFrameToPerspectiveCamera } from './frameCamera.js';
import {
  vitrumSceneToThree,
  applyEnvironment,
  applyVitrumMaterialToMesh,
  findMeshByPrimitiveId,
} from '@vitrum/three-bindings';
import type { BdptLightSubpathTracer } from './bdpt/runBdptLightSubpathPass.js';
import { BdptLightPathBuffer } from './bdptLightPathBuffer.js';
import { bdptForceGpuBind, isSoftwareGlRenderer } from './bdpt/isSoftwareGlRenderer.js';
import { driveForkMaterialUniforms } from './forkUniformBridge.js';
import { ForkAccess } from './forkAccess.js';
import {
  MAX_TILE_GRID,
  TileVariancePass,
  computeAdaptiveTileRepeatFactors,
} from './adaptiveTileWeights.js';
import {
  OIDNFinalDispatcher,
  type DenoisedFrame,
  type OIDNBridgeLoader,
} from './oidnFinalDispatcher.js';
import { IblBakerCache } from './iblBaker.js';
import { auditPtWebglSceneForTlas, type PtWebglTlasAudit } from './sceneTlasAudit.js';
import type { SkyParams } from '@vitrum/scene-lighting';
import type { DataTexture, Texture } from 'three';

export interface PTEngineWebGL2Options extends EngineOptions {
  readonly device: WebGLRenderer;
  /**
   * Test-only OIDN bridge loader override. When the host selects
   * `denoiser: 'oidn-final'`, the engine constructs an
   * {@link OIDNFinalDispatcher}; production code lets the dispatcher
   * lazy-import `@vitrum/shared-denoisers` on first kick. Tests pass a
   * synthetic loader returning a mock bridge to avoid pulling in
   * `onnxruntime-web`. Not part of the public Engine contract — typed
   * here for the W11 follow-up integration test.
   */
  readonly oidnBridgeLoader?: OIDNBridgeLoader;
}

const DEFAULT_MAX_BOUNCES = 12;
const DEFAULT_MAX_SAMPLES_PER_PIXEL = 4096;
const DEFAULT_TILE_SIZE = 3;

export type PTEngineWebGL2QualityMode = 'interactive' | 'final' | 'capture' | 'safe';

export interface PTEngineWebGL2Telemetry {
  readonly qualityMode: PTEngineWebGL2QualityMode;
  readonly renderer: string;
  readonly requestedWidth: number;
  readonly requestedHeight: number;
  readonly renderWidth: number;
  readonly renderHeight: number;
  readonly samplesPerFrame: number;
  readonly tileSize: number;
  readonly batchMs: number;
  readonly msPerSample: number | null;
  readonly sppDelta: number;
  readonly sppPerSecond: number | null;
  readonly estimatedRenderTargetBytes: number;
  readonly renderTargetBudgetBytes: number;
  readonly guardrail: string | null;
  readonly additiveAccumulation: boolean;
  readonly pixelAdaptiveSampling: boolean;
}

export type PTEngineWebGL2FrameOutput = FrameOutput & {
  readonly telemetry?: PTEngineWebGL2Telemetry | undefined;
};

interface DeviceLimits {
  readonly maxTextureSize: number;
  readonly maxRenderbufferSize: number;
  readonly renderer: string;
}

/**
 * Typed surface of the fork we depend on. WebGLPathTracer's published types
 * use Three.js Scene/Camera types that diverge slightly between three.js
 * @types versions; this wrapper interface lets us cast once at construction
 * and call methods on the wrapper using our local types. Add methods here as
 * we depend on them — keep the surface minimal.
 */
interface WebGLPathTracerCompat {
  setScene(scene: unknown, camera: unknown): void;
  setCamera(camera: unknown): void;
  setSize(width: number, height: number): void;
  reset(): void;
  renderSample(): void;
  /** Re-reads `scene.environment` / `scene.environmentIntensity` /
   *  `scene.environmentRotation` (and the matching background fields) into the
   *  fork's IBL uniforms WITHOUT touching geometry, materials, or the BVH.
   *  Internally calls `reset()` (one accumulator-clear) — no BVH rebuild,
   *  no geometry re-upload. Used by `PTEngineWebGL2.updateEnvironment()` to
   *  service host-driven timeOfDay scrubs cheaply. */
  updateEnvironment?(): void;
  /** Re-pack MaterialsTexture from the cached scene without BVH rebuild (PR-8). */
  updateMaterials?(): void;
  /** Re-pack light buffers from the cached scene without BVH rebuild (PR-8). */
  updateLights?(): void;
  configureAdditiveAccumulation?(enabled: boolean, blendFrames: boolean): void;
  renderBdptLightSubpathPass?(
    lightPathTarget: import('three').WebGLRenderTarget,
    maxLightBounces: number,
    frameSeed: number,
  ): void;
  /** Optional fork field — the wrapper stores a reference to the THREE scene
   *  most recently passed to `setScene()`. updateEnvironment() reads
   *  `scene.environment*` off this reference, so the host MUST mutate the
   *  same scene object the wrapper has cached, not pass in a new one. */
  scene?: unknown;
  dispose?(): void;
  samples: number;
  tileRepeatFactors?: Uint8Array | null;
  tiles: { setScalar: (n: number) => void; set(x: number, y: number): void; x: number; y: number };
  bounces: number;
  filterGlossyFactor: number;
  fastUpdate: boolean;
  domElement?: HTMLCanvasElement;
}

interface RenderSizePlan {
  readonly width: number;
  readonly height: number;
  readonly estimatedBytes: number;
  readonly guardrail: string | null;
}

function isEmitterOnlyPatch(patch: Partial<SceneEmitter>): boolean {
  if ('kind' in patch && patch.kind !== undefined) return false;
  if ('meshPrimitiveId' in patch && patch.meshPrimitiveId !== undefined) return false;
  if ('transform' in patch && patch.transform !== undefined) return false;
  return Object.keys(patch).some((k) => k !== 'id');
}

function isMaterialOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  if (patch.material === undefined) return false;
  for (const key of Object.keys(patch) as (keyof ScenePrimitive)[]) {
    if (key === 'id' || key === 'material') continue;
    if ((patch as Record<string, unknown>)[key] !== undefined) return false;
  }
  return true;
}

function isTransformOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  const rec = patch as Record<string, unknown>;
  if (rec['transform'] === undefined) return false;
  for (const key of Object.keys(rec)) {
    if (key === 'id' || key === 'transform') continue;
    if (rec[key] !== undefined) return false;
  }
  return true;
}

type PathTracerGenerateResult = {
  bvhChanged?: boolean;
  bvh?: unknown;
  needsMaterialIndexUpdate?: boolean;
  geometry?: {
    attributes: {
      normal?: { array: ArrayLike<number> };
      tangent?: { array: ArrayLike<number> };
      uv?: { array: ArrayLike<number> };
      color?: { array: ArrayLike<number> };
    };
  };
};

/** Regenerate merged geometry + BVH via fork generator — no full `setScene`. */
function refreshPathTracerSceneGeometry(
  pathTracer: WebGLPathTracer,
  threeRoot: ThreeScene,
): boolean {
  const internal = pathTracer as unknown as {
    _generator?: { initialized?: boolean; generate: () => PathTracerGenerateResult };
    _pathTracer?: {
      material: {
        bvh: { updateFrom: (b: unknown) => void };
        attributesArray: {
          updateFrom: (
            normal: { array: ArrayLike<number> } | undefined,
            tangent: { array: ArrayLike<number> } | undefined,
            uv: { array: ArrayLike<number> } | undefined,
            color: { array: ArrayLike<number> } | undefined,
          ) => void;
        };
        materialIndexAttribute: { updateFrom: (attr: unknown) => void };
      };
    };
  };
  const gen = internal._generator;
  if (gen?.initialized !== true) {
    return false;
  }
  threeRoot.updateMatrixWorld(true);
  const result = gen.generate();
  const mat = internal._pathTracer?.material;
  if (result.bvhChanged === true && result.bvh != null && mat != null) {
    mat.bvh.updateFrom(result.bvh);
    const attrs = result.geometry?.attributes;
    if (attrs != null) {
      mat.attributesArray.updateFrom(attrs.normal, attrs.tangent, attrs.uv, attrs.color);
    }
    if (result.needsMaterialIndexUpdate === true && result.geometry != null) {
      const materialIndex = (result.geometry as { attributes: { materialIndex?: unknown } }).attributes
        .materialIndex;
      if (materialIndex != null) {
        mat.materialIndexAttribute.updateFrom(materialIndex);
      }
    }
  }
  pathTracer.reset();
  return true;
}

function isPositionsOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  const rec = patch as Record<string, unknown>;
  if (rec['positions'] === undefined) return false;
  for (const key of Object.keys(rec)) {
    if (key === 'id' || key === 'positions' || key === 'normals') continue;
    if (rec[key] !== undefined) return false;
  }
  return true;
}

function applyPositionsPatchToMesh(mesh: TMesh, patch: Partial<MeshPrimitive>): boolean {
  const positions = patch.positions;
  if (positions == null) return false;
  const posAttr = mesh.geometry.getAttribute('position');
  const vertCount = positions.length / 3;
  if (posAttr != null && posAttr.count !== vertCount) {
    return false;
  }
  mesh.geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  if (patch.normals != null) {
    mesh.geometry.setAttribute('normal', new BufferAttribute(new Float32Array(patch.normals), 3));
  }
  return true;
}

function patchPrimitiveInScene(scene: Scene, id: string, patch: Partial<ScenePrimitive>): Scene {
  const idx = scene.primitives.findIndex((p) => String(p.id) === id);
  if (idx < 0) throw new Error(`updatePrimitive: primitive "${id}" not found in current scene`);
  const current = scene.primitives[idx]!;
  if (patch.id !== undefined && String(patch.id) !== String(current.id)) {
    throw new Error(`updatePrimitive: primitive "${id}" id cannot be changed`);
  }
  if (patch.kind !== undefined && patch.kind !== current.kind) {
    throw new Error(
      `updatePrimitive: primitive "${id}" kind cannot change from "${current.kind}" to "${patch.kind}"`,
    );
  }
  const next = scene.primitives.slice();
  next[idx] = { ...current, ...patch } as ScenePrimitive;
  return { ...scene, primitives: next };
}

function patchEmitterInScene(scene: Scene, id: string, patch: Partial<SceneEmitter>): Scene {
  const idx = scene.emitters.findIndex((e) => String(e.id) === id);
  if (idx < 0) throw new Error(`updateEmitter: emitter "${id}" not found in current scene`);
  const current = scene.emitters[idx]!;
  if (patch.id !== undefined && String(patch.id) !== String(current.id)) {
    throw new Error(`updateEmitter: emitter "${id}" id cannot be changed`);
  }
  if (patch.kind !== undefined && patch.kind !== current.kind) {
    throw new Error(
      `updateEmitter: emitter "${id}" kind cannot change from "${current.kind}" to "${patch.kind}"`,
    );
  }
  const next = scene.emitters.slice();
  next[idx] = { ...current, ...patch } as SceneEmitter;
  return { ...scene, emitters: next };
}

interface SchedulerOptions {
  readonly qualityMode: PTEngineWebGL2QualityMode;
  readonly adaptive: boolean;
  readonly targetBatchMs: number;
  readonly minSamplesPerFrame: number;
  readonly maxSamplesPerFrame: number;
  readonly initialSamplesPerFrame: number;
  readonly initialTileSize: number;
  readonly maxTileSize: number;
  readonly renderTargetBudgetBytes: number;
}

/** RGBA16F texel size: 4 channels × 2 bytes per channel. */
const BYTES_PER_RGBA16F_PIXEL = 8;
/** Number of full-resolution render targets the WebGL path tracer allocates:
 *  primary accumulation, depth, normal, motion vector. */
const ESTIMATED_RENDER_TARGET_COUNT = 4;
/** Per-renderer overhead in driver metadata + mip alignment + GL state, used
 *  to budget memory when computing the host's adaptive render-size plan. */
const DEFAULT_RENDER_TARGET_OVERHEAD_BYTES = 64 * 1024 * 1024;

/** Below 360p (≈230k pixels) the adaptive-tiling dispatch's per-tile overhead
 *  dominates the per-pixel cost, so we disable tiling entirely. The threshold
 *  is the standard SD frame area. */
const MIN_RESOLUTION_FOR_TILING = 640 * 360;

function extensionNumber(
  extensions: Readonly<Record<string, unknown>> | undefined,
  key: string,
  fallback: number,
): number {
  const value = extensions?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function extensionBoolean(
  extensions: Readonly<Record<string, unknown>> | undefined,
  key: string,
  fallback: boolean,
): boolean {
  const value = extensions?.[key];
  return typeof value === 'boolean' ? value : fallback;
}

function extensionQualityMode(
  extensions: Readonly<Record<string, unknown>> | undefined,
): PTEngineWebGL2QualityMode {
  const value = extensions?.['vitrum.ptWebgl.qualityMode'];
  return value === 'interactive' || value === 'final' || value === 'capture' || value === 'safe'
    ? value
    : 'capture';
}

function clampInt(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.floor(value)));
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function defaultSchedulerOptions(
  extensions: Readonly<Record<string, unknown>> | undefined,
): SchedulerOptions {
  const qualityMode = extensionQualityMode(extensions);
  const modeDefaults: Record<PTEngineWebGL2QualityMode, Omit<SchedulerOptions, 'qualityMode'>> = {
    interactive: {
      adaptive: true,
      targetBatchMs: 40,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 48,
      initialSamplesPerFrame: 6,
      initialTileSize: 1,
      maxTileSize: 4,
      renderTargetBudgetBytes: 1024 * 1024 * 1024,
    },
    final: {
      adaptive: true,
      targetBatchMs: 120,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 96,
      initialSamplesPerFrame: 12,
      initialTileSize: 1,
      maxTileSize: 4,
      renderTargetBudgetBytes: 2 * 1024 * 1024 * 1024,
    },
    capture: {
      adaptive: false,
      targetBatchMs: 0,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 1,
      initialSamplesPerFrame: 1,
      initialTileSize: DEFAULT_TILE_SIZE,
      maxTileSize: DEFAULT_TILE_SIZE,
      renderTargetBudgetBytes: 512 * 1024 * 1024,
    },
    safe: {
      adaptive: true,
      targetBatchMs: 8,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 4,
      initialSamplesPerFrame: 1,
      initialTileSize: DEFAULT_TILE_SIZE,
      maxTileSize: 4,
      renderTargetBudgetBytes: 256 * 1024 * 1024,
    },
  };
  const base = modeDefaults[qualityMode];
  const requestedSamplesPerFrame = extensionNumber(
    extensions,
    'vitrum.ptWebgl.samplesPerFrame',
    base.initialSamplesPerFrame,
  );
  const requestedTileSize = extensionNumber(extensions, 'vitrum.ptWebgl.tileSize', base.initialTileSize);
  const maxSamplesPerFrame = clampInt(
    extensionNumber(extensions, 'vitrum.ptWebgl.maxSamplesPerFrame', base.maxSamplesPerFrame),
    base.minSamplesPerFrame,
    128,
  );
  return {
    qualityMode,
    adaptive: extensionBoolean(extensions, 'vitrum.ptWebgl.adaptiveScheduler', base.adaptive),
    targetBatchMs: Math.max(0, extensionNumber(extensions, 'vitrum.ptWebgl.targetBatchMs', base.targetBatchMs)),
    minSamplesPerFrame: base.minSamplesPerFrame,
    maxSamplesPerFrame,
    initialSamplesPerFrame: clampInt(requestedSamplesPerFrame, base.minSamplesPerFrame, maxSamplesPerFrame),
    initialTileSize: clampInt(requestedTileSize, 1, base.maxTileSize),
    maxTileSize: clampInt(extensionNumber(extensions, 'vitrum.ptWebgl.maxTileSize', base.maxTileSize), 1, 8),
    renderTargetBudgetBytes: Math.max(
      64 * 1024 * 1024,
      extensionNumber(extensions, 'vitrum.ptWebgl.renderTargetBudgetBytes', base.renderTargetBudgetBytes),
    ),
  };
}

/**
 * Internal state-setter token. The factory constructs a `StateSlot`, passes it
 * to `PTEngineWebGL2`, then calls `slot.set('ready')`. External callers cannot
 * obtain the slot; only the factory transitions `initializing` → `ready`.
 */
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

/**
 * Walk a Three.js subtree and dispose every Mesh's geometry + material(s).
 * Module-private — only `PTEngineWebGL2` uses this on the converted scene
 * root it owns (no external consumer should depend on it).
 */
function disposeObject3DTree(obj: Object3D): void {
  obj.traverse((o) => {
    const mesh = o as TMesh;
    if (mesh.isMesh === true) {
      mesh.geometry?.dispose();
      const m = mesh.material as TMaterial | TMaterial[] | undefined;
      if (Array.isArray(m)) m.forEach((x) => {
        x.dispose?.();
      });
      else m?.dispose?.();
    }
  });
}

interface PTEngineWebGL2Init {
  readonly renderer: WebGLRenderer;
  readonly pathTracer: WebGLPathTracer;
  readonly camera: PerspectiveCamera;
}

export class PTEngineWebGL2 implements Engine {
  readonly #slot: StateSlot;

  readonly #renderer: WebGLRenderer;
  readonly #pathTracer: WebGLPathTracer;
  readonly #camera: PerspectiveCamera;

  readonly #maxBouncesLimit: number;
  readonly #maxSamplesLimit: number;
  readonly #causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly #mneeMaxIterations: number;
  readonly #mneeMaxChainLength: number;
  readonly #spectralRendering: boolean;
  readonly #radianceClamp: number;
  // Sprint 10c — BDPT option. Stored from extensions['vitrum.ptWebgl.bdpt'].
  // Forwarded to fork uBdptEnabled / uBdptMaxLightBounces / uBdptLightPathTex uniforms.
  readonly #bdpt: boolean;
  readonly #bdptMaxLightBounces: number;
  /** When true, skip the fork GPU light-subpath pass (CPU bounce-0 fill only). */
  readonly #bdptCpuFill: boolean;
  /** Sprint 10c — most-recently-supplied BDPT light-path texture. Set via
   *  {@link bdptAdvanceFrame}; null until the host calls that method.
   *  When non-null + `#bdpt === true`, every renderFrame's connect pass
   *  reads this texture for cached light vertices. */
  readonly #limits: DeviceLimits;
  readonly #schedulerOptions: SchedulerOptions;

  #vitrumScene: Scene | null = null;
  #lastTlasAudit: PtWebglTlasAudit | null = null;
  #threeSceneRoot: ThreeScene | null = null;
  #cameraSignature = '';
  #samplesPerFrame: number;
  #tileSize: number;
  #lastRenderWidth = 0;
  #lastRenderHeight = 0;
  #contextLost = false;
  #contextLostHandler: ((ev: Event) => void) | null = null;
  #lastTargetSpp = 16;
  #lastTelemetry: PTEngineWebGL2Telemetry | undefined;
  #additiveAccumulation = false;
  #pixelAdaptiveSampling = false;
  #pixelAdaptiveCadence = 4;
  #tileVariancePass: TileVariancePass | null = null;
  #tileFactorsScratch: Uint8Array = new Uint8Array(MAX_TILE_GRID * MAX_TILE_GRID);

  /** T3.E telemetry — fired at the end of each renderFrame() with the
   *  scheduler's per-frame stats. Empty by default; subscribers register
   *  via {@link onFrame}. */
  readonly #frameSubs: Array<(s: FrameStats) => void> = [];
  /** T3.E telemetry — fired with the running SPP / target on each frame
   *  that performed work. Subscribers register via {@link onProgress}. */
  readonly #progressSubs: Array<(p: ProgressStats) => void> = [];

  /** W11 follow-up — OIDN final-pass dispatcher. Non-null iff the host
   *  selected `denoiser: 'oidn-final'` AND supplied
   *  `extensions['vitrum.ptWebgl.oidnModelUrl']`. The dispatcher is
   *  invalidated on setScene / reset / updateEnvironment (any state
   *  change that resets the accumulator). See
   *  {@link OIDNFinalDispatcher} for the kick-and-return state machine. */
  readonly #oidnDispatcher: OIDNFinalDispatcher | null;

  /** Per-engine analytic-sky bake cache. Replaces the previous module-level
   *  singleton in `iblBaker.ts` so multi-engine hosts no longer share GPU
   *  state across renderers. Hosts that need a sky equirect for
   *  `scene.environment` should call `bakeSkyEquirect` on the engine
   *  instance (defined below). Disposed in {@link dispose}. */
  readonly #iblBakerCache: IblBakerCache;

  constructor(opts: PTEngineWebGL2Options, gpu: PTEngineWebGL2Init, slot: StateSlot) {
    this.#slot = slot;
    this.#maxBouncesLimit = opts.maxBounces ?? DEFAULT_MAX_BOUNCES;
    this.#maxSamplesLimit = opts.maxSamplesPerPixel ?? DEFAULT_MAX_SAMPLES_PER_PIXEL;
    // RFE-05: strategy is forwarded to fork uniforms and mirrored in `capabilities.causticStrategy`.
    this.#causticStrategy = opts.causticStrategy ?? 'none';
    const causticOpts = opts.causticOptions ?? {};
    const mneeIter = typeof causticOpts.mneeMaxIterations === 'number' ? causticOpts.mneeMaxIterations : 8;
    const mneeChain = typeof causticOpts.mneeMaxChainLength === 'number' ? causticOpts.mneeMaxChainLength : 3;
    this.#mneeMaxIterations = Math.max(1, mneeIter);
    this.#mneeMaxChainLength = Math.max(1, mneeChain);
    this.#spectralRendering = opts.extensions?.['vitrum.ptWebgl.spectralRendering'] === true;
    const requestedRadianceClamp = opts.extensions?.['vitrum.ptWebgl.radianceClamp'];
    this.#radianceClamp = typeof requestedRadianceClamp === 'number' && Number.isFinite(requestedRadianceClamp)
      ? Math.max(0, requestedRadianceClamp)
      : 0;
    // Sprint 10c: BDPT option from extensions.
    // 'vitrum.ptWebgl.bdpt' (boolean) enables BDPT mode for PT_FINAL caustic renders.
    // 'vitrum.ptWebgl.bdptMaxLightBounces' (1–3) controls light-subpath depth.
    // The lightPathTex is not passed via options (it requires a live WebGL texture
    // object); hosts that need BDPT must call driveForkMaterialUniforms() directly
    // with their ForkBridgeBdptOptions including the ping-pong texture reference.
    this.#bdpt = opts.extensions?.['vitrum.ptWebgl.bdpt'] === true;
    const requestedBdptBounces = opts.extensions?.['vitrum.ptWebgl.bdptMaxLightBounces'];
    this.#bdptMaxLightBounces = typeof requestedBdptBounces === 'number' && requestedBdptBounces >= 1
      ? Math.min(3, Math.floor(requestedBdptBounces))
      : 3;
    this.#bdptCpuFill = opts.extensions?.['vitrum.ptWebgl.bdptCpuFill'] === true;
    this.#schedulerOptions = defaultSchedulerOptions(opts.extensions);
    this.#samplesPerFrame = this.#schedulerOptions.initialSamplesPerFrame;
    this.#tileSize = this.#schedulerOptions.initialTileSize;
    this.#renderer = gpu.renderer;
    this.#pathTracer = gpu.pathTracer;
    this.#camera = gpu.camera;
    this.#limits = this.#detectDeviceLimits();
    const adaptiveRequested = opts.extensions?.['vitrum.ptWebgl.pixelAdaptiveSampling'] === true;
    this.#additiveAccumulation =
      opts.extensions?.['vitrum.ptWebgl.additiveAccumulation'] === true || adaptiveRequested;
    this.#pixelAdaptiveSampling = adaptiveRequested;
    this.#pixelAdaptiveCadence = Math.max(
      1,
      extensionNumber(opts.extensions, 'vitrum.ptWebgl.pixelAdaptiveCadence', 4),
    );
    this.#tileFactorsScratch.fill(1);
    if (this.#pixelAdaptiveSampling) {
      this.#tileVariancePass = new TileVariancePass(MAX_TILE_GRID);
    }
    const tracerCompat = gpu.pathTracer as unknown as WebGLPathTracerCompat;
    tracerCompat.configureAdditiveAccumulation?.(
      this.#additiveAccumulation,
      this.#additiveAccumulation,
    );
    this.#contextLostHandler = () => {
      this.#contextLost = true;
      this.#samplesPerFrame = 1;
      this.#tileSize = Math.max(this.#tileSize, DEFAULT_TILE_SIZE);
    };
    this.#renderer.domElement?.addEventListener?.('webglcontextlost', this.#contextLostHandler);
    this.#pathTracer.renderDelay = 0;
    this.#pathTracer.minSamples = 0;
    this.#pathTracer.fadeDuration = 0;

    // W11 follow-up — wire 'oidn-final' denoiser mode. The factory below
    // validates the model URL at construction; this branch trusts that
    // validation has already run, so a 'oidn-final' opts.denoiser without
    // the matching model URL is a programming error in the factory itself.
    // Color-only readback (no albedo / normal aux inputs) — the pt-webgl
    // fork's primary render target is plain `WebGLRenderTarget`, not MRT,
    // so the shader's `gAlbedo` / `gNormalDepth` outputs aren't captured
    // host-side. See {@link OIDNFinalDispatcher} doc for context.
    if (opts.denoiser === 'oidn-final') {
      const modelUrl = opts.extensions?.['vitrum.ptWebgl.oidnModelUrl'];
      const epsRaw = opts.extensions?.['vitrum.ptWebgl.oidnExecutionProviders'];
      const eps = Array.isArray(epsRaw)
        ? (epsRaw.filter((p) => p === 'webnn' || p === 'webgpu' || p === 'wasm') as Array<'webnn' | 'webgpu' | 'wasm'>)
        : undefined;
      const dispatcherOpts =
        eps !== undefined && eps.length > 0
          ? { modelUrl: modelUrl as string, executionProviders: eps }
          : { modelUrl: modelUrl as string };
      this.#oidnDispatcher = new OIDNFinalDispatcher(dispatcherOpts, opts.oidnBridgeLoader);
    } else {
      this.#oidnDispatcher = null;
    }

    // Per-engine IBL bake cache. Construct with default LRU capacity (matches
    // the previous module-singleton sizing). Capacity is overridable from
    // extensions for hosts that scrub atmospheric params more aggressively
    // than the default day-cycle bucket count.
    const requestedCacheCapacity = opts.extensions?.['vitrum.ptWebgl.iblBakerMaxEntries'];
    const cacheOpts =
      typeof requestedCacheCapacity === 'number' && Number.isFinite(requestedCacheCapacity)
        ? { maxEntries: Math.max(1, Math.floor(requestedCacheCapacity)) }
        : undefined;
    this.#iblBakerCache = new IblBakerCache(cacheOpts);
  }

  /**
   * Bake (or fetch the cached bake of) an analytic Preetham sky equirect using
   * this engine's per-instance {@link IblBakerCache}. The returned
   * `DataTexture` is suitable for `scene.environment` and is owned by the
   * engine — DO NOT dispose it directly. Eviction and {@link dispose} handle
   * cleanup.
   *
   * The previous module-level free-function `bakeSkyEquirect` was removed
   * on 2026-05-18 (W6-E2 follow-up); every host should call this per-engine
   * method so cache contents stay bound to the renderer that produced them.
   */
  bakeSkyEquirect(params: SkyParams): DataTexture {
    if (this.#slot.get() === 'disposed') {
      throw new Error('bakeSkyEquirect: engine is disposed');
    }
    return this.#iblBakerCache.bake(this.#renderer, params);
  }

  /**
   * Inspect the engine's IBL bake cache — primarily for tests, debug overlays,
   * and hosts that want to surface cache-occupancy telemetry. Mutating the
   * returned reference is not supported; use {@link bakeSkyEquirect} and
   * {@link dispose} as the public seams.
   */
  get iblBakerCache(): IblBakerCache {
    return this.#iblBakerCache;
  }

  /**
   * Sprint 10c — host-driven BDPT frame advance. Updates the fork's
   * `uBdptLightPathTex` uniform so the next `renderFrame` call's
   * connection pass reads from the supplied texture for cached light
   * vertices.
   *
   * Hosts are expected to populate the texture via their own light-subpath
   * draw pass BEFORE calling this method; the engine doesn't own that
   * draw call (it's a fork shader the host must drive). See
   * {@link BdptLightPathBuffer} in `@vitrum/pt-webgl/bdptLightPathBuffer`
   * for the recommended host-side texture lifecycle.
   *
   * Calling this method on an engine that didn't opt in to BDPT (no
   * `extensions['vitrum.ptWebgl.bdpt']: true` at construction) is a
   * no-op — the fork's `FEATURE_BDPT` define is unset and the connection
   * GLSL is compiled out.
   *
   * @param lightPathTex - The light-subpath texture from your host helper
   *   (typically `BdptLightPathBuffer.texture`). Pass `null` to disable
   *   BDPT for the next frame as a safety guard.
   */
  /**
   * Populate a {@link BdptLightPathBuffer} via the fork GPU light-subpath pass when
   * hardware GL is available; otherwise CPU bounce-0 fill.
   */
  fillBdptLightPath(buffer: BdptLightPathBuffer, frameSeed: number): void {
    this.#assertLive('fillBdptLightPath');
    if (!this.#bdpt) return;
    const scene = this.#vitrumScene;
    if (scene == null) {
      throw new Error('fillBdptLightPath: call setScene() first');
    }
    const tracer = this.#pathTracer as unknown as WebGLPathTracerCompat;
    const useGpu =
      !this.#bdptCpuFill &&
      (!isSoftwareGlRenderer(this.#limits.renderer) || bdptForceGpuBind()) &&
      typeof tracer.renderBdptLightSubpathPass === 'function';
    buffer.fillFromScene(
      this.#renderer,
      scene,
      frameSeed,
      useGpu ? (tracer as BdptLightSubpathTracer) : null,
    );
  }

  bdptAdvanceFrame(lightPathTex: Texture | null): void {
    if (!this.#bdpt) return;
    const softwareGl = isSoftwareGlRenderer(this.#limits.renderer) && !bdptForceGpuBind();
    driveForkMaterialUniforms(
      this.#pathTracer,
      {
        strategy: this.#causticStrategy,
        mneeMaxIterations: this.#mneeMaxIterations,
        mneeMaxChainLength: this.#mneeMaxChainLength,
        spectralRendering: this.#spectralRendering,
        radianceClamp: this.#radianceClamp,
      },
      {
        enabled: this.#bdpt && !softwareGl,
        maxLightBounces: this.#bdptMaxLightBounces,
        lightPathTex: softwareGl ? null : lightPathTex,
      },
    );
  }

  get state(): EngineState {
    return this.#slot.get();
  }

  /** C2 — whether the current scene structurally needs TLAS (pt-webgl still uses merged BVH). */
  getSceneTlasAudit(): PtWebglTlasAudit | null {
    return this.#lastTlasAudit;
  }

  get capabilities(): EngineCapabilities {
    const experimental = new Set<string>();
    if (this.#bdpt) experimental.add('bdpt-approximate');
    if (this.#spectralRendering) experimental.add('spectral-jakob-hanika-placeholder');
    if (this.#lastTlasAudit?.needsTlas) experimental.add('merged-bvh-only');
    return {
      supportsIncrementalScene: true,
      incrementalPatchSupport: {
        transform: true,
        positions: true,
        material: true,
        emitter: true,
        topology: false,
      },
      supportsAuxBuffers: false,
      accumulates: true,
      maxSamplesPerPixel: this.#maxSamplesLimit,
      maxBounces: this.#maxBouncesLimit,
      supportedAnalyticShapes: new Set(),
      supportedPrimitiveKinds: new Set<ScenePrimitive['kind']>(['mesh', 'skinned-mesh']),
      supportedEmitterKinds: new Set<SceneEmitter['kind']>([
        'directional',
        'rect-area',
        'disc-area',
        'point',
        'spot',
        'mesh-area',
      ]),
      supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri']),
      presentationMode: 'offscreen-texture',
      ...(experimental.size > 0 ? { experimentalFeatures: experimental } : {}),
      causticStrategy: this.#causticStrategy,
    };
  }

  #assertLive(method: string): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error(`${method}: engine is disposed`);
    }
    if (this.#vitrumScene == null) {
      throw new Error(`${method}: call setScene() before ${method}`);
    }
  }

  #detectDeviceLimits(): DeviceLimits {
    const gl = this.#renderer.getContext();
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    const rendererFromDebug = debugInfo == null
      ? null
      : gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    const rendererFallback = gl.getParameter(gl.RENDERER);
    const maxTextureSize = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE));
    const maxRenderbufferSize = Number(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE));
    return {
      maxTextureSize: Number.isFinite(maxTextureSize) && maxTextureSize > 0 ? maxTextureSize : 4096,
      maxRenderbufferSize: Number.isFinite(maxRenderbufferSize) && maxRenderbufferSize > 0
        ? maxRenderbufferSize
        : 4096,
      renderer: typeof rendererFromDebug === 'string' && rendererFromDebug.length > 0
        ? rendererFromDebug
        : typeof rendererFallback === 'string'
          ? rendererFallback
          : 'unknown WebGL renderer',
    };
  }

  #estimateRenderTargetBytes(width: number, height: number): number {
    return (
      width *
      height *
      BYTES_PER_RGBA16F_PIXEL *
      ESTIMATED_RENDER_TARGET_COUNT +
      DEFAULT_RENDER_TARGET_OVERHEAD_BYTES
    );
  }

  #planRenderSize(width: number, height: number): RenderSizePlan {
    const requestedWidth = Math.max(1, Math.floor(width));
    const requestedHeight = Math.max(1, Math.floor(height));
    const maxDimension = Math.max(1, Math.min(this.#limits.maxTextureSize, this.#limits.maxRenderbufferSize));
    let scale = Math.min(1, maxDimension / requestedWidth, maxDimension / requestedHeight);
    let guardrail: string | null = scale < 1
      ? `capped to WebGL max render dimension ${maxDimension}`
      : null;
    let plannedWidth = Math.max(1, Math.floor(requestedWidth * scale));
    let plannedHeight = Math.max(1, Math.floor(requestedHeight * scale));
    let estimatedBytes = this.#estimateRenderTargetBytes(plannedWidth, plannedHeight);
    if (estimatedBytes > this.#schedulerOptions.renderTargetBudgetBytes) {
      const targetBytes = Math.max(
        1,
        this.#schedulerOptions.renderTargetBudgetBytes - DEFAULT_RENDER_TARGET_OVERHEAD_BYTES,
      );
      const pixelBytes = Math.max(1, plannedWidth * plannedHeight * BYTES_PER_RGBA16F_PIXEL * ESTIMATED_RENDER_TARGET_COUNT);
      const memoryScale = Math.min(1, Math.sqrt(targetBytes / pixelBytes));
      scale *= memoryScale;
      plannedWidth = Math.max(1, Math.floor(requestedWidth * scale));
      plannedHeight = Math.max(1, Math.floor(requestedHeight * scale));
      estimatedBytes = this.#estimateRenderTargetBytes(plannedWidth, plannedHeight);
      guardrail = guardrail == null
        ? `downscaled to fit ${Math.round(this.#schedulerOptions.renderTargetBudgetBytes / 1024 / 1024)} MiB render-target budget`
        : `${guardrail}; downscaled to fit render-target budget`;
    }
    return {
      width: plannedWidth,
      height: plannedHeight,
      estimatedBytes,
      guardrail,
    };
  }

  /**
   * Run a variance-aware adaptive tile-repeat pass at most once every
   * `#pixelAdaptiveCadence` samples. Writes the resulting per-tile repeat
   * factors to `#pathTracer.tileRepeatFactors`. Falls back to clearing the
   * factors when adaptive sampling is disabled.
   */
  #updateAdaptiveTileFactors(
    sppBefore: number,
    tilesX: number,
    tilesY: number,
    tileCount: number,
    w: number,
    h: number,
  ): void {
    const tracerCompat = this.#pathTracer as unknown as WebGLPathTracerCompat;
    if (this.#pixelAdaptiveSampling && this.#additiveAccumulation && this.#tileVariancePass != null) {
      if (sppBefore >= 2 && sppBefore % this.#pixelAdaptiveCadence === 0) {
        computeAdaptiveTileRepeatFactors(
          this.#tileVariancePass,
          this.#renderer,
          this.#pathTracer.target.texture as unknown as import('three').Texture,
          w,
          h,
          tilesX,
          tilesY,
          this.#tileFactorsScratch,
        );
        tracerCompat.tileRepeatFactors = this.#tileFactorsScratch.subarray(0, tileCount);
      }
    } else if (!this.#pixelAdaptiveSampling) {
      tracerCompat.tileRepeatFactors = null;
    }
  }

  #updateScheduler(batchMs: number): void {
    if (!this.#schedulerOptions.adaptive || this.#schedulerOptions.targetBatchMs <= 0) return;
    if (this.#contextLost) {
      this.#samplesPerFrame = 1;
      this.#tileSize = Math.min(this.#schedulerOptions.maxTileSize, Math.max(this.#tileSize, DEFAULT_TILE_SIZE));
      return;
    }
    const target = this.#schedulerOptions.targetBatchMs;
    if (batchMs > target * 1.35) {
      this.#samplesPerFrame = Math.max(
        this.#schedulerOptions.minSamplesPerFrame,
        Math.floor(this.#samplesPerFrame * 0.5),
      );
      if (batchMs > target * 2 && this.#tileSize < this.#schedulerOptions.maxTileSize) {
        this.#tileSize += 1;
      }
      return;
    }
    if (batchMs < target * 0.55 && this.#samplesPerFrame < this.#schedulerOptions.maxSamplesPerFrame) {
      this.#samplesPerFrame = Math.min(
        this.#schedulerOptions.maxSamplesPerFrame,
        Math.max(this.#samplesPerFrame + 1, Math.ceil(this.#samplesPerFrame * 1.2)),
      );
      if (batchMs < target * 0.25 && this.#tileSize > this.#schedulerOptions.initialTileSize) {
        this.#tileSize -= 1;
      }
    }
  }

  setScene(scene: Scene): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('setScene: engine is disposed');
    }
    if (this.#threeSceneRoot != null) {
      disposeObject3DTree(this.#threeSceneRoot);
    }
    // OIDN dispatcher: scene swap invalidates the accumulator, so the
    // cached denoised image is also stale. Drop it.
    this.#oidnDispatcher?.invalidate();
    this.#vitrumScene = scene;
    this.#lastTlasAudit = auditPtWebglSceneForTlas(scene);
    if (this.#lastTlasAudit.needsTlas) {
      console.warn(`[vitrum/pt-webgl] ${this.#lastTlasAudit.detail}`);
    }
    this.#cameraSignature = '';
    const threeScene = vitrumSceneToThree(scene);
    this.#threeSceneRoot = threeScene;
    const tracerCompat = this.#pathTracer as unknown as WebGLPathTracerCompat;
    tracerCompat.setScene(threeScene, this.#camera);
    // Sprint 10c: BDPT bridge — enabled=true only when the extension flag is set.
    // lightPathTex is null here (no WebGL texture at setScene time); hosts that
    // need per-frame BDPT must call driveForkMaterialUniforms() after supplying
    // their ping-pong texture from the light-subpath draw pass.
    driveForkMaterialUniforms(
      this.#pathTracer,
      {
        strategy: this.#causticStrategy,
        mneeMaxIterations: this.#mneeMaxIterations,
        mneeMaxChainLength: this.#mneeMaxChainLength,
        spectralRendering: this.#spectralRendering,
        radianceClamp: this.#radianceClamp,
      },
      {
        enabled: this.#bdpt,
        maxLightBounces: this.#bdptMaxLightBounces,
        lightPathTex: null, // populated per-frame by host after light-subpath draw pass
      },
    );
  }

  /**
   * Apply an environment-only update without rebuilding geometry, materials,
   * or the BVH. Hosts call this for fast timeOfDay scrubs where only the
   * HDRI texture / intensity / rotation changes.
   *
   * Cost: one accumulator-clear (one frame's worth of work) — no BVH rebuild
   * and no GPU geometry / material re-upload. Compare to `setScene()`, which
   * disposes the entire converted three.js subtree, rebuilds it via
   * `vitrumSceneToThree`, and triggers a full BVH rebuild inside
   * WebGLPathTracer.
   *
   * Implementation: mutates the engine's INTERNAL THREE.Scene root via the
   * shared `applyEnvironment` helper from @vitrum/three-bindings (matches
   * the env-application logic vitrumSceneToThree uses on the freshly created
   * scene), then calls `WebGLPathTracer.updateEnvironment()` which reads
   * the env uniforms back off the cached scene reference. The fork's
   * `updateEnvironment()` already calls `reset()` internally.
   *
   * Pre-conditions: a successful `setScene()` must have been called first;
   * otherwise the engine has no internal scene to mutate.
   *
   * Pass `null` to clear the environment (equivalent to `{ kind: 'none' }`).
   */
  updateEnvironment(env: SceneEnvironment | null): void {
    if (this.#slot.get() === 'disposed') {
      throw new Error('updateEnvironment: engine is disposed');
    }
    if (this.#threeSceneRoot == null || this.#vitrumScene == null) {
      throw new Error('updateEnvironment: call setScene() before updateEnvironment()');
    }
    // Mutate the existing internal THREE.Scene in place (no rebuild, no
    // dispose, no replacement). The fork caches a reference to this scene
    // when setScene() was called, so it will see the new env on its next
    // updateEnvironment() call.
    applyEnvironment(this.#threeSceneRoot, env);
    const tracerCompat = this.#pathTracer as unknown as WebGLPathTracerCompat;
    if (typeof tracerCompat.updateEnvironment === 'function') {
      tracerCompat.updateEnvironment();
    } else {
      // Fork API gap: should never happen with the vendored
      // three-gpu-pathtracer fork (WebGLPathTracer.updateEnvironment was
      // present at fork-time and continues to exist). Reset the accumulator
      // anyway so the host sees fresh samples on the next render.
      console.warn(
        '[pt-webgl] WebGLPathTracer.updateEnvironment() not available; ' +
          'falling back to accumulator reset only — IBL uniforms will not update until next setScene()',
      );
      tracerCompat.reset();
    }
    // OIDN dispatcher: an env swap clears the accumulator (the fork's
    // updateEnvironment() calls reset() internally), so any cached
    // denoised image is stale relative to the new lighting. Drop it.
    this.#oidnDispatcher?.invalidate();
    // Cache the env on the vitrum scene record so subsequent reads of the
    // engine's vitrum scene state reflect the env update. The Scene type's
    // environment field is non-nullable; collapse a null env to `{ kind: 'none' }`.
    const nextEnv: SceneEnvironment = env ?? { kind: 'none' };
    this.#vitrumScene = { ...this.#vitrumScene, environment: nextEnv };
  }

  updatePrimitive(_id: string, _patch: Partial<ScenePrimitive>): void {
    this.#assertLive('updatePrimitive');
    if (this.#vitrumScene == null || this.#threeSceneRoot == null) {
      throw new Error('updatePrimitive: call setScene() before updatePrimitive()');
    }
    if (isMaterialOnlyPrimitivePatch(_patch)) {
      const mesh = findMeshByPrimitiveId(this.#threeSceneRoot, _id);
      if (mesh == null) {
        throw new Error(`updatePrimitive: primitive "${_id}" not found in internal THREE scene`);
      }
      applyVitrumMaterialToMesh(mesh, _patch.material!);
      const tracerCompat = this.#pathTracer as unknown as WebGLPathTracerCompat;
      if (typeof tracerCompat.updateMaterials === 'function') {
        tracerCompat.updateMaterials();
      } else {
        tracerCompat.reset();
      }
      this.#oidnDispatcher?.invalidate();
      this.#vitrumScene = patchPrimitiveInScene(this.#vitrumScene, _id, _patch);
      return;
    }
    if (isTransformOnlyPrimitivePatch(_patch)) {
      const mesh = findMeshByPrimitiveId(this.#threeSceneRoot, _id);
      if (mesh == null) {
        throw new Error(`updatePrimitive: primitive "${_id}" not found in internal THREE scene`);
      }
      const transform = (_patch as Partial<MeshPrimitive>).transform;
      if (transform != null && transform.length >= 16) {
        const m = new Matrix4().fromArray(Array.from(transform));
        mesh.matrix.copy(m);
        mesh.matrixWorld.copy(m);
        mesh.matrixAutoUpdate = false;
      }
      if (!refreshPathTracerSceneGeometry(this.#pathTracer, this.#threeSceneRoot)) {
        const next = patchPrimitiveInScene(this.#vitrumScene, _id, _patch);
        this.setScene(next);
        return;
      }
      this.#oidnDispatcher?.invalidate();
      this.#vitrumScene = patchPrimitiveInScene(this.#vitrumScene, _id, _patch);
      return;
    }
    if (isPositionsOnlyPrimitivePatch(_patch)) {
      const mesh = findMeshByPrimitiveId(this.#threeSceneRoot, _id);
      if (mesh == null) {
        throw new Error(`updatePrimitive: primitive "${_id}" not found in internal THREE scene`);
      }
      const meshPatch = _patch as Partial<MeshPrimitive>;
      if (!applyPositionsPatchToMesh(mesh, meshPatch)) {
        const next = patchPrimitiveInScene(this.#vitrumScene, _id, _patch);
        this.setScene(next);
        return;
      }
      if (!refreshPathTracerSceneGeometry(this.#pathTracer, this.#threeSceneRoot)) {
        const next = patchPrimitiveInScene(this.#vitrumScene, _id, _patch);
        this.setScene(next);
        return;
      }
      this.#oidnDispatcher?.invalidate();
      this.#vitrumScene = patchPrimitiveInScene(this.#vitrumScene, _id, _patch);
      return;
    }
    const next = patchPrimitiveInScene(this.#vitrumScene, _id, _patch);
    this.setScene(next);
  }

  updateEmitter(_id: string, _patch: Partial<SceneEmitter>): void {
    this.#assertLive('updateEmitter');
    if (this.#vitrumScene == null || this.#threeSceneRoot == null) {
      throw new Error('updateEmitter: call setScene() before updateEmitter()');
    }
    if (isEmitterOnlyPatch(_patch)) {
      const next = patchEmitterInScene(this.#vitrumScene, _id, _patch);
      this.#vitrumScene = next;
      const tracerCompat = this.#pathTracer as unknown as WebGLPathTracerCompat;
      if (typeof tracerCompat.updateLights === 'function') {
        tracerCompat.updateLights();
      } else {
        tracerCompat.reset();
      }
      this.#oidnDispatcher?.invalidate();
      return;
    }
    const next = patchEmitterInScene(this.#vitrumScene, _id, _patch);
    this.setScene(next);
  }

  #makeCameraSignature(input: FrameInput): string {
    const viewport = input.viewport;
    return [
      ...input.viewMatrix,
      ...input.projMatrix,
      ...input.cameraPosition,
      viewport.width,
      viewport.height,
      viewport.devicePixelRatio,
    ].join(',');
  }

  renderFrame(input: FrameInput): PTEngineWebGL2FrameOutput {
    this.#assertLive('renderFrame');
    if (this.#slot.get() === 'paused') {
      const spp = this.#pathTracer.samples;
      const cap = Math.max(1, Math.min(this.#lastTargetSpp, this.#maxSamplesLimit));
      const primaryRadiance = ForkAccess.getRenderTexture(this.#pathTracer);
      if (primaryRadiance == null) {
        return {
          kind: 'skipped',
          samplesAccumulated: 0,
          isConverged: false,
          telemetry: this.#lastTelemetry,
        };
      }
      return {
        kind: 'rendered',
        primaryRadiance: asBackendTexture<'webgl', typeof primaryRadiance>(primaryRadiance),
        samplesAccumulated: spp,
        isConverged: spp >= cap,
        telemetry: this.#lastTelemetry,
      };
    }

    const cameraSignature = this.#makeCameraSignature(input);
    if (cameraSignature !== this.#cameraSignature) {
      applyFrameToPerspectiveCamera(this.#camera, input);
      (this.#pathTracer as unknown as WebGLPathTracerCompat).setCamera(this.#camera);
      this.#cameraSignature = cameraSignature;
      this.#oidnDispatcher?.invalidate();
    }

    const q = input.quality ?? {};
    const b = Math.min(q.bounces ?? this.#maxBouncesLimit, this.#maxBouncesLimit);
    const targetSpp = Math.min(q.samplesTarget ?? 16, this.#maxSamplesLimit);
    this.#lastTargetSpp = targetSpp;
    this.#pathTracer.bounces = b;
    this.#pathTracer.transmissiveBounces = Math.min(b, this.#maxBouncesLimit);
    this.#pathTracer.filterGlossyFactor = q.filteredGlossyFactor ?? 0;

    const factor = q.resolutionFactor ?? 1;
    const requestedWidth = Math.max(1, Math.floor(input.viewport.width * factor));
    const requestedHeight = Math.max(1, Math.floor(input.viewport.height * factor));
    const sizePlan = this.#planRenderSize(requestedWidth, requestedHeight);
    const w = sizePlan.width;
    const h = sizePlan.height;
    if (this.#tileSize <= 1 || w * h <= MIN_RESOLUTION_FOR_TILING) {
      this.#pathTracer.tiles.set(1, 1);
    } else {
      this.#pathTracer.tiles.set(this.#tileSize, this.#tileSize);
    }
    if (w !== this.#lastRenderWidth || h !== this.#lastRenderHeight) {
      this.#renderer.setSize(w, h, false);
      this.#lastRenderWidth = w;
      this.#lastRenderHeight = h;
    }

    let spp = this.#pathTracer.samples;
    const sppBefore = spp;
    const batchStart = nowMs();
    const samplesThisFrame = Math.min(this.#samplesPerFrame, Math.max(0, Math.ceil(targetSpp - spp)));

    const tilesX = Math.max(1, Math.floor(this.#pathTracer.tiles.x));
    const tilesY = Math.max(1, Math.floor(this.#pathTracer.tiles.y));
    const tileCount = tilesX * tilesY;

    this.#updateAdaptiveTileFactors(sppBefore, tilesX, tilesY, tileCount, w, h);

    for (let i = 0; i < samplesThisFrame && spp < targetSpp; i += 1) {
      this.#pathTracer.renderSample();
      spp = this.#pathTracer.samples;
    }
    const batchMs = Math.max(0, nowMs() - batchStart);
    const sppDelta = Math.max(0, spp - sppBefore);
    this.#updateScheduler(batchMs);
    const msPerSample = sppDelta > 0 ? batchMs / sppDelta : null;
    this.#lastTelemetry = {
      qualityMode: this.#schedulerOptions.qualityMode,
      renderer: this.#limits.renderer,
      requestedWidth,
      requestedHeight,
      renderWidth: w,
      renderHeight: h,
      samplesPerFrame: samplesThisFrame,
      tileSize: this.#tileSize <= 1 || w * h <= MIN_RESOLUTION_FOR_TILING ? 1 : this.#tileSize,
      batchMs,
      msPerSample,
      sppDelta,
      sppPerSecond: sppDelta > 0 && batchMs > 0 ? (sppDelta * 1000) / batchMs : null,
      estimatedRenderTargetBytes: sizePlan.estimatedBytes,
      renderTargetBudgetBytes: this.#schedulerOptions.renderTargetBudgetBytes,
      guardrail: this.#contextLost
        ? 'webgl context loss observed; scheduler reduced workload'
        : sizePlan.guardrail,
      additiveAccumulation: this.#additiveAccumulation,
      pixelAdaptiveSampling: this.#pixelAdaptiveSampling,
    };
    // T3.E telemetry hooks. We fire AFTER #lastTelemetry is populated so
    // subscribers can also peek via the FrameOutput.telemetry passthrough.
    if (this.#frameSubs.length > 0) {
      const stats: FrameStats = {
        frameTimeMs: batchMs,
        spp,
        // GPU-memory budget — pt-webgl wraps three-gpu-pathtracer, whose
        // render-target textures are opaque to us. We reuse the scheduler's
        // existing `sizePlan.estimatedBytes` (a worst-case `width × height ×
        // RGBA16F × renderTargetCount + overhead` estimate, already plumbed
        // into the telemetry payload) as the scalar `estimatedGpuMemoryBytes`.
        // The structured `gpuMemoryBytes` breakdown is intentionally omitted:
        // the underlying fork doesn't expose per-pass texture handles, so a
        // by-category split would either be invented or stale.
        estimatedGpuMemoryBytes: sizePlan.estimatedBytes,
      };
      for (const sub of this.#frameSubs) {
        try { sub(stats); } catch { /* swallow */ }
      }
    }
    if (this.#progressSubs.length > 0 && sppDelta > 0) {
      const target = Math.max(1, targetSpp);
      const progress: ProgressStats = {
        kind: 'pt-spp',
        current: spp,
        target,
        fraction: Math.min(1, spp / target),
      };
      for (const sub of this.#progressSubs) {
        try { sub(progress); } catch { /* swallow */ }
      }
    }

    const isConverged = spp >= targetSpp;
    // W11 follow-up: kick the OIDN dispatcher on every converged frame.
    // The dispatcher is internally idempotent — it short-circuits on
    // in-flight or already-completed inferences for the current
    // invalidation cohort, so calling unconditionally per converged
    // frame is the right shape (no need to gate here on "first converged
    // frame"). Readback uses the additive-accumulation branch when the
    // engine is in sum/count mode so the bridge sees averaged HDR RGB.
    if (this.#oidnDispatcher != null && isConverged && spp > 0) {
      this.#oidnDispatcher.kickIfReady(
        this.#renderer,
        this.#pathTracer.target as unknown as WebGLRenderTarget,
        w,
        h,
        this.#additiveAccumulation,
      );
    }

    const primaryRadiance = ForkAccess.getRenderTexture(this.#pathTracer);
    if (primaryRadiance == null) {
      return {
        kind: 'skipped',
        samplesAccumulated: 0,
        isConverged: false,
        telemetry: this.#lastTelemetry,
      };
    }
    return {
      kind: 'rendered',
      primaryRadiance: asBackendTexture<'webgl', typeof primaryRadiance>(primaryRadiance),
      samplesAccumulated: spp,
      isConverged,
      telemetry: this.#lastTelemetry,
    };
  }

  reset(): void {
    if (this.#slot.get() === 'disposed') return;
    this.#pathTracer.reset();
    // OIDN dispatcher: the accumulator just cleared, so the cached
    // denoised image (and any in-flight inference) is stale.
    this.#oidnDispatcher?.invalidate();
  }

  /**
   * HDR accumulation buffer backing `primaryRadiance` / fork canvas preview.
   * Texels are running averages unless additive accumulation is enabled (then RGB sum / alpha count).
   */
  getAccumulationRenderTarget(): WebGLRenderTarget {
    this.#assertLive('getAccumulationRenderTarget');
    return this.#pathTracer.target as unknown as WebGLRenderTarget;
  }

  /**
   * W11 follow-up — returns the most recently completed OIDN-denoised
   * RGB image from the internal {@link OIDNFinalDispatcher}, or null when:
   *
   *  - the engine was not constructed with `denoiser: 'oidn-final'`;
   *  - the dispatcher has not yet completed its first inference for the
   *    current accumulator cohort;
   *  - the accumulator was just invalidated (setScene / reset /
   *    updateEnvironment) and no converged frame has rendered since.
   *
   * Layout matches `OIDNDenoiseInputs.color`: row-major interleaved RGB,
   * `width × height × 3` floats. Hosts that want to display the
   * denoised result can pass this to `writeTonemappedRgbToCanvas`
   * (the cornell-box demo uses exactly that path today, just with a
   * host-side `denoiseFinal` call — this method lets the engine own
   * the denoise instead).
   */
  getDenoisedFrame(): DenoisedFrame | null {
    return this.#oidnDispatcher?.getLatestDenoised() ?? null;
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

  // ── Telemetry (T3.E) ───────────────────────────────────────────────────

  /** Subscribe to per-frame stats (frameTimeMs + running SPP). Returns an
   *  unsubscribe function. Subscribers that throw are swallowed. */
  onFrame(cb: (stats: FrameStats) => void): () => void {
    this.#frameSubs.push(cb);
    return () => {
      const i = this.#frameSubs.indexOf(cb);
      if (i >= 0) this.#frameSubs.splice(i, 1);
    };
  }

  /** Subscribe to SPP-accumulation progress (`kind: 'pt-spp'`). Returns
   *  an unsubscribe function. Subscribers that throw are swallowed.
   *  Fired only on frames that actually advanced SPP — paused / no-work
   *  frames don't emit. */
  onProgress(cb: (progress: ProgressStats) => void): () => void {
    this.#progressSubs.push(cb);
    return () => {
      const i = this.#progressSubs.indexOf(cb);
      if (i >= 0) this.#progressSubs.splice(i, 1);
    };
  }

  dispose(): void {
    if (this.#slot.get() === 'disposed') return;
    if (this.#contextLostHandler != null) {
      try {
        this.#renderer.domElement?.removeEventListener?.('webglcontextlost', this.#contextLostHandler);
      } catch {}
      this.#contextLostHandler = null;
    }
    if (this.#tileVariancePass != null) {
      this.#tileVariancePass.dispose();
      this.#tileVariancePass = null;
    }
    if (this.#threeSceneRoot != null) {
      disposeObject3DTree(this.#threeSceneRoot);
      this.#threeSceneRoot = null;
    }
    this.#oidnDispatcher?.dispose();
    this.#iblBakerCache.dispose();
    this.#pathTracer.dispose();
    this.#frameSubs.length = 0;
    this.#progressSubs.length = 0;
    this.#slot.set('disposed');
  }
}

export const createPTEngine_WebGL2: EngineFactory<PTEngineWebGL2Options> = async (
  opts: PTEngineWebGL2Options,
): Promise<Engine> => {
  if (opts.device == null || typeof (opts.device as { getContext?: unknown }).getContext !== 'function') {
    throw new TypeError(
      'createPTEngine_WebGL2: device must be a THREE.WebGLRenderer instance (got null/undefined or an object without a getContext() method)',
    );
  }
  const glContext = (opts.device as { getContext(): unknown }).getContext();
  if (!(glContext instanceof WebGL2RenderingContext)) {
    throw new TypeError(
      'createPTEngine_WebGL2: device.getContext() must return a WebGL2RenderingContext; WebGL1 is not supported',
    );
  }

  const maxBounces = opts.maxBounces;
  if (maxBounces !== undefined && maxBounces < 1) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxBounces structural cap must be >= 1 (got ${maxBounces})`,
    );
  }

  const maxSpp = opts.maxSamplesPerPixel;
  if (maxSpp !== undefined && maxSpp < 1) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxSamplesPerPixel structural cap must be >= 1 (got ${maxSpp})`,
    );
  }

  // W11 follow-up — validate 'oidn-final' requires a model URL up front.
  // The OIDNFinalDispatcher constructor also throws on a missing URL, but
  // surfacing the error from the engine factory (with a pointer to the
  // exact extensions key) gives hosts a clearer integration error than a
  // late-bound throw from the dispatcher.
  if (opts.denoiser === 'oidn-final') {
    const modelUrl = opts.extensions?.['vitrum.ptWebgl.oidnModelUrl'];
    if (typeof modelUrl !== 'string' || modelUrl.length === 0) {
      throw new Error(
        "createPTEngine_WebGL2: denoiser: 'oidn-final' requires " +
          "extensions['vitrum.ptWebgl.oidnModelUrl'] (a URL or path to the OIDN ONNX model file). " +
          'See packages/shared-denoisers/src/oidnBridge.ts for model variants ' +
          '(oidn_rt_hdr.onnx for color-only, oidn_rt_hdr_alb_nrm.onnx for the aux-input variant).',
      );
    }
  }

  const renderer = opts.device;
  RectAreaLightUniformsLib.init();

  const pathTracer = new WebGLPathTracer(
    renderer as unknown as ConstructorParameters<typeof WebGLPathTracer>[0],
  );
  pathTracer.renderDelay = 0;
  pathTracer.minSamples = 1;
  pathTracer.dynamicLowRes = false;
  pathTracer.multipleImportanceSampling = true;
  pathTracer.tiles.set(1, 1);

  const camera = new PerspectiveCamera();

  const slot = makeStateSlot();
  const engine = new PTEngineWebGL2(
    opts,
    { renderer, pathTracer, camera },
    slot,
  );
  slot.set('ready');
  return engine;
}
