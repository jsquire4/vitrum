/**
 * Per-frame render orchestration for {@link HybridEngine} (W4e).
 */
import type { FrameInput, FrameOutput, FrameStats, EngineDebugSurface, EngineState, Scene } from '@vitrum/core';
import { asBackendTexture } from '@vitrum/core';
import type * as THREE from 'three';
import type { DDGI } from './ddgi/DDGI.js';
import { packDDGIGridParams } from './ddgi/ddgiGridUbo.js';
import { propagateBvhToGiSubsystems } from './HybridEngineGiPropagation.js';
import type { RCSubsystem } from './HybridEngineRC.js';
import type { Tunables } from './HybridEngineTuning.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';
import type { GpuSkinningSubsystem } from './skin/GpuSkinningSubsystem.js';

export const HYBRID_FRAME_SKIP_OUTPUT: FrameOutput = {
  kind: 'skipped',
  samplesAccumulated: 0,
  isConverged: false,
};

export interface HybridEngineFrameDiag {
  initStart: number;
  initCount: number;
  disposeCount: number;
  skipNoPipeline: number;
  skipNoBvh: number;
  skipNoSwapView: number;
  skipFrameInterval: number;
  framesDispatched: number;
  lastReportTs: number;
}

/** Runtime-mutable lighting cluster (the engine's `updateLighting()` mutates
 *  these). Grouped so a new lighting field is one edit in the engine's
 *  `_lightingSnapshot()` rather than one per dependency builder. */
export interface HybridLightingDeps {
  primaryLightDir: [number, number, number];
  primaryLightIntensity: number;
  skyTint: [number, number, number];
  skyIrradiance: number;
}

/** Tuple-typed denoiser-filter cluster (firefly clamp + per-channel atrous
 *  sigmas). Lives outside the number-only {@link Tunables} table because the
 *  values are tuples; grouped for the same one-edit reason as the lighting
 *  cluster. */
export interface HybridDenoiserFilterDeps {
  indirectFireflyClamp: readonly [number, number, number];
  atrousDirectSigmas: readonly [number, number, number];
  atrousIndirectSigmas: readonly [number, number, number];
}

export interface HybridEngineFrameDeps extends HybridLightingDeps, HybridDenoiserFilterDeps {
  state: EngineState;
  debug: boolean;
  dbg: HybridEngineFrameDiag | null;
  pipeline: WalkaroundGPUPipeline | null;
  bvhBuffers: SceneBVHBuffers | null;
  /** When true, caller already called reset() and frame should skip. */
  consumeRebuildKeyChange: () => boolean;
  targetFrameIntervalMs: number | null;
  getLastFrameTs: () => number;
  setLastFrameTs: (ts: number) => void;
  width: number;
  height: number;
  skinning: GpuSkinningSubsystem | null;
  lastScene: Scene | null;
  runSkinning: () => void;
  ddgiOn: boolean;
  isLayerEnabled: (layer: string) => boolean;
  ddgi: DDGI;
  ddgiTraversalScene: THREE.Scene | null;
  ensureThreeSceneRoot: () => THREE.Scene | null;
  device: GPUDevice;
  tunables: Tunables;
  rc: RCSubsystem | null;
  rcWeight: number;
  frameSubs: ReadonlyArray<(stats: FrameStats) => void>;
  verbose: boolean;
  debugTimings: Array<{ t: number; ms: number }>;
  debugSurface: EngineDebugSurface;
  presentLastFrame: (view: GPUTextureView) => void;
}

export function fingerprintHybridPipelineRebuildKey(
  key: string | number | null | undefined,
): string {
  if (key === null || key === undefined) return '__null';
  if (typeof key === 'number') return Number.isNaN(key) ? '__n:NaN' : `__n:${key}`;
  return `__s:${key}`;
}

export function getPreferredSwapChainFormat(): GPUTextureFormat {
  return (typeof navigator !== 'undefined' && 'gpu' in navigator
    ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
        .getPreferredCanvasFormat?.() ?? 'bgra8unorm'
    : 'bgra8unorm');
}

function runDdgiAndRc(deps: HybridEngineFrameDeps, input: FrameInput): void {
  const ddgiLayerOn = deps.ddgiOn && deps.isLayerEnabled('ddgi');
  const ddgiScene = ddgiLayerOn
    ? (deps.ddgiTraversalScene ?? deps.ensureThreeSceneRoot())
    : null;

  // Per-frame BVH ⇒ GI-subsystem cascade — same owner the post-update /
  // post-publish paths use, but tlas-sync-only (allowRcSceneRebuild=false so
  // merged-mode RC is NOT rebuilt every frame), and the DDGI sync gated on the
  // resolved ddgi-layer scene exactly as before.
  propagateBvhToGiSubsystems({
    ddgi: deps.ddgi,
    rc: deps.rc,
    bvhBuffers: deps.bvhBuffers,
    lastScene: deps.lastScene,
    syncDdgi: ddgiLayerOn && ddgiScene != null,
    allowRcSceneRebuild: false,
    ensureThreeSceneRoot: deps.ensureThreeSceneRoot,
  });

  if (ddgiLayerOn && ddgiScene != null) {
    void deps.ddgi.updateFrame({
      scene: ddgiScene,
      device: deps.device,
      enabled: true,
    });
    if (deps.ddgi.ready) {
      deps.ddgi.pass.setGlassMixScale(deps.tunables.glassMixScale);
    }
  }

  const pipeline = deps.pipeline!;
  if (deps.rc) {
    deps.rc.dispatchFrame({
      sunDirection: deps.primaryLightDir,
      sunColor: [
        deps.primaryLightIntensity,
        deps.primaryLightIntensity,
        deps.primaryLightIntensity,
      ],
      frameSeed: input.frameSeed,
      triIntersectEpsilon: deps.tunables.triIntersectEpsilon,
    });
    pipeline.setRCInputs(deps.rc.buildRCInputs(deps.rcWeight));
  } else {
    pipeline.setRCInputs(null);
  }

  if (!ddgiLayerOn) {
    pipeline.setDDGIInputs(null);
  } else if (deps.ddgi.ready) {
    const atlas = deps.ddgi.pass.getReadAtlasGPUTextures?.();
    if (atlas) {
      pipeline.setDDGIInputs({
        irradianceTex: atlas.irradiance,
        visibilityTex: atlas.visibility,
        gridParams: packDDGIGridParams(deps.ddgi.probeGrid.params),
      });
    }
  }
}

function emitFrameTelemetry(
  deps: HybridEngineFrameDeps,
  pipeline: WalkaroundGPUPipeline,
  dt: number,
  now: number,
): void {
  if (deps.frameSubs.length > 0) {
    const gpu = pipeline.lastGpuTimings;
    const gpuTotal = gpu?.['total'];
    const memBreakdown = deps.debugSurface.estimatedGpuMemoryBytes?.() ?? undefined;
    const stats: FrameStats = {
      frameTimeMs: dt,
      ...(gpuTotal !== undefined ? { gpuTimeMs: gpuTotal } : {}),
      ...(gpu ? { passTimings: gpu } : {}),
      spp: 1,
      ...(memBreakdown
        ? { gpuMemoryBytes: memBreakdown, estimatedGpuMemoryBytes: memBreakdown.total }
        : {}),
    };
    for (const sub of deps.frameSubs) {
      try {
        sub(stats);
      } catch (err) {
        if (deps.verbose) console.warn('[HybridEngine] onFrame subscriber threw', err);
      }
    }
  }

  if (deps.debug) {
    deps.debugTimings.push({ t: now, ms: dt });
    if (deps.debugTimings.length > 240) deps.debugTimings.shift();

    if (typeof window !== 'undefined') {
      const w = window as unknown as { __WGPU__?: { walkaround?: { frameTimings: unknown } } };
      if (w.__WGPU__?.walkaround) {
        const ft = w.__WGPU__.walkaround.frameTimings as Array<{ t: number; ms: number }>;
        if (Array.isArray(ft)) {
          ft.push({ t: now, ms: dt });
          if (ft.length > 240) ft.shift();
        }
      }
    }
  }
}

export function runHybridEngineFrame(deps: HybridEngineFrameDeps, input: FrameInput): FrameOutput {
  if (deps.state === 'paused' || deps.state === 'disposed' || deps.state === 'error') {
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  if (deps.consumeRebuildKeyChange()) {
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  const dbg = deps.dbg;
  const pipeline = deps.pipeline;
  const bvh = deps.bvhBuffers;
  if (!pipeline) {
    if (dbg) dbg.skipNoPipeline++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }
  if (!bvh) {
    if (dbg) dbg.skipNoBvh++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  const now = performance.now();
  if (
    deps.targetFrameIntervalMs !== null &&
    deps.getLastFrameTs() !== 0 &&
    now - deps.getLastFrameTs() < deps.targetFrameIntervalMs
  ) {
    if (dbg) dbg.skipFrameInterval++;
    const skipSwapView = input.swapChainView as GPUTextureView | undefined;
    if (skipSwapView) {
      deps.presentLastFrame(skipSwapView);
    }
    return HYBRID_FRAME_SKIP_OUTPUT;
  }
  deps.setLastFrameTs(now);

  const t0 = now;
  const swapView = input.swapChainView as GPUTextureView | undefined;
  const swapFmt =
    (input.swapChainFormat as GPUTextureFormat | undefined) ?? getPreferredSwapChainFormat();

  if (!swapView) {
    if (dbg) dbg.skipNoSwapView++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  if (dbg) {
    dbg.framesDispatched++;
    if (dbg.lastReportTs === 0) dbg.lastReportTs = now;
    if (now - dbg.lastReportTs > 5_000) {
      const elapsed = (now - dbg.lastReportTs) / 1_000;
      const gpu = pipeline.lastGpuTimings ?? {};
      const gpuTotal = gpu['total'];
      console.log('[hybrid:debug] rate (5s window)', {
        framesDispatched: dbg.framesDispatched,
        fps: (dbg.framesDispatched / elapsed).toFixed(2),
        skipReasons: {
          noPipeline: dbg.skipNoPipeline,
          noBvh: dbg.skipNoBvh,
          noSwapView: dbg.skipNoSwapView,
          frameInterval: dbg.skipFrameInterval,
        },
        gpuTotalMs: gpuTotal !== undefined ? +gpuTotal.toFixed(2) : 'n/a',
        gpuPerPassMs: gpu,
      });
      dbg.framesDispatched = 0;
      dbg.skipNoPipeline = 0;
      dbg.skipNoBvh = 0;
      dbg.skipNoSwapView = 0;
      dbg.skipFrameInterval = 0;
      dbg.lastReportTs = now;
    }
  }

  deps.runSkinning();
  runDdgiAndRc(deps, input);

  pipeline.renderFrame({
    viewMatrix: new Float32Array(input.viewMatrix),
    projMatrix: new Float32Array(input.projMatrix),
    prevViewMatrix: new Float32Array(input.prevViewMatrix ?? input.viewMatrix),
    prevProjMatrix: new Float32Array(input.prevProjMatrix ?? input.projMatrix),
    cameraPos: input.cameraPosition as [number, number, number],
    screenWidth: deps.width,
    screenHeight: deps.height,
    frameSeed: input.frameSeed,
    totalEmissivePower: bvh.totalEmissivePower ?? 1.0,
    emitterCount: bvh.emitters?.count ?? 0,
    primaryLightDir: deps.primaryLightDir,
    primaryLightIntensity: deps.primaryLightIntensity,
    skyTint: deps.skyTint,
    skyIrradiance: deps.skyIrradiance,
    ...deps.tunables,
    indirectFireflyClamp: deps.indirectFireflyClamp,
    bvhMode: bvh.bvhMode === 'tlas' ? 1 : 0,
    tlasNodeCount: bvh.tlas?.nodeCount ?? 0,
    atrousDirectSigmas: deps.atrousDirectSigmas,
    atrousIndirectSigmas: deps.atrousIndirectSigmas,
    swapChainView: swapView,
    swapChainFormat: swapFmt,
  });

  emitFrameTelemetry(deps, pipeline, performance.now() - t0, now);

  return {
    kind: 'rendered',
    primaryRadiance: asBackendTexture<'webgpu', GPUTextureView>(swapView),
    samplesAccumulated: 1,
    isConverged: false,
  };
}
