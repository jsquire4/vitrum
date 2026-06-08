// Top-level drop-in factory.
//
// Given a canvas + scene (THREE.Scene or vitrum Scene), createEngine()
//   - probes WebGPU vs WebGL2,
//   - picks the walkaround-hybrid backend (real-time GI) or pt-webgl
//     backend (converged path tracer),
//   - derives scale-sensitive defaults (Möller-Trumbore epsilon, camera-
//     move-reset threshold, emitter dist² floor, GTAO sigma) from the
//     scene's AABB diagonal D,
//   - constructs and owns the backend's device handle (GPUDevice for
//     walkaround, THREE.WebGLRenderer for pt-webgl) so the host doesn't
//     have to plumb GPU primitives,
//   - returns the @vitrum/core Engine contract with an idempotent dispose.
//
// Note on resize: HybridEngine (WebGPU) requires setSize(w, h) for in-flight
// resizes; attachVitrum() owns the ResizeObserver and calls it automatically.
// Generic PT engines honour FrameInput.viewport per-frame. createEngine()
// itself does NOT attach an observer.

import type { Scene, Engine, AdapterProfile } from '@vitrum/core';
import { auditSceneNeedsTlas, detectGpu } from '@vitrum/core';
import {
  createWalkaroundEngine_Hybrid,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
  type HybridEngineOptions,
} from '@vitrum/walkaround-hybrid';
import { probeAdapterProfile } from './adapterProfile.js';
// pt-webgl is the WebGL2 path-tracer backend; it (transitively) pulls the whole
// three-gpu-pathtracer stack + `three`. Import the runtime factory LAZILY (only
// `constructPathTracer` needs it) so a host taking exclusively the WebGPU path —
// e.g. `createProgressiveEngine` (walkaround + pt-webgpu, no WebGL2) — never has
// to resolve or typecheck against that module graph.
import {
  createPTEngine_WebGPU,
  ptWebgpuRequiredLimitsForAdapter,
  type PTEngineWebGPUOptions,
} from '@vitrum/pt-webgpu';

import { computeSceneAABB, type SceneAABB } from './sceneAABB.js';
import { wrapWithIdempotentDispose } from './idempotentDispose.js';
import type { GIStatePersistable } from './idempotentDispose.js';
import { configureWebGpuCanvas } from './configureWebGpuCanvas.js';
import {
  DEFAULT_PRIMARY_LIGHT_DIR,
  DEFAULT_PRIMARY_LIGHT_INTENSITY,
  DEFAULT_SKY_IRRADIANCE,
  DEFAULT_SKY_TINT,
  deriveScaleDefaults,
  pickBackend,
  type EnginePreference,
  type ScaleDefaults,
} from './createEngineScale.js';
import {
  isThreeScene,
  sceneFromThreeSceneLike,
  type ThreeSceneLike,
} from './threeSceneBridge.js';

export type { EnginePreference, ScaleDefaults };
export { pickBackend, deriveScaleDefaults };
export type { ThreeSceneLike } from './threeSceneBridge.js';

type WebGL2PathTracerAdvancedOptions = Record<string, unknown>;

interface ThreeWebGLRendererLike {
  dispose(): void;
  forceContextLoss?: () => void;
}

interface ThreeRuntimeModule {
  readonly WebGLRenderer: new (opts: {
    readonly canvas: HTMLCanvasElement;
    readonly antialias?: boolean;
    readonly preserveDrawingBuffer?: boolean;
  }) => ThreeWebGLRendererLike;
}

type WebGL2PathTracerOptionsLike = WebGL2PathTracerAdvancedOptions & {
  readonly device: ThreeWebGLRendererLike;
};

interface PtWebglModuleLike {
  readonly createPTEngine_WebGL2: (opts: WebGL2PathTracerOptionsLike) => Promise<Engine>;
}

/** When scene layout needs TLAS, default walkaround `bvhMode` unless host set one. */
export function mergeWalkaroundTlasExtension(
  advanced: Partial<HybridEngineOptions> | undefined,
  needsTlas: boolean,
): Partial<HybridEngineOptions> | undefined {
  if (!needsTlas) return advanced;
  const wh = advanced?.extensions?.['walkaround-hybrid'];
  if (wh?.bvhMode != null) return advanced;
  return {
    ...advanced,
    extensions: {
      ...(advanced?.extensions ?? {}),
      'walkaround-hybrid': { ...wh, bvhMode: 'tlas' },
    },
  };
}

// Re-exported for unit-test access (tests import it from this module's path).
// @internal — not part of the public `@vitrum/engine` API surface.
export { wrapWithIdempotentDispose } from './idempotentDispose.js';

export interface CreateEngineOptions {
  /** Canvas the engine renders into. Used to obtain the GPU context. */
  readonly canvas: HTMLCanvasElement;

  /** Scene description. Either a vitrum Scene or a THREE.Scene; THREE
   *  scenes are auto-converted via @vitrum/three-bindings. */
  readonly scene: Scene | ThreeSceneLike;

  /** Quality vs speed hint:
   *    'realtime' — prefer walkaround-hybrid (WebGPU; ~60fps target).
   *    'quality'  — prefer pt-webgl (WebGL2 path tracer; converged).
   *    'quality-webgpu' — prefer pt-webgpu when WebGPU is available, else pt-webgl.
   *    'auto'     — pick walkaround-hybrid if WebGPU + tris < 500k,
   *                 else pt-webgl. Default. */
  readonly prefer?: EnginePreference;

  /** Backend-specific overrides. Merged on top of the createEngine()-
   *  derived defaults; user-supplied keys win. Most users leave empty. */
  readonly advanced?: Partial<HybridEngineOptions> | WebGL2PathTracerAdvancedOptions | Partial<PTEngineWebGPUOptions>;

  /** Debug overlay opt-in. Forwarded to backend as `debug: true`. */
  readonly debug?: boolean;

  /** Phase-0 productization — callback invoked once with the graceful-
   *  degradation {@link AdapterProfile} when the walkaround-hybrid backend is
   *  selected (before device acquisition). Lets hosts read the JSON for a HUD
   *  / CI artifact (§4.1 / §10.3). Not called for the pt-webgl / pt-webgpu
   *  backends (they have their own tier selection). */
  readonly onAdapterProfile?: (profile: AdapterProfile) => void;
}

export async function createEngine(opts: CreateEngineOptions): Promise<Engine & Partial<GIStatePersistable>> {
  if (opts.canvas == null) {
    throw new TypeError('createEngine: opts.canvas is required');
  }
  if (opts.scene == null) {
    throw new TypeError('createEngine: opts.scene is required');
  }

  const sceneInputIsThree = isThreeScene(opts.scene);
  const vitrumScene: Scene = sceneInputIsThree
    ? await sceneFromThreeSceneLike(opts.scene)
    : (opts.scene);

  const aabb = computeSceneAABB(vitrumScene);
  const tlasAudit = auditSceneNeedsTlas(vitrumScene);
  const gpu = await detectGpu({ publishToWindow: false });
  const backend = pickBackend(
    opts.prefer ?? 'auto',
    gpu.isWebGPU,
    aabb.triangleCount,
    tlasAudit.needsTlas,
  );
  if (tlasAudit.needsTlas && backend === 'pt-webgl') {
    console.warn(`[vitrum/createEngine] ${tlasAudit.detail}`);
  }

  if (backend === 'walkaround-hybrid') {
    return await constructWalkaround(opts, vitrumScene, aabb, sceneInputIsThree, tlasAudit.needsTlas);
  }
  if (backend === 'pt-webgpu') {
    return await constructPathTracerWebGPU(opts, vitrumScene, sceneInputIsThree);
  }
  return await constructPathTracer(opts, vitrumScene, sceneInputIsThree);
}

// ────────────────────────────────────────────────────────────────────────────
// Backend constructors
// ────────────────────────────────────────────────────────────────────────────

/**
 * A pre-acquired adapter + device the caller owns. When passed to a backend
 * constructor, that constructor reuses them INSTEAD of minting (and disposing)
 * its own. This is the seam `createProgressiveEngine` uses to stand BOTH the
 * walkaround (realtime) and pt-webgpu (converged) engines up on ONE shared
 * GPUDevice — a prerequisite for cross-engine texture compatibility (the
 * progressive seed handoff binds the walkaround's output texture into the
 * converged engine's `seedAccumulator`, which is only legal same-device).
 *
 * When `null`/absent (the default `createEngine()` path) the constructor mints
 * its own adapter+device and destroys the device on dispose, exactly as before —
 * this whole type is invisible to the createEngine code path.
 *
 * `ownsDeviceLifecycle: false` tells the constructor NOT to `device.destroy()`
 * on the returned engine's dispose: the SHARED-device owner (the progressive
 * facade) destroys it once, after disposing both sub-engines.
 *
 * @internal — consumed only by `createProgressiveEngine`; not public API.
 */
export interface SharedDeviceCtx {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  /** Always false here — the facade owns the device. Present for intent clarity. */
  readonly ownsDeviceLifecycle: false;
}

/** @internal — reused by `createProgressiveEngine` to build the realtime engine
 *  on a shared device. Not part of the public `@vitrum/engine` API. */
export async function constructWalkaround(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  aabb: SceneAABB,
  sceneInputIsThree: boolean,
  needsTlas: boolean,
  shared?: SharedDeviceCtx,
): Promise<Engine & Partial<GIStatePersistable>> {
  const adapter = shared?.adapter ?? await navigator.gpu.requestAdapter();
  if (adapter == null) {
    throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported support');
  }

  // Phase-0 productization — probe the adapter's graceful-degradation profile
  // BEFORE requesting the device. This (a) lets us throw the actionable
  // Class-D error instead of failing opaquely inside HybridEngine init when
  // the adapter can't bind the hybrid pipeline, (b) selects the lite tier on
  // a hybrid-incapable-but-lite-capable adapter, and (c) requests the device
  // with the matching `requiredLimits` (previously NO limits were requested,
  // so full hybrid silently relied on adapter defaults — a latent gap).
  const profile = await probeAdapterProfile(adapter);
  opts.onAdapterProfile?.(profile);

  // `recommendedRealtimeTier === 'unavailable'` captures BOTH the below-lite
  // case AND the software-adapter (SwiftShader) case — a software adapter can
  // report passing limits but must never run hybrid (§4.4/§10.4), so gating on
  // the tier verdict (not just the lite-limit boolean) closes that hole.
  if (profile.recommendedRealtimeTier === 'unavailable') {
    // Class D — software rasterizer or below the lite floor. Never init
    // hybrid here (it would fail opaquely mid-init). Point the host at a
    // path-tracer fallback.
    throw new Error(
      `createEngine: this adapter cannot run the walkaround-hybrid realtime ` +
      `engine (recommendedRealtimeTier='${profile.recommendedRealtimeTier}', ` +
      `isSoftwareAdapter=${profile.isSoftwareAdapter}, ` +
      `maxStorageBuffersPerStage=${profile.maxStorageBuffersPerStage}, ` +
      `maxStorageTexturesPerStage=${profile.maxStorageTexturesPerStage}). ` +
      `Pass prefer:'quality' (pt-webgl) or prefer:'quality-webgpu' (pt-webgpu) ` +
      `to use a path-tracer backend on this hardware.`,
    );
  }

  // Full when the adapter meets the full limits; otherwise lite (the profile
  // already guaranteed hybridLiteCapable above).
  const useLite = !profile.hybridCapable;
  // Reuse a shared device when given (progressive facade), else mint our own.
  // A shared device is built with the limit UNION (which includes the FULL
  // hybrid floor), so `profile.hybridCapable` is true and `useLite` is false —
  // the shared path always runs full hybrid, as it must to satisfy the union.
  const device = shared?.device ?? await adapter.requestDevice({
    requiredLimits: useLite ? HYBRID_LITE_LIMITS : HYBRID_WEBGPU_REQUIRED_LIMITS,
  });

  const D = aabb.diagonal;
  const scaleDefaults = deriveScaleDefaults(D);

  // The HybridEngine still expects mutable [number, number, number] for
  // these direction/tint fields (predates the readonly Vec3 alias on
  // @vitrum/core/scene). Spread into a fresh tuple to satisfy the older
  // signature without leaking mutable refs to our shared defaults.
  const primaryLightDir: [number, number, number] = [
    DEFAULT_PRIMARY_LIGHT_DIR[0], DEFAULT_PRIMARY_LIGHT_DIR[1], DEFAULT_PRIMARY_LIGHT_DIR[2],
  ];
  const skyTint: [number, number, number] = [
    DEFAULT_SKY_TINT[0], DEFAULT_SKY_TINT[1], DEFAULT_SKY_TINT[2],
  ];

  // T3.H removal: pass `threeScene` ONLY when the user gave us one — when
  // they passed a vitrum Scene the engine's setScene() path synthesizes the
  // THREE.Scene internally on first BVH build. Removes the round-trip
  // through vitrumSceneToThree() that we previously did at the facade.
  const threeSceneForCtor = sceneInputIsThree
    ? (opts.scene as unknown as Parameters<typeof createWalkaroundEngine_Hybrid>[0]['threeScene'])
    : undefined;

  // Phase-0 — the recommended realtime tier becomes the DEFAULT qualityTier
  // (a preset ceiling). Only applied when it is a concrete preset id — at this
  // point the profile already passed the hybridLiteCapable gate, so it is
  // 'ultra' (full) or 'medium' (lite). An explicit `advanced.qualityTier`
  // overrides it (the advanced spread is last).
  const recommendedTier =
    profile.recommendedRealtimeTier === 'ultra' ||
    profile.recommendedRealtimeTier === 'high' ||
    profile.recommendedRealtimeTier === 'medium' ||
    profile.recommendedRealtimeTier === 'low'
      ? profile.recommendedRealtimeTier
      : undefined;

  const advancedHybrid = opts.advanced as Partial<HybridEngineOptions> | undefined;
  const merged: HybridEngineOptions = {
    device,
    width: Math.max(1, opts.canvas.width),
    height: Math.max(1, opts.canvas.height),
    primaryLightDir,
    primaryLightIntensity: DEFAULT_PRIMARY_LIGHT_INTENSITY,
    skyTint,
    skyIrradiance: DEFAULT_SKY_IRRADIANCE,
    ...(threeSceneForCtor != null ? { threeScene: threeSceneForCtor } : {}),
    cameraMoveResetThresholdSq: scaleDefaults.cameraMoveResetThresholdSq,
    temporalAccumAlpha: scaleDefaults.temporalAccumAlpha,
    debug: opts.debug ?? false,
    // Phase-0 — resource tier + default quality preset. The lite-aware TLAS
    // merge below + the advanced spread both run AFTER these, so a host's
    // explicit `advanced.tier` / `advanced.qualityTier` still win.
    tier: useLite ? 'lite' : 'full',
    ...(recommendedTier !== undefined ? { qualityTier: recommendedTier } : {}),
    ...mergeWalkaroundTlasExtension(
      advancedHybrid,
      // Lite forces merged BVH inside HybridEngine regardless; don't auto-set
      // the TLAS extension when lite, so a needs-TLAS scene still runs merged
      // on a weak adapter (the engine warns about reduced instanced fidelity).
      needsTlas && !useLite,
    ),
    // Theme-H — the audit tuning knobs moved to the nested `tuning` namespace
    // (`Partial<Tunables>`). Placed LAST (after the `advanced` spread) and
    // deep-merged so the host's `advanced.tuning` overrides PER-KEY on top of
    // the scale-derived floors — matching the pre-Theme-H per-key flat override
    // (a wholesale `tuning` replace would drop scale floors the host omitted).
    tuning: {
      emitterDist2Floor: scaleDefaults.emitterDist2Floor,
      triIntersectEpsilon: scaleDefaults.triIntersectEpsilon,
      ...(advancedHybrid?.tuning ?? {}),
    },
  };

  const engine = await createWalkaroundEngine_Hybrid(merged);
  engine.setScene(vitrumScene);

  // A2 — configure the canvas's WebGPU context so the attachVitrum RAF tick
  // can acquire a fresh GPUTextureView per frame and pass it as
  // FrameInput.swapChainView. HybridEngine.renderFrame skips the WebGPU path
  // when input.swapChainView is undefined (HybridEngine.ts:979). Without
  // this configure step, a host using attachVitrum() against a WebGPU
  // backend gets a black canvas. We configure here (not in attachVitrum)
  // because createEngine owns the GPUDevice handle.
  configureWebGpuCanvas(opts.canvas, device);

  return wrapWithIdempotentDispose(engine, () => {
    // Don't destroy a device we don't own — the shared-device owner (the
    // progressive facade) destroys it once after disposing both sub-engines.
    if (shared == null) {
      try { device.destroy(); } catch {}
    }
  });
}

/** @internal — reused by `createProgressiveEngine` to build the converged
 *  engine on a shared device. Not part of the public `@vitrum/engine` API. */
export async function constructPathTracerWebGPU(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  _sceneInputIsThree: boolean,
  shared?: SharedDeviceCtx,
): Promise<Engine> {
  const adapter = shared?.adapter ?? await navigator.gpu.requestAdapter();
  if (adapter == null) {
    throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported support');
  }
  const device = shared?.device ?? await adapter.requestDevice({
    requiredLimits: ptWebgpuRequiredLimitsForAdapter(adapter),
  });

  const advancedWebGPU = opts.advanced as Partial<PTEngineWebGPUOptions> | undefined;
  const merged: PTEngineWebGPUOptions = {
    device,
    // A shared device is built with the limit UNION (≥ the full pt-webgpu
    // floor of 10 buffers / 5 textures), so force the full trace tier here —
    // the union exists precisely so both engines run at full fidelity, and the
    // auto-resolver would also pick 'full' from these limits. Explicit is safer
    // (it surfaces a clear throw if a caller ever passes an under-spec device).
    ...(shared != null ? { traceTier: 'full' as const } : {}),
    ...advancedWebGPU,
  };

  const engine = await createPTEngine_WebGPU(merged);
  engine.setScene(vitrumScene);

  // The converged engine renders offscreen (presentationMode:'offscreen-texture');
  // it does not present to the canvas. When standing alone (createEngine) we keep
  // the historical canvas-configure for attachVitrum swap-chain plumbing. Under a
  // shared device the realtime engine owns the canvas, so skip it here to avoid
  // re-configuring the same context twice.
  if (shared == null) {
    configureWebGpuCanvas(opts.canvas, device);
  }

  return wrapWithIdempotentDispose(engine, () => {
    if (shared == null) {
      try { device.destroy(); } catch {}
    }
  });
}

async function constructPathTracer(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  _sceneInputIsThree: boolean,
): Promise<Engine> {
  const renderer = await createWebGL2RendererForCanvas(opts.canvas);

  const advancedWebGL2 = opts.advanced as WebGL2PathTracerAdvancedOptions | undefined;
  const merged: WebGL2PathTracerOptionsLike = {
    device: renderer,
    ...(advancedWebGL2 ?? {}),
  };

  // Lazy runtime import — keeps the WebGL2 path-tracer stack out of the module
  // graph for hosts that only ever take the WebGPU path (see the import note).
  const { createPTEngine_WebGL2 } = await import('@vitrum/pt-webgl') as unknown as PtWebglModuleLike;
  const engine = await createPTEngine_WebGL2(merged);
  engine.setScene(vitrumScene);

  return wrapWithIdempotentDispose(engine, () => {
    try { renderer.dispose(); } catch {}
    // Some pt-webgl test paths hand back a renderer with forceContextLoss.
    const ext = (renderer as unknown as { forceContextLoss?: () => void }).forceContextLoss;
    if (typeof ext === 'function') {
      try { ext.call(renderer); } catch {}
    }
  });
}

async function createWebGL2RendererForCanvas(
  canvas: HTMLCanvasElement,
): Promise<ThreeWebGLRendererLike> {
  // Late dynamic import keeps the @vitrum/engine bundle leaner for users
  // who only ever take the WebGPU path. The peer-dep guarantees `three`
  // resolves; if it doesn't, we surface a friendly error pointing at the
  // peer-dep block in package.json.
  let three: ThreeRuntimeModule;
  try {
    three = await import('three') as unknown as ThreeRuntimeModule;
  } catch (err) {
    throw new Error(
      'createEngine: failed to load three.js. @vitrum/engine has `three` as a peer dependency; install it in your host. Original error: ' + String(err),
    );
  }
  const renderer = new three.WebGLRenderer({
    canvas,
    antialias: false,
    preserveDrawingBuffer: false,
  });
  return renderer;
}
