// Top-level drop-in factory.
//
// Given a canvas + vitrum Scene, createEngine()
//   - probes WebGPU vs WebGL2,
//   - picks the walkaround-hybrid backend (real-time GI) or pt-webgl2
//     backend (converged path tracer),
//   - derives scale-sensitive defaults (Möller-Trumbore epsilon, camera-
//     move-reset threshold, emitter dist² floor, GTAO sigma) from the
//     scene's AABB diagonal D,
//   - constructs and owns the backend's device handle (GPUDevice for
//     walkaround, WebGL2RenderingContext for pt-webgl2) so the host doesn't
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
import {
  createPTEngine_WebGPU,
  ptWebgpuRequiredLimitsForAdapter,
  type PTEngineWebGPUOptions,
} from '@vitrum/pt-webgpu';
import type { PTEngineWebGL2Options } from '@vitrum/pt-webgl2';

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

export type { EnginePreference, ScaleDefaults };
export { pickBackend, deriveScaleDefaults };

type WebGL2PathTracerAdvancedOptions = Partial<Omit<PTEngineWebGL2Options, 'device'>>;

interface PtWebgl2ModuleLike {
  readonly createPTEngine_WebGL2: (opts: PTEngineWebGL2Options) => Promise<Engine>;
}

export type CreateEngineBackendId = 'walkaround-hybrid' | 'pt-webgpu' | 'pt-webgl2';

/** Engine returned by {@link createEngine} with its chosen backendId attached.
 *  Hosts that need to know which backend was selected (e.g. to type-narrow
 *  `opts.advanced` for backend-specific API calls) read this field.
 *
 *  The `advanced` cast to a backend-specific options type is safe ONLY when
 *  `engine.backendId` matches the target backend — gate such casts on this field. */
export interface EngineWithBackendId extends Engine, Partial<GIStatePersistable> {
  readonly backendId: CreateEngineBackendId;
}

export type CreateEngineErrorPhase =
  | 'create:walkaround-hybrid'
  | 'create:pt-webgpu'
  | 'create:pt-webgl2'
  | 'canvas-configure'
  | 'attach:resize'
  | 'attach:swapchain'
  | 'attach:renderFrame';

export interface CreateEngineErrorEvent {
  readonly phase: CreateEngineErrorPhase;
  readonly backend?: CreateEngineBackendId;
  readonly recoverable: boolean;
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

/**
 * Strip ownership-critical keys (`device`, `canvas`, `context`) from an
 * `advanced` option bag before spreading it over createEngine's own
 * factory-derived values.  If `advanced` supplied any of these keys they
 * would silently override the device that createEngine minted and owns — the
 * dispose path would destroy a device the HOST owns (or a completely alien
 * object), so the engine-minted one would leak.  We strip and warn so the
 * bug surfaces at construction instead of at GC/teardown time.
 *
 * @internal — used by the two device-owning constructors (walkaround-hybrid +
 * pt-webgpu).  The WebGL2 path already uses `Omit<…, 'device'>` in its type
 * so it is safe without this helper.
 */
const OWNERSHIP_CRITICAL_KEYS = ['device', 'canvas', 'context'] as const;
type OwnershipCriticalKey = (typeof OWNERSHIP_CRITICAL_KEYS)[number];

export function stripOwnershipCriticalKeys<T extends Record<string, unknown>>(
  advanced: T | undefined,
  backend: CreateEngineBackendId,
): Omit<T, OwnershipCriticalKey> {
  if (advanced == null) return {} as Omit<T, OwnershipCriticalKey>;
  const stripped = { ...advanced } as Record<string, unknown>;
  const overridden: string[] = [];
  for (const key of OWNERSHIP_CRITICAL_KEYS) {
    if (key in stripped) {
      overridden.push(key);
      delete stripped[key];
    }
  }
  if (overridden.length > 0) {
    console.warn(
      `[vitrum/createEngine] advanced.${overridden.join('/')} was supplied but ` +
      `createEngine owns the ${backend} device lifecycle — the supplied ` +
      `${overridden.join('/')} key(s) have been ignored to prevent a double-dispose. ` +
      `To bring your own device, use the backend factory directly.`,
    );
  }
  return stripped as Omit<T, OwnershipCriticalKey>;
}

export interface CreateEngineOptions {
  /** Canvas the engine renders into. Used to obtain the GPU context. */
  readonly canvas: HTMLCanvasElement;

  /** Scene description in the host-agnostic @vitrum/core contract. */
  readonly scene: Scene;

  /** Quality vs speed hint:
   *    'realtime' — prefer walkaround-hybrid (WebGPU; ~60fps target).
   *    'quality'  — prefer pt-webgl2 (WebGL2 path tracer; converged).
   *    'quality-webgpu' — prefer pt-webgpu when WebGPU is available, else pt-webgl2.
   *    'auto'     — pick walkaround-hybrid if WebGPU + tris < 500k,
   *                 else a path-tracer backend. Default. */
  readonly prefer?: EnginePreference;

  /** Backend-specific overrides. Merged on top of the createEngine()-
   *  derived defaults; user-supplied keys win. Most users leave empty. */
  readonly advanced?: Partial<HybridEngineOptions> | WebGL2PathTracerAdvancedOptions | Partial<PTEngineWebGPUOptions>;

  /** Debug overlay opt-in. Forwarded to backend as `debug: true`. */
  readonly debug?: boolean;

  /** Phase-0 productization — callback invoked once with the graceful-
   *  degradation {@link AdapterProfile} when the walkaround-hybrid backend is
   *  selected (before device acquisition). Lets hosts read the JSON for a HUD
   *  / CI artifact (§4.1 / §10.3). Not called for the pt-webgl2 / pt-webgpu
   *  backends (they have their own tier selection). */
  readonly onAdapterProfile?: (profile: AdapterProfile) => void;

  /** Host-visible error report for recoverable fallback and canvas plumbing
   *  failures. Recoverable events are still handled internally; unrecoverable
   *  events are reported immediately before the original error is re-thrown. */
  readonly onError?: (error: unknown, event: CreateEngineErrorEvent) => void;
}

export async function createEngine(opts: CreateEngineOptions): Promise<EngineWithBackendId> {
  if (opts.canvas == null) {
    throw new TypeError('createEngine: opts.canvas is required');
  }
  if (opts.scene == null) {
    throw new TypeError('createEngine: opts.scene is required');
  }

  const vitrumScene: Scene = opts.scene;

  const aabb = computeSceneAABB(vitrumScene);
  const tlasAudit = auditSceneNeedsTlas(vitrumScene);
  const gpu = await detectGpu({ publishToWindow: false });
  const backend = pickBackend(
    opts.prefer ?? 'auto',
    gpu.isWebGPU,
    aabb.triangleCount,
    tlasAudit.needsTlas,
  );
  // When the audit recommends a TLAS-capable backend but we resolved to pt-webgl2
  // (the only merged-BVH backend), surface the recommendation + detail so the host
  // can switch to walkaround-hybrid or pt-webgpu for correct instancing behaviour.
  if (tlasAudit.recommendation === 'prefer-tlas-backend' && backend === 'pt-webgl2') {
    console.warn(`[vitrum/createEngine] ${tlasAudit.detail}`);
  }

  if (backend === 'walkaround-hybrid') {
    try {
      const engine = await constructWalkaround(opts, vitrumScene, aabb, tlasAudit.needsTlas);
      return attachBackendId(engine, 'walkaround-hybrid');
    } catch (err) {
      reportCreateEngineError(opts, err, {
        phase: 'create:walkaround-hybrid',
        backend: 'walkaround-hybrid',
        recoverable: true,
      });
      const fallbackBackend = gpu.isWebGPU ? 'pt-webgpu' : 'pt-webgl2';
      console.warn(
        `[vitrum/createEngine] walkaround-hybrid unavailable; falling back to ${fallbackBackend}.`,
        err,
      );
      warnCrossBackendAdvanced(opts.advanced, 'walkaround-hybrid', fallbackBackend);
      return await constructPathTracerFallback(opts, vitrumScene, gpu.isWebGPU);
    }
  }
  if (backend === 'pt-webgpu') {
    try {
      const engine = await constructPathTracerWebGPU(opts, vitrumScene);
      return attachBackendId(engine, 'pt-webgpu');
    } catch (err) {
      reportCreateEngineError(opts, err, {
        phase: 'create:pt-webgpu',
        backend: 'pt-webgpu',
        recoverable: true,
      });
      console.warn('[vitrum/createEngine] pt-webgpu unavailable; falling back to pt-webgl2.', err);
      warnCrossBackendAdvanced(opts.advanced, 'pt-webgpu', 'pt-webgl2');
      return await constructPathTracerWebGLFallback(opts, vitrumScene);
    }
  }
  return await constructPathTracerWebGLFallback(opts, vitrumScene);
}

/** Attach a `backendId` property to an engine returned by a constructor.
 *  The property is non-enumerable-but-readable so it doesn't interfere with
 *  spread/clone patterns the host might use on the Engine object.
 *  @internal */
function attachBackendId(
  engine: Engine & Partial<GIStatePersistable>,
  backendId: CreateEngineBackendId,
): EngineWithBackendId {
  return Object.defineProperty(engine, 'backendId', {
    value: backendId,
    writable: false,
    enumerable: false,
    configurable: false,
  }) as EngineWithBackendId;
}

function reportCreateEngineError(
  opts: CreateEngineOptions,
  error: unknown,
  event: CreateEngineErrorEvent,
): void {
  try {
    opts.onError?.(error, event);
  } catch { /* host error callback must not propagate — ignore */ }
}

/**
 * Warn when `advanced` is non-empty and the resolved backend differs from the
 * backend the host most likely targeted (because the preferred backend fell
 * back). The `advanced` keys are applied to a different backend than the one
 * they were written for — most keys will be silently ignored, but a few may
 * accidentally match, producing subtle misbehaviour.
 *
 * Note: there is no explicit `targetBackend` field on `advanced`; this heuristic
 * fires on any non-empty `advanced` when a fallback occurs, which is the honest
 * minimum without per-key introspection.
 *
 * @internal Exported for unit-test access. Not part of the public API.
 */
export function warnCrossBackendAdvanced(
  advanced: CreateEngineOptions['advanced'],
  preferredBackend: CreateEngineBackendId,
  resolvedBackend: CreateEngineBackendId,
): void {
  if (advanced == null) return;
  const keys = Object.keys(advanced as Record<string, unknown>).filter(
    (k) => (advanced as Record<string, unknown>)[k] !== undefined,
  );
  if (keys.length === 0) return;
  console.warn(
    `[vitrum/createEngine] advanced options (keys: ${keys.join(', ')}) were supplied ` +
    `but the preferred backend '${preferredBackend}' was unavailable — they are now ` +
    `being applied to the fallback backend '${resolvedBackend}'. Keys authored for ` +
    `'${preferredBackend}' may be silently ignored or misinterpreted by '${resolvedBackend}'. ` +
    `Pass prefer:'${resolvedBackend}' explicitly to suppress this warning.`,
  );
}

function destroyOwnedWebGpuDevice(shared: SharedDeviceCtx | undefined, device: GPUDevice): void {
  if (shared == null) {
    try { device.destroy(); } catch { /* best-effort destroy — ignore */ }
  }
}

function disposeOwnedWebGL2Context(gl: WebGL2RenderingContext): void {
  try { gl.getExtension('WEBGL_lose_context')?.loseContext(); } catch { /* best-effort context loss — ignore */ }
}

async function constructPathTracerFallback(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  tryWebGpuFirst: boolean,
): Promise<EngineWithBackendId> {
  if (tryWebGpuFirst) {
    try {
      const engine = await constructPathTracerWebGPU(opts, vitrumScene);
      return attachBackendId(engine, 'pt-webgpu');
    } catch (err) {
      reportCreateEngineError(opts, err, {
        phase: 'create:pt-webgpu',
        backend: 'pt-webgpu',
        recoverable: true,
      });
      console.warn('[vitrum/createEngine] pt-webgpu fallback unavailable; falling back to pt-webgl2.', err);
    }
  }
  return await constructPathTracerWebGLFallback(opts, vitrumScene);
}

async function constructPathTracerWebGLFallback(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
): Promise<EngineWithBackendId> {
  try {
    const engine = await constructPathTracer(opts, vitrumScene);
    return attachBackendId(engine, 'pt-webgl2');
  } catch (err) {
    reportCreateEngineError(opts, err, {
      phase: 'create:pt-webgl2',
      backend: 'pt-webgl2',
      recoverable: false,
    });
    throw err;
  }
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
      `Pass prefer:'quality' (pt-webgl2) or prefer:'quality-webgpu' (pt-webgpu) ` +
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

  let engine: (Engine & Partial<GIStatePersistable>) | null = null;
  try {
    const advancedHybridRaw = opts.advanced as Partial<HybridEngineOptions> | undefined;
    const advancedHybrid = stripOwnershipCriticalKeys(advancedHybridRaw, 'walkaround-hybrid') as Partial<HybridEngineOptions>;
    const merged: HybridEngineOptions = {
      device,
      width: Math.max(1, opts.canvas.width),
      height: Math.max(1, opts.canvas.height),
      primaryLightDir,
      primaryLightIntensity: DEFAULT_PRIMARY_LIGHT_INTENSITY,
      skyTint,
      skyIrradiance: DEFAULT_SKY_IRRADIANCE,
      cameraMoveResetThresholdSq: scaleDefaults.cameraMoveResetThresholdSq,
      temporalAccumAlpha: scaleDefaults.temporalAccumAlpha,
      debug: opts.debug ?? false,
      // Phase-0 — resource tier + default quality preset. The lite-aware TLAS
      // merge below + the advanced spread both run AFTER these, so a host's
      // explicit `advanced.tier` / `advanced.qualityTier` still win on normal
      // createEngine builds. Shared progressive builds re-force full below.
      tier: useLite ? 'lite' : 'full',
      ...(recommendedTier !== undefined ? { qualityTier: recommendedTier } : {}),
      ...mergeWalkaroundTlasExtension(
        advancedHybrid,
        // Lite forces merged BVH inside HybridEngine regardless; don't auto-set
        // the TLAS extension when lite, so a needs-TLAS scene still runs merged
        // on a weak adapter (the engine warns about reduced instanced fidelity).
        needsTlas && !useLite,
      ),
      ...(shared != null ? { tier: 'full' as const } : {}),
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

    engine = await createWalkaroundEngine_Hybrid(merged);
    engine.setScene(vitrumScene);

    // A2 — configure the canvas's WebGPU context so the attachVitrum RAF tick
    // can acquire a fresh GPUTextureView per frame and pass it as
    // FrameInput.swapChainView. HybridEngine.renderFrame skips the WebGPU path
    // when input.swapChainView is undefined (HybridEngine.ts:979). Without
    // this configure step, a host using attachVitrum() against a WebGPU
    // backend gets a black canvas. We configure here (not in attachVitrum)
    // because createEngine owns the GPUDevice handle.
    configureWebGpuCanvas(opts.canvas, device, (err) => {
      reportCreateEngineError(opts, err, {
        phase: 'canvas-configure',
        backend: 'walkaround-hybrid',
        recoverable: true,
      });
    });

    const built = engine;
    engine = null;
    return wrapWithIdempotentDispose(built, () => {
      // Don't destroy a device we don't own — the shared-device owner (the
      // progressive facade) destroys it once after disposing both sub-engines.
      destroyOwnedWebGpuDevice(shared, device);
    });
  } catch (err) {
    try { engine?.dispose(); } catch { /* best-effort cleanup before re-throw — ignore */ }
    destroyOwnedWebGpuDevice(shared, device);
    throw err;
  }
}

/** @internal — reused by `createProgressiveEngine` to build the converged
 *  engine on a shared device. Not part of the public `@vitrum/engine` API. */
export async function constructPathTracerWebGPU(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  shared?: SharedDeviceCtx,
): Promise<Engine> {
  const adapter = shared?.adapter ?? await navigator.gpu.requestAdapter();
  if (adapter == null) {
    throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported support');
  }
  const advancedWebGPURaw = opts.advanced as Partial<PTEngineWebGPUOptions> | undefined;
  const advancedWebGPU = stripOwnershipCriticalKeys(advancedWebGPURaw, 'pt-webgpu') as Partial<PTEngineWebGPUOptions>;
  const device = shared?.device ?? await adapter.requestDevice({
    requiredLimits: ptWebgpuRequiredLimitsForAdapter(adapter, {
      restirPtReuse: advancedWebGPURaw?.restirPtReuse === true,
    }),
  });

  let engine: Engine | null = null;
  try {
    const merged: PTEngineWebGPUOptions = {
      device,
      ...advancedWebGPU,
      // A shared device is built with the limit UNION (≥ the full pt-webgpu
      // per-stage buffer/texture floor), so force the full trace tier here —
      // the union exists precisely so both engines run at full fidelity, and the
      // auto-resolver would also pick 'full' from these limits. Explicit is safer
      // (it surfaces a clear throw if a caller ever passes an under-spec device).
      ...(shared != null ? { traceTier: 'full' as const } : {}),
    };

    engine = await createPTEngine_WebGPU(merged);
    engine.setScene(vitrumScene);

    // The converged engine renders offscreen (presentationMode:'offscreen-texture');
    // it does not present to the canvas. When standing alone (createEngine) we keep
    // the historical canvas-configure for attachVitrum swap-chain plumbing. Under a
    // shared device the realtime engine owns the canvas, so skip it here to avoid
    // re-configuring the same context twice.
    if (shared == null) {
      configureWebGpuCanvas(opts.canvas, device, (err) => {
        reportCreateEngineError(opts, err, {
          phase: 'canvas-configure',
          backend: 'pt-webgpu',
          recoverable: true,
        });
      });
    }

    const built = engine;
    engine = null;
    return wrapWithIdempotentDispose(built, () => {
      destroyOwnedWebGpuDevice(shared, device);
    });
  } catch (err) {
    try { engine?.dispose(); } catch { /* best-effort cleanup before re-throw — ignore */ }
    destroyOwnedWebGpuDevice(shared, device);
    throw err;
  }
}

async function constructPathTracer(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
): Promise<Engine> {
  const gl = createWebGL2ContextForCanvas(opts.canvas);

  let engine: Engine | null = null;
  try {
    const advancedWebGL2 = opts.advanced as WebGL2PathTracerAdvancedOptions | undefined;
    const merged: PTEngineWebGL2Options = {
      device: gl,
      ...(advancedWebGL2 ?? {}),
    };

    // Lazy runtime import keeps the WebGL2 path-tracer stack out of the module
    // graph for hosts that only ever take the WebGPU path.
    const { createPTEngine_WebGL2 } = await import('@vitrum/pt-webgl2') as unknown as PtWebgl2ModuleLike;
    engine = await createPTEngine_WebGL2(merged);
    engine.setScene(vitrumScene);

    const built = engine;
    engine = null;
    return wrapWithIdempotentDispose(built, () => {
      disposeOwnedWebGL2Context(gl);
    });
  } catch (err) {
    try { engine?.dispose(); } catch { /* best-effort cleanup before re-throw — ignore */ }
    disposeOwnedWebGL2Context(gl);
    throw err;
  }
}

function createWebGL2ContextForCanvas(
  canvas: HTMLCanvasElement,
): WebGL2RenderingContext {
  const gl = canvas.getContext('webgl2', {
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (gl == null) {
    throw new Error('createEngine: WebGL2 is unavailable; canvas.getContext("webgl2") returned null.');
  }
  return gl;
}
