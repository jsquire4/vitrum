// @internal — walkaround-hybrid backend constructor.
//
// Extracted from createEngine.ts (C1 / I1.1) to keep the public facade thin.
// Re-exported from createEngine.ts for createProgressiveEngine back-compat;
// this module is NOT part of the public @vitrum/engine API surface.

import type { Scene, Engine } from '@vitrum/core';
import {
  createWalkaroundEngine_Hybrid,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
  type HybridEngineOptions,
} from '@vitrum/walkaround-hybrid';
import { probeAdapterProfile } from '../adapterProfile.js';
import { configureWebGpuCanvas } from '../configureWebGpuCanvas.js';
import {
  DEFAULT_PRIMARY_LIGHT_DIR,
  DEFAULT_PRIMARY_LIGHT_INTENSITY,
  DEFAULT_SKY_IRRADIANCE,
  DEFAULT_SKY_TINT,
  deriveScaleDefaults,
} from '../createEngineScale.js';
import {
  mergeWalkaroundTlasExtension,
  resolveAdvancedForBackend,
  stripOwnershipCriticalKeys,
  reportCreateEngineError,
  attachBackendId,
  wrapWithIdempotentDispose,
  type CreateEngineOptions,
  type SharedDeviceCtx,
  type EngineWithBackendId,
} from '../createEngineInternals.js';
import type { GIStatePersistable } from '../idempotentDispose.js';
import type { SceneAABB } from '../sceneAABB.js';

function destroyOwnedWebGpuDevice(shared: SharedDeviceCtx | undefined, device: GPUDevice): void {
  if (shared == null) {
    try { device.destroy(); } catch { /* best-effort destroy — ignore */ }
  }
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
    const advancedHybridRaw = resolveAdvancedForBackend(
      opts,
      'walkaround-hybrid',
    ) as Partial<HybridEngineOptions> | undefined;
    const advancedHybrid = stripOwnershipCriticalKeys(
      advancedHybridRaw,
      'walkaround-hybrid',
      opts.onWarning,
    ) as Partial<HybridEngineOptions>;
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
      ...(opts.onWarning != null ? { onWarning: opts.onWarning } : {}),
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

/** Thin dispatch-table wrapper: builds, tags with backendId, returns EngineWithBackendId.
 *  @internal */
export async function constructWalkaroundForDispatch(
  opts: CreateEngineOptions,
  vitrumScene: Scene,
  aabb: SceneAABB,
  needsTlas: boolean,
  shared?: SharedDeviceCtx,
): Promise<EngineWithBackendId> {
  const engine = await constructWalkaround(opts, vitrumScene, aabb, needsTlas, shared);
  return attachBackendId(engine, 'walkaround-hybrid');
}
