// createProgressiveEngine.ts — Track A, increment 3 of the progressive
// walkaround→PT handoff (P8).
//
// Mints ONE shared GPUDevice and stands BOTH a realtime engine
// (@vitrum/walkaround-hybrid — smooth GI while the camera moves) and a converged
// engine (@vitrum/pt-webgpu — ground-truth path tracing) up on it, then wires a
// {@link ProgressiveHandoffCoordinator} that hands the display off realtime→PT
// once the camera settles. The shared device is the load-bearing piece: the
// coordinator's seed handoff binds the walkaround's resolved output TEXTURE into
// the converged engine's `seedAccumulator`, which is only legal when both engines
// allocate against the SAME device (a cross-device texture bind throws). It is
// the missing facade that makes increments 1 (`seedAccumulator`) + 2
// (`seedFromRealtime` wiring) usable end-to-end.
//
// HOST-OWNS-LIFECYCLE: the facade OWNS the device it mints (it is the one piece
// the host cannot reasonably plumb itself — the device must satisfy the LIMIT
// UNION of both backends). `dispose()` tears down both sub-engines, then destroys
// the device exactly once. The host still owns the frame cadence (it calls
// `coordinator.frame(input)` per RAF tick) and the canvas presentation.

import type {
  Scene,
  Engine,
} from '@vitrum/core';
import { auditSceneNeedsTlas } from '@vitrum/core';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  type HybridEngineOptions,
} from '@vitrum/walkaround-hybrid';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  type PTEngineWebGPUOptions,
} from '@vitrum/pt-webgpu';

import {
  constructWalkaround,
  constructPathTracerWebGPU,
  type CreateEngineOptions,
  type SharedDeviceCtx,
} from './createEngine.js';
import { computeSceneAABB } from './sceneAABB.js';
import { configureWebGpuCanvas } from './configureWebGpuCanvas.js';
import {
  ProgressiveHandoffCoordinator,
  type ProgressiveHandoffOptions,
} from './progressiveHandoff.js';
import type { AdapterProfile } from '@vitrum/core';
import {
  isThreeScene,
  sceneFromThreeSceneLike,
  type ThreeSceneLike,
} from './threeSceneBridge.js';

export interface CreateProgressiveEngineOptions {
  /** Canvas the REALTIME engine presents into (the converged engine renders
   *  offscreen). Used to obtain the WebGPU context for swap-chain plumbing. */
  readonly canvas: HTMLCanvasElement;

  /** Scene description. Either a vitrum {@link Scene} or a THREE.Scene; THREE
   *  scenes are auto-converted via @vitrum/three-bindings (once — both engines
   *  receive the SAME converted vitrum scene). */
  readonly scene: Scene | ThreeSceneLike;

  /** Backend-specific overrides for the REALTIME (walkaround-hybrid) engine,
   *  merged on top of the scale-derived defaults exactly as `createEngine`'s
   *  `advanced` is. Most hosts leave empty. */
  readonly realtimeOptions?: Partial<HybridEngineOptions>;

  /** Backend-specific overrides for the CONVERGED (pt-webgpu) engine. The full
   *  trace tier is forced (the shared device satisfies it); a host can still
   *  tune e.g. `maxBounces`, `spectral`, `bdpt`, `causticStrategy`. */
  readonly convergedOptions?: Partial<PTEngineWebGPUOptions>;

  /** Forwarded to {@link ProgressiveHandoffCoordinator}: consecutive still
   *  frames before handing off to the converged engine. Default 6. */
  readonly stillFramesBeforeHandoff?: number;

  /** Virtual-sample weight of the realtime seed prior on each handoff (passed to
   *  `seedAccumulator` via the coordinator). Higher = trust the realtime seed
   *  longer; it decays as W/(W+M) so it never biases the converged mean.
   *  Default 4. */
  readonly seedWeight?: number;

  /** SEED the converged accumulator from the realtime engine on each handoff (the
   *  whole point of the shared device). Default `true` here — the facade exists
   *  to make seeding work; pass `false` only to A/B the pop-hiding win against an
   *  unseeded (black-start) converged run. */
  readonly seedFromRealtime?: boolean;

  /** Forwarded to the coordinator — hide the realtime→1-sample pop by
   *  accumulating the converged image BEHIND the still realtime frame until it is
   *  clean. Default false (the standard interactive-PT switch-and-refine UX). */
  readonly settleBehindRealtime?: boolean;

  /** Forwarded to the coordinator — with {@link settleBehindRealtime}, the sample
   *  count at which the display switches to the converged engine. Default 64. */
  readonly convergedDisplaySamples?: number;

  /** Forwarded to the coordinator — max-abs camera-delta below which a frame
   *  counts as "still". Default 1e-5. */
  readonly cameraEpsilon?: number;

  /** Debug overlay opt-in, forwarded to BOTH sub-engines as `debug: true`. */
  readonly debug?: boolean;

  /** Invoked once with the shared device's graceful-degradation
   *  {@link AdapterProfile} (probed from the union device) before the engines are
   *  built. Lets a host read the tier verdict for a HUD / CI artifact. */
  readonly onAdapterProfile?: (profile: AdapterProfile) => void;
}

export interface ProgressiveEngineHandle {
  /** The coordinator the host drives per RAF tick (`coordinator.frame(input)`).
   *  Also the scene-mutation authority (it forwards setScene/updatePrimitive/… to
   *  both engines and re-arms the handoff). */
  readonly coordinator: ProgressiveHandoffCoordinator;
  /** The realtime (walkaround-hybrid) engine, for direct introspection. Do NOT
   *  call `dispose()` on it directly — use the handle's `dispose()`. */
  readonly realtime: Engine;
  /** The converged (pt-webgpu) engine, for direct introspection. Do NOT call
   *  `dispose()` on it directly — use the handle's `dispose()`. */
  readonly converged: Engine;
  /** Tear down both engines, then destroy the shared device. Idempotent. */
  dispose(): void;
}

/**
 * The device {@link GPUSupportedLimits} a progressive engine must satisfy: the
 * per-key MAXIMUM of the walkaround-hybrid FULL floor and the pt-webgpu FULL
 * floor. The shared device must satisfy BOTH, so the requested device limits are
 * the union (max), not either set alone.
 *
 * Pure + exported so a host can preflight an adapter (`computeProgressiveLimitUnion`
 * + compare to `adapter.limits`) before committing, and so the union is unit-
  * testable without a GPU. The pt-webgpu full trace layout currently dominates
  * the buffer floor, while walkaround-hybrid dominates the texture floor.
 */
export function computeProgressiveLimitUnion(): Record<string, number> {
  // The two FULL-tier requiredLimits sets.
  const hybridFull = HYBRID_WEBGPU_REQUIRED_LIMITS;
  const ptWebgpuFull: Record<string, number> = {
    maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    maxStorageTexturesPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  };
  const union: Record<string, number> = {};
  for (const set of [hybridFull, ptWebgpuFull]) {
    for (const [key, val] of Object.entries(set) as [string, number][]) {
      union[key] = Math.max(union[key] ?? 0, val);
    }
  }
  return union;
}

/**
 * Build a progressive walkaround→PT engine pair on one shared GPUDevice.
 *
 * @throws if WebGPU is unavailable, if the adapter cannot satisfy the limit UNION
 *   of both backends (a clear error naming the gap — Class-A-only; graceful
 *   degradation to a single backend is the host's call via `createEngine`), or if
 *   either built engine fails the progressive capability preflight.
 */
export async function createProgressiveEngine(
  opts: CreateProgressiveEngineOptions,
): Promise<ProgressiveEngineHandle> {
  if (opts.canvas == null) {
    throw new TypeError('createProgressiveEngine: opts.canvas is required');
  }
  if (opts.scene == null) {
    throw new TypeError('createProgressiveEngine: opts.scene is required');
  }
  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error(
      'createProgressiveEngine: WebGPU is unavailable (navigator.gpu is undefined). ' +
        'The progressive engine requires WebGPU for BOTH the realtime and converged ' +
        'backends. Use createEngine({ prefer: "quality" }) for a WebGL2 path tracer.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (adapter == null) {
    throw new Error('createProgressiveEngine: navigator.gpu.requestAdapter() returned null (no WebGPU adapter).');
  }

  // ── Limit-union preflight (Class-A-only) ──────────────────────────────────
  // The shared device must satisfy BOTH backends' FULL floors. Compute the union
  // and check the adapter BEFORE requesting the device, so we can throw a clear,
  // gap-naming error instead of letting requestDevice reject opaquely.
  const union = computeProgressiveLimitUnion();
  const unmet: string[] = [];
  for (const [key, wanted] of Object.entries(union) as [string, number][]) {
    const cap = (adapter.limits as unknown as Record<string, number | undefined>)[key];
    if (typeof cap !== 'number' || cap < wanted) {
      unmet.push(`${key}: need ≥${wanted}, adapter has ${cap ?? 'undefined'}`);
    }
  }
  if (unmet.length > 0) {
    throw new Error(
      'createProgressiveEngine: this adapter cannot satisfy the device-limit UNION of ' +
        'the walkaround-hybrid (realtime) + pt-webgpu (converged) backends, which is ' +
        'required to run both on one shared device. Unmet limits: ' +
        unmet.join('; ') +
        '. The progressive engine is Class-A-only (discrete GPU / native browser WebGPU); ' +
        'graceful degradation to a single backend is the host\'s call — use ' +
        'createEngine({ prefer: "realtime" }) or createEngine({ prefer: "quality-webgpu" }) ' +
        'on this hardware.',
    );
  }

  const device = await adapter.requestDevice({ requiredLimits: union });

  // From here on the device is allocated; EVERY subsequent throw (profile probe,
  // THREE→vitrum scene conversion, either sub-engine build, the capability
  // preflight) must destroy it so it never leaks. The try opens immediately after
  // acquisition for exactly that reason.
  let realtime: (Engine & { dispose(): void }) | null = null;
  let converged: (Engine & { dispose(): void }) | null = null;
  try {
    // Surface the shared device's profile for HUD / CI (probed from the device, so
    // its reported limits reflect the union we requested). This is the SINGLE
    // invocation of the host's callback — it is deliberately NOT forwarded into the
    // realtime sub-build below (which would call it a second time off the adapter),
    // honouring the "invoked once" contract on the option.
    if (opts.onAdapterProfile != null) {
      // Lazy import avoids a cycle (adapterProfile imports backend limit consts;
      // this module is in the same package, so a static import is fine — but the
      // probe is async and only needed when the host asked for it).
      const { probeAdapterProfile } = await import('./adapterProfile.js');
      opts.onAdapterProfile(await probeAdapterProfile(device));
    }

    // Convert THREE → vitrum ONCE; both engines ingest the SAME vitrum scene (the
    // handoff requires both to hold an identical scene — see the coordinator's
    // scene-authority forwarding).
    const sceneInputIsThree = isThreeScene(opts.scene);
    const vitrumScene: Scene = sceneInputIsThree
      ? await sceneFromThreeSceneLike(opts.scene)
      : opts.scene;

    const aabb = computeSceneAABB(vitrumScene);
    const needsTlas = auditSceneNeedsTlas(vitrumScene).needsTlas;

    const shared: SharedDeviceCtx = { adapter, device, ownsDeviceLifecycle: false };

    // The two sub-builds reuse createEngine's OWN scene-handling / options-merging
    // (the scale-derived hybrid defaults, the TLAS extension merge, the pt-webgpu
    // tier resolution) by routing through the shared-device seam — no replication.
    // Each gets its own synthesized CreateEngineOptions carrying its `advanced`.
    const realtimeBuildOpts: CreateEngineOptions = {
      canvas: opts.canvas,
      scene: opts.scene,
      ...(opts.realtimeOptions != null ? { advanced: opts.realtimeOptions } : {}),
      ...(opts.debug != null ? { debug: opts.debug } : {}),
      // onAdapterProfile is intentionally NOT forwarded — the facade already
      // invoked it once above (off the shared device). Forwarding it here would
      // fire the host callback a second time (off the adapter), breaking the
      // "invoked once" contract.
    };
    realtime = await constructWalkaround(
      realtimeBuildOpts,
      vitrumScene,
      aabb,
      sceneInputIsThree,
      needsTlas,
      shared,
    );

    const convergedBuildOpts: CreateEngineOptions = {
      canvas: opts.canvas,
      scene: opts.scene,
      ...(opts.convergedOptions != null ? { advanced: opts.convergedOptions } : {}),
      ...(opts.debug != null ? { debug: opts.debug } : {}),
    };
    converged = await constructPathTracerWebGPU(
      convergedBuildOpts,
      vitrumScene,
      sceneInputIsThree,
      shared,
    );

    // ── Capability preflight ────────────────────────────────────────────────
    // The seed handoff requires the realtime engine to expose a seed SOURCE and
    // the converged engine a seed SINK. Assert both — a clear throw here beats a
    // silent no-op handoff at runtime (the coordinator would degrade to a black
    // reset). pt-webgpu advertises supportsAccumulatorSeed unconditionally (the
    // accum buffers exist on both tiers); we still assert it so the contract is
    // enforced at the seam, not assumed.
    if (realtime.capabilities.supportsProgressiveSeedSource !== true) {
      throw new Error(
        'createProgressiveEngine: the realtime engine does not advertise ' +
          'supportsProgressiveSeedSource — it cannot provide a seed texture for the ' +
          'converged engine\'s accumulator. The progressive seed handoff requires it.',
      );
    }
    if (converged.capabilities.supportsAccumulatorSeed !== true) {
      throw new Error(
        'createProgressiveEngine: the converged engine does not advertise ' +
          'supportsAccumulatorSeed — its accumulator cannot be seeded from the realtime ' +
          'engine. The progressive seed handoff requires it.',
      );
    }

    // Belt-and-braces: configure the canvas context against the shared device for
    // the REALTIME engine's swap-chain presentation (constructWalkaround already
    // did this, but it is idempotent + best-effort, so a host that swapped the
    // canvas is still covered).
    configureWebGpuCanvas(opts.canvas, device);

    const coordinatorOpts: ProgressiveHandoffOptions = {
      realtime,
      converged,
      seedFromRealtime: opts.seedFromRealtime ?? true,
      ...(opts.seedWeight != null ? { seedWeight: opts.seedWeight } : {}),
      ...(opts.stillFramesBeforeHandoff != null
        ? { stillFramesBeforeHandoff: opts.stillFramesBeforeHandoff }
        : {}),
      ...(opts.settleBehindRealtime != null
        ? { settleBehindRealtime: opts.settleBehindRealtime }
        : {}),
      ...(opts.convergedDisplaySamples != null
        ? { convergedDisplaySamples: opts.convergedDisplaySamples }
        : {}),
      ...(opts.cameraEpsilon != null ? { cameraEpsilon: opts.cameraEpsilon } : {}),
    };
    const coordinator = new ProgressiveHandoffCoordinator(coordinatorOpts);

    const builtRealtime = realtime;
    const builtConverged = converged;
    let disposed = false;
    return {
      coordinator,
      realtime: builtRealtime,
      converged: builtConverged,
      dispose(): void {
        if (disposed) return;
        disposed = true;
        // Sub-engine disposes are no-ops on the device (shared-device path); the
        // facade destroys it once below, after both are torn down.
        try { builtRealtime.dispose(); } catch {}
        try { builtConverged.dispose(); } catch {}
        try { device.destroy(); } catch {}
      },
    };
  } catch (err) {
    // Build failed after acquiring the device (or after one sub-engine built):
    // tear down whatever exists, then the device, so we never leak it.
    try { realtime?.dispose(); } catch {}
    try { converged?.dispose(); } catch {}
    try { device.destroy(); } catch {}
    throw err;
  }
}
