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
  BackendSupportDetails,
  BackendSupportMode,
  EngineCapabilities,
  EngineState,
  EngineWarning,
  Scene,
  Engine,
  FrameInput,
} from '@vitrum/core';
import { auditSceneNeedsTlas, validateScene } from '@vitrum/core';
import type { RuntimeEngineWithBackendId as EngineWithBackendId } from './createEngineInternals.js';
import type { GIStatePersistable } from './idempotentDispose.js';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  nrcWebGpuRequiredLimitsForConfig,
  resolveHybridNrcConfig,
  validateHybridEngineAdvancedOptions,
  type NrcConfig,
  type HybridEngineOptions,
} from '@vitrum/walkaround-hybrid';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  validatePtWebgpuAdvancedOptions,
  type PTEngineWebGPUOptions,
} from '@vitrum/pt-webgpu';

import {
  constructWalkaround,
  constructPathTracerWebGPU,
  type CreateEngineOptions,
  type CreateEngineErrorEvent,
  type SharedDeviceCtx,
} from './createEngine.js';
import { computeSceneAABB } from './sceneAABB.js';
import { configureWebGpuCanvas } from './configureWebGpuCanvas.js';
import {
  ProgressiveHandoffCoordinator,
  assertProgressiveHandoffConfiguration,
  type ProgressiveHandoffController,
  type ProgressiveHandoffControllerDelta,
  type ProgressiveHandoffOptions,
} from './progressiveHandoff.js';
import type { AdapterProfile } from '@vitrum/core';
import { requiredWalkaroundNeuralDeviceFeatures } from './neuralFeatureNegotiation.js';
export interface CreateProgressiveEngineOptions {
  /** Canvas the REALTIME engine presents into (the converged engine renders
   *  offscreen). Used to obtain the WebGPU context for swap-chain plumbing. */
  readonly canvas: HTMLCanvasElement;

  /** Scene description in the host-agnostic @vitrum/core contract. */
  readonly scene: Scene;

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

  /** Forwarded to the coordinator — optional scene animation controller (for
   *  example a glTF controller). The coordinator advances it once per frame and
   *  routes patches to both sub-engines. */
  readonly controller?: ProgressiveHandoffController;

  /** Forwarded to the coordinator — seconds passed to `controller.advance()` per
   *  frame. Default 1/60. */
  readonly controllerDeltaSeconds?: ProgressiveHandoffControllerDelta;

  /** Forwarded to the coordinator — `loop` option for controller animation.
   *  Default true. */
  readonly controllerLoop?: boolean;

  /** Debug overlay opt-in, forwarded to BOTH sub-engines as `debug: true`. */
  readonly debug?: boolean;

  /** Invoked once with the shared device's graceful-degradation
   *  {@link AdapterProfile} (probed from the union device) before the engines are
   *  built. Lets a host read the tier verdict for a HUD / CI artifact. */
  readonly onAdapterProfile?: (profile: AdapterProfile) => void;

  /** Construction-time warnings from either sub-engine. Runtime warnings are
   * available through the adapted engine's `onWarning` fan-out. Host callback
   * exceptions are contained by the backend construction facade. */
  readonly onWarning?: (warning: EngineWarning) => void;

  /** Host-visible construction/plumbing error callback.
   *
   * Mirrors the `onError` parameter on {@link CreateEngineOptions}: the second
   * argument carries the phase/backend/recoverability record for sub-engine
   * construction errors and the progressive facade's final best-effort canvas
   * configure. One-argument callbacks remain source-compatible in TypeScript. */
  readonly onError?: (error: unknown, event: CreateEngineErrorEvent) => void;
}

const CREATE_PROGRESSIVE_ENGINE_OPTION_KEYS = {
  canvas: true,
  scene: true,
  realtimeOptions: true,
  convergedOptions: true,
  stillFramesBeforeHandoff: true,
  seedWeight: true,
  seedFromRealtime: true,
  settleBehindRealtime: true,
  convergedDisplaySamples: true,
  cameraEpsilon: true,
  controller: true,
  controllerDeltaSeconds: true,
  controllerLoop: true,
  debug: true,
  onAdapterProfile: true,
  onWarning: true,
  onError: true,
} as const satisfies Readonly<Record<keyof CreateProgressiveEngineOptions, true>>;

function assertProgressivePlainDataObject(
  value: unknown,
  label: string,
  allowedKeys: ReadonlySet<string>,
): asserts value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be a plain object`);
  }
  const prototype: unknown = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(`${label} must have Object.prototype or null prototype`);
  }
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string') {
      throw new TypeError(`${label} contains unsupported symbol key ${String(key)}`);
    }
    if (!allowedKeys.has(key)) {
      throw new TypeError(`${label} contains unknown key ${JSON.stringify(key)}`);
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor)) {
      throw new TypeError(`${label}.${key} must be an own data property`);
    }
    if (!descriptor.enumerable) {
      throw new TypeError(`${label}.${key} must be enumerable`);
    }
  }
}

/**
 * Validate the progressive facade before scene traversal, adapter acquisition,
 * capability reads, or canvas-context access. Backend bags are validated by the
 * same pure validators used by their concrete factories.
 */
export function validateCreateProgressiveEngineOptions(
  value: unknown,
): asserts value is CreateProgressiveEngineOptions {
  assertProgressivePlainDataObject(
    value,
    'createProgressiveEngine options',
    new Set(Object.keys(CREATE_PROGRESSIVE_ENGINE_OPTION_KEYS)),
  );

  if (value.canvas == null || typeof value.canvas !== 'object') {
    throw new TypeError(
      'createProgressiveEngine: opts.canvas is required and must be an object',
    );
  }
  if (typeof (value.canvas as { readonly getContext?: unknown }).getContext !== 'function') {
    throw new TypeError('createProgressiveEngine: opts.canvas must expose getContext()');
  }
  for (const dimension of ['width', 'height'] as const) {
    const component = (value.canvas as {
      readonly width?: unknown;
      readonly height?: unknown;
    })[dimension];
    if (typeof component !== 'number' || !Number.isSafeInteger(component) || component < 0) {
      throw new RangeError(
        `createProgressiveEngine: opts.canvas.${dimension} must be a non-negative safe integer ` +
          `(got ${String(component)})`,
      );
    }
  }
  if (value.scene == null || typeof value.scene !== 'object') {
    throw new TypeError(
      'createProgressiveEngine: opts.scene is required and must be an object',
    );
  }

  if (value.realtimeOptions !== undefined) {
    validateHybridEngineAdvancedOptions(value.realtimeOptions);
  }
  if (value.convergedOptions !== undefined) {
    validatePtWebgpuAdvancedOptions(
      value.convergedOptions as Partial<Omit<PTEngineWebGPUOptions, 'device'>>,
    );
  }

  for (const option of [
    'seedFromRealtime',
    'settleBehindRealtime',
    'controllerLoop',
    'debug',
  ] as const) {
    if (value[option] !== undefined && typeof value[option] !== 'boolean') {
      throw new TypeError(
        `createProgressiveEngine: opts.${option} must be a boolean when supplied`,
      );
    }
  }
  for (const callback of ['onAdapterProfile', 'onWarning', 'onError'] as const) {
    if (value[callback] !== undefined && typeof value[callback] !== 'function') {
      throw new TypeError(
        `createProgressiveEngine: opts.${callback} must be a function when supplied`,
      );
    }
  }
  if (
    value.controller !== undefined &&
    (
      value.controller === null ||
      (typeof value.controller !== 'object' && typeof value.controller !== 'function') ||
      typeof (value.controller as { readonly advance?: unknown }).advance !== 'function'
    )
  ) {
    throw new TypeError(
      'createProgressiveEngine: opts.controller must expose advance() when supplied',
    );
  }
  if (
    value.controllerDeltaSeconds !== undefined &&
    typeof value.controllerDeltaSeconds !== 'number' &&
    typeof value.controllerDeltaSeconds !== 'function'
  ) {
    throw new TypeError(
      'createProgressiveEngine: opts.controllerDeltaSeconds must be a number or function',
    );
  }

  assertProgressiveHandoffConfiguration(
    value,
  );
}

export interface ProgressiveEngineHandle {
  /** Resolved converged-backend profile forwarded without inference. */
  readonly backendProfileId?: 'pt-webgpu' | 'pt-webgpu-lite';
  /** Adapter/loader profile identity forwarded without inference when present. */
  readonly profileId?: 'pt-webgpu' | 'pt-webgpu-lite';
  /** The coordinator the host drives per RAF tick (`coordinator.frame(input)`).
   *  Also the scene-mutation authority (it forwards setScene/updatePrimitive/… to
   *  both engines and re-arms the handoff). */
  readonly coordinator: ProgressiveHandoffCoordinator;
  /** The realtime (walkaround-hybrid) engine, for direct introspection. Do NOT
   *  call `dispose()` on it directly — use the handle's `dispose()`. */
  readonly realtime: Engine & Partial<GIStatePersistable>;
  /** The converged (pt-webgpu) engine, for direct introspection. Do NOT call
   *  `dispose()` on it directly — use the handle's `dispose()`. */
  readonly converged: Engine;
  /** Tear down both engines, then destroy the shared device. Idempotent. */
  dispose(): void;
}


const SUPPORT_MODE_RANK: Readonly<Record<BackendSupportMode, number>> = {
  native: 0,
  'fallback-rebuild': 1,
  'fallback-generated-mesh': 2,
  approximate: 3,
  unsupported: 4,
};

/** Runtime-immutable set view for a capability object retained by the facade.
 *  The closure owns the mutable backing Set; hosts receive no mutation method. */
function runtimeReadonlySet<T>(items: Iterable<T>): ReadonlySet<T> {
  const values = new Set(items);
  const view: ReadonlySet<T> = {
    get size() { return values.size; },
    has: (value) => values.has(value),
    entries: () => values.entries(),
    keys: () => values.keys(),
    values: () => values.values(),
    forEach: (callback, thisArg) => {
      for (const value of values) callback.call(thisArg, value, value, view);
    },
    [Symbol.iterator]: () => values[Symbol.iterator](),
  };
  return Object.freeze(view);
}

function weakestSupportMode(
  first: BackendSupportMode | undefined,
  second: BackendSupportMode | undefined,
): BackendSupportMode | undefined {
  if (first == null || second == null) {
    return first === 'unsupported' || second === 'unsupported' ? 'unsupported' : undefined;
  }
  return SUPPORT_MODE_RANK[first] >= SUPPORT_MODE_RANK[second] ? first : second;
}

function combineSupportMap<K extends PropertyKey>(
  first: Readonly<Partial<Record<K, BackendSupportMode>>>,
  second: Readonly<Partial<Record<K, BackendSupportMode>>>,
): Readonly<Partial<Record<K, BackendSupportMode>>> {
  const result: Partial<Record<K, BackendSupportMode>> = {};
  const keys = new Set<K>([
    ...(Reflect.ownKeys(first) as K[]),
    ...(Reflect.ownKeys(second) as K[]),
  ]);
  for (const key of keys) {
    const combined = weakestSupportMode(first[key], second[key]);
    if (combined != null) result[key] = combined;
  }
  return Object.freeze(result);
}

function intersectSet<T>(first: ReadonlySet<T>, second: ReadonlySet<T>): ReadonlySet<T> {
  return runtimeReadonlySet([...first].filter((value) => second.has(value)));
}

function intersectOptionalSet<T>(
  first: ReadonlySet<T> | undefined,
  second: ReadonlySet<T> | undefined,
): ReadonlySet<T> | undefined {
  return first != null && second != null ? intersectSet(first, second) : undefined;
}

function unionSet<T>(first: ReadonlySet<T> | undefined, second: ReadonlySet<T> | undefined): ReadonlySet<T> {
  return runtimeReadonlySet([...(first ?? []), ...(second ?? [])]);
}

function greatestCommonDivisor(a: number, b: number): number {
  let x = a;
  let y = b;
  while (y !== 0) {
    const remainder = x % y;
    x = y;
    y = remainder;
  }
  return x;
}

function combineSupportDetails(
  realtime: BackendSupportDetails | undefined,
  converged: BackendSupportDetails | undefined,
  sceneFallbackAvailable: boolean,
  lightingAvailable: boolean,
): BackendSupportDetails | undefined {
  if (realtime == null || converged == null) return undefined;

  const denoisers = combineSupportMap(realtime.denoisers, converged.denoisers) as
    BackendSupportDetails['denoisers'];
  const shapeModes = new Set([
    ...Object.keys(realtime.denoiserSpatialShapeRequirements ?? {}),
    ...Object.keys(converged.denoiserSpatialShapeRequirements ?? {}),
  ]) as Set<keyof BackendSupportDetails['denoisers']>;
  const denoiserSpatialShapeRequirements: Record<string, {
    readonly minWidth: number;
    readonly minHeight: number;
    readonly widthMultiple: number;
    readonly heightMultiple: number;
  }> = {};
  for (const mode of shapeModes) {
    const first = realtime.denoiserSpatialShapeRequirements?.[mode];
    const second = converged.denoiserSpatialShapeRequirements?.[mode];
    if (first == null && second == null) continue;
    const widthMultipleA = first?.widthMultiple ?? 1;
    const widthMultipleB = second?.widthMultiple ?? 1;
    const heightMultipleA = first?.heightMultiple ?? 1;
    const heightMultipleB = second?.heightMultiple ?? 1;
    denoiserSpatialShapeRequirements[mode] = {
      minWidth: Math.max(first?.minWidth ?? 1, second?.minWidth ?? 1),
      minHeight: Math.max(first?.minHeight ?? 1, second?.minHeight ?? 1),
      widthMultiple:
        (widthMultipleA / greatestCommonDivisor(widthMultipleA, widthMultipleB)) * widthMultipleB,
      heightMultiple:
        (heightMultipleA / greatestCommonDivisor(heightMultipleA, heightMultipleB)) * heightMultipleB,
    };
  }

  const mutationMode = (
    key: keyof BackendSupportDetails['mutations'],
    sceneBacked: boolean,
  ): BackendSupportMode => {
    const first = realtime.mutations[key];
    const second = converged.mutations[key];
    if (sceneBacked && sceneFallbackAvailable && (first === 'unsupported' || second === 'unsupported')) {
      return 'fallback-rebuild';
    }
    return weakestSupportMode(first, second) ?? 'unsupported';
  };

  return {
    primitives: combineSupportMap(realtime.primitives, converged.primitives),
    emitters: combineSupportMap(realtime.emitters, converged.emitters),
    environments: combineSupportMap(realtime.environments, converged.environments),
    analyticShapes: combineSupportMap(realtime.analyticShapes, converged.analyticShapes),
    materials: combineSupportMap(realtime.materials, converged.materials),
    shadows: combineSupportMap(realtime.shadows, converged.shadows),
    denoisers,
    ...(shapeModes.size > 0 ? { denoiserSpatialShapeRequirements } : {}),
    mutations: {
      transform: mutationMode('transform', true),
      positions: mutationMode('positions', true),
      material: mutationMode('material', true),
      emitter: mutationMode('emitter', true),
      topology: mutationMode('topology', true),
      addPrimitive: mutationMode('addPrimitive', true),
      removePrimitive: mutationMode('removePrimitive', true),
      environment: mutationMode('environment', true),
      resize: mutationMode('resize', false),
      lighting: lightingAvailable ? mutationMode('lighting', false) : 'unsupported',
    },
    ...(realtime.thinFilmLayerLimit != null && converged.thinFilmLayerLimit != null
      ? { thinFilmLayerLimit: Math.min(realtime.thinFilmLayerLimit, converged.thinFilmLayerLimit) }
      : {}),
  };
}

function patchSupport(capabilities: EngineCapabilities, key: keyof NonNullable<
  EngineCapabilities['incrementalPatchSupport']
>): boolean {
  return capabilities.incrementalPatchSupport?.[key] ?? capabilities.supportsIncrementalScene;
}

/** Compose only guarantees that remain true across BOTH presentation phases. */
export function composeProgressiveCapabilities(
  realtime: Engine,
  converged: Engine,
  sceneFallbackAvailable = false,
): EngineCapabilities {
  const first = realtime.capabilities;
  const second = converged.capabilities;
  const supportsPrimitivePatch = sceneFallbackAvailable || (
    typeof realtime.updatePrimitive === 'function' &&
    typeof converged.updatePrimitive === 'function'
  );
  const supportsEmitterPatch = sceneFallbackAvailable || (
    typeof realtime.updateEmitter === 'function' &&
    typeof converged.updateEmitter === 'function'
  );
  const supportsAddRemove = sceneFallbackAvailable || (
    first.supportsAddRemovePrimitive === true &&
    second.supportsAddRemovePrimitive === true &&
    typeof realtime.addPrimitive === 'function' &&
    typeof converged.addPrimitive === 'function' &&
    typeof realtime.removePrimitive === 'function' &&
    typeof converged.removePrimitive === 'function'
  );
  const supportsLighting =
    typeof realtime.updateLighting === 'function' &&
    typeof converged.updateLighting === 'function';
  const supportedPrimitiveKinds = intersectOptionalSet(
    first.supportedPrimitiveKinds,
    second.supportedPrimitiveKinds,
  );
  const supportedEnvironmentKinds = intersectOptionalSet(
    first.supportedEnvironmentKinds,
    second.supportedEnvironmentKinds,
  );
  const supportDetails = combineSupportDetails(
    first.supportDetails,
    second.supportDetails,
    sceneFallbackAvailable,
    supportsLighting,
  );
  return {
    supportsIncrementalScene: supportsPrimitivePatch || supportsEmitterPatch,
    supportsAddRemovePrimitive: supportsAddRemove,
    incrementalPatchSupport: {
      transform: supportsPrimitivePatch && (sceneFallbackAvailable || (patchSupport(first, 'transform') && patchSupport(second, 'transform'))),
      positions: supportsPrimitivePatch && (sceneFallbackAvailable || (patchSupport(first, 'positions') && patchSupport(second, 'positions'))),
      material: supportsPrimitivePatch && (sceneFallbackAvailable || (patchSupport(first, 'material') && patchSupport(second, 'material'))),
      emitter: supportsEmitterPatch && (sceneFallbackAvailable || (patchSupport(first, 'emitter') && patchSupport(second, 'emitter'))),
      topology: supportsPrimitivePatch && (sceneFallbackAvailable || (patchSupport(first, 'topology') && patchSupport(second, 'topology'))),
    },
    supportsAuxBuffers: first.supportsAuxBuffers && second.supportsAuxBuffers,
    accumulates: true,
    supportsAccumulatorSeed: false,
    supportsProgressiveSeedSource: false,
    maxSamplesPerPixel: Math.min(first.maxSamplesPerPixel, second.maxSamplesPerPixel),
    maxBounces: Math.min(first.maxBounces, second.maxBounces),
    supportedAnalyticShapes: intersectSet(first.supportedAnalyticShapes, second.supportedAnalyticShapes),
    supportedEmitterKinds: intersectSet(first.supportedEmitterKinds, second.supportedEmitterKinds),
    ...(supportedPrimitiveKinds != null ? { supportedPrimitiveKinds } : {}),
    ...(supportedEnvironmentKinds != null ? { supportedEnvironmentKinds } : {}),
    presentationMode: 'swapchain-optional',
    activeFeatures: unionSet(first.activeFeatures, second.activeFeatures),
    ...(supportDetails != null ? { supportDetails } : {}),
    // Intentionally omit inverseRendering. The progressive facade delegates
    // inverse sessions to the converged engine only; it is not a guarantee
    // shared by both phases and therefore must not be presented as a composite
    // capability.
    causticStrategy: first.causticStrategy === second.causticStrategy
      ? first.causticStrategy
      : 'none',
    debugSurface: false,
  };
}

function progressiveState(realtime: EngineState, converged: EngineState): EngineState {
  if (realtime === 'error' || converged === 'error') return 'error';
  if (realtime === 'disposed' && converged === 'disposed') return 'disposed';
  if (realtime === 'disposed' || converged === 'disposed') return 'error';
  if (realtime === 'initializing' || converged === 'initializing') return 'initializing';
  if (realtime === 'uninitialized' || converged === 'uninitialized') return 'uninitialized';
  if (realtime === 'paused' && converged === 'paused') return 'paused';
  if (realtime === 'ready' && converged === 'ready') return 'ready';
  return 'error';
}
/**
 * Subscribe `cb` to a telemetry channel on BOTH sub-engines and return a single
 * unsubscribe that drains both. The progressive facade fans out the identical
 * pattern for `onFrame`, `onError`, and `onWarning`; this collapses the three
 * near-identical bodies into one. A sub-engine that lacks the channel simply
 * contributes no subscription.
 */
function fanOut<Cb>(
  engines: readonly (Engine | undefined)[],
  channel: (engine: Engine) => ((cb: Cb) => (() => void)) | undefined,
  cb: Cb,
): () => void {
  const unsubs = engines
    .map((engine) => (engine != null ? channel(engine)?.(cb) : undefined))
    .filter((fn): fn is () => void => typeof fn === 'function');
  return () => {
    for (const unsub of unsubs) unsub();
  };
}

/**
 * Adapt a {@link ProgressiveEngineHandle} to the {@link EngineWithBackendId}
 * surface that `attachVitrum` / `<VitrumCanvas>` drive. Scene mutations and the
 * per-frame render route through the coordinator (the single scene-mutation +
 * handoff authority); telemetry fans out to both sub-engines via {@link fanOut}.
 *
 * Lives here (next to `createProgressiveEngine`) rather than in the React
 * component so the non-React vanilla path can consume the same adapter and the
 * fan-out logic is single-source. `getPresentationSource` (R2) is preserved: the
 * coordinator returns the converged offscreen texture after handoff and null
 * while the realtime swapchain engine presents itself — this is what unfreezes
 * the canvas after handoff.
 */
export function progressiveHandleAsEngine(handle: ProgressiveEngineHandle): EngineWithBackendId {
  const coordinator = handle.coordinator;
  const capabilities = composeProgressiveCapabilities(
    handle.realtime, handle.converged, coordinator.getScene() != null,
  );
  const convergedProfile = handle.converged as {
    readonly backendProfileId?: 'pt-webgpu' | 'pt-webgpu-lite';
    readonly profileId?: 'pt-webgpu' | 'pt-webgpu-lite';
  };
  const backendProfileId = handle.backendProfileId ?? convergedProfile.backendProfileId;
  const profileId = handle.profileId ?? convergedProfile.profileId;
  const exportGIState = handle.realtime.exportGIState?.bind(handle.realtime);
  const importGIState = handle.realtime.importGIState?.bind(handle.realtime);
  const engine = {
    backendId: 'progressive' as const,
    ...(backendProfileId != null ? { backendProfileId } : {}),
    ...(profileId != null ? { profileId } : {}),
    get state() { return progressiveState(handle.realtime.state, handle.converged.state); },
    get capabilities() { return capabilities; },
    setScene: (scene: Scene) => coordinator.setScene(scene),
    getScene: () => coordinator.getScene() ?? handle.realtime.getScene?.() ?? null,
    updatePrimitive: (id: string, patch: Parameters<NonNullable<EngineWithBackendId['updatePrimitive']>>[1]) =>
      coordinator.updatePrimitive(id, patch),
    addPrimitive: (primitive: Parameters<NonNullable<EngineWithBackendId['addPrimitive']>>[0]) =>
      coordinator.addPrimitive(primitive),
    removePrimitive: (id: Parameters<NonNullable<EngineWithBackendId['removePrimitive']>>[0]) =>
      coordinator.removePrimitive(id),
    updateEmitter: (id: string, patch: Parameters<NonNullable<EngineWithBackendId['updateEmitter']>>[1]) =>
      coordinator.updateEmitter(id, patch),
    updateEnvironment: (environment: Parameters<NonNullable<EngineWithBackendId['updateEnvironment']>>[0]) =>
      coordinator.updateEnvironment(environment),
    ...(typeof handle.realtime.updateLighting === 'function' &&
    typeof handle.converged.updateLighting === 'function'
      ? {
          updateLighting: (
            opts: Parameters<NonNullable<EngineWithBackendId['updateLighting']>>[0],
          ) => coordinator.updateLighting(opts),
        }
      : {}),
    setSize: (width: number, height: number) => coordinator.setSize(width, height),
    renderFrame: (input: FrameInput) => coordinator.frame(input).output,
    captureFrame: (options?: Parameters<NonNullable<EngineWithBackendId['captureFrame']>>[0]) =>
      coordinator.captureFrame(options),
    // V1-1 / R2 — presentation source for attachVitrum's offscreen blit. The
    // coordinator returns the converged (offscreen pt-webgpu) engine's texture
    // once it hands off, and null while the swapchain realtime engine is driving
    // (it presents itself). This is what unfreezes the canvas after handoff.
    getPresentationSource: () => coordinator.getPresentationSource(),
    reset: () => {
      handle.realtime.reset();
      handle.converged.reset();
      coordinator.reset();
    },
    pause: () => {
      handle.realtime.pause();
      handle.converged.pause();
    },
    ...(typeof handle.converged.createInverseSession === 'function'
      ? { createInverseSession: handle.converged.createInverseSession.bind(handle.converged) }
      : {}),
    ...(typeof handle.converged.getRestirPtResultBuffer === 'function'
      ? { getRestirPtResultBuffer: handle.converged.getRestirPtResultBuffer.bind(handle.converged) }
      : {}),
    ...(exportGIState != null
      ? { exportGIState }
      : {}),
    ...(importGIState != null
      ? { importGIState }
      : {}),
    resume: () => {
      handle.realtime.resume();
      handle.converged.resume();
    },
    dispose: () => handle.dispose(),
    onFrame: (cb: Parameters<NonNullable<EngineWithBackendId['onFrame']>>[0]) =>
      fanOut([handle.realtime, handle.converged], (e) => e.onFrame?.bind(e), cb),
    onProgress: (cb: Parameters<NonNullable<EngineWithBackendId['onProgress']>>[0]) =>
      handle.converged.onProgress?.(cb) ?? (() => {}),
    onError: (cb: Parameters<NonNullable<EngineWithBackendId['onError']>>[0]) =>
      fanOut([handle.realtime, handle.converged], (e) => e.onError?.bind(e), cb),
    onWarning: (cb: Parameters<NonNullable<EngineWithBackendId['onWarning']>>[0]) =>
      fanOut([handle.realtime, handle.converged], (e) => e.onWarning?.bind(e), cb),
  };
  return engine;
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
export interface ProgressiveLimitUnionOptions {
  /** Include converged-engine ReSTIR-PT reuse reservoirs in the shared-device floor. */
  readonly restirPtReuse?: boolean;
  /** Include the realtime engine's opt-in NRC bind-group/buffer/workgroup floor. */
  readonly nrcEnabled?: boolean;
  /** Resolved against DEFAULT_NRC_CONFIG when NRC is enabled. */
  readonly nrcConfig?: Partial<NrcConfig>;
  /** Include the opt-in CWBVH closest-hit traversal buffers in the shared-device floor. */
  readonly cwbvhClosest?: boolean;
}

export function computeProgressiveLimitUnion(
  options: ProgressiveLimitUnionOptions = {},
): Record<string, number> {
  // The two FULL-tier requiredLimits sets.
  const hybridFull = options.nrcEnabled === true
    ? nrcWebGpuRequiredLimitsForConfig(options.nrcConfig ?? {})
    : HYBRID_WEBGPU_REQUIRED_LIMITS;
  const ptBufferFloor = options.restirPtReuse === true
    ? (options.cwbvhClosest === true
        ? PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE
        : PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE)
    : (options.cwbvhClosest === true
        ? PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE
        : PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE);
  const ptWebgpuFull: Record<string, number> = {
    maxStorageBuffersPerShaderStage: ptBufferFloor,
    maxStorageTexturesPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  };
  const union: Record<string, number> = {};
  for (const set of [hybridFull, ptWebgpuFull]) {
    for (const [key, val] of Object.entries(set)) {
      union[key] = Math.max(union[key] ?? 0, val);
    }
  }
  return union;
}

/**
 * Preflight an adapter against the progressive limit UNION and return the list of
 * unmet limits (empty ⇒ the adapter satisfies the union). Each entry is a
 * human-readable `"<key>: need ≥<wanted>, adapter has <cap>"` string.
 *
 * Single source of truth for the identical union-comparison loop that both
 * {@link createProgressiveEngine} and `negotiateWebGPUDevice`'s `'progressive'`
 * target ran inline (D2-4). Callers keep their own gap-naming error message.
 */
export function checkProgressiveLimitUnion(
  adapter: { readonly limits: GPUSupportedLimits },
  options: ProgressiveLimitUnionOptions = {},
): string[] {
  const union = computeProgressiveLimitUnion(options);
  const unmet: string[] = [];
  for (const [key, wanted] of Object.entries(union)) {
    const cap = (adapter.limits as unknown as Record<string, number | undefined>)[key];
    if (typeof cap !== 'number' || cap < wanted) {
      unmet.push(`${key}: need ≥${wanted}, adapter has ${cap ?? 'undefined'}`);
    }
  }
  return unmet;
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
  validateCreateProgressiveEngineOptions(opts);
  validateScene(opts.scene);

  // Scene traversal is pure and belongs before GPU acquisition: malformed
  // geometry must never consume an adapter/device or touch the canvas.
  const vitrumScene: Scene = opts.scene;
  const aabb = computeSceneAABB(vitrumScene);
  const needsTlas = auditSceneNeedsTlas(vitrumScene).needsTlas;

  const failConstruction = (error: unknown): never => {
    reportProgressiveEngineError(opts, error, {
      phase: 'create:progressive',
      backend: 'progressive',
      recoverable: false,
    });
    throw error;
  };

  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    failConstruction(new Error(
      'createProgressiveEngine: WebGPU is unavailable (navigator.gpu is undefined). ' +
        'The progressive engine requires WebGPU for BOTH the realtime and converged ' +
        'backends. Use createEngine({ prefer: "quality" }) for a WebGL2 path tracer.',
    ));
  }

  let requestedAdapter: GPUAdapter | null = null;
  try {
    requestedAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (error) {
    return failConstruction(error);
  }
  if (requestedAdapter == null) {
    return failConstruction(new Error(
      'createProgressiveEngine: navigator.gpu.requestAdapter() returned null (no WebGPU adapter).',
    ));
  }
  const adapter = requestedAdapter;

  // ── Limit-union preflight (Class-A-only) ──────────────────────────────────
  // The shared device must satisfy BOTH backends' FULL floors. Compute the union
  // and check the adapter BEFORE requesting the device, so we can throw a clear,
  // gap-naming error instead of letting requestDevice reject opaquely.
  const restirPtReuse = opts.convergedOptions?.restirPtReuse === true;
  const cwbvhClosest = opts.convergedOptions?.bvhTraversal === 'cwbvh-closest';
  const nrcEnabled = opts.realtimeOptions?.nrcEnabled === true;
  const nrcConfig = resolveHybridNrcConfig(opts.realtimeOptions ?? {});
  const unionOptions = { restirPtReuse, cwbvhClosest, nrcEnabled, nrcConfig };
  const union = computeProgressiveLimitUnion(unionOptions);
  const unmet = checkProgressiveLimitUnion(adapter, unionOptions);
  if (unmet.length > 0) {
    failConstruction(new Error(
      'createProgressiveEngine: this adapter cannot satisfy the device-limit UNION of ' +
        'the walkaround-hybrid (realtime) + pt-webgpu (converged) backends, which is ' +
        'required to run both on one shared device. Unmet limits: ' +
        unmet.join('; ') +
        '. The progressive engine is Class-A-only (discrete GPU / native browser WebGPU); ' +
        'graceful degradation to a single backend is the host\'s call — use ' +
        'createEngine({ prefer: "realtime" }) or createEngine({ prefer: "quality-webgpu" }) ' +
        'on this hardware.',
    ));
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      requiredLimits: union,
      requiredFeatures: requiredWalkaroundNeuralDeviceFeatures(adapter, opts.realtimeOptions),
    });
  } catch (error) {
    return failConstruction(error);
  }

  // From here on the device is allocated; every subsequent throw (profile probe,
  // either sub-engine build, the capability preflight) must destroy it so it
  // never leaks. The try opens immediately after acquisition for exactly that
  // reason.
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
      try {
        opts.onAdapterProfile(await probeAdapterProfile(device));
      } catch {
        // Host telemetry callbacks must not break progressive construction.
      }
    }

    // Both engines ingest the SAME vitrum scene; the handoff requires both to
    // hold identical scene state (see the coordinator's scene-authority
    // forwarding).
    const shared: SharedDeviceCtx = { adapter, device, ownsDeviceLifecycle: false };

    // Both sub-builds receive the progressive callback so construction-phase
    // errors (canvas-configure, adapter/device failures) surface through the
    // host's single callback with the same phase/backend/recoverability event
    // shape as createEngine().
    const subBuildOnError = opts.onError != null
      ? (err: unknown, ev: CreateEngineErrorEvent): void => reportProgressiveEngineError(opts, err, ev)
      : undefined;

    // The two sub-builds reuse createEngine's OWN scene-handling / options-merging
    // (the scale-derived hybrid defaults, the TLAS extension merge, the pt-webgpu
    // tier resolution) by routing through the shared-device seam — no replication.
    // Each gets its own synthesized CreateEngineOptions carrying its `advanced`.
    const realtimeBuildOpts: CreateEngineOptions = {
      canvas: opts.canvas,
      scene: vitrumScene,
      ...(opts.realtimeOptions != null
        ? {
            advanced: opts.realtimeOptions,
            advancedBackend: 'walkaround-hybrid' as const,
          }
        : {}),
      ...(opts.debug != null ? { debug: opts.debug } : {}),
      // onAdapterProfile is intentionally NOT forwarded — the facade already
      // invoked it once above (off the shared device). Forwarding it here would
      // fire the host callback a second time (off the adapter), breaking the
      // "invoked once" contract.
      ...(subBuildOnError != null ? { onError: subBuildOnError } : {}),
      ...(opts.onWarning != null ? { onWarning: opts.onWarning } : {}),
    };
    realtime = await constructWalkaround(
      realtimeBuildOpts,
      vitrumScene,
      aabb,
      needsTlas,
      shared,
    );

    const convergedBuildOpts: CreateEngineOptions = {
      canvas: opts.canvas,
      scene: vitrumScene,
      ...(opts.convergedOptions != null
        ? {
            advanced: opts.convergedOptions,
            advancedBackend: 'pt-webgpu' as const,
          }
        : {}),
      ...(opts.debug != null ? { debug: opts.debug } : {}),
      ...(subBuildOnError != null ? { onError: subBuildOnError } : {}),
      ...(opts.onWarning != null ? { onWarning: opts.onWarning } : {}),
    };
    converged = await constructPathTracerWebGPU(
      convergedBuildOpts,
      vitrumScene,
      shared,
    );

    // ── Capability preflight ────────────────────────────────────────────────
    // The seed handoff requires the realtime engine to expose a seed SOURCE and
    // the converged engine a seed SINK. Assert both — a clear throw here beats a
    // silent no-op handoff at runtime (the coordinator would degrade to a black
    // reset). pt-webgpu advertises supportsAccumulatorSeed unconditionally (the
    // accum buffers exist on both tiers); we still assert it so the contract is
    // enforced at the seam, not assumed.
    if (
      realtime.capabilities.supportsProgressiveSeedSource !== true ||
      typeof realtime.getProgressiveSeedTexture !== 'function'
    ) {
      throw new Error(
        'createProgressiveEngine: the realtime engine does not advertise ' +
          'a callable, advertised progressive seed source — it cannot provide a seed texture for the ' +
          'converged engine\'s accumulator. The progressive seed handoff requires it.',
      );
    }
    if (
      converged.capabilities.supportsAccumulatorSeed !== true ||
      typeof converged.seedAccumulator !== 'function'
    ) {
      throw new Error(
        'createProgressiveEngine: the converged engine does not advertise ' +
          'a callable, advertised accumulator seed sink — it cannot be seeded from the realtime ' +
          'engine. The progressive seed handoff requires it.',
      );
    }

    // Belt-and-braces: configure the canvas context against the shared device for
    // the REALTIME engine's swap-chain presentation (constructWalkaround already
    // did this, but it is idempotent + best-effort, so a host that swapped the
    // canvas is still covered).
    // H31-b — thread the host onError callback so canvas-configure failures are
    // surfaced with the same structured event shape as createEngine().
    configureWebGpuCanvas(opts.canvas, device, (err) => {
      reportProgressiveEngineError(opts, err, {
        phase: 'canvas-configure',
        backend: 'walkaround-hybrid',
        recoverable: true,
      });
    });

    const coordinatorOpts: ProgressiveHandoffOptions = {
      realtime,
      converged,
      scene: vitrumScene,
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
      ...(opts.controller != null ? { controller: opts.controller } : {}),
      ...(opts.controllerDeltaSeconds != null
        ? { controllerDeltaSeconds: opts.controllerDeltaSeconds }
        : {}),
      ...(opts.controllerLoop != null ? { controllerLoop: opts.controllerLoop } : {}),
    };
    const coordinator = new ProgressiveHandoffCoordinator(coordinatorOpts);

    const builtRealtime = realtime;
    const builtConverged = converged;
    const convergedProfile = builtConverged as {
      readonly backendProfileId?: 'pt-webgpu' | 'pt-webgpu-lite';
      readonly profileId?: 'pt-webgpu' | 'pt-webgpu-lite';
    };
    let disposed = false;
    return {
      coordinator,
      ...(convergedProfile.backendProfileId != null
        ? { backendProfileId: convergedProfile.backendProfileId }
        : {}),
      ...(convergedProfile.profileId != null
        ? { profileId: convergedProfile.profileId }
        : {}),
      realtime: builtRealtime,
      converged: builtConverged,
      dispose(): void {
        if (disposed) return;
        disposed = true;
        // Sub-engine disposes are no-ops on the device (shared-device path); the
        // facade destroys it once below, after both are torn down.
        try { builtRealtime.dispose(); } catch { /* best-effort sub-engine cleanup — ignore */ }
        try { builtConverged.dispose(); } catch { /* best-effort sub-engine cleanup — ignore */ }
        try { device.destroy(); } catch { /* best-effort device destroy — ignore */ }
      },
    };
  } catch (err) {
    // Build failed after acquiring the device (or after one sub-engine built):
    // tear down whatever exists, then the device, so we never leak it.
    try { realtime?.dispose(); } catch { /* best-effort cleanup — ignore */ }
    try { converged?.dispose(); } catch { /* best-effort cleanup — ignore */ }
    try { device.destroy(); } catch { /* best-effort device destroy — ignore */ }
    throw err;
  }
}

function reportProgressiveEngineError(
  opts: CreateProgressiveEngineOptions,
  error: unknown,
  event: CreateEngineErrorEvent,
): void {
  try {
    opts.onError?.(error, event);
  } catch { /* host error callback must not propagate — ignore */ }
}
