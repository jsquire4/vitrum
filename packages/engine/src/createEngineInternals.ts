// Shared types and utilities for the createEngine facade and backend modules.
//
// Extracted from createEngine.ts (C1) to break the circular import that would
// arise if backends/walkaround.ts, backends/ptWebgpu.ts, and backends/ptWebgl2.ts
// imported from createEngine.ts while createEngine.ts also imported from them.
//
// Import order:
//   createEngineInternals (no vitrum/engine imports)
//     ↑
//   backends/walkaround.ts, backends/ptWebgpu.ts, backends/ptWebgl2.ts
//     ↑
//   createEngine.ts   (re-exports everything for back-compat)
//
// @internal — not part of the public @vitrum/engine API surface.

import type { Engine, AdapterProfile, EngineWarning, Scene } from '@vitrum/core';
import type { HybridEngineOptions } from '@vitrum/walkaround-hybrid';
import type { SceneAABB } from './sceneAABB.js';
import { wrapWithIdempotentDispose } from './idempotentDispose.js';
import type { GIStatePersistable } from './idempotentDispose.js';
import type { PTEngineWebGL2Options } from '@vitrum/pt-webgl2';
import type { PTEngineWebGPUOptions } from '@vitrum/pt-webgpu';
import type { EngineBackendId, EnginePreference } from './createEngineScale.js';

// Re-export wrapWithIdempotentDispose here so backends don't need to import
// idempotentDispose.ts directly (reduces backend import surface).
export { wrapWithIdempotentDispose };

// ────────────────────────────────────────────────────────────────────────────
// Core types
// ────────────────────────────────────────────────────────────────────────────

export type WebGL2PathTracerAdvancedOptions = Partial<Omit<PTEngineWebGL2Options, 'device'>>;

export type CreateEngineBackendId = EngineBackendId;

const CREATE_ENGINE_BACKEND_IDS = new Set<CreateEngineBackendId>([
  'walkaround-hybrid',
  'pt-webgpu',
  'pt-webgl2',
]);

/**
 * Construction failed because the requested backend is unavailable on the
 * current host. Only this error class is eligible for createEngine fallback;
 * configuration, scene, shader, and implementation errors must propagate.
 * @internal
 */
export class BackendUnavailableError extends Error {
  readonly backend: CreateEngineBackendId;

  constructor(
    backend: CreateEngineBackendId,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = 'BackendUnavailableError';
    this.backend = backend;
  }
}

/** @internal */
export function isBackendUnavailableError(
  error: unknown,
): error is BackendUnavailableError {
  return error instanceof BackendUnavailableError;
}

function assertPlainDataObject(
  value: unknown,
  label: string,
  allowedKeys?: ReadonlySet<string>,
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
    if (allowedKeys != null && !allowedKeys.has(key)) {
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

function assertBackendId(value: unknown, label: string): asserts value is CreateEngineBackendId {
  if (typeof value !== 'string' || !CREATE_ENGINE_BACKEND_IDS.has(value as CreateEngineBackendId)) {
    throw new RangeError(
      `${label} must be one of ${Array.from(CREATE_ENGINE_BACKEND_IDS, (entry) => JSON.stringify(entry)).join(', ')} ` +
        `(got ${String(value)})`,
    );
  }
}

function describeValidationValue(value: unknown): string {
  return value !== null && typeof value === 'object'
    ? Object.prototype.toString.call(value)
    : String(value);
}

export interface CreateEngineGltfAssetHint {
  readonly recommendedBackend?: {
    readonly backend: CreateEngineBackendId;
  };
}

export interface CreateEngineAdvancedByBackend {
  readonly 'walkaround-hybrid'?: Partial<HybridEngineOptions>;
  readonly 'pt-webgpu'?: Partial<PTEngineWebGPUOptions>;
  readonly 'pt-webgl2'?: WebGL2PathTracerAdvancedOptions;
}

/** Engine returned by {@link createEngine} with its chosen backendId attached.
 *  Hosts that need to know which backend was selected (e.g. to type-narrow
 *  `opts.advanced` for backend-specific API calls) read this field.
 *
 *  The `advanced` cast to a backend-specific options type is safe ONLY when
 *  `engine.backendId` matches the target backend — gate such casts on this field. */
export interface EngineWithBackendId extends Engine, Partial<GIStatePersistable> {
  readonly backendId: CreateEngineBackendId;
  /** Concrete backend profile selected at construction. Present for backends
   *  whose runtime profile is narrower than their backend identity. */
  readonly backendProfileId?: 'pt-webgpu' | 'pt-webgpu-lite';
  /** Legacy-compatible profile identity. Forwarded verbatim when a backend supplies it. */
  readonly profileId?: 'pt-webgpu' | 'pt-webgpu-lite';
}

/** Runtime identity accepted by lifecycle hosts. `progressive` is a composite
 * walkaround + pt-webgpu facade, not a backend selectable by createEngine. */
export type RuntimeEngineBackendId = CreateEngineBackendId | 'progressive';

/** Engine accepted by attachVitrum and other lifecycle surfaces, including
 * composite facades that are not selectable createEngine backends. */
export interface RuntimeEngineWithBackendId extends Engine, Partial<GIStatePersistable> {
  readonly backendId: RuntimeEngineBackendId;
  readonly backendProfileId?: 'pt-webgpu' | 'pt-webgpu-lite';
  readonly profileId?: 'pt-webgpu' | 'pt-webgpu-lite';
}

export type CreateEngineErrorPhase =
  | 'create:walkaround-hybrid'
  | 'create:pt-webgpu'
  | 'create:progressive'
  | 'create:pt-webgl2'
  | 'canvas-configure'
  | 'attach:scene-controller'
  | 'attach:resize'
  | 'attach:gi-export'
  | 'attach:gi-import'
  | 'attach:auto-recreate'
  | 'attach:initial'
  | 'attach:swapchain'
  | 'attach:present'
  | 'attach:frame-preparation'
  | 'attach:renderFrame';

export interface CreateEngineErrorEvent {
  readonly phase: CreateEngineErrorPhase;
  readonly backend?: RuntimeEngineBackendId;
  readonly recoverable: boolean;
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

  /** Optional glTF planning result. When `prefer` is left as `'auto'`, the
   *  createEngine backend picker follows `gltfAsset.recommendedBackend.backend`
   *  instead of falling back to the generic triangle-count heuristic. This is
   *  structural on purpose: @vitrum/engine does not need to import adapter
   *  runtime code to consume the adapter's recommendation. */
  readonly gltfAsset?: CreateEngineGltfAssetHint;

  /** Legacy backend-specific overrides. Merged on top of the createEngine()-
   *  derived defaults; user-supplied keys win. Most users leave empty.
   *
   *  For auto backend selection, prefer `advancedByBackend` so overrides are
   *  consumed only by the backend they were authored for. If using this legacy
   *  field with auto selection, set `advancedBackend` to make the target
   *  explicit and receive a structured warning if a different backend is chosen. */
  readonly advanced?: Partial<HybridEngineOptions> | WebGL2PathTracerAdvancedOptions | Partial<PTEngineWebGPUOptions>;

  /** Explicit target for legacy `advanced`. When the selected backend differs,
   *  the legacy bag is ignored and a structured warning is emitted instead of
   *  applying keys to the wrong backend. */
  readonly advancedBackend?: CreateEngineBackendId;

  /** Backend-keyed override bags for predictable auto selection and fallback.
   *  When present, the selected backend consumes only its own entry. */
  readonly advancedByBackend?: CreateEngineAdvancedByBackend;

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

  /** Host-visible nonfatal warning report. Mirrors contract-affecting
   *  `console.warn` output such as fallback, ignored advanced options, and
   *  backend capability downgrades. */
  readonly onWarning?: (warning: EngineWarning) => void;
}

const CREATE_ENGINE_OPTION_KEYS = {
  canvas: true,
  scene: true,
  prefer: true,
  gltfAsset: true,
  advanced: true,
  advancedBackend: true,
  advancedByBackend: true,
  debug: true,
  onAdapterProfile: true,
  onError: true,
  onWarning: true,
} as const satisfies Readonly<Record<keyof CreateEngineOptions, true>>;

const CREATE_ENGINE_PREFERENCES: ReadonlySet<string> = new Set([
  'auto',
  'realtime',
  'quality',
  'quality-webgpu',
]);

function assertNoOwnershipCriticalKeys(
  value: Record<string, unknown>,
  label: string,
): void {
  for (const key of OWNERSHIP_CRITICAL_KEYS) {
    if (Object.prototype.hasOwnProperty.call(value, key)) {
      throw new TypeError(
        `${label}.${key} is ownership-critical and cannot be supplied through createEngine; ` +
          'use the concrete backend factory to provide a device, canvas, or context',
      );
    }
  }
}

/**
 * Validate the createEngine payload before scene traversal, capability probes,
 * adapter requests, or canvas-context acquisition. This deliberately rejects
 * accessor-backed and ambiguous configuration instead of normalising it.
 * @internal
 */
export function validateCreateEngineOptionsShape(
  value: unknown,
): asserts value is CreateEngineOptions {
  assertPlainDataObject(
    value,
    'createEngine options',
    new Set(Object.keys(CREATE_ENGINE_OPTION_KEYS)),
  );

  if (value.canvas == null || typeof value.canvas !== 'object') {
    throw new TypeError('createEngine: opts.canvas is required and must be an object');
  }
  if (typeof (value.canvas as { readonly getContext?: unknown }).getContext !== 'function') {
    throw new TypeError('createEngine: opts.canvas must expose getContext()');
  }
  for (const dimension of ['width', 'height'] as const) {
    const component = (value.canvas as { readonly width?: unknown; readonly height?: unknown })[dimension];
    if (typeof component !== 'number' || !Number.isSafeInteger(component) || component < 0) {
      throw new RangeError(
        `createEngine: opts.canvas.${dimension} must be a non-negative safe integer ` +
          `(got ${String(component)})`,
      );
    }
  }
  if (value.scene == null || typeof value.scene !== 'object') {
    throw new TypeError('createEngine: opts.scene is required and must be an object');
  }

  const preference: unknown = value.prefer;
  if (
    preference !== undefined &&
    (typeof preference !== 'string' || !CREATE_ENGINE_PREFERENCES.has(preference))
  ) {
    throw new RangeError(
      `createEngine: opts.prefer must be one of ` +
        `${Array.from(CREATE_ENGINE_PREFERENCES, (entry) => JSON.stringify(entry)).join(', ')} ` +
        `(got ${describeValidationValue(preference)})`,
    );
  }
  if (value.debug !== undefined && typeof value.debug !== 'boolean') {
    throw new TypeError('createEngine: opts.debug must be a boolean when supplied');
  }
  for (const callback of ['onAdapterProfile', 'onError', 'onWarning'] as const) {
    if (value[callback] !== undefined && typeof value[callback] !== 'function') {
      throw new TypeError(`createEngine: opts.${callback} must be a function when supplied`);
    }
  }

  if (value.gltfAsset !== undefined) {
    assertPlainDataObject(
      value.gltfAsset,
      'createEngine options.gltfAsset',
      new Set(['recommendedBackend']),
    );
    if (value.gltfAsset.recommendedBackend !== undefined) {
      assertPlainDataObject(
        value.gltfAsset.recommendedBackend,
        'createEngine options.gltfAsset.recommendedBackend',
        new Set(['backend']),
      );
      assertBackendId(
        value.gltfAsset.recommendedBackend.backend,
        'createEngine options.gltfAsset.recommendedBackend.backend',
      );
    }
  }

  if (value.advancedBackend !== undefined) {
    assertBackendId(value.advancedBackend, 'createEngine options.advancedBackend');
  }
  if (value.advanced !== undefined) {
    assertPlainDataObject(value.advanced, 'createEngine options.advanced');
    assertNoOwnershipCriticalKeys(value.advanced, 'createEngine options.advanced');
  }
  if (value.advancedByBackend !== undefined) {
    assertPlainDataObject(
      value.advancedByBackend,
      'createEngine options.advancedByBackend',
      CREATE_ENGINE_BACKEND_IDS,
    );
    for (const backend of CREATE_ENGINE_BACKEND_IDS) {
      const advanced = value.advancedByBackend[backend];
      if (advanced === undefined) continue;
      assertPlainDataObject(
        advanced,
        `createEngine options.advancedByBackend[${JSON.stringify(backend)}]`,
      );
      assertNoOwnershipCriticalKeys(
        advanced,
        `createEngine options.advancedByBackend[${JSON.stringify(backend)}]`,
      );
    }
  }

  const legacyAdvancedKeys = value.advanced == null ? [] : Object.keys(value.advanced);
  if (value.advancedByBackend !== undefined && (value.advanced !== undefined || value.advancedBackend !== undefined)) {
    throw new TypeError(
      'createEngine: advancedByBackend cannot be combined with legacy advanced or advancedBackend',
    );
  }
  if (legacyAdvancedKeys.length > 0 && value.advancedBackend === undefined) {
    throw new TypeError(
      'createEngine: non-empty legacy advanced requires advancedBackend; use advancedByBackend for fallback-safe options',
    );
  }
  if (value.advancedBackend !== undefined && value.advanced === undefined) {
    throw new TypeError('createEngine: advancedBackend requires an advanced option bag');
  }
}

// ────────────────────────────────────────────────────────────────────────────
// SharedDeviceCtx
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

// ────────────────────────────────────────────────────────────────────────────
// BackendConstructor + dispatch table type
// ────────────────────────────────────────────────────────────────────────────

/**
 * The common shape every backend constructor implements.
 * `canvas` + `scene` + `opts` + optional `shared` → Promise<EngineWithBackendId>.
 *
 * The dispatch table (`BACKEND_CONSTRUCTORS`) maps each `CreateEngineBackendId`
 * to a value of this type, replacing the if/else chain in `createEngine`.
 *
 * @internal
 */
export type BackendConstructor = (
  opts: CreateEngineOptions,
  scene: Scene,
  aabb: SceneAABB,
  needsTlas: boolean,
  shared?: SharedDeviceCtx,
) => Promise<EngineWithBackendId>;

// ────────────────────────────────────────────────────────────────────────────
// Utility functions shared by createEngine.ts and backends
// ────────────────────────────────────────────────────────────────────────────

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

const OWNERSHIP_CRITICAL_KEYS = ['device', 'canvas', 'context'] as const;
type OwnershipCriticalKey = (typeof OWNERSHIP_CRITICAL_KEYS)[number];

/**
 * Strip ownership-critical keys (`device`, `canvas`, `context`) from an
 * `advanced` option bag before spreading it over createEngine's own
 * factory-derived values. If `advanced` supplied any of these keys they
 * would silently override the device that createEngine minted and owns — the
 * dispose path would destroy a device the HOST owns (or a completely alien
 * object), so the engine-minted one would leak. The function rejects the
 * entire option bag before construction; it never publishes a silently
 * stripped configuration.
 *
 * @internal — used by the two device-owning constructors (walkaround-hybrid +
 * pt-webgpu).  The WebGL2 path already uses `Omit<…, 'device'>` in its type
 * so it is safe without this helper.
 */
export function stripOwnershipCriticalKeys<T extends Record<string, unknown>>(
  advanced: T | undefined,
  backend: CreateEngineBackendId,
): Omit<T, OwnershipCriticalKey> {
  if (advanced == null) return {} as Omit<T, OwnershipCriticalKey>;
  const overridden: string[] = [];
  for (const key of OWNERSHIP_CRITICAL_KEYS) {
    if (Object.prototype.hasOwnProperty.call(advanced, key)) {
      overridden.push(key);
    }
  }
  if (overridden.length > 0) {
    throw new TypeError(
      `[vitrum/createEngine] advanced ownership-critical key(s) ` +
        `${overridden.join(', ')} cannot be supplied for ${backend}; ` +
        'use the concrete backend factory to provide host-owned handles',
    );
  }
  return { ...advanced };
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
function advancedKeys(advanced: unknown): string[] {
  if (advanced == null || typeof advanced !== 'object') return [];
  return Object.keys(advanced as Record<string, unknown>).filter(
    (k) => (advanced as Record<string, unknown>)[k] !== undefined,
  );
}

/**
 * Resolve the backend-specific options bag that should be merged into a
 * constructor. This is the guardrail for `createEngine({ prefer:'auto' })`:
 * backend-keyed options are selected precisely, while legacy untagged
 * `advanced` remains compatible but emits diagnostics when its target is
 * ambiguous or mismatched.
 *
 * @internal Exported for unit-test access. Not part of the public API.
 */
export function resolveAdvancedForBackend(
  opts: Pick<CreateEngineOptions, 'advanced' | 'advancedBackend' | 'advancedByBackend' | 'prefer' | 'onWarning'>,
  backend: CreateEngineBackendId,
): CreateEngineOptions['advanced'] | undefined {
  const legacyKeys = advancedKeys(opts.advanced);

  if (opts.advancedByBackend != null) {
    if (legacyKeys.length > 0) {
      throw new TypeError(
        `[vitrum/createEngine] advancedByBackend cannot be combined with legacy ` +
          `advanced keys (${legacyKeys.join(', ')})`,
      );
    }
    return opts.advancedByBackend[backend];
  }

  if (legacyKeys.length === 0) return undefined;

  if (opts.advancedBackend != null && opts.advancedBackend !== backend) {
    emitCreateEngineWarning(opts.onWarning, {
      code: 'createEngine.advanced-target-backend-mismatch',
      backend: 'createEngine',
      phase: 'construction',
      method: 'createEngine',
      message:
        `[vitrum/createEngine] legacy advanced keys (${legacyKeys.join(', ')}) were tagged ` +
        `for '${opts.advancedBackend}', but createEngine selected '${backend}'. The legacy ` +
        `advanced bag was ignored for '${backend}'. Use advancedByBackend for fallback-safe ` +
        `per-backend overrides.`,
      details: { advancedBackend: opts.advancedBackend, selectedBackend: backend, keys: legacyKeys },
    });
    return undefined;
  }

  if ((opts.prefer == null || opts.prefer === 'auto') && opts.advancedBackend == null) {
    throw new TypeError(
      `[vitrum/createEngine] legacy advanced keys (${legacyKeys.join(', ')}) require ` +
        'advancedBackend under auto selection; use advancedByBackend for fallback-safe options',
    );
  }

  return opts.advanced;
}

/** @internal */
export function reportCreateEngineError(
  opts: CreateEngineOptions,
  error: unknown,
  event: CreateEngineErrorEvent,
): void {
  try {
    opts.onError?.(error, event);
  } catch { /* host error callback must not propagate — ignore */ }
}

/** @internal */
export function emitCreateEngineWarning(
  onWarning: ((warning: EngineWarning) => void) | undefined,
  warning: EngineWarning,
  ...consoleArgs: readonly unknown[]
): void {
  console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
  try {
    onWarning?.(warning);
  } catch { /* host warning callback must not propagate — ignore */ }
}

/** Attach a `backendId` property to an engine returned by a constructor.
 *  The property is non-enumerable-but-readable so it doesn't interfere with
 *  spread/clone patterns the host might use on the Engine object.
 *  @internal */
export function attachBackendId(
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

// ────────────────────────────────────────────────────────────────────────────
// NOTE on the engine=null sentinel + dispose-on-throw pattern (C1/D1.6):
// the three backend constructors (backends/walkaround.ts, ptWebgpu.ts,
// ptWebgl2.ts) each repeat the same ~8-line guard:
//   let engine = null; try { engine = await create(...); ...;
//   const built = engine; engine = null; return wrap(built, onDispose); }
//   catch (err) { try { engine?.dispose(); } catch {} ; cleanup(); throw err; }
// A `withEngineLifetime` helper was prototyped during the 2026-06-10 sweep and
// REJECTED: it erases each backend's specific engine type (walkaround returns
// Engine & Partial<GIStatePersistable>) through the helper's generic seam, in
// the highest-blast-radius public path, to save 3 small duplications. The
// repetition is intentional; keep the guard semantics identical at all three
// sites if you touch one.
// ────────────────────────────────────────────────────────────────────────────
