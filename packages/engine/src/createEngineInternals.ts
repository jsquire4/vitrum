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
export function stripOwnershipCriticalKeys<T extends Record<string, unknown>>(
  advanced: T | undefined,
  backend: CreateEngineBackendId,
  onWarning?: (warning: EngineWarning) => void,
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
    const message =
      `[vitrum/createEngine] advanced.${overridden.join('/')} was supplied but ` +
      `createEngine owns the ${backend} device lifecycle — the supplied ` +
      `${overridden.join('/')} key(s) have been ignored to prevent a double-dispose. ` +
      `To bring your own device, use the backend factory directly.`;
    emitCreateEngineWarning(
      onWarning,
      {
        code: 'createEngine.advanced-ownership-key-ignored',
        backend: 'createEngine',
        phase: 'construction',
        method: 'createEngine',
        message,
        details: { backend, keys: overridden },
      },
    );
  }
  return stripped as Omit<T, OwnershipCriticalKey>;
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
  onWarning?: (warning: EngineWarning) => void,
): void {
  if (advanced == null) return;
  const keys = Object.keys(advanced as Record<string, unknown>).filter(
    (k) => (advanced as Record<string, unknown>)[k] !== undefined,
  );
  if (keys.length === 0) return;
  const message =
    `[vitrum/createEngine] advanced options (keys: ${keys.join(', ')}) were supplied ` +
    `but the preferred backend '${preferredBackend}' was unavailable — they are now ` +
    `being applied to the fallback backend '${resolvedBackend}'. Keys authored for ` +
    `'${preferredBackend}' may be silently ignored or misinterpreted by '${resolvedBackend}'. ` +
    `Pass prefer:'${resolvedBackend}' explicitly to suppress this warning.`;
  emitCreateEngineWarning(
    onWarning,
    {
      code: 'createEngine.advanced-cross-backend',
      backend: 'createEngine',
      phase: 'fallback',
      method: 'createEngine',
      message,
      details: { preferredBackend, resolvedBackend, keys },
    },
  );
}

function advancedKeys(advanced: unknown): string[] {
  if (advanced == null || typeof advanced !== 'object') return [];
  return Object.keys(advanced as Record<string, unknown>).filter(
    (k) => (advanced as Record<string, unknown>)[k] !== undefined,
  );
}

function backendAdvancedKeys(advancedByBackend: CreateEngineAdvancedByBackend | undefined): string[] {
  if (advancedByBackend == null) return [];
  const keys: string[] = [];
  for (const backend of Object.keys(advancedByBackend) as CreateEngineBackendId[]) {
    if (advancedKeys(advancedByBackend[backend]).length > 0) keys.push(backend);
  }
  return keys;
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
  const keyedBackends = backendAdvancedKeys(opts.advancedByBackend);

  if (opts.advancedByBackend != null) {
    if (legacyKeys.length > 0) {
      emitCreateEngineWarning(opts.onWarning, {
        code: 'createEngine.advanced-legacy-ignored',
        backend: 'createEngine',
        phase: 'construction',
        method: 'createEngine',
        message:
          `[vitrum/createEngine] advancedByBackend was supplied, so legacy advanced ` +
          `keys (${legacyKeys.join(', ')}) are ignored. Put backend-specific keys under ` +
          `advancedByBackend['${backend}'] to make auto selection deterministic.`,
        details: { selectedBackend: backend, legacyKeys, advancedByBackend: keyedBackends },
      });
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
    emitCreateEngineWarning(opts.onWarning, {
      code: 'createEngine.advanced-auto-backend-ambiguous',
      backend: 'createEngine',
      phase: 'construction',
      method: 'createEngine',
      message:
        `[vitrum/createEngine] legacy advanced keys (${legacyKeys.join(', ')}) are being ` +
        `applied to auto-selected backend '${backend}'. Pass advancedBackend or ` +
        `advancedByBackend to make this deterministic across machines and scene changes.`,
      details: { selectedBackend: backend, keys: legacyKeys },
    });
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
