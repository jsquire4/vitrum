// Idempotent-dispose engine proxy.
//
// Wraps a backend Engine so that:
//   • dispose() is idempotent (engine.dispose() + postDispose() fire at most once);
//   • renderFrame() after dispose returns a no-op skipped frame (host RAF loops
//     that race the dispose don't crash);
//   • every OPTIONAL Engine method is conditionally exposed (only when the
//     backend implements it AND the relevant capability is advertised) and its
//     post-dispose behaviour matches the backend-agnostic contract.
//
// The optional-method forwarding is data-driven (OPTIONAL_METHOD_PROXIES below):
// each entry pairs the backend method name with the exact disposed-behaviour the
// facade must reproduce. Divergent disposed-behaviours (no-op vs empty-unsub vs
// throw) are encoded as DATA — do not unify them.

import type { Engine, GIStatePersistable as GIStatePersistableCore } from '@vitrum/core';
import type { GIStateSnapshot } from '@vitrum/walkaround-hybrid';

// Concrete alias: bind the generic core interface to the walkaround-hybrid
// snapshot type. All engine-package consumers that import `GIStatePersistable`
// from this module (vanilla.ts, createEngine.ts, giStateProxy.test.ts) get the
// concrete form `{ exportGIState(): Promise<GIStateSnapshot|null>; importGIState(...) }`
// — unchanged from before this move. The generic form is available from
// @vitrum/core as `GIStatePersistableCore<TSnapshot>` for hosts that only need
// the protocol contract without the concrete snapshot type.
export type GIStatePersistable = GIStatePersistableCore<GIStateSnapshot>;

/**
 * Disposed-behaviour kinds for an optional Engine method, captured as data.
 *
 *   • 'noop'         — when disposed, swallow the call and return undefined.
 *                      (updatePrimitive/updateEmitter/addPrimitive/removePrimitive/
 *                       updateEnvironment/setSize/updateLighting/seedAccumulator)
 *   • 'null'         — when disposed, return null synchronously (for methods
 *                      whose contract return type is `T | null` — distinct from
 *                      noop whose return is `undefined`).
 *                      (getRestirPtResultBuffer/getScene/getProgressiveSeedTexture)
 *   • 'promise-null' — when disposed, return `Promise.resolve(null)` (for async
 *                      methods whose contract return type is `Promise<T | null>`).
 *                      (captureFrame/exportGIState)
 *   • 'value-false'  — when disposed, return `false` synchronously (for methods
 *                      whose contract return type is `boolean`).
 *                      (importGIState)
 *   • 'empty-unsub'  — the method returns an unsubscribe fn; when disposed, return
 *                      a no-op unsubscribe `() => {}` without forwarding.
 *                      (onFrame/onProgress/onError/onWarning)
 *   • 'throw'        — when disposed, throw (the engine is torn down; refuse).
 *                      (createInverseSession)
 */
type DisposedBehavior = 'noop' | 'null' | 'promise-null' | 'value-false' | 'empty-unsub' | 'throw';

type OptionalMethodName =
  | 'updatePrimitive'
  | 'updateEmitter'
  | 'addPrimitive'
  | 'removePrimitive'
  | 'updateEnvironment'
  | 'setSize'
  | 'updateLighting'
  | 'onFrame'
  | 'onProgress'
  | 'onError'
  | 'onWarning'
  | 'createInverseSession'
  | 'getRestirPtResultBuffer'
  | 'getPresentationSource'
  // Scene read-back — optional on Engine; all three shipping backends implement it.
  | 'getScene'
  // Async GPU→CPU pixel readback — optional on Engine; all three shipping backends.
  | 'captureFrame'
  // Progressive walkaround→PT seed source/sink (P8).
  | 'getProgressiveSeedTexture'
  | 'seedAccumulator'
  // GI-state persistence — walkaround-hybrid only (not on Engine interface;
  // carried via GIStatePersistable and cast at the forwarding site).
  | 'exportGIState'
  | 'importGIState';

interface OptionalMethodProxy {
  readonly method: OptionalMethodName;
  readonly disposedBehavior: DisposedBehavior;
  /** Extra gating beyond `engine[method] != null`. `caps` carries the
   *  precomputed capability flags. Defaults to always-eligible. */
  readonly eligible?: (caps: CapabilityGates) => boolean;
  /** Message thrown when `disposedBehavior === 'throw'`. */
  readonly throwMessage?: string;
}

interface CapabilityGates {
  readonly primitivePatchAdvertised: boolean;
  readonly emitterPatchAdvertised: boolean;
  readonly addRemoveAdvertised: boolean;
  readonly seedSourceAdvertised: boolean;
  readonly accumulatorSeedAdvertised: boolean;
}

// The table reproduces EXACTLY the per-method disposed-behaviour of the prior
// hand-coded proxy. Each row is behaviour-preserving — see the kind docs above.
// Exported for the compile-time conformance gate in optionalMethodConformance.test.ts.
export const OPTIONAL_METHOD_PROXIES: readonly OptionalMethodProxy[] = [
  { method: 'updatePrimitive', disposedBehavior: 'noop', eligible: (c) => c.primitivePatchAdvertised },
  { method: 'updateEmitter', disposedBehavior: 'noop', eligible: (c) => c.emitterPatchAdvertised },
  // Whole-primitive add/remove is gated on the dedicated capability (was
  // write-only — set by every backend but never consulted, so the facade
  // silently dropped addPrimitive/removePrimitive even when supported).
  { method: 'addPrimitive', disposedBehavior: 'noop', eligible: (c) => c.addRemoveAdvertised },
  { method: 'removePrimitive', disposedBehavior: 'noop', eligible: (c) => c.addRemoveAdvertised },
  { method: 'updateEnvironment', disposedBehavior: 'noop' },
  { method: 'setSize', disposedBehavior: 'noop' },
  { method: 'updateLighting', disposedBehavior: 'noop' },
  { method: 'onFrame', disposedBehavior: 'empty-unsub' },
  { method: 'onProgress', disposedBehavior: 'empty-unsub' },
  // H61 class — GPU/runtime error subscription. Returns an unsubscribe fn →
  // 'empty-unsub' disposed behaviour (same as onFrame/onProgress). Without this
  // row the createEngine facade silently hid onError, so vanilla.ts's
  // `engine.onError ? ...` check was always false and device-loss recovery was
  // silently dead. All three shipping backends implement onError.
  { method: 'onError', disposedBehavior: 'empty-unsub' },
  // Structured non-fatal warning subscription. Same disposed behavior as the
  // other subscription methods.
  { method: 'onWarning', disposedBehavior: 'empty-unsub' },
  // WS5 — inverse-rendering (differentiable RT) sessions. After dispose the
  // proxy refuses to open a NEW session (the engine is torn down). Existing
  // sessions do not outlive the engine: an in-flight or later step rejects,
  // and the host should dispose each session before or with the engine.
  {
    method: 'createInverseSession',
    disposedBehavior: 'throw',
    throwMessage: 'createInverseSession: engine is disposed',
  },
  // H61 — debug accessor for the ReSTIR-PT reuse output buffer
  // (pt-webgpu, gated by active feature 'pt-webgpu-restir-pt-reuse').
  // Added to the Engine contract in H14-C; without this row the createEngine
  // facade silently hid it. After dispose the buffer is destroyed → null
  // (the contract type is `unknown | null`, so null is the correct sentinel,
  // not undefined; use 'null' behavior not 'noop').
  { method: 'getRestirPtResultBuffer', disposedBehavior: 'null' },

  // Offscreen presentation handoff. The source texture is only valid while the
  // backend is live; after disposal the facade must return the contract's null
  // sentinel without touching destroyed GPU state.
  { method: 'getPresentationSource', disposedBehavior: 'null' },

  // Scene read-back — optional on Engine; all three shipping backends implement
  // it. Disposed → null: the contract says no method except state/capabilities
  // is valid after dispose; the facade gives a uniform null regardless of whether
  // the backend nulls its own scene reference on teardown.
  { method: 'getScene', disposedBehavior: 'null' },

  // Async GPU→CPU pixel readback — optional on Engine; all three shipping
  // backends implement it. Disposed → Promise.resolve(null): no GPU resources
  // remain; matches vanilla.ts's missing-method fallback semantics and the
  // contract's "returns null before the first frame" guarantee.
  { method: 'captureFrame', disposedBehavior: 'promise-null' },

  // Progressive walkaround→PT seed source (P8). Gated on the dedicated
  // capability. Disposed → null (no seed to expose; the engine is torn down).
  {
    method: 'getProgressiveSeedTexture',
    disposedBehavior: 'null',
    eligible: (c) => c.seedSourceAdvertised,
  },

  // Progressive walkaround→PT seed sink (P8). Gated on the dedicated capability.
  // Disposed → noop (matches the other mutating-method disposed semantics).
  {
    method: 'seedAccumulator',
    disposedBehavior: 'noop',
    eligible: (c) => c.accumulatorSeedAdvertised,
  },

  // GI-state export (walkaround-hybrid only). Disposed → Promise.resolve(null):
  // the atlases are torn down; no state to export. The engine is cast to
  // Engine & Partial<GIStatePersistable> at the forwarding site.
  { method: 'exportGIState', disposedBehavior: 'promise-null' },

  // GI-state import (walkaround-hybrid only). Disposed → false: the atlases are
  // torn down; the import is a no-op and returns the "not applied" sentinel.
  { method: 'importGIState', disposedBehavior: 'value-false' },
];

/** Wrap an engine so that calling .dispose() multiple times is a no-op
 *  beyond the first call. The plan calls this out as an explicit
 *  acceptance criterion ("engine.dispose() followed by engine.dispose()
 *  is idempotent").
 *
 *  H31-c — `onDisposeError` is an optional channel for throws that originate
 *  inside `engine.dispose()` or `postDispose()`. Without it those errors are
 *  silently swallowed (the prior behaviour — both catch blocks rethrew nothing).
 *  The channel is intentionally optional: existing callers keep their silent
 *  behaviour by default.
 *
 *  @internal Exported for unit-test access only. Not part of the public
 *  `@vitrum/engine` API surface; consumers should use {@link createEngine}
 *  / {@link attachVitrum}. */
export function wrapWithIdempotentDispose(
  engine: Engine,
  postDispose: () => void,
  onDisposeError?: (err: unknown) => void,
): Engine & Partial<GIStatePersistable> {
  let disposed = false;
  const backendProfileId = (engine as {
    readonly backendProfileId?: 'pt-webgpu' | 'pt-webgpu-lite';
  }).backendProfileId;
  const profileId = (engine as {
    readonly profileId?: 'pt-webgpu' | 'pt-webgpu-lite';
  }).profileId;
  const patchSupport = engine.capabilities.incrementalPatchSupport;
  const primitivePatchAdvertised = patchSupport == null
    ? engine.capabilities.supportsIncrementalScene
    : (
        patchSupport.transform
        || patchSupport.positions
        || patchSupport.material
        || patchSupport.topology
      );
  const emitterPatchAdvertised = patchSupport == null
    ? engine.capabilities.supportsIncrementalScene
    : patchSupport.emitter;
  // Whole-primitive add/remove is gated on the dedicated capability.
  const addRemoveAdvertised = engine.capabilities.supportsAddRemovePrimitive === true;
  const seedSourceAdvertised = engine.capabilities.supportsProgressiveSeedSource === true;
  const accumulatorSeedAdvertised = engine.capabilities.supportsAccumulatorSeed === true;
  const caps: CapabilityGates = {
    primitivePatchAdvertised,
    emitterPatchAdvertised,
    addRemoveAdvertised,
    seedSourceAdvertised,
    accumulatorSeedAdvertised,
  };

  const proxy: Engine = {
    get state() { return engine.state; },
    get capabilities() { return engine.capabilities; },
    setScene(scene) { if (!disposed) engine.setScene(scene); },
    renderFrame(input) {
      if (disposed) {
        // Returning a no-op output keeps host RAF loops from crashing if
        // they race the dispose. The host is expected to stop rendering
        // when state === 'disposed'.
        return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
      }
      return engine.renderFrame(input);
    },
    reset() { if (!disposed) engine.reset(); },
    pause() { if (!disposed) engine.pause(); },
    resume() { if (!disposed) engine.resume(); },
    dispose() {
      if (disposed) return;
      disposed = true;
      try { engine.dispose(); } catch (err) { try { onDisposeError?.(err); } catch { /* callback must not propagate — ignore */ } }
      try { postDispose(); } catch (err) { try { onDisposeError?.(err); } catch { /* callback must not propagate — ignore */ } }
    },
    // T3.G followup — pass the underlying engine.debug surface through
    // unchanged. Methods are bound to the engine instance, so calling
    // proxy.debug.atlasTexture() reads live state. After dispose, the
    // surface still exists but most methods will return null / empty
    // because the underlying _ddgi / _pipeline / _bvhBuffers are torn down.
    ...(engine.debug ? { debug: engine.debug } : {}),
    // Construction identity is immutable metadata, so preserve it across the
    // lifecycle proxy just like state/capabilities. GLTF compatibility routing
    // must use this resolved profile instead of inferring tier from features.
    ...(backendProfileId != null ? { backendProfileId } : {}),
    ...(profileId != null ? { profileId } : {}),
  };

  // Data-driven optional-method forwarding. Each row reproduces the exact
  // disposed-behaviour of the prior hand-coded proxy (see OPTIONAL_METHOD_PROXIES).
  // GI-state methods (exportGIState/importGIState) are not on the Engine interface;
  // both giEngine and proxyWithGI are cast to Engine & Partial<GIStatePersistable>
  // so the table loop can reach them without type errors.
  type EngineWithGI = Engine & Partial<GIStatePersistable>;
  const giEngine = engine as EngineWithGI;
  const proxyWithGI = proxy as EngineWithGI;
  for (const spec of OPTIONAL_METHOD_PROXIES) {
    const impl = (giEngine as Record<OptionalMethodName, unknown>)[spec.method];
    if (impl == null) continue;
    if (spec.eligible && !spec.eligible(caps)) continue;
    (proxyWithGI as Record<OptionalMethodName, unknown>)[spec.method] =
      makeForward(giEngine, spec, () => disposed);
  }

  return proxy;
}

/** Build a forwarding wrapper for one optional method that reproduces the
 *  given disposed-behaviour. Variadic args are passed through verbatim.
 *  The engine parameter is `Engine & Partial<GIStatePersistable>` so the
 *  table-loop can forward GI-state methods (not on the Engine interface). */
function makeForward(
  engine: Engine & Partial<GIStatePersistable>,
  spec: OptionalMethodProxy,
  isDisposed: () => boolean,
): (...args: unknown[]) => unknown {
  const impl = (engine as Record<OptionalMethodName, unknown>)[spec.method] as (...args: unknown[]) => unknown;
  switch (spec.disposedBehavior) {
    case 'noop':
      return (...args: unknown[]) => {
        if (isDisposed()) return undefined;
        return impl.apply(engine, args);
      };
    case 'null':
      return (...args: unknown[]) => {
        if (isDisposed()) return null;
        return impl.apply(engine, args);
      };
    case 'promise-null':
      return (...args: unknown[]) => {
        if (isDisposed()) return Promise.resolve(null);
        return impl.apply(engine, args);
      };
    case 'value-false':
      return (...args: unknown[]) => {
        if (isDisposed()) return false;
        return impl.apply(engine, args);
      };
    case 'empty-unsub':
      return (...args: unknown[]) => {
        if (isDisposed()) return () => {};
        return impl.apply(engine, args);
      };
    case 'throw':
      return (...args: unknown[]) => {
        if (isDisposed()) {
          throw new Error(spec.throwMessage ?? `${spec.method}: engine is disposed`);
        }
        return impl.apply(engine, args);
      };
  }
}
