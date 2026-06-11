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

import type { Engine } from '@vitrum/core';
import type { GIStateSnapshot } from '@vitrum/walkaround-hybrid';

/**
 * Optional GI-state persistence surface — implemented by the walkaround-hybrid
 * backend only (DDGI probe-atlas export/import; see `HybridEngine.exportGIState`).
 * Forwarded through the facade so `createEngine()` users can reach the shipped
 * feature without dropping to the concrete HybridEngine. Backends with no GI
 * state simply don't provide these (the facade return type marks them optional).
 */
export interface GIStatePersistable {
  exportGIState(): Promise<GIStateSnapshot | null>;
  importGIState(snapshot: GIStateSnapshot): boolean;
}

/**
 * Disposed-behaviour kinds for an optional Engine method, captured as data.
 *
 *   • 'noop'        — when disposed, swallow the call and return undefined.
 *                     (updatePrimitive/updateEmitter/addPrimitive/removePrimitive/
 *                      updateEnvironment/setSize/updateLighting)
 *   • 'null'        — when disposed, return null (for methods whose contract
 *                     return type is `T | null` — distinct from noop whose
 *                     return is `undefined`).
 *                     (getRestirPtResultBuffer)
 *   • 'empty-unsub' — the method returns an unsubscribe fn; when disposed, return
 *                     a no-op unsubscribe `() => {}` without forwarding.
 *                     (onFrame/onProgress)
 *   • 'throw'       — when disposed, throw (the engine is torn down; refuse).
 *                     (createInverseSession)
 */
type DisposedBehavior = 'noop' | 'null' | 'empty-unsub' | 'throw';

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
  | 'createInverseSession'
  | 'getRestirPtResultBuffer';

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
}

// The table reproduces EXACTLY the per-method disposed-behaviour of the prior
// hand-coded proxy. Each row is behaviour-preserving — see the kind docs above.
const OPTIONAL_METHOD_PROXIES: readonly OptionalMethodProxy[] = [
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
  // WS5 — inverse-rendering (differentiable RT) sessions. After dispose the
  // proxy refuses to open a NEW session (the engine is torn down); an
  // already-open session the host holds keeps working until the host disposes
  // it. Sessions outlive a single frame but not the engine.
  {
    method: 'createInverseSession',
    disposedBehavior: 'throw',
    throwMessage: 'createInverseSession: engine is disposed',
  },
  // H61 — debug/experimental accessor for the ReSTIR-PT reuse output buffer
  // (pt-webgpu, gated by the 'pt-webgpu-restir-pt-reuse' experimental feature).
  // Added to the Engine contract in H14-C; without this row the createEngine
  // facade silently hid it. After dispose the buffer is destroyed → null
  // (the contract type is `unknown | null`, so null is the correct sentinel,
  // not undefined; use 'null' behavior not 'noop').
  { method: 'getRestirPtResultBuffer', disposedBehavior: 'null' },
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
  const caps: CapabilityGates = {
    primitivePatchAdvertised,
    emitterPatchAdvertised,
    addRemoveAdvertised,
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
  };

  // Data-driven optional-method forwarding. Each row reproduces the exact
  // disposed-behaviour of the prior hand-coded proxy (see OPTIONAL_METHOD_PROXIES).
  for (const spec of OPTIONAL_METHOD_PROXIES) {
    const impl = engine[spec.method];
    if (impl == null) continue;
    if (spec.eligible && !spec.eligible(caps)) continue;
    (proxy as Record<OptionalMethodName, unknown>)[spec.method] =
      makeForward(engine, spec, () => disposed);
  }

  // GI-state persistence (walkaround-hybrid only) — forwarded when the backend
  // implements it, with dispose-safe fallbacks that match the methods' own no-op
  // semantics (export → null, import → false when atlases aren't available).
  const giEngine = engine as Engine & Partial<GIStatePersistable>;
  if (typeof giEngine.exportGIState === 'function') {
    (proxy as Partial<GIStatePersistable>).exportGIState = () =>
      disposed ? Promise.resolve(null) : giEngine.exportGIState!();
  }
  if (typeof giEngine.importGIState === 'function') {
    (proxy as Partial<GIStatePersistable>).importGIState = (snapshot) =>
      disposed ? false : giEngine.importGIState!(snapshot);
  }

  // Scene read-back (`getScene`). A value-returning read forwarded here (not via
  // the OPTIONAL_METHOD_PROXIES table, which only models noop/empty-unsub/throw)
  // so a host can reach the backend's retained canonical `Scene` through the
  // wrapped engine instead of shadowing scene state. Disposed → null: the
  // contract says no method except state/capabilities is valid after dispose, so
  // the facade gives a uniform null regardless of whether the backend nulls its
  // own scene reference on teardown (pt-webgpu) or keeps it (pt-webgl2/hybrid).
  const sceneEngine = engine;
  if (typeof sceneEngine.getScene === 'function') {
    proxy.getScene = () => (disposed ? null : sceneEngine.getScene!());
  }

  // captureFrame — async GPU→CPU pixel readback. Forwarded here (not via the
  // OPTIONAL_METHOD_PROXIES table, which only models synchronous disposed-behaviours)
  // because the return type is `Promise<CapturedFrame | null>`. Disposed →
  // `Promise.resolve(null)` (no GPU resources remain; matches vanilla.ts's
  // missing-method fallback semantics and the contract's "returns null before the
  // first frame" guarantee). Without this forwarding `engine.captureFrame` is
  // always undefined through the facade, so vanilla.ts:593 returns
  // `Promise.resolve(null)` on every call — readback is dead.
  if (typeof engine.captureFrame === 'function') {
    proxy.captureFrame = (opts) =>
      disposed ? Promise.resolve(null) : engine.captureFrame!(opts);
  }

  // Progressive walkaround→PT seed source/sink (P8). These value-returning /
  // bespoke-disposed-semantics methods are forwarded here (not via the
  // OPTIONAL_METHOD_PROXIES table, which only models noop/empty-unsub/throw) so
  // a host driving the handoff over `createEngine`-wrapped engines — most
  // importantly `createProgressiveEngine`, which builds its coordinator over
  // these wrapped engines — can actually reach them. WITHOUT this forwarding the
  // ProgressiveHandoffCoordinator's `realtime.getProgressiveSeedTexture?.()` /
  // `converged.seedAccumulator?.()` resolve to undefined and the seed silently
  // no-ops (the two arms become byte-identical). Each is gated on its capability
  // so a backend that doesn't advertise it stays unforwarded.
  const seedEngine = engine;
  if (
    engine.capabilities.supportsProgressiveSeedSource === true &&
    typeof seedEngine.getProgressiveSeedTexture === 'function'
  ) {
    proxy.getProgressiveSeedTexture = () =>
      // Disposed → null (the engine is torn down; there is no seed to expose).
      disposed ? null : seedEngine.getProgressiveSeedTexture!();
  }
  if (
    engine.capabilities.supportsAccumulatorSeed === true &&
    typeof seedEngine.seedAccumulator === 'function'
  ) {
    proxy.seedAccumulator = (seed, opts) => {
      // Disposed → no-op (matches the other mutating-method disposed semantics).
      if (disposed) return;
      seedEngine.seedAccumulator!(seed, opts);
    };
  }

  return proxy;
}

/** Build a forwarding wrapper for one optional method that reproduces the
 *  given disposed-behaviour. Variadic args are passed through verbatim. */
function makeForward(
  engine: Engine,
  spec: OptionalMethodProxy,
  isDisposed: () => boolean,
): (...args: unknown[]) => unknown {
  const impl = engine[spec.method] as (...args: unknown[]) => unknown;
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
