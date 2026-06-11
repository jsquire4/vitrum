// giState.ts — GI-state persistence interface for the @vitrum/core contract.
//
// Defines the generic GIStatePersistable<TSnapshot> interface so the engine
// facade (packages/engine) can declare and proxy the export/import surface
// WITHOUT importing from @vitrum/walkaround-hybrid (which would create a
// package-level dependency inversion: core ← engine ← walkaround-hybrid,
// but core must NOT import walkaround-hybrid).
//
// The concrete GIStateSnapshot type lives in @vitrum/walkaround-hybrid. The
// facade narrows to it by supplying the type argument explicitly:
//   `Engine & Partial<GIStatePersistable<GIStateSnapshot>>`
//
// Callers that only need the storage/restore contract — without caring about
// the concrete backend snapshot shape — can use the unbound default
// (`GIStatePersistable` or `GIStatePersistable<unknown>`).
//
// TypeScript method declarations (not arrow-function properties) are bivariant
// in their parameter types regardless of `strictFunctionTypes`. Declaring the
// methods here with method syntax (not function-property syntax) therefore
// ensures that `GIStatePersistable<GIStateSnapshot>` is assignable to
// `GIStatePersistable<unknown>`, which lets `createEngine.ts` continue to use
// the unbound form while the facade narrows to the concrete snapshot type.
//
// DESIGN: we do NOT re-export GIStateSnapshot from core. The import direction
// must remain core ← engine ← walkaround-hybrid. Core only exports the
// generic capability interface; the facade supplies the concrete type argument.

/**
 * GI-state persistence surface — the backend-agnostic contract for baked
 * global-illumination state export/import.
 *
 * `TSnapshot` is the concrete snapshot type produced and consumed by the
 * implementing backend.  The engine facade ({@link @vitrum/engine}) exposes
 * this with `TSnapshot = GIStateSnapshot` (from `@vitrum/walkaround-hybrid`).
 * Hosts working at the contract level can omit the type argument (defaulting
 * to `unknown`) when the concrete snapshot shape does not matter — only the
 * round-trip storage / restore semantics do.
 *
 * **Method declaration form (bivariant parameter checking):** the methods are
 * declared as TypeScript method declarations (not function-property
 * assignments), so they use bivariant parameter checking regardless of
 * `strictFunctionTypes`. This means `GIStatePersistable<GIStateSnapshot>` is
 * safely assignable to `GIStatePersistable<unknown>`, keeping the facade and
 * engine-package types compatible without `any`.
 *
 * **Implemented by:** `@vitrum/walkaround-hybrid` (DDGI probe-atlas
 * export/import; see `HybridEngine.exportGIState`).  Backends with no GI
 * state simply do not implement this interface; the facade marks the methods
 * optional on its return type.
 *
 * **Forwarding semantics (after dispose):**
 *  - `exportGIState` → `Promise.resolve(null)` (atlases are torn down).
 *  - `importGIState` → `false` (engine is torn down; import is a no-op
 *    and returns the "not applied" sentinel).
 */
export interface GIStatePersistable<TSnapshot = unknown> {
  /** Export the engine's current baked GI state as a serialisable snapshot.
   *  Returns `null` when the engine has not yet converged or when GI state
   *  is unavailable.  The caller may serialise the snapshot via the backend
   *  package's `serializeGIState` helper and store it (e.g. in IndexedDB). */
  exportGIState(): Promise<TSnapshot | null>;

  /** Restore a previously exported GI state.  Returns `true` when the
   *  snapshot was accepted and applied, `false` otherwise (incompatible
   *  dimensions, version mismatch, or engine disposed).  The import is
   *  synchronous: the GPU atlases are overwritten in the same call; the
   *  next `renderFrame` renders from the restored state. */
  importGIState(snapshot: TSnapshot): boolean;
}
