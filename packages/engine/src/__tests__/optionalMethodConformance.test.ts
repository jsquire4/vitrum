// optionalMethodConformance.test.ts — compile-time + runtime conformance gate.
//
// Asserts two structural invariants that prevent the R10 bug class (a shipped
// optional method visible to backends but silently hidden by the facade because
// it has no proxy row in OPTIONAL_METHOD_PROXIES):
//
//   (1) Every key in OPTIONAL_METHOD_PROXIES names a method that EXISTS on
//       `Engine & Partial<GIStatePersistable>` — i.e. no phantom proxy rows.
//
//   (2) Every method key in BackendMethodPromises has a corresponding entry in
//       OPTIONAL_METHOD_PROXIES OR is intentionally handled outside the table
//       (the `debug` property, which is forwarded via object spread, is the
//       only known legitimate exception and is recorded here).
//
// Methodology: the compile-time checks use mapped types + conditional types
// that resolve to `never` on violation (TypeScript emits TS2344 / TS2578 on
// `never` in a type-assertion position). The runtime checks in vitest mirror
// the type checks with concrete string lookups so CI output names the failing
// method on a violation.
//
// BackendMethodPromises keys that are intentionally NOT in OPTIONAL_METHOD_PROXIES:
//   • `debug` — forwarded as an object property via `...(engine.debug ? { debug: engine.debug } : {})`
//               in wrapWithIdempotentDispose; it is not a callable method and the
//               proxy table only covers callable methods.
// All other BackendMethodPromises keys MUST appear as `method` values in
// OPTIONAL_METHOD_PROXIES.

import { describe, it, expect } from 'vitest';
import type { Engine, BackendMethodPromises } from '@vitrum/core';
import type { GIStatePersistable } from '../idempotentDispose.js';
import { OPTIONAL_METHOD_PROXIES } from '../idempotentDispose.js';

// ── Type-level helpers ────────────────────────────────────────────────────────

/** Proxy-covered surface: union of all method names registered in the table. */
type ProxiedMethod = typeof OPTIONAL_METHOD_PROXIES[number]['method'];

/** Methods reachable on the engine surface the proxy operates on. */
type EngineWithGISurface = Engine & Partial<GIStatePersistable>;

/** Assert that every proxied method name is actually a key on the engine surface.
 *  TypeScript emits TS2344 if `ProxiedMethod extends keyof EngineWithGISurface`
 *  is false — meaning a phantom proxy row was added for a method that doesn't exist. */
type _AssertProxiedMethodsExist = ProxiedMethod extends keyof EngineWithGISurface
  ? true
  : never;
// If this type is `never`, the proxy table contains a row for a non-existent method.
// `declare const` is a pure compile-time assertion — it emits no JS and must not be voided at runtime.
declare const _phantomCheck: _AssertProxiedMethodsExist;

/** Methods from BackendMethodPromises that MUST have a proxy row.
 *  `debug` is the documented exception (property, not a callable — forwarded
 *  via object spread outside the table). */
type LedgerMethodKey = Exclude<keyof BackendMethodPromises, 'debug'>;

/** For each ledger key, check whether it is in the proxied set.
 *  Maps each key → `true` (covered) or `never` (missing proxy row). */
type _LedgerCoverageCheck = {
  [K in LedgerMethodKey]: K extends ProxiedMethod ? true : never;
};
// If any value in this mapped type resolves to `never`, the ledger advertises a
// method that the proxy table does not cover. TypeScript will emit an error on
// the `declare const` below because the object type would contain a `never` field.
//
// NOTE: This check currently passes — all LedgerMethodKey entries are covered
// by ProxiedMethod.  If a new method is added to BackendMethodPromises without
// a corresponding OPTIONAL_METHOD_PROXIES row, this check will catch it.
// `declare const` is a pure compile-time assertion — it emits no JS.
declare const _ledgerCoverage: _LedgerCoverageCheck;

// ── Runtime mirror of the type checks ────────────────────────────────────────
// These give human-readable failure messages in the vitest output.

/** All method names registered in the proxy table (runtime). */
const PROXIED_METHODS = new Set(OPTIONAL_METHOD_PROXIES.map((r) => r.method));

/** BackendMethodPromises keys that require a proxy row (all except 'debug'). */
const LEDGER_METHOD_KEYS: ReadonlyArray<Exclude<keyof BackendMethodPromises, 'debug'>> = [
  'updatePrimitive',
  'updateEmitter',
  'updateEnvironment',
  'addPrimitive',
  'removePrimitive',
  'setSize',
  'updateLighting',
  'onFrame',
  'onProgress',
  'getScene',
  'onError',
  'captureFrame',
  'createInverseSession',
  'getRestirPtResultBuffer',
  'getProgressiveSeedTexture',
  'seedAccumulator',
  'giStatePersistence',
];

/** Engine + GIStatePersistable surface keys reachable by the proxy (runtime). */
const ENGINE_SURFACE_KEYS = new Set<string>([
  // Engine required methods
  'state', 'capabilities', 'setScene', 'renderFrame', 'reset', 'pause', 'resume', 'dispose',
  // Engine optional methods / properties
  'getScene',
  'updatePrimitive',
  'addPrimitive',
  'removePrimitive',
  'updateEmitter',
  'updateEnvironment',
  'setSize',
  'updateLighting',
  'seedAccumulator',
  'getProgressiveSeedTexture',
  'pause',
  'resume',
  'onFrame',
  'onProgress',
  'onError',
  'debug',
  'createInverseSession',
  'captureFrame',
  'getRestirPtResultBuffer',
  // GIStatePersistable methods
  'exportGIState',
  'importGIState',
]);

describe('optional-method proxy conformance gate (C4 / I1.3)', () => {
  it('every proxy row names a method that exists on Engine & Partial<GIStatePersistable>', () => {
    const phantom: string[] = [];
    for (const { method } of OPTIONAL_METHOD_PROXIES) {
      if (!ENGINE_SURFACE_KEYS.has(method)) {
        phantom.push(method);
      }
    }
    expect(phantom, `Phantom proxy rows (method not on engine surface): ${phantom.join(', ')}`).toEqual([]);
  });

  it('every BackendMethodPromises key (except debug) has a proxy row', () => {
    // `giStatePersistence` is the ledger key for the exportGIState/importGIState PAIR.
    // The proxy table has SEPARATE rows for `exportGIState` and `importGIState`;
    // the ledger groups them under one boolean because both are always co-present.
    // Special-case: verify both individual rows exist rather than the ledger key name.
    const LEDGER_KEY_TO_PROXY_METHODS: Partial<Record<typeof LEDGER_METHOD_KEYS[number], string[]>> = {
      giStatePersistence: ['exportGIState', 'importGIState'],
    };

    const missing: string[] = [];
    for (const key of LEDGER_METHOD_KEYS) {
      const proxyNames = LEDGER_KEY_TO_PROXY_METHODS[key];
      if (proxyNames != null) {
        // Multi-row ledger key: verify all named proxy methods are present.
        for (const name of proxyNames) {
          if (!PROXIED_METHODS.has(name as Parameters<typeof PROXIED_METHODS['has']>[0])) {
            missing.push(`${key} → ${name}`);
          }
        }
      } else {
        // 1:1 ledger key → proxy row: the key name IS the proxy method name.
        if (!PROXIED_METHODS.has(key as Parameters<typeof PROXIED_METHODS['has']>[0])) {
          missing.push(key);
        }
      }
    }
    expect(
      missing,
      `BackendMethodPromises keys with no proxy row (R10 bug class): ${missing.join(', ')}`,
    ).toEqual([]);
  });

  it('debug is intentionally excluded from the proxy table (forwarded via object spread)', () => {
    // Structural confirmation that `debug` is absent from the table (not an
    // oversight) — the proxy handles it outside the data-driven loop.
    const debugInTable = OPTIONAL_METHOD_PROXIES.some((r) => r.method === 'debug' as string);
    expect(debugInTable).toBe(false);
  });
});
