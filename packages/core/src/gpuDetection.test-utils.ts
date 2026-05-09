/**
 * gpuDetection.test-utils — FOR UNIT TESTS ONLY.
 *
 * Production code must never import from this file. Only `*.test.ts` /
 * `*.spec.ts` files should reference it. Any lint pre-commit hook should
 * treat imports of this module outside test files as a tripwire.
 *
 * The reset function is defined in `gpuDetection.ts` (where `cached` lives)
 * and re-exported here under the conventional `__*ForTests` name so the
 * test-only intent is clear at every call site.
 */

import { _resetCacheUnsafe } from './gpuDetection.js';

/**
 * Reset the memoized detection cache + clear `window.__WG__`.
 * Call in `afterEach` / `beforeEach` to prevent cross-test state leakage.
 */
export function __resetGpuDetectionForTests(): void {
  _resetCacheUnsafe();
}
