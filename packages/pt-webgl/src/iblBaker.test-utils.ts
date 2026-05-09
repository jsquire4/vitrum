/**
 * iblBaker.test-utils — FOR UNIT TESTS ONLY.
 *
 * Production code must never import from this file. Only `*.test.ts` /
 * `*.spec.ts` files should reference it. Any lint pre-commit hook should
 * treat imports of this module outside test files as a tripwire.
 *
 * Re-exports `_skyEquirectCacheSize` under the conventional `__*ForTests`
 * name so the test-only intent is clear at every call site, mirroring the
 * pattern in `@vitrum/core`'s `gpuDetection.test-utils.ts`.
 */

import { _skyEquirectCacheSize } from './iblBaker.js';

/**
 * Return the current number of baked sky equirect entries in the LRU cache.
 * Call in tests to verify cache eviction behaviour without reaching into
 * internal module state directly.
 *
 * Do NOT call from production code — cache internals are subject to change.
 */
export function __skyEquirectCacheSizeForTests(): number {
  return _skyEquirectCacheSize();
}
