/**
 * Neutral WGSL module type — shared across subsystem files that need
 * `WgslModule` without a dependency on the pipeline/ orchestration layer.
 *
 * Extracted from `pipeline/wgslComposer.ts` (R6 E sweep, I5.1, 2026-06-11):
 * the type is a pure data shape with no pipeline logic, so subsystem files
 * (ppg/*.wgsl.ts, ddgi/ddgiSampleWgsl.ts, shaders/*.wgsl.ts) can import it
 * here rather than reaching into pipeline/. `wgslComposer` re-exports the type
 * from this file for back-compatibility with any existing callers.
 */

export interface WgslModule {
  /** Stable identifier — referenced by other modules' `requires` arrays. */
  readonly name: string;
  /** Raw WGSL source for this module (without its own deps prepended). */
  readonly source: string;
  /** Names of modules this module depends on, in the order they should be
   *  emitted relative to each other when reached via this module. */
  readonly requires: readonly string[];
}
