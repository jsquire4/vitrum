// Standalone home for the `PTEngineWebGL2QualityMode` type alias.
//
// Lifted out of `ptEngineWebGL2.ts` (Task 4.4 Theme A) so the extracted
// `AdaptiveScheduler` can reference the quality-mode union WITHOUT importing the
// engine module (which would create a cycle: engine → adaptiveScheduler →
// engine). The engine re-exports this type so the public surface is unchanged.

/** Render quality preset selecting the scheduler's batch budget, tile size,
 *  and adaptive cadence defaults. */
export type PTEngineWebGL2QualityMode = 'interactive' | 'final' | 'capture' | 'safe';
