// @vitrum/dev — shared types for debug overlay components.
//
// As of T3.E + the T3.G followup, FrameStats / ProgressStats / EngineDebugSurface
// all live on @vitrum/core. We re-export them here so existing imports
// (`import type { FrameStats } from '@vitrum/dev'`) stay working without
// the consumer needing to know which package owns the canonical type.

import type { Engine, EngineDebugSurface, FrameStats, ProgressStats } from '@vitrum/core';

export type { EngineDebugSurface, FrameStats, ProgressStats };

/**
 * Engine with the optional T3.E telemetry hooks and T3.G debug surface.
 * Cast a real Engine to this type to access debug APIs — all fields are
 * optional, so it's safe to cast even when the engine doesn't implement them.
 */
export interface DebuggableEngine extends Engine {
  /** Subscribe to per-frame stats (T3.E). Returns an unsubscribe function. */
  onFrame?(cb: (stats: FrameStats) => void): () => void;

  /** Subscribe to long-running progress notifications (T3.E). */
  onProgress?(cb: (progress: ProgressStats) => void): () => void;

  /** Optional debug introspection surface (T3.G followup; HybridEngine
   *  ships an implementation as of 2026-05-12). */
  debug?: EngineDebugSurface;
}
