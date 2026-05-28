/**
 * SpatialReservoirPass — ReSTIR-DI spatial reuse (two ping-pong passes for fidelity).
 *
 * Owns both `spatial-1` and `spatial-2` labels: emits two compute dispatches
 * with the shared frame/scene/ubo bind groups. NEIGHBORS=5; the visual win
 * is the dominant variance reducer in the pipeline.
 *
 * The shared dispatch body ({@link SharedBindGroupPass}) loops over
 * `passLabels`, so the two-dispatch behaviour falls out of the base class
 * declaring both labels — no per-pass dispatch override needed.
 */

import { SharedBindGroupPass } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class SpatialReservoirPass extends SharedBindGroupPass {
  readonly id = 'spatial-2' as const; // last dispatch id — shade depends on this.
  readonly dependencies: readonly string[] = ['temporal'];
  readonly passLabels: readonly PassLabel[] = ['spatial-1', 'spatial-2'];
}
