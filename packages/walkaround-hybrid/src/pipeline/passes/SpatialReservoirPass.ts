/**
 * SpatialReservoirPass — ReSTIR-DI spatial reuse (1 or 2 ping-pong passes).
 *
 * Owns up to both `spatial-1` and `spatial-2` labels: emits one compute
 * dispatch per label with the shared frame/scene/ubo bind groups. NEIGHBORS=5
 * (compile-time const, intentionally fixed for Phase 0); the spatial reuse is
 * the dominant variance reducer in the pipeline.
 *
 * Phase-0 productization — the ping-pong PASS COUNT is host/preset-driven via
 * the constructor `passCount` arg (the spatial NEIGHBOR count stays fixed at 5).
 * `passCount: 2` (ultra/high) is the full-fidelity variance reducer; `1`
 * (medium/low) halves the spatial cost. The terminal label `spatial-2` is kept
 * for BOTH counts so the `shade` dependency + the `id` stay stable; a 1-pass
 * config emits only `['spatial-2']` (a single dispatch).
 *
 * The shared dispatch body ({@link SharedBindGroupPass}) loops over
 * `passLabels`, so the variable-dispatch behaviour falls out of the base class
 * declaring the chosen labels — no per-pass dispatch override needed.
 *
 * R2 — `buildPassLayout` MUST be built with the same `diSpatialPasses` so the
 * timestamp slot layout matches the labels emitted here (asserted by the
 * pass-layout parity test).
 */

import { SharedBindGroupPass } from '../Pass.js';
import { diSpatialPassLabels } from './passOrder.js';
import type { PassLabel } from '../timestampQueries.js';

export class SpatialReservoirPass extends SharedBindGroupPass {
  readonly id = 'spatial-2' as const; // last dispatch id — shade depends on this.
  readonly dependencies: readonly string[] = ['temporal'];
  readonly passLabels: readonly PassLabel[];

  constructor(pipeline: GPUComputePipeline, passCount: 1 | 2 = 2) {
    super(pipeline);
    this.passLabels = diSpatialPassLabels(passCount);
  }
}
