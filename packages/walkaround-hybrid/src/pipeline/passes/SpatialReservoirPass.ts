/**
 * SpatialReservoirPass — ReSTIR-DI spatial reuse (two ping-pong passes for fidelity).
 *
 * Owns both `spatial-1` and `spatial-2` labels: emits two compute dispatches
 * with the shared frame/scene/ubo bind groups. NEIGHBORS=5; the visual win
 * is the dominant variance reducer in the pipeline.
 */

import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class SpatialReservoirPass implements Pass {
  readonly id = 'spatial-2' as const; // last dispatch id — shade depends on this.
  readonly dependencies: readonly string[] = ['temporal'];
  readonly passLabels: readonly PassLabel[] = ['spatial-1', 'spatial-2'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { encoder, computeDesc, frameBindGroup, sceneBindGroup, uboBindGroup, wgX, wgY } = ctx;
    for (const label of this.passLabels) {
      const pass = encoder.beginComputePass(computeDesc(label));
      pass.setPipeline(this._pipeline);
      pass.setBindGroup(0, frameBindGroup);
      pass.setBindGroup(1, sceneBindGroup);
      pass.setBindGroup(2, uboBindGroup);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }
  }

  dispose(): void {}
}
