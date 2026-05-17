/**
 * TemporalReservoirPass — ReSTIR-DI temporal reuse (merge with previous-frame reservoir).
 *
 * Reuses the shared frame/scene/ubo bind groups; no pass-private state.
 */

import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class TemporalReservoirPass implements Pass {
  readonly id = 'temporal' as const;
  readonly dependencies: readonly string[] = ['ris'];
  readonly passLabels: readonly PassLabel[] = ['temporal'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { encoder, computeDesc, frameBindGroup, sceneBindGroup, uboBindGroup, wgX, wgY } = ctx;
    const pass = encoder.beginComputePass(computeDesc('temporal'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {}
}
