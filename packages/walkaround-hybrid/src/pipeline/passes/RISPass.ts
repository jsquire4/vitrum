/**
 * RISPass — ReSTIR-DI primary-ray-cast + initial candidate sampling.
 *
 * Casts primary rays through the BVH, samples emitter candidates via
 * importance sampling, and writes the current-frame reservoir. Uses the
 * shared frame/scene/ubo bind groups; no pass-private bind groups.
 */

import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class RISPass implements Pass {
  readonly id = 'ris' as const;
  readonly dependencies: readonly string[] = ['sample-budget'];
  readonly passLabels: readonly PassLabel[] = ['ris'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { encoder, computeDesc, frameBindGroup, sceneBindGroup, uboBindGroup, wgX, wgY } = ctx;
    const pass = encoder.beginComputePass(computeDesc('ris'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {}
}
