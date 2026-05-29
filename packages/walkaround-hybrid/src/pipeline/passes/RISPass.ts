/**
 * RISPass — ReSTIR-DI primary-ray-cast + initial candidate sampling.
 *
 * Casts primary rays through the BVH, samples emitter candidates via
 * importance sampling, and writes the current-frame reservoir. Uses the
 * shared frame/scene/ubo bind groups, PLUS a RIS-only group(3) carrying the
 * light-tree storage buffer for spatially-aware DI light SELECTION. The
 * light-tree group is bound only here (not on temporal/spatial/shade) so the
 * extra storage buffer lands on the RIS pipeline layout alone — keeping the
 * heavier shade pass at the `maxStorageBuffersPerShaderStage = 16` floor.
 *
 * The kernel reads `ubo.lightTreeEnabled` at runtime: when `0` it ignores the
 * group(3) buffer entirely (flat power-CDF path), so the group is always bound
 * (a 1-node placeholder backs it when disabled) without a pipeline recompile.
 */

import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class RISPass implements Pass {
  readonly id = 'ris' as const;
  readonly dependencies: readonly string[] = ['sample-budget'];
  readonly passLabels: readonly PassLabel[] = ['ris'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const {
      encoder, computeDesc,
      frameBindGroup, sceneBindGroup, uboBindGroup, lightTreeBindGroup,
      wgX, wgY,
    } = ctx;
    const pass = encoder.beginComputePass(computeDesc('ris'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.setBindGroup(3, lightTreeBindGroup); // RIS-only DI light-selection tree
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {}
}
