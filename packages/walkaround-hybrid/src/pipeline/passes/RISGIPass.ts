/**
 * RISGIPass — Sprint 16 ReSTIR-GI RIS pass.
 *
 * Half-resolution dispatch (W/2 × H/2). Reuses the shared frame/scene/ubo
 * bind groups + the hybrid-layers (DDGI) bind group at slot 3.
 *
 * Runs after the DI spatial passes (consumes the spatially-fused DI
 * reservoir from `bgFrame`) so the GI reservoir is built on the
 * variance-reduced primary visibility.
 */

import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class RISGIPass implements Pass {
  readonly id = 'gi-ris' as const;
  readonly dependencies: readonly string[] = ['spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['gi-ris'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const {
      encoder,
      computeDesc,
      frameBindGroup,
      sceneBindGroup,
      uboBindGroup,
      hybridLayersBindGroup,
      halfWgX,
      halfWgY,
    } = ctx;
    const pass = encoder.beginComputePass(computeDesc('gi-ris'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.setBindGroup(3, hybridLayersBindGroup);
    pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
    pass.end();
  }

  dispose(): void {}
}
