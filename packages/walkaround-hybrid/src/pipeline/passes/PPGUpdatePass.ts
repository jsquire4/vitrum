/**
 * PPGUpdatePass — Müller 2017 §3.3 path-guiding "update" pass (AFTER shade).
 *
 * Same stub shape as {@link PPGGuidePass}: dispatch is a `(0,0,0)` no-op
 * until the sTree/dTree GPU buffers are wired. The CPU reads back the
 * atomic buffer at the end of each rebuild cycle and calls
 * splitOverflowLeaves + refineDTree to adapt the tree.
 *
 * Gated on `opts.ppgEnabled`.
 */

import type { Pass, PassDispatchContext, PassGateOptions, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class PPGUpdatePass implements Pass {
  readonly id = 'ppg-update' as const;
  readonly dependencies: readonly string[] = ['shade'];
  readonly passLabels: readonly PassLabel[] = ['ppg-update'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(opts: PassGateOptions): boolean {
    return opts.ppgEnabled;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { encoder, computeDesc } = ctx;
    const pass = encoder.beginComputePass(computeDesc('ppg-update'));
    pass.setPipeline(this._pipeline);
    pass.dispatchWorkgroups(0, 0, 0); // no-op stub until sTree GPU buffer is wired
    pass.end();
  }

  dispose(): void {}
}
