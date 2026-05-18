/**
 * PPGGuidePass — Müller 2017 §3.2/3.4 path-guiding "guide" pass (BEFORE shade).
 *
 * Currently a no-op stub (`dispatchWorkgroups(0,0,0)`). The full bind-group
 * wiring (ppgLeafFlux, ppgLeafSolidAng, ppgTotalFlux, ppgSampleOut) requires
 * the serialised sTree/dTree buffers from the CPU-side PPGModelHandle; the
 * pipeline is compiled and reserved here until that wiring lands.
 *
 * Gated on `opts.ppgEnabled` — the orchestrator only runs this when both the
 * host opted in AND the PPG pipelines compiled successfully.
 */

import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassGateOptions } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class PPGGuidePass implements Pass {
  readonly id = 'ppg-guide' as const;
  /** Runs after gi-spatial-2 (which feeds shade) and before shade — declare
   *  gi-spatial-2 as the dependency so the topo sort places it correctly.
   *  The shade pass independently depends on gi-spatial-2; lex ordering
   *  within the same tier puts ppg-* labels right before `shade`. */
  readonly dependencies: readonly string[] = ['gi-spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['ppg-guide'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(opts: PassGateOptions): boolean {
    return opts.ppgEnabled;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { encoder, computeDesc } = ctx;
    const pass = encoder.beginComputePass(computeDesc('ppg-guide'));
    pass.setPipeline(this._pipeline);
    pass.dispatchWorkgroups(0, 0, 0); // no-op stub until sTree GPU buffer is wired
    pass.end();
  }

  dispose(): void {}
}
