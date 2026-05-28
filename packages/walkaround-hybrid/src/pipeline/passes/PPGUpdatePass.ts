/**
 * PPGUpdatePass — Müller 2017 §3.3 path-guiding "update" pass (AFTER shade).
 *
 * W9 — wires the real flat-buffer leaf-locator kernel. Reads per-pixel training
 * samples `(pos, dir, Li)` from PPG sample buffers and atomically increments
 * the dTree leaf flux counter for each. The CPU reads back the atomic buffer
 * at the end of each rebuild cycle and calls `splitOverflowLeaves` +
 * `refineDTree` to adapt the tree (topology changes are CPU-side per §5).
 *
 * Gated on `opts.ppgEnabled`.
 *
 * Bind group layout (from `layout: 'auto'` on the WGSL kernel):
 *   group(0):
 *     binding(0) samplesPosBuf  (storage, read)
 *     binding(1) samplesDirBuf  (storage, read)
 *     binding(2) samplesLiBuf   (storage, read)   ← DEVIATION 3 — L_i binding
 *     binding(3) fluxAtomicsBuf (storage, read_write)
 *     binding(4) sTreeBuf       (storage, read)
 *     binding(5) dTreeBuf       (storage, read)
 *     binding(6) dTreeOffsets   (storage, read)
 *   group(1):
 *     binding(0) updateUboBuffer (uniform)
 */

import { buildPpgUpdateBindGroups } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassGateOptions,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class PPGUpdatePass implements Pass {
  readonly id = 'ppg-update' as const;
  readonly dependencies: readonly string[] = ['shade'];
  readonly passLabels: readonly PassLabel[] = ['ppg-update'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(opts: PassGateOptions): boolean {
    return opts.ppgEnabled;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, resources, width, height } = ctx;
    const ppg = resources.ppg;
    // Contract invariant — see PPGGuidePass for the same rationale.
    if (!ppg.sTreeBuf || !ppg.dTreeBuf || !ppg.dTreeOffsetsBuf ||
        !ppg.fluxAtomicsBuf || !ppg.samplesPosBuf || !ppg.samplesDirBuf ||
        !ppg.samplesLiBuf || !ppg.updateUboBuffer) {
      throw new Error(
        '[PPG] update dispatch invariant violated: PPG resources are not allocated. ' +
        'This indicates ppgEnabled=true was claimed but allocatePPGResources was never called.',
      );
    }

    const [bg0, bg1] = buildPpgUpdateBindGroups(
      device,
      (i) => this._pipeline.getBindGroupLayout(i),
      {
        samplesPosBuf: ppg.samplesPosBuf,
        samplesDirBuf: ppg.samplesDirBuf,
        samplesLiBuf: ppg.samplesLiBuf,
        fluxAtomicsBuf: ppg.fluxAtomicsBuf,
        sTreeBuf: ppg.sTreeBuf,
        dTreeBuf: ppg.dTreeBuf,
        dTreeOffsetsBuf: ppg.dTreeOffsetsBuf,
        updateUboBuffer: ppg.updateUboBuffer,
      },
    );

    const sampleCount = width * height;
    const wgCount = Math.max(1, Math.ceil(sampleCount / 64));

    const pass = encoder.beginComputePass(computeDesc('ppg-update'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.dispatchWorkgroups(wgCount, 1, 1);
    pass.end();
  }

  dispose(): void {}
}
