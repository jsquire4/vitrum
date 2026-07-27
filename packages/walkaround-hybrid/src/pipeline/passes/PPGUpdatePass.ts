/**
 * PPGUpdatePass — Müller 2017 §3.3 path-guiding "update" pass (immediately after initial GI RIS).
 *
 * Reads ReSTIR-GI initial reservoirs `(xv, normalize(xs - xv), w_sum, M)` and deposits
 * the unbiased arbitrary-bin histogram estimator `w_sum / M`. It must execute
 * before GI temporal/spatial reuse mutates that reservoir.
 * atomic buffer at the end of each rebuild cycle and calls `splitOverflowLeaves` +
 * `refineDTree` to adapt the tree (topology changes are CPU-side per §5).
 *
 * Gated on `opts.ppgEnabled`.
 *
 * Bind group layout (from `layout: 'auto'` on the WGSL kernel):
 *   group(0):
 *     binding(0) reservoirGiCurrentBuffer (storage, read)
 *     binding(1) fluxAtomicsBuf           (storage, read_write)
 *     binding(2) queryArenaBuf             (storage, read)
 *       versioned header + sTree + dTrees + dTree offsets
 *     binding(3) cellSampleCountsBuf       (storage, read_write)  [A2]
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
  readonly dependencies: readonly string[] = ['gi-ris'];
  readonly passLabels: readonly PassLabel[] = ['ppg-update'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(opts: PassGateOptions): boolean {
    // PPG on AND a train frame (`frameCount % ppgDispatchInterval === 0`;
    // absent ⇒ every frame). Skipping the update pass on off-interval frames
    // skips one window of flux accumulation into `fluxAtomicsBuf`; the CPU
    // refine/readback (`PPGCoordinator.maybeRunTrainingRefine`) runs on its own
    // cadence and merely sees fewer accumulated samples — the tree topology and
    // gi-ris guided sampling are unaffected.
    return opts.ppgEnabled && (opts.ppgTrainThisFrame ?? true);
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, resources, width, height } = ctx;
    const ppg = resources.ppg;
    // Contract invariant: PPG resources are allocated whenever this pass is
    // registered. If a host bypasses the initialization branch, fail loudly
    // instead of silently skipping training.
    if (!('queryArenaBuf' in ppg) || !resources.restirGI.reservoirGiCurrentBuffer) {
      throw new Error(
        '[PPG] update dispatch invariant violated: PPG resources are not allocated. ' +
        'This indicates ppgEnabled=true was claimed but allocatePPGResources was never called.',
      );
    }

    const ppgBindGroupResources = {
      reservoirGiCurrentBuffer: resources.restirGI.reservoirGiCurrentBuffer,
      fluxAtomicsBuf: ppg.fluxAtomicsBuf,
      queryArenaBuf: ppg.queryArenaBuf,
      cellSampleCountsBuf: ppg.cellSampleCountsBuf,
      updateUboBuffer: ppg.updateUboBuffer,
    };
    const buildBgs = (): readonly [GPUBindGroup, GPUBindGroup] => buildPpgUpdateBindGroups(
      device,
      (i) => this._pipeline.getBindGroupLayout(i),
      ppgBindGroupResources,
    );
    const bgPair = ctx.resourceCache?.bindGroup('pass:ppg-update', [
      ppgBindGroupResources.reservoirGiCurrentBuffer,
      ppgBindGroupResources.fluxAtomicsBuf,
      ppgBindGroupResources.queryArenaBuf,
      ppgBindGroupResources.cellSampleCountsBuf,
      ppgBindGroupResources.updateUboBuffer,
    ], buildBgs);
    const [bg0, bg1] = bgPair ?? buildBgs();

    const sampleCount = Math.max(1, Math.floor(width / 2)) * Math.max(1, Math.floor(height / 2));
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
