/**
 * PPGGuidePass — Müller 2017 §3.2/3.4 path-guiding "guide" pass (BEFORE shade).
 *
 * W9 — wires the real flat-buffer traversal kernel. The CPU side (HybridEngine)
 * uploads serialised sTree + dTree buffers each rebuild cycle; this pass binds
 * them and dispatches one workgroup per 64 pixels to write a per-pixel
 * (dir, pdf) guide sample to `ppg.sampleOutBuf`.
 *
 * Gated on `opts.ppgEnabled` — the orchestrator only runs this when both the
 * host opted in AND the PPG pipelines compiled successfully. When PPG is
 * disabled the pass is not registered at all (see WalkaroundGPUPipeline).
 *
 * Bind group layout (from `layout: 'auto'` on the WGSL kernel):
 *   group(0):
 *     binding(0) sTreeBuf     (storage, read)
 *     binding(1) dTreeBuf     (storage, read)
 *     binding(2) dTreeOffsets (storage, read)
 *     binding(3) sampleOut    (storage, read_write)
 *   group(1):
 *     binding(0) guideUboBuffer (uniform)
 */

import type {
  Pass,
  PassDispatchContext,
  PassGateOptions,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class PPGGuidePass implements Pass {
  readonly id = 'ppg-guide' as const;
  /** Runs after gi-spatial-2 (which feeds shade) and before shade — declare
   *  gi-spatial-2 as the dependency so the topo sort places it correctly.
   *  The shade pass independently depends on gi-spatial-2; lex ordering
   *  within the same tier puts ppg-* labels right before `shade`. */
  readonly dependencies: readonly string[] = ['gi-spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['ppg-guide'];

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
    // Contract invariant: PPG resources are allocated whenever the pass is
    // registered (the orchestrator only registers PPG passes when the
    // pipelines compiled AND ppgEnabled=true; allocation runs in the same
    // initialize() branch). If a host bypasses that path, the bind-group
    // build below throws a clear "buffer undefined" error — failing loudly
    // is better than a silent no-op (W9 — no more dispatchWorkgroups(0,0,0)).
    if (!ppg.sTreeBuf || !ppg.dTreeBuf || !ppg.dTreeOffsetsBuf ||
        !ppg.sampleOutBuf || !ppg.guideUboBuffer) {
      throw new Error(
        '[PPG] guide dispatch invariant violated: PPG resources are not allocated. ' +
        'This indicates ppgEnabled=true was claimed but allocatePPGResources was never called.',
      );
    }

    // Build bind groups for both groups from the auto layout.
    const bg0 = device.createBindGroup({
      label: 'ppg-guide-bg0',
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ppg.sTreeBuf } },
        { binding: 1, resource: { buffer: ppg.dTreeBuf } },
        { binding: 2, resource: { buffer: ppg.dTreeOffsetsBuf } },
        { binding: 3, resource: { buffer: ppg.sampleOutBuf } },
      ],
    });
    const bg1 = device.createBindGroup({
      label: 'ppg-guide-bg1',
      layout: this._pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: ppg.guideUboBuffer } }],
    });

    const pixelCount = width * height;
    // Workgroup size in the WGSL is 64 (1-D); ceil-div for partial tail.
    const wgCount = Math.max(1, Math.ceil(pixelCount / 64));

    const pass = encoder.beginComputePass(computeDesc('ppg-guide'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.dispatchWorkgroups(wgCount, 1, 1);
    pass.end();
  }

  dispose(): void {}
}
