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
 *
 * Checkerboard sparse-RIS (host opt-in; default OFF): RIS SEEDS the per-pixel
 * reservoir — the primary BVH cast + the M_LIGHT=64 emitter-candidate loop, the
 * pipeline's most expensive initial-candidate stage (~19% of the walkaround
 * frame, dzn RTX-4090). When checkerboard is ON this pass COMPACTS the dispatch
 * to ~half the threads — `ceil(ceil(W/2)/8) × ceil(H/8)` workgroups — so the
 * primary cast + candidate sampling genuinely run for ONE pixel per active-parity
 * slot instead of every pixel (compaction, NOT a shader early-return — an
 * early-returned thread still occupies its warp and saves no BVH traversal). The
 * RIS shader decodes the compacted global_invocation_id back into the true
 * full-res active-parity pixel `px = gid.x*2 + ((gid.y + frameParity)&1)`,
 * `py = gid.y`, EXACTLY the (px+py)&1 == frameParity set ShadePass shades +
 * SpatialReservoirPass refines this frame.
 *
 * Gap-parity correctness: RIS does NOT write the gap-parity reservoir slots, so
 * each retains the carried-forward reservoir RIS wrote when that pixel was last
 * active (the parity flips each frame). The FULL-RATE temporal pass — which must
 * STAY full-rate — reads that carried-forward reservoir as its `cur` and keeps
 * combining it with the reprojected previous-frame history, so every pixel has a
 * VALID reservoir for temporal/spatial/shade to consume. The net effect for a
 * gap pixel is one missed fresh candidate that frame (an effectively half-rate
 * candidate cadence reconstructed by the temporal pass + the denoiser). This
 * mirrors {@link ShadePass} / {@link SpatialReservoirPass} so the active-parity
 * decode + the motion fallback are shared across all three compacted passes. OFF
 * (default, OR a fast-motion frame — see WalkaroundGPUPipeline motion fallback)
 * ⇒ the full-res `wgX/wgY` dispatch, byte-identical to before.
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
    // Checkerboard ON — compact the X dispatch to the active-parity columns.
    // Each row has at most ceil(W/2) active-parity pixels; the shader decodes
    // compacted gid.x -> px = gid.x*2 + ((gid.y + frameParity)&1) and the few
    // overshoot threads (px >= W) hit the existing bounds guard. Y stays
    // full-res (one compacted thread per row). 8x8 workgroup matches ris.wgsl
    // @workgroup_size(8,8,1) — identical compaction to ShadePass/Spatial. OFF ⇒
    // the full-res wgX/wgY dispatch, byte-identical to before.
    const dx = ctx.checkerboardOn ? Math.ceil(Math.ceil(ctx.width / 2) / 8) : wgX;
    const dy = ctx.checkerboardOn ? Math.ceil(ctx.height / 8) : wgY;
    const pass = encoder.beginComputePass(computeDesc('ris'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.setBindGroup(3, lightTreeBindGroup); // RIS-only DI light-selection tree
    pass.dispatchWorkgroups(dx, dy, 1);
    pass.end();
  }

  dispose(): void {}
}
