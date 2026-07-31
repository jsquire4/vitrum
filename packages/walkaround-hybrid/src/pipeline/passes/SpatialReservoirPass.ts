/**
 * SpatialReservoirPass — ReSTIR-DI spatial reuse (1 or 2 ping-pong passes).
 *
 * Owns up to both `spatial-1` and `spatial-2` labels: emits one compute
 * dispatch per label with the shared frame/scene/ubo bind groups. NEIGHBORS=5
 * (compile-time const, intentionally fixed for Phase 0); the spatial reuse is
 * the dominant variance reducer in the pipeline.
 *
 * Phase-0 productization — the ping-pong PASS COUNT is host/preset-driven via
 * the constructor `passCount` arg (the spatial NEIGHBOR count stays fixed at 5).
 * `passCount: 2` (ultra/high) is the full-fidelity variance reducer; `1`
 * (medium/low) halves the spatial cost. The terminal label `spatial-2` is kept
 * for BOTH counts so the `shade` dependency + the `id` stay stable; a 1-pass
 * config emits only `['spatial-2']` (a single dispatch).
 *
 * R2 — `buildPassLayout` MUST be built with the same `diSpatialPasses` so the
 * timestamp slot layout matches the labels emitted here (asserted by the
 * pass-layout parity test).
 *
 * Checkerboard sparse-spatial (host opt-in; default OFF): the spatial reuse is
 * the pipeline's dominant cost — each thread does castPrimary(center) + 5×
 * castPrimary(neighbor) = 6 BVH traversals, ×passCount. When checkerboard is ON
 * this pass COMPACTS each dispatch to ~half the threads —
 * `ceil(ceil(W/2)/8) × ceil(H/8)` workgroups — so those 6 BVH re-casts run for
 * ONE pixel per active-parity slot instead of every pixel, genuinely skipping
 * the gap-parity work (compaction, NOT a shader early-return — early-returned
 * threads still occupy their warps and save no BVH traversals). The spatial
 * shader decodes the compacted global_invocation_id back into the true full-res
 * active-parity pixel `px = gid.x*2 + ((gid.y + frameParity)&1)`, `py = gid.y`,
 * which lands EXACTLY on the (px+py)&1 == frameParity set ShadePass shades and
 * reads this frame. OFF ⇒ the full-res `wgX/wgY` dispatch, byte-identical to
 * before. This mirrors {@link ShadePass} exactly so the active-parity decode is
 * shared across the spatial and shade passes.
 */

import { SharedBindGroupPass, dispatchSharedBindGroupPass } from '../Pass.js';
import type { PassDispatchContext } from '../Pass.js';
import { diSpatialPassLabels } from './passOrder.js';
import type { PassLabel } from '../timestampQueries.js';

export class SpatialReservoirPass extends SharedBindGroupPass {
  readonly id = 'spatial-2' as const; // last dispatch id — shade depends on this.
  readonly dependencies: readonly string[] = ['temporal'];
  readonly passLabels: readonly PassLabel[];
  private readonly _roundTwoPipeline: GPUComputePipeline;
  private readonly _passCount: 1 | 2;

  constructor(
    roundOnePipeline: GPUComputePipeline,
    roundTwoPipelineOrPassCount: GPUComputePipeline | 1 | 2 = roundOnePipeline,
    passCount: 1 | 2 = 2,
  ) {
    super(roundOnePipeline);
    if (typeof roundTwoPipelineOrPassCount === 'number') {
      // Compatibility for pass-level harnesses using the historical
      // `(pipeline, passCount)` constructor. Production always supplies the
      // independently-specialized round-two pipeline.
      this._roundTwoPipeline = roundOnePipeline;
      this._passCount = roundTwoPipelineOrPassCount;
    } else {
      this._roundTwoPipeline = roundTwoPipelineOrPassCount;
      this._passCount = passCount;
    }
    this.passLabels = diSpatialPassLabels(this._passCount);
  }

  override dispatch(ctx: PassDispatchContext): void {
    const checkerboardReservoirDispatch =
      ctx.checkerboardOn && ctx.restirReservoirScale === 1;
    if (checkerboardReservoirDispatch) {
      // The compact dispatch overwrites only active-parity destinations.
      // Seed the whole round-one destination from temporal/current first so
      // gap pixels remain valid inputs to round two and valid terminal history
      // for a one-round configuration.
      const { reservoirCurrentBuffer, reservoirSpatialBuffer } =
        ctx.resources.restirDI;
      ctx.encoder.copyBufferToBuffer(
        reservoirCurrentBuffer,
        0,
        reservoirSpatialBuffer,
        0,
        reservoirCurrentBuffer.size,
      );
    }
    const dispatchOverride = checkerboardReservoirDispatch
      ? { x: ctx.checkerboardWgX, y: ctx.checkerboardWgY }
      : { x: ctx.restirDiWgX, y: ctx.restirDiWgY };
    const firstLabel = this.passLabels[0]!;
    dispatchSharedBindGroupPass(ctx, this._pipeline, {
      label: firstLabel,
      ...(dispatchOverride !== undefined ? { dispatchOverride } : {}),
    });
    if (this._passCount === 1) return;

    dispatchSharedBindGroupPass(ctx, this._roundTwoPipeline, {
      label: this.passLabels[1]!,
      frameBindGroupOverride: ctx.diSpatialReverseFrameBindGroup,
      ...(dispatchOverride !== undefined ? { dispatchOverride } : {}),
    });
  }
}
