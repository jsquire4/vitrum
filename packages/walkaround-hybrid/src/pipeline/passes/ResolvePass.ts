/**
 * ResolvePass — sparse-shade gap-fill pass.
 *
 * `checkerboardOn` is driven by the pipeline's `_checkerboard` flag (host
 * opt-in via HybridEngineOptions.checkerboardRendering; default OFF). When OFF
 * the pass is passthrough — every pixel copies through from `writeAccum` to
 * `common.resolvedTexture` (byte-identical to the pre-checkerboard wire-in).
 * When ON, shade.wgsl writes only the SHADED half of the checkerboard and this
 * pass reprojects the GAP pixels from the previous frame's radiance via the
 * motion-vector G-buffer slot. The `frameParity` (frameCount & 1) it packs is
 * the SAME phase shade.wgsl reads from the WalkaroundUBO, so the gap pixels
 * shade skips are exactly the pixels reprojected here.
 *
 * Layout note: resolve uses `@workgroup_size(8, 8, 1)` —
 * dispatch with `wgX/wgY` (`ceil(W/8)`), NOT the 16×16-sized counts.
 */

import { buildResolveBindGroup, type UboRef } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';
import { RESOLVE_UBO } from './uboLayouts.js';

export class ResolvePass implements Pass {
  readonly id = 'resolve' as const;
  readonly dependencies: readonly string[] = ['temporalAccum'];
  readonly passLabels: readonly PassLabel[] = ['resolve'];

  private readonly _pipeline: GPUComputePipeline;
  private readonly _uboRef: UboRef;
  /** Host opt-in (pipeline `_checkerboard`). OFF (default) ⇒ passthrough
   *  (every pixel shaded ⇒ byte-identity); ON ⇒ checkerboard gap-fill. */
  private readonly _checkerboard: boolean;

  constructor(
    pipeline: GPUComputePipeline,
    uboRef: UboRef,
    checkerboard: boolean,
  ) {
    this._pipeline = pipeline;
    this._uboRef = uboRef;
    this._checkerboard = checkerboard;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, bglCache, resources, frameState, frameCount, width, height } = ctx;

    // ResolveUniforms: u32 W, u32 H, u32 frameParity, u32 checkerboardOn (16 bytes).
    // W2-C13 follow-up: byte-identical to the prior Uint32Array write —
    // defineUbo packs four u32 fields contiguously at offsets 0/4/8/12.
    const resolveUboBytes = new ArrayBuffer(RESOLVE_UBO.sizeBytes);
    RESOLVE_UBO.pack(new DataView(resolveUboBytes), 0, {
      screenW:        width,
      screenH:        height,
      frameParity:    frameCount & 1,
      checkerboardOn: this._checkerboard ? 1 : 0,
    });
    device.queue.writeBuffer(this._uboRef.buf!, 0, resolveUboBytes);
    const bg = buildResolveBindGroup(
      device, bglCache,
      this._uboRef.buf!,
      frameState.writeAccum.createView(),           // current radiance (post-accum)
      frameState.readAccum.createView(),            // prev radiance (other ping-pong slot)
      resources.common.motionVectorTexture.createView(),
      resources.common.resolvedTexture.createView(),
    );
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'resolve');
  }

  dispose(): void {}
}
