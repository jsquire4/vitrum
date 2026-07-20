/**
 * ResolvePass — sparse-shade gap-fill pass.
 *
 * `checkerboardOn` is driven PER FRAME by `ctx.checkerboardOn` — the pipeline's
 * `cbActiveThisFrame` (= host opt-in `_checkerboard` via
 * HybridEngineOptions.checkerboardRendering AND the camera-move being below
 * `checkerboardMotionThresholdSq`; default OFF, and forced OFF on a fast-motion
 * frame). This is the SAME per-frame flag the shade + spatial passes read for
 * their compacted dispatch, so resolve gap-fills EXACTLY the frames (and pixels)
 * shade/spatial left sparse — including the motion fallback, where all four go
 * full-rate together. When OFF the pass is passthrough — every pixel copies
 * through from `writeAccum` to
 * `common.resolvedTexture` (byte-identical to the pre-checkerboard wire-in).
 * When ON, shade.wgsl writes only the SHADED half of the checkerboard and this
 * pass reprojects the GAP pixels from the previous frame's radiance via the
 * motion-vector G-buffer slot. The `frameParity` (frameCount & 1) it packs is
 * the SAME phase shade.wgsl reads from the WalkaroundUBO, so the gap pixels
 * shade skips are exactly the pixels reprojected here.
 *
 * The constructor still takes the host opt-in flag, but it is only a fast
 * disable for the default-OFF case (when the host never opts in, ctx.checkerboardOn
 * is always false too); the per-frame `ctx.checkerboardOn` is the authoritative
 * gate so the motion fallback stays consistent across passes.
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
import { cachedBindGroup } from '../PipelineResourceCache.js';
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
    const { device, bglCache, resources, frameState, frameCount, width, height, resourceCache } = ctx;

    // ResolveUniforms: u32 W, u32 H, u32 frameParity, u32 checkerboardOn (16 bytes).
    // W2-C13 follow-up: byte-identical to the prior Uint32Array write —
    // defineUbo packs four u32 fields contiguously at offsets 0/4/8/12.
    //
    // The gate is the PER-FRAME `ctx.checkerboardOn` (`cbActiveThisFrame`), not
    // the constructor `_checkerboard`: on a fast-motion frame the pipeline forces
    // checkerboard off for shade + spatial, and resolve must passthrough that
    // frame too (otherwise it would gap-fill pixels shade rendered full-rate).
    // `_checkerboard` is ANDed in as a defensive fast-path: when the host never
    // opted in, ctx.checkerboardOn is always false anyway, so this stays
    // byte-identical to the prior `_checkerboard ? 1 : 0` for the default path.
    const checkerboardOn = this._checkerboard && ctx.checkerboardOn;
    const resolveUboBytes = new ArrayBuffer(RESOLVE_UBO.sizeBytes);
    RESOLVE_UBO.pack(new DataView(resolveUboBytes), 0, {
      screenW:        width,
      screenH:        height,
      frameParity:    frameCount & 1,
      checkerboardOn: checkerboardOn ? 1 : 0,
    });
    device.queue.writeBuffer(this._uboRef.buf!, 0, resolveUboBytes);
    const buildBg = (): GPUBindGroup => buildResolveBindGroup(
      device, bglCache,
      this._uboRef.buf!,
      resourceCache?.textureView(frameState.writeAccum) ?? frameState.writeAccum.createView(),
      resourceCache?.textureView(frameState.readAccum) ?? frameState.readAccum.createView(),
      resourceCache?.textureView(resources.common.motionVectorTexture) ?? resources.common.motionVectorTexture.createView(),
      resourceCache?.textureView(resources.common.resolvedTexture) ?? resources.common.resolvedTexture.createView(),
    );
    const bg = cachedBindGroup(resourceCache, 'pass:resolve', [
      this._uboRef.buf,
      frameState.writeAccum,
      frameState.readAccum,
      resources.common.motionVectorTexture,
      resources.common.resolvedTexture,
    ], buildBg);
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'resolve');
  }

  dispose(): void {}
}
