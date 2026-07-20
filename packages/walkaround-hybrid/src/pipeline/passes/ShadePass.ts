/**
 * ShadePass — re-traces primary ray, evaluates ReSTIR, computes DI + indirect bounce.
 *
 * Reads the final DI reservoir (post-spatial-2) and the final GI reservoir
 * (post-gi-spatial-2). Writes hdrColorTexture (direct), hdrIndirectTexture
 * (indirect), hdrTotalTexture (direct+indirect), gNormalDepthTexture, and
 * the albedoTexture for downstream demodulation.
 *
 * Full-res dispatch with the hybrid-layers (DDGI) group bound at slot 3;
 * the dispatch body is the shared {@link SharedBindGroupPass}.
 *
 * Checkerboard sparse-shade (host opt-in; default OFF): when ON this pass
 * COMPACTS its dispatch to ~half the threads — `ceil(ceil(W/2)/8) × ceil(H/8)`
 * workgroups — so only one thread is launched per active-parity pixel, instead
 * of the previous full-res dispatch that early-returned the gap-parity threads
 * (which wasted compute: early-returned threads still occupy their warps). The
 * shade shader decodes the compacted `global_invocation_id` back into the true
 * full-res active-parity pixel
 *   `px = gid.x*2 + ((gid.y + frameParity) & 1)`,  `py = gid.y`
 * which lands EXACTLY on the `(px+py)&1 == frameParity` set the early-return
 * path shaded (and ResolvePass gap-fills), so the rendered image is unchanged.
 * OFF ⇒ the full-res `wgX/wgY` dispatch, byte-identical to before.
 */

import { SharedBindGroupPass, dispatchSharedBindGroupPass } from '../Pass.js';
import type { PassDispatchContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class ShadePass extends SharedBindGroupPass {
  readonly id = 'shade' as const;
  readonly dependencies: readonly string[] = ['spatial-2', 'gi-spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['shade'];

  protected override readonly useHybridLayers = true;
  protected override readonly useShadeHybridLayers = true;

  override dispatch(ctx: PassDispatchContext): void {
    if (!ctx.checkerboardOn) {
      // OFF path — full-res dispatch, byte-identical to before.
      super.dispatch(ctx);
      return;
    }
    // Checkerboard ON — compact the X dispatch to the active-parity columns.
    // Each row has at most ceil(W/2) active-parity pixels; the shader's decode
    // maps compacted gid.x -> px = gid.x*2 + ((gid.y + frameParity)&1) and
    // guards the few overshoot threads (px >= W) with the existing bounds
    // check. Y stays full-res (one compacted thread per row). Workgroup size
    // is 8×8 (matches shade.wgsl @workgroup_size(8, 8, 1)).
    dispatchSharedBindGroupPass(ctx, this._pipeline, {
      label: 'shade',
      useHybridLayers: this.useHybridLayers,
      useShadeHybridLayers: this.useShadeHybridLayers,
      dispatchOverride: { x: ctx.checkerboardWgX, y: ctx.checkerboardWgY },
    });
  }
}
