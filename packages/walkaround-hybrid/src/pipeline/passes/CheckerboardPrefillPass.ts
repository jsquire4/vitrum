/**
 * CheckerboardPrefillPass — fills hdrColorTexture gap pixels BEFORE the
 * denoiser-adapter slot.
 *
 * When checkerboard rendering is on, ShadePass writes only the active-parity
 * half of `hdrColorTexture`; the complementary gap-parity pixels hold stale
 * radiance from the previous frame.  The four "real" denoisers (svgf-real,
 * bmfr, neural, oidn-final) all read `hdrColorTexture` directly at the
 * denoiser-adapter slot — they do not participate in the temporal-accumulator
 * ping-pong and therefore receive structurally corrupted input (half the
 * pixels are wrong-frame data).
 *
 * This pass fills the gap pixels in `hdrColorTexture` via temporal
 * reprojection from the previous-frame accumulator (`readAccum`) through the
 * motion vectors.  The denoiser then sees a fully-populated `hdrColorTexture`.
 *
 * Gating:
 *   - OFF when `checkerboardOn` is false (byte-identical to today's default
 *     path; the pass is simply skipped and hdrColorTexture is untouched).
 *   - OFF for the default `atrous` / `atrous-variance` / `none` denoisers,
 *     which consume hdrColorTexture through the temporal-accumulator path and
 *     are less sensitive to per-pixel gaps.
 *   - ON when checkerboard is active AND the active denoiser is one of
 *     `['svgf-real', 'bmfr', 'neural', 'oidn-final']`.
 *
 * Dependencies:
 *   - `gtao-upsample` — matches the former sole dependency of
 *     `denoiser-adapter`.  This pass runs in the topological slot immediately
 *     before the denoiser-adapter:
 *       … → gtao-upsample → cb-prefill → denoiser-adapter → …
 *
 * Bind group (`cb-prefill` BGL, 4 bindings):
 *   0  CbPrefillUniforms UBO  (16 bytes: screenW/H, frameParity, _pad)
 *   1  readAccum              (previous-frame accumulated radiance, sampled)
 *   2  motionVectorTexture    (rgba32float, sampled)
 *   3  hdrColorTexture        (rgba16float, storage write — gap pixels only)
 *
 * The pass has its own `_cbPrefillUboRef` buffer (16 bytes). Values are the
 * same semantics as `RESOLVE_UBO` (frameParity, screenW/H) but packed into
 * the `CbPrefillUniforms` layout (no `checkerboardOn` field — this pass
 * only runs when checkerboard is on).
 */

import { buildCbPrefillBindGroup, type UboRef } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassGateOptions,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';
import { CB_PREFILL_UBO } from './uboLayouts.js';

/** Denoiser IDs that read `hdrColorTexture` directly and therefore require
 *  the pre-denoiser gap-fill when checkerboard rendering is active. */
const REAL_DENOISER_IDS: ReadonlySet<string> = new Set([
  'svgf-real',
  'bmfr',
  'neural',
  'oidn-final',
]);

export class CheckerboardPrefillPass implements Pass {
  readonly id = 'cb-prefill' as const;
  /** Dependency chain: gtao-upsample → cb-prefill → (denoiser-adapter). */
  readonly dependencies: readonly string[] = ['gtao-upsample'];
  readonly passLabels: readonly PassLabel[] = ['cb-prefill'];

  private readonly _pipeline: GPUComputePipeline;
  private readonly _uboRef: UboRef;
  /** Host opt-in flag (pipeline `_checkerboard`). When false the pipeline
   *  never sets `checkerboardOn`, so this pass is never actually needed;
   *  the fast-path `gates()` check short-circuits before the denoiser lookup. */
  private readonly _checkerboard: boolean;

  constructor(
    pipeline: GPUComputePipeline,
    uboRef: UboRef,
    checkerboard: boolean,
  ) {
    this._pipeline    = pipeline;
    this._uboRef      = uboRef;
    this._checkerboard = checkerboard;
  }

  gates(opts: PassGateOptions): boolean {
    // Fast-path: checkerboard is off at the host level → always skip.
    if (!this._checkerboard) return false;
    // Per-frame: the motion-fallback may have forced full-rate this frame.
    if (!opts.checkerboardOn) return false;
    // Only the four real denoisers need the gap-fill.
    return REAL_DENOISER_IDS.has(opts.denoiserMode);
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, bglCache, frameState, frameCount, width, height, resourceCache } = ctx;

    // Pack CbPrefillUniforms: screenW/H + frameParity + padding.
    const uboBytes = new ArrayBuffer(CB_PREFILL_UBO.sizeBytes);
    CB_PREFILL_UBO.pack(new DataView(uboBytes), 0, {
      screenW:     width,
      screenH:     height,
      frameParity: frameCount & 1,
      _pad:        0,
    });
    device.queue.writeBuffer(this._uboRef.buf!, 0, uboBytes);

    const buildBg = (): GPUBindGroup => buildCbPrefillBindGroup(
      device, bglCache,
      this._uboRef.buf!,
      resourceCache?.textureView(frameState.readAccum) ?? frameState.readAccum.createView(),
      resourceCache?.textureView(ctx.resources.common.motionVectorTexture) ?? ctx.resources.common.motionVectorTexture.createView(),
      resourceCache?.textureView(ctx.resources.common.hdrColorTexture) ?? ctx.resources.common.hdrColorTexture.createView(),
    );
    const bg = resourceCache?.bindGroup('pass:cb-prefill', [
      this._uboRef.buf,
      frameState.readAccum,
      ctx.resources.common.motionVectorTexture,
      ctx.resources.common.hdrColorTexture,
    ], buildBg) ?? buildBg();

    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'cb-prefill');
  }

  dispose(): void {}
}
