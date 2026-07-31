/**
 * RISGIPass — Sprint 16 ReSTIR-GI RIS pass.
 *
 * Half-resolution dispatch (W/2 × H/2). Reuses the shared frame/scene/ubo
 * bind groups + the hybrid-layers (DDGI) bind group at slot 3.
 *
 * Runs after the DI spatial passes (consumes the spatially-fused DI
 * reservoir from `bgFrame`) so the GI reservoir is built on the
 * variance-reduced primary visibility.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * NRC (Müller et al. 2021) — COMPILE-TIME structural gate
 * ════════════════════════════════════════════════════════════════════════════
 * When the engine was created with `nrcEnabled`, compilePipelines built the
 * gi-ris pipeline with a 5th `@group(4)` NRC bind group (MLP weights/biases +
 * hash tables + level descs + record gather + config UBO) and the inline-MLP
 * shader variant. In that case this pass binds slot 4 from the host-owned NRC
 * bind group (supplied via the getter). When NRC is OFF (default) the getter is
 * undefined, the pass dispatches the verbatim 4-group shared path, and the
 * pipeline structure is byte-for-byte pre-NRC. Binding slot 4 only when the
 * structural variant was compiled is what keeps the default render untouched —
 * the GRIS-class regression discipline (f8df9a4).
 */

import { dispatchSharedBindGroupPass, type Pass, type PassDispatchContext, type PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class RISGIPass implements Pass {
  readonly id = 'gi-ris' as const;
  readonly dependencies: readonly string[] = ['spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['gi-ris'];

  private readonly _pipeline: GPUComputePipeline;
  /** Clears NRC per-slot claims immediately before the NRC GI-RIS dispatch. */
  private readonly _nrcClearSlotClaims?: (encoder: GPUCommandEncoder) => void;

  constructor(
    pipeline: GPUComputePipeline,
    nrcClearSlotClaims?: (encoder: GPUCommandEncoder) => void,
  ) {
    this._pipeline = pipeline;
    if (nrcClearSlotClaims !== undefined) this._nrcClearSlotClaims = nrcClearSlotClaims;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    this._nrcClearSlotClaims?.(ctx.encoder);
    dispatchSharedBindGroupPass(ctx, this._pipeline, {
      label: 'gi-ris',
      // gi-ris binds its dedicated GI reservoir+frame group at slot 0 (not the
      // shared frameBindGroup), the DDGI hybrid-layers group at slot 3, and —
      // only when the compile-time NRC variant was built — the host-owned NRC
      // @group(4) group at slot 4. Half-res dispatch (W/2 × H/2).
      frameBindGroupOverride: ctx.risGiFrameBindGroup,
      useHybridLayers: true,
      restirGi: true,
    });
  }

  dispose(): void {}
}
