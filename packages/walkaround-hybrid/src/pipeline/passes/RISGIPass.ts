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

import { dispatchSharedBindGroupPass } from '../Pass.js';
import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class RISGIPass implements Pass {
  readonly id = 'gi-ris' as const;
  readonly dependencies: readonly string[] = ['spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['gi-ris'];

  private readonly _pipeline: GPUComputePipeline;
  /** When NRC is compile-time on, returns the host-owned @group(4) NRC bind
   *  group to bind at slot 4. Undefined (default) ⇒ NRC OFF, no slot-4 bind. */
  private readonly _nrcBindGroup?: () => GPUBindGroup;

  constructor(pipeline: GPUComputePipeline, nrcBindGroup?: () => GPUBindGroup) {
    this._pipeline = pipeline;
    if (nrcBindGroup !== undefined) this._nrcBindGroup = nrcBindGroup;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    if (this._nrcBindGroup === undefined) {
      // NRC OFF (default) — verbatim 4-group half-res shared dispatch.
      dispatchSharedBindGroupPass(ctx, this._pipeline, {
        label: 'gi-ris',
        useHybridLayers: true,
        halfRes: true,
      });
      return;
    }
    // NRC ON — bind frame/scene/ubo/hybrid + the NRC @group(4), half-res.
    const {
      encoder, computeDesc,
      frameBindGroup, sceneBindGroup, uboBindGroup, hybridLayersBindGroup,
      halfWgX, halfWgY,
    } = ctx;
    const pass = encoder.beginComputePass(computeDesc('gi-ris'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.setBindGroup(3, hybridLayersBindGroup);
    pass.setBindGroup(4, this._nrcBindGroup());
    pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
    pass.end();
  }

  dispose(): void {}
}
