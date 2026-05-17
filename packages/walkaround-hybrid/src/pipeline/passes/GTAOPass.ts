/**
 * GTAOPass — Sprint 15 half-resolution GTAO compute pass.
 *
 * Packs the GTAOUniforms struct (tanFovHalf, radiusPx, intensity, depthThresh,
 * bilateralDepthSigma, _pad...) per frame from PipelineFrameInputs, then
 * dispatches at half-res (W/2 × H/2 with 8×8 workgroups). Reads
 * gNormalDepthTexture (written by shade), writes aoHalfTexture.
 *
 * E1 — also reads hdrAlbedoOut (Item 24) for Jiménez 2016 §5.2 multi-bounce.
 */

import { buildGTAOBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class GTAOPass implements Pass {
  readonly id = 'gtao' as const;
  readonly dependencies: readonly string[] = ['shade'];
  readonly passLabels: readonly PassLabel[] = ['gtao'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, inputs, width, height } = ctx;

    // Pack GTAOUniforms: (tanFovHalf, radiusPx, intensity, depthThresh,
    // bilateralDepthSigma, _pad0, _pad1, _pad2). radiusPx/intensity/depthThresh/
    // bilateralDepthSigma are now host-configurable via HybridEngineOptions.gtao
    // (audit M1 + B3).
    const camY = inputs.projMatrix[5] ?? 1.0; // (1/tan(fov/2)) at the y-FOV
    const tanFovHalf = camY > 1e-6 ? 1.0 / camY : 0.5;
    const gtaoUboBytes = new Float32Array([
      tanFovHalf,
      inputs.gtaoRadiusPx,
      inputs.gtaoIntensity,
      inputs.gtaoDepthThreshold,
      inputs.gtaoBilateralDepthSigma,
      0, 0, 0,
    ]);
    device.queue.writeBuffer(resources.gtao.gtaoUboBuffer, 0, gtaoUboBytes);

    const halfW = Math.max(1, Math.floor(width / 2));
    const halfH = Math.max(1, Math.floor(height / 2));
    const wgGtaoX = Math.ceil(halfW / 8);
    const wgGtaoY = Math.ceil(halfH / 8);

    const bg = buildGTAOBindGroup(
      device, bglCache,
      resources.common.gNormalDepthTexture.createView(),
      resources.gtao.aoHalfTexture.createView(),
      resources.gtao.gtaoUboBuffer,
      // E1 — hdrAlbedoOut for Jiménez 2016 §5.2 multi-bounce term.
      resources.common.albedoTexture.createView(),
    );
    const pass = encoder.beginComputePass(computeDesc('gtao'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgGtaoX, wgGtaoY, 1);
    pass.end();
  }

  dispose(): void {}
}
