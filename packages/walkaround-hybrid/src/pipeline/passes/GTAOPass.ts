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

import { defineUbo } from '@vitrum/shared-samplers';
import { buildGTAOBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

// W2-C13 follow-up — GTAOUniforms: 5 active f32 followed by 3 explicit f32
// pad slots to reach the 32-byte buffer size that `gtao.wgsl` declares and
// that `gtaoUpsample.wgsl` re-binds (`resourceManager` allocates 32 B). The
// std140 floor for 5×f32 alone is only 20 B (rounded up to 16), so the three
// trailing fields are kept explicit to preserve the buffer's tail-zero
// invariant byte-identically.
const GTAO_UBO = defineUbo([
  { name: 'tanFovHalf',          type: 'f32' },
  { name: 'radiusPx',            type: 'f32' },
  { name: 'intensity',           type: 'f32' },
  { name: 'depthThresh',         type: 'f32' },
  { name: 'bilateralDepthSigma', type: 'f32' },
  { name: '_pad0',               type: 'f32' },
  { name: '_pad1',               type: 'f32' },
  { name: '_pad2',               type: 'f32' },
] as const);

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
    // (audit M1 + B3). W2-C13 follow-up: byte-identical to the prior
    // Float32Array([..., 0, 0, 0]) write — the trailing 3 explicit pad
    // fields keep the buffer at 32 B to match what gtaoUpsample reads.
    const camY = inputs.projMatrix[5] ?? 1.0; // (1/tan(fov/2)) at the y-FOV
    const tanFovHalf = camY > 1e-6 ? 1.0 / camY : 0.5;
    const gtaoUboBytes = new ArrayBuffer(GTAO_UBO.sizeBytes);
    GTAO_UBO.pack(new DataView(gtaoUboBytes), 0, {
      tanFovHalf,
      radiusPx:            inputs.gtaoRadiusPx,
      intensity:           inputs.gtaoIntensity,
      depthThresh:         inputs.gtaoDepthThreshold,
      bilateralDepthSigma: inputs.gtaoBilateralDepthSigma,
      _pad0: 0, _pad1: 0, _pad2: 0,
    });
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
