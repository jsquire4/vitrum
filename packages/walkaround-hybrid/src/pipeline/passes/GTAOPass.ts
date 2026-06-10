/**
 * GTAOPass — Sprint 15 low-resolution GTAO compute pass.
 *
 * Packs the GTAOUniforms struct (tanFovHalf, radiusPx, intensity, depthThresh,
 * bilateralDepthSigma, gtaoDownscale, _pad...) per frame from
 * PipelineFrameInputs, then dispatches at W/ds × H/ds with 8×8 workgroups,
 * where ds = `ctx.gtaoDownscale` (2 for `gtaoMode:'on'` half-res, 4 for
 * `gtaoMode:'quarter'` quarter-res). Reads gNormalDepthTexture (written by
 * shade), writes aoHalfTexture (sized W/ds × H/ds by `createGtaoFrameResources`).
 *
 * E1 — also reads hdrAlbedoOut (Item 24) for Jiménez 2016 §5.2 multi-bounce.
 */

import { buildGTAOBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassGateOptions,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { GTAO_UBO } from './uboLayouts.js';

export class GTAOPass implements Pass {
  readonly id = 'gtao' as const;
  readonly dependencies: readonly string[] = ['shade'];
  readonly passLabels: readonly PassLabel[] = ['gtao'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  /** Phase-0 productization — gate off when the quality preset disables GTAO
   *  (`gtaoMode:'off'` ⇒ `gtaoEnabled:false`). Absent flag ⇒ on (default). */
  gates(opts: PassGateOptions): boolean {
    return opts.gtaoEnabled !== false;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, inputs, width, height, gtaoDownscale, resourceCache } = ctx;
    // AO compute downscale: 2 = half-res (`gtaoMode:'on'`), 4 = quarter-res
    // (`gtaoMode:'quarter'`). Clamp ≥ 1 so a bad value can't divide by zero.
    const ds = Math.max(1, Math.floor(gtaoDownscale));

    // Pack GTAOUniforms: (tanFovHalf, radiusPx, intensity, depthThresh,
    // bilateralDepthSigma, gtaoDownscale, _pad1, _pad2). radiusPx/intensity/
    // depthThresh/bilateralDepthSigma are host-configurable via
    // HybridEngineOptions.gtao (audit M1 + B3); gtaoDownscale (2/4) is the
    // half/quarter-res selector. The 2 trailing explicit pad fields keep the
    // buffer at 32 B to match what gtaoUpsample reads.
    const camY = inputs.camera.projMatrix[5] ?? 1.0; // (1/tan(fov/2)) at the y-FOV
    const tanFovHalf = camY > 1e-6 ? 1.0 / camY : 0.5;
    const gtaoUboBytes = new ArrayBuffer(GTAO_UBO.sizeBytes);
    GTAO_UBO.pack(new DataView(gtaoUboBytes), 0, {
      tanFovHalf,
      radiusPx:            inputs.gtao.gtaoRadiusPx,
      intensity:           inputs.gtao.gtaoIntensity,
      depthThresh:         inputs.gtao.gtaoDepthThreshold,
      bilateralDepthSigma: inputs.gtao.gtaoBilateralDepthSigma,
      gtaoDownscale:       ds,
      _pad1: 0, _pad2: 0,
    });
    device.queue.writeBuffer(resources.gtao.gtaoUboBuffer, 0, gtaoUboBytes);

    // Dispatch at the AO compute resolution (W/ds × H/ds). This is the line
    // that makes `gtaoMode:'quarter'` genuinely cheaper than `'on'`: ds=4
    // dispatches 1/16 the workgroups of full-res (1/4 of the half-res 'on'
    // count) over the matching W/4 × H/4 aoHalfTexture.
    const lowW = Math.max(1, Math.floor(width / ds));
    const lowH = Math.max(1, Math.floor(height / ds));
    const wgGtaoX = Math.ceil(lowW / 8);
    const wgGtaoY = Math.ceil(lowH / 8);

    // D6 — class (a): all four bindings are stable resources (the GTAO UBO
    // buffer identity is fixed; its CONTENTS are rewritten above each frame via
    // writeBuffer, which is independent of the bind group). Memoize the group;
    // the cache key is the resource identities so a resize-driven texture
    // reallocation auto-invalidates.
    const buildGtaoBg = (): GPUBindGroup => buildGTAOBindGroup(
      device, bglCache,
      resourceCache?.textureView(resources.common.gNormalDepthTexture)
        ?? resources.common.gNormalDepthTexture.createView(),
      resourceCache?.textureView(resources.gtao.aoHalfTexture)
        ?? resources.gtao.aoHalfTexture.createView(),
      resources.gtao.gtaoUboBuffer,
      // E1 — hdrAlbedoOut for Jiménez 2016 §5.2 multi-bounce term.
      resourceCache?.textureView(resources.common.albedoTexture)
        ?? resources.common.albedoTexture.createView(),
    );
    const bg = resourceCache?.bindGroup('pass:gtao', [
      resources.common.gNormalDepthTexture,
      resources.gtao.aoHalfTexture,
      resources.gtao.gtaoUboBuffer,
      resources.common.albedoTexture,
    ], buildGtaoBg) ?? buildGtaoBg();
    const pass = encoder.beginComputePass(computeDesc('gtao'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgGtaoX, wgGtaoY, 1);
    pass.end();
  }

  dispose(): void {}
}
