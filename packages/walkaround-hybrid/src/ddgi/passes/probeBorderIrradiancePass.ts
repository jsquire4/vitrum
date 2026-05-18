/**
 * ProbeBorderIrradiancePass — DDGI irradiance atlas border-pixel fill.
 *
 * Phase 4 of the 5-stage DDGI probe update chain. Implements
 * Majercik 2019 §3.2: after blend, each probe cell's interior pixels are
 * correct but the 1-pixel border surrounding the cell is zeroed. Without
 * border replication, bilinear sampling at every probe-cell edge blends
 * with black, darkening probe lookups at exactly the texels most often
 * sampled.
 *
 * This pass reads from a scratch copy of the post-blend atlas
 * ({@link ProbePassContext.irrScratchTex}, populated by the orchestrator
 * with a `copyTextureToTexture` before dispatch) and writes the border
 * pixels back into the live write-side atlas. The scratch hop is required
 * because WebGPU forbids binding the same texture as both `texture_2d`
 * (read) and `texture_storage_2d` (write) in a single pipeline.
 *
 * Dispatch shape: `dispatchWorkgroups(probeCount, 1, 1)`. Each workgroup
 * has 48 threads covering the (IRR_STRIDE)² = 100 positions of one cell's
 * border ring.
 *
 * References: Majercik et al. 2019 — "Dynamic Diffuse Global Illumination
 * with Ray-Traced Irradiance Probes" §3.2.
 */

import { PROBE_UPDATE_BORDER_IRR_WGSL } from '../wgsl/probeUpdateBorder.wgsl.js';
import type { ProbePass, ProbePassContext } from './probePassTypes.js';

export class ProbeBorderIrradiancePass implements ProbePass {
  readonly id = 'ddgi-border-irr';

  private _pipeline: GPUComputePipeline | null = null;

  async compile(device: GPUDevice): Promise<void> {
    const module = device.createShaderModule({ code: PROBE_UPDATE_BORDER_IRR_WGSL });
    this._pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'probeUpdateBorderIrradiance' },
    });
  }

  dispatch(encoder: GPUCommandEncoder, ctx: ProbePassContext): void {
    const pipeline = this._pipeline!;
    const { device, irrScratchTex, irrWriteTex } = ctx;
    if (!irrScratchTex) {
      throw new Error('[DDGI] ProbeBorderIrradiancePass.dispatch: ctx.irrScratchTex is null');
    }

    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: irrScratchTex.createView() },
        { binding: 1, resource: irrWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
        { binding: 2, resource: { buffer: ctx.borderIrrUboBuf } },
      ],
    });
    const pass = encoder.beginComputePass({ label: this.id });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    // One workgroup per probe. Each workgroup has 48 threads covering the
    // (IRR_STRIDE)² = 100 positions of one cell's border ring.
    pass.dispatchWorkgroups(ctx.probeCount, 1, 1);
    pass.end();
  }

  dispose(): void {
    this._pipeline = null;
  }
}
