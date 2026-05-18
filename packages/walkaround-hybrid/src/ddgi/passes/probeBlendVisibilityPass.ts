/**
 * ProbeBlendVisibilityPass — DDGI octahedral visibility (mean, mean²) blend.
 *
 * Phase 3 of the 5-stage DDGI probe update chain. Mirrors
 * {@link ProbeBlendIrradiancePass} but writes to the visibility atlas
 * (mean distance + mean-squared distance for Chebyshev visibility queries
 * in the shading pass).
 *
 * The visibility atlas is allocated as `rgba16float` because WebGPU does
 * not support `rg16float` as a storage texture format — the WGSL shader
 * declares its storage texture as `rgba16float` to match.
 *
 * Dispatch shape: `dispatchWorkgroups(activeCount, 1, 1)` — one workgroup
 * per active probe cell.
 */

import { PROBE_UPDATE_BLEND_VIS_WGSL } from '../wgsl/probeUpdateBlend.wgsl.js';
import type { ProbePass, ProbePassContext } from './probePassTypes.js';

export class ProbeBlendVisibilityPass implements ProbePass {
  readonly id = 'ddgi-blend-vis';

  private _pipeline: GPUComputePipeline | null = null;

  async compile(device: GPUDevice): Promise<void> {
    const module = device.createShaderModule({ code: PROBE_UPDATE_BLEND_VIS_WGSL });
    this._pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'probeUpdateBlendVisibility' },
    });
  }

  dispatch(encoder: GPUCommandEncoder, ctx: ProbePassContext): void {
    const pipeline = this._pipeline!;
    const { device, visReadTex, visWriteTex, linearSampler } = ctx;

    const bg0 = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ctx.rayResultsBuf } },
        { binding: 1, resource: { buffer: ctx.activeProbesBuf } },
        { binding: 2, resource: { buffer: ctx.gridParamsBuf } },
        { binding: 3, resource: { buffer: ctx.blendParamsBuf } },
      ],
    });
    const bg1 = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: visReadTex.createView() },
        { binding: 1, resource: linearSampler },
        { binding: 2, resource: visWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
      ],
    });

    const pass = encoder.beginComputePass({ label: this.id });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.dispatchWorkgroups(ctx.activeCount, 1, 1);
    pass.end();
  }

  dispose(): void {
    this._pipeline = null;
  }
}
