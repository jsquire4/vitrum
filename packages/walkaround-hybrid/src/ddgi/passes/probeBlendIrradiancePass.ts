/**
 * ProbeBlendIrradiancePass — DDGI octahedral irradiance blend.
 *
 * Phase 2 of the 5-stage DDGI probe update chain. Reads per-ray radiance
 * from `rayResultsBuf` (populated by {@link ProbeRaysPass}) and the previous
 * frame's irradiance atlas, blends them with EWMA hysteresis, and writes
 * the result to the write-side atlas.
 *
 * Dispatch shape: `dispatchWorkgroups(activeCount, 1, 1)` with workgroup
 * size (8,8,1) declared inside `probeUpdateBlend.wgsl` — covers one probe
 * cell per workgroup.
 */

import { PROBE_UPDATE_BLEND_IRR_WGSL } from '../wgsl/probeUpdateBlend.wgsl.js';
import type { ProbePass, ProbePassContext } from './probePassTypes.js';

export class ProbeBlendIrradiancePass implements ProbePass {
  readonly id = 'ddgi-blend-irr';

  private _pipeline: GPUComputePipeline | null = null;

  async compile(device: GPUDevice): Promise<void> {
    const module = device.createShaderModule({ code: PROBE_UPDATE_BLEND_IRR_WGSL });
    this._pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'probeUpdateBlendIrradiance' },
    });
  }

  dispatch(encoder: GPUCommandEncoder, ctx: ProbePassContext): void {
    const pipeline = this._pipeline!;
    const { device, irrReadTex, irrWriteTex, linearSampler } = ctx;

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
        { binding: 0, resource: irrReadTex.createView() },
        { binding: 1, resource: linearSampler },
        { binding: 2, resource: irrWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
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
