/**
 * ProbeBorderVisibilityPass — DDGI visibility atlas border-pixel fill.
 *
 * Phase 5 of the 5-stage DDGI probe update chain. Sibling of
 * {@link ProbeBorderIrradiancePass} for the visibility atlas. See that
 * file's header for the rationale (Majercik 2019 §3.2 border replication,
 * scratch-hop to satisfy WebGPU's no-self-bind constraint).
 *
 * Dispatch shape: `dispatchWorkgroups(probeCount, 1, 1)`. Each workgroup
 * runs 256 threads × 2 passes covering all 324 positions of the
 * (VIS_STRIDE)² visibility cell border ring.
 */

import { PROBE_UPDATE_BORDER_VIS_WGSL } from '../wgsl/probeUpdateBorder.wgsl.js';
import type { ProbePass, ProbePassContext } from './probePassTypes.js';

export class ProbeBorderVisibilityPass implements ProbePass {
  readonly id = 'ddgi-border-vis';

  private _pipeline: GPUComputePipeline | null = null;

  async compile(device: GPUDevice): Promise<void> {
    const module = device.createShaderModule({ code: PROBE_UPDATE_BORDER_VIS_WGSL });
    this._pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'probeUpdateBorderVisibility' },
    });
  }

  dispatch(encoder: GPUCommandEncoder, ctx: ProbePassContext): void {
    const pipeline = this._pipeline!;
    const { device, visScratchTex, visWriteTex } = ctx;
    if (!visScratchTex) {
      throw new Error('[DDGI] ProbeBorderVisibilityPass.dispatch: ctx.visScratchTex is null');
    }

    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: visScratchTex.createView() },
        { binding: 1, resource: visWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
        { binding: 2, resource: { buffer: ctx.borderVisUboBuf } },
      ],
    });
    const pass = encoder.beginComputePass({ label: this.id });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    // One workgroup per probe. 256 threads × 2 passes covers all 324 positions.
    pass.dispatchWorkgroups(ctx.probeCount, 1, 1);
    pass.end();
  }

  dispose(): void {
    this._pipeline = null;
  }
}
