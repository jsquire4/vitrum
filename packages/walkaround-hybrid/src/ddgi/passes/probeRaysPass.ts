/**
 * ProbeRaysPass — DDGI ray-casting compute pass.
 *
 * Phase 1 of the 5-stage DDGI probe update chain. For each active probe,
 * fires `RAYS_PER_PROBE` rays via inline BVH traversal and writes the
 * radiance + hit-distance result for each ray to `rayResultsBuf`.
 *
 * Per-probe dispatch shape:
 *   - `dispatchWorkgroups(activeCount)` — one workgroup per active probe.
 *   - Workgroup size is declared inside `probeUpdateRays.wgsl`.
 *
 * Owns its compiled compute pipeline (created in {@link compile}). All
 * other GPU resources are supplied per dispatch via {@link ProbePassContext}.
 */

import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';
import type { ProbePass, ProbePassContext } from './probePassTypes.js';

export class ProbeRaysPass implements ProbePass {
  readonly id = 'ddgi-probe-rays';

  private _pipeline: GPUComputePipeline | null = null;
  private readonly _maxMaterials: number;

  constructor(maxMaterials: number) {
    this._maxMaterials = maxMaterials;
  }

  /**
   * Compile the pipeline. Called once from `ProbeUpdatePass.init()`.
   * Throws if shader compilation fails — caller catches + reports.
   */
  async compile(device: GPUDevice): Promise<void> {
    // M9: compile with the host-specified material array size so scenes with
    // more than 64 materials don't overflow the uniform buffer.
    const module = device.createShaderModule({
      code: makeProbeUpdateRaysWGSL(this._maxMaterials),
    });
    this._pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'probeUpdateRays' },
    });
  }

  dispatch(encoder: GPUCommandEncoder, ctx: ProbePassContext): void {
    const pipeline = this._pipeline!;
    const { device, irrReadTex, linearSampler } = ctx;

    const bg0 = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ctx.bvhBuf } },
        { binding: 1, resource: { buffer: ctx.posBuf } },
        { binding: 2, resource: { buffer: ctx.idxBuf } },
        { binding: 3, resource: { buffer: ctx.normBuf } },
        { binding: 4, resource: { buffer: ctx.matIdBuf } },
      ],
    });
    const bg1 = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: ctx.materialsBuf } },
        { binding: 1, resource: { buffer: ctx.lightsBuf } },
      ],
    });
    const bg2 = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: ctx.rayResultsBuf } },
        { binding: 1, resource: { buffer: ctx.activeProbesBuf } },
        { binding: 2, resource: irrReadTex.createView() },
        { binding: 3, resource: linearSampler },
        { binding: 4, resource: { buffer: ctx.gridParamsBuf } },
        { binding: 5, resource: { buffer: ctx.frameParamsBuf } },
      ],
    });

    const pass = encoder.beginComputePass({ label: this.id });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg0);
    pass.setBindGroup(1, bg1);
    pass.setBindGroup(2, bg2);
    // Dispatch one workgroup per active probe.
    pass.dispatchWorkgroups(ctx.activeCount);
    pass.end();
  }

  dispose(): void {
    // Compute pipelines are GC-managed by WebGPU; null the ref so a
    // second dispose() is a no-op and re-init() forces fresh compile.
    this._pipeline = null;
  }
}
