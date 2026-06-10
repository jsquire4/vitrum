/**
 * Probe-update compute pass dispatch (W4c — extracted from probeUpdatePass.ts).
 */
import type { ProbeGrid } from './probeGrid.js';
import type { ProbeUpdateGpuState } from './probeUpdateGpuState.js';
import { DDGI_BORDER_UBO } from './probeUpdateUbos.js';

export interface ProbeUpdateBorderUboInput {
  probeCount: number;
  atlasWidth: number;
  atlasHeight: number;
  gridDimX: number;
  gridDimY: number;
  gridDimZ: number;
}

export function packProbeUpdateBorderUbo(input: ProbeUpdateBorderUboInput): ArrayBuffer {
  const data = new ArrayBuffer(DDGI_BORDER_UBO.sizeBytes);
  DDGI_BORDER_UBO.pack(new DataView(data), 0, {
    numProbes: input.probeCount,
    atlasWidth: input.atlasWidth,
    atlasHeight: input.atlasHeight,
    _pad0: 0,
    gridDimX: input.gridDimX,
    gridDimY: input.gridDimY,
    gridDimZ: input.gridDimZ,
    _pad1: 0,
  });
  return data;
}

export function uploadProbeUpdateBorderUbo(
  device: GPUDevice,
  gpu: ProbeUpdateGpuState,
  grid: ProbeGrid,
  atlas: GPUTexture,
  which: 'irr' | 'vis',
): void {
  const data = packProbeUpdateBorderUbo({
    probeCount: grid.probeCount,
    atlasWidth: atlas.width,
    atlasHeight: atlas.height,
    gridDimX: grid.params.dims.x,
    gridDimY: grid.params.dims.y,
    gridDimZ: grid.params.dims.z,
  });
  // Irradiance is SH (seam-free, no border pass), so only the visibility border
  // UBO exists; `which` is retained for call-site clarity but is always 'vis'.
  const buf = gpu.borderVisUboBuf;
  device.queue.writeBuffer(buf, 0, data);
}

export function dispatchProbeUpdateRaysPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  irrReadTex: GPUTexture,
): void {
  const bg0 = gpu.device.createBindGroup({
    layout: gpu.raysPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.bvhBuf } },
      { binding: 1, resource: { buffer: gpu.posBuf } },
      { binding: 2, resource: { buffer: gpu.idxBuf } },
      { binding: 3, resource: { buffer: gpu.normBuf } },
      { binding: 4, resource: { buffer: gpu.matIdBuf } },
      { binding: 5, resource: { buffer: gpu.tlasNodesBuf } },
      { binding: 6, resource: { buffer: gpu.tlasInstIdxBuf } },
      { binding: 7, resource: { buffer: gpu.tlasBlasRootsBuf } },
      { binding: 8, resource: { buffer: gpu.tlasW2lBuf } },
      { binding: 9, resource: { buffer: gpu.tlasL2wBuf } },
      { binding: 10, resource: { buffer: gpu.traceParamsBuf } },
    ],
  });
  const bg1 = gpu.device.createBindGroup({
    layout: gpu.raysPipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: gpu.materialsBuf } },
      { binding: 1, resource: { buffer: gpu.lightsBuf } },
      // H18 Stage 2 — area-emitter NEE triangles (dummy 16-byte buf when count==0).
      { binding: 2, resource: { buffer: gpu.emitterTrisBuf } },
    ],
  });
  const bg2 = gpu.device.createBindGroup({
    layout: gpu.raysPipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
      { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
      { binding: 2, resource: irrReadTex.createView() },
      { binding: 3, resource: gpu.linearSampler },
      { binding: 4, resource: { buffer: gpu.gridParamsBuf } },
      { binding: 5, resource: { buffer: gpu.frameParamsBuf } },
      // Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
      // A 1×1 placeholder view is bound when hasEnv=0 so the bind group is
      // always valid; the WGSL sampleSkyColor gates on frameParams.hasEnv
      // before sampling, so the placeholder is never actually read.
      { binding: 6, resource: gpu.envMapView },
      { binding: 7, resource: gpu.envSamplerForProbe },
    ],
  });

  const pass = encoder.beginComputePass({ label: 'ddgi-probe-rays' });
  pass.setPipeline(gpu.raysPipeline);
  pass.setBindGroup(0, bg0);
  pass.setBindGroup(1, bg1);
  pass.setBindGroup(2, bg2);
  pass.dispatchWorkgroups(activeCount);
  pass.end();
}

export function dispatchProbeUpdateBlendIrrPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  irrReadTex: GPUTexture,
  irrWriteTex: GPUTexture,
): void {
  const bg0 = gpu.device.createBindGroup({
    layout: gpu.blendIrrPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
      { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
      { binding: 2, resource: { buffer: gpu.gridParamsBuf } },
      { binding: 3, resource: { buffer: gpu.blendParamsBuf } },
    ],
  });
  const bg1 = gpu.device.createBindGroup({
    layout: gpu.blendIrrPipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: irrReadTex.createView() },
      { binding: 1, resource: gpu.linearSampler },
      { binding: 2, resource: irrWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
    ],
  });

  const pass = encoder.beginComputePass({ label: 'ddgi-blend-irr' });
  pass.setPipeline(gpu.blendIrrPipeline);
  pass.setBindGroup(0, bg0);
  pass.setBindGroup(1, bg1);
  pass.dispatchWorkgroups(activeCount, 1, 1);
  pass.end();
}

export function dispatchProbeUpdateBlendVisPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  activeCount: number,
  visReadTex: GPUTexture,
  visWriteTex: GPUTexture,
): void {
  const bg0 = gpu.device.createBindGroup({
    layout: gpu.blendVisPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.rayResultsBuf } },
      { binding: 1, resource: { buffer: gpu.activeProbesBuf } },
      { binding: 2, resource: { buffer: gpu.gridParamsBuf } },
      { binding: 3, resource: { buffer: gpu.blendParamsBuf } },
    ],
  });
  const bg1 = gpu.device.createBindGroup({
    layout: gpu.blendVisPipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: visReadTex.createView() },
      { binding: 1, resource: gpu.linearSampler },
      { binding: 2, resource: visWriteTex.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
    ],
  });

  const pass = encoder.beginComputePass({ label: 'ddgi-blend-vis' });
  pass.setPipeline(gpu.blendVisPipeline);
  pass.setBindGroup(0, bg0);
  pass.setBindGroup(1, bg1);
  pass.dispatchWorkgroups(activeCount, 1, 1);
  pass.end();
}

export function dispatchProbeUpdateBorderVisPass(
  encoder: GPUCommandEncoder,
  gpu: ProbeUpdateGpuState,
  probeCount: number,
  scratchTex: GPUTexture,
  writeAtlas: GPUTexture,
): void {
  const bg = gpu.device.createBindGroup({
    layout: gpu.borderVisPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: scratchTex.createView() },
      { binding: 1, resource: writeAtlas.createView({ format: 'rgba16float', mipLevelCount: 1 }) },
      { binding: 2, resource: { buffer: gpu.borderVisUboBuf } },
    ],
  });
  const pass = encoder.beginComputePass({ label: 'ddgi-border-vis' });
  pass.setPipeline(gpu.borderVisPipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(probeCount, 1, 1);
  pass.end();
}
