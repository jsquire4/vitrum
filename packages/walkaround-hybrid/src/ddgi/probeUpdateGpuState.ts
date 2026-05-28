/**
 * GPU resource bundle for {@link ProbeUpdatePass} (W4c).
 */
import type { ProbeUpdateBvhGpuBuffers } from './probeUpdateBvhBuffers.js';

export interface ProbeUpdateGpuState extends ProbeUpdateBvhGpuBuffers {
  device: GPUDevice;
  raysPipeline: GPUComputePipeline;
  blendIrrPipeline: GPUComputePipeline;
  blendVisPipeline: GPUComputePipeline;
  borderIrrPipeline: GPUComputePipeline;
  borderVisPipeline: GPUComputePipeline;
  irrScratchTex: GPUTexture | null;
  visScratchTex: GPUTexture | null;
  traceParamsBuf: GPUBuffer;
  materialsBuf: GPUBuffer;
  lightsBuf: GPUBuffer;
  gridParamsBuf: GPUBuffer;
  frameParamsBuf: GPUBuffer;
  blendParamsBuf: GPUBuffer;
  borderIrrUboBuf: GPUBuffer;
  borderVisUboBuf: GPUBuffer;
  rayResultsBuf: GPUBuffer;
  activeProbesBuf: GPUBuffer;
  linearSampler: GPUSampler;
}
