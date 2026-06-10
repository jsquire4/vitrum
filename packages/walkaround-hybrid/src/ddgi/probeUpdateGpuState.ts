/**
 * GPU resource bundle for {@link ProbeUpdatePass} (W4c).
 */
import type { ProbeUpdateBvhGpuBuffers } from './probeUpdateBvhBuffers.js';

export interface ProbeUpdateGpuState extends ProbeUpdateBvhGpuBuffers {
  device: GPUDevice;
  raysPipeline: GPUComputePipeline;
  blendIrrPipeline: GPUComputePipeline;
  blendVisPipeline: GPUComputePipeline;
  borderVisPipeline: GPUComputePipeline;   // irradiance is SH (seam-free) — no irr border pass
  irrScratchTex: GPUTexture | null;
  visScratchTex: GPUTexture | null;
  traceParamsBuf: GPUBuffer;
  materialsBuf: GPUBuffer;
  lightsBuf: GPUBuffer;
  /** H18 Stage 2 — packed EmitterTri array for rect/disc area-emitter NEE in the
   *  probe-ray kernel. Matches the RC `rc_emitters` layout (5 × vec4f = 80 bytes
   *  per tri). A 16-byte dummy (1 element) when emitterCount == 0 so the bind
   *  group is always valid. */
  emitterTrisBuf: GPUBuffer;
  /** Number of valid emitter triangles in emitterTrisBuf (0 when sun-only). */
  emitterTrisCount: number;
  gridParamsBuf: GPUBuffer;
  frameParamsBuf: GPUBuffer;
  blendParamsBuf: GPUBuffer;
  borderVisUboBuf: GPUBuffer;
  rayResultsBuf: GPUBuffer;
  activeProbesBuf: GPUBuffer;
  linearSampler: GPUSampler;
}
