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
  /**
   * Wave 4 — HDRI into DDGI probe misses (2026-06-10).
   *
   * Env-map texture view for the `ddgiEnvMap` binding (@group(2) @binding(6)).
   * A 1×1 placeholder rgba16float view is always bound so the bind group is
   * valid even when no HDRI is loaded (hasEnv=0 in FrameParams gates the
   * sample path in WGSL — the placeholder is never actually read).
   *
   * Owned by ProbeUpdatePass (not the caller) only when `envMapOwnedByPass`
   * is true; callers that provide a long-lived external view set this false.
   */
  envMapView: GPUTextureView;
  /** Whether `envMapView` references a pass-owned placeholder texture that
   *  must be destroyed on dispose / view swap. */
  envMapOwnedByPass: boolean;
  /** Placeholder or caller-supplied texture backing `envMapView`. Null when
   *  the view was supplied externally (non-owned). Destroyed by dispose only
   *  when `envMapOwnedByPass` is true. */
  envMapPlaceholderTex: GPUTexture | null;
  /** Sampler for ddgiEnvMap (@group(2) @binding(7)). When the caller provides
   *  a view, it should also provide a matching sampler; otherwise the pass's
   *  own `linearSampler` is reused (clamp, linear). */
  envSamplerForProbe: GPUSampler;
}
