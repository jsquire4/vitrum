/**
 * GPU resource bundle for {@link ProbeUpdatePass} (W4c).
 */
import type { ProbeUpdateBvhGpuBuffers } from './probeUpdateBvhBuffers.js';

/**
 * D6.4 — Bind-group cache entry for the probe-update passes.
 *
 * Bind groups are created once per unique resource-identity set and reused
 * until any participating resource changes (BVH rebuild, emitter update,
 * atlas resize, etc.). The cache is keyed on the buffer/texture object
 * references that appear in each bind group.
 *
 * Layout:
 *  - raysGroup0 / raysGroup1: stable (keyed on bvhBuf + materialsBuf + material atlas views)
 *  - raysGroup2:              per-frame (keyed on irrReadTex + rayResultsBuf + envMapView)
 *  - blendIrrGroup0/1:        stable/per-frame (keyed on rayResultsBuf / irrRead+irrWriteTex)
 *  - blendVisGroup0/1:        stable/per-frame
 *  - borderGroup0:            per-atlas (keyed on scratchTex + visWriteTex)
 */
export interface DispatchBindGroupCache {
  // Rays pass — group 0: BVH buffers (11 entries). Epoch key = bvhBuf.
  raysG0Key: GPUBuffer | null;
  raysG0: GPUBindGroup | null;
  // Rays pass — group 1: materials + lights + emitters + material atlas.
  // Key = materialsBuf + emitterTrisBuf + atlas views.
  raysG1Key0: GPUBuffer | null;
  raysG1Key1: GPUBuffer | null;
  raysG1KeyAtlas: GPUTextureView | null;
  raysG1KeyAtlasMeta: GPUTextureView | null;
  raysG1: GPUBindGroup | null;
  // Rays pass — group 2: per-frame (changes every atlas swap).
  // Key = irrReadTex + rayResultsBuf + activeProbesBuf + envMapView.
  raysG2KeyTex: GPUTexture | null;
  raysG2KeyBuf: GPUBuffer | null;
  raysG2KeyProbes: GPUBuffer | null;
  raysG2KeyEnv: GPUTextureView | null;
  raysG2IrrView: GPUTextureView | null;
  raysG2: GPUBindGroup | null;
  // Blend irr — group 0: buffers (stable). Key = rayResultsBuf + activeProbesBuf.
  blendIrrG0Key: GPUBuffer | null;
  blendIrrG0KeyProbes: GPUBuffer | null;
  blendIrrG0: GPUBindGroup | null;
  // Blend irr — group 1: atlas textures (per-frame). Key = irrReadTex + irrWriteTex.
  blendIrrG1KeyRead: GPUTexture | null;
  blendIrrG1KeyWrite: GPUTexture | null;
  blendIrrG1ReadView: GPUTextureView | null;
  blendIrrG1WriteView: GPUTextureView | null;
  blendIrrG1: GPUBindGroup | null;
  // Blend vis — group 0: buffers (stable). Key = rayResultsBuf + activeProbesBuf.
  blendVisG0Key: GPUBuffer | null;
  blendVisG0KeyProbes: GPUBuffer | null;
  blendVisG0: GPUBindGroup | null;
  // Blend vis — group 1: atlas textures (per-frame). Key = visReadTex + visWriteTex.
  blendVisG1KeyRead: GPUTexture | null;
  blendVisG1KeyWrite: GPUTexture | null;
  blendVisG1ReadView: GPUTextureView | null;
  blendVisG1WriteView: GPUTextureView | null;
  blendVisG1: GPUBindGroup | null;
  // Border vis — group 0: per-atlas scratch + write. Key = scratchTex + writeAtlas.
  borderG0KeyScratch: GPUTexture | null;
  borderG0KeyWrite: GPUTexture | null;
  borderG0ScratchView: GPUTextureView | null;
  borderG0WriteView: GPUTextureView | null;
  borderG0: GPUBindGroup | null;
}

export interface ProbeUpdateGpuState extends ProbeUpdateBvhGpuBuffers {
  /** D6.4 — bind-group cache; allocated lazily on first dispatch. */
  bgCache?: DispatchBindGroupCache;
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
  /** DDGI-local copy of the readable material texture atlas for probe-hit emission maps. */
  materialTextureAtlas: GPUTexture;
  materialTextureAtlasView: GPUTextureView;
  /** Per-triangle material-map metadata texture paired with {@link materialTextureAtlas}. */
  materialTextureAtlasMeta: GPUTexture;
  materialTextureAtlasMetaView: GPUTextureView;
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
