/**
 * GPU resource bundle for {@link ProbeUpdatePass} (W4c).
 */
import type { ProbeUpdateBvhGpuBuffers } from './probeUpdateBvhBuffers.js';

/**
 * D6.4 / D6-7 — Generic keyed bind-group cache for the probe-update passes.
 *
 * Bind groups are created once per unique resource-identity set and reused
 * until any participating resource changes (BVH rebuild, emitter update,
 * atlas resize, ping-pong swap, etc.). Each slot is keyed by an ordered list
 * of the exact buffer/texture/view object references its bind group binds —
 * an entry is rebuilt iff ANY key reference changes (`Object.is` per key).
 *
 * This replaces the former ~45-field flat struct + paired invalidation `if`s
 * (one comparison chain + one write-back per slot). The single keyed list per
 * slot makes the compare-set and the assign-set impossible to drift apart —
 * the drift-bug class this file documented.
 *
 * Slots (id → keyed on):
 *  - `raysG0`       BVH buffers (11 entries) — keyed on `bvhBuf` (all 11 rebuild atomically).
 *  - `raysG1`       materials/lights/emitters/atlas/tangent/color — keyed on all seven identities.
 *  - `raysG2`       per-frame — keyed on irrReadTex + rayResultsBuf +
 *                     activeProbesBuf + envMapView + envSamplerForProbe.
 *  - `classifyG0`   ray/state resources — keyed on rayResultsBuf +
 *                     activeProbesBuf + exact active-prefix byte length +
 *                     irrReadTex + irrWriteTex.
 *  - `blendIrrG0`   stable buffers — keyed on rayResultsBuf +
 *                     activeProbesBuf.
 *  - `blendIrrG1`   atlas textures — keyed on irrReadTex + irrWriteTex.
 *  - `blendVisG0`   stable buffers — keyed on rayResultsBuf +
 *                     activeProbesBuf.
 *  - `blendVisG1`   atlas textures — keyed on visReadTex + visWriteTex.
 *  - `borderG0`     per-atlas — keyed on scratchTex + writeAtlas.
 */
export interface DispatchBindGroupCacheEntry {
  keys: readonly unknown[];
  group: GPUBindGroup;
}

export type DispatchBindGroupCache = Map<string, DispatchBindGroupCacheEntry>;

/**
 * Return the cached bind group for `id` when every key in `keys` is
 * reference-identical to the cached entry's keys (same length + `Object.is`
 * per position); otherwise build a fresh group via `build`, cache it, and
 * return it. Preserves the exact "rebuild iff any keyed resource changed"
 * invalidation the probe-update passes rely on.
 */
export function getOrCreateBindGroup(
  cache: DispatchBindGroupCache,
  id: string,
  keys: readonly unknown[],
  build: () => GPUBindGroup,
): GPUBindGroup {
  const hit = cache.get(id);
  if (hit && hit.keys.length === keys.length && hit.keys.every((k, i) => Object.is(k, keys[i]))) {
    return hit.group;
  }
  const group = build();
  cache.set(id, { keys: [...keys], group });
  return group;
}

export interface ProbeUpdateGpuState extends ProbeUpdateBvhGpuBuffers {
  /** D6.4 — bind-group cache; allocated lazily on first dispatch. */
  bgCache?: DispatchBindGroupCache;
  device: GPUDevice;
  raysPipeline: GPUComputePipeline;
  classifyRelocatePipeline: GPUComputePipeline;
  blendIrrPipeline: GPUComputePipeline;
  blendVisPipeline: GPUComputePipeline;
  borderVisPipeline: GPUComputePipeline;   // irradiance is SH (seam-free) — no irr border pass
  visScratchTex: GPUTexture | null;
  traceParamsBuf: GPUBuffer;
  materialsBuf: GPUBuffer;
  lightsBuf: GPUBuffer;
  /** Allocated byte capacity of the runtime-sized DDGI light storage buffer. */
  lightsCapacityBytes: number;
  /** H18 Stage 2 — packed EmitterTri array for rect/disc area-emitter NEE in the
   *  probe-ray kernel. Matches the RC `rc_emitters` layout (5 × vec4f = 80 bytes
   *  per tri). A one-record dummy when emitterCount == 0 so the bind group is
   *  always valid under layout:auto storage-array validation. */
  emitterTrisBuf: GPUBuffer;
  /** Number of valid emitter triangles in emitterTrisBuf (0 when sun-only). */
  emitterTrisCount: number;
  /** DDGI-local copy of the atlas-backed material textures for probe-hit maps. */
  materialTextureAtlas: GPUTexture;
  materialTextureAtlasView: GPUTextureView;
  /** Per-triangle material-map metadata texture paired with {@link materialTextureAtlas}. */
  materialTextureAtlasMeta: GPUTexture;
  materialTextureAtlasMetaView: GPUTextureView;
  /** DDGI-local per-vertex authored/generated tangent.xyzw texture for probe-hit normal/bump TBN. */
  bvhTangentTexture: GPUTexture;
  bvhTangentTextureView: GPUTextureView;
  /** DDGI-local per-vertex COLOR_0 rgba texture for probe-hit base-color/alpha parity. */
  bvhVertexColorTexture: GPUTexture;
  bvhVertexColorTextureView: GPUTextureView;
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
