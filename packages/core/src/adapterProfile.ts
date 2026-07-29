// @vitrum/core — graceful-degradation adapter capability report.
//
// Pure DATA shape only. The actual probe that fills this in
// (`probeAdapterProfile`) lives in `@vitrum/engine`, because computing the
// capability booleans requires the real limit thresholds that live in
// `@vitrum/walkaround-hybrid` (`HYBRID_WEBGPU_REQUIRED_LIMITS` /
// `HYBRID_LITE_LIMITS`) and `@vitrum/pt-webgpu` (PT_WEBGPU_* consts) — packages
// that `@vitrum/core` deliberately does NOT depend on. Keeping the *type* here
// honors design principle #1 ("the contract is the thing that's fixed"): hosts
// can type-annotate an `AdapterProfile` without pulling `@vitrum/engine` in.

import type { WgpuAdapterKind } from './wgpuSupport.js';

/** Recommended realtime preset ceiling for an adapter (roadmap Class A–E).
 *  This is a *ceiling*, not a lock — a host can always downshift via an
 *  explicit `qualityTier`. `'unavailable'` means the hybrid realtime engine
 *  must not be initialized on this adapter (Class D/E — software rasterizer or
 *  below-lite limits); the host should fall back to a path-tracer backend. */
export type RealtimeTier = 'ultra' | 'high' | 'medium' | 'low' | 'unavailable';

/** Recommended converged-/hero-render backend for an adapter. */
export type HeroBackendRec = 'pt-webgpu-full' | 'pt-webgpu-lite' | 'pt-webgl2' | 'none';

/** pt-webgpu trace-tier recommendation derived from adapter limits. */
export type PtWebgpuTierRec = 'full' | 'lite' | 'none';

/**
 * Graceful-degradation adapter capability report. Produced by
 * `probeAdapterProfile()` (exported from `@vitrum/engine`). Pure data — every
 * field is derived from `navigator.gpu` adapter/device limits + adapter info,
 * compared against the real backend limit thresholds. No GPU work happens when
 * reading this; the probe runs once and the host caches the result.
 */
export interface AdapterProfile {
  /** True when `navigator.gpu` exposed an adapter. */
  readonly hasWebGPU: boolean;
  /** Meets the current `HYBRID_WEBGPU_REQUIRED_LIMITS` — full hybrid path. */
  readonly hybridCapable: boolean;
  /** Meets the reduced `HYBRID_LITE_LIMITS` (Deliverable 3) but not full.
   *  When this is false the hybrid realtime engine is unavailable on the
   *  adapter (Class D). */
  readonly hybridLiteCapable: boolean;
  /** pt-webgpu trace tier the adapter can satisfy. */
  readonly ptWebgpuTier: PtWebgpuTierRec;
  /** Adapter `maxStorageBuffersPerShaderStage` (the dominant hybrid limit). */
  readonly maxStorageBuffersPerStage: number;
  /** Adapter `maxStorageTexturesPerShaderStage` (the dominant hybrid limit). */
  readonly maxStorageTexturesPerStage: number;
  /** SwiftShader-class heuristic (vendor==='google' && arch==='swiftshader').
   *  Software adapters never initialize the hybrid engine (§4.4/§10.4). */
  readonly isSoftwareAdapter: boolean;
  /** Three-state hardware verdict re-exported from `wgpuSupport`. */
  readonly adapterKind: WgpuAdapterKind;
  /** Whether WebGL2 is available as a fallback path (drives `recommendedHeroBackend`
   *  when WebGPU is absent). */
  readonly hasWebGL2: boolean;
  /** Preset ceiling for the realtime hybrid engine. */
  readonly recommendedRealtimeTier: RealtimeTier;
  /** Recommended converged-/hero-render backend. */
  readonly recommendedHeroBackend: HeroBackendRec;
  /** Raw limit bag (forward-compat; lets hosts read anything not surfaced
   *  above without a second probe). */
  readonly limits: Readonly<Record<string, number>>;
}
