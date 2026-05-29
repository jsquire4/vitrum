/**
 * Walkaround per-frame uniform layout + global constants.
 *
 * Split out of common.wgsl.ts (T9-stepA): the canonical `WalkaroundUBO`
 * struct shared by every ReSTIR compute pass (ris/temporal/spatial/shade),
 * the global PI/INV_PI/INFINITY/BVH constants, and the `emitterGeometry`
 * geometry-term helper whose dist² floor is read from the UBO (audit M12).
 *
 * Layout offsets are pinned; host-side packers (pipeline/uboUpdater.ts) rely
 * on them. Bump documentation here if the layout changes.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const WALKAROUND_UBO_WGSL = /* wgsl */ `

// ============================================================
// Constants
// ============================================================
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
const INFINITY = 1e20;
const BVH_STACK_DEPTH = 60u;
const LEAFNODE_FLAG = 0xFFFF0000u;

// Audit M12 follow-up: the emitter-geometry-term distance² floor is now
// a runtime UBO value (ubo.emitterDist2Floor) rather than a compile-
// time constant. Hosts set it via HybridEngineOptions.emitterDist2Floor
// and a sensible default is computed from the scene AABB:
//   max(sceneDiag × 1e-3, 0.0001)²
// Cornell preserves 0.01 (≈10 cm minimum effective distance).
//
// CRITICAL: the floor must be applied identically in BOTH the RIS
// reservoir construction (ris.wgsl computePHat + M_LIGHT loop) AND
// shade direct-light evaluation (shade.wgsl) so the importance-sampled
// pHat matches the evaluated pHat. Both call:
//   emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor)
// with the same UBO value.

// Sprint 18 follow-up — ReSTIR-GI per-pixel unbiased weight (W) cap.
//
// W = w_sum / (M · p̂(z)) is unbounded when the chosen sample's p̂ at the
// visible point is small (grazing cosTheta from the visible normal to the
// reconnection direction, or near-zero luminance after a sky-miss bounce).
// A single pixel with a tiny p̂ produces a huge spike that:
//   1. Passes through the per-channel atrous-indirect chain (the spike is
//      a 1-pixel impulse, which the 5×5 kernel only attenuates to ~1/13 of
//      its peak per step; spread across 4 iterations the spike becomes a
//      multi-pixel halo rather than disappearing).
//   2. Defeats the temporal accumulator: alpha=0.02 means the spike's
//      contribution to the running average is 2% per frame, but new spikes
//      arrive on different pixels every ~5 frames, so the average never
//      settles to a stable fixed point — the user sees a never-converging
//      image with shifting bright dots and wavy bands.
//
// Capping W at 16 keeps single-sample contribution bounded without biasing
// the indirect magnitude.  Previous tighter cap of 4 was masking a real
// firefly source (near-light DDGI atlas spikes) while also truncating the
// heavy tail of legitimately bright samples — the result was a uniformly
// dim indirect signal (~17× too low on Cornell white walls) AND visible
// fireflies, because the cap reduced spike *amplitude* without addressing
// spike *prevalence*.  The proper fix is upstream: cap irrAtXs at the
// reconnection vertex (see risGi.wgsl).  W can then breathe at 16, which
// admits the legitimate variance the unbiased estimator needs while still
// bounding pathological cases for variance-bounded convergence.
//
// 2026-05-18 sweep — the Cornell-tuned cap value (16.0) now lives on the
// WalkaroundUBO as restirGiWCap. Library consumers on different scene
// scales override it via HybridEngineOptions.restirGiWCap.

// ============================================================
// WalkaroundUBO — canonical per-frame uniform layout shared by every
// ReSTIR compute pass (ris/temporal/spatial/shade). Defined here so
// the four passes do not drift; each shader binds with:
//   @group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
// (Atrous/welford/svgf do not bind @group(2); the struct definition is
// inert in those modules.)
//
// Layout offsets are pinned and host-side packers (pipeline/uboUpdater.ts)
// rely on them. Bump documentation if the layout changes.
// ============================================================
struct WalkaroundUBO {
  viewMatrix:                 mat4x4f, //  offset 0
  projMatrix:                 mat4x4f, //  offset 64
  prevViewMatrix:             mat4x4f, //  offset 128
  cameraPos:                  vec3f,   //  offset 192
  frameSeed:                  u32,     //  offset 204
  screenSize:                 vec2u,   //  offset 208
  emitterCount:               u32,     //  offset 216
  totalEmPower:               f32,     //  offset 220
  sunDirection:               vec3f,   //  offset 224
  sunIntensity:               f32,     //  offset 236 — matches BVH build
  skyTint:                    vec3f,   //  offset 240 — diffuse sky dome RGB
  skyIrradiance:              f32,     //  offset 252 — sky dome brightness
  // Library-generality tunables (audit follow-up). All scalar f32/u32
  // packed tightly after the existing 256-byte block. Defaults preserve
  // Cornell behaviour; hosts override via HybridEngineOptions.
  emitterDist2Floor:          f32,     //  offset 256 — audit M12
  directFireflyClamp:         f32,     //  offset 260 — audit B4
  causticBoost:               f32,     //  offset 264 — audit B1
  causticVisClamp:            f32,     //  offset 268 — audit B1
  temporalMClampDI:           u32,     //  offset 272 — audit M6
  spatialReuseRadiusPx:       f32,     //  offset 276 — audit M7
  spatialDepthTolFloor:       f32,     //  offset 280 — audit M8
  triIntersectEpsilon:        f32,     //  offset 284 — D12: Möller-Trumbore coplanarity floor
  // 2026-05-18 sweep — eight more Cornell-tuned magic constants migrated
  // from WGSL kernels into the UBO so hosts can override per scene.
  glassMixScale:              f32,     //  offset 288 — probeUpdateRays glass-transmission perceptual mix
  restirGiWCap:               f32,     //  offset 292 — risGi/spatialGi unbiased-weight cap
  restirGiIrrClamp:           f32,     //  offset 296 — risGi DDGI irradiance read clamp
  restirGiMClamp:             u32,     //  offset 300 — temporalGi prev-frame M-clamp
  restirGiSpatialRadiusPx:    f32,     //  offset 304 — spatialGi disc-sample radius (half-res pixels)
  restirGiSpatialNormalDotMin:f32,     //  offset 308 — spatialGi normal-alignment cosine min
  restirGiSpatialCoplanarTol: f32,     //  offset 312 — spatialGi tangent-plane distance tolerance
  _padPreVec3:                f32,     //  offset 316 — pad to align vec3f to 16-byte boundary
  indirectFireflyClamp:       vec3f,   //  offset 320 — shade indirect channel per-channel HDR clamp
  _padEnd:                    f32,     //  offset 332 — align vec3 to 336
  bvhMode:                    u32,     //  offset 336 — 0 merged world BVH, 1 TLAS+local BLAS
  tlasNodeCount:              u32,     //  offset 340 — TLAS node count (0 → merged path)
  // T5 — stained-glass opt-in flags (repurposed pad slot; byte-size unchanged).
  // Bit 0 = sun-caustic enabled, bit 1 = sky-aperture enabled. Default 0 (both
  // OFF) → a generic scene gets ZERO stained-glass caustic/aperture physics.
  // Hosts opt in via HybridEngineOptions.stainedGlass; the bits gate the
  // early-return in lo_sg_caustic / lo_sg_aperture (stainedGlassShade.wgsl.ts).
  stainedGlassFlags:          u32,     //  offset 344 — T5 (was _tracePad0)
  // PPG (Müller 2017) guided-sampling controls (W9 guided-sampling landing).
  // ppgEnabled is the runtime gate the gi-ris RIS source-pdf mixture reads:
  // when 0, the Bernoulli mixing weight α collapses to 0 so the gi-ris
  // candidate loop samples a pure cosine hemisphere and the explicit RIS
  // weight reduces EXACTLY to today's luminance(Lo) cosine shortcut
  // (ppg-OFF bit-identity). When 1, gi-ris draws α-fraction of candidates
  // from the learned dTree and evaluates the defensive mixture pdf
  // p_src = α·p_guide + (1−α)·p_cos for an unbiased estimator (§3.4).
  ppgEnabled:                 u32,     //  offset 348 — PPG guided-sampling gate (was _tracePad1)
  ppgMixAlpha:                f32,     //  offset 352 — MIS mixing weight α ∈ [0,1] (Müller §3.4)
  // Light-tree DI light-SELECTION gate (was _ppgPad0/_ppgPad1). When
  // lightTreeEnabled == 1 the RIS candidate loop draws emitters via the
  // spatially-aware light tree (sampleLightTree) and uses the tree selection
  // pdf as the WRS source pmf; when 0 it falls back to the flat power-CDF path
  // (sampleEmitterIdx). Either way the estimator is unbiased — the WRS weight
  // divides p̂ by the EXACT pdf the selection used. lightTreeNodeCount bounds
  // the GPU descent loop.
  lightTreeEnabled:           u32,     //  offset 356 — DI light-tree selection gate (was _ppgPad0)
  lightTreeNodeCount:         u32,     //  offset 360 — packed light-tree node count (was _ppgPad1)
  _ppgPad2:                   u32,     //  offset 364 — pad to 16-byte boundary at 368
  // ── ReGIR (Boksansky 2021 grid-based reservoirs) — DI light-SELECTION
  // grid that decouples per-pixel cost from light count. The grid is co-located
  // in the SAME @group(3) light-tree storage buffer (RIS stays at 16 storage
  // buffers); the grid-build pass writes it, RIS samples from the containing
  // cell instead of traversing the tree per pixel. regirEnabled gates the path:
  // 0 ⇒ RIS uses the light-tree path bit-identically (the grid-build pass also
  // early-returns). Unbiased: RIS divides p̂ by the EXACT per-cell selection pmf
  // (q̂_c(e)/Ŝ) the grid stored — same discipline as the tree pmf. See regir.wgsl.
  regirOrigin:                vec3f,   //  offset 368 — world-space grid AABB min (vec3f 16-aligned)
  regirInvCellSize:           f32,     //  offset 380 — 1 / cellSize (uniform cubic cells)
  regirDims:                  vec3u,   //  offset 384 — grid cell counts (x, y, z)
  regirEnabled:               u32,     //  offset 396 — ReGIR DI-selection gate (0 ⇒ light-tree fallback)
  regirCandidatesPerCell:     u32,     //  offset 400 — M: WRS candidates per sub-reservoir
  regirSurvivorsPerCell:      u32,     //  offset 404 — K: survivors stored per cell
  regirGridFloatOffset:       u32,     //  offset 408 — float offset of the grid region in the combined buffer
  _regirPad:                  u32,     //  offset 412 — struct size 416 bytes (416 % 16 == 0)
};

// T5 — stained-glass opt-in flag bit masks. Bit 0 gates the sun-caustic term,
// bit 1 gates the sky-aperture term. Mirrors the host-side packStainedGlassFlags
// helper in pipeline/uboUpdater.ts (the two MUST agree on bit positions).
const SG_FLAG_SUN_CAUSTIC: u32 = 1u;   // bit 0
const SG_FLAG_SKY_APERTURE: u32 = 2u;  // bit 1

// Emitter geometry term G with a configurable dist² clamp applied at
// every call site. Use this everywhere instead of inlining
// nlDotL / dist² directly.  The floor is now read from the WalkaroundUBO
// (audit M12) rather than the old scene-scale-baked constant.
fn emitterGeometry(nlDotL: f32, dist2: f32, dist2Floor: f32) -> f32 {
  let dist2_clamped = max(dist2, dist2Floor);
  return nlDotL / dist2_clamped;
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const WALKAROUND_UBO_MODULE: WgslModule = {
  name: "walkaroundUbo",
  source: WALKAROUND_UBO_WGSL,
  requires: [],
};
