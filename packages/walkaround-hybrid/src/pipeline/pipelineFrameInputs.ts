/**
 * Per-frame input types for {@link WalkaroundGPUPipeline.renderFrame}.
 *
 * Extracted from `WalkaroundGPUPipeline.ts` (D3.5 decomposition, 2026-06-11)
 * to reduce that file's size and allow importers (e.g. `Pass.ts`, test harnesses)
 * to pull only these types without the full pipeline. Re-exported from
 * `WalkaroundGPUPipeline.ts` for back-compat.
 */

/** Camera matrices + position for one frame. */
interface PipelineFrameCamera {
  /** Camera view matrix (column-major mat4x4f, 16 floats). The pipeline
   *  composes VP = projMatrix * viewMatrix internally; do NOT pre-multiply. */
  viewMatrix: Float32Array;
  /** Camera projection matrix (column-major mat4x4f, 16 floats). */
  projMatrix: Float32Array;
  /** Previous-frame view-projection matrix (prevProj * prevView). Pass
   *  the current projection/view product on the first frame to avoid a
   *  one-frame ghost from uninitialized previous-frame state. */
  prevViewProjMatrix: Float32Array;
  /** World-space camera position [x, y, z]. */
  cameraPos: [number, number, number];
}

/** Swap-chain + frame-seed for one frame. */
interface PipelineFrameScreen {
  /** Render-target dimensions in pixels. Used by all compute kernels for
   *  workgroup dispatch sizing — must match the swap chain's actual size. */
  screenWidth: number;
  screenHeight: number;
  /** u32 frame counter / per-frame randomness seed. Drives PCG hash inits
   *  for ray jitter, RIS candidate sampling, and temporal reservoir update.
   *  Caller may use a frame index, performance.now()|0, or any monotone u32. */
  frameSeed: number;
  /** The WebGPU swap-chain texture view to render into for this frame.
   *  Caller must obtain via context.getCurrentTexture().createView()
   *  inside the same animation-frame callback that calls renderFrame. */
  swapChainView: GPUTextureView;
  /** The format of swapChainView. The composite pass's render-pipeline
   *  is recompiled if this changes (rare — usually fixed at canvas mount). */
  swapChainFormat: GPUTextureFormat;
}

/** Lighting scalars, emitter budget, and light-tree configuration. */
interface PipelineFrameLighting {
  /** Sum of (Le * area) over all emitter triangles, computed at BVH build
   *  time. Used by RIS importance-sampling weight normalization. Must match
   *  the value baked into the emitter CDF in SceneBVHBuffers. */
  totalEmissivePower: number;
  /** Number of entries in the emitter list (length of EmitterTri[] array
   *  in SceneBVHBuffers.emitters). Used by RIS to bound candidate selection. */
  emitterCount: number;
  /** Primary directional light direction [x, y, z] in world space, normalized.
   *  Today this is the sun for cathedral-window glass tracing; the field is
   *  named generically because the path tracer is light-source-agnostic. */
  primaryLightDir: [number, number, number];
  /** Primary directional light irradiance multiplier (linear, unitless).
   *  Uploaded to the shade-side WalkaroundUBO each frame (offset ~220) to
   *  scale Lo_emit on primary-panel hits in the glass/stained-glass path.
   *  NOTE (H21): directional/sun emitters NEVER enter the rect-area emitter
   *  CDF (`collectRectAreaEmitterTrisFromCore` collects only `rect-area` and
   *  `disc-area` kinds), so this field does NOT need to stay in sync with
   *  BVH-build time emitter baking. The "must match at BVH-build time" claim
   *  in earlier docs was an over-broad carry-over from the stained-glass path;
   *  `updateLighting({primaryLightIntensity})` correctly mutates it without
   *  rebuilding the emitter list. */
  primaryLightIntensity: number;
  /** Cone radius in radians for the direct sun sample. Derived from
   *  `DirectionalEmitter.angularDiameter / 2` when authored; otherwise the
   *  legacy real-sun radius is used. */
  sunAngularRadius?: number;
  /** 1 when the current scene contains contract-level transmission and the
   * generic bounded refractive-caustic estimator should run. */
  genericRefractiveCaustics?: number;
  /** Bounded Newton work for `causticStrategy:'manifold-nee'`. */
  mneeMaxIterations?: number;
  /** Maximum fixed-offset specular chain length (1..8). */
  mneeMaxChainLength?: number;
  /** Bounded SMS recurrence trials used for inverse-basin correction. */
  mneeMultiplicityTrials?: number;
  /** Diffuse-sky-dome RGB tint, derived from computeLightingState. Replaces
   *  four formerly-hardcoded sky tints in WGSL. Consumed by sky-aperture
   *  probe + second-bounce sky-miss paths. */
  skyTint: [number, number, number];
  /** Sky-dome irradiance scalar paired with skyTint. ~0.5×sun at noon. */
  skyIrradiance: number;
  /** Audit M12 — emitter geometry term dist² floor; default `0.01` for
   *  Cornell-scale; hosts on different scales should pass `(diag * 1e-3)²`. */
  emitterDist2Floor: number;
  /** Audit B4 — per-channel max HDR radiance clamp on the direct channel.
   *  Default 4.0 calibrated to Le=12. */
  directFireflyClamp: number;
  /** Audit B1 — stained-glass caustic boost. Cornell uses 22.0; generic
   *  scenes pass 1.0 (no boost). */
  causticBoost: number;
  /** Audit B1 — clamp applied to the tinted-visibility vector before the
   *  caustic-boost multiplication. Cornell uses 0.6; generic scenes pass 1.0. */
  causticVisClamp: number;
  /** Light-tree gate for ReSTIR-DI initial-candidate light SELECTION (UBO
   *  offset 356). `1` ⇒ the RIS candidate loop draws lights via the
   *  spatially-aware light tree (`sampleLightTree`) and divides the WRS weight
   *  by the tree selection pdf; `0` ⇒ RIS uses the flat power-CDF path
   *  (`sampleEmitterIdx` + the `emitterPmf` weight) verbatim. Built `1` only
   *  when the tree has ≥ 2 emitters AND the host left light-tree selection on;
   *  see `SceneBVHBuffers.lightTreeEnabled`. The estimator is unbiased in BOTH
   *  states because the WRS weight always divides p̂ by the EXACT pdf the
   *  selection used. */
  lightTreeEnabled: number;
  /** Number of nodes in the packed light tree (UBO offset 360). Bounds the
   *  `sampleLightTree` descent loop. `0` when the tree is disabled. */
  lightTreeNodeCount: number;
}

/** ReSTIR-DI temporal + spatial reuse tuning knobs. */
interface PipelineFrameRestirDI {
  /** Audit M6 — ReSTIR-DI temporal M-clamp; Cornell default 20. */
  temporalMClampDI: number;
  /** Audit M7 — ReSTIR-DI spatial reuse radius in pixels; Cornell default 30. */
  spatialReuseRadiusPx: number;
  /** Audit M8 — ReSTIR-DI spatial depth-tolerance world-units floor; Cornell
   *  default 0.05 (5 cm). Hosts on different scales should pass
   *  `sceneDiagonal * 1e-3`. */
  spatialDepthTolFloor: number;
}

/** ReSTIR-GI / GRIS tuning + reuse gate. */
interface PipelineFrameRestirGI {
  /** 2026-05-18 sweep — ReSTIR-GI per-pixel unbiased weight cap (risGi,
   *  spatialGi). Cornell default 16.0. */
  restirGiWCap: number;
  /** 2026-05-18 sweep — DDGI irradiance clamp applied at the ReSTIR-GI
   *  reconnection vertex (risGi). Cornell default 5.0. */
  restirGiIrrClamp: number;
  /** 2026-05-18 sweep — ReSTIR-GI temporal previous-frame M clamp
   *  (temporalGi). Cornell default 50. */
  restirGiMClamp: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse disc radius (half-res
   *  pixels). Cornell default 12.0. */
  restirGiSpatialRadiusPx: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse normal-alignment cosine
   *  minimum (spatialGi). Cornell default 0.906 ≈ cos(25°). */
  restirGiSpatialNormalDotMin: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse tangent-plane distance
   *  tolerance (spatialGi). Cornell default 0.05 (5 cm world units). */
  restirGiSpatialCoplanarTol: number;
  /** GRIS DDGI-proxy reconnection-shift reuse gate (UBO offset 412). `1` ⇒
   *  the GI spatial + temporal reuse passes apply the bounded GRIS DDGI-proxy
   *  reconnection shift, its change-of-variables Jacobian, a reconnection-
   *  visibility ray, and the bounded all-technique transformed-density balance MIS (Lin et al.
   *  2022). `0`/omitted ⇒ the reuse runs the legacy clamped-Jacobian path
   *  BIT-FOR-BIT (the GRIS branch is gated behind `ubo.grisReuse == 1`).
   *  Host opt-in via HybridEngineOptions.grisReuse — the same
   *  OFF-is-bit-identical pattern as RC/PPG/ReGIR. */
  grisReuse?: number;
}

/** GTAO + adaptive-sampling tuning knobs. */
interface PipelineFrameGtao {
  /** Audit M1 — GTAO sampling radius in pixels; Cornell default 32. */
  gtaoRadiusPx: number;
  /** Audit M1 — GTAO intensity exponent; Cornell default 2.0. */
  gtaoIntensity: number;
  /** Audit M1 — GTAO depth threshold in world units; Cornell default 2.0. */
  gtaoDepthThreshold: number;
  /** Audit B3 — GTAO upsample bilateral depth sigma (world units);
   *  Cornell default 0.25 (= 1/√(2*4); see legacy `exp(-Δ * 4)`). */
  gtaoBilateralDepthSigma: number;
  /** Audit M2 — adaptive-sampling tier classifier low-variance threshold;
   *  Cornell default 0.01.  Variance below this → tier 1 (converged). */
  adaptiveSamplingThresholdLow: number;
  /** Audit M2 — adaptive-sampling tier classifier high-variance threshold;
   *  Cornell default 0.10.  Variance above this → tier 4 (high noise). */
  adaptiveSamplingThresholdHigh: number;
}

/** Denoiser filter parameters, firefly clamps, and stained-glass gate. */
export interface PipelineFrameFilter {
  /** D12 — Möller-Trumbore coplanarity epsilon.  Controls the `abs(det) < ε`
   *  near-zero determinant threshold in `intersectTriangle`.  Default `1e-5`
   *  (metre-scale).  Reduce for millimetre-scale geometry. */
  triIntersectEpsilon: number;
  /** 2026-05-18 sweep — probe-side glass-transmission perceptual mix scale.
   *  Cornell default 0.7. */
  glassMixScale: number;
  /** 2026-05-18 sweep — per-channel HDR clamp on the indirect channel
   *  (shade.wgsl). Cornell default [1.0, 1.0, 1.0]. */
  indirectFireflyClamp: readonly [number, number, number];
  /** 2026-05-19 B3a — atrous DIRECT-channel sigmas [sigmaN, sigmaZ, sigmaC].
   *  Cornell default `[128.0, 5.0, 0.05]`. Consumed by the AtrousDenoiser
   *  direct-path chain. */
  atrousDirectSigmas: readonly [number, number, number];
  /** 2026-05-19 B3a — atrous INDIRECT-channel sigmas [sigmaN, sigmaZ, sigmaC].
   *  Cornell default `[32.0, 20.0, 0.5]`. Consumed by AtrousIndirectPass. */
  atrousIndirectSigmas: readonly [number, number, number];
  /** T5 / SHADOW-01 — shade flag bitfield (lands at UBO offset 344).
   *  Bit 0 = sun-caustic enabled, bit 1 = sky-aperture enabled, bit 2 = direct
   *  sun visibility disabled for a scene directional emitter with
   *  `castShadow:false`. Default 0 keeps generic scenes on the standard path. */
  stainedGlassFlags: number;
}

/** BVH traversal mode + TLAS configuration. */
interface PipelineFrameBvh {
  /** PR-3 — 0 = merged world BVH, 1 = TLAS + local BLAS traversal. */
  bvhMode: number;
  /** PR-3 — TLAS node count from CPU pack (0 forces merged path in WGSL). */
  tlasNodeCount: number;
}

/** Optional per-frame NRC gate. */
interface PipelineFrameNrc {
  /** NRC (Müller et al. 2021) cache gate (UBO offset 364 — the former
   *  `_ppgPad2` slot). `1` ⇒ the GI suffix may TERMINATE into the learned
   *  neural radiance cache (spread heuristic + MLP query) and radiance records
   *  self-train it. `0`/omitted ⇒ the gi-ris suffix runs the verbatim
   *  DDGI-atlas estimate and the UBO bytes are unchanged — **OFF is
   *  bit-identical**. Host opt-in via HybridEngineOptions.nrcEnabled; FORBIDDEN
   *  on tier:'lite'. NRC is a BIASED cache (not a converged-mean-preserving
   *  reuse) — see HARDWARE-VALIDATION-NEEDS.md V20. WIRED (2026-05-29): the gi-ris
   *  NRC variant runs the MLP query + writes self-training records;
   *  `NrcSubsystem.trainFromRecords` runs an MLP `trainStep` AND the hash-grid
   *  encode-backward + table Adam each frame — so with the gate at 1 the suffix
   *  uses the (biased) learned MLP prediction when the spread heuristic fires. */
  nrcEnabled?: number;
}

/** Per-frame tonemap / exposure / output-colorspace dials (2026-06-10). */
interface PipelineFrameComposite {
  /** Tonemap operator mode index — matches TONEMAP_MODE_INDEX from
   *  @vitrum/shared-samplers: 0=aces(default) 1=agx 2=reinhard 3=linear 4=none. */
  tonemapMode: number;
  /** Linear-exposure multiplier applied before the tonemap operator. Default: 1.0. */
  exposure: number;
  /** Output color space: 0 = srgb (default, OETF applied), 1 = linear (OETF skipped). */
  outputColorSpace: number;
}

/**
 * Per-frame inputs to {@link WalkaroundGPUPipeline.renderFrame}.
 *
 * Fields are grouped into named sub-objects so each sprint's new field
 * lands in the right semantic bucket rather than growing a flat list.
 * UBO byte layout is unchanged — the sub-objects are TypeScript-only;
 * `uboUpdater.ts` unpacks them field-by-field as before.
 */
export interface PipelineFrameInputs {
  /** Camera matrices and world-space position. */
  camera: PipelineFrameCamera;
  /** Swap-chain targets and per-frame seed. */
  screen: PipelineFrameScreen;
  /** Lighting scalars, emitter budget, and light-tree gate. */
  lighting: PipelineFrameLighting;
  /** ReSTIR-DI temporal + spatial reuse tuning. */
  restirDI: PipelineFrameRestirDI;
  /** ReSTIR-GI / GRIS tuning and reconnection gate. */
  restirGI: PipelineFrameRestirGI;
  /** GTAO + adaptive-sampling tuning. */
  gtao: PipelineFrameGtao;
  /** Denoiser filter parameters, firefly clamps, and stained-glass gate. */
  filter: PipelineFrameFilter;
  /** BVH traversal mode and TLAS configuration. */
  bvh: PipelineFrameBvh;
  /** NRC cache gate (optional; absent ⇒ OFF, bit-identical). */
  nrc: PipelineFrameNrc;
  /** Per-frame tonemap / exposure / output-colorspace dials (2026-06-10). */
  composite: PipelineFrameComposite;
}
