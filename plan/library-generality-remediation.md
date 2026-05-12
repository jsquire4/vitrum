# Library generality remediation plan

**Status**: open, work-in-progress
**Created**: 2026-05-11
**Source**: three parallel audits run on 2026-05-11 (WGSL-shader generality, host-pipeline generality, recent-commits code-quality). Plus an empirical reservoir probe on the live Cornell walkaround engine.

## Scope

`walkaround-hybrid` was developed iteratively against the Cornell box test scene and inherited additional Cornell-specific calibration from the Phase-6 stained-glass scene work. The library is shipped as part of `vitrum`, a generalised path-tracing/GI engine; magic numbers tuned to "Cornell at Le=12 in a 2×2 m room" do not belong in shipping library code.

This plan tracks the findings from the audit pass and which have landed vs which remain.

## Severity rubric

- **blocker** — library produces visibly wrong output for any scene ≥2× different in scale or intensity from Cornell.
- **major** — degrades visibly for moderately different scenes.
- **minor** — small bias / cosmetic / dead code.

---

## Landed in `956cab0` (salvaged from remediation worktree, audit fixes that didn't conflict with main)

| ID | Status | What |
|---|---|---|
| B2 | ✅ landed | `probeUpdateRays.wgsl` `sampleSkyColor` now reads `frameParams.skyTint × .skyIrradiance` instead of the hardcoded daylight gradient |
| M9 | ✅ landed | `DDGI_MAX_MATERIALS` is now compile-time-injectable via `makeProbeUpdateRaysWGSL(maxMaterials)`; runtime warning when scene exceeds the cap |
| M11 | ✅ landed | `ProbeGrid.computeFromBounds` accepts `maxProbesPerAxis` override; wired through `DDGI` → `HybridEngine` |
| M13 | ✅ landed | Probe-pass `NORMAL_BIAS` now `gridParams.spacing × 0.001` instead of hardcoded `0.02` |
| M14 | ✅ landed | Glass-step in `probeUpdateRays.wgsl` and `probeRayCast.wgsl` now `gridParams.spacing × 0.01` (and a configurable `slabStepSize` parameter) instead of hardcoded `0.5 m` |
| Q1 | ✅ landed | `plan/phase-6-status.md` — Sprints 9/10a/11 moved from "deferred" to "complete" |
| (M6/M7/M8 partial) | ✅ landed | minor sigma + depth-tolerance cleanup in `spatial.wgsl` / `temporal.wgsl` |

## Landed in this batch (cleanup + correctness surgery on main HEAD)

| ID | Status | What |
|---|---|---|
| B5 | ✅ landed | dropped the empirical `* 0.08` trim on `Lo_skyAperture` in `shade.wgsl`. The upstream `skyVisScalar × skyTint × skyIrradiance × albedo × INV_PI` is already the correct outgoing radiance for a Lambertian receiver under the sky dome |
| B6 | ✅ landed | reservoir copy is no longer a separate command-buffer submission. Folded `copyBufferToBuffer` calls into the main `encoder` before `queue.submit`, eliminating the temporal-reservoir race at high FPS |
| B7 | ✅ landed | `HybridEngine` now throws `TypeError` at construction if `denoiser` is anything other than `'atrous' | 'svgf'`. The `@vitrum/core` contract advertises additional modes ('none', 'bmfr', 'oidn-final') that this backend does not implement; previously they silently coerced to SVGF |
| Q2 | ✅ landed | `WalkaroundGPUPipeline.ts` tier-output comment corrected — risGi.wgsl consumes the tier |
| Q3 | ✅ landed | `timestampQueries.ts` doc-table updated for post-Sprint-18-follow-up slot counts (24/25/26/27) |
| Q4 | ✅ landed | dead `_prevAccum`/`_gNormalDepth`/`_motionVectors` params removed from `buildSVGFVarianceBindGroup` |
| Q5 | ✅ landed | `compositeLinearSampler` renamed to `compositeSampler` (it was `nearest`, not `linear`); dead `void PPG_DIRECTIONS` import removed |

---

## Still open — substantial UBO / `HybridEngineOptions` work

These need real API design + UBO field additions + host-side plumbing. They were *not* landed tonight because the work depends on a coherent `HybridEngineOptions` surface that should be designed in one pass rather than incrementally accumulated. Each item lists the audit reference, the current scene-dependence, and the suggested remediation pattern.

### Blockers

#### B1 — `shade.wgsl:254–256` `CAUSTIC_BOOST = 22.0` + `visClamp = 0.6`

A 22× direct-sun multiplier compensating for Brown-Beer-Lambert attenuation in the Cornell stained-glass scene. For scenes without glass, the path is multiplied by 0 and harmless; for scenes *with* glass but a different sun intensity, the boost produces a physically unjustified contribution.

**Remediation**: expose `HybridEngineOptions.caustic?: { boost?: number; visClamp?: number }`. Defaults `1.0` / `1.0` (no boost, no clamp). Plumb through `WalkaroundUBO` (`ubo.causticBoost`, `ubo.causticVisClamp`). Cornell stained-glass scenes opt in explicitly.

#### B3 — `gtaoUpsample.wgsl:27` `exp(-depthDelta * 4.0)`

σ=0.25 m bilateral depth weight, half-scale of Cornell. 10× larger scenes → no edge stopping. 10× smaller → all taps rejected.

**Remediation**: add `bilateralDepthSigma` field to the GTAO-upsample UBO (`GtaoUpsampleUBO`). Host computes default from scene AABB (1% of diagonal). Expose `HybridEngineOptions.gtao.bilateralDepthSigma?: number` override.

#### B4 — `shade.wgsl:431` `min(directRadiance, vec3f(4.0))`

Firefly clamp comment is explicitly "for Le=12". Scenes with brighter emitters (sun, automotive headlamps, stage lights) get silently clipped to 4.0.

**Remediation**: add `directFireflyClamp` field to `WalkaroundUBO`. Host computes default as `4.0 × luminance(maxEmitterLe)` after building the emitter buffer. Expose `HybridEngineOptions.directFireflyClamp?: number` override.

#### B8 — `WalkaroundGPUPipeline.ts:148` `CAMERA_MOVE_RESET_THRESHOLD_SQ = 1.0`

1.0 world-units² is fine for Cornell (~2 m room). For 100 m scenes the camera must physically jump a full metre before history resets → permanent ghosting. For micro-scale scenes the threshold is never crossed → temporal accumulator never resets.

**Remediation**: expose `HybridEngineOptions.cameraMoveResetThreshold?: number`. Default `1.0` for backwards compatibility; document the scene-scale dependency.

### Majors

#### M1 — GTAO uniforms `radiusPx=32`, `intensity=2`, `depthThresh=2.0`

`depthThresh=2.0` is in world units and tuned to Cornell. `radiusPx=32` is resolution-dependent. Both should be configurable + scene-derived.

**Remediation**: `HybridEngineOptions.gtao?: { radiusPx?, intensity?, depthThresholdWorldUnits?, bilateralDepthSigma? }` with defaults derived from scene AABB / render-target dimensions.

#### M2 — sample-budget tier thresholds `0.01 / 0.10`

Tuned to Cornell's variance dynamic range. HDR scenes will permanently classify every pixel to tier 4; low-light scenes to tier 1.

**Remediation**: either normalise variance by current frame's peak radiance before thresholding, or expose `HybridEngineOptions.adaptiveSamplingThresholds?: [low: number, high: number]`.

#### M3 — temporal accumulator α=0.01

Framerate-dependent. At 30 FPS doubles temporal lag; at 120 FPS halves it.

**Remediation**: expose `HybridEngineOptions.temporalAccumAlpha?: number`. Optionally derive default as `1 − exp(−dt × k)` so the temporal *feel* is FPS-independent.

#### M4 — `TARGET_FRAME_INTERVAL_MS = 1000/60 − 1` (60 FPS hard cap)

VR (90 Hz) and high-refresh desktop (120 Hz) hosts are throttled to 60 FPS regardless of capability.

**Remediation**: `HybridEngineOptions.targetFps?: number | null` (null = no cap). Default 60.

#### M5 — `defaultIsSceneReady` triangle threshold (`total >= 200`)

Procedural / sparse-geometry hosts may have 50 triangles and a perfectly valid BVH; the heuristic silently blocks them.

**Remediation**: change to `total > 0` (any geometry → ready). Hosts with stricter readiness requirements supply their own `isSceneReady`.

#### M6/M7/M8 (remaining) — temporal/spatial-DI tuning constants

`temporal.wgsl:40` `M_CLAMP = 20u` is framerate-dependent. `spatial.wgsl:43` `RADIUS = 30 pixels` is resolution-dependent. `spatial.wgsl:144–146` 5 cm absolute depth tolerance is scene-scale-dependent.

**Remediation**: plumb via `WalkaroundUBO` fields (`temporalMClampDI`, `spatialReuseRadiusFraction`, `depthToleranceAbsoluteMin`). Host computes defaults from target FPS + render-target height + scene AABB.

#### M10 — `PPG_MAX_SPATIAL_CELLS = 10_000`

Large outdoor scenes need more spatial cells for path-guiding convergence. Module-level constant is not overridable by host.

**Remediation**: pipe `HybridEngineOptions.ppgMaxSpatialCells?: number` through `createFrameResources` → `createPPGBuffers({ maxCells })`.

#### M12 — `common.wgsl:55` `EMITTER_DIST2_FLOOR = 0.01`

Now 0.01 (10 cm), but still scene-scale-dependent. Should be `(sceneDiagonal × 1e-3)²` per host configuration.

**Remediation**: add `emitterDist2Floor` field to `WalkaroundUBO`. Host computes default from scene AABB. Expose `HybridEngineOptions.emitterDist2Floor?: number` override.

### Minors

- `M_GI_BASE`, `RESTIR_GI_W_CAP`, `SPATIAL_RADIUS_GI`, `K_SPATIAL_GI`, `M_CLAMP_GI`, `M_CLAMP_SPATIAL`, `ITA_ALPHA`, `ITA_BBOX_EXPAND`, `ITA_SPIKE_BOUND_MULT` — all currently constants in `risGi.wgsl` / `temporalGi.wgsl` / `spatialGi.wgsl` / `indirectTemporalAccum.wgsl`. **These should not be parameterised now** — the ReSTIR-GI subsystem has a separate outstanding correctness bug (spatial-reuse lock-in producing per-pixel fireflies) that should be addressed first; tuning constants are downstream of that fix.
- `compositeLinearSampler` rename → `compositeSampler` (Q5, landed).
- Dead `void PPG_DIRECTIONS` import (Q5, landed).
- `isGlass = matColor.a > 0.3` threshold replicated across four shaders — derive from packing convention constant rather than inline.
- `PPG_GUIDE_INDIRECT_BLEND = 0.35` (data-driven blend would be better, but minor impact).
- `probeUpdateRays.wgsl:24` glass transmission probe scale `0.7` (already addressed in the M14 work).
- `probeRayCast.wgsl` `gThick = max(0.001, sMat.thickness)` absolute 1 mm thickness floor.

---

## Out of scope for this remediation plan

- **The ReSTIR-GI Cornell rendering bug** (spatial-reuse lock-in causing per-pixel firefly pattern). Verified via empirical reservoir probe: each pixel's reservoir converges to a different chosen reconnection vertex; spatial reuse rejects most neighbours in corner geometry. This is a correctness issue, not a generality issue. Tracked separately.
- Code-quality findings in files owned by an in-flight ReSTIR-GI rewrite (the rewrite agent's branch `worktree-agent-a193d8b569f806c50` exists but was forked from a stale base and was not merged).
- Performance optimisation.

---

## How to use this doc

Pick the highest-priority remaining item from the list above. Open a focused branch. Add the `HybridEngineOptions` field with a JSDoc explaining the trade-off. Plumb through `uboUpdater.ts` → `WalkaroundUBO` (in `common.wgsl.ts`) → the consuming shader. Compute a sensible default in `HybridEngine.ts` (typically derived from scene AABB or max emitter power). Keep Cornell rendering byte-for-byte equivalent at default values.

The remediation agent's worktree at `.claude/worktrees/agent-a06812aa6c5b09b98` (uncommitted, on stale base `273994c`) contains worked-through reference implementations for several of the open items. Treat it as a design sketch — its diffs do not apply cleanly to main HEAD due to substantial divergence in the pipeline glue layer, but the API field naming and the shader-side template patterns are useful starting points.
