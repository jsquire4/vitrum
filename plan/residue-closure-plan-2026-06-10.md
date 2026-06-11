# Residue Closure Plan — 2026-06-10 (R8)

> The final work order: closes every item the trust campaign left honestly open
> (plan/trust-remediation-plan-2026-06-10.md "Known-open residuals" + the A/B-bucket
> tail in plan/road-to-100.md). V28-B recapture runs LAST, after all render-changing
> work lands (user decision). §0 governs: implement fully; decisions made inline below.

## Round R8-A (parallel, file-disjoint)

1. **A4-progressive — true Hachisuka SPPM** (pt-webgpu, WGSL owner). Replace the
   streaming-window gather with per-pixel progressive statistics: persistent per-pixel
   {τ.rgb, R², N} buffer (8 f32/px, full-tier); per frame: photon pass re-deposits into
   the hash grid, gather applies the SPPM update τ' = (τ+Φ_M)·(R'²/R²),
   R'² = R²(N+αM)/(N+M), N' = N+αM (α=2/3), contributes τ'/(N_e·πR'²); stats reset
   with the accumulator (camera moves already reset accumulation — SPPM's static-view
   assumption holds by construction). Limits-guarded like the grid.
2. **texCoord (uv1) on pt-webgl2** (pt-webgl2). The last documented unkept promise:
   upload uv1 as a new attributes-array layer; per-map uv-set selection bitmask packed
   into the spare D3 texel lane; GLSL selects uv vs uv1 per map. Flip the pinned
   "documents the absence" test into a consumption test.
3. **Device-loss recovery protocol** (engine + docs). docs/device-loss-recovery.md
   (protocol + code example per backend kind) + attachVitrum opt-in
   `autoRecreateOnDeviceLoss` (fatal device-lost/context-lost → dispose + recreate with
   retained options/scene, surfaced via onEngineError first). Closes R7d item 32.
4. **Sun-NEE default-on** (walkaround shade.wgsl). Direct directional-light NEE for
   opaque surfaces in shade (shadow ray + BRDF eval), default-on when a directional
   emitter exists — sharp sun shadows no longer locked behind the stained-glass flag.
   DECISION: implement as default; the stained-glass caustic terms stay flag-gated.
5. **walkaround captureFrame 'output'** (walkaround engine/pipeline). Dispatch the
   composite (tonemap UBO) into an on-demand offscreen target + readback instead of
   rejecting.
6. **Neural weights v2** (tools/). Extend the CPU dataset generator with randomized
   scenes (geometry/materials/lights/cameras); train a v2 checkpoint within a wall-clock
   budget; vendor + document quality honestly. "Production" quality is acknowledged
   iterative — this round ships the diverse-data pipeline + best-effort v2.

## Round R8-B (parallel, file-disjoint; after A commits)

7. **A6 — NRC semantics** (walkaround neural). Fix the spread-termination predicate to
   Müller's a0 camera-pdf footprint semantics (audit: fires at every vertex → vacuous;
   comments state the effect backwards); retarget training from the DDGI estimate
   (pure distillation) to the ReSTIR-GI reconnection radiance the risGiNrc pass already
   computes; document the bias bound.
8. **B2 — DDGI glossy-aware probe bounce** (walkaround ddgi). Metals at probe-ray hits
   currently fold as Lambertian; implement the specular complement at the probe level
   (reflected-direction previous-frame field lookup via the existing indirectFeedback
   atlas read) or justify the documented-ceiling + lo_indirectSpecular complement as
   the correct architecture — agent reads first, implements the honest option.
9. **B1 tail — glass refracted GI** (walkaround risGi + shade transmission lobe; owns
   shade.wgsl this round). Glass primaries get a reservoir built at the first diffuse
   surface through a 1-interface refraction walk (castPrimary-through-glass); shade
   consumes it through the transmission lobe. Multi-interface documented as remaining.
10. **A8 — GRIS default decision** (walkaround options/README). DECISION: biased
    default retained for the realtime regime; the unbiased GRIS variant becomes a
    first-class documented option with the default's bias sources quantified in the
    option JSDoc + README (clamped Jacobian [0.1,10], no reuse-visibility,
    centroid-p̂). Closes the gate without wrecking frame time.

## Round R8-C (LAST — after all render-changing work)

11. **V28-B recapture** (external rig, ~/projects/wsl-gpu). Re-run the T1 GPU smoke +
    baseline/benchmark captures against the final tree; update baselines; report
    deltas with the render-changing commit map (every delta should trace to a named
    intentional change — anything unexplained is a finding).

Gates for every round: 13-package suite, typecheck, WGSL 51/51+, GLSL 6/6,
behavioral 26/26+ (new configs added where features land: sun-NEE, glass-GI).
