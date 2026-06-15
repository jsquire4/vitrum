# Road-to-100 Campaign 2 — 2026-06-09

> Successor to the §H remediation (closed; see items_to_fix.md §H). This campaign covers
> EVERY remaining open item: the V28 GPU-validation tail, the road-to-100 feature/fidelity
> ledger, the open known bugs, and the review leads. Distribution/governance stays excluded
> per the standing scope. Execution = Opus agents in 3 file-disjoint waves; lead verifies
> every claim by code-read/harness-rerun before commit (sub-agent reports are hypotheses).

## Decision resolutions (road-to-100 "Open decisions", resolved this campaign — flag to user)

- **A4 caustics:** MNEE stays the sole validated caustic path; photon-map remains the
  honest 'approximate' opt-in (c9d88fa demotion stands). No progressive photon map build.
- **A7 RC:** stays a validated OPT-IN (2026-06-07 gates); not a default contributor.
  Regime documented; the unconsumed TSL remnants already pruned.
- **A8 GRIS default:** validation-gated flip — V28-adjacent A/B quantifies default-path
  bias vs GRIS-on; flip the default only on a clean A/B (Wave C).
- **B1 approach:** ReSTIR-GI full-BRDF p̂ + widened packed material lane (real roughness/
  metalness), NOT specular probes.
- **A6/A10:** remain opt-in/experimental until trained assets exist; this campaign makes
  the training PIPELINE runnable end-to-end (dataset capture + train smoke), not a shipped
  checkpoint.

## Wave A (dispatched 2026-06-09, 8 Opus agents, file-disjoint)

| # | Scope | Items |
|---|-------|-------|
| A-1 | **V28 GPU validation** (wsl-gpu harnesses; lavapipe + dzn) | V28 steps 0–4 (T1 smoke run DIRECTLY, not via push), the 12 A/Bs, golden re-seeds via oracles |
| A-2 | **walkaround glossy/metal GI** (B1, XL) | widen material payload (real roughness/metal), GGX GI target in risGi/reservoir p̂, metals get direct + glossy GI (remove the isGlass/isMetal punts), shade consumption |
| A-3 | **pt-webgpu ReSTIR-PT + BDPT quality** (A1 + A9) | composite resolve into beauty w/ MIS + spatial-reuse pass; BDPT glossy light-subpath, parallel build, bounce-cap raise |
| A-4 | **pt-webgl2 completion** (A5 + B4 + H4-full + H2-followon + flags) | BDPT host driver (ping-pong subpath passes), mesh-area NEE in the lights texture, power-weighted forward-hit MIS via per-light selection prob, Jakob/Cauchy uploads, plumb-or-honestly-gate the dead feature flags |
| A-5 | **PPG spatial adaptivity** (A2) + PPG refine-runaway investigation | feed per-cell sample counts GPU→sTree split; investigate/fix the ~7-cycle flux ramp |
| A-6 | **DDGI radiometry research** (cardinal under-read) + B5 Beer-Lambert probes | derive octahedral solid-angle/cosine quadrature analytically (analytic==FD harness), fix the cardinal 23–60% under-read; real Beer-Lambert in probe glass |
| A-7 | **Verification pass** | the 4 uncited review leads (neural resize, OIDN resize race, out-of-range indices, progressiveHandoff desync) + base-GI size-200 estimator instability |
| A-8 | **Neural training pipeline** (A10 scoped) | dataset-capture path via wsl-gpu pt-webgpu renders, train.py end-to-end smoke on a tiny set, weight-load round-trip into the engine |

## Wave A RESULTS (2026-06-10 — all 8 agents returned; every load-bearing claim lead-verified by code-read; full typecheck green; suites: wh 1202, pt-webgpu 507, pt-webgl2 116, shared-bvh 133, engine 169, core 93, shared-denoisers 176 — all green)

- **A-1 V28 validation:** T1 smoke PASS direct-invoke (56.7 dB both backends, all 3 CPU oracles 100%, goldens current); item-level PASS: H32 (TLAS glass, CPU-oracle 100% both polarities), H14-E (packer-level), H52 (pt-webgl2 lobes lit, zero-lobe control 0.0), H6 (rotation matrices vs analytic 1e-7), H28 (ReLU fix accepted / aliased rejected by validation). DEFERRED (WSL full-tier-blocked): H13 glass anchor, H52/H14-E full-tier cross-backend, PPG/NRC radiometric. 5 new harnesses in ~/projects/wsl-gpu.
- **A-2 B1 SHIPPED:** per-tri `bvh_material` r32uint texture (rough[31:24]|metal[23:16]; texture not buffer — stays off the 16-storage-buffer floor); metals now get DI + analytic NEE + specular indirect (`evalGGXSpecularOnly` re-weight of the chosen GI sample, routed un-demodulated); GI reservoir p̂ DELIBERATELY left Lambertian (specular = deterministic re-weight → no p̂/consumption bias, no demod conflict); diffuse-default byte-identical (packs the old 0.85/0.05/0 defaults + spec term gated to 0). Glass refracted GI explicitly out of scope. NRC xsRough wiring = follow-up.
- **A-3 A1+A9 SHIPPED:** spatial GRIS pass (5-neighbor full-GBH, no /p_src) + beauty compositing via E0-direct/indirect estimator split (producer-dropped specular pixels fall through to the full path); OFF-paths byte-identical at runtime; BDPT light subpath now real-BSDF (glossy VNDF), vertex rows 3→4, cap 3→8, isotropic point emitter; serial one-shared-path build RETAINED (documented). wsl-gpu restir-pt compile gate extended.
- **A-4 pt-webgl2 SHIPPED:** A5 BDPT driver (3-column ping-pong subpath protocol, blit-preserve, inert-warn removed); B4 mesh-area NEE (area-proportional selection → triangle-independent pdf → forward-hit MIS from one global `uTotalEmissiveArea`; fold kept as the BSDF strategy — exactly-one-MIS-estimate algebra documented); H4 verified ALREADY-correct (power-weighted forward pdf recomputed inline — no change needed); H2 Cauchy live (per-material Jakob lane later landed in the 2026-06-15 parity wave); flags: cameraType+dof PLUMBED, fog/backgroundMap/randomType/debugMode honestly GATED.
- **A-5 A2 SHIPPED + runaway FIXED:** per-cell atomic sample counters (UBO pad slot reused, 16B unchanged) → readback → `splitOverflowLeaves`; children seeded via cloneDTree; growth-aware GPU clears (postSplit count — a second latent bug found+fixed). Runaway root-caused by CPU harness: decay=1 diverges linearly (matches the ~7-cycle 3× signature); fix = `PPG_FLUX_DECAY = 0.5` per-window (analytic steady state F/(1−d) pinned by test).
- **A-6 DDGI cardinal bias ROOT-CAUSED + FIXED:** the SH math is EXACT (harness <2% all cardinals — the convolution-coefficient hypothesis REJECTED); the bug was the receiver's octahedral-era `max(0,dot(n,probeDir))` weight double-applying a cosine and asymmetrically starving cube-corner probes for axis-aligned normals (harness reproduced −21% cardinal / −2.6% diagonal; fix → −6% = the spatial floor). Removed in `ddgiSampleWgsl.ts` with the derivation. B5 Beer-Lambert probes done (true exp(−σ·t/d), thickness-clamped). V28 re-confirm: re-run ddgi-white-bounce-ab — expect cardinals ≤~8%.
- **A-7 leads:** #1 neural-resize NOT-REAL as described, but two REAL adjacents found+fixed (InferenceGraph double-init guard; 7-buffers-per-teardown relu input leak); #2 OIDN resize race NOT-REAL (cohort tokens guard); #3 OOB indices REAL→FIXED (silent (0,0,0) collapse → filtered+warned); #4 progressiveHandoff desync NOT-REAL (mutations force reset(); converged engine rebuilds); #5 size-200 instability: the 1/dist² suspect REFUTED (base estimator is scale-invariant) — refined to the Cornell-tuned `restirGiIrrClamp=5`/`restirGiWCap=16` bimodal clipping (B15); V28 clamp-sweep scenario specced.
- **A-8 A10 pipeline:** capture→train→export→load CLOSES (CPU-smoke dataset; --dry-run exports a valid 535,107-param `.vitrum-model` — the "vi-neural-weights.json" reference was stale, binary format is the real loader; round-trip test green). Real dataset + torch training = the remaining hardware/provisioning tail.
- **Lead fixes during verification:** 4 stale PPG test fixtures updated (cellSampleCountsBuf + second readback chain in the mock device).

## Wave B (after A lands; conflicts with A-2/A-3 files)

B3 directional IBL (walkaround equirect CDF import), B6 GTAO view-space reconstruction,
B7 planar-SAH epsilon (+optional SBVH spike), B8 light-tree orientation cones,
B9 GGX multiscatter (all 3 backends), B10 physical refraction throughput, B12 lite-tier
env importance + area-MIS, B15 scene-scale-aware clamp defaults, B16 DI BRDF candidate,
H46-A maxBounces→DDGI control, H20-A sky-only present path, **A3 true spectral transport
(pt-webgpu, XL — solo agent)**.

## Wave C (after B)

D3 reserved-field consumption (aoMap/bump/displacement/lightMap/envMapIntensity/
anisotropy/soft-sun/castShadow — per-field, golden-breaking), A8 GRIS default flip
(if the Wave-A/B A/B is clean), D6 bind-group memoization, residual polish + ledger
re-grades + fidelity-matrix refresh.

## Standing rules

No git push (T1 smoke invoked directly by the validator). Agents do NOT commit — the lead
verifies by code-read + suite runs and commits per scope. Render-changing work ships with
the A/B evidence or an explicit V28-queue entry. Every ledger row a fix flips must move in
the same commit (ledgerVsCapabilities enforces core rows). WGSL stays naga-conservative.
