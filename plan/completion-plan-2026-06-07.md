# Vitrum completion plan — DO IT ALL (2026-06-07)

**Purpose.** The complete, phased, parallelized punch-list to take vitrum from
"release candidate with a bounded list" to a defensible, shippable v1. Authored
to be executed by agent fan-out (each Track is file-disjoint enough to run
concurrently) and to be **fully validatable without human sight** — every
fidelity item names its objective oracle, never an eyeball.

**Governing doctrine (load-bearing, because the maintainer cannot see):** no
fidelity claim is closed by looking at it. Every correctness item is gated by an
objective oracle — a CPU/analytic ground-truth reference, a cross-backend
path-traced reference, energy conservation, or a self-validating residual→0
harness. The two oracles already built this session (`ddgi-white-bounce-ab`,
`checkerboard-motion-ab`) are the template. **Phase 0 builds the rest of the
oracle suite first**, because every later fix is gated by one.

**Sequencing rule.** Phase 0 (oracles) unblocks Phase 1 (P0 correctness).
Phases 2/3/4 are largely file-disjoint and run concurrently after Phase 1.
Phase 5 converges. Within a phase, Tracks run in parallel.

**Effort scale (from the 2026-06-06 audit):** "weeks of focused work to a
defensible v1, not months." The one strategic fork is Phase 4 (THREE-decouple):
v1 requirement vs documented internal dependency — a maintainer decision.

---

## ✅ FIXED P0 (2026-06-07, `254c284`) — default GI was DEAD; DDGI surface/normal bias restores it
**FIXED + GPU-validated.** Root cause: `ddgiSample` sampled the receiver at the EXACT surface; on a grid-boundary plane (Cornell walls/floors) the trilinear collapses onto in-plane probes (perpendicular probe-dir → cosine weight 0) → `totalWeight<1e-4` → returns 0 → the whole DDGI→ReSTIR-GI handoff zeroed → default GI dead (rode entirely on off-default RC). Fix: the standard Majercik 2019 §4 surface/normal bias (`biasedPos = worldPos + n·(gridSpacing·0.25)`) inside `ddgiSample`, applied to cell-select + cosine + visibility. **Validated on dzn:** default Cornell RC-off indirect **0.0004 → 0.237** (635×); 0-E RC-off **0.000 → 0.495**. Root-caused by elimination (gidiag): visibility-disable=no change (ruled out Chebyshev), force-Lo→0.80 (reservoir/blend proven working), offset-xs→0.25 (the fix). walkaround-hybrid 1210 green, no golden moved.
- **RESIDUAL (LARGELY RESOLVED 2026-06-07):** the dominant contributor was NOT octahedral
  quadrature — it was the producer MULTI-BOUNCE FEEDBACK (clamp + ×π), root-caused + FIXED
  in `c098c8e` and GPU-validated 71%→14% vs a CPU f64 anchor (see 1-A below). 0-E re-anchored
  (fix-invariant same-channel interior mean, no-GI + wrong-bounce discrimination — the old
  "RC-off=zero-GI" variant was correctly retired) and the post-fix re-run is ORACLE_SOUND
  with floorEnergyRatio 0.580→0.661. **Remaining tail — CHARACTERIZED as inherent-
  approximation (NOT bugs), 2026-06-07:**
  - *Axis-aligned cardinal under-read (~33% on ±x/±y/−Y vs ≤6% on diagonals).* ROOT: the
    octahedral SEAM (cardinals map to the diamond vertices = square edge-midpoints). **FIXED
    2026-06-07 via the SH reparameterization** (f6fc831 + 6d80641): L2 spherical harmonics
    for irradiance is seam-free — cardinals 33%→**1.4%** vs an MC ground truth (visibility
    stays octahedral). Also 4× less irradiance-atlas memory + no border pass. See
    [[ddgi-sh-irradiance-2026-06-07]]. HONEST nuance: the seam fix is large for single-bounce/
    directional indirect but MASKED in smooth multi-bounce (where the floor residual is the
    convergence gap, not the seam). Receiver perf MEASURED (dzn timestamp-query): SH 0.385
    ns/call vs octahedral 0.463 → SH ~17% FASTER (cache-local 3x3 block in a 4x-smaller atlas
    + branchless eval beat octahedral's 1 sample + branchy encode) — the 9-read "cost" was a
    false fear, so SH wins on quality + storage + receiver perf. The earlier IRR_CELL 8→16
    experiment confirmed more octahedral resolution is NOT free (starves at 192 rays) — SH was
    the right lever. Also: the VISIBILITY octahedral atlas was measured for the same cardinal
    seam and has NONE (cardinals≈diagonals; VIS_CELL=16 resolves the equator) → cube-map
    visibility skipped. GPU timestamp-query works on dzn (`wsl-gpu/scripts/gpu-timestamp.ts`) —
    perf is no longer a blind spot.
  - *Absolute walkaround interior indirect ~3.8× below pt in the 0-E scene.* The rect-area
    emitter IS fed to the probe field as a flux-equivalent `fixture` point approx (via
    `coreEmittersToDDGILights`) — NOT a missing-light gap. The gap is dominated by coarse-DDGI
    convergence vs fully-converged pt (5³ probes, single-probe crude feedback, hysteresis EMA
    not reaching the high-order bounce tail a path tracer does) — INHERENT to a realtime probe
    GI, which the contract treats as a fast preview with pt as the hero. The 0-E oracle
    correctly gates on GI DISTRIBUTION (floor/interior balance), not absolute magnitude.

## ~~CONFIRMED P0 — default GI DEAD~~ → FIXED above (`254c284`)
**ROOT CAUSE LOCALIZED** via harness shader-patch A/B on dzn (bisect `--gidiag`):
- Disabling the risGi final-visibility reject → NO change (indirect still 0.0004) → visibility test RULED OUT.
- Forcing the candidate `Lo`=const on a bounce HIT → indirect jumps **0.0004 → 0.80** → the reservoir/RIS/blend/downstream pipeline ALL WORK, and the GI bounce rays DO hit.
- ∴ the ONLY failing link is **`Lo = sampleDDGIAtPoint(xs, ns)·albedo·INV_PI` ≈ 0** — i.e. `ddgiSample` (the production DDGI consumer) returns ~0 at the GI reconnection vertices (Cornell-wall surface points), DESPITE the DDGI atlas being populated (mean 3.4).
This is the WHOLE default realtime GI being dead (relies entirely on off-default RC). It is a SEPARATE, MORE SEVERE issue than 1-A (the DDGI axis-aligned under-read is 0.40×, not ~0) — but likely the same subsystem. **NEXT (top priority):** why does `ddgiSample` return ~0 at wall surface points in the full pipeline when the `ddgi-white-bounce-ab` harness got 0.6–1.3 at the same class of point? Candidates: probe-grid bounds (wall points at/outside the grid extent → trilinear edge/out-of-bounds → 0), the Chebyshev visibility term suppressing, or an atlas-binding/grid-param mismatch in the risGi bind group vs the producer. Gated by the bisect `--gidiag` + a new `ddgiSample`-tap. Harness: `wsl-gpu/scripts/tlas-zero-gi-bisect.ts --gidiag=1 GIDIAG_WHAT=vis|forcelo|hitrate`.

## ~~ORACLE-SURFACED FINDING (0-E) — default ReSTIR-GI indirect ~0 [P0-suspect]~~ → CONFIRMED above
The Phase 0-E cross-backend oracle, immediately on first run, surfaced a
radiometric gap NO string-golden or existing test caught (the whole point of the
oracle suite): on an enclosed Cornell with the DEFAULT config (RC off, `ultra`
tier), the walkaround **indirect tap (`hdrIndirect`, ReSTIR-GI) reads ~0** while
the DDGI atlas is fully populated (mean 3.4, 100% coverage). Only the RC path
(rcEnabled=true, OFF-default) energizes indirect (mean ~1.96, which matches
pt-webgpu at floor-ratio 1.013). VERIFIED so far: ReSTIR-GI runs by default
(`RISGIPass.gates()` → true; ultra tier giSpatialPasses=2 — NOT a "GI off"
artifact); corroborated by an independent lead obs (bisect cornell5 rc=0 indirect
0.000356); `risGi.wgsl:13` confirms the GI reservoir candidate radiance comes
from sampling the DDGI atlas — so a populated atlas SHOULD yield non-zero
reservoirs. NOT YET root-caused: WHY risGi → ~0 (reservoirs empty? the
DDGI-sample-at-GI-hit returning 0? a normal/visibility reject?). Could relate to
the open G-P0.1 ReSTIR-DI p̂ class or the 1-A DDGI axis-aligned sampling. **TOP
next investigation** — gated by oracle 0-E + the bisect indirect tap. If real,
the default realtime GI relies entirely on an off-default opt-in (RC) — a
significant P0 the audit's flat-wall tests + string goldens missed.

## ⚡ REVISED SCOPE (2026-06-07, after the first execution wave)
**The audit ledger overstates the open work.** 3 of the first 4 agents found their
target ALREADY FIXED by the 2026-06-06 G-sweep (`178f80d`) — three-bindings
G-P0.4 (all 5 asymmetries), the fork P2 sweep (G-P2.1/2.2/2.3/2.4 — dispose
leaks, dead-GLSL removal, the `length` ReferenceError, the lint gate), and the
two "zero-test" packages (already had thin suites). The audit was filed BEFORE
`178f80d` landed, so its P1/P2 list is mostly closed. **The genuinely-remaining
real work is narrower than the audit implied:**
- **Phase 0 oracle suite** — genuinely new (0-F denoiser ✅ built; 0-D/0-E/0-G remain).
- **1-A DDGI octahedral solid-angle** — real, confirmed by the oracle this session.
- **2-A PPG defects** (refine-loop runaway + base-GI instability) — real, V17.
- **2-B WebGL-BDPT** `lightRec.point` compile — verify vs `178f80d` then fix.
- **Phase 4 THREE-decouple T1–T5** — real, intricate, byte-identity-gated,
  and deferred from the current maturity sweep by the 2026-06-08 instruction.
The hygiene/coverage phases (3) are mostly DONE; treat them as verify-and-add-
coverage, not implement. Re-scope effort: closer to a defensible v1 than the
audit's "weeks" implied, MINUS the genuinely-deep items (DDGI math, T1–T5).

## ⚡ RECONCILIATION (2026-06-08) — concurrent agent's "harden THREE ingestion" pass (`646ead9` + green uncommitted)
A second agent landed a hardening sweep (19 files, +969; verified by code-read +
combined typecheck/vitest all green: core 73, three-bindings 130, engine 142,
walkaround-hybrid 1221). It built ON the SH migration cleanly (SH integrity
re-verified: IRR_CELL=3, 9-site receiver eval, blend projects coeffs, irr border
still removed). What it CLOSED / ADVANCED vs this plan:
- **1-C ✅ CLOSED.** `three-bindings/mesh.ts` (+238) comprehensively hardens the
  THREE→core ingestion: grouped multi-material geometry → stable per-group
  primitives, `stripEmissive` (no emissive double-count), InstancedMesh/
  SkinnedMesh + rotation/scale/parent-world transforms, host ShaderMaterial
  conversion — all pinned by `sceneFromThreeJS.test.ts` (11 cases).
- **3-B ✅ PARTIAL.** New: `hybridEngineGeometryUpdate.test.ts` (the
  `HybridEnginePrimitiveUpdates` "0 tests" gap — transform/positions/topology/
  material fast paths), `engineContract.test.ts`, `backendContractMatrix.test.ts`,
  `sceneFromThreeJS.test.ts`. **RECONCILED 2026-06-08:** the basic package
  script/suite gaps are now CLOSED for `scene-lighting` (`test` script +
  `sceneLighting.test.ts`), `stained-glass-extensions` (`test` script +
  `stainedGlassExtensions.test.ts`, incl. `packCameUBO` layout pins),
  `walkaround-rc` (`test` script + rc bindings/solid-angle/kernel math/cascade
  tests), and `examples/two-engines-one-scene` (`test` script + 2 test files).
  REMAINING 3-B is validation depth, not zero-test plumbing: three-bindings
  STEP/CUBICSPLINE semantics, deeper two-engines backend smoke, and the RC
  behavior/acceptance script that proves the kernels under the GPU behavior
  path.
- **A3 ✅ CLOSED.** `supportsIncrementalScene` → true; `updatePrimitive` geometry
  fast paths (transform-only no-recompile, positions-only refit, topology
  rebuild, material fast path) + disposed/initializing guards.
- **5-B ✅ ADVANCED.** Backend promise ledger + `supportDetails` + per-backend
  add/remove + walkaround mutation-fidelity + "DDGI SH vs path-tracer"
  environment rows are now test-pinned.
- **5-A/5-D 🟡 IN PROGRESS (uncommitted).** pt-webgpu drops `'experimental-backend'`
  from its `experimentalFeatures` (maturity promotion); `patchScene` rejects
  analytic shape/params/fallbackMesh on mesh-like primitives (input hardening);
  pre-push hook made node-path robust (NOT the 0-H analytic-oracle extension).
- **T3-adjacent ADVANCE (not closed):** `rebuildEmitterBuffersFromCoreScene`
  (core-first incremental emitter/material rebuild). BUT **T2 `vitrumSceneToThree`
  (`HybridEngineLifecycle.ts:337`) + T3 `findMeshByPrimitiveId` (8×) are STILL
  PRESENT** — the deep internal core→THREE-synthesis decouple is UNCHANGED.
NOTE: that agent's uncommitted source changes are green but NOT yet committed —
leave them be; this plan edit touches only the plan doc.

## ✅ RECONCILIATION (2026-06-08) — non-THREE maturity tail closed (`75391f2` + follow-up)
The current non-THREE-decouple maturity sweep closes the remaining Phase 3 code
and harness tail:
- **3-A ✅ CLOSED.** Walkaround bind-group / texture-view memoization,
  DDGI binding-state cache reuse, BvhBufferHost texture-view reuse, and
  GPU-memory accounting tests landed in `75391f2`.
- **3-B ✅ CLOSED for default mechanical coverage.** `examples/two-engines-one-scene`
  now has backend smoke that instantiates both pt-webgl and walkaround-hybrid
  against the same core scene with deterministic stubs. `walkaround-rc` has
  `test:rc-behavior` via `run-rc-behavior-mechanical.mjs`. THREE animation
  import now has STEP coverage plus an explicit, tested Smooth→LINEAR demotion
  because THREE tracks do not carry glTF CUBICSPLINE tangent triples.
- **3-C ✅ CLOSED.** Strict fork shader smoke is wired, the pre-push hook runs
  it, fork static spectral uploads are regression-pinned, BDPT is honestly
  labeled opt-in approximate, and stained-glass shadow normal perturbation is
  feature-gated off by default.
- **0-G code harness ✅ CLOSED; validation remains opt-in.** The pt-webgl oracle
  now has `capturePtWebgl.mjs` plus `run-pt-webgl-fidelity-oracle.mjs`, which
  either validates committed paired PNG fixtures mechanically or, with
  `VITRUM_PTWEBGL_ORACLE_CAPTURE=1`, captures pt-webgpu baselines and pt-webgl
  candidates from the two-engines example before running strict acceptance.
  Formal hardware/browser capture is validation evidence, not more library code.

## STATUS LEGEND
- ✅ DONE + oracle-validated · 🟡 partially done / needs verify · 🔴 open · ⚖️ needs maintainer decision

---

## PHASE 0 — Validation foundation (the oracle suite)  [UNBLOCKS EVERYTHING]

> Why first: with no sight, the oracle IS the acceptance test. Each later fix
> references one of these. These are wsl-gpu harnesses + (where possible)
> in-tree numeric tests so correctness leaves the "string-goldens only" regime
> the audit flagged (the systemic reason G-P0.1/0.2/0.3/P1.1 sat green).

| Track | Oracle | Covers | State |
|---|---|---|---|
| **0-A** | DDGI vs CPU f64 path-trace anchor (`ddgi-white-bounce-ab`, diagonal/axis-aligned split) | DDGI bounce + octahedral sampling | ✅ exists |
| **0-B** | Checkerboard scripted-motion A/B (`checkerboard-motion-ab`, PSNR + parity-comb) | temporal reconstruction quality | ✅ exists |
| **0-C** | RC emitter-NEE + cRc gate A/B (`tlas-zero-gi-bisect`, rect/sun) + two-scene gate | RC light model | ✅ exists |
| **0-D** | **ReSTIR-DI/GI vs reference** — reuse-unbiased + p̂-consistency + variance-reduction | the G-P0.1 class | ✅ DONE (`restir-fidelity-oracle.ts`, lead-verified both directions: clean PASS/exit0, --inject-bug FAIL/exit1) |
| **0-E** | **Cross-backend ground truth** — pt-webgpu converged vs walkaround GI, GI-channel-isolated + exposure-anchored | whole walkaround GI stack | ✅ DONE (`xbackend-groundtruth-ab.ts`, lead-verified: correct 1.013, zero-GI rejected 1038×, wrong-bounce 72.6×). SURFACED the default-indirect-~0 P0-suspect above. |
| **0-F** | **Denoiser oracle** — identity + variance-reduction-without-bias, discrimination-proven | shared-denoisers | ✅ DONE (`denoiserFidelityOracle.test.ts`, lead-verified: rejects passthrough + biased) |
| **0-G** | **pt-webgl fork oracle** — WebGL2 capture for the fork's default radiometry (G-P0.3) + caustics + BDPT | the whole fork | ✅ CODE HARNESS DONE. `capturePtWebgl.mjs` captures the fork through the two-engines pt-webgl page; `run-pt-webgl-fidelity-oracle.mjs` validates committed fixtures mechanically by default and, with `VITRUM_PTWEBGL_ORACLE_CAPTURE=1`, captures pt-webgpu baselines + pt-webgl candidates before strict metrics + env-gated vitest acceptance. Residual hardware-GL BDPT eye↔light connection capture is validation evidence, not a missing code harness. |
| **0-H** | **In-tree shader-execution harness** — bridge the "npm test never runs shaders" gap: a headless-WebGPU vitest path (lavapipe in CI) that runs the smallest radiometric kernels against analytic references, so a subset of fidelity leaves the external-only harness | systemic test-architecture gap | ⚖️ scope (CI WebGPU availability) |

**Parallelization:** 0-D, 0-E, 0-F, 0-G are independent → 4 concurrent agents.
0-H is a scoping spike (decide feasibility before investing).

---

## PHASE 1 — P0 default-path correctness  [gated by Phase 0 oracles]

> What every user renders by default must be physically correct. Most P0 items
> are already validated; the open ones are below. Each is days-scale with the
> right independent-reference A/B.

- **1-A ✅ DDGI under-read — DOMINANT cause ROOT-CAUSED + FIXED (`c098c8e`, 2026-06-07).**
  The ~2× under-energy was NOT primarily octahedral quadrature (the earlier producer
  hypothesis) — it was the PRODUCER MULTI-BOUNCE FEEDBACK (`probeUpdateRays`), two defects
  that together starved every bounce past the first:
  (1) the feedback cell guard `baseProbeIdx3 + 1 < dims` returned `indirect=0` for EVERY
      hit on enclosing geometry (room walls/floor lie on/just past the grid boundary), so
      wall→wall→receiver multi-bounce was DISABLED — the field was effectively single-bounce
      and the floor (lit almost only by wall bounce) was the worst-hit surface;
  (2) the atlas stores E/π (cosine-weighted MEAN; the blend pass + `ddgiSample` both ×π to
      reconstruct E) but the feedback added that raw E/π to `direct`=E without ×π → the
      indirect was π× too weak per bounce. A stale "atlas holds irradiance E" comment masked
      it. Fix = clamp the cell index to [0,dims-1] (consistent with the receiver) + ×π.
  **GPU-validated** vs a CPU f64 multi-bounce anchor (`wsl-gpu/scripts/ddgi-indirect-pi-ab.ts`,
  dzn RTX-4090, 400 ticks): mean luminance error vs ground truth **71% [base] → 63% [clamp
  only] → 13.7% [clamp+×π]**, 6/6 normals; both fixes required (clamp alone barely helps,
  ×π alone inert); CLAMPPI sits just below the anchor (not over-correcting). Post-fix re-pin
  + no-patch re-run reproduced the numbers exactly; T1 smoke PASS; 53 DDGI vitest green.
  **RESIDUAL (research-grade, sharply isolated):** after the fix the axis-aligned floor −Y
  is still ~33% under (vs ≤6% on diagonals/ceiling) — the known octahedral-atlas
  border-wrap + cosine-MEAN-vs-INTEGRAL confound (orthogonal to the feedback; tracked with
  V27). 0-E cross-backend re-run (post-fix, dzn) CONFIRMS: ORACLE_SOUND (no-GI rejected 33×,
  wrong-bounce 3.3×), floorEnergyRatio **0.580 → 0.661** (floor/interior balance moved toward
  1.0), exposure scale 4.57 → 3.81 (walkaround interior indirect brightened ~1.2× from the
  restored multi-bounce — more modest than the empty-box harness's 3× because the 0-E scene's
  multi-bounce structure differs; the empty-box CPU-anchored harness shows the full 71%→14%).
- **1-B ✅ DDGI coloured-bounce.** Validated 2–4% on interior normals (`8aa444a`).
  Re-confirm after 1-A.
- **1-C ✅ CLOSED (2026-06-08, `646ead9`).** `three-bindings/mesh.ts` rework +
  `sceneFromThreeJS.test.ts` (11 cases) close the asymmetries: grouped
  multi-material→per-group primitives, `stripEmissive`, InstancedMesh/SkinnedMesh
  + rotation/scale/parent-world transforms, ShaderMaterial host conversion.
- **1-D 🟡 Re-validate G-P0.1 / G-P0.2 / G-P0.3 on current main** with the
  oracles (they passed in the 2026-06-06 G-sweep; re-run post-bounce-fix so the
  default-path correctness ledger is current, not historical).

**Parallelization:** 1-A (the real work) | 1-C | 1-D run concurrently.

---

## PHASE 2 — P1 opt-in features: finish or honestly demote  [concurrent with 3/4]

> The contract advertises these; either deliver them or demote via the existing
> `supportDetails` / `experimentalFeatures` mechanism (a day). North-star
> (roadmap §0.5): no deferring/removing fidelity features — implement what's
> promised.

- **2-A ✅ PPG defects RESOLVED by the DDGI fix (2026-06-07 cascade) — no PPG edit needed.** Both V17 defects (refine-loop runaway + size-200 base-GI instability) were DOWNSTREAM SYMPTOMS of the dead-GI P0: `ddgiSample` returning ~0 at the scene's axis-aligned surfaces gave PPG's dTree a degenerate near-zero, high-relative-variance training signal → dTree positive-feedback runaway + seed-dependent base variance. Post `254c284` the indirect is dense/well-conditioned (~0.13 plateau) → V17 re-run (dzn): refine-loop max 0.146 (was 0.45 ramp — GONE), size-200 base var ratio 1.008× (was 0.485× — GONE), PPG unbiased + ~6–14% variance reduction across all clean regimes. Root cause code-verified (matches the confirmed ddgiSample-zero). RESIDUAL (tracked, NOT a PPG blocker): a size-256×seed-100000 ppg-OFF base-reservoir variance quirk — isolated, research-grade. V17 → PASS. (Agent re-run; lead-corroborated via the shared root cause — a dedicated V17 run can re-confirm PASS if needed.)
- **B3 🟡 Checkerboard perf-win STILL UNMEASURED → stays off-default.** The vitem harness JSON exposes no wall-clock field and dzn lacks timestamp queries, so the actual shade-pass GPU-time saving isn't measured. The decision (enable medium/low) was gated on confirming the win — until a timestamp-capable host or a wall-clock-instrumented harness measures it, checkerboard stays OFF-default (built + motion-validated comb-free; enabling a perf feature whose win is unmeasured is premature).
- **2-A (orig) 🔴 PPG production-readiness.** Training writer EXISTS (`PPGCoordinator.ts:266`)
  and PPG is live + unbiased + gives variance reduction on a guiding-favourable
  scene (V17, this session). Two real defects block production: (1) **refine-loop
  runaway** (~7 refine cycles then ramps ~3× via GI-reservoir↔dTree positive
  feedback); (2) **base-GI size-200 estimator instability** (independent of PPG).
  Fix both; **Oracle: V17 occluded-scene A/B** (`g-p11-ppg-occluded.mjs`) —
  unbiased + variance ratio <1 + stable across refine cycles + scene sizes.
- **2-B 🟡 WebGL BDPT.** (a) ✅ ALREADY FIXED — `LightRecord` now HAS `vec3 point`
  (`light_sampling_functions.glsl.js:37`); `FEATURE_BDPT=1` compiles (verified
  2026-06-07 + G-sweep G-P1.2). (b) 🔴 Hardware-GL eye↔light connections are
  engine-disabled on ANGLE → needs a Windows-Chrome / real-GL connection render
  (environmental — **Oracle 0-G** fork WebGL2 capture runner now exists) + a
  BDPT variance-win A/B vs unidirectional on a hidden-emitter caustic scene.
- **2-C ⚖️ Fork caustic strategies (G-P1.3).** Phenomenological. pt-webgpu
  photon-map already DEMOTED (this session). Decide: demote the fork caustic
  modes to `approximate` (consistent with the pt-webgpu decision — fast) or
  implement physically. Recommend demote (MNEE is the validated reference).
- **2-D 🔴 shared-denoisers loose ends (G-P1.4).** Close the cited gaps; gate
  with **Oracle 0-F**. Includes the BMFR row honoring (roadmap §0.5 item 1 —
  real Koskela-2019 BMFR shipped; verify it's consumed + oracle-validated).
- **2-E 🟡 Checkerboard.** Built off-default, motion-validated comb-free
  (`dc266c5`). Optional tightening: the same octahedral-edge work as 1-A would
  improve the aggressive-pan transient. Then the enable-by-default decision
  (⚖️) — objective gate is **Oracle 0-B** (worst-frame PSNR + comb sub-perceptible).
- **2-F 🟡 fork stained-glass NEE bias.** Always-on shadow-ray normal-map
  perturbation biases NEE visibility (`attenuate_hit_function.glsl.js:182-205`).
  Gate it (opt-in) or justify physically. **Oracle 0-G**.
- **2-G 🟡 updateLighting DDGI sun desync.** Audit-reported open; appears FIXED
  (`HybridEngine.ts:1571` now republishes DDGI sun on `primaryLightDir`). Verify
  + add a regression test (runtime sun-scrub direct/indirect agreement).

**Parallelization:** 2-A | 2-B | 2-D | 2-F are file-disjoint → 4 concurrent.
2-C/2-E/2-G are small (decision/verify).

---

## PHASE 3 — Professional hygiene + test coverage (P2)  [concurrent with 2/4]

> "One-or-two-day sweep with outsized professionalism return" — but the test
> gaps are the real risk (they're why P0 bugs sat green).

- **3-A ✅ G-P2.6 performance hygiene CLOSED**: bind-group/view memoization,
  SVGF allocation/memory-accounting coverage, BVH/DDGI external memory sections,
  fork static CIE upload caching, and `BDPT_CONTRIBUTION_CLAMP` documentation
  are in-tree.
- **3-B ✅ G-P2.7 test gaps CLOSED for default mechanical coverage**:
  - ✅ `HybridEnginePrimitiveUpdates` geometry-update fast paths now tested
    (`hybridEngineGeometryUpdate.test.ts`, `646ead9`).
  - ✅ GI-state `serialize/deserialize` round-trip (`giStateSnapshot.test.ts`,
    incl. the v3 SH-break rejection) + TLAS rotation/scale transforms
    (`sceneFromThreeJS.test.ts` "preserves rotation+scale+parent-world").
  - ✅ `scene-lighting` basic suite/script closed (`package.json` has
    `test`; `__tests__/sceneLighting.test.ts` pins the host lighting math).
  - ✅ `stained-glass-extensions` basic suite/script closed (`package.json` has
    `test`; `__tests__/stainedGlassExtensions.test.ts` pins
    `SURFACE_TEXTURE_ID`, userData keys, and `packCameUBO` std140 layout).
  - ✅ `examples/two-engines-one-scene` basic script gap + backend smoke closed
    (`benchmarkScenes.test.ts`, `prBenchHarness.test.ts`,
    `sceneContractSmoke.test.ts`, `backendSmoke.test.ts`).
  - ✅ walkaround-rc basic suite/script + behavior runner closed
    (`test:rc-behavior` → `run-rc-behavior-mechanical.mjs`).
  - ✅ three-bindings STEP/Smooth coverage closed. Smooth is intentionally
    demoted to LINEAR because THREE tracks lack glTF CUBICSPLINE tangent
    triples; true CUBICSPLINE remains supported in `@vitrum/core` for loaders
    that provide tangent-tripled sampler values.
- **3-C ✅ G-P2.1–2.5 finish CLOSED**: strict fork shader smoke + pre-push
  wiring, fork resource/static-upload tests, BDPT clamp documentation, and
  feature gates are in-tree.

**Parallelization:** 3-A | each 3-B package | 3-C → many concurrent agents.

---

## PHASE 4 — THREE-decouple (T1–T5)  [DEFERRED from the current maturity sweep]

> The GI-signal DATA paths are decoupled (emitter/DDGI/per-tri-material/RC). The
> INGESTION / RESOLVER / UPDATE layer is still THREE. **Current instruction
> (2026-06-08): maturity work excludes THREE-decouple. Keep this track open as
> deferred, not closed and not part of the current "knock out maturity work"
> execution queue.**

- **4-DECISION ⚖️ DEFERRED** — v1 requirement vs documented internal THREE
  dependency remains a product/architecture call. Do not count T1–T5 as closed;
  do not execute as part of the current non-THREE-decouple maturity sweep.
- **T1 🟡 STEP 1 DONE + byte-A/B validated (`aa9a7ac`, 2026-06-07).** `buildMaterialResolver`'s `coreMaterials`/`resolveMaterialId` are now built CORE-NATIVELY from `scene.primitives` (one slot per mesh-like primitive in order) — the THREE traversal no longer drives the material-slot ordering (it only gathers the legacy `materials` fallback). Byte-identical: T1 GPU smoke **999 dB** on both backends + RC/TLAS/DDGI traversal oracles PASS. Removed the dead `DEFAULT_CORE_MATERIAL`. **REMAINING T1:** the returned THREE `materials` list is still consumed by **RC's material packer + the DDGI fallback** (`sceneBvhFromCore.ts:163`), and `sceneRoots` for the world-bake (though `mergeWorldSpaceFromCore` is the THREE-free analogue already used for geometry/emitters). Step 2 = make RC's cascade-material packer + the DDGI fallback read `coreMaterials`, then drop `sceneRoots`/`materials` from `buffersFromScenePack` + `buildMaterialResolver`. Then T2 (setScene synthesis), T3 (findMeshByPrimitiveId).
- **T1 (orig) 🔴** `buildMaterialResolver` traverses THREE (`sceneBvhFromCore.ts:54-118`). **SCOPED 2026-06-07 (byte-identity-critical):** the `coreMaterials` DATA already comes from the core scene (`coreByName` map, lines 76-97), but the material SLOT ORDERING + dedup is driven by `root.traverseVisible(sceneRoots)` + `materials.indexOf(mat)` (THREE object/material identity). The per-triangle packers index by `geo.triMaterialIds`, produced by THIS ordering — so the decouple must REPRODUCE the exact slot order from `scene.primitives` (matching `vitrumSceneToThree`'s object-creation order, `vitrumSceneToThree.ts:312/329/341`) + dedup by core MaterialSpec identity, NOT a THREE traversal. The ordering used here also DIFFERS from `mergeWorldSpaceFromCore`'s structural dedup (the already-decoupled geometry path) — T1 must either unify them or reproduce the THREE order exactly. **A half-done ordering change silently corrupts per-triangle material assignment** → gate EVERY step with the byte/converged-A/B oracle (`ddgi-white-bounce-ab` set-equivalence pattern + a per-tri material-id A/B vs the THREE path). Step 1: replicate the THREE traversal order over `scene.primitives`, byte-A/B confirm, THEN drop the `sceneRoots` param. **MAPPED 2026-06-07 — the dedup is TRIVIAL:** `vitrumMaterialToThree` creates a FRESH THREE material per primitive (`vitrumSceneToThree.ts:379/396/408` — no sharing), so `materials.indexOf(mat)` NEVER dedups → the slot ordering is just ONE SLOT PER mesh/skinned/instanced PRIMITIVE in `scene.primitives` order (= the order `vitrumSceneToThree` adds them, lines 701-710; `root.traverseVisible` visits flat-added objects in add-order). So core-native `buildMaterialResolver(scene)` = iterate `scene.primitives`, push `p.material`→coreMaterials + `id`→index, `resolveMaterialId(id)=index`. **INTERCONNECTION (the real T1 scope):** `resolveMaterialId` defines the slots (`packSceneFromCore` builds `geo.triMaterialIds` from it), but `materials` (THREE) is ALSO passed to `buffersFromScenePack(scene, sceneRoots, geo, materials, coreMaterials)` as the LEGACY fallback (coreMaterials is preferred per the RestirBvhSnapshot doc). So full T1 = (a) core-native resolver, (b) confirm `buffersFromScenePack` uses coreMaterials not the THREE `materials` when coreMaterials is populated, (c) drop `sceneRoots`/`materials` from all three. SAFEST step: build the core-native ordering ALONGSIDE the THREE one + assert equality in a test, THEN remove the THREE path. Byte-A/B-gated each step (per-tri material-id A/B).
- **T1 MATERIAL PATHS ✅ (2026-06-07, byte-A/B validated):** the material decouple is essentially complete — `buildMaterialResolver` coreMaterials/resolveMaterialId core-native (`aa9a7ac`), RC RESTIR path → `packCascadeMaterialsFromCore` (`df0d757`), DDGI already core-first (`probeUpdatePass.ts:444-449`). All 999 dB + RC/TLAS/DDGI oracles PASS. The THREE `materials` list now survives ONLY as the snapshot legacy-fallback + the T3 `updatePrimitive` consumer (`HybridEnginePrimitiveUpdates.ts:991` `bvh.buildMaterials`). `sceneRoots` stays until T3. The REMAINING THREE coupling is genuinely T2 (setScene synthesis) + T3 (updatePrimitive findMeshByPrimitiveId) — the ingestion/update layer, not the material packing.
- **T2 🔴** `setScene` synthesizes `vitrumSceneToThree` every build (`HybridEngineLifecycle.ts:323`).
- **T3 🔴** all 4 `updatePrimitive` paths + skin-refit use `findMeshByPrimitiveId`
  (`HybridEnginePrimitiveUpdates.ts:281/334/461/543/622`).
- **T4/T5 🔴** the resolver/update-layer core-native replacements + the
  `_ddgiTraversalScene`/`_synthesizedThreeScene` double-graph removal (ties to G-P2.2).
- **Oracle:** set-equivalence / converged-A/B (the SAH-order-differing producer-swap
  pattern proven in the emitter decouple, `46a0078`) — byte-or-converged identity
  vs the THREE path.

**Parallelization:** T1/T2/T3 are sequential-ish (resolver → setScene → updates)
but each is one focused workstream; can overlap with Phases 2/3.

---

## PHASE 5 — Release convergence  [after 1–4]

- **5-A** Fidelity-matrix promotion: every `experimental` row → `supported` as its
  Phase-0 oracle confirms it (the matrix already tracks this; promote on green).
- **5-B** Contract/API final pass: the API trio landed (`c9d88fa`); audit the
  contract for any remaining advertised-but-unimplemented surface; ensure
  `supportDetails`/`experimentalFeatures` honestly reflect every demote decision.
- **5-C** Examples: at least one example per backend that exercises the full
  stack end-to-end (working test scenes in `examples/`), each with a smoke test
  that actually runs. `two-engines-one-scene` now has a `test` script plus
  scene/harness/backend smoke coverage. Full browser/GPU page captures remain
  validation evidence, not a default `npm test` requirement.
- **5-D** Docs accuracy: CHANGELOG current; per-package READMEs match shipped
  reality (the audit found descriptions denying shipped features); CREDITS
  complete; the maturity wording consistent ("experimental backend", not
  "pre-alpha").
- **5-E ⚖️ Packaging** (maintainer call — no publish without instruction):
  npm-publish readiness (workspace → published), version, license, the
  `file:./packages/*` → published-dep flip. Distribution posture is yours.
- **5-F** Frontier (roadmap §8, de-prioritized, tracked not blocking v1): NRC
  default-on, ReSTIR-PT compositing/spatial-reuse follow-ups, cached light field
  extensions. Each needs verified-feasibility before scheduling (no SOTA cargo-cult).

---

## PARALLELIZATION MAP (how to actually run it)

```
Phase 0  ──┬─ 0-D ReSTIR oracle ─┐
           ├─ 0-E xbackend GT ───┤  (4 concurrent agents)
           ├─ 0-F denoiser ──────┤
           └─ 0-G fork WebGL2 ───┘   0-H spike (scope CI WebGPU)
                     │ (oracles ready)
Phase 1  ──┬─ 1-A DDGI octahedral (the work) ─┐
           ├─ 1-C three-bindings ─────────────┤  (3 concurrent)
           └─ 1-D re-validate P0 sweep ────────┘
                     │
   ┌─────────────────┼─────────────────┐  (Phases 2/3 overlap; Phase 4 deferred)
Phase 2          Phase 3            Phase 4 (deferred)
 2-A PPG          3-A perf-hygiene   4-DECISION ⚖️ deferred
 2-B WebGL-BDPT   3-B test-gaps×N    T1→T2→T3→T4/5 deferred
 2-D denoisers    3-C P2 residue
 2-F stained-glass
 (2-C/E/G small)
                     │ (all green, oracles confirm)
Phase 5  ── 5-A matrix promote · 5-B contract · 5-C examples · 5-D docs · 5-E ⚖️ pkg · 5-F frontier
```

## MAINTAINER DECISIONS — RESOLVED 2026-06-07; PHASE 4 UPDATED 2026-06-08
1. **Phase 4 — DEFERRED for the current maturity sweep.** THREE-decouple T1–T5
   stays open and explicitly out-of-scope for the current non-THREE-decouple work.
2. **2-C — DEMOTE.** Fork caustics → `approximate` (honest label; MNEE is the reference). No physical WebGL2 caustic impl.
3. **2-E — ENABLE on medium/low tiers.** Checkerboard ON in the `medium`+`low` quality presets, OFF for `ultra`+`high`. PRECONDITION: measure the real shade-pass GPU-time saving first (timestamps were unavailable on dzn — must confirm the win is real on a timestamp-capable adapter before flipping the presets; if the saving is marginal, report back).
4. **5-E — DEFERRED.** Packaging/publish posture parked (maintainer will revisit).
5. **0-H / validation — HYBRID.** `npm test` stays GPU-free; the pre-push lavapipe+dzn GPU smoke is the shader-execution gate, EXTENDED with analytic oracles (octahedral round-trip, energy conservation, residual→0). No GPU in CI `npm test`.

Everything else is engineering I execute and self-validate against the oracles
above — no sight required.
