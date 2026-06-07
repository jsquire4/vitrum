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
- **Phase 4 THREE-decouple T1–T5** — real, intricate, byte-identity-gated.
The hygiene/coverage phases (3) are mostly DONE; treat them as verify-and-add-
coverage, not implement. Re-scope effort: closer to a defensible v1 than the
audit's "weeks" implied, MINUS the genuinely-deep items (DDGI math, T1–T5).

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
| **0-D** | **ReSTIR-DI/GI vs reference** — RIS-only vs reuse bias + an independent unbiased estimator on a shared scene; p̂-consistency probe across RIS/temporal/spatial passes | the G-P0.1 class | 🔴 build |
| **0-E** | **Cross-backend ground truth** — pt-webgpu converged (the physical reference) vs walkaround GI on a shared scene, exposure-aligned; relative-L2 on the indirect channel | whole walkaround GI stack | 🔴 build |
| **0-F** | **Denoiser oracle** — SVGF/BMFR/OIDN: converged-input identity (denoise(converged)≈converged) + variance-reduction-without-bias on a noisy input vs its own converged reference | shared-denoisers | 🔴 build |
| **0-G** | **pt-webgl fork oracle** — a WebGL2 capture path (currently none; lavapipe is WebGPU-only) so the fork's default-path radiometry (G-P0.3) + caustics + BDPT get the same ground-truth treatment | the whole fork | 🔴 build (largest; unblocks pt-webgl promotion) |
| **0-H** | **In-tree shader-execution harness** — bridge the "npm test never runs shaders" gap: a headless-WebGPU vitest path (lavapipe in CI) that runs the smallest radiometric kernels against analytic references, so a subset of fidelity leaves the external-only harness | systemic test-architecture gap | ⚖️ scope (CI WebGPU availability) |

**Parallelization:** 0-D, 0-E, 0-F, 0-G are independent → 4 concurrent agents.
0-H is a scoping spike (decide feasibility before investing).

---

## PHASE 1 — P0 default-path correctness  [gated by Phase 0 oracles]

> What every user renders by default must be physically correct. Most P0 items
> are already validated; the open ones are below. Each is days-scale with the
> right independent-reference A/B.

- **1-A 🔴 DDGI octahedral axis-aligned sampling bias.** Floors/walls under-read
  DDGI irradiance 23–60% (diagonals 2–4%) — production `ddgiSample` octahedral
  EDGE sampling (`ddgiSampleWgsl.ts:133-135`), NOT the border-store (that
  off-by-one hypothesis was ground-truth-rejected + reverted 2026-06-07).
  Derive the correct seam reflection / sample-position (full Cigolle 2014 oct
  seam map; the equator/pole folds need more than an edge mirror). **Oracle 0-A**
  (target: axis-aligned ≤5% like the diagonals).
- **1-B ✅ DDGI coloured-bounce.** Validated 2–4% on interior normals (`8aa444a`).
  Re-confirm after 1-A.
- **1-C 🟡 G-P0.4 three-bindings asymmetries.** Skinned double-transform FIXED;
  ShaderMaterial guards added (Codex). Verify the remaining "silent data drops /
  wrong-space conversions" the audit cited are closed; add round-trip tests.
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

- **2-A 🔴 PPG production-readiness.** Training writer EXISTS (`PPGCoordinator.ts:266`)
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
  (environmental — **Oracle 0-G** fork WebGL2 capture) + a BDPT variance-win A/B
  vs unidirectional on a hidden-emitter caustic scene.
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

- **3-A 🟡 G-P2.6 performance hygiene** (the largest P2): bind-group/view
  memoization (~20-30 createBindGroup + dozens of createView per frame, change
  only on resize/setScene); SVGF's ~80-90 MB unconditional textures when the
  denoiser isn't svgf-real; `estimatedGpuMemoryBytes` omits BVH/TLAS/light-tree;
  fork re-uploads static CIE tables every call; document `BDPT_CONTRIBUTION_CLAMP`.
- **3-B 🔴 G-P2.7 test gaps** (highly parallel — disjoint packages):
  - `HybridEnginePrimitiveUpdates` (~1,055 lines, zero tests; mocks return null
    so refit bodies are unreachable — needs real fixtures).
  - `scene-lighting` + `stained-glass-extensions` (zero tests; incl. `packCameUBO`
    GPU wire contract).
  - `examples/two-engines-one-scene` (2 test files, no `test` script → never run).
  - walkaround-rc kernels (`probeRayCast`/`cascadeMerge` — string pins only;
    behavior test env-gated off → the numeric oracles from Phase 0-C make these
    runnable).
  - GI-state `serialize/deserialize` round-trip; TLAS rotation/scale instance
    transforms (fixtures are identity/translation only); three-bindings
    STEP/CUBICSPLINE animation import.
- **3-C 🟡 G-P2.1–2.5 finish** (mostly ✅): verify the fork lint gate is green +
  add fork lint to a pre-push hook (so it can't rot again); close remaining
  dispose leaks (`_synthesizedThreeScene` in dispose, fork `_lowResPathTracer` /
  `_colorBackground` / CubeToEquirect leak); finish the dead-code classification
  + stale-comment cluster (G-P2.4/2.5 — most marked ✓, sweep the residue).

**Parallelization:** 3-A | each 3-B package | 3-C → many concurrent agents.

---

## PHASE 4 — THREE-decouple (T1–T5)  [⚖️ STRATEGIC DECISION FIRST · concurrent with 2/3]

> The GI-signal DATA paths are decoupled (emitter/DDGI/per-tri-material/RC). The
> INGESTION / RESOLVER / UPDATE layer is still THREE. **Maintainer decision: is
> host-agnosticism a v1 requirement, or a documented internal dependency?**

- **4-DECISION ⚖️** — v1 requirement → execute T1–T5; else document the THREE
  dependency honestly in the contract + capabilities and **skip to Phase 5**.
- **T1 🔴** `buildMaterialResolver` traverses THREE (`sceneBvhFromCore.ts:54-118`). **SCOPED 2026-06-07 (byte-identity-critical):** the `coreMaterials` DATA already comes from the core scene (`coreByName` map, lines 76-97), but the material SLOT ORDERING + dedup is driven by `root.traverseVisible(sceneRoots)` + `materials.indexOf(mat)` (THREE object/material identity). The per-triangle packers index by `geo.triMaterialIds`, produced by THIS ordering — so the decouple must REPRODUCE the exact slot order from `scene.primitives` (matching `vitrumSceneToThree`'s object-creation order, `vitrumSceneToThree.ts:312/329/341`) + dedup by core MaterialSpec identity, NOT a THREE traversal. The ordering used here also DIFFERS from `mergeWorldSpaceFromCore`'s structural dedup (the already-decoupled geometry path) — T1 must either unify them or reproduce the THREE order exactly. **A half-done ordering change silently corrupts per-triangle material assignment** → gate EVERY step with the byte/converged-A/B oracle (`ddgi-white-bounce-ab` set-equivalence pattern + a per-tri material-id A/B vs the THREE path). Step 1: replicate the THREE traversal order over `scene.primitives`, byte-A/B confirm, THEN drop the `sceneRoots` param.
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
  that actually runs (fixes the `two-engines-one-scene` no-`test`-script gap).
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
   ┌─────────────────┼─────────────────┐  (Phases 2/3/4 overlap — file-disjoint)
Phase 2          Phase 3            Phase 4
 2-A PPG          3-A perf-hygiene   4-DECISION ⚖️
 2-B WebGL-BDPT   3-B test-gaps×N    T1→T2→T3→T4/5
 2-D denoisers    3-C P2 residue
 2-F stained-glass
 (2-C/E/G small)
                     │ (all green, oracles confirm)
Phase 5  ── 5-A matrix promote · 5-B contract · 5-C examples · 5-D docs · 5-E ⚖️ pkg · 5-F frontier
```

## MAINTAINER DECISIONS — RESOLVED 2026-06-07
1. **Phase 4 — DO IT.** Host-agnosticism (THREE-decouple T1–T5) IS a v1 requirement. Execute.
2. **2-C — DEMOTE.** Fork caustics → `approximate` (honest label; MNEE is the reference). No physical WebGL2 caustic impl.
3. **2-E — ENABLE on medium/low tiers.** Checkerboard ON in the `medium`+`low` quality presets, OFF for `ultra`+`high`. PRECONDITION: measure the real shade-pass GPU-time saving first (timestamps were unavailable on dzn — must confirm the win is real on a timestamp-capable adapter before flipping the presets; if the saving is marginal, report back).
4. **5-E — DEFERRED.** Packaging/publish posture parked (maintainer will revisit).
5. **0-H / validation — HYBRID.** `npm test` stays GPU-free; the pre-push lavapipe+dzn GPU smoke is the shader-execution gate, EXTENDED with analytic oracles (octahedral round-trip, energy conservation, residual→0). No GPU in CI `npm test`.

Everything else is engineering I execute and self-validate against the oracles
above — no sight required.
