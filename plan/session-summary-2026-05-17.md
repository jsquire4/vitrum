# Session summary — 2026-05-17 premium-grade refactor

A single autonomous `/implement-plan` + `/loop` session shipped essentially the entire `premium-grade-refactor-20260517.md` plan (W1–W13) PLUS every closeable `items_to_fix.md` entry. This doc indexes the 54 session branches + 90 commits so reviewers can navigate.

**Status:** all branches LOCAL; no remote pushes per CLAUDE.md doctrine. User decides push timing + merge order.

## Quick stats

- **54 branches** stacked on `29ccf96` (W1-R6 head).
- **90 unique commits.**
- **WalkaroundGPUPipeline.ts:** 1574 → 964 LOC during W1 alone (later splits shrink it further on respective branches).
- **pt-webgpu/wgsl/pathTraceBruteforce.wgsl.ts:** 1908 LOC monolith → 76 LOC orchestrator + 15 structured modules.
- **shared-denoisers/svgfRealWebGPU.ts:** 838 → 400 LOC.
- **HybridEngine.ts:** various reductions across W1-R2 (frame resources), W3-D12 (extensions), A3 (updatePrimitive); fully decomposed split (W4-A1) deferred — see "Remaining work" below.
- **Tests grew** 354 (baseline) → 1000+ across the workspace. Specific package counts vary by branch.
- **2 new packages:** `@vitrum/walkaround-rc` (RC extraction), `@vitrum/stained-glass-extensions` (host-app domain).
- **Items_to_fix.md verified-open list:** A1, A2, A3, A4, B3, B4, C1, C2, C3, C4 all CLOSED. B1 (PPG) shipped Phase 1+2 (numerical Phase 3 deferred). B2 (RC) CLOSED.

## Branches grouped by theme

### W1 — Pass + Resource + Denoiser registry (stacked R1→R6)

The foundation. Every other branch builds on `feat/refactor-w1r6-wgsl-include-graph` @ `29ccf96`.

| Branch | Commit | What |
|---|---|---|
| `feat/refactor-w1r1-pass-registry-foundation` | `11c4698` | Pass + PassRegistry + Denoiser foundation |
| `feat/refactor-w1r2-frame-resources-split` | `834f441` | FrameResources 45-field god-struct → 8 sub-structs |
| `feat/refactor-w1r3r4-denoiser-migration` | `987b3fb` | 6 denoiser entries in registry; 16+ string-switch sites removed |
| `feat/refactor-w1r5-declarative-pass-order` | `60dd48b` | 18 Pass classes; renderFrame becomes pass-loop |
| `feat/refactor-w1r6-wgsl-include-graph` | `29ccf96` | composeWgsl + WGSL_MODULES registry; 9 prepend patterns gone |

### Items_to_fix surgical closures

| Branch | Commit | What |
|---|---|---|
| `feat/items-to-fix-A1-A2-A4` | `11fb8f3` | createEngine.updateEnvironment proxy, attachVitrum.swapChainView, FrameInput.viewport doc + auto-setSize |
| `feat/items-to-fix-B3-neural-inputpacker` | `96ab815` | INPUT_PACKER_WGSL wired + dispose leak fix + shape-debate comment strip |
| `feat/a3-hybridengine-incremental-updates` | `d0d22b0` | HybridEngine.updatePrimitive + updateEmitter material-only fast path |

### W2 — Dedup canonicalisation

Each canonicalises one duplicated algorithm into the shared lib that logically owns it.

| Branch | Commit | What |
|---|---|---|
| `feat/w2-small-dedups-c10-c11-c12` | `8a40ffe` | Luminance + B3-spline kernel + GTAOUniforms |
| `feat/w2-geometry-primitives-c1-c3-c4` | `3446817` | BVH traversal WGSL into shared-bvh + octEncode bug fix in rc + atlas-coord dedup |
| `feat/w2-c2-bvh-builder-shared` | `44598ad` | THREE-independent `buildArrayBvh` in shared-bvh; pt-webgpu's becomes 19-LOC re-export |
| `feat/w2-c5-material-entry-unification` | `23d9c3a` | Canonical 16-float MaterialEntry in shared-bvh |
| `feat/w2-c6-c15-pcg-hash-primitives` | `651619e` | PCG/Fresnel/ONB/cosineHemisphere/hash unified across pt-webgpu + walkaround |
| `feat/w2-c7-c8-c9-restir-phat-canonicalise` | `7f4c417` | ReSTIR p̂ + castPrimary shared modules |
| `feat/w2-c13-ubo-codegen` | `44e8497` | `defineUbo` codegen + 2 UBOs migrated; ~20 follow-ups documented |
| `feat/w2-c14-spectral-cdf-reexport` | `e853f48` | shared-samplers CIE/CDF re-export; pt-webgl drops reimplementation |

### W3 — Contract hygiene

| Branch | Commit | What |
|---|---|---|
| `feat/w3-contract-hygiene-d5-d6-d13-d15` | `becfbae` | drop isHardwareGpu + brand Mat4 + drop raw-WGSL re-exports + drop underscored exports |
| `feat/w3-d1-material-readonly-mutationproxy` | `34e4665` | Material → MaterialSpec readonly (only 2 mutation sites; much lighter than expected) |
| `feat/w3-d2-d3-stained-glass-extensions-package` | `0f323bb` | New `@vitrum/stained-glass-extensions` package; AnalyticShape opened; dichroicLUTs externalised |
| `feat/w3-d4-denoiser-discriminated-union` | `4928d49` | DenoiserConfig DU with per-mode required config (neural→weights, oidn-final→modelUrl) |
| `feat/w3-d7-frame-output-discriminated-union` | `40cd837` | FrameOutput {kind:'skipped'|'rendered'} DU |
| `feat/w3-d8-engine-capabilities` | `1355be1` | EngineCapabilities-driven feature query; D9 createEngine.updateEnvironment forward implicit |
| `feat/w3-d12-hybridengine-options-via-extensions` | `c72e2d6` | 30 walkaround knobs → `extensions['walkaround-hybrid']` + warn-once shim |
| `feat/w3-d16-uniform-telemetry` | `8c96e4b` | Canonical FrameStats + pt-webgpu onFrame + qualityModes capability |
| `feat/w3-d17-d18-cleanup` | `9ea12c9` | D17 KEEP Material.extensions (used by three-bindings) + D18 DROP supportsMotionBlur |
| `feat/w3-d19-backendtexture-branding` | `5863cda` | BackendTexture<TBackend> branded |

### W4 — God-file dissolution

| Branch | Commit | What |
|---|---|---|
| `feat/w4-a4-pt-webgpu-shader-split` | `f640de5` | pathTraceBruteforce 1908→76 LOC + 15 structured WGSL modules |
| `feat/w4-a5-shade-wgsl-split` | `7b2e628` | shadeMain 383→144 (5 lo_* helpers extracted) |
| `feat/w4-a7-svgfreal-split` | `489111d` | svgfRealWebGPU 838→400 LOC + texture upload dedup across 3 drivers |
| `feat/w4-a8-probeupdatepass-split` | `6d3ade3` | 5 per-phase DDGI Pass classes |
| `feat/w4-a9-ptengine-webgpu-split` | `e9b759d` | PTEngineWebGPU class → 4 engine/* modules |

### W6 — Hidden globals

| Branch | Commit | What |
|---|---|---|
| `feat/w6-hidden-globals-pt-webgl` | `a6a3c90` | iblBakerCache instance + ForkAccess (encapsulates fork-private `_pathTracer.material.uniforms`) |
| `feat/w6-hidden-globals-e1-e3` | `7d85bb0` | sharedWebGpuDevice flipped to test-only; window.__WGPU__ writes removed from engine |

### W7 — Stale + misplaced + dead-code

| Branch | Commit | What |
|---|---|---|
| `feat/w7-dead-code-batch` | `0444729` | G2/G3/G5/G7/G9 deletions/un-exports |
| `feat/w7-misplaced-code-moves` | `84f474b` | H1/H4/H5/H7/H8 file moves preserving git history |
| `feat/w7-tail-g6-g8-stale` | `9199dcd` | spatialFilter exported; shared-samplers triaged (5 production / 36 oracles); Sprint 12 stale comments fixed |
| `feat/w7-h2-h3-plus-future-doc-accuracy` | `69f1237` | DDGI atlas-octahedral + SceneBvh out of shared-bvh; sprint-future doc rewrite |
| `feat/w7-h6-surface-textures-split` | `01fbf89` | glassVisibility (library) + stained-glass/surfaceMods (host) split |

### W8 — RC extraction

| Branch | Commit | What |
|---|---|---|
| `feat/w8-rc-extract-to-new-package` | `9126d59` | RC → `@vitrum/walkaround-rc`; 5 `__gpuBuffer` reach-throughs rewritten to raw GPUBuffer |

### W9 — PPG GPU dTree

| Branch | Commit | What |
|---|---|---|
| `feat/w9-ppg-gpu-dtree-finish` | `3327ee8` | Phase 1: serialiseDTree/STree + real WGSL traversal kernels + real dispatch (was dispatchWorkgroups(0,0,0)) |
| `feat/w9-phase2-ppg-mis-in-shade` | `adcf11a` | Phase 2: power-heuristic MIS in shade.wgsl lo_indirect (ReSTIR + PPG candidates; sentinel fallback today) |

### W10 — Neural full finish

| Branch | Commit | What |
|---|---|---|
| `feat/w10-neural-full-finish` | `452a0a6` | New `examples/neural-denoiser/` workspace + acceptance test + buildRandomWeightsForSpec helper + README updates |

### W11 — OIDN wires (both backends)

| Branch | Commit | What |
|---|---|---|
| `feat/w11-oidn-wire` | `74fad35` | HybridEngine side: kick-and-return async pattern; registerBuiltinDenoisers options bag |
| `feat/w11-pt-webgl-oidn` | `7e98d90` | pt-webgl side: OIDNFinalDispatcher; cornell-box can drop manual wire |

### W12 — Dev overlays

| Branch | Commit | What |
|---|---|---|
| `feat/w12-dev-overlays` | `af02305` | 4 stub React components implemented (DDGIAtlasViewer, BVHVisualizer, GISignalSplit, MaterialInspector) + 2 new debug-surface methods (ddgiAtlasReadback, giSignalReadback) |

### W13 — Documentation reconciliation

| Branch | Commit | What |
|---|---|---|
| `feat/docs-c1-c3-claude-md-changelog` | `8cacb78` | First docs reconciliation: CLAUDE.md + CHANGELOG.md catchup |
| `chore/w13-readme-audit-plan-archive` | `413923a` | walkaround README updated; 11 plan docs archived to plan/archive/ |
| `chore/final-docs-catchup-and-cornell-oidn-simplify` | `7243ebc` | Second CLAUDE.md catchup |
| `chore/missing-package-readmes` | `ec58f48` | 7 READMEs added (core/engine/dev/shared-bvh/shared-denoisers/shared-samplers/three-bindings) |

### Hygiene + bug fixes

| Branch | Commit | What |
|---|---|---|
| `fix/examples-typecheck` | `ac1b593` | 12 typecheck errors in cornell-box + two-engines-one-scene + pt-webgpu process ref |
| `fix/gpu-tests-opt-in` | `006debc` | npm test no longer requires Playwright; test:gpu opt-in |
| `fix/pt-webgl-tests-three-gpu-pathtracer-dep` | `ce87517` | Worktree-path resolution for sibling repo (4-tier lookup + env override) |
| `chore/gitignore-cron-lock` | `df2d877` | .claude/scheduled_tasks.lock ignored (later reverted by user) |

### Decisional follow-ups

| Branch | Commit | What |
|---|---|---|
| `feat/pt-webgpu-bsdfsample-struct` | `00155af` | BsdfSample {wi,pdf,value} unification across 3 samplers (W4-A4 deferred follow-up) |

## Remaining work

Marginal items deferred at session-end:

- **W4-A1** HybridEngine fully decomposed beyond what W3-D12 + W6-E3 + A3 achieved. The orchestration core that remains is genuinely irreducible without changing the engine contract.
- **W4-A6** common.wgsl split — conflicts with W2-C6's primitive extractions. Diminishing returns at this point.
- **~18 remaining UBO migrations** per the W2-C13 follow-up list (currently in flight as `feat/w2-c13-followup-ubo-migrations` at session end).
- **W9 Phase 3** — numerical validation that PPG beats vanilla NEE on caustic-heavy scenes. Render-budget work.
- **pt-webgpu glossy BSDF sampling/PDF mismatch** — math correctness, was scoped to the 2026-05-11 sweep (not the 05-17 structural one); the W4-A4 split + BsdfSample struct made the architectural fix easier; the actual math fix is a separate workstream.

## Merge strategy

Every branch stacks on R6 = `29ccf96`. Suggested sequence (least-conflict → most-conflict):

1. **W1 chain first** (R1 → R2 → R3+R4 → R5 → R6).
2. **Hygiene fixes** (fix/examples-typecheck, fix/gpu-tests-opt-in, fix/pt-webgl-tests, chore/gitignore-cron-lock).
3. **W7 cleanup** (dead-code, misplaced, tail, H2/H3, H6).
4. **W6 hidden globals + W12 dev overlays + W11 ×2 OIDN.**
5. **W2 dedup chain** (small → geometry → PCG/hash → MaterialEntry → BVH builder → spectral CDF → ReSTIR p̂ → UBO codegen + follow-ups).
6. **W3 contract chain** (D5/D6/D13/D15 → D7 → D8 → D12 → D16 → D17/D18 → D19 → D1 → D2/D3 → D4).
7. **W4 splits** (A4 → A5 → A7 → A8 → A9; A4 BsdfSample follow-up).
8. **W9 Phase 1 + Phase 2.**
9. **W8 RC extraction.**
10. **A3 incremental updates** (depends on W3-D8 EngineCapabilities).
11. **W10 neural example** (depends on B3 + DenoiserConfig D4).
12. **W13 docs + chore/missing-package-readmes + chore/final-docs-catchup** last.
13. **chore/session-summary** (this doc) anywhere late.

Estimate: ~2-3 days of careful sequential merging with conflict resolution. Or: squash-merge each branch to a single commit and let `git merge` resolve.
