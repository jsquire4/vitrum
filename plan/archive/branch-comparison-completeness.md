# Branch Completeness Comparison — main vs feat/plan-gaps

**Date**: 2026-05-09
**Branch A**: `main` at commit `6a0da62` (the `2cf887a` session + subsequent work; 28 commits total)
**Branch B**: `feat/plan-gaps` at commit `9cfc70f`; 8 commits ahead of main; 121 files changed; net -8,172 LOC

---

## Summary table

| Item | main (A) | feat/plan-gaps (B) | Notable diff |
|---|---|---|---|
| Sprint 1 | Scaffold + constants + HDRI presets | Scaffold + constants + HDRI presets + `PT_PREVIEW_RESOLUTION_FACTOR` + `PT_POSTPROCESS_WARMUP_SAMPLES` + `suggestSkipPostProcess` in core/pt-webgl | B adds runtime warmup signal; A missing resolutionFactor + warmup constant |
| Sprint 2 | cellPower, fork patch spec | cellPower, fork patch spec | Identical |
| Sprint 3 | CDF/MIS in shared-samplers; fork-patch spec; DoD item 3.3 open | CDF/MIS in shared-samplers; fork-patch spec; **3.3 marked complete in backlog** | B marks back-face NEE done in backlog only (doc commit) |
| Sprint 4 | lobeMask + liteMode in shared-samplers | Same | Identical |
| Sprint 5 | cameUniformPacker + MRT spec | Same | Identical |
| Sprint 6 | spatialFilter.wgsl + roughRefractionGLSL | Same | Identical |
| Sprint 7 | HG phase + SSS helpers | Same | Identical |
| Sprint 8 | Cauchy + jakobHanika in shared-samplers | Same | Identical |
| Sprint 9 | Shaders authored; dispatch **deferred** | Shaders authored; **dispatch wired in renderFrame** | B has live sampleBudget + resolve passes dispatched |
| Sprint 10a | SVGF shaders + bindings; dispatch **deferred** | SVGF shaders + bindings; **à-trous replaced by SVGF in WalkaroundGPUPipeline** | B wires SVGF variance + 5 atrous iterations live |
| Sprint 10b | oidnBridge.ts; host-side deferred | Same | Identical |
| Sprint 10c | bdptVertex + bdptMIS + 35 tests + fork-patch spec | Not present (deferred doc + no implementation) | A implemented; B is deferred |
| Sprint 11 | PPG shaders + buffers + 81 tests; dispatch deferred; **setPPGEnabled wires pipeline rebuild** | PPG shaders + buffers + 82 tests; dispatch deferred | A has pipeline rebuild on toggle; B does not |
| Sprint 12 | cieCmf + wavelengthSampling + cauchyIor + 50 tests + fork-patch spec | Deferred doc only | A implemented; B is deferred |
| Sprint 13 | WGSL inference kernels + InferenceGraph + UNet spec + 101 tests + training scaffold | Deferred doc only; **training scaffold deleted** | A implemented; B is deferred |
| babylon-bindings | Not present | Stub package present | B adds `@vitrum/babylon-bindings` |
| vitrumSceneToThree | Not present in three-bindings | Present; also wired in HybridEngine.setScene | B promotes to three-bindings; HybridEngine uses it for BVH + DDGI |
| SpectralCurve / Material fields (RFE-01) | Full: spectralAttenuation + dispersionAbbeNumber + scatteringCoefficient etc. (494-line scene.ts) | Minimal: spectralAttenuation + dispersionAbbeNumber only (290-line scene.ts) | A has complete RFE-01..05 fields; B has only Tier 1 |
| VITRUM_SPECTRAL_EXTENSION_KEY | Not exported from three-bindings | Exported from three-bindings/spectral.ts | B adds extension key |
| examples | cornell-box + stained-glass-mini | cornell-box + shared + two-engines-one-scene | B adds shared builder + dual-engine example; A has stained-glass-mini |
| phase-6-roadmap-backlog.md | Not present | Present (135-line DoD checklist) | B adds backlog tracker |
| sprint-N-benchmark.md stubs | Not present | All 13 benchmark stubs present | B adds benchmark templates |
| sprints-1-11-audit.md | Present | Deleted | A has the audit source doc |
| external-requests-status.md | Present | Deleted | A has the RFE tracking doc |
| neural-denoiser-training/ | Present (4 .md files in tools/) | Deleted | B removed training scaffold docs |
| H-1 PPG stride fix | Fixed (`leafIdx * 64u`) | Not fixed (`leafIdx * 32u`) | |
| H-2 PPG clamp fix | Fixed (`0xFFFFFFFFu`) | Not fixed (`0xFFFFFFu`) | |
| M-3 HWC↔NCHW tests | Fixed (direct round-trip tests added) | Not fixed | |
| L-3 BDPT audit comment | Added in index.ts | N/A (no BDPT) | |

---

## Sprint-by-sprint detail

### Sprint 1 — PT preview speed wins
- **A**: `PT_PREVIEW_BOUNCES = 3`, `PT_FINAL_BOUNCES = 10` in `packages/pt-webgl/src/constants.ts`. No `PT_PREVIEW_RESOLUTION_FACTOR`, no `PT_POSTPROCESS_WARMUP_SAMPLES`, no `suggestSkipPostProcess` field in `@vitrum/core`.
- **B**: Adds `PT_PREVIEW_RESOLUTION_FACTOR = 0.5` and `PT_POSTPROCESS_WARMUP_SAMPLES = 8` to `packages/pt-webgl/src/constants.ts:30,33`. Adds `hdriPresets.ts` with Poly Haven preset table + `loadHdriEquirect`. Adds `FrameOutput.suggestSkipPostProcess?: boolean` to `packages/core/src/frame.ts:172`. Emits it from `packages/pt-webgl/src/index.ts:193,219`. Cornell example wired with `PT_PREVIEW_OPTIONS` and `?hdri=` param.
- **Files differ**: `packages/pt-webgl/src/constants.ts`, `packages/pt-webgl/src/hdriPresets.ts` (B only), `packages/core/src/frame.ts`, `packages/pt-webgl/src/index.ts`.

### Sprints 2–8
- **A and B**: Vitrum-side scaffold identical. Fork-patch spec docs identical. Runtime integration deferred on both (fork not yet applied). Test suites match with minor count differences (see Test Count Delta section).

### Sprint 9 — Convergence + Welford
- **A**: Shaders authored (`sampleBudget.wgsl.ts`, `resolve.wgsl.ts`). `createVarianceBuffer` exported. Dispatch wiring **deferred** per `plan/sprint-9-walkaround-integration.md`. Neither shader is dispatched in `renderFrame`.
- **B**: Shaders authored. Dispatch **wired**. `WalkaroundGPUPipeline.ts` contains `sampleBudgetPipeline` and `resolvePipeline` compiled at line 262-263. Both dispatched in `renderFrame` at lines 485-521 (Pass 5.6 and Pass 5.7 per the comment block). `bindGroupBuilders.ts` and `bindGroupLayouts.ts` carry the corresponding BGL/BG builders.
- **Files differ**: `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts`, `packages/walkaround-hybrid/src/pipeline/bindGroupBuilders.ts`, `packages/walkaround-hybrid/src/pipeline/bindGroupLayouts.ts`, `packages/walkaround-hybrid/src/pipeline/pipelineCompiler.ts`, `packages/walkaround-hybrid/src/pipeline/resourceManager.ts`.

### Sprint 10a — SVGF
- **A**: `svgf.wgsl.ts` + `svgfBindings.ts` in shared-denoisers. Walkaround dispatch **deferred**. `WalkaroundGPUPipeline` uses plain à-trous (`atrousPipeline`) at line 188.
- **B**: Same shared-denoisers code. Walkaround dispatch **wired**. `WalkaroundGPUPipeline` has `svgfVariancePipeline` and `svgfAtrousPipeline` at lines 191-192. Pass 5 in `renderFrame` (lines 398-444) runs SVGF variance then 5 à-trous iterations; the old à-trous pipeline is gone. `timestampQueries.ts` updated for the new pass count.
- **Files differ**: Same as Sprint 9 list plus `packages/walkaround-hybrid/src/pipeline/timestampQueries.ts`.

### Sprint 10b — OIDN
- **A and B**: Identical (`oidnBridge.ts`, host-side deferred).

### Sprint 10c — BDPT
- **A**: `packages/shared-samplers/src/bdptVertex.ts` + `packages/shared-samplers/src/bdptMIS.ts` implemented. 35 tests in `__tests__/bdpt.test.ts`. `plan/sprint-10c-pt-fork-patch.md` fork-patch spec (314 lines). Exports in `index.ts` with L-3 audit comment. `plan/sprint-10c-deferred.md` archived.
- **B**: No BDPT source files. `plan/sprint-10c-deferred.md` present, `plan/sprint-10c-pt-fork-patch.md` **deleted**.

### Sprint 11 — PPG
- **A**: All PPG files + 81 tests. `HybridEngine.setPPGEnabled()` triggers `_rebuildPipeline()` immediately (wired in `05ad4be`). `createFrameResources` gates PPG buffer allocation on `ppgEnabled`.
- **B**: All PPG files + 82 tests. `setPPGEnabled` does not wire pipeline rebuild (that change is absent). One additional test in sprint11-ppg.test.ts (607 lines vs 587).

### Sprint 12 — Hero-wavelength spectral
- **A**: `src/cieCmf.ts` + `src/wavelengthSampling.ts` + `src/cauchyIor.ts` in shared-samplers. 50 tests in `__tests__/spectral.test.ts`. `plan/sprint-12-pt-fork-patch.md` (302 lines). `plan/sprint-12-deferred.md` archived.
- **B**: No spectral source files. `plan/sprint-12-deferred.md` present. `plan/sprint-12-pt-fork-patch.md` **deleted**.

### Sprint 13 — Neural denoiser
- **A**: 5 WGSL inference kernels in `packages/walkaround-hybrid/src/neural/wgsl/`. `InferenceGraph.ts`. `unetArchitecture.ts`. 101 tests. `tools/neural-denoiser-training/` (4 .md scaffold docs). `plan/sprint-13-walkaround-integration.md`. `plan/sprint-13-deferred.md` archived.
- **B**: No neural source files. `tools/neural-denoiser-training/` **deleted**. `plan/sprint-13-deferred.md` present. `plan/sprint-13-walkaround-integration.md` **deleted**.

---

## Audit finding status

| Finding | main (A) | feat/plan-gaps (B) |
|---|---|---|
| **H-1** PPG leaf stride mismatch | **Fixed**: `ppgUpdate.wgsl.ts:175` uses `leafIdx * 64u`; audit comment added at line 168-175 | **Unfixed**: `ppgUpdate.wgsl.ts:164` still uses `leafIdx * 32u` |
| **H-2** PPG radiance clamp at 256 nits | **Fixed**: `ppgUpdate.wgsl.ts:209` uses `0xFFFFFFFFu`; audit comment at 207-208 | **Unfixed**: `ppgUpdate.wgsl.ts:195` still uses `0xFFFFFFu` |
| **M-1** Light tree CDF misleading doc | Untouched on both branches — no rename, doc only, GPU unaffected | Same |
| **M-2** SVGF sigmaColor provenance claim | Untouched on both — doc only | Same |
| **M-3** HWC↔NCHW round-trip tests missing | **Fixed**: `_hwcToNchw`/`_nchwToHwc` exported and directly tested in `__tests__/oidnBridge.test.ts:117+` | **Unfixed**: test file has no direct layout transform tests |
| **M-4** PPG O(N) brute-force scan | Untouched (acknowledged); `plan/sprint-11-ppg-integration.md` notes it as optimization | Same |
| **M-5** mixturePdf zero-probability undocumented | Untouched on both | Same |
| **M-6** SVGF atrous iteration count not a shader uniform | Untouched on both | Same |
| **L-1** jakobHanika TODO not tracked in plan | Sprint 12 tracking entry in `plan/phase-6-roadmap.md:395-398` (pre-existing); `TODO` still in `jakobHanika.ts:49` on both | Same |
| **L-2** Light tree degenerate centroid no warning | Untouched on both | Same |
| **L-3** BDPT exports in index.ts before integration testing | **Addressed**: audit comment added in `packages/shared-samplers/src/index.ts:24-26` marking exports as deferred | N/A — no BDPT exports exist on B |

---

## RFE status

### RFE-01 — Spectral Attenuation and Dispersion
- **A**: `SpectralCurve` interface + `Material.spectralAttenuation` + `Material.dispersionAbbeNumber` in `packages/core/src/scene.ts`. `VITRUM_SPECTRAL_EXTENSION_KEY` not in three-bindings.
- **B**: Same `SpectralCurve` + both fields. Also exports `VITRUM_SPECTRAL_EXTENSION_KEY` from `packages/three-bindings/src/spectral.ts:9`.

### RFE-02 — Volume Scattering (Henyey-Greenstein)
- **A**: `Material.scatteringCoefficient`, `scatteringAnisotropy`, `scatteringCoefficientRGB` in `packages/core/src/scene.ts:247-269`.
- **B**: These fields are **not present** in `packages/core/src/scene.ts` (290-line file vs 494-line on A).

### RFE-03 — Subsurface Volume
- **A**: Implemented per `plan/external-requests-status.md` (fields in core + capabilities flag).
- **B**: Not present. `plan/external-requests-status.md` deleted.

### RFE-04 — Animated Scene Clock
- **A**: Implemented per `plan/external-requests-status.md`.
- **B**: Not present.

### RFE-05 — Bounded-Volume Emitters
- **A**: Implemented per `plan/external-requests-status.md`.
- **B**: Not present.

*Note: RFE-02 through RFE-05 status on B is inferred from the 290-line scene.ts (vs 494 lines on A) and absence of `external-requests-status.md`. The missing fields were not individually verified line-by-line but the file-size delta and absence of the tracking doc are consistent with none of RFE-02..05 being present.*

---

## Specific question answers

### 1. Sprint 9 runtime wiring — does feat/plan-gaps dispatch sampleBudget + resolve?

Yes. In `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` (commit `9a45390`):
- `sampleBudgetPipeline` compiled at line 262, dispatched at line 497 (`Pass 5.6`).
- `resolvePipeline` compiled at line 263, dispatched at line 520 (`Pass 5.7`).
- Both have dedicated bind group builders (`buildSampleBudgetBindGroup`, `buildResolveBindGroup`) and layouts in `bindGroupLayouts.ts`.
- The composite pass (Pass 6) reads `this.res.resolvedTexture` — the resolve output — not the raw atrous output.

### 2. Sprint 10a SVGF runtime — does feat/plan-gaps replace à-trous with SVGF?

Yes. In the same `WalkaroundGPUPipeline.ts` (commit `9cfc70f`):
- The old `atrousPipeline` field is gone. `svgfVariancePipeline` (line 191) and `svgfAtrousPipeline` (line 192) replace it.
- Pass 5 in `renderFrame` runs `svgfVariancePipeline` once (lines 405-428), then loops 5 iterations of `svgfAtrousPipeline` (lines 430-444).
- The SVGF atrous output feeds directly into temporal accumulation (Pass 5.5) and then into the sampleBudget + resolve passes.
- `timestampQueries.ts` updated to account for the variance pass + 5 atrous slots.

### 3. Sprint 3 back-face NEE — does the implementation exist on feat/plan-gaps that's missing on main?

No. Commit `c7131a1` (on feat/plan-gaps) is a **docs-only commit**: it updates `plan/phase-6-roadmap-backlog.md` to mark backlog item 3.3 as complete. Its commit message says "implementing 4x front-facing resampling in the three-gpu-pathtracer fork" — meaning the fork (at `~/projects/three-gpu-pathtracer/`) was patched, not vitrum itself. No new TypeScript or WGSL files changed. The vitrum-side CDF/MIS code that supports the fork patch exists on both branches identically. Branch A has `plan/sprint-3-pt-fork-patch.md` describing the fork changes; Branch B has the same doc plus the backlog checkbox ticked.

### 4. Plan docs reorganization principle

The reorg principle (read from `plan/phase-6-roadmap-backlog.md`): **separate tracking from spec**. Branch B introduces:
- `plan/phase-6-roadmap-backlog.md` — a single flat DoD checklist with checkboxes, one item per sprint, cross-linked to the narrative roadmap. Updated as sprints land.
- `plan/sprint-N-benchmark.md` stubs — fill-in-the-blank template for recording measured outcomes after fork patches are applied. One per sprint, all 13 present.

The deleted documents (`sprint-10c-pt-fork-patch.md`, `sprint-12-pt-fork-patch.md`, `sprint-11-ppg-integration.md`, `sprint-13-walkaround-integration.md`, `sprints-1-11-audit.md`, `external-requests-status.md`) are specs and audit artifacts for work that Branch B treats as **deferred or not-yet-triggered**. The reorg strips the spec docs for deferred sprints to reduce noise, replacing them with compact `sprint-N-deferred.md` files. The remaining fork-patch and integration specs (sprints 2–10b) are retained because their work is in the active queue.

### 5. `tools/neural-denoiser-training/` deletion — moved or removed?

Removed, not moved. All four files (`README.md`, `dataset_spec.md`, `export_weights.md`, `train.py.md`) are deleted in Branch B's diff; they do not appear anywhere else in the Branch B tree. No replacement directory was created. The deletion is consistent with Branch B treating Sprint 13 as fully deferred with no in-repo training infrastructure. Branch A retains all four files.

### 6. `vitrumSceneToThree` (commit baea01c) — what does this do that main doesn't have?

Three changes absent from Branch A:

1. **Function moved into `@vitrum/three-bindings`**: `vitrumSceneToThree` and `disposeVitrumThreeSceneRoot` are implemented in `packages/three-bindings/src/vitrumSceneToThree.ts` and exported from `packages/three-bindings/src/index.ts:21`. On Branch A, the function does not exist in three-bindings at all.

2. **`@vitrum/pt-webgl` depends on three-bindings and re-exports**: `packages/pt-webgl/src/index.ts` imports and re-exports `vitrumSceneToThree` via three-bindings. pt-webgl's `package.json` lists `@vitrum/three-bindings` as a dependency.

3. **`HybridEngine.setScene` uses it for BVH + DDGI**: When the scene has mesh primitives, `buildSceneBVH` calls `vitrumSceneToThree(this._lastScene!)` at `packages/walkaround-hybrid/src/HybridEngine.ts:701` to derive the THREE scene graph used for BVH construction. The same derived scene graph feeds DDGI probe traversal. On Branch A, `HybridEngine` builds BVH from `this._threeScene` (a THREE.Scene the host is expected to have set directly), meaning the walkaround engine on Branch A is **not** wired to accept a core `Scene` object for BVH purposes.

---

## Test count delta

| Package | main (A) | feat/plan-gaps (B) |
|---|---|---|
| pt-webgl | 18 | 21 |
| shared-bvh | 11 | 11 |
| shared-denoisers | 75 | 69 |
| shared-samplers | 143 | 54 |
| three-bindings | 1 | 1 |
| walkaround-hybrid | 294 | 194 |
| **Total** | **542** | **350** |

Counts verified by running `npm test --workspaces --if-present` in each worktree. Branch A's shared-samplers count (143) includes 54 base + 35 BDPT + 50 spectral + 4 HDRI preset tests; Branch B's 54 are base only. Branch A's walkaround-hybrid count (294) includes 193 base + 101 Sprint 13 neural tests; Branch B's 194 are base + 1 extra PPG test only. Branch A's shared-denoisers count (75) includes the M-3 HWC↔NCHW round-trip tests not present on B.
