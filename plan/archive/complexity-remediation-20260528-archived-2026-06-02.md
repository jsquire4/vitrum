> **Archived 2026-06-02 — completed/superseded. Current state: CLAUDE.md + git log on main.**

# Complexity-Remediation Plan — 2026-05-28 (full sweep)

Source: `/complexity-sweep` (19 scanners) → `/plan-implementation`. Raw findings + verified dead-code classification: `~/.claude/projects/-home-jsquire4-projects-vitrum/memory/in-flight-sweep-2026-05-28.md`.

## Scope (user-approved)
Full sweep: **5 self-verified latent bugs (A1–A5) + 13 structural themes**. Maximize parallelization while respecting the documented merge-race hazard: contract/shared-layer changes land FIRST (Wave 0), parallel backend work forks from that commit.

## Locked architecture decisions
- **T13 `updateLighting`** — core stays opaque (`Record<string,unknown>`); `@vitrum/engine` re-exports `LightingOptions`; `HybridEngine.updateLighting` adds an unknown-key `console.warn`. Rationale: `updateLighting` is HybridEngine's cheap per-frame time-of-day scrub, distinct from the universal `updateEnvironment` channel — don't bake one backend's vocab into the contract. Promote to core only on a 2nd-backend need.
- **T9 bind-groups** — staged-both. Step A: `SharedBindGroupPass` base (5-way dispatch dedup) + `buildPpgGuide/UpdateBindGroup` helpers. Step B: data-driven descriptor table generating BOTH the BGL factory and builder for the UNIFORM families (frame/scene/ubo/gtao/gi), with per-entry `note` preserving dead-binding rationale, escape hatches for the 3 non-uniform builders (atrous/accum/hybrid-layers), and a builder↔layout entry-count parity test.
- **T5 stained-glass** — extract `lo_sun_caustic`/`lo_sky_aperture` into a new opt-in `shaders/stainedGlassShade.wgsl.ts` WGSL module gated by an `sgParams.enabled` UBO bit, pulled into `shade` via the include-graph — mirroring the proven `SAMPLE_CASCADE_C0_MODULE` + `rcParams.enabled` precedent. Default-off. `shade.wgsl` carries zero stained-glass knowledge.

## Verified bugs (each fixed with a regression test)
- **A1** `shared-bvh/scenePack.ts:445,589` — leaf-flag `(x & 0xffff0000)===0xffff0000` is a dead branch (always false; also wrongly requires low-16 zero). Fix = new shared `isLeafSplit(word) => (word>>>16)===0xffff` used by scenePack/buildArrayBvh/bvhCommon. Affects 2nd+ primitive leaf-offset rebasing in concatenated BLAS packs.
- **A2** `shared-samplers/heroWavelengthTables.ts:18` — hardcoded `array<f32, 82>` for the 81-length CMF tables → fixed-size-array constructor mismatch in the **main pt-webgpu kernel**; latent because GPU compile is opt-in. Fix = derive size from `table.length`.
- **A3** BDPT bounce-0 tangent — `b[i]*x` should be `b[i]*y` (the computed `y` is currently dead). Duplicated in `pt-webgl/bdpt/bdptSceneEmittersCpu.ts:68-70` AND `pt-webgpu/bdpt/bdptEmitterPickCpu.ts:206-209`.
- **A4** `pt-webgl/ptEngineWebGL2.ts:160` — `isEmitterOnlyPatch` guards on phantom field `meshPrimitiveId`; core field is `meshId`. A meshId repoint wrongly takes the light-only fast path and skips the BVH rebuild. (Rides with the pt-webgl branch — see conflict note below.)
- **A5** `walkaround-hybrid/ppg/dTree.ts:357-360` — merged dTree children orphaned in-place; `nodes` grows unbounded across refine cycles. Fix = compaction.

## Wave plan (merge-race-safe DAG)

### Wave 0 — SERIAL foundation (one branch off `main`; everything forks from its merge commit)
Order: **B1+A1 → T2 → A2 → T13**.
- **B1+A1** (shared-bvh): add `isLeafSplit` to `bvhCommon.ts` + export; rewrite `scenePack.ts:445,589` and audit `buildArrayBvh.ts`/`bvhCommon.ts` leaf checks to use it. TDD: failing multi-primitive leaf-rebase test first.
- **T2** (core + 3 backends): new `core/src/scene/patchScene.ts` — `patchPrimitiveInScene`/`patchEmitterInScene` (carry pt-webgpu's stricter analytic-param validation) + `filterPatchByCapabilities` consuming `supported*Kinds`. Export from `core/index.ts`. Delete the 3 backend copies (`pt-webgpu/scene/patchScene.ts`, `pt-webgl/ptEngineWebGL2.ts:158-302` patch fns, `walkaround-hybrid/scenePatch.ts`) → thin re-export/consume. `isEmitterOnlyPatch` (A4) stays a pt-webgl predicate (Wave 2).
- **A2** (shared-samplers): `heroWavelengthTables.ts` derive `array<f32,${table.length}>`.
- **T13** (core/engine/hybrid): re-export `LightingOptions` from `@vitrum/engine`; unknown-key warn in `HybridEngine.updateLighting`.

**→ AUDIT CHECKPOINT 0:** `/audit` changed files; `npm run typecheck` + `npm test` green before any Wave-1 branch forks.

### Wave 1 — PARALLEL (7 branches, fork from Wave-0 merge)
- **W1-a / T3+dead** (shared-bvh, shared-samplers, pt-webgpu wgsl, hybrid restir): route pt-webgpu through canonical `safeInvDir`/Möller-Trumbore (delete copies in `pt-webgpu/wgsl/common.wgsl.ts`+`intersection*.wgsl.ts`); canonical TS `luminance` at `emitterList.ts:39`/`bdptEmitterPickCpu.ts:11`/`environmentPacking.ts:102`; delete `tlasBridge.ts` dead `refitSceneTlas`, re-export `buildSceneTlas`+types from shared-bvh; ReSTIR double-BVH-build dedup (drive emitter list from scenePack arrays).
- **W1-b / A3** (pt-webgl/bdpt, pt-webgpu/bdpt): `b*x`→`b*y` both sites; consumes dead `y`.
- **W1-c / A5+PPG-dead** (hybrid/ppg): dTree compaction; delete `aabbContains`, `computeMISWeights`, `PPG_MIS_ALPHA_MIN/MAX`, `PPGModelHandle`.
- **W1-d / T10+T20 denoisers** (shared-denoisers): new `albedoModulation.wgsl.ts` (demod/remod shared), `acquireDenoiseDevice` helper (3× preamble), delete dead imports `svgfRealWebGPU.ts:43-44`, collapse 5× D3-rename headers.
- **W1-e / T15 three-bindings** (three-bindings): `convertFirstMaterial` (drop dead `isPhys`), `buildGeometry`/`disposeMaterialTextures`/`rectAreaLightBasis` helpers.
- **W1-f / T9-stepA-WGSL** (hybrid/shaders): split `common.wgsl.ts` (695 LOC, 9 concerns) into focused `WGSL_MODULES`; passes `requires` minimum set. (Disjoint from passes/ → no conflict with bind-group work.)
- **W1-g / T10-DDGI** (hybrid/ddgi): blend/border shaders read `ddgiAtlasLayout` via factory fns (kill hardcoded IRR_CELL/VIS_CELL/strides); fold border irr/vis into one parameterized factory; delete dead re-exports `probeUpdatePass.ts:39-44`, `resourceManager.ts:28-29`.

**→ AUDIT CHECKPOINT 1** after all W1 branches merge.

### Wave 2 — PARALLEL (fork from Wave-1 merge)
- **W2-a / pt-webgl hoist + A4** (`ptEngineWebGL2.ts`): extract `scenePatch.ts` (the `:158-302` block), telemetry/config-parse helpers, centralize ForkAccess casts; **A4 rename `meshPrimitiveId`→`meshId` at `:160` rides HERE** (same file region as the extraction).
- **W2-b / pt-webgpu hoist** (`index.ts`): `scene/incrementalPatch.ts`, `buildFrameOutput()` helper (kills 3× literal at :1038/:1099/:1257), `GpuResources` sub-struct, typed mutators replacing readonly-cast ×4.
- **W2-c / three-bindings dedup**: `findMeshByPrimitiveId` canonical in three-bindings (walkaround-hybrid consumes, drops its copy).
- **W2-d / T16 emitter round-trip** (`HybridEngineLifecycle.ts`): new `coreEmittersToDDGILights` from the core scene (fix π-error lossy THREE round-trip); keep `collectDDGILightsFromThreeRoot` only for the raw-threeScene escape hatch.

**→ AUDIT CHECKPOINT 2** after all W2 branches merge.

### Wave 3 — SERIAL tail (converged hybrid god-files + fork-gated themes)
Order: **T4 → T6 → BVH→GI propagation → T5 → T9-stepB → GpuSkinningHost → dead-code/comments**.
- **T4** drop `as unknown as` casts at `HybridEngine.ts:185/194/209` (members already public); add typed `get frameResources()` for the `:489` `_res` reach-in.
- **T6** extract the DI reservoir-builder surface (`HybridEngine.ts:953-1012/1205-1268`).
- **BVH→GI propagation** single `propagateBvhToGiSubsystems` owner (collapses 4 drifted copies across ~11 hybrid files).
- **T5** new `stainedGlassShade.wgsl.ts` opt-in module + `sgParams` UBO (via `defineUbo`) + `HybridEngineOptions.stainedGlass` flags; `shade.wgsl` calls `lo_sg_*`.
- **T9-stepB** descriptor table for uniform BGL families + parity test + escape hatches; `SharedBindGroupPass` base (5 passes) + PPG bind-group helpers.
- **GpuSkinningSubsystem** narrow `GpuSkinningHost` interface replacing the whole-`HybridEngine` back-pointer (breaks the import cycle).
- **Dead-code/comments (last):** safe-rm `gpuSkinLbs.wgsl.ts` (+ correct stale CLAUDE.md "What's next" item 4), `surfaceTextureIds.ts` shim, `rcParamsLayout.ts` wrapper; delete `traceWgslForTier`, `_resetCacheUnsafe`, `cellPowerArray`; remove ESLint dead locals; mark test oracles `@internal` (`dTreeSample`, bdpt CPU mirrors, `fillBdptLightPathCpu`); `knip.json` fixes (`src/__tests__` entries, `eslint:true`); T12/T19 stale-comment sweep.

**→ AUDIT CHECKPOINT 3** + full `/retrospective`.

## Testing strategy
- **Gates every wave:** `npm run typecheck` + `npm test` green.
- **TDD (failing test first):** A1 (multi-primitive leaf rebase), A3 (tangent uses `y` — strengthen the degenerate-axis emitter fixtures that currently can't catch it), A4 (meshId change triggers rebuild), A5 (node count bounded across refine cycles), T9 parity.
- **New test files:** `shared-bvh/__tests__/scenePackLeafRebase.test.ts`; `hybrid/ppg/__tests__/dTreeCompaction.test.ts`; `core/__tests__/patchScene.test.ts` (incl. analytic-param validation); `hybrid/pipeline/__tests__/bindGroupParity.test.ts`; `pt-webgpu/__tests__/heroWavelengthWgslSize.test.ts` (cheap, non-GPU) + extend `wgslSmoke.gpu.test.ts` for A2.
- **Reference renders** (`tools/reference-renders/`): A3 BDPT Cornell (both pt-webgl + pt-webgpu); T5 Cornell-stained-glass with `sgParams.enabled` must be **bit-identical** to today's inline, generic scene with it off must drop caustic/aperture terms to **exactly zero**.
- **Existing tests to update:** `sprint18-indirectCombine.test.ts` (T5 moves the `Lo_*` names it pins); `bdptSceneEmittersCpu.test.ts`/`bdptLightPathPackWebGL.test.ts` (A3 — strengthen axes); `pt-webgpu/__tests__/{bdptEmitterPickCpu,bdptLightSubpathOracle,patchScene,traceTier,updateEmitterIncremental}.test.ts`.
- **Behavior-preserving themes** (T3/T4/T6/T7/T8/T10/T14/T15/BVH→GI): characterization tests pin outputs (packer buffer snapshots, denoiser output hashes, scene round-trips) before refactor; existing coverage noted per item.

## Execution strategy & regression safety
- **Parallelization:** each Wave-1/Wave-2 branch is an independent git worktree off the prior wave's merge; no two concurrent branches edit the same file (verified by the conflict table). Wave 0 and Wave 3 are serial single branches.
- **Merge-race mitigation:** every branch forks from the LATEST merged wave commit, never from stale `main` — this is the exact failure mode the repo history warns about.
- **Commit batching:** one focused commit per theme/sub-item; co-located dead code rides with its theme.
- **Failure modes:** a refactor silently changes behavior → caught by characterization tests / reference renders; a bug fix regresses → caught by its TDD test; a deletion breaks an import → caught by typecheck (and the dead-code verification already confirmed zero consumers).
- **Audit checkpoints are gates, not suggestions** — fix any God-file/mixed-concern/export-sprawl regression before the next wave forks.

## Definition of done
- All 5 bugs fixed with passing regression tests; A2 covered by a WGSL-compile test.
- All 13 themes landed; 3 forks implemented as decided.
- Verified dead code removed; "keep" items retained (test oracles `@internal`, codegen artifacts, public API, the NOT_DEAD false-positives untouched).
- `npm run typecheck` + `npm test` green; reference renders A/B'd and justified.
- Stale CLAUDE.md "What's next" item 4 corrected; `knip.json` config fixed.
- `/audit` clean after each wave; `/retrospective` at the end.
