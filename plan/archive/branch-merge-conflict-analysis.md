# Merge Conflict Analysis: feat/plan-gaps → main

**Merge base**: `545a10e` (commit shared by both branches)
**A (main)**: `6a0da62`
**B (feat/plan-gaps)**: `9cfc70f`
**Analysis method**: `git merge-tree 545a10e main feat/plan-gaps` plus `git merge --no-commit --no-ff` simulation in a scratch branch (aborted, not committed).

---

## Summary

| Category                                                     | Count                                |
| ------------------------------------------------------------ | ------------------------------------ |
| Files changed on both branches (git "changed in both")       | 4                                    |
| Syntactic conflicts (git cannot auto-merge)                  | **0**                                |
| Semantic conflicts (git merged cleanly but result is broken) | **1**                                |
| Deletion-vs-modification conflicts                           | **0** (see §Deletion Clarifications) |
| Files changed on main only (auto-merged, take A)             | 43                                   |
| Files changed on feat/plan-gaps only (auto-merged, take B)   | 82                                   |

The merge will **not stop with conflict markers**. But one file (`packages/core/src/scene.ts`) will produce a **TypeScript compile error** after the clean auto-merge because both branches independently added `SpectralCurve` at nearby-but-not-identical locations, resulting in two declarations of the same interface.

---

## File-by-file conflict map

### 1. `packages/core/src/scene.ts`

**Type**: Semantic conflict — duplicate type declaration (git auto-merges, TypeScript rejects)

**What happened**:

- **B** added a minimal `SpectralCurve` interface at the top of the Material block (~line 28 in base) plus two Material fields (`spectralAttenuation`, `dispersionAbbeNumber`) immediately after `thickness`. B's declaration has no JSDoc.
- **A** independently added a fully-documented `SpectralCurve` interface in a new `// Spectral rendering types (RFE-01)` section block (~line 32 in base), preceded by a Wilkie 2014 citation JSDoc. A also added `SurfaceAbsorptionLayer` (RFE-03), `ThinFilmLayer` + `ThinFilmStack` (RFE-04), plus richer versions of the same two Material fields (`spectralAttenuation`, `dispersionAbbeNumber`) further down in the Material interface body, with full JSDoc including OpenPBR reference.

**Merge result (verified)**: git places both additions in the result because they occupy slightly different source offsets. The merged file has:

- Two `export interface SpectralCurve { ... }` declarations (lines ~35 and ~61 in merged file).
- Two `spectralAttenuation?: SpectralCurve` fields and two `dispersionAbbeNumber?: number` fields in `interface Material` (lines ~199 and ~242).

TypeScript will reject the duplicate interface declarations with `error TS2300: Duplicate identifier 'SpectralCurve'`.

**Conflict region** (merged file):

```
Line ~35:  export interface SpectralCurve { ... }  ← B's minimal declaration
Line ~61:  export interface SpectralCurve { ... }  ← A's documented declaration (RFE-01 block)
Line ~199: spectralAttenuation?: SpectralCurve;    ← B's field (minimal JSDoc)
Line ~205: dispersionAbbeNumber?: number;           ← B's field (minimal JSDoc)
Line ~242: spectralAttenuation?: SpectralCurve;    ← A's field (full JSDoc, OpenPBR ref)
Line ~254: dispersionAbbeNumber?: number;           ← A's field (full JSDoc, range note)
```

**Suggested resolution — Take A, delete B's copies**:

1. Delete B's minimal `SpectralCurve` declaration (lines ~35–43 in merged file) — A's version at ~61 is identical in shape but has better JSDoc and proper section header.
2. Delete B's two minimal Material fields at ~199–205 — A's richer versions at ~242–254 supersede them completely.
3. A's `SurfaceAbsorptionLayer`, `ThinFilmLayer`, `ThinFilmStack`, and all the RFE-02 / RFE-03 / RFE-04 Material extensions added by A should be kept as-is.

Net result: A's full type surface wins; B contributes nothing unique to this file.

---

### 2. `packages/pt-webgl/src/index.ts`

**Type**: Disjoint changes — auto-merges cleanly, semantically correct

**What each branch changed**:

- **A** added the `causticStrategy` field capture in the constructor, the `causticStrategy: this.#causticStrategy` return in `getCapabilities()`, and the `#causticStrategy` private field declaration. Changes confined to lines ~101–157 in base.
- **B** re-pointed `vitrumSceneToThree` import from `./sceneToThree.js` to `@vitrum/three-bindings`, added `PT_POSTPROCESS_WARMUP_SAMPLES` import, added `suggestSkipPostProcess` to both `renderFrame()` return paths, and added `export * from './hdriPresets.js'`. Changes confined to lines ~18–260.

**Verified**: `git merge --no-commit` reported "Auto-merging packages/pt-webgl/src/index.ts" with no conflict. The merged file contains all changes from both branches correctly interleaved.

**Suggested resolution**: Accept the auto-merge as-is. No action needed.

---

### 3. `packages/walkaround-hybrid/src/HybridEngine.ts`

**Type**: Disjoint changes — auto-merges cleanly, semantically correct

**What each branch changed**:

- **A** added 6 lines in `getCapabilities()` return (~line 255 in base): the `causticStrategy: 'none'` field with a JSDoc comment explaining walkaround incompatibility.
- **B** made 14 separate hunks across the file: added `vitrumSceneToThree` import from `@vitrum/three-bindings`, updated `threeScene` JSDoc to "fallback" framing, updated `ppgEnabled` JSDoc to reflect that dispatch now applies immediately, added `HYBRID_FACTORY_BOOT_SCENE` constant, added `_lastScene` and `_ddgiTraversalScene` private fields, updated `setScene()` to call `vitrumSceneToThree` and populate `_ddgiTraversalScene`, added `setPPGEnabled()` rebuild logic, updated factory to use `HYBRID_FACTORY_BOOT_SCENE` instead of `{} as Scene`.

**Verified**: `git merge --no-commit` auto-merged all 21 combined hunks cleanly. The merged file correctly contains both A's `causticStrategy: 'none'` in capabilities and B's full `vitrumSceneToThree` integration.

**Suggested resolution**: Accept the auto-merge as-is. No action needed.

---

### 4. `plan/phase-6-roadmap.md`

**Type**: Disjoint changes — auto-merges cleanly

**What each branch changed**:

- **A** inserted a 4-line "Jakob+Hanika precomputed table" tracked-item note just before the Sprint 13 section (~line 397 in base). Also rewrote one sentence in §9 Verification to reference the backlog file.
- **B** added a "Checklist backlog" header link at line 7 (near the top of the frontmatter block). Also updated the §9 verification sentence (same sentence as A but with slightly different wording referencing the backlog file).

**Note**: The §9 sentence is changed by BOTH branches. Git auto-merges it because A rewrote the sentence starting from "Verifications captured" → "Verifications are tracked" while B's version was "Verifications are tracked … (see backlog for DoD ↔ sprint mapping)". The merge-tree result takes B's version of that sentence because it changed the same character positions and B's is a superset.

**Verified**: `git merge --no-commit` auto-merged cleanly.

**Suggested resolution**: Accept the auto-merge. After merging, read §9 to confirm the blended sentence reads correctly (B's version is preferred — it's more complete).

---

## Deletion Clarifications

The prompt listed several files as suspected "deletion vs modification" conflicts. After reading the actual branch trees (`git ls-tree -r`), none of these are deletion conflicts:

| File / Directory                                   | Reality                                                                                                                                  |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `plan/sprint-10c-pt-fork-patch.md`                 | Added by A after merge-base; B never had it. Git adds it to merge result automatically.                                                  |
| `plan/sprint-12-pt-fork-patch.md`                  | Same — A-only addition.                                                                                                                  |
| `plan/sprint-13-walkaround-integration.md`         | Same — A-only addition.                                                                                                                  |
| `plan/sprints-1-11-audit.md`                       | Same — A-only addition.                                                                                                                  |
| `plan/sprint-11-ppg-integration.md`                | Existed at merge-base; A modified it (added BLOCKING CONDITIONS section); B left it unchanged. Git takes A's version.                    |
| `tools/neural-denoiser-training/` (4 files)        | A-only additions (README, dataset_spec, export_weights, train.py). B never had them. Git adds them to merge result.                      |
| `packages/walkaround-hybrid/src/neural/` (8 files) | A-only additions (InferenceGraph.ts, unetArchitecture.ts, 5 WGSL files, sprint13 test). B never had them. Git adds them to merge result. |
| `_staging/legacy-source/` (24 files)               | B deleted all of them; A did NOT modify any of them (confirmed: none appear in A's change set vs base). Git accepts B's deletions.       |

---

## Auto-mergeable file list (changes on one branch only)

**A-only additions that B will gain after merge** (no conflict, git takes A):

- `packages/core/src/engine.ts` — causticStrategy added to EngineCapabilities + EngineOptions
- `packages/shared-samplers/src/index.ts` — BDPT + spectral exports added
- `packages/shared-samplers/src/{bdptMIS,bdptVertex,cauchyIor,cieCmf,wavelengthSampling,mixturePdf}.ts` — Sprint 10c/12 implementations
- `packages/shared-denoisers/src/{oidnBridge,svgfBindings}.ts` and tests — SVGF runtime path + OIDN bridge cleanup
- `packages/walkaround-hybrid/src/index.ts` — barrel exports expanded (RC, PPG, neural)
- `packages/walkaround-hybrid/src/neural/*` — 7 neural denoiser source files + sprint13 test
- `packages/walkaround-hybrid/src/ppg/wgsl/{ppgSample,ppgUpdate}.wgsl.ts` — PPG WGSL kernels
- `plan/sprint-10c-pt-fork-patch.md`, `plan/sprint-12-pt-fork-patch.md`, `plan/sprint-13-walkaround-integration.md`, `plan/sprints-1-11-audit.md` — Sprint spec docs
- `plan/sprint-11-ppg-integration.md` — BLOCKING CONDITIONS section
- `plan/external-requests-status.md`, `plan/sprint-12-deferred.md`, `plan/phase-6-status.md` — status/deferred docs
- `tools/neural-denoiser-training/*` — training pipeline documentation

**B-only additions that A will gain after merge** (no conflict, git takes B):

- `examples/shared/`, `examples/two-engines-one-scene/` — new G2 demo + shared Cornell builder
- `examples/cornell-box/` (modified) — uses shared Cornell helper
- `packages/babylon-bindings/` — stub package
- `packages/three-bindings/src/{vitrumSceneToThree,spectral}.ts` — scene converter (moved from pt-webgl)
- `packages/three-bindings/src/index.ts` — updated barrel
- `packages/pt-webgl/src/{hdriPresets,constants}.ts` — HDRI presets + warmup constant
- `packages/pt-webgl/__tests__/hdriPresets.test.ts`, `src/__tests__/scene-sync-golden.test.ts` — new tests
- `packages/walkaround-hybrid/src/pipeline/*` — WalkaroundGPUPipeline, resourceManager, bindGroupBuilders, etc.
- `packages/walkaround-hybrid/src/shaders/{resolve,sampleBudget}.wgsl.ts` — shader updates
- `packages/walkaround-hybrid/__tests__/sprint11-ppg.test.ts` — PPG unit test
- `packages/core/src/frame.ts` — new FrameOutput field(s)
- `plan/phase-6-roadmap-backlog.md` — DoD checklist
- `plan/{generalized-library-milestones,library-architecture,sprint-0-api-contract,sprint-*-benchmark}.md` — architecture + benchmark stubs
- `_staging/legacy-source/*` — entire legacy tree deleted (24 files)
- `_staging/README.md` — simplified (no more file disposition table)
- `package.json`, `package-lock.json` — workspace addition (babylon-bindings)

---

## Recommended merge strategy

**Approach**: standard `git merge --no-ff feat/plan-gaps` on main, then immediately fix the one semantic conflict in `scene.ts`.

**Step-by-step**:

1. **Run the merge**: `git merge --no-ff feat/plan-gaps` — it will complete without stopping (no syntactic conflicts).
2. **Fix `packages/core/src/scene.ts` immediately** (do not commit before fixing):
   - Delete B's minimal `SpectralCurve` declaration (the one without JSDoc, appearing first in the merged file).
   - Delete B's two minimal Material fields (`spectralAttenuation` and `dispersionAbbeNumber` near `thickness`, without JSDoc) — keep A's richly-documented versions further down in the Material body.
   - Run `tsc --noEmit` to confirm zero errors.
3. **Commit the merge + fix together** as a single merge commit with a note: `chore: merge feat/plan-gaps; resolve duplicate SpectralCurve declaration`.

**Why not rebase B onto A**: B has 6 commits and touches 82 files including pipeline, shader, and example work. Rebasing would re-introduce the scene.ts collision on every commit that touches Material types. The merge approach isolates the fix to one post-merge edit.

**Why not cherry-pick**: The 82-file change set on B is cohesive (Sprint 9 SVGF runtime, G2 example, scene converter move, babylon stub). Cherry-picking individual commits would risk partial states where `@vitrum/three-bindings` exports `vitrumSceneToThree` but `pt-webgl` still imports from `./sceneToThree.js`.
