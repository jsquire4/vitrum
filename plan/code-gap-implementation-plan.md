# Code Gap Implementation Plan — vitrum

> **Parallel-first execution.** See `plan/code-gap-parallel-schedule.md` for wave table.  
> **Orchestrator:** `plan/ORCHESTRATOR_AGENT.md` + `node tools/gap-scan/orchestrator-run.mjs`  
> **Per-wave prompts:** `plan/waves/Wnnn/agents/agent-NN.md` (generated)  
> **Machine-readable:** `plan/code-gap-tasks.jsonl` + `plan/code-gap-parallel-schedule.json`

> **Worktree:** `/home/jsquire4/projects/vitrum-gap-remediation` on branch `feat/gap-remediation`  
> **Validation:** 28 GPU/render tasks **deferred** — `plan/VALIDATION-DEFERRED.md`  
> **Active scheduled tasks:** 473 (after dedup + deferral)  
**Parallel waves:** 70 · **Agent cap:** 25 per wave  
**Generated:** 2026-06-17

---

## 1. Parallel execution protocol (MANDATORY)

### 1.1 Orchestrator model
- Work proceeds in **waves** (`W000`, `W001`, …). Each wave has **at most 25 tasks** — dispatch **≤25 agents**, one task per agent.
- Tasks within a wave are **file-disjoint** — safe to run concurrently up to the cap.
- **Wave gate:** after each wave, run integration (§1.5). Do not start wave N+1 until wave N is done + integrated.
- **Super-batches** (`B00`, …): optional deeper integration every ~4 waves (see parallel-schedule.md).

### 1.2 Agent pool sizing
| Setting | Value |
|---------|------:|
| **Max concurrent agents** | **25** |
| Bootstrap W000 | 8 agents (all P0) |
| Typical wave | up to 25 agents |
| Mutex waves | 1 agent (promiseLedger, materialAtlas, etc.) |

### 1.3 Agent assignment

| Model | When | Rule |
|-------|------|------|
| **Wave pool (recommended)** | Fixed pool of ≤25 agents | Orchestrator publishes current `waveId`; each agent claims one unclaimed `taskId` from that wave only. |
| **Lane-pinned** | Long-running workers | Agent owns a **lane**; claims only tasks in current wave matching its lane. Still ≤25 total active. |

### 1.4 Claiming work
1. Read `plan/code-gap-parallel-schedule.json` → `waves[current]`.
2. Pick an unclaimed `taskId` from that wave whose `lane` you own (or any if wave pool).
3. Record claim in `plan/code-gap-lane-workers.md` → `claims` table: `taskId | agent | started | status`.
4. **Hard rule:** do not edit any file not listed in your task's **Files to modify**.
5. **Hard rule:** if two tasks share a file, they are in different waves — never start early.

### 1.5 Skipped duplicates
Do not execute: `PTGL-003`, `PTWG-037`, `WH-034`, `CORE-041`. Phase-0 aliases cover these.

### 1.6 Integration commands (run after EVERY wave)
```bash
cd /home/jsquire4/projects/vitrum && npm run typecheck
# Plus package tests for lanes touched in the wave (orchestrator lists lanes in wave JSON)
```

### 1.7 Per-task loop (within your claim)
1. Read **Problem**, **Files**, **Steps**.
2. Execute steps only — no drive-by refactors.
3. Run task **Tests**.
4. Mark done in `plan/code-gap-task-progress.md`.
5. **Commits:** orchestrator commits each wave (`--commit --yes`). Workers do not commit. **Never push** unless user explicitly requests.

### 1.8 Code-first policy (validation deferred)

GPU A/B, golden PNG, reference-render capture, behavioral-gate **render** proof, and
wsl-gpu harness tasks are **excluded from the schedule**. See `plan/VALIDATION-DEFERRED.md`
(28 tasks). Priority is **landing code**; validation sprint follows merge.

### 1.9 Disposition codes

| Code | Meaning | Required outcome |
|------|---------|------------------|
| `BUG` | Incorrect runtime behavior | Fix + regression test |
| `IMP` | Missing feature / native path | Implement + test |
| `DOC` | Contract/JSDoc/ledger text wrong | Text fix only |
| `TEST` | Missing test or gate | Add test/gate only |
| `ACC` | Permanent unsupported | EngineWarning + ledger; do not implement |
| `RT` | Route to other backend | gltf planner only |
| `TOG` | Fidelity toggle | Default preserves current behavior |
| `VERIFY` | Unit-test / code-read proof | Close or file follow-up (render VERIFY deferred) |
| `SKIP` | Deferred validation or duplicate | No agent dispatch |
| `DECIDE` | Architecture/product call | Record decision; align ledger |

### 1.10 Road-to-100 overlay (phases 8–12)

See `plan/code-gap-road-to-100-crosswalk.md` for mapping from `plan/road-to-100.md`
buckets A–D, Phases 2–5, and F1–F6 to `RT100-*` task IDs.

- **Phase 8–11 validation/decision tasks:** **SKIP** (deferred) in code-first mode
- **Phase 9:** Implementation tails (`RT100-ADJ-*`, `RT100-WA-*`, `RT100-PT*`) — **active**
- **Phase 10:** `RT100-5D-DOC` only (doc sync); render proof skipped
- **Phase 12:** SOTA perf IMP (`RT100-LD-SAMPLING-01`, `RT100-WBVH-01`, F3)

Generic `MAT-WH-*` tasks cover per-field promotion; `RT100-WA-3D/3E` are the
authoritative walkaround atlas epics per road-to-100 Phase 3 scope decision.

### 1.11 Mutex hotspots (serialize — never parallelize)
| File / area | Max concurrent editors |
|-------------|------------------------|
| `packages/core/src/engine/promiseLedger.ts` | **1** globally |
| `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts` | **1** |
| `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts` | **1** |
| `packages/pt-webgpu/src/inverse/inverseSession.ts` | **1** |
| `packages/walkaround-hybrid/src/HybridEngine.ts` | **1** |

### 1.12 Bootstrap wave W000 (8 agents, mandatory first)
All 8 `P0-*` tasks run in parallel — **no other work until W000 integration passes.**

- `P0-001-PTWG-037` lane `pt-webgpu`
- `P0-002-PTGL-003` lane `pt-webgl2`
- `P0-003-WH-034` lane `walkaround-hybrid`
- `P0-004-DENO-001` lane `core`
- `P0-005-LEDGER-01` lane `core`
- `P0-006-LEDGER-02` lane `core`
- `P0-007-LEDGER-03` lane `core`
- `P0-008-TOOL-001` lane `tools`

### 1.13 Dispatch loop (W001 onward)
```
repeat until all waves done:
  wave = read plan/code-gap-parallel-schedule.json → waves[current]
  spawn min(wave.taskCount, 25) agents
  each agent: claim taskId → execute → mark done
  integration: npm run typecheck + lane tests
  current += 1
```
Typical wave size: **25 tasks** (file-disjoint batch). Total waves: **70**.

---

## 2. Phase overview (logical grouping — waves may cross phases)

| Phase | Name | Tasks | Notes |
|-------|------|------:|-------|
| 0 | P0 correctness | 8 | **Wave W000:** all 8 run in parallel |
| 1 | Contract + engine | 80 | Interleaved with other lanes |
| 2 | pt-webgl2 + FP | 45 | FP-01→06 chain is sequential |
| 3 | pt-webgpu | 79 | Independent of walkaround lane |
| 4 | walkaround | 177 | Largest lane; internal file mutex |
| 5 | glTF | 60 | Fully parallel with other lanes |
| 6 | Inverse | 16 | inverseSession/adjoint mutex |
| 7 | Tools + shared | 0 | Behavioral gate, radiometric-ab |
| 8 | RT100 V28 validation | 14 | **Deferred** — not scheduled |
| 9 | RT100 impl/promotion | 10 | Adjoint, atlas tail, PT mutations |
| 10 | RT100 proof | 4 | Golden PNG, doc sync, full-tier gate |
| 11 | RT100 decisions | 2 | NRC tier, neural weights |
| 12 | RT100 SOTA perf | 6 | LD sampling, WBVH, denoiser:auto |

**Scheduling truth:** `70 waves`, not phase order. Phases are taxonomy only.

---

## 3. Task register (full)

## P0-001-PTWG-037

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | BUG |
| Lane | pt-webgpu |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
pt-webgpu lite tier: material-only updatePrimitive() calls host.setScene(nextScene) and returns without writing materialsBuffer — GPU stays stale.

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sceneMutationRouter.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/sceneMutationRouter.ts.
2. Find lines 315-318: `if (host.isLiteTier?.() === true && canFastPathMaterialPatch(fastPathPatch)) { host.setScene(nextScene); return; }`.
3. Delete that entire if-block (4 lines). Do NOT replace with another early return.
4. Confirm execution falls through to fastPaths array; material handler at ~530-592 calls packFoldedMaterialEntry + device.queue.writeBuffer to sceneBuffers.materialsBuffer.
5. Grep uploadSceneBuffers.ts / gpuResources for lite-tier materialsBuffer creation — if null on lite setScene, ensure materials buffer is allocated on lite path before this fix can work.
6. Run updatePrimitiveIncremental tests; add lite-tier material patch test if missing.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/__tests__/updatePrimitiveIncremental.test.ts
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] Lite material-only patch no longer calls full setScene.
- [ ] materialsBuffer receives queue.writeBuffer on lite.
- [ ] updatePrimitiveIncremental.test.ts green including lite case.
- [ ] Root typecheck green.

---
## P0-002-PTGL-003

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | BUG |
| Lane | pt-webgl2 |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
pt-webgl2 tryFastPathMaterialMutation passes stale geoPack to packMeshAreaLights after materials repacked into nextGeoPack.

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/mutateSceneTextures.ts.
2. At line 216 change `packMeshAreaLights(nextScene, geoPack)` to `packMeshAreaLights(nextScene, nextGeoPack)`.
3. Verify return still sets geoPack: nextGeoPack at line ~232.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run src/scene/meshAreaLights.test.ts
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] Line 216 uses nextGeoPack.
- [ ] meshAreaLights tests green.
- [ ] typecheck green.

---
## P0-003-WH-034

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | BUG |
| Lane | walkaround-hybrid |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
walkaround material-only patch returns applySubsystems:false; DDGI probe cache only invalidated on emissive/transmission-threshold — not beer/attenuation/thickness.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`
2. `packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open HybridEnginePrimitiveUpdates.ts function materialPatch (~line 1400+).
2. Add helper ddgiAffectingMaterialChanged(prev, next, patch) returning true when patch touches: attenuationColor, attenuationDistance, thickness, transmission, transmissionMap, thicknessMap, baseColor (beer tint), roughness, metallic, or any map field in MATERIAL_ATLAS_FIELDS.
3. After computing prevMaterial/nextMaterial, if ddgiAffectingMaterialChanged OR emitterAffectingMaterialChanged: call ctx.ddgi.invalidateProbeCache().
4. Keep applySubsystems: false — geometry unchanged.
5. Update comment at lines 1477-1480 documenting DDGI invalidation on beer/attenuation edits.
6. Add mutationMatrix.test.ts: patch attenuationDistance only → expect invalidateProbeCache mock called.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run src/__tests__/mutationMatrix.test.ts
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] attenuationDistance-only patch invalidates probe cache.
- [ ] applySubsystems remains false.
- [ ] mutationMatrix test pins behavior.

---
## P0-004-DENO-001

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | IMP |
| Lane | core |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
walkaround NoneDenoiser exists but VALID_DENOISERS omits 'none' — construction throws.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineOptions.ts`
2. `packages/walkaround-hybrid/src/HybridEngineConfig.ts`
3. `packages/walkaround-hybrid/__tests__/hybridEngineTuning.test.ts`
4. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Add 'none' as first entry in VALID_DENOISERS in HybridEngineOptions.ts line 22.
2. Update HybridEngineConfig.ts validation error message to list none.
3. Confirm pipeline/denoisers/none.ts registered in denoiser index.
4. Update promiseLedger WALKAROUND_DENOISERS row for none if missing.
5. Fix hybridEngineTuning.test.ts: denoiser:'none' must construct without throw.
6. Add test: denoiser:'none' skips DenoiserAdapterPass denoise stage.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run __tests__/hybridEngineTuning.test.ts __tests__/denoiserRegistry.test.ts
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] 'none' in VALID_DENOISERS.
- [ ] Construction succeeds.
- [ ] Ledger aligned.

---
## P0-005-LEDGER-01

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | DOC |
| Lane | core |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
promiseLedger.ts:907-911 stale comment claims point/spot DDGI-only approximate.

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open promiseLedger.ts lines 907-911.
2. Delete stale comment block about point/spot DDGI-only approximate.
3. WALKAROUND_EMITTERS already grades point/spot native — do not change ledger rows.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] Stale comment removed.
- [ ] ledgerVsCapabilities green.

---
## P0-006-LEDGER-02

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | DOC |
| Lane | core |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
core engine/index.ts JSDoc says walkaround captureFrame('output') rejects.

### Files to modify (exact paths)
1. `packages/core/src/engine/index.ts`
2. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/index.ts captureFrame JSDoc (~402-411).
2. Replace walkaround 'output' rejects text with: supports colorSpace:'output' via captureOutputFrame (tonemapped present).
3. Fix matching promiseLedger method comment at ~977 if still wrong.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] JSDoc matches HybridEngine.captureFrame behavior.

---
## P0-007-LEDGER-03

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | DOC |
| Lane | core |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
promiseLedger updateEnvironment note says HDRI intensity-only; Wave 4/5 env NEE is live.

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`
2. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Read HybridEngine.updateEnvironment implementation.
2. Rewrite promiseLedger.ts lines ~966-970 to describe actual behavior: scalar sky + HDRI CDF rebuild when resolver provides pixels.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] Comment matches updateEnvironment code.

---
## P0-008-TOOL-001

| Field | Value |
|-------|-------|
| Phase | 0 |
| Priority | P0 |
| Disposition | TEST |
| Lane | tools |
| Wave | W000 (#0) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
behavioral gate has zero pt-webgl2 configs.

### Files to modify (exact paths)
1. `tools/behavioral-gate/gate.mjs`

### Steps (execute in order — do not skip, do not improvise)
1. Open tools/behavioral-gate/gate.mjs.
2. Add import for pt-webgl2 engine factory (mirror existing pt-webgpu import pattern).
3. Add PTGL_CONFIGS array with at least: ptgl/default, ptgl/spectral, ptgl/mesh-area.
4. Add run branch: label prefix ptgl/ uses WebGL2 factory.
5. Add EXPECTATION_TABLE entries for each ptgl/* label.
6. Document in gate header: pt-webgl2 needs WebGL2 context; skip via env if unavailable.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum && node tools/behavioral-gate/gate.mjs --filter ptgl 2>&1 | head -50
```

### Done when (ALL boxes required before next task)
- [ ] ≥3 ptgl/* configs defined.
- [ ] Gate imports without error.

---
## CORE-001

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | DOC |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-001: JSDoc HDRI PARTIAL vs ledger native

### Files to modify (exact paths)
1. `packages/core/src/scene/environment.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/environment.ts:45-51.
2. Gap: JSDoc HDRI PARTIAL vs ledger native
3. Fix: Update HdriEnvironment JSDoc to match ledger
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-001 fix applied.
- [ ] Tests green.

---
## CORE-002

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | DOC |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-002: envMapIntensity JSDoc drift

### Files to modify (exact paths)
1. `packages/core/src/scene/material.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/material.ts:273-275.
2. Gap: envMapIntensity JSDoc drift
3. Fix: Align JSDoc with WALKAROUND_MATERIALS native row
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-002 fix applied.
- [ ] Tests green.

---
## CORE-003

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-003: castShadow comment approximate vs native

### Files to modify (exact paths)
1. `packages/core/src/scene/primitives.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/primitives.ts:65-67.
2. Gap: castShadow comment approximate vs native
3. Fix: Fix comment after code-read of castShadow handling
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-003 fix applied.
- [ ] Tests green.

---
## CORE-004

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | ACC |
| Lane | core |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-004: receiveShadow @reserved unsupported

### Files to modify (exact paths)
1. `packages/core/src/scene/primitives.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/primitives.ts:39-44.
2. Gap: receiveShadow @reserved unsupported
3. Fix: IMP receiver occlusion OR permanent ACC + partitionSceneBySupport warning
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-004 fix applied.
- [ ] Tests green.

---
## CORE-005

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | ACC |
| Lane | core |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-005: displacement* @reserved

### Files to modify (exact paths)
1. `packages/core/src/scene/material.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/material.ts:225-233.
2. Gap: displacement* @reserved
3. Fix: IMP on PT backend OR permanent ACC in ledger
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-005 fix applied.
- [ ] Tests green.

---
## CORE-006

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-006: texture Phase 2 reserved

### Files to modify (exact paths)
1. `packages/core/src/inverse.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/inverse.ts:47-49.
2. Gap: texture Phase 2 reserved
3. Fix: IMP texture optimization in pt-webgpu inverse
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-006 fix applied.
- [ ] Tests green.

---
## CORE-007

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-007: ssim/lpips throw

### Files to modify (exact paths)
1. `packages/core/src/inverse.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/inverse.ts:83-85.
2. Gap: ssim/lpips throw
3. Fix: IMP perceptual losses OR remove from InverseLoss union
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-007 fix applied.
- [ ] Tests green.

---
## CORE-008

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-008: materials Partial

### Files to modify (exact paths)
1. `packages/core/src/engine/capabilities.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/capabilities.ts:78-80.
2. Gap: materials Partial
3. Fix: Full matrix OR export isMaterialFieldAudited()
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-008 fix applied.
- [ ] Tests green.

---
## CORE-009

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-009: swapchain-optional unused

### Files to modify (exact paths)
1. `packages/core/src/engine/capabilities.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/capabilities.ts:106-107.
2. Gap: swapchain-optional unused
3. Fix: Remove enum OR assign backend
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-009 fix applied.
- [ ] Tests green.

---
## CORE-010

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-010: supportsAuxBuffers false vs partial aux

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:893-898.
2. Gap: supportsAuxBuffers false vs partial aux
3. Fix: Wire variance OR narrow contract
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-010 fix applied.
- [ ] Tests green.

---
## CORE-011

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-011: analytic fallback vs native shapes

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:921,947.
2. Gap: analytic fallback vs native shapes
3. Fix: Split accepted vs native-traced in capabilities
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-011 fix applied.
- [ ] Tests green.

---
## CORE-012

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-012: spectral/volume unsupported WH

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:475-482.
2. Gap: spectral/volume unsupported WH
3. Fix: partitionSceneBySupport warnings with codes
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-012 fix applied.
- [ ] Tests green.

---
## CORE-013

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-013: alphaMode blend not GI participant

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:354-358.
2. Gap: alphaMode blend not GI participant
3. Fix: Document permanent OR IMP GI transport
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-013 fix applied.
- [ ] Tests green.

---
## CORE-014

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-014: quantized baseColor/roughness/metallic

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:340-344.
2. Gap: quantized baseColor/roughness/metallic
3. Fix: TOG EngineFidelityProfile full lanes
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-014 fix applied.
- [ ] Tests green.

---
## CORE-015

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-015: mutations.environment approximate

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:959.
2. Gap: mutations.environment approximate
3. Fix: Document OR promote after env fast-path
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-015 fix applied.
- [ ] Tests green.

---
## CORE-016

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-016: shadingModel unlit

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:505-507.
2. Gap: shadingModel unlit
3. Fix: IMP unlit-as-emissive OR mark unsupported
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-016 fix applied.
- [ ] Tests green.

---
## CORE-017

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-017: scatteringCoefficientRGB approx PTGL

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:567-569.
2. Gap: scatteringCoefficientRGB approx PTGL
3. Fix: Port pt-webgpu σ_s path
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-017 fix applied.
- [ ] Tests green.

---
## CORE-018

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-018: pt-webgl2 emitterCastShadow approx

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:749-752.
2. Gap: pt-webgl2 emitterCastShadow approx
3. Fix: Wire castShadow emissive-hit paths
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-018 fix applied.
- [ ] Tests green.

---
## CORE-019

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-019: pt-webgpu emitterCastShadow specialty

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:767-769.
2. Gap: pt-webgpu emitterCastShadow specialty
3. Fix: Complete BDPT/MNEE/SPPM shadow flags
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-019 fix applied.
- [ ] Tests green.

---
## CORE-020

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-020: pt-webgpu setSize false

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:845-846.
2. Gap: pt-webgpu setSize false
3. Fix: Add setSize API OR document viewport-only
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-020 fix applied.
- [ ] Tests green.

---
## CORE-021

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-021: PT updateLighting false

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:830,846.
2. Gap: PT updateLighting false
3. Fix: IMP OR map to updateEnvironment
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-021 fix applied.
- [ ] Tests green.

---
## CORE-022

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-022: PT denoisers limited

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:788-805.
2. Gap: PT denoisers limited
3. Fix: Document construction warn behavior
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-022 fix applied.
- [ ] Tests green.

---
## CORE-023

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-023: getRestirPtResultBuffer when off

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:1129-1134.
2. Gap: getRestirPtResultBuffer when off
3. Fix: Gate methodPromises on experimental flag
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-023 fix applied.
- [ ] Tests green.

---
## CORE-024

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-024: updateEnvironment JSDoc

### Files to modify (exact paths)
1. `packages/core/src/engine/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/index.ts:196-204.
2. Gap: updateEnvironment JSDoc
3. Fix: Rewrite for all backends
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-024 fix applied.
- [ ] Tests green.

---
## CORE-025

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-025: partition plain strings

### Files to modify (exact paths)
1. `packages/core/src/scene/partitionSceneBySupport.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/partitionSceneBySupport.ts:46-48.
2. Gap: partition plain strings
3. Fix: Return EngineWarning[]
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-025 fix applied.
- [ ] Tests green.

---
## CORE-026

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-026: analytic null AABB

### Files to modify (exact paths)
1. `packages/core/src/scene/tlasAudit.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/tlasAudit.ts:134-154.
2. Gap: analytic null AABB
3. Fix: Tessellate bounds in primitiveBounds
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-026 fix applied.
- [ ] Tests green.

---
## CORE-027

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-027: TLAS audit ignores analytic

### Files to modify (exact paths)
1. `packages/core/src/scene/tlasAudit.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/tlasAudit.ts:33-38.
2. Gap: TLAS audit ignores analytic
3. Fix: Include tessellated triangle estimates
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-027 fix applied.
- [ ] Tests green.

---
## CORE-028

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-028: low-poly tessellation

### Files to modify (exact paths)
1. `packages/core/src/scene/analyticToMesh.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/analyticToMesh.ts:33-58.
2. Gap: low-poly tessellation
3. Fix: Expose segment count + warn on pt-webgpu
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-028 fix applied.
- [ ] Tests green.

---
## CORE-029

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | core |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-029: anisotropyMap comment stale

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:576-578.
2. Gap: anisotropyMap comment stale
3. Fix: Verify atlas wired; fix comment
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-029 fix applied.
- [ ] Tests green.

---
## CORE-030

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W017 (#17) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-030: thinFilmStack cap 35 vs 8

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:574-575.
2. Gap: thinFilmStack cap 35 vs 8
3. Fix: Export per-backend cap in capabilities
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-030 fix applied.
- [ ] Tests green.

---
## CORE-031

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W018 (#18) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-031: extensions native WH / unsupported PT

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts:487-489.
2. Gap: extensions native WH / unsupported PT
3. Fix: Document PT ignores extensions
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-031 fix applied.
- [ ] Tests green.

---
## CORE-032

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-032: angularDiameter pt-webgpu only

### Files to modify (exact paths)
1. `packages/core/src/scene/emitters.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/emitters.ts:42-49.
2. Gap: angularDiameter pt-webgpu only
3. Fix: IMP soft sun WH/pt-webgl2 OR @reserved
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-032 fix applied.
- [ ] Tests green.

---
## CORE-033

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-033: point/spot distance/decay uneven

### Files to modify (exact paths)
1. `packages/core/src/scene/emitters.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/emitters.ts:69-80.
2. Gap: point/spot distance/decay uneven
3. Fix: Per-emitter-field supportDetails rows
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-033 fix applied.
- [ ] Tests green.

---
## CORE-034

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-034: AnimationClip no consumer

### Files to modify (exact paths)
1. `packages/core/src/scene/animation.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/animation.ts:5-8.
2. Gap: AnimationClip no consumer
3. Fix: Document host duty OR Engine.playAnimation
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-034 fix applied.
- [ ] Tests green.

---
## CORE-035

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-035: CPU LBS only

### Files to modify (exact paths)
1. `packages/core/src/skinSolver.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/skinSolver.ts:39-40.
2. Gap: CPU LBS only
3. Fix: Document GPU skinning is backend path
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-035 fix applied.
- [ ] Tests green.

---
## CORE-036

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-036: photon-map energy biased

### Files to modify (exact paths)
1. `packages/core/src/engine/factory.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/factory.ts:110-120.
2. Gap: photon-map energy biased
3. Fix: Surface fidelity tier in capabilities
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-036 fix applied.
- [ ] Tests green.

---
## CORE-037

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-037: exportGIState null vague

### Files to modify (exact paths)
1. `packages/core/src/engine/giState.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/giState.ts:57-60.
2. Gap: exportGIState null vague
3. Fix: Discriminated return OR warn on null
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-037 fix applied.
- [ ] Tests green.

---
## CORE-038

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-038: debugSurface boolean

### Files to modify (exact paths)
1. `packages/core/src/engine/debug.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/debug.ts:12-77.
2. Gap: debugSurface boolean
3. Fix: List implemented debug methods
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-038 fix applied.
- [ ] Tests green.

---
## CORE-039

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-039: Mat4 brand not deep freeze

### Files to modify (exact paths)
1. `packages/core/src/scene/math.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/math.ts:14-21.
2. Gap: Mat4 brand not deep freeze
3. Fix: Document OR defensive copy getScene
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-039 fix applied.
- [ ] Tests green.

---
## CORE-040

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-040: getScene retained reference

### Files to modify (exact paths)
1. `packages/core/src/engine/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/index.ts:121-131.
2. Gap: getScene retained reference
3. Fix: Optional getScene({copy:true})
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-040 fix applied.
- [ ] Tests green.

---
## CORE-042

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-042: swapChainView walkaround-only

### Files to modify (exact paths)
1. `packages/core/src/frame.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/frame.ts:131-137.
2. Gap: swapChainView walkaround-only
3. Fix: Clarify FrameInput JSDoc
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-042 fix applied.
- [ ] Tests green.

---
## CORE-043

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-043: path-replay downgrade list

### Files to modify (exact paths)
1. `packages/core/src/inverse.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/inverse.ts:210-214.
2. Gap: path-replay downgrade list
3. Fix: Shared eligible-field table in core
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-043 fix applied.
- [ ] Tests green.

---
## CORE-044

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-044: denoiserState walkaround-shaped

### Files to modify (exact paths)
1. `packages/core/src/engine/telemetry.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/telemetry.ts:66-67.
2. Gap: denoiserState walkaround-shaped
3. Fix: Generalize OR extension bag
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-044 fix applied.
- [ ] Tests green.

---
## CORE-045

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-045: material deep-merge vs emitter shallow

### Files to modify (exact paths)
1. `packages/core/src/scene/patchScene.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/scene/patchScene.ts:103-112.
2. Gap: material deep-merge vs emitter shallow
3. Fix: Document OR deep-merge emitters
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-045 fix applied.
- [ ] Tests green.

---
## CORE-046

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | core |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-046: detectGpu memoized

### Files to modify (exact paths)
1. `packages/core/src/gpuDetection.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/gpuDetection.ts:245-264.
2. Gap: detectGpu memoized
3. Fix: Auto-invalidate on device.lost
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-046 fix applied.
- [ ] Tests green.

---
## CORE-047

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
CORE-047: AdapterProfile engine-only

### Files to modify (exact paths)
1. `packages/core/src/adapterProfile.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/adapterProfile.ts:1-10.
2. Gap: AdapterProfile engine-only
3. Fix: Document requires @vitrum/engine
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] CORE-047 fix applied.
- [ ] Tests green.

---
## ENG-001

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-001: onAdapterProfile walkaround-only

### Files to modify (exact paths)
1. `packages/engine/src/createEngineInternals.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/createEngineInternals.ts:118-122.
2. Gap: onAdapterProfile walkaround-only
3. Fix: Call for PT backends OR rename option
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-001 fix applied.
- [ ] Tests green.

---
## ENG-002

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-002: walkaround fallback wrong advanced bag

### Files to modify (exact paths)
1. `packages/engine/src/createEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/createEngine.ts:130-155.
2. Gap: walkaround fallback wrong advanced bag
3. Fix: Require advancedByBackend on fallback
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-002 fix applied.
- [ ] Tests green.

---
## ENG-003

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-003: pt-webgpu configures offscreen canvas

### Files to modify (exact paths)
1. `packages/engine/src/backends/ptWebgpu.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/backends/ptWebgpu.ts:74-87.
2. Gap: pt-webgpu configures offscreen canvas
3. Fix: Skip configureWebGpuCanvas when offscreen
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-003 fix applied.
- [ ] Tests green.

---
## ENG-004

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | DOC |
| Lane | engine |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-004: Class-A only throws

### Files to modify (exact paths)
1. `packages/engine/src/createProgressiveEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/createProgressiveEngine.ts:226-236.
2. Gap: Class-A only throws
3. Fix: Document; optional realtime-only fallback
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-004 fix applied.
- [ ] Tests green.

---
## ENG-005

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-005: no advancedByBackend prop

### Files to modify (exact paths)
1. `packages/engine/src/react/VitrumCanvas.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/react/VitrumCanvas.tsx:89-90.
2. Gap: no advancedByBackend prop
3. Fix: Add prop plumbed to attachVitrum
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-005 fix applied.
- [ ] Tests green.

---
## ENG-006

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-006: engine recreates on gltf change

### Files to modify (exact paths)
1. `packages/engine/src/react/VitrumCanvas.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/react/VitrumCanvas.tsx:141-165.
2. Gap: engine recreates on gltf change
3. Fix: Ref-stabilize gltfOptions
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-006 fix applied.
- [ ] Tests green.

---
## ENG-007

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-007: debug live after dispose

### Files to modify (exact paths)
1. `packages/engine/src/idempotentDispose.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/idempotentDispose.ts:242-247.
2. Gap: debug live after dispose
3. Fix: Null debug on proxy post-dispose
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-007 fix applied.
- [ ] Tests green.

---
## ENG-008

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | VERIFY |
| Lane | engine |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-008: setSize noop disposed

### Files to modify (exact paths)
1. `packages/engine/src/idempotentDispose.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/idempotentDispose.ts:108-109.
2. Gap: setSize noop disposed
3. Fix: Verify pt-webgpu never proxied setSize
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-008 fix applied.
- [ ] Tests green.

---
## ENG-009

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-009: no WebGPU silent pt-webgl2

### Files to modify (exact paths)
1. `packages/engine/src/createEngineScale.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/createEngineScale.ts:47-49.
2. Gap: no WebGPU silent pt-webgl2
3. Fix: Throw/warn realtime-unavailable
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-009 fix applied.
- [ ] Tests green.

---
## ENG-010

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-010: 500k threshold undocumented

### Files to modify (exact paths)
1. `packages/engine/src/createEngineScale.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/createEngineScale.ts:8-9.
2. Gap: 500k threshold undocumented
3. Fix: Export constant + document
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-010 fix applied.
- [ ] Tests green.

---
## ENG-011

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-011: AABB rest-pose skinned

### Files to modify (exact paths)
1. `packages/engine/src/sceneAABB.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/sceneAABB.ts:101-110.
2. Gap: AABB rest-pose skinned
3. Fix: Optional posed AABB via solveSkin
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-011 fix applied.
- [ ] Tests green.

---
## ENG-012

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-012: lite merged BVH needsTlas

### Files to modify (exact paths)
1. `packages/engine/src/backends/walkaround.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/backends/walkaround.ts:154-157.
2. Gap: lite merged BVH needsTlas
3. Fix: EngineWarning when useLite&&needsTlas
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-012 fix applied.
- [ ] Tests green.

---
## ENG-013

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-013: shared device forces full

### Files to modify (exact paths)
1. `packages/engine/src/backends/walkaround.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/backends/walkaround.ts:159.
2. Gap: shared device forces full
3. Fix: Allow lite override with warning
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-013 fix applied.
- [ ] Tests green.

---
## ENG-014

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-014: progressive hardcodes pt-webgpu

### Files to modify (exact paths)
1. `packages/engine/src/gltf.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/gltf.ts:143-146.
2. Gap: progressive hardcodes pt-webgpu
3. Fix: Align gltf progressive profile
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-014 fix applied.
- [ ] Tests green.

---
## ENG-015

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-015: target none without limits

### Files to modify (exact paths)
1. `packages/engine/src/negotiateWebGPUDevice.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/negotiateWebGPUDevice.ts:54-56.
2. Gap: target none without limits
3. Fix: Default target to backend
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-015 fix applied.
- [ ] Tests green.

---
## ENG-016

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-016: recreate preserves GI not inverse

### Files to modify (exact paths)
1. `packages/engine/src/lifecycle/vanilla.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/lifecycle/vanilla.ts:576-585.
2. Gap: recreate preserves GI not inverse
3. Fix: Document limitations
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-016 fix applied.
- [ ] Tests green.

---
## ENG-017

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-017: no alpha cross-fade

### Files to modify (exact paths)
1. `packages/engine/src/progressiveHandoff.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/progressiveHandoff.ts:18-22.
2. Gap: no alpha cross-fade
3. Fix: IMP blend OR remove from claims
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-017 fix applied.
- [ ] Tests green.

---
## ENG-018

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-018: configure failure swallowed

### Files to modify (exact paths)
1. `packages/engine/src/configureWebGpuCanvas.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/configureWebGpuCanvas.ts:40-43.
2. Gap: configure failure swallowed
3. Fix: EngineWarning + skip render
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-018 fix applied.
- [ ] Tests green.

---
## ENG-019

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-019: serializeGIState re-export

### Files to modify (exact paths)
1. `packages/engine/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/index.ts:111-116.
2. Gap: serializeGIState re-export
3. Fix: Move to optional subpath
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-019 fix applied.
- [ ] Tests green.

---
## ENG-020

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | engine |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-020: publishToWindow false

### Files to modify (exact paths)
1. `packages/engine/src/gpuDetection.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/gpuDetection.ts.
2. Gap: publishToWindow false
3. Fix: Document embedder opt-out
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-020 fix applied.
- [ ] Tests green.

---
## ENG-021

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-021: ResizeObserver not in createEngine

### Files to modify (exact paths)
1. `packages/engine/src/createEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/createEngine.ts:15-18.
2. Gap: ResizeObserver not in createEngine
3. Fix: JSDoc backend resize table
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-021 fix applied.
- [ ] Tests green.

---
## ENG-022

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-022: add/remove gated

### Files to modify (exact paths)
1. `packages/engine/src/idempotentDispose.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/idempotentDispose.ts:100-106.
2. Gap: add/remove gated
3. Fix: Verify all backends capability true
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-022 fix applied.
- [ ] Tests green.

---
## ENG-023

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-023: seedFromRealtime default mismatch

### Files to modify (exact paths)
1. `packages/engine/src/progressiveHandoff.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/progressiveHandoff.ts:101-108.
2. Gap: seedFromRealtime default mismatch
3. Fix: Align defaults OR warn
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-023 fix applied.
- [ ] Tests green.

---
## ENG-024

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-024: reject-unsupported passes approximate

### Files to modify (exact paths)
1. `packages/engine/src/gltf.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/gltf.ts:217-220.
2. Gap: reject-unsupported passes approximate
3. Fix: reject-approximate mode OR document
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-024 fix applied.
- [ ] Tests green.

---
## ENG-025

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-025: ResizeObserver walkaround-only

### Files to modify (exact paths)
1. `packages/engine/src/lifecycle/vanilla.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/lifecycle/vanilla.ts:458-461.
2. Gap: ResizeObserver walkaround-only
3. Fix: Document pt-webgpu viewport requirement
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-025 fix applied.
- [ ] Tests green.

---
## ENG-026

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-026: updateEnvironment comment stale

### Files to modify (exact paths)
1. `packages/engine/src/backends/walkaround.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/backends/walkaround.ts.
2. Gap: updateEnvironment comment stale
3. Fix: Reconcile with implementation
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-026 fix applied.
- [ ] Tests green.

---
## ENG-027

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-027: lifetime guard duplicated

### Files to modify (exact paths)
1. `packages/engine/src/createEngineInternals.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/createEngineInternals.ts:436-437.
2. Gap: lifetime guard duplicated
3. Fix: Accept OR typed helper
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-027 fix applied.
- [ ] Tests green.

---
## ENG-028

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-028: LightingOptions walkaround on main export

### Files to modify (exact paths)
1. `packages/engine/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/index.ts:98-103.
2. Gap: LightingOptions walkaround on main export
3. Fix: Subpath @vitrum/engine/walkaround
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-028 fix applied.
- [ ] Tests green.

---
## ENG-029

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-029: requires scene or gltf

### Files to modify (exact paths)
1. `packages/engine/src/react/VitrumCanvas.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/react/VitrumCanvas.tsx:175-177.
2. Gap: requires scene or gltf
3. Fix: Document deferred scene via engine prop
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-029 fix applied.
- [ ] Tests green.

---
## ENG-030

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | IMP |
| Lane | engine |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
ENG-030: detectWebGL2Sync headless

### Files to modify (exact paths)
1. `packages/engine/src/adapterProfile.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/engine/src/adapterProfile.ts:132-136.
2. Gap: detectWebGL2Sync headless
3. Fix: Return false without document
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] ENG-030 fix applied.
- [ ] Tests green.

---
## PTGL-001

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-001: map fields block fast path

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/mutateSceneTextures.ts:53.
2. Gap: map fields block fast path
3. Fix: Atlas delta upload for material maps
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-001 fix applied.
- [ ] Tests green.

---
## PTGL-002

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-002: transform/positions/topology null fast path

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/mutateSceneTextures.ts:178.
2. Gap: transform/positions/topology null fast path
3. Fix: Incremental BVH patches from pt-webgpu
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-002 fix applied.
- [ ] Tests green.

---
## PTGL-004

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-004: displacement blocks unimplemented

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/mutateSceneTextures.ts:35.
2. Gap: displacement blocks unimplemented
3. Fix: Remove blocker OR implement
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-004 fix applied.
- [ ] Tests green.

---
## PTGL-005

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-005: addPrimitive full setScene

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/index.ts:476.
2. Gap: addPrimitive full setScene
3. Fix: Incremental splice
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-005 fix applied.
- [ ] Tests green.

---
## PTGL-006

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-006: removePrimitive full rebuild

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/index.ts:491.
2. Gap: removePrimitive full rebuild
3. Fix: Incremental removal
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-006 fix applied.
- [ ] Tests green.

---
## PTGL-007

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-007: incrementalPatchSupport contradiction

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/capabilities.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/capabilities.ts:94.
2. Gap: incrementalPatchSupport contradiction
3. Fix: Align capabilities with mutations
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-007 fix applied.
- [ ] Tests green.

---
## PTGL-008

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-008: displacement warns only

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/index.ts:87.
2. Gap: displacement warns only
3. Fix: Tessellate OR hard-drop
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-008 fix applied.
- [ ] Tests green.

---
## PTGL-009

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-009: _NO_TEXTURE unused

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/materialsTexture.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/materialsTexture.ts:168.
2. Gap: _NO_TEXTURE unused
3. Fix: Use consistently
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-009 fix applied.
- [ ] Tests green.

---
## PTGL-010

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-010: no displacement layer

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/materialsTexture.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/materialsTexture.ts:279.
2. Gap: no displacement layer
3. Fix: Add if in scope
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-010 fix applied.
- [ ] Tests green.

---
## PTGL-011

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-011: doubleSided not in side

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/materialsTexture.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/materialsTexture.ts:435.
2. Gap: doubleSided not in side
3. Fix: Read extensions.doubleSided
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-011 fix applied.
- [ ] Tests green.

---
## PTGL-012

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-012: matte dead field

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/materialsTexture.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/materialsTexture.ts:445.
2. Gap: matte dead field
3. Fix: Remove GLSL branch
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-012 fix applied.
- [ ] Tests green.

---
## PTGL-013

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-013: ImageBitmap unreadable

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/texturesArray.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/texturesArray.ts:7.
2. Gap: ImageBitmap unreadable
3. Fix: CPU decode before ingest
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-013 fix applied.
- [ ] Tests green.

---
## PTGL-014

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-014: SAMPLED_MAP_KEYS incomplete

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/texturesArray.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/texturesArray.ts:23.
2. Gap: SAMPLED_MAP_KEYS incomplete
3. Fix: Extend to all consumed maps
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-014 fix applied.
- [ ] Tests green.

---
## PTGL-015

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-015: nearest ignores sampler

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/texturesArray.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/texturesArray.ts:250.
2. Gap: nearest ignores sampler
3. Fix: Respect TextureRef filters
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-015 fix applied.
- [ ] Tests green.

---
## PTGL-016

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-016: unreadable maps silent drop

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/texturesArray.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/texturesArray.ts:308.
2. Gap: unreadable maps silent drop
3. Fix: Strict mode throw
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-016 fix applied.
- [ ] Tests green.

---
## PTGL-017

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-017: emissiveMap needs CPU texels

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/meshAreaLights.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/meshAreaLights.ts:200.
2. Gap: emissiveMap needs CPU texels
3. Fix: Require decode in pipeline
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-017 fix applied.
- [ ] Tests green.

---
## PTGL-018

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-018: barycentric subdivision fallback

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/meshAreaLights.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/meshAreaLights.ts:384.
2. Gap: barycentric subdivision fallback
3. Fix: Expand texel subtriangle coverage
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-018 fix applied.
- [ ] Tests green.

---
## PTGL-019

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-019: synthetic emissiveMap material

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/meshAreaLights.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/meshAreaLights.ts:230.
2. Gap: synthetic emissiveMap material
3. Fix: Sample atlas in packer
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-019 fix applied.
- [ ] Tests green.

---
## PTGL-020

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-020: CDF missing sin θ

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/equirectHdrInfo.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/equirectHdrInfo.ts:18.
2. Gap: CDF missing sin θ
3. Fix: Add solid-angle Jacobian
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-020 fix applied.
- [ ] Tests green.

---
## PTGL-021

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-021: skin warn not EngineWarning

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/solveSkinPrimitives.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/solveSkinPrimitives.ts:54.
2. Gap: skin warn not EngineWarning
3. Fix: Emit structured warning
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-021 fix applied.
- [ ] Tests green.

---
## PTGL-022

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-022: no bone-only fast path

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/solveSkinPrimitives.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/solveSkinPrimitives.ts:20.
2. Gap: no bone-only fast path
3. Fix: positionsRefit after solveSkin
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-022 fix applied.
- [ ] Tests green.

---
## PTGL-023

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-023: fold keying prim.id only

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/foldEmissiveEmitters.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/scene/foldEmissiveEmitters.ts:42.
2. Gap: fold keying prim.id only
3. Fix: meshId semantics across prims
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-023 fix applied.
- [ ] Tests green.

---
## PTGL-024

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-024: realtime denoisers degrade

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/index.ts:976.
2. Gap: realtime denoisers degrade
3. Fix: Document in README
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-024 fix applied.
- [ ] Tests green.

---
## PTGL-025

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-025: OIDN not turnkey

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/index.ts:257.
2. Gap: OIDN not turnkey
3. Fix: Capability gate + clear error
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-025 fix applied.
- [ ] Tests green.

---
## PTGL-026

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-026: refraction PDF TODO

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js:48.
2. Gap: refraction PDF TODO
3. Fix: Implement PDF attenuation
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-026 fix applied.
- [ ] Tests green.

---
## PTGL-027

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-027: env MIS transmissive gap

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/glsl/render/direct_light_contribution_function.glsl.js`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/glsl/render/direct_light_contribution_function.glsl.js:125.
2. Gap: env MIS transmissive gap
3. Fix: Transmissive-aware env sample
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-027 fix applied.
- [ ] Tests green.

---
## PTGL-028

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
PTGL-028: legacy GGX not VNDF

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/glsl/shader/bsdf/ggx_functions.glsl.js`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgl2/src/glsl/shader/bsdf/ggx_functions.glsl.js:15.
2. Gap: legacy GGX not VNDF
3. Fix: Port Heitz VNDF from pt-webgpu
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTGL-028 fix applied.
- [ ] Tests green.

---
## PTWG-001

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: no setSize()

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Add setSize(width,height) mirroring viewport resize without scene rebuild; reset accum with EngineWarning
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-001 complete.
- [ ] Tests green.

---
## PTWG-002

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: no updateLighting()

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Add updateLighting(LightingOptions) bulk API
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-002 complete.
- [ ] Tests green.

---
## PTWG-003

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: resize resets accum silently

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Emit EngineWarning on resize accum reset
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-003 complete.
- [ ] Tests green.

---
## PTWG-004

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | TEST |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite vs ledger incrementalPatchSupport drift

### Files to modify (exact paths)
1. `packages/core/src/__tests__/ledgerVsCapabilities.test.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/__tests__/ledgerVsCapabilities.test.ts.
2. Implement: Tier-aware ledger test for lite mutations
3. Add/update test in packages/core/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-004 complete.
- [ ] Tests green.

---
## PTWG-005

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | TEST |
| Lane | pt-webgpu |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite capabilities untested

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/__tests__/capabilities.test.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/__tests__/capabilities.test.ts.
2. Implement: Extend promiseLedger.test for lite tier
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-005 complete.
- [ ] Tests green.

---
## PTWG-006

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: bdpt on lite silently ignored

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Construction EngineWarning when bdpt:true on lite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-006 complete.
- [ ] Tests green.

---
## PTWG-007

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: causticStrategy on lite no-op

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Validate causticStrategy at construction on lite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-007 complete.
- [ ] Tests green.

---
## PTWG-008

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: inverse on lite undefined

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Gate inverse to FD-only on lite tier
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-008 complete.
- [ ] Tests green.

---
## PTWG-009

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: inverse channels hardcoded 3

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Plumb alpha channel count from target image
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-009 complete.
- [ ] Tests green.

---
## PTWG-010

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: SPPM capacity fallback silent

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sppm/sppmSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/sppm/sppmSubsystem.ts.
2. Implement: EngineWarning when photon capacity clamped
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-010 complete.
- [ ] Tests green.

---
## PTWG-011

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: BDPT eye-stack OOM silent disable

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/bdpt/bdptSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/bdpt/bdptSubsystem.ts.
2. Implement: onWarning when BDPT disabled OOM
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-011 complete.
- [ ] Tests green.

---
## PTWG-012

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P1 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: getRestirPtResultBuffer undocumented

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Export on public interface + package README
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-012 complete.
- [ ] Tests green.

---
## PTWG-013

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: AO double-counts GI

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/bsdf.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/bsdf.wgsl.ts.
2. Implement: Fix AO vs GI coupling or downgrade ledger
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-013 complete.
- [ ] Tests green.

---
## PTWG-014

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lightMap camera-only

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts.
2. Implement: Add NEE lightMap term or mark approximate in ledger
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-014 complete.
- [ ] Tests green.

---
## PTWG-015

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: bump finite-difference stepping

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/bsdf.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/bsdf.wgsl.ts.
2. Implement: Analytic normal gradient or ledger approximate
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-015 complete.
- [ ] Tests green.

---
## PTWG-016

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: spectral emission flat-spectrum

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/spectral.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/spectral.wgsl.ts.
2. Implement: Jakob basis on emission
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-016 complete.
- [ ] Tests green.

---
## PTWG-017

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
pt-webgpu: Kulla-Conty LUT clamps

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/kullaConty.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/kullaConty.wgsl.ts.
2. Implement: Furnace test before promotion
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-017 complete.
- [ ] Tests green.

---
## PTWG-018

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: thickness closed-surface vs volume

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/transport.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/transport.wgsl.ts.
2. Implement: Thin-shell tracer for closed meshes
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-018 complete.
- [ ] Tests green.

---
## PTWG-019

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: extension maps approximate

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/scene/materialsBuffer.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/scene/materialsBuffer.ts.
2. Implement: Audit each extension map callsite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-019 complete.
- [ ] Tests green.

---
## PTWG-020

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: extension maps approximate 2

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/bsdf.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/bsdf.wgsl.ts.
2. Implement: Same audit specialty paths
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-020 complete.
- [ ] Tests green.

---
## PTWG-021

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: aoMapIntensity no indirect coupling in adjoint

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Document in inverse header
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-021 complete.
- [ ] Tests green.

---
## PTWG-022

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: alpha blend high variance

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts.
2. Implement: Opacity-aware NEE
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-022 complete.
- [ ] Tests green.

---
## PTWG-023

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: unlit terminal approximate

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/bsdf.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/bsdf.wgsl.ts.
2. Implement: Emissive-light participation for unlit
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-023 complete.
- [ ] Tests green.

---
## PTWG-024

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: texture LOD no ray differentials

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts.
2. Implement: Document approximate LOD
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-024 complete.
- [ ] Tests green.

---
## PTWG-025

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: missing adjoint hook FD only

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Warn when path-replay requested without hook
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-025 complete.
- [ ] Tests green.

---
## PTWG-026

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: bounces mismatch mid-session

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Freeze bounces at session construction
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-026 complete.
- [ ] Tests green.

---
## PTWG-027

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: transport fields FD; ior analytic unused

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Wire ior in brdfAdjoint
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-027 complete.
- [ ] Tests green.

---
## PTWG-028

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite merged transform FD wrong

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Diagnostic when lite + transform inverse
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-028 complete.
- [ ] Tests green.

---
## PTWG-029

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: mesh-area capped stream downgrade

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Warn with emitter metadata on downgrade
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-029 complete.
- [ ] Tests green.

---
## PTWG-030

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: env downgrades path-replay

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Document inverse scenes need env:none
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-030 complete.
- [ ] Tests green.

---
## PTWG-031

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: anisotropy coupled downgrade

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Document downgrade codes
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-031 complete.
- [ ] Tests green.

---
## PTWG-032

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: iridescence coupled params

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Document coupled downgrade
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-032 complete.
- [ ] Tests green.

---
## PTWG-033

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: texture kind throws Phase 2

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Implement texture opt or reserved diagnostic
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-033 complete.
- [ ] Tests green.

---
## PTWG-034

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: emitter geometry params not in adjoint

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointEligible.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointEligible.ts.
2. Implement: Extend ADJOINT_ELIGIBLE_EMITTER_FIELDS
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-034 complete.
- [ ] Tests green.

---
## PTWG-035

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | TOG |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: FD forward-only not central

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/finiteDifference.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/finiteDifference.ts.
2. Implement: Optional central difference toggle
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-035 complete.
- [ ] Tests green.

---
## PTWG-036

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite tier overview

### Files to modify (exact paths)
1. `packages/pt-webgpu/README.md`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/README.md.
2. Implement: Document lite tier limits
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-036 complete.
- [ ] Tests green.

---
## PTWG-038

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite positions:true vs transform throw

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/capabilities.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/capabilities.ts.
2. Implement: Align capabilities with throw behavior
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-038 complete.
- [ ] Tests green.

---
## PTWG-039

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: merged multi-material fidelity

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/scene/uploadSceneBuffers.ts.
2. Implement: Warn on merged multi-material
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-039 complete.
- [ ] Tests green.

---
## PTWG-040

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite tangents zero

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/scene/uploadSceneBuffers.ts.
2. Implement: Warn when anisotropy/normalMap on lite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-040 complete.
- [ ] Tests green.

---
## PTWG-041

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite directional/light tree

### Files to modify (exact paths)
1. `packages/pt-webgpu/README.md`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/README.md.
2. Implement: Document lite lighting limits
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-041 complete.
- [ ] Tests green.

---
## PTWG-042

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite directional 2

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/capabilities.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/capabilities.ts.
2. Implement: Capability rows for lite lights
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-042 complete.
- [ ] Tests green.

---
## PTWG-043

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite light tree

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/restir/lightTree.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/restir/lightTree.ts.
2. Implement: Document lite omits light tree
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-043 complete.
- [ ] Tests green.

---
## PTWG-044

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite MNEE stub

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Gate MNEE off on lite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-044 complete.
- [ ] Tests green.

---
## PTWG-045

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite variance texture misleading

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/gpuResources.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/gpuResources.ts.
2. Implement: Omit varianceMomentsBuffer on lite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-045 complete.
- [ ] Tests green.

---
## PTWG-046

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: webgpuLimits comment stale

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/gpuResources.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/gpuResources.ts.
2. Implement: Update post-B12 limits comment
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-046 complete.
- [ ] Tests green.

---
## PTWG-047

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: procedural-sky bake resolution

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/environment/proceduralSky.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/environment/proceduralSky.ts.
2. Implement: Document 256×128 constant
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-047 complete.
- [ ] Tests green.

---
## PTWG-048

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: BDPT serial single workgroup

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/bdpt/bdptDispatch.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/bdpt/bdptDispatch.ts.
2. Implement: Roadmap: parallel dispatch (document current limitation)
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-048 complete.
- [ ] Tests green.

---
## PTWG-049

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: bdptAdvanceFrame no layout validation

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/bdpt/bdptSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/bdpt/bdptSubsystem.ts.
2. Implement: Assert stride at runtime
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-049 complete.
- [ ] Tests green.

---
## PTWG-050

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | TEST |
| Lane | pt-webgpu |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: CPU/GPU BDPT emitter parity

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/bdpt/__tests__/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/bdpt/__tests__/.
2. Implement: Keep oracles green
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-050 complete.
- [ ] Tests green.

---
## PTWG-051

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: SPPM ring buffer not true progressive tail

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sppm/sppmSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/sppm/sppmSubsystem.ts.
2. Implement: Document streaming-window semantics
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-051 complete.
- [ ] Tests green.

---
## PTWG-052

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: SPPM_CELL_CAPACITY 32 bias

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sppm/sppmSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/sppm/sppmSubsystem.ts.
2. Implement: Adaptive cell size
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-052 complete.
- [ ] Tests green.

---
## PTWG-053

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: MNEE chain length default 3

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/mnee/mneeSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/mnee/mneeSubsystem.ts.
2. Implement: Diagnostic on truncate
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-053 complete.
- [ ] Tests green.

---
## PTWG-054

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | TEST |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: ReSTIR-PT off-default

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/restirPt/__tests__/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/restirPt/__tests__/.
2. Implement: Tests + warnings when disabled
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-054 complete.
- [ ] Tests green.

---
## PTWG-055

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | TEST |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: ReSTIR-PT partial init

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/restirPt/restirPtSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/restirPt/restirPtSubsystem.ts.
2. Implement: Init validation
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-055 complete.
- [ ] Tests green.

---
## PTWG-056

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | TEST |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: ReSTIR-PT composite drift

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/restirPt/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/restirPt/.
2. Implement: Composite parity test
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-056 complete.
- [ ] Tests green.

---
## PTWG-057

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | TEST |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: ReSTIR-PT warnings

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Warn when restirPtReuse without buffers
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-057 complete.
- [ ] Tests green.

---
## PTWG-058

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite varianceMomentsBuffer unused

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/gpuResources.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/gpuResources.ts.
2. Implement: Omit buffer alloc on lite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-058 complete.
- [ ] Tests green.

---
## PTWG-059

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: add/remove full repack

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sceneMutationRouter.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/sceneMutationRouter.ts.
2. Implement: Document fallback-rebuild for add/remove
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-059 complete.
- [ ] Tests green.

---
## PTWG-060

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: skin fallthrough

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sceneMutationRouter.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/sceneMutationRouter.ts.
2. Implement: GPU skin fast path after skinned positions patch
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-060 complete.
- [ ] Tests green.

---
## PTWG-061

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: env resize

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Rebuild env CDF on resize when HDRI
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-061 complete.
- [ ] Tests green.

---
## PTWG-062

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite patch field filter

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sceneMutationRouter.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/sceneMutationRouter.ts.
2. Implement: Reject unsupported lite patch fields at throw
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-062 complete.
- [ ] Tests green.

---
## PTWG-063

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: mutation router coverage

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/__tests__/updatePrimitiveIncremental.test.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/__tests__/updatePrimitiveIncremental.test.ts.
2. Implement: Cover all fast paths
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-063 complete.
- [ ] Tests green.

---
## PTWG-064

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: adjoint full-tier only

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Reject adjoint on lite
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-064 complete.
- [ ] Tests green.

---
## PTWG-065

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | BUG |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: unknown fieldCode maps to baseColor

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/optimizer.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/optimizer.ts.
2. Implement: Throw on unknown inverse param fieldCode
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-065 complete.
- [ ] Tests green.

---
## PTWG-066

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: env adjoint missing

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Env map adjoint or permanent downgrade doc
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-066 complete.
- [ ] Tests green.

---
## PTWG-067

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | pt-webgpu |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: partition vs getScene divergence

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Test getScene matches last setScene
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-067 complete.
- [ ] Tests green.

---
## PTWG-068

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: Beer slab

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/transport.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/transport.wgsl.ts.
2. Implement: Verify beer slab thickness
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-068 complete.
- [ ] Tests green.

---
## PTWG-069

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | BUG |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: implicit emitter desync

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/scene/meshAreaLights.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/scene/meshAreaLights.ts.
2. Implement: Resync mesh-area on material patch
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-069 complete.
- [ ] Tests green.

---
## PTWG-070

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: lite lights

### Files to modify (exact paths)
1. `packages/pt-webgpu/README.md`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/README.md.
2. Implement: Document lite emitter limits
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-070 complete.
- [ ] Tests green.

---
## PTWG-071

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: partition warnings

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Structured warnings on setScene partition
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-071 complete.
- [ ] Tests green.

---
## PTWG-072

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W019 (#19) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: ledger specular approximate

### Files to modify (exact paths)
1. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/core/src/engine/promiseLedger.ts.
2. Implement: Verify specular* rows match shader
3. Add/update test in packages/core/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-072 complete.
- [ ] Tests green.

---
## PTWG-073

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: emitterCastShadow

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/shaders/pathTrace.wgsl.ts.
2. Implement: Wire castShadow in specialty legs
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-073 complete.
- [ ] Tests green.

---
## PTWG-074

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: resize naming

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Clarify resize vs setSize in JSDoc
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-074 complete.
- [ ] Tests green.

---
## PTWG-075

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: OIDN turnkey

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Capability gate + clear error at construction
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-075 complete.
- [ ] Tests green.

---
## PTWG-076

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | DOC |
| Lane | pt-webgpu |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: traceTier defaults

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/index.ts.
2. Implement: Document default traceTier full
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-076 complete.
- [ ] Tests green.

---
## PTWG-077

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: mutation + inverse interaction

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/__tests__/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/__tests__/.
2. Implement: Integration test
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-077 complete.
- [ ] Tests green.

---
## PTWG-078

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | pt-webgpu |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: spectral + BDPT

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/__tests__/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/__tests__/.
2. Implement: Combined mode test
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-078 complete.
- [ ] Tests green.

---
## PTWG-079

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | pt-webgpu |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu: SPPM + env

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/__tests__/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/__tests__/.
2. Implement: Combined mode test
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-079 complete.
- [ ] Tests green.

---
## PTWG-080

| Field | Value |
|-------|-------|
| Phase | 3 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
pt-webgpu: behavioral gate coverage

### Files to modify (exact paths)
1. `tools/behavioral-gate/gate.mjs`

### Steps (execute in order — do not skip, do not improvise)
1. Open tools/behavioral-gate/gate.mjs.
2. Implement: Ensure all pt/* configs pass post-changes
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] PTWG-080 complete.
- [ ] Tests green.

---
## WH-001

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | ACC |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: displacement* unsupported on walkaround

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Line 142: keep unsupported; add partitionSceneBySupport warning code displacement-unsupported; update ledger permanent ACC
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-001 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-002

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | RT |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: spectralAttenuation unsupported

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Line 142: gltf featureReport routes to pt-webgpu; emit EngineWarning on walkaround setScene if field present
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-002 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-003

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | RT |
| Lane | walkaround-hybrid |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: dispersionAbbeNumber unsupported

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Same as WH-002 for dispersionAbbeNumber
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-003 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-004

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | RT |
| Lane | walkaround-hybrid |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: scattering* unsupported

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Route scatteringCoefficient*, scatteringAnisotropy, scatteringCoefficientRGB to pt-webgpu in gltf planner
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-004 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-005

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | RT |
| Lane | walkaround-hybrid |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: frontLayer/backLayer unsupported

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: partitionSceneBySupport warning layered-material-unsupported
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-005 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-006

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | RT |
| Lane | walkaround-hybrid |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: thinFilmStack unsupported

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Route thinFilmStack to pt-webgpu; warn on walkaround
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-006 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-007

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: extensions partial consume

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Line 50: warn on unknown extension keys via onWarning with code unknown-material-extension
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-007 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-008

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: specularIntensity GI approximate

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/restirGiMaterial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/restirGiMaterial.wgsl.ts.
2. Implement: Add specularIntensity channel to GI suffix packing in restirGiMaterial.wgsl + CPU packer
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-008 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-009

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: clearcoatRoughness GI approximate

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/restirGiMaterial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/restirGiMaterial.wgsl.ts.
2. Implement: Pack clearcoatRoughness into GI material suffix
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-009 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-010

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: sheenColor GI approximate

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/restirGiMaterial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/restirGiMaterial.wgsl.ts.
2. Implement: Pack sheenColor into GI suffix
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-010 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-011

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: alpha blend GI non-participant

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/shade.wgsl.ts.
2. Implement: Add HybridEngineOptions.alphaInGi toggle default false; when true enable stochastic alpha GI transport
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-011 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-012

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: emissiveMap texel PDF approximate

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/emitterList.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/emitterList.ts.
2. Implement: Implement alias table for emissive texel importance sampling
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-012 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-013

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: transmissionMap scalar-only in GI

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/packingHelpers.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/packingHelpers.ts.
2. Implement: Sample transmissionMap in GI emitter payloads not scalar transmission only
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-013 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-014

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: texCoord uv0/uv1 only

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts.
2. Implement: Extend BVH UV channels + atlas metadata for uv2/uv3
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-014 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-015

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: unreadable textures dropped

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts.
2. Implement: Line 324: strict mode throw OR async readback before atlas upload
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-015 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-016

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: texCoord 0/1 in metadata

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts.
2. Implement: Line 384: pack texCoord index per map in atlas metadata
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-016 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-017

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: TBN uv0/uv1 only

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts.
2. Implement: Line 494: third/fourth UV set in TBN construction
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-017 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-018

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: unlit approximate

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/shade.wgsl.ts.
2. Implement: Line 319: glTF KHR_materials_unlit terminal — emissive + baseColor only, skip lighting
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-018 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-019

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: lightMap camera-only

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Optional DDGI lightMap fold via engine option foldLightMapIntoDdgi default false
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-019 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-020

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: scalar vs texture alpha split

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/packingHelpers.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/packingHelpers.ts.
2. Implement: Line 592: unify alpha mask traversal for scalar opacity and alphaMap
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-020 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-021

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: causticStrategy warned not rejected

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Lines 554-571: throw TypeError at construction when causticStrategy !== none (not just EngineWarning)
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-021 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-022

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: maxSamplesPerPixel ignored

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Lines 572-588: expose explicit capability maxSamplesPerPixel: Infinity with documented no-op; remove ambiguous warn-only
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-022 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-023

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: causticOptions ignored

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Lines 590-599: throw when causticOptions provided (mirror causticStrategy reject)
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-023 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-024

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: maxBounces DDGI-only semantics

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Lines 529-551: document in HybridEngineOptions JSDoc that maxBounces gates DDGI indirect feedback not path depth
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-024 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-025

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: NRC biased experimental

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineOptions.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineOptions.ts.
2. Implement: Add quality guard: reject nrcEnabled unless experimentalQuality:true
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-025 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-026

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GRIS default off bias sources

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineOptions.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineOptions.ts.
2. Implement: Document restirPtReuse/GRIS toggles; add EngineFidelityProfile hook FP-06
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-026 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-027

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: lite tier OIDN/svgf not forbidden

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineConfig.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineConfig.ts.
2. Implement: Document VRAM cost in JSDoc when lite + heavy denoiser
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-027 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-028

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: lite forces merged BVH

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineConfig.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineConfig.ts.
2. Implement: Warn when useLite && scene needs TLAS (needsTlas heuristic)
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-028 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-029

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: svgf-real experimental objId

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/denoisers/svgfRealPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/denoisers/svgfRealPass.ts.
2. Implement: Finish conservative object-id reprojection path or downgrade ledger
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-029 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-030

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: supportsAuxBuffers false but partial aux

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Line 710-715: either wire variance to FrameOutput or stop emitting motionVectors/normalDepth
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-030 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-031

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: FrameInput.viewport ignored

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Lines 2015-2038: already warns once — add capability flag honorsViewport:false
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-031 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-032

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: stainedGlass caustic knobs inert

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineConfig.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineConfig.ts.
2. Implement: Validate mutual deps: stainedGlass extensions require rcEnabled or explicit opt-out warn
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-032 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-033

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: rcWeight clamp only when RC on

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineRC.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineRC.ts.
2. Implement: Validate rcWeight at parse in HybridEngineConfig when rcEnabled false → reject non-default rcWeight
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-033 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-035

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: normals-only triggers topologyRebuild

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts.
2. Implement: Add normals-only fast path: upload normals slice without full topology rebuild
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-035 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-036

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: empty patch silent no-op

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts.
2. Implement: Emit EngineWarning code empty-primitive-patch when patch object has no recognized keys
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-036 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-037

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: instances/params/shape full setScene

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts.
2. Implement: Port instanced TLAS refit from pt-webgpu sceneMutationRouter instanced-topology fast path
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-037 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-038

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: material patch throws if not ready

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts.
2. Implement: Queue material patches until pipeline init complete instead of throw
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-038 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-039

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: material patch throws if not ready (emitter)

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts.
2. Implement: Same queue pattern for updateEmitter
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-039 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-040

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | ACC |
| Lane | walkaround-hybrid |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: receiveShadow ignored

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts.
2. Implement: Emit partition warning; ledger permanent unsupported — do not implement receiver occlusion on walkaround
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-040 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-041

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: castShadow GI approximate

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/packingHelpers.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/packingHelpers.ts.
2. Implement: Verify MATERIAL_FLAG bit 1 set on all castShadow:true paths; fix if missing
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-041 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-042

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GPU skin bindMatrix

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/gpuSkin/gpuSkinBvh.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/gpuSkin/gpuSkinBvh.wgsl.ts.
2. Implement: Apply inverse bind matrix in skinning compute before BVH refit
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-042 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-043

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GPU skin morph targets

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/gpuSkin/gpuSkinBvh.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/gpuSkin/gpuSkinBvh.wgsl.ts.
2. Implement: Blend morph deltas in compute shader pre-BVH
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-043 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-044

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GPU skin tangents

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/gpuSkin/gpuSkinBvh.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/gpuSkin/gpuSkinBvh.wgsl.ts.
2. Implement: Skin tangents with inverse-transpose for normal maps
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-044 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-045

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P1 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: updatePrimitive during initializing

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Document: throws until setScene completes; or buffer patches in pending queue
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-045 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-046

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: ddgiMaxMaterials cap silent drop

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts.
2. Implement: Throw EngineError or LOD when material count exceeds ddgiMaxMaterials
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-046 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-047

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: unknown DDGI light kinds skipped

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/probeUpdateFrameParams.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/probeUpdateFrameParams.ts.
2. Implement: Reject unknown light kinds at setScene with structured warning
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-047 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-048

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: MAX_LIGHTS=16 truncation

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts.
2. Implement: Emit setScene warning when lights.length > 16 with code ddgi-lights-truncated
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-048 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-049

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: maxBounces==1 diffuse-only

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts.
2. Implement: Document direct-only probe behavior in DDGI README + ledger
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-049 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-050

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: area shadow opaque-first glass ignored

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts.
2. Implement: Implement ddgiTraceShadowTransmittance for glass shadows
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-050 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-051

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: glossy probe SH complement not GGX-filtered

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts.
2. Implement: Prefiltered radiance for glossy lobes in probe update
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-051 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-052

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: Beer path-length approx

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts.
2. Implement: Sample thicknessMap in probe rays for beer attenuation
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-052 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-053

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | BUG |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: DDGI updateFrame no device silent skip

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts.
2. Implement: Fail init when device null instead of silent skip
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-053 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-054

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: opaque HDRI without resolver

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/environment/resolveHybridEnvironment.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/environment/resolveHybridEnvironment.ts.
2. Implement: Require resolveEnvironmentMap callback when hdri opaque env set
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-054 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-055

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: DDGI texCoord limitation

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts.
2. Implement: Align with WH-014 uv2/uv3
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-055 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-056

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: indirectFeedback coupling

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ddgi/ddgiSubsystem.ts.
2. Implement: Document/setIndirectFeedback coupling to maxBounces
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-056 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-057

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: phantom zero emitter structural

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/emitterList.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/emitterList.ts.
2. Implement: Keep inert guards; add regression test zero-power emitters excluded
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-057 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-058

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: transmission classification scalar not map

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/emitterList.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/emitterList.ts.
2. Implement: Sample transmissionMap for glass classification
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-058 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-059

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: checkerboard history lag

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/restirDiTemporal.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/restirDiTemporal.ts.
2. Implement: Tune motion threshold or expose restirDiMotionThreshold option
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-059 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-060

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TEST |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: centroid pHat closed

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/__tests__/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/__tests__/.
2. Implement: Add regression pin for centroid pHat vs brute force
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-060 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-061

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: glass skips lo_direct

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/shade.wgsl.ts.
2. Implement: OIT direct path for glass lo_direct
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-061 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-062

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: CDF clamp edge bias

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/emitterList.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/emitterList.ts.
2. Implement: Exact binary search instead of clamped linear scan
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-062 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-063

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | BUG |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: dangling mesh-area emitter

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/emitterList.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/emitterList.ts.
2. Implement: Fail setScene when mesh-area emitter references missing primitive
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-063 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-064

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: ReGIR needs ≥2 emitters

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/restirDiSpatial.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/restirDiSpatial.ts.
2. Implement: Document minimum emitter count for ReGIR grid
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-064 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-065

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: ReGIR grid boundary clamp bias

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/restirDiSpatial.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/restirDiSpatial.ts.
2. Implement: Wrap/jitter grid cells at boundaries
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-065 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-066

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: Jacobian clamp [0.1,10]

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/jacobianShift.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/jacobianShift.wgsl.ts.
2. Implement: Document bias; enable GRIS via restirPtReuse or remove clamp when GRIS on
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-066 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-067

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: no reconnection visibility OFF

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts.
2. Implement: GRIS variant with reconnection visibility
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-067 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-068

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: OFF spatial no MIS denominator

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts.
2. Implement: Full GBH MIS denominator in OFF mode
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-068 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-069

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: temporal OFF same biases

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiTemporal.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiTemporal.wgsl.ts.
2. Implement: GRIS temporal path
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-069 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-070

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: glass walk 1-interface

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts.
2. Implement: Multi-interface Snell walk for glass
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-070 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-071

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: glass straight-through approx

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts.
2. Implement: Snell refraction instead of straight-through
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-071 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-072

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: glass rough approx

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts.
2. Implement: Rough glass microfacet in GI walk
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-072 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-073

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: blend excluded from reservoirs

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiTrace.wgsl.ts.
2. Implement: Document alpha blend exclusion from ReSTIR-GI reservoirs
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-073 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-074

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: NRC skips glass

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/neural/nrc/nrcTrainPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/neural/nrc/nrcTrainPass.ts.
2. Implement: Share risGi glass walk in NRC training rays
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-074 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-075

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GI suffix reflected-dir proxy

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts.
2. Implement: MIS suffix sample instead of reflected-dir proxy
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-075 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-076

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: rich payload heuristic

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/packingHelpers.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/packingHelpers.ts.
2. Implement: Use BVH material flags not heuristic for rich payload
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-076 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-077

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: spatial no object-id

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts.
2. Implement: Add objId similarity test in spatial reuse
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-077 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-078

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: temporal no object-id

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiTemporal.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiTemporal.wgsl.ts.
2. Implement: Add objId test in temporal reuse
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-078 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-079

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: PPG disabled on glass

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ppg/ppgGuidePass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ppg/ppgGuidePass.ts.
2. Implement: Train PPG on glass diffuse hits
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-079 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-080

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GRIS visibility skipGlass:true

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/restirGiSpatial.wgsl.ts.
2. Implement: Optional tinted reconnection when skipGlass false
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-080 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-081

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GTAO view-axis approx

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts.
2. Implement: Per-pixel unprojection for view axis
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-081 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-082

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GTAO horizon approx

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts.
2. Implement: Improved horizon search
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-082 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-083

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: shadingTerms composer ordering

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/wgslCompose.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/wgslCompose.ts.
2. Implement: Declarative injection order for shadingTerms modules
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-083 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-084

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: RC/ReSTIR-GI mix energy

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/shade.wgsl.ts.
2. Implement: Validate MIS normalization RC+GI; fix if sum ≠ 1
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-084 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-085

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: RC/ReSTIR-GI mix energy 2

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineRC.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineRC.ts.
2. Implement: Cross-check rcWeight application
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-085 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-086

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: OIT no GI/ReSTIR

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts.
2. Implement: Stochastic transparency GI coupling
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-086 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-087

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: OIT emissive approx

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts.
2. Implement: Accurate emissive in OIT pass
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-087 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-088

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: OIT ReSTIR gap

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts.
2. Implement: ReSTIR-DI on transparent surfaces
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-088 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-089

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: firefly clamps bias HDR

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/shade.wgsl.ts.
2. Implement: Auto scene-scale clamps tied to luminance telemetry
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-089 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-090

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: atrous-variance ≠ Schied SVGF

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/denoisers/atrousVariance.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/denoisers/atrousVariance.ts.
2. Implement: Rename mode description or document difference from svgf-real
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-090 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-091

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: BMFR screen-space position proxy

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/denoisers/bmfrPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/denoisers/bmfrPass.ts.
2. Implement: Use world-space G-buffer position
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-091 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-092

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: OIDN stale-by-one-frame

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/denoisers/oidnFinalPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/denoisers/oidnFinalPass.ts.
2. Implement: Document capture-mode-only freshness requirement
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-092 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-093

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: OIDN no model actionable error

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/denoisers/oidnFinalPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/denoisers/oidnFinalPass.ts.
2. Implement: Throw at construction with download URL when model missing
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-093 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-094

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: neural unsupported layer kinds skipped

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/neural/inferenceGraph.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/neural/inferenceGraph.ts.
2. Implement: Throw at init on unsupported layer kinds
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-094 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-095

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: starter neural weights only

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/neural/weights.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/neural/weights.ts.
2. Implement: Ship production checkpoint or document training pipeline
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-095 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-096

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: PPG dTree indexing

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ppg/dTree.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ppg/dTree.ts.
2. Implement: Fix indexing bugs + oracle tests
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-096 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-097

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: PPG MIS

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ppg/ppgGuidePass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ppg/ppgGuidePass.ts.
2. Implement: Tune α·p_guide+(1-α)·p_cos MIS
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-097 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-098

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: PPG decay

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts.
2. Implement: Adaptive decay cadence
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-098 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-099

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TEST |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: PPG oracle

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ppg/__tests__/`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ppg/__tests__/.
2. Implement: CPU oracle vs GPU guide distribution
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-099 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-100

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: PPG train buffer missing throws

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts.
2. Implement: Graceful degrade when train buffer unavailable
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-100 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-101

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: NRC trains biased GI

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/neural/nrc/nrcTrainPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/neural/nrc/nrcTrainPass.ts.
2. Implement: Unbiased reference target for distillation
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-101 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-102

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: NRC spreadC semantics

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/neural/nrc/nrcInferPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/neural/nrc/nrcInferPass.ts.
2. Implement: Fix spreadC per A6 roadmap
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-102 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-103

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: NRC distillation target

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/neural/nrc/nrcTrainPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/neural/nrc/nrcTrainPass.ts.
2. Implement: Align target with ReSTIR-GI reservoir
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-103 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-104

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: RC separate BVH ~50ms

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineRC.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineRC.ts.
2. Implement: Unify TLAS packer with ReSTIR/DDGI shared BVH
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-104 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-105

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: RC BVH sync

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineRC.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineRC.ts.
2. Implement: syncRestirBvhBuffers on every mutation
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-105 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-106

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: RC sun kinds

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineRC.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineRC.ts.
2. Implement: Document supported RC sun/directional kinds
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-106 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-107

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: RC MIS weight

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineRC.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineRC.ts.
2. Implement: Validate rcWeight MIS against oracle
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-107 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-108

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: G-buffer placeholders primary-ray

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/frameResources.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/frameResources.ts.
2. Implement: Real deferred G-buffer in primary-ray mode
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-108 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-109

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: G-buffer placeholders 2

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/PassRegistry.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/PassRegistry.ts.
2. Implement: Wire G-buffer passes
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-109 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-110

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: neural frame resources empty

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/frameResources.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/frameResources.ts.
2. Implement: Wire neural tensor slots
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-110 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-111

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: SVGF 1×1 when inactive

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/frameResources.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/frameResources.ts.
2. Implement: Lazy alloc SVGF buffers on denoiser mode switch
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-111 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-112

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: PPG flux readback fire-and-forget

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts.
2. Implement: Await readback or double-buffer flux
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-112 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-113

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: frame interval cap >60Hz

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Document frame interval cap behavior
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-113 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-114

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: env placeholder until updateEnvironment

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: First-frame scalar sky default instead of black
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-114 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-115

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: optional subsystem placeholder when disabled

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/frameResources.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/frameResources.ts.
2. Implement: Debug assert when enabled subsystem has null buffer
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-115 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-116

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: GI snapshot v1/v2 incompatible

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/giStateSnapshot.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/giStateSnapshot.ts.
2. Implement: Migration path v2→v4
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-116 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-117

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: checkerboard prefill denoiser list

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/PassRegistry.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/pipeline/PassRegistry.ts.
2. Implement: Document which denoisers support checkerboard prefill
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-117 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-118

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: timestamp 0n on lavapipe

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Telemetry guard when GPU timestamps unavailable
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-118 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-119

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: procedural-sky 256×128 bake

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/environment/proceduralSkyBake.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/environment/proceduralSkyBake.ts.
2. Implement: Higher res bake or analytic eval
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-119 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-120

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W012 (#12) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: opaque HDRI without resolver mandatory

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/environment/resolveHybridEnvironment.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/environment/resolveHybridEnvironment.ts.
2. Implement: Throw at setScene when hdri without resolver (duplicate WH-054 harden)
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-120 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-121

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: sun NEE skipGlass:true

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/shaders/shade.wgsl.ts.
2. Implement: Optional tinted transmittance for sun NEE
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-121 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-122

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | BUG |
| Lane | walkaround-hybrid |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: directional intensity not synced to primaryLightIntensity

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Derive directional emitter UBO from primaryLightIntensity on updateLighting
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-122 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-123

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TEST |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: mutation matrix completeness

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts.
2. Implement: Cover all patch kinds in matrix test
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-123 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-124

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: BVH castShadow split

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/bvhCore.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/bvhCore.ts.
2. Implement: Verify castShadow bit propagation TLAS+BLAS
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-124 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-125

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | DOC |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: alpha policy

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Document alpha blend/mask/scissor policy in README
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-125 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-126

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: sky-only env path

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/environment/resolveHybridEnvironment.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/environment/resolveHybridEnvironment.ts.
2. Implement: Test sky-only without HDRI
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-126 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-127

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: ppgDispatchInterval validation

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngineConfig.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngineConfig.ts.
2. Implement: Reject ppgDispatchInterval < 1 or > maxFrames
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-127 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## WH-128

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | VERIFY |
| Lane | walkaround-hybrid |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround: cross-cutting env+mutation

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-hybrid/src/HybridEngine.ts.
2. Implement: Integration test: setScene → updateEnvironment → updatePrimitive material
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] WH-128 fix implemented per steps.
- [ ] Tests green.
- [ ] typecheck green.

---
## GLTF-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-001: no bundled Draco

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/compression.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/compression.ts:137.
2. Gap: no bundled Draco
3. Fix: Optional peer dep + default hook
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-001 fix applied.
- [ ] Tests green.

---
## GLTF-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-002: meshopt unresolved

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/compression.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/compression.ts:262.
2. Gap: meshopt unresolved
3. Fix: Fail fast reject-unsupported
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-002 fix applied.
- [ ] Tests green.

---
## GLTF-003

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-003: Draco fail skip primitive

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/compression.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/compression.ts:440.
2. Gap: Draco fail skip primitive
3. Fix: Strip in strict modes
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-003 fix applied.
- [ ] Tests green.

---
## GLTF-004

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-004: Draco incomplete attrs

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/compression.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/compression.ts:386.
2. Gap: Draco incomplete attrs
3. Fix: Require complete semantics
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-004 fix applied.
- [ ] Tests green.

---
## GLTF-005

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-005: external URI not fetched

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/textures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/textures.ts:127.
2. Gap: external URI not fetched
3. Fix: Fetch in assetLoader
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-005 fix applied.
- [ ] Tests green.

---
## GLTF-006

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-006: Node raw-image no decode

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/textures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/textures.ts:194.
2. Gap: Node raw-image no decode
3. Fix: Default decodePixels in bridge
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-006 fix applied.
- [ ] Tests green.

---
## GLTF-007

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-007: basisu/webp inactive

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/textures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/textures.ts:261.
2. Gap: basisu/webp inactive
3. Fix: Auto-enable from featureReport
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-007 fix applied.
- [ ] Tests green.

---
## GLTF-008

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-008: decodeSceneTextures raw-image only

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/texturePipeline.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/texturePipeline.ts:493.
2. Gap: decodeSceneTextures raw-image only
3. Fix: ImageBitmap CPU path
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-008 fix applied.
- [ ] Tests green.

---
## GLTF-009

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-009: raw-image no decodePixels

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/texturePipeline.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/texturePipeline.ts:508.
2. Gap: raw-image no decodePixels
3. Fix: Require in reject-degraded
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-009 fix applied.
- [ ] Tests green.

---
## GLTF-010

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P1 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-010: walkaround atlas subset

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/texturePipeline.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/texturePipeline.ts:234.
2. Gap: walkaround atlas subset
3. Fix: Expand OR report per field
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-010 fix applied.
- [ ] Tests green.

---
## GLTF-011

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-011: unknown extensions unsupportedOptional

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/featureReport.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/featureReport.ts:745.
2. Gap: unknown extensions unsupportedOptional
3. Fix: Sync allowlist with gltfToScene
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-011 fix applied.
- [ ] Tests green.

---
## GLTF-012

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-012: emissiveMap.texelPdf approximate

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/featureReport.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/featureReport.ts:628.
2. Gap: emissiveMap.texelPdf approximate
3. Fix: Promote when NEE aligned
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-012 fix applied.
- [ ] Tests green.

---
## GLTF-013

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | DOC |
| Lane | gltf-adapter |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-013: lite all texture unsupported

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/featureReport.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/featureReport.ts:232.
2. Gap: lite all texture unsupported
3. Fix: Document profile
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-013 fix applied.
- [ ] Tests green.

---
## GLTF-014

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-014: vertex colors approximate

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/featureReport.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/featureReport.ts:228.
2. Gap: vertex colors approximate
3. Fix: Promote when validated
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-014 fix applied.
- [ ] Tests green.

---
## GLTF-015

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W017 (#17) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-015: sampler policy approximate

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/featureReport.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/featureReport.ts:609.
2. Gap: sampler policy approximate
3. Fix: Per-texture sampler creation
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-015 fix applied.
- [ ] Tests green.

---
## GLTF-016

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-016: cameras ignored

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/gltfToScene.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/gltfToScene.ts:468.
2. Gap: cameras ignored
3. Fix: Export Scene cameras optional
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-016 fix applied.
- [ ] Tests green.

---
## GLTF-017

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-017: TEXCOORD_2+ stripped

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/gltfToScene.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/gltfToScene.ts:1549.
2. Gap: TEXCOORD_2+ stripped
3. Fix: Extend UV sets OR split meshes
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-017 fix applied.
- [ ] Tests green.

---
## GLTF-018

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-018: morph COLOR/TEXCOORD ignored

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/gltfToScene.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/gltfToScene.ts:1802.
2. Gap: morph COLOR/TEXCOORD ignored
3. Fix: Map OR warn in featureReport
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-018 fix applied.
- [ ] Tests green.

---
## GLTF-019

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-019: unknown light type skipped

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/gltfToScene.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/gltfToScene.ts:1908.
2. Gap: unknown light type skipped
3. Fix: featureReport issue row
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-019 fix applied.
- [ ] Tests green.

---
## GLTF-020

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-020: doubleSided in extensions

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/materials.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/materials.ts:585.
2. Gap: doubleSided in extensions
3. Fix: First-class OR PTGL-011
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-020 fix applied.
- [ ] Tests green.

---
## GLTF-021

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-021: unknown KHR warn only

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/materials.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/materials.ts:526.
2. Gap: unknown KHR warn only
3. Fix: Extend KNOWN_KHR_EXTENSIONS
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-021 fix applied.
- [ ] Tests green.

---
## GLTF-022

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
GLTF-022: variant bindings stale throws

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/sceneController.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/gltf-adapter/src/sceneController.ts:334.
2. Gap: variant bindings stale throws
3. Fix: Recovery API + validation
4. Add regression test if behavior changes.
5. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] GLTF-022 fix applied.
- [ ] Tests green.

---
## MAT-WH-baseColor

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W020 (#20) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.baseColor is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote baseColor to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.baseColor row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] baseColor disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-roughness

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W021 (#21) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.roughness is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote roughness to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.roughness row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] roughness disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-metallic

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W022 (#22) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.metallic is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote metallic to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.metallic row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] metallic disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-shadingModel

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W023 (#23) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.shadingModel is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote shadingModel to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.shadingModel row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] shadingModel disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-alphaMode

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W024 (#24) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.alphaMode is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote alphaMode to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.alphaMode row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] alphaMode disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-alphaCutoff

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W025 (#25) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.alphaCutoff is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote alphaCutoff to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.alphaCutoff row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] alphaCutoff disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-opacity

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W026 (#26) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.opacity is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote opacity to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.opacity row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] opacity disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-transmission

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W027 (#27) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.transmission is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote transmission to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.transmission row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] transmission disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-ior

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W028 (#28) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.ior is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote ior to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.ior row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] ior disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-attenuationColor

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W029 (#29) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.attenuationColor is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote attenuationColor to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.attenuationColor row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] attenuationColor disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-attenuationDistance

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W030 (#30) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.attenuationDistance is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote attenuationDistance to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.attenuationDistance row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] attenuationDistance disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-thickness

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W031 (#31) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.thickness is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote thickness to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.thickness row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] thickness disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-baseColorMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W032 (#32) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.baseColorMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote baseColorMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.baseColorMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] baseColorMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-normalMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W033 (#33) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.normalMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote normalMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.normalMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] normalMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-normalScale

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W034 (#34) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.normalScale is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote normalScale to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.normalScale row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] normalScale disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-roughnessMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W035 (#35) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.roughnessMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote roughnessMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.roughnessMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] roughnessMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-metallicMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W036 (#36) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.metallicMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote metallicMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.metallicMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] metallicMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-transmissionMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W037 (#37) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.transmissionMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote transmissionMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.transmissionMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] transmissionMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-thicknessMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W038 (#38) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.thicknessMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote thicknessMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.thicknessMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] thicknessMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-emissiveMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W039 (#39) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.emissiveMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote emissiveMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.emissiveMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] emissiveMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-alphaMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W040 (#40) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.alphaMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote alphaMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.alphaMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] alphaMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-aoMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W041 (#41) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.aoMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote aoMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.aoMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] aoMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-aoMapIntensity

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W042 (#42) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.aoMapIntensity is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote aoMapIntensity to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.aoMapIntensity row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] aoMapIntensity disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-clearcoatMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W043 (#43) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.clearcoatMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote clearcoatMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.clearcoatMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] clearcoatMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-clearcoatRoughnessMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W044 (#44) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.clearcoatRoughnessMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote clearcoatRoughnessMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.clearcoatRoughnessMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] clearcoatRoughnessMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-clearcoatNormalMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W045 (#45) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.clearcoatNormalMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote clearcoatNormalMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.clearcoatNormalMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] clearcoatNormalMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-clearcoatNormalScale

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W046 (#46) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.clearcoatNormalScale is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote clearcoatNormalScale to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.clearcoatNormalScale row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] clearcoatNormalScale disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-sheenColorMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W047 (#47) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.sheenColorMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote sheenColorMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.sheenColorMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] sheenColorMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-sheenRoughnessMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W048 (#48) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.sheenRoughnessMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote sheenRoughnessMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.sheenRoughnessMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] sheenRoughnessMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-iridescenceMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W049 (#49) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.iridescenceMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote iridescenceMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.iridescenceMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] iridescenceMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-iridescenceThicknessMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W050 (#50) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.iridescenceThicknessMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote iridescenceThicknessMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.iridescenceThicknessMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] iridescenceThicknessMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-anisotropyMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W051 (#51) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.anisotropyMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote anisotropyMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.anisotropyMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] anisotropyMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-specularColorMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W052 (#52) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.specularColorMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote specularColorMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.specularColorMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] specularColorMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-specularIntensityMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W053 (#53) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.specularIntensityMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote specularIntensityMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.specularIntensityMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] specularIntensityMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-bumpMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W054 (#54) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.bumpMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote bumpMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.bumpMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] bumpMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-bumpScale

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W055 (#55) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.bumpScale is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote bumpScale to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.bumpScale row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] bumpScale disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-lightMap

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W056 (#56) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.lightMap is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote lightMap to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.lightMap row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] lightMap disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-lightMapIntensity

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W057 (#57) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.lightMapIntensity is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote lightMapIntensity to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.lightMapIntensity row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] lightMapIntensity disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-sheen

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W058 (#58) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.sheen is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote sheen to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.sheen row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] sheen disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-sheenColor

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W059 (#59) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.sheenColor is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote sheenColor to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.sheenColor row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] sheenColor disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-sheenRoughness

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W060 (#60) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.sheenRoughness is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote sheenRoughness to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.sheenRoughness row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] sheenRoughness disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-clearcoat

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W061 (#61) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.clearcoat is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote clearcoat to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.clearcoat row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] clearcoat disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-clearcoatRoughness

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W062 (#62) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.clearcoatRoughness is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote clearcoatRoughness to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.clearcoatRoughness row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] clearcoatRoughness disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-iridescence

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W063 (#63) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.iridescence is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote iridescence to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.iridescence row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] iridescence disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-iridescenceIor

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W064 (#64) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.iridescenceIor is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote iridescenceIor to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.iridescenceIor row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] iridescenceIor disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-iridescenceThicknessRange

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W065 (#65) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.iridescenceThicknessRange is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote iridescenceThicknessRange to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.iridescenceThicknessRange row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] iridescenceThicknessRange disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-specularIntensity

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W066 (#66) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.specularIntensity is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote specularIntensity to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.specularIntensity row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] specularIntensity disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-specularColor

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W067 (#67) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.specularColor is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote specularColor to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.specularColor row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] specularColor disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-anisotropy

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W068 (#68) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.anisotropy is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote anisotropy to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.anisotropy row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] anisotropy disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## MAT-WH-anisotropyRotation

| Field | Value |
|-------|-------|
| Phase | 4 |
| Priority | P2 |
| Disposition | TOG |
| Lane | core |
| Wave | W069 (#69) |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround MaterialSpec.anisotropyRotation is approximate (quantized atlas); ledger says approximate.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
3. `packages/core/src/engine/promiseLedger.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Decide per FP-06: promote anisotropyRotation to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.anisotropyRotation row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] anisotropyRotation disposition recorded in ledger.
- [ ] Test covers chosen path.

---
## FP-01

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
EngineFidelityProfile type does not exist — quality scattered across options.

### Files to modify (exact paths)
1. `packages/core/src/engine/fidelityProfile.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Create packages/core/src/engine/fidelityProfile.ts with EngineFidelityProfile type.
2. Fields: materialStorage quantized|full, emissiveImportance, alphaInGi, grisEnabled, restirPtReuse, traceTier.
3. Export from packages/core/src/engine/index.ts.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] Type exported from @vitrum/core.

---
## FP-02

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | FP-01, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
createEngine does not map fidelity profile.

### Files to modify (exact paths)
1. `packages/engine/src/createEngineInternals.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Add mapFidelityProfile(backend, options) called during engine construction.
2. Merge result into effective engine options.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] Profile mapping wired.

---
## FP-03

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | FP-02, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
capabilities.supportDetails not derived from active profile.

### Files to modify (exact paths)
1. `packages/engine/src/createEngineInternals.ts`
2. `packages/core/src/engine/capabilities.ts`

### Steps (execute in order — do not skip, do not improvise)
1. After mapFidelityProfile, patch supportDetails.materials rows affected by profile.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] Capabilities reflect active profile.

---
## FP-04

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | IMP |
| Lane | gltf-adapter |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | FP-01, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
gltf backend ranking ignores fidelity profile.

### Files to modify (exact paths)
1. `packages/gltf-adapter/src/featureReport.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Add rankGltfBackends(report, profile) using profile constraints.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] Ranking function exported and tested.

---
## FP-05

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P2 |
| Disposition | DOC |
| Lane | core |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | FP-01, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Pipeline-rebuild-required toggles undocumented.

### Files to modify (exact paths)
1. `packages/core/src/engine/fidelityProfile.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Document which profile fields require engine recreation vs runtime toggle.

### Tests (run exactly in this order)
```bash
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] JSDoc lists rebuild-required fields.

---
## FP-06

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | FP-01, FP-03, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Runtime UBO bits for material storage not wired.

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`
2. `packages/walkaround-hybrid/src/HybridEngine.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Add HybridEngine UBO bit materialStorageQuantized from profile.
2. shade.wgsl reads bit to select atlas decode path.

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] UBO bit wired end-to-end.

---
## MUT-01

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W004 (#4) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | PTGL-002, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-01: pt-webgl2 transform fallback

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task PTGL-002 first.
2. Verify MUT-01 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-01 closed when PTGL-002 done and ledger verified.

---
## MUT-02

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W005 (#5) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | PTGL-002, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-02: pt-webgl2 positions fallback

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task PTGL-002 first.
2. Verify MUT-02 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-02 closed when PTGL-002 done and ledger verified.

---
## MUT-03

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | PTGL-002, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-03: pt-webgl2 topology fallback

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task PTGL-002 first.
2. Verify MUT-03 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-03 closed when PTGL-002 done and ledger verified.

---
## MUT-04

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W007 (#7) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | PTGL-001, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-04: pt-webgl2 material maps fallback

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task PTGL-001 first.
2. Verify MUT-04 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-04 closed when PTGL-001 done and ledger verified.

---
## MUT-05

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W002 (#2) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | PTWG-001, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-05: pt-webgpu resize unsupported

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/index.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task PTWG-001 first.
2. Verify MUT-05 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-05 closed when PTWG-001 done and ledger verified.

---
## MUT-06

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W009 (#9) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | PTWG-038, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-06: pt-webgpu lite transform throw

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sceneMutationRouter.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task PTWG-038 first.
2. Verify MUT-06 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-06 closed when PTWG-038 done and ledger verified.

---
## MUT-07

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | P0-001-PTWG-037, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-07: pt-webgpu lite material GPU stale

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/sceneMutationRouter.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task P0-001-PTWG-037 first.
2. Verify MUT-07 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-07 closed when P0-001-PTWG-037 done and ledger verified.

---
## MUT-08

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | WH-037, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-08: walkaround topology fallback

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task WH-037 first.
2. Verify MUT-08 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-08 closed when WH-037 done and ledger verified.

---
## MUT-09

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W001 (#1) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | P0-003-WH-034, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-09: walkaround material DDGI skip

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task P0-003-WH-034 first.
2. Verify MUT-09 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-09 closed when P0-003-WH-034 done and ledger verified.

---
## MUT-10

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W013 (#13) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | WH-120, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-10: walkaround opaque HDRI

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/environment/resolveHybridEnvironment.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task WH-120 first.
2. Verify MUT-10 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-10 closed when WH-120 done and ledger verified.

---
## MUT-11

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | CORE-004, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-11: receiveShadow unsupported

### Files to modify (exact paths)
1. `packages/core/src/scene/primitives.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task CORE-004 first.
2. Verify MUT-11 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-11 closed when CORE-004 done and ledger verified.

---
## MUT-12

| Field | Value |
|-------|-------|
| Phase | 2 |
| Priority | P1 |
| Disposition | IMP |
| Lane | core |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | CORE-005, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Mutation matrix MUT-12: displacement unsupported

### Files to modify (exact paths)
1. `packages/core/src/scene/material.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Complete linked task CORE-005 first.
2. Verify MUT-12 acceptance: mutation ledger row matches runtime behavior.

### Tests (run exactly in this order)
```bash
npm run typecheck
npm test
```

### Done when (ALL boxes required before next task)
- [ ] MUT-12 closed when CORE-005 done and ledger verified.

---
## INV-001

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: ssim loss throws

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Implement SSIM loss kernel OR remove from InverseLoss union
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-001 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-002

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W017 (#17) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: lpips loss throws

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Implement LPIPS loss OR remove from union
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-002 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-003

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W018 (#18) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: texture kind throws Phase 2

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Implement texture optimization session
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-003 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-004

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: bounces>1 path-replay downgrade

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Multi-bounce adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-004 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-005

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W017 (#17) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: spectral path-replay downgrade

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Spectral adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-005 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-006

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W018 (#18) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: ior/transmission/thickness/attenuation FD only

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Wire transport adjoint in WGSL
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-006 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-007

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W019 (#19) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: opacity/alphaCutoff FD only

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Visibility adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-007 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-008

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W020 (#20) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: normalScale/bumpScale FD only

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Normal-map adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-008 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-009

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W021 (#21) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: envMapIntensity FD only

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Env-map adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-009 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-010

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W022 (#22) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: texture maps partial adjoint

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Map local chain + pixel grads per header comment
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-010 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-011

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W019 (#19) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: environmentKind!==none downgrade

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Env light adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-011 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-012

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W020 (#20) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: non-mesh primitive targets downgrade

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Analytic adjoint OR reject at session creation
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-012 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-013

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W021 (#21) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: layered/volume/spectral domains

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/inverseSession.ts.
2. Implement: Specialty BRDF adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-013 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-014

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W023 (#23) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: soft-sun/mesh-area/texel-PDF emitters partial

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Extend emitter sampling in adjoint
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-014 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-015

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W024 (#24) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: forward MIS not mirrored in adjoint

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: MIS parity in adjoint pass
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-015 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## INV-016

| Field | Value |
|-------|-------|
| Phase | 6 |
| Priority | P2 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W025 (#25) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
inverse: indirect/multi-bounce not in path-replay

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Research-scope adjoint path or document permanent FD
3. Add inverseSession regression test.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/
cd /home/jsquire4/projects/vitrum && npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] INV-016 implemented or permanently documented as FD-only.
- [ ] inverse tests green.

---
## SBVH-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: CPU pick O(n)

### Files to modify (exact paths)
1. `packages/shared-bvh/src/pick.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/pick.ts.
2. Fix: Implement BVH-accelerated pick or document O(n) permanent
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-001 complete.
- [ ] Tests green.

---
## SBVH-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: rest-pose skinned pick

### Files to modify (exact paths)
1. `packages/shared-bvh/src/pick.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/pick.ts.
2. Fix: Use posed positions when skinned mesh
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-002 complete.
- [ ] Tests green.

---
## SBVH-003

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: analytic sphere proxy

### Files to modify (exact paths)
1. `packages/shared-bvh/src/buildArrayBvh.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/buildArrayBvh.ts.
2. Fix: Tighter analytic bounds
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-003 complete.
- [ ] Tests green.

---
## SBVH-004

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: emissive decode dup

### Files to modify (exact paths)
1. `packages/shared-bvh/src/emissiveDecode.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/emissiveDecode.ts.
2. Fix: Dedup with walkaround copy
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-004 complete.
- [ ] Tests green.

---
## SBVH-005

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: fingerprint sampling

### Files to modify (exact paths)
1. `packages/shared-bvh/src/fingerprint.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/fingerprint.ts.
2. Fix: Document collision probability
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-005 complete.
- [ ] Tests green.

---
## SBVH-006

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: empty TLAS throw

### Files to modify (exact paths)
1. `packages/shared-bvh/src/tlas.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/tlas.ts.
2. Fix: EngineWarning instead of throw on empty scene
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-006 complete.
- [ ] Tests green.

---
## SBVH-007

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: validateBvhEncoding internal

### Files to modify (exact paths)
1. `packages/shared-bvh/src/validateBvhEncoding.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/validateBvhEncoding.ts.
2. Fix: Keep @internal; document
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-007 complete.
- [ ] Tests green.

---
## SBVH-008

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-bvh |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-bvh: pick primitive id stability

### Files to modify (exact paths)
1. `packages/shared-bvh/src/pick.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-bvh/src/pick.ts.
2. Fix: Regression test pick id
3. Test in packages/shared-bvh.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SBVH-008 complete.
- [ ] Tests green.

---
## SSAMP-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-samplers: ReGIR CPU-only here

### Files to modify (exact paths)
1. `packages/shared-samplers/src/regir.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-samplers/src/regir.ts.
2. Fix: Document walkaround consumes CPU grid only
3. Test in packages/shared-samplers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-samplers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SSAMP-001 complete.
- [ ] Tests green.

---
## SSAMP-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-samplers: zero-emitter light tree throw

### Files to modify (exact paths)
1. `packages/shared-samplers/src/lightTree.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-samplers/src/lightTree.ts.
2. Fix: Graceful empty tree
3. Test in packages/shared-samplers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-samplers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SSAMP-002 complete.
- [ ] Tests green.

---
## SSAMP-003

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-samplers: UBO padding drift

### Files to modify (exact paths)
1. `packages/shared-samplers/src/uboCodegen.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-samplers/src/uboCodegen.ts.
2. Fix: Pin padding in tests
3. Test in packages/shared-samplers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-samplers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SSAMP-003 complete.
- [ ] Tests green.

---
## SSAMP-004

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-samplers: Jakob approx

### Files to modify (exact paths)
1. `packages/shared-samplers/src/spectral.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-samplers/src/spectral.ts.
2. Fix: Document approximation bounds
3. Test in packages/shared-samplers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-samplers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SSAMP-004 complete.
- [ ] Tests green.

---
## SSAMP-005

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W014 (#14) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-samplers: BDPT MIS oracle vs pt-webgl2

### Files to modify (exact paths)
1. `packages/shared-samplers/src/bdpt.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-samplers/src/bdpt.ts.
2. Fix: Keep oracle parity test green
3. Test in packages/shared-samplers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-samplers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SSAMP-005 complete.
- [ ] Tests green.

---
## SSAMP-006

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-samplers: Preetham bake not live GLSL

### Files to modify (exact paths)
1. `packages/shared-samplers/src/preetham.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-samplers/src/preetham.ts.
2. Fix: Wire live eval or document bake-only
3. Test in packages/shared-samplers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-samplers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SSAMP-006 complete.
- [ ] Tests green.

---
## SDENO-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-denoisers |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-denoisers: SVGF one-shot unused

### Files to modify (exact paths)
1. `packages/shared-denoisers/src/svgfReal.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-denoisers/src/svgfReal.ts.
2. Fix: Remove dead export or wire
3. Test in packages/shared-denoisers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-denoisers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SDENO-001 complete.
- [ ] Tests green.

---
## SDENO-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-denoisers |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-denoisers: OIDN dynamic import

### Files to modify (exact paths)
1. `packages/shared-denoisers/src/oidnBridge.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-denoisers/src/oidnBridge.ts.
2. Fix: Document lazy load
3. Test in packages/shared-denoisers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-denoisers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SDENO-002 complete.
- [ ] Tests green.

---
## SDENO-003

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-denoisers |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-denoisers: shared device opt-in

### Files to modify (exact paths)
1. `packages/shared-denoisers/src/sharedWebGpuDevice.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-denoisers/src/sharedWebGpuDevice.ts.
2. Fix: Document reuseSharedWebGpuDevice===true
3. Test in packages/shared-denoisers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-denoisers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SDENO-003 complete.
- [ ] Tests green.

---
## SDENO-004

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-denoisers |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-denoisers: BMFR not in engine union

### Files to modify (exact paths)
1. `packages/shared-denoisers/src/bmfr.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-denoisers/src/bmfr.ts.
2. Fix: Add to HybridEngineOptions or document external-only
3. Test in packages/shared-denoisers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-denoisers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SDENO-004 complete.
- [ ] Tests green.

---
## SDENO-005

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-denoisers |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-denoisers: HDR bilateral not in union

### Files to modify (exact paths)
1. `packages/shared-denoisers/src/hdrLuminanceBilateral.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-denoisers/src/hdrLuminanceBilateral.ts.
2. Fix: Same as SDENO-004
3. Test in packages/shared-denoisers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-denoisers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SDENO-005 complete.
- [ ] Tests green.

---
## SDENO-006

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-denoisers |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shared-denoisers: atrous variance naming

### Files to modify (exact paths)
1. `packages/shared-denoisers/src/atrousVariance.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/shared-denoisers/src/atrousVariance.ts.
2. Fix: Clarify vs svgf-real in README
3. Test in packages/shared-denoisers.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/shared-denoisers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SDENO-006 complete.
- [ ] Tests green.

---
## RC-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-rc |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround-rc: placeholder env/atlas

### Files to modify (exact paths)
1. `packages/walkaround-rc/src/cascadeDispatch.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-rc/src/cascadeDispatch.ts.
2. Fix: Wire real env inputs
3. Test in packages/walkaround-rc.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-rc && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] RC-001 complete.
- [ ] Tests green.

---
## RC-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-rc |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround-rc: placeholder emitters

### Files to modify (exact paths)
1. `packages/walkaround-rc/src/cascadeDispatch.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-rc/src/cascadeDispatch.ts.
2. Fix: Wire emitter buffer
3. Test in packages/walkaround-rc.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-rc && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] RC-002 complete.
- [ ] Tests green.

---
## RC-003

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-rc |
| Wave | W017 (#17) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround-rc: merged TLAS placeholders

### Files to modify (exact paths)
1. `packages/walkaround-rc/src/cascadeDispatch.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-rc/src/cascadeDispatch.ts.
2. Fix: Use shared TLAS packer
3. Test in packages/walkaround-rc.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-rc && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] RC-003 complete.
- [ ] Tests green.

---
## RC-004

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-rc |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround-rc: mesh-area NEE gap

### Files to modify (exact paths)
1. `packages/walkaround-rc/src/receiver.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-rc/src/receiver.wgsl.ts.
2. Fix: Document or wire mesh lights
3. Test in packages/walkaround-rc.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-rc && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] RC-004 complete.
- [ ] Tests green.

---
## RC-005

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-rc |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround-rc: cascade rays power-of-two throw

### Files to modify (exact paths)
1. `packages/walkaround-rc/src/cascadePyramid.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-rc/src/cascadePyramid.ts.
2. Fix: Clear error message
3. Test in packages/walkaround-rc.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-rc && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] RC-005 complete.
- [ ] Tests green.

---
## RC-006

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | walkaround-rc |
| Wave | W018 (#18) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
walkaround-rc: RC env rotation

### Files to modify (exact paths)
1. `packages/walkaround-rc/src/cascadeDispatch.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/walkaround-rc/src/cascadeDispatch.ts.
2. Fix: Sync with DDGI env rotation
3. Test in packages/walkaround-rc.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/walkaround-rc && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] RC-006 complete.
- [ ] Tests green.

---
## SL-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | scene-lighting |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
scene-lighting: per-mode intensity multipliers

### Files to modify (exact paths)
1. `packages/scene-lighting/src/lightingModes.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/scene-lighting/src/lightingModes.ts.
2. Fix: Document multipliers
3. Test in packages/scene-lighting.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/scene-lighting && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SL-001 complete.
- [ ] Tests green.

---
## SL-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | scene-lighting |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
scene-lighting: stained-glass sky defaults

### Files to modify (exact paths)
1. `packages/scene-lighting/src/stainedGlassSky.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/scene-lighting/src/stainedGlassSky.ts.
2. Fix: Export defaults
3. Test in packages/scene-lighting.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/scene-lighting && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SL-002 complete.
- [ ] Tests green.

---
## SL-003

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | scene-lighting |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
scene-lighting: heuristic skyIrradiance

### Files to modify (exact paths)
1. `packages/scene-lighting/src/skyIrradiance.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/scene-lighting/src/skyIrradiance.ts.
2. Fix: Document heuristic
3. Test in packages/scene-lighting.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/scene-lighting && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SL-003 complete.
- [ ] Tests green.

---
## SG-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | stained-glass-extensions |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
stained-glass-extensions: packCameUBO no consumer

### Files to modify (exact paths)
1. `packages/stained-glass-extensions/src/packCameUBO.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/stained-glass-extensions/src/packCameUBO.ts.
2. Fix: Wire consumer or move to host
3. Test in packages/stained-glass-extensions.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/stained-glass-extensions && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SG-001 complete.
- [ ] Tests green.

---
## SG-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | stained-glass-extensions |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
stained-glass-extensions: segment cap warn-only

### Files to modify (exact paths)
1. `packages/stained-glass-extensions/src/segmentCap.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/stained-glass-extensions/src/segmentCap.ts.
2. Fix: Throw in strict mode
3. Test in packages/stained-glass-extensions.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/stained-glass-extensions && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] SG-002 complete.
- [ ] Tests green.

---
## DEV-001

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | dev |
| Wave | W015 (#15) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
dev: Denoiser toggle walkaround-only

### Files to modify (exact paths)
1. `packages/dev/src/react/DenoiserABToggle.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/dev/src/react/DenoiserABToggle.tsx.
2. Fix: Document backend guard
3. Test in packages/dev.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] DEV-001 complete.
- [ ] Tests green.

---
## DEV-002

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | dev |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
dev: gpu blit format

### Files to modify (exact paths)
1. `packages/dev/src/overlays/gpuBlit.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/dev/src/overlays/gpuBlit.ts.
2. Fix: Match engine swapchain format
3. Test in packages/dev.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] DEV-002 complete.
- [ ] Tests green.

---
## DEV-003

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | dev |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
dev: MaterialInspector partial fields

### Files to modify (exact paths)
1. `packages/dev/src/react/MaterialInspector.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/dev/src/react/MaterialInspector.tsx.
2. Fix: Expand to full MaterialSpec
3. Test in packages/dev.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] DEV-003 complete.
- [ ] Tests green.

---
## DEV-004

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | dev |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
dev: pickPrimitive optional

### Files to modify (exact paths)
1. `packages/dev/src/react/PickOverlay.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/dev/src/react/PickOverlay.tsx.
2. Fix: Hide when unsupported
3. Test in packages/dev.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] DEV-004 complete.
- [ ] Tests green.

---
## DEV-005

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | dev |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
dev: frameMonitor requires return value

### Files to modify (exact paths)
1. `packages/dev/src/react/FrameMonitor.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/dev/src/react/FrameMonitor.tsx.
2. Fix: Document onFrame requirement
3. Test in packages/dev.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] DEV-005 complete.
- [ ] Tests green.

---
## DEV-006

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | IMP |
| Lane | dev |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
dev: GI overlay no skip reason

### Files to modify (exact paths)
1. `packages/dev/src/react/GISignalSplit.tsx`

### Steps (execute in order — do not skip, do not improvise)
1. Open packages/dev/src/react/GISignalSplit.tsx.
2. Fix: Show skip reason from engine
3. Test in packages/dev.
4. npm run typecheck

### Tests (run exactly in this order)
```bash
cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] DEV-006 complete.
- [ ] Tests green.

---
## TOOL-002

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | TEST |
| Lane | tools |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | P0-008-TOOL-001, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
behavioral gate missing oidn/neural/svgf-real walkaround configs

### Files to modify (exact paths)
1. `tools/behavioral-gate/gate.mjs`

### Steps (execute in order — do not skip, do not improvise)
1. Add opt-in wh/oidn, wh/neural, wh/svgf-real configs behind env flag VITRUM_HEAVY_DENOISER_GATE=1

### Tests (run exactly in this order)
```bash
node tools/behavioral-gate/gate.mjs --filter wh/oidn
```

### Done when (ALL boxes required before next task)
- [ ] Configs documented and gated.

---
## TOOL-003

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | TEST |
| Lane | tools |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | P0-008-TOOL-001, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
shader-gate compiles WGSL only not GLSL pt-webgl2

### Files to modify (exact paths)
1. `tools/shader-gate/gate.mjs`

### Steps (execute in order — do not skip, do not improvise)
1. Add glslGate pass mirroring wgsl compose for pt-webgl2 composeTraceGlsl

### Tests (run exactly in this order)
```bash
npm run shader-gate
```

### Done when (ALL boxes required before next task)
- [ ] GLSL paths in shader-gate.

---
## TOOL-004

| Field | Value |
|-------|-------|
| Phase | 5 |
| Priority | P2 |
| Disposition | TEST |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
gltf-material-sweep limited fixtures

### Files to modify (exact paths)
1. `tools/gltf-material-sweep/sweep.mjs`

### Steps (execute in order — do not skip, do not improvise)
1. Expand SWEEP_MAPS to cover all 65 material fields

### Tests (run exactly in this order)
```bash
node tools/gltf-material-sweep/sweep.mjs --dry-run
```

### Done when (ALL boxes required before next task)
- [ ] Sweep map list complete.

---
## TOOL-005

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | DOC |
| Lane | tools |
| Wave | W006 (#6) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
benchmark-runner no in-repo Playwright

### Files to modify (exact paths)
1. `tools/benchmark-runner/README.md`

### Steps (execute in order — do not skip, do not improvise)
1. Document host Playwright requirement for browser capture modes

### Tests (run exactly in this order)
```bash
```

### Done when (ALL boxes required before next task)
- [ ] README updated.

---
## TOOL-006

| Field | Value |
|-------|-------|
| Phase | 1 |
| Priority | P2 |
| Disposition | TEST |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
radiometric-ab oracles not CI-gated

### Files to modify (exact paths)
1. `package.json`

### Steps (execute in order — do not skip, do not improvise)
1. Add optional npm script test:radiometric-ab behind env flag

### Tests (run exactly in this order)
```bash
npm run test:radiometric-ab
```

### Done when (ALL boxes required before next task)
- [ ] Script wired.

---
## RT100-V28-000

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P1 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] V28-B baseline recapture on wsl-gpu. See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `HARDWARE-VALIDATION-NEEDS.md`
2. `tools/reference-renders/`

### Steps (execute in order — do not skip, do not improvise)
1. Recapture reference renders for all render-changing landings.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] V28-B baseline set captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-A1

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] A1 radiometric/variance GPU A/B (road-to-100 Bucket A). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run A1 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] A1 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-A2

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] A2 radiometric/variance GPU A/B (road-to-100 Bucket A). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run A2 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] A2 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-A3

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] A3 radiometric/variance GPU A/B (road-to-100 Bucket A). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run A3 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] A3 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-A4

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] A4 radiometric/variance GPU A/B (road-to-100 Bucket A). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run A4 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] A4 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-A5

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] A5 radiometric/variance GPU A/B (road-to-100 Bucket A). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run A5 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] A5 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-A7

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] A7 radiometric/variance GPU A/B (road-to-100 Bucket A). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run A7 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] A7 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-B1

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] B1 radiometric/variance GPU A/B (road-to-100 Bucket B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run B1 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] B1 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-B2

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] B2 radiometric/variance GPU A/B (road-to-100 Bucket B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run B2 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] B2 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-B4

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] B4 radiometric/variance GPU A/B (road-to-100 Bucket B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run B4 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] B4 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-B8

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] B8 radiometric/variance GPU A/B (road-to-100 Bucket B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run B8 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] B8 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-B15

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] B15 radiometric/variance GPU A/B (road-to-100 Bucket B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run B15 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] B15 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V28-B16

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] B16 radiometric/variance GPU A/B (road-to-100 Bucket B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `HARDWARE-VALIDATION-NEEDS.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run B16 A/B scenario on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] B16 V28 evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-V19-GRIS

| Field | Value |
|-------|-------|
| Phase | 8 |
| Priority | P1 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] GRIS-on unbiasedness + biased-default GPU quantification (A8/F6). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/benchmark-runner/`

### Steps (execute in order — do not skip, do not improvise)
1. Run converged-unbiasedness harness on wsl-gpu.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] V19 GRIS A/B report written.
- [ ] Deferred — no agent dispatch.

---
## RT100-ADJ-001

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgpu |
| Wave | W003 (#3) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | P0-008-TOOL-001, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgpu inverse adjoint path replay OPEN: alpha-map adjoint, normal/bump/transmission adjoints, env light terms, indirect paths (road-to-100 §2C Adjoint row).

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`
2. `packages/pt-webgpu/src/inverse/pathTraceAdjoint.wgsl.ts`
3. `packages/pt-webgpu/src/inverse/inverseSession.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100.md §2C integrator audit Adjoint row Still OPEN list.
2. Implement or permanently downgrade each OPEN adjoint domain with structured diagnostic.
3. Extend brdfAdjoint.test.ts + inverseSession.test.ts per domain.
4. Update promiseLedger inverse downgrade matrix comments.

### Tests (run exactly in this order)
```bash
cd packages/pt-webgpu && npx vitest run src/inverse/
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] Each OPEN adjoint row closed or downgraded with test.

---
## RT100-WA-3D

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W010 (#10) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | P0-003-WH-034, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Walkaround Phase 3D tail: remaining atlas gaps — bump/displacement policy rows, morph-target UV deform refresh, narrower atlas refresh optimization (road-to-100 §3D footguns).

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
2. `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`
3. `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100.md Phase 3D — atlas slices largely landed; identify remaining gaps vs Master checklist.
2. Implement bump map consumption if still missing from shade/ReSTIR paths.
3. Add morph-target UV deform detection → full atlas refresh or documented limitation.
4. Optional: narrower atlas refresh for map-handle-only edits (cost footgun).

### Tests (run exactly in this order)
```bash
cd packages/walkaround-hybrid && npx vitest run
cd packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] 3D footguns closed or documented with tests.
- [ ] Ledger rows match CONSUMED_MATERIAL_FIELDS.

---
## RT100-WA-3E

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-WA-3D |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] Walkaround Phase 3E promotion via material-furnace/reference A/B. See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/reference-renders/`

### Steps (execute in order — do not skip, do not improvise)
1. Run promotion A/B per extension lobe.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] 3E promotion evidence captured.
- [ ] Deferred — no agent dispatch.

---
## RT100-WA-ALPHA

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | IMP |
| Lane | walkaround-hybrid |
| Wave | W011 (#11) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | RT100-WA-3D, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Walkaround Phase 3C alpha: transparent ReSTIR/GI promotion + layered transport (road-to-100 Master checklist alpha rows).

### Files to modify (exact paths)
1. `packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts`
2. `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`
3. `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100 alpha row: OIT direct sun done; ReSTIR/GI transport still approximate.
2. Implement stochastic alpha GI transport OR document permanent OIT-split with ledger ACC.
3. Add behavioral gate wh/alpha-blend if not present.

### Tests (run exactly in this order)
```bash
cd packages/walkaround-hybrid && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] Alpha GI policy implemented or permanently documented.

---
## RT100-PTGL-MUT

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | IMP |
| Lane | pt-webgl2 |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | P0-002-PTGL-003, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
pt-webgl2 geometry mutations: port TLAS/refit/splice from pt-webgpu sceneMutationRouter (road-to-100 §2D — transform/positions/topology still fallback-rebuild).

### Files to modify (exact paths)
1. `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`
2. `packages/pt-webgpu/src/sceneMutationRouter.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Study pt-webgpu fast paths: transform, positions, topology-resize, instanced-topology.
2. Port applicable paths to pt-webgl2 WebGL2 buffer upload model.
3. Promote promiseLedger mutation rows from fallback-rebuild to native where implemented.
4. Extend updatePrimitiveIncremental tests on pt-webgl2.

### Tests (run exactly in this order)
```bash
cd packages/pt-webgl2 && npx vitest run
cd packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts
```

### Done when (ALL boxes required before next task)
- [ ] transform/positions/topology native or explicitly documented fallback.

---
## RT100-PTWG-MAT

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | IMP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | P0-001-PTWG-037 |
| Blocks | (none) |

### Problem
PTWG-MAT-01 integrator audit: extension lobes must match across BDPT/SPPM/ReSTIR-PT/MNEE paths — material-furnace promotion tail.

### Files to modify (exact paths)
1. `packages/pt-webgpu/src/wgsl/pathTrace/`
2. `packages/pt-webgpu/src/__tests__/extensionLobeReference.test.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100.md §2C integrator audit table.
2. For each path row not ✅: wire evaluateBrdf/sampleNextBounceDirection parity.
3. Run extensionLobeReference + bdptGlossyLightSubpath + restirPtSpecialtyReference tests.
4. Promote renderer-fidelity-matrix rows when unit/oracle tests pass (GPU furnace deferred).

### Tests (run exactly in this order)
```bash
cd packages/pt-webgpu && npx vitest run
npm run shader-gate
```

### Done when (ALL boxes required before next task)
- [ ] All integrator audit rows ✅ or documented approximate.

---
## RT100-GLTF-PICK

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | IMP |
| Lane | engine |
| Wave | W008 (#8) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | FP-04, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Arbitrary glTF Phase 4: pickBackend must use feature report not triangle-count alone (road-to-100 §4 + createEngineScale footgun).

### Files to modify (exact paths)
1. `packages/engine/src/createEngineScale.ts`
2. `packages/gltf-adapter/src/featureReport.ts`
3. `packages/engine/src/gltf.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100.md Phase 4 and trap table pickBackend row.
2. Wire rankGltfBackends / evaluateGltfBackendCompatibility into createEngine preference path.
3. Add test: textured hero asset must not auto-route to walkaround when PT supports features.

### Tests (run exactly in this order)
```bash
cd packages/engine && npx vitest run
cd packages/gltf-adapter && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] pickBackend uses compatibility report for glTF assets.

---
## RT100-EMISSIVE-PDF

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | WH-012, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
B4 tail: full energy-weighted emissive texel alias/PDF with forward-hit MIS parity (road-to-100 — not just area-PDF).

### Files to modify (exact paths)
1. `packages/shared-samplers/src/meshAreaLights.ts`
2. `packages/pt-webgpu/src/scene/meshAreaLights.ts`
3. `packages/pt-webgl2/src/scene/meshAreaLights.ts`
4. `packages/walkaround-hybrid/src/restir/emitterList.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100 B4 Done= tail about texel alias/PDF.
2. Implement alias table for emissive texel importance on CPU packers.
3. Ensure forward NEE PDF matches forward-hit MIS weight.
4. Extend meshAreaMis.test.ts across backends.

### Tests (run exactly in this order)
```bash
npm test --workspaces --if-present
```

### Done when (ALL boxes required before next task)
- [ ] Alias/PDF path live on all three backends.
- [ ] meshAreaMis parity tests green.

---
## RT100-A9-BDPT

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-PTWG-MAT |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] BDPT radiometric/material-furnace oracle for light-subpath connections. See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/radiometric-ab/`

### Steps (execute in order — do not skip, do not improvise)
1. Run radiometric oracle vs forward-traced reference.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] BDPT radiometric oracle green.
- [ ] Deferred — no agent dispatch.

---
## RT100-PTWG-FURNACE

| Field | Value |
|-------|-------|
| Phase | 9 |
| Priority | P1 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-PTWG-MAT |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] PTWG approximate ledger rows: material-furnace + reference-render promotion. See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/reference-renders/`

### Steps (execute in order — do not skip, do not improvise)
1. Furnace promotion for approximate PTWG material rows.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] Furnace promotion report.
- [ ] Deferred — no agent dispatch.

---
## RT100-5A-GOLDEN

| Field | Value |
|-------|-------|
| Phase | 10 |
| Priority | P1 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] Phase 5A: golden PNG + real Khronos asset render sweep. See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/reference-assets/gltf/`

### Steps (execute in order — do not skip, do not improvise)
1. Render 64spp per fixture; compare golden PNG.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] Golden renders for Khronos set.
- [ ] Deferred — no agent dispatch.

---
## RT100-5C-GPU-MUT

| Field | Value |
|-------|-------|
| Phase | 10 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V28-000 |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] Phase 5C: GPU mutation matrix observability on wsl-gpu. See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/benchmark-runner/`

### Steps (execute in order — do not skip, do not improvise)
1. Observe real GPU buffers on mutation scenarios.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] GPU mutation matrix report.
- [ ] Deferred — no agent dispatch.

---
## RT100-5D-DOC

| Field | Value |
|-------|-------|
| Phase | 10 |
| Priority | P2 |
| Disposition | DOC |
| Lane | repo-root |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
Phase 5D documentation sync: fidelity matrix, items_to_fix §H, road-to-100 stale addenda (H1–H5), READMEs cite ledger (road-to-100 C5).

### Files to modify (exact paths)
1. `plan/renderer-fidelity-matrix.md`
2. `plan/road-to-100.md`
3. `items_to_fix.md`
4. `README.md`
5. `packages/*/README.md`

### Steps (execute in order — do not skip, do not improvise)
1. Remove deleted pt-webgl column from fidelity matrix; ensure pt-webgl2 column accurate.
2. Strike/reconcile stale road-to-100 addendum bullets (e.g. H1–H5 inert — closed in items_to_fix).
3. Close or strike items_to_fix §H entries verified fixed.
4. README maturity claims cite BACKEND_PROMISE_LEDGER not prose.

### Tests (run exactly in this order)
```bash
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] Doc sync checklist complete.
- [ ] No stale OPEN claims contradicting items_to_fix.

---
## RT100-GATE-FULL

| Field | Value |
|-------|-------|
| Phase | 10 |
| Priority | P2 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-5A-GOLDEN |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] Phase 5E: full-tier behavioral gate + walkaround glTF render lanes. See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/behavioral-gate/gate.mjs`

### Steps (execute in order — do not skip, do not improvise)
1. Run glTF fixtures at full tier when adapter allows.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] Full-tier gate path documented or implemented.
- [ ] Deferred — no agent dispatch.

---
## RT100-A6-DECIDE

| Field | Value |
|-------|-------|
| Phase | 11 |
| Priority | P1 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-V19-GRIS |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] NRC default-on tier decision (requires GPU quality A/B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `plan/road-to-100.md`

### Steps (execute in order — do not skip, do not improvise)
1. Run NRC quality A/B vs DDGI reference.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] Decision recorded.
- [ ] Deferred — no agent dispatch.

---
## RT100-A10-WEIGHTS

| Field | Value |
|-------|-------|
| Phase | 11 |
| Priority | P1 |
| Disposition | SKIP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | (none) |
| Blocks | (none) |

### Problem
[DEFERRED — code-first] Production neural checkpoint decision (requires quality A/B). See plan/VALIDATION-DEFERRED.md.

### Files to modify (exact paths)
1. `tools/neural-denoiser-training/`

### Steps (execute in order — do not skip, do not improvise)
1. Quality A/B vs starter-v1 checkpoint.
2. No work in code-first campaign — task deferred to validation sprint.

### Tests (run exactly in this order)
```bash
# deferred
```

### Done when (ALL boxes required before next task)
- [ ] Production checkpoint decision made.
- [ ] Deferred — no agent dispatch.

---
## RT100-LD-SAMPLING-01

| Field | Value |
|-------|-------|
| Phase | 12 |
| Priority | P2 |
| Disposition | IMP |
| Lane | shared-samplers |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | RT100-PTWG-MAT, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
F1/LD-SAMPLING-01: Owen-Sobol or PMJ02 + blue-noise screen scramble in shared-samplers; pt-webgpu + pt-webgl2 integration.

### Files to modify (exact paths)
1. `packages/shared-samplers/src/`
2. `packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts`
3. `packages/pt-webgl2/src/glsl/`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100.md F1 section.
2. Generate LD tables CPU-side; upload as textures/buffers.
3. Per-dimension assignment audit (bounce/lobe/light).
4. Revive or replace dead pt-webgl2 RANDOM_TYPE branches.
5. GPU RMSE A/B deferred — see plan/VALIDATION-DEFERRED.md.

### Tests (run exactly in this order)
```bash
cd packages/shared-samplers && npx vitest run
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] LD sampling integrated both PT backends.

---
## RT100-WBVH-01

| Field | Value |
|-------|-------|
| Phase | 12 |
| Priority | P2 |
| Disposition | IMP |
| Lane | — |
| Wave | — |
| Parallel wave | no |
| Agent cap | 25 |
| Depends on | RT100-PTGL-MUT |
| Blocks | (none) |

### Problem
F2/WBVH-01: compressed wide BVH (CWBVH) opt-in builder + traversal behind capability flag.

### Files to modify (exact paths)
1. `packages/shared-bvh/src/`
2. `packages/shared-bvh/wgsl/`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100.md F2 section.
2. Implement CWBVH build + WGSL traversal in shared-bvh.
3. CPU brute-force oracle vs binary BVH in package vitest (wsl-gpu oracle deferred).
4. Per-backend opt-in until parity proven.

### Tests (run exactly in this order)
```bash
cd packages/shared-bvh && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] CWBVH behind capability flag.
- [ ] Oracle parity on test scenes.

---
## RT100-F3-DENO-AUTO

| Field | Value |
|-------|-------|
| Phase | 12 |
| Priority | P3 |
| Disposition | IMP |
| Lane | engine |
| Wave | W016 (#16) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | RT100-A10-WEIGHTS, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
F3: denoiser:auto default when weights resolve; turnkey OIDN/neural without host wiring.

### Files to modify (exact paths)
1. `packages/engine/src/createEngine.ts`
2. `packages/walkaround-hybrid/src/HybridEngineOptions.ts`
3. `packages/shared-denoisers/`

### Steps (execute in order — do not skip, do not improvise)
1. Read road-to-100 F3 section.
2. Add denoiser:auto union value.
3. Resolve bundled or downloadable weights at engine construction.
4. Clear error when assets missing.

### Tests (run exactly in this order)
```bash
cd packages/engine && npx vitest run
```

### Done when (ALL boxes required before next task)
- [ ] denoiser:auto documented and functional when weights present.

---
## RT100-F4-WAVEFRONT

| Field | Value |
|-------|-------|
| Phase | 12 |
| Priority | P3 |
| Disposition | ACC |
| Lane | repo-root |
| Wave | W017 (#17) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | RT100-LD-SAMPLING-01, __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
F4: wavefront PT rearchitecture — profile-gated research item, not arbitrary-glTF blocker (road-to-100 post-100).

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts`

### Steps (execute in order — do not skip, do not improvise)
1. Run divergence profiling on reference scenes before scheduling.
2. Document gate criteria in plan/road-to-100.md or archive.
3. Do not implement unless profiling justifies; mark ACC in roadmap.

### Tests (run exactly in this order)
```bash
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] F4 decision documented as deferred or scoped.

---
## RT100-F5-VOLUMES

| Field | Value |
|-------|-------|
| Phase | 12 |
| Priority | P3 |
| Disposition | ACC |
| Lane | core |
| Wave | W018 (#18) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
F5: heterogeneous volumes (null-collision delta tracking) — product-gated; stained-glass may never need (road-to-100 post-100).

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `packages/core/src/scene/`

### Steps (execute in order — do not skip, do not improvise)
1. Confirm product scope with user before contract extension.
2. If in scope: add AnalyticShape/Material.extensions volume primitive first.
3. Else: document permanent unsupported + planner routing.

### Tests (run exactly in this order)
```bash
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] F5 scope decision recorded.

---
## RT100-F-BRIDGE

| Field | Value |
|-------|-------|
| Phase | 12 |
| Priority | P3 |
| Disposition | DOC |
| Lane | repo-root |
| Wave | W019 (#19) |
| Parallel wave | yes |
| Agent cap | 25 |
| Depends on | __BOOTSTRAP_P0__ |
| Blocks | (none) |

### Problem
F-BRIDGE: experimental no-hardware-RT bridge levers — track as research backlog, not 100% blockers (road-to-100 §F-BRIDGE table).

### Files to modify (exact paths)
1. `plan/road-to-100.md`
2. `plan/roadmap.md`

### Steps (execute in order — do not skip, do not improvise)
1. Ensure F-BRIDGE table remains in road-to-100 with feasibility notes.
2. Cross-link active performance track (LD-SAMPLING, WBVH) vs bridge items.
3. No implementation unless promoted by user.

### Tests (run exactly in this order)
```bash
npm run typecheck
```

### Done when (ALL boxes required before next task)
- [ ] F-BRIDGE backlog visible and separated from Phase 0–6 closure.

---
