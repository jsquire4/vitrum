# Implementation Plan — 2026-06-10 complexity-sweep remediation (full scope)

**Repo:** `/home/jsquire4/projects/vitrum` · branch `main` · clean at `4fefb40`
**Source findings:** `~/.claude/projects/-home-jsquire4-projects-vitrum/memory/in-flight-sweep-2026-06-10.md` (D1–D13, I1–I5, dead-code, reminisce)
**Scope:** FULL — all 7 themes (A layout SSOT · B god-file decomp · C facade hardening · D WGSL dedup · E subsystem facades · F extensibility tables · G dead code/hygiene). Nothing deferred. Behavior-preserving throughout (one justified exception: D10.9, isolated with A/B capture).

## Pre-flight facts (verified by code-read during planning)

- `BVH_NODE_FLOATS` / `BVH_NODE_STRIDE` do **not** yet exist in `shared-bvh` — theme A genuinely creates them.
- Both gate scripts present: `npm run shader-gate` (51 WGSL via naga) + `npm run shader-gate:glsl` (pt-webgl2).
- `knip.json` exists; tools/** entry-pattern gap is real (G fixes knip.json, does not delete the scripts).
- 18 wsl-gpu oracle scripts hardcode `walkaround-hybrid/src/{restir,rc,ddgi}` absolute import paths (list in Global rule 4).

---

## Global rules (apply to EVERY task — non-negotiable)

1. **Verify-first.** Task step 1 is always: READ the cited file:lines and confirm the finding reproduces. If it does not reproduce, record it as a **false positive** in the round log and skip — do not "fix" a phantom. A grep hit is a pointer, not a finding.
2. **Behavior-preserving.** No render-output change. Each round adds/keeps: byte-identity composed-WGSL goldens, UBO goldens, characterization tests on extracted code. Theme A adds NEW pin tests (field-order / sentinel round-trip / stride-constant parity).
3. **WGSL gate.** Any `.wgsl.ts` change → `npm run shader-gate`; any pt-webgl2 GLSL change → `npm run shader-gate:glsl`. **UBO-leak lesson (d0ef37b):** never extract a WGSL fn that reads a UBO field — parameterize the read (pass the value in), or the field leaks into every includer and naga rejects. The pre-push T1 GPU smoke is the final arbiter.
4. **Oracle repoint.** Renaming/moving ANY `walkaround-hybrid/src/{restir,rc,ddgi}` production module → grep `~/projects/wsl-gpu/scripts` for the old path and repoint the oracles + render worker **in the same task**. Repoint-sensitive set (18 scripts): `v28-h15-uv-seam`, `ddgi-white-bounce-ab`, `v28-h32-tlas-glass-shadow`, `ddgi-uniform-energy-octattrib`, `ddgi-sh-compile-check`, `ddgi-indirect-pi-ab`, `ddgi-receiver-perf`, `v27-octahedral-isolate`, `capture-ftlas1-refs`, `ddgi-core-bvh-ab`, `ddgi-visibility-seam-ab`, `ddgi-bvh-bruteforce-ab`, `restir-tlas-bvh-bruteforce-ab`, `rc-merged-bvh-bruteforce-ab`, `v27-octahedral-asym`, `ddgi-uniform-energy-ab`, `ddgi-uniform-energy-diag`, `v27-octahedral-sign-ab`. **Tell a stale-import failure from a real one:** error line is `await import(...)` (module-not-found) = stale path, not a traversal regression.
5. **Dead-code = verify-true-deadness.** Classify each candidate (truly-dead / unwired-intended / oracle-referenced / codegen / public-API) with grep evidence. `@vitrum/core` public exports and `tools/**` scripts are NEVER dead.
6. **Parallelism discipline.** Agents on the same package must be **file-disjoint**. No interacting signature changes in parallel. **Implementation agents do NOT run `git commit`/`git push`/captures** — the orchestrator commits at round boundaries.
7. **Do NOT touch intentional divergences:** GGX/fresnel per-backend roughness floors (D9.14); à-trous L1 vs SVGF L2 edge-stop (D12.5); regir `rb_*` binding-forced dup (D5.6, comment-only); temporalGiCommon mid-file-anchor (D5.11, comment-only); adjointPass/restirPtHybridShift microfacet copies (D9.14, mirror-comments only); `MATERIAL_VEC4_STRIDE` pt-webgpu-local divergence (I2.6).
8. **Per-round mechanical gates:** `npm run typecheck`, `npm test`, shader-gate(s). **T1 GPU smoke** at every push point (round boundary commits).
9. **gitnexus:** `gitnexus_impact({target, direction:"upstream"})` before editing any shared symbol; report HIGH/CRITICAL before proceeding. `gitnexus_detect_changes()` before each commit.
10. **Audit gate.** After each round: run `/audit` on changed files; fix arch issues (new god-files, mixed concerns, export sprawl) before the next round.

---

## Round dependency graph

```
R0 (dead-code + knip + hygiene, theme G) ─┐
                                          ├─> R1 (theme A: layout SSOT + pin tests)
                                          │        │
                                          │        ├─> R2 (theme D: WGSL dedup, order 4→5→3→1+2→7; 8b last)
                                          │        ├─> R3 (theme B: god-file decomp — SEQUENCED per file)
                                          │        └─> R4 (theme F: extensibility tables)
                                          │
                                          └─> R5 (theme C: engine↔backend facade — I1)   [needs R1 constants, R0 dead exports gone]
                                                   │
R2,R3,R4,R5 ──────────────────────────────────────┴─> R6 (theme E: subsystem facades — I5/I3/I2 seams)
                                                          │
                                                          └─> R7 (final sweep: residual hygiene, T1 + retrospective)
```

**Why this order:** R0 first — dead-code shrinks the surface every later round touches. R1 second — exported stride/layout constants + pin tests are consumed by B's extractions and D's WGSL interpolation. R2/R3/R4 parallel-capable across packages but R3 internally sequenced on shared god-files. D5 intra-order honored (4→5→3→1+2→7, 8b last). R5 after R1+R0. R6 last structural round (depends on B's extracted seams).

---

## R0 — Dead code, knip, low-risk hygiene (theme G)

| Task | Finding IDs | Files | Verify-first | Change | Gate | Parallel-with |
|------|-------------|-------|--------------|--------|------|---------------|
| G0.1 Fix knip false-positives | dead-code §FALSE POS | `knip.json` | Read knip.json; `npx knip` to see tools/** flagged | Add `tools/**` + gltfTypes + PipelineFrame*/orchestrator internal types to entry/ignore patterns | `npx knip` | G0.2,G0.3,G0.4 |
| G0.2 Delete truly-dead **exports** | dead-code §TRULY DEAD exports | engine `__testing`; wh `LIGHTING_OPTION_KEYS`,`enrichMeshVertexRangesWithMatrix`,`collectRectAreaLightEmitterTris`,`buildEmitterList`,`PPG_MAX_SPATIAL_CELLS`,`PPG_DEFAULT_SPATIAL_CELLS`; pt-webgl2 `BVH_LEAF_FLAG`,`BVH_DISTANCE_FUNCTIONS`; pt-webgpu `rgba16fBufferToRgbF32`,`LITE_{POINT,SPOT,RECT}_VEC4S`,`FRAME_PARAMS_F32_SLOTS`,`deviceErrorStubMethods`,`fillBdptLightPathCpu`,`lambertianDirectionalPdf`,`bdptLightLuminance`,`bdptHasEnvironmentEmitter`,`bdptEnvironmentPower`; shared-denoisers `RGBA32F_BPP` | For EACH: grep whole repo + `~/projects/wsl-gpu/scripts` + `tools/**`; confirm 0 non-def hits; classify | Delete export + dead body | typecheck, test | G0.1 |
| G0.3 Delete truly-dead **files** | dead-code §TRULY DEAD files; D10.6 | pt-webgl2 `glsl/fullscreenVert.ts`; 7 unimported pt-webgl2 GLSL `index.js` barrels | grep each filename across repo+tools+wsl-gpu = 0 importers | safe-rm each | typecheck, shader-gate:glsl | G0.1 |
| G0.4 Delete shim/orphan modules | D2.8, D6.12, D6.13, D2.2 | wh `pbrScalars.ts`; `ddgi/ddgiRestirBvh.ts`; `debug/pickPrimitive.ts`; `collectDDGILightsFromThreeRoot`+`ThreeObjectLike` → legacy/ | grep callers; repoint importers to direct source then delete shims | Repoint imports → delete shims; move Three-orphan to legacy/ | typecheck, test; **oracle-repoint check** | after G0.2 if symbol overlap |
| G0.5 Unused deps + unlisted dep | dead-code §UNUSED/UNLISTED | pt-webgl2 (drop `@vitrum/shared-denoisers`), root devDeps (`@typescript-eslint/*`, `@vitest/coverage-v8`), examples (`@vitejs/plugin-react`), tools/gpu-env (add playwright) | grep imports per dep before removal | Edit package.json deps | `npm install` clean, typecheck | last in R0 |

**Commit point** → typecheck + test + shader-gate(s) + knip + **T1 smoke**. `/audit`.

---

## R1 — Theme A: layout/stride single-source-of-truth + pin tests *(hard gate for R2/R3)*

| Task | Finding IDs | Files | Verify-first | Change | Gate |
|------|-------------|-------|--------------|--------|------|
| A1.1 Export BVH node/vertex/mat4 stride consts | D11.2, D11.3, I2.2 | shared-bvh new `strides.ts`; consumers: scenePack(5 sites), refitBvhBounds, validateBvhEncoding, tlas, pt-webgpu(6), sceneMutationRouter, pt-webgl2(3), dev | gitnexus_impact upstream per consumer; read each literal site | Add `BVH_NODE_FLOATS=8`, `VERTEX_STRIDE_F32`, `MAT4_STRIDE_F32`, `BVH_NODE_STRIDE_U32`; replace 12+ literals | **NEW stride-parity test**; typecheck, test, shader-gate |
| A1.2 stride-3 index reads parameterized | D11.4, I2.3 | `refitBvhBounds.ts:56-76`, `bvhCore.ts` | confirm hardcoded stride-3 | `BvhIndexStride` param; `collapseIndicesToStride3` in shared-bvh; **bvhCore = restir subtree → oracle-repoint** | NEW round-trip test; oracle rerun |
| A1.3 WGSL stride interpolation | D5.5, D5.12, D5.13, I2.4 | bvhIntersect (literal 60 ×3), tlasTraversal, lightTree.wgsl `${'16'}`, regir `REGIR_FLOATS_PER_SURVIVOR` ×2 | confirm literal-vs-const divergence risk | Template-interpolate from TS consts; **no UBO-read inlining** | shader-gate; byte-identity golden identical |
| A1.4 FrameParams / RestirPt / Sppm field-order pin tests | I4.3, I4.4, I4.5, D8.11 | pt-webgpu frameParamsLayout.generated, RestirPtParams, SppmStats | read generated table vs hand-WGSL struct order | Export `*_WGSL_FIELDS`; **NEW field-order test**; size asserts; slot<=128 load-time assert | NEW tests; typecheck |
| A1.5 UBO sentinel round-trip tests | I3.3, D3.17, D4.11, D12.11 | walkaroundUbo/uboUpdater/constants/PipelineFrameInputs; cbPrefill/resolve UBO; temporalAccum AccumUBO | read uboUpdater offset-by-comment layout | Layout table; **NEW sentinel round-trip test per field**; `temporalAccumBindings.ts`; CB_PREFILL co-located in uboLayouts | NEW tests; shader-gate |
| A1.6 Stride alias cleanups | I2.5, D11.6, D11.8 | `DDGI_MATERIAL_ENTRY_FLOATS` alias+test; sceneAabbFromBvh param; tlas stack 64-vs-60 consts | confirm alias pure dup; over-narrow type | Delete alias+fix test; accept `ArrayBuffer\|Float32Array`; named stack consts + divergence comment | typecheck, test |

**Commit point** → all NEW pin tests green + full gates + **T1 smoke**. `/audit`. R2/R3 do not start before this merges.

---

## R2 — Theme D: WGSL dedup *(D5 order 4→5→3→1+2→7; 8b last)*

Every task: confirm dup by reading both sites; parameterize any UBO read; `npm run shader-gate` per task; byte-identity golden identical.

| Task | Finding IDs | Change | Notes |
|------|-------------|--------|-------|
| D-a (4) clampCoord shared | D5.4 | extract helper module (cbPrefill, resolve) | |
| D-b (5) regir stride interpolation | D5.5 | TS-interpolate REGIR_FLOATS_PER_SURVIVOR (coordinates w/ A1.3) | |
| D-c (3) reservoirGi finalise merge | D5.3 | single fn with `gris: bool` arg (flag passed as arg, NOT UBO read) | |
| D-d (1+2) DDGIGridUBO + sampleDDGIAtPoint dedup | D5.1, D5.2 | new `ddgiGridUbo.wgsl.ts` module; sampleDDGIAtPoint → ddgiSample | HIGH drift risk (churn 56/25/11); GI subtree → oracle check; sequential, single agent |
| D-e (7) surfaceTextures tinted-leaf | D5.7 | extract `_bvhTintedLeaf` | after D-d; churn 13, careful |
| D-f (8b) shade lo_* helpers | D5.8 | extract 8 lo_* helpers + cross-ref guard for unused @group(3) compat decls | **after D-d lands** |
| D-g pt-webgpu WGSL dedups | D9.1,2,3,4,11,13,17 | computeAnisotropicAxes; sampleMaterialLayer stamp; buildShadingTangentFrame; mneeChainFdJacobian4x4; concentricDiscSample→kernelCore; rotateY→connectCore; bdptLightSubpath reformat | parallel (different package) |
| D-h harness/WGSL separation | D9.5,6,7; D7 WGSL | mneeNewton/restirPtShift/HybridShift → `.harness.wgsl.ts` siblings; sppm host consts → `sppmParams.ts`; `mlpParamsStructWgsl` | move-only; confirm no production import of harness |
| D-i dead WGSL + lite mismatch | D9.9, D9.12 | delete dead `sppmGather`; audit kernelLite `photonMapContribution` signature vs causticLite stub + lite-composition gate check | |
| D-j doc/comment-only | D5.6,9,10,11; D9.14,15,16 | regir rb_* sync note; gtao PI note; motionVectors docblock; temporalGiCommon comment; adjoint/hybridShift mirror-comments; restirPtCompose body-found assert; intersection.wgsl anchor → named const exported from shared-bvh | comment/assert only — do NOT merge intentional dups |

Also: D9.8/I4.1 `SPPM_GROUP4_BINDINGS_WGSL` → `SPPM_GROUP3_BINDINGS_WGSL` rename + JSDoc fix; D9.10 caustic receiver-gate helpers.

**Commit point** → shader-gates + goldens + full gates + **T1 smoke**. `/audit`.

---

## R3 — Theme B: god-file decomposition *(same-file tasks SEQUENCED, one agent per chain)*

### B-chain-1: HybridEngine.ts (one agent, sequential)
1. Option-parsing 132-482 → `HybridEngineConfig.ts` (D2.1/D3.6) + `VALID_DENOISERS as const` (D3.8).
2. `readRgba16fWalkaround` → `util/gpuReadback.ts` (D3.7).
3. GI-state export/import → `HybridEngineGIState.ts` (D2.1).
4. `syncDdgiLightsFromScene()` dedup vs Lifecycle (D2.4) — DDGI subtree → oracle check.
5. `_buildFrameDeps` rebuild-key → renderFrame boundary (D2.5).
6. Delete 4 test-seam getters, retarget tests (D2.7).
7. `applyNormalTransformAndUpload` (D2.3).
8. `buildRcFrameInputs` (D2.9).
9. `getMemoryBreakdown()` on pipeline (I3.1); `ATROUS_*_SIGMAS` → pipeline/constants.ts (I3.2).

### B-chain-2: WalkaroundGPUPipeline.ts (one agent, AFTER chain-1)
1. `initialize()` → `_initResources/_initPasses/_initSubsystems` (D3.1/D4.4).
2. `renderFrame` → `buildFrameContext/dispatchPasses/tickSubsystemTraining` (D3.2); single checkerboardState record (D4.5).
3. `captureOutputFrame` → `FrameCaptureHelper` (D3.3/D4.2).
4. `PipelineFrame*` interfaces → `pipelineFrameInputs.ts` (D3.5).
5. `welfordPing` getter on Denoiser iface, drop instanceof (D3.4); typed `PassOwnedUboRef` (D4.3).

### B-chain-3: gpuResources.ts (one agent, pt-webgpu)
1. `ReservoirResources/SppmResources/PresentResources` sub-objects (D8.1).
2. `#makeGroup0LayoutEntries`/`#makeGroup0BindGroupEntries` — dedup the ×3 layout, historical F1 site (D8.2/I4.7).
3. Hoist binding-helper aliases (D8.3).

### B-parallel (file-disjoint)
| Task | Finding IDs | Scope |
|------|-------------|-------|
| B-p1 pt-webgpu index.renderFrame | D8.4,5,10,15 | `#ensurePerFrameResources`/`#encodePathTracePasses`; `isReadyToRender()`; `AdjointPass` class; `#computeSppmSceneStats` — after B-chain-3 |
| B-p2 pt-webgl2 glResources | D10.1,4,8 | BdptSubpathBuilder/AccumBlender/PresentPass; RENDER_MAIN named sections; per-section material packers — shader-gate:glsl |
| B-p3 wh denoisers | D4.6,7,8,9 | svgfReal builders + `_selectPingPong`; oidnFinal 3-stage + rgba helpers → shared-denoisers; neural `_tensorBuffers` |
| B-p4 wh-gi extractions | D6.3,5,6,8 | probeUpdatePass init() split; bvhSceneHelpers split + THREE→legacy; bvhCore Map-keyed match; probeUpdateRays sub-templates — **all oracle-repoint** |
| B-p5 wh-neural | D7.1,2,3,4,5,6,9 | cpuForwardBackward; main() split; HashGridTableTrainer in LIVENESS; group option on nrcQueryWgsl; disposed-buffer typing; unpackRecords(); patchReLUInPlaceAliasing; packAdamUbo shared (D7.8) |
| B-p6 pt-webgpu scene host | D8.6,8,9 | readonly/mutable `GpuSceneHandles` split; `synthesizeImplicitEmitters`; hoist write + `applyScenePackCounts` — after B-chain-3 |
| B-p7 shared-bvh/periphery | D11.1,5,7,10; D13.1,4,5,6,7,8 | scenePack invertMat4→mathUtils + splicePack.ts; emitterClassify.ts; pickPrimitiveCpu mathUtils; worldSpaceMerge factory ids; cascadeDispatch split; gltfToScene/convertMaterial/accessors splits; dev/vanilla per-overlay + bvhStats shared (D13.10) |

**Commit per chain/task (orchestrator).** Gates per commit + oracle reruns where GI/restir/rc touched. **T1 smoke at end of round.** `/audit`.

---

## R4 — Theme F: extensibility tables *(parallel, file-disjoint)*

| Task | Finding IDs | Change |
|------|-------------|--------|
| F1 pipeline spec tables | D3.10,11,12,15,16; D4.10,11,12,13 | PASS_LABELS array; destroyQueue at alloc; PPGFrameResources factory; PIPELINE_SPECS loop; BGLCache record type; GTAO dispatchSingleBindGroup; passOrder doc + passEntry; denoiser lifecycle typing |
| F2 pt-webgpu registry | D8.7,12,13,14 | SceneBufferRegistry; RPT bind-group invariant doc+assert; buildLightTreeInputForScene required params; lite-warnings from capability diff |
| F3 pt-webgl2 tables | D10.2,3,5,7,9,10,11,12,13,14 | SCENE_TEXTURE_BINDINGS typed table; FrameUniforms exhaustive mapping; uniform manifest; uv1 merge→shared-bvh; **D10.9 per-triangle materialIndex (render-affecting — isolated commit + before/after reference A/B captured by orchestrator)**; lightsTexture slot writer + k===24 assert; RGBA8 present target; explicit handle channels; allocGlTexture shared; field grouping + #resolveRegime |
| F4 ledger/dispose tables | D1.3,4 | ALL_MUTATIONS_FALLBACK_REBUILD + COMMON_METHOD_PROMISES spreads; table-ify 6 forwarding blocks via new DisposedBehavior rows |
| F5 shared-denoisers | D12.1,4,7,10 + hygiene D12.2,3,8,9 | uploadRgbAsRgba32fPacked; hdrBilateral defineUbo; direct pipeline cache; MAX_DEVICE_ACQUIRE_RETRIES; comment/style fixes |
| F6 periphery | D13.2,3,9,11,12,13 | CascadeUniformInputs object; rcLightEval.wgsl factor; frame-monitor onFrame preference; halfBitsToFloat dep decision; skyParams yScale/zBias params; came reserved-slot doc |

**Commit point** → full gates + D10.9 A/B capture + **T1 smoke**. `/audit`.

---

## R5 — Theme C: engine↔backend facade hardening (I1) *(after R0+R1)*

| Task | Finding IDs | Change |
|------|-------------|--------|
| C1 BackendConstructor dispatch table | I1.1, D1.2, D1.6 | Split createEngine.ts / backends/walkaround.ts / backends/ptWebgpu.ts; dispatch table; `advanced`→Record per-backend validated; `withEngineLifetime` helper |
| C2 attach lifecycle machine | D1.1, D1.5 | EngineAutoRecreateMachine + buildEngineFromOpts (dedup createEngine spread); QualityOption named type; unify onError/onEngineError (phase param) — after C1 |
| C3 GIState → core capability | I1.2, D1.7 | GIStatePersistable/GIStateSnapshot generic in core; LightingOptions re-export note; fix Engine.updateLighting JSDoc |
| C4 method-promise conformance | I1.3, I1.4, I1.5 | Mapped-type conformance proxies↔ledger; add shipped optional methods to BackendMethodPromises; EngineFactory-typed PtWebgl2ModuleLike; **createProgressiveEngine forward onError** (verify gap still live first) |

gitnexus_impact upstream on createEngine/attachVitrum/OPTIONAL_METHOD_PROXIES (HIGH blast radius — report before editing).

**Commit point** → full gates + **T1 smoke**. `/audit`.

---

## R6 — Theme E: subsystem facades & shims *(after R3)*

| Task | Finding IDs | Change |
|------|-------------|--------|
| E1 type/ownership seams | I5.1, I5.2 | WgslModule type → neutral path; NRC owns getNrcBindGroupLayout |
| E2 DDGI facade | I5.3 | exportAtlasData/importAtlasData + gridParams on DDGI facade; demote pass/probeGrid getters — **oracle-repoint** |
| E3 RC light layout test | I5.4 | layout test / exported constants for packRCLights↔RCLightBuffer |
| E4 bind-state | D3.14, D3.13 | DDGIBindingState rename/split; audit + remove inert placeholder slots 0-4 (verify unreferenced first) |
| E5 misc shims | D6.4,7,11,14; D2.6 | DispatchBindGroupCache; packingHelpers @deprecated + repackBVHMaterialRange→bvhCore; aabbHelpers; GpuSkinning cpuFallback; rc/packingHelpers.ts + delete tombstone setScene; probeGrid dual-AABB removal (D6.10, verify dead) |
| E6 doc-only seam notes | I3.4,5,6; I4.6; I5.5; D12.5,6,9; D11.9,11 | group-1 comment 10→11; PipelineSubsystem shapes doc; svgf edge-stop self-doc; bmfr MUST-MATCH comments; symbol-refs not line-refs; tlasSceneHitTraversal JSDoc; FULL_SPHERE_CONE export + gate _powerPrefixSumDebug |

**Commit point** → full gates + oracle rerun (E2) + **T1 smoke**. `/audit`.

---

## R7 — Final sweep & retrospective

- Residual hygiene; full-repo `/audit`.
- Full gate suite: typecheck + test + shader-gate + shader-gate:glsl + knip + **T1 GPU smoke** (lavapipe + dzn).
- Confirm every reference render unchanged except justified D10.9 A/B.
- `/retrospective`; memory entry. Push only on user instruction.

---

## Test strategy

| Layer | What | New for this sweep |
|-------|------|--------------------|
| Byte-identity composed-WGSL goldens | extracted WGSL composes identically | extend to ddgiGridUbo/clampCoord/tinted-leaf modules (R2) |
| UBO goldens | packed bytes unchanged | extend to temporalAccum/cbPrefill/resolve (R1) |
| Characterization tests | extracted TS fns reproduce prior outputs | one per B/F extraction |
| **Theme-A pin tests (NEW)** | stride-const == old literal; sentinel write→pack→read round-trip; generated-slot-order == WGSL-field-order; size asserts | R1 — the core safety net; TDD: write against current literals first |
| Cross-backend parity | tonemap GLSL vs WGSL numeric parity (I2.8); MATERIAL_VEC4_STRIDE parity (I4.2) | guards intentional regime-split copies |
| wsl-gpu oracles | CPU brute-force RC/TLAS/DDGI == GPU | repoint+rerun on every GI/restir/rc move |
| T1 GPU smoke | compiles the runtime pass graph — only check that catches UBO-leak naga rejects | every commit/round boundary |
| Reference-render A/B | D10.9 only | orchestrator captures before/after |

## Risk register

| # | Risk | L | Mitigation |
|---|------|---|------------|
| 1 | WGSL extraction leaks UBO field → naga reject (d0ef37b class); vitest goldens stay green | HIGH | Parameterize every UBO read; shader-gate per task; T1 at boundaries |
| 2 | Stale oracle import mislabeled as traversal regression | HIGH | Repoint rule in-task; `await import()` failure = stale, match-rate failure = real |
| 3 | Byte-identity green while both sides share a bug | MED | Pin tests assert against independent literals; T1 carries independent CPU oracles |
| 4 | Parallel agents collide on same package | MED | Sequential single-agent god-file chains; file-disjoint parallel; agents never git-mutate |
| 5 | D10.9 changes render output | MED (intended) | Isolated commit; before/after A/B; accept only if it fixes last-writer-wins + visually justified |
| 6 | Deleting unwired-intended/oracle-referenced export | MED | verify-true-deadness; public-core & tools never dead |
| 7 | Constant substitution flips a stride at 1 of 12+ sites | LOW-MED | stride-parity pin test per const; gitnexus_impact; typecheck |
| 8 | "Fixing" an intentional divergence | LOW | Rule 7 list; comment-only at those sites |
| 9 | Phantom agent finding | MED | verify-first step 1 on every task; record false positives |

## Agent dispatch structure

All impl agents Sonnet, file-disjoint, no git/captures. Orchestrator commits at boundaries, runs D10.9 captures, T1 smoke, /audit, gitnexus_detect_changes.

| Round | Parallel agents | Sequential |
|-------|----------------|------------|
| R0 | 3 (knip / dead exports+files / shims); deps last | — |
| R1 | 2 (strides shared-bvh / UBO+frameparams pin tests); A1.3 after A1.1 | consts → tests |
| R2 | 3 (D5 wh-shaders chain / pt-webgpu WGSL / harness+doc) | D5 order 4→5→3→1+2→7, 8b last |
| R3 | 4 parallel (pt-webgl2 / denoisers / wh-gi / neural+shared+periphery) | 3 god-file chains: HybridEngine → Pipeline; gpuResources independent; B-p1/B-p6 after chain-3 |
| R4 | 4-5 (pipeline tables / pt-webgpu registry / pt-webgl2 [D10.9 isolated] / ledger+denoisers / periphery) | — |
| R5 | 2 (facade split / core+conformance); C2 after C1 | createEngine→attach |
| R6 | 3 (seam types+DDGI / RC+bindstate / shims+docs) | — |
| R7 | orchestrator only | — |

## Definition of done

- All D1–D13, I1–I5, dead-code findings fixed or recorded as verified false positives — nothing deferred.
- Intentional divergences untouched (comment-only where flagged).
- All gates green: typecheck, test, shader-gate, shader-gate:glsl, knip, T1 GPU smoke (lavapipe+dzn).
- All NEW theme-A pin tests present and green; goldens unchanged.
- Reference renders unchanged except justified D10.9 A/B.
- All 18 repoint-sensitive oracles green after any GI/restir/rc move.
- `/audit` clean after every round.
