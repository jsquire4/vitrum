# vitrum Complexity-Sweep — Implementation Plan (2026-06-02)

**Branch (to create):** `feat/complexity-sweep-2026-06-02` | **Base:** `main`
**Verdict:** Issues found — codebase well-decomposed (4 prior sweeps); **no new mandatory god-file rewrites**. Work = pervasive WGSL/cross-engine duplication + ~8 verified latent bugs + coupling/contract refinement + dead code + stale comments + 4 user-approved architecture forks.

## User-approved decisions (Phase 4 + fork walk)
- **Scope = FULL** + walk forks one-by-one; behavior-preserving tests mandatory.
- **Fork A:** move `walkaround-rc/src/bvhCompute.ts` → `walkaround-hybrid/src/rc/`; drop RC's `@vitrum/three-bindings` dep; KEEP `giReceiver`/`walkaroundDiffuseLighting` (intentional TSL public API).
- **Fork B:** move `solveSkin`/`combineSkinMatrices`/`mat3InverseTranspose` → `@vitrum/core`.
- **Fork C = Option 1:** interface hygiene + frame-struct grouping; **NO** HybridEngine field-storage rewrite (`_cfg` consolidation already done = Task 4.2).
- **Fork D:** all WGSL dedup via existing `composeWgsl requires:[]` + whole-function extraction + TS-`buildX` splicing; **no** net-new composer tooling.

## Stale-memory corrections (code beat memory this sweep)
- W2-C13 `defineUbo` migration is **DONE** (atrousVarianceBindings:65,161). Do not re-raise.
- HybridEngine `_cfg` consolidation is **DONE** (Task 4.2 / Theme A, HybridEngine.ts:583). Do not re-raise.

## Verified findings (orchestrator read the cited code)
octahedral `sign()` regression (ddgiSampleWgsl:93,122) · probeUpdatePass dispose omits 6 buffers (315-320 vs 587-608) · fusedMlpTrainer per-step UBO leak (311-326, .destroy only @261) · sampleSky byte-identical (connect/connectLite) · walkaround-rc THREE coupling (bvhCompute:39-48) · applyDDGIShading `_injectedMaterials` not reset (106, dispose 70-77) · prevProjMatrix internal field dead (decl 378, fwd 519, never packed) · gtao dual-override (HybridEngineTuning:184 vs 132).

---

## Global gates (every phase)
`npm run typecheck` clean + `npm test` green before commit. Per-theme commits. **Radiometric rule:** a WGSL dedup is *structural* (no GPU A/B) only if the **composed output bytes are identical**; if they differ → V-numbered entry in `HARDWARE-VALIDATION-NEEDS.md` + real-GPU A/B owed. Native tools over bash; `safe-rm` for any file deletion; `package.json` edits via Edit (protected file).

---

## Phase 0 — Git hygiene + before-state
- Branch `feat/complexity-sweep-2026-06-02` off clean `main`. Baseline `typecheck`+`test` (record per-workspace pass counts so characterization additions are visibly additive). Note current highest V-number in `HARDWARE-VALIDATION-NEEDS.md`.

## Phase 1 — Shared-package foundations (unblocks P2/P3)
- **1A — GGX → shared-samplers/bsdfPrimitives `[RADIOMETRIC → GPU A/B]`.** Hoist canonical `distributionGGX`/`geometrySchlickGGX`/`geometrySmith` **with the `max(·,1e-6)` defensive floors** folded in. walkaround `ggxBrdf.wgsl.ts` keeps `evalGGX`, pulls the 3 fns via `requires:['bsdfPrimitives']`, deletes locals (composed bytes change → floor added). pt-webgpu `material.wgsl.ts` deletes `ggxD`/`smithG1`, renames call sites (same floor → effectively structural via rename). One combined GPU A/B for both engines.
- **1B — fresnelSchlick unify `[RADIOMETRIC → GPU A/B]` (REFINED).** pt-webgpu `common.wgsl.ts` pulls `BSDF_PRIMITIVES_WGSL`; delete pt-webgpu `material.wgsl.ts:310-315` local. **REFINEMENT: preserve the defensive `clamp(1-cosθ,0,1)`** — either add the clamp to the shared form (consistent with 1A's defensive-floor choice) or prove by call-site analysis that all callers pass cosθ∈[0,1] (clamp is a no-op). Do NOT silently drop the guard. If shared gains the clamp, walkaround's composed output may change too → fold into the same A/B. Keep `frDielectric` local.
- **1C — pcgHashToF32 → shared-samplers/wgsl/pcg `[STRUCTURAL]`.** Add the exact body from walkaround-rc probeRayCast.wgsl:133; RC imports it, deletes local; snapshot composed RC shader byte-identical.
- **1D — regir helper de-dup `[STRUCTURAL, TS]`.** `export dist2ToAabb`(lightTree:362)+`nodeImportance`(380); regir.ts:70-89 imports them, deletes copies (VERIFIED char-identical).
- **1E — invertMat4 shared home `[STRUCTURAL]`.** Single home in shared-bvh; pt-webgpu `math/mat4.ts:19` re-exports it.
- **1F — Fork B: skinSolver → core `[STRUCTURAL]`.** New `core/src/skinSolver.ts` + re-export; delete three-bindings copy; repoint 4 consumers (vitrumSceneToThree, three-bindings/index, GpuSkinningSubsystem, HybridEngine) to `@vitrum/core`. Keep all 19 skin tests.
- **1G — Fork A: RC bvhCompute → walkaround-hybrid `[STRUCTURAL]`.** Move file → `walkaround-hybrid/src/rc/bvhCompute.ts`; repoint HybridEngineRC import; drop `@vitrum/three-bindings` from walkaround-rc package.json + index re-exports; keep TSL files. One-line cross-ref comment on the intentional thickness-0.1 delta vs ddgi/probeUpdateMaterials.
- Parallel: {1A+1B} | 1C | 1D | 1E | 1F | 1G. Land 1F/1G + workspace typecheck before P2.

## Phase 2 — T1 WGSL dedup (consumes P1)
Every move: snapshot composed output before → move → assert byte-identical after (structural); if differs → radiometric V-entry.
- **2A pt-webgpu:** 2A.1 sampleSky→connectCore `[STRUCTURAL, verified]` · 2A.2 kernel core (generatePrimaryRay/projectToNdc/causticMode/RRResult/russianRoulette)→kernelCore `[verify byte-identical]` · 2A.3 caustic `gatherRadius` → `photonGatherRadius` FrameParams field, host default 0.35 (numerically identical; re-pin UBO offset golden).
- **2B walkaround-hybrid:** 2B.1 risGiNrc TS-splice from RIS_GI_WGSL · 2B.2 spatialGiCommon (sampleDiscPx+constants) · 2B.3 `finaliseGIReservoirW`→reservoirGi (dup 4×) · 2B.4 `refreshPhase0Cache`→reservoirGi · 2B.5 gtaoCommon (GTAOUniforms struct) · 2B.6 sampleBudget→`requires:['welfordTail']` · 2B.7 PPG (ppgTreeLayout module + octahedralCore requires + RESERVOIR_GI_STRIDE TS const) · 2B.8 neural CPU dedup (nrcOneBlob, eb*, heInit export, f16BitsToF32 import; harness CPU-forward ×3 = test-only, low priority). All `[STRUCTURAL]`, byte-identity goldens.
- **2C DDGI:** 2C.1 octahedral `sign()→select()` (ddgiSampleWgsl:93,122) `[RADIOMETRIC → GPU A/B]` — output changes at axis-aligned dirs only; unit-test the 6 axis dirs + composed snapshot.
- Parallel across 2A/2B/2C; within 2B all 8 independent. Do pt-webgpu P2 after P1 pt-webgpu (rebaseline goldens once).

## Phase 3 — T2 cross-engine dedup (all `[STRUCTURAL]`)
- **3A** OIDNDispatcherCore + shared types → shared-denoisers, parameterized by readback callback; both engines instantiate. + state-machine characterization test.
- **3B** scenePack: `resolveOneTransform` shared by packSceneFromCore + resolveInstanceTransforms; `packOneMeshLikePrimitive` returns `{slice,warnings}` always (kills re-derivation). Characterization on fixture w/ non-identity transforms.
- **3C** bdpt: pt-webgl bdptSceneEmittersCpu math → shared-samplers; pt-webgpu bdptConnectionMisFull re-exports 3 leaf fns (keep `assembleMergedConnectionPath` local).
- **3D** denoiser host: `makePerDevicePipelineCache<T>` (dup ×4); promote `buildAtrousVarianceAtrousBindGroup` to bindGroupBuilders; `checkShaderCompile` helper (dup ×3); `readRgba32fToRgb` in webGpuTextureUpload.
- Parallel: 3A | 3B | 3C | 3D.

## Phase 4 — T7 verified bugs (behavior-improving)
- **4A leaks `[no A/B]`:** 4A.1 probeUpdatePass dispose +6 destroys (VERIFIED) · 4A.2 fusedMlpTrainer hoist per-step UBOs to persistent fields (VERIFIED) · 4A.3 applyDDGIShading reset `_injectedMaterials` (const→let, VERIFIED) · 4A.4 ProbeUpdatePass `_initAttempted` guard. Each + dispose/reinit characterization.
- **4B `[STRUCTURAL]`:** remove the **internal** dead `prevProjMatrix` field (WalkaroundGPUPipeline:378) + forward (Orchestrator:519); **KEEP** the `FrameInput.prevProjMatrix` contract field (public + consumed in nrcGateBitIdentity.test:35). UBO goldens prove no packed-byte change.
- **4C:** 4C.1 unetArchitecture derive paramCount (lock 426075) · 4C.2 NeuralDenoiser.resize teardown window · 4C.3 NeuralDenoiser inline WGSL → modules (lower priority, structural golden).

## Phase 5 — T3 interface hygiene + frame-struct grouping (Fork C=Opt1)
NO field-storage rewrite. DO NOT rename test seams `_initSeq`/`_pendingTeardown`/`_initRunning`/`_disposed`.
- **5A** DDGI forwarding methods (setSunIntensityMultiplier/setGlassMixScale/getReadAtlasGPUTextures/gridParams); narrow `.pass`/`.probeGrid`.
- **5B** `BvhUpdateSink` interface (7 pipeline methods); type PrimitiveUpdateContext.pipeline to it.
- **5C** `getDebugTextures():PipelineDebugTextures`; narrow `frameResources` getter to internal.
- **5D** `PipelineSubsystem` interface {initialize,dispose,enabled}; 5 coordinators implement; **add ReGIRCoordinator.dispose** (real leak fix).
- **5E** regroup 31-field HybridEngineFrameDeps + 46-field PipelineFrameInputs → struct-of-structs. **Land LAST + ALONE**; UBO byte-identity goldens are the proof. Minor: add lightTree/regirBuild to BIND_GROUP_TABLE (parity); note RegisterPassesDeps thunks.

## Phase 6 — T6 contract hygiene (all `[STRUCTURAL]`, compile-dominated)
6A advanced union += `Partial<PTEngineWebGPUOptions>` (drop cast) · 6B `satisfies Partial<EngineCapabilities>` on promiseLedger · 6C `LightingUpdateOptions` type + close/document causticOptions index-sig · 6D move wgpuSupport setTimeout → gpuDetection · 6E move mergeWalkaroundTlasExtension → createEngine, drop createEngineScale backend dep · 6F CameraLike iface + capabilities.presentationMode check in vanilla.ts (drop THREE import). All independent.

## Phase 7 — T4 misc + T5 stale comments + T8 dead code
- **7A T4:** drop dead ThreeStdMat/ThreePhysMat exports; scene-lighting lightingState → Vec3 triple (drop THREE) + fix "both backends consume" comment; dev useKeyToggle (×3) + useDebugDevice (×2) hooks.
- **7B T5 (comments fixed in same commit as paired move):** sTree dTreeAccumulateFlux export + fix false "circular import" comment; createCommonFrameResources → pipeline/constants WALKAROUND_UBO_SIZE_BYTES + fix comment; delete 4 @deprecated probeUpdate consts (test-only, verify-then-delete); reword uboLayouts/transposedConv2d/svgfRealWebGPU/MotionVectorsPass-JSDoc/MaterialInspector-TODO/cameUniformUploader-layout.
- **7C T8 (verify-then-act; grep each before acting):** drop `export` on 18 dead exports + 9 dead types; remove 5 dead devDeps (engine/dev react-dom etc); add missing deps (@eslint/js, @webgpu/types root, playwright tools/gpu-env); **KEEP** false-positives (7 material.wgsl compose exports, 5 BDPT oracles, layout-contract consts, gpuSkinBvh negative oracle); 7C.5 fork example devDeps = **default KEEP** unless grep proves examples removed. File deletions (if any) via safe-rm.

## Phase 8 — Verification + HARDWARE-VALIDATION-NEEDS
- Full typecheck + test green; test count grew (additive goldens/characterization).
- Append 3 radiometric V-entries: **GGX defensive-floor unification** (1A), **pt-webgpu fresnelSchlick** (1B, if the guard-preserving form changes bytes), **DDGI octahedral sign(0) fix** (2C.1). Capturable in **2 GPU sessions** (pt-webgpu trace covers GGX+fresnel; walkaround DDGI scene with axis-aligned surfaces). caustic gatherRadius = awareness-only (numerically identical at default 0.35).
- Confirm all structural WGSL goldens + UBO goldens (uboUpdater, nrcGateBitIdentity, FrameParams, 416-byte WALKAROUND_UBO) green. Any unexpected golden diff → reclassify radiometric + add V-entry.
- Update CLAUDE.md "What's done" + memory with the landing summary (stale-context rule, same pass).

## Genuinely blocked: none.
Two judgment calls (not blocked): 7C.5 fork example devDeps → default keep; 4C.3 NeuralDenoiser WGSL extraction → lower priority but in-scope (structural golden), not deferred.
