# Implementation Plan — Complexity-Sweep Refactor (2026-05-30)

> Authoritative findings: `~/.claude/projects/-home-jsquire4-projects-vitrum/memory/in-flight-sweep-2026-05-30.md`
> Scope: all 12 themes (A–L). User has approved fixing **all** findings in one plan.

## Governing principles (apply to EVERY task)

1. **Behavior-preserving is non-negotiable.** ~39 unpushed, unit-pinned-but-NOT-GPU-validated radiometric commits sit underneath `main` (V1–V24, see `HARDWARE-VALIDATION-NEEDS.md`). No refactor here may perturb rendered output or radiometry. **Pin behavior BEFORE touching code.**
   - **WGSL dedup** → assert the *composed shader string* is **byte-identical** before vs after. The repo already does this style of check (`packages/walkaround-hybrid/__tests__/wgslCompose.test.ts`, `packages/pt-webgpu/src/__tests__/wgslContract.test.ts` + `wgslLiteContract.test.ts`). Snapshot the composed string into a golden, refactor, assert equality.
   - **TS dedup/restructure** → characterization/golden test (capture current output for representative inputs, assert unchanged across the refactor).
2. **Two genuine correctness fixes live in Theme L.** They DO change behavior intentionally and get their own targeted tests. Keep them isolated and clearly labeled.
3. **Fidelity is paramount — restructure, never remove.** Dead-code scan found ZERO truly-dead symbols. Only the explicitly-named cosmetic down-scopings and the two genuinely-orphaned items (`worldFromHalfPx_temporal` dead helper; stale docblocks) may be removed. `bilinearUpsample` is a wired extension point → **document, do not delete.**
4. **Mechanical gate after every task:** `npm run typecheck` (all workspaces) + `npm test` (vitest, all workspaces). A task is not "done" until both are green.
5. **Respect package boundaries.** The `three-gpu-pathtracer` fork boundary stays intact. Shared-package helpers land BEFORE their consumers adopt them.
6. **gitnexus impact** before editing any exported symbol with non-trivial blast radius (god-orchestrator splits especially).

---

## DECISIONS — RESOLVED (2026-05-30)

**D1 — Theme H tunables → ✅ NESTED `tuning?: Partial<Tunables>` namespace, type derived from the existing `Tunables` source.**
Confirmed by code-read: each knob is currently declared 3× (the `Tunables` interface + a `TUNABLE_DEFINITIONS` row [already keyed `K extends keyof Tunables`] + a redundant flat field on `HybridEngineOptions`). `readTunables()` already reads a `grouped` source + a flat fallback. Resolution:
- Add `tuning?: Partial<Tunables>` to `HybridEngineOptions`; **delete the ~25 hand-declared flat audit-knob fields** (their type now comes from `Tunables` — ONE source).
- `readTunables()` reads `opts.tuning?.[k] ?? def.default`. Fold the existing flat + ad-hoc `grouped` (`opts.gtao?.radiusPx`, `opts.caustic?.boost`) audit-knob reads into the `tuning` path so the hand-maintained `grouped` map shrinks. (Tuple-shaped accessors like `adaptiveSamplingThresholds[0/1]` may keep a thin mapping — handle case-by-case.)
- **This IS an API surface change** (callers move flat audit knobs under `tuning`). Pre-publish → zero migration cost. Headline options (`denoiser`, `qualityMode`, `rcEnabled`…) stay flat at top level. Update examples/tests that pass flat audit knobs.
- Behavior pin: characterization over the full knob matrix — every tunable must resolve to the same value via `tuning` as it did flat; defaults unchanged.

**D2 — Theme I `bilinearUpsample` → ✅ DOCUMENT as extension point** ("fully plumbed; no spec currently emits this"). Do not delete (constraint #3).

**D3 — Theme J BDPT-CPU oracle → ✅ INTRA-PACKAGE dedup only.** Collapse the ×3 in pt-webgpu and the ×N in pt-webgl within each package; leave the cross-backend mirror as two single-sourced copies (respects the "SceneHit vs IntersectionResult genuinely hard" caution). A shared-samplers hoist is a separate deliberate move, not this sweep.

**D4 — Theme A `_cfg` depth → ✅ INCREMENTAL.** Introduce `readonly _cfg: ParsedHybridEngineConfig`, migrate only the ~30 tunable-cluster members that hop 6 layers; leave genuine per-instance mutable runtime state as fields. Behind characterization tests.

---

## PHASING OVERVIEW (ordered by DEPENDENCY, not importance)

| Phase | Themes | Why here | Parallelizable? |
|------|--------|----------|-----------------|
| 0 | L (2 correctness), G (stale comments) | Cheap, low-risk, clears noise; L is isolated behavior change done first while context is fresh | L and G fully independent → 2 branches |
| 1 | F (shared-bvh helpers), K (shared WGSL frag + defineUbo), D (shared WGSL fragments) | Shared-package primitives must land before consumers adopt | F, K, D-fragments largely independent → 3 branches |
| 2 | E (pass base-helpers), C (WGSL whole-body dedup), J (oracle dedup intra-pkg) | Consume Phase-1 fragments; mid-size, mechanical | E, C, J independent → 3 branches |
| 3 | B (updatePrimitive cascade ×3), H (config ingestion) | B depends on stable backend shapes; H is the API-shape decision (D1) | B-per-backend ×3 independent; H separate |
| 4 | A, I (god-orchestrator splits) | Largest + riskiest; each behind characterization tests; sequence LAST | A-per-file and I-per-subsystem can parallelize but each needs its own char-test first |

After **each phase**: run `/audit` on changed files. Fix any God-file / mixed-concern / export-sprawl regressions before the next phase. These are gates, not suggestions.

---

## PHASE 0 — Correctness fixes + stale-comment sweep (cheap, clears noise)

### Task 0.1 — [Theme L, CORRECTNESS] dTree `u0`-reuse → oracle matches GPU
- **Change:** In `packages/walkaround-hybrid/src/ppg/dTree.ts:247,254`, `dTreeSample` reuses `u0` for both the flux-descent (`remaining = u0 * totalFlux`) AND the leaf v-coord (`vSample = node.v0 + u0 * (node.v1 - node.v0)`), correlating leaf position with the descent path. The production GPU sampler (`ppgGuide.wgsl:259-270`) draws **fresh** randoms. Fix: stratify by carrying the rescaled residual of `u0` after descent, OR take a 3rd independent random, so the CPU oracle matches the GPU path.
- **Blast radius:** LOW. This is an **oracle-fidelity** fix — the production GPU path is already correct; only the CPU oracle diverged. gitnexus_impact on `dTreeSample` first to confirm callers are tests/oracle only.
- **Behavior pin / test:** This is an *intentional* behavior change to the oracle. Add a targeted test in `packages/walkaround-hybrid/__tests__/ppgGuidedSampling.test.ts` (exists): assert that over many samples the CPU `dTreeSample` leaf-v distribution is **decorrelated** from descent path AND matches the GPU sampler's statistical contract (uniform within leaf, independent of which leaf). Pin the new (correct) behavior; the OLD correlated behavior is the bug being removed.
- **Gate:** typecheck + test. **A/B note:** no GPU render changes (GPU was already right); the A/B is oracle-vs-GPU agreement, which IMPROVES. Flag in commit msg that this closes an oracle-divergence, not a render change.

### Task 0.2 — [Theme L, CORRECTNESS] `FusedMlpTrainer.dispose()` — fix GPU-buffer leak
- **Change:** `packages/walkaround-hybrid/src/nrc/fusedMlpTrainer.ts` allocates ~16 GPU buffers in `build()` with no dispose path (leaks until device teardown — contradicts host-owns-lifecycle). Add `FusedMlpTrainer.dispose()` that destroys all allocated buffers; call it from `NrcSubsystem.dispose()` (`nrc/nrcSubsystem.ts:497-498`).
- **Blast radius:** LOW-MEDIUM. gitnexus_impact on `NrcSubsystem.dispose` + `FusedMlpTrainer`. Verify no double-dispose (idempotent guard).
- **Behavior pin / test:** Characterization — a test that builds a trainer, calls `dispose()`, asserts buffers destroyed (mock `GPUBuffer.destroy` spy count == allocated count) and a second `dispose()` is a no-op. Assert `NrcSubsystem.dispose()` forwards to trainer dispose. No render-path change.
- **Gate:** typecheck + test.

### Task 0.3 — [Theme G] Stale/misleading comment sweep (NO code change)
- **Changes (comments/docblocks only):**
  - `neural/*` — replace "Bug 1..8 fix / Sprint 13 deleted scaffold" archaeology with plain invariant statements.
  - `ppg/serialise.ts:18-101` — delete the ~50-line rejected 8-f32-sTree design walkthrough (code uses 16-f32).
  - `ppg/ppgConstants.ts:10-46` — remove "the prompt" authority refs; fix depth-8/256 arithmetic error; correct `PPG_MIS_ALPHA` "fixed 0.5" wording (now guides).
  - `shared-samplers jakobHanika.ts:36-37` — header says finite-difference Jacobian; code is analytic (chain rule). Correct header. *(Verified: code is analytic.)*
  - `pt-webgpu emitterPacking.ts:430-431` — comment says default irradiance `[1,1,1]`; code (V22) returns `[0,0,0]`. Rewrite.
  - `pt-webgpu denoise/oidnFinalDispatcher.ts:90-93` — error msg references removed `extensions['vitrum.ptWebgpu.oidnModelUrl']`; live API is `oidn:{modelUrl}`. Update.
  - `pt-webgl ptEngineWebGL2.ts:464-467` — delete orphaned `#bdptLightPathTex` docblock (field doesn't exist).
  - `pt-webgl oidnFinalDispatcher.ts:199,201-212` — JSDoc `_cohortId` vs actual `#cohortId`; rename refs, move field decl above first use.
  - `walkaround ddgi/DDGI.ts:176-181` — **VERIFIED**: stale "finally takes effect / 2× unless preset corrected to 8" hedge; `DEFAULT_STRIDE=8` and presets span divisor 2/4/8/32 (`HybridEngineQualityPreset.ts:101/113/127/140`). Reconcile comment to shipped preset spread, delete hedge.
  - `pt-webgl bdptSceneEmittersCpu.ts` header + `nrc/fusedMlpTrainer.ts:11` — "NOT wired into path tracer" headers are now wired (via `nrcSubsystem`). Correct.
- **Behavior pin / test:** None needed (comments only). **Care:** for `oidnFinalDispatcher.ts:199` the field-decl move is a real code move (hoist `#cohortId` above its first use) — confirm typecheck stays green; treat that one sub-item as a tiny code change, not a comment edit.
- **Gate:** typecheck + test (catches the one field-move sub-item).
- **Parallelizable:** Fully independent of 0.1/0.2. Separate branch/agent.

**→ `/audit` checkpoint after Phase 0.**

---

## PHASE 1 — Shared-package primitives (land BEFORE consumers)

### Task 1.1 — [Theme F] shared-bvh helpers
- **Changes (in `packages/shared-bvh/src/`):**
  - `deriveSceneAABBFromBvhPositions()` — single source for the scene-AABB-from-BVH-positions copy currently in `WalkaroundGPUPipeline:1466-1488`, `PPGCoordinator:39-67`, `ReGIRCoordinator:62-92`. **(Helper lands here; consumers adopt in Phase 4 where those files are already being touched — OR adopt now if low-risk. Land the helper + adopt the 3 call sites; they're mechanical.)**
  - `scenePack.ts:255-352 vs 708-851` — make `packSceneFromCore` inline delegate to `packOneMeshLikePrimitive` (vec4 expansion + buildArrayBvh + node-extract are byte-identical).
  - `scenePack.ts:354-398,1050-1104,887-951` — extract `resolveInstanceTransforms()` + `mapPrimitivesById()` (3 TLAS-instance collectors rebuild Map+invert+transformAabb).
  - `scenePack.ts:419-595` — extract `rebaseLeafTriOffset()` + `copyVec4Strided()` out of the 175-LOC `spliceResizedPrimitiveBlas` (5 parallel-array prefix/changed/suffix copies, leaf-rebase ×4).
  - `index.ts:37-58` — add `expandIndicesToStride4()` helper for the stride-3→4 expansion that "every caller must post-process."
- **NOTE / leave-as-is:** the two binned-SAH builders (`buildArrayBvh.ts:274-471` vs `tlas.ts:188-343`) and the WGSL any/closest traversal copies (`bvhIntersect.wgsl.ts`) are flagged "tooling-constrained / inherent-given-WGSL." **Do the intra-module leaf-body dedup only**; do NOT force a generic `buildBinnedSah<Primitive>` in this sweep (the finding itself caveats it). If trivially clean, propose it; otherwise leave with a one-line note.
- **Behavior pin / test:** shared-bvh has the densest test coverage. Add characterization goldens: for `packSceneFromCore` vs `packOneMeshLikePrimitive` delegation, assert the packed buffers (positions, nodes, indices) are **byte-identical** for a representative multi-mesh + TLAS scene before/after. For `spliceResizedPrimitiveBlas`, golden the resulting buffer for a known resize op. For `deriveSceneAABBFromBvhPositions`, assert AABB equals current inline result on a fixture.
- **Gate:** typecheck + test. **Extra A/B care:** `spliceResizedPrimitiveBlas` (175-LOC, leaf-rebase) — any off-by-one in offset rebasing silently corrupts BVH traversal. Golden the exact buffer.
- **Parallelizable:** scenePack changes are one cluster (same file, sequence internally); `deriveSceneAABBFromBvhPositions` + `expandIndicesToStride4` are independent. → up to 2 sub-streams.

### Task 1.2 — [Theme K] shared-samplers/denoisers structure + defineUbo migration
- **Changes:**
  - **`shared-samplers lightTree.ts:506-679`** — split the ReGIR core (Boksansky 2021, 4 exports) into `regir.ts`. Re-export from index to preserve public surface. (gitnexus_impact on the 4 exports first.)
  - **`shared-denoisers atrousVarianceBindings.ts:70-82,158-172`** — migrate hand-rolled DataView UBO packers + hand-maintained `=16` sizes to `defineUbo` (from `shared-samplers/src/uboCodegen.ts` — **verified location**). Completes the incomplete W2-C13.
  - **`shared-denoisers atrous.wgsl.ts:62-110` + `atrousVariance.wgsl.ts:243-281` + `spatialFilter.wgsl.ts:165-197`** — extract the edge-stop weight body (normal/depth/chroma) into a shared `edgeStop.wgsl.ts` fragment; have all 3 include it.
- **Behavior pin / test:** For the `defineUbo` migration — golden the **emitted byte layout** (offsets + total size) of the old hand-rolled packer, assert `defineUbo` produces an identical buffer for representative uniform values. This is the highest-risk K item (a layout mismatch silently corrupts denoiser params). For `edgeStop.wgsl` — assert composed `atrous`/`atrousVariance`/`spatialFilter` shader strings are **byte-identical** before/after. For `regir.ts` — re-export equivalence (import path unchanged for consumers).
- **Gate:** typecheck + test.
- **Parallelizable:** regir split, defineUbo migration, edgeStop fragment are 3 independent units. → 3 sub-streams.

### Task 1.3 — [Theme D] shared WGSL fragments into the include-graph
- **Changes:**
  - Extend the walkaround include-graph (`WGSL_MODULES` / `composeWgsl`, verified present in `packages/walkaround-hybrid/src/shaders/`) to emit `sceneBindings` + `ddgiBindings` fragments. Replace the hand-copied scene `group(1)` 8-buffer decl + `DDGIGridUBO` across the 8+ files (`shade/ris/temporal/spatial/risGi/risGiNrc` + GRIS variants).
  - **`pt-webgpu material.wgsl.ts:441` + `bdpt/bdptLightSubpath.wgsl.ts:7-9`** — replace inline Rec.709 `dot(c,vec3f(.2126,.7152,.0722))` ×2 with the canonical `luminance()` already in scope (W2-C10 missed these). **Keep the `1e-20`-floor variant in bdpt** if it differs — verify the bdpt one isn't a deliberately-floored variant before collapsing.
  - **`pt-webgpu caustic.wgsl.ts:69-77,251-255`** — route hand-decoded packed material through canonical `decodeMaterial()` / `materialScalar()`.
- **NOTE / leave-as-is:** `walkaround surfaceTextures.wgsl.ts:159-272` `bvhTraceTintedVisibility` (needs per-hit tint not bool) — flagged hard-to-dedup. Keep lockstep with shared-bvh, add a cross-ref comment; do NOT force-merge.
- **Behavior pin / test:** Byte-identity on composed shader strings for all 8 walkaround consumers via `wgslCompose.test.ts` (extend it). For pt-webgpu, extend `wgslContract.test.ts` + `wgslLiteContract.test.ts` to assert composed `material`/`bdpt`/`caustic` strings unchanged. **CRITICAL for caustic.wgsl:** the hand-decode might not be bit-identical to `decodeMaterial()` — if the composed string changes, this is a real behavior change → flag for GPU A/B and verify the decode math is genuinely equivalent (don't assume).
- **Gate:** typecheck + test.
- **Parallelizable:** walkaround scene/ddgi fragments (1 unit), pt-webgpu luminance (1 unit), pt-webgpu caustic decode (1 unit) → 3 sub-streams. **caustic gets extra scrutiny.**

**→ `/audit` checkpoint after Phase 1.**

---

## PHASE 2 — Consumers adopt Phase-1 primitives; WGSL whole-body dedup

### Task 2.1 — [Theme E] pass/denoiser dispatch helpers (adopt existing `SharedBindGroupPass`)
- **Changes (in `packages/walkaround-hybrid/src/pipeline/`):**
  - Add `dispatchSingleBindGroup(ctx, pipeline, bg, label, {wg16?})`; adopt in the 7 single-bind-group passes (MotionVectors/GTAOUpsample/IndirectCombine/IndirectTemporalAccum/TemporalAccum/Resolve/SampleBudget) — currently only 3/~10 use `SharedBindGroupPass`.
  - Add `runAtrousChain()` for the à-trous ping-pong loop duplicated ×4 (`atrous.ts:69-88`, `atrousVariance.ts:287-313`, `svgfReal.ts:255-291`, `AtrousIndirectPass.ts:81-99`).
  - Add `extraGroups` option to `dispatchSharedBindGroupPass` for the GRIS slot-1 bind ×4 (`SpatialGIReservoirPass:84/96/114`, `TemporalGIReservoirPass:58`) + `RISGIPass` NRC slot-4 ad-hoc.
  - `SpatialGIReservoirPass.ts:73-118` — local closure for the 7-line dispatch block ×3.
  - `neural.ts:204-235` — replace borrowed labels `'welford-temporal'`/`'atrous-variance-variance'` with real `neural-pack`/`neural-unpack` labels; fix `passLabels` over-declaring 5 vs 2 dispatched.
  - `GTAOPass`/`ResolvePass`/`SampleBudgetPass` — co-locate the `defineUbo` struct literal with the `*.wgsl.ts` module (currently duplicates the WGSL struct layout, kept-in-sync by comment).
  - `IndirectTemporalAccumPass.ts:28-39` — replace the synthetic `'denoiser-adapter'` dep (purely pins timestamp slot order) with an **assertion test** that `composePassLabels == runtime dispatch order`.
- **Behavior pin / test:** Pass dispatch is GPU-side; pin via the existing pass-label / dispatch-order tests. For label renames (`neural.ts`), update the label assertions and add the `composePassLabels == dispatch order` assertion (this is itself the fix for `IndirectTemporalAccumPass`). Characterization: assert each refactored pass produces the same bind-group + workgroup-dims descriptor (mock device, capture dispatch args, compare before/after).
- **Gate:** typecheck + test. **Care:** the `neural.ts` label change could break any consumer keying on the old borrowed labels — gitnexus_impact on those label strings first.
- **Parallelizable:** dispatch helper, atrous helper, neural labels, defineUbo co-location are independent → up to 4 sub-streams.

### Task 2.2 — [Theme C] WGSL compile-time-variant whole-body dedup (biggest LOC win)
- **Changes:**
  - **`pt-webgpu intersectionLite.wgsl.ts` vs `intersection.wgsl.ts`** — **VERIFIED byte-identical (340 lines, empty diff).** Extract `PT_WEBGPU_INTERSECTION_CORE_WGSL`; compose FULL + LITE from it.
  - **`pt-webgpu material.wgsl.ts:25-79 vs 81-138`** — FrameParams 41-field struct + bindings 0-11 written twice → shared struct const; FULL appends bindings 12-13.
  - **`pt-webgpu kernelLite.wgsl.ts:136-249 vs kernel.wgsl.ts:350-436`** — material-decode + thin-film TMM prologue byte-identical → shared `shadePrologue()` helper.
  - **`walkaround risGiNrc.wgsl.ts:65-323`** — ~250/356 lines verbatim from `RIS_GI_WGSL` (already drifting: 4 vs 19 commits). Build a parameterized gi-ris body builder, inject the Lo-suffix (NRC vs DDGI).
  - **`walkaround temporalGi.wgsl.ts:62-217 vs 241-484` (+spatialGi)** — OFF/GRIS pair duplicates `worldFromHalfPx_temporal`/`projectToPrevHalfPx`/consts → `temporalGiCommon` module. **Delete the genuinely-dead `worldFromHalfPx_temporal`** (unused in BOTH copies — one of the two allowed deletions).
- **Behavior pin / test:** **Byte-identity on the FINAL composed shader string for every variant.** This is the canonical use of the existing `wgslContract.test.ts` / `wgslLiteContract.test.ts` / `wgslCompose.test.ts`. Snapshot each composed FULL and LITE shader to a golden string BEFORE refactor; after refactor assert `composed === golden`. Zero tolerance — a single differing byte fails.
- **Gate:** typecheck + test. **Extra A/B care:** `risGiNrc` is ALREADY DRIFTED from `RIS_GI_WGSL` (4 vs 19 commits) — the parameterized builder must reproduce *each* current variant byte-for-byte, drift included. Do NOT "reconcile" the drift here (that would be a radiometric change); preserve both current outputs exactly. If the drift turns out to be a latent bug, file it separately — do not fix it inside a dedup.
- **Parallelizable:** intersection, material, kernel prologue (pt-webgpu) and risGiNrc, temporalGi (walkaround) → up to 5 independent sub-streams. intersection is the cleanest (start there to validate the byte-identity harness).

### Task 2.3 — [Theme J] light-sampling CPU oracle dedup (intra-package only — see D3)
- **Changes:**
  - **`pt-webgpu bdpt/bdptEmitterPickCpu.ts:59-157,188-437` + `emitterPacking.ts:459-516`** — collapse the per-emitter-kind unrolled stride walk ×3 (in-file) + the re-walk in light-tree input into a shared flat-emitter generator **within pt-webgpu**.
  - **`pt-webgl bdpt/bdptSceneEmittersCpu.ts:50-163`** — collapse the in-package duplication. (Cross-backend hoist to shared-samplers = **D3 decision; default B = keep as two single-sourced copies.**)
  - **`pt-webgpu denoise`** — `OidnReadbackFn` type declared identically in `rgba16fReadback.ts:47` + `oidnFinalDispatcher.ts:54` → declare once, re-export.
- **Behavior pin / test:** Characterization — the emitter generator must yield the **same flat emitter list (same order, same values)** as the current unrolled walks for representative scenes (point/area/directional/env mix). Golden the list. Emitter pick order feeds RNG-correlated sampling, so order must be preserved exactly.
- **Gate:** typecheck + test.
- **Parallelizable:** pt-webgpu and pt-webgl are independent packages → 2 sub-streams. `OidnReadbackFn` re-export is trivial, fold into either.

**→ `/audit` checkpoint after Phase 2.**

---

## PHASE 3 — updatePrimitive cascade + config ingestion

### Task 3.1 — [Theme B] updatePrimitive fast-path cascade (same shape ×3 backends)
- **Changes:**
  - **`pt-webgpu index.ts:716-934`** — 218-LOC, 6 fast-path if-branches, null-triple restated 5×, repeated warn-drain+reset+setScene tail → ordered `{eligible, apply}` handler array + hoisted tail.
  - **`pt-webgl ptEngineWebGL2.ts:1111-1254`** — 6 `if(isXOnlyPatch)` branches, fallback-to-setScene copy-pasted 5×, epilogue 5× → handler table + shared epilogue.
  - **`HybridEngine.ts:933-1001`** — 4 branches repeat identical bvh/scene/subsystems/return epilogue; `topologyFields` stringly-probed + duplicated in `PrimitiveUpdates.ts:230-237` → uniform result applied once; hoist the topology-field const to one location.
- **Behavior pin / test:** Characterization is **essential** — `updatePrimitive` has many branches (material-only fast path, geometry-throws, topology-change). For each backend, drive a representative matrix of patch shapes (material-only, transform-only, positions-only, topology-change, null fields) and assert the **observable outcome** is identical before/after: which fast-path taken, what gets re-uploaded, what throws, the return value. gitnexus_impact on `updatePrimitive`/`updateEmitter` in each backend first.
- **Gate:** typecheck + test. **Care:** the handler-table refactor must preserve **branch ORDER** (first-eligible-wins) and the exact throw conditions for geometry/topology changes — these are contract-pinned (`HybridEngine.updatePrimitive` geometry case throws explicitly). Do not relax a throw into a silent setScene.
- **Parallelizable:** 3 backends fully independent → 3 branches/agents. Hoist the shared `topologyFields` const (HybridEngine + PrimitiveUpdates) as one coordinated sub-task.

### Task 3.2 — [Theme H] config god-object / option ingestion (gated on D1)
- **Changes:**
  - **`HybridEngineOptions.ts:76-870`** — per **D1 = NESTED**. Add `tuning?: Partial<Tunables>`; delete the ~25 hand-declared flat audit-knob fields (type now derives from `Tunables`). Rewire `readTunables()` to read `opts.tuning?.[k] ?? def.default` and fold the existing flat/`grouped` audit-knob reads into the `tuning` path. Update examples/tests passing flat audit knobs. **API surface change — pre-publish, no deprecation layer.**
  - **`HybridEngine.ts:154-401`** — split `parseHybridEngineOptions` (195-LOC, mixes lite-throws + denoiser-throws + 45-field defaulting) into `validateHybridEngineOptions` + `deriveHybridEngineConfig`. **Behavior-preserving split** (no validation logic change).
  - **`pt-webgl ptEngineWebGL2.ts:219-235,343,541-547,611`** — consolidate the ~12 bag-keyed reads (the designed seam) into frozen-config parsers; document the bag↔typed boundary. **Do not remove the bag** (designed extension seam).
  - **`createEngine.ts:96-522`** — extract `idempotentDispose.ts` (the 133-LOC `wrapWithIdempotentDispose` proxy) + `configureWebGpuCanvas()` (duplicated WebGPU canvas-configure block).
  - **`createEngine.ts:389-462`** — replace the hand-coded 11-optional-method proxy with a data-driven `{method, disposedBehavior}` table (current per-method shotgun surgery + inconsistent disposed-behavior). **Pin each method's current disposed-behavior exactly** (some throw, some no-op) — the table must reproduce per-method behavior, not unify it.
  - **`promiseLedger.ts:52-239`** — `incrementalPatchSupport` block byte-identical ×3 → shared `ALL_PATCHES_SUPPORTED` const; move the 25-line rationale comments to a doc.
- **Behavior pin / test:** `parseHybridEngineOptions` split — characterization over the full option matrix (lite-mode throws, denoiser-mode throws, all 45 defaults), assert parsed config identical. `createEngine` proxy table — for each of the 11 methods, assert pre-dispose forwarding AND post-dispose behavior identical (the `promiseLedger.test.ts` already asserts real-engine == ledger; extend it). `promiseLedger` const — assert the 3 sites still report identical support.
- **Gate:** typecheck + test.
- **Parallelizable:** HybridEngineOptions, HybridEngine.parse split, pt-webgl bag, createEngine extract, promiseLedger const → up to 5 sub-streams, but HybridEngineOptions + parse split touch overlapping files (coordinate). createEngine + promiseLedger independent.

**→ `/audit` checkpoint after Phase 3.**

---

## PHASE 4 — God-orchestrator splits (largest + riskiest — LAST, each behind char-tests)

> Each task in this phase: **write the characterization test FIRST** (capture current observable behavior over a representative driver), get it green, THEN refactor, THEN assert the char-test still green. gitnexus_impact (HIGH/CRITICAL likely) before each — report blast radius to user.

### Task 4.1 — [Theme A] `WalkaroundGPUPipeline.ts` (top churn: 85 commits, 1046 LOC)
- **Change:** Expose `bvhHost` getter instead of the 6 BVH-refit pass-throughs to `_bvhHost`. Extract `registerPasses()` as a free function. (`initialize()` 303 LOC — extract sub-steps if cleanly separable; do not force.)
- **Behavior pin:** Characterization driver that runs a full frame through the pipeline (mock device), captures the pass order + each pass's dispatch descriptor, asserts identical before/after. The `PASS_ORDER` registry + `wgslCompose` order pins already exist — lean on them.
- **A/B FLAG:** pipeline drives the actual render. Any reordering of pass registration changes output → **assert `PASS_ORDER` byte-identical.**

### Task 4.2 — [Theme A] `HybridEngine.ts` (gated on D4)
- **Change:** Per **D4 (default B)** — introduce `readonly _cfg: ParsedHybridEngineConfig`; migrate the ~30 splatted tunable members that hop 6 layers onto it. Leave genuine per-instance mutable runtime state as fields. Document `PipelineInitHost`/`FrameDeps` wide back-refs as accepted-coupling (the finding labels them accepted).
- **Behavior pin:** Characterization — construct HybridEngine with a representative options set, assert every tunable resolves to the same value via `_cfg` as it did via the splatted field. Run a mock frame, assert deps/inputs identical.
- **A/B FLAG:** highest field-count change. The tunable values feed radiometry — assert each resolved tunable is bit-identical.

### Task 4.3 — [Theme A] `pt-webgpu index.ts` (1180 LOC, 52 commits)
- **Change:** Extract `FrameParamsPacker` (`#buildParamsBuffer:527-608`) + `SceneMutationRouter` (`setScene`/`add`/`remove`/`updatePrimitive`/`updateEmitter`). (updatePrimitive itself already cleaned in 3.1 — this is the routing extraction.)
- **Behavior pin:** Golden the `#buildParamsBuffer` output (the packed FrameParams buffer) for representative frame inputs — assert byte-identical. Characterize SceneMutationRouter dispatch (which mutation → which internal op).
- **A/B FLAG:** `FrameParamsPacker` emits the GPU uniform buffer — **byte-identity mandatory.**

### Task 4.4 — [Theme A] `pt-webgl ptEngineWebGL2.ts` (1150 LOC, 46 commits, 8 responsibilities)
- **Change:** Lift patch-routing into `scenePatch.ts routePrimitivePatch()` (coordinate with 3.1's pt-webgl handler table — do 3.1 first, then this extracts it to its own file). Extract `AdaptiveScheduler` class.
- **Behavior pin:** Characterization over patch-routing matrix (reuse 3.1's tests, now asserting against the extracted module). AdaptiveScheduler — assert same schedule decisions for a sequence of frame timings.
- **A/B FLAG:** AdaptiveScheduler governs sample budget/convergence cadence — assert identical scheduling decisions.

### Task 4.5 — [Theme I] `InferenceGraph` / NRC / PPG god-orchestrators
- **Changes:**
  - **`neural/InferenceGraph.ts:105-754`** — extract `TensorDimSolver` + `LayerResourceAllocator`; **compute tensor dims ONCE at init** (currently recomputes per-layer per-frame at line 347). *(This is a perf fix that must be behavior-identical — same dims, computed once.)*
  - **`neural/InferenceGraph` + `unetArchitecture`** — `bilinearUpsample` layer kind: **document as extension point** (D2) — "fully plumbed; no spec currently emits it." Do NOT delete.
  - **`nrc/nrcSubsystem.ts:344-482`** — extract `HashGridTableTrainer` peer class; allocate the 3 UBOs at init instead of 3 throwaway per frame. (Coordinate with 0.2's `dispose()` — the new peer class also needs dispose wired.)
  - **`nrc/fusedMlpTrainer.ts:454-516`** — split FD/loss/readback debug helpers into `FusedMlpTrainerProbe`. (Coordinate with 0.2 — dispose covers the split.)
  - **`nrc/nrcEncoding.wgsl.ts:140-182`** — collapse the trilinear hash-grid scatter ×4 sites; **delete the ptr-arg `nrcHashLevelBackward`** (undispatchable oracle that can drift) and **point the oracle at the CPU reference** (one of the genuinely-orphaned removals — verify it's truly undispatchable first via gitnexus).
  - **`ppg/dTree.ts:48-185`** — `pushFourChildren()` helper for the identical `childExtents`+push in `buildSubtree`/`buildSubtreeChildrenOnly` + the 3rd BFS layout in `compactDTree`.
  - **`ppg/ppgGuide.wgsl.ts:154-195` + `ppgPdf.wgsl.ts:143-176` + `ppgUpdate`** — parameterize the flux-proportional dTree sampler + sTree descent ×3 by buffer-prefix + RNG-fn (group(0) vs group(3) bind-limit constraint).
- **Behavior pin:** InferenceGraph — golden the inference output for a fixed input + weights, assert bit-identical (the dims-once change must not alter results). NRC trainer split — characterization that training step produces identical buffer mutations. ppg WGSL — byte-identity on composed `ppgGuide`/`ppgPdf`/`ppgUpdate` strings. dTree `pushFourChildren` — golden the built tree structure.
- **A/B FLAG:** **`nrcHashLevelBackward` deletion** — verify via gitnexus_impact that NOTHING dispatches it (the finding says undispatchable; confirm before deleting). The CPU-oracle re-point must keep the oracle test green. **InferenceGraph dims-once** — assert identical inference output (a dim mismatch would silently corrupt).

**→ `/audit` checkpoint after Phase 4 (final).** Then full-workspace `npm run typecheck` + `npm test`, and a final knip pass to confirm no new dead exports introduced.

---

## DEAD-CODE / COSMETIC DOWN-SCOPING (fold into the phase that touches each file)

- Make **non-export** (NOT delete), only where the symbol is verified used intra-file: `lambertianDirectionalPdf`, `rgba16fBufferToRgbF32`, the 7 PT_WEBGPU material binding consts, `beerTextureSize`/`emissiveTextureSize`, `getTableEntry`, `BMFR_WORKGROUP_SIZE`, `grisReuseMis` luminance, packing helpers, `TUNABLE_DEFINITIONS` (only if D1=B keeps it internal), the 9 unused-exported types, `AUTO_REALTIME_TRIANGLE_BUDGET`.
- **knip config:** add `src/**/*Harness.ts` + `tools/gpu-env/*` to entry globs to stop harness false-positives (do early — it makes the final knip gate trustworthy). → fold into Phase 0.
- **devDeps:** verify `react-dom` + `@types/react-dom` (engine & dev) against examples BEFORE removing; remove the redundant root `@typescript-eslint/eslint-plugin`. **Leave** three-gpu-pathtracer fork devDeps (example tooling). → Phase 3 (config) or standalone.

---

## DEFINITION OF DONE

- [ ] All 12 themes addressed per the tasks above (or explicitly deferred via a Decisions-needed outcome).
- [ ] Every WGSL dedup proven byte-identical on the composed string (golden assertions in `wgslCompose`/`wgslContract`/`wgslLiteContract` tests).
- [ ] Every TS restructure pinned by characterization/golden tests asserting identical observable behavior.
- [ ] Two Theme-L correctness fixes have their own targeted tests; dTree fix proven to make CPU oracle match GPU; trainer dispose proven idempotent + wired from NrcSubsystem.dispose.
- [ ] `npm run typecheck` green across all workspaces.
- [ ] `npm test` green across all workspaces.
- [ ] `/audit` clean after each phase (no new God-files / mixed-concern / export-sprawl).
- [ ] Final knip pass: no NEW dead exports; harness false-positives suppressed via entry-glob config.
- [ ] No contract-promised feature removed; `bilinearUpsample` documented not deleted; only the two genuinely-orphaned removals (`worldFromHalfPx_temporal`, undispatchable `nrcHashLevelBackward` after gitnexus confirmation) deleted.
- [ ] Per-task A/B-flagged items (caustic decode, risGiNrc drift preservation, FrameParamsPacker, PASS_ORDER, InferenceGraph dims-once, HybridEngine tunables) verified byte/bit-identical OR escalated to the GPU A/B queue if a composed-string change is unavoidable.
- [ ] Commits do NOT get pushed (no remote push without user instruction).

## PARALLELIZATION SUMMARY (independent streams = candidate branches/agents)

- **Phase 0:** {0.1 dTree} ∥ {0.2 trainer dispose} ∥ {0.3 comments} — 3 branches.
- **Phase 1:** {1.1 shared-bvh} ∥ {1.2 shared-samplers/denoisers} ∥ {1.3 shared WGSL frags} — 3 branches (each with internal sub-streams).
- **Phase 2:** {2.1 pass helpers} ∥ {2.2 WGSL whole-body} ∥ {2.3 oracle dedup} — 3 branches; 2.2 has 5 independent shader sub-streams.
- **Phase 3:** {3.1 ×3 backends} ∥ {3.2 config, gated D1} — up to 4 streams.
- **Phase 4:** 4.1 ∥ 4.2 ∥ 4.3 ∥ 4.4 ∥ 4.5 — 5 streams, BUT each must land behind its own char-test and a fresh gitnexus_impact; coordinate 4.4 after 3.1 (pt-webgl) and 4.5 after 0.2 (NRC dispose).

## CROSS-TASK COORDINATION (do-X-before-Y)

- 0.2 (trainer dispose) **before** 4.5 (NRC split — the new peer classes also need dispose wired).
- 1.1/1.2/1.3 (shared primitives) **before** their consumers in Phases 2 & 4.
- 3.1 pt-webgl handler table **before** 4.4 (extracts it to `scenePatch.ts`).
- knip entry-glob config (Phase 0) **before** the final knip gate (Phase 4 close).
- D1 resolved **before** 3.2; D3 resolved **before** 2.3; D4 resolved **before** 4.2.
