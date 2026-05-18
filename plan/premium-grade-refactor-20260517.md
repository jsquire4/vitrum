# Premium-grade refactor plan — 2026-05-17

> **Status:** drafted, awaiting human review.
> **Input:** complexity sweep findings synthesized in
> `~/.claude/projects/-home-jsquire4-projects-vitrum/memory/complexity-sweep-20260517.md`
> (Themes A–I, 178 raw findings).
> **Out of scope:** math/physics/numerical correctness — tracked separately in
> `~/.claude/projects/-home-jsquire4-projects-vitrum/memory/in-flight-sweep-20260511.md`.
> **Goal:** dissolve the structural perimeter that holds vitrum below
> $200/mo subscription-tier library quality. The algorithm cores are
> library-grade; this plan only touches structure, API hygiene, scaffold
> finishing, dedup, and dead-code.

---

## 0. Locked decisions (do not relitigate)

These were settled by the dispatching agent before this plan was drafted:

1. **RC → extract.** Create `@vitrum/walkaround-rc`. Rewrite the three.js
   `StorageBufferAttribute.__gpuBuffer` reach-through at
   `rc/cascadeDispatch.ts:182, 411-415, 460, 512-513` to raw `GPUBuffer`
   ownership. Drop RC from `walkaround-hybrid`. Rename / update the
   `walkaround-hybrid` README to honestly state "DDGI + ReSTIR-DI +
   ReSTIR-GI + GTAO + per-channel SVGF (no RC)".
2. **PPG → finish.** Write `serialiseDTree(dTree): Float32Array` (CPU
   producer); write a flat-traversal WGSL kernel replacing the uniform-grid
   `u32(sqrt(N))` stubs at `ppgGuide.wgsl.ts:122-132` and
   `ppgUpdate.wgsl.ts:100-114`; plumb the sTree GPU buffer from
   `HybridEngine`; replace `dispatchWorkgroups(0,0,0)` at
   `WalkaroundGPUPipeline.ts:884, 907` with real dispatches; add the
   MIS-with-PPG-guidance indirect-bounce path in `shade.wgsl`. The CPU
   `ppg/dTree.ts` + `ppg/sTree.ts` is well-built and does not need rewrite.
3. **Neural → finish.** Wire `INPUT_PACKER_WGSL` from
   `neural/inputPacker.ts` (currently 0 importers) into
   `InferenceGraph._runInputPack` (currently 3 planar `copyBufferToBuffer`
   calls); fix `InferenceGraph.dispose()` to actually free weights/biases/UBOs
   (admitted leak); strip the 137 lines of comment-only shape-debate at
   `unetArchitecture.ts:80-217`; verify + fix remaining skip-connection
   shape mismatches flagged by the 05-11 sweep; add a `'neural'`-mode
   example in `examples/` proving the wiring.
4. **OIDN → wire `'oidn-final'` as a real mode.** Add a dispatch path in
   `HybridEngine` and `pt-webgl/PTEngineWebGL2`; the
   `shared-denoisers/oidnBridge.ts` implementation is already
   code-complete. Demote the underscored test-only exports (`_hwcToNchw`,
   `_nchwToHwc`) out of the public surface.
5. **Dev stub overlays → finish.** Implement all 4 (`DDGIAtlasViewer`,
   `BVHVisualizer`, `GISignalSplit`, `MaterialInspector` picker) against the
   existing `HybridEngine.debug.*` surface (already implemented at
   `HybridEngine.ts:1237-1294` as of commit `38463d1`). Default to
   finishing; only remove an overlay if the underlying debug primitive is
   genuinely missing and unimplementable, with a 1-line note.

These five dispositions are not options to be reconsidered. They drive the
shape of the workstreams below.

---

## 1. Premium-library acceptance criteria

A change in any workstream is "done" only when it meets ALL of:

1. **Behavior preserved** — `npm run typecheck` workspace-wide is clean;
   `npm test` passes every package's vitest suite (~660+ tests, the 2-3
   GPU-only skipped tests stay skipped); any test deleted is replaced by
   one that covers the same behavior at the new abstraction.
2. **Reference renders** — for any change that alters the visual output of
   any backend (PT or walkaround), capture before/after reference renders in
   `tools/reference-renders/` and A/B them per the CLAUDE.md testing
   protocol. Numerical regression acceptable only with explicit visual
   justification documented in the PR body.
3. **No band-aids** — symptoms-suppressing fixes are forbidden. If a
   refactor reveals a deeper bug, file it against the 05-11 sweep tracking
   list and either fix it in scope or coordinate with that track.
4. **Citation hygiene** — every new algorithm reference points to its
   prior-work source in source comment + package README + project-level
   `CREDITS.md`.
5. **Audit-checkpoint clean** — `/audit` run after each implementation
   wave; issues fixed before moving to the next wave (mandatory per
   plan-implementation skill rules — structural debt compounds).

---

## 2. Workstreams (13 total)

| #   | Workstream                                                                                         | Theme          | Wave | Parallel-safe with | First-PR candidate                                       |
| --- | -------------------------------------------------------------------------------------------------- | -------------- | ---- | ------------------ | -------------------------------------------------------- |
| W1  | Pass + Resource + Denoiser registry abstraction                                                    | B              | 1    | W2, W3, W7         | **YES — first PR**                                       |
| W2  | Dedup canonicalisation (BVH/oct/luminance/MaterialEntry)                                           | C              | 1    | W1, W3, W7         |                                                          |
| W3  | Contract hygiene (Material mutability, AnalyticShape, denoiser DU, BackendTexture brand)           | D              | 1    | W1, W2, W7         |                                                          |
| W4  | God-file dissolution (HybridEngine, WalkaroundGPUPipeline, pathTraceBruteforce, etc.)              | A              | 2    | W5, W6, W11        | depends on W1                                            |
| W5  | Per-pass BGL (eliminate dead bindings, drop shared PipelineLayouts)                                | I              | 2    | W4, W6             | depends on W1                                            |
| W6  | Hidden-globals de-singletonization (sharedWebGpuDevice, iblBaker, **WGPU**)                        | E              | 2    | W4, W5, W11        | depends on W3                                            |
| W7  | Stale + misplaced + dead-code sweep                                                                | G + H          | 1    | all                |                                                          |
| W8  | Scaffold-1: RC extraction → `@vitrum/walkaround-rc`                                                | F (RC)         | 3    | W9, W10, W11, W12  | depends on W2 (BVH canonicalize) + W4 (pass abstraction) |
| W9  | Scaffold-2: PPG GPU dTree traversal finish                                                         | F (PPG)        | 3    | W8, W10, W11, W12  | depends on W1, W4, W5                                    |
| W10 | Scaffold-3: Neural denoiser finish                                                                 | F (Neural)     | 4    | W11                | depends on W1, W3, W5                                    |
| W11 | Scaffold-4: OIDN `'oidn-final'` wire                                                               | F (OIDN)       | 3    | W8, W9, W10, W12   | depends on W3                                            |
| W12 | Scaffold-5: Dev overlays finish (DDGIAtlasViewer, BVHVisualizer, GISignalSplit, MaterialInspector) | D14            | 3    | W8, W9, W10, W11   | depends on W3                                            |
| W13 | Documentation reconciliation (CLAUDE.md, CHANGELOG.md, plan/, memory/)                             | G (stale-docs) | 4    | —                  | depends on everything                                    |

### Wave structure

- **Wave 1 (foundation, parallel):** W1, W2, W3, W7 — abstractions and
  hygiene before any structural rewrite.
- **Wave 2 (structure, parallel-after-Wave-1):** W4, W5, W6 — dissolve
  god-files, switch to per-pass BGL, remove hidden globals.
- **Wave 3 (scaffold-finish, parallel-after-Wave-2):** W8, W9, W11, W12 —
  the non-neural scaffold completions.
- **Wave 4 (neural + docs):** W10, W13 — neural is the riskiest scaffold;
  docs reconcile against shipped state.

---

## 3. Per-workstream specifications

### W1 — Pass + Resource + Denoiser registry abstraction (Theme B)

**Goal:** make adding one pass a single-file change instead of the current
~25-30 edits across 6-9 files (verified at sweep time).

**Sub-tasks:**

1. Define a `Pass` interface in `walkaround-hybrid/src/pipeline/Pass.ts`:
   - `id: string`, `bindGroupLayout(): GPUBindGroupLayout`,
     `bindGroup(res: FrameResources): GPUBindGroup`,
     `pipeline(device, common): GPUComputePipeline`,
     `dispatch(encoder, res): void`, `dependencies(): readonly string[]`,
     `gates(opts): boolean` (decide whether to run based on EngineOptions).
2. Refactor `FrameResources` god-struct (`resourceManager.ts:319-763`, 41
   sibling fields) into per-algorithm sub-structs:
   `FrameResources.ddgi`, `.restirDI`, `.restirGI`, `.svgf`, `.gtao`,
   `.ppg`, `.neural`, `.common`. Each sub-struct has its own allocator and
   destructor. `createFrameResources` becomes a thin assembler over the
   per-algorithm allocators.
3. Replace the 4-string `denoiser` union switched across 5 files with a
   **Denoiser Registry** in `walkaround-hybrid/src/pipeline/denoisers/`.
   Each denoiser exports `{ id, allocate(device, dims), bindGroupLayout,
pipeline, dispatch, destroy }`. `HybridEngine` consults the registry by
   id. Adding a denoiser = adding a registry entry + sub-struct allocator;
   no edits elsewhere.
4. Replace the position-encoded pass order in
   `WalkaroundGPUPipeline.renderFrame` (lines 648-1198) with a declarative
   `PASS_ORDER: readonly Pass[]` array; `renderFrame` iterates and
   topologically-sorts by `dependencies()`. The duplicate declaration in
   `timestampQueries.buildPassLayout()` consumes the same array — single
   source of truth.
5. Replace the hand-rolled WGSL concatenation in `pipelineCompiler.ts` (9
   prepend patterns, anti-duplication-by-comment at line 131) with an
   include-graph: each WGSL module declares its `requires: string[]`; the
   compiler topo-sorts and concatenates once. Drop `COMMON_WGSL` as a
   universal prepend; passes opt-in.

**File paths (writes):**

- `packages/walkaround-hybrid/src/pipeline/Pass.ts` (new)
- `packages/walkaround-hybrid/src/pipeline/PassRegistry.ts` (new)
- `packages/walkaround-hybrid/src/pipeline/denoisers/{index,atrous,atrousVariance,svgfReal,neural,oidnFinal,none}.ts` (new)
- `packages/walkaround-hybrid/src/pipeline/resourceManager.ts` (refactor: split FrameResources)
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` (slim renderFrame to a pass loop)
- `packages/walkaround-hybrid/src/pipeline/pipelineCompiler.ts` (include-graph)
- `packages/walkaround-hybrid/src/pipeline/timestampQueries.ts` (consume PASS_ORDER)

**Test plan:**

- **Behavior preservation:** every existing walkaround test runs unchanged
  and passes.
- **New tests:** `Pass.test.ts` covering registry registration,
  dependency cycle detection, gate evaluation, topological order
  determinism; `denoiserRegistry.test.ts` covering id collision,
  allocate/dispose round-trip, dispatch sanity.
- **Reference renders:** capture full hero scene before W1 starts; capture
  after; A/B must be numerically identical to within FP roundoff (this is a
  pure structural refactor — bits should not change).

**Reference-render captures:**

- `tools/reference-renders/W1-pre/hero-{1080p,720p}.png`
- `tools/reference-renders/W1-post/hero-{1080p,720p}.png`
- Diff report appended to `tools/reference-renders/sweep-2026-05-11-diff-report.md` pattern.

**Estimated sprints:** 2 (one for Pass+Registry+ResourceSplit, one for include-graph + renderFrame slim).

**Risk:** medium — touches every pass, but mechanical with strong test
coverage.

---

### W2 — Dedup canonicalisation (Theme C)

**Goal:** ~15 duplicated algorithms reduced to one canonical source each.

**Sub-tasks (each is a small focused PR):**

1. **C1 BVH WGSL traversal** — consolidate into
   `@vitrum/shared-bvh/wgsl/bvhTraverse.wgsl.ts` (canonical). Replace 4
   in-place copies (ddgi/probeUpdateRays, rc/probeRayCast,
   pt-webgpu/pathTraceBruteforce, walkaround/common.wgsl) with imports of
   the canonical string. Verify byte-equivalence after compilation.
2. **C2 BVH CPU builder** — pull a backend-agnostic
   `buildBvhFromTriangles` into `@vitrum/shared-bvh`. Make
   `pt-webgpu/scene/buildCpuBvh.ts` delegate to it. Make
   `shared-bvh/bvhCommon.ts` the three-adapter (the comment at H3 said so;
   make it true).
3. **C3 octEncode/octDecode** — canonical in
   `shared-samplers/wgsl/octahedralCore.wgsl.ts` already. Replace the
   buggy `sign(n.x)` re-inlined copy in `rc/wgsl/probeRayCast.wgsl.ts:320`
   and `rc/wgsl/cascadeMerge.wgsl.ts:44`; replace the TS copy in
   `rc/octahedralSolidAngles.ts:48` with `octahedralCore.ts` import. (Note:
   the `rc/` code paths move to `@vitrum/walkaround-rc` in W8 — coordinate
   ordering: do C3 first, then W8 carries the canonical version.)
4. **C4 Atlas-coord arithmetic** — single source in `shared-bvh`'s
   `probeAtlasUv`; 3 in-place copies replaced.
5. **C5 MaterialEntry struct** — single canonical packing in
   `@vitrum/walkaround-hybrid/src/scene/materialEntry.ts`; 3 incompatible
   copies (ddgi/probeUpdatePass, rc/bvhCompute, restir/bvhCompute) unified.
   Default values consistent (roughness 0.5 not "0.5 here, 1.0 there").
6. **C6 PCG/safe_normalize/safeInvDir/intersectTriangle/fresnelSchlick/buildONB**
   — canonical in `@vitrum/shared-samplers/wgsl/` (where the math
   primitives logically belong); `pt-webgpu/wgsl/common.wgsl.ts` and
   `walkaround-hybrid/shaders/common.wgsl.ts` both import. Capitalization
   normalized (`buildONB` everywhere).
7. **C7/C8/C9 ReSTIR `computePHat`** — pull into a single canonical WGSL
   string + helper module, imported by ris/temporal/spatial/risGi/temporalGi/spatialGi/shade.
   Eliminate `castPrimary`/`castPrimary_t` triple.
8. **C10 Rec.709 luminance vector** — `const REC709_LUMINANCE = vec3(...)`
   in `shared-samplers/wgsl/luminance.wgsl.ts`. All 6+ open-codings import.
9. **C11 B3-spline kernel** — single `ATROUS_KERNEL_WGSL` in
   `shared-denoisers/wgsl/atrousKernel.wgsl.ts`; both atrous +
   atrous-variance import.
10. **C12 GTAOUniforms struct** — single struct decl; gtao.wgsl +
    gtaoUpsample.wgsl import.
11. **C13 UBO packing codegen** — write a `defineUbo<T>(spec)` codegen
    helper in `@vitrum/shared-samplers` (or new
    `@vitrum/shared-ubo`) producing { interface, SIZE_BYTES, pack(dv,
    offset, value), WGSL struct string } from a single spec. Apply to ≥5
    hand-mirrored UBOs in shared-denoisers; leave the rest as
    drop-in-replace candidates over time.
12. **C14 spectral CDF** — re-export
    `computeCmfIntegralWavelengthSpace`/`buildCmfCdf` from
    `@vitrum/shared-samplers` index; `pt-webgl/forkUniformBridge.ts`
    imports them instead of re-implementing.
13. **C15 Hash function** — `pcgHash` / `floatHash` in
    `shared-samplers/wgsl/hash.wgsl.ts`; replace 3+ `sin(...)*43758.5453`
    open-codings.

**File paths:** see sub-task list above; touches `shared-bvh`,
`shared-samplers`, `shared-denoisers`, `walkaround-hybrid`, `pt-webgpu`,
`pt-webgl`.

**Test plan:**

- **Behavior preservation:** ALL existing tests must pass. Any test
  asserting bit-exact WGSL output replaced with import-graph assertion.
- **New tests:**
  - Per-canonicalisation: a regression test that the canonical WGSL string
    is byte-equal to the previous in-place copy (snapshot test).
  - `materialEntryPacking.test.ts`: byte layout, default values.
  - `defineUbo` codegen: round-trip pack-then-decode equality.
- **Reference renders:** mandatory — even though math should be identical,
  octahedral C3 fixes a _bug_ (sign(0) collapse at axis boundaries). The
  hero render is the verification. Capture before/after the C3 fix
  specifically.

**Reference-render captures:**

- `tools/reference-renders/W2-C3-pre/hero-axis-aligned.png`
- `tools/reference-renders/W2-C3-post/hero-axis-aligned.png`
- (others: bit-identical, skip captures)

**Estimated sprints:** 2 (small focused PRs can be batched).

**Risk:** low per-item, but C3 + C5 (different default roughness) can
silently change images — the reference-render gate catches this.

---

### W3 — Contract hygiene (Theme D)

**Goal:** `@vitrum/core` becomes the load-bearing contract the project
already calls it; no host-app vocabulary, no leaked sentinels, no optional
quicksand.

**Sub-tasks:**

1. **D1 Material mutability** — split `Material` into `MaterialSpec`
   (readonly, the contract) and `MaterialMutationProxy` (the runtime live
   handle backends produce). The prose-encoded "mutability contract"
   becomes a type relationship. Update three-bindings + pt-webgl +
   pt-webgpu + walkaround-hybrid to consume `MaterialSpec` in builders and
   issue mutations via the proxy.
2. **D2 AnalyticShape** — remove `'h-channel-came'` from the core enum;
   move to `@vitrum/stained-glass-extensions` (new tiny package) or expose
   via `AnalyticShape.extensions` (per CLAUDE.md key design principle #3).
   Strip internal codes ("T3.E", "RFE-01"…"Gap 5") from JSDoc; replace
   with prior-art citations or remove. (The codes are useful in tracking
   docs; not on the public API.)
3. **D3 dichroicLUTs** — extract from three-bindings to
   `@vitrum/stained-glass-extensions`. three-bindings becomes
   host-app-agnostic again.
4. **D4 Denoiser discriminated union** — replace the string union with
   `Denoiser = { kind: 'none' } | { kind: 'atrous' } | { kind:
'atrous-variance' } | { kind: 'svgf-real' } | { kind: 'neural';
weights: NeuralWeights } | { kind: 'oidn-final'; modelUrl: string }`.
   Per-mode requirements are type-enforced; the registry in W1 consumes
   this DU.
5. **D5 isHardwareGpu** — drop the REQUIRED+DEPRECATED field; staging dir
   is slated for deletion (it goes in W7). Remove from `GpuDetection`.
6. **D6 Mat4 brand** — `type Mat4 = Float32Array & { __mat4: true }`; add
   `asMat4(arr): Mat4` constructor with a length-16 assert. Same for
   `Vec3`/`Vec4`/`Quat` if they exist.
7. **D7 FrameOutput discriminated union** — replace
   `primaryRadiance: BackendTexture | null` + the
   `samplesAccumulated === 0` sentinel with
   `output: { kind: 'skipped'; reason: string } | { kind: 'rendered';
texture: BackendTexture; samplesAccumulated: number }`.
8. **D8 Engine interface** — replace the 6 optional members with a
   discriminated `EngineCapabilities` object the engine declares once at
   construction; callers query capabilities and dispatch. Drop `typeof
engine.foo?.X === 'function'` checks everywhere.
9. **D9 createEngine proxy** — proxy `updateEnvironment` (currently
   dropped); add to the unit-test that every Engine method round-trips
   through the proxy.
10. **D10 / D11** — HybridEngine consumes `FrameInput.viewport` (size
    changes don't require teardown); `attachVitrum` plumbs `swapChainView`
    (walkaround returns real output when host-driven).
11. **D12 HybridEngineOptions** — move the ~30 walkaround-specific fields
    behind `EngineOptions.extensions['walkaround-hybrid']` per the
    CLAUDE.md generalization principle. Hosts construct generic
    `EngineOptions` + extensions, every backend reads its own bucket.
12. **D13 Public packages stop re-exporting raw WGSL strings.** All
    `COMMON_WGSL`, `RIS_WGSL`, etc. become package-internal; pass
    registration via W1's Pass interface is the supported extension point.
13. **D15** — `_hwcToNchw`/`_nchwToHwc`/`_powerPrefixSumDebug`/
    `_skyEquirectCacheSize` move out of public index files; tests import
    from internal paths.
14. **D16** — uniform telemetry: PT and walkaround both implement
    `onFrame`/`onProgress`/`FrameStats` exactly. Quality-modes and
    adaptive-scheduler on pt-webgl either move into the contract or
    documented as PT-only `EngineCapabilities.qualityModes`.
15. **D17 Material.extensions** — at least one backend consumes it
    (three-bindings reads `material.extensions.dichroicLUT` after the D3
    move). Otherwise delete the scaffolding.
16. **D18 Motion blur** — `supportsMotionBlur` + `FrameInput.shutterTime`
    consumed by at least one backend or dropped. Probably dropped — no
    consumer in roadmap.
17. **D19 BackendTexture branding** — `BackendTexture<TFormat>` becomes a
    branded type; WebGPU adapter narrows it to `GPUTextureView` at
    backend boundary; first error is a TS error not a runtime cast
    failure.

**File paths:** `packages/core/src/{scene,engine,frame,gpuDetection}.ts`
(all four), every backend's adapter (`pt-webgl/index.ts`,
`pt-webgpu/index.ts`, `walkaround-hybrid/HybridEngine.ts`,
`three-bindings/index.ts`), `packages/engine/src/createEngine.ts`,
new `packages/stained-glass-extensions/` (if D2/D3 produces it).

**Test plan:**

- **Behavior preservation:** every existing test passes. Tests asserting
  string union values updated to assert discriminated union shape.
- **New tests:** D7 round-trip (skipped frame yields skipped, rendered
  frame yields rendered); D6 brand mistype is a compile error (use
  expect-type / TS-only test); D12 extensions round-trip across `engine`
  facade; D19 BackendTexture format narrows correctly.
- **Reference renders:** none required — pure type/contract change.

**Estimated sprints:** 2 (D1, D2/D3, D4 split into two; rest fold in).

**Risk:** medium-high — contract change touches every consumer. Mitigation:
each sub-task is its own PR; type errors are caught at compile-time, not
runtime.

---

### W4 — God-file dissolution (Theme A)

**Goal:** the 7 god-files in the sweep table become focused modules
≤300 LOC each (with WGSL string files allowed to be longer per
necessity).

**Sub-tasks (one per god-file):**

1. **A1 HybridEngine.ts (1743 LOC, 12 jobs)** — extract:
   - `HybridEngineOptionsParser` (option validation + default expansion)
   - `HybridEngineTunables` (runtime mutable knobs)
   - `HybridEngineRebuildKey` (the 70+-field rebuild-trigger hasher)
   - `HybridEngineDebugSurface` (the `engine.debug` block at lines
     1237-1294)
   - `HybridEngineLightingState` (lighting mutation API + LightTree
     ownership)
   - The 3 `as unknown as` reach-throughs into `_pipeline._res` go away
     because W1 made FrameResources publicly grouped.
2. **A2 WalkaroundGPUPipeline.ts (1574 LOC, 41 private fields, 550-LOC
   renderFrame)** — W1 already slims renderFrame to a Pass-loop; this
   subtask finishes the dissolution by moving each algorithm's pipeline
   construction into its `Pass` module (PassRegistry pattern from W1).
   `WalkaroundGPUPipeline` becomes a thin orchestrator over registered
   passes.
3. **A3 resourceManager.ts createFrameResources (440 LOC)** — addressed by
   W1 (per-algorithm sub-structs). This subtask is verification.
4. **A4 pt-webgpu pathTraceBruteforce.wgsl (1908 LOC)** — split into
   `frameParams.wgsl`, `bindings.wgsl`, `bsdf/{layered,glossy,dielectric,
conductor,diffuse,mis}.wgsl`, `bvh.wgsl` (imports
   shared-bvh/bvhTraverse from C1), `lights/{rectArea,meshArea,emissive,
environment}.wgsl`, `mnee.wgsl`, `photonMap.wgsl`, `materialDecode.wgsl`,
   `main.wgsl`. Shared `BsdfSample` struct used by all sample/PDF/eval
   triples.
5. **A5 shade.wgsl (485 LOC, 380-LOC shadeMain)** — extract per-lighting-term
   helpers: `lo_emit`, `lo_direct`, `lo_sunCaustic`, `lo_skyAperture`,
   `lo_indirect`. Top-level `shadeMain` becomes a clean composition. Dead
   group-3 DDGI bindings disappear when W5 (per-pass BGL) lands.
6. **A6 common.wgsl (857 LOC, 11 subsystems)** — split by subsystem into
   `pcg.wgsl`, `reservoir.wgsl`, `ggx.wgsl`, `cameraHelpers.wgsl`,
   `samplerState.wgsl`. BVH primitives go to shared (W2 C1/C6); the 80
   lines of inline sprint commentary become commit messages or are
   deleted.
7. **A7 svgfRealWebGPU.ts (838 LOC)** — separate concerns:
   `svgfRealCpuEmulation.ts` (the CPU oracle), `svgfRealPipelineCache.ts`
   (compute pipeline cache), `svgfRealTextureUpload.ts` (the 7 helpers,
   shared with `atrousVarianceWebGPU.ts` — both consume
   `shared-denoisers/webGpuTextureCopy.ts` only), `svgfRealDriver.ts`
   (the 255-LOC `runSVGFRealWebGPU`).
8. **A8 probeUpdatePass.ts (1037 LOC)** — split:
   `probeRaysPass.ts`, `probeBlendIrradiancePass.ts`,
   `probeBlendVisibilityPass.ts`, `probeBorderIrradiancePass.ts`,
   `probeBorderVisibilityPass.ts`, `haltonShoemakeQuaternion.ts` (moves to
   `shared-samplers` per H5), `ddgiAtlasCoords.ts` (already exists; remove
   inline copies). Each pass is its own W1 Pass module.
9. **A9 pt-webgpu PTEngineWebGPU (470 LOC class, 8 concerns)** — extract
   `PTEngineWebGPUResources` (mirrors W1's grouped resources pattern),
   `PTEngineWebGPUSampleSchedule`, `PTEngineWebGPUTelemetry`. Aligns with
   what `pt-webgl` already has via the fork.

**File paths:** see sub-tasks. Touches `walkaround-hybrid/src/`
extensively; `pt-webgpu/src/`; `shared-denoisers/src/`.

**Test plan:**

- **Behavior preservation:** the full test suite must pass after each
  sub-task. This is the most fragile workstream — recommend per-god-file
  PRs.
- **New tests:** none mandatory; the existing suite is what validates
  preservation. Some unit tests of newly extracted modules are nice (e.g.,
  `HybridEngineRebuildKey.test.ts`) but optional.
- **Reference renders:** mandatory before each sub-task's merge. Hero
  scene + cornell-box + hero-product-viz at 1080p (the 3 examples in
  `examples/`). Bit-equality expected; numerical regression must be
  visually justified.

**Reference-render captures:**

- `tools/reference-renders/W4-A{N}-pre/{hero,cornell-box,hero-product-viz}-1080p.png`
- `tools/reference-renders/W4-A{N}-post/{...}.png`
- A/B diff committed in the PR body.

**Estimated sprints:** 4 (A1+A2+A3 in one, A4 alone, A5+A6 in one, A7+A8+A9 in one).

**Risk:** high mechanical-error rate due to volume of moved code. Heavy
test-driven mitigation; reference renders catch silent visual breaks.

---

### W5 — Per-pass BGL (Theme I)

**Goal:** every shader declares only the bindings it actually uses; shared
PipelineLayouts go away; the "for BGL compat, unused" dead bindings
disappear.

**Sub-tasks:**

1. **I1** — ris/temporal/spatial/shade currently share a 10-binding
   `group(0)` for layout compat. After W1 + W4-A5, each Pass declares its
   own BGL via its `bindGroupLayout()`. Verify ris/spatial drop ~70% of
   bindings (the dead ones per the sweep).
2. **I2 indirectCombine `ic_gNormalDepth`** — declared "for BGL compat,
   unused" — drop with the per-pass BGL move.
3. **I3 atrousVariance double-binding** — the `varIn_*` vs `atrous_*`
   prefix workaround for two entry points goes away when each entry point
   has its own BGL.

**File paths:** `walkaround-hybrid/src/shaders/*.wgsl.ts` (binding
declarations), `walkaround-hybrid/src/pipeline/bindGroupLayouts.ts`,
`walkaround-hybrid/src/pipeline/bindGroupBuilders.ts`,
`walkaround-hybrid/src/pipeline/BGLCache.ts` (probably collapses).

**Test plan:**

- Behavior preservation only. Existing tests cover correctness; this is a
  bind-group reshape.
- **Reference renders:** mandatory bit-equality at hero scene; if not
  bit-equal something migrated wrong.

**Estimated sprints:** 1.

**Risk:** medium — bind-group misindexing is a footgun, but it produces
loud GPU validation errors in test.

---

### W6 — Hidden-globals de-singletonization (Theme E)

**Goal:** CLAUDE.md key principle #2 ("host owns lifecycle") becomes
_actually true_. No module-level GPU resources, no
`window.__WGPU__` writes from engine internals.

**Sub-tasks:**

1. **E1 sharedWebGpuDevice singleton** — remove module-level
   `cachedDevice`/`pendingDevice`/`deviceGeneration` from
   `shared-denoisers/sharedWebGpuDevice.ts`. The 16-attempt retry loop
   and `SUPERSEDED_MSG` sentinel are symptoms of treating the device as
   owned-by-library when CLAUDE.md says it's host-owned. All denoiser
   drivers take a `GPUDevice` parameter (or a `DeviceProvider` callback);
   `reuseSharedWebGpuDevice` option is dropped.
2. **E2 iblBaker module-level Map** — convert to an
   `IblBakerCache` instance the host (pt-webgl's `PTEngineWebGL2`)
   constructs and owns. Renderer-scoped, not process-scoped.
3. **E3 HybridEngine writes to `window.__WGPU__`** — relocate to a
   host-bridge `WindowDebugBridge` that the host (the example app or dev
   panel) wires up by subscribing to `engine.onFrame`. The engine never
   touches `window`.
4. **E4 cascadeDispatch.\_sharedDispatcher** — zero callers, deleted in W7
   dead-code sweep; verify after W8 RC extraction.
5. **E5 `gpuBufferOf(attr)` \_\_gpuBuffer reach-through** — eliminated by W8
   (RC extraction rewrites to raw `GPUBuffer` ownership). Cross-stream
   dependency.
6. **E6 pt-webgl `_pathTracer.material.uniforms`** — encapsulate the
   fork-private accessor behind a single `ForkUniformBridge` class;
   `forkUniformBridge.ts` and `ptEngineWebGL2.ts` both go through it.
   When the fork eventually upstreams an official accessor, one file
   changes.

**File paths:** `packages/shared-denoisers/src/sharedWebGpuDevice.ts`,
`packages/pt-webgl/src/iblBaker.ts`, `packages/walkaround-hybrid/src/HybridEngine.ts`,
`packages/pt-webgl/src/{ptEngineWebGL2,forkUniformBridge}.ts`.

**Test plan:**

- New tests: `iblBakerCache.test.ts` cache instance isolation;
  `sharedWebGpuDevice.test.ts` removed (module deleted).
- Behavior preservation across denoiser drivers (each must accept a passed
  device).

**Estimated sprints:** 1.

**Risk:** low — singletons are caught at construction time.

---

### W7 — Stale + misplaced + dead-code sweep (Themes G + H)

**Goal:** delete what's dead, move what's misplaced, fix what's stale.
Pure hygiene — runs in parallel with everything from Wave 1.

**Sub-tasks (dead-code; verify zero-importers via `npx knip` + grep):**

- **G1 neural/inputPacker.ts** — DO NOT delete; wire it into
  `InferenceGraph._runInputPack` per W10.
- **G2 bdptMIS `_partial` variants** — fold into test fixtures; delete from src.
- **G3 jakobHanika `rgbToSpectralCoefficients` alias** — keep the canonical
  name; delete the alias. Update header to honestly say "3-channel
  Gaussian-peak approximation, see Jakob & Hanika 2019 for the full
  spectral-uplifting paper this approximates".
- **G4 pt-webgl `_skyEquirectCacheSize`** — verified zero callers, delete.
- **G5 shared-bvh `validateBvhEncoding`** — keep but un-export from
  `index.ts` (only tests use it).
- **G6 shared-denoisers `spatialFilter.wgsl.ts`** — either export from
  `index.ts` (some pass should consume it) or delete. Check W8/W5 first.
- **G7 three-bindings internal helpers** — un-export
  `extractAttribute`/`extractIndex`/`warnOnce` from `index.ts`.
- **G8 shared-samplers TS dead exports (~85%)** — audit each: keep the
  ones that are spec/oracle for the WGSL impls (and document them as such
  in the package README); delete the rest. Light tree, BDPT, jakobHanika,
  hgPhase, equiAngular, mixturePdf, cauchyIor, hero-wavelength to be
  individually triaged.
- **G9 pt-webgpu uploadSceneBuffers re-exports** — remove redundant
  passthrough.
- **G10 CLAUDE.md "PPG hard-throws"** — already false per sweep dead-code
  agent. Update CLAUDE.md in W13.
- **G11 CLAUDE.md "no neural mode"** — already false. Update in W13.
- **G12 three-bindings Vector3 scratch pool** — convert to per-instance.
- **G13 bounceLimit double-clamp** — single source of truth: the WGSL
  constant pulls from a TS-exported const, codegen-injected by the include
  preprocessor (W1).
- **G14 THIN_FILM_LAYER_LIMIT loop bounds** — use the constant
  consistently.
- **G15 gatherRadius magic number** — lift to
  `WalkaroundExtensions.gatherRadius` extension option.
- **G16 RC TSL_TO_RAW_MAPPING.md** — moves to `@vitrum/walkaround-rc/notes/`
  (internal-only, not in src/) via W8.

**Sub-tasks (misplaced):**

- **H1 rc/applyDDGIShading.ts** — move to `walkaround-hybrid/src/ddgi/`
  (imports only from ddgi). Do this BEFORE W8 RC extraction so the move
  doesn't snag.
- **H2 shared-bvh/wgsl/octahedral.wgsl.ts** — DDGI-specific atlas helpers;
  move to `walkaround-hybrid/src/ddgi/wgsl/atlasOctahedral.wgsl.ts`.
  Keep the canonical `octahedralCore` in shared-samplers.
- **H3 shared-bvh/sceneBvh.ts** — module name + header admit it's a DDGI
  wrapper; move to `walkaround-hybrid/src/ddgi/sceneBvh.ts` and shrink
  the shared-bvh surface.
- **H4 HybridEngine `collectDDGILightsFromRectAreaLights`** — move to
  `three-bindings/lights.ts`. HybridEngine imports it.
- **H5 Halton-Shoemake quaternion** — move from `probeUpdatePass` to
  `shared-samplers/wgsl/quaternionSampling.wgsl.ts` + TS counterpart.
- **H6 surfaceTextures.wgsl** — split: `bvhTraceTintedVisibility` stays in
  walkaround-hybrid; the 8 stained-glass procedural patterns move to
  `@vitrum/stained-glass-extensions` (the W3 D2/D3 destination).
- **H7 bdptVertex.ts `BDPT_MAX_LIGHT_BOUNCES`/`BDPT_MAX_EYE_BOUNCES`** —
  these are pt-webgl-fork budgets, move there.
- **H8 demodulateAlbedo/remodulateAlbedo** — move from
  `atrousVarianceWebGPU.ts` to a new `shared-denoisers/denoiserMath.ts`.

**Sub-tasks (stale):**

- **I (sweep stale findings)** — clean up `restir/bvhCompute.ts:100,259`
  ("deferred until Sprint 9/10" — shipped); reword
  `jakobHanika.ts:49` "TODO post-Sprint-12" — Sprint 12 shipped.
- **Stale plan/ files** — move ~40 completed-sprint artifacts to
  `plan/archive/` per the 05-11 sweep. (This is partly W13.)
- **Stale memory/ files** — 6 of 8 `project_*` files reference the
  pre-extraction stainedGlass host app. Mark as archived in memory;
  retain the two non-stale (`project_design_principles.md`, the
  `feedback_*.md` set). Done via memory hygiene, coordinated by W13.
- **Stale worktree branches** — delete
  `worktree-agent-a06812aa6c5b09b98`, `worktree-agent-a193d8b569f806c50`
  (referenced by stale in-flight memory; in-flight-plan.md claims a
  ReSTIR-GI rewrite is in flight but the work shipped in commits 1203df9
  / 7d307f6).
- **Disagreement between two RFE trackers** — `plan/external-requests-status.md`
  vs `external_requests/IMPLEMENTATION-STATUS.md` reconciled to one
  canonical doc.

**File paths:** scattered; ~30 files touched. Each sub-task is a small
focused PR.

**Test plan:** delete-only sub-tasks: test that the symbol no longer
exports + test suite still passes. Move-only sub-tasks: import path
update; tests still pass.

**Reference renders:** none required (pure hygiene; no visual change). The
H1/H2/H3 moves verify byte-equal WGSL output via existing snapshot tests.

**Estimated sprints:** 2 (G batch in one, H batch in one). Can run
opportunistically across all waves.

**Risk:** very low.

---

### W8 — Scaffold-1: RC extraction → `@vitrum/walkaround-rc`

**Goal:** RC becomes a real, independent package that does not reach into
three.js renderer privates.

**Sub-tasks:**

1. **Create the package** — `packages/walkaround-rc/` with its own
   `package.json` (workspace symlinked), `tsconfig.json`, `__tests__/`,
   README. Move all `walkaround-hybrid/src/rc/*` files. Drop `rc/` from
   `walkaround-hybrid/`.
2. **Rewrite the three.js `__gpuBuffer` reach-through.** The 5 sites at
   `cascadeDispatch.ts:182, 411-415, 460, 512-513` use `attr.__gpuBuffer`
   to grab the GPU buffer the Three.js WebGPU renderer allocated. Replace
   with raw `GPUBuffer` ownership: `RCDispatcher` allocates and owns its
   own buffers; the data marshaling between Three.js storage attributes
   (if RC still consumes scene data through three) becomes an explicit
   copy via `device.queue.writeBuffer`. The `gpuBufferOf(attr)` helper is
   deleted.
3. **Decide the RC public API.** Per the dispatching agent: do not block.
   Reasonable shape: export `RCDispatcher`, `RCConfig`, `RCDispatchOutput`;
   document host-driven usage. The package is independent enough that the
   stained-glass app or other hosts could wire it standalone.
4. **Update walkaround-hybrid README** — replace "DDGI + ReSTIR + RC hybrid"
   with "DDGI + ReSTIR-DI + ReSTIR-GI + GTAO + per-channel SVGF + PPG
   guide (no RC — see `@vitrum/walkaround-rc` for radiance-cascades)".
5. **`TSL_TO_RAW_MAPPING.md`** — moves to
   `packages/walkaround-rc/notes/TSL_TO_RAW_MAPPING.md` (internal, not
   under `src/`). It documents the migration story for future readers.
6. **C3 octahedral fix carries forward** — the canonical octEncode/octDecode
   from W2 is what the new package imports.

**File paths:**

- New: `packages/walkaround-rc/{package.json, tsconfig.json, src/**, __tests__/**, README.md, notes/TSL_TO_RAW_MAPPING.md}`
- Delete: `packages/walkaround-hybrid/src/rc/*`
- Update: `packages/walkaround-hybrid/{README.md, package.json}`,
  workspace `package.json` if RC is symlinked.

**Test plan:**

- All existing `rc/` tests move to the new package and pass.
- New test: `RCDispatcher.test.ts` verifies raw-GPUBuffer ownership (no
  reach-through into private renderer state).
- Integration test: scene with RC enabled standalone renders correctly.

**Reference renders:** if RC ships standalone with an example, capture
that. The walkaround-hybrid hero render does NOT change (RC was never
wired into HybridEngine per sweep finding B7).

**Estimated sprints:** 1.

**Risk:** medium — the `__gpuBuffer` rewrite is the load-bearing piece. If
RC's three.js storage-attribute integration was the _only_ way to get the
data on GPU, replacing it with raw buffers means writing the data plumbing
ourselves. Test: standalone-render integration test catches this.

**Depends on:** W2-C1 (BVH canonicalisation — the new package imports the
canonical traversal), W2-C3 (octahedral fix), W4-A1 (HybridEngine
slimming clears the RC removal hook), W7-H1 (applyDDGIShading move out of
rc/).

---

### W9 — Scaffold-2: PPG GPU dTree traversal finish

**Goal:** PPG `ppgEnabled: true` actually does path guiding. The
production code path matches the CPU side that's already well-built.

**Sub-tasks:**

1. **Write `serialiseDTree(dTree): Float32Array`** — CPU producer. Output
   layout: per-cell header (`leafCount`, `totalFlux`, `pad0`, `pad1`)
   followed by N leaves of `(uvMinU, uvMinV, uvMaxU, uvMaxV, flux,
solidAng, _pad, _pad)` packed as f32 octets. Documented at the top of
   `ppg/dTree.ts`. Add tests: shape-correctness, round-trip via a CPU
   deserialiser (already exists conceptually — extract).
2. **Write GPU sTree GPU buffer producer** — CPU `sTree.ts` serialise
   step: pack each cell's centroid + extents + (offset, leafCount) into a
   storage buffer. Bind under a new `ppg/types.ts` layout
   `PpgSTreeBufferEntry`.
3. **Replace uniform-grid stub at `ppgGuide.wgsl.ts:122-132`** — currently
   `col = leafIdx % u32(sqrt(N))`. Replace with: sample leaf proportional
   to flux (already correct), then read the leaf's actual `uvMin`/`uvMax`
   from the serialised dTree buffer, jitter inside that rectangle. PDF
   stays `(leafFlux / totalFlux) / solidAng`.
4. **Replace uniform-grid stub at `ppgUpdate.wgsl.ts:100-114`** — currently
   `uIdx = u32(clamp(uv.x * leafCount, ...))`. Replace with: locate the
   actual leaf containing `uv` via tree traversal (the serialised dTree
   is a quadtree per cell; binary-search-style descent in a flat array).
   Add `vIdx` consumption (currently unused, line 114 says "reserved for
   2-D leaf indexing in a future tree serialisation pass" — that future
   is now).
5. **Plumb sTree GPU buffer from HybridEngine** — `HybridEngine` calls
   `serialiseSTree(sTree)` after each rebuild cycle, uploads to
   `device.queue.writeBuffer`, binds via a Pass module (per W1's pattern).
6. **Replace `dispatchWorkgroups(0, 0, 0)` stubs at
   WalkaroundGPUPipeline.ts:884, 907** with real dispatches: one
   workgroup per pixel for guide, one per L_i sample for update.
7. **PPG MIS path in shade.wgsl** — for the indirect bounce, sample
   direction from PPG when `ppgEnabled`; combine with BSDF sample via
   power-heuristic one-sample MIS (the `bsdf-only` fallback is already
   the current code path, so we keep it as the α=0 case).
8. **Add tests** — `ppgGuide.test.ts` and `ppgUpdate.test.ts` exercising
   serialiseDTree round-trip, GPU traversal correctness via CPU oracle,
   PDF non-negativity, MIS combine correctness against ground truth.

**File paths:** `packages/walkaround-hybrid/src/ppg/{dTree,sTree,
ppgGuide.wgsl,ppgUpdate.wgsl,types,serialise,passModule}.ts`,
`packages/walkaround-hybrid/src/HybridEngine.ts`,
`packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`.

**Test plan:**

- Pre-existing CPU `ppg/dTree.test.ts`, `ppg/sTree.test.ts` continue to
  pass.
- New GPU-side tests, CPU-oracle parity.
- Integration: run path-guided render on a hard scene (caustics via
  point light through a glass cell) and verify variance reduces vs BSDF-only
  baseline at equal SPP.

**Reference renders:**

- `tools/reference-renders/W9-pre-ppg-off/scene-caustics-1080p-1024spp.png`
- `tools/reference-renders/W9-post-ppg-on/scene-caustics-1080p-1024spp.png`
- Variance-reduction expected; document the ratio in the PR body.

**Estimated sprints:** 2.

**Risk:** medium — the GPU dTree traversal is novel-to-this-codebase. The
CPU oracle catches algorithmic errors; reference renders catch visual
issues.

**Depends on:** W1 (Pass/registry — PPG becomes a real Pass module), W4
(Pass-loop renderFrame so dispatching is declarative), W5 (per-pass BGL so
sTree+dTree bindings don't pollute shared layouts).

---

### W10 — Scaffold-3: Neural denoiser finish

**Goal:** `denoiser: 'neural'` mode actually runs the U-Net. The 137-line
shape-debate goes away. `dispose()` no longer leaks. An example wires it
end to end.

**Sub-tasks:**

1. **Wire INPUT_PACKER_WGSL into InferenceGraph.\_runInputPack** — replace
   the 3 planar `copyBufferToBuffer` calls (currently incorrect layout
   per F2) with a `device.createComputePipeline` over `INPUT_PACKER_WGSL`;
   bind noisy/albedo/normals; output to enc_input in interleaved layout
   matching what `unetArchitecture.ts` expects.
2. **Fix InferenceGraph.dispose()** — destroy every weights buffer, biases
   buffer, and per-layer uniform buffer. Currently leaks all of them
   (admitted in `InferenceGraph.ts` comment). Add a leak test:
   construct/dispose 100 graphs, assert `device.queue` reports no
   outstanding buffers.
3. **Strip the 137-line shape-debate at unetArchitecture.ts:80-217.** Keep
   ~10 lines of resolved narrative (the FINAL LAYER GRAPH at lines
   202-216 documenting the adopted architecture) and the architectural
   citation. Move the resolved-debate to a one-time architectural decision
   record in `packages/walkaround-hybrid/notes/unet-shape-resolution.md`.
4. **Verify and fix skip-connection shape mismatches** — per the 05-11
   sweep, skip connections at every level pair H/N with H/2N. Walk
   through the FINAL LAYER GRAPH the code adopts, verify each skip-add at
   runtime. The current architecture (post-debate-resolution) claims
   skips match — verify with a shape-assertion test.
5. **Verify bind-group bindings match shader bindings** — 05-11 sweep
   says "bind-group binding indices don't match shader bindings". Read
   each WGSL shader, read the BG builder, assert each binding pair.
6. **Verify uniform buffer is written** — 05-11 sweep says "uniform
   buffer is allocated but never written". Add the write.
7. **Add a `'neural'` mode example** — `examples/neural-denoiser/`
   showing a noisy 1spp render denoised by the U-Net. Ships with a
   minimal trained-weights file (or random weights for build verification
   — the denoising won't be useful but the pipeline will run).
8. **Demote internal-only debug exports.**

**File paths:**

- `packages/walkaround-hybrid/src/neural/{InferenceGraph,
inputPacker,unetArchitecture,weights}.ts`
- `packages/walkaround-hybrid/notes/unet-shape-resolution.md` (new)
- `examples/neural-denoiser/{package.json,index.html,src/**}` (new)

**Test plan:**

- `InferenceGraph.dispose()` no-leak test.
- `inputPacker.test.ts` — pack 3 separate buffers, assert interleaved
  output matches CPU oracle.
- Shape-assertion test per skip-connection pair.
- BG-binding match test.
- Example renders successfully (build-time test).

**Reference renders:**

- `tools/reference-renders/W10-pre/cornell-box-1spp.png`
- `tools/reference-renders/W10-post-neural/cornell-box-1spp-denoised.png`
- The denoised result with random weights won't look great — expected.
  Smoke-test only.

**Estimated sprints:** 2 (mostly because the architecture may need
revisits if shape mismatches reveal deeper bugs).

**Risk:** **high.** This is the workstream most likely to reveal "the
scaffold doesn't quite work and the architecture itself needs revision"
discoveries. Mitigation: time-box the architecture-rework subtask; if it
balloons, file a separate stream and ship the rest of W10 (dispose fix +
comment strip + input packer) cleanly. Per CLAUDE.md "no half-implementations"
— do not ship `'neural'` mode as "kind of works"; if it doesn't fully work,
explicitly mark the example as a build-smoke-test and document the gap.

**Depends on:** W1, W3 (denoiser DU), W5 (per-pass BGL).

---

### W11 — Scaffold-4: OIDN `'oidn-final'` wire

**Goal:** `denoiser: 'oidn-final'` is a real mode in both HybridEngine
and pt-webgl.

**Sub-tasks:**

1. **HybridEngine** — add the dispatch path. After the standard denoise
   chain (atrous or svgf-real), if `denoiser.kind === 'oidn-final'`,
   readback HDR buffer → call
   `oidnBridge.denoiseFinal({color, albedo, normal})` → upload the
   denoised result to a presentation texture → present.
2. **pt-webgl PTEngineWebGL2** — analogous: after sample accumulation,
   readback the HDR target → call OIDN → present. This is more natural
   because pt-webgl already has `readbackHdr.ts`.
3. **Demote underscored test-only exports** — `_hwcToNchw`,
   `_nchwToHwc` move out of `oidnBridge.ts` public surface; tests import
   from a `__test__` re-export.
4. **Denoiser registry** — `oidn-final` is a real entry in W1's registry.
5. **Document the modelUrl convention** — `oidn_rt_hdr.onnx`,
   `oidn_rt_hdr_alb_nrm.onnx`, etc. README in shared-denoisers.
6. **Add an example** — `examples/oidn-final/` showing 16spp + OIDN final
   pass vs 16spp without.

**File paths:**

- `packages/walkaround-hybrid/src/HybridEngine.ts`
- `packages/walkaround-hybrid/src/pipeline/denoisers/oidnFinal.ts` (new, per W1)
- `packages/pt-webgl/src/ptEngineWebGL2.ts`
- `packages/shared-denoisers/src/{oidnBridge,index}.ts`
- `examples/oidn-final/**` (new)

**Test plan:**

- Integration: render hero scene at 16spp + OIDN; compare against a 1024spp
  reference. Should match within bounded PSNR.
- Demote test: `_hwcToNchw` is not in `index.ts` exports;
  `__test__/oidnBridge.test.ts` imports from internal path.

**Reference renders:**

- `tools/reference-renders/W11/hero-1024spp-reference.png`
- `tools/reference-renders/W11/hero-16spp-oidn.png`
- PSNR / SSIM committed in PR body.

**Estimated sprints:** 1.

**Risk:** low. `oidnBridge.ts` is code-complete; this is purely wiring.

**Depends on:** W3-D4 (denoiser DU), W1 (registry).

---

### W12 — Scaffold-5: Dev overlays finish

**Goal:** the 4 stub overlay components (DDGIAtlasViewer, BVHVisualizer,
GISignalSplit, MaterialInspector) become real, useful debug UI.

**Sub-tasks:**

1. **DDGIAtlasViewer** — wire to `engine.debug.atlasTexture()` (already
   implemented per HybridEngine.ts:1238). Per-frame: allocate a 2D canvas
   overlay, blit the GPUTexture via copyTextureToBuffer → readback →
   draw to canvas. Click on probe cell → emit `onProbeSelected(cellId)`.
2. **BVHVisualizer** — wire to `engine.debug.bvhNodes()` (already
   implemented per HybridEngine.ts:1246). Render AABB wireframes
   overlaid on the 3D scene; color by depth (the `depth=0` placeholder in
   `bvhNodes` becomes a real depth via parent-traversal — small fix in
   HybridEngine.ts:1268).
3. **GISignalSplit** — wire to `engine.debug.giSignalTextures()` (already
   implemented per HybridEngine.ts:1273-1294). Split-screen 4-up: direct,
   indirect, ao, total.
4. **MaterialInspector picker** — wire to `engine.debug.pickPrimitive(x,
y)`. Currently NOT implemented in HybridEngine.debug; add the
   implementation: read the gNormalDepth texture's primitive-id channel
   at (x, y). Click on screen → show material editor for the picked
   material.

**File paths:**

- `packages/dev/src/react/{DDGIAtlasViewer,BVHVisualizer,GISignalSplit,
MaterialInspector}.tsx`
- `packages/walkaround-hybrid/src/HybridEngine.ts` (add `pickPrimitive`
  - parent-link traversal for bvhNodes depth)

**Test plan:**

- Each overlay has a render test (happy-dom + minimal mock engine).
- MaterialInspector picker has a coord → primitive-id test.

**Reference renders:** none required (UI components, not visual renderer
changes).

**Estimated sprints:** 1.

**Risk:** low. Surface already exists for 3 of 4; picker requires the
gNormalDepth primitive-id readback which is a one-pass addition.

**Depends on:** W3 (debug surface in the contract; already there per
commit 38463d1).

---

### W13 — Documentation reconciliation

**Goal:** every doc reflects shipped reality. Stale memory archived. Plan
folder cleaned.

**Sub-tasks:**

1. **CLAUDE.md** — update Phase/Sprint counts (Sprint 13 → 18+W{N}),
   correct test count, mention `pt-webgpu` in the package list, refresh
   "What's next." Remove the "PPG hard-throws" claim (false per 05-17
   sweep finding G10) and the "no `'neural'` mode" claim (false per G11).
2. **CHANGELOG.md** — add the missing ~30 entries since "Sprint 11 PPG
   dispatch": Sprints 14-18, ReSTIR-GI temporal/spatial, per-channel
   SVGF, GTAO, firefly+dim-magnitude root-cause fix, adaptive sampling
   wiring. Plus an entry per workstream W1-W12 as they land.
3. **plan/** — archive the ~40 completed-sprint files to `plan/archive/`
   per the 05-11 sweep recommendation. Retain `library-architecture.md`,
   `phase-7-restir-gi.md`, `walkaround-without-three.md`,
   `renderer-fidelity-matrix.md`, `generalized-library-milestones.md`,
   `binding-babylon-sketch.md`, `d2-e6-pt-webgpu-ppg-performance.md`,
   `pt-webgpu-deep-audit.md`, `hardware-gpu-validation-spec.md`. Mark
   Phase-6 docs "closed".
4. **memory/** — archive the 6 stainedGlass-referencing `project_*` files
   as `project_*_archived.md`; retain `project_design_principles.md` +
   `feedback_*` set + the two sweep memory entries.
5. **in-flight files** — delete `in-flight-plan.md` (ReSTIR-GI claim is
   false; work shipped in commits 1203df9 / 7d307f6).
6. **Worktree branches** — delete `worktree-agent-a06812aa6c5b09b98`,
   `worktree-agent-a193d8b569f806c50`.
7. **RFE trackers** — reconcile `plan/external-requests-status.md` and
   `external_requests/IMPLEMENTATION-STATUS.md` to one canonical doc.
8. **`CREDITS.md`** — re-audit; ensure every algorithm has its prior-art
   citation. Particularly: PPG (Müller 2017), GTAO (Jiménez 2016), OIDN
   (Áfra 2019), ReSTIR-GI (Ouyang 2021), per-channel SVGF (Schied 2017),
   Estévez-Kulla 2018 (light tree — IF the actual implementation is
   Shirley-1996 Median-split per the 05-11 sweep finding, the CREDITS
   entry should reflect the _true_ implementation).
9. **READMEs** — every package's README aligned to its true state. The
   walkaround-hybrid README in particular gets the W8 RC-removal update.

**File paths:** `CLAUDE.md`, `CHANGELOG.md`, `plan/**`, `memory/**`,
`CREDITS.md`, every `packages/*/README.md`, every `examples/*/README.md`.

**Test plan:** none — doc-only.

**Reference renders:** none.

**Estimated sprints:** 1.

**Risk:** none.

---

## 4. Parallel-vs-serial map

```
WAVE 1 (parallel, ~2 sprints)
├── W1: Pass/Resource/Denoiser registry      [first PR, foundational]
├── W2: Dedup canonicalisation               [parallel with W1, W3, W7]
├── W3: Contract hygiene                     [parallel with W1, W2, W7]
└── W7: Stale + misplaced + dead-code        [parallel, opportunistic]

WAVE 2 (parallel, ~2-3 sprints; depends on Wave 1)
├── W4: God-file dissolution                 [depends on W1]
├── W5: Per-pass BGL                         [depends on W1]
└── W6: Hidden-globals de-singletonization   [depends on W3]

WAVE 3 (parallel, ~1-2 sprints; depends on Wave 2)
├── W8: RC extraction                        [depends on W2-C1/C3, W4-A1, W7-H1]
├── W9: PPG GPU dTree finish                 [depends on W1, W4, W5]
├── W11: OIDN 'oidn-final' wire              [depends on W1, W3]
└── W12: Dev overlays finish                 [depends on W3 — already shipped]

WAVE 4 (~2 sprints)
├── W10: Neural denoiser finish              [depends on W1, W3, W5; highest risk]
└── W13: Documentation reconciliation        [depends on everything]
```

**Total estimated sprints:** 8-10 across 13 workstreams. Some streams can
collapse into single sprints when small; some (W4 god-file dissolution,
W10 neural) span 2 sprints each.

---

## 5. Audit checkpoints (per plan-implementation skill rules — non-negotiable)

After each wave: run `/audit` on all changed files. Fix structural debt
(remnant god-files, mixed concerns, public export sprawl) before starting
the next wave. Structural debt compounds — catching at the wave boundary
is orders of magnitude cheaper than after Wave 4.

- **End of Wave 1:** run `/audit` on W1's new Pass/Registry modules; W2's
  canonicalisation imports; W3's contract changes; W7's cleanup. Fix any
  remaining god-export, mixed-concern, or stale-comment debt before
  Wave 2.
- **End of Wave 2:** run `/audit` on the dissolved god-files. Confirm no
  module exceeds 300 LOC (excluding WGSL strings); confirm
  `WalkaroundGPUPipeline` is now a thin orchestrator; confirm no
  remaining `as unknown as` private-state reach-throughs.
- **End of Wave 3:** run `/audit` on the 4 finished scaffolds; confirm
  public surface is uniform, examples build, integration tests pass.
- **End of Wave 4:** run `/audit` on neural scaffold (most likely to need
  revisits) and confirm docs are aligned.

---

## 6. First PR — concrete next action

**W1 — Pass + Resource + Denoiser registry abstraction.**

Rationale: every other workstream either depends on it (W4, W5, W9, W10,
W11) or is unblocked-but-faster-after-it (W2 dedup canonicalisations land
into well-named module homes; W3 contract changes are simpler when the
denoiser registry consumes the DU). Starting elsewhere — e.g., diving
straight into W4 god-file dissolution — risks producing smaller god-files
without the abstractions that prevent regrowth.

**Branch:** `feat/refactor-pass-registry-w1`
**Worktree:** `worktree-w1-pass-registry` (per CLAUDE.md user has
standing discretion on worktree placement; place at `~/projects/vitrum-w1-pass-registry/`).
**Tag (on merge):** `refactor/W1-pass-registry`.

**Acceptance criteria for the first PR:**

1. `Pass` interface + `PassRegistry` exist and are used by at least one
   pass (start with `shade` — the central pass).
2. `Denoiser` discriminated-union type defined in `@vitrum/core`.
3. `FrameResources` split into `.ddgi`, `.svgf`, `.gtao`, etc. sub-structs
   (even if `WalkaroundGPUPipeline` still does the orchestration directly
   for non-`shade` passes — full migration in W4).
4. All ~660 tests pass; typecheck clean.
5. Hero reference render bit-identical before/after.

---

## 7. Risk register

| Risk                                                                                                  | Likelihood                  | Impact                                          | Mitigation                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W10 neural reveals architectural bugs beyond shape mismatches                                         | medium-high                 | scope creep                                     | time-box the architecture-rework subtask; if it balloons, ship the dispose-fix + input-packer-wire + comment-strip cleanly and explicitly mark `'neural'` mode as build-smoke-only in the example README. Do NOT ship "kind of works" — that's a band-aid per user preferences. |
| W4 god-file dissolution silently breaks visual output                                                 | high (volume of moved code) | high (looks like W4 worked but image regressed) | mandatory before/after reference renders per sub-task; bit-equality check in CI                                                                                                                                                                                                 |
| W8 RC `__gpuBuffer` rewrite turns out to need three.js storage-attribute integration we can't replace | low                         | medium                                          | standalone-render integration test; if the integration was load-bearing, document it as a limitation and explore an alternative data-marshaling path in scope                                                                                                                   |
| W1 Pass abstraction doesn't fit ReSTIR/PPG (multi-pass per algorithm with internal state)             | medium                      | medium                                          | design Pass interface with `PassGroup` extension upfront; first-class support for multi-pass algorithms. Validate with ReSTIR-DI (3 passes — RIS + temporal + spatial) before locking.                                                                                          |
| W3 contract changes break stained-glass host app (pre-extraction consumer)                            | low (extracted in 05-12)    | low                                             | the stained-glass app is no longer a consumer per CLAUDE.md; verify in W3-D2 (AnalyticShape `h-channel-came`)                                                                                                                                                                   |
| W6 sharedWebGpuDevice removal breaks denoiser drivers in their test harnesses                         | medium                      | low                                             | drivers accept passed device; tests pass a vitest-managed device. Standard refactor.                                                                                                                                                                                            |
| W2 C5 MaterialEntry unification changes default roughness silently (0.5 vs 1.0)                       | high                        | medium                                          | reference renders catch this; document the chosen default (probably 0.5 per the more common usage) in PR body                                                                                                                                                                   |
| Cross-stream interleaving (W2-C3 + W8 race)                                                           | low                         | medium                                          | sequence W2-C3 before W8 explicitly (W7 dependency note)                                                                                                                                                                                                                        |

---

## 8. Out-of-scope reminders

Do not relitigate or fold these into this plan:

- Numerical/math/physics bugs from `in-flight-sweep-20260511.md`. They have
  their own tracking. Some refactors (W2 C3, W4-A4 BSDF split) will
  _expose_ them; if so, file in the 05-11 tracker and coordinate; do not
  fix in-stream unless the fix is a one-line opportunistic improvement.
- Upstream PRs to `gkjohnson/three-gpu-pathtracer`.
- npm publish.
- Remote pushes without explicit user instruction.
- SOTA-cargo-cult additions (per CLAUDE.md): every new algorithm requires
  verified-feasibility (public source, language portable to web, not
  RTX-hardware-locked) before scheduling. This plan introduces NO new
  algorithms; it only finishes the ones already in tree.

---

## 9. Definition of done (whole plan)

All of:

- 13 workstreams merged to main; each via its own PR; each with green CI.
- Workspace `npm run typecheck` clean; `npm test` ~660+ tests pass (the
  2-3 GPU-only skipped tests stay skipped — verify same set).
- Reference renders for every visual-output-affecting change captured and
  A/B'd; numerical regressions are documented and visually justified or
  reverted.
- `CLAUDE.md`, `CHANGELOG.md`, `CREDITS.md`, every `README.md`, all
  `plan/*` docs aligned to shipped state.
- The 4 scaffold subsystems (RC extracted, PPG finished, Neural finished,
  OIDN wired) all have working integration tests and at least one
  example each.
- `/audit` clean at each wave boundary and at final merge.

---

## 10. Appendix — pre-flight verification done before drafting this plan

To honor the CLAUDE.md "scans are not investigation — always READ the
files" principle, the following files were directly read before locking
sub-task specifics:

- `packages/walkaround-hybrid/src/ppg/ppgGuide.wgsl.ts:110-139` —
  confirmed uniform-grid stub at line 122-132.
- `packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts:85-115` —
  confirmed uniform-grid stub at line 100-114; `vIdx` reserved.
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:878-910`
  — confirmed `dispatchWorkgroups(0, 0, 0)` at 884 and 907.
- `packages/walkaround-hybrid/src/neural/unetArchitecture.ts:70-217` —
  confirmed 137 lines of shape-debate ending in "FINAL LAYER GRAPH"
  resolution.
- `packages/walkaround-hybrid/src/neural/InferenceGraph.ts:265-275, 625-660`
  — confirmed `_runInputPack` does planar copy; `inputPacker.ts` is
  referenced only in comments.
- `packages/shared-denoisers/src/oidnBridge.ts:154-250, 336-380` —
  confirmed `denoiseFinal`/`preloadOIDNModel` complete; `_hwcToNchw` /
  `_nchwToHwc` exported from src.
- `packages/dev/src/react/DDGIAtlasViewer.tsx` — confirmed stub state +
  TODO note pointing at engine.debug.atlasTexture.
- `packages/walkaround-hybrid/src/HybridEngine.ts:1235-1295` — confirmed
  `engine.debug` IS implemented (atlasTexture, visibilityAtlasTexture,
  bvhNodes, giSignalTextures). pickPrimitive missing — flagged in W12.
- `packages/core/src/engine.ts:200-235` — confirmed `EngineDebugSurface`
  interface in contract.
- `packages/walkaround-hybrid/src/rc/cascadeDispatch.ts:175-205` —
  confirmed `gpuBufferOf` reach-through into `attr.__gpuBuffer`.
- Workspace file sizes verified match sweep table (1743, 1574, 813, 485,
  857, 1037, 838, 1908 LOC for the 8 god-files).
- Git branch listing confirms 2 stale worktree branches present.
- Existing recent commits: most recent meaningful structural commit is
  `38463d1 feat(core,walkaround-hybrid,dev,engine): T3.G followup — wire
engine.debug surface` — confirms the debug surface IS shipped.

This pre-flight investigation took ~2 dozen file reads. The plan is
grounded in the current code state, not the sweep findings alone.
