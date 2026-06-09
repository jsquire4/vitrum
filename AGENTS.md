# Agent brief — vitrum

> If you're a new Codex agent working in this repo, **read this first.**

## What this project is

`vitrum` is a WebGPU + WebGL2 path tracing & global illumination engine for the browser. The white-whale ambition is to own the entire SOTA-browser-rendering stack — from BVH construction to physically-based path tracing to real-time global illumination to denoising — under one consistent, host-agnostic API contract.

## Where you are right now

Read in this order to onboard:

1. `README.md` — package architecture overview
2. `plan/library-architecture.md` — package responsibilities, dependencies
3. `plan/archive/phase-7-restir-gi-archived-2026-05-28.md` — historical Phase-7 walkaround GI record (Sprints 15–18 shipped: GTAO, ReSTIR-GI RIS/temporal/spatial, per-channel SVGF)
4. `packages/core/src/scene/` + `src/frame.ts` + `src/engine/` (scene and engine are directories since the A-6/A-7 splits) — the locked-in API contract types
5. `CREDITS.md` — attribution to ~30 prior works the engine builds on
6. The other plan/ docs (`library-architecture.md`, `walkaround-without-three.md`, `renderer-fidelity-matrix.md`, `roadmap.md`, `premium-grade-refactor-20260517.md`) are the active docs. Completed-sprint artifacts live in `plan/archive/` (the 2026-05-28 + 2026-05-30 context sweeps archived completed docs there, including `backend-maturity-matrix-2026-05-26`, `w8-rc-mis-composition`, and the superseded `complexity-remediation-20260526` draft).

## What's done

- **2026-05-24 reconciliation:** merge-race backlog items E1-E7 were re-landed.
  Treat any older "NOT IN HEAD" caveats below as historical audit notes.

- **Phase 6 (Sprints 0–13) complete**; **Phase 7 walkaround-hybrid (Sprints 14–18) shipped**: layered BSDF fork patch, half-res GTAO + bilateral upsample (S15), ReSTIR-GI RIS (S16), ReSTIR-GI temporal+spatial reuse (S17), per-channel SVGF on direct + indirect (S18), plus extensive firefly / dim-magnitude root-cause work and library-generality remediation. Workspace `tsc --noEmit` clean; **all** vitest tests pass across workspaces — the 2–3 previously-skipped GPU-only paths were enabled via happy-dom / vitest browser mode; GPU-browser tests are opt-in via env flag so default `npm test` no longer requires Playwright.
- **Packages**: `core`, `shared-bvh`, `shared-samplers` (light tree, BDPT, spectral), `shared-denoisers` (à-trous-variance, `svgf-real` Schied 2017, OIDN bridge), `pt-webgl2` (native WebGL2 converged PT — release-candidate track), `pt-webgpu` (WebGPU-native PT peer), `walkaround-rc` (Radiance Cascades subsystem — cascade pyramid + dispatch), `walkaround-hybrid` (realtime GI — release-candidate track: DDGI + ReSTIR-DI/GI + GTAO + SVGF + opt-in RC/PPG/neural), `engine` (`createEngine` / `attachVitrum` facade), `stained-glass-extensions` (stained-glass-specific contracts), `dev` (debug overlays).
- **Extraction**: `_staging/legacy-source/` contains only host-app React/Redux files intentionally not extracted (see `_staging/README.md`).
- **External RFEs**: 01–05 (contract-layer) plus 06/07/08/09/10/12/14 fork patches applied per `external_requests/IMPLEMENTATION-STATUS.md`.
- **M7 — DDGI Coherent Physical Model shipped** (sweep 2026-05-11 Items 2, 4, 6, 20): producer no longer pre-multiplies `albedo/π`; Lambertian cosine blend kernel (`w` for irradiance, `w²` for visibility) replaces `pow(w,8)` / `pow(w,50)`; per-frame Halton-Shoemake SO(3) `randomRotation`; RC GI receiver applies `materialColor · PI_INV` before injecting via `emissiveNode`. Cited at `probeUpdateFrameParams.ts:19-60` (Halton-{2,3,5} sequence) and `applyDDGIShading.ts:172-175` (receiver applies `albedo · PI_INV` exactly once).
- **`svgf-real` (Schied 2017) denoiser pipeline shipped** (`@vitrum/shared-denoisers` + `@vitrum/walkaround-hybrid`): real reprojection + variance-from-moments + 7×7 spatial fallback + à-trous wavelet ×5; opt-in via `denoiser: 'svgf-real'` alongside the default `'atrous-variance'` mode.
- **`HybridEngine.updateLighting()` + `HybridEngine.setSize()`** runtime APIs shipped (no engine recreation for time-of-day scrubbing or canvas resize).
- **Neural U-Net denoiser revived** (T2.H2 + B3 + F3 + F4): real `'neural'` mode in `HybridEngineOptions.denoiser`; INTERLEAVED `inputPacker.ts` compute shader now wired into `InferenceGraph._runInputPack` (the previous planar-vs-interleaved layout mismatch was real and is fixed); `dispose()` releases weight/bias/uniform buffers (F4); 137-line shape-debate header stripped from `unetArchitecture.ts` (F3); `train.py` / `export_weights.py` are real PyTorch tools.
- **A3 shipped — `HybridEngine.updatePrimitive` + `updateEmitter`** (`feat/a3-hybridengine-incremental-updates`, `d0d22b0`): material-only fast path re-uploads one slot in the materials texture without touching DDGI or the temporal accumulator. Geometry-change case throws explicitly (BVH leaf rebuild left as a follow-up). Unblocks a walkaround SceneSync in stainedGlass for material edits.

- **C1 shipped — SkinnedMesh contract + per-frame CPU solver + morph targets** (`7d0f4d7` + `f3a91ce`, 2026-05-19): `SkinnedMeshPrimitive` added to `@vitrum/core`'s discriminated union; `sceneFromThreeJS` converts THREE.SkinnedMesh; `@vitrum/three-bindings`'s `solveSkin` does LBS with morph-delta pre-blend. Per-frame pose updates flow through `engine.updatePrimitive(id, { positions, normals })`. Downstream consumers (`sceneAABB`, `_coreSceneSuppliesMeshes`, `summarizeScene`, `vitrumSceneToThree`) accept skinned meshes. 19 tests pin the math + adapter shape. GPU compute variant + inverse-transpose normal for scaled bones deferred — baseline already enables real-time single-hero-character skinning.

- **C2 — TLAS pipeline (2026-05-27 deepening):** `packSceneFromCore` + `fingerprintBuffers`; hybrid ReSTIR/DDGI/RC share TLAS+BLAS traversal; `RCSubsystem.syncRestirBvhBuffers`; incremental refit via `rebuildPrimitiveBlas` / `refitTlasTransforms`. CPU: `buildTlas` / `refitTlas` / `tlasIntersect` (`5cf642b`). pt-webgpu uses the same packer (`WG-6`).

### W1 — Pass + Resource + Denoiser registry refactor (all 6 rounds shipped)

- R1 — Pass + Denoiser registry foundation (`11c4698`, `feat/refactor-w1r1-pass-registry-foundation`).
- R2 — FrameResources god-struct split into 8 per-algorithm sub-structs (`834f441`, `feat/refactor-w1r2-frame-resources-split`).
- R3+R4 — denoiser migration to self-contained registry entries (`987b3fb`, `feat/refactor-w1r3r4-denoiser-migration`).
- R5 — declarative `PASS_ORDER` via `PassRegistry`; 18 self-contained Pass classes (21 today, after later passes); `renderFrame` becomes a registry-iteration loop (`60dd48b`, `feat/refactor-w1r5-declarative-pass-order`).
- R6 — declarative WGSL include-graph; 27-entry `WGSL_MODULES` registry (54 today); `composeWgsl` topo-sort replaces 9 hand-rolled `COMMON_WGSL + X_WGSL` concat patterns (`29ccf96`, `feat/refactor-w1r6-wgsl-include-graph`).
- Aggregate: `WalkaroundGPUPipeline.ts` 1574 → 964 LOC at W1 close (-39%; has since grown with later passes, ~1534 today); 354 → 424 tests; typecheck clean every round.

### W2 — shared-package dedup / hoisting

- C1 — BVH traversal WGSL canonicalised into `shared-bvh` (`feat/w2-geometry-primitives-c1-c3-c4`, `1a07b2b`).
- C2 — THREE-independent `buildArrayBvh` hoisted from `pt-webgpu` into `shared-bvh` (`feat/w2-c2-bvh-builder-shared`, `44598ad`).
- C3 — canonical octEncode/octDecode (was buggy `sign(0)→0` collapse) (`8fec673`).
- C4 — `probeAtlasUv` reused instead of 3 inline atlas-coord derivations (`3446817`).
- C6 — shared WGSL primitives (`pcg.wgsl.ts`, `bsdfPrimitives.wgsl.ts`) re-landed 2026-05-24 (merge-race E4 closed).
- C15 — pixel-hash function canonicalised into `shared-samplers/wgsl/hash` (`651619e`).
- C10 — Rec.709 luminance vector canonicalised across `shared-samplers` / `shared-denoisers` / `walkaround-hybrid` (`feat/w2-small-dedups-c10-c11-c12`, `54ae795`).
- C11 — duplicated B3-spline atrous kernel collapsed (`76120b9`).
- C12 — `GTAOUniforms` struct deduped across `gtao` + `gtaoUpsample` shaders (`8a40ffe`).

> **Note — May 17 merge-race losses were all re-landed (2026-05-24; re-verified by code-read 2026-05-28).** The 2026-05-17 sprint had several feature commits silently dropped from HEAD by merge races (W2-C6, W3-D6, W3-D7, W3-D19, W6-E1, W6-E6, W7-G5). All have since been re-applied and are present in HEAD — the per-item bullets below were updated to reflect this. The old inline "NOT IN HEAD" labels on D6/D7/D19/E1/E6 were stale and have been corrected.

### W3 — contract hygiene (surgical core/* moves)

- D5/D13/D15 — stopped re-exporting raw WGSL strings (internal-only now); moved underscored test-only exports off public surface (`feat/w3-contract-hygiene-d5-d6-d13-d15`, `e845cc5..becfbae`).
- D6 — **in HEAD (re-landed 2026-05-24).** `Mat4` is branded: `packages/core/src/scene/math.ts:11` is `export type Mat4 = Float32Array & { readonly [MAT4_BRAND]: 'Mat4' }`, with `asMat4()` + `isMat4()` guards.
- D7 — **in HEAD (re-landed 2026-05-24).** `FrameOutput` is the `{kind:'skipped'|'rendered'}` discriminated union `FrameSkipped | FrameRendered` at `packages/core/src/frame.ts:150-185`.
- D8 — typed `EngineCapabilities`-driven feature query replaces `typeof` checks across `core` + `engine` (`feat/w3-d8-engine-capabilities`, `1355be1`).
- D16 — canonical `FrameStats` + `qualityModes` capability in `@vitrum/core`; `pt-webgpu` now emits `onFrame` stats (`feat/w3-d16-uniform-telemetry`, `197510c..8c96e4b`).
- D17 — verified `Material.extensions` IS used by `three-bindings` dichroic LUT — keep (`80c2388`); D18 — dropped `supportsMotionBlur` + `FrameInput.shutterTime` (no consumer in roadmap) (`9ea12c9`).
- D19 — **in HEAD (re-landed 2026-05-24).** `BackendTexture<TBackend, THandle>` nominal brand + `asBackendTexture` / `asBackendTextureFormat` / `narrowToBackendTexture` / `narrowToBackendTextureFormat` helpers at `packages/core/src/frame.ts:191-228`.

### W4 — god-file dissolution (first wave)

- A4 — `pt-webgpu` `pathTraceBruteforce.wgsl.ts` split (1908 → 76 LOC orchestrator + 15 structured modules) (`feat/w4-a4-pt-webgpu-shader-split`, `f640de5`).

### W6 — hidden-globals de-singletonization

- E1 — **in HEAD (re-landed 2026-05-24).** `reuseSharedWebGpuDevice` is opt-in (`=== true`) in all three shared-denoiser dispatchers (`svgfRealWebGPU.ts:195`, `atrousVarianceWebGPU.ts:232`, `hdrLuminanceBilateralWebGPU.ts:58`).
- E2 — module-level `iblBaker` cache replaced with per-engine `IblBakerCache` instance (`feat/w6-hidden-globals-pt-webgl`, `6a5106f`).
- E3 — `window.__WGPU__` write removed from `HybridEngine.renderFrame` (use `onFrame`) (`7d85bb0`).
- E6 — **in HEAD (re-landed 2026-05-24).** `packages/pt-webgl/src/forkAccess.ts` exists (`ForkAccess` static class); `forkUniformBridge.ts:115` + `ptEngineWebGL2.ts:1141,1283` go through `ForkAccess.getMaterial` / `ForkAccess.getRenderTexture`.

### W7 — dead-code + misplaced-code cleanup

- Dead-code batch (G2/G3/G5/G7/G9 — all in HEAD as of 2026-05-19) (`feat/w7-dead-code-batch`): deleted deprecated `bdptConnectionMIS_partial` / `buildBDPTStrategyPDFs_partial` (G2 ✓); `rgbToApproxSpectralCoefficients` aliasing (G3 ✓); un-exported `validateBvhEncoding` from `shared-bvh/index.ts` (G5 — was a merge-race casualty per `items_to_fix.md` E7, **re-applied 2026-05-19**); un-exported internal helpers `extractAttribute`/`extractIndex`/`warnOnce` from three-bindings (G7 ✓); deleted redundant emitter-constants re-exports from `uploadSceneBuffers` (G9 ✓).
- Misplaced-code moves (H1/H4/H5/H7/H8) (`feat/w7-misplaced-code-moves`): albedo demodulate/remodulate extracted to its own file (H8, `84f474b`); BDPT bounce-budget constants moved to pt-webgl (H7, `bf5e524`); Halton-Shoemake quaternion sampler moved to shared-samplers (H5, `bae8066`); plus H1, H4.
- Tail (G6/G8/I) (`feat/w7-tail-g6-g8-stale`): `SPATIAL_FILTER_WGSL` exported from package index (G6, `1d27529`); shared-samplers exports triaged + test-oracle items marked `@internal` (G8, `539b049`); stale "post-Sprint X" comments reworded (I, `9199dcd`).
- H2/H3 (`feat/w7-h2-h3-plus-future-doc-accuracy`): DDGI-specific atlas-octahedral helpers + `SceneBvh` wrapper class moved out of `shared-bvh` into `walkaround-hybrid/ddgi` (`ea2e975..084de2b`); `plan/sprint-{neural,ppg}-future.md` reworded to reflect that those subsystems are revived/in-flight, not deleted (`69f1237`).

### W11 — OIDN wire (both engine sides)

- `'oidn-final'` real denoiser mode wired in `HybridEngine` (`feat/w11-oidn-wire`, `74fad35`).
- `'oidn-final'` denoiser mode wired in `PTEngineWebGL2` (`feat/w11-pt-webgl-oidn`, `7e98d90`). Engines now expose an in-engine OIDN dispatcher; the OIDN bridge is no longer a zero-consumer public API.

### W12 — dev overlays finished

- 4 stubs in `@vitrum/dev` replaced with real React components (`feat/w12-dev-overlays`): `DDGIAtlasViewer` (`10183dd`), `BVHVisualizer` (`090695a`), `GISignalSplit` (`76f019c`), `MaterialInspector` (`af02305`).

### W13 — documentation reconciliation

- `chore/w13-readme-audit-plan-archive`: per-package README accuracy audit (`fc882f6`); 11 completed-sprint plan documents archived into `plan/archive/` (`413923a`).

### Items-to-fix landings

- A1/A2/A4 (`feat/items-to-fix-A1-A2-A4`): `createEngine` proxy now forwards `updateEnvironment` (A1, `0a24fd2`); `attachVitrum` plumbs `swapChainView` + `swapChainFormat` into `FrameInput` (A2, `1cd8a03`); `HybridEngine` `FrameInput.viewport` contract documented as informational-only — hosts must call `setSize()` directly (A4, `11fb8f3`).
- A3 (`feat/a3-hybridengine-incremental-updates`, `d0d22b0`): `HybridEngine.updatePrimitive` + `updateEmitter` material-only fast path (geometry case throws).
- B3 + F3 + F4 (`feat/items-to-fix-B3-neural-inputpacker`): `INPUT_PACKER_WGSL` wired into `InferenceGraph._runInputPack` (interleaved layout, `954ed1b`); `dispose()` destroys weights/biases/uniform buffers (`72b42d2`); 137-line shape-debate header stripped from `unetArchitecture.ts` (`96ab815`).
- B4 — closed by W11 (both engine sides now consume the OIDN bridge in-engine).

### Repo / infra housekeeping

- `fix/pt-webgl-tests-three-gpu-pathtracer-dep` (`ce87517`): factory + `materialsTextureSpectral` tests were made resilient to the old missing sibling `three-gpu-pathtracer` repo (now superseded by the absorbed workspace package).
- `fix/gpu-tests-opt-in` (`006debc`): GPU-browser tests now opt-in via env flag; default `npm test` no longer requires Playwright.
- `fix/examples-typecheck` (`ac1b593`): `cornell-box` + `two-engines-one-scene` examples + `pt-webgpu` process ref typecheck clean.
- `chore/gitignore-cron-lock` (`df2d877`): `.Codex/scheduled_tasks.lock` gitignored.
- Branch / worktree cleanup: 5 stale branches deleted; 2 zombie worktrees removed (items_to_fix C2 follow-up).

## Where things actually stand (read this before claiming "ready")

The 2026-05-11 deep math/physics sweep + the 2026-05-17 complexity sweep + the 2026-05-17 judge-mode audit + the 2026-05-18 structural sweep have largely been worked through. The `items_to_fix.md` file at the repo root is the **authoritative** open-bug list — each entry was re-verified by opening the cited file before being kept. After W1–W7 / W11 / W12 / W13 / items-to-fix landings + the 2026-05-18 sweep landings (HybridEngine decomp, WalkaroundGPUPipeline split, pathTraceBruteforce split, core/scene + core/engine splits, Cornell-magic UBO migration, iblBaker per-instance hoist, scene-lighting package extract, renderFrame denoiser-pass collapse, webGpuTextureUpload migration, Möller-Trumbore canonical hoist), **all of Sections A / B / C of `items_to_fix.md` are closed, and the Section E merge-race backlog (E1–E7) is closed as well** (per the 2026-05-24 reconciliation at the top of that file). Two notable closures for reference:

- **B2 — RC into HybridEngine** — W8 sprint **shipped end-to-end (2026-05-18)**: Phase 1A (cascade data types THREE-free), Phase 1B (`RCDispatcher.dispatchFrameRaw` raw-GPU entry), Phase 2 (`HybridEngineOptions.rcEnabled` + per-engine `RCSubsystem`), Phase 3 (shade.wgsl `sampleCascadeC0` + Track-A balance-heuristic MIS via `rcWeight` option), and Phase 4 (gated `rcAcceptance.gpu.test.ts` + reference-render landings in `tools/reference-renders/W8-rc-{off,on}/`) all landed. The harness for the actual GPU capture lives in `tools/benchmark-runner/` once it grows an `rc-acceptance` mode; the host-side wiring + MIS math is pinned by `packRCParams` tests + the `wgslCompose` order pin. See [plan/archive/w8-rc-mis-composition-archived-2026-05-30.md](./plan/archive/w8-rc-mis-composition-archived-2026-05-30.md) for the full sprint trace.
- **`pt-webgpu` glossy BSDF sampling/PDF mismatch** — **fixed** (`a7dd51a`; Heitz 2018 VNDF in `bsdf.wgsl.ts`). Deep-audit findings are closed (`plan/archive/pt-webgpu-deep-audit-archived-2026-05-28.md`). Remaining pt-webgpu work is **fidelity promotion** (renderer matrix rows still `experimental`) and adapter-tier limits (lite vs full), not baseline path-tracer correctness.

Treat the open items as real, prioritise honestly. Don't paper over with band-aids that suppress symptoms.

## What's next

**Maturity label (do not call the library "pre-alpha"):** root `README.md` places vitrum on the **release-candidate track** for `@vitrum/engine`, `walkaround-hybrid`, and `pt-webgl2`. `@vitrum/pt-webgpu` is a **peer PT backend** with closed deep-audit findings; treat "experimental" as per-feature fidelity tier (`plan/renderer-fidelity-matrix.md`), not as "the whole repo is a prototype."

**Programs PR + WG (2026-05-26 signoffs):** primary-release and WebGPU-PT-parity implementation waves are landed in code; see `plan/archive/PR-signoff-2026-05-26-archived-2026-05-28.md`, `plan/archive/WG-signoff-2026-05-26-archived-2026-05-28.md`, and `plan/archive/backend-maturity-matrix-2026-05-26-archived-2026-05-30.md`.

**Honest remaining deep-pipeline work** (ignore npm / release governance):

1. **Fidelity promotion on pt-webgpu** — spectral, thin-film, SSS, caustics, multi-emitter rows are implemented with mechanical tests but still tagged `experimental` until gap-closure scenarios promote them to `supported` in `plan/renderer-fidelity-matrix.md`. (SVGF-real is now `unsupported` on both converged backends — regime mismatch, not a gap.)
2. **Host animation workflows** — walkaround + pt-webgl + pt-webgpu all expose transform/positions incremental patches via `incrementalPatchSupport`. pt-webgl/pt-webgpu also absorb vertex/index-count and instance-count changes via targeted BLAS/TLAS or geometry regeneration (no host `setScene()`); walkaround absorbs vertex/index-count but throws on instance-count/params/shape (P5 follow-up).
3. **GI subsystem BVH alignment** — RC moving-instance merged refit without teardown is now WIRED into GI propagation (+ filter parity with ReSTIR); optional merged-BVH fallback cleanup remains. GPU A/B = V13.
4. **GPU skinning compute** — fully shipped: `GpuSkinningSubsystem` skins positions AND normals (inverse-transpose via `GPU_SKIN_BVH_WITH_NORMALS_WGSL`/`mat3InverseTranspose`), with a CPU-`solveSkin` fallback for non-identity-bind meshes. GPU A/B = V11.
5. **PPG** — now actually GUIDES: gi-ris draws from the learned dTree with a defensive `α·p_guide+(1−α)·p_cos` MIS (was train-only). Remaining tuning + adaptive dispatch cadence tracked in `plan/archive/d2-e6-pt-webgpu-ppg-performance-archived-2026-05-28.md` (archived — its original items shipped). GPU A/B = V17.
6. **Denoisers** — BMFR is implemented (real Householder-QR feature regression). BMFR is no longer a gap; the type union's only contract-without-impl entries now are whatever future modes get added.

Older active docs: `plan/renderer-fidelity-matrix.md`. (Archived under `plan/archive/`: `primary-release-and-webgpu-pt-parity-2026-05-26` and `d2-e6-pt-webgpu-ppg-performance` — their items shipped.)

## Former path-tracer fork

`packages/three-gpu-pathtracer/` and `@vitrum/pt-webgl` were removed in favor of the native `@vitrum/pt-webgl2` backend. Do **not** rely on or create sibling checkout branches for vitrum work; older archived sprint docs may mention that retired workflow. Ported kernels keep provenance in source comments and `CREDITS.md`.

## Conventions

- **No upstream PRs yet.** Do not create upstream PRs to provenance projects without explicit user instruction.
- **No npm publish yet.** Local-only via npm workspaces (`file:./packages/*`). Do not publish without explicit user instruction.
- **No remote pushes without instruction.** Do not push `~/projects/vitrum` without the user saying so.

## Key design principles (in priority order)

1. **The contract is the thing that's fixed.** Backends are swappable; scene bindings are swappable; denoisers are composable. Public types in `@vitrum/core` are the load-bearing interface.
2. **The host owns lifecycle.** Engine accepts a device handle but does NOT own the device. Engine accepts frame inputs but does NOT own the cadence. This is the design choice that makes the library survive Canvas remounts, route changes, tab visibility transitions.
3. **Generalize over time.** Today's contract handles the most pressing concrete needs. Each Phase 6 sprint generalizes one more dimension. Use `Material.extensions`, `EngineOptions.extensions`, and the `AnalyticShape` discriminated union as explicit extension points.
4. **Cite prior work.** Every algorithm has provenance. Citation goes in three places: source code comment at the implementation site, package README, and the project-level `CREDITS.md`.

## Testing protocol

For any algorithmic change to a backend or shared package: capture a "before" reference render of the relevant test scene, make the change, capture an "after" reference render, A/B them. Numerical regression is acceptable only if visually justified. Reference renders live in `tools/reference-renders/`; any new example should target the core `Scene` contract.

Mechanical checks: **`npm run typecheck`** (TypeScript, all packages with a `typecheck` script), **`npm test`** (Vitest in packages that define tests). Release notes: **[CHANGELOG.md](./CHANGELOG.md)**.

## Memory location

This project's per-session memory: `/home/jsquire4/.Codex/projects/-home-jsquire4-projects-vitrum/memory/` (already seeded with foundational entries — read `MEMORY.md` there for the index).

## When in doubt

The user's pattern: ask one question at a time, surface options before locking decisions, don't sandbag with half-implementations. They tolerate longer timelines for better outcomes. They do not want SOTA-cargo-cult — every proposed technique needs verified-feasibility (public source, language portable to web, not RTX-hardware-locked) before scheduling. See `MEMORY.md` index in the memory directory for the full set of working preferences.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **vitrum** (16792 symbols, 27205 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/vitrum/context` | Codebase overview, check index freshness |
| `gitnexus://repo/vitrum/clusters` | All functional areas |
| `gitnexus://repo/vitrum/processes` | All execution flows |
| `gitnexus://repo/vitrum/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
