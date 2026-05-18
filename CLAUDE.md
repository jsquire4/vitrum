# Agent brief — vitrum

> If you're a new Claude Code agent working in this repo, **read this first.**

## What this project is

`vitrum` is a WebGPU + WebGL2 path tracing & global illumination engine for the browser. The white-whale ambition is to own the entire SOTA-browser-rendering stack — from BVH construction to physically-based path tracing to real-time global illumination to denoising — under one consistent, host-agnostic API contract.

## Where you are right now

Read in this order to onboard:

1. `README.md` — package architecture overview
2. `plan/library-architecture.md` — package responsibilities, dependencies
3. `plan/phase-7-restir-gi.md` — current Phase-7 walkaround GI work (Sprints 15–18 shipped: GTAO, ReSTIR-GI RIS/temporal/spatial, per-channel SVGF)
4. `packages/core/src/{scene,frame,engine}.ts` — the locked-in API contract types
5. `CREDITS.md` — attribution to ~30 prior works the engine builds on
6. The other plan/ docs (`generalized-library-milestones.md`, `walkaround-without-three.md`, `pt-webgpu-deep-audit.md`, `d2-e6-pt-webgpu-ppg-performance.md`, `renderer-fidelity-matrix.md`, `premium-grade-refactor-20260517.md`) are the active docs. Completed-sprint artifacts live in `plan/archive/` (the W13 archival pass moved 11 stale plan docs there).

## What's done

- **Phase 6 (Sprints 0–13) complete**; **Phase 7 walkaround-hybrid (Sprints 14–18) shipped**: layered BSDF fork patch, half-res GTAO + bilateral upsample (S15), ReSTIR-GI RIS (S16), ReSTIR-GI temporal+spatial reuse (S17), per-channel SVGF on direct + indirect (S18), plus extensive firefly / dim-magnitude root-cause work and library-generality remediation. Workspace `tsc --noEmit` clean; **all** vitest tests pass (~660+ across workspaces — the 2–3 previously-skipped GPU-only paths were enabled via happy-dom / vitest browser mode; GPU-browser tests are opt-in via env flag so default `npm test` no longer requires Playwright).
- **Packages**: `core`, `three-bindings`, `shared-bvh`, `shared-samplers` (light tree, BDPT, spectral), `shared-denoisers` (à-trous-variance, `svgf-real` Schied 2017, OIDN bridge), `pt-webgl` (wraps three-gpu-pathtracer fork — production PT), `pt-webgpu` (pre-alpha prototype WebGPU PT — internal, not production), `walkaround-hybrid` (DDGI + RC + ReSTIR-DI + ReSTIR-GI + PPG + real neural U-Net denoiser + GTAO + per-channel SVGF), `engine` (`createEngine` / `attachVitrum` facade), `dev` (debug overlays).
- **Extraction**: `_staging/legacy-source/` contains only host-app React/Redux files intentionally not extracted (see `_staging/README.md`).
- **External RFEs**: 01–05 (contract-layer) plus 06/07/08/09/10/12/14 fork patches applied per `external_requests/IMPLEMENTATION-STATUS.md`.
- **M7 — DDGI Coherent Physical Model shipped** (sweep 2026-05-11 Items 2, 4, 6, 20): producer no longer pre-multiplies `albedo/π`; Lambertian cosine blend kernel (`w` for irradiance, `w²` for visibility) replaces `pow(w,8)` / `pow(w,50)`; per-frame Halton-Shoemake SO(3) `randomRotation`; RC GI receiver applies `materialColor · PI_INV` before injecting via `emissiveNode`. Cited at `probeUpdatePass.ts:670` (Halton-{2,3,5} sequence) and `applyDDGIShading.ts:145-148` (receiver applies `albedo · PI_INV` exactly once).
- **`svgf-real` (Schied 2017) denoiser pipeline shipped** (`@vitrum/shared-denoisers` + `@vitrum/walkaround-hybrid`): real reprojection + variance-from-moments + 7×7 spatial fallback + à-trous wavelet ×5; opt-in via `denoiser: 'svgf-real'` alongside the default `'atrous-variance'` mode.
- **`HybridEngine.updateLighting()` + `HybridEngine.setSize()`** runtime APIs shipped (no engine recreation for time-of-day scrubbing or canvas resize).
- **Neural U-Net denoiser revived** (T2.H2 + B3 + F3 + F4): real `'neural'` mode in `HybridEngineOptions.denoiser`; INTERLEAVED `inputPacker.ts` compute shader now wired into `InferenceGraph._runInputPack` (the previous planar-vs-interleaved layout mismatch was real and is fixed); `dispose()` releases weight/bias/uniform buffers (F4); 137-line shape-debate header stripped from `unetArchitecture.ts` (F3); `train.py` / `export_weights.py` are real PyTorch tools.
- **A3 shipped — `HybridEngine.updatePrimitive` + `updateEmitter`** (`feat/a3-hybridengine-incremental-updates`, `d0d22b0`): material-only fast path re-uploads one slot in the materials texture without touching DDGI or the temporal accumulator. Geometry-change case throws explicitly (BVH leaf rebuild left as a follow-up). Unblocks a walkaround SceneSync in stainedGlass for material edits.

### W1 — Pass + Resource + Denoiser registry refactor (all 6 rounds shipped)

- R1 — Pass + Denoiser registry foundation (`11c4698`, `feat/refactor-w1r1-pass-registry-foundation`).
- R2 — FrameResources god-struct split into 8 per-algorithm sub-structs (`834f441`, `feat/refactor-w1r2-frame-resources-split`).
- R3+R4 — denoiser migration to self-contained registry entries (`987b3fb`, `feat/refactor-w1r3r4-denoiser-migration`).
- R5 — declarative `PASS_ORDER` via `PassRegistry`; 18 self-contained Pass classes; `renderFrame` becomes a registry-iteration loop (`60dd48b`, `feat/refactor-w1r5-declarative-pass-order`).
- R6 — declarative WGSL include-graph; 27-entry `WGSL_MODULES` registry; `composeWgsl` topo-sort replaces 9 hand-rolled `COMMON_WGSL + X_WGSL` concat patterns (`29ccf96`, `feat/refactor-w1r6-wgsl-include-graph`).
- Aggregate: `WalkaroundGPUPipeline.ts` 1574 → 964 LOC (-39%); 354 → 424 tests; typecheck clean every round.

### W2 — shared-package dedup / hoisting

- C1 — BVH traversal WGSL canonicalised into `shared-bvh` (`feat/w2-geometry-primitives-c1-c3-c4`, `1a07b2b`).
- C2 — THREE-independent `buildArrayBvh` hoisted from `pt-webgpu` into `shared-bvh` (`feat/w2-c2-bvh-builder-shared`, `44598ad`).
- C3 — canonical octEncode/octDecode (was buggy `sign(0)→0` collapse) (`8fec673`).
- C4 — `probeAtlasUv` reused instead of 3 inline atlas-coord derivations (`3446817`).
- C6 — PCG / cosineHemisphere / ONB / Fresnel / GGX primitives canonicalised into `shared-samplers` (`feat/w2-c6-c15-pcg-hash-primitives`, `da286d7`).
- C15 — pixel-hash function canonicalised into `shared-samplers/wgsl/hash` (`651619e`).
- C10 — Rec.709 luminance vector canonicalised across `shared-samplers` / `shared-denoisers` / `walkaround-hybrid` (`feat/w2-small-dedups-c10-c11-c12`, `54ae795`).
- C11 — duplicated B3-spline atrous kernel collapsed (`76120b9`).
- C12 — `GTAOUniforms` struct deduped across `gtao` + `gtaoUpsample` shaders (`8a40ffe`).

### W3 — contract hygiene (surgical core/* moves)

- D5/D6/D13/D15 — Mat4 branded as `Float32Array & __mat4Brand`; stopped re-exporting raw WGSL strings (internal-only now); moved underscored test-only exports off public surface (`feat/w3-contract-hygiene-d5-d6-d13-d15`, `e845cc5..becfbae`).
- D7 — `FrameOutput` replaced null+sentinel with discriminated union `{kind:'skipped'|'rendered'}` (`feat/w3-d7-frame-output-discriminated-union`, `40cd837`).
- D8 — typed `EngineCapabilities`-driven feature query replaces `typeof` checks across `core` + `engine` (`feat/w3-d8-engine-capabilities`, `1355be1`).
- D16 — canonical `FrameStats` + `qualityModes` capability in `@vitrum/core`; `pt-webgpu` now emits `onFrame` stats (`feat/w3-d16-uniform-telemetry`, `197510c..8c96e4b`).
- D17 — verified `Material.extensions` IS used by `three-bindings` dichroic LUT — keep (`80c2388`); D18 — dropped `supportsMotionBlur` + `FrameInput.shutterTime` (no consumer in roadmap) (`9ea12c9`).
- D19 — `BackendTexture` / `BackendTextureFormat` branded with backend type parameter (`feat/w3-d19-backendtexture-branding`, `5863cda`).

### W4 — god-file dissolution (first wave)

- A4 — `pt-webgpu` `pathTraceBruteforce.wgsl.ts` split (1908 → 76 LOC orchestrator + 15 structured modules) (`feat/w4-a4-pt-webgpu-shader-split`, `f640de5`).

### W6 — hidden-globals de-singletonization

- E1 — `reuseSharedWebGpuDevice` default flipped to `false`; singleton renamed for test-only clarity (`feat/w6-hidden-globals-e1-e3`, `3dbe11a`).
- E2 — module-level `iblBaker` cache replaced with per-engine `IblBakerCache` instance (`feat/w6-hidden-globals-pt-webgl`, `6a5106f`).
- E3 — `window.__WGPU__` write removed from `HybridEngine.renderFrame` (use `onFrame`) (`7d85bb0`).
- E6 — fork-private `_pathTracer` access encapsulated behind a `ForkAccess` indirection (`a6a3c90`).

### W7 — dead-code + misplaced-code cleanup

- Dead-code batch (G2/G3/G5/G7/G9) (`feat/w7-dead-code-batch`): deleted deprecated `bdptConnectionMIS_partial` / `buildBDPTStrategyPDFs_partial` (G2); dropped `rgbToApproxSpectralCoefficients` alias + clarified Jakob-Hanika header (G3); un-exported `validateBvhEncoding` (G5, tests-only); un-exported internal helpers `extractAttribute`/`extractIndex`/`warnOnce` from three-bindings (G7); deleted redundant emitter-constants re-exports from `uploadSceneBuffers` (G9).
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

- `fix/pt-webgl-tests-three-gpu-pathtracer-dep` (`ce87517`): factory + `materialsTextureSpectral` tests resilient to missing sibling `three-gpu-pathtracer` repo (worktree-path resolution).
- `fix/gpu-tests-opt-in` (`006debc`): GPU-browser tests now opt-in via env flag; default `npm test` no longer requires Playwright.
- `fix/examples-typecheck` (`ac1b593`): `cornell-box` + `two-engines-one-scene` examples + `pt-webgpu` process ref typecheck clean.
- `chore/gitignore-cron-lock` (`df2d877`): `.claude/scheduled_tasks.lock` gitignored.
- Branch / worktree cleanup: 5 stale branches deleted; 2 zombie worktrees removed (items_to_fix C2 follow-up).

## Where things actually stand (read this before claiming "ready")

The 2026-05-11 deep math/physics sweep + the 2026-05-17 complexity sweep + the 2026-05-17 judge-mode audit have largely been worked through. The `items_to_fix.md` file at the repo root is the **authoritative** open-bug list — each entry was re-verified by opening the cited file before being kept. After W1–W7 / W11 / W12 / W13 / items-to-fix landings, **Sections A and C of `items_to_fix.md` are empty of open items**; only two Section B items remain:

- **B1 — PPG GPU dispatch is `dispatchWorkgroups(0, 0, 0)`.** `ppgEnabled = true` compiles the pipeline but produces zero PPG samples. Either wire the sTree/dTree GPU producer (W9 work) or remove the option. (`packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:829`)
- **B2 — RC subsystem ships 1500+ LOC but is unwired into `HybridEngine`.** Either wire RC into the combined shading sum (W8 work) or move the subtree to `examples/standalone-rc/` and drop it from `walkaround-hybrid` public exports.
- **`pt-webgpu` glossy BSDF mix-around-mirror sampling/PDF mismatch** — still real, but `pt-webgpu` is explicitly labelled pre-alpha prototype, not the production PT path (`pt-webgl` is). Tracked in `plan/pt-webgpu-deep-audit.md`.

Treat the open items as real, prioritise honestly. Don't paper over with band-aids that suppress symptoms.

## What's next

Two parallel queues:

1. **`items_to_fix.md` remaining Section B items** — B1 (PPG dispatchWorkgroups 0,0,0) → W9; B2 (RC unwired) → W8. See the file at the repo root for verified-current locations, symptoms, fix sketches, and acceptance criteria. **A3 follow-up:** geometry-change BVH leaf rebuild (the current fast path throws on geometry diffs).
2. **`plan/premium-grade-refactor-20260517.md` remaining workstreams** — W1 / W2 / W3 (D5/D6/D7/D8/D13/D15/D16/D17/D18/D19) / W4 (A4) / W6 (E1/E2/E3/E6) / W7 (G2/G3/G5/G6/G7/G8/G9 + H1–H5/H7/H8 + I) / W11 / W12 / W13 are done. Remaining: **W3 D1/D2/D3/D4/D12** (other contract items); **W4 A5/A6/A7/A8/A9** (remaining god-file splits — `HybridEngine`, `WalkaroundGPUPipeline`, `probeUpdatePass`, etc.); **W5** (per-pass BGLs); **W8** (RC extraction → `@vitrum/walkaround-rc`); **W9** (PPG GPU dTree finish); **W10** (Neural denoiser full finish — B3 was partial).

Older active docs: `phase-7-restir-gi.md`, `d2-e6-pt-webgpu-ppg-performance.md`, `pt-webgpu-deep-audit.md`. Sprint 10c (BDPT dispatch) and Sprint 14 (layered BSDF) remain gated.

## Sibling repository: the path-tracer fork

`~/projects/three-gpu-pathtracer/` — local working copy of a fork of `gkjohnson/three-gpu-pathtracer` at branch `phase4-normalmap-shadow-rays`. The Phase 4 normalMap-perturbed-NEE-shadow-ray patch is committed there. `@vitrum/pt-webgl` will wrap this fork as its WebGL2 backend implementation.

When `@vitrum/pt-webgl` reaches the point of importing from it, the cleanest pattern is `npm install file:../three-gpu-pathtracer` from the package directory, with a clear note in pt-webgl's README about the version pin. The fork's remote: `git@github.com:jsquire4/three-gpu-pathtracer.git`.

## Conventions

- **No upstream PRs yet.** The fork stays local until vitrum is prime-time-ready. Do not create upstream PRs to `gkjohnson/three-gpu-pathtracer` without explicit user instruction.
- **No npm publish yet.** Local-only via npm workspaces (`file:./packages/*`). Do not publish without explicit user instruction.
- **No remote pushes without instruction.** Both `~/projects/vitrum` and `~/projects/three-gpu-pathtracer` have remotes; do not push without the user saying so.

## Key design principles (in priority order)

1. **The contract is the thing that's fixed.** Backends are swappable; scene bindings are swappable; denoisers are composable. Public types in `@vitrum/core` are the load-bearing interface.
2. **The host owns lifecycle.** Engine accepts a device handle but does NOT own the device. Engine accepts frame inputs but does NOT own the cadence. This is the design choice that makes the library survive Canvas remounts, route changes, tab visibility transitions.
3. **Generalize over time.** Today's contract handles the most pressing concrete needs. Each Phase 6 sprint generalizes one more dimension. Use `Material.extensions`, `EngineOptions.extensions`, and the `AnalyticShape` discriminated union as explicit extension points.
4. **Cite prior work.** Every algorithm has provenance. Citation goes in three places: source code comment at the implementation site, package README, and the project-level `CREDITS.md`.

## Testing protocol

For any algorithmic change to a backend or shared package: capture a "before" reference render of the relevant test scene, make the change, capture an "after" reference render, A/B them. Numerical regression is acceptable only if visually justified. Reference renders live in `tools/reference-renders/`. Working test scenes go in `examples/`.

Mechanical checks: **`npm run typecheck`** (TypeScript, all packages with a `typecheck` script), **`npm test`** (Vitest in packages that define tests). Release notes: **[CHANGELOG.md](./CHANGELOG.md)** (pre-alpha versioning called out there).

## Memory location

This project's per-session memory: `/home/jsquire4/.claude/projects/-home-jsquire4-projects-vitrum/memory/` (already seeded with foundational entries — read `MEMORY.md` there for the index).

## When in doubt

The user's pattern: ask one question at a time, surface options before locking decisions, don't sandbag with half-implementations. They tolerate longer timelines for better outcomes. They do not want SOTA-cargo-cult — every proposed technique needs verified-feasibility (public source, language portable to web, not RTX-hardware-locked) before scheduling. See `MEMORY.md` index in the memory directory for the full set of working preferences.
