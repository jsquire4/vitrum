# Agent brief — vitrum

> If you're a new Claude Code agent working in this repo, **read this first.**

## What this project is

`vitrum` is a WebGPU + WebGL2 path tracing & global illumination engine for the browser. The white-whale ambition is to own the entire SOTA-browser-rendering stack — from BVH construction to physically-based path tracing to real-time global illumination to denoising — under one consistent, host-agnostic API contract.

## Onboarding order

1. `README.md` — package architecture overview
2. `plan/library-architecture.md` — package responsibilities, dependencies
3. `packages/core/src/scene/` + `src/frame.ts` + `src/engine/` — the locked-in API contract types
4. `CREDITS.md` — attribution to ~30 prior works the engine builds on
5. Active plan docs: `plan/road-to-100.md` (current code-completion queue), `plan/renderer-fidelity-matrix.md` (implemented backend contract), and `plan/roadmap.md` (future product additions). Completed sprint records live in `plan/archive/`.

## Packages

- `core` — the locked-in host-agnostic contract: scene/frame/engine types, `EngineCapabilities` (+ fine-grained `BackendSupportDetails`), backend promise ledger, `solveSkin`, `analyticPrimitiveToMesh`, `createInverseSession`
- `engine` — `createEngine` / `attachVitrum` facade + `createProgressiveEngine` (shared-device walkaround→PT handoff)
- `walkaround-hybrid` — realtime GI (release-candidate track): DDGI + ReSTIR-DI/GI + GTAO + SVGF + opt-in RC/PPG/neural; GI-state export/import; analytic primitives via generated-mesh fallback
- `pt-webgl2` — native WebGL2 converged PT (release-candidate track)
- `pt-webgpu` — WebGPU-native converged PT peer: spectral, BDPT, SSS random walk, MNEE manifold caustics, light-tree NEE, ReSTIR-PT (off-default), path-replay adjoint
- `walkaround-rc` — Radiance Cascades subsystem (cascade pyramid + dispatch + receiver)
- `shared-bvh`, `shared-samplers`, `shared-denoisers`, `scene-lighting`, `stained-glass-extensions`, `dev` (debug overlays)

## Where things stand (2026-07-24)

- **Implemented profiles:** `engine`, `walkaround-hybrid`, `pt-webgl2`, and `pt-webgpu` implement the feature surface they advertise. Unsupported combinations are reported as unsupported instead of being routed through a provisional mode.
- **Contract closure:** core scene validation is used by adapters, engines, and mutation paths. The renderers consume advertised material, texture, lighting, animation, volume, spectral, caustic, and denoiser fields. Radiance Cascades uses runtime-sized light tables and strict raw-dispatch validation.
- **Authoritative current queue:** `plan/road-to-100.md`. Add a row only for a source-read, reachable runtime gap with an implementation change. `items_to_fix.md` is historical audit context, not an open backlog.
- **Future scope:** `plan/roadmap.md` holds deliberate additions such as native point/line rendering and heterogeneous media. Do not turn future scope into a present implementation defect.
- **THREE removal: complete.** Runtime engines consume `@vitrum/core` scenes only.

Do not reopen completed work from old plans or generated evidence. Reopen it only from current production source and a concrete failure mode.

## Hard-won process lessons

- **Structural WGSL dedup needs a naga-compile gate, not just string goldens.** The pre-push T1 GPU smoke (`wsl-gpu`, lavapipe + dzn) is the only check that compiles the runtime pass graph — it runs automatically on `git push` and has caught real regressions the vitest byte-identity goldens missed.
- **A byte-identity test can be GREEN while both sides share a pre-existing bug** — the T1 smoke carries independent CPU brute-force oracles for RC/TLAS/DDGI traversal for exactly this reason.
- **Self-validating math harnesses (residual→0, analytic==FD) land cleanly where scene-dependent radiometric A/Bs dead-end** — prefer them for subtle radiometric work.
- **Merge races have silently dropped landed commits before** (2026-05-17). Verify presence by code-read, not by changelog.
- **The wsl-gpu T1 oracles import vitrum production files by HARDCODED ABSOLUTE PATH** (e.g. `…/walkaround-hybrid/src/restir/bvhCore.ts`). Renaming/moving a production module silently breaks them: deno throws "module not found" → the script exits 1 → the smoke mislabels it as a "<subsystem> traversal regression" (it's a stale import, NOT a real assertion failure — tell them apart by the error line being an `await import(...)`, not a match-rate). **When you rename/move a walkaround-hybrid/rc/restir/ddgi production module, grep `~/projects/wsl-gpu/scripts` for its path and repoint the oracles.** Verified 2026-06-09: the THREE-decouple (`restir/sceneBvhFromCore.ts` → `restir/bvhCore.ts` + `legacy/three/`) broke all 3 GI brute-force oracles + the render worker this way; production traversal was correct all along (oracles now 100% vs ground truth after repointing).

## Former path-tracer fork

`packages/three-gpu-pathtracer/` and `@vitrum/pt-webgl` were removed in favor of the native `@vitrum/pt-webgl2` backend. Do not recreate sibling checkouts or package aliases for the old fork; use source provenance comments and `CREDITS.md` for attribution when touching ported kernels.

## Conventions

- **No upstream PRs yet.** Do not create upstream PRs to provenance projects without explicit user instruction.
- **No npm publish yet.** Local-only via npm workspaces (`file:./packages/*`). Do not publish without explicit user instruction.
- **No remote pushes without instruction.** Do not push `~/projects/vitrum` without the user saying so.

## Key design principles (in priority order)

1. **The contract is the thing that's fixed.** Backends are swappable; scene bindings are swappable; denoisers are composable. Public types in `@vitrum/core` are the load-bearing interface.
2. **The host owns lifecycle.** Engine accepts a device handle but does NOT own the device. Engine accepts frame inputs but does NOT own the cadence. This is the design choice that makes the library survive Canvas remounts, route changes, tab visibility transitions.
3. **Generalize over time.** Use `Material.extensions`, `EngineOptions.extensions`, and the `AnalyticShape` discriminated union as explicit extension points.
4. **Cite prior work.** Every algorithm has provenance. Citation goes in three places: source code comment at the implementation site, package README, and the project-level `CREDITS.md`.

## Testing protocol

For any algorithmic change to a backend or shared package: capture a "before" reference render of the relevant test scene, make the change, capture an "after" reference render, A/B them. Numerical regression is acceptable only if visually justified. Reference renders live in `tools/reference-renders/`; any new example should target the core `Scene` contract.

Mechanical checks: **`npm run typecheck`** (TypeScript, all packages with a `typecheck` script), **`npm test`** (Vitest in packages that define tests). The pre-push hook runs the T1 GPU smoke automatically. Release notes: **[CHANGELOG.md](./CHANGELOG.md)**.

## Memory location

This project's per-session memory: `/home/jsquire4/.Codex/projects/-home-jsquire4-projects-vitrum/memory/` (read `MEMORY.md` there for the index).

## When in doubt

The user's pattern: ask one question at a time, surface options before locking decisions, don't sandbag with half-implementations. They tolerate longer timelines for better outcomes. They do not want SOTA-cargo-cult — every proposed technique needs verified-feasibility (public source, language portable to web, not RTX-hardware-locked) before scheduling. See `MEMORY.md` index in the memory directory for the full set of working preferences.

<!-- gitnexus:start -->
# GitNexus disabled in current Codex work

GitNexus is intentionally not part of the operating workflow for this repo right now. The environment has been unreliable, and the active Road-to-100 goal requires direct source reads, grep/find, tests, shader gates, and WSL GPU validation instead. Do not use GitNexus impact/query/detect_changes instructions as a gate for edits or commits until the user explicitly re-enables that workflow.

<!-- gitnexus:end -->
