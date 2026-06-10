# Agent brief — vitrum

> If you're a new Claude Code agent working in this repo, **read this first.**

## What this project is

`vitrum` is a WebGPU + WebGL2 path tracing & global illumination engine for the browser. The white-whale ambition is to own the entire SOTA-browser-rendering stack — from BVH construction to physically-based path tracing to real-time global illumination to denoising — under one consistent, host-agnostic API contract.

## Onboarding order

1. `README.md` — package architecture overview
2. `plan/library-architecture.md` — package responsibilities, dependencies
3. `packages/core/src/scene/` + `src/frame.ts` + `src/engine/` — the locked-in API contract types
4. `CREDITS.md` — attribution to ~30 prior works the engine builds on
5. Active plan docs: `plan/roadmap.md` (§0.5 = locked priorities), `plan/renderer-fidelity-matrix.md`, `plan/walkaround-without-three.md`. Completed-sprint artifacts live in `plan/archive/`.

## Packages

- `core` — the locked-in host-agnostic contract: scene/frame/engine types, `EngineCapabilities` (+ fine-grained `BackendSupportDetails`), backend promise ledger, `solveSkin`, `analyticPrimitiveToMesh`, `createInverseSession`
- `engine` — `createEngine` / `attachVitrum` facade + `createProgressiveEngine` (shared-device walkaround→PT handoff)
- `walkaround-hybrid` — realtime GI (release-candidate track): DDGI + ReSTIR-DI/GI + GTAO + SVGF + opt-in RC/PPG/neural; GI-state export/import; analytic primitives via generated-mesh fallback
- `pt-webgl2` — native WebGL2 converged PT (release-candidate track)
- `pt-webgpu` — WebGPU-native converged PT peer: spectral, BDPT, SSS random walk, MNEE manifold caustics, light-tree NEE, ReSTIR-PT (off-default), path-replay adjoint
- `walkaround-rc` — Radiance Cascades subsystem (cascade pyramid + dispatch + receiver)
- `shared-bvh`, `shared-samplers`, `shared-denoisers`, `scene-lighting`, `stained-glass-extensions`, `dev` (debug overlays)

## Where things stand (2026-06-10)

- **Maturity:** release-candidate track for `engine` / `walkaround-hybrid`. `pt-webgl2` RC label RESTORED — all §H P0/P1 gaps (H1–H5, H6–H7 upload-gap cluster) were fixed in the 2026-06-09 validation campaign plus the v1-closure campaign. `pt-webgpu` peer PT backend. SVGF-real intentionally `unsupported` on both converged backends. In-repo naga shader compile gate (48 WGSL shaders, `npm run shader-gate`). CI rewritten (typecheck + test + lint + shader-gate). **Big validation tail: V28-B** — GPU A/B recapture needed for every render-changing Wave 2–5 landing (tonemap default, SPPM, skinning, env NEE, spectral MIS, Preetham, RC estimator, DDGI probe HDRI). These are improvement confirmations, not regression suspects.
- **v1-closure campaign (2026-06-10, commits 6e90443–caab499) key landings:** real SPPM photon map (A4); Preetham procedural sky (pt-webgpu); skinning native on both PT backends; tonemap/exposure/outputColorSpace wired on all 3 backends (BEHAVIOR CHANGE: default is now `aces@1.0@sRGB`, not raw HDR — migrate with `tonemap:'none'`); BDPT estimator coherence both backends; env pillar complete on walkaround (HDRI `approximate→native`; DI NEE candidate in RIS; DDGI probe HDRI sampling; risGiNrc env parity); RC finished (A7; correct MC estimator; point/spot lights; chromatic sun); lifecycle hardening; contract honesty sweep; in-repo shader compile gate.
- **Authoritative sources:** open bugs → `items_to_fix.md` §H (H1–H62 + v1-closure updates; see STATUS 2026-06-10 block); pending GPU validation → `HARDWARE-VALIDATION-NEEDS.md` V28-B; road-to-100 priorities → `plan/road-to-100.md` + `plan/v1-closure-plan-2026-06-10.md`; per-session state → the memory dir's `MEMORY.md`.
- **Remaining work (2026-06-10):** (1) **feature completion** — A6 NRC posture (distillation target, spreadC semantics); A8 GRIS default (biased default, GRIS exists off-default); A10 weights not shipped; B2 DDGI diffuse-only bounce; B12 lite-tier fidelity cliff; sun-NEE default gate; pt-webgl2 BDPT orchestration (H5 kernel exists, host driver not wired); (2) **fidelity ceilings** — glossy GI in walkaround (B1 partial — metals lit but glass GI empty); analytic NEE on pt-webgl2 needs NEE for mesh-area; (3) **provisioning** — C4 examples gap (zero runnable examples); A10/C2 neural weights; (4) **test infrastructure** (H53-class: recording mock, in-repo naga vitest gate, size-validating GPU stub). See `plan/road-to-100.md` for the full ledger.
- **THREE removal: COMPLETE (2026-06-09).** `packages/pt-webgl`, `packages/three-bindings`, `packages/three-gpu-pathtracer`, legacy Three examples, Three bridge subpaths deleted. Runtime engines consume `@vitrum/core` scenes only.
- Recent headline landings: v1-closure campaign above; MNEE manifold caustics COMPLETE; V24 GPU path-replay adjoint; progressive walkaround→PT handoff; ReSTIR-PT wired (OFF-default); DDGI GI-state cache; BackendSupportDetails; B3/B6/B7/B8/B9/B10/B15/B16 (Wave B fidelity wave). Full ledger: `plan/archive/claude-md-history-archived-2026-06-05.md` + git log.

Treat open items as real, prioritise honestly. Don't paper over with band-aids that suppress symptoms.

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

This project's per-session memory: `/home/jsquire4/.claude/projects/-home-jsquire4-projects-vitrum/memory/` (read `MEMORY.md` there for the index).

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
