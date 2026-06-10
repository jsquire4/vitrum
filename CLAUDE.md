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

- **Maturity headline: behavioral gate 26/26 configs non-black, zero GPU errors on lavapipe.** Release-candidate track for `engine` / `walkaround-hybrid` / `pt-webgl2` / `pt-webgpu`. SVGF-real intentionally `unsupported` on both converged backends (regime mismatch). In-repo naga shader compile gate (51 WGSL shaders, `npm run shader-gate`). CI: typecheck + test + lint + shader-gate + behavioral-gate.
- **Trust-remediation rounds R7a–R7d (2026-06-10, commits a1b85b1–1a8ab08) key landings:** behavioral gate permanent (26/26); anisotropic GGX live on pt-webgpu; `onError` engine error surface (silent-GPU-error class dead); `@vitrum/gltf-adapter` new package (glTF 2.0 → core Scene); `captureFrame` pixel-readback API; `pickPrimitive` real on all 3 backends; `CameraLike`/`QualityTier`/presets public; examples/ tree + debugging runbook; IES dead chain removed; spectral×photon-map gather spectralized; giState v4 (PPG warm restore); SPPM streaming-window corrected.
- **v1-closure campaign (2026-06-10, commits 6e90443–caab499) key landings:** SPPM photon gather (A4, streaming-window); Preetham sky (pt-webgpu); skinning native both PT backends; tonemap/exposure/outputColorSpace on all 3 backends (BEHAVIOR CHANGE: default `aces@1.0@sRGB` — migrate with `tonemap:'none'`); BDPT estimator coherence both backends; env pillar complete on walkaround; RC finished (A7); lifecycle hardening; in-repo shader compile gate.
- **Authoritative sources:** open bugs → `items_to_fix.md` §H; pending GPU validation → `HARDWARE-VALIDATION-NEEDS.md` V28-B; road-to-100 priorities → `plan/road-to-100.md`; per-session state → memory dir `MEMORY.md`.
- **Remaining work tail (verified, not papered over):** (1) A4-progressive (true Hachisuka SPPM — current is streaming-window); (2) A6 NRC semantics (distillation target + spreadC); (3) A8 GRIS default (biased default, GRIS off-default); (4) A10 production neural weights (starter only shipped); (5) B2 DDGI diffuse-only bounce; (6) glass refracted GI (B1 tail); (7) `TextureRef.texCoord` on pt-webgl2 (documented unkept promise); (8) V28-B GPU A/B recapture (all render-changing landings — improvement confirmations, not regression suspects); (9) H-residue (H5 BDPT host driver, H21/H24-cluster/H32/H34/H35). Full ledger: `plan/road-to-100.md`.
- **THREE removal: COMPLETE (2026-06-09).** Runtime engines consume `@vitrum/core` scenes only.

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
