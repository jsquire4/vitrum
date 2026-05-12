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
6. The other plan/ docs (`generalized-library-milestones.md`, `walkaround-without-three.md`, `pt-webgpu-deep-audit.md`, `d2-e6-pt-webgpu-ppg-performance.md`, `renderer-fidelity-matrix.md`) are the active docs. Completed-sprint artifacts live in `plan/archive/`.

## What's done

- **Phase 6 (Sprints 0–13) complete**; **Phase 7 walkaround-hybrid (Sprints 14–18) shipped**: layered BSDF fork patch, half-res GTAO + bilateral upsample (S15), ReSTIR-GI RIS (S16), ReSTIR-GI temporal+spatial reuse (S17), per-channel SVGF on direct + indirect (S18), plus extensive firefly / dim-magnitude root-cause work and library-generality remediation. Workspace `tsc --noEmit` clean; ~660+ vitest tests pass (2–3 skipped — intentional GPU-only paths).
- **Packages**: `core`, `three-bindings`, `shared-bvh`, `shared-samplers` (light tree, BDPT, spectral), `shared-denoisers` (SVGF, OIDN bridge), `pt-webgl` (wraps three-gpu-pathtracer fork — production PT), `pt-webgpu` (pre-alpha prototype WebGPU PT — internal, not production), `walkaround-hybrid` (DDGI + RC + ReSTIR-DI + ReSTIR-GI + PPG + neural denoiser scaffold + GTAO + per-channel SVGF).
- **Extraction**: `_staging/legacy-source/` contains only host-app React/Redux files intentionally not extracted (see `_staging/README.md`).
- **External RFEs**: 01–05 (contract-layer) plus 06/07/08/09/10/12/14 fork patches applied per `external_requests/IMPLEMENTATION-STATUS.md`.

## Where things actually stand (read this before claiming "ready")

A 2026-05-11 deep math/physics sweep found multiple load-bearing bugs that the green test suite does not catch — see `memory/in-flight-sweep.md` for the verified list. Highlights:

- **DDGI receiver double-applies albedo and 1/π** (producer writes `L_o = albedo·(direct+indirect)/π`; consumer multiplies again by `albedo/π`).
- **DDGI atlas border padding is allocated but never written**; bilinear at every probe-cell edge blends with zero.
- **What ships as "SVGF" is à-trous + a variance scalar lookup** — variance pass declares 4 textures it never samples; depth uploaded to `.r` but read from `.w`; no per-pixel disocclusion-reset history.
- **PPG enable hard-throws at pipeline compile** because the injector still searches for a string that no longer exists in `shade.wgsl`.
- **`pt-webgpu` glossy BSDF** is a mix-around-mirror lerp paired with a textbook GGX half-vector PDF — sampling/PDF mismatch.
- **Neural denoiser** is decorative scaffolding (no `'neural'` mode in `HybridEngine`; scaffold itself has shape mismatches).

Treat these as real, prioritise honestly. Don't paper over with band-aids that suppress symptoms (e.g., the recent `disable DDGI gain` and the hard-coded `randomRotation = (0,0,0)`).

## What's next

Pick from `memory/in-flight-sweep.md` blockers, or active docs `phase-7-restir-gi.md` / `d2-e6-pt-webgpu-ppg-performance.md` / `pt-webgpu-deep-audit.md`. Sprint 10c (BDPT dispatch) and Sprint 14 (layered BSDF) remain gated.

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
