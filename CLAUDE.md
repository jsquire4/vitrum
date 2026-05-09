# Agent brief — vitrum

> If you're a new Claude Code agent working in this repo, **read this first.**

## What this project is

`vitrum` is a WebGPU + WebGL2 path tracing & global illumination engine for the browser. It is being extracted from the [stainedGlass](../stainedGlass) renderer (its proving-ground host application) into a host-agnostic library. The white-whale ambition is to own the entire SOTA-browser-rendering stack — from BVH construction to physically-based path tracing to real-time global illumination to denoising — under one consistent API contract.

The stainedGlass app is the integration test + demo. It is NOT the project. **The renderer is the project.**

## Where you are right now

Read in this order to onboard:

1. `README.md` — package architecture overview
2. `plan/library-architecture.md` — package responsibilities, dependencies, migration plan from stainedGlass
3. `plan/phase-6-roadmap.md` — full Phase 6 sprint plan (originally in stainedGlass; moved here because the work is library extraction, not app feature work)
4. `plan/sprint-0-api-contract.md` — Sprint 0 (the prerequisite sprint) and its Definition of Done
5. `packages/core/src/{scene,frame,engine}.ts` — the locked-in API contract types
6. `CREDITS.md` — attribution to ~30 prior works the engine builds on
7. The other plan/ docs (`glorious-hybrid.md`, `path-tracer-library-readiness.md`, etc.) are historical context for why the architecture is what it is.

## What's done

- Sprint 0 step 1: `@vitrum/core` types committed (Scene, Material, ScenePrimitive, SceneEmitter, SceneEnvironment, FrameInput, FrameOutput, Engine, EngineCapabilities, EngineState, EngineFactory, EngineOptions).
- Monorepo scaffold: 8 package skeletons (`core`, `three-bindings`, `shared-bvh`, `shared-samplers`, `shared-denoisers`, `pt-webgl`, `pt-webgpu`, `walkaround-hybrid`).
- Plan docs in place: library-architecture, sprint-0, phase-6-roadmap.

## What's next (Sprint 0 remaining work)

Per `plan/sprint-0-api-contract.md`:

1. `@vitrum/pt-webgl/src/index.ts` stub — implements the `Engine` interface, every method throws `Not implemented` initially. Accepts a `WebGLRenderingContext` via a `createPTEngine_WebGL2(options)` factory. Returns sensible `EngineCapabilities`.
2. `@vitrum/three-bindings/src/index.ts` stub — `sceneFromThreeJS(threeScene: THREE.Scene): Scene`.
3. stainedGlass `src/rendering/vitrum-bridge/` placeholder hooks (`useVitrumPTEngine`, `useVitrumWalkaroundEngine`).
4. `tsc --noEmit` clean across workspace.

Sprint 0 then closes; Phase 6 Sprint 1 begins.

## Cross-repo conventions

- **The stainedGlass repo lives at `../stainedGlass`.** Its `CLAUDE.md` notes the split. Some plan docs reference it.
- **Three-gpu-pathtracer fork lives at `~/projects/stainedGlass/vendor/three-gpu-pathtracer/`** today (gitignored). It will eventually move to `~/projects/three-gpu-pathtracer/` and be wrapped by `@vitrum/pt-webgl` rather than directly imported by stainedGlass.
- **No upstream PRs yet.** The fork stays local until vitrum is prime-time-ready. Do not create upstream PRs to gkjohnson without explicit user instruction.
- **No npm publish yet.** Local-only via npm workspaces (`file:../packages/*`). Do not publish without explicit user instruction.
- **Local-only repos.** Do not push to GitHub remotes without explicit user instruction.

## Key design principles (in priority order)

1. **The contract is the thing that's fixed.** Backends are swappable; scene bindings are swappable; denoisers are composable. Public types in `@vitrum/core` are the load-bearing interface.
2. **The host owns lifecycle.** Engine accepts a device handle but does NOT own the device. Engine accepts frame inputs but does NOT own the cadence. This is the design choice that makes the library survive Canvas remounts, route changes, tab visibility transitions.
3. **Generalize over time.** Today's contract handles what stainedGlass needs. Each Phase 6 sprint generalizes one more dimension. Use `Material.extensions`, `EngineOptions.extensions`, and the `AnalyticShape` discriminated union as explicit extension points.
4. **Cite prior work.** Every algorithm has provenance. Citation goes in three places: source code comment at the implementation site, package README, and the project-level `CREDITS.md`.

## Testing protocol

For any algorithmic change to a backend or shared package: capture a "before" reference render of the stainedGlass demo scene, make the change, capture an "after" reference render, A/B them. Numerical regression is acceptable only if visually justified. Reference renders live in `tools/reference-renders/`.

## Memory location

This project's per-session memory: `/home/jsquire4/.claude/projects/-home-jsquire4-projects-vitrum/memory/` (created on first save). The stainedGlass project's memory at `/home/jsquire4/.claude/projects/-home-jsquire4-projects-stainedGlass/memory/` should NOT be cross-imported — they're separate projects with separate concerns.

## When in doubt

The user's pattern (per stainedGlass memory) is: ask one question at a time, surface options before locking decisions, don't sandbag with half-implementations. They tolerate longer timelines for better outcomes. They do not want SOTA-cargo-cult — every proposed technique needs verified-feasibility (public source, language portable to web, not RTX-hardware-locked) before scheduling.
