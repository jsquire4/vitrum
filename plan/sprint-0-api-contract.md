# Sprint 0 — Library API contract

**Status**: in progress (initial draft committed 2026-05-09)
**Effort**: 2–3 days from kickoff to first stainedGlass-app integration test

## Goal

Lock the public API contract in `@vitrum/core` such that every subsequent Phase 6 sprint deliverable lands in a vitrum package without unwinding work. After Sprint 0:

- `@vitrum/core/src/scene.ts` — Scene, Material, ScenePrimitive, SceneEmitter, SceneEnvironment types
- `@vitrum/core/src/frame.ts` — FrameInput, FrameOutput, Viewport types
- `@vitrum/core/src/engine.ts` — Engine, EngineCapabilities, EngineState, EngineFactory, EngineOptions types
- `@vitrum/core/src/index.ts` — barrel export

These files exist as of the initial Sprint 0 commit. **Step 1 is done.**

## Remaining Sprint 0 work

### Step 2 — first backend stub (~1 day)

Create `@vitrum/pt-webgl/src/index.ts` as a stub that:
- Implements the `Engine` interface (every method exists, may throw `Not implemented` initially)
- Accepts a `WebGLRenderingContext` via `createPTEngine_WebGL2(options)` factory
- Returns sensible `EngineCapabilities` for the WebGL2 + three-gpu-pathtracer-fork use case
- Does NOT yet wrap the actual three-gpu-pathtracer code — that's Sprint 1's work

### Step 3 — three-bindings stub (~0.5 day)

Create `@vitrum/three-bindings/src/index.ts` exporting:
- `sceneFromThreeJS(threeScene: THREE.Scene): Scene` — walks the scene graph, converts every `THREE.Mesh`/`THREE.Light` to the corresponding `@vitrum/core` type
- Initially throws on unsupported types (instanced meshes, custom shaders); each Phase 6 sprint adds support

### Step 4 — stainedGlass migration scaffold (~0.5 day)

In the stainedGlass repo, add a `src/rendering/vitrum-bridge/` directory containing:
- `useVitrumPTEngine.ts` — React hook that mirrors the current `usePathtracer` lifecycle but creates an `@vitrum/pt-webgl` engine instead of a raw three-gpu-pathtracer instance
- `useVitrumWalkaroundEngine.ts` — same for walkaround-hybrid

Both hooks throw initially (the engines don't render yet). The point is to fix the import sites.

### Step 5 — verification (~0.5 day)

- Run stainedGlass `npm install` with vitrum linked via npm workspaces (or `file:` protocol)
- Verify `import { Scene } from '@vitrum/core'` compiles in stainedGlass
- Verify `tsc --noEmit` is clean across the entire workspace
- Add a `tools/benchmark-runner` placeholder so future sprints can drop benchmarks in

## Definition of done

- [x] `@vitrum/core/src/scene.ts` exists, fully typed, documented
- [x] `@vitrum/core/src/frame.ts` exists, fully typed, documented
- [x] `@vitrum/core/src/engine.ts` exists, fully typed, documented
- [x] `@vitrum/core/src/index.ts` re-exports
- [x] Monorepo `package.json` workspace config
- [x] All stub package `package.json` files
- [x] LICENSE (MIT)
- [x] CREDITS.md with foundational citations
- [x] README.md with architecture overview
- [x] plan/library-architecture.md (this file's sibling)
- [ ] `@vitrum/pt-webgl/src/index.ts` stub implementing Engine interface (throws)
- [ ] `@vitrum/three-bindings/src/index.ts` stub
- [ ] stainedGlass `src/rendering/vitrum-bridge/` placeholder hooks
- [ ] `tsc --noEmit` clean across workspace
- [ ] Sprint 0 committed; Phase 6 doc updated to reference vitrum packages

## Open contract decisions to revisit

These are explicitly TBD until Phase 6 sprints expose the requirements. Sprint 0 leaves them as commented-out fields or extension keys; later sprints lock the shape:

1. **Incremental scene updates**: `updatePrimitive(id, patch)` is in the Engine type, but no backend implements it yet. Sprint 3 (light tree) will be the first sprint that benefits from it (light tree CDF rebuild on emitter update). Lock the diff API shape during Sprint 3.
2. **Spectral payload**: Sprint 8 ships RGB-as-3λ; Sprint 12 (conditional) ships hero-wavelength. The `Material.extensions.spectral` slot is reserved; the actual fields lock when Sprint 8 commits.
3. **Motion blur input**: `FrameInput.shutterTime` is in the type but no engine consumes it. Sprint 10's Scene Fidelity #1 will lock the per-primitive motion data shape.
4. **Volumetric medium**: Sprint 7 introduces a global volume; the Scene type may grow a `volume?: SceneVolume` field. Decide between "global only" or "per-region" at Sprint 7 kickoff.
5. **Backend texture handle type**: currently `unknown`. Each backend's package declares its actual type via TypeScript module augmentation. Lock the augmentation pattern when `@vitrum/pt-webgl` ships its first real implementation in Sprint 1.

## What Sprint 0 explicitly does NOT do

- Move any actual stainedGlass engine code into vitrum packages. That's Phase 6 sprints' job, one feature at a time.
- Pull `vendor/three-gpu-pathtracer` out of stainedGlass into vitrum. That happens when `@vitrum/pt-webgl` first actually uses it (Sprint 1 or 2).
- Replace stainedGlass's PathTracingLayer / WalkaroundStage React components. They keep working as-is until the migration shim hooks (`useVitrumPTEngine`, `useVitrumWalkaroundEngine`) reach feature parity.
- Publish anything to npm. Local-only until prime time.
