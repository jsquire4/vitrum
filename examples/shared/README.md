# `@vitrum-examples/shared`

Internal helper package — **not** published. Contains the THREE.js scene builders that the runnable vitrum examples (`cornell-box`, `two-engines-one-scene`, `hero-*`) consume so each example doesn't re-implement its own test scene.

## Exports

| Symbol | What it builds |
|--------|----------------|
| `buildCornellBoxThreeScene()` | The classic Cornell box: red/green/white walls, single area-light ceiling, two boxes. Used as the minimal demo for `@vitrum/pt-webgl` and `@vitrum/pt-webgpu`. |
| `buildComplexThreeScene()` | A heavier scene with mixed materials (PBR + glass + emissive) used by `two-engines-one-scene` and the hybrid GI walkthroughs. |
| `buildBenchmark200kThreeScene(targetTriangles?)` | Cornell + subdivided floor (~200k tris by default) for `PR-hybrid-200k-static`. |
| `buildTlas10InstThreeScene()` | Cornell + one `InstancedMesh` ×10 for `PR-hybrid-tlas-10-inst`. |
| `countThreeSceneTriangles(scene)` | CPU triangle counter for benchmark assertions. |

Both functions return a `THREE.Scene`. Examples pass that scene to `@vitrum/three-bindings`' `sceneFromThreeJS` to get the engine-agnostic `@vitrum/core` `Scene` representation.

## Why a shared package?

Each runnable example is a separate vite app. Hoisting scene construction here keeps the examples focused on demonstrating engine API surface rather than re-asserting Cornell-box geometry coordinates. It also makes any A/B comparison across engines (e.g. `two-engines-one-scene`) a one-line `import` instead of a copy.

## Not a vitrum public surface

This package lives under `examples/`, not `packages/`. It is never published, never imported from `@vitrum/*` packages, and exists only for example-level scene reuse. If you reach into it from a vitrum package, that's a bug.
