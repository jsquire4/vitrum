# Changelog

All notable changes to this monorepo will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) **once** packages leave **pre-alpha** (see root `README.md`). Until then, versions may remain `0.0.0` and breaking API moves are noted here without minor/major bumps.

## [Unreleased]

### Added

- **`scripts/run-fork-shader-smoke.mjs`** + root **`npm run fork-shader-smoke`** — runs sibling `three-gpu-pathtracer` shader string regression (`VITRUM_FORK_DIR` override).
- **`tools/benchmark-runner/scenario-presets.mjs`** — canonical scenario metadata for gap-closure captures.
- **`plan/renderer-fidelity-matrix.md`** — capability honesty matrix (WebGL vs WebGPU).
- **`CONTRIBUTING.md`** — contributor commands and evidence gates.
- **`.github/workflows/ci.yml`** + root **`npm run release:dry-run`** — CI-equivalent mechanical checks and package dry-run scaffold.
- Workspace **`npm run typecheck`** — runs `tsc --noEmit` in each package that defines a `typecheck` script.
- **`plan/binding-babylon-sketch.md`** — future second-binding checklist against `@vitrum/core`.
- **`plan/walkaround-without-three.md`** — module-by-module note on THREE coupling and RC re-composition scope.
- **`@vitrum/walkaround-hybrid`**: exported **`hostScene/types.ts`** seam types (`WalkaroundBVHSceneRoot`, `WalkaroundDDGIScene`, `WalkaroundThreeHostScene`).
- Unit tests: **`@vitrum/three-bindings`** (`sceneFromThreeJS` smoke), **`@vitrum/pt-webgl`** (factory validation).
- **`@vitrum/pt-webgpu`**: initial functional backend prototype (no longer stub), including:
  - progressive accumulation compute path,
  - CPU-built BVH + GPU BVH traversal,
  - multi-bounce diffuse/specular/emissive baseline shading,
  - directional + point direct-light support,
  - package test suite (`scenePack`, `buildCpuBvh`).
- **`packages/pt-webgpu/README.md`** documenting current capabilities, limits, and next steps.

### Changed

- **Fork (`three-gpu-pathtracer`)**: spectral Beer–Lambert reads packed μ(λ); NEE threads hero wavelength; `transmissionEval` PDF uses Walter-style GGX Jacobian with matching GGX half-vector sampling (shader smoke checks extended).
- **`@vitrum/pt-webgpu`**: material stride **22** vec4s with thin-film extinction + incident IOR; **multi-light** storage buffers and WGSL direct-light loops; `brdfDirectionalPdf` includes an active transmission-hemisphere term; Playwright adapter passes scenario query params and viewport size from env.
- **`@vitrum/pt-webgl`**: package is explicitly `private` while it depends on the sibling fork via `file:`.
- **`@vitrum/three-bindings`**: RFE `userData` stamping tests on production `vitrumSceneToThree`; duplicate `pt-webgl` scene converter removed.
- **`HybridEngine`**: RC note in file header now points to `plan/walkaround-without-three.md` instead of a dangling TODO.
- Repo docs now reflect that **`@vitrum/pt-webgpu`** has moved from stub to active prototype (`README.md`, `plan/library-architecture.md`, `plan/phase-6-status.md`).
