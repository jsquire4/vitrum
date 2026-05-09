# Changelog

All notable changes to this monorepo will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) **once** packages leave **pre-alpha** (see root `README.md`). Until then, versions may remain `0.0.0` and breaking API moves are noted here without minor/major bumps.

## [Unreleased]

### Added

- Workspace **`npm run typecheck`** — runs `tsc --noEmit` in each package that defines a `typecheck` script.
- **`plan/binding-babylon-sketch.md`** — future second-binding checklist against `@vitrum/core`.
- **`plan/walkaround-without-three.md`** — module-by-module note on THREE coupling and RC re-composition scope.
- **`@vitrum/walkaround-hybrid`**: exported **`hostScene/types.ts`** seam types (`WalkaroundBVHSceneRoot`, `WalkaroundDDGIScene`, `WalkaroundThreeHostScene`).
- Unit tests: **`@vitrum/three-bindings`** (`sceneFromThreeJS` smoke), **`@vitrum/pt-webgl`** (factory validation).

### Changed

- **`HybridEngine`**: RC note in file header now points to `plan/walkaround-without-three.md` instead of a dangling TODO.
