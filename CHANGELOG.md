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
- **`@vitrum/core`**: Tier 1 **`SpectralCurve`** + optional **`Material.spectralAttenuation`** / **`dispersionAbbeNumber`** (RFE 01 contract surface).
- **`@vitrum/three-bindings`**: **`VITRUM_SPECTRAL_EXTENSION_KEY`** stub export.
- **`@vitrum/babylon-bindings`**: stub package with **`sceneFromBabylonScene`** (`Not implemented`).
- **`examples/shared`**, **`examples/two-engines-one-scene`** — shared Cornell builder and **G2** demo (one `Scene` → pt-webgl + walkaround-hybrid).
- **`@vitrum/pt-webgl`**: Cornell → core **golden summary** unit test; spectral fields forwarded on **`MeshPhysicalMaterial.userData.vitrumSpectral`**.

### Changed

- **`createWalkaroundEngine_Hybrid`**: factory bootstraps with a valid empty **`Scene`** instead of `{} as Scene`.
- **`examples/cornell-box`**: uses **`@vitrum-examples/shared`** for the Cornell THREE scene.
- **`_staging/`**: **`legacy-source/`** tree removed; README points at git history and packages.
- **`HybridEngine`**: RC note in file header now points to `plan/walkaround-without-three.md` instead of a dangling TODO.
