# Changelog

All notable changes to this monorepo will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) **once** packages leave **pre-alpha** (see root `README.md`). Until then, versions may remain `0.0.0` and breaking API moves are noted here without minor/major bumps.

## [Unreleased]

### Added

- **`OIDNDenoiseOptions.tensorNames`** — optional ONNX I/O tensor name overrides (`@vitrum/shared-denoisers`).
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
- **`@vitrum/core`**: `GpuDetection.adapterKind` (`WgpuAdapterKind`), optional `detectGpu({ publishToWindow })`, `probeWebGPU()` returns `adapterKind`; `probeWebGPU` / `isSwiftShaderAdapter` re-exported; JSDoc on `Material` mutability vs `readonly material` on primitives.
- **`@vitrum/walkaround-hybrid`**: `HybridEngineOptions.verbose` gates ReSTIR pipeline init / compile `console.log` calls (still enabled when `debug` is true).
- **`@vitrum/pt-webgpu`**: WGSL sampling via `@vitrum/shared-samplers` (`HAMMERSLEY_WGSL`, `OCTAHEDRAL_CORE_WGSL`); deprecated re-export alias `OCTAHEDRAL_WGSL`.

### Changed

- **Docs / packaging honesty**: root `README.md` (denoisers list, walkaround capability wording), `@vitrum/shared-denoisers` / `@vitrum/pt-webgpu` package descriptions, `packages/walkaround-hybrid` README + `package.json` + `src/index.ts` header — align marketing with shipped code (BMFR not exported; RC toolbox vs `HybridEngine` composition).
- **`@vitrum/walkaround-hybrid`**: Sprint 10a **SVGF** path wired in `WalkaroundGPUPipeline` (default): temporal Welford luminance pass + `svgfVarianceMain` + five `svgfAtrousMain` iterations; `HybridEngineOptions.denoiser` (`'svgf'` default, `'atrous'` legacy). New frame textures: Welford ping-pong, `svgfVarianceEstimateTexture`, zero `motionVectorTexture`. `welfordTemporal.wgsl.ts` + expanded GPU timestamp pass slots (14).
- **`@vitrum/shared-denoisers`**: `svgf.wgsl` **à-trous** pass now decodes packed G-buffer like walkaround `atrous.wgsl` (normal `xyz*2-1`, depth `.w`).
- **`tools/benchmark-runner/README.md`**: documents `VITRUM_STRICT_GAP_CLOSURE=1` strict exit behavior (with `run-gap-closure-verification.mjs`).
- **`plan/library-architecture.md`**: fork consumption via sibling `file:` `three-gpu-pathtracer`; BMFR call-out; `@vitrum/pt-webgl` / `pt-webgpu` dependency bullets per actual `package.json`.
- **`@vitrum/pt-webgpu`**: `pathTraceBruteforce.wgsl` — shared `traceMeshBvh` / `traceAnalyticShapes` backing for `traceClosest` and `traceAny` (D2 deduplication).
- **`@vitrum/walkaround-hybrid`**: Sprint 11 PPG spatial lookup — kd-tree buffer + GPU traversal in `ppgSample` and `ppgUpdate`; host `writePpgKdTree` and `buildPpgKdTreeGpuBytes` (E6).
- **Fork (`three-gpu-pathtracer`)**: spectral Beer–Lambert reads packed μ(λ); NEE threads hero wavelength; `transmissionEval` PDF uses Walter-style GGX Jacobian with matching GGX half-vector sampling (shader smoke checks extended).
- **`@vitrum/walkaround-hybrid`**: RC `cascadeDispatch` / `bvhCompute` / `nodeMaterialUpgrade` typing cleanups (backend view, `instanceof` material packing, `Record` copy).
- **`@vitrum/babylon-bindings`** (removed 2026-05-11): stub package deleted — zero external consumers and no implementation work scheduled. Babylon binding remains a future option via `plan/binding-babylon-sketch.md`.
- **`@vitrum/pt-webgpu`**: material stride **22** vec4s with thin-film extinction + incident IOR; **multi-light** storage buffers and WGSL direct-light loops; `brdfDirectionalPdf` includes an active transmission-hemisphere term; Playwright adapter passes scenario query params and viewport size from env.
- **`@vitrum/pt-webgl`**: package is explicitly `private` while it depends on the sibling fork via `file:`.
- **`@vitrum/three-bindings`**: RFE `userData` stamping tests on production `vitrumSceneToThree`; duplicate `pt-webgl` scene converter removed.
- **`HybridEngine`**: RC note in file header now points to `plan/walkaround-without-three.md` instead of a dangling TODO.
- Repo docs now reflect that **`@vitrum/pt-webgpu`** has moved from stub to active prototype (`README.md`, `plan/library-architecture.md`, `plan/phase-6-status.md`).
- **`@vitrum/pt-webgpu`**: **`disc-area`** emitters → rect-area prototype packing (orthogonal half-spans √(π)·r/2 so quad patch area equals π·r²; not uniform disc sampling); **`supportedEmitterKinds`** includes **`disc-area`**; **`renderFrame`** reuses **`GPUBindGroup`** across frames until **`setScene`**, accum texture/realloc or **`dispose`**; **`paused`** **`isConverged`** uses **`samplesTarget`** like the accumulating path.
- **`@vitrum/three-bindings`**: **`vitrumSceneToThree`** **`disc-area`** surrogate + **`mesh-area`** radiance merged into **`MeshPhysicalMaterial.emissive`**; **`sceneFromThreeJS`** adds **`mesh-area`** for emissive meshes and zeros primitive **`emissive`** to prevent double emission.
- **`@vitrum/walkaround-hybrid`** (`InferenceGraph`): clarified buffer contract; **`run()`** enforces **`inputTensors`/`outputTensors`** presence in maps; per-layer **`GPUBindGroup`** cache reduces allocation churn until **`dispose`**.
- **`@vitrum/pt-webgl`**: **`transmissiveBounces`** now **`min(per-frame`** **`bounces,`** **`maxBouncesLimit)`** (**no** hardcoded **`12`** cap vs structural limit).
