# Complexity sweep — corrected findings & remediation plan

Derived from verified code reads (May 2026). Items framed as **bug** vs **missing feature** vs **design debt**. Execution order respects dependencies.

---

## A. Corrected finding list

### A1. Bugs / incorrect behavior

| ID | Location | Issue | Verification |
|----|----------|--------|---------------|
| B1 | `packages/pt-webgpu/src/index.ts` | Paused `renderFrame`: `isConverged` uses `samplesAccumulated >= maxSamplesLimit` while active path uses `targetSpp = min(samplesTarget ?? 16, maxSamplesLimit)`. Inconsistent convergence semantics across pause/active. | Code ~292–301 vs ~486–494 |
| B2 | `HybridEngine`: `HybridEngineOptions.pipelineRebuildKey` | Option is documented but **never read** — hosts cannot rely on it. | `rg` only finds type field |

*B1-note:* `configureAdditiveAccumulation(enabled, enabled)` is **two real parameters** (second is `displayDivideByAlpha` per fork API) — **not** a duplicate-arg typo; only remediate if you want independent control.

### A2. Missing or partial implementation

| ID | Location | Gap |
|----|----------|-----|
| M1 | `three-bindings/vitrumSceneToThree.ts` | `disc-area`, `mesh-area` emitters hit `default` → warn + skip. |
| M2 | `three-bindings/index.ts` (`sceneFromThreeJS`) | Emissive meshes not converted to `MeshAreaEmitter` (comment admits gap). |
| M3 | `packages/pt-webgpu` | Scene pack + WGSL + `capabilities.supportedEmitterKinds` omit **`disc-area`** though `@vitrum/core` defines it. |
| M4 | `InferenceGraph.ts` | Comment promises lazy allocation for intermediate tensors in `run()`; **no allocation** occurs — callers must pre-pass all output buffers via `outputs` map. |
| M5 | `WalkaroundGPUPipeline.ts` | `cellPowerBuffer` uploaded/destroyed but **never bound** into any shader / bind group. |
| M6 | `WalkaroundGPUPipeline.updateEmitters` | **Never called** in-repo; if used, **`cellPower` would diverge** (not refreshed). |
| M7 | `HybridEngine.renderFrame` | `FrameInput.quality.bounces` clamped then **`void`** — bounce count not wired to GPU pipeline (fixed at shader compile time). |
| M8 | `packages/babylon-bindings` | Public API **`sceneFromBabylonScene` always throws** — placeholder only. |
| M9 | `shared-samplers/jakobHanika.ts` | Placeholder spectral upsampling; in-code TODO for full Jakob/Hanika table pending distribution. |

### A3. Design / performance debt (not strictly “broken”)

| ID | Location | Issue |
|----|----------|--------|
| D1 | `pt-webgpu/index.ts` | New `GPUBindGroup` every `renderFrame` — allocation churn. |
| D2 | `pt-webgpu` | `PROTOTYPE_MAX_BOUNCES = 8` clamps constructor; factory warns but `capabilities.maxBounces` reflects clamp — OK if documented; hostile if callers expect uncapped Structural cap without reading warn. |
| D3 | `pt-webgpu` | Manual `paramsF32`/WGSL `FrameParams` layout — drift risk vs shader. |
| D4 | `pt-webgl/index.ts` | `transmissiveBounces = min(b, 12)` independently of `@vitrum/core` structural `maxBounces`. |
| D5 | `restir/bvhCompute.ts` | `DEFAULT_PROXY_MESH_NAMES` hardcodes stained-glass-style mesh names as **library default**. |
| D6 | `InferenceGraph.run` | `createBindGroup` per layer **per call** — hot-path cost when integrated. |
| D7 | `HybridEngine` | Init failure maps to `state === 'disposed'` — conflates user dispose vs fatal error. |
| D8 | `oidnBridge.ts` | Assumes ONNX tensor names `color`/`output`/`albedo`; brittle across exports. |
| E1 | Examples | `examples/cornell-box/src/main.ts` monolith (~700 LOC) mixes URL parsing, capture globals, renderer — maintainability only. |

### A4. Intentionally excluded from remediation (unless product changes)

- **`TextureRef = unknown`, `extensions` bags** — core escape hatch by design.
- **`detectGpu` writing `window.__WG__`** — documented; optional refactor to inject publisher later.
- **Large monolithic WGSL** — refactor is large; schedule only if editing pain threshold reached.

---

## B. Remediation plan

### Phase 0 — Contract truth & quick wins (1–2 days)

1. **B2 `pipelineRebuildKey`:** Either implement (compare to previous key in `setScene`/`reset` and force pipeline rebuild + internal `reset()`), or **remove** from `HybridEngineOptions` and document breaking change in CHANGELOG. *Tests:* unit test that changing key triggers `_teardownPipeline` + reinit path.
2. **B1 pt-webgpu pause convergence:** Use the **same** `targetSpp` rule as active path when paused (read `FrameInput.quality?.samplesTarget` with same `?? 16` default, clamp to `maxSamplesLimit`). *Tests:* extend existing pt-webgpu tests.
3. **M4 InferenceGraph:** Fix **comments only** in one PR *or* implement lazy allocation (Phase 3) — do not leave misleading “lazy in run()” text.
4. **D5 proxy defaults:** Change `buildSceneBVH` default to **`empty Set`** for `proxyMeshNames`; require hosts to pass names explicitly. *Tests:* `restir`/shared BVH tests expecting old default need updating.

### Phase 1 — Three-bindings parity (core ↔ THREE)

1. **M1 `vitrumSceneToThree`:** Add `disc-area` (map to closest THREE light or document use of custom helper + `RectAreaLight` approximation). Add **`mesh-area`** by resolving `meshId` to scene mesh and sampling / emissive panel pattern as project requires (may need material emissive read from attached mesh).
2. **M2 `sceneFromThreeJS`:** Detect emissive `MeshPhysicalMaterial` (and standard path if applicable) and emit **`MeshAreaEmitter`** referencing mesh id with stable `SceneNodeId`.
3. **Tests:** Round-trip / golden tests in `three-bindings/__tests__` for new emitter kinds; ensure `pt-webgl` path tracer scene still builds.

*Dependency:* M1/M2 before or in parallel with M3 (pt-webgpu) if using same scenes.

### Phase 2 — Walkaround GPU pipeline wiring

1. **M5/M6:** Either **bind `cellPowerBuffer`** in the appropriate compute bind group + consume in WGSL (RIS/shade as per sprint spec), **or** stop allocating/uploading until needed. If **binding**, extend **`updateEmitters`** to re-upload **`cellPower`** alongside emitters/CDF or remove `updateEmitters` until light edits are supported.
2. **M7 `HybridEngine` bounces:** Pass effective bounce count into `WalkaroundGPUPipeline` / UBO / shader defines (may require **pipeline recompile** on bounce change — then tie to `pipelineRebuildKey` or document “creation-time only”). If dynamic recompile is too heavy, **stop reading** `quality.bounces` in `HybridEngine` and set `capabilities` to reflect fixed bounces — **honest capability**.

### Phase 3 — pt-webgpu completeness

1. **M3 disc-area:** Extend `uploadSceneBuffers` / packing, WGSL sampling, **`supportedEmitterKinds`**, and tests (`scenePack` / factory capabilities).
2. **D1 bind group caching:** Cache `GPUBindGroup` when scene buffers + accum views + dimensions unchanged; invalidate on `setScene`, resize, or pipeline rebuild.
3. **D3 layout safety:** Introduce shared layout (e.g. `ParamsBufferWriter` mirror of WGSL struct, or codegen comment block) — single source of indices.

### Phase 4 — Inference graph (neural denoiser path)

1. Implement **automatic intermediate buffer sizing** from `InferenceGraphSpec` + `layer.params` (element counts), **or** formally require `outputs` to list every tensor and **enforce** via validation at `initialize()`/`run()` with clear errors.
2. **D6:** Cache bind groups per layer immutable inputs; recreate only when dimensions/weights change.
3. **Tests:** Minimal 2-layer graph on mock `GPUDevice` or headless Dawn if available — assert no throw and buffer lifecycle.

### Phase 5 — Secondary packages & polish

1. **M8 Babylon:** Either `@vitrum/babylon-bindings` **minimal walker** (`Mesh` → `mesh` primitives, lights parity with three-bindings subset) **or** **remove**/don’t publish until ready; avoid exported throw-only API without `experimental` naming.
2. **D8 oidnBridge:** Options field `tensorNames?: { color, normal?, albedo?, output? }` with sane defaults.
3. **M9 Jakob/Hanika:** Legal review table import → TypeScript lut + parity tests vs reference implementation snippets.
4. **D7 HybridEngine:** Add `'error'` to `EngineState` in **`@vitrum/core`** **only if** all backends migrated; alternately document “`disposed` after fatal init” in `HybridEngine` JSDoc + host guide (smaller blast radius).
5. **E1 Cornell example:** Split `captureConfig` / URL parse / RAF loop into `cornell/` modules — optional cleanup sprint.

---

## C. Definition of done (global)

- [ ] Every **B*** item resolved or superseded by ADR.
- [ ] Every **M*** item implemented **or** capability matrix updated so **no false advertising** (`supportedEmitterKinds`, `supportsIncrementalScene`, etc.).
- [ ] **`npm run typecheck`** and **`npm test`** green.
- [ ] **`CHANGELOG.md`** entries for behavioral or API changes.
- [ ] New behavior covered by Vitest **where deterministic** (CPU-side packers, convergence math, InferenceGraph allocator).

---

## D. Suggested execution order

`Phase 0` → `Phase 1` (bindings) ⇄ `Phase 3` (pt-webgpu disc) in parallel if staffed → `Phase 2` (walkaround) → `Phase 4` (neural) → `Phase 5` (babylon OIDN jakob example polish).
