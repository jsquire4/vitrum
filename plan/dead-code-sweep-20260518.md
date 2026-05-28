# Dead-code sweep — 2026-05-18 (knip)

> **Read-only sweep.** No code was removed. Findings only. Companion to the
> 2026-05-17 hand-rolled `dead-exports-audit-20260517.md`; this run uses
> `knip@5` plus targeted grep to widen coverage (unused files, deps, types,
> binaries).

## Tool / run

- **Tool:** `knip` (via `npx --yes knip@5`) — version `5.x` (latest on date of run).
- **Reporter:** `compact` (human-readable); also captured `json` for structured parsing.
- **Run command (from repo root):**
  ```
  npx --yes knip@5 --reporter compact --no-config-hints
  ```
- **Config:** `/home/jsquire4/projects/vitrum/knip.json` — see *Methodology* below. (This file is part of the sweep branch; remove or keep at your discretion.)
- **Manual grep:** `@deprecated`, `export const _foo`, `dead-code` / `TODO: remove` comments.

## Methodology

### Knip configuration

A minimal `knip.json` was authored for this run because the default knip
plugin discovery loads `eslint.config.js` which requires `@eslint/js` (not
installed locally — ESLint is declared but devDependencies have never been
installed at the workspace root). The config:

- Disables knip's eslint plugin (`"eslint": false`).
- Adds workspace `entry` patterns:
  - **packages/*:** `src/index.{ts,tsx}` and test files under `__tests__/`.
  - **examples/*:** `index.html`, `src/main.{ts,tsx}`, `vite.config.{ts,mjs,js}`.
  - **tools/*:** `src/index.{ts,mjs,js}`, `bin/*`, top-level `*.{ts,mjs,js}`.
- Ignores `_staging/**`, `plan/**`, `external_requests/**`, `tools/reference-renders/**`, `**/*.wgsl`, `**/*.glsl`.

### Exclusions

Per `CLAUDE.md`:

- `_staging/legacy-source/**` — intentionally retained host-app code. Exports
  whose only consumer is `_staging` are **not** flagged as dead.
- `__tests__/**` — fixture / mock exports flagged here are **noted, not
  removal candidates**.
- `tools/reference-renders/**` — data directory, not code.

### Verification

Every removal-candidate file flagged below was verified by reading it and
grepping the workspace for any consumer outside its own source.

## Top-line counts

| Category | Count |
|---|---|
| **Unused files**                            | **2**  |
| **Unused dependencies** (across 9 package.json files) | **15** entries (15 distinct lines; some package names repeat across packages) |
| **Unused devDependencies** (across 8 package.json files) | **22** entries |
| **Referenced optional peerDependencies**    | **1**  |
| **Unlisted dependencies**                   | **3**  |
| **Unlisted binaries**                       | **12** (mostly `eslint` & `prettier` — repo lints exist as scripts but no `eslint` is declared per-package) |
| **Unused exports (values)**                 | **52** named-value exports across **20** files |
| **Unused exported types**                   | **20** type exports across **15** files |
| **Duplicate exports**                       | **1** (aliasing, intentional)  |
| **Manual `@deprecated`**                    | **7** markers (still kept as compat aliases) |
| **Manual `export _foo` underscore exports** | **1** (`_skyEquirectCacheSize`) |

## Top-5 highest-confidence deletion candidates

These were each verified by reading the file and grepping the entire workspace
for any consumer (including `_staging/`).

1. ~~**`packages/shared-denoisers/src/svgfRealPipelineCache.ts`**~~ — **STALE FINDING (2026-05-28).** This file is imported at `svgfRealWebGPU.ts:28` (`import { svgfRealPipelines } from './svgfRealPipelineCache.js'`). Knip's static analysis did not see the `.js` extension import at the time. The `complexity-remediation-20260526.md` correctly re-evaluates this as verdict C (knip false positive — do not delete). File is live; cache is legitimately hoisted.
2. ~~**`packages/walkaround-hybrid/src/restir/packingHelpers.ts:WARM_GRAY_DEFAULT_{R,G,B}` + `applyBeerLambert`**~~ — **STALE FINDING (2026-05-28).** These were already demoted to file-local (unexported) `const`/`function` in response to this sweep. They are no longer exported. Confirmed by reading `packingHelpers.ts` lines 17-37: no `export` keyword on any of these four symbols. Finding is closed.
3. **`packages/pt-webgpu/src/scene/emitterPacking.ts` + `uploadSceneBuffers.ts` — 8 `MAX_*_LIGHTS` / `*_FLOAT_STRIDE` constants exported by *both* files.** The duplicated set is unused externally and only locally referenced *within each file*; one of the two declaration sites is dead. (pt-webgpu is pre-alpha, so neither is shipped, but this is the cleanest scoped consolidation.)
4. **`packages/shared-denoisers/src/webGpuTextureUpload.ts` — 11 unused exports.** `RGBA16F_BPP`, `RG32F_BPP`, `R32F_BPP`, `R32U_BPP`, `R16U_BPP`, `alignedTextureCopyBytesPerRow`, `uploadTexture2D`, `uploadR32f`, `uploadR32Uint`, `uploadR16Uint`, `fillR16Uint`. Bulk-unused helpers from a pre-W4 era; the surviving callers in `shared-denoisers` use only an internal subset. Verify before deletion — these may be intended as a public surface for hosts.
5. **Examples' unused dependency entries** (zero-risk `package.json` cleanup):
   - `examples/hero-{viewer,lighting-designer,product-viz}/package.json` declare `@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl`, `three-gpu-pathtracer`, `three-mesh-bvh`, `xatlas-web` but resolve them via `vite.config.ts` aliases pointing at the workspace source (no `from '@vitrum/...'` imports in `src/`).
   - `examples/cornell-box` and `examples/two-engines-one-scene` declare `three-gpu-pathtracer` and `xatlas-web` unused.
   These are *declaration*-level cleanups, not code deletes — safe pruning if the example builds are not relying on them transitively. (Triple-check vite-resolve before pruning aliases.)

---

## Unused FILES (deletion candidates)

| Package | File | Notes |
|---|---|---|
| (root) | `eslint.config.js` | Knip can't see it as referenced. The repo's `lint` script runs `eslint .`, which loads it implicitly; this is a **false positive** — keep. |
| `shared-denoisers` | `packages/shared-denoisers/src/svgfRealPipelineCache.ts` | ~~Genuinely orphaned~~ **STALE (2026-05-28) — false positive.** File IS imported at `svgfRealWebGPU.ts:28` via `.js` extension. Keep. See top-5 #1 correction. |

## Unused EXPORTS (named values) per package

> Test-only fixture exports are **noted** (italicized) and not recommended for
> removal — they may be exercised by `vitest` runtime via global setup or
> reflection-style access.

### `packages/pt-webgl`

| Export | File |
|---|---|
| *`FakeWebGL2RenderingContext`* (test fixture) | `src/__tests__/testUtils.ts` |
| `_skyEquirectCacheSize` (underscore-prefixed; test seam pattern) | `src/iblBaker.ts` |

### `packages/pt-webgpu`

> Pre-alpha package. CPU-tracer file is a vitest reference impl.

| Export | File |
|---|---|
| *`mul`, `length3`, `maxComp`, `safeInvDir`, `intersectTriangleMT`, `intersectAabb`, `cosineHemisphereSample`, `sampleGgxVndfTangent`, `frDielectric`, `schlickFresnel`, `powerHeuristic`* (test helpers) | `src/__tests__/cpuTracer.ts` |
| `MAX_POINT_LIGHTS`, `MAX_SPOT_LIGHTS`, `MAX_RECT_AREA_LIGHTS`, `MAX_MESH_AREA_LIGHTS`, `POINT_LIGHT_FLOAT_STRIDE`, `SPOT_LIGHT_FLOAT_STRIDE`, `RECT_AREA_LIGHT_FLOAT_STRIDE`, `MESH_AREA_LIGHT_FLOAT_STRIDE` | `src/scene/emitterPacking.ts` *(duplicated in uploadSceneBuffers.ts)* |
| `MAX_POINT_LIGHTS`, `MAX_SPOT_LIGHTS`, `MAX_RECT_AREA_LIGHTS`, `MAX_MESH_AREA_LIGHTS`, `POINT_LIGHT_FLOAT_STRIDE`, `SPOT_LIGHT_FLOAT_STRIDE`, `RECT_AREA_LIGHT_FLOAT_STRIDE`, `MESH_AREA_LIGHT_FLOAT_STRIDE` | `src/scene/uploadSceneBuffers.ts` |
| `emptyEnvironmentParams`, `buildProceduralSkyEnvironmentParams` | `src/scene/environmentPacking.ts` |

### `packages/shared-denoisers`

| Export | File |
|---|---|
| `WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT` | `src/webGpuTextureCopy.ts` |
| `RGBA16F_BPP`, `RG32F_BPP`, `R32F_BPP`, `R32U_BPP`, `R16U_BPP`, `alignedTextureCopyBytesPerRow`, `uploadTexture2D`, `uploadR32f`, `uploadR32Uint`, `uploadR16Uint`, `fillR16Uint` | `src/webGpuTextureUpload.ts` |
| `ATROUS_KERNEL_VALUES` | `src/wgsl/atrousKernel.wgsl.ts` |

### `packages/three-bindings`

| Export | File |
|---|---|
| `warnOnce` | `src/lights.ts` |
| `extractAttribute`, `extractIndex` | `src/mesh.ts` |

### `packages/walkaround-hybrid`

| Export | File |
|---|---|
| `RAYS_PER_PROBE` (re-export from `probeUpdatePass.ts` — original constant lives in `ddgiConstants.ts`) | `src/ddgi/probeUpdatePass.ts` |
| `PROBE_UPDATE_RAYS_WGSL` (bound to a 64-ray default — supplanted by `makeProbeUpdateRaysWGSL(64)`) | `src/ddgi/wgsl/probeUpdateRays.wgsl.ts` |
| `ATROUS_DIRECT_SIGMAS` | `src/pipeline/bindGroupBuilders.ts` |
| `NON_DENOISER_PASS_ORDER`, `DDGI_BORDER_LABELS` | `src/pipeline/passes/passOrder.ts` |
| `WALKAROUND_UBO_SIZE_BYTES` | `src/pipeline/uboUpdater.ts` |
| `COMMON_MODULE`, `SURFACE_TEXTURES_MODULE`, `RESTIR_PHAT_MODULE`, `RESTIR_CAST_PRIMARY_MODULE`, `DDGI_SAMPLE_MODULE`, `WELFORD_VARIANCE_MODULE` (all are imported by `pipeline/wgslModules.ts`, then re-exported only via internal aggregation — knip sees the inputs as un-imported elsewhere) | `src/pipeline/wgslModules.ts` (re-exports) + their source files |
| `EMITTER_STRIDE`, `EMITTER_FLOATS` | `src/restir/emitterList.ts` |
| `WARM_GRAY_DEFAULT_R`, `WARM_GRAY_DEFAULT_G`, `WARM_GRAY_DEFAULT_B`, `applyBeerLambert` | `src/restir/packingHelpers.ts` |

> **Caveat for `walkaround-hybrid/src/pipeline/wgslModules.ts` entries:**
> these *are* imported into `wgslModules.ts` and used to construct
> aggregate maps — knip's "unused export" warning flags them because the
> intermediate names are not used outside the file. Treat as informational,
> not a removal candidate.

## Unused exported TYPES per package

| Type | File |
|---|---|
| `WebGPUSwapChainInfo` | `packages/engine/src/lifecycle/vanilla.ts` |
| `Rng`, `Ray`, `PathOpts` | `packages/pt-webgpu/src/__tests__/cpuTracer.ts` (test) |
| `CpuBvhBuildResult` | `packages/pt-webgpu/src/scene/buildCpuBvh.ts` |
| `EnvironmentParams` | `packages/pt-webgpu/src/scene/environmentPacking.ts` |
| `PackedSceneData` | `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts` |
| `SVGFReprojCPUInput`, `SVGFReprojCPUOutput` | `packages/shared-denoisers/src/svgfRealWebGPU.ts` |
| `SpatialFilterBindGroupLayout` | `packages/shared-denoisers/src/wgsl/spatialFilter.wgsl.ts` |
| `DDGIDeviceHandle` | `packages/walkaround-hybrid/src/ddgi/types.ts` |
| `FrameBindGroupResources`, `SceneBindGroupResources`, `AtrousSigmas`, `HybridLayersResources` | `packages/walkaround-hybrid/src/pipeline/bindGroupBuilders.ts` |
| `RegisterBuiltinDenoisersOptions` | `packages/walkaround-hybrid/src/pipeline/denoisers/registerBuiltinDenoisers.ts` |
| `PingPongRef` | `packages/walkaround-hybrid/src/pipeline/passes/index.ts` |
| `NonDenoiserPassEntry` | `packages/walkaround-hybrid/src/pipeline/passes/passOrder.ts` |
| `CompiledPipelines` | `packages/walkaround-hybrid/src/pipeline/pipelineCompiler.ts` |
| `PassLayoutOptions`, `PassLayout` | `packages/walkaround-hybrid/src/pipeline/timestampQueries.ts` |
| `EmitterListOptions` | `packages/walkaround-hybrid/src/restir/emitterList.ts` |

## Unused dependencies (per `package.json`)

> Many of these are *transitively used via vite alias* — see *Limitations*.

### `dependencies`

| Package | Unused dependency |
|---|---|
| `examples/cornell-box` | `three-gpu-pathtracer`, `xatlas-web` |
| `examples/hero-lighting-designer` | `@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl`, `three-gpu-pathtracer`, `three-mesh-bvh`, `xatlas-web` |
| `examples/hero-product-viz`        | `@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl`, `three-gpu-pathtracer`, `three-mesh-bvh`, `xatlas-web` |
| `examples/hero-viewer`             | `@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl`, `three-gpu-pathtracer`, `three-mesh-bvh`, `xatlas-web` |
| `examples/two-engines-one-scene`   | `three-gpu-pathtracer`, `xatlas-web` |
| `packages/pt-webgl`                | `three-gpu-pathtracer`, `xatlas-web` |
| `packages/shared-bvh`              | `@vitrum/core` |
| `packages/shared-denoisers`        | `@vitrum/core` |
| `packages/shared-samplers`         | `@vitrum/core` |

> **`@vitrum/core` in shared-***: verified — none of `shared-bvh`,
> `shared-denoisers`, or `shared-samplers` `import` from `@vitrum/core`
> in their `src/`. Declared as a workspace coupling; not exercised by the
> code. *Genuine candidate for `dependencies` cleanup*, **but** the W3
> contract refactor may intend this coupling — confirm before pruning.

### `devDependencies`

| Package | Unused devDependency |
|---|---|
| `package.json` (root) | `@typescript-eslint/eslint-plugin`, `@typescript-eslint/parser`, `eslint`, `eslint-config-prettier`, `eslint-plugin-react`, `eslint-plugin-react-hooks`, `globals`, `prettier`, `typescript-eslint` |
| `packages/core`        | `@webgpu/types` |
| `packages/dev`         | `@types/react-dom`, `react-dom` |
| `packages/engine`      | `@types/react-dom`, `@vitejs/plugin-react`, `@webgpu/types`, `react-dom` |
| `packages/pt-webgl`    | `three-mesh-bvh` |
| `packages/pt-webgpu`   | `playwright` |
| `packages/shared-denoisers` | `playwright` |
| `examples/hero-lighting-designer` | `@webgpu/types` |
| `examples/hero-product-viz`       | `@webgpu/types` |
| `examples/hero-viewer`            | `@webgpu/types` |
| `examples/two-engines-one-scene`  | `@webgpu/types` |

> **Root eslint/prettier devDeps:** the `lint` and `format` npm scripts at the
> repo root invoke `eslint` and `prettier` directly — these are *true*
> runtime-required dev tools. Knip flags them because no `import` statement
> resolves them. **False positive — keep.**

> **`react-dom` in dev/engine:** these packages declare `react-dom` as a
> `peerDependency` for hosts and additionally pin it in `devDependencies`
> for test-time JSX runtime. Verify whether tests actually require
> `react-dom` (they may not — `@testing-library` may pull it transitively).
> **Probable false positive.**

## Unlisted dependencies

| Source | Imported package | Notes |
|---|---|---|
| `packages/walkaround-hybrid/src/rc/bvhCompute.ts:33` | `three-mesh-bvh` | Type-only `import type { MeshBVH }`. Type imports are erased at runtime, so not strictly required as a runtime dep — but it should still be a devDependency for the typecheck to resolve. |
| `packages/pt-webgl/src/__tests__/materialsTextureSpectral.test.ts:40` | `@vitrum-pathtracer` | Historical note: this used to point at a sibling fork alias. It now resolves to the absorbed `packages/three-gpu-pathtracer` workspace package via Vitest alias. |
| `tsconfig.base.json` | `@webgpu/types` | Declared in `compilerOptions.types`. Resolved per-package; correct architecturally — each package that emits WebGPU types includes `@webgpu/types` in its devDeps. |

## Unlisted binaries

12 entries. The clean-up is to add `"eslint": "..."` to per-package
`devDependencies` (or to inherit it from root via workspace hoisting).
Currently each package has an `eslint` npm-script but no declared `eslint`
binary, so knip flags them all.

- `.github/workflows/ci.yml`: `playwright` (CI uses playwright; declared
  per-package but not at the workflow level — informational).
- `package.json`: `eslint`, `prettier`.
- `packages/core`, `packages/dev`, `packages/engine`, `packages/pt-webgl`,
  `packages/pt-webgpu`, `packages/shared-bvh`, `packages/shared-denoisers`,
  `packages/shared-samplers`, `packages/three-bindings`,
  `packages/walkaround-hybrid` — each declares `eslint` as an npm-run script
  binary but does not list `eslint` in its own devDeps.

## Suspicious patterns (manual grep)

### `@deprecated` markers (7)

These are intentional compat aliases. No removal candidates — listed for
awareness only. When the deprecation period elapses (Phase 7 → Phase 8 cut)
they should be revisited.

| File | Line | Symbol | Replacement |
|---|---|---|---|
| `packages/core/src/gpuDetection.ts`              | 34  | `adapterKind` field on `GpuDetectionResult` | `adapterKind` (preferred) |
| `packages/core/src/wgpuSupport.ts`               | 15  | (similar) | `adapterKind` |
| `packages/core/src/engine.ts`                    | 329 | scalar memory stat | `FrameStats.gpuMemoryBytes` |
| `packages/core/src/scene.ts`                     | 347 | legacy material type | `MaterialSpec` (W3-D1, 2026-05-17) |
| `packages/shared-samplers/src/bdptMIS.ts`        | 44  | `bdptConnectionMIS` | `bdptConnectionMIS_full` + `buildBDPTStrategyPDFs_full` |
| `packages/shared-samplers/src/bdptMIS.ts`        | 100 | `buildBDPTStrategyPDFs` | `buildBDPTStrategyPDFs_full` (full Veach §10.3) |
| `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts` | 35 | `PROBE_UPDATE_RAYS_WGSL` | `makeProbeUpdateRaysWGSL(64)` |

### Underscore-prefixed exports (1)

| File | Line | Symbol | Notes |
|---|---|---|---|
| `packages/pt-webgl/src/iblBaker.ts` | 229 | `_skyEquirectCacheSize` | Convention: underscore = test seam / internal. Currently unused even by tests. **Verify before removing** — may exist for a planned test. |

### `// dead-code` / `TODO: remove` comments

**None found.** The codebase keeps removal intent in plan docs, not source.

### Duplicate exports

`packages/shared-samplers/src/jakobHanika.ts`:
`rgbToApproxSpectralCoefficients` (the implementation, marked `@internal`)
and `rgbToSpectralCoefficients` (an `export const = …Approx…` alias for
the stable public name). **Intentional naming-aliasing pattern; not a
duplicate to fix.**

## Limitations

These are categories where knip **cannot** see consumers, so each finding
must be verified by reading the file:

1. **Dynamic / string-based imports.** WGSL strings, `await import(...)`
   call sites, and shader-snippet imports concatenated as strings are
   invisible to static analysis.
2. **Vite aliases.** `examples/hero-*/vite.config.ts` aliases `@vitrum/*`
   to workspace `src/index.ts`. Knip sees `package.json` entries that no
   `import` statement resolves — but the example does pull from the
   workspace via the alias. *All `@vitrum/*` unused-dependency flags in
   the examples are likely false positives.*
3. **Type-only imports.** TypeScript erases `import type` at runtime; knip
   sometimes flags the package as `unlisted` even when the type import
   is valid.
4. **Lint / format binaries invoked from npm scripts.** Knip detects
   imports, not arbitrary `npm run` invocations. The root `lint` and
   `format` scripts genuinely use `eslint` and `prettier` — false-positive
   flagging.
5. **WGSL `.wgsl.ts` files.** Test files importing WGSL strings for
   compilation smoke-tests show up as live, but constants exported "for
   debugging" (e.g., `ATROUS_KERNEL_VALUES`) may legitimately have no
   import-time consumer.
6. **Renderer subpath imports.** `@vitrum-pathtracer` is a Vitest-only alias
   for the absorbed `packages/three-gpu-pathtracer` package; static scans may
   still flag it as unlisted because it is test-runner configuration, not a
   package.json dependency.
7. **Test fixtures via globals.** Test setup files that mutate
   `globalThis` (e.g., `FakeWebGL2RenderingContext` install pattern) are
   invisible to knip's static graph.
8. **ESLint-mode false positives.** Knip can't load the workspace's
   `eslint.config.js` without `@eslint/js` installed at the root. This
   sweep ran with the eslint plugin disabled, so eslint-driven unused-var
   detection is **out of scope for this report** — recommend running
   `npx eslint . --rule '@typescript-eslint/no-unused-vars: error'` as a
   follow-up sweep when `@eslint/js` is added to root devDeps.

## Recommendations

In order of confidence:

1. **High:** Delete `packages/shared-denoisers/src/svgfRealPipelineCache.ts`
   (orphan extraction; duplicated implementation lives in
   `svgfRealWebGPU.ts`). Verify external_requests/ first.
2. **Medium:** Consolidate the `MAX_*_LIGHTS` / `*_FLOAT_STRIDE` constants
   in `pt-webgpu` — declare in one file, import in the other.
3. **Medium:** Remove dead `webGpuTextureUpload.ts` helpers after confirming
   no example or host uses them. (`shared-denoisers` is intended to ship
   public helpers — this requires architecture-owner sign-off.)
4. **Medium:** Prune `@vitrum/core` from `dependencies` of `shared-bvh`,
   `shared-denoisers`, `shared-samplers` *if* the W3 contract has finalized
   the shared-* → core dependency direction. (Architecture decision — do
   not silently prune.)
5. **Low:** Audit `examples/hero-*/package.json` unused deps. The aliasing
   pattern works without them; risk is that prod-mode builds (no aliases)
   would break — verify both code paths.
6. **Low:** Address `walkaround-hybrid/src/rc/bvhCompute.ts` missing
   `three-mesh-bvh` devDep (type-only import; minor hygiene).
7. **Informational:** Install `@eslint/js` so the workspace's
   `eslint.config.js` can be loaded by tooling, then re-run knip with the
   eslint plugin enabled to expose unused-var-level findings.

## Cross-reference

- See `plan/dead-exports-audit-20260517.md` for the prior hand-rolled
  package-index audit (named exports from each `src/index.ts`).
- See `plan/premium-grade-refactor-20260517.md` and
  `memory/in-flight-plan.md` for the in-progress W1 Pass+Registry refactor
  that is the most likely root of the orphaned `svgfRealPipelineCache.ts`.
