# Vitrum Audit Findings — 2026-05-09

## Summary

- **Total findings: 17 (HIGH: 4, MEDIUM: 9, LOW: 4)**
- **Tests passing:** YES — 50 tests (11 shared-bvh + 39 walkaround-hybrid), all green
- **tsc clean:** YES — zero errors across workspace
- **Public API surface:** Has issues — see H-2, H-3, M-1, M-4
- **Test coverage:** 2 of 8 packages have tests; 6 have none

---

## HIGH Severity Findings

### H-1 `FrameOutput.primaryRadiance` contract violated by HybridEngine skip path

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:274–278`

**Issue:** `HybridEngine.renderFrame()` returns `{ primaryRadiance: null, ... }` on any skip condition (pipeline not ready, paused, no swap view, frame-rate throttle). `FrameOutput.primaryRadiance` is typed as `BackendTexture = unknown` — callers can assign and return `null` without a type error — but `unknown` is not intended to permit `null`; it is the opaque-handle type that the host passes to post-processing. Any host that pattern-matches on `FrameOutput.primaryRadiance` expecting a non-null texture will silently mishandle skip frames. The contract doc says nothing about null being a valid sentinel.

**Why HIGH:** API contract break. Hosts following the documented contract have no way to distinguish a "skip frame" from a "rendered frame with a null texture." This silently corrupts any downstream pipeline that reads `primaryRadiance` on every frame.

**Suggested fix:** Introduce a discriminated `isSkipFrame: boolean` field on `FrameOutput`, or change the return type for skip paths to throw/return a sentinel `FrameOutput` with `samplesAccumulated: 0` and a documented placeholder texture. Add JSDoc to `FrameOutput.primaryRadiance` explicitly stating whether `null` is a valid skip-frame sentinel. Requires no change to `@vitrum/core` types — just document the nullable contract or make it explicit: `primaryRadiance: BackendTexture | null`.

---

### H-2 `_skyEquirectCacheSize` test-utility leaked into production barrel

**File:** `packages/pt-webgl/src/index.ts:134`

**Issue:** `_skyEquirectCacheSize()` is explicitly documented in `iblBaker.ts` as "for tests and debug overlays only," yet it is exported from the production barrel alongside `bakeSkyEquirect` and `clearSkyEquirectCache`. The underscore prefix convention signals internal-only intent, but the export makes it part of the published API surface, discoverable to any consumer via autocomplete.

**Why HIGH:** Public API contract. The function's name and docstring say test-only; the export says otherwise. Any consumer that calls `_skyEquirectCacheSize()` in production code has a fragile dependency on internal state. This is the exact pattern that `@vitrum/core`'s `gpuDetection.test-utils.ts` was created to avoid (see `_resetCacheUnsafe` — intentionally NOT re-exported from `core/index.ts`).

**Suggested fix:** Remove `_skyEquirectCacheSize` from the barrel export in `packages/pt-webgl/src/index.ts`. Create `packages/pt-webgl/src/iblBaker.test-utils.ts` that re-exports it under `__skyEquirectCacheSizeForTests()`, mirroring the pattern in `@vitrum/core`. Any test that needs it imports from the test-utils file.

---

### H-3 `PTEngineWebGL2._markReady()` is a public method on a non-exported class

**File:** `packages/pt-webgl/src/index.ts:81–83`

**Issue:** `PTEngineWebGL2` is a private class (not exported), but `_markReady()` is declared as a plain (public) method. The factory `createPTEngine_WebGL2` calls it directly on the instance (`engine._markReady()`). Because the class is unexported, the method is not literally accessible from outside the module — but within the module it is callable on any `Engine` interface reference after an unsafe cast. More importantly, the `_markReady` call pattern means state management is split between the constructor (which sets `#state = 'initializing'`) and the factory (which sets it to `'ready'`). If a future refactor extracts the class or adds additional factories, it is easy to forget to call `_markReady`, leaving engines perpetually in `'initializing'`.

**Why HIGH:** State invariant fragility. The engine is returned to the host while state is `'initializing'` and becomes `'ready'` in a separate synchronous step, but there is no compile-time enforcement that the factory actually calls `_markReady()`. A missed call is a latent bug that manifests as engines that never become `'ready'`.

**Suggested fix:** Either (a) move `_markReady()` inside `createPTEngine_WebGL2` as a private local function that writes to `#state` via a constructor parameter, or (b) add a constructor overload that accepts `EngineState` as an initial state. The goal: no `public` methods whose sole caller is the factory.

---

### H-4 `HybridEngine.setScene` ignores the `Scene` argument (no-op with no warning)

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:241–245`

**Issue:** `setScene(_scene: Scene)` accepts the `@vitrum/core` `Scene` parameter but documents explicitly "The `_scene` parameter is accepted to satisfy the interface but is not inspected." The engine then traverses `this._threeScene` (the `THREE.Scene` passed at construction) for BVH construction. A host that follows the `Engine` contract — calling `setScene(convertedScene)` after `sceneFromThreeJS()` — gets a pipeline reinit but no BVH update from the passed scene. The contract says `setScene` triggers a "full BVH/light-tree rebuild" from the scene passed. That invariant is violated.

**Why HIGH:** Public API contract break. Hosts using the standard `sceneFromThreeJS → setScene` flow will silently get an engine that ignores scene changes unless the host also separately updates `opts.threeScene`. There is no documentation telling the host to pass `threeScene` instead.

**Suggested fix:** Add a JSDoc note to `HybridEngine.setScene` stating the THREE.Scene coupling explicitly AND add a `console.warn` in the method body when `_scene` contains primitives that differ from what `this._threeScene` would produce. Long-term: Sprint 1 should wire the `Scene` parameter into BVH construction so the contract is properly honoured. As a minimum, the README and JSDoc must be updated to state this limitation clearly.

---

## MEDIUM Severity Findings

### M-1 `materialColors` deprecated sentinel field is still in the public API

**File:** `packages/walkaround-hybrid/src/restir/bvhCompute.ts:121–125`

**Issue:** `SceneBVHBuffers.materialColors` is tagged `@deprecated SENTINEL — always a 4-byte zero-length placeholder.` The field is exported as part of the public `SceneBVHBuffers` type (re-exported from `walkaround-hybrid/src/index.ts`). The deprecation JSDoc says "Slated for removal in a future coordinated rename / surface cleanup." Without a target sprint, this sentinel will accumulate external consumers.

**Why MEDIUM:** API surface debt. Every consumer of `SceneBVHBuffers` must reason about a field that must never be uploaded to the GPU. The sentinel is confusing and the deprecation has no target date.

**Suggested fix:** Remove `materialColors` from `SceneBVHBuffers` in the next sprint that touches `bvhCompute.ts`. It has no GPU callers (verified: `WalkaroundGPUPipeline.ts:208` explicitly skips it). Internal callers in `_initPipeline` must also be updated.

---

### M-2 `warnedTypes` Set in `three-bindings` is module-level global state

**File:** `packages/three-bindings/src/index.ts:177`

**Issue:** `const warnedTypes = new Set<string>()` is module-level, so any light type that fires `warnOnce(typeName, ...)` is permanently suppressed for the lifetime of the module. In tests and in long-running apps where scenes change, the warning will not re-fire if the same unsupported light type appears in a new scene after a hot reload or scene swap. This is especially problematic for tests that call `sceneFromThreeJS` multiple times.

**Why MEDIUM:** Test isolation and dev-experience issue. It also means a developer adding an `AmbientLight` will see the warning once and then never again across scene reloads, making it easy to miss a misconfigured scene.

**Suggested fix:** Either pass `warnedTypes` as a parameter to `sceneFromThreeJS` (caller supplies, default to a fresh `Set` per call), or clear the set on each `sceneFromThreeJS` call. The simplest fix: remove the dedup — `console.warn` deduplication is the browser's job.

---

### M-3 `HybridEngine.resume()` silently ignores calls in `'uninitialized'` state

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:466–469`

**Issue:** `resume()` only transitions from `'paused'` → `'ready'`. If called when `_state` is `'uninitialized'` (e.g., before `setScene` has been called for the first time), it silently no-ops. `pause()` has the same narrow guard. By contrast, `PTEngineWebGL2.pause()` and `PTEngineWebGL2.resume()` call `#assertLive` which throws for disposed but otherwise accepts any live state. The two implementations have different guard semantics.

**Why MEDIUM:** API inconsistency. A host that calls `resume()` too early gets no error and no state change. `HybridEngine` should throw or document that `resume()` is a no-op outside `'paused'` state.

**Suggested fix:** Either add `console.warn('[HybridEngine] resume() called in state:', this._state)` for non-paused calls, or align with `PTEngineWebGL2`'s `#assertLive` pattern. Document the guard semantics explicitly in JSDoc.

---

### M-4 Factory naming convention inconsistency: underscore vs camelCase

**File:** `packages/pt-webgl/src/index.ts:149` vs `packages/walkaround-hybrid/src/HybridEngine.ts:637`

**Issue:** The two factories are named `createPTEngine_WebGL2` (underscore before platform suffix) and `createWalkaroundEngine_Hybrid` (underscore before variant suffix). `EngineFactory` type in `@vitrum/core/engine.ts` does not enforce naming, but having an underscore in a public export name is unusual in the TypeScript ecosystem and inconsistent across the two packages. Neither factory conforms to the `EngineFactory<T>` shape (they take narrowed options types, which is correct, but are not declared as `EngineFactory<T>` explicitly).

**Why MEDIUM:** API consistency/discovery issue. Hosts scanning both packages will see inconsistent naming patterns. Factory types should be explicitly declared as `const createPTEngine_WebGL2: EngineFactory<PTEngineWebGL2Options>` so type inference aligns with the contract type.

**Suggested fix:** Standardise naming: `createWebGL2Engine` and `createHybridEngine`, or at least make the underscore-before-suffix pattern consistent. Also add explicit `EngineFactory<T>` type annotations to each factory so callers benefit from the type contract.

---

### M-5 `three-bindings` multi-material mesh silently discards all materials after index 0

**File:** `packages/three-bindings/src/index.ts:146`

**Issue:** `Array.isArray(obj.material) ? obj.material[0] : obj.material` silently uses only the first material for any multi-material mesh. No warning is emitted. A Three.js mesh with 3 material groups will be silently converted to a single-material mesh.

**Why MEDIUM:** Silent data loss. Users with multi-material meshes (e.g., combined geometry with different surface types) will get wrong renders without any indication of why.

**Suggested fix:** Add a `console.warn` when `Array.isArray(obj.material) && obj.material.length > 1` noting that multi-material meshes are not yet supported and only the first material will be used.

---

### M-6 Test gap: `@vitrum/three-bindings` has zero tests

**File:** `packages/three-bindings/` — no `__tests__/` directory, no test script in `package.json`

**Issue:** `sceneFromThreeJS` is the primary public API. It has non-trivial logic: mesh traversal, material type guards, emitter direction computation for directional/rect-area/spot lights, environment resolution. None of it is tested. The RectAreaLight `uAxis`/`vAxis` derivation from `matrixWorld.elements` column vectors is particularly fragile.

**Why MEDIUM:** Critical path untested. The light conversion math (especially RectAreaLight column extraction from `me[0..2]` for uAxis and `me[4..6]` for vAxis) is not intuitive and has no regression cover.

**Suggested fix:** Add at minimum: (1) a test for each light type that verifies the emitter fields are correctly populated, (2) a test for the RectAreaLight matrix decomposition, (3) a test for the unsupported-type error paths, (4) a test for the environment resolution.

---

### M-7 Test gap: `@vitrum/pt-webgl`, `@vitrum/core`, `@vitrum/shared-samplers`, `@vitrum/shared-denoisers` have zero tests

**Files:** `packages/pt-webgl/`, `packages/core/`, `packages/shared-samplers/`, `packages/shared-denoisers/`

**Issue:** `@vitrum/core` is the load-bearing contract; none of its types have runtime-behavior tests. `@vitrum/pt-webgl` has `debounceMsForEditRate`, `getSunIntensity`, `skyParamsFor`, `worldSunPosition` — all pure functions that are straightforward to test.

**Why MEDIUM:** Coverage gap on pure functions that are critical to scene correctness (sun intensity, sky params, debounce logic).

**Suggested fix:** For `pt-webgl`, add unit tests for `debounceMsForEditRate` (burst detection threshold), `getSunIntensity` (bucket boundaries), `skyParamsFor` (sunY cap, turbidity range), `worldSunPosition` (distance scaling). For `@vitrum/core`, add a structural test verifying `EngineState` exhaustiveness.

---

### M-8 `HybridEngine` `_initPipeline` error is silently swallowed after logging

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:614–616`

**Issue:**
```typescript
} catch (err) {
  console.error('[HybridEngine] init failed:', err);
}
```
When `buildSceneBVH` or `pipeline.initialize` throws, the engine stays in `'initializing'` state forever. The host has no way to know initialization failed — `engine.state` never becomes `'ready'`, `renderFrame()` returns skip frames, and there is no error event or state transition to `'disposed'` or a hypothetical `'error'` state.

**Why MEDIUM:** Silent failure. Hosts polling `engine.state` for a `'ready'` transition will poll indefinitely. The 5s scene-readiness timeout will fire and the engine will attempt BVH build, but if that succeeds while `pipeline.initialize` subsequently fails, the engine is stuck.

**Suggested fix:** On catch in `buildBVHWhenReady`, transition `this._state` to `'disposed'` (or add `'error'` state to `EngineState`) so the host can detect and handle the failure. At minimum, emit a `console.error` that instructs the host to check `engine.state`.

---

### M-9 `@vitrum/shared-bvh` exports `@vitrum/core` dep transitively but `core` is not listed in `peerDependencies`

**File:** `packages/shared-bvh/package.json`

**Issue:** `@vitrum/shared-bvh` lists `@vitrum/core` as a `dependencies` entry (hard dep, monorepo file: path). This is fine for internal use, but `SceneBVHCommonResult.materials` exposes `THREE.Material[]` which means the package already structurally couples to `three`. The `@vitrum/core` hard dep is not wrong, but `three` is listed as both a `dependency` and a `peerDependency` — that's redundant; peer deps alone are sufficient for library packages that don't bundle.

**Why MEDIUM:** Package hygiene. Library packages should list runtime requirements as `peerDependencies` only, letting the host app control the version. Listing `three` in both `dependencies` and `peerDependencies` can cause version duplication in host apps.

**Suggested fix:** Remove `three` from `dependencies` in `shared-bvh/package.json`; keep it only in `peerDependencies`. Same audit applies to `pt-webgl` and `walkaround-hybrid`.

---

## LOW Severity Findings

### L-1 `HybridEngine.capabilities` is a field, not a getter — breaks the Engine interface intent

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:141`

**Issue:** `HybridEngine` declares `readonly capabilities: EngineCapabilities` as a plain field set in the constructor. `PTEngineWebGL2` implements `capabilities` as a getter that returns a fresh object each call. The `Engine` interface declares `readonly capabilities: EngineCapabilities` — both patterns satisfy it. But field vs getter is different: `HybridEngine.capabilities` is a single frozen object; `PTEngineWebGL2` returns a new object per access. Neither is wrong, but they're inconsistent.

**Why LOW:** Style/consistency. Not a correctness issue. A host that caches `engine.capabilities` will get the right answer either way.

**Suggested fix:** Document the chosen pattern per backend. If the intent is "capabilities are immutable post-construction," a field is fine — just add a JSDoc note so future maintainers don't convert it to a dynamic getter accidentally.

---

### L-2 TODO comment in `HybridEngine.ts` references "Step 4" with no sprint context

**File:** `packages/walkaround-hybrid/src/HybridEngine.ts:20`

**Issue:** `TODO Step 4: re-wire RC composition here when the RC cascade is extracted.` has no sprint reference, no issue link, no date. The extraction plan is in `plan/` but the TODO doesn't cite it.

**Why LOW:** Stale-context risk. "Step 4" is meaningful only to whoever wrote it.

**Suggested fix:** Replace with `// TODO (Phase 6 Sprint N): re-wire RC composition — see plan/phase-6-roadmap.md §RC` so future agents can locate the context.

---

### L-3 `pt-webgl` package description says "Stub — not yet implemented" but the package has substantial utility code

**File:** `packages/pt-webgl/package.json:4`

**Issue:** The description field reads "Stub — not yet implemented." But the package exports `bakeSkyEquirect`, `computeLightingState`, `skyParamsFor`, `debounceMsForEditRate`, and several intensity table utilities — all working code. Only the `Engine` class itself is a stub (all methods throw "Not implemented"). The description is misleading.

**Why LOW:** Documentation accuracy.

**Suggested fix:** Change description to "WebGL2 path-tracer backend for vitrum. Engine stub with working utility modules (IBL baker, sky params, lighting tables). Sprint 1 wires the engine class."

---

### L-4 `pt-webgl` factory name uses mixed convention (`createPTEngine_WebGL2`) — the underscore is non-idiomatic

**File:** `packages/pt-webgl/src/index.ts:149`

**Issue:** Already flagged as M-4 for consistency. Separately flagging the naming style: `createPTEngine_WebGL2` uses both a camelCase prefix and an underscore+variant suffix, which is non-idiomatic TypeScript/JavaScript. Compounded by `createWalkaroundEngine_Hybrid` doing the same.

**Why LOW:** Code style. No runtime impact.

**Suggested fix:** Track under M-4.

---

## Coverage Gap Matrix

| Package | Has tests? | Critical paths tested? | Recommended additions |
|---|---|---|---|
| `@vitrum/core` | No | No | EngineState exhaustiveness; `detectGpu` + `probeWebGPU` mock tests; `isSwiftShaderAdapter` |
| `@vitrum/shared-bvh` | **Yes** (11 tests) | **Yes** — single-root invariant, matId reorder safety, stride-3/4, proxy substitution, empty-scene | Filter edge cases; `SceneBvh.update` dirty-check path |
| `@vitrum/shared-samplers` | No | No | Hammersley sequence value spot-checks |
| `@vitrum/shared-denoisers` | No | No | WGSL string contains expected entry points (same pattern as rc-bindings tests) |
| `@vitrum/three-bindings` | No | No | All light types; multi-material warning; RectAreaLight matrix decomp; SkinnedMesh/InstancedMesh errors |
| `@vitrum/pt-webgl` | No | No | `debounceMsForEditRate`, `getSunIntensity` bucket boundaries, `skyParamsFor`, factory validation guards |
| `@vitrum/pt-webgpu` | No | Stub only | N/A until Sprint 1 |
| `@vitrum/walkaround-hybrid` | **Yes** (39 tests) | **Structural only** — binding shapes, WGSL entry points, class API surface. No GPU behavioral tests. | `HybridEngine` state machine; `sceneFromThreeJS → setScene` round-trip; error path when init fails |

---

## Type-Safety Hot Spots

All `any` / `as unknown as` / non-null assertions in production source files. Each has a justification check:

| Location | Pattern | Verdict |
|---|---|---|
| `core/wgpuSupport.ts:56` | `adapter: any` in `readAdapterInfo` | **Benign** — WebGPU `GPUAdapter` types are inconsistent across `@webgpu/types` versions; the `any` is necessary and the function is defensive. Comment explains. |
| `core/wgpuSupport.ts:125` | `gpu as any` before `.requestAdapter()` | **Benign** — same WebGPU typing issue. Comment present. |
| `rc/cascadeDispatch.ts:169` | `as unknown as Record<string, unknown>` to access `__gpuBuffer` | **Benign** — Three.js internal property, not in .d.ts. Comment explains pattern and risk of upstream rename. |
| `rc/cascadeDispatch.ts:211,357` | `gl as any` twice | **Benign** — accessing `backend.device` on Three.js `WebGPURenderer`, which is not in the stable .d.ts. Comment explains. |
| `pipeline/timestampQueries.ts:75` | `import.meta as unknown as { env?: { DEV?: boolean } }` | **Benign** — Vite-specific env pattern; typed guard needed. |
| `pipeline/timestampQueries.ts:96` | `device as unknown as { adapterInfo?... }` | **Benign** — `adapterInfo` not in `@webgpu/types`; comment explains. |
| `HybridEngine.ts:315` | `pipeline as unknown as { lastGpuTimings?... }` | **Benign** — accessing debug field on pipeline; guarded by `this._debug`. Could instead add `lastGpuTimings` to `WalkaroundGPUPipeline`'s public type (it already declares it as `public`). Minor cleanup opportunity. |
| `HybridEngine.ts:430` | `window as unknown as { __WGPU__?... }` | **Benign** — debug bridge, guarded by `this._debug && typeof window !== 'undefined'`. |
| `HybridEngine.ts:599` | `this._ddgi.pass as unknown as { setSunIntensityMultiplier... }` | **RISK** — `setSunIntensityMultiplier` exists on `ProbeUpdatePass` but is accessed via an `as unknown` cast on `this._ddgi.pass` because `pass` is typed as `ProbeUpdatePass`. This is unnecessary: `ProbeUpdatePass` exports `setSunIntensityMultiplier` as a public method. The cast can be removed entirely — call `this._ddgi.pass.setSunIntensityMultiplier(...)` directly. |
| `rc/applyDDGIShading.ts:27,69,70,125,127` | `AnyNode = any`, multiple `as unknown as THREE.Texture` | **Benign** — three/tsl typings do not model `wgslFn` return types. Documented. The `as unknown as THREE.Texture` cast on `StorageTexture` is fragile but needed for TSL compatibility. |
| `rc/walkaroundDiffuseLighting.ts:47,95` | `AnyNode = any` | **Benign** — same TSL typing limitation. |
| `rc/giReceiver.ts:36,115,167` | `AnyNode = any`, symbol-tagged material cast | **Benign** — TSL typing + brand tag pattern. |
| `lib/nodeMaterialUpgrade.ts:47,63,65` | `as unknown as`, `src/dst as any` | **Benign** — Three.js NodeMaterial property copy; `any` necessary because TSL material types are structurally identical but not intersection-compatible. |
| `shared-bvh/bvhCommon.ts:390` | `sgg as unknown as { applyWorldTransforms? }` | **Benign** — accessing a Three.js internal property. Comment notes the risk of upstream rename. |
| `probeUpdatePass.ts:329–333, 377, 433, 438, 451, 459` | `this._gpu!.*` (non-null assertion, 10+ uses) | **All benign** — `this._gpu` is only `null` before `init()` or after `dispose()`. All call sites are within methods that check `if (!this._gpu) return` or are only called from `runFrame` which gates on `this._gpu` being set. However the pattern is defensive-by-convention only. |

**Actionable item from hot-spot review:** `HybridEngine.ts:599` — the `as unknown as` cast is unnecessary and can be removed. `this._ddgi.pass.setSunIntensityMultiplier(this._primaryLightIntensity)` compiles cleanly since `setSunIntensityMultiplier` is public on `ProbeUpdatePass`.

---

## Three.js Coupling Map

| Package | `three` usage | Coupling type | Notes |
|---|---|---|---|
| `@vitrum/core` | **None** | — | Clean. `BackendTexture = unknown` intentionally opaque. |
| `@vitrum/shared-bvh` | `THREE.*`, `MeshBVH`, `StaticGeometryGenerator` | **Required** — BVH construction walks `THREE.Scene`; cannot decouple without replacing the input contract | `SceneBVHCommonResult.materials: THREE.Material[]` leaks into the public type. Acceptable for now. |
| `@vitrum/shared-samplers` | **None** | — | Clean. Pure WGSL string module. |
| `@vitrum/shared-denoisers` | **None** | — | Clean. Pure WGSL string module. |
| `@vitrum/three-bindings` | `THREE.*` | **Core purpose** — package is the Three.js → `@vitrum/core` adapter | All coupling is intentional. |
| `@vitrum/pt-webgl` | `THREE.*` + `three/examples/jsm/objects/Sky` | **Required** for IBL baker, sky params, lighting state (all `THREE.Vector3`, `THREE.Color`, `THREE.WebGLRenderer`) | `LightingState.sunDirection: THREE.Vector3` leaks into the public type. Consider returning `[number, number, number]` instead. |
| `@vitrum/pt-webgpu` | None | Stub | — |
| `@vitrum/walkaround-hybrid` | `THREE.*`, `three/webgpu`, `three/tsl` | Complex coupling: `HybridEngine` requires `THREE.Scene` at construction (passed to DDGI + BVH build). RC subsystem requires `three/tsl`. DDGI atlas requires `three/webgpu`. ReSTIR-only path does NOT require TSL. | The README documents the split correctly. The issue is that `HybridEngine` constructor accepts `threeScene: THREE.Scene` directly — this is the deepest coupling point. |

---

## Strict-TS Workaround Review

The following patterns were added by extraction agents to satisfy tsc — each is reviewed for "benign vs hides a bug":

| Workaround | Location | Verdict |
|---|---|---|
| `?? 0n` BigInt defaults | Not found in any file | N/A — this pattern does not appear in the codebase. |
| `?? 1.0` on `totalEmissivePower` | `HybridEngine.ts:416` | **Benign** — `SceneBVHBuffers.totalEmissivePower` is typed `number` (not optional), so `?? 1.0` is defensive but harmless. |
| `?? 0` on `bvh.emitters?.count` | `HybridEngine.ts:417` | **Mild concern** — `bvh.emitters` is `StorageBufferHandle` (not optional per `SceneBVHBuffers`), so the `?.` optional chain is unnecessary. The RHS `?? 0` defaults to 0 emitters if the field were somehow absent — the pipeline would then sample from a dummy emitter buffer, which is handled in `bvhCompute.ts:666` ("add a dummy one so the buffer is non-empty"). Net: benign but the `?.` is a false safety signal. |
| `?? 'bgra8unorm'` on swap chain format | `HybridEngine.ts:302` | **Benign** — correct fallback for browsers that don't expose `getPreferredCanvasFormat`. |
| `img.width ?? 80` / `img.height ?? 480` | `probeUpdatePass.ts:476` | **Risk** — if `StorageTexture.image` has not yet been set by Three.js's backend, the texture will be created at 80×480 regardless of the actual atlas size. If the atlas is later allocated at a different size, the `GPUTexture` in the cache will be stale. The `WeakMap` cache keyed on the `StorageTexture` object means once the wrong size is baked in, it persists until the `StorageTexture` object is GC'd. This is an edge-case race, not a guaranteed bug, but the fallback values are magic numbers that need a comment explaining their provenance (IRR atlas default = `IRR_STRIDE * probeCount_max` etc.). |

**Actionable:** The `img.width ?? 80` fallback in `probeUpdatePass.ts:476` deserves a comment linking it to the atlas layout constants in `ddgiAtlasLayout.ts`, and ideally a `console.warn` if the fallback fires (indicating the atlas was not yet initialized when the texture was first accessed).
