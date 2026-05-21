# Fix-Everything Tracker (1:1)

Source of truth: user-provided issue list from the approved remediation plan request.

Status legend: `[ ]` pending, `[x]` fixed.

## Engine / Core / Dev
- [ ] `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` imports/uses symbols that do not exist in current code.
- [ ] `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` has implicit-any callback parameter in `meshVertexRanges?.find((r) => ...)`.
- [ ] `packages/engine/src/lifecycle/vanilla.ts` viewport dimensions use CSS pixels instead of DPR-applied physical pixels.
- [ ] `packages/engine/src/lifecycle/vanilla.ts` captures `opts.quality` once and ignores later changes.
- [ ] `packages/engine/src/react/VitrumCanvas.tsx` re-initializes on camera identity changes.
- [ ] `packages/engine/src/react/VitrumCanvas.tsx` props omit `debug`/`advanced` options.
- [ ] `packages/engine/src/createEngine.ts` adapter request does not use `{ powerPreference: 'high-performance' }`.
- [ ] `packages/engine/src/createEngine.ts` realtime route does not gate SwiftShader via `adapterKind`.
- [ ] `packages/engine/src/createEngine.ts` proxy wrapper omits `updateEnvironment`.
- [ ] `packages/engine/src/createEngine.ts` post-dispose `renderFrame` reports `isConverged: true`.
- [ ] `packages/engine/src/lifecycle/vanilla.ts` unconditional top-level `import * as THREE from 'three'`.
- [ ] `packages/engine/src/sceneAABB.ts` can return `±Infinity` for empty instanced mesh.
- [ ] `packages/engine/src/sceneAABB.ts` excludes analytic primitives from bounds in common cases.
- [ ] `packages/engine/src/sceneAABB.ts` excludes emitters from bounds.
- [ ] `packages/core/src/gpuDetection.ts` cached first probe ignores later options.
- [ ] `packages/core/src/gpuDetection.ts` references missing `_resetCacheUnsafe`.
- [ ] `packages/dev/src/react/DDGIAtlasViewer.tsx` warning fires every rerender.
- [ ] `packages/dev/src/react/FrameTimeHUD.tsx` stale ring buffer closure after `averageWindow` change.
- [ ] `packages/dev/src/react/DenoiserABToggle.tsx` UI state can diverge when setter unavailable.
- [ ] `packages/dev/src/react/MaterialInspector.tsx` shallow clone aliases nested material handles.
- [ ] `packages/dev/src/react/MaterialInspector.tsx` `hexToVec3` lacks validation and can emit `NaN`.
- [ ] `packages/dev/src/vanilla.ts` default overlay positions collide.
- [ ] `packages/dev/src/vanilla.ts` accepts `scene` option but discards it.
- [ ] `packages/dev/src/react/BVHVisualizer.tsx` mutates ref during render.

## Engine / Dev Test Gaps
- [ ] `packages/engine/__tests__/telemetryProxy.test.ts` does not verify real `createEngine` proxy forwarding.
- [ ] `packages/engine/__tests__/debugSurface.test.ts` does not verify `engine.debug` survives proxying.
- [ ] `packages/engine/__tests__/sceneAABB.test.ts` misses empty-instanced / infinity / analytic-only / emitter-only.
- [ ] `packages/engine/__tests__/createEngine.test.ts` misses SwiftShader routing, adapter mismatch, proxy behavior.
- [ ] `packages/dev/__tests__/imports.test.ts` import-only; no behavior/render assertions.
- [ ] `packages/dev/__tests__/vanilla.test.ts` no overlay collision or keyboard toggle coverage.
- [ ] `packages/dev/__tests__/ringBuffer.test.ts` no zero/negative capacity or non-finite input coverage.

## Package Metadata
- [ ] `packages/core/package.json` includes `README.md` in `files` but file is missing.
- [ ] `packages/dev/package.json` includes `README.md` in `files` but file is missing.
- [ ] `packages/engine/package.json` includes `README.md` in `files` but file is missing.

## PT-WebGPU
- [ ] `packages/pt-webgpu/src/wgsl/common.wgsl.ts` references `params.triIntersectEpsilon` without params definition.
- [ ] `packages/pt-webgpu/src/index.ts` identity fallback on `invertMat4` failure silently produces wrong rays.
- [ ] `packages/pt-webgpu/src/index.ts` directional RGB collapsed into scalar average `lightDir.w`.
- [ ] `packages/pt-webgpu/src/scene/emitterPacking.ts` `discAreaPackedAsRect` can divide by zero-length normal.
- [ ] `packages/pt-webgpu/src/scene/emitterPacking.ts` instanced mesh-area packing uses only first instance.
- [ ] `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` dead helper `sampleMeshAreaLight`.
- [ ] `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` glossy sampling/PDF mismatch.
- [ ] `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` transmission PDF mismatch.
- [ ] `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` inconsistent direct-light MIS across classes.
- [ ] `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` normal world/local transform invalid under non-uniform scale.
- [ ] `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` motion vectors written as `rgba16float` vs documented RG32F.
- [ ] `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` variance moments format mismatches Welford `{mean, M2}`.
- [ ] `packages/pt-webgpu/src/index.ts` bind-group storage buffers exceed SwiftShader limit.

## Shared BVH / Samplers
- [ ] `packages/shared-bvh/src/sceneBvh.ts` zero-mesh update does not clear cached buffers.
- [ ] `packages/shared-bvh/src/bvhCommon.ts` triangle material id from first index vertex only.
- [ ] `packages/shared-bvh/src/bvhCommon.ts` multi-material snapshot uses only `material[0]`.
- [ ] `packages/shared-samplers/src/jakobHanika.ts` analytic placeholder instead of coefficient-table implementation.
- [ ] `packages/shared-samplers/src/bdptMIS.ts` partial strategy PDF does not enumerate full Veach strategies.
- [ ] `packages/shared-samplers/src/wavelengthSampling.ts` `wavelengthToRGB` divides by `Y_INTEGRAL` instead of mixture pdf.
- [ ] `packages/shared-samplers/src/equiAngular.ts` PDF evaluates clamped `t` mismatching sample geometry.

## Shared Denoisers
- [ ] `packages/shared-denoisers/src/svgfRealWebGPU.ts` creates `r16uint` history output while WGSL expects `r32uint`.
- [ ] `packages/shared-denoisers/src/wgsl/atrousVariance.wgsl.ts` binds prev-radiance/motion textures never sampled.
- [ ] `packages/shared-denoisers/src/wgsl/atrousVariance.wgsl.ts` and `atrous.wgsl.ts` decode normals as `xyz*2-1` against world-normal contract.
- [ ] `packages/shared-denoisers/src/wgsl/atrousVariance.wgsl.ts` inconsistent depth channel usage (`.x` vs `.w` expectation).
- [ ] `packages/shared-denoisers/src/atrousVarianceWebGPU.ts` fallback normal upload incompatible with shader decode.
- [ ] `packages/shared-denoisers/src/svgfRealWebGPU.ts` one-shot fallback uploads identical curr/prev depth/normal/object IDs.

## Walkaround-Hybrid Integration
- [ ] `packages/walkaround-hybrid/src/HybridEngineLifecycle.ts` orphaned.
- [ ] `packages/walkaround-hybrid/src/HybridEngineOptions.ts` orphaned and type-drifted.
- [ ] `packages/walkaround-hybrid/src/HybridEngineTuning.ts` orphaned from live constructor/UBO wiring.
- [ ] `packages/walkaround-hybrid/src/HybridEngine.ts` accepts `denoiser: 'neural'` while registry disabled.
- [ ] `packages/walkaround-hybrid/src/HybridEngine.ts` accepts `ppgEnabled` but does not forward to pipeline init.
- [ ] `packages/walkaround-hybrid/src/pipeline/denoisers/atrousVariance.ts` denoiser uses wrong depth channel from normal-depth texture.
- [ ] `packages/walkaround-hybrid/src/pipeline/denoisers/svgfReal.ts` depth from `.r` while shade writes depth in `.w`.
- [ ] `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts` hardcodes sun direction.
- [ ] `packages/walkaround-hybrid/src/shaders/risGi.wgsl.ts` hardcoded irradiance clamp.
- [ ] `packages/walkaround-hybrid/src/shaders/common.wgsl.ts` compile-time ReSTIR GI cap not host tunable.
- [ ] `packages/walkaround-hybrid/src/shaders/spatialGi.wgsl.ts` hardcoded spatial tolerances/radius not host tunable.
- [ ] `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts` hardcoded indirect clamp not host tunable.
- [ ] `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts` hardcoded glass transmission scale.
- [ ] `packages/walkaround-hybrid/src/pipeline/passes/PPGGuidePass.ts` dispatch `(0,0,0)` no-op.
- [ ] `packages/walkaround-hybrid/src/pipeline/passes/PPGUpdatePass.ts` dispatch `(0,0,0)` no-op.
- [ ] `packages/walkaround-hybrid/src/pipeline/denoisers/neural.ts` registered disabled.
- [ ] `packages/walkaround-hybrid/src/pipeline/denoisers/oidnFinal.ts` registered disabled.
- [ ] `packages/walkaround-hybrid/src/HybridEngine.ts` runtime light direction updates do not propagate to DDGI probe-light upload.
- [ ] `packages/walkaround-hybrid/src/shaders/temporal.wgsl.ts` lacks per-pixel disocclusion reset gate.
- [ ] `packages/walkaround-hybrid/tsconfig.json` excludes `__tests__`.
- [ ] `packages/walkaround-hybrid/__tests__` missing coverage for orphaned extracted modules.

## Three-Bindings
- [ ] `packages/three-bindings/src/vitrumSceneToThree.ts` drops sheen/clearcoat/iridescence fields.
- [ ] `packages/three-bindings/src/vitrumSceneToThree.ts` procedural-sky path degrades to flat background.
- [ ] `packages/three-bindings/src/vitrumSceneToThree.ts` non-texture HDRI fallback does not fully reset previous environment state.
- [ ] `packages/three-bindings/src/vitrumSceneToThree.ts` disc-area emitters approximated as rect-area and do not round-trip.
- [ ] `packages/three-bindings/src/gltfLoader.ts` camera extraction returns first camera object, not scene-referenced active camera.
- [ ] `packages/three-bindings/__tests__` lacks emitter conversion and advanced material round-trip coverage.
- [ ] `packages/three-bindings/package.json` includes `README.md` in `files` but file is missing.

## Benchmark / Tooling / CI
- [ ] `tools/benchmark-runner/run-gap-closure-verification.mjs` marks pass on capture success without baseline image comparison.
- [ ] `tools/benchmark-runner/run-gap-closure-verification.mjs` variant expansion only handles caustic variants.
- [ ] `tools/benchmark-runner/run-gap-closure-verification.mjs` env forwarding misses documented knobs.
- [ ] `tools/benchmark-runner/capture-adapter-playwright.mjs` drops `candidate/baseline` caustic labels.
- [ ] `tools/benchmark-runner/capture-adapter-playwright.mjs` timeout flow still screenshots after readiness failure.
- [ ] `tools/reference-renders/baseline` lacks baseline PNGs for strict gap-closure.
- [ ] `tools/benchmark-runner/scenario-presets.mjs` walkaround scenarios mismatch default cornell-box capture host.
- [ ] `examples/cornell-box/vite.config.ts` capture middleware allows unbounded body writes and no auth/size limits.
- [ ] `examples/cornell-box/vite.config.ts` default port conflicts with other tooling defaults.
- [ ] `scripts/capture-cornell-suite.sh` `--bdpt` param not consumed by `examples/cornell-box/src/main.ts`.
- [ ] `examples/cornell-box/src/main.ts` capture-ready fallback can report ready after denoise failure on wrong output surface.
- [ ] `examples/hero-viewer/src/main.ts` initial camera aspect uses window size, not canvas size.
- [ ] `examples/hero-product-viz/src/main.ts` resize updates projection without explicit engine viewport update.
- [ ] `examples/hero-lighting-designer/src/main.ts` resize updates projection without explicit engine viewport update.
- [ ] `examples/*/package.json` `three@^0.184.x` conflicts with root override `three@0.171.0`.
- [ ] `examples/cornell-box/package.json` and `examples/two-engines-one-scene/package.json` missing `typecheck` scripts.
- [ ] Root `package.json` lint command is effectively a no-op because workspaces lack lint scripts.
- [ ] Root `tsconfig.json` include scope omits examples/tools/staging.
- [ ] Root `tsconfig.json` paths include only subset of internal packages.
- [ ] `.github/workflows/ci.yml` checks out fork at `main`, risking branch mismatch vs intended fork state.
- [ ] `.github/workflows/ci.yml` missing gap-closure benchmark stage.

## `_staging` Full Rehabilitation
- [ ] `_staging/legacy-source` snapshot has unresolved imports and does not compile standalone.
- [ ] `_staging/legacy-source/src/rendering/scene/ptEnvironment.ts` imports missing local `lightingIntensityTable`.
- [ ] `_staging/legacy-source/src/rendering/scene/lighting/renderers/sunPathTraced.tsx` hardcodes sun constants instead of shared package constants.
- [ ] `_staging/legacy-source/src/rendering/scene/walkaround/HybridLayeredStage.tsx` and `.../RestirStage.tsx` use inconsistent time-of-day selector path.
- [ ] `_staging/legacy-source/src/rendering/scene/walkaround/engines/restir/RestirStage.tsx` updates `prevProjMatrix` with current projection in-frame.
- [ ] `_staging/legacy-source/src/rendering/scene/walkaround/lib/useSceneBVH.ts` keeps `opts` in effect deps causing debounce rebuild churn.
- [ ] `_staging/legacy-source/src/rendering/scene/walkaround/useSceneBVH.ts` duplicates similarly named hook with divergent behavior/types.
- [ ] `_staging/legacy-source/src/rendering/scene/walkaround/engines/restir/RestirStage.tsx` cleanup nulls global `window.__WGPU__`.
- [ ] `_staging` has no CI/test coverage and is not typechecked.
