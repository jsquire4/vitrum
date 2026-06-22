# Gap Closure Execution Plan

Generated 2026-06-17 as the practical replacement for the oversized Cursor
orchestrator bundle. That bundle found real issues, but its 501-task / 71-wave /
25-agent plan was too broad, stale in places, and unsafe to run as written. The
old generated bundle was removed from `plan/` during the cleanup pass so agents
do not restart that stale queue.

This plan is intentionally smaller. It is meant to close the remaining code
gaps quickly, keep long-tail proof work visible, and stop the team from spending
days on open-ended polishing without shrinking the blocker list.

## Goal

Reach **contract-complete arbitrary glTF + backend truthfulness**:

1. Known runtime correctness gaps are fixed or explicitly rejected by capability
   checks / structured diagnostics.
2. `loadGltfAsset` / `loadGltfForEngine` can predictably ingest, diagnose,
   route, and render arbitrary glTF assets on an appropriate backend.
3. Walkaround, pt-webgpu, and pt-webgl2 do not silently claim native support for
   approximate or unsupported material/transport rows.
4. Remaining items after the code waves are only validation, production learned
   assets, or explicitly parked SOTA/performance work.

This plan does **not** chase every possible SOTA improvement. If a task does not
help the contract-complete goal, park it.

## Stop Rules

Stop implementation work and switch to validation/signoff when all are true:

- The verified active queue below is empty.
- `plan/road-to-100.md` has no remaining implementation-distance item except
  validation/provisioning/permanent-unsupported rows.
- `npm run typecheck`, relevant package tests, `npm run shader-gate`, and the
  focused behavioral glTF lane pass on the available WSL adapter.
- Any item that still sounds like "implementation" has either a filed task below
  or an explicit reason it is not part of contract-complete.

## Operating Rules

- **Lead agent validates and commits.** Workers may edit, but the lead verifies.
- **Use at most 3 concurrent workers.** More than 3 creates conflict churn and
  burns context faster than it closes gaps.
- **No task begins without source revalidation.** Open the cited file and confirm
  the gap still exists on current `main`.
- **No multi-file shared-core parallelism.** Anything touching
  `packages/core/src/engine/promiseLedger.ts`,
  `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`,
  `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`,
  `packages/pt-webgpu/src/inverse/inverseSession.ts`, or
  `packages/walkaround-hybrid/src/HybridEngine.ts` runs alone.
- **One wave, one coherent commit.** Do not pile unrelated long-tail work into a
  green wave.
- **No exploratory sprawl.** If a worker discovers a new issue, record it in the
  candidate queue and finish the assigned task first.
- **No validation-as-implementation theater.** Render captures/A-B proof are
  important, but they do not replace fixing a confirmed code bug.

## Triage Labels

- **CLOSE-NOW:** confirmed runtime behavior gap, bounded implementation.
- **TRUTH:** capability/diagnostic/ledger mismatch causing false claims.
- **PROOF:** renderer/fidelity evidence required to promote a row.
- **DECIDE:** product/default-tier decision; do not implement blindly.
- **PARK:** post-contract SOTA/performance work or intentionally unsupported row.

## Verified Fast-Start Queue

These came from the Cursor plan and were rechecked against current source on
2026-06-17. Revalidate again immediately before editing, but they are credible
first targets.

**Wave 0 status (2026-06-17): CLOSED.** VQ-001 through VQ-004 are implemented
and package-gated: `npm run typecheck`,
`npm test --workspace @vitrum/pt-webgpu`,
`npm test --workspace @vitrum/pt-webgl2`, and
`npm test --workspace @vitrum/walkaround-hybrid` are green in WSL Node 20.

### VQ-001 — pt-webgpu lite material patch should not full-rebuild

- Label: **CLOSE-NOW**
- Status: **CLOSED 2026-06-17** — lite material-only patches now use the
  existing material-buffer fast path and advertise `material: native`.
- Files:
  - `packages/pt-webgpu/src/sceneMutationRouter.ts`
  - `packages/pt-webgpu/src/__tests__/updatePrimitiveIncremental.test.ts`
- Verified closure:
  - Lite scalar material patches write the resident material buffer through the
    fast path; texture/descriptor/displacement fields still emit the structured
    repack diagnostic and use a scene repack for coherence.
  - Regression coverage observes `queue.writeBuffer` to the material buffer and no
    `setScene()` fallback for the scalar material path.
- Focused gate:
  - `cd packages/pt-webgpu && npx vitest run src/__tests__/updatePrimitiveIncremental.test.ts`
  - root `npm run typecheck`

### VQ-002 — pt-webgl2 mesh-light material mutation must use `nextGeoPack`

- Label: **CLOSE-NOW**
- Status: **CLOSED 2026-06-17** — mesh-light repack now uses `nextGeoPack`, with
  a regression test covering changed emissive material radiance.
- Files:
  - `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`
  - `packages/pt-webgl2/src/scene/meshAreaLights.test.ts` or nearby mutation test
- Verified closure:
  - Mesh-area light repack calls `packMeshAreaLights(nextScene, nextGeoPack)`.
  - Regression coverage mutates emissive material radiance and observes the packed
    mesh-light texture using the new material state.
- Focused gate:
  - `cd packages/pt-webgl2 && npx vitest run src/scene/mutateSceneTextures.test.ts`
  - root `npm run typecheck`

### VQ-003 — walkaround `denoiser:'none'` option should construct

- Label: **CLOSE-NOW/TRUTH**
- Status: **CLOSED 2026-06-17** — `none` is accepted as a first-class
  pass-through denoiser mode and stale rejection tests were updated.
- Files:
  - `packages/walkaround-hybrid/src/HybridEngineOptions.ts`
  - `packages/walkaround-hybrid/src/HybridEngineConfig.ts`
  - `packages/walkaround-hybrid/src/pipeline/denoisers/*`
  - tests near denoiser registry/tuning
- Verified closure:
  - `VALID_DENOISERS` includes `'none'`, and construction/config validation accepts
    it as a public no-op denoiser mode.
  - Registry/config tests cover routing to the no-op adapter.
- Focused gate:
  - `cd packages/walkaround-hybrid && npx vitest run` focused denoiser/config tests
  - root `npm run typecheck`

### VQ-004 — walkaround material-only edits should refresh DDGI probe state

- Label: **CLOSE-NOW**
- Status: **CLOSED 2026-06-18** — material-slice edits refresh DDGI's
  material snapshot without forcing RC geometry propagation; receiver
  radiance/visibility plus roughness/metallic changes invalidate DDGI probes.
- Files:
  - `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`
  - `packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts`
- Fields covered:
  - Scalar slice path: `attenuationColor`, `attenuationDistance`, `thickness`,
    `transmission`, Beer/tint-relevant `baseColor`, `roughness`, and
    `metallic`.
  - Atlas-backed maps still route through the full rebuild path when map handles
    or atlas metadata change.
- Verified closure:
  - `attenuationDistance`-only patch calls `invalidateProbeCache()`.
  - Roughness-only scalar patch refreshes DDGI snapshot and invalidates probes
    without emitter rebuild.
- Focused gate:
  - `cd packages/walkaround-hybrid && npx vitest run src/__tests__/mutationMatrix.test.ts`
  - root `npm run typecheck`

## Wave Schedule

### Wave 0 — Revalidate And Land Fast Correctness

Concurrency: 3 workers max.

- Worker A: VQ-001
- Worker B: VQ-002
- Worker C: VQ-003
- Lead/local after workers: VQ-004, because it touches a sensitive walkaround
  mutation path and should be reviewed in one place.

Wave gate:

```bash
npm run typecheck
npm test --workspace @vitrum/pt-webgpu
npm test --workspace @vitrum/pt-webgl2
npm test --workspace @vitrum/walkaround-hybrid
```

If full package tests are too slow, run focused tests first, then at least one
root typecheck before commit.

### Wave 1 — Truthfulness Cleanup

Start only after Wave 0 is green. Re-read `plan/road-to-100.md` and
`packages/core/src/engine/promiseLedger.ts`; do not use stale Cursor line
numbers.

Status 2026-06-17: **CLOSED for the verified Wave-0 deltas.** The Road row that
still claimed pt-webgpu lite material patches were fallback-rebuild was updated,
and a stale ledger-test label for walkaround point/spot emitters was reconciled
with the current native row. `ledgerVsCapabilities` and `materialNativeEvidence`
passed.

Targets:

- Remove duplicated/stale Road rows that make closed work look open.
- Fix any ledger/capability mismatch discovered while landing Wave 0.
- Ensure warnings for parked approximations are structured and source-pathed.

Concurrency: 1 if `promiseLedger.ts` is touched; otherwise 2.

Wave gate:

```bash
npm run typecheck
cd packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts src/__tests__/materialNativeEvidence.test.ts
```

### Wave 2 — Adjoint Scope Closure, Not Full Inverse Research

Purpose: stop inverse work from being an endless tail.

Status 2026-06-17: **CLOSED by source revalidation and gate.** Current source
already downgrades unsupported path-replay domains to finite difference with
structured diagnostics for hook absence, spectral/multi-bounce render regimes,
transport, alpha visibility, normal/bump/clearcoat-normal scale, environment
terms, receiver material gaps, unsupported emitters, and unsupported primitive
kinds. Focused inverse/adjoint tests and `shader-gate` passed.

Allowed work:

- Add missing downgrade diagnostics for unreplayed domains.
- Close small single-bounce direct-light derivatives where forward code is
  already mirrored and tests are local.
- Add tests proving unsupported path-replay requests fall back to finite
  difference with the right reason.

Not allowed in this wave:

- Indirect/path-space full adjoint.
- Spectral/volume/layered adjoint.
- Exact texel-PDF light-selection research.

Concurrency: 1, because `inverseSession.ts` / `adjointPass.wgsl.ts` are mutex.

Wave gate:

```bash
cd packages/pt-webgpu && npx vitest run src/__tests__/inverseSession.test.ts src/__tests__/adjointHarness.test.ts
npm run shader-gate
npm run typecheck
```

### Wave 3 — glTF Proof Harness, Not New Material Features

Purpose: prove the API path handles representative arbitrary assets without
turning this into renderer research.

Status 2026-06-17: **CLOSED for available WSL proof lanes.**
`gltf-material-sweep` passed and `behavioral-gate -- --filter gltf` passed all
six pt-webgpu glTF configs with finite non-black output and zero GPU errors.
The real public-asset lane now also has committed pt-webgpu lavapipe golden PNGs
for BoxTextured, CesiumMilkTruck Draco, and MeshoptCubeTest meshopt assets; the
real-asset import/decode sweep reports those `behavioral-gate` proof labels and
golden paths through shared metadata. The synthetic topology lanes for point/line
fallback and triangle strip/fan conversion now also have checked manifests and
shared proof metadata via `npm run gltf-topology-proof-check`. Recommended
backend/browser captures remain in the validation queue, as intended.

2026-06-17 follow-up: `behavioral-gate -- --filter gltf-material-sweep` still
passes on lavapipe as an auto-tier/lite PNG proof. The documented
`--require-full-tier` path now actively requests `traceTier:"full"` instead of
checking after the engine has already auto-selected lite; on this WSL lavapipe
adapter it fails honestly because the adapter reports
`maxStorageBuffersPerShaderStage=8` while pt-webgpu full tier requires at least
34. The same full-tier assertion is now committed as a companion WSL dzn status
artifact via
`npm run behavioral-gate:dzn -- --filter gltf-material-sweep --require-full-tier`
(`tier=full`, zero GPU errors, golden RMSE 0.544); `npm run proof-check`
validates `tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json`.
2026-06-20 follow-up: `pt/material-lobes` and `pt/material-lobe-maps` now add
focused scalar-lobe and CPU-readable map-backed behavioral scenes for clearcoat,
sheen, iridescence, anisotropy, and dielectric specular panels. Both lanes have
lavapipe goldens and dzn full-tier `dzn-full` golden/status artifacts via
`npm run behavioral-gate:dzn -- --filter material-lobes --require-full-tier`
and
`npm run behavioral-gate:dzn -- --filter material-lobe-maps --require-full-tier`
(`tier=full`, zero GPU errors, golden RMSE 0.000). Treat broader
specialty-integrator and radiometric A/B promotion as the remaining validation
work.

Allowed work:

- Add or tighten real-asset glTF boot/readback fixtures.
- Add golden-PNG comparison hooks if the harness is already close.
- Improve diagnostics for assets that must route away from a backend.

Not allowed:

- Implementing new material lobes.
- Native point/line primitives.
- Native arbitrary UV arrays.

Wave gate:

```bash
npm run gltf-material-sweep
npm run gltf-topology-proof-check
npm run proof-check
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json npm run behavioral-gate -- --filter gltf
npm run typecheck
```

### Wave 4 — Final Ledger Reconciliation

Purpose: make the repo stop looking less complete than it is.

Status 2026-06-17: **CLOSED for this execution pass.** Road/ledger residue found
during Waves 0-3 was reconciled, and every remaining implementation-looking row
in this plan is now either closed, validation/proof, product-decision, or parked
post-contract work. Final gates run in WSL Node 20: `npm run typecheck`,
`npm test`, `npm run shader-gate`, `npm run gltf-material-sweep`, and
`behavioral-gate -- --filter gltf`.

Tasks:

- Reconcile `plan/road-to-100.md`, `plan/renderer-fidelity-matrix.md`, and
  `promiseLedger.ts`.
- Every remaining implementation-looking row must be one of:
  - an active task in this plan,
  - a validation/proof item,
  - a product decision,
  - explicitly parked.

Concurrency: 1.

Wave gate:

```bash
npm run typecheck
npm test
npm run shader-gate
```

### Wave 5 — Source-Reader Bounded Tails

Status 2026-06-20: **CLOSED.** A follow-up source-reader pass found four
bounded code tails that were implementation work rather than proof work:

- pt-webgpu inverse now treats scalar `displacementScale` / `displacementBias`
  as finite-difference inverse parameters, matching the shared-BVH vertex
  displacement renderer path, while requested path replay still downgrades with
  a structured geometry diagnostic.
- pt-webgpu inverse receives active full/lite material support details, so
  runtime-profile-unsupported material fields are rejected before session start.
- walkaround material atlas and DDGI probe-hit material sampling now transform
  derived fallback tangent/bitangent frames through the TLAS local-to-world
  direction matrix before normal/bump-map evaluation; DDGI smooth probe-hit
  normals use the TLAS inverse-transpose normal transform before material
  response/direct-lighting.
- walkaround default and NRC ReSTIR-GI producers now run selected-sample final
  visibility through the RGB transparent-tint helper and scalarize the result
  for reservoir weighting, matching the direct-light tint path without
  broadening the GI shader closure to the full common aggregate.
- TLAS stained-glass tinted visibility reconstructs world-space hit distance
  for the caller's finite `tMax`, and glTF anisotropy-only texture assets now
  trigger generated tangents just like normal/clearcoat-normal/bump maps.

Focused gates:

```bash
npm test --workspace @vitrum/pt-webgpu -- inverseSession.test.ts
npm test --workspace @vitrum/walkaround-hybrid -- smoothNormals.test.ts pcgHashPolicy.test.ts
npm test --workspace @vitrum/gltf-adapter -- gltfAdapter.test.ts
npm run shader-gate
npm run typecheck
```

### Wave 6 — Public Callback Predictability

Status 2026-06-21: **CLOSED.** A source scan of the remaining raw public
callback surfaces found three bounded predictability tails:

- `createProgressiveEngine()` now guards its single `onAdapterProfile` telemetry
  callback, matching the already-guarded walkaround construction path, so a host
  HUD/CI callback throw cannot abort progressive engine construction after the
  shared device is allocated.
- `decodeSceneTextures()` now guards `onDiagnostic` and `onWarning` callbacks
  while keeping diagnostics and warnings in the returned result. Host UI/logging
  code can no longer abort arbitrary-glTF texture normalization.
- glTF accessor/material/animation/compression/texture-acquisition diagnostic
  emitters now treat diagnostic sinks as advisory, preserving returned warning
  data where applicable if a host or helper callback throws.
- `mergeWorldSpaceFromCore()` now routes malformed-triangle warnings into both
  `WorldSpaceMergeResult.warnings` and the optional `onWarning` callback while
  preserving existing `console.warn` output; the callback is guarded so advisory
  warning sinks cannot break DDGI/RC/ReSTIR ingestion.
- pt-webgl2 and pt-webgpu inverse-session diagnostic callbacks are guarded, so
  path-replay downgrade diagnostics cannot change session construction or method
  selection when host diagnostic code throws.

Focused gates:

```bash
npm exec -- vitest run \
  packages/engine/__tests__/createProgressiveEngineErrors.test.ts \
  packages/gltf-adapter/src/gltfAssetApi.test.ts \
  packages/shared-bvh/src/__tests__/worldSpaceMergeInvalidTriangles.test.ts \
  packages/pt-webgl2/src/__tests__/inverseSession.test.ts \
  packages/pt-webgpu/src/__tests__/inverseSession.test.ts
```

### Wave 7 — CPU Texture Data-Type Semantics

Status 2026-06-21: **CLOSED.** A direct source sweep of CPU-readable texture
decoders found a small but real cross-backend predictability gap: several
renderer readers treated plain `Uint16Array` handles, and even
`dataType:'uint16'`, as half-float payloads. The glTF decode bridge and
pt-webgpu inverse alpha helper already interpret `uint16` as normalized
integer data, so the same authored CPU texture could produce different values
depending on which backend/subsystem read it.

Implementation:

- `shared-bvh` vertex displacement now treats unhinted `Uint16Array` and
  explicit `dataType:'uint16'` height maps as normalized `[0,1]` data; true
  half-float displacement remains available via `dataType:'float16'` or
  `'half-float'`.
- `shared-bvh` emitter texture classification now uses the same policy for
  average emissive/color estimates and UV texel sampling.
- `shared-samplers` environment-map pixels now use normalized `uint16` by
  default, with explicit `float16`/`half-float` for HDR half-float handles.
- `pt-webgl2` material texture arrays and `walkaround-hybrid` material atlases
  now use normalized `uint16` by default and reserve half-float for explicit
  hints.
- `walkaround-hybrid` alpha-blend diagnostics now classify `uint16` alpha
  coverage consistently instead of silently treating values like `0x3c00` as
  opaque half-float `1.0`.
- A stale `gltfToScene` comment was reconciled with the actual decode-report
  contract: browser `ImageBitmap` handles are ready for pt-webgpu, while
  pt-webgl2 and walkaround still need CPU-readable decoded pixels.

Focused gates:

```bash
npm test --workspace @vitrum/shared-bvh -- scenePack.test.ts emitterClassify.test.ts
npm test --workspace @vitrum/shared-samplers -- environmentMapPixels.test.ts
npm test --workspace @vitrum/pt-webgl2 -- texturesArray.test.ts equirectHdrInfo.test.ts
npm test --workspace @vitrum/walkaround-hybrid -- materialTextureAtlas.test.ts consumedMaterialFields.test.ts
npm test --workspace @vitrum/pt-webgpu -- environmentPacking.test.ts inverseSession.test.ts
npm run typecheck
```

### Wave 8 — pt-webgl2 BDPT Reverse-PDF Recheck

Status 2026-06-21: **SOURCE-VERIFIED NON-GAP / GUARDED.** A subagent flagged
WebGL2 multi-vertex BDPT as if generated surface light vertices still exposed
Lambertian reverse PDFs in material-rich paths. Direct code-read narrowed this:

- `packages/pt-webgl2/src/index.ts` clamps WebGL2 BDPT light subpaths to
  `BDPT_MAX_LIGHT_BOUNCES = 3`; `maxLightBounces > 1` is still explicit
  research-mode opt-in.
- `packages/pt-webgl2/src/glsl/render/bdpt_light_subpath.glsl.js` stores a
  current-column `cos(theta)/pi` placeholder because WebGL2 cannot rewrite the
  previous light-path column during that draw.
- `packages/pt-webgl2/src/glsl/render/bdpt_connection.glsl.js` overwrites
  `mRev[c]` and `mRev[c - 1]` with material-aware straddle PDFs before the
  Veach MIS sweep. With the current three-column cap, every generated surface
  vertex that can participate in the multi-vertex path is either `L_c` or
  `L_{c-1}` for a connection and therefore receives an override.

Implementation:

- Added a structural GLSL composition guard that pins the three-column host cap,
  the light-subpath placeholder, and both connection-side overrides together.
  If the WebGL2 cap is raised later, this test is the tripwire for implementing
  broader internal-light-chain reverse-PDF recomputation.

Focused gate:

```bash
npm test --workspace @vitrum/pt-webgl2 -- composeTraceGlsl.test.ts bdptDriver.test.ts
```

### Wave 9 — Walkaround ReSTIR Emitter Shadow Lane

Status 2026-06-21: **CLOSED.** Direct source-read of the walkaround emitter path
found a bounded implementation miss in an otherwise-closed promise row:
material-backed ReSTIR-DI mesh emitters hardcoded `castShadowDisabled:false`.
That meant a source primitive or explicit `mesh-area` emitter with
`castShadow:false` could still cast the ReSTIR direct-light shadow ray, even
though analytic emitters, DDGI, RC, pt-webgpu, and pt-webgl2 already carried the
shadow-disabled lane.

Implementation:

- `coreEmitterBuffers()` now builds its ReSTIR emitter-only merged stream with
  `splitMaterialsByCastShadow:true`, matching the render-BVH material flag lane.
- `buildMeshAreaLeOverrides()` carries both the Le override and the explicit
  mesh-area emitter shadow flag into the production material slice.
- `buildEmitterListFromCore()` propagates the material shadow flag through
  ordinary mesh-material emitters, bounded barycentric subdivision, and exact
  emissive-map texel-cell subemitters.
- `directLightEmitterCore.test.ts` pins merged and TLAS scalar emitters,
  exact texel-cell emissive-map subemitters, and explicit mesh-area emitters.

Focused gate:

```bash
npm test --workspace @vitrum/walkaround-hybrid -- directLightEmitterCore.test.ts
npm test --workspace @vitrum/walkaround-hybrid -- bvhCoreMaterialResolver.test.ts directLightEmitterCore.test.ts roughMetalPacking.test.ts
npm run typecheck --workspace @vitrum/walkaround-hybrid
```

### Wave 10 — pt-webgpu Inverse Emitter Profile Gate

Status 2026-06-21: **CLOSED.** Direct source-read of
`packages/pt-webgpu/src/inverse/inverseSession.ts` found one bounded
truthfulness miss in the path-replay classifier: material-target inverse fits
passed the active runtime emitter-support table into the direct-light replay
diagnostic, but emitter-target fits accidentally dropped that table before
calling the same lighting check. A point-light color/intensity fit in a scene
with a contributing mesh-area light could therefore keep path replay even when
the active pt-webgpu profile reported `mesh-area: unsupported`.

Implementation:

- `diagnosePathReplaySlot()` now threads `emitterSupportDetails` into the
  emitter-target diagnostic path, matching the material-target path.
- `inverseSession.test.ts` pins a point-emitter fit that downgrades with
  `path-replay-unsupported-lighting` when another contributing mesh-area emitter
  is unsupported by the active runtime profile.

Focused gate:

```bash
npm test --workspace @vitrum/pt-webgpu -- inverseSession.test.ts
npm run typecheck --workspace @vitrum/pt-webgpu
```

### Wave 11 — Runtime Truthfulness And Texture Policy Tails

Status 2026-06-22: **CLOSED.** A source-verified pass over the latest external
feedback left three bounded implementation tails, all smaller than the remaining
validation/proof queue:

- Walkaround unsupported-material diagnostics now retain per-primitive authorship
  on both `setScene()` and `updatePrimitive()` paths. The aggregate `fields`
  list remains compact, while `details.primitiveFields[]` identifies the exact
  primitive id and unsupported field list that triggered permanent unsupported
  material families.
- pt-webgpu full-tier directional-light parity now uses the authored
  `DirectionalEmitter.angularDiameter` in the in-medium NEE path and ReSTIR-PT
  suffix direct-light replay, matching the existing surface direct-light
  soft-cone sampling instead of silently treating those paths as hard delta
  directionals.
- The glTF decode bridge now has an explicit `npotRepeatWrapPolicy`:
  `warn` preserves the old diagnostic-only posture, `resize-to-pot` resamples
  decoded CPU handles to deterministic power-of-two dimensions, and
  `clamp-sampler` rewrites the returned sampler wrap to `clamp-to-edge`. The
  option threads through `decodeSceneTextures()`, `loadGltfAndDecodeTextures()`,
  `loadGltfForEngine()`, and `@vitrum/engine/gltf` type exports.

Focused gates run in WSL Node 20:

```bash
npm test --workspace @vitrum/walkaround-hybrid -- consumedMaterialFields.test.ts mutationMatrix.test.ts
npm test --workspace @vitrum/pt-webgpu -- oracle.directionalConeSample.test.ts restirPtReuseContract.test.ts volumetricSss.test.ts restirPtReuseWiring.test.ts wgslContract.test.ts
npm test --workspace @vitrum/gltf-adapter -- gltfAssetApi.test.ts
npm test --workspace @vitrum/engine -- gltfStrictPtWebgpuTier.test.ts
npm run typecheck --workspace @vitrum/walkaround-hybrid
npm run typecheck --workspace @vitrum/gltf-adapter
npm run typecheck --workspace @vitrum/engine
```

### Wave 29 — Validation Queue Proof-Citation Hardening

Status 2026-06-22: **CLOSED.** Three delegated source audits plus direct lead
verification found no bounded runtime implementation gap in orchestration,
future-contract truthfulness, or the renderer proof rows. The one actionable
gap was proof-ledger hardening: several validation rows were backed by real
checkers and source/tests, but the machine-readable Road queue did not cite the
full dependency chain, so stale evidence claims could survive as long as the
top-level PNG/status artifacts still existed.

Implementation:

- Strengthened `tools/road-to-100/check-validation-queue.mjs` with mandatory
  proof-artifact and source-needle checks for `VQ-PT-WEBGPU-RUNTIME-GOLDENS`,
  `VQ-GLTF-REAL-WEBGPU`, `VQ-GLTF-MATERIAL-TOPOLOGY`, and the non-BDPT parts
  of `VQ-RADIOMETRIC-PT`.
- Expanded `tools/road-to-100/validation-queue.json` so those rows cite the
  actual proof scripts, behavioral-gate source, manifests, dzn status JSONs,
  and package oracle tests they depend on.
- Kept the active implementation queue empty: these edits do not promote any
  validation/provisioning/future-contract tail into runtime implementation.

Gates:

```bash
npm run road-to-100-source-check
npm run proof-check
npm run typecheck
npm run shader-gate
npm test
git diff --check
```

### Wave 30 — Walkaround Behavioral Matrix Proof-Citation Guard

Status 2026-06-22: **CLOSED.** After Wave 29, the remaining weakest
committed-proof row was `VQ-WALKAROUND-BEHAVIORAL-MATRIX`: the focused dzn
status JSONs and PNGs were real, but the Road queue/checker did not force the
gate source labels or the transparent-OIT approximation contract files that
keep the row honest.

Implementation:

- Added required walkaround behavioral proof rows to
  `tools/road-to-100/check-validation-queue.mjs`, covering every focused
  `wh/*` dzn status artifact and corresponding `dzn-full` PNG golden.
- The checker now fails if a focused shard loses `PASS`, `exitStatus:0`,
  `gpuErrors:0`, `nan:false`, `goldenStatus:'ok'`, `goldenVariant:'dzn-full'`,
  or its expected gate label.
- Expanded `VQ-WALKAROUND-BEHAVIORAL-MATRIX` proof artifacts to cite the
  behavioral gate scripts plus the transparent-OIT shader and contract test
  sources that pin camera-visible local lighting without ReSTIR/GI reservoir
  participation.

Focused gates:

```bash
npm run road-to-100-validation-status
npm run behavioral-gate:dzn-status-check
npm test --workspace @vitrum/walkaround-hybrid -- transparentAlphaTransportContract.test.ts
npm run proof-check
git diff --check
```

### Wave 31 — Mutation Matrix Proof-Citation Guard

Status 2026-06-22: **CLOSED.** The active implementation queue remained empty:
source revalidation showed the pt-webgpu mutation router and walkaround mutation
paths already implement the current contract. The remaining weak spot was that
`VQ-MUTATION-MATRIX` cited only status JSONs and PNGs, so a stale Road row could
miss loss of the gate labels, routing tests, or backend mutation source that
make those captures meaningful.

Implementation:

- Added required mutation proof artifacts to
  `tools/road-to-100/check-validation-queue.mjs`, including the behavioral gate,
  dzn status checker, pt-webgpu mutation router/patch tests, and walkaround
  mutation/GI propagation sources.
- The checker now verifies every pt-webgpu and walkaround mutation gate label
  plus the matching `mutationGolden(...)` binding for material, environment,
  emitter, transform, topology, instanced-count, add-primitive, and
  remove-primitive.
- The checker also pins source-level mutation routing needles: pt-webgpu
  add/remove/updatePrimitive/updateEmitter/updateEnvironment, bind-group/lite
  texture invalidation, and walkaround TLAS/material/DDGI/RC propagation hooks.

Focused gates:

```bash
npm run road-to-100-validation-status
npm run behavioral-gate:dzn-status-check
npm test --workspace @vitrum/pt-webgpu -- mutationDesyncs.test.ts
npm test --workspace @vitrum/walkaround-hybrid -- mutationMatrix.test.ts
npm run proof-check
git diff --check
```

### Wave 28 — pt-webgl2 Mesh-Light MIS Prose Reconciliation

Status 2026-06-22: **CLOSED.** Source revalidation of the pt-webgl2 mesh-area
light path found the runtime and tests already use emitted-power-proportional
triangle/texel-cell selection for mesh-light NEE, but the source header in
`meshAreaLights.ts` still described the older area-proportional derivation. This
was not a runtime bug, but it was a source-level contradiction in the renderer
math documentation.

Implementation:

- Rewrote the `meshAreaLights.ts` double-count/MIS derivation to match the live
  emitted-power selection and forward-hit PDF reconstruction.

Focused gate:

```bash
npm test --workspace @vitrum/pt-webgl2 -- meshAreaMis.test.ts
```

### Wave 27 — RC Material-Atlas Parity Fix

Status 2026-06-22: **CLOSED.** Source revalidation of the walkaround/RC
material path found a real bounded code gap: the RC probe-ray shader consumed
the walkaround material atlas with a stale 53-texel per-triangle metadata stride
while the atlas producer and main walkaround material shader use 56 texels. That
misaddressed every RC material-map metadata load after triangle 0. The same RC
scalar-map helper also returned only the sampled map channel for roughness and
metallic maps, while the main material shader correctly multiplies the authored
scalar fallback by the sampled channel.

Implementation:

- Updated `probeRayCast.wgsl.ts` to use the 56-texel material-map metadata
  stride and to return `fallback * mapChannel` for scalar material maps.
- Added RC WGSL structural tests that compare the RC shader stride against the
  atlas producer and main material shader, and pin the scalar-map multiplier.
- Kept runtime package boundaries intact: `@vitrum/walkaround-rc` does not gain
  a dependency on `@vitrum/walkaround-hybrid`; the test reads sibling source
  files to pin parity.

Focused gates:

```bash
npm test --workspace @vitrum/walkaround-rc -- probeRayCastWgsl.test.ts rcKernelMath.test.ts
npm test --workspace @vitrum/walkaround-hybrid -- consumedMaterialFields.test.ts materialTextureAtlas.test.ts rcMergedRefit.test.ts giPropagationRcMerged.test.ts
npm run shader-gate
npm run validate:gpu:smoke
git diff --check
```

### Wave 26 — Walkaround Texture-Map Ledger Prose Reconciliation

Status 2026-06-22: **CLOSED.** Direct source read of
`BACKEND_PROMISE_LEDGER` found a stale walkaround `baseColorMap` comment still
claiming "the rest of the texture-map family is not sampled" even though the
same ledger, `CONSUMED_MATERIAL_FIELDS`, and shader/material-atlas evidence now
list normal, ORM, AO, alpha, emissive, transmission, thickness, light,
specular, clearcoat, sheen, anisotropy, iridescence, bump, and displacement map
families as consumed with approximate semantics. The code/test contract was
already right; the stale prose was the contradiction.

Implementation:

- Reworded the `baseColorMap` ledger comment to keep the approximate boundary
  on scalar Beer/transmission tint, mapped-emitter PDFs, and transparent
  transport instead of incorrectly saying other texture-map families are
  unsampled.
- Strengthened `road-to-100-validation-status` so the stale sentence fails the
  queue check if it returns.

Focused gates:

```bash
npm test --workspace @vitrum/walkaround-hybrid -- consumedMaterialFields.test.ts
npm test --workspace @vitrum/core -- ledgerVsCapabilities.test.ts
npm run road-to-100-validation-status
git diff --check
```

### Wave 25 — Learned Systems Training-Pipeline Evidence Guard

Status 2026-06-22: **CLOSED.** Source revalidation of
`VQ-LEARNED-SYSTEMS` found the production posture is intentionally still
provisioning-gated: the repo has tracked research checkpoints, runtime shape
validation, opt-in neural/NRC/PPG/GRIS guards, and no bundled production
checkpoint. The weak point was evidence drift: the learned-systems proof checker
validated checkpoint bytes and runtime truthfulness, but the Road queue did not
require the training script, exporter, smoke capture tool, dataset contract, or
runtime round-trip test as cited proof artifacts.

Implementation:

- Added a learned-systems training-pipeline assertion that reads
  `train.py`, `export_weights.py`, `capture-dataset.mjs`, `dataset_spec.md`,
  and `neuralWeightsRoundTrip.test.ts`, then fails closed if the lazy PyTorch
  dry-run path, canonical parameter count, vitrum-model binary writer/exporter,
  smoke-capture caveat, production dataset sizing/capture gap, or runtime
  load/validate/InferenceGraph round-trip evidence drifts.
- Expanded `VQ-LEARNED-SYSTEMS` proof artifacts to cite the training/export/
  capture/dataset/round-trip evidence directly.
- Strengthened `tools/road-to-100/check-validation-queue.mjs` so the learned
  row cannot silently drop those citations while still honestly remaining
  `provisioning-needed` until a production checkpoint and quality A/B manifest
  exist.

Focused gates:

```bash
npm run learned-systems-proof-check
npm run road-to-100-validation-status
npm test --workspace @vitrum/walkaround-hybrid -- learnedSystemConfig.test.ts capabilitiesPartition.test.ts hybridLiteTier.test.ts neuralWeightsRoundTrip.test.ts
npm run proof-check
git diff --check
```

### Wave 24 — BDPT Estimator Boundary Guard

Status 2026-06-22: **CLOSED.** Source revalidation of
`VQ-RADIOMETRIC-PT` found the remaining BDPT promotion blocker is not a narrow
shader typo. The explicit connection kernel already carries a full Veach 10.3
MIS sweep, the light-subpath builder uses real material sampling at light
vertices, and both have oracle tests. The measured multi-vertex drift remains
because the path-tracing megakernel still accumulates the ordinary eye-path
estimator at full weight while adding extra light-subpath connections. That is a
larger estimator-composition redesign, so the opt-in multi-vertex branch must
stay a research diagnostic rather than a production-promoted default.

Implementation:

- Updated the pt-webgpu path-tracing kernel comment at the BDPT accumulation
  site to state the exact non-promotion boundary: explicit eye-light connection
  MIS exists, but the ordinary eye-path estimator is not yet composed into the
  same strategy family.
- Expanded `VQ-RADIOMETRIC-PT` proof artifacts to cite the BDPT harness/checker,
  host-status/result JSONs, pt-webgpu warning source, BDPT connection/light
  subpath WGSL, and the oracle tests that pin the current implementation.
- Strengthened `tools/road-to-100/check-validation-queue.mjs` so the row fails
  if it loses those proof/source citations or stops documenting the ordinary
  eye-path estimator boundary.

Focused gates:

```bash
npm test --workspace @vitrum/pt-webgpu -- bdptConnectionMisFull.test.ts bdptGlossyLightSubpath.test.ts h51WarnCoercions.test.ts
npm run shader-gate
npm run radiometric-ab:proof-check
npm run road-to-100-validation-status
npm run proof-check
git diff --check
```

### Wave 23 — Walkaround A/B Evidence-Citation Guard

Status 2026-06-22: **CLOSED.** Source revalidation of
`VQ-WALKAROUND-RADIOMETRIC-AB` found no bounded implementation row hiding behind
the partial status. The current walkaround A/B harness and checker already pin a
full four-case proof set: A8/GRIS is `NEGLIGIBLE`, SUN is analytic `PASS`, GLASS
is `PASS`, and GLOSSY is a material-effect `FINDING` with
`promotion.defaultReady:false` because the DDGI cache is not GGX-filtered
radiance.

Implementation:

- Expanded `VQ-WALKAROUND-RADIOMETRIC-AB` proof artifacts to cite the harness,
  proof metadata, proof checker, radiometric README, host-status/result JSONs,
  `restirPtReuse` option source, GGX/specular GI shader sources, and the glossy
  material GI regression test.
- Strengthened `tools/road-to-100/check-validation-queue.mjs` so the row fails
  if it loses the required source/proof citations, drifts off `PASS-PARTIAL`, or
  drops the do-not-promote glossy `FINDING` boundary.
- Kept the row partial-proof-green: browser/real-adapter or higher-quality
  case-specific references are still needed before promoting GRIS/ReSTIR-GI,
  PPG, NRC, or rich-material GI quality claims.

Focused gates:

```bash
npm run radiometric-ab:proof-check
npm run road-to-100-validation-status
npm run proof-check
git diff --check
```

### Wave 22 — Renderer Fidelity Evidence-Citation Guard

Status 2026-06-22: **CLOSED.** Source revalidation of
`VQ-RENDERER-FIDELITY-PROOF` found `renderer-fidelity-proof-check` already
fail-closes the substantive claims: all pt-webgpu `supported` rows need runtime
proof artifacts, pt-webgl2 specialty rows remain explicitly unpromoted while
browser/WebGL2 capture is `HOST-BLOCKED`, material-furnace rows are source/oracle
guarded, and stale package-level pt-webgpu/pt-webgl2 prose is rejected. The weak
point was the Road queue: it cited the matrix and only three PNGs, not the full
set of source files, dzn status files, browser-blocked artifacts, and baseline
goldens the checker actually reads.

Implementation:

- Expanded `VQ-RENDERER-FIDELITY-PROOF` proof artifacts to cite the renderer
  proof checker, fidelity playbook, architecture/README/hardware docs, browser
  pt-webgl2 status and manifest, four dzn status files, all pt-webgpu baseline
  golden PNGs used by the checker, and the pt-webgl2 material-furnace source and
  oracle test files.
- Strengthened `tools/road-to-100/check-validation-queue.mjs` so the renderer
  fidelity row fails if any of those evidence files stops being cited by the
  machine-readable queue.
- Kept the row partial-proof-green: the remaining work is still browser/real
  adapter reference A/B for pt-webgl2 specialty promotion, not a hidden current
  implementation row.

Focused gates:

```bash
npm run renderer-fidelity-proof-check
npm run road-to-100-validation-status
npm run proof-check
git diff --check
```

### Wave 21 — BDPT Multi-Vertex Non-Promotion Artifact Guard

Status 2026-06-22: **CLOSED.** Source revalidation of
`VQ-RADIOMETRIC-PT` found the default `bdpt:true` path remains endpoint-only and
radiometrically neutral against `bdpt:false`, while `maxLightBounces > 1` still
requires `bdptOptions.experimentalMultiVertex:true` and is intentionally
research-only. The remaining multi-vertex estimator work is not a small shader
patch: the current research branch is not yet weighted against the regular
eye-path strategy, and the committed A/B snapshot records +13% to +17% mean
drift when those extra light-subpath connections are enabled.

Implementation:

- Added explicit `controls.multiVertexPromotion` metadata to
  `tools/radiometric-ab/ab-bdpt.mjs` and the committed
  `results-bdpt.json`: `defaultReady:false`, warning code, blocker,
  required estimator, and evidence path.
- Strengthened `radiometric-ab:proof-check` to require that metadata in the
  result snapshot and to require the matching structured warning source.
- Strengthened the Road validation queue JSON expectations for
  `VQ-RADIOMETRIC-PT`, and updated the radiometric README so the human-facing
  proof note matches the machine artifact.

Focused gates:

```bash
npm test --workspace @vitrum/pt-webgpu -- h51WarnCoercions.test.ts
npm run radiometric-ab:proof-check
npm run road-to-100-validation-status
npm run proof-check
git diff --check
```

### Wave 20 — Learned Systems Evidence-Citation Guard

Status 2026-06-22: **CLOSED.** Source revalidation of
`VQ-LEARNED-SYSTEMS` found the learned-system proof checker already validates
the two committed research checkpoints byte-for-byte, rejects unregistered model
files, keeps `productionCheckpoint:null` fail-closed, and requires production
quality A/B metadata before any production checkpoint can count. The weak point
was evidence accounting: the Road queue cited only part of the source and docs
that the checker reads to prove neural/NRC/PPG/GRIS default-off truthfulness.

Implementation:

- Expanded `VQ-LEARNED-SYSTEMS` proof artifacts to cite the full evidence set:
  `HybridEngineOptions.ts`, root and package READMEs,
  `plan/library-architecture.md`, neural-training README, and
  `HARDWARE-VALIDATION-NEEDS.md` in addition to the existing source, checkpoint,
  and regression-test files.
- Strengthened `tools/road-to-100/check-validation-queue.mjs` so the learned
  systems row fails if any of those evidence files stops being cited by the
  machine-readable queue.
- No production checkpoint was fabricated and no default-tier NRC/PPG promotion
  was claimed; the row remains provisioning-needed until quality A/B evidence
  exists.

Focused gates:

```bash
npm run learned-systems-proof-check
npm run road-to-100-validation-status
npm run proof-check
git diff --check
```

### Wave 19 — Mutation Matrix Road Artifact Guard

Status 2026-06-22: **CLOSED.** Source revalidation of `VQ-MUTATION-MATRIX`
found the dedicated dzn status checker already enforced eight pt-webgpu and
eight walkaround mutation lanes, observable before/after pixel deltas, zero
failures, and committed `dzn-full` goldens. The Road checker only required the
two status artifacts to say `PASS` plus the PNGs to exist, which left room for a
future edit to weaken coverage without failing the umbrella Road gate.

Implementation:

- Strengthened `tools/road-to-100/check-validation-queue.mjs` so
  `VQ-MUTATION-MATRIX` requires all eight mutation kinds for both `pt` and
  `wh`: material, environment, emitter, transform, topology, instanced-count,
  add-primitive, and remove-primitive.
- The checker now reads both committed dzn status JSON artifacts and requires
  `verdict: PASS`, `exitStatus: 0`, zero failures, `goldenVariant: dzn-full`,
  per-lane `rawStatus: OK`, matching `mutationKind`, committed golden status,
  and observable mutation deltas.
- The checker also requires every per-lane `dzn-full` PNG artifact to be cited
  by the Road queue and keeps the Road text honest about pixel-delta and
  post-mutation-golden proof.

Focused gates:

```bash
npm run road-to-100-validation-status
npm run behavioral-gate:dzn-status-check
npm run proof-check
git diff --check
```

### Wave 18 — Scoped Adjoint Downgrade Taxonomy Guard

Status 2026-06-22: **CLOSED.** Source revalidation of
`VQ-ADJOINT-SCOPED-PATH-REPLAY` found the implementation already separates
supported direct-light path replay from finite-difference fallbacks for render
regimes, transport, visibility, geometry, light-selection, and environment
cases. The weak point was the Road validation checker: it mostly checked prose
needles rather than the concrete diagnostic taxonomy.

Implementation:

- Strengthened `tools/road-to-100/check-validation-queue.mjs` so
  `VQ-ADJOINT-SCOPED-PATH-REPLAY` fails if `inverseSession.ts` or
  `inverseSession.test.ts` lose the concrete downgrade codes:
  `path-replay-unsupported-render-regime`,
  `path-replay-unsupported-transport`,
  `path-replay-unsupported-visibility`,
  `path-replay-unsupported-geometry`,
  `path-replay-unsupported-light-selection`, and
  `path-replay-unsupported-environment`.
- Added checker requirements for the regression test names that pin transport,
  alpha-visibility, and displacement finite-difference downgrade coverage.

Focused gates:

```bash
npm run road-to-100-validation-status
npm test --workspace @vitrum/pt-webgpu -- inverseSession.test.ts
git diff --check
```

### Wave 17 — Walkaround Glossy Finding Proof Guard

Status 2026-06-22: **CLOSED.** Source revalidation of the walkaround `GLOSSY`
radiometric A/B row found the rich-material GI code path is present but
intentionally approximate: shade owns a GGX specular-indirect term, DDGI owns a
reflected-direction SH complement, and the committed A/B result is a
non-promotable `FINDING` because DDGI stores cosine-weighted irradiance rather
than GGX-filtered radiance.

Implementation:

- Added structured promotion metadata to the walkaround glossy result and
  harness output: `promotion.defaultReady:false`, blocker
  `ddgi-irradiance-cache-not-ggx-filtered-radiance`, and required evidence
  `material-furnace-reference-ab-and-browser-real-adapter-recapture`.
- Strengthened `radiometric-ab:proof-check` so a glossy `FINDING` must carry
  that exact non-promotion metadata.
- Strengthened the Road validation queue JSON expectations for
  `VQ-WALKAROUND-RADIOMETRIC-AB`.
- Reconciled stale `tools/radiometric-ab/README.md` text: GLASS is now the
  committed `PASS`; the aggregate remains `PASS-PARTIAL` only because GLOSSY is
  still a finding.

Focused gates:

```bash
npm run radiometric-ab:proof-check
npm run road-to-100-validation-status
git diff --check
```

### Wave 16 — BDPT Multi-Vertex Research Guard

Status 2026-06-22: **CLOSED.** Source revalidation of the opt-in
`bdptOptions.maxLightBounces > 1` path found no small shader patch that would
honestly promote multi-vertex BDPT: the extra light-subpath strategies are
currently added beside the regular eye-path estimator instead of being weighted
against that strategy set. The safe default remains endpoint-only and
radiometrically neutral; the multi-vertex branch remains reproducible research
evidence, not a promotable default.

Implementation:

- Added structured runtime warning fields to
  `pt-webgpu.bdpt-multivertex-research-mode`: `promotionReady:false`, blocker
  `not-weighted-against-regular-eye-path-strategy`, and evidence path
  `tools/radiometric-ab/results-bdpt.json`.
- Added `BDPT_MULTIVERTEX_RESEARCH_PROOF` and checker coverage so
  `npm run radiometric-ab:proof-check` fails if the committed A/B finding,
  source warning, blocker, or evidence path drift.
- Updated the Road-to-100 validation queue and queue checker so
  VQ-RADIOMETRIC-PT distinguishes closed safe-default BDPT from the remaining
  estimator-redesign tail.

Focused gates:

```bash
npm test --workspace @vitrum/pt-webgpu -- h51WarnCoercions.test.ts
npm run radiometric-ab:proof-check
npm run road-to-100-validation-status
```

Broad gates:

```bash
npm run proof-check
npm run typecheck
npm run shader-gate
npm test --workspace @vitrum/pt-webgpu
```

### Wave 15 — Future-Contract Code Queue Guard

Status 2026-06-22: **CLOSED.** A fresh source pass plus three delegated
source-only classifications found no bounded implementation row hiding in the
seven future-contract tails. Each row would require either a new public core
contract, a new transport/physics program, or both:

- real displacement/microdisplacement,
- true transparent layered GI transport,
- walkaround spectral/thin-film/layered/volumetric transport,
- native point/line primitives,
- arbitrary UV arrays,
- native instanced skinned/morphed primitives,
- full-path analytic adjoint parity.

Implementation:

- Added `codeNowBounded:false` and concrete `decisionBlockers` to every
  `futureContractRows[]` entry in `tools/road-to-100/validation-queue.json`.
- Strengthened `tools/road-to-100/check-validation-queue.mjs` so
  `npm run road-to-100-validation-status` fails if a future-contract row lacks
  blocker metadata or silently becomes an implied active implementation task.
- Updated `plan/road-to-100.md` to make the queue-control rule explicit.

Focused gate:

```bash
npm run road-to-100-validation-status
```

### Wave 14 — ReSTIR-PT Glossy Research Finding

Status 2026-06-22: **CLOSED.** The active implementation queue stayed empty, but
`VQ-RADIOMETRIC-PT` still named glossy ReSTIR-PT research-mode proof as a vague
remaining tail. Source read showed the opt-in branch is real and intentionally
gated by `restirPtReuseOptions.experimentalGlossyReuse:true`; the missing piece
was committed radiometric evidence for that branch.

Implementation:

- Added `tools/radiometric-ab/ab-restir-pt-glossy-research.mjs`, a full-tier
  repaired-Cornell A/B for `restirPtReuse:true` plus
  `experimentalGlossyReuse:true` against the base path.
- Captured `tools/radiometric-ab/results-restir-pt-glossy-research.json`.
  Verdict is `FINDING`: `globalRelErr=108.42%`, `roiRelErr=297.66%`,
  `varRatio=7.7140`, and `promotion.defaultReady=false`.
- Wired the artifact into `tools/radiometric-ab/proofs.mjs`,
  `tools/radiometric-ab/check-results.mjs`, and
  `tools/road-to-100/validation-queue.json`.

Focused gate:

```bash
npm run radiometric-ab:restir-pt-glossy-research
npm run radiometric-ab:proof-check
npm run road-to-100-validation-status
```

### Wave 13 — Machine-Readable Validation Queue

Status 2026-06-22: **CLOSED.** After Wave 12, the active runtime implementation
queue was source-verified empty. The remaining risk was process drift: validation,
provisioning, and future-contract rows kept being rediscovered as if they were
fresh code gaps.

Implementation:

- Added `tools/road-to-100/validation-queue.json` as the machine-readable queue:
  active implementation rows are empty; validation/proof rows carry commands,
  artifacts, and remaining blockers; future-contract rows are segregated.
- Added `tools/road-to-100/check-validation-queue.mjs`.
- Added `tools/road-to-100/check-source-gap-markers.mjs`, a production-source
  marker guard that classifies every intentional TODO/FIXME/stub/not-implemented
  hit and fails on any new unclassified marker.
- Wired `npm run road-to-100-source-check` to run the queue checker and marker
  guard, so `npm run proof-check` fails if a proof row loses its artifact, an
  npm command disappears, HOST-BLOCKED/PASS/PASS-PARTIAL statuses drift, a
  future-contract row sneaks back into the active implementation queue, or a new
  production stub marker appears without source-read classification.

### Wave 12 — pt-webgpu Bump Sampler Policy Parity

Status 2026-06-22: **CLOSED.** Direct source-read found one remaining bounded
texture-policy implementation tail in pt-webgpu: bump-map finite differences
still sampled raw-UV height values at base LOD, and the upload path therefore
emitted a `texture-sampler-policy-approximation` warning for authored bump
sampler policies.

Implementation:

- Forward full-tier bump-map height samples now use
  `sampleMaterialLayerLinearRawUvPolicy()`, preserving the one-source-texel raw-UV
  finite-difference step while applying the same per-map mip/filter policy,
  nearest `textureLoad`, and explicit-LOD linear sampling as regular linear
  material maps.
- The adjoint pass mirrors the same raw-UV policy-aware bump sampler, so scoped
  inverse replay no longer diverges from forward shading for authored bump
  sampler policy.
- pt-webgpu upload warnings no longer classify bump sampler policy as
  approximate, and warning-propagation tests pin that no structured sampler
  approximation is emitted for this case.

Focused gate:

```bash
npm test --workspace @vitrum/pt-webgpu -- materialTextureArray.test.ts materialTextures.test.ts adjointHarness.test.ts uploadSceneWarnings.test.ts wgslContract.test.ts
```

## Parked Long-Tail Items

These are real, but they should not block contract-complete unless the user
explicitly widens the target.

Current source-verification pass (2026-06-22, after Wave 12): the active runtime
implementation queue is empty. The only source-verified "code gap" phrasing left
in the Road is native promotion or future contract expansion, not broken
renderability/API behavior:

Follow-up implementation narrowing (2026-06-21): while checking the inverse
classifier tails, two bounded false downgrades were code-closed in
`packages/pt-webgpu/src/inverse/inverseSession.ts` and pinned by
`inverseSession.test.ts`: readable all-opaque base-color/standalone alpha
coverage plus stably-opaque `mask` scalars now remain on path replay, and
positive authored `transmission` is treated as dormant when a readable
`transmissionMap` has zero R coverage everywhere. Active fractional alpha,
positive transmission coverage, layered/volume/spectral transport, and true
visibility/transport derivatives remain finite-difference/proof tails.

- pt-webgl2 dimension-changing topology/list edits are supported through
  resident-storage `fallback-rebuild` with structured diagnostics; true targeted
  primitive splice/refit remains a native-mutation promotion task.
- glTF skinned/morphed `EXT_mesh_gpu_instancing` is renderable through
  fallback-expanded skinned/morphed primitives; native instanced-skinned
  primitives remain a performance/core-contract expansion.
- glTF high UV sets are supported for `TEXCOORD_1` plus a single lossless high-UV
  remap into `uv1`; arbitrary UV arrays remain a future core/backend contract.
- glTF `COLOR_1+` and native point/line topologies remain explicit diagnostics
  or generated-mesh fallbacks, not silent drops.

- Production neural denoiser checkpoint and default-on `denoiser:'auto'`.
- NRC default-on decision after quality/convergence A/B.
- GRIS default flip after unbiasedness/error/perf evidence.
- Low-discrepancy sampling / PMJ / Sobol convergence upgrade.
- Compressed wide BVH / CWBVH traversal.
- Wavefront path tracing.
- Heterogeneous volumes / NanoVDB-class media.
- Native point/line primitive contract.
- Native arbitrary UV-array contract.
- True instanced-skinned/morphed primitive contract.
- Real displacement tessellation or geometry displacement pipeline.

## Validation Queue

These are not code blockers, but they are required before a high-confidence
"100%" signoff.

WSL smoke status (2026-06-21): direct lavapipe and dzn T1 smoke both pass,
including the Cornell DDGI non-regression capture plus RC/ReSTIR-TLAS/DDGI
brute-force GPU oracles. The dzn direct run completed in about 210 seconds, so
the old 120-second `npm run validate:gpu:smoke` wrapper timeout was too short
for the current hybrid-capture path. `scripts/validate-gpu.mjs` now defaults the
smoke timeout to 300 seconds, preserving a bounded pre-push gate without
misclassifying a slow dzn PASS as a validation failure. Keep full
browser/real-adapter validation in the queue; do not treat T1 smoke as a
replacement for V28-B recaptures.

- V28-B render-changing recaptures for PT and walkaround changes.
- GRIS-on unbiasedness and biased-default error quantification.
- PPG favorable-scene A/B.
- NRC quality/convergence A/B.
- Neural denoiser quality A/B once production weights exist.
- Baseline/lite/spectral/skinned/analytic dzn execution proof. Focused dzn
  status artifacts now cover `pt/default`, `wh/default`, `pt/lite-tier`,
  `pt/spectral`, `pt/spectral+photon`, `pt/spectral+bdpt`, `pt/skinned-mesh`,
  `pt/gltf-skinned-animation`, `wh/skinned-mesh`, and `pt/analytic-sphere` with
  finite non-black output and zero GPU errors. The status checker now derives
  the real behavioral-gate label inventory from `gate.mjs` and fails if any
  non-self-test lane lacks committed dzn status coverage. The analytic lane also
  fixed a stale behavioral fixture that used an object-shaped sphere instead of
  the core `AnalyticPrimitive` contract. Remaining work is reference/radiometric
  promotion, not boot/render validity.
- BDPT material/radiometric A/B. Full-tier dzn boot/render proof now covers
  `pt/bdpt` and `pt/spectral+bdpt` through
  `behavioral-gate:dzn -- --filter bdpt --require-full-tier`; material furnace,
  multi-vertex radiometry, and broader promotion evidence remain.
- ReSTIR-PT reuse A/B. Full-tier dzn boot/render proof now covers
  `pt/restirPtReuse` through
  `behavioral-gate:dzn -- --filter restirPtReuse --require-full-tier`; the
  repaired-Cornell default equal-spp A/B is now recaptured and PASS
  (`globalRelErr=7.91%`, `varRatio=0.9403`) after restricting default reuse to
  diffuse-safe visible vertices. Glossy/metallic visible-vertex reuse remains
  available only through `restirPtReuseOptions.experimentalGlossyReuse:true` and
  still needs research-mode promotion evidence.
- SPPM/MNEE caustic A/B. Full-tier dzn boot/render proof now covers
  `pt/caustic-manifold`, `pt/caustic-photon`, and `pt/spectral+photon` through
  the focused `caustic` and `photon` dzn status artifacts; the SPPM-vs-MNEE
  radiometric A/B is now freshly recaptured on dzn full-tier and passes the
  committed loose convergence proof (`finalRelErr=23.4%`, threshold <500%).
  Tighter/equal-quality caustic promotion remains future evidence work.
- Analytic emitter/environment render proof. Focused dzn status artifacts now
  cover pt-webgpu point/disc/spot/directional emitters, pt-webgpu HDRI and
  procedural-sky environments, the matching lite point/HDRI rows, and walkaround
  directional-sun/HDRI rows. Reference-quality radiometric sweeps remain.
- Walkaround behavioral dzn proof. The old broad `--filter wh/` aggregate now
  times out on this WSL/dzn host before all rows finish, so the durable proof
  is sharded per row: `wh/default`, `wh/rcEnabled`, `wh/ppgEnabled`,
  `wh/gtao-off`, `wh/checkerboard`, `wh/skinned-mesh`, `wh/hdri-env`,
  `wh/rect-area-emitter`, `wh/directional-sun`, `wh/glass-gi`, and
  `wh/transparent-oit` each have a committed PASS status artifact with zero GPU
  errors. The walkaround mutation shard
  (`behavioral-gate:dzn -- --filter wh/mutation --require-full-tier`) now also
  proves material, transform, and emitter mutation rows with visible pixel
  deltas and zero GPU validation errors. The dzn status checker verifies those
  shard artifacts instead of requiring the fragile monolithic run. The original broad run had found and fixed an RC-only validation bug: the
  RC material atlas/meta bindings were declared filterable even though their
  `rgba32float` textures are unfilterable and read only with `textureLoad`.
- pt-webgpu full-tier material-furnace/reference-render sweeps. WSL lavapipe is
  adapter-limited for this lane (`maxStorageBuffersPerShaderStage=8`; full tier
  requires 34); use `behavioral-gate:dzn`, browser, or another real/full-tier
  adapter validation lane.
- Real glTF golden PNG sweep on recommended backends (pt-webgpu WSL public-asset
  lavapipe golden lane is covered; dzn full-tier execution now passes all 11
  glTF lanes against explicit `dzn-full` real-asset goldens; the pt-webgl2
  browser lane now has fail-closed Playwright status for real BoxTextured,
  CesiumMilkTruck Draco, and MeshoptCubeTest meshopt construction/texture
  readiness, including browser Draco/meshopt decoder-hook telemetry, but WSL
  canvas readback stalls before PNGs can be captured).
- Mutation matrix on real GPU/browser harness: pt-webgpu material/environment/
  emitter/transform/topology/instanced-count/add-primitive/remove-primitive mutation lanes now pass on the default WSL runtime and dzn full tier
  (`behavioral-gate:dzn -- --filter mutation --require-full-tier`) after fixing
  the primitive-less full-tier BVH/TLAS placeholder buffers. The dzn mutation
  status artifact is committed at
  `tools/behavioral-gate/behavioral-gate-dzn-mutation-status.json` and checked by
  `npm run behavioral-gate:dzn-status-check`. The pt-webgpu cached-bind-group
  invalidation seam for reallocating mutation fast paths is now source/test
  pinned in `packages/pt-webgpu/src/__tests__/mutationDesyncs.test.ts`
  (topology resize and instanced-count changes invalidate before commit/reset).
  Walkaround denoiser-history reset is now source/shader pinned: frame-zero
  mutation resets drive Welford `forceReset`, BMFR `hasHistory=0`, and
  SVGF-real reprojection `forceReset` through the old UBO pad slot, with
  focused shared-denoiser and walkaround tests. Walkaround material-only
  mutations now also refresh DDGI's `RestirBvhSnapshot` material payload without
  RC geometry propagation, and roughness/metallic edits invalidate DDGI probe
  cache because probe rays consume those fields. Walkaround material/transform/
  emitter mutation rows now have a focused dzn proof at
  `tools/behavioral-gate/behavioral-gate-dzn-wh-mutation-status.json`; that run
  also closed the DDGI no-TLAS placeholder-buffer bug in `rebuildProbeBvhFromRestir`.
  Broader cross-backend/browser mutation-matrix promotion remains.

## How To Use This Plan

1. Start at Wave 0.
2. Before editing a task, revalidate the cited source and write a one-line
   "still open" note in the work log or commit message.
3. If the source no longer matches, mark the task stale and do not edit.
4. Keep each wave small enough to review.
5. After each wave, update this file only if the active queue changes.

The main discipline: close confirmed gaps, then stop. Do not let newly noticed
research tails reset the finish line.

## Appendix A — Candidate Backlog From Cursor

This appendix preserves the useful breadth of Cursor's larger plan without
letting it become the execution driver. Treat every item here as **candidate
work**: re-open the cited source on current `main`, prove the gap still exists,
then either promote it into the active queue above or mark it stale/parked.

The original Cursor bundle was generated from a moving 2026-06-16 code snapshot.
Some items have already been closed by later commits; others are validation or
post-contract work. Do not execute this appendix top-to-bottom.

### A1 — Promote To Active Queue If Still Open

These are plausible implementation gaps or truthfulness gaps. They are smaller
than the parked SOTA work and should be checked soon.

Status 2026-06-17: **AUDITED/CLOSED.** Every A1 candidate has been revalidated
against current source and classified below. No additional bounded
implementation item survived this pass beyond VQ-001 through VQ-004; remaining
work is validation-deferred, explicitly decided, or parked future contract/SOTA
scope.

| Candidate | Area | Revalidate By | Disposition |
|-----------|------|---------------|-------------|
| Lite material mutation fast path | pt-webgpu | `sceneMutationRouter.ts`; lite material patch tests | **CLOSED 2026-06-17.** VQ-001 landed; lite material patches now write the material buffer and advertise `native`. |
| Mesh-light repack uses stale geometry/material pack | pt-webgl2 | `mutateSceneTextures.ts`; mesh-area light mutation tests | **CLOSED 2026-06-17.** VQ-002 landed; mesh-light repack uses `nextGeoPack` with a radiance regression. |
| Public `denoiser:'none'` mismatch | walkaround | `HybridEngineOptions.ts`; denoiser registry | **CLOSED 2026-06-17.** VQ-003 landed; `none` is a supported pass-through mode. |
| DDGI material snapshot/invalidation for material edits | walkaround | `HybridEnginePrimitiveUpdates.ts`; mutation matrix | **CLOSED 2026-06-18.** VQ-004 follow-up landed; material-slice edits refresh DDGI's snapshot without RC geometry propagation, and radiance/visibility plus roughness/metallic changes invalidate DDGI probes. |
| `supportsAuxBuffers` truthfulness | core / backends | `promiseLedger.ts`; backend capabilities; frame outputs | **CLOSED BY EXISTING SOURCE/TESTS.** Ledger records full-tier WebGL2/pt-webgpu aux support and walkaround `false`; lite downgrades are runtime-profile rows. Pinned by `ledgerVsCapabilities` and backend promise tests. |
| pt-webgl2 behavioral gate coverage | tools | `tools/behavioral-gate/gate.mjs`; available WebGL2 harness | **VALIDATION-DEFERRED.** Current behavioral gate is WebGPU/Deno-lavapipe oriented. pt-webgl2 has package behavior tests plus GLSL gate; real WebGL2/browser render proof stays in the validation queue. |
| pt-webgl2 procedural/fog/background feature flags | pt-webgl2 | `featureTypes.ts`; active GLSL defines | **CLOSED BY EXISTING SOURCE/TESTS.** `featureTypes.ts` pins fog/backgroundMap/random/debug to safe defaults with explicit rationale; upload/compose tests cover the flags and procedural-sky environment bake. |
| pt-webgl2 mutation capability overclaim | pt-webgl2/core | `PT_WEBGL2_MUTATIONS`; `incrementalPatchSupport`; mutation tests | **CLOSED BY EXISTING SOURCE/TESTS.** Boolean incremental support means the method absorbs the patch; `supportDetails.mutations` distinguishes `native` material/emitter/env/resize from fallback-rebuild geometry rows. |
| pt-webgpu API asymmetry (`setSize`, `updateLighting`) | pt-webgpu/core | backend class + `EngineCapabilities` | **DECIDED/TRUTHFUL.** Optional methods remain absent; ledger and core tests explicitly record `setSize:false`, `updateLighting:false`. No facade shim should imply unsupported behavior. |
| Engine adapter-profile reporting parity | engine | `createEngineInternals.ts`; backend wrappers | **DECIDED/TRUTHFUL.** `onAdapterProfile` is documented as walkaround-only in `CreateEngineOptions`; glTF strict pt-webgpu paths probe runtime lite/full profile separately before construction. Broadening this is a future API decision, not a hidden gap. |
| `VitrumCanvas` advanced backend options parity | engine/react | `VitrumCanvas.tsx`; `createEngine` options | **CLOSED BY EXISTING SOURCE/TESTS.** `VitrumCanvas` forwards `advanced` through both direct `attachVitrum` and glTF-created `engineOptions`; mount tests cover forwarding. |
| Progressive fallback option correctness | engine | `createProgressiveEngine.ts`; tests | **CLOSED BY EXISTING SOURCE.** Progressive has explicit `realtimeOptions` and `convergedOptions` bags; there is no backend fallback path in this Class-A shared-device facade. |
| Debug/proxy post-dispose behavior | engine | `idempotentDispose.ts`; proxy-table tests | **DECIDED/TRUTHFUL.** Debug is a property forwarded by object spread, not a proxied callable. Proxy tests pin post-dispose behavior for callable optional methods and the live debug-surface behavior. |
| glTF ignored-camera diagnostics | glTF adapter | `featureReport.ts`; `gltfToScene.ts`; diagnostics tests | **CLOSED BY EXISTING SOURCE/TESTS.** Import diagnostics and compatibility issues carry `cameras[0]`; strict modes reject only degraded policy as intended. |
| glTF texture-source provenance | glTF adapter | `texturePipeline.ts`; feature report | **CLOSED BY EXISTING SOURCE/TESTS.** Texture decode report entries carry material field, texture/image indices, alternate texture-source extension metadata, and source paths. |
| Spec-gloss alpha-to-roughness bake | glTF adapter | `materials.ts`; spec-gloss tests | **CLOSED BY SOURCE/TESTS.** Decode path bakes glossiness alpha into a CPU-linear roughness map; strict mode no longer reports the alpha degradation only after every authored spec-gloss texture path has a matching baked linear `roughnessMap`, so unrelated roughness maps cannot satisfy the legacy alpha issue. |
| High-UV remap edge cases | glTF adapter | `gltfToScene.ts`; texture sweep tests | **CLOSED/NARROWED.** Single high `TEXCOORD_N` remaps to `uv1` when lossless; conflict/missing-accessor diagnostics remain. Native arbitrary UV arrays are parked as a future contract. |
| Skinned/morphed instancing fallback expansion | glTF adapter | `featureReport.ts`; `gltfToScene.ts`; `sceneController.ts` | **CLOSED BY SOURCE/TESTS.** `EXT_mesh_gpu_instancing` on skinned/morphed primitives fallback-expands into one skinned/morphed primitive per authored instance transform with `fallback-expanded-gpu-instancing`; `reject-unsupported` accepts the renderable fallback and `reject-degraded` rejects it for hosts that require native instanced skinning. True native instanced-skinned/morphed primitives remain future performance/core-contract work. |
| Emissive-map texel-PDF warning | glTF adapter / PT | `featureReport.ts`; emitter packing tests | **CLOSED/TRUTHFUL.** Compatibility emits `emissiveMap.texelPdf=approximate` on applicable non-lite profiles; exact texel-PDF emitter sampling remains proof/promotion tail. |

### A2 — Proof / Promotion Backlog

These should not block code waves unless the implementation is missing. They are
for promotion from "implemented/approximate" to "trusted/native".

| Candidate | Area | Required Evidence |
|-----------|------|-------------------|
| V28-B recaptures | all render-changing lanes | Before/after captures with deltas attributed to intended changes. |
| GRIS-on unbiasedness | walkaround ReSTIR-GI | Converged reference comparison and biased-default error quantification. |
| Walkaround radiometric A/B harness | walkaround/tools | **UPDATED 2026-06-21.** The SUN fixture now uses the directional-light radiance baseline `Lo = I * cos(theta) * albedo / pi`, runs as diffuse-only, and disables sky/GTAO/denoising for that analytic case. `VITRUM_WALKAROUND_AB_CASES` can rerun selected cases while preserving the other committed results. Current WSL native-Deno validation completes and records `PASS-PARTIAL`: A8 is `NEGLIGIBLE`, SUN is `PASS`, GLASS is `PASS`, and GLOSSY remains a material-effect `FINDING`; rerun on browser/real-adapter or higher-quality case-specific references before promotion. |
| PPG quality | walkaround PPG | Favorable-scene A/B showing convergence or variance win. |
| NRC quality/default tier | walkaround NRC | Quality/convergence A/B after warm-up gate; decide default/off/experimental. |
| Neural denoiser quality | shared/walkaround | **Checkpoint classification guard added 2026-06-21:** every committed `.vitrum-model` is now manifest-pinned by role/size/SHA/param count and `learned-systems-proof-check` fails on unregistered or production-like weights without a passing production A/B manifest. Remaining work is still the actual production checkpoint plus quality A/B; otherwise keep opt-in. |
| BDPT / ReSTIR-PT material and radiometric proof | pt-webgpu/pt-webgl2 | Safe-default BDPT, SPPM, and ReSTIR-PT committed snapshots are checked by `npm run radiometric-ab:proof-check`; `npm run radiometric-ab:pt-host-status` now recaptures all three native-Deno full-tier A/B cases and records a committed `PASS` host-status artifact. Focused dzn full-tier behavioral status separately proves `pt/bdpt`, `pt/spectral+bdpt`, and off-default `pt/restirPtReuse` boot/render finite non-black with zero GPU errors. The repaired-Cornell ReSTIR-PT equal-spp default A/B passes (`globalRelErr=7.91%`, `varRatio=0.9403`) after moving glossy/metal visible-vertex reuse behind `experimentalGlossyReuse`. The ReSTIR-PT specialty fixture pins scalar plus map-backed-effective clearcoat/sheen/iridescence/aniso/specular one-sample producer/finalize/resolve identity. The opt-in multi-vertex BDPT branch is now a structured non-promotable research finding (`promotionReady:false`, blocker `not-weighted-against-regular-eye-path-strategy`) tied to `results-bdpt.json`. Remaining work is GPU/radiometric material-furnace promotion and a redesigned multi-vertex BDPT estimator weighted against the regular eye-path strategy. |
| SPPM / MNEE caustic radiometric proof | pt-webgpu | Focused dzn full-tier behavioral status proves `pt/caustic-manifold`, `pt/caustic-photon`, and `pt/spectral+photon` boot/render finite non-black with zero GPU errors. The committed SPPM-vs-MNEE radiometric A/B is freshly recaptured and passes the loose convergence proof (`finalRelErr=23.4%`, trend improving by 80 frames). Remaining work is tighter/equal-quality caustic promotion evidence, not baseline proof existence. |
| Baseline/lite/spectral/skinned/analytic execution | pt-webgpu/walkaround | Focused dzn status now proves default pt/walkaround, explicit pt-webgpu lite fallback, spectral combos, skinned/glTF-skinned animation, and full-tier analytic sphere lanes boot/render finite non-black with zero GPU errors. The dzn status checker now fails if any real behavioral-gate label lacks committed coverage. Remaining work is reference-quality or radiometric promotion where applicable. |
| Analytic emitter/environment proof | pt-webgpu/walkaround | Focused dzn status now proves point/disc/spot/directional, HDRI, and procedural-sky lanes boot/render finite non-black with zero GPU errors on their selected full/lite/walkaround rows. Remaining work is reference-quality radiometric sweep coverage. |
| Walkaround behavioral matrix | walkaround | The broad `behavioral-gate:dzn -- --filter wh/ --require-full-tier` aggregate is host-blocked on this WSL/dzn runner, but each walkaround row now has an individual committed PASS status artifact (`wh/default`, `wh/rcEnabled`, `wh/ppgEnabled`, `wh/gtao-off`, `wh/checkerboard`, `wh/skinned-mesh`, `wh/hdri-env`, `wh/rect-area-emitter`, `wh/directional-sun`, `wh/glass-gi`, `wh/transparent-oit`) with zero GPU errors. The focused `wh/mutation` dzn shard additionally proves material/transform/emitter mutation rows with visible deltas and zero GPU errors, and `npm run behavioral-gate:dzn-status-check` verifies all of those shards. Remaining work is A/B quality proof, not boot/render validity. |
| pt-webgpu full-tier material furnace | pt-webgpu | Scalar and CPU-readable map-backed clearcoat/sheen/iridescence/aniso/specular full-tier captures are now covered by `pt/material-lobes` and `pt/material-lobe-maps` dzn golden proof, and the ReSTIR-PT specialty fixture pins the matching scalar/map-backed-effective reservoir identity. Remaining work is GPU/radiometric A/B before promotion. |
| pt-webgl2 material furnace | pt-webgl2 | Source/oracle proof is guarded by `npm run renderer-fidelity-proof-check`: GGX white furnace, thickness/SSS packing and shader consumption, procedural-sky bake into the HDRI/CDF path, and emissive-map mesh-area MIS all have source/test needles. Remaining work is browser/real-adapter reference A/B before any supported-row promotion. |
| Rich-material GI | walkaround | A/B showing receiver-lobe material target improves or preserves correctness. |
| Transparent OIT visual proof | walkaround | `wh/transparent-oit` now has a committed `dzn-full` PNG golden plus a dzn PASS status proving a fractional alpha-blend pane with sun, point-light, and finite-area lighting against that golden, while preserving the approximation warning. Remaining work is reference-quality A/B proof for alpha shadow transmittance and layered-transport boundaries. |
| Real glTF golden sweep | glTF/tools | pt-webgpu WSL public-asset golden lane covered; recommended-backend/browser import/decode readiness is proven for textured GLB, Draco, and meshopt rows. The pt-webgl2 browser harness now defaults to paused public `engine.captureFrame({ colorSpace:'output' })` readback, fail-closes WSL readback timeouts quickly with an `engine-captureFrame-output` attempt in every `HOST-BLOCKED` row, keeps locator screenshot / clipped screenshot / canvas data URL fallbacks for explicit modes, and supports opt-in Chromium GL/ANGLE args via `VITRUM_CHROMIUM_EXTRA_ARGS`; this WSL Playwright host still blocks on pixel readback, so PNG goldens/tolerance remain a browser-host validation item, not a source-code import gap. |
| Mutation matrix on real GPU | engine/backends | pt-webgpu mutation rows and walkaround material/transform/emitter rows now have committed dzn full-tier status artifacts. Remaining work is browser/adapter promotion plus any additional backend-specific mutation rows not covered by those shards. |
| Browser/adapter coverage | tools | Browser or real-adapter validation for rows WSL lavapipe cannot prove. |

### A3 — Product Decisions

These need a call, not blind implementation.

| Decision | Default Recommendation | Why |
|----------|------------------------|-----|
| NRC default tier | Keep opt-in until A/B proves quality/perf | It is biased/learned and scene-dependent. |
| Neural denoiser shipping | Do not advertise production default until checkpoint exists | Code path is not the same as a production model. |
| GRIS default flip | Keep biased realtime default until validation says otherwise | Unbiased path has cost; decision should be evidence-based. |
| pt-webgl2 caustic naming | Keep approximate wording unless true MNEE/SPPM parity lands | Avoid claiming algorithmic equivalence. |
| Walkaround transparent transport | Keep approximate/unsupported for true layered transport | OIT is not full reservoir/GI participation. |
| Displacement support | Keep approximate vertex-displacement wording unless tessellation/microdisplacement lands | CPU-readable maps are applied before BVH construction, but no new geometry is synthesized. |
| Native point/line primitives | Keep generated-mesh fallback | Good enough for arbitrary glTF routing; native contract is a new feature. |
| Arbitrary UV arrays | Keep narrow remap + diagnostics | Native array support touches core/backend contracts. |
| Instanced skinned/morphed glTF | Keep fallback-expanded primitives; do not claim native instanced skinning | Arbitrary glTF is renderable through generated skinned/morphed primitives today; a first-class instanced-skinned primitive remains a new performance/core-contract feature. |

### A4 — Parked SOTA / Performance Track

Do not let these block contract-complete. They are useful after the current
closure campaign.

| Candidate | Area | Why Parked |
|-----------|------|------------|
| Low-discrepancy sampling / PMJ / Sobol | shared samplers + PT backends | Convergence upgrade, not contract correctness. |
| Compressed wide BVH / CWBVH | shared-bvh + backends | Throughput architecture project. |
| Wavefront path tracing | pt-webgpu | Major rearchitecture; only if profiling demands. |
| Heterogeneous volumes | core + pt-webgpu | New primitive/medium contract. |
| Production OIDN/UNet assets | shared denoisers | Provisioning/licensing/model-quality project. |
| FSR2-class temporal upscaling | tools/backends | Performance/product feature, not API correctness. |
| SHaRC/radiance-cache experiments | walkaround/PT | Research/SOTA bridge, not current closeout. |
| Shadow-map NEE assist | hybrid/PT | Optimization with new raster/PT coupling. |
| Subgroup/f16 traversal compaction | PT/shared-bvh | Adapter-tier performance feature. |

### A5 — Stale-Or-Likely-Closed Cursor Items

Check these only if they reappear in a failing test or stale doc. The current
Road suggests they are already closed or narrowed.

- glTF point/line modes: now fallback-generated mesh with diagnostics.
- Draco/meshopt hook smoke tests: real codec smoke coverage exists.
- Runtime pt-webgpu profile selection: strict modes handle full vs lite.
- glTF texture decode report provenance: decoded dimensions/source provenance
  now present.
- Spec-gloss alpha roughness issue: direct asset callers now bake/remove the
  compatibility issue.
- Walkaround point/spot/direct emitter cast-shadow rows: current Road marks
  native/closed.
- RC UV1 emissive atlas fallback: closed by RC probe material sampling tests that
  pin UV1 atlas metadata and emissive-map probe response.
- ReGIR mapped emitter target: closed for the bounded walkaround emitter path;
  mapped emitter sampling has direct shader/source coverage, while global
  GI/RC/DDGI exact texel-PDF promotion stays in proof.
- Soft-sun adjoint: closed for scoped direct-light replay by the directional-cone
  sampling oracle; indirect/transport adjoints remain finite-difference lanes.
- Alpha shadow/transmittance in RC/walkaround: closed for the bounded
  approximation (transparent shadows and DDGI/visibility tint tests), not true
  layered transparent transport.

### A6 — Backlog Promotion Rule

Move an appendix item into the active queue only when all are true:

1. The gap is still visible in current source.
2. The expected behavior is clear.
3. The edit scope is bounded to a small set of files.
4. A focused regression test can be written before or with the fix.
5. It advances contract-complete, not just SOTA polish.

If any condition fails, keep it in `PROOF`, `DECIDE`, or `PARK`.
