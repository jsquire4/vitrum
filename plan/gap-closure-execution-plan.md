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
- Current evidence:
  - `sceneMutationRouter.ts` still has:
    `if (host.isLiteTier?.() === true && canFastPathMaterialPatch(...)) { host.setScene(nextScene); return; }`
- Desired behavior:
  - Lite material-only patches write the lite material buffer through the existing
    material fast path, or explicitly prove the lite path cannot safely patch and
    emit a truthful unsupported diagnostic.
- Minimum tests:
  - Add/confirm a lite-tier material-only patch test that observes
    `queue.writeBuffer` to the material buffer and no `setScene()` fallback.
  - `cd packages/pt-webgpu && npx vitest run src/__tests__/updatePrimitiveIncremental.test.ts`
  - root `npm run typecheck`

### VQ-002 — pt-webgl2 mesh-light material mutation must use `nextGeoPack`

- Label: **CLOSE-NOW**
- Status: **CLOSED 2026-06-17** — mesh-light repack now uses `nextGeoPack`, with
  a regression test covering changed emissive material radiance.
- Files:
  - `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`
  - `packages/pt-webgl2/src/scene/meshAreaLights.test.ts` or nearby mutation test
- Current evidence:
  - In `tryFastPathMaterialMutation`, one mesh-light repack path still calls
    `packMeshAreaLights(nextScene, geoPack)` after creating `nextGeoPack`.
- Desired behavior:
  - Mesh-area light repack sees the updated material table.
- Minimum tests:
  - Regression where an emissive/material mutation changes mesh-light radiance and
    the packed light texture observes the new material state.
  - `cd packages/pt-webgl2 && npx vitest run src/scene/meshAreaLights.test.ts`
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
- Current evidence:
  - `pipeline/denoisers/none.ts` exists.
  - `VALID_DENOISERS` omits `'none'`.
- Desired behavior:
  - Either `'none'` is a supported public denoiser option, or the unused denoiser
    implementation is made internal/dead. Prefer support because the pipeline
    already has the mode.
- Minimum tests:
  - Construction/config accepts `denoiser:'none'`.
  - Denoiser registry routes to the no-op adapter.
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
- Current evidence:
  - Material patch invalidates DDGI for emitter-affecting changes
    (transmission-threshold/emissive radiance), but beer/tint fields that change
    probe lighting are not clearly covered.
- Desired behavior:
  - Material-only patches refresh DDGI's material snapshot without forcing
    geometry subsystem rebuilds.
  - Material-only patches that affect DDGI radiance, visibility, or the glossy
    probe BRDF invalidate the probe cache.
- Fields covered:
  - Scalar slice path: `attenuationColor`, `attenuationDistance`, `thickness`,
    `transmission`, Beer/tint-relevant `baseColor`, `roughness`, and
    `metallic`.
  - Atlas-backed maps still route through the full rebuild path when map handles
    or atlas metadata change.
- Tests:
  - `attenuationDistance`-only patch calls `invalidateProbeCache()`.
  - Roughness-only scalar patch refreshes DDGI snapshot and invalidates probes
    without emitter rebuild.
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

## Parked Long-Tail Items

These are real, but they should not block contract-complete unless the user
explicitly widens the target.

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

WSL smoke status (2026-06-17): `npm run validate:gpu:smoke` completed on both
lavapipe and dzn. The prior dzn hybrid-capture timeout did not reproduce in this
run; both backend captures passed non-regression, the dzn/lavapipe cross-check
passed, and the RC/ReSTIR/DDGI BVH brute-force oracles passed on lavapipe. Keep
full browser/real-adapter validation in the queue; do not treat this smoke as a
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
  `behavioral-gate:dzn -- --filter restirPtReuse --require-full-tier`; equal-spp
  variance reduction and specialty/radiometric promotion remain.
- SPPM/MNEE caustic A/B. Full-tier dzn boot/render proof now covers
  `pt/caustic-manifold`, `pt/caustic-photon`, and `pt/spectral+photon` through
  the focused `caustic` and `photon` dzn status artifacts; caustic radiometric
  convergence remains.
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
  errors. The dzn status checker
  verifies those shard artifacts instead of requiring the fragile monolithic
  run. The original broad run had found and fixed an RC-only validation bug: the
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
  cache because probe rays consume those fields. Broader combined real-adapter
  mutation-matrix promotion remains.

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
| PPG quality | walkaround PPG | Favorable-scene A/B showing convergence or variance win. |
| NRC quality/default tier | walkaround NRC | Quality/convergence A/B after warm-up gate; decide default/off/experimental. |
| Neural denoiser quality | shared/walkaround | Production checkpoint plus quality A/B; otherwise keep opt-in. |
| BDPT / ReSTIR-PT material and radiometric proof | pt-webgpu/pt-webgl2 | Safe-default BDPT, SPPM, and ReSTIR-PT committed snapshots are checked by `npm run radiometric-ab:proof-check`; focused dzn full-tier behavioral status now proves `pt/bdpt`, `pt/spectral+bdpt`, and off-default `pt/restirPtReuse` boot/render finite non-black with zero GPU errors. The ReSTIR-PT specialty fixture now pins scalar plus map-backed-effective clearcoat/sheen/iridescence/aniso/specular one-sample producer/finalize/resolve identity. Remaining work is GPU/radiometric material-furnace promotion, equal-spp ReSTIR-PT variance proof, and multi-vertex BDPT promotion evidence. |
| SPPM / MNEE caustic radiometric proof | pt-webgpu | Focused dzn full-tier behavioral status now proves `pt/caustic-manifold`, `pt/caustic-photon`, and `pt/spectral+photon` boot/render finite non-black with zero GPU errors. Remaining work is caustic radiometric convergence / forward-traced oracle A/B. |
| Baseline/lite/spectral/skinned/analytic execution | pt-webgpu/walkaround | Focused dzn status now proves default pt/walkaround, explicit pt-webgpu lite fallback, spectral combos, skinned/glTF-skinned animation, and full-tier analytic sphere lanes boot/render finite non-black with zero GPU errors. The dzn status checker now fails if any real behavioral-gate label lacks committed coverage. Remaining work is reference-quality or radiometric promotion where applicable. |
| Analytic emitter/environment proof | pt-webgpu/walkaround | Focused dzn status now proves point/disc/spot/directional, HDRI, and procedural-sky lanes boot/render finite non-black with zero GPU errors on their selected full/lite/walkaround rows. Remaining work is reference-quality radiometric sweep coverage. |
| Walkaround behavioral matrix | walkaround | The broad `behavioral-gate:dzn -- --filter wh/ --require-full-tier` aggregate is host-blocked on this WSL/dzn runner, but each walkaround row now has an individual committed PASS status artifact (`wh/default`, `wh/rcEnabled`, `wh/ppgEnabled`, `wh/gtao-off`, `wh/checkerboard`, `wh/skinned-mesh`, `wh/hdri-env`, `wh/rect-area-emitter`, `wh/directional-sun`, `wh/glass-gi`, `wh/transparent-oit`) with zero GPU errors, and `npm run behavioral-gate:dzn-status-check` verifies those shards. Remaining work is A/B quality proof, not boot/render validity. |
| pt-webgpu full-tier material furnace | pt-webgpu | Scalar and CPU-readable map-backed clearcoat/sheen/iridescence/aniso/specular full-tier captures are now covered by `pt/material-lobes` and `pt/material-lobe-maps` dzn golden proof, and the ReSTIR-PT specialty fixture pins the matching scalar/map-backed-effective reservoir identity. Remaining work is GPU/radiometric A/B before promotion. |
| pt-webgl2 material furnace | pt-webgl2 | Thickness/SSS/procedural-sky/emissive panels against references. |
| Rich-material GI | walkaround | A/B showing receiver-lobe material target improves or preserves correctness. |
| Transparent OIT visual proof | walkaround | `wh/transparent-oit` now has a committed `dzn-full` PNG golden plus a dzn PASS status proving a fractional alpha-blend pane with sun, point-light, and finite-area lighting against that golden, while preserving the approximation warning. Remaining work is reference-quality A/B proof for alpha shadow transmittance and layered-transport boundaries. |
| Real glTF golden sweep | glTF/tools | pt-webgpu WSL public-asset golden lane covered; recommended-backend/browser render, mean luminance, no GPU errors, and PNG tolerance remain for final promotion. |
| Mutation matrix on real GPU | engine/backends | Real buffers, bind groups, denoiser history, GI propagation together. |
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
- RC UV1 emissive atlas fallback: recent commits indicate closure.
- ReGIR mapped emitter target: recent commits indicate closure.
- Soft-sun adjoint: recent commits indicate closure for scoped direct-light
  replay.
- Alpha shadow/transmittance in RC/walkaround: recent commits indicate closure
  of the bounded approximation, not true layered transport.

### A6 — Backlog Promotion Rule

Move an appendix item into the active queue only when all are true:

1. The gap is still visible in current source.
2. The expected behavior is clear.
3. The edit scope is bounded to a small set of files.
4. A focused regression test can be written before or with the fix.
5. It advances contract-complete, not just SOTA polish.

If any condition fails, keep it in `PROOF`, `DECIDE`, or `PARK`.
