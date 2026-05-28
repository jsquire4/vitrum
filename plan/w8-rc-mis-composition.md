# W8 — RC + MIS composition in HybridEngine

**Status:** Shipped (2026-05-18, all four phases). Trilinear interpolation in sampleCascadeC0 is a Phase-3b follow-up (nearest-probe only today).
**Acceptance:** items_to_fix.md B2 — HybridEngine runs a Cornell scene with `denoiser: 'atrous-variance'`, `rcEnabled: true`, and visibly produces higher-quality first-bounce indirect than DDGI-only.

## Why this is bigger than the audit said

`items_to_fix.md` B2 framed this as "wire RC into the combined shading sum." Reading the code on 2026-05-18 shows it's deeper:

- **RC's current design injects GI into THREE materials via TSL nodes.** `giReceiver.ts` wraps receiver meshes with `MeshPhysicalNodeMaterial` and a custom `lightingNode` built from `walkaroundDiffuseLighting.ts`. Neither path is reachable from a WGSL shade pass.
- **HybridEngine's shade pass is pure WGSL.** `shade.wgsl` reads bind-group resources by index; it has no concept of a TSL-wrapped material.
- **RC's dispatch path still imports THREE.** `cascadeDispatch.ts` takes a `WebGPURenderer`, uses `THREE.Vector3` / `THREE.Color` / `THREE.Texture`, and reaches into `renderer.backend.get(attribute)` to extract raw GPUBuffers.
- **Three GI sources overlap.** DDGI's probe atlas, ReSTIR-GI's reservoir, and RC's cascade pyramid all estimate the same `L_indirect`. Adding them naively triple-counts.

So "wire RC into HybridEngine" splits into four phases:

## Phase 1A — Strip THREE coupling from RC core (data + buffer wrappers)

**Files:**
- `packages/walkaround-hybrid/src/rc/cascadePyramid.ts` — replace `THREE.Box3` / `THREE.Vector3` with plain `{ min: [x,y,z], max: [x,y,z] }` / `[x,y,z]` types.
- `packages/walkaround-hybrid/src/rc/cascadeBuffers.ts` — drop `THREE.Box3` cache, use the plain shape above.

**Acceptance:** typecheck clean; existing `rc-bindings.test.ts` + `cascadeMergeWeights.test.ts` + `rcSolidAngles.test.ts` still pass.

## Phase 1B — Strip THREE coupling from RC dispatch

**Files:**
- `packages/walkaround-hybrid/src/rc/cascadeDispatch.ts` — replace `WebGPURenderer` constructor param with a raw `GPUDevice`; replace the `renderer.backend.get(attribute)` GPU-buffer extraction with caller-supplied raw `GPUBuffer` handles in `RCDispatchOpts`.
- `packages/walkaround-hybrid/src/rc/bvhCompute.ts` — make `buildRCSceneBVH` return raw `GPUBuffer`s alongside (or instead of) `StorageBufferAttribute`s, matching `restir/bvhCompute.ts`'s already-raw output.

**Out of scope for Phase 1:**
- `packages/walkaround-hybrid/src/rc/giReceiver.ts` and `walkaroundDiffuseLighting.ts` are pure TSL injection paths intended for a HOST app's THREE-material flow. They have **no role** in HybridEngine's shade.wgsl pipeline. Leave them as-is and document them as a host-only RC integration option in the package README. (Future: relocate to a separate `@vitrum/walkaround-rc-three-bindings` package.)

**Acceptance:** RC dispatch no longer imports `WebGPURenderer`; `RCDispatcher` constructs from a `GPUDevice` directly; the same tests pass.

## Phase 2 — HybridEngine integration

**Files:**
- `packages/walkaround-hybrid/src/HybridEngineOptions.ts` — add `readonly rcEnabled?: boolean` (default `false`).
- `packages/walkaround-hybrid/src/HybridEngine.ts` — allocate `CascadeBufferManager` + `RCDispatcher` per device; schedule cascade dispatch each frame in `renderFrame`; expose cascade C0 `GPUBuffer` to the pipeline.
- `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts` (or a new `pipeline/passes/RCDispatchPass.ts`) — declare an `rcDispatch` pass that runs before `shade` when `rcEnabled` is true.

**Acceptance:** with `rcEnabled: true` and Cornell scene, cascade buffers are allocated + dispatched per frame without errors; cascade C0 buffer is visible in the pipeline's `FrameResources.rc` sub-struct; shade.wgsl is not yet reading it (no visual change).

## Phase 3 — `shade.wgsl` Lo_rc sample + MIS composition

**Files:**
- `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts` (or new sub-module per the W4-A5 splitting pattern: `shaders/sampleCascadeC0.wgsl.ts`) — add a binding for the cascade C0 `array<vec4f>` storage buffer; implement `sampleCascadeC0(worldPos, normal)` via trilinear interpolation across the probe grid (probe position is derived from `probeOriginWorld + iVec * roomSize / probeCount` — same formula as the cascade producer).
- Add MIS composition logic in `Lo_indirect`. Two implementation tracks:
  - **Track A (cheap):** balance-heuristic with constant per-source weights — `w_ddgi`, `w_rc`, `w_restirgi` summing to 1.0. Hosts override via HybridEngineOptions. Default `[1/3, 1/3, 1/3]` if all three are active.
  - **Track B (paper-accurate):** per-pixel variance-weighted MIS — use ReSTIR's stored `Var(L)` from the temporal accumulator; pick the source with lowest variance per pixel. Requires adding a variance estimator for the DDGI atlas read (currently none).

**Recommendation:** Track A first (matches the audit's `combined indirect sum` framing). Track B becomes a follow-up.

**Acceptance:** with `rcEnabled: true` and `w_rc = 1.0`, Cornell scene renders using ONLY RC's first-bounce indirect (DDGI + ReSTIR-GI weights zero). The result should match a reference render captured from the legacy standalone RC engine path.

## Phase 4 — Reference renders + acceptance test

**Files:**
- `tools/reference-renders/W8-rc-off/cornell-1080p-1024spp.png` (capture)
- `tools/reference-renders/W8-rc-on/cornell-1080p-1024spp.png` (capture)
- `packages/walkaround-hybrid/__tests__/rcAcceptance.gpu.test.ts` (new — opt-in via `GPU=1` env)

**Acceptance:** Visible first-bounce indirect detail in `W8-rc-on` that's not in `W8-rc-off`. Numerical RMSE / SSIM acceptance threshold codified in the test.

## Composition math sketch (Phase 3 reference)

Today, `shade.wgsl`'s `Lo_indirect` ≈ `Lo_ddgi(x, n) + Lo_restirgi(x, n)`. The two estimate disjoint sample sets per the ReSTIR-GI temporal/spatial reuse, so adding them is roughly OK (the audit team flagged this as "approximately disjoint via ReSTIR's resampling weighting").

Adding RC requires:
```
Lo_indirect = w_ddgi · Lo_ddgi + w_rc · Lo_rc + w_restirgi · Lo_restirgi
where w_ddgi + w_rc + w_restirgi = 1
```
Balance-heuristic MIS over different sampling strategies (here DDGI/RC/ReSTIR-GI are three estimators of the same integral) is a valid combination as long as the weights sum to one and each estimator is unbiased. (Veach 1997, §9.2, Eq. 9.7.)

**Caveat:** DDGI and RC are biased estimators (both produce smoothed irradiance via interpolation across probes/cascades). Strict MIS bounds only apply for unbiased estimators. In practice the literature treats this combination as a perceptual quality choice — see Majercik 2019 §7 + Sannikov 2023.

## Citations to update

- `CREDITS.md` — add Sannikov 2023 "Radiance Cascades" if not already cited (verified 2026-05-18: it's there as RC-1).
- `packages/walkaround-hybrid/README.md` — once Phase 3 ships, update "Radiance Cascades (RC) are implemented under src/rc/ for standalone dispatch and material-wrapper flows; composition back into HybridEngine's shade pass is tracked" to reflect the live composition path.
- `CLAUDE.md` — update "Where things actually stand" to remove B2 once Phase 4 ships.
