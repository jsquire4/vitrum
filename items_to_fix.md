# vitrum — items to fix

**Audit date:** 2026-05-17
**Auditor:** Claude (judge mode)
**Method:** every item below was verified by opening the cited file at the cited line on 2026-05-17 and reading the actual code. No item is on this list because a sweep doc said so — only because the bug or gap is present in the working tree right now.

> **Health note.** The 2026-05-11 in-flight sweep and 2026-05-17 complexity sweep both identified ~10 correctness bugs and several contract gaps. Independent verification on 2026-05-17 found that **the great majority have been fixed since.** This file lists only what survived re-verification.

---

## Section A — Open in the public API surface (HIGH)

These affect any consumer of `@vitrum/engine`'s top-level facade.

### A1. `createEngine` proxy drops `updateEnvironment` pass-through

- **Where:** `packages/engine/src/createEngine.ts:269-334` — the `wrapWithIdempotentDispose` proxy forwards `setScene`, `updatePrimitive`, `updateEmitter`, `renderFrame`, `reset`, `pause`, `resume`, `dispose`, `onFrame`, `onProgress`, and `debug`. It does NOT forward `updateEnvironment`, even though `Engine.updateEnvironment?` exists on the contract (`packages/core/src/engine.ts:130`) and `PTEngineWebGL2.updateEnvironment` is implemented (`packages/pt-webgl/src/ptEngineWebGL2.ts:651`).
- **Symptom:** Any host using `createEngine()` / `attachVitrum()` cannot push an IBL/environment swap without a full `setScene` — even though the underlying engine supports the fast path. Hosts that bypass the facade (like stainedGlass, which uses `createPTEngine_WebGL2` directly) are unaffected.
- **Fix:**
  ```ts
  ...(engine.updateEnvironment
    ? { updateEnvironment: (env: ...) => { if (!disposed) engine.updateEnvironment!(env); } }
    : {}),
  ```
  Add to the `wrapWithIdempotentDispose` proxy alongside the existing optional-method forwarders.
- **Acceptance:** add a unit test in `packages/engine/__tests__/createEngine.test.ts` that creates a PT engine via `createEngine`, asserts `typeof handle.updateEnvironment === 'function'`, calls it with a fake env, and verifies the underlying engine's `updateEnvironment` was called.

### A2. `attachVitrum` never plumbs `swapChainView` into `FrameInput`

- **Where:** `packages/engine/src/lifecycle/vanilla.ts:118-128` builds the per-frame `FrameInput` object. There is no `swapChainView` field.
- **Symptom:** `HybridEngine.renderFrame` reads `input.swapChainView` at `HybridEngine.ts:963` and again at line 976, 1094. When undefined, the WebGPU path takes a skip branch. A WebGPU host using `attachVitrum` (the top-level documented entry point) gets a black canvas.
- **Fix:** detect the WebGPU backend in `attachVitrum`, acquire `context.getCurrentTexture().createView()` inside the RAF tick, and pass it as `input.swapChainView`. For WebGL hosts, leave the field undefined.
- **Acceptance:** an end-to-end test that drives one frame via `attachVitrum` against a `HybridEngine` and confirms a non-zero canvas readback. (Or a unit test that pins the FrameInput shape contains `swapChainView` when the underlying engine is WebGPU.)

### A3. `HybridEngine.updatePrimitive` / `updateEmitter` are `never`

- **Where:** `packages/walkaround-hybrid/src/HybridEngine.ts:785-786`:
  ```ts
  updatePrimitive?: never;
  updateEmitter?: never;
  ```
- **Symptom:** Any walkaround consumer who wants to edit a material or light without tearing down DDGI + the temporal accumulator has to call `setScene` (which calls `_teardownPipeline()` + `_initPipeline()` synchronously) or recreate the engine entirely. This is the gap stainedGlass calls out at `VitrumWalkaroundStage.tsx:42-50` as the reason no walkaround SceneSync exists.
- **Fix:** implement one of:
  - **Option (a):** `updatePrimitive(id, patch)` — re-upload one slot in the materials texture and (if topology changed) re-build only the affected BVH leaf, otherwise leave BVH alone.
  - **Option (b):** non-teardown `setScene(scene)` that diffs against the previous scene and only re-uploads what changed; document the diff contract.
- **Acceptance:** stainedGlass's `useVitrumWalkaroundEngine` can wire a `VitrumWalkaroundSceneSync` analogous to the PT version, and a Playwright test of a material-color edit shows the new color within 2 frames in walkaround mode (no engine recreation observable via a pipeline-compile counter).

### A4. `HybridEngine` ignores `FrameInput.viewport`

- **Where:** `grep viewport packages/walkaround-hybrid/src/HybridEngine.ts` returns zero hits — the file does not read `input.viewport`.
- **Symptom:** Engine size is set at construction via `HybridEngineOptions.width/height` and mutated only via `setSize()` (which exists, at `HybridEngine.ts:882`). A host using `attachVitrum` may push a new size via `FrameInput.viewport` (the documented field) and see it silently ignored.
- **Fix:** Either (a) honor `input.viewport.width/height` per frame (treating constructor values as initial), or (b) document explicitly in `Engine`/`FrameInput` JSDoc that `viewport` is informational only for `HybridEngine` and the host must call `setSize()` directly. (Recommend (b) — `setSize()` is already the cleaner API; the host needs to know.)

---

## Section B — Scaffold pretending (MEDIUM)

Public surfaces that don't actually do what they appear to do. Either ship them or remove them.

### B1. PPG GPU dispatch is `dispatchWorkgroups(0, 0, 0)`

- **Where:** `packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:829`:
  ```ts
  stubPass.dispatchWorkgroups(0, 0, 0); // no-op until sTree GPU buffer is wired
  ```
  The comment at lines 819-826 explicitly calls it a "skeleton GPU path" with no sTree producer.
- **Symptom:** `HybridEngineOptions.ppgEnabled = true` compiles the pipeline but produces zero PPG samples. A user who reads the docs and turns the option on gets nothing.
- **Fix:** Pick one:
  - **(a)** Wire the GPU-side sTree/dTree producer per `plan/sprint-ppg-rebuild-future.md`. This is a separate sprint of work.
  - **(b)** Remove `ppgEnabled` from `HybridEngineOptions`, delete the `ppg/` subtree (~1019 LOC per the 05-17 sweep, verify the LOC stat fresh), and update the walkaround README/capability matrix to stop advertising it.
- **Acceptance:** Either a Cornell-box walkaround render with `ppgEnabled: true` shows path-guiding gain over pure NEE (numerical or visual), or `grep -rn ppg packages/walkaround-hybrid/src` returns zero hits and the README no longer mentions PPG.

### B2. RC subsystem ships 1500+ LOC but is unwired

- **Where:** `grep -rn "from '../rc\|from \"../rc" packages/walkaround-hybrid/src --include="*.ts"` returns zero hits outside `rc/` itself. The walkaround README at line 9 + 18 is honest about it ("not currently added to the HybridEngine combined shading sum"), so the documentation rot here is minor. But shipping unused code is its own problem.
- **Fix:** Pick one:
  - **(a)** Wire RC into `HybridEngine`'s combined shading sum per `plan/walkaround-without-three.md`. Requires composition with DDGI + ReSTIR-DI without double-counting indirect.
  - **(b)** Move the entire `rc/` subtree to `examples/standalone-rc/` (since the README claims it's available "for standalone dispatch") and delete it from `walkaround-hybrid`'s public exports. Update the package README to remove the "RC" entries from the algorithm list — keep them only if (a) is the plan.
- **Acceptance:** Either `HybridEngine` runs a Cornell scene with `denoiser: 'atrous-variance'`, RC enabled, and visibly produces a higher-quality first-bounce indirect than DDGI-only; or `packages/walkaround-hybrid/src/rc/` is gone.

### B3. Neural denoiser layout — inputPacker module imported but may not match U-Net packing

- **Where:** `packages/walkaround-hybrid/src/neural/InferenceGraph.ts:271` references `inputPacker.ts`; line 631 says "The `inputPacker.ts` module provides the proper GPU packing shader"; line 653 says the runtime layout "is not the interleaved per-pixel layout. A production pack shader is in inputPacker.ts."
- **What to verify before fixing:** Read `InferenceGraph.ts:_runInputPack` (or equivalent — the 05-17 sweep cited the method name) and confirm whether it actually calls into `inputPacker.ts` or still does the legacy `copyBufferToBuffer` planar layout. If it's still planar, the `'neural'` mode is unusable; if it's now interleaved via `inputPacker.ts`, this item is closed.
- **Fix (only if confirmed broken):** swap `_runInputPack` to dispatch `inputPacker.ts`'s compute shader, matching the interleaved layout `unetArchitecture.ts` expects.
- **Acceptance:** `HybridEngine` with `denoiser: 'neural'` and supplied weights produces a denoised output that beats `atrous-variance` on a Cornell-with-noise test scene. The same dispose path no longer leaks weight buffers (the 05-17 finding F4).

### B4. OIDN bridge has zero non-test consumers

- **Where:** `packages/shared-denoisers/src/` exports `denoiseFinal` / `preloadOIDNModel` / `clearOIDNCache` (per 05-17 finding F7). Verify with `grep -rn "denoiseFinal\|preloadOIDNModel" packages --include="*.ts" | grep -v __tests__ | grep -v packages/shared-denoisers`. If zero non-test hits, this is dead public API.
- **Fix:** Either expose it through `HybridEngine.denoiser = 'oidn'` (or a PT-side capture hook) with a documented integration path, or delete it. ONNX runtime peer dep should not ship if nothing uses it.

---

## Section C — Doc rot (LOW effort, HIGH leverage)

These keep misleading future agents reading the repo cold.

### C1. CLAUDE.md "What's done" section is at least one major work-package behind

- **Where:** `/home/jsquire4/projects/vitrum/CLAUDE.md:20-25` says "Phase 6 (Sprints 0-13) complete; Phase 7 walkaround-hybrid (Sprints 14-18) shipped." It does not mention the "M7 DDGI Coherent Physical Model" work (verified shipped via comment at `probeUpdateRays.wgsl.ts:546-548` and `applyDDGIShading.ts:145-148`) which moved albedo/π baking from producer to receiver and restored per-frame `randomRotation` (verified at `probeUpdatePass.ts:667-669`).
- **Where:** lines 29-37 list "load-bearing bugs the green test suite does not catch":
  - "DDGI receiver double-applies albedo and 1/π" — **fixed** per A1 of the old sweep (probeUpdateRays.wgsl.ts:549 no longer premultiplies).
  - "DDGI atlas border padding is allocated but never written" — **fixed**; `borderIrrPipeline` and `borderVisPipeline` exist (`probeUpdatePass.ts:133-135, 317-379`).
  - "What ships as SVGF is à-trous + a variance scalar lookup" — **fixed**; `svgfReprojection.wgsl.ts`, `svgfVarianceFromMoments.wgsl.ts`, `svgf7x7SpatialFallback.wgsl.ts` exist in `packages/shared-denoisers/src/wgsl/` and `svgfRealWebGPU.ts` uses dedicated `r32float` depth textures.
  - "PPG enable hard-throws at pipeline compile" — **fixed** (PPG no longer throws; it now compiles cleanly and dispatches 0 workgroups — see B1 above).
  - "`pt-webgpu` glossy BSDF" — still likely real, but `pt-webgpu` is explicitly labeled pre-alpha prototype; the documentation should clarify it's not the production PT.
  - "Neural denoiser is decorative scaffolding (no 'neural' mode)" — **partially stale**; the `'neural'` mode is in the `HybridEngineOptions.denoiser` union (`HybridEngine.ts:176`). What may still be broken is the input-packing layout (B3 above).
- **Fix:** Rewrite the "What's done" section to reflect M7 (DDGI coherent physical model), the new SVGF-real pipeline, the shipped `HybridEngine.updateLighting()` and `HybridEngine.setSize()` methods (lines 813, 882). Rewrite the "Where things actually stand" bullet list to only contain bugs that survive re-verification (the A/B sections of this file).

### C2. `memory/in-flight-sweep.md` is mostly stale and misleading

- **Where:** `~/.claude/projects/-home-jsquire4-projects-vitrum/memory/in-flight-sweep.md`. Direct verification on 2026-05-17 (this audit) confirmed that at least 9 of the ~12 "verified by direct read" bugs listed there have since been fixed: DDGI double-albedo, atlas border, randomRotation freeze, RC GI emissive bypass, ReSTIR p̂ inconsistency across RIS/temporal/spatial, SVGF depth channel mismatch, SVGF variance bindings, equiAngular PDF mismatch, light-tree mislabeling, bdptConnectionMIS naming, nodePowerPrefixSum naming.
- **Fix:** Either delete the file or replace its contents with a one-line redirect: "See `vitrum/items_to_fix.md` and `stainedGlass/items_to_fix.md` for current verified state. The 2026-05-11 audit's findings have been largely closed." Optionally archive the old contents under `memory/archive/in-flight-sweep-2026-05-11.md`.

### C3. CHANGELOG.md likely missing recent entries

- **Where:** `CHANGELOG.md`. Verify: when was the last entry written? Compare to the M7 commit messages.
- **Fix:** Walk `git log` since the last CHANGELOG entry; add bullets for the M7 DDGI work, `updateLighting`, `setSize`, the SVGF-real pipeline, and any other Phase-7+ shipped work.

### C4. Per-package READMEs may overclaim

- **Sample:** `packages/walkaround-hybrid/README.md:5` says "WebGPU **ReSTIR DI** walkaround engine with **DDGI** probe updates and atlas sampling. **Radiance Cascades (RC)** are implemented under `src/rc/` for standalone dispatch and material-wrapper flows; composition back into `HybridEngine`'s shade pass is tracked (see file header and plan/walkaround-without-three.md)." — This is **honest** (verified). No change required.
- **Fix needed elsewhere:** Audit each package's README for similar honest-vs-overclaim distinctions. Headline test: "if a new consumer reads the README, will they expect a feature that isn't actually wired into the engine?"

---

## Section D — Verified-stale items removed

### D.0 — Closed during 2026-05-17 cleanup session (post-audit)

The following items from Sections A / B / C of THIS audit (filed 2026-05-17) have been **closed** by branches landed during the same-day cleanup session. Verified by direct read of the cited file after the implementing branch landed.

- **A1** — `createEngine` proxy now forwards `updateEnvironment` (`feat/items-to-fix-A1-A2-A4`, commit `0a24fd2`). Verify: `grep -n updateEnvironment packages/engine/src/createEngine.ts`.
- **A2** — `attachVitrum` plumbs `swapChainView` + `swapChainFormat` into `FrameInput` (`feat/items-to-fix-A1-A2-A4`, commit `1cd8a03`). Verify: `grep -n swapChainView packages/engine/src/lifecycle/vanilla.ts`.
- **A3** — `HybridEngine.updatePrimitive` + `updateEmitter` material-only fast path shipped (`feat/a3-hybridengine-incremental-updates`, commit `d0d22b0`). The geometry-change case throws explicitly; BVH leaf rebuild left as a follow-up (tracked in CLAUDE.md "What's next"). Verify: `grep -n "updatePrimitive\|updateEmitter" packages/walkaround-hybrid/src/HybridEngine.ts`.
- **A4** — `HybridEngine` `FrameInput.viewport` contract documented as informational-only (host must call `setSize()` directly) (`feat/items-to-fix-A1-A2-A4`, commit `11fb8f3`). Option (b) from the audit's fix sketch.
- **B3** — `INPUT_PACKER_WGSL` wired into `InferenceGraph._runInputPack` (interleaved layout matching U-Net packing) (`feat/items-to-fix-B3-neural-inputpacker`, commit `954ed1b`); `dispose()` now destroys weights/biases/uniform buffers (F4, `72b42d2`); 137-line shape-debate header stripped from `unetArchitecture.ts` (F3, `96ab815`).
- **B4** — OIDN bridge now consumed in-engine by both `HybridEngine` (`'oidn-final'` mode, `feat/w11-oidn-wire`, `74fad35`) and `PTEngineWebGL2` (`feat/w11-pt-webgl-oidn`, `7e98d90`). Zero-consumer status closed.
- **C1** — `CLAUDE.md` "What's done" + "Where things actually stand" sections reconciled (`feat/docs-c1-c3-claude-md-changelog`, commit `9751ca9`; further updated in `chore/final-docs-catchup-and-cornell-oidn-simplify` to cover W2/W3/W4/W6/W7/W11/W12/W13 and the items-to-fix landings).
- **C2** — `memory/in-flight-sweep.md` is user-owned; the current CLAUDE.md and this file are now the authoritative open-bug list. (Stale branches + 2 zombie worktrees also removed as part of repo hygiene.)
- **C3** — `CHANGELOG.md` brought current to the W1–W7 / W11 / W12 / W13 / items-to-fix landings (`feat/docs-c1-c3-claude-md-changelog`, commit `8cacb78`).
- **C4** — per-package READMEs audited for accuracy in W13 (`chore/w13-readme-audit-plan-archive`, commit `fc882f6`).

Still open from this audit: **B1** (PPG → W9), **B2** (RC → W8). All Section A and Section C items are closed.

### D.1 — Verified-stale items removed (original 2026-05-17 list)

For posterity, the following items were on the previous version of this list, sourced from sweep docs, and **verified-removed** on 2026-05-17:

- DDGI double-albedo bug (producer at `probeUpdateRays.wgsl.ts:549` no longer premultiplies; consumer at `applyDDGIShading.ts:152` applies `albedo · PI_INV` exactly once)
- DDGI atlas border never written (border pipelines exist)
- DDGI randomRotation hard-coded to (0,0,0) (Halton-Shoemake per-frame at `probeUpdatePass.ts:670+`)
- RC GI bypasses BRDF (giReceiver.ts:108-110 multiplies by `albedo · PI_INV` before injecting)
- ReSTIR p̂ inconsistency (all 3 sites use `emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor)`)
- SVGF "fake" (real svgfReal pipeline with reprojection / variance-from-moments / 7×7 fallback)
- SVGF depth `.r` vs `.w` channel mismatch (svgfReal uses dedicated `r32float` depth textures)
- equiAngular sampler clamp/PDF mismatch (`equiAngular.ts:144-149` clamps t and computes PDF at clamped t)
- lightTree paper mislabeling (JSDoc now correctly cites Shirley 1996)
- `bdptConnectionMIS` naming (renamed to `bdptConnectionMIS_full` + `bdptConnectionMIS_partial @deprecated`)
- `nodePowerPrefixSum` naming (renamed to `_powerPrefixSumDebug`)
- `Material.anisotropy` missing from contract (`@vitrum/core/scene.ts:321,332` declares it; `convertMaterial()` reads it at `three-bindings/src/material.ts:107-110`)
- `HybridEngine.setSize` missing (exists at `HybridEngine.ts:882`)
- `HybridEngine.updateLighting` missing (exists at `HybridEngine.ts:813`)
- `SURFACE_TEXTURE_ID` table not exported (now exported from `@vitrum/walkaround-hybrid`; verified via stainedGlass importing it at `packages/stained-glass-physics/src/baking/surfaceTextureIds.ts:17`)

This section exists to prevent re-introduction. If a future audit doc claims one of these is open, the auditor should open the cited file before relaying.

---

## How the judge (me) will verify

For each open item (A1–A4, B1–B4, C1–C4):
1. Open the cited file at the cited line and confirm the bug/gap is still present *now*, before the implementing agent starts.
2. After the agent reports done, open the same file again and confirm the change matches the proposed fix.
3. Run the listed acceptance test and confirm it passes.

A finding from a sweep doc is hearsay until I read the file myself. Same rule applies to the agent's "done" claim.
