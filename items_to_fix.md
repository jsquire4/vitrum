# vitrum — items to fix

**Audit date (original):** 2026-05-17
**Status reconciliation (2026-05-24):** every item in Sections A / B / C is closed, and the Section E merge-race backlog (E1–E7) is now closed as well. The descriptions below remain for posterity so future agents can see what was once broken and where the fix landed.

> **OPEN ITEMS (2026-06-06):** a new full-codebase audit filed **Section G (G-P0.1 … G-P2.7)** in [full-codebase-audit-2026-06-06.md](./full-codebase-audit-2026-06-06.md). A working-tree implementation sweep addresses the P0/P1 findings and most P2 hygiene items. **Native-WSL mechanical validation is DONE (2026-06-06):** typecheck clean, full vitest green (after re-pinning the goldens the sweep legitimately moved + fixing 2 sweep-introduced test bugs), fork shader smoke PASSES, layout codegen in sync. A second-pass 20-agent audit of the post-sweep tree fixed three further defects same-session (pt-webgpu lightTreeBuffer destroy leak; walkaround-rc dummy-TLAS buffer leak; three-bindings skinned-mesh double-world-transform — regression-pinned by `skinnedWorldTransform.test.ts`) and filed two NEW open items: `updateLighting({primaryLightDir})` does not re-sync the DDGI sun direction (HybridEngine.ts:1524-1548), and ~~the fork's always-on stained-glass shadow-ray perturbation biases NEE visibility (attenuate_hit_function.glsl.js:182-205)~~ — **RESOLVED (2026-06-08, fix landed `178f80d` + `75391f2`; verified by code-read):** the perturbation is NOT always-on. It is gated behind `#if FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION` (default `0` at `PhysicalPathTracingMaterial.js:69`), the fork shader-smoke-check asserts it must never be enabled by default (`shader-smoke-check.js:200-205`), and a tangent-length guard skips it when no valid TBN exists (`attenuate_hit_function.glsl.js:185`). The default shadow/NEE path is unbiased; the perturbation is an opt-in caustic-relief feature with a documented bias trade-off only when a host compiles it in. So of the two items filed here, only the `updateLighting` DDGI sun-sync one is unverified-open. **GPU A/B captures DONE (2026-06-06, `daa9716` — full numbers in HARDWARE-VALIDATION-NEEDS.md top block):** G-P0.1 PASS, G-P0.2 PASS, G-P0.3 PASS, G-P1.2 PASS-on-rig (hardware-GL BDPT connections runtime = Windows-Chrome follow-up), G-P1.1 wiring FIXED but radiometric INSPECT (PPG live + unbiased; no variance gain on the open Cornell — needs a guiding-favourable scene). The captures surfaced + `daa9716` fixed: a **pt-webgpu P0** (within-leaf closest-hit last-writer-wins — face-on thin slabs black; **all pre-`daa9716` pt-webgpu reference renders/baselines embed it → re-capture before the next fidelity gate**), the **PPG enable-wiring gap**, and a **fork P0 HEAD blocker** (`activeLayerWeight` dangling call; GLSL call-closure gate added to the fork smoke). ~~NEW INSPECT: walkaround TLAS path produced zero ReSTIR signal headless~~ **RESOLVED (2026-06-06, `0bedd92`) — and INVERTED:** TLAS was fine (the original report conflated an up-facing-emitter scene confound); the REAL bug was **ReSTIR merged mode black since PR-3 (29942f9, 2026-05-26)** — the merged-mode scene-BGL placeholders were 16 bytes vs binding 6's 32-byte `array<BVHNode>` minimum → invalid bind group → every ReSTIR pass no-opped (the DDGI-`ea88803`/RC dummy-buffer class, never applied to the ReSTIR scene BGL; hidden because the `>1 mesh → tlas` auto-rule kept merged out of every multi-mesh test). Blast radius was single-mesh core scenes + `tier:'lite'`. Fixed (dummy 16→32B) + commit-bisect + intervention-proven on dzn: merged/auto/tlas now all render `direct=0.7320` identically on the bisect Cornell (`wsl-gpu/scripts/tlas-zero-gi-bisect.ts`). Remaining non-validation work is concentrated in the larger G-P2.6 allocator/view memoization and lazy-allocation follow-up; everything in Sections A–F below remains closed.

> **DECISION-QUEUE PASS (2026-06-07).** The 5 parked GPU-evidence decisions were resolved by the user and most executed this session: (1) **DDGI bounce → Lambertian colour bounce FIXED** (`8aa444a`, `radiance=(direct+indirect)·baseColor/π`; dims indirect ~4× + adds chroma — host calibration review + DDGI reference re-capture noted in HVN); (2) **photon-map caustic DEMOTED to approximate** (`c9d88fa`, honest-labeling only); (3) **API trio LANDED** (`c9d88fa`: backend-typed factories, `engine.getScene?()`, `negotiateWebGPUDevice`); (4) **RC out-of-model weight** — user chose the full fix (gate `cRc` on RC-has-energy + extend RC's probe cast to sample emitters) — IN PROGRESS; (5) **checkerboard rendering** (ResolvePass) — design plan complete, implementation pending. NEW open items from the V17 PPG occluded-scene A/B (`queue-2026-06-07/v17-ppg`): **PPG refine-loop runaway** (stable ~7 refine cycles then ramps ~3× via GI-reservoir↔dTree positive feedback) and **base-GI size-200 estimator instability** (ppg-OFF diverges for some seeds, control var 0.485× — independent of PPG). Both need follow-up before PPG is production-ready.

> **PT-WEBGL2 NATIVE-RENDER BRINGUP (2026-06-08).** The THREE-free `@vitrum/pt-webgl2` backend now renders a **correct, converging Cornell** on a real WebGL2 driver (llvmpipe via ANGLE, numeric readback — no visual claims). Root-cause chain fixed this session (commit `3363b87` + the prior bringup series `fbc8b68`/`ed85e45`/`cda5a03`): THREE-compat GLSL3 #define bridges → individual-uniform + dummy-sampler wiring → struct-member sampler binding names → **per-vertex (not per-triangle) materialIndex** → **emissive-emitter fold** (three-bindings strips emissive into a `mesh-area` emitter for NEE backends, but the fork integrator lights by hitting `surf.emission` AND `packLightsTexture` filters `mesh-area` out → light fell through the crack → black; `foldMeshAreaEmittersIntoMaterials` re-attaches radiance pt-webgl2-only). Convergence proof: 8spp meanLum 0.444/nonBlack 0.277 → 128spp meanLum 0.449 (stable)/nonBlack 0.986/0 NaN = unbiased MC convergence. 83/83 vitest green. Harness: `wsl-gpu/webgl2-capture/run-ptwebgl2-probe.mjs` (`SPP=<n>` env to vary samples). **FORK-VS-NATIVE A/B PASS (2026-06-08):** same Cornell through the original `three-gpu-pathtracer` `WebGLPathTracer` vs pt-webgl2 at matched bounces(8)/spp(128)/res/camera → converged ratios native/fork meanLum 1.0023, maxLum 1.0002, per-channel R 1.0037 / G 1.0025 / B 0.9952 (sub-percent, channels straddle 1.0 = noise not bias; ratios tightened from ~1.03@48spp as both converged). The THREE-free port is numerically faithful to the fork, colour bleed included. Harness: `wsl-gpu/webgl2-capture/run-ptwebgl2-vs-fork.mjs`.
>
> **PT-WEBGL2 A/B COVERAGE MAP (2026-06-09) — "what's left before dropping the fork".** Expanded the fork-vs-native A/B (`wsl-gpu/webgl2-capture/run-ptwebgl2-vs-fork.mjs`, `SCENE=cornell|glass|envmap`) across hard scenes: **(1) Diffuse GI — FAITHFUL** (meanLum 1.0023, per-channel <0.5%). **(2) Glass/transmission — FAITHFUL** (Cornell + glass sphere, refraction/Fresnel/IOR/attenuation; 96spp meanLum 1.0048, per-channel ~1%, channels straddle 1.0 = noise). **(3) Env-map/IBL — math FAITHFUL after a real fix** (`43a5b34`: `envMapInfo.totalSum` was never set → env-sampling GLSL early-outed on `totalSum==0` → IBL silently rendered fully black; now lit, 240spp meanLum 0.984 converging, visibility split identical to fork on a high-variance small-sun scene). **(G2) IBL data-bridge — RESOLVED** (`dc38f69`, 2026-06-09): `three-bindings` emitted `hdri` as a `THREE.Texture` (correct for the fork-wrapping `@vitrum/pt-webgl`, which reads it back via `vitrumSceneToThree`), but THREE-free backends need a raw `{width,height,data}` payload → IBL silently dropped through the on-ramp (empirically: meanLum 0). Fixed with `equirectTextureToPayload` (duck-types `.image.{data,width,height}`; RGBA→RGB stride, HalfFloat decode, rows reversed iff `flipY` — mirrors the fork's `preprocessEnvMap`) + a non-breaking `sceneFromThreeJS({ environmentPayload: 'raw' })` opt-in (default `'texture'` preserves pt-webgl). Verified: the 'raw' on-ramp lights pt-webgl2 byte-identically to the manual payload (totalSum 7891) and faithfully to the fork (ratio 0.97). pt-webgpu has the identical latent gap — the same on-ramp option fixes it. **(G3) Textures — RESOLVED** (`63a6dab` + `5eaf664`, 2026-06-09, core path): the materials packer hardcoded every map id to -1 (no atlas) → textured materials rendered flat. Wired end-to-end across 3 packages: **shared-bvh** propagates per-vertex UVs through `mergeWorldSpaceFromCore` (`WorldSpaceMergeResult.uvs`, stride 2 — without this the UV layer was zero and textures sampled one texel); **pt-webgl2** adds `texturesArray.ts` (gather unique map handles, duck-type pixels — raw `{w,h,data}` OR DataTexture `.image`, RGBA/RGB/R stride + HalfFloat — nearest-resample to a common dim, pack an RGBA32F `sampler2DArray` + handle→layer map), assigns each material map's atlas LAYER index + writes the UV-transform mat3 (`texel 55+2k`, THREE `setUvTransform`), and fixes the attributes packer to read UVs at stride 2. Verified on the textured A/B (4-quadrant-albedo panel): meanLum ratio 0.995, per-quadrant orientation matches the fork (top bright, TL<TR, BL<BR — no UV flip), bright quadrants within 1%. **OPEN follow-up (G3.1):** DOM `Image`/`ImageBitmap` pixel sources need a host-side canvas-readback bridge (the `DataTexture`/raw forms cover procedural + baked textures; canvas can't run in the Node unit tests). Math fidelity for all implemented paths (diffuse/glass/IBL/textures) is proven.
>
> **RESOLVED — pt-webgl2 G1: gNormalDepth aux g-buffer diverged under the running-average regime** (fix `567a1f9`, 2026-06-09; approach (b)). `gNormalDepth = vec4(normalEnc, linearDepth)` packs depth (>1) in `.a`, and WebGL2 weights each MRT attachment's blend by its OWN output alpha → the colour-blend recurrence diverged → garbage/NaN (−46175 at 8spp, NaN at 128spp). Fixed by disabling blend on JUST the aux attachments via `OES_draw_buffers_indexed.disableiOES(BLEND, 1/2)` so they overwrite (latest sample = correct, primary-hit is deterministic) while the colour attachment keeps its EXACT running-average untouched. **LESSON:** a first attempt (`c95377b`: write g-buffer on sample 0 + freeze via `drawBuffers=[0,NONE,NONE]`) silently 3×-brightened OPEN/env-lit scenes (closed Cornell unaffected) — bisect-confirmed; a g-buffer fix that perturbs colour is net-worse than the unused-aux bug it fixes. The redo leaves the colour path byte-identical. Verified: Cornell g-buffer sane (maxDepth 4.2, normal [0,1], 0 NaN), env A/B ratio 0.972. Extension absent → g-buffer best-effort, colour always correct.

> **Health note (2026-05-24).** Historical "NOT IN HEAD" merge-race items have been re-landed: W2-C6 shared WGSL primitives, W3-D6 Mat4 branding, W3-D7 FrameOutput discriminated union, W3-D19 backend texture brands/helpers, W6-E6 ForkAccess indirection, and W7-G5 export hygiene. Section E is retained as an audit trail only.

---

## Section A — Public API surface (all closed 2026-05-17 → 2026-05-18)

These affected any consumer of `@vitrum/engine`'s top-level facade. All four
items shipped during the post-audit cleanup session — see Section D.0 for the
commit map. The descriptions are kept below for posterity.

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

## Section B — Scaffold pretending (all closed)

Public surfaces that didn't actually do what they appeared to. All four items
shipped — B1 (PPG GPU dispatch) closed 2026-05-18; B2 (RC into HybridEngine)
closed via the W8 sprint (2026-05-18, see CLAUDE.md "Where things actually
stand"); B3 (neural inputPacker) and B4 (OIDN bridge consumers) both closed
during the items-to-fix landings. Descriptions kept below for posterity.

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

## Section C — Doc rot (all closed)

All four items closed during the items-to-fix landings + W13 documentation
reconciliation. Descriptions kept below for posterity.

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

## Section E — Open items discovered 2026-05-19

### E1–E7 — seven sprint-of-2026-05-17 commits lost via merge races

All seven landed as standalone feature commits in the May 17 sprint but their
changes did not survive into HEAD because subsequent commits were branched
from pre-feature parents and silently overwrote them on merge. The features
still exist as commits in `git log`; they just have zero footprint in the
current code. All seven are type-system / structural / API-hygiene changes —
none is a runtime correctness bug, the pre-sprint behaviour is what's
running.

**Status as of 2026-05-24:**
- E1–E7 — **closed/re-landed**. Keep these entries as historical provenance.

The audit so far has spot-checked W2/W3/W6/W7 sub-bullets. W4/W11/W12/W13
have not been comprehensively audited — additional losses may surface.

### E1. W3-D7 FrameOutput discriminated union — closed (re-landed)

- **Where:** `packages/core/src/frame.ts`. Commit `40cd837` (W3-D7, 2026-05-17) introduced `FrameOutput = FrameSkipped | FrameRendered` with a `kind` discriminant. Commit `9ea12c9` (W3-D18, ~50 min later) branched from `80c2388` (D17, pre-D7) instead of from D7, and when its frame.ts changes merged they overwrote D7's. The net result: D7's code is unreachable from HEAD even though the commit is in the history.
- **Symptom:** the `FrameOutput` contract is back to `primaryRadiance: BackendTexture | null` + `samplesAccumulated === 0` skip sentinel — the exact pattern D7 was supposed to eliminate. Hosts that didn't check `samplesAccumulated > 0` before reading `primaryRadiance` would silently dereference null; the type system can't catch them.
- **Verification:** `git show HEAD:packages/core/src/frame.ts | grep -c FrameRendered` → 0. `grep -rn "FrameRendered\|FrameSkipped\|kind: 'rendered'" packages --include="*.ts"` → 0 consumers anywhere.
- **Fix:** re-apply D7's contract change, then propagate to:
  - producers: `walkaround-hybrid/src/HybridEngine.ts`, `pt-webgl/src/ptEngineWebGL2.ts`, `pt-webgpu/src/index.ts` (and any examples returning FrameOutput from a stub).
  - consumers: `engine/src/createEngine.ts` (the post-dispose stub at the `disposed` guard), examples (`cornell-box`, `two-engines-one-scene`, `hero-product-viz`), tests that fake `renderFrame()`.
  - new test: re-add `packages/engine/__tests__/frameOutputShape.test.ts` pinning the union.
- **Closure note (2026-05-24):** re-landed in the single-wave contract migration; producers and test stubs now emit `kind: 'rendered' | 'skipped'`.

### E2. W3-D6 Mat4 brand — closed (re-landed)

- **Where:** `packages/core/src/scene/math.ts:10`. Commit `e845cc5` (W3-D6, 2026-05-17) brand-typed `Mat4` as `Float32Array & __mat4Brand` and added an `asMat4(arr): Mat4` constructor that throws on length ≠ 16. Commit `cead5ab` (2026-05-18) split the monolithic `scene.ts` into 6 sibling files (`scene/{math,emitters,environment,primitives,material,index}.ts`). The split was done against a pre-D6 working copy, and the new `scene/math.ts` was created with the unbranded `Mat4 = Float32Array` form.
- **Symptom:** any `Float32Array` of any length silently satisfies the `Mat4` type. The D6 motivation — preventing 9-element upper-3×3 normal matrices from being passed as 16-element Mat4 — is gone. `asMat4` doesn't exist; callers that should have routed through it for length validation just `as Mat4` cast.
- **Verification:** `grep -c "__mat4Brand" packages/core/src/scene/math.ts` → 0; `grep -rn "asMat4" packages/*/src --include="*.ts"` → 0 callers.
- **Fix:** re-port the D6 changes to `scene/math.ts` (brand + asMat4), then re-update the 21 implicit-cast construction sites that D6 originally fixed (three-bindings, pt-webgpu, engine).
- **Closure note (2026-05-24):** re-landed with `asMat4`/`isMat4`, and callsites migrated across packages/examples.

### E7. W7-G5 validateBvhEncoding un-export — **RE-APPLIED 2026-05-19**

Re-applied during the same session that filed the audit. `packages/shared-bvh/src/index.ts` now uses selective named exports (`buildSceneBVH`, `refitBvhBounds`, the two interface types) from `bvhCommon.js` instead of `export * from`. `validateBvhEncoding` leaves the public surface. Both shared-bvh test files that needed it now import from `'../bvhCommon.js'` directly. All downstream packages (walkaround-hybrid, walkaround-rc, pt-webgpu, pt-webgl, engine) typecheck clean — the external surface of @vitrum/shared-bvh that they actually use is unchanged.

### E6. W6-E6 ForkAccess indirection — closed (re-landed)

- **Where:** `packages/pt-webgl/src/`. Commit `a6a3c90` (W6-E6, 2026-05-17) added `forkAccess.ts` (static `ForkAccess` class with `getMaterial(tracer)` + `getRenderTexture(tracer)`) and migrated `forkUniformBridge.ts` + `ptEngineWebGL2.ts` off direct `tracer._pathTracer.material.uniforms` reach-throughs.
- **Symptom:** if the upstream three-gpu-pathtracer fork renames `_pathTracer` (or upstreams an official accessor), `forkUniformBridge.ts` and `ptEngineWebGL2.ts` break in lockstep and have to be hand-patched at every reach-through site — the exact problem the indirection was meant to solve.
- **Verification:** `packages/pt-webgl/src/forkAccess.ts` is missing. `grep -n "_pathTracer" packages/pt-webgl/src/forkUniformBridge.ts` returns the pre-E6 direct-cast form: `const tracer = pathTracer as { _pathTracer?: { material?: PathTracerMaterialLike } }; const material = tracer._pathTracer?.material ?? null;`.
- **Fix:** re-apply `a6a3c90`. Test-only helper `makeForkPathTracerStubForTests` also needs re-adding (the original commit added it).
- **Closure note (2026-05-24):** `forkAccess.ts` restored and pt-webgl bridge/engine callers now route through it.

### E5. W6-E1 reuseSharedWebGpuDevice default flip — closed (re-landed)

- **Where:** `packages/shared-denoisers/src/{atrousVarianceWebGPU,hdrLuminanceBilateralWebGPU,svgfRealWebGPU}.ts`. Commit `3dbe11a` (W6-E1, 2026-05-17) was a breaking change: flipped `reuseSharedWebGpuDevice`'s effective default from `true` to `false` so callers without an explicit `device` would error rather than silently reuse a process-wide singleton (a design-principle violation — host-owns-lifecycle). The diff changed `opts.reuseSharedWebGpuDevice !== false` to `opts.reuseSharedWebGpuDevice === true` in all three dispatchers.
- **Symptom:** the pre-E1 implicit-singleton behaviour is back. Callers without a `device` arg silently reuse the global, masking the host-ownership contract violation E1 was meant to surface.
- **Verification:** `grep -n "reuseSharedWebGpuDevice !== false" packages/shared-denoisers/src/*.ts` returns 3 hits (the pre-E1 form). The post-E1 form (`=== true`) is not in HEAD.
- **Fix:** re-apply `3dbe11a` and update any test that called the dispatchers without `device` (those would have been updated in the original commit).
- **Closure note (2026-05-24):** denoiser dispatchers now require explicit opt-in (`reuseSharedWebGpuDevice === true`) when no device is supplied.

### E4. W2-C6 shared-samplers WGSL primitives — closed (re-landed)

- **Where:** `packages/shared-samplers/src/wgsl/`. Commit `da286d7` (W2-C6, 2026-05-17) introduced `pcg.wgsl.ts` (80 LOC: `pcgInit`, `pcgNext`, `rand_f32`, `rand_f32_2`, `rand_f32_3`) and `bsdfPrimitives.wgsl.ts` (108 LOC: `buildONB`, `sampleCosineHemisphere`, `cosineHemispherePdf`, `fresnelSchlick`) as canonical sources, and migrated walkaround-hybrid + pt-webgpu off their local copies. Neither file is in HEAD; both local copies are back. Verified 2026-05-19: `ls packages/shared-samplers/src/wgsl/` returns only `hammersley.wgsl.ts`, `luminance.wgsl.ts`, `octahedralCore.wgsl.ts`. `grep -A5 "fn pcgInit" packages/pt-webgpu/src/wgsl/common.wgsl.ts packages/walkaround-hybrid/src/shaders/common.wgsl.ts` returns byte-identical 6-line definitions in both files.
- **Symptom:** parallel WGSL primitives evolving in two places. The original C6 motivation — capitalisation drift (`buildONB` vs `buildOnb`) and arg-order drift (`sampleCosineHemisphere(n, rng)` vs `cosineHemisphereSample(rng, n)`) — has had time to re-accumulate.
- **Fix:** re-apply `da286d7` cleanly. The original commit's stat: `pcg.wgsl.ts` +80, `bsdfPrimitives.wgsl.ts` +108, walkaround-hybrid common -75 LOC, pt-webgpu common -35 LOC, pathTraceBruteforce -38 LOC.
- **Closure note (2026-05-24):** canonical `pcg.wgsl.ts` + `bsdfPrimitives.wgsl.ts` restored in shared-samplers; duplicated local copies removed from walkaround-hybrid/pt-webgpu common modules.

### E3. W3-D19 BackendTexture brand — closed (re-landed)

- **Where:** `packages/core/src/frame.ts:193`. Commit `5863cda` (W3-D19, 2026-05-17) replaced `export type BackendTexture = unknown` with `BackendTexture<TBackend>` + `BackendTextureFormat<TBackend>` nominal brands keyed by `unique symbol`. The D17/D18 merge race that dropped D7 (see E1) also overwrote D19 because D19 was branched off D7.
- **Symptom:** WebGPU `swapChainView` can be silently set to a WebGL `WebGLTexture` and vice versa — the type system can't catch it. `as*BackendTexture` constructors + `narrowTo*` helpers in `walkaround-hybrid` / `pt-webgpu` / `pt-webgl` are also gone (verified 2026-05-19: `grep -rn "asWebGPUBackendTexture\|narrowToWebGPU\|asWebGLBackendTexture" packages/*/src` → 0).
- **Verification:** `packages/core/src/frame.ts:193` reads `export type BackendTexture = unknown;` — the pre-D19 form.
- **Fix:** re-apply D19 (5 of the original commit's files: core types + 3 backend brand helpers + 1 example). Companion test `packages/engine/__tests__/backendTextureBrand.test.ts` also needs re-adding.
- **Closure note (2026-05-24):** backend texture/format brand helpers restored in `core/frame.ts`, with helpers consumed at backend boundaries.

---

## Section D — Verified-stale items removed

### D.0 — Closed during 2026-05-17 cleanup session (post-audit)

The following items from Sections A / B / C of THIS audit (filed 2026-05-17) have been **closed** by branches landed during the same-day cleanup session. Verified by direct read of the cited file after the implementing branch landed.

- **A1** — `createEngine` proxy now forwards `updateEnvironment` (`feat/items-to-fix-A1-A2-A4`, commit `0a24fd2`). Verify: `grep -n updateEnvironment packages/engine/src/createEngine.ts`.
- **A2** — `attachVitrum` plumbs `swapChainView` + `swapChainFormat` into `FrameInput` (`feat/items-to-fix-A1-A2-A4`, commit `1cd8a03`). Verify: `grep -n swapChainView packages/engine/src/lifecycle/vanilla.ts`.
- **A3** — `HybridEngine.updatePrimitive` + `updateEmitter` are implemented. Transform/positions fast paths are in-tree; material/emitter edits route through incremental patch APIs with rebuild fallback where needed. Verify: `grep -n "updatePrimitive\|updateEmitter" packages/walkaround-hybrid/src/HybridEngine.ts`.
- **A4** — `HybridEngine` `FrameInput.viewport` contract documented as informational-only (host must call `setSize()` directly) (`feat/items-to-fix-A1-A2-A4`, commit `11fb8f3`). Option (b) from the audit's fix sketch.
- **B3** — `INPUT_PACKER_WGSL` wired into `InferenceGraph._runInputPack` (interleaved layout matching U-Net packing) (`feat/items-to-fix-B3-neural-inputpacker`, commit `954ed1b`); `dispose()` now destroys weights/biases/uniform buffers (F4, `72b42d2`); 137-line shape-debate header stripped from `unetArchitecture.ts` (F3, `96ab815`).
- **B4** — OIDN bridge now consumed in-engine by both `HybridEngine` (`'oidn-final'` mode, `feat/w11-oidn-wire`, `74fad35`) and `PTEngineWebGL2` (`feat/w11-pt-webgl-oidn`, `7e98d90`). Zero-consumer status closed.
- **C1** — `CLAUDE.md` "What's done" + "Where things actually stand" sections reconciled (`feat/docs-c1-c3-claude-md-changelog`, commit `9751ca9`; further updated in `chore/final-docs-catchup-and-cornell-oidn-simplify` to cover W2/W3/W4/W6/W7/W11/W12/W13 and the items-to-fix landings).
- **C2** — `memory/in-flight-sweep.md` is user-owned; the current CLAUDE.md and this file are now the authoritative open-bug list. (Stale branches + 2 zombie worktrees also removed as part of repo hygiene.)
- **C3** — `CHANGELOG.md` brought current to the W1–W7 / W11 / W12 / W13 / items-to-fix landings (`feat/docs-c1-c3-claude-md-changelog`, commit `8cacb78`).
- **C4** — per-package READMEs audited for accuracy in W13 (`chore/w13-readme-audit-plan-archive`, commit `fc882f6`).
- **B1** — PPG dispatch is now a real `dispatchWorkgroups(wgCount, 1, 1)` call wired through the W9 GPU flat-buffer traversal kernel. Verified at `packages/walkaround-hybrid/src/pipeline/passes/PPGGuidePass.ts:94` (and the file header's "no more dispatchWorkgroups(0,0,0)" callout). The `ppg-dispatch.test.ts` suite pins the wiring (5 tests passing, run 2026-05-18). The earlier "stub" comment block referenced at `WalkaroundGPUPipeline.ts:819-829` is gone — that file's PPG integration now flows through `PPGCoordinator` (post W1-R5 + walkaround-pipeline-split).

- **B2** — RC subsystem now wired into HybridEngine via the W8 sprint (Phases 1A/1B/2/3/4 all landed 2026-05-18). `RCSubsystem` constructed in `HybridEngine.ts:388` when `opts.rcEnabled === true`; cascade-0 buffer composed into `shade.wgsl` via Track-A balance-heuristic MIS; `rcAcceptance.gpu.test.ts` harness scaffold + reference-render landings in `tools/reference-renders/W8-rc-{off,on}/`. See [plan/w8-rc-mis-composition.md](./plan/w8-rc-mis-composition.md) for the full sprint trace. The W8 follow-up — extracting `walkaround-hybrid/src/rc/` into a standalone `@vitrum/walkaround-rc` package — also shipped 2026-05-18 (verify: `ls packages/walkaround-rc/src/`).

All Section A, B, and C items from this audit are closed.

### D.1 — Verified-stale items removed (original 2026-05-17 list)

For posterity, the following items were on the previous version of this list, sourced from sweep docs, and **verified-removed** on 2026-05-17:

- DDGI double-albedo bug (producer at `probeUpdateRays.wgsl.ts:477-497` no longer premultiplies; consumer at `applyDDGIShading.ts:172-175` applies `albedo · PI_INV` exactly once)
- DDGI atlas border never written (border pipelines exist)
- DDGI randomRotation hard-coded to (0,0,0) (Halton-Shoemake per-frame at `probeUpdateFrameParams.ts:19-60`)
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

---

## Section F — Open follow-ups (post-2026-05-30 wave)

### T3.G — `engine.debug.pickPrimitive(x, y)` — DONE 2026-06-04

- **STATUS: DONE + unit-tested.** Implemented option (a) — exact CPU ray-cast (better than the "ray-AABB approximate" first sketched; triangle-exact, not just bounds). `pickPrimitiveCpu` (`walkaround-hybrid/src/debug/pickPrimitive.ts`): unproject (x,y) through the retained last-frame camera (`HybridEngine._lastFrameCamera`, copied each `renderFrame`) → `invertMat4(proj·view)` far-plane ray (NDC z=+1, robust to both GL/WebGPU depth conventions) → closest-hit over `_lastScene.primitives` — Möller–Trumbore for mesh/instanced/skinned (rest-pose; deformation ignored, debug-only), world bounding-sphere for analytic (or `fallbackMesh` triangles when present). Wired into the debug surface (`HybridEngineDebug.ts` `pickPrimitive` + new `pickScene`/`pickCamera`/`pickSize` deps). `MaterialInspector` self-wires click-to-pick via an optional `canvas` prop (CSS-pixel click → canvas backing-store pixels → `pickPrimitive` → internal `pickedId`, no external `selectedPrimitiveId` needed); the stale "T3.G followup" warning + `vanilla.ts` "stubbed" comment were corrected. 7 unit tests (`pickPrimitive.test.ts`: centre-pixel hit [the acceptance case], miss→null, depth ordering, analytic sphere, transform-honoured, empty scene, zero-viewport). Full workspace typecheck + 1216 vitest green. Option (b) (GPU primitive-ID readback) deferred — CPU is exact for meshes and adds zero GPU-readback latency.
- **Acceptance (met):** `engine.debug.pickPrimitive(x, y)` returns the correct primitive ID for a known mesh (unit-tested); `MaterialInspector` opens on canvas click without external `selectedPrimitiveId` wiring (the `canvas` prop path).

### F-RC1 — RC merged-mode GPU traversal unreliable on multi-node trees (FIXED 2026-06-04)

- **STATUS: FIXED + oracle-validated.** ROOT CAUSE (NOT "traversal flakiness" as first guessed — an F-TLAS1-CLASS STRIDE BUG, upload direction): `buildSceneBVH` emits STRIDE-3 indices (12 B/tri); `RCSubsystem._uploadBVH` uploaded them RAW; but the RC probe shader declares `rc_geom_index: array<vec4u>` and `bvhIntersectFirstHit` reads each entry at a 16-byte (stride-4) stride (`bvhIntersect.wgsl.ts:327`) → GPU reads `bvhIndex.xyz[t]` from byte 16·t while data lives at 12·t → wrong for tri≥1 (the 17.75 dB / 3× lit-slot divergence). DDGI over the SAME builder works because it pads via `padTriangleIndicesToVec4` (`ddgi/probeUpdateBvhBuffers.ts:76`); RC's merged upload was the lone consumer that forgot to. FIX (`HybridEngineRC.ts:_uploadBVH`): pad to stride-4 via the same `padTriangleIndicesToVec4` (stride-3 `bvh.indices` retained for the refit fast path; TLAS path already stride-4). PROVEN: a CPU brute-force oracle (`wsl-gpu rc-merged-bvh-bruteforce-ab.ts`, modeled on F-TLAS1's) gives FIXED **100.00%** (2304/2304) vs the re-introduced bug **47.92%** (Δ52pp); production buffer now 768 B = stride-4 (was 576). RC GPU SMOKE ADDED: the oracle is wired into the every-push T1 smoke (`wsl-gpu t1-smoke.mjs:runRcMergedOracle`) — closes the "RC had zero GPU coverage" gap that kept this latent. All re-run by the lead.

- **Where:** `packages/walkaround-rc/src/wgsl/probeRayCast.wgsl.ts` (`rcTraceFirstHit` → `BVH_INTERSECT_WGSL`/`TLAS_TRAVERSAL_WGSL`). RC's merged-mode probe traversal.
- **Symptom:** rendering the SAME geometry through two valid-but-differently-shaped SAH trees (three-mesh-bvh `MeshBVH` vs the THREE-free `buildArrayBvh`) yields a **3× difference in lit probe slots** (THREE 5756 vs core 17783) and **PSNR 17.75 dB / |Δ mean| 34%** on a 6-prim Cornell (`wsl-gpu scripts/rc-core-bvh-ab.ts` CHECK 2). A CORRECT BVH traversal returns identical closest-hits on any valid tree over the same geometry, so this is a traversal bug, not a tree bug. **Diagnosis it is RC-traversal-specific, not the builder:** `buildArrayBvh`'s multi-node tree is independently validated — the DDGI core-BVH A/B renders it at 82.66 dB (`ddgi-core-bvh-ab.ts`). So RC's traversal mishandles tree shape; DDGI's (over the same builder) does not.
- **Latency of discovery:** RC merged-mode GPU was NEVER exercised — `rcEnabled:false` in every harness, no GPU compile/bind gate (W8 pinned only CPU `packRCParams` + `wgslCompose` order). The shader didn't even compile (missing `safe_normalize`) or bind (16-byte TLAS placeholder) until `592ef7f` fixed those; THIS traversal issue is the next layer.
- **Fix:** audit `rcTraceFirstHit` / the shared `BVH_INTERSECT_WGSL` stack traversal as RC assembles it (stack depth, node-order/leaf-flag assumptions, `triEps`); compare against the DDGI/ReSTIR consumers of the same WGSL (which traverse correctly) for what RC composes differently. **Add a real RC GPU smoke** (an `rcEnabled:true` Cornell in the wsl-gpu T1 path) so RC merged-mode regressions are caught — RC currently has zero GPU coverage.
- **Acceptance:** RC merged-mode renders a multi-node Cornell to within converged-A/B tolerance of a CPU brute-force (all-triangles, no BVH) reference; an `rcEnabled` GPU smoke is wired and green.

### F-RC2 — RC merged-BVH THREE-decouple — DONE 2026-06-04 (re-attempted after F-RC1 fix, oracle-validated)

- **STATUS: DONE + landed + render-validated.** Re-attempted now that F-RC1 is fixed. The RC brute-force oracle (`rc-merged-bvh-bruteforce-ab.ts`) scores BOTH the THREE `setScene` path AND the new core `setSceneFromCore` path at **100.00%** (2304/2304) vs the independent CPU all-triangles ground truth — both hit the same 1444 triangles at the same world points through DIFFERENTLY-shaped SAH trees (tree-shape-invariant correctness; buggy stride-3 still discriminated at 47.92%). The converged THREE-vs-core RC A/B (`rc-core-bvh-ab.ts` CHECK-2, previously "inconclusive" at 17.75 dB) is now **999 dB pixel-identical** — confirming that divergence WAS the F-RC1 stride bug, not harness edge-grazing. CPU set-equivalence: 9 tests (merged tri SET + per-tri materialId + world AABB + packed cascade-material bytes, incl. ei=4→`emissive·1` + the thickness→0.1 floor). The core path inherits the F-RC1 stride-4 pad via the shared `_setSceneFromBVH`→`_uploadBVH` (768-byte index buffer confirmed). Wired into production (`HybridEngineGiPropagation` routes merged-mode RC to `setSceneFromCore` when a core Scene is present, falling back to THREE for the raw-`threeScene` escape hatch). typecheck clean; wh 1185 + shared-bvh 286 green; T1 smoke 999 dB + RC oracle gate PASS. ALL re-run by the lead. **So the RC merged-BVH decouple was RIGHT all along — it was BLOCKED by F-RC1, not wrong.**
- **History (the blocker, now resolved):** Implemented + CPU/byte-validated 2026-06-04, then REVERTED (not committed) because it couldn't be render-validated while F-RC1 stood. **F-RC1 is now FIXED** (the stride bug — which, NOTE, was the REAL cause of F-RC2's `rc-core-bvh-ab.ts` CHECK-2 17.75 dB divergence; the "harness floating-point edge-grazing" explanation in that harness was WRONG — the stride bug corrupted both THREE and core paths differently). So F-RC2 is re-attemptable: re-apply the decouple (recoverable from session `97e46b32`) + run a now-STABLE converged RC A/B (the traversal is correct + has a GPU smoke). The packing (`buildRCSceneBVHFromCore` + `packCascadeMaterialsFromCore` with the `toProductionEmissiveRadiance` ei-collapse fix — RC's probe kernel DOES read `mat.emissive`, so the fix is load-bearing here unlike DDGI) is correct: CPU set-equivalence (8 tests, incl. ei=4 → `emissive·1` and the `thickness→0.1` floor) + the GPU CHECK 1 byte-identical 999 dB. The blocker is purely that swapping `buildSceneBVH(sceneRoots)` → `mergeWorldSpaceFromCore(scene)` changes the tree shape, which F-RC1's unreliable RC traversal renders differently — so the decouple's converged render can't be proven equivalent.
- **Note on production reach:** unlike DDGI (whose standalone path is dead in-engine), production RC merged mode DOES reach `buildRCSceneBVH` for single-mesh core scenes (`HybridEngineGiPropagation` merged branch + `allowRcSceneRebuild`). So this is a real production decouple once unblocked.
- **Fix:** after F-RC1, re-apply the decouple (the diff is recoverable from session `97e46b32`) + gate it on a conclusive multi-node converged A/B. Mirrors the landed emitter (`46a0078`) + standalone-DDGI (`15070cd`) decouples.
- **Acceptance:** RC core path wired into `propagateBvhToGiSubsystems` (core-first when a core Scene is present) with a converged RC A/B (THREE vs core) at high PSNR on a multi-node scene.

### F-TLAS1 — ReSTIR-TLAS `bvhIndex.xyz` vertex lanes packed STRIDE-3 from a STRIDE-4 buffer (FIXED 2026-06-04)

- **STATUS: FIXED + validated (commit lands with this entry).** `buffersFromScenePack` now feeds the packer a stride-3 extraction (`triIndices3[t*3+k]=geo.indices[t*4+k]`). PROVEN CORRECT end-to-end: a CPU brute-force oracle (`wsl-gpu scripts/restir-tlas-bvh-bruteforce-ab.ts` — independent all-triangles closest-hit using the stride-4 ground truth + Möller-Trumbore, no BVH) gives the FIXED GPU `traceTlasFirstHit` a **100.00%** match (2304/2304 rays) vs **59.68%** for the re-introduced bug (Δ40 pp), identical on lavapipe + dzn. Unit-pinned (`materialPackingCoreEquivalence.test.ts` now asserts `bvhIndex.xyz==geo.indices[t*4+k]`). The T1 smoke golden (`wsl-gpu captures/golden/hybrid-{lavapipe,dzn}.png`) was RE-CAPTURED against the corrected render (smoke now 999 dB vs the new golden; the old golden encoded the bug, regressing 81→11 dB). The bug landed via `048a2bb` (2026-05-26); `e374a7c` preserved it byte-identically.
- **RESIDUAL — RESOLVED 2026-06-04 via wsl-gpu (NOT a hardware-Chrome host — the companion app is the GPU validator):** the 5 multi-mesh hybrid-TLAS reference renders that encoded the bug were RE-CAPTURED on dzn (RTX 4090 — the wsl-gpu deno worker DOES reach the ≥16-buffer adapter; `capture-worker/render-hybrid.ts` was generalized with `renderHybridScene(scene)` → `wsl-gpu scripts/capture-ftlas1-refs.ts`, scenes built faithfully from the real `@vitrum-examples/shared` host builders + `scenario-presets.mjs` params, verified on dzn + recognizable-geometry by-eye + the fix proven by the CPU oracle). Re-captured: `PR-hybrid/{tlas-on/PR-hybrid-tlas-10-inst,200k-static/PR-hybrid-200k-static,material-edit/PR-hybrid-material-churn}.png` + `session-20260527/hero-viewer-realtime.png` + its byte-copy `baseline/hero-viewer-realtime.png`. These are now wsl-gpu/dzn captures (byte-different from the old Chrome SHA256s — the Chrome `benchmark:pr-hybrid-refs` path can't reach a ≥16-buffer adapter in WSL, which is why wsl-gpu exists; `PR-hybrid/manifest.json` hashes updated, and those hashes are asserted by nothing). VERIFIED NOT affected: all pt-webgl/pt-webgpu refs (own packers), all `W8-rc-*` (walkaround-rc path), and anything captured < 2026-05-26.
- **BVH-index stride AUDIT (2026-06-04) — CLEAN, no new bug.** Audited EVERY storage-buffer `array<vec4u>` bvhIndex consumer for the F-TLAS1/F-RC1 stride class (CPU buffer fed to a 16-byte-stride shader read): **restir-TLAS** `packBVHIndexWFromCore(triIndices3)` (F-TLAS1-fixed); **restir-merged** `packBVHIndexW(shared.indices)` (stride-3 in, stride-3 read — correct by construction); **RC merged** `_uploadBVH` padded (F-RC1-fixed) + **RC core** (F-RC2) padded + **RC-TLAS** uses the fixed `snap.bvhIndex`; **DDGI** snapshot uses `snap.bvhIndex` + standalone pads via `padTriangleIndicesToVec4`; **pt-webgpu** uses stride-4-native `geo.indices` AND throws at upload if not 16-byte-aligned (structurally immune). `padTriangleIndicesToVec4` verified correct. **pt-webgl is OUT OF CLASS** (WebGL2 texture-packed BVH — `texelFetch1D`/`DataTexture`, no storage buffers). **HARDENING — all 3 GI subsystems now brute-force-gated on every push** (`wsl-gpu t1-smoke.mjs`): `runRcMergedOracle` (F-RC1), `runRestirTlasOracle` (F-TLAS1), AND `runDdgiOracle` — the DDGI oracle covers BOTH DDGI upload paths (TLAS `rebuildProbeBvhFromRestir(snap.bvhIndex)` + merged standalone `rebuildProbeBvhFromScene`→`padTriangleIndicesToVec4`), each FIXED 100% vs an independent CPU all-triangles ground truth, discriminating BOTH stride sub-classes (stride3of4 = F-TLAS1 + rawStride3 = F-RC1). pt-webgpu self-guards via its upload-time stride-4 throw; pt-webgl is out of class (texture-packed). So EVERY storage-buffer `array<vec4u>` BVH-index consumer is now gated — closing the missing-standing-gate gap that let F-TLAS1 + F-RC1 go latent for weeks.

- **Severity:** **CONFIRMED HIGH (production geometry).** Runtime impact RESOLVED 2026-06-04 by a decisive test: applying the fix locally regresses the T1 smoke from **81→11 dB (mae 0.22)** on BOTH lavapipe and dzn (the two fixed renders still agree, cross-check 26.81 dB). So the committed golden ENCODES the bug, and the production Cornell-DDGI render (core TLAS path) IS materially corrupted — not latent. The pre-fix render is "subtly wrong, not obvious garbage" (DDGI is diffuse GI over probe rays — mangled tri vertices skew the irradiance without an obvious silhouette break), which is why it went unnoticed and the non-regression smoke passed against a buggy golden. Surfaced by the ReSTIR-material decouple agent; **all links + the impact independently verified by the lead.** The fix (feed the packer the stride-3 `geo.indices[t*4+k]` extraction — same as `bvhIndicesStride3`) is provably correct (correct global indices → correct vertices); it CHANGES the render, so it needs the T1 golden (and any other ReSTIR/DDGI reference renders) RE-CAPTURED.
- **The bug:** `buffersFromScenePack` (`restir/sceneBvhFromCore.ts`) packs `bvhIndex` via `packBVHIndexW*(geo.indices, …)`. `geo.indices` (`ScenePackResult.indices`) is **stride-4** (`scenePack.ts:551` "stride 4 (vec4u/triangle)"; allocated `*4` at :593; written `[tri*4+i]` at :628). But `packBVHIndexWTri` (`restir/packingHelpers.ts:113-115`) reads `indices[tri*3 + k]` — a **stride-3** read. So `bvhIndex.xyz[tri]` = `geo.indices[tri*3+k]`, which is correct ONLY for tri 0 (3·0==4·0); tri ≥1 reads across triangle boundaries (e.g. tri 1 gets `[geo.indices[3],[4],[5]]` = `[tri0's .w pad, v0_tri1, v1_tri1]` instead of `[v0,v1,v2]_tri1`). The new `packBVHIndexWFromCore` faithfully reproduces this (which is WHY the decouple's byte-identity test + T1 smoke passed — both sides share the same wrong read).
- **GPU consumes it for geometry:** `shared-bvh/wgsl/bvhIntersect.wgsl.ts:327-334` — `idx = (*bvh_index)[triIdx].xyz; intersectTriangle(bvh_position[idx.x], [idx.y], [idx.z])`. `bvh_index` IS the packed buffer (the shader also reads `idxEntry.w` for material color at :340, a lane only `packBVHIndexW` produces). Reached in production TLAS mode via `tlasTraversal.wgsl.ts:190 bvhIntersectFirstHitAtRoot`. **Merged mode is FINE** — it feeds genuine stride-3 `shared.indices` (`buildSceneBVH`), so the bug is TLAS-path-only (multi-mesh / instanced scenes; `resolveReSTIRBvhMode` returns 'tlas' for >1 mesh or any instanced).
- **THE PUZZLE — RESOLVED 2026-06-04: answer (a).** The goldens were captured WITH the bug, so the T1 non-regression smoke passes against a buggy golden — which **indicts the T1 smoke's correctness value for the ReSTIR/DDGI path** (it has been pinning corrupted geometry). This means the fix's golden re-capture MUST be validated against an independent CORRECT reference (CPU brute-force / known-geometry oracle), not just "re-snapshot what the fix produces". The decisive evidence: fix-applied → smoke 81→11 dB (see Severity).
- **Fix (small) + validation (involved):** pass the already-correctly-extracted stride-3 indices to the packer in the TLAS path — `bvhIndicesStride3` is built via `geo.indices[t*4+k]` at `sceneBvhFromCore.ts:~154` (reorder so it precedes the `packBVHIndexW*` call, or add a stride param). Add a unit test pinning `bvhIndex.xyz[tri] == [geo.indices[tri*4+0..2]]`. **The fix CHANGES the render** (corrects geometry) → the T1 golden must be RE-CAPTURED against a *correct* reference (a CPU brute-force / known-geometry oracle), NOT the existing (possibly-buggy) golden. Single-mesh / tri-0-only scenes mask it — use a transformed multi-mesh ReSTIR-TLAS scene.
- **Acceptance:** the puzzle is resolved (real-vs-latent established with evidence); `bvhIndex.xyz` matches the stride-4 source per a unit test; a multi-mesh ReSTIR-TLAS render matches a CPU brute-force reference; the T1 golden is re-captured if it was buggy.
