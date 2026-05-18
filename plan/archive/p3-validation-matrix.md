# P3 validation matrix — drive on RTX 4090 via Claude-in-Chrome

The P3 deferred-items remediation made changes across all rendering backends. Most are byte-equivalent at the data-flow level; a few are real algorithmic changes that need visual verification on hardware. This matrix gives you a step-by-step validation plan you can drive through Claude-in-Chrome on the 4090.

> **Per memory rule:** No visual claims from single screenshots. Path-traced and temporal pipelines have transient states. Drive validation by **telemetry counters** (sample-per-second, last-GPU-timings, accumulated-sample-count, etc.) and **A/B comparisons against a known-good baseline render**, not eyeballs on isolated frames.

## What changed in P3 (one-liners per commit)

- **P3-A.1** — `PTEngineWebGL2` class extracted from `pt-webgl/src/index.ts` (832→43 lines) into its own file. Pure mechanical move.
- **P3-A.2** — `passIdx:number` replaced with named-pass `PassLayout` in walkaround-hybrid. Fixes a latent telemetry bug where the GPU-timing dict labelled passes generically (`denoise-5`..`denoise-11`) instead of by their real name (`welford-temporal`, `svgf-atrous-N`, `atrous-N`).
- **P3-B.1a** — `uploadSceneBuffers.ts` (1058 lines) split into `materialPacking.ts` + `environmentPacking.ts` + `emitterPacking.ts` + thin orchestrator. Pure mechanical, byte-equivalent.
- **P3-B.1b** — `first*` legacy single-light extractors deleted from pt-webgpu; WGSL `FrameParams` repacked to drop `.w`-stuffing tricks (env map dims, MNEE knobs, caustic strategy, hasEnvironmentMap, light-active flags). Read sites updated to read from the light storage arrays at index 0.
- **P3-B.2** — `pathTraceBruteforce.wgsl.ts` `main()` (~417 lines) split into four sub-helpers: `decodeMaterial`, `sampleNextBounceDirection`, `russianRoulette`, `accumulateFrame`. Multi-light NEE loop and thin-film TMM blocks intentionally NOT extracted (too much shared local state).
- **P3-B.3** — Cross-engine `extractThreePbrScalars` helper added to `@vitrum/three-bindings`. DDGI `_uploadMaterials` refactored to use it. Byte-equivalent material upload.
- **P3-B.4** — `three/webgpu` `StorageTexture` import dropped from `probeGrid.ts` + `probeUpdatePass.ts`. ProbeGrid now holds plain `AtlasTextureSlot {width, height}` records; the compute path is pure raw WebGPU. TSL coupling isolated to `applyDDGIShading.ts`.
- **P3-C.1** — Sprint 9 adaptive sampling (sampleBudget + resolve passes + RIS/shade modifications). Real algorithmic change — biggest verification surface.
- **P3-C.2** — Orphaned `PPG_SAMPLE_WGSL` deleted. The live `shadePpgGuide.wgsl.ts` already provides the path-guided indirect bounce with real `@group(3)` bindings.

## Validation steps

Each step lists a route, an action, and the telemetry signals to check.

### Step 1 — pt-webgl Cornell box parity (P3-A.1 + P3-A.2)

**Route:** `examples/cornell-box` (`npm run dev --workspace examples/cornell-box`)

**Action:** Open `http://localhost:5173/` (or whatever Vite picks). Wait for the path-traced render to converge (~60s of accumulation).

**Signals:**

- HUD shows `samples/sec` non-zero and stable. Expected: similar to baseline (whatever the user's reference render targets).
- HUD shows `sample count` climbing monotonically until `samplesTarget` is reached.
- DevTools console: no shader-compile errors. No `WebGL: ` warnings beyond the expected ones (RAFFLE / extension probes).

**Acceptance:** Final converged image matches a known reference render (capture once locally, then re-render after each P3 change to A/B). The renderer string and `qualityMode` in the HUD telemetry block match the baseline.

**Risk surface (low):** P3-A.1 is mechanical class extraction; P3-A.2 only changes a label-to-slot mapping in walkaround-hybrid, not pt-webgl. This step is a smoke test for the broader pt-webgl path.

---

### Step 2 — walkaround-hybrid pass-timing labels (P3-A.2)

**Route:** `examples/two-engines-one-scene` (`npm run dev --workspace examples/two-engines-one-scene`)

**Action:** Switch to the walkaround engine (URL param: `?engine=walkaround-hybrid`). Open DevTools, find the GPU-timing dev panel (whatever surface displays `lastGpuTimings`).

**Signals:**

- The timing dict keys should now be the real pass labels: `ris`, `temporal`, `spatial-1`, `spatial-2`, `shade`, `ppg-update` (only if PPG enabled), `welford-temporal`, `svgf-variance`, `svgf-atrous-0`..`svgf-atrous-4`, `temporalAccum`, `composite` — NOT the old generic `denoise-5`..`denoise-11`.
- Each ms value is plausible (welford-temporal < 1ms, svgf-atrous-N each ~0.5–2ms, shade is the dominant cost).
- Switching denoiser mode (`?denoiser=atrous` vs `?denoiser=svgf`) should swap the SVGF labels for `atrous-0` / `atrous-1` / `atrous-2`.

**Acceptance:** Labels match the actual passes running for the active denoiser config. No `denoise-N` slot labels remain.

**Risk surface (low):** Tested by 17 unit tests in `__tests__/passLayout.test.ts`. The only risk is a runtime-only mismatch with the actual GPU pass order, which the dev panel will surface directly.

---

### Step 3 — pt-webgpu rendering parity (P3-B.1a + P3-B.1b + P3-B.2)

**Route:** `examples/two-engines-one-scene` with pt-webgpu engine (`?engine=pt-webgpu` — verify the host UI exposes this; if not, the import-only anchor in `main.ts:12` means the engine is still loaded but not driving the canvas. In that case open the scene through whatever pt-webgpu probe exists, OR temporarily wire `createPTEngine_WebGPU(...)` into the engine-switch handler).

**Action:** Render the scene with multiple lights (1 point + 1 spot + 1 rect-area + 1 mesh-area emitter) + HDRI environment. Let it converge.

**Signals to check (telemetry, not eyeballs):**

- No shader-compile errors. The new `FrameParams` struct has 20 u32 + 4 vec4f + 3 mat4x4f (336 bytes used, 512 buffer); if the WGSL string fails to compile, the engine init throws on `getCompilationInfo`.
- Per-light accumulators (point/spot/rect/mesh) should each appear in the final image. The `first*` light extractors are gone, but the multi-light loop ALREADY iterates the same storage arrays at the same offsets — the change is which slot the WGSL reads "light 0" from.
- HDRI sky aperture sampling still works (use a scene with a textured environment, look at glossy reflections). Sample count climbs steadily.
- Photon-map mode (`?caustic=photon-map`) emits caustics under glass. The photon launcher reads `pointLights[0]` / `spotLights[0..1]` now instead of `params.pointLightPos` / `params.spotLightPos` — verify by toggling a scene with one point light + one glass primitive and watching for the caustic spot.
- MNEE mode (`?caustic=manifold-nee`) finds the same caustic patches. The MNEE knobs now read from `params.mneeMaxIterations` (u32) instead of `params.rectAreaU.w` (f32). Verify by toggling between modes; output should be similar.

**Acceptance:** A/B against a saved reference render of the same scene with the previous commit. Sample-per-second should be within ±5% (no algorithm change).

**Risk surface (HIGH for B.1b):** This is the biggest non-pt-webgl risk. Every `params.*.w` read site was rewritten; if any site was missed or wired wrong (e.g. reading `mneeMaxIterations` from the wrong u32 slot), the affected feature will silently degrade. The `frameParamsLayout.test.ts` pins the offsets at the TS level, but the WGSL bindings need a real-device run.

**Risk surface (MEDIUM for B.2):** The bounce-loop refactor preserves all the local var reads/writes (the agent verified each), but a subtle scoping bug in the WGSL split could cause variables to read stale values. Watch for: emissive surfaces appearing dimmer, glossy reflections losing the thin-film tint, or subsurface scattering looking weirdly cool.

---

### Step 4 — walkaround DDGI material upload parity (P3-B.3 + P3-B.4)

**Route:** `examples/two-engines-one-scene` walkaround engine, with a scene containing a glass primitive (transmission > 0) so the DDGI material's `isGlass` flag exercises.

**Action:** Open the DDGI debug view (if a probe-visualization mode exists). Otherwise watch the indirect-light contribution on a non-emissive surface near the glass.

**Signals:**

- Glass primitives still tint indirect light correctly (the `attenuationColor` field flows through `extractThreePbrScalars` → DDGI buffer).
- Probe atlases allocate at the right size (check `probeGrid.params.irradianceAtlasW` × `irradianceAtlasH` in the dev panel).
- No "three/webgpu storage texture not allocated" errors; the new `AtlasTextureSlot` pattern lazily creates the GPUTexture inside `probeUpdatePass._getOrCreateAtlasTexture` per slot.

**Acceptance:** DDGI-lit indirect bounce visually matches the pre-P3 baseline (capture before/after).

**Risk surface (low):** B.3 is byte-equivalent — `extractThreePbrScalars` returns the same defaults as the inline code did. B.4 is a refactor of the texture-handle indirection that the compute path doesn't actually depend on (TSL path is preserved via `applyDDGIShading.ts`'s own StorageTexture cache).

---

### Step 5 — walkaround Sprint 9 adaptive sampling (P3-C.1) **— BIGGEST RISK SURFACE**

**Route:** `examples/two-engines-one-scene` walkaround engine.

**Action:** Render a scene. Sprint 9 adds two new passes (`sample-budget` before RIS, `resolve` after temporalAccum) and modifies RIS + shade to act on the per-pixel tier.

**Signals (telemetry — these matter most):**

1. **Pass layout:** GPU-timing dict now has `sample-budget` and `resolve` as keys, in addition to all the previous ones. New slot count should be 17 (was 15).
2. **Sample budget output:** `tierTexture` (r32uint) should be written every frame. The shader writes tier ∈ {1, 2, 4} based on the welford variance estimate. If the user has a tier-visualization mode, look for tier=4 (red?) at undersampled regions and tier=1 (green?) at converged regions.
3. **Checkerboard pattern:** With `frameParity = frameCount & 1`, half the pixels are shaded each frame. The other half come from the resolve pass via motion-vector reprojection.
4. **Camera-motion regression:** When the camera moves, motion vectors should produce non-zero values, and the resolve pass should successfully reproject the previous-frame radiance. Watch for ghosting trails — those are expected per the Sprint 9 DoD ("ghosting acceptable per the existing variance-clamped AABB").
5. **Static scene regression:** With the camera still, the converged image should match the pre-Sprint-9 baseline (full-resolution shading). Welford variance settles, tier transitions to all-1, and effectively all pixels are reused from the previous frame.

**Acceptance:**

- A/B static render against P3-B.2 baseline: should converge to the same image (within ±2% pixel difference).
- Performance: sample-per-second should be HIGHER than baseline once tier=1 dominates the frame (the whole point of adaptive sampling).

**Risk surfaces:**

- **RIS modification:** The "tier=1 → reuse prev reservoir" branch is the deepest semantic change. If RIS resamples even with tier=1, perf gains evaporate; if it reuses too aggressively, dark/light spots appear.
- **shade.wgsl checkerboard:** The "gap pixel" branch writes a sentinel to hdrColor. The denoise chain (svgf-variance + atrous) reads from hdrColor. If the denoiser interprets the sentinel as "real" radiance, the denoised output gets corrupted streaks. If the user's agent implementation chose to make the resolve pass run BEFORE the denoise instead of after, watch for which-pass-comes-when in the encoder.
- **Motion vectors:** Resolve relies on accurate motion vectors. These are written by RIS's gbuffer-output. Verify they're non-zero during camera motion.
- **Ghosting:** Expected per spec; should fade out within ~30 frames after camera stops.

---

### Step 6 — mechanical-checks sanity rerun

After each P3 commit lands, the user (or CI) can rerun:

```
npm run typecheck           # 9 packages, no errors
npm test                    # workspaces; should be ~661–680 passing + 3 skipped
npm run fork-shader-smoke   # green
```

These give a fast sanity check that nothing broke at the TS / WGSL-compile level. They DO NOT validate visual correctness — that's what the steps above are for.

---

## Triage tree

If something looks wrong:

1. **Run mechanical checks first.** If typecheck or tests fail, the regression is in TS code; check the last commit's diff.
2. **Check the GPU-timing dict.** If a pass is missing from `lastGpuTimings`, it's not being dispatched (pipeline compile error, or the encoder is skipping). Check DevTools console.
3. **Compare against the previous commit.** `git checkout <prev>` → run again → A/B. The P3 commits are small enough that bisecting by reverting one and re-rendering should pinpoint the regression.
4. **Look for shader compile errors in console.** Both pt-webgpu and walkaround-hybrid surface compile errors as runtime exceptions on engine init.
5. **For Sprint 9 weirdness (Step 5):** the simplest disabling toggle is to revert the C.1 commit. The rest of P3 stands without it. Other P3 work is structural and shouldn't gate on this.

---

_Generated as part of the P3-V task — drive via Claude-in-Chrome on the 4090 to validate the P3 series before pushing to origin._
