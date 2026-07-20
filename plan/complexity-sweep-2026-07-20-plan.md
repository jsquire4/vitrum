# Complexity-sweep remediation plan — 2026-07-20

**Source report:** `~/.claude/projects/-home-jsquire4-projects-vitrum/memory/in-flight-sweep-2026-07-20.md`
**Base:** `main` clean @ `207a1dea` | **Constraints:** `CLAUDE.md` (naga gate, T1 GPU smoke, wsl-gpu oracle hardcoded paths, no-push, no-defer)
**Planner verification:** every task below was source-verified by opening the cited file:line on 2026-07-20. Rejected findings and their evidence are in the final section.

## Locked ground rules (apply to every wave)

- **No deferring.** Every non-rejected finding gets a task in this plan. Intentional/mitigated findings are recorded as accepted-intentional with rationale, not silently dropped.
- **Behavior preservation is non-negotiable.** Each task states its strategy: (a) **byte-identity golden** (WGSL/UBO string goldens stay green), (b) **characterization test added first** (write the pinning test, see it green on old code, then refactor), or (c) **pure-mechanical with existing coverage** (name the covering suite). Radiometric changes require a GPU A/B per the testing protocol.
- **composeWgsl ordering constraint.** `composeWgsl` emits shared `WgslModule`s *before* consumer bindings. Any shared helper that references a **consumer binding** (e.g. `scene`, `ubo`, atlas textures) MUST be a **raw-string template interpolated into the consumer body** (the `temporalGiCommon.wgsl.ts` / `spatialGiCommon.wgsl.ts` precedent), NOT a `WgslModule`. Binding-free pure math MAY be a `WgslModule`.
- **wsl-gpu oracle hardcoded paths.** The T1 oracles in `~/projects/wsl-gpu/scripts` import these wh production modules by **absolute path** (verified 2026-07-20):
  `HybridEngineRC.ts`, `index.ts`, `ddgi/DDGI.ts`, `ddgi/ddgiAtlasLayout.ts`, `ddgi/ddgiSampleWgsl.ts`, `ddgi/probeUpdateBvhBuffers.ts`, `ddgi/wgsl/ddgiSH.wgsl.ts`, `ddgi/wgsl/probeUpdateBlend.wgsl.ts`, `ddgi/wgsl/probeUpdateRays.wgsl.ts`, `rc/bvhCore.ts`, `restir/bvhCompute.ts`, `restir/bvhCore.ts`, `restir/restirBvhSnapshot.ts`, `restir/sceneBvhFromCore.ts`, `neural/wgsl/relu.wgsl.ts`, `skin/gpuSkinBvh.wgsl.ts`, plus `src/rc/` and `src/shaders` directories.
  **Rule:** if a task renames/moves ANY file on that list, it MUST (1) `grep -rn '<old-path>' ~/projects/wsl-gpu/scripts`, (2) edit each hit to the new path, (3) note this in the task's on-disk proof. **`packingHelpers.ts`, `reservoirGiLayout.ts`, `emitterList.ts`, `consumedMaterialFields.ts` are NOT on the oracle list** — moving symbols out of them is oracle-safe (but moving the files themselves is not; keep the files, extract symbols into new modules and re-export).
- **run-ptwebgl2-h1** (~60s, bit-deterministic) MUST run for any task touching `pt-webgl2` source — mock-GL is blind to texture/format/readback/sampler-order classes.
- **`grep -c` exits 1 on zero matches** — never chain it with `&&`. Use `grep -c ... || true`.

## Per-wave agent-dispatch protocol

1. **File-disjoint dispatch only.** Within a wave, no two agents touch the same file or make interacting signature changes to the same package. Task groups below are pre-partitioned for disjointness.
2. **No agent git usage — including `git stash`.** Agents edit the working tree only. The orchestrator owns all commits.
3. **On-disk proof required.** Each agent's final message MUST quote the exact new/changed lines it wrote (file:line + verbatim quote) — the "marker" for that deliverable.
4. **Orchestrator verifies before commit.** After a wave, the orchestrator `grep`s one unique marker string per deliverable (e.g. the new function name at its new site) to confirm the edit landed on disk, THEN runs the wave gate, THEN commits. (Parallel-agent edits have silently vanished 3× historically — this greps them.)
5. **Neutral prompts.** Dispatch states the problem + the prescribed fix from this plan; no "do it fast" / pre-decided tradeoffs.

---

# USER DECISIONS — ALL FIVE RESOLVED (2026-07-20)

All five decisions are resolved by the user; nothing below is blocked. Summary:
- **D1 (thin-film limit): Option B** — keep 8 (pt-webgpu) / 35 (pt-webgl2); declare per-backend `thinFilmLayerLimit` in `BackendSupportDetails` + emit a `thin-film-layer-limit-exceeded` structured warning. WGSL `8u` unchanged. Implemented in **T1-1**.
- **D2 (BDPT default): raise `BDPT_SAFE_DEFAULT_LIGHT_BOUNCES` 1→2 unconditionally** (`index.ts:373`) + re-pin frameParamsPacker default goldens + loop pin test + JSDoc caveat + GPU A/B. Implemented in **R4/V2-5**.
- **D3 (HDR emissive): build it** — new task **T1-6** (dedicated `rgba16float` emissive array in pt-webgpu).
- **D4 (naga/Firefox): Option B** — keep `ptr<storage>` shipping; add a verbatim-compile tracked-metric gate mode + document Chromium-only + file the refactor as a tracked road-to-100 program (do NOT refactor). Implemented in **T8**.
- **D5 (neural preprocessing): align training to runtime** — decode normals `[0,1]→[-1,1]`+renormalize in `train.py`; switch color targets to HDR in `capture-dataset.mjs`. No runtime change. Implemented in **R7/V3-7**.

Per-decision detail (state, choice, prescriptive tasks, verification) follows.

## D1 — thin-film layer limit (pt-webgpu 8 vs pt-webgl2 35) — **RESOLVED: Option B (per-backend declared limit + truthfulness warning)**
- **State (verified 2026-07-20):** silent capability mismatch. `pt-webgpu/src/scene/materialPacking.ts:4` `THIN_FILM_LAYER_LIMIT = 8`; `pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts:1130` `const THIN_FILM_LAYER_LIMIT = 8u;` (pinned by `packages/pt-webgpu/src/__tests__/wgslContract.test.ts:318` — `expect(PT_WEBGPU_TRACE_WGSL).toContain('const THIN_FILM_LAYER_LIMIT = 8u;')`; NOTE: the earlier draft's `src/wgsl/__tests__/` path was wrong — the file is `packages/pt-webgpu/src/__tests__/wgslContract.test.ts`); `pt-webgl2/src/scene/materialsTexture.ts:76` `const THIN_FILM_LAYER_LIMIT = 35;` (consumed at :613, :690, :691). A scene with 9–35 thin-film layers renders differently per backend, no diagnostic.
- **DECISION — Option B (keep 8 vs 35; declare per-backend + warn on exceed):** do NOT reconcile to a shared scalar value. Keep pt-webgpu at 8 and pt-webgl2 at 35 (no WGSL stride/loop widening, no fidelity regression, `wgslContract.test.ts:318` stays pinned at `8u`). Instead:
  1. **Declare a per-backend `thinFilmLayerLimit` in `@vitrum/core` `BackendSupportDetails`.** `BackendSupportDetails` (`packages/core/src/engine/capabilities.ts:78–107`) currently carries no numeric-limit rows — add an additive optional numeric field `readonly thinFilmLayerLimit?: number;` to the interface (additive/optional → back-compat, existing hosts unaffected). Populate it in the per-backend support-details literals in `packages/core/src/engine/promiseLedger.ts` (the pt-webgpu vs walkaround/pt-webgl2 detail objects near the `thinFilmStack` rows at `promiseLedger.ts:501/599/725`): pt-webgpu → `thinFilmLayerLimit: 8`, pt-webgl2 → `thinFilmLayerLimit: 35`. (Read `promiseLedger.ts` and place the field in each backend's `BackendSupportDetails` literal; keep the source-of-truth constants importable if the backends already export them, else keep the numbers co-located with the existing `thinFilmStack` mode row per backend.)
  2. **Emit a structured truthfulness warning** when a scene's thin-film layer count exceeds the active backend's declared limit. At each backend's thin-film pack site (`materialPacking.ts:128` `Math.min(thinFilmLayers.length, THIN_FILM_LAYER_LIMIT)`; `materialsTexture.ts:613/690`), when `thinFilmLayers.length > THIN_FILM_LAYER_LIMIT`, push a structured warning (reuse each backend's existing structured-warning channel — the same channel the material-texture-array warnings drain through) with a stable code (e.g. `thin-film-layer-limit-exceeded`) naming the requested count, the active-backend limit, and that excess layers are dropped. This closes the *silent* part: the host learns the layers were truncated and by which backend.
- **T1-1 impact (make concrete):** the shared **scalar-derivation** extraction in T1-1 proceeds unchanged (helpers hoisted to `@vitrum/shared-samplers`). The **limit** is NOT a shared const — it becomes **per-backend data sourced from `BackendSupportDetails`**, NOT a shared `THIN_FILM_LAYER_LIMIT` constant. The per-backend `THIN_FILM_LAYER_LIMIT` locals (materialPacking.ts:4 = 8, materialsTexture.ts:76 = 35) and the WGSL `= 8u` stay in place. T1-1's thin-film sub-task is now: (a) add the `thinFilmLayerLimit` field to core + populate both backends; (b) wire the exceed-warning at both pack sites; (c) add pin-tests.
- **Verification (this decision's tasks carry these):**
  - Core: `npm run typecheck`; add a core pin-test asserting each backend's `supportDetails.thinFilmLayerLimit` matches its packer constant (pt-webgpu 8, pt-webgl2 35) — fail on future drift.
  - pt-webgpu: `wgslContract.test.ts:318` stays green at `8u` (unchanged); a test asserting a >8-layer thin-film scene emits the `thin-film-layer-limit-exceeded` structured warning with the requested count + limit 8.
  - pt-webgl2: a test asserting a >35-layer scene emits the warning with limit 35; **run-ptwebgl2-h1** (pt-webgl2 packing touched) — anchor unchanged (warning-only, no texel-byte change for ≤35-layer scenes).
- **No longer blocks anything** — all of T1-1 (scalar extraction + this limit sub-task) is unblocked.

## D2 — BDPT default policy (V2-5) — **RESOLVED: raise `BDPT_SAFE_DEFAULT_LIGHT_BOUNCES` 1 → 2 unconditionally**
- **State (verified 2026-07-20):** `const BDPT_SAFE_DEFAULT_LIGHT_BOUNCES = 1;` at `packages/pt-webgpu/src/index.ts:373` (confirmed on read; neighbours `EXPERIMENTAL_MAX_BOUNCES = 8`, `BDPT_MAX_LIGHT_BOUNCES = 8` at 371–372). The kernel connection loop `for lvi=1u; lvi<maxLv` with `maxLv=1` performs **zero light-path connections at defaults** → BDPT is inert unless the host raises light bounces.
- **DECISION — raise the default to 2, unconditionally (no experimental-ack gate on the raise itself):** change `BDPT_SAFE_DEFAULT_LIGHT_BOUNCES` from `1` to `2` at `index.ts:373`. With `maxLv=2` the kernel connection loop executes `lvi=1` → real light-path connections happen out of the box. BDPT remains an opt-in feature (`bdpt:true`); this only fixes its default light-bounce count so it is not silently inert when enabled.
- **R4 BDPT sub-task (now concrete — no longer warn-vs-raise):**
  1. **Change the default:** `index.ts:373` `1 → 2`.
  2. **Update golden tests pinning defaults:** the `frameParamsPacker` golden tests that pin the default packed FrameParams (the default `bdptLightBounces`/`maxLv` byte) must be re-pinned to reflect `2`. Read the frameParamsPacker golden/pin tests (`packages/pt-webgpu/src/**/frameParamsPacker*` + any FrameParams byte golden) and update the expected default byte; document each re-pin as an intended default change (not a regression).
  3. **Add a pin test asserting `maxLv=2` ⇒ the kernel connection loop executes `lvi=1`** — i.e. a test that at defaults the packed light-bounce count is `2` and that the connection loop is non-empty (assert the packed `maxLv`/`bdptLightBounces` field is `2`, so `for lvi=1u; lvi<maxLv` runs at least once). This guards against a future silent revert to inert BDPT.
  4. **Document the known-bias caveat in the option's JSDoc** (BDPT sidecar is additive, not eye-path-reweighted; multi-vertex BDPT is research-mode per CLAUDE.md) and **emit the existing experimental warning path** for `bdpt:true` (BDPT is already flagged experimental — keep/route that warning; do NOT add a new "inert at <2" warning since the default is now 2).
  5. **GPU A/B (testing protocol):** capture a before/after reference render on a scene where BDPT contributes (e.g. a small emitter behind a partial occluder / hard-to-reach light path) to confirm the default change produces the intended connection contribution and no regression on non-BDPT scenes. **Note:** old pt-webgpu baselines embed the inert-BDPT (`=1`) default — any BDPT-enabled baseline captured before this change is stale and must be re-captured on the next refresh.
- **No longer blocks R4** — the SPPM fallback (V2-1) and atomic setScene (V2-2) proceed in parallel; the BDPT sub-task above is fully specified and unblocked.

## D3 — HDR emissive texture array (V2-6) — **RESOLVED: build it — new task T1-6 (separate rgba16float emissive array in pt-webgpu)**
- **State (verified 2026-07-20):** material textures are 8-bit only — exactly two `texture_2d_array`s: sRGB `rgba8unorm-srgb` (baseColor **+ emissive**) and linear `rgba8unorm` (normal + ORM), created at `pt-webgpu/src/scene/uploadSceneBuffers.ts:2018–2030`. Emissive shares the sRGB 8-bit array (`materialTextures.ts:491` `emissiveIdx` indexes the sRGB layer space; `materialTextures.ts:24/384` "sRGB variant — baseColor + emissive"). In WGSL, emissive samples the sRGB array binding `@group(3) @binding(3) var materialTextures: texture_2d_array<f32>` (`material.wgsl.ts:394`) via `sampleMaterialEmissive`/`sampleMaterialLayer` at `material.wgsl.ts:578–582`. → **textured HDR emissive values are clamped to [0,1]** (8-bit sRGB). Constant emissive is unaffected.
- **DECISION — build the rgba16float emissive array feature in this campaign as new task T1-6** (placed alongside the T1 material work; see the new **T1-6** section under WAVE T1). Prescriptive files/anchors are specified there (all verified on read 2026-07-20).
- **No longer blocks Wave R** — T1-6 lands in the T1 wave; Wave R proceeds independently.

## D4 — naga / Firefox portability (V2-7) — **RESOLVED: Option B (keep `ptr<storage>` shipping; track + document + verbatim-compile gate mode)**
- **State (verified 2026-07-20):** walkaround production shaders ship with `ptr<storage>` fn params; the shader gate validates `applyNagaFix`-patched *derivatives*, NOT the shipped source. In `tools/shader-gate/gate.mjs` the walkaround/RC entries are wrapped in `applyNagaFix(...)` (confirmed: `gate.mjs:321` `wgsl: applyNagaFix(raw)`, `:540` `applyNagaFix(PROBE_RAY_CAST_WGSL)`, `:739` `applyNagaFix(desc.code)`; the `ptr<storage>` intent is called out in comments at `gate.mjs:314` and `:524`). `ptr<storage>` fn params need `unrestricted_pointer_parameters` (no WebGPU enable path) → Tint/Chrome accepts, **naga/Firefox REJECTS** → shipped walkaround shaders are **Chromium-only** and the gate green-lights it silently. (pt-webgpu shaders ARE gate-validated verbatim; this is walkaround-only.)
- **DECISION — Option B: keep `ptr<storage>` shipping; do NOT refactor the traversal core in this campaign.** Instead make the Chromium-only reality honest, tracked, and measured. This plan file only **PRESCRIBES** the following doc/gate edits (the implementing wave performs them):
  1. **Verbatim-compile gate mode in `tools/shader-gate/gate.mjs`:** add a mode that ALSO attempts *un-patched* (verbatim, no `applyNagaFix`) compilation of the walkaround/RC shaders and **REPORTS** (does NOT fail) the naga-incompatibility count as a tracked metric — e.g. "walkaround/RC = Chromium-only, N shaders reject on naga verbatim". The existing `applyNagaFix`-patched pass stays the gating (green) path; the verbatim pass is a tracked, non-fatal metric. **Scheduled in T8** (tooling wave — added to T8's task list; see T8).
  2. **Document the Chromium-only constraint** in `packages/walkaround-hybrid/README.md` (verified present) and `plan/renderer-fidelity-matrix.md` (verified present): state that walkaround production shaders use `ptr<storage>` fn params requiring `unrestricted_pointer_parameters` and are therefore Chromium-only (naga/Firefox reject).
  3. **File the `ptr<storage>` refactor as a tracked program entry in `plan/road-to-100.md`** (verified present) — record it as scoped/known; do NOT plan the refactor itself here (it remains a future HIGH-risk, T1-smoke-gated program).
- **Scope guard:** this plan PRESCRIBES the doc/gate-visibility edits ONLY. It does NOT plan or perform the traversal-core `ptr<storage>` refactor.
- **Verification:** the gate-mode edit — run `gate.mjs` (both modes) before/after, confirm the gating pass exit code is unchanged and the new verbatim metric prints a count; the three doc edits are prose (verify by read). No source/shader behavior change.
- **No longer blocks anything** — Wave R proceeds; the T8 gate mode + three doc edits are added to T8 as noted.

## D5 — neural training/runtime encoding alignment (V3-7) — **RESOLVED: concrete task added in R7 (align training preprocessing to runtime)**
- **State (verified 2026-07-20):** the runtime pack shader `packages/walkaround-hybrid/src/shaders/neuralPack.wgsl.ts` decodes normals `[0,1]→[-1,1]` and renormalizes (`:57` `let nd_remapped = nd * 2.0 - 1.0;`, `:58` `select(normalize(nd_remapped), vec3f(0,1,0), ...)`) and packs the color input as **raw linear HDR** (`:60–62` `noisyOut[base+0u] = n.r;` etc., where `n` is loaded linearly from `noisyTex`). Training does NEITHER:
  - `tools/neural-denoiser-training/train.py` `_load_rgb` (`:226–230`) loads every PNG (incl. normals) as raw `[0,1]` with `np.array(img)/255.0` and no `[-1,1]` decode; `__getitem__` (`:203–223`) stacks `noisy/albedo/normal` all through `_load_rgb`, so the **normal channel is `[0,1]`, not `[-1,1]`+renormalized** — misaligned with `neuralPack.wgsl:57–58`.
  - `tools/neural-denoiser-training/capture-dataset.mjs` `renderColor` (`:572–589`) writes the color targets as **reinhard-LDR 8-bit PNG** (`:583–585` `out[i] = toByte(reinhard(c[0]))` etc.; `reinhard` at `:532`), while the runtime color input is raw linear HDR — the training color target does NOT match the runtime color encoding.
- **DECISION — align training preprocessing to the runtime encoding (R7 sub-task, no runtime change):**
  1. **train.py normal decode:** in `_load_rgb`/`__getitem__`, decode the **normal** PNG `[0,1]→[-1,1]` and renormalize (mirror `neuralPack.wgsl:57–58`): after loading the normal tensor via `_load_rgb` at `train.py:208` (and the mirror in `dry_run` `load` at `:441–443` if that path is kept consistent), apply `n = normalize(n*2 - 1)` on the 3-channel normal only — do NOT apply the decode to `noisy`/`albedo`/`clean` (those stay as-is). Cleanest: keep `_load_rgb` generic (`[0,1]`) and add the `*2-1`+normalize decode to the **normal** branch in `__getitem__` (`:208`) so only normals are remapped, matching the runtime pack.
  2. **capture-dataset.mjs color target → HDR:** switch the color captures (noisy at low-spp and clean at high-spp) from reinhard-LDR 8-bit PNG to an **HDR target loaded linearly** — e.g. write a float `.bin` (or EXR) of the linear radiance instead of `encodePNG(reinhard(...))`. Concretely: at `renderColor` (`:572–589`) stop applying `reinhard`/`toByte` for the color targets and emit the linear float buffer; adjust the write sites (`capture-dataset.mjs:656` noisy, `:659` clean) to write the HDR color format. **Keep albedo/normal as PNG** (`:657–658`) — only the two color targets go HDR. Then in `train.py` load the HDR color targets **linearly** (add an HDR loader path for `noisy`/`clean`; keep the `[0,1]` PNG path for albedo/normal).
- **Flags (record in the task):** this lands the *correct* preprocessing but **re-training is required** and **no production neural checkpoint exists yet** (CLAUDE.md) — the checkpoint + quality A/B remain **separately scoped future steps**. **No runtime changes** — `neuralPack.wgsl` is the source-of-truth reference and stays unchanged; this task only edits the two training-pipeline files.
- **Verification:** a preprocessing unit test asserting (a) a `[0,1]` PNG normal decodes to the runtime `[-1,1]`+renormalized encoding (compare against the `neuralPack.wgsl:57–58` formula), and (b) the color target is HDR/linear (not reinhard-LDR). `train.py` runs its `dry_run`/smoke path green on the new format. No naga/GPU gate (no runtime change).
- **No longer blocks anything** — scheduled as the R7 neural sub-task (touches `neural/` training code only; disjoint from the R7 denoiser fixes).

---

# WAVE R — Release blockers (correctness + lifecycle; do FIRST, before Wave 0)

**Source:** the external maturity review, source-verified by 3 Opus agents (memory file §"External maturity review — VERIFIED verdicts", V1/V2/V3). Every item below was CONFIRMED or PARTIAL on source read; REFUTED items (pt-webgpu `rgba16fReadback.ts` already-correct) are recorded as no-action.

**Why first:** these are shipping-blocking correctness/lifecycle defects (canvas freezes after handoff, dispose leaks, silent-zero caustics, sceneless-on-throw, wrong-primitive UV1, glTF spec violations, denoiser leaks). They take precedence over all refactor waves. Task groups R1–R7 are **file-disjoint** and may run as parallel agents. **Every Wave R fix ships a regression/pin test in the same task.**

**Overlaps merged (each fix appears exactly once, in the earliest wave):**
- The **bmfr try/finally** fix (V3-4 leak class) is already **Wave-0 0A** — NOT duplicated here; R7 covers only the *other* denoiser leaks (svgf one-shot, bmfr overlap knob, `webGpuTextureUpload` try/finally, oidnBridge promise cache). 0A stays in Wave 0.
- The **atomic-setScene** fix (R4) composes with the **`uploadSceneBuffers` registry-driving** task in **T2-A/D10-2** — R4 lands the try/rollback *safety*; T2-A later drives the same buffer list off `SCENE_BUFFER_REGISTRY`. Cross-referenced in both.
- `compressedWideBvh` tMin/comparator (Wave-0 0E) is unaffected by R5 (R5's shared-bvh work is TLAS-singular + UV1-range, different fns).

## R1 — engine lifecycle (packages/engine) [V1-2, V1-3, V1-6, V1-7]
All four touch `packages/engine/src/` and are internally sequenced (same files: `lifecycle/vanilla.ts`, `configureWebGpuCanvas`, `backends/ptWebgl2.ts`). **Run R1 as ONE agent** (the vanilla.ts fixes interact) — do not split V1-2/V1-3 across agents.
- **V1-2 (CONFIRMED) — dispose during auto-recreate leaks the late replacement.** `vanilla.ts` only checks `disposed` at :729 (post-install); a dispose that arrives during the `await` installs+subscribes a fresh engine that never gets torn down. **Fix:** immediately after the `await` at `:692`, re-check `disposed`; if disposed, dispose the late engine + reset the recreate state machine, and return before install/subscribe.
- **V1-3 (PARTIAL) — retry-cap branch doesn't stop RAF/dispose.** `vanilla.ts:582–589` (cap branch) does not cancel RAF or dispose; the H31-d 5-consecutive-throw self-stop (`787–801`) halts RAF ~5 frames later but the engine is never disposed. **Fix:** in the cap branch set `stopped=true`, cancel RAF, unsubscribe, and dispose the engine (mirror the self-stop teardown).
- **V1-6 (CONFIRMED runtime; type-level mitigated) — ptWebgl2 ownership-key leak.** `backends/ptWebgl2.ts:60–64` spreads `advanced` AFTER `device:gl`, and pt-webgl2 is the only backend NOT calling `stripOwnershipCriticalKeys`. **Fix (preferred):** run `advanced` through `stripOwnershipCriticalKeys` before spreading (or, minimally, spread `device` last so an `advanced.device` cannot override the owned handle).
- **V1-7 (CONFIRMED) — configureWebGpuCanvas swallows failure; walkaround goes permanently black.** The catch (`configureWebGpuCanvas:40–47`) swallows; walkaround is swapchain-required and skips frames without `swapChainView` → black forever on configure failure. The helper comment's "skip frames cleanly" is only valid for offscreen. **Fix:** add a `required: true` param that re-throws (walkaround call site passes it), or raise a non-recoverable `EngineError` at `walkaround.ts:187`. Keep offscreen's swallow behavior.
- **Tests (same task):** add `packages/engine/src/__tests__/lifecycleRecreate.test.ts` (or extend the existing vanilla lifecycle test) pinning: (V1-2) a dispose mid-recreate results in the late engine's `.dispose` being called + no lingering subscription; (V1-3) hitting the retry cap cancels RAF + disposes; (V1-6) an `advanced.device` in options does NOT reach the backend (assert stripped); (V1-7) a configure failure at a walkaround call site throws (and at an offscreen call site still swallows).
- **Behavior preservation:** (b) new pin tests FIRST where the current behavior is observable (assert the leak/black-frame on old code, then fix); (c) V1-6 is a spread-order change covered by the new assertion.
- **Verify:** `npm run typecheck`; `npm test` (engine).

## R2 — presentation gap: offscreen-backend blit + progressive present (packages/engine) [V1-1]
**This is the largest design task in the wave.** It is disjoint from R1's files (new module + `VitrumCanvas.tsx` + `vanilla.ts` present-path — coordinate the one `vanilla.ts` touch with R1 by sequencing R2's `vanilla.ts` edit after R1 lands, since both edit `vanilla.ts`; keep them in the same agent-batch only if file-disjoint by function, else run R1 then R2). 
- **State (verified):** pt-webgpu is `presentationMode: 'offscreen-texture'` (`index.ts:812`), zero `swapChainView` refs in the package; `attachVitrum` (`vanilla.ts:782`) discards the `renderFrame` output — **no blit anywhere in `engine/src`**. `VitrumCanvas` progressive returns the coordinator output which `attachVitrum` discards → **canvas freezes on the last walkaround frame after handoff.** `createEngineScale`/`pickBackend` CAN return pt-webgpu under default `'auto'` (≥500k tris or material-driven), so this hits default configurations.
- **Blit mechanism (concrete spec):**
  - Offscreen backends already expose (or must expose) the **device handle** and the **`primaryRadiance` texture** in their `FrameOutput`/engine surface. Confirm on read: pt-webgpu's `FrameOutput` carries a `GPUTexture` (or a `getPrimaryRadianceTexture()` accessor) and the engine exposes its `GPUDevice`. If not exposed, add a minimal accessor on the backend (`presentSourceTexture()` / `getDevice()`) — this is part of R2.
  - **New helper module: `packages/engine/src/presentOffscreen.ts`.** Signature:
    ```ts
    export interface OffscreenPresenter { present(source: GPUTexture): void; dispose(): void; }
    export function createOffscreenPresenter(args: {
      device: GPUDevice;
      canvas: HTMLCanvasElement;           // or OffscreenCanvas
      context: GPUCanvasContext;           // configured by the presenter (format from navigator.gpu.getPreferredCanvasFormat())
      format?: GPUTextureFormat;           // default = preferred canvas format
    }): OffscreenPresenter;
    ```
    Internals: on construction, configure `context` (format = preferred), build a **textured-quad render pipeline** (fullscreen-triangle vertex shader + a fragment shader sampling `source` via a linear sampler) and a persistent bind-group-layout; cache the pipeline + sampler. `present(source)` acquires `context.getCurrentTexture().createView()`, encodes a single render pass drawing the quad sampling `source`, submits. Reuse the existing dev `gpuTextureBlit`/fullscreen-quad WGSL if one exists (grep `engine/dev` + `pt-webgl2 PresentPass` for a reusable quad) — do NOT invent a second quad shader if a shared one exists; otherwise add the minimal WGSL in this module. `dispose()` frees the pipeline/sampler/bind-group.
  - **attachVitrum wiring (`vanilla.ts`):** when `capabilities.presentationMode === 'offscreen-texture'`, lazily construct the `OffscreenPresenter` (device + canvas + a `GPUCanvasContext` from the canvas) on first frame, and after each `renderFrame` call `presenter.present(frameOutput.primaryRadiance)`. Walkaround/swapchain backends keep the existing path (no presenter). Dispose the presenter in `attachVitrum`'s teardown.
  - **VitrumCanvas / progressive wiring:** `progressiveHandleAsEngine.renderFrame` must return a `FrameOutput` whose `primaryRadiance` is the *current* backend's present texture (walkaround before handoff, pt-webgpu after). `attachVitrum` then blits it every frame regardless of which sub-engine is active — this fixes the post-handoff freeze. Ensure the presenter's `source` follows the active backend across handoff (the presenter is device-scoped; the shared-device progressive facade means one device — assert device identity, which the facade already preflights).
- **Tests (same task):** add `packages/engine/src/__tests__/presentOffscreen.test.ts` — with a stubbed `GPUDevice`/`GPUCanvasContext` (mirror the engine test-stub style), assert: (1) a presenter is created only for `presentationMode==='offscreen-texture'`; (2) `present()` acquires the current canvas texture and encodes exactly one render pass per frame; (3) after a simulated progressive handoff, `present` is called with the pt-webgpu source (not the frozen walkaround texture); (4) teardown disposes the presenter. If a real-GPU assertion is needed it belongs in the **T1 GPU smoke** (add a note), but the unit test pins the wiring on the stub.
- **Behavior preservation:** (b) new — the freeze is currently observable via "attachVitrum discards output"; pin that the output is now consumed. Swapchain-backend path must be byte-unchanged (assert no presenter constructed for walkaround).
- **Verify:** `npm run typecheck`; `npm test` (engine); note the real-GPU present as a **T1 smoke** follow-up.

## R3 — HybridEngineLifecycle teardown pair + cascadeDispatch shader-module null (walkaround-hybrid + walkaround-rc) [V1-4, V1-5]
File-disjoint from R1/R2 (wh + rc). V1-4 and V1-5 are in different packages → may be two sub-agents.
- **V1-4 (CONFIRMED both parts) — deferred teardown never completes + RC/skinning leak on dispose-races-init.** In `HybridEngineLifecycle`: the poll-exit `return` at `:333` is **before** the `try` (which opens at `:357`), so the `finally` (`585–620`) never runs → `_pendingTeardown` never completes. AND the deferred teardown calls only `teardownPipeline` + `disposeDdgi`, whereas the synchronous dispose also does `_skinning.dispose()` + `_rc.dispose()` → **RC + skinning leak** on a dispose-races-init. **Fix:** move the `try` before the poll (or run the deferred finalisation inside the poll-exit block so `finally` fires); AND add `disposeRc` + `disposeSkinning` to the `PipelineInitHost` interface, wire them, and call them in the deferred teardown (mirror the synchronous dispose).
- **V1-5 (CONFIRMED) — cascadeDispatch reuses old-device shader modules after device change.** `cascadeDispatch` `invalidateBindings` (`521–524`) does not null `_castShaderModule`/`_mergeShaderModule`; a rebuild after a device change reuses old-device modules → cross-device validation error. **Fix:** null both `_castShaderModule` and `_mergeShaderModule` in `invalidateBindings`.
- **wsl-gpu note:** `cascadeDispatch.ts` is under `src/rc/` — **on the oracle directory list.** Do NOT rename/move; internals only. `grep -rn 'cascadeDispatch\|HybridEngineLifecycle' ~/projects/wsl-gpu/scripts` and record (expect no path breakage — internal edits only). If V1-5 shares the file with T2-C's `CascadeUniforms` codegen work, sequence R3 (this bug) FIRST.
- **Tests (same task):** (V1-4) add a `HybridEngineLifecycle` teardown test that triggers a dispose during init and asserts `_pendingTeardown` resolves AND `_rc.dispose`/`_skinning.dispose` were called (mirror existing lifecycle-race tests); (V1-5) add a cascadeDispatch test asserting `_castShaderModule`/`_mergeShaderModule` are nulled after `invalidateBindings`, forcing a rebuild.
- **Behavior preservation:** (b) new pin tests FIRST (assert the leak/never-resolve on old code, then fix).
- **Verify:** `npm run typecheck`; `npm test` (walkaround-hybrid, walkaround-rc); **T1 GPU smoke** at push (RC oracle) since V1-5 touches the rc device path.

## R4 — pt-webgpu SPPM fallback + atomic setScene (+ BDPT default 1→2 per D2) (pt-webgpu) [V2-1, V2-2, V2-5]
All in `packages/pt-webgpu/src`. Composes with **T2-A/D10-2** (registry-driving) — R4 lands the safety, T2-A later drives the same buffer set off `SCENE_BUFFER_REGISTRY`; cross-referenced there.
- **V2-1 (CONFIRMED) — SPPM fallback miswired; caustics silently ~zero.** `gpuResources:1688–1709` warns "falling back to manifold-nee" and returns `false`; `index.ts:1519–22` installs placeholders but **never mutates `causticStrategy`**; the packer packs the mode from static config (`frameParamsPacker:181–186`); the kernel's `caustic==2u` branch runs `photonMapContribution` against the placeholder buffers → **the fallback never actually happens** and caustics render ~zero. **Fix:** on the ceiling-miss set `#causticModeOverride = 'manifold-nee'` and pass it into `#buildParamsBuffer` so the packed mode reflects the fallback (kernel then takes the manifold-nee path, not the placeholder photon path).
- **V2-2 (CONFIRMED) — non-atomic setScene leaks + leaves engine sceneless on throw.** `index.ts:1348` destroys the old `sceneBuffers` BEFORE `:1349` `uploadPackedScene`; `uploadPackedScene` has zero `try` in its body and does ~30 sequential creates — a mid-throw leaks all created buffers AND leaves the engine sceneless. **Fix:** upload to a **local** buffer set first, then swap-and-destroy the old on success; inside `uploadPackedScene` track `created[]` and `catch { destroy all created; throw }`. (This is exactly the try/rollback that **T2-A** will later fold into the registry-driven uploader — do the safety here.)
- **V2-5 (CONFIRMED) — BDPT default inert. [RESOLVED per D2 — raise default 1→2 unconditionally].** See D2 for the full spec. Concretely: (1) change `BDPT_SAFE_DEFAULT_LIGHT_BOUNCES` `1 → 2` at `pt-webgpu/src/index.ts:373`; (2) re-pin the `frameParamsPacker` default goldens (the default packed `bdptLightBounces`/`maxLv` byte) to `2`, documenting each as an intended default change; (3) add a pin test asserting `maxLv=2` ⇒ the connection loop `for lvi=1u; lvi<maxLv` executes `lvi=1` (packed default light-bounce count == 2, loop non-empty); (4) document the known-bias caveat in the BDPT option JSDoc + keep the existing `bdpt:true` experimental warning (do NOT add an "inert at <2" warning — the default is now 2); (5) GPU A/B: before/after reference render on a BDPT-contributing scene per the testing protocol (old pt-webgpu BDPT baselines embed the inert `=1` default — re-capture on next refresh). This sub-task is UNBLOCKED and runs alongside V2-1/V2-2.
- **Tests (same task):** (V2-1) a test asserting that on a forced ceiling-miss the packed FrameParams caustic mode is `manifold-nee` (1u), not `photon` (2u) — byte-check the packed params; (V2-2) a test stubbing `uploadPackedScene` to throw mid-sequence and asserting all created buffers were destroyed AND the previous scene buffers are still intact (engine not sceneless); (V2-5) the `maxLv=2`⇒`lvi=1`-loop pin test + re-pinned frameParamsPacker default goldens per D2.
- **Behavior preservation:** (b) V2-1/V2-2 new pin tests FIRST (assert the silent-zero / leak on old code); (a) V2-1 touches the packed params → keep the FrameParams byte-golden green except the intended mode byte.
- **Verify:** `npm run typecheck`; `npm test` (pt-webgpu); naga (caustic-path packing unchanged structurally); **T1 GPU smoke** at push.

## R5 — shared-bvh TLAS singular-skip + UV1 range-id (shared-bvh) [V2-3, V2-4]
Both in `packages/shared-bvh/src`. **Characterization tests FIRST for both** (per the shared-bvh discipline). Disjoint from Wave-0 0E (that fix is `compressedWideBvh`; these are `scenePack`/`worldSpaceMerge`).
- **V2-3 (CONFIRMED) — TLAS singular-instance handling diverges between build paths.** The initial pack **skips** non-invertible instances (`scenePack:966–973` `continue`), but the rebuild path **inserts an identity fallback** (`:1213–1219`). There is also a pre-existing count-vs-membership mismatch on both paths. **Fix:** make the rebuild path **mirror the skip behavior** (skip non-invertible, do not insert identity), and revisit `liveCounts` so the instance count matches actual membership on both paths.
- **V2-4 (CONFIRMED) — UV1 range drift attaches the wrong primitive's UV1.** The producer skips all-filtered instances (`validTriangles.length===0 → continue :816`, no range pushed), but `mergeUv1FromCore` (`:994–1043`) does NOT replicate that guard → `rangeIdx` desyncs and UV1 attaches to the wrong primitive. **Fix:** record the **source-primitive id** on `MergedMeshVertexRange` and drive the consumer off ranges **by id** (this kills the replicated skip logic instead of duplicating it — the robust fix).
- **wsl-gpu:** shared-bvh is not on the wh oracle list, but the **T1 smoke's TLAS/BVH oracles compile against it** → T1 smoke is the real gate. `grep -rn 'scenePack\|mergeUv1FromCore\|worldSpaceMerge' ~/projects/wsl-gpu/scripts` and record; do not move files.
- **Tests (same task):** (V2-3) `packages/shared-bvh/src/__tests__/tlasSingularInstance.test.ts` — build a scene with a non-invertible instance, assert BOTH the initial-pack and rebuild paths agree (instance skipped, `liveCounts` matches membership); see it FAIL on current code (rebuild inserts identity), then fix. (V2-4) `packages/shared-bvh/src/__tests__/uv1RangeId.test.ts` — a scene with an all-filtered primitive between two UV1-bearing primitives, assert the third primitive's UV1 attaches to the correct primitive by id; FAIL on current code, then fix. Keep the T1 CPU brute-force TLAS oracle at 100% post-change.
- **Behavior preservation:** (b) characterization tests FIRST for both; **T1 smoke (TLAS/BVH oracles)** at push is mandatory.
- **Verify:** `npm run typecheck`; `npm test` (shared-bvh); the two new tests; **T1 GPU smoke** at push.

## R6 — gltf MAT2/MAT3 column padding + pre-allocation ceilings (gltf-adapter) [V3-1, V3-2]
Both in `packages/gltf-adapter/src`. Disjoint files (`accessors.ts` vs `texturePipeline.ts`) → two sub-agents possible.
- **V3-1 (CONFIRMED) — glTF MAT2/MAT3 small-component column padding missing (spec violation).** `accessors.ts` computes `elementByteLength = compSize*count` (`:91`) and reads contiguously (`:201–208`); glTF spec §3.6.2.4 requires **4-byte column alignment** for matrix accessors whose component size < 4 bytes. **Fix:** read per-column with `colStride = ceil(compSize*colCount/4)*4` when `compSize < 4 && (type===MAT2 || type===MAT3)`; **mirror the same in `accessorBufferViewRange`** so range computation matches.
- **V3-2 (CONFIRMED) — decode ceilings applied AFTER allocation.** `normalizeDecodedPixels` allocates a full-res `Float32Array` (`texturePipeline:1476`) BEFORE `resizeDecodedTextureToMaxSize` (`:839`); `validateDecodedTexturePixels` only checks a minimum length; the spec-gloss bake does the same (`:1242`). A hostile/huge texture allocates unbounded before the clamp. **Fix:** clamp dims **before** allocation (fuse the resize into the decode loop) + add a configurable **pixel-budget rejection** pre-decode (reject over-budget textures before allocating).
- **Tests (same task):** (V3-1) `packages/gltf-adapter/src/__tests__/matrixColumnPadding.test.ts` — a **hand-built MAT3 accessor over a BYTE component** buffer with column padding, assert the decoded matrices match the spec-padded layout (and a MAT2/BYTE fixture); FAIL on current contiguous read, then fix. (V3-2) a test with an over-budget decoded-texture fixture asserting rejection/clamp occurs before the full-res allocation (spy the allocation size or assert the budget-rejection path).
- **Behavior preservation:** (b) new fixtures FIRST — the MAT3/BYTE fixture currently decodes wrong; pin the correct spec-padded output. glTF sweep suites must stay green (non-padded matrices unaffected).
- **Verify:** `npm run typecheck`; `npm test` (gltf-adapter); glTF sweep gate.

## R7 — shared-denoisers lifecycle/chaining + neural preprocessing (shared-denoisers; +neural note) [V3-3, V3-4, V3-5, V3-6; D5/V3-7 adjacent]
In `packages/shared-denoisers/src`. **The bmfr try/finally leak (V3-4's *sibling* class) is already Wave-0 0A** — R7 does NOT redo it. R7 covers svgf one-shot, the bmfr *overlap knob*, `webGpuTextureUpload` try/finally, and the oidnBridge promise cache. The neural preprocessing fix (D5/V3-7) is disjoint (`neural/` + `neuralPack.wgsl`) and may run as a sub-agent here or standalone.
- **V3-3 (CONFIRMED) — one-shot SVGF can't chain frames.** No prev depth/normals/objectIds inputs (uploads current to both, `:377–408` "prev same as curr for one-shot"); the `finally` destroys `moments`/`history-out` before returning → chaining is impossible. **Fix:** add **optional `prev*` inputs** (prev depth/normals/objectIds) and **return** `{ rgb, momentsOut, historyLengthOut }` so the caller can feed them back next frame (don't destroy them in `finally` when the caller opts into chaining).
- **V3-4 (CONFIRMED, opt-in only) — BMFR overlap non-determinism.** When `blockStride < blockSize`, overlapping workgroups `textureStore` the same texels (`bmfr.wgsl:325`), last-writer-wins nondeterministic; default `stride==blockSize` is safe. **Fix:** either add an **accumulate + resolve** pass (so overlap sums deterministically) **or remove the overlap knob** (clamp `blockStride>=blockSize`). Prescribe: **remove/clamp the knob** unless a consumer needs overlap (grep consumers first) — the resolve pass is more work for an opt-in-only path. (Distinct from 0A's try/finally leak fix — 0A does not touch the overlap knob.)
- **V3-5 (PARTIAL) — `webGpuTextureUpload` readback leaks on rejection.** `rgba16fReadback.ts` was **REFUTED** (already `try/finally` correct — no-action, record it). But `shared-denoisers/webGpuTextureUpload.ts` `readRgba16fToRgb`/`readRgba32fToRgb` are CONFIRMED (`mapAsync` with no `try/finally`; the staging buffer leaks on rejection; used by svgf + bmfr). **Fix:** wrap each in `try { ... } finally { buf.destroy(); }` in both readback fns.
- **V3-6 (CONFIRMED) — OIDN concurrent first-use creates a duplicate leaked session.** `_sessionCache` stores resolved sessions; two concurrent creates both pass the `undefined` check → duplicate session created, the overwritten one leaks untracked. **Fix:** cache a `Promise<session>` — set the promise in the cache **before** the `await`, and `delete` it on rejection (so a failed create doesn't poison the cache).
- **D5 / V3-7 (RESOLVED — concrete task; touches training-pipeline files ONLY, no runtime change).** Full spec + verified anchors in **§D5**. Two edits: **(1) `tools/neural-denoiser-training/train.py`** — decode the **normal** channel `[0,1]→[-1,1]`+renormalize (mirror `neuralPack.wgsl:57–58`) in `__getitem__` at the normal-load site (`train.py:208`; keep `_load_rgb` at `:226–230` generic for noisy/albedo/clean); **(2) `tools/neural-denoiser-training/capture-dataset.mjs`** — switch the two **color** targets (noisy + clean) from reinhard-LDR PNG (`renderColor:583–585`, `reinhard:532`, writes at `:656`/`:659`) to an **HDR target** (float `.bin` or EXR of linear radiance) and load it **linearly** in `train.py`; keep albedo/normal as PNG (`:657–658`). **Flags:** re-training is required AND no production checkpoint exists yet (CLAUDE.md) — this lands correct preprocessing but does NOT produce a shippable neural tier; the checkpoint + quality A/B are **separately scoped future steps**. `neuralPack.wgsl` stays UNCHANGED (it is the source-of-truth reference for the alignment).
- **Tests (same task):** (V3-3) a test that runs svgf twice feeding frame-1's returned `moments`/`historyLength` into frame-2 and asserts the second frame consumes them (no "prev same as curr" fallback) + that the returned handles are not destroyed when chaining; (V3-4) a test asserting `blockStride<blockSize` is rejected/clamped (or, if the resolve pass is chosen, that overlapping writes sum deterministically across two runs); (V3-5) stub `mapAsync` to reject and assert `buf.destroy()` ran in both readback fns; (V3-6) two concurrent `getSession` calls resolve to the **same** cached session (no duplicate created) and a rejected create removes the cache entry; (V3-7) a preprocessing unit test asserting a `[0,1]` PNG normal decodes to the runtime `[-1,1]` normalized encoding and the color target is HDR (not reinhard-LDR).
- **Behavior preservation:** (b) new pin tests FIRST (assert the leak/duplicate/mis-encoding on old code); (c) the svgf one-shot default path (no prev inputs) must be byte-unchanged — the prev inputs are optional.
- **Verify:** `npm run typecheck`; `npm test` (shared-denoisers); (V3-7) the preprocessing unit test asserting the `[0,1]→[-1,1]`+renormalize normal decode matches `neuralPack.wgsl:57–58` and the color target is HDR/linear (not reinhard-LDR), plus `train.py` `dry_run`/smoke green on the new format. Note: neural changes need re-training + a future quality A/B (no naga/GPU gate — no runtime change).

**Wave-R commit stubs** (one per group; orchestrator commits after marker-grep + gate):
- `fix(engine): re-check disposed mid-recreate; stop RAF+dispose on retry cap; strip ptWebgl2 ownership keys; re-throw on required configure`
- `feat(engine): present offscreen-backend primaryRadiance to canvas + fix post-handoff freeze`
- `fix(wh,rc): complete deferred lifecycle teardown (rc/skinning dispose); null cascade shader modules on invalidate`
- `fix(pt-webgpu): honor SPPM manifold-nee fallback; make setScene atomic (rollback on throw)`
- `fix(shared-bvh): mirror TLAS singular-instance skip on rebuild; UV1 range-by-source-id`
- `fix(gltf-adapter): MAT2/MAT3 column padding; clamp decode dims before allocation + pixel budget`
- `fix(shared-denoisers): svgf chaining; bmfr overlap knob; readback try/finally; oidn promise cache`
- `fix(neural): align training preprocessing to runtime normal/HDR encoding (retraining required)`
- *(BDPT policy V2-5 lands inside the R4 commit above — raise default 1→2 + re-pin frameParamsPacker default goldens + loop pin test + GPU A/B; per D2.)*

---

# WAVE 0 — Real bugs, dead code, stale docs

All Wave-0 items were planner-verified. Groups 0A–0F are file-disjoint and may run in parallel; 0-DEAD and 0-DOC are disjoint from all.

## 0A — bmfrWebGPU resource leak (shared-denoisers) [REAL BUG]
- **File:** `packages/shared-denoisers/src/bmfrWebGPU.ts` (fn `runBmfrWebGPU`, verified lines 129–214).
- **Verified defect:** textures (141–145), `ubo` (171), and `destroyEphemeral` (from `acquireDenoiseDevice` at 129) are freed only on the happy path (210–212). If `readRgba16fToRgb` (204) or any earlier GPU call throws, all leak. The 3 sibling dispatchers use `try/finally`.
- **Fix (exact shape):** wrap from the first resource acquisition through the return in `try { ... } finally { for (const t of [...]) t.destroy(); ubo.destroy(); destroyEphemeral(); }`. Declare `colorTex/normalTex/worldPosTex/historyTex/outTex/ubo` and `destroyEphemeral` with `let ...: T | undefined` BEFORE the `try`, assign inside, and null-guard each in `finally` (`t?.destroy()`, `ubo?.destroy()`, `destroyEphemeral?.()`). Keep the successful `return result` inside the `try`.
- **Behavior preservation:** (c) pure-mechanical on the success path — `finally` runs the identical teardown; existing `bmfr*` vitest suites cover the happy path. Add a characterization test `packages/shared-denoisers/src/__tests__/bmfrWebGPU.leak.test.ts` that stubs `readRgba16fToRgb` to throw and asserts every created texture's `.destroy` was called and `destroyEphemeral` ran (mirror the stub style already used in the denoiser tests).
- **Verify:** `npm run typecheck`; `npm test` (shared-denoisers).

## 0B — pt-webgl2 exhaustiveness gate re-enable (composeTraceGlsl) [REAL BUG]
- **File:** `packages/pt-webgl2/src/glsl/composeTraceGlsl.ts:301–307` (verified).
- **Verified defect:** `type _ExhaustivenessCheck = _AllFrameKeys extends (_ManifestFrameKey | _HandledSeparately) ? true : never;` is declared but the consuming assert `const _exhaustive: _ExhaustivenessCheck = true;` is **commented out** (307), so a new `FrameUniforms` key silently escapes the manifest with no compile error.
- **Fix (exact shape):** replace line 307's comment with a live assert that TypeScript will reject if `_ExhaustivenessCheck` resolves to `never`:
  ```ts
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _exhaustive: _ExhaustivenessCheck = true;
  ```
  Confirm it currently compiles (i.e. the manifest is exhaustive today); if `npm run typecheck` fails, the manifest has a real gap — add the missing key(s) to `_ManifestFrameKey` or `_HandledSeparately` as appropriate, do NOT re-comment the gate.
- **Behavior preservation:** (c) compile-time-only; no runtime change (the GLSL output is unaffected — `_exhaustive` is a type-level binding). run-ptwebgl2-h1 unaffected but still run it since the file is pt-webgl2.
- **Verify:** `npm run typecheck`; run-ptwebgl2-h1; `npm test` (pt-webgl2).

## 0C — mockGl false-coverage comment + optional enrichment (pt-webgl2) [REAL STALE HAZARD]
- **Files:** `packages/pt-webgl2/src/__tests__/mockGl.ts:114` (`getActiveUniform: () => null`, verified) and `packages/pt-webgl2/src/gl/glResources.ts:50–52, 609` (comments claiming "mock-GL tests pin sampler-unit assignment by this order", verified).
- **Verified defect:** `getActiveUniform` returns `null`, so `GlProgram.assignSamplerUnits` is never exercised by the mock-GL suite — the load-bearing comment in `glResources.ts` is **false**; a reorder of the sampler table would NOT break a mock-GL test.
- **Fix (choose the low-risk correction; do BOTH parts):**
  1. **Correct the comment (required, cheap):** in `glResources.ts:50–52` and `:609`, change "The mock-GL tests pin sampler-unit assignment by this order" to point at the real guard: "The **pre-push T1 GPU smoke** compiles the pass graph and pins sampler-unit order; mock-GL does not exercise `assignSamplerUnits` (its `getActiveUniform` returns null)." — this removes the stale-context hazard.
  2. **Enrich the mock (optional, only if it lands green without golden churn):** give `mockGl.getActiveUniform` a minimal deterministic implementation returning `{ name, size: 1, type }` derived from the recorded `__shaderSources` sampler declarations, so `assignSamplerUnits` runs. If enrichment perturbs any existing pin/golden, **do part 1 only** and record enrichment as accepted-not-done with that reason.
- **Behavior preservation:** part 1 is comment-only; part 2 gated on all existing pt-webgl2 tests staying green.
- **Verify:** `npm run typecheck`; run-ptwebgl2-h1; `npm test` (pt-webgl2).

## 0D — thin-film layer limit reconcile [MOVED to User Decisions §D1]
- The thin-film 8-vs-35 reconcile is now a top-of-plan **User Decision (D1)**; its implementation task lives inside **T1-1** (shared material scalar derivations), gated on D1's chosen option. See §D1 and T1-1. Nothing to do in Wave 0 for this item.

## 0E — compressedWideBvh tMin/comparator inconsistency (shared-bvh) [REAL BUG — ORACLE-ADJACENT]
- **File:** `packages/shared-bvh/src/compressedWideBvh.ts` — `intersectCompressedWideBvhFirstHit` (verified 493–567) vs `intersectCompressedWideBvhAnyHit` (verified 569–625).
- **Verified defect:** firstHit uses `tMin = opts.tMin ?? triEps` (502) and accepts `hit.t >= tMin` (549); anyHit uses `tMin = opts.tMin ?? 1e-4` (578) and `hit.t > tMin` (618). Two divergences: the **default tMin** (`triEps` default 1e-5 vs `1e-4`) and the **comparator** (`>=` vs `>`). A ray grazing at exactly `tMin` is a hit in firstHit but a miss in anyHit → shadow/occlusion vs closest-hit disagree at the epsilon boundary.
- **Fix (deliberate reconcile — DECIDE the correct convention, document it):**
  - Standardize both on `const tMin = opts.tMin ?? triEps;` (single default source) and the **`>=`** acceptance (firstHit's, which is the more inclusive/standard self-intersection guard). Rationale to put in a comment: anyHit must not report a *miss* where firstHit reports a *hit* at the same epsilon, or shadow rays under-occlude relative to the closest-hit geometry.
  - This is oracle-adjacent (`compressedWideBvh` is imported by the CWBVH repeat proof — see recent commit `9d40f9b8`). Do NOT move the file. `grep -rn 'compressedWideBvh' ~/projects/wsl-gpu/scripts` and confirm the oracle still imports the same path (it does — no move here). The **T1 smoke's CWBVH oracle is the real gate** for this change.
- **Behavior preservation:** (b) characterization test FIRST — add `packages/shared-bvh/src/__tests__/cwbvhTMinParity.test.ts` constructing a ray that grazes a triangle at exactly `t == 1e-4` and one at `t == triEps`, assert firstHit-hit ⇒ anyHit-hit (the invariant), see it FAIL on current code, then fix. Keep the CPU brute-force oracle in mind: the T1 smoke's CWBVH oracle must stay 100% post-change.
- **Verify:** `npm run typecheck`; `npm test` (shared-bvh); the new parity test; rely on **T1 GPU smoke** (CWBVH oracle) at push time.

## 0F — examples createDracoMesh dropped call (examples/gltf-viewer) [DEAD-CODE / stale helper]
- **File:** `examples/gltf-viewer/src/main.ts` (report cited `main.ts:310`; actual path has `/src/`). Verified: `createDracoMesh` defined at 310–312, referenced only in a **type position** at 260 (`ReturnType<typeof createDracoMesh>`); the runtime path inlines `new module.Mesh()` at 267.
- **Verified classification:** the helper's runtime call was dropped in favor of the inline `new module.Mesh()` — the inline is correct and working; `createDracoMesh` is effectively dead except as a `typeof` anchor for the `mesh` local's type.
- **Fix (exact shape):** either (preferred) **use the helper** — replace `mesh = new module.Mesh();` (267) with `mesh = createDracoMesh(module);` so the type anchor and the runtime path share one source; OR delete `createDracoMesh` and change 260's type to `InstanceType<typeof DracoDecoderModule>['Mesh'] extends ...` — messier. **Prescribe: use the helper at 267.** Net behavior identical (`createDracoMesh` returns `new module.Mesh()`).
- **Behavior preservation:** (c) mechanical — the helper body IS `new module.Mesh()`. Examples have no vitest; verify by `npm run typecheck` (examples tsconfig) and a build (`npm run build` in the example if defined).
- **Verify:** `npm run typecheck`; example build if present.

## 0-DEAD — dead-code removal (10 truly-dead + 4 test-oracle + directives)
File-disjoint from 0A–0F. Each item was flagged by the dead-code agent; the orchestrator MUST source-read each before deletion (verify-before-delete rule — knip over-reports).
- **TRULY_DEAD (delete, after per-item read confirms zero non-type consumers):**
  1. `collectPrimitiveMaterialFieldUses` in `pt-webgl2/src/index.ts:185` AND `pt-webgpu/src/index.ts:136` — **VERIFIED orphaned** (grep found only the two definitions, no call sites; superseded by `collectUnsupportedMaterialFieldUses`). Delete both. Confirm `collectUnsupportedMaterialFieldUses` exists and is the live path before deleting.
  2. unused import `PT_WEBGPU_COMMON_WGSL` in `pt-webgpu/src/wgsl/.../restirPtCompose.wgsl.ts:352` — remove the import line only.
  3. unused import type `OIDNBridgeLike` in `pt-webgl2/src/.../oidnFinalDispatcher.ts:7` — remove.
  4. unused local `compSize` in `gltf-adapter/src/accessors.ts:472` — remove the binding.
  5. unused destructured `sampler` in `gltf-adapter/src/featureReport.ts:3703` — remove from the destructure.
  6. 3 stale `eslint-disable` directives: `VitrumCanvas.tsx:419`, `equirectHdrInfo.ts:167`, `oracle.liteRectMis.test.ts:381` — remove each (confirm no longer needed by running lint after).
  - (createDracoMesh handled in 0F, not here.)
- **TEST_ORACLE dead bindings (4) — do NOT delete; wire them into their assertions** (they are independent-reference oracles whose value is being computed-but-unused):
  1. `INV_PI` in `extensionLobeReference.test.ts:12` — use it in the reference expression instead of an inline literal.
  2. `ratio` in `oracle.bdptConnectionCosine.test.ts:386` — the assertion recomputes inline; assert against `ratio`.
  3. `diagnostics` in `gltfExtensionPolicy.test.ts:169` — assert on it.
  4. `buildRisGiFrameBindGroup` import in `pipelineBindGroupFactory.test.ts:19` — add the missing coverage call, or if genuinely redundant with an existing case, remove the import (read first).
- **NOT_DEAD (no action):** `tools/gltf-material-sweep/fixture.mjs:119` `field` param — used inside a live exported fn (per agent; orchestrator confirms by read, then leaves as-is).
- **Behavior preservation:** (c) removals of unused symbols/imports — covered by `npm run typecheck` + full vitest + `npx eslint` (must stay 0 dead-code msgs where removed). Test-oracle rewires: the tests must stay green.
- **Verify:** `npm run typecheck`; full `npm test`; lint.

## 0-DOC — stale-doc / comment fixes (grouped, cheap, disjoint)
Each is a comment/doc string with no runtime effect (verify each by read):
1. `uboUpdater.ts:167–169` (wh pipeline) — `packWalkaroundUBO` doc says 416 bytes; header (1–3) and code write 432. Fix the 416 → 432. (D5-10)
2. `unetArchitecture.ts:94` + `neural/wgsl/skipConnection.wgsl.ts:7` — reference extinct `InferenceGraph._validateSkipShapes()`; real symbol is `tensorDimSolver.validateSkipShapes:103`. Update both comments. (D7-1)
3. `gpuTextureBlit.ts:9` (engine/dev) — stale docstring references non-existent `useGpuTextureBlit`. Correct or remove. (D2 minor)
4. `frameParamsPacker.ts:34–57` (pt-webgpu) — hand-maintained prose slot ledger redundant with the runtime guard; delete the narrative, keep the guard. (D9-8)
5. `gpuResources.ts:1274, 1854` (pt-webgpu) — hard-coded cross-file line-number citations already rotted; replace with symbolic references (name the symbol, not the line). (D9-7)
6. `run-gate-dzn.mjs:18` + `check-ledger.mjs:905` (tools) — hardcoded `/home/jsquire4` paths; derive from `import.meta.url`/repo-root or relativize. (D17-10). NOTE: `check-ledger.mjs` is an anti-regression pin file — change ONLY the path literal, touch no pin strings.
- **Behavior preservation:** doc-only except #6 (path derivation) — for #6 add/keep it working by running the tool once locally after. (b for #6: confirm the gate still runs from repo root.)
- **Verify:** `npm run typecheck`; run `run-gate-dzn.mjs` help/dry path for #6.

**Wave-0 commit stubs** (one per group, orchestrator commits after marker-grep + gate):
- `fix(denoisers): wrap runBmfrWebGPU teardown in try/finally (leak on readback throw)`
- `fix(pt-webgl2): re-enable FrameUniforms exhaustiveness gate in composeTraceGlsl`
- `fix(pt-webgl2): correct false mock-GL sampler-pin comment; point at T1 smoke`
- `fix(shared-bvh): reconcile CWBVH firstHit/anyHit tMin default + comparator`
- `refactor(examples): route gltf-viewer Draco mesh alloc through createDracoMesh`
- `chore: remove verified dead code + wire test oracles into assertions`
- `docs: fix stale UBO byte count, InferenceGraph refs, rotted line citations, hardcoded paths`
- *(thin-film moved to User Decision D1; implemented under T1-1 once chosen — not a Wave-0 commit)*

---

# WAVE T2 — Table-driving / codegen (kills whole drift-bug classes)

These convert hand-maintained parallel lists into single-source tables/codegen — the H1/H41 offset-drift class. Highest leverage, mostly pure-mechanical with strong existing coverage. File-disjoint groups.

## T2-A — pt-webgpu buffer/texture registries (D9-6, D10-2, D10-4)
- **D10-2 `scene/uploadSceneBuffers.ts`:** `SCENE_BUFFER_REGISTRY` exists but `uploadPackedScene` + `destroy` + `gpuMemoryBytes` hand-list the same ~30 buffers (4 parallel lists, verified 2198 LOC). Fix: drive all three off the registry (iterate entries with `{name, upload, destroy, memoryBytes}`); split into `scene/packScene.ts` + `scene/uploadScene.ts` if the split lands cleanly, else keep one file and just register-drive. **Composes with R4/V2-2:** Wave R already made `uploadPackedScene` atomic (local-then-swap, `created[]` + `catch{destroy all; throw}`). PRESERVE that try/rollback when register-driving — fold the `created[]` tracking into the registry iteration (the registry entries make the rollback list self-maintaining); do NOT regress the atomicity R4 landed. **Behavior:** (b) add a characterization test asserting the set of destroyed buffers == set of uploaded buffers == registry keys BEFORE refactor, and that the R4 rollback-on-throw test still passes.
- **D10-4 `scene/materialTextures.ts:480–651`:** ordered 23-map list enumerated 7×. Fix: single `TEXTURE_MAP_SLOTS` table `{index, wrap, mip, filter, uvMeta}` driving all writes + both uv-fit passes. **Behavior:** (a) the packed texture output must be byte-identical — add/keep a golden on the packed material-texture bytes; assert unchanged.
- **D9-6 `gpuResources.ts` lite entries 1320–1337 + rebuildGroup2Only 1436–1457:** extract `#makeGroup2BindGroupEntries` + a shared bindings-0-11 prefix (`#makeGroup0BindGroupEntries`). **Behavior:** (c) bind-group entry lists are structural; covered by pt-webgpu engine-contract test + T1 smoke.
- **Verify:** typecheck; pt-webgpu tests; naga (no WGSL change expected); T1 smoke at push.

## T2-B — pt-webgl2 material/texture/light tables (D11-4, D11-5, D11-12)
- **D11-4 `scene/materialsTexture.ts:377–457, 719–826`:** `packLayerIds` 6-positional-arg literal + `packTextureTransforms` 30+ hardcoded absolute texel offsets (the package's most fragile stride-drift surface). Fix: data-driven map table + **named offsets sourced from `materialStride.js`** (single source of truth for texel positions). **Behavior:** (a) byte-identity — the materials texture bytes must be identical; run-ptwebgl2-h1 is the deterministic gate.
- **D11-5 `inverse/finiteDifferenceSession.ts:401–593`:** material field set enumerated 4× (set/read/clamp/patch). Fix: single `MATERIAL_PARAM_DESCRIPTORS` table `{path, kind, clampRange, read, write}` driving all four switches. **Overlaps I5-3** — coordinate: if T1-3 (shared inverse scaffolding) runs, this table becomes the pt-webgl2 provider feeding the shared param parser. Sequence: do the local table here in T2; T1-3 later hoists the *shared* scaffolding and consumes this table. **Behavior:** (b) characterization test enumerating a param through all four operations before refactor.
- **D11-12 `scene/lightsTexture.ts:143–310`:** 5 per-kind arms re-spell s0/s1 texel writes with a fragile `k++` cursor. Fix: `writePositionType(texel, ...)` / `writeColorIntensity(texel, ...)` helpers with an explicit cursor object. **Behavior:** (a) byte-identity lights texture; run-ptwebgl2-h1.
- **Verify:** typecheck; run-ptwebgl2-h1 (MANDATORY — all three touch pt-webgl2 texture packing); pt-webgl2 tests.

## T2-C — walkaround-rc codegen extension (D16-7, D16-8)
- **D16-7 `wh/rc/packingHelpers.ts` vs `probeRayCast` RCLight struct:** 16-float layout hand-mirrored TS↔WGSL while `rcParamsLayout.generated.ts` codegen already solved this class for RCParams. Fix: **extend `tools/.../generate-wgsl-layouts` (or the RC codegen)** to emit RCLight, replacing the hand-mirror. **Behavior:** (a) generated output must byte-match the current hand-written layout — commit the generated file, diff == 0.
- **D16-8 `cascadeDispatch.ts:239–299`:** `buildCascadeUniformDataInto` raw magic slot indices (`ui[29..31]`) + ASCII layout comment duplicating the WGSL struct (H1/H41 offset-drift class). Fix: bring `CascadeUniforms` under the same codegen; replace magic indices with generated named offsets. **Behavior:** (a) byte-identity of the packed cascade uniform buffer — the **T1 smoke's RC oracle** is the real gate; add a CPU pin on the packed bytes first.
- **wsl-gpu note:** `cascadeDispatch.ts` and `rc/packingHelpers.ts` are under `src/rc/` which IS on the oracle directory list — do NOT move/rename these files; only change their internals + add a generated layout module. Re-run `grep -rn 'rc/' ~/projects/wsl-gpu/scripts` to confirm no path assumptions about generated file names.
- **Verify:** typecheck; wh tests; naga; **T1 smoke (RC oracle)** at push — mandatory for T2-C.

## T2-D — cross-cutting hand-tables from the reminisce hotspot list
The reminisce flagged: `passOrder` (3 sources of truth), `PassLabel` + `MAX_PASS_COUNT`, tunables triple-declared, FrameParams TS-vs-WGSL offsets, ReGIR OFF-state, timestamp decode.
- **D5-9 ReGIR OFF literal:** `ReGIRCoordinator.ts:154–163` ≡ `uboUpdater.ts` `REGIR_OFF`. Fix: export `REGIR_OFF` from one site, import at the other. (c) trivial, covered by wh tests.
- **D5-8 timestamp decode:** `timestampQueries.ts:283–303` vs `373–383` BigInt64 decode loop (already drifting). Fix: `decodeTimestampSlots(...)` helper. (b) pin a decode result first.
- **passOrder / PassLabel / MAX_PASS_COUNT / tunables:** verify the "3 sources of truth" claim by reading each site (planner did NOT deep-verify these — treat as candidate). If confirmed, make one the source and derive the others; if already single-sourced (stale reminisce), record as rejected. (b) characterization test pinning pass order before any change.
- **Verify:** typecheck; wh + pt-webgpu tests; T1 smoke (pass-graph) at push.

**T2 commit stubs:**
- `refactor(pt-webgpu): drive scene buffer/material-texture writes off single registries`
- `refactor(pt-webgl2): table-drive material/texture/light texel packing (byte-identical)`
- `refactor(walkaround-rc): extend WGSL layout codegen to RCLight + CascadeUniforms`
- `refactor(wh,pt-webgpu): single-source ReGIR-OFF, timestamp decode, pass-order tables`

---

# WAVE T6 — walkaround-hybrid ownership moves (ORACLE REPOINTS REQUIRED)

**Run T6 alone (not parallel with other wh-touching waves)** — it moves the shared-foundation seam. Sub-tasks are file-disjoint internally but all touch wh import graphs. **Every task here MUST do the oracle-repoint dance.**

## T6-1 — Split shared foundations out of `restir/` (I3-1)
- **Verified imports (2026-07-20):** `rc/bvhCore.ts:19` imports `packUVIntoVec4W` from `restir/packingHelpers`; `ppg/ppgConstants.ts:96` + `ppg/ppgUpdate.wgsl.ts:53` import from `restir/reservoirGiLayout`; `ddgi/DDGI.ts:35–36` imports `makeRestirBvhSnapshot`/`SceneBVHBuffers` from `restir/restirBvhSnapshot`+`restir/bvhCore`; `ddgi/probeUpdatePass.ts:38,70` imports from `restir/restirBvhSnapshot` + `EMITTER_TRI_STRIDE_BYTES` from `restir/emitterList`; `probeUpdateBvhBuffers.ts:6` imports `RestirBvhSnapshot`.
- **Fix (precise, oracle-safe):**
  - Create `packages/walkaround-hybrid/src/bvh/bvhPacking.ts` and move the **generic** `packUVIntoVec4W` (+ any generic `packBVH*` from `packingHelpers.ts` that has no restir semantics) there. Keep `restir/packingHelpers.ts` as a file (it's an oracle-safe non-listed module) and **re-export** the moved symbol for back-compat, OR repoint all consumers. Repoint `rc/bvhCore.ts:19` → `../bvh/bvhPacking.js`.
  - Create `packages/walkaround-hybrid/src/gi/giLayout.ts` and move `reservoirGiLayout` contents (or re-export). Repoint `ppg/ppgConstants.ts:96` and `ppg/ppgUpdate.wgsl.ts:53`.
  - **BVH-snapshot ownership:** `restir/restirBvhSnapshot.ts` and `restir/bvhCore.ts` ARE on the oracle hardcoded-path list. **DO NOT MOVE THESE FILES.** Leave them in place; the ddgi/rc consumers already import from the stable paths. Only the *generic packing/layout* symbols move.
- **wsl-gpu repoint:** `grep -rn 'restir/packingHelpers\|restir/reservoirGiLayout' ~/projects/wsl-gpu/scripts`. These two are NOT on the listed oracle paths — expected zero hits. If any hit, repoint. `bvhCore.ts`/`restirBvhSnapshot.ts` stay put → oracles unaffected. **Confirm by grep and record the grep output as on-disk proof.**
- **Behavior preservation:** (c) pure re-home of exported constants/fns; covered by wh typecheck + tests. If any symbol is re-exported for back-compat, add a `threeDecoupleSeams`-style pin test asserting the new module exports it.
- **Verify:** typecheck; wh tests; **T1 smoke** (confirms oracles still resolve — the primary safety net for moves).

## T6-2 — Move CPU pack-utils off `pipeline/` upward-dep (I3-2)
- **Verified:** `restir/bvhCore.ts:30` imports `packMaterialTextureAtlas` from `pipeline/materialTextureAtlas` (violates restir's no-upward-dep charter); `ddgi/probeUpdatePass.ts:50–51` imports `uploadTangentTexture`/`uploadVertexColorTexture` from `pipeline/`.
- **Fix:** move the **pure-CPU pack halves** (`packMaterialTextureAtlas` and the CPU portions of the tangent/vertex-color builders) to a shared BVH-build location: `packages/walkaround-hybrid/src/bvh/` (co-located with T6-1's `bvhPacking.ts`) or `@vitrum/shared-bvh` if the deps allow (materialTextureAtlas imports only core types → shared-bvh-eligible). Keep the **GPU-upload halves** pipeline-side. Repoint `restir/bvhCore.ts:30` and `ddgi/probeUpdatePass.ts:50–51`.
- **wsl-gpu repoint:** `bvhCore.ts` and `probeUpdatePass.ts`... `probeUpdatePass.ts` is NOT on the list but `probeUpdateBvhBuffers.ts` IS — confirm the moved symbols aren't reached by an oracle. `bvhCore.ts` IS on the list but we're not moving it, only changing its import line — oracle imports the file by path, so its internal import change is transparent to the oracle (deno resolves transitively). Still `grep -rn 'materialTextureAtlas\|uploadTangentTexture' ~/projects/wsl-gpu/scripts` and repoint any hit.
- **Behavior preservation:** (c) re-home; wh typecheck + tests + T1 smoke.
- **Verify:** typecheck; wh tests; T1 smoke.

## T6-3 — Hoist `toProductionEmissiveRadiance` ei-collapse guard (D6-8)
- **Verified byte-identical 3-line guard** at `restir/bvhCore.ts:94`, `restir/packingHelpers.ts:509`, `ddgi/probeUpdateMaterials.ts:71` (+ inline in `emitterHelpers`).
- **Fix:** hoist a single `toProductionEmissiveRadiance(...)` to `@vitrum/shared-bvh` (binding-free CPU math). Import from all sites. **Do not move bvhCore.ts** — only change its import + delete its local copy.
- **wsl-gpu:** `bvhCore.ts` internal edit only; oracle transparent. Grep for the symbol name in wsl-gpu (expect zero) and record.
- **Behavior preservation:** (c) identical math; shared-bvh + wh tests + T1 smoke (which independently oracles emitter Le).
- **Verify:** typecheck; shared-bvh + wh tests; T1 smoke.

## T6-4 — Move deprecated PbrMaterialLike packers to test-support (D6-4)
- **Verified:** `restir/packingHelpers.ts:161–480` holds ~9 `@deprecated` `PbrMaterialLike` packers retained as **test oracles** inside the production module.
- **Fix:** move them to a test-support file `packages/walkaround-hybrid/src/restir/__tests__/support/legacyPbrPackers.ts` (NOT deleted — legit independent-reference pattern per verify-before-delete rule). **First** grep all consumers (`grep -rn 'PbrMaterialLike\|<each fn name>' packages/`); repoint test imports. If ANY production (non-test) consumer exists, do NOT move that symbol — leave it, note why.
- **wsl-gpu:** `packingHelpers.ts` not on the oracle list, but grep anyway; expect zero.
- **Behavior preservation:** (c) test-only relocation; the consuming tests must stay green.
- **Verify:** typecheck; wh tests.

**T6 commit stubs:**
- `refactor(wh): extract generic BVH-packing + GI-layout foundations out of restir/`
- `refactor(wh): move CPU pack-utils off the pipeline/ upward-dep`
- `refactor(shared-bvh): single-source toProductionEmissiveRadiance ei-collapse guard`
- `refactor(wh): relocate deprecated PbrMaterialLike oracle packers to test-support`

---

# WAVE T3 — God-file splits (largest files; concern separation)

Each god-file is a single package/file → naturally file-disjoint across agents. Splits are **structure-only** where possible; each keeps a characterization/contract test green.

## T3-A — HybridEngine material-approximation warner (D3-1, D3-2, D3-3)
- **Verified 3122 LOC.** `HybridEngine.ts:942–1224` = ~280-LOC inline warning subsystem (9 `_warned*` Sets + 8 `_warn*` methods) + re-invocation at `1545–1609`; `1129–1224` triple-repeated 5-arm nested ternary.
- **Fix:** extract `MaterialApproximationWarner` module (mirrors the existing `DdgiSync`/`GIState` extraction pattern) with a **table-driven `warnOnce`** keyed by `diagnostic.code`; move the 9 Sets + 8 methods + the nested-ternary lookup into it. HybridEngine holds one `#materialWarner` field and delegates. D3-2's nested ternary → a single lookup table keyed by `diagnostic.code`.
- **NOT in scope:** the rejected HybridEngine field-storage rewrite (do-not-reattempt). This is warner extraction only.
- **wsl-gpu:** `HybridEngine.ts` is NOT on the oracle path list (its subsystems are). Safe. Do not rename the file.
- **Behavior preservation:** (b) characterization test FIRST — pin the exact warning strings + once-only semantics for each of the 9 approximation classes (drive each code, assert message + that a 2nd call is silent), green on old code, then extract.
- **Verify:** typecheck; wh tests; new warner characterization test.

## T3-B — pt-webgpu index.ts + gpuResources.ts (D9-1, D9-2, D9-3, D9-5, D9-9)
- **Verified index.ts 2641 LOC, gpuResources.ts 2127 LOC.**
- **Fix (index.ts):** extract `ptWebgpuCapabilities()` pure builder (the 165-line capabilities literal 726–890), `validatePtWebgpuOptions()` module (factory validation 2302–2641), `LiteSceneValidator` (scene validation), and move the lite scene-analysis free fns (180–219) into `scene/`. D9-3: split `#ensurePerFrameResources` (1465–1710) into per-subsystem `ensureXPerFrame()`; put the `u32[bdptEnabled]=0` byte-poke (1493) behind a `packFrameParams` flag. D9-5: emit **structured** displacement warnings at source in `buildPackedScene`/`uploadPackedScene` instead of regex-recovering from prose (3 drain loops at 129–134/1287–1346/1381–1397).
- **Fix (gpuResources.ts):** D9-2 — make the extracted sub-objects (`Reservoir`/`Sppm`/`Present`) **real owners** (`gpu.reservoir.x`) and delete the ~90 lines of pass-through accessor pairs (362–419); move the SPPM (~350 LOC) + ReSTIR-PT (~300 LOC) method clusters INTO their sub-objects. This is the larger change — **do it as its own commit** after the accessor-deletion is proven by the engine-contract test.
- **wsl-gpu:** neither file on the oracle list. Safe.
- **Behavior preservation:** (b) the pt-webgpu **engineContract test** (2638 LOC — flagged as a god-test but it IS the coverage) pins the public surface; run it green before and after each extraction. Capabilities builder: (a) assert the built capabilities object deep-equals the pre-extraction literal. `#ensurePerFrameResources` split: pin the packed FrameParams bytes (byte-identity) before/after.
- **Verify:** typecheck; pt-webgpu tests; naga (bdpt poke path); T1 smoke at push.

## T3-C — pt-webgpu inverseSession + emitterPacking + others (D10-1, D10-3, D9 inverse split)
- **Verified inverseSession.ts 2398 LOC.**
- **Fix:** D10-1 — extract `inverse/pathReplayDiagnostics.ts` (the ~1400-line diagnostic subsystem: 40 pure fns) + `inverse/paramResolution.ts`; collapse the 11 near-identical `materialIssueFor*` fns to a **table-driven helper**; session class drops to ~700 lines. D10-3 — `scene/emitterPacking.ts:461–677`: shared `packMeshAreaTrianglesToVec4` + grouped-arg record for the 11-arg `pushTriangle` closure; merge the near-identical `packEmitterArrays` vs `packMeshAreaAdjointReplayArrays` cap+sort+pack tails.
- **Behavior preservation:** (b) the inverse-session test (`inverseSession.test.ts`, 4629 LOC) + adjoint FD-vs-analytic self-validating harnesses pin correctness — run green before/after. emitterPacking: (a) byte-identity of packed emitter arrays.
- **Verify:** typecheck; pt-webgpu tests (inverse + adjoint harnesses); T1 smoke.

## T3-D — pt-webgl2 index.ts + composeTraceGlsl.ts + mutateSceneTextures.ts (D11-1, D11-2, D11-3, D11-6)
- **Verified index.ts 1445 LOC, composeTraceGlsl.ts 1356 LOC.**
- **Fix:** D11-1 — extract `options.validate.ts`, `mutationFallbackWarnings.ts`, `frameUniformsPacker.ts` from index.ts. D11-3 — split `glsl/renderMain.glsl.ts` (the ~800 lines of `RENDER_MAIN_*` constants + caustic heuristics) + `glsl/uniformManifest.ts` out of composeTraceGlsl; composer → ~300 lines. D11-2 — `scene/mutateSceneTextures.ts:635–1206`: shared `resolveAtlasSwap` + `commitResidentGeometryTextures` for the 3 fast-path fns' triplicated atlas-refresh block. D11-6 — `gl/glResources.ts`: extract `uploadFrameUniforms` free fn (the ~130-line inline in `drawAccumStep`); move `BLEND_FRAG` to its own module.
- **CRITICAL:** composeTraceGlsl output must stay byte-identical — the composed GLSL string is the load-bearing artifact. **(a) byte-identity golden on the composed trace GLSL** + **run-ptwebgl2-h1** (deterministic render) is the real gate. Note: this touches the same file as Wave-0 0B (exhaustiveness gate) — **0B must land first**; T3-D builds on the re-enabled gate. Do NOT parallelize 0B and T3-D.
- **wsl-gpu:** pt-webgl2 not on the wh oracle list; run-ptwebgl2-h1 is the gate.
- **Behavior preservation:** (a) composed-GLSL golden + run-ptwebgl2-h1; (b) engineContract.test.ts (2638 LOC) for the index split.
- **Verify:** typecheck; run-ptwebgl2-h1 (MANDATORY); pt-webgl2 tests.

## T3-E — gltf-adapter god files (D15-1..7, I4-2)
- **Verified featureReport.ts 4083 LOC, gltfToScene.ts 3169 LOC.**
- **Fix:** D15-1 — 4-way split of `featureReport.ts`: `types.ts` / `assetInventory.ts` / `backendCompatibility.ts` / `compatibilityMessages.ts`. D15-2 — `evaluateGltfBackendProfileCompatibility` (657–1124, ~470 lines) → per-category emitter fns. D15-3 — `analyzePrimitives`/`analyzeMaterials`/`analyzeAnimations` → per-item body helpers. D15-4 — `gltfToScene.ts:500–1501` 12-section fn → named functions per banner; extract `buildPrimitiveFromMeshPrimitive`. D15-5 — `sceneController.ts`: `#commitSceneChange` helper + variant collaborator. D15-6 — `texturePipeline.ts` → `textureCodecs.ts` + `textureDecodeReport.ts`. D15-7 — `assetLoader.ts:335–723` → `backendCompatibilityReconcile.ts`. D15-8/I4-2 — centralize the duplicated compatibility-issue predicates (`isTextureReadinessIssue` ×2, `*SatisfiedByDecode` mirrors, and `engine/gltf.ts:497–568` re-implementing `engineBridge.ts:738–785`) in ONE module and export; call from `gltf.ts`.
- **wsl-gpu:** gltf-adapter not on the oracle list. Safe.
- **Behavior preservation:** (b) the gltf sweep suites + `featureReport`/`gltfToScene` tests pin the emitted `Scene` + compatibility reports — run green before/after. I4-4's proposed pin-test (adapter tables vs `BackendSupportDetails`) is a NEW test — add it here (see T1-4).
- **Verify:** typecheck; gltf-adapter tests; glTF sweep gate.

## T3-F — walkaround-rc probeRayCast + featureReport-adjacent (D16-1)
- **Verified probeRayCast.wgsl.ts 1512 LOC** and it is **on the oracle directory list** (`src/rc/`... actually `walkaround-rc/src/wgsl/` — verify: the oracle list showed `wh/src/rc/` and `wh/src/shaders`, NOT walkaround-rc. Re-grep before touching.).
- **Fix:** split `rcMaterialAtlas.wgsl.ts` + `rcBrdf.wgsl.ts` out of `probeRayCast.wgsl.ts` (62 fns). **KEEP the `probeRayCast.wgsl.ts` file path** (report says wsl-gpu oracle references it) — extract INTO sibling modules that probeRayCast imports/composes, do NOT rename probeRayCast itself.
- **CRITICAL composeWgsl constraint:** the atlas-sampling + BRDF helpers reference **consumer bindings** (atlas textures, scene). Per the ordering rule, the extracted shared helpers that reference consumer bindings MUST be **raw-string templates interpolated into probeRayCast's body**, NOT standalone `WgslModule`s. Only binding-free math (e.g. pure BRDF math with no texture reads) may be a `WgslModule`. Verify each extracted fn's binding dependencies before choosing its form.
- **wsl-gpu:** `grep -rn 'probeRayCast\|walkaround-rc/src/wgsl' ~/projects/wsl-gpu/scripts`; if probeRayCast is referenced, keep its path and repoint nothing (only internals change). Record grep.
- **Behavior preservation:** (a) byte-identity WGSL golden on the composed probeRayCast output + naga compile + **T1 smoke RC oracle**. This is the gate that catches composeWgsl-ordering regressions.
- **Verify:** typecheck; wh/rc tests; naga; **T1 smoke (RC oracle)** MANDATORY.

## T3-G — tooling god files (D17-1, D17-2, D17-3)
- **Verified gate.mjs 2944 LOC, check-validation-queue.mjs 3359 LOC, check-ledger.mjs 2167 LOC.**
- **Fix:** D17-1 — extract `sceneBuilders`/`gltfFixtures`/`ptNagaGapFix`/`readback`/`goldens` + shared `applyMutation` from `gate.mjs`; the `runPtConfig`/`runWhConfig` char-identical ~55-line mutation dispatch chains → shared `applyMutation`. D17-2 — split the ~450-line hardcoded manifest tables in `check-validation-queue.mjs` into a data module; derive shared rows from source-of-truth modules. D17-3 — group `check-ledger.mjs` into per-document check fns + pinned-string tables. **D17-3 CRITICAL:** the prose-string pins (incl. pt-webgl removal pins) are **intentional anti-regression** — do NOT delete or alter any pin string; only reorganize them into tables.
- **Behavior preservation:** (b) run each tool before/after and diff its output/exit code — these are the gates themselves, so their output must be identical. For `check-ledger.mjs`, every pin must still fire.
- **Verify:** run `gate.mjs`, `check-validation-queue.mjs`, `check-ledger.mjs` and diff output vs pre-refactor.

**T3 commit stubs:** one per sub-task (A–G), e.g.:
- `refactor(wh): extract MaterialApproximationWarner from HybridEngine`
- `refactor(pt-webgpu): extract capabilities/options-validate/lite-scene-validator from index`
- `refactor(pt-webgpu): make gpuResources sub-objects real owners; drop passthrough accessors`
- `refactor(pt-webgpu): split inverseSession diagnostics + param resolution`
- `refactor(pt-webgl2): split renderMain/uniformManifest out of composeTraceGlsl`
- `refactor(gltf-adapter): 4-way split featureReport + section-fn extraction in gltfToScene`
- `refactor(walkaround-rc): split rcMaterialAtlas/rcBrdf out of probeRayCast (path preserved)`
- `refactor(tools): decompose gate/validation-queue/ledger god files (pins preserved)`

---

# WAVE T4 — WGSL dedup round (naga-gated; largest structural dup)

**Run after T3-F/T2-C** (they touch WGSL too). All items are byte-identity-or-golden-refresh; each states which. The **composeWgsl raw-string-vs-WgslModule rule applies to every extraction**.

## T4-1 — risGiNrc ≡ risGi dedup (D8-1, THE LARGEST) [byte-identity via raw-string template]
- **Verified risGi.wgsl.ts 719 LOC, risGiNrc.wgsl.ts 793 LOC**; `risGiNrc:154–755` is a near-verbatim copy of `risGi:53–691` incl. the ~250-line glass-refraction walk, with 3 surgical NRC deltas.
- **Fix:** create a **raw-string template builder** (the `temporalGiCommon.wgsl.ts` precedent — verified that file exists) `buildRisGiBody({ nrc: boolean, ...interpolationSlots })` that emits the shared body with the 3 NRC deltas as interpolated slots. Both `risGi.wgsl.ts` and `risGiNrc.wgsl.ts` call it. **MUST be raw-string interpolation, NOT a WgslModule** — the body references `scene`/`ubo` consumer bindings.
- **wsl-gpu:** these are under `wh/src/shaders` (on the oracle directory list). Do NOT rename the files. `grep -rn 'risGi' ~/projects/wsl-gpu/scripts` and confirm no path breakage.
- **Behavior preservation:** (a) **byte-identity** — the composed `risGi` and `risGiNrc` WGSL strings must be byte-for-byte unchanged. Add/keep goldens on both composed outputs; assert diff==0. Then **naga compile** both + **T1 smoke** (the string golden is insufficient per CLAUDE.md — naga is mandatory).
- **Verify:** typecheck; wh tests (WGSL goldens); naga gate (both shaders); T1 smoke MANDATORY.

## T4-2 — Shared atlas-decode WGSL (D6-1, D8-5, D16-2, D8-3) [the 62-texel ABI, THREE copies]
- **Verified:** the 62-texel material-atlas ABI is hand-cloned in `ddgi/wgsl/probeUpdateRays.wgsl.ts` (D6-1, `DDGI_MATERIAL_MAP_*` + ~40 `ddgiSample*` fns), `shaders/materialAtlas.wgsl.ts`/`shade`, and `walkaround-rc/.../probeRayCast.wgsl.ts:340–370` (`RC_MATERIAL_MAP_*`, D16-2). Three parallel copies of the same texel offsets.
- **Fix (careful — verify byte-identity of all three FIRST):** promote the texel-offset **constants** (`*_MATERIAL_MAP_*`) to a shared WGSL const module in `@vitrum/shared-bvh` (binding-free → legal `WgslModule`). The **decode functions** that read atlas textures reference consumer bindings → they must be **raw-string templates** shared via a builder, interpolated into each consumer. Do this in two steps: (1) offsets to shared const module; (2) decode-fn template. If the three copies are NOT byte-identical (subtle drift), reconcile deliberately and record the reconciliation as a golden-refresh with A/B.
- **D8-3** (`traceSceneFirstHitAlphaMask` ×3 differing only in discard predicate) and **D8-5** (barycentric+Beer decode ×2) fold in here as predicate-interpolated templates.
- **wsl-gpu:** `probeUpdateRays.wgsl.ts` and `ddgiSampleWgsl.ts` are **on the oracle list**; do NOT rename. probeRayCast path preserved (T3-F). Repoint nothing (internals only), but `grep` and record.
- **Behavior preservation:** (a) byte-identity if the three copies match today (assert composed-WGSL goldens for shade/probeUpdate/probeRayCast unchanged); (a→golden-refresh) if reconciling drift — state which goldens re-pin + run the **T1 smoke DDGI + RC oracles** + a GPU A/B. naga on all three shaders MANDATORY.
- **Verify:** typecheck; wh + rc tests; naga (3 shaders); **T1 smoke (DDGI + RC oracles)** MANDATORY.

## T4-3 — Pure-math WGSL helpers to shared WgslModules (D8-2, D8-4, D8-6)
- **Verified byte-identical:** D8-2 — `transparentOit.wgsl.ts:82–108` ≡ `shadingTerms.wgsl.ts:150–176` (SpotConeFalloff + PointSpotAttenuation); D8-6 — `regir.wgsl.ts:90–98` ≡ `:248–256` (cell-centroid math). D8-4 — `shadingTerms.wgsl.ts:536–579` vs `714–753` (4-corner bilinear GI gather).
- **Fix:** the falloff/attenuation/centroid math is **binding-free pure math** → extract to shared `WgslModule`s (legal, emitted before consumers). D8-4's `giBilinearGather` reads GI textures (consumer bindings) → **raw-string template**, not WgslModule.
- **wsl-gpu:** files under `wh/src/shaders`; no rename; grep + record.
- **Behavior preservation:** (a) byte-identity composed WGSL goldens for each touched shader; naga; T1 smoke.
- **Verify:** typecheck; wh tests; naga; T1 smoke.

## T4-4 — temporalGi/spatialGi shared preamble extension (D8-7)
- **Verified:** OFF-vs-GRIS bodies share a binding block + preamble + verbatim 11-line M-clamp comment; the two-body split itself is deliberate (accepted).
- **Fix:** extend the existing `*GiCommon` modules with a `SCENE_GROUP_BINDINGS` fragment + shared preamble (the two-body split STAYS — only the shared preamble/bindings dedupe). Because these reference `scene` bindings, keep them as raw-string fragments interpolated into both bodies (they already are — `temporalGiCommon`/`spatialGiCommon` verified present).
- **Behavior preservation:** (a) byte-identity composed WGSL; naga; T1 smoke.
- **Verify:** typecheck; wh tests; naga; T1 smoke.

**T4 commit stubs:**
- `refactor(wh-shaders): dedup risGiNrc/risGi via raw-string template (byte-identical)`
- `refactor(wh,rc,shared-bvh): single-source 62-texel atlas-decode ABI (offsets + templates)`
- `refactor(wh-shaders): hoist pure-math falloff/centroid to shared WgslModules`
- `refactor(wh-shaders): share temporalGi/spatialGi preamble + scene-binding fragment`

---

# WAVE T1 — Cross-backend shared modules + parity tests

**Run after T2/T3** (they stabilize the per-backend tables this consumes). These extract genuinely-shared scaffolding and add parity/pin tests — the highest-value seam work.

## T1-1 — Shared material scalar derivations (I5-1) [D1 RESOLVED — Option B]
- **Verified duplicated verbatim:** `sigmaAFromAttenuation` (webgpu materialPacking.ts:216–219 ≡ webgl2 materialsTexture.ts:93–109), `sampleSpectralCurve` (webgpu :64–76 ≡ webgl2 :155–169), 32-sample 380–780nm grid, dispersion-from-Abbe, `emissiveIntensity` default.
- **Fix (scalar-derivation extraction):** create a shared module (in `@vitrum/shared-samplers` — it already single-sources Jakob-Hanika) exporting these **backend-agnostic scalar derivations**. Both backends import; **layouts stay per-backend** (the texel packing differs; only the math is shared).
- **Thin-film limit sub-task (D1 = Option B — per-backend declared limit + exceed-warning; NOT a shared const):**
  1. Add `readonly thinFilmLayerLimit?: number;` to `BackendSupportDetails` (`packages/core/src/engine/capabilities.ts:78–107`, additive/optional); populate the per-backend support-details literals in `packages/core/src/engine/promiseLedger.ts` (near the `thinFilmStack` rows at `:501/:599/:725`): pt-webgpu → `8`, pt-webgl2 → `35`.
  2. The per-backend `THIN_FILM_LAYER_LIMIT` locals stay put (materialPacking.ts:4 = `8`, materialsTexture.ts:76 = `35`) and the WGSL `const THIN_FILM_LAYER_LIMIT = 8u;` (material.wgsl.ts:1130) is UNCHANGED — `wgslContract.test.ts:318` stays pinned at `8u` (no re-pin, no WGSL widening).
  3. Emit a structured `thin-film-layer-limit-exceeded` warning at each pack site (materialPacking.ts:128; materialsTexture.ts:613/690) when `thinFilmLayers.length > THIN_FILM_LAYER_LIMIT`, naming requested count + active-backend limit + that excess layers are dropped (reuse each backend's existing structured-warning channel).
- **Behavior preservation:** (b) add a **cross-backend parity test** for the scalar derivations (shared module vs pre-extraction inline copies produce identical values); (a) byte-identity of each backend's packed material bytes (pt-webgpu golden + run-ptwebgl2-h1) for ≤limit scenes (warning path adds no texel bytes). D1 pin-tests: a core test asserting each backend's `supportDetails.thinFilmLayerLimit` matches its packer constant (8 / 35); a pt-webgpu test for the >8-layer exceed-warning; a pt-webgl2 test for the >35-layer exceed-warning.
- **Verify:** typecheck; core + both backends' tests; run-ptwebgl2-h1; **no naga change** for thin-film (WGSL const unchanged under D1=B).

## T1-2 — Shared vector math + emitter canonicalizer + parity (I5-2)
- **Verified:** webgl2 re-defines Rec.709 luminance inline (`lightsTexture.ts:71`) + private cross/normalize/tangentBasis; webgpu has its own. Silent parity gaps: webgl2 spot radius hardcoded 0 (`lightsTexture.ts:230`), mesh-area filtered out of the analytic list (`:129`).
- **Fix:** dedupe cross/normalize/tangentBasis/luminance into `@vitrum/shared-samplers` (Rec.709 luminance is duplicated 14+ times per reminisce — this is the anchor copy). Add a shared `emitterToCanonical(scene)` normalizer both backends feed, + a **parity test** asserting both backends' canonical emitter sets match for a fixture. **File the two feature-parity gaps** (spot-radius=0, mesh-area exclusion) as tracked items in `items_to_fix.md` — these are behavior gaps, not dedup (do NOT silently "fix" spot radius without an A/B; record + schedule).
- **Behavior preservation:** (c) for the pure-math dedup (identical formulas); (b) parity test is new. The spot-radius/mesh-area gaps are DOCUMENTED not changed here.
- **Verify:** typecheck; both backends; run-ptwebgl2-h1.

## T1-3 — Shared inverse scaffolding (I5-3, D11-5 overlap)
- **Verified:** webgl2 `finiteDifferenceSession.ts` re-implements Adam (126–152), L2/L1 loss (106–124), `parseParamPath` (327), MATERIAL_SCALAR/RGB_FIELDS tables (53–99), `defaultClampRange` (475), material/emitterPatch (540–594) — all conceptually duplicated in pt-webgpu `inverse/`.
- **Fix:** extract backend-agnostic scaffolding (optimizer, L1/L2 loss, param parser, field-metadata tables) into a shared location — `@vitrum/core` inverse (core already owns `createInverseSession`) or a new `@vitrum/shared-inverse`. Backends provide **only the gradient source** (FD for webgl2, adjoint for webgpu — that split is documented+correct, stays). Consume the `MATERIAL_PARAM_DESCRIPTORS` table from T2-B/D11-5 as the shared field metadata.
- **Behavior preservation:** (b) the inverse-session tests on both backends (webgpu `inverseSession.test.ts`, webgl2 FD tests) pin numeric convergence — run green before/after; add a parity test that the shared optimizer produces identical steps given identical gradients.
- **Verify:** typecheck; both backends' inverse tests.

## T1-4 — gltf-adapter capability tables → BackendSupportDetails pin (I4-4, I4-2, I5 clean seams)
- **Verified:** `featureReport.ts:485–560` hand-encodes `VERTEX_COLOR_SUPPORT`, `PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS` (30+ keys), `GltfBackendProfile` registry — per-backend capabilities that core's `BackendSupportDetails` should own; no compile-time link → drift.
- **Fix (non-breaking first step):** add a **pin-test** asserting the adapter tables agree with each backend's declared `BackendSupportDetails` (fail on drift). Longer-term source-from-core is a follow-up, but the pin-test closes the *silent* drift now. Also land I4-2's predicate export (from T3-E's centralized predicate module) so `gltf.ts` calls the adapter's predicates instead of re-implementing.
- **Behavior preservation:** (b) the pin-test is new; adapter behavior unchanged.
- **Verify:** typecheck; gltf-adapter + core tests; glTF sweep.

## T1-5 — OIDN deriveState + progressive hardening (I2-2, I5-5, I2-3)
- **Verified:** pt-webgl2/pt-webgpu delegate to `OIDNDispatcherCore` with byte-identical `getState()` ladders; walkaround-hybrid `oidnFinal.ts` reimplements its own (justified by sync contract + extra 'warming-up' state).
- **Fix:** I2-2 — extract `OIDNDispatcherCore.deriveState()`; extend the state enum with `'warming-up'`; wh reuses the status mapping but keeps its own orchestration. I5-5 — optional coordinator-level device-identity check for standalone (non-facade) progressive wiring (low-risk hardening). I2-3 — OPTIONAL/low-urgency: hoist `defineUbo` + `Ubo*` types into a leaf pkg or core (only if restructuring anyway — record as accepted-intentional/deferred-low-value if it forces churn, since shared-denoisers depending on shared-samplers for defineUbo is not a defect).
- **Behavior preservation:** (a) the byte-identical getState ladders must produce identical states — assert both backends' state sequences unchanged; (c) wh reuse covered by wh OIDN tests.
- **Verify:** typecheck; both backends + wh tests; run-ptwebgl2-h1.

## T1-6 — pt-webgpu HDR emissive texture array (rgba16float) [D3 RESOLVED — build it] [FEATURE]
**New task from User Decision D3.** pt-webgpu textured emissive is currently clamped to 8-bit sRGB because emissive shares the sRGB `rgba8unorm-srgb` array; this adds a dedicated `rgba16float` emissive array so HDR emissive texture values survive. All anchors verified on source-read 2026-07-20.
- **State (verified):** two arrays today — sRGB `rgba8unorm-srgb` (baseColor **+ emissive**) and linear `rgba8unorm` (normal + ORM), created at `scene/uploadSceneBuffers.ts:2018–2030`. Emissive `emissiveIdx` indexes the **sRGB** layer space (`scene/materialTextures.ts:491`) and is sampled in WGSL from `@group(3) @binding(3) var materialTextures: texture_2d_array<f32>` (`wgsl/pathTrace/material.wgsl.ts:394`) via `sampleMaterialEmissive`→`sampleMaterialLayer` at `material.wgsl.ts:578–582`.
- **Fix (prescriptive — files + verified insertion points):**
  1. **`scene/materialTextureArray.ts` — float format support.** `createMaterialTextureArray` (`:398–404`) already takes a `format` param and builds the array generically; the mip pipeline `generateTextureArrayMips` (`:244–314`) also takes `format` and works for a color-renderable float target. What is 8-bit-specific is the raw-data upload normalizers `normalizeRawRgba8`/`normalizeRawNumericRgba8`/`normalizeRawTextureUpload` (`:130–194`) which quantize to `Uint8Array` bytes. Add a **float upload path** (e.g. `normalizeRawTextureUploadFloat` producing `Float16`/`Float32` rows with the matching `bytesPerRow`) selected when the requested `format` is `rgba16float`; the `copyExternalImageToTexture` path (`:465–470`) is format-agnostic and needs no change. NOTE: `rgba16float` IS color-renderable (mip generation via render pass works); confirm on the target device limits. The 1×1 dummy (`createDummyArray:318–341`) already honors `format` — for a float format write a float white texel instead of the `Uint8Array([255,255,255,255])` at `:327`.
  2. **`scene/uploadSceneBuffers.ts:2018–2030` — third array creation.** Alongside the sRGB array (`:2018–2023`) and linear array (`:2025–2030`), create a third `materialEmissiveArray = createMaterialTextureArray(device, packed.materialTextureEmissiveSources, 'rgba16float', packed.materialTextureEmissiveSourceInfos)`. This requires the pack step to split the emissive sources into their own source list + info list (new `packed.materialTextureEmissiveSources`/`...SourceInfos` in the pack producer, and a new `emissiveIdx` index space pointing at the emissive array's layers — mirror the sRGB `indexOf(m.emissiveMap, ...)` at `materialTextures.ts:491` but into the emissive array's layer space). Add the emissive array handle + dispose to the `UploadedSceneBuffers` registry entry (there are dispose handles for the sRGB/linear arrays near `uploadSceneBuffers.ts:222/228` — add an emissive one), and drain its `warnings`/`structuredWarnings` into the material-texture warning accumulators (`:2044–2051`). Also feed its `layerUvScales` into `applyMaterialTextureUvFitScales` (`:2031–2035`) if emissive uses UV-fit (it currently reads the sRGB scales for emissive — repoint emissive to the new array's scales).
  3. **WGSL sampling side — `wgsl/pathTrace/material.wgsl.ts`.** Add a new binding for the emissive float array. Current group-3 texture bindings: `@binding(3)` sRGB `materialTextures`, `@binding(4)` shared `materialTexSampler`, `@binding(5)` linear `materialTexturesLinear` (`material.wgsl.ts:394–396`). `@binding(6..9)` are FREE (next used is `@binding(10)` meshTangents / `@binding(11)` meshVertexColors at `:397–398`). **Add `@group(3) @binding(6) var materialTexturesEmissive: texture_2d_array<f32>;`** (a `rgba16float` array is still sampled as `texture_2d_array<f32>`). Repoint `sampleMaterialEmissive` (`:578–582`) to sample the new binding — either add a `sampleMaterialLayer` variant that reads `materialTexturesEmissive` (mirror the `materialTexturesLinear` variant at `:585–665`) or parameterize the array. Emissive descriptor slots (`emissiveIdx` in vec4[0].w, UV-scale vec4[7].zw, wrap vec4[13].zw, mip/filter policies at materialTextures.ts:557/583/606/629) stay the same layout but now index the emissive array's layers.
  4. **Bind group layouts + entries.** Add binding 6 (emissive `texture_2d_array<f32>`) to the pt-webgpu group-3 bind-group LAYOUT and the group-3 bind-group ENTRIES wherever they're built (grep `binding: 5` / `materialTexturesLinear` in the pt-webgpu bind-group factory + `gpuResources.ts` group-3 entry builders; the adjoint pass also declares these arrays at `wgsl/pathTrace/adjointPass.wgsl.ts:224/227` — decide whether the adjoint path needs the emissive-float array too, or keep adjoint on the sRGB emissive for now and document).
  5. **Capabilities truthfulness update.** Reflect HDR-emissive-texture support in the pt-webgpu capabilities/`BackendSupportDetails` (and note pt-webgl2 does NOT get this in this task) so the fidelity matrix is honest.
- **Tests + A/B:** unit test on `materialTextureArray.ts` float path (a raw-float source uploads to `rgba16float` without 8-bit quantization; a >1.0 emissive value survives round-trip); a pack test asserting emissive sources land in the emissive array's index space (not the sRGB array). **GPU A/B** with an **HDR emissive fixture** (emissive texture with values >1.0): before = clamped/dim, after = full HDR emission; assert no regression on LDR-emissive and textureless scenes (dummy-array byte-identity).
- **wsl-gpu:** none of these files are on the wh oracle path list; the **T1 GPU smoke** compiles the pt-webgpu pass graph (validates the new binding).
- **Behavior preservation:** (a) LDR/textureless scenes must stay byte-identical (new binding + dummy float array must not perturb existing renders); (b) new float-path + pack tests FIRST.
- **Verify:** typecheck; pt-webgpu tests; naga (new binding); **T1 GPU smoke** at push; GPU A/B on the HDR emissive fixture.

**T1 commit stubs:**
- `refactor(shared-samplers): single-source material scalar derivations + per-backend thin-film limit + exceed-warning + parity test`
- `refactor(shared-samplers): dedupe vector math + shared emitterToCanonical + parity test`
- `refactor(core/inverse): extract backend-agnostic inverse scaffolding; backends supply gradients`
- `test(gltf-adapter): pin capability tables against BackendSupportDetails; wire predicate module`
- `refactor(denoisers): OIDNDispatcherCore.deriveState + progressive device-identity check`
- `feat(pt-webgpu): dedicated rgba16float emissive texture array (HDR textured emission)`

---

# WAVE T5 — Mechanical TS dedup (broad, low-risk, high count)

All are pure-mechanical extractions with existing coverage. Partitioned by package for disjointness. Run late (after god-file splits settle the files).

## T5-A — engine + dev (D2-1..7)
- vanilla.ts:386–837 → `createAutoRecreateController` + dedupe the twin warning blocks (598–733). VitrumCanvas.tsx:85–143 → move `progressiveHandleAsEngine` next to `createProgressiveEngine` + `fanOut` telemetry helper. progressiveHandoff.ts:263–370 → private `#applyToBothEngines`. negotiateWebGPUDevice.ts:226–245 + createProgressiveEngine.ts:216–237 → exported `checkProgressiveLimitUnion(adapter): string[]`. dev: unify NumberRing/RingBuffer to one O(1)-sum impl + back-compat alias; FrameTimeHUD/overlayFrameTime → shared `observeFrameTime(engine, ring, onSample)`. Add `engine/src/__tests__/fixtures/stubEngine.ts` and adopt across the 10 test files.
- **Behavior preservation:** (c) covered by engine vitest; the stub fixture is a test-only consolidation (tests stay green).

## T5-B — walkaround-hybrid non-shader (D3-4,5,6, D4-1..4, D5-1..8,11)
- D3-5 → promote mat4/mat3/`transformPoint` helpers to shared-bvh, consume from HybridEnginePrimitiveUpdates + HybridEngineScaleAwareClamps. D3-4 → characterization test pinning the (patchShape→path) matrix (confirm `mutationMatrix.test.ts` covers atlas-rebuild predicates). D3-6 → named intermediate group objects in HybridEngineRC dispatchFrame. D4-1 route gpuMemoryEstimate through `addSection`. D4-2 delete bvhTangent/bvhVertexColor forks, use shared `bvhTextureLimits`. D4-3 BvhBufferHost per-resource descriptor table. D4-4 shared `uploadElementTexture`. D5-1 use `checkShaderCompile`. D5-2 generalize `dispatchSharedBindGroupPass` with optional slot3Group. D5-3 add `checkerboardWgX/Y` to PassDispatchContext. D5-4 `sourceMetaFields` helper. D5-5 `writeScalarMeta`. D5-6 shared `cachedBindGroup`. D5-7 split/optionalize OptionalSubsystemBindingState input type. D5-11 `_createOutputTexture` in oidnFinal.
- **wsl-gpu:** `HybridEngineRC.ts` IS on the oracle list — do NOT rename; internals only. shared-bvh additions (mat4 promote) don't move oracle files. grep + record.
- **Behavior preservation:** (b) D3-4 characterization test FIRST; (c) rest covered by wh vitest + (for descriptor-table/bind-group changes) T1 smoke.

## T5-C — pt-webgl2 non-god (D11-8,9,11)
- D11-8 parameterized `respecifyTexture` + `finalizeGeometryBuild` for the 13 upload wrappers. D11-9 hoist `emitSubTriangle` helper in meshAreaLights. D11-11 `resolveHandleHint` + `makeDecoder` in texturesArray. D11-13 `#traceTier` — keep (unwired-intended, roadmap B12).
- **Behavior preservation:** (a) byte-identity where texture bytes involved + **run-ptwebgl2-h1 MANDATORY**.

## T5-D — pt-webgpu non-god (D10-7, D9-4, sceneMutationRouter)
- D10-7 `sampleCosineHemisphere` helper in bdptEmitterPickCpu (168–266). D9-4 sceneMutationRouter updatePrimitive (283–799) → named private methods per fast path + `#resyncMeshAreaEmitters` + `#resolveBonesPatch`.
- **Behavior preservation:** (b) sceneMutationRouter is correctness-sensitive — the pt-webgpu mutation tests + engineContract pin it; run green before/after.

## T5-E — shared-bvh (D12-1..10)
- D12-1 `resolveReadableTexture`. D12-2/D12-10 shared `textureDecode.ts` (halfToFloat + decoderFor + fold BYTES_PER_ELEMENT/`__vitrum_hint__`). D12-4 `finalizeSplicedPack` + `rebaseIndexWords`. D12-5 `makeLeaf` closure. D12-6 `resolveDisplacedGeometry` in vertexDisplacement. D12-7 stride param on computeLocalAabb (default 3). D12-8 call `mapPrimitivesById` instead of inlining. D12-9 split `materialSignature.ts` (+ optional `worldTransforms.ts`) out of worldSpaceMerge. (D12-3 handled in Wave-0 0E.)
- **wsl-gpu:** shared-bvh isn't on the wh oracle list, but the T1 smoke's TLAS/BVH oracles compile against it — **T1 smoke is the gate**. Grep + record.
- **Behavior preservation:** (b) shared-bvh has strong pack/traversal tests + the T1 CPU brute-force oracles — pin packed bytes before/after; T1 smoke at push.

## T5-F — shared-denoisers + rc + samplers residue (D14-1,3,4, D16-3,4,5,6, D7-2, D1-1,2, D10-6, D10-5)
- D14-1 `buildAtrousChain` + `makeResourceTracker` (atrous/svgf shared ping-pong). D14-3 cross-reference comments (albedo-demod divergence). D14-4 document/drop oidnBridge 'color' fallback. D16-3 field-registry-driven signature in cascadeDispatch. D16-4 `resolvePlaceholderTexture` helper. D16-5 named `ResolvedCastBindings`/`ResolvedBvhBuffers` interfaces. D16-6 delete `_RC_LIGHT_EVAL_WGSL` dead const (verify 0 consumers first). D7-2 (OPTIONAL) ppg descent single builder — **mitigated** (MUST-MATCH comments + drift test); do only if it lands byte-identical, else accepted-intentional. D1-1 `blendMorphStream` helper in skinSolver. D1-2 scene-lighting skyParams 0.5 → named const/option. D10-6 `symmetricDeriv3` helper (safe subset only). D10-5 pathTraceBruteforce — accepted-intentional (order load-bearing, naga+golden gated, low priority — record, do NOT extract unless byte-identity holds).
- **wsl-gpu:** cascadeDispatch under `src/rc/`... verify against the list (`wh/src/rc` was listed; walkaround-rc may differ) — grep + record; keep file paths.
- **Behavior preservation:** (b) skinSolver.test pins morph blending; (a) cascade signature/atrous chain byte-identity + T1 smoke; ppg drift test gates D7-2.

**T5 commit stubs:** one per group (A–F).

---

# WAVE T8 — Tooling residue (after T3-G)

Disjoint from all source waves. (D17-1/2/3 god-splits are in T3-G; these are the rest.)
- D17-4 shared `ptNagaGapFix.mjs` (gate.mjs:1892–1990 ↔ radiometric-ab/helpers.mjs byte-identical). (a) both callers produce identical output.
- D17-5 promote WH GPU-harness scaffolding (perspective/lookAt/patchDevice/readback/quad/cornell) to a shared harness module (5 files). (c) harness tests.
- D17-6 move the 3 scratch harnesses (walkaround-diag/sun-control/material-check.mjs) to `tools/diagnostics/` marked ad-hoc (do NOT delete — capture value).
- D17-7 harnesses consume `assetManifest.mjs`; export a shared threshold const (checkers MAY keep independent pins deliberately — do not force-unify checker pins).
- D17-8 the 6 tool files deep-importing `../../packages/*/src/*.ts` (9 hits) — import via package entrypoints where public; else add a stable tool-facing barrel + a `CLAUDE.md` watch-list entry (same silent-break class as wsl-gpu oracles). **grep the 9 hits, list them in the task.**
- D17-9 add `gate.mjs --json` mode; repoint `run-gate.mjs:122–204` off stdout-scraping.
- D17-11 import core's `selectPtWebgpuTraceTier`/`adapterProfile` authority into probe-deno.ts/probe-adapter.mjs/gate.mjs (replace re-encoded 10/5, 8/4, 16/8 math).
- **D4 (RESOLVED — Option B) verbatim-compile gate mode:** add a mode to `tools/shader-gate/gate.mjs` that ALSO attempts **un-patched** (verbatim, no `applyNagaFix`) compilation of the walkaround/RC shaders (the entries currently wrapped in `applyNagaFix(...)` at `gate.mjs:321/540/739`) and **REPORTS** (does NOT fail) the naga-incompatibility count as a tracked metric ("walkaround/RC = Chromium-only, N shaders reject on naga verbatim"). The existing `applyNagaFix`-patched pass stays the gating (green) path; verbatim is a non-fatal tracked metric. **Behavior:** the gating pass exit code must be unchanged; the new metric prints a count. See §D4.
- **D4 docs (RESOLVED — Option B):** document the Chromium-only `ptr<storage>` constraint in `packages/walkaround-hybrid/README.md` and `plan/renderer-fidelity-matrix.md`; file the `ptr<storage>` refactor as a tracked program entry in `plan/road-to-100.md` (do NOT plan the refactor). Prose edits — verify by read. (These may also run in an R-DOC step; T8 is the natural home.)
- **Behavior preservation:** (b) run each tool before/after, diff output/exit; the D4 gate mode must not change the gating exit code.
- **Commit stub:** `refactor(tools): shared naga-gap patch, harness module, manifest/tier authority, gate --json`
- **Commit stub (D4):** `feat(shader-gate): verbatim naga-incompat metric for walkaround; docs(walkaround): Chromium-only ptr<storage> tracked`

---

# ACCEPTED-INTENTIONAL (recorded, no task) — with rationale

- **D8-8 `bvh_material` declared in 6 files + primary-hit smooth-normal ×5** — composeWgsl-ordering-forced duplication, documented in-code. OPTIONAL binding-free wrapper on the DI side only if it lands byte-identical; else accept. (Low priority; not scheduled unless an agent confirms byte-identity trivially.)
- **D7-2 ppg descent dup** — mitigated by MUST-MATCH comments + `ppgDescentDrift` test. Optional single-builder only if byte-identical (see T5-F).
- **D10-5 pathTraceBruteforce compose fns** — order load-bearing, naga+golden gated, low priority; accept.
- **D10-6 brdfAdjoint `*WithAnisotropy` FD wrappers** — kept explicit for auditability; only `symmetricDeriv3` subset extracted.
- **D14-3 atrous/svgf albedo-demod divergence** — intentional (different algos); add cross-reference comment only (in T5-F).
- **I2-1, I2-4, I3-3, I4-1, I4-3, I5-4, I5-5(core), I1** — CLEAN reference seams; no action (I5-5 device-identity check is an OPTIONAL hardening in T1-5).
- **I2-3 defineUbo hoist** — low-value; do only if restructuring forces it (T1-5 note).

# OUT-OF-SCOPE (do-NOT-reattempt list — reasons)

- **GGX / fresnelSchlick merge** — intentional per-backend roughness-floor divergence; merging would change rendering. (D13 confirms documented in-code.)
- **pt-webgpu Möller-Trumbore unification** — needs a SceneHit/IntersectionResult contract first; not this sweep.
- **HybridEngine field-storage rewrite** — explicitly rejected; T3-A does warner extraction only, not field rewrite.
- **photonGatherRadius surfacing** — frozen-UBO churn for net-zero.
- **withEngineLifetime facade** — rejected.
- **D10.11 class (RGBA32F present target)** — the comment may be the bug and the code the intent (load-bearing FLOAT readback); do NOT "fix" comment/code-disagreement findings of this shape.

# REJECTED FINDINGS (planner source-read, false positive) — with evidence

- None fully rejected on this pass. Two findings were **path-corrected** (not rejected): the report's `examples/gltf-viewer/main.ts` is actually `examples/gltf-viewer/src/main.ts` (verified — 0F uses the corrected path); the report's `createDracoMesh` "may be a bug" is precisely a **dead-helper / dropped-call** (verified: referenced only in a `typeof` position at :260, runtime inlines `new module.Mesh()` at :267 — the inline is correct; 0F routes through the helper, no behavioral bug).
- **Candidate-only (NOT yet source-verified by planner — agents MUST verify before acting):** D2-D17 individual extraction line ranges were spot-verified (god-file sizes + the 6 Wave-0 bugs + oracle paths + composeWgsl precedent confirmed), but the exact interior line ranges of the ~80 mechanical extractions in T4/T5/T3 are taken from the domain agents' reports. Per the investigation-rigor rule, **each implementing agent opens the cited file:line and confirms the dup/structure before extracting**; if a finding is stale on read, the agent records it as rejected-with-evidence in its report rather than forcing a refactor. D2-D5 "3 sources of truth" table claims (passOrder/PassLabel/tunables in T2-D) are explicitly flagged candidate — verify single-source status first.

---

# FINAL FULL-GATE CHECKLIST (run before any push; push only on user instruction)

1. `npm run typecheck` (all packages) — 0 errors.
2. `npm test` (full Vitest, all packages) — green; new characterization/parity/leak tests included, **including every Wave R regression/pin test**:
   - **R1:** `engine/src/__tests__/lifecycleRecreate.test.ts` — dispose-mid-recreate disposes the late engine (V1-2); retry-cap cancels RAF + disposes (V1-3); `advanced.device` stripped from options (V1-6); walkaround configure-failure throws / offscreen still swallows (V1-7).
   - **R2:** `engine/src/__tests__/presentOffscreen.test.ts` — presenter created only for `offscreen-texture`; one present pass/frame; present uses pt-webgpu source after handoff (freeze fixed); teardown disposes (V1-1).
   - **R3:** `HybridEngineLifecycle` deferred-teardown resolves + `_rc`/`_skinning` disposed (V1-4); cascadeDispatch `invalidateBindings` nulls both shader modules (V1-5).
   - **R4:** packed FrameParams caustic mode = `manifold-nee` on ceiling-miss (V2-1); `uploadPackedScene` throw destroys all created buffers + keeps prior scene (V2-2); BDPT default=2 → `maxLv=2`⇒`lvi=1`-loop pin test + re-pinned frameParamsPacker default goldens (V2-5, D2).
   - **R5:** `shared-bvh/src/__tests__/tlasSingularInstance.test.ts` (V2-3); `shared-bvh/src/__tests__/uv1RangeId.test.ts` (V2-4).
   - **R6:** `gltf-adapter/src/__tests__/matrixColumnPadding.test.ts` (hand-built MAT3/BYTE + MAT2/BYTE fixtures, V3-1); decode pixel-budget/clamp-before-alloc test (V3-2).
   - **R7:** svgf frame-chaining test (V3-3); bmfr overlap-knob reject/clamp test (V3-4); `webGpuTextureUpload` readback try/finally test (V3-5); OIDN concurrent-first-use single-session test (V3-6); neural preprocessing `[0,1]→[-1,1]`+HDR encoding test (V3-7/D5).
3. **naga gate** (shader-gate) — all WGSL compiles; expected 51/51 (or updated count if modules added — **T1-6 adds the `rgba16float` emissive binding**, so expect +N shaders/new binding validated; thin-film WGSL `8u` is UNCHANGED under D1=B). The D4 verbatim-metric mode adds a tracked (non-gating) naga-reject count for walkaround/RC.
4. **glsl gate** — pt-webgl2 GLSL compiles.
5. **behavioral gate** (`tools/behavioral-gate/gate.mjs`) — expected 29/29 (or refreshed if a golden legitimately re-pinned, e.g. the **frameParamsPacker BDPT default re-pin (D2)** or an atlas-decode reconcile — document each refresh). Thin-film is NOT re-pinned (D1=B keeps `8u`).
6. **run-ptwebgl2-h1** (~60s, bit-deterministic) — ran for EVERY pt-webgl2-touching wave (0B, 0C, T2-B, T3-D, T5-C, T1-1/2/5); T1-1's thin-film sub-step (D1=B) is warning-only → anchor unchanged. Anchor unchanged everywhere (or A/B-justified).
7. **wsl-gpu oracles repointed** for every moved file on the hardcoded-path list (T6, T2-C, T3-F, T4); `grep -rn '<old>' ~/projects/wsl-gpu/scripts` returns 0 stale hits.
8. **pre-push T1 GPU smoke** (lavapipe + dzn) — the only gate that compiles the runtime pass graph + runs the RC/TLAS/DDGI/CWBVH CPU brute-force oracles at 100%. MANDATORY for **R3 (rc device path), R4 (pt-webgpu), R5 (shared-bvh TLAS/BVH oracles)**, and for T6, T2-C, T3-B/F, T4-*, T5-B/E. R2's offscreen present should also get a real-GPU present check here. Distinguish a real assertion failure (match-rate mismatch) from a stale-import failure (`await import(...)` module-not-found ⇒ an un-repointed oracle path).
9. `npx eslint` / `knip` — 0 dead-code regressions (0-DEAD removals reflected).

**Wave dependency order:** **R → 0** → T2 → T6 → T3 → T4 → T1 → T5 → T8. (Wave R release-blockers land FIRST; 0B before T3-D; T6 solo among wh waves; T3-F/T2-C before T4; T2 tables before T1 consumers.)
**Wave R internal notes:** R1–R7 are file-disjoint and parallel EXCEPT: R2's `vanilla.ts` present-path edit must sequence after R1's `vanilla.ts` lifecycle edits (same file); R3's V1-5 cascadeDispatch edit must land before T2-C's `CascadeUniforms` codegen (bug before refactor); R4's atomic `uploadPackedScene` must be preserved when T2-A register-drives it. **All five User Decisions are RESOLVED — nothing is blocked.** **D2** (BDPT default 1→2) is a concrete R4/V2-5 sub-task. **D1** (thin-film per-backend limit + warning) is a concrete T1-1 sub-task (WGSL `8u` unchanged → no naga change). **D3** (HDR emissive) is the new **T1-6** feature task. **D4** (naga/Firefox Option B) is the T8 verbatim-metric gate mode + three doc edits (no traversal-core refactor). **D5** (neural preprocessing) is the R7/V3-7 training-pipeline task (no runtime change).
