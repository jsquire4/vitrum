# V1 Closure Plan — the road to an actual 100%

> Authored 2026-06-10 from a 10-agent full-source audit (every package read file-by-file;
> every load-bearing claim re-verified by lead code-read or executed command — verification
> method noted per item). Supersedes the *sequencing* of `plan/road-to-100.md`; that file
> remains the feature ledger and gets its status corrections from §1 below.
>
> **The core problem this plan fixes:** "how close to 100%?" has answered ~70% for two weeks
> because (a) every audit grows the denominator, (b) audits land mid-wave on a broken tree,
> and (c) "100%" was never frozen. This plan freezes it.

## 0. Definition of done (the frozen denominator)

**USER DECISION 2026-06-10: "Everything fully implemented."** No demotions as a path to
100% — advertised surfaces get genuinely implemented. Capability-reporting remains the
mechanism for *interim* honesty while an item is in flight, never the end state.

**v1.0 = 100% means, and ONLY means:**

1. **Every advertised surface is fully implemented.** Every core-contract field, option,
   emitter kind, environment kind, and quality dial is fully consumed by each backend that
   lists it. Until an item lands, hosts get a runtime warning (no silent drops, ever) — but
   the warning is scaffolding, not the deliverable. Consequences of this decision:
   - **Skinning:** implement real pose solving in pt-webgl2 + pt-webgpu (CPU `solveSkin` at
     ingestion + `bones`-patch re-solve + BVH refit), not a demotion to `approximate`.
   - **procedural-sky (pt-webgpu):** implement a real analytic sky (wire the orphaned
     `scene-lighting` Preetham params as the consumer), not a re-grade.
   - **Photon-map caustics (A4):** real progressive density estimation, or REMOVE the
     option entirely (implement-don't-fudge); no 21%-energy approximation survives.
   - **Neural/OIDN weights (A10/C2):** a trained checkpoint ships or is fetchable; the
     pipeline alone no longer closes the item.
   - **Lite tier (B12):** texture-pack env CDFs + tri-lights into the binding budget so
     lite scenes light correctly; the documented cliff no longer closes the item.
   - **Multi-directional emitters (pt-webgpu):** pack N, not warn-and-drop.
2. **Zero verified open bugs.** Everything in §3 closed; `items_to_fix.md` §H residue closed
   or explicitly re-scoped as post-1.0 with a capability/doc surface.
3. **Default paths robust:** context loss, oversized scenes, resize/dispose edges, and
   incremental-mutation desyncs handled or fail-loud.
4. **Apparatus green and binding:** working CI, an in-repo shader compile gate, packer↔shader
   stride parity tests, suite green, typecheck green, lint in the gate.
5. **Runnable examples for every public entry point** (C4).
6. **Docs/comments match code** (zero stale claims in contract docs, ledgers, plan files).

**Explicitly POST-1.0 (does not count against 100%)** — performance/variance optimizations
that are not advertised surfaces: SBVH (B7 tail), parallel BDPT subpath build (A9 tail —
the serial build is correct, just slow), multi-hero-wavelength spectral (hero-λ IS spectral
transport; the quad is variance), D6 bind-group memoization beyond the cheap wins, and
real-GPU radiometric A/B campaigns (tracked in `HARDWARE-VALIDATION-NEEDS.md`, per the
standing scope exclusion).

Once §1–§7 are done, the maturity question is answered against THIS list — not against an
open-ended SOTA ideal. New findings after the freeze go to a v1.1 ledger, not into the
v1.0 denominator.

---

## 1. Road-to-100 status corrections (apply to `plan/road-to-100.md` in Wave 1)

| Item | Ledger says | Code-truth (this audit) |
|---|---|---|
| B3 directional IBL | ✅ DONE | **Partial.** `envRadiance` on GI-escape is live (`risGi.wgsl.ts:264`), but `envImportanceSample`/`envDirectionalPdf` + CDF bindings 15-19 have ZERO call sites (lead-verified grep). No env NEE; DDGI probes (`probeUpdateRays.wgsl.ts:497-508`) and risGiNrc (`risGiNrc.wgsl.ts:299`) still see procedural/scalar sky. The uncommitted `hdri:'native'` ledger promotion is premature — hold at `approximate` until Wave 4. |
| A5 pt-webgl2 BDPT driver | ✅ DONE | **Wired but incorrect:** vertex rows built by 3 fragments with 3 RNG streams (`composeTraceGlsl.ts:342-398` + `pcg.glsl.js:6-10`) — incoherent light vertices. |
| A9 pt-webgpu BDPT subpath | ✅ mostly done | **New defect:** throughput/pdfFwd stored for a sampled-then-discarded `nextDir` (`bdptLightSubpath.wgsl.ts:351-431`); estimator internally inconsistent. |
| B14 DDGI emitters | ◻ open | **Half-landed + over-corrects:** H18 rect/disc NEE live, but the point-proxy fixture path was never removed → double-count (`probeUpdateRays.wgsl.ts:587` sums both; `coreEmittersToDDGILights.ts:155` still maps rect→fixture). Mesh-area emissive still invisible. |
| A2 PPG | ✅ DONE | Confirmed in code — but `HybridEngineOptions.ts:656` JSDoc still claims "single global cell"; delete. New defect: flux atomics can u32-wrap (`ppgUpdate.wgsl.ts:241`). |
| B11 | ◻ open | References deleted `three-bindings`; rewrite or drop. |
| B16 DI BRDF candidate | ✅ DONE | Confirmed — with a sampler/pdf floor mismatch (`ggxBrdf.wgsl.ts:205` α=max(r²,1e-3) vs `:223` r floored at 0.01, Schlick-k G1 vs exact Smith). |
| "Foundations ~90%" header | — | Re-state per §0: features ~85%, professional-for-others ~70%; both converge to 100% via this plan. |

---

## 2. Wave 1 — Stabilize the tree (verified ✅, BLOCKING everything)

1. **Finish the D3 material-stride work coherently** (fidelity-paramount: finish, don't revert):
   - Packer writes texels 85-92 (ao/light/bump layer ids + intensities/scale + envMapIntensity + 3 mat3 transforms); bump `MATERIAL_PIXELS` 85→93 (`materialsTexture.ts:21,198-200`).
   - `readMaterialInfo` assigns the new struct fields (`material_struct.glsl.js:90-118` currently uninitialized).
   - Consume them in `get_surface_record_function.glsl.js` (ao modulation, lightMap additive at emissive-on-hit, bump perturbation) — without this the fields are dead weight.
   - Repoint the 3 stale stride sites: `thin_film_tmm.glsl.js:5`, `inside_fog_volume_function.glsl.js:10`, `util_functions.glsl.js:45` — single-source the constant into the GLSL via template injection.
   - **Add the parity test:** packer `MATERIAL_PIXELS` must equal the GLSL-injected stride (greps the composed shader). This is occurrence #4 of the stride class; the test is non-negotiable.
2. **Fix `material.wgsl.ts:381`** unescaped backticks (file fails to parse — lead-verified via tsc; kills 17/23 engine test files).
3. **Resolve the `hdri` ledger/test conflict** (`promiseLedger.ts:253` vs `engineContract.test.ts:128`): per §1, revert the promotion to `approximate` (re-promote in Wave 4 when env NEE + probe HDRI land). Core suite green again (currently 1 failed — lead-verified by running vitest).
4. Commit; push per standing instruction only when user says so.

**Effort: S–M (a day).**

## 3. Wave 2 — Contract honesty at field level

All verified ✅ unless marked ◻ (agent-reported, re-verify before edit).

1. **Material-field consumption warnings.** Build a per-backend `CONSUMED_MATERIAL_FIELDS`
   set; on ingestion, `warnOnce` for every supplied-but-unconsumed field. Covers walkaround's
   wholesale map-ignoring (zero `TextureRef` reads in the package), pt-webgpu's dozen dropped
   maps/scalars ◻ (`transmissionMap`, `alphaMap`, clearcoat/sheen/iridescence maps,
   `specularColor/Intensity`, `thickness`…), vertex `colors` on all three.
2. **Implement skinning on pt-webgl2 + pt-webgpu** (lead-verified gap: `solveSkin` consumed
   only by walkaround; PT backends pack rest pose despite `'native'` grades). CPU
   `solveSkin` at ingestion + `bones`/`morphTargets` patch re-solve + BVH refit. Interim
   runtime warning until it lands.
3. **Implement `quality.tonemap`/`exposure`/`outputColorSpace`** (lead-verified dead on all
   three; operators already exist in shared-samplers — wire `applyTonemap` into each
   backend's composite/present path; walkaround un-hardcode ACES@1.0 `composite.wgsl.ts:106`).
   Contract says these exist; implement rather than retract (fidelity-paramount).
4. **`emissiveIntensity` default divergence:** pt-webgl2 `?? 0.0` → `?? 1.0`
   (`materialsTexture.ts:155`) to match pt-webgpu/walkaround/contract.
5. **pt-webgpu directional emitters:** pack N directionals (array, not the single
   `.find(...)` slot — `emitterPacking.ts:166`); NEE + MIS over all of them.
6. **Lite tier:** narrow `supportDetails` to match the actual group-0-only binding ◻
   (`index.ts:461`); per-scene warn when packed content can't bind; stop uploading what can
   never bind.
7. **`procedural-sky` on pt-webgpu:** implement a real analytic sky (Preetham via the
   orphaned `scene-lighting` package as parameter source; bake to equirect → reuse the HDRI
   path) replacing the heuristic tint ◻ (`environmentPacking.ts:64-86`). Interim re-grade
   to `'approximate'` only until it lands.
8. **`supportsAuxBuffers` on walkaround:** false (or expose variance) ◻ — flag definition
   requires variance AND motion vectors (`capabilities.ts:91`).
9. **pt-webgl2 `denoiser` option:** warn-degrade like pt-webgpu does ◻ (currently zero reads,
   zero warns).
10. **HDRI RGBA stride detection** ◻ (`environmentPacking.ts:100-119` assumes stride 3;
    detect w·h·4 and read stride 4; fix the misleading all-black warning).
11. **`AnalyticPrimitive.fallbackMesh` honored by pt-webgl2 partition** ◻
    (`partitionSceneBySupport.ts:76-87` drops without consulting it, contradicting
    `primitives.ts:54-58`).
12. **Stale-doc sweep** (all lead-verified or excerpt-quoted): `ppgMaxSpatialCells` JSDoc;
    `environment.ts:45-50` walkaround-HDRI "no-op" note; `material.ts` @reserved notes
    overtaken by working tree; walkaround `index.ts:33` phantom `/three` subpath;
    shared-samplers `lightTree.ts:18` pre-B8 header; `oidnDispatcherCore.ts:46` phantom
    constructor throw; `bmfrWebGPU.ts:71` wrong worldPos-omitted claim; scene-lighting
    `index.ts:9` false "currently consumed by" claim; D5 leftovers.

**Effort: M (2–4 days). Highest trust-per-hour in the plan.**

## 4. Wave 3 — Close the verified bug tail

Correctness bugs, ordered render-impact-first. ✅ = lead-verified, ◻ = agent excerpt, verify-then-fix.

1. ✅ **Caustic module strides:** point 2→3 (`caustic.wgsl.ts:219,418,641`), photon point
   `:947` and spot `:957` (spot is 4), spot emission axis negation `:965`. Single-source the
   stride constants shared with `kernel.wgsl.ts`.
2. ◻ **pt-webgpu BDPT estimator coherence:** extend with the direction actually traced
   (apply `cosScatter/pdfScatter` to the traced segment; store pdfs for that segment), or
   restructure to sample-once-per-vertex (`bdptLightSubpath.wgsl.ts:351-431`).
3. ◻ **pt-webgl2 BDPT RNG coherence:** seed the subpath RNG per-column (vertex id), not per
   fragment-row, so one vertex's 3 rows come from one stream (`composeTraceGlsl.ts:342-398`).
4. ✅ **DDGI rect/disc double-count:** stop mapping rect/disc→fixture in
   `coreEmittersToDDGILights.ts:155-180` now that H18 NEE covers them (keep point/spot
   fixtures). + **mesh-area emissive into the probe NEE list** (finishes B14;
   `bvhSceneHelpers.ts:352` currently excludes them).
5. ✅ **`hdrLuminanceBilateral` i32/u32 fix** (`hdrLuminanceBilateral.wgsl.ts:51`) — and the
   Wave 6 compile gate that would have caught it.
6. ◻ **Mutation-desync cluster (pt-webgpu):** material fast path rejects texture-map patches
   (route to repack) (`incrementalPatch.ts:63`); implicit emissive emitters tracked by the
   H11 staleness repair (`emitterPacking.ts:482`); `emissive` patches re-pack NEE radiance;
   `angularDiameter` in `MutableSceneBufferFields`; ReSTIR-PT reservoir clear on
   reset/setScene (`index.ts:1039`).
7. ◻ **Mutation-desync cluster (walkaround):** shared-material `bvhIndex` full-range upload
   (`HybridEnginePrimitiveUpdates.ts:1027-1053`); inverse-transpose + normalize in both
   normal-rotation helpers (`:246-314`).
8. ◻ **B16 floor unify:** one α floor + exact Smith G1 in `ggxVndfReflectionPdf`
   (`ggxBrdf.wgsl.ts:204-227`).
9. ◻ **Spectral MIS halves:** spectralize the BSDF-side connections (`connect.wgsl.ts:264,276`)
   to match NEE/miss.
10. ◻ **Small-but-real:** PPG flux u32 saturation (`ppgUpdate.wgsl.ts:241`); SampleBudget
    reads the Welford ping-pong's fresh side (`SampleBudgetPass.ts:76`); restirPtSpatial
    passes `pdfSrc` not `W` (`restirPtSpatial.wgsl.ts:210`); GI-state import validates
    origin/spacing/dims (`probeUpdatePass.ts:763`); stained-glass visibility TLAS-blind
    (`surfaceTextures.wgsl.ts:159` — gate or implement).
11. ◻ **Lifecycle edges:** pipeline `dispose()` clears `_initialized`
    (`WalkaroundGPUPipeline.ts:1875`); OIDN resize stale-write guard (`oidnFinal.ts:458`);
    neural resize vs fixed InferenceGraph (`neural.ts:224`); PPG resize drops
    `ppgMaxSpatialCells` + readback race (`PPGCoordinator.ts:147,232`); `attachVitrum`
    ResizeObserver updates canvas backing store (`vanilla.ts:274`); progressive seed dims
    contract (`progressiveHandoff.ts:286`); `advanced.device` override guard + cross-backend
    `advanced` fallback warning (`createEngine.ts:388-527`).
12. **Honest-skinning fidelity item** from Wave 2.2: `solveSkin` at PT ingestion + bones-patch
    path → re-grade.

**Effort: L (≈1 week).**

## 5. Wave 4 — Finish the environment-lighting pillar (the one true fidelity wave)

The machinery exists; this is consumption wiring. Completes B3 for real.

1. **Env as DI NEE candidate** in walkaround RIS: call `envImportanceSample` for an M_ENV
   candidate with `envDirectionalPdf` in the MIS bookkeeping (`ris.wgsl.ts`).
2. **HDRI into DDGI probe misses** (`probeUpdateRays.wgsl.ts:497` — bind the equirect +
   sample by direction; keep procedural fallback when no map).
3. **risGiNrc parity:** add `environmentSample` to its requires; `envRadiance` on escape
   (`risGiNrc.wgsl.ts:299,382`).
4. **RC env binding** (decision-gated by A7): pass env view/sampler through `RCFrameInputs`
   (`HybridEngineRC.ts:62`, `cascadeDispatch.ts:452`), or document RC as sun+emissive-only.
5. **`updateEnvironment` directional fast-path** (B3 remaining) — rebuild CDFs on swap.
6. **Then** re-promote the ledger `hdri` grade and re-pin the contract test (closes the
   Wave 1.3 hold).
7. Optional same-wave: default-sun NEE in shade (currently locked behind the stained-glass
   flag — `stainedGlassShade.wgsl.ts:68`) ◻ — decision-gate.

**Effort: L (≈1 week incl. A/B harness runs on the rig).**

## 6. Wave 5 — Provision & harden

1. **Examples (C4):** one runnable core-Scene example per public entry point (`createEngine`,
   `attachVitrum`, `VitrumCanvas`, `createProgressiveEngine`, both PT factories, hybrid
   factory). Required by §0.5.
2. **Robustness:** WebGL2 context-loss listeners + recoverable error surface (zero handling
   today — lead-verified grep); `MAX_TEXTURE_SIZE`/array-layer validation at pt-webgl2 upload;
   `maxStorageBufferBindingSize`/`maxBufferSize` guards in pt-webgpu `createStorageBuffer`;
   mesh-area-light expansion cap + warn ◻ (`emitterPacking.ts:375`).
3. **Perf hygiene (agent-reported, verify then fix):** dedupe the double
   `packEmitterArrays`/`environmentParams` per pack (~133 MB transient per setScene with a
   4K HDRI — `uploadSceneBuffers.ts:394,405`, `emitterPacking.ts:550,565`); D6 bind-group
   memoization if cheap.
4. **Weights (A10/C2, in scope per §0):** run the real training pipeline
   (`tools/neural-denoiser-training`) on a captured dataset on the rig; vendor/fetch the
   checkpoint; fix the in-place ReLU read/read_write binding (H28) first; quality A/B. For
   OIDN, vendor or auto-fetch a color-only ONNX model so `oidn-final` is turnkey.
5. **Lite tier (B12, in scope per §0):** texture-pack env CDF + tri-lights into the ≤10
   storage-buffer budget so lite scenes get analytic lights + HDRI.
5. **Orphan/dormant packages:** scene-lighting (orphan — wire into a consumer or mark
   standalone-util in README), stained-glass-extensions dormant packer (@reserved is fine;
   fix the false header), dev-package duplication nits.

**Effort: M–L.**

## 7. Wave 6 — Fix the apparatus (what stops the next regression class)

1. ✅ **CI is dead:** `.github/workflows/ci.yml` checks out the deleted fork and calls two
   nonexistent scripts (lead-verified). Replace with: install → typecheck → vitest →
   lint → shader-gate (below). This is the only item with zero code risk and maximal payoff.
2. **In-repo shader compile gate:** naga-wasm (or deno+wgpu in CI) parse/validate over EVERY
   exported composed WGSL string + the pt-webgl2 composed GLSL (glslang). Would have caught
   the backtick break, hdrLuminanceBilateral, and the D0 stride field-accessor class.
3. **Stride/layout parity tests:** packer constant ↔ shader-injected constant for materials
   (Wave 1), emitters, light tree, reservoirs — one shared test pattern.
4. **Upload-gap recorder generalization:** extend `uploadGapGuard`-style by-name uniform
   recording to all pt-webgl2 programs; size-validating GPU stub for pt-webgpu (H53–H56 top-3).
5. **Lint into `verify:mechanical`**; fix the two hoisting-only vitest devDeps; document the
   WSL node-on-PATH footgun in CONTRIBUTING.
6. Optional: coverage reporting (visibility only, not a gate).

**Effort: M.**

## 8. Wave 7 — The 100% gate

1. Re-run the §H + V28 ledgers against the tree; close or re-scope residue per §0.
2. Update `road-to-100.md` (corrections from §1 + final statuses), `items_to_fix.md`,
   CHANGELOG, package READMEs, memory.
3. **Re-run the standing maturity prompt against the §0 definition.** Pass = every check in
   §0 green. That run prints 100% or it prints the named residue; nothing else counts.

---

## Decision gates (asked one at a time as reached, per working preferences)

| Gate | Wave | Options (recommendation first) — per §0 "fully implemented", fudge-keeping options are off the table |
|---|---|---|
| A4 photon-map | 3 | Build real progressive photon map (XL) / REMOVE the `photon-map` option (MNEE sole caustic path). Both satisfy §0; "keep the fudge" does not. |
| A7 RC role | 4 | Keep + finish (fix receiver math, bind env, add point/spot) / retire the subsystem in favor of DDGI⊕ReSTIR-GI. |
| A8 GRIS default | 3 | GRIS default-on (unbiased default fits §0) / keep biased default with the bias quantified+documented. |
| Sun NEE default | 4 | Unlock from stained-glass flag, default-on / keep flag-gated. |
| A6 NRC posture | 5 | Fix the spread/training-target semantics + ship as opt-in-with-weights / default-on after A/B. |

## Sequencing summary

Wave 1 (day) → Wave 2 (≈1 week with skinning/sky/tonemap now implementation work) →
Wave 3 (week) → Wave 4 (week) → Waves 5+6 (1–2 weeks; weights training + lite packing are
the long poles, parallelizable with 6) → Wave 7 (gate). **Under the "everything fully
implemented" definition this is a multi-month arc (realistically 6–10 focused weeks), and
the honest percentage will climb wave by wave instead of jumping** — that is the cost of
the stricter definition, accepted 2026-06-10. New findings after this freeze go to a v1.1
ledger; they do not move the v1.0 denominator.
