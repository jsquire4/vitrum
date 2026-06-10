# §H Remediation Plan — 2026-06-09

> Execution plan for every item in `items_to_fix.md` §H (H1–H59) plus the road-to-100
> addendum buckets it concretizes (B13–B16, C4–C5, D10). Drafted by six domain planning
> agents (each read the cited code before planning), then lead-reviewed: every load-bearing
> fix approach was checked against the source; lead corrections are marked **[LEAD]**.
>
> **Effort legend:** XS = <1h · S = hours · M = 1–3 days · L = ~1 week · XL = multi-week.
> **Rules of engagement:** ◻ items must be re-verified against current code before editing
> (the §H verification protocol). Render-changing fixes (notably H1/H3/H13/H15/H17/H32/H33/
> H51-D/H52/H14-E) require before/after A/Bs against INDEPENDENT references — never against
> goldens produced by the same code path (the F-TLAS1 lesson).

---

## 0. Global sequencing (waves)

Waves order by: (1) stop-the-bleeding consumer-facing bugs, (2) truth reconciliation +
test gates so fixes can't silently regress, (3) render-changing correctness fixes with
A/B captures, (4) feature-completion / fidelity (overlaps road-to-100), (5) decision-gated
and L/XL items.

| Wave | Contents | Effort |
|---|---|---|
| **W1 — Surgical fixes + truth-now** | H49, H43, H45, H38, H50-doc, H36-doc, H37-a/c, H59-e, H24-A/C, H46/H47 warns+wire, H48 warns, H51-A/B/C warns, H12, H31 mechanical sub-items, H7-b/c/d, H58 | ~3–4 days |
| **W2 — Ledger + test gates** | H39/H40/H41-ledger (truth-now + consistency test), H42, H44, H53/H53b, H54, H55, H56, H59-a/b/c/d | ~1.5 weeks |
| **W3 — Render-changing correctness** | H1+H4+H40-flip, H2, H3, H9, H16, H17, H15, H19, H13, H32, H33, H35, H30, H24-B, H6, H8, H22 | ~1.5–2 weeks |
| **W4 — Feature completion** | H10+H11, H18, H23, H41-Option-A, H25, H26+H27+H56-update, H28, H29, H14 cluster, H51-D, H31-g, H34 cluster, H57 examples | ~2–3 weeks |
| **W5 — Decision-gated / large** | H5 (BDPT pt-webgl2), H7-a (additive regime), H52 (Disney lobes), H20-Option-A, H34-h | per decision |

Within each wave, items are independent unless a "Depends on" says otherwise — parallelize
freely. The W2 test gates (H53/H53b especially) are deliberately BEFORE W3 so the
render-changing fixes land with regression protection (the GL uniform-completeness test is
written in characterization mode first, flipped to enforcement when H1 lands).

---

## 1. Consolidated decision queue

> **STATUS (2026-06-09): ALL RECOMMENDATIONS ACCEPTED by the user.** Every D-item below is
> resolved to its recommended option. D3 resolves to "investigate first, likely delete the
> dead regime"; D7 resolves to "ledger demotion now (W2) AND the analytic-NEE implementation
> (W4)". Individual calls can be revisited before their wave starts if new evidence appears.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | **H5** pt-webgl2 `bdpt` | (A) gate option off honestly · (B) implement the host subpath driver (L) · (C) dummy-bind stopgap, document inert | **A** now; B as a scoped roadmap item. Not a removal of an implemented feature — honest labeling of an unimplemented one |
| D2 | **H7-e** pt-webgl2 caustic naming | (X) rename `'manifold-nee'` → honest name (breaking) · (Y) keep name, document the heuristic accurately | **Y**; track real MNEE port as road-to-100 fidelity item |
| D3 | **H7-a** additive accumulation regime | (A) enable on floatBlend devices (changes default path!) · (B) delete the dead regime (road-to-100 D1) · (C) investigate original intent first | **C → likely B.** [LEAD] The planner's "one-char fix" assumption is unsafe — `'normal'` is already the float-blend running-average path; enabling `'additive'` would change default accumulation on most hardware |
| D4 | **H13** delta-refraction pdf | (A) return 0 for delta lobes (standard, full-NEE weight) · (B) implement a real smooth BTDF pdf+sampler pair (couples to road-to-100 A3) | **A** — unbiased, S effort; B is the spectral-BSDF workstream |
| D5 | **H27** NRC training target | (A) full path-traced suffix (Müller §5, 2–4× cost) · (B) one-bounce Lo (direct@xs + DDGI bounce) · (C) decorrelated-DDGI only | **B** — real fidelity gain at bounded cost; A later if NRC graduates |
| D6 | **H28** neural ReLU aliasing | (A) ping-pong output buffers (host-only, +12–80 MB) · (B) fuse ReLU into conv kernels | **A** — host-only change; precede with the real-adapter repro |
| D7 | **H41** walkaround point/spot direct light | (A) implement analytic NEE loop in ReSTIR-DI shade (M) · (B) honest ledger demotion + warn only | **B immediately (W2), A in W4** — both |
| D8 | **H20** empty-scene behavior | (A) implement sky-only present path (M) · (B) fix the JSDoc to match reality | **B now**; A filed as road-to-100 item |
| D9 | **H31-d** RAF self-stop threshold | consecutive-throw frames before stopping | **N=5** (confirm) |
| D10 | **H31-f** analytic scene-AABB | (A) null-skip degenerate contribution · (B) derive real bounds from `decodeAnalyticParams` | **A now, B follow-up** |
| D11 | **H34-d** `tlas:false` multi-primitive guard | (A) warn + auto-upgrade to tlas · (B) throw | **A** (warn+upgrade); throw under `debug:true` |
| D12 | **H34-h** SceneBvh fingerprint-before-merge | requires a `sceneVersionTag` interface change on `updateFromCore` callers | accept the interface change (only caller is probeUpdatePass) |
| D13 | **H35** denoiser failure surface | (A) `FrameStats.denoiserState` core contract addition · (B) debug-only surface | **A** — additive, already-subscribed signal |
| D14 | **H36** orphan packages | scene-lighting: keep+relabel as host-utility vs relocate vs delete; stained-glass-extensions: prune `VITRum_USER_DATA_KEYS`, keep `SURFACE_TEXTURE_ID`, mark `packCameUBO` reserved | **keep+relabel / prune-and-keep** per the verify-true-deadness policy |
| D15 | **H46** walkaround `maxBounces` | (A) wire into DDGI feedback control · (B) warn-on-non-default + honest capabilities | **B now**; A needs an architecture pass |
| D16 | **H50** zero-consumer contract fields | (A) implement consumption per field (B-bucket, golden-breaking) · (B) JSDoc `@reserved` markers now | **B now**, consumption stays road-to-100 D3/B-bucket with the verified field list |
| D17 | **H52** Disney lobes in pt-webgpu | implement clearcoat/sheen/iridescence (L, stride bump) vs document asymmetry | **implement** (north-star), scheduled W5; sequence after H13 + H51-D |
| D18 | **H59-c** pt-webgl-fidelity baselines | delete vs archive-relabel | **archive-relabel** (sole cutover A/B reference) |

---

## 2. Section A — pt-webgl2 (H1–H8, H40, H49, H44/H51 pt-webgl2 parts)

*(line numbers re-verified at `2ac1edb` by the planning agent)*

#### H1 + H4 + H40 — `lights.count` upload + MIS reconciliation + ledger flip (ONE unit)
- **Fix:** (1) add `GlProgram.setUint` (`gl.uniform1ui`); (2) in `GlResources.drawAccumStep` after `#bindSceneTextures`, `prog.setUint('lights.count', scene.lightCount)` (`UploadedSceneTextures.lightCount` already exists); (3) H4: `intersectLightAtIndex` hardcodes `discretePdf = 1.0` at `light_sampling_functions.glsl.js:84,132,174` while NEE selection is power-weighted → set forward-hit `discretePdf = 1.0/float(lights.count)` as the conservative interim (document residual bias for unequal-power multi-light; full fix = store selection prob in the lights texture, follow-up); (4) flip the H40 ledger row in the same commit (the W2 consistency test will force this).
- **Files:** `gl/glProgram.ts`, `gl/glResources.ts:213`, `glsl/shader/sampling/light_sampling_functions.glsl.js`, `core/src/engine/promiseLedger.ts`.
- **Validation:** RENDER-CHANGING (analytic lights turn on). A/B vs an independent CPU/cross-backend reference on a two-rect-area-light Cornell (new pt-webgl2 variant of the wsl-gpu multiemitter capture). The H53 recording-mock uniform-completeness test flips from characterization to enforcement here.
- **Effort:** M · **Depends:** none — the section foundation · **Risk:** scenes get brighter (correct); residual MIS bias documented.

#### H2 — spectral CMF upload
- **Fix:** in `drawAccumStep` when spectral enabled: `setFloatArray` for `uCmfX/Y/Z` (81), the three CDFs (82, convert Float64→Float32), `setFloat` for the three integrals — all sourced from `@vitrum/shared-samplers` exports (`CIE_X_TABLE` etc., verified exported). Also upload `u_jakobCoeffs` + `iorCauchyA/B/C` (defaults 0 = documented no-op).
- **Files:** `gl/glResources.ts`, `index.ts`. · **Validation:** RENDER-CHANGING (black → spectral). A/B: before = all-black assertion; after = rfe08-style chromatic attenuation; cross-backend pt-webgpu spectral comparison as the hardware follow-up. · **Effort:** S · **Depends:** same-PR as H1 to avoid conflicts.

#### H3 — `backgroundAlpha` upload
- **Fix:** `prog.setFloat('backgroundAlpha', 1.0)` in `drawAccumStep` (opaque background; make configurable only if DOF compositing ever needs it). Note `backgroundBlur` default-0 is correct.
- **Validation:** RENDER-CHANGING (visible sky/HDRI appears). Covered by the H1 A/B scene + recording-mock assertion. · **Effort:** S.

#### H5 — `bdpt: true` never driven → **DECISION D1** (rec: gate off; driver = roadmap item)
- Option A files: `options.ts`, `index.ts`, `capabilities.ts`, `promiseLedger.ts`, dummy-bind for `uBdptLightPathTex` if FEATURE_BDPT can still compile. · **Effort:** A=S, B=L, C=S.

#### H6 — env `intensity`/`rotationY`
- **Fix:** read `environment.intensity ?? 1` (keep the `envMap != null ? : 0` outer gate); add `makeRotationYMat4` to `mat4.ts`; thread via existing `FrameUniforms.environmentRotation`. GLSL already consumes both. · **Effort:** S · no render change at defaults.

#### H7 — cluster
- **H7-a additive regime → DECISION D3** [LEAD: investigate before touching — do NOT treat as a one-char flip].
- **H7-b** delete `#dummy2dTex`/`#dummy2dArrTex` in `dispose()` (S).
- **H7-c** remove the duplicate `partitionSceneBySupport` inside `buildSceneTextures`; caller pre-filters; move warnings to `index.ts` (S).
- **H7-d** real `frameTimeMs` via `performance.now()` around `drawAccumStep` (S).
- **H7-e caustic naming → DECISION D2** (rec: document honestly, keep option name; photon-map's escaped-ray energy-add gets an explicit known-approximation comment + road-to-100 entry).

#### H8 — materialIndex aliasing invariant
- **Fix:** dev-mode assertion in `bvhTextureAdapter.ts` that no vertex carries two material ids (rec over comment-only — this bug class has recurred). · **Effort:** S.

#### H49 — specularColor/Intensity
- **Fix:** `materialsTexture.ts:167-168` → `m.specularColor ?? DEFAULT_SPECULAR_COLOR` / `m.specularIntensity ?? 1.0`; delete the false "core has no field" comments. Unit test pinning packed bytes for `specularColor:[1,0,0]`. Byte-identical at defaults. · **Effort:** S.

#### H44/H51 pt-webgl2 parts
- Root README pt-webgl2 row corrected AFTER H1/H2 land (interim: a "under repair, see items_to_fix §H" callout). New `packages/pt-webgl2/README.md` (entry point, capabilities, limitations, snippet). `procedural-sky` stays honestly `unsupported`. Verify `emissiveIntensity` multiplication in `materialsTexture.ts` (~line 155) — comment if correct. · **Effort:** S.

**Section ordering:** H1+H4+H40 → H2/H3 (same PR) → S-cluster (H6/H49/H7-b,c,d,e/H8) → D1/D3 decisions → README. **Total: M–L (3–5 days)**, dominated by the A/B harness + recording-mock test, not the code.

---

## 3. Section B — pt-webgpu (H9–H14, H48, H51 coercions, H52)

#### H9 — `bdptAdvanceFrame` kills rendering
- **Fix:** split `buildBindGroups` into the existing full build + a new `rebuildGroup2Only(sb, lightPathBuffer)` that recreates ONLY group 2 (groups 0/1/3 stay valid). `bdptAdvanceFrame` calls it instead of nulling; add a pointer-equality fast-out for stable host buffers. The `index.ts:850` gate then sees all groups non-null.
- **Validation:** recording-device-stub test (exactly one `createBindGroup` for slot-5 buffer; subsequent frame sets group 2); wsl-gpu T1 BDPT smoke. · **Effort:** S · **Risk:** retain/pass `sb`; scene reloads already do the full build first.

#### H10 — emissive-fold desync (emitter + material fast paths)
- **Fix:** extract `packFoldedMaterialEntry(primitive, scene)` from `buildPackedScene` (`uploadSceneBuffers.ts:286-295`); `updateEmitter` re-writes affected material slots; the material fast path re-applies the fold (currently packs RAW material) when the primitive backs a mesh-area emitter and `cameraVisibleEmitters` is on.
- **Validation:** unit test (fold present after setScene → preserved after `updateEmitter` color patch → preserved after `updatePrimitive` roughness patch → InverseSession round keeps emitter visible). · **Effort:** M · **Risk:** fold must apply exactly once — helper mirrors `buildPackedScene` precisely.

#### H11 — stale mesh-area emitter triangles on geometry/transform fast paths
- **Fix:** shared `hasMeshAreaEmitterForPrimitive(scene, id)`; the 4 geometry/transform fast-path branches re-run `packEmitterArrays` + light-tree rebuild when true. · **Validation:** skinned-emitter positions-patch test asserting `meshAreaLightsBuffer` matches new world-space tris; RENDER-CHANGING for animated emitters (A/B still). · **Effort:** S · **Depends:** after H10 (shared infra).

#### H12 — lite-tier capability truth
- **Fix:** branch `capabilities()` on `#traceTier`: lite → empty analytic set, emitters = `directional` only (params-UBO-only — verified `kernel.wgsl.ts:341-342`), env = `none`/`procedural-sky`, drop `pt-webgpu-bdpt` from experimentalFeatures. Runtime value is authoritative; the static ledger reconciliation is H39's (W2). · **Effort:** S.

#### H13 — delta-refraction pdf → **DECISION D4** (rec: return 0)
- **Fix:** `bsdf.wgsl.ts:69-76` opposite-hemisphere branch → `return 0.0`; keep the existing `max(pdf,1e-6)` guards at connection sites (verified present at `connect.wgsl.ts:242`). MIS resolves to full-NEE weight on delta transmission — unbiased. · **Validation:** RENDER-CHANGING on transmissive scenes; A/B vs high-spp independent reference; WGSL golden re-pin. · **Effort:** S.

#### H14 — cluster
- **H14-A** emissive-BVH-mesh lighting: synthesize an implicit mesh-area emitter for primitives with `emissiveIntensity>0` and no explicit emitter (guard against double-count when one exists). M.
- **H14-B** ReSTIR-PT producer NEE: add spot + mesh-area loops mirroring kernel/connect implementations; extract shared NEE helpers. M. After H51-D (shared spot stride).
- **H14-C** `getRestirPtResultBuffer` typed surface: optional method on the core Engine contract gated by the experimental feature; do NOT block on road-to-100 A1 compositing. S.
- **H14-D** try/finally + destroy on all mapAsync paths (`rgba16fReadback.ts`, adjoint transients). S.
- **H14-E** decouple HDRI radiance from `environmentSun.w` (dedicated HDRI-intensity UBO lane); RENDER-CHANGING at sun.w=0 + HDRI — A/B required; audit walkaround's reading of the same lane first. S.
- **H14-F** once-gate the buffer-ceiling warns. S.

#### H48 — denoiser silent no-op → warn on both PT backends (+ JSDoc). S.

#### H51 — coercions: **A** warn on maxBounces clamp (cap stays); **B** warn on distinct roughnessMap+metallicMap (ORM single-slot is intentional); **C** warn on ignored `extensions` keys (graduation follow-up tracked); **D** pack spot `penumbra/distance/decay` + point `distance/decay` (stride 12→16; smoothstep inner/outer cone + ranged decay falloff; `distance=0` = no cutoff preserved). D is RENDER-CHANGING + stride bump → goldens re-pinned, layout-verification test added. A/B/C = S; D = M.

#### H52 — Disney lobes → **DECISION D17** (rec: implement; W5)
- vec4 stride 23→26 (clearcoat/sheen/iridescence lanes), WGSL decode + three lobes (GGX clearcoat IOR 1.5, Charlie sheen, thin-film iridescence F0 port from the GLSL reference — beware WGSL `all()` for vec comparisons), pdf contributions added. Sequence AFTER H13 and H51-D; check `caustic.wgsl.ts` material indexing after the bump. RENDER-CHANGING; A/B vs pt-webgl2 same-scene as the independent cross-check. **Effort:** L.

**Section ordering:** H9 → H13 → H48/H12/H51-A/B/C → H10→H11 + H14-C/D/F → H51-D → H14-A/B/E → H52. **Total: XL** (dominated by H52 + H14 + H10/H11).

---

## 4. Section C — walkaround-hybrid core (H15–H24, H41, H46, H47)

#### H15 — UV plumb-through (road-to-100 B13)
- **Fix:** merged path: pass `{array: merged.uvs}` (already in scope) as `packUVIntoPositionW`'s second arg at `bvhCore.ts:246` (`packUVIntoPositionW` already handles stride-2 arrays, verified `packingHelpers.ts:90`). TLAS path (`:169`): lift `uvs` from the merge already computed at `:182-185` or extend `ScenePackResult` with `uvs` — both call sites in ONE commit.
- **Validation:** RENDER-CHANGING for textured scenes. A/B vs pt-webgpu/pt-webgl2 rendering the same textured scene (independent backend = the reference). T1 smoke for compile only. · **Effort:** S · do FIRST in this section.

#### H16 — DDGI invalidate + sun-dir desync (one commit)
- **Fix:** Part A: `_invalidating` flag on DDGI → `packProbeUpdateBlendParams` passes `hysteresis = 0.0` for the next blend window (`mix(coeff, prev, 0)` = full replace — the kernel already supports it); clear flag after upload; fix the JSDoc. Optional stronger variant: zero-write the atlases. Part B: verify `setLights` propagation to the probe pass actually lands pre-next-frame; add a dirty flag if not — closes the open Section-G sun-dir item together.
- **Validation:** behavioral: lighting change converges within one stride window (8 frames), not hundreds; `probeUpdateFrameParams.test.ts` golden updated; unit test pins hysteresis=0-exactly-once. · **Effort:** S.

#### H17 — skin kernel translation [LEAD-corrected math]
- **Fix:** seed `var sp = vec4f(0.0)` (zero, matching CPU `solveSkin`'s accumulate-from-zero); after the loop `sp.w` = Σwᵢ ≈ 1 for normalized weights — set `sp.w = 1.0` explicitly before `matrixWorld * sp` for robustness against unnormalized weights. Read `solveSkin` first to confirm exact convention, then mirror it exactly.
- **Validation:** CPU-vs-kernel unit A/B on a translated merged-mode single-bone cube (f32 tolerance). RENDER-CHANGING for translated merged-mode skinned meshes; check whether any committed reference uses one before claiming no re-capture. · **Effort:** S.

#### H18 — DDGI emitter blindness (road-to-100 B14), staged
- **Stage 1 (S):** warn on >16 lights truncation; document the unknown-kind guard.
- **Stage 2 (M):** port the RC probe-cast emitter NEE pattern (`probeRayCast.wgsl.ts:241-287` + dispatch wiring) into `probeUpdateRays.wgsl.ts` after `evalDirectLighting`; bind the existing `ExtraEmitterTri[]` buffer (reuse `collectRectAreaEmitterTrisFromCore`); guard on `emitterCount > 0`; test both BVH modes.
- **Validation:** RENDER-CHANGING. Rect-area-lit (no sun) scene: DDGI indirect goes 0 → non-zero; extend the wsl-gpu DDGI brute-force oracle with an emitter-lit case. · **Depends:** after H16 (clean convergence) and H22 (placeholder resolved).

#### H19 — stale GPU normals on fast paths
- **Fix:** per-primitive normal recompute + ranged `writeBuffer` into `bvhNormals` using `meshVertexRanges` on positions/transform fast paths (skin path exempt — already writes per frame). Full-rebuild fallback rejected (O(scene)). · **Validation:** buffer-readback unit test vs expected transformed normals. · **Effort:** S–M.

#### H20 — empty scene → **DECISION D8** (rec: JSDoc truth-fix now; sky-only present path filed to road-to-100). S.

#### H21 — `primaryLightIntensity` CDF desync
- **Plan-revision from code-read:** sun/directional emitters never enter the rect-area emitter CDF (`collectRectAreaEmitterTrisFromCore` collects rect/disc only), so the audited "contract violation" reduces to a stale/over-broad UBO contract comment. Fix the comment at `WalkaroundGPUPipeline.ts:434-438` + `updateLighting` JSDoc; re-verify after H41-A lands (which adds analytic lights to the direct path). · **Effort:** S.

#### H22 — phantom emitter: verify the RIS `emitterCount`/power gate by code-read FIRST; if gated → comment; if live → `power=0` sentinel or count-guard. S.

#### H23 — mesh-area emitter color/intensity
- **Fix:** inject emitter-derived Le for triangles of a mesh-area-referenced primitive at pack time (extend `packBVHEmissiveLeFromCore`/material resolution — the pt-webgl2 emissive-fold pattern); warn when `primitiveId` matches nothing. Update the pinned `directLightEmitterCore.test.ts` expectation. RENDER-CHANGING; A/B mesh-area-on-non-emissive-mesh scene. · **Effort:** M · after H15, H41.

#### H24 — engineering sub-items
- **H24-A** warn on unknown-primitive-id → material 0 (`bvhCore.ts:90-97`). XS.
- **H24-B** gate `_ready` on `_gpuOk`; add a `state()` accessor ('initializing'/'ready'/'failed'). S — with H16.
- **H24-C** drop the length-based probe-BVH change gate; rebuild on any `updateFromCore` (not hot-path). XS.
- Remaining H24 P2s (denoiser `state()` consumers → covered by H35; RC sun chroma; 80-byte stride; texture-height guard; GRIS dead lanes) → W4 hygiene batch, S each; GRIS lane removal coordinates with road-to-100 A1/A8 owners.

#### H41 — point/spot direct → **DECISION D7** (rec: B now + A in W4)
- **Option B (W2, XS):** ledger demote to `'approximate'` + JSDoc (DDGI-only routing) + setScene warn.
- **Option A (W4, M):** additive analytic-NEE loop in shade (separate from the RIS area pool — no pdf contamination; the pt-webgl2 architectural split), `Le/(d²+ε)·cos·vis` + spot cone falloff, using existing shadow-ray infra. RENDER-CHANGING; A/B vs pt-webgpu same scene; rect-area non-regression A/B.

#### H46 / H47 — option surface
- `maxBounces` → **DECISION D15** (rec: warn + honest capabilities); `maxSamplesPerPixel: Infinity` documented intentional; `causticStrategy` warn on non-'none'. H47: thread `ppgMaxSpatialCells` → `allocatePPGResources`; fix the JSDoc default (1 024, not 16 384); rename/clarify the ceiling constant; note dependency on H25/A2 for it to matter. Both S.

**Section total: ~2 days of S-items + 2 M-sessions (H18, H23/H41-A).**

---

## 5. Section D — PPG / NRC / neural (H25–H29, H56)

#### H29 — dTree stride single-sourcing (FIRST — H25 testing needs it)
- **Fix:** convert `PPG_UPDATE_WGSL` to `buildPpgUpdateWgsl(maxDTreeNodesPerCell)` (the existing `RESERVOIR_GI_STRIDE` template pattern); pipelineCompiler passes the live allocation value; add `PPG_DEFAULT_SPATIAL_CELLS = 1_024` beside the ceiling constant. Byte-identical at default. · **Effort:** S.

#### H25 — dTree interior flux + pdf consistency (gates road-to-100 A2)
- **Fix (both halves, one change):** (a) CPU post-order interior-flux propagation in `_mergeFluxAndRefine` after the leaf-flux merge, before serialize (reverse-index pass over BFS-ordered nodes = bottom-up; parents then carry subtree sums and the GPU sampler's existing `cBase+4u` child reads become correct at every level) — chosen over per-descent GPU subtree summation for cost + testability; (b) `ppgEvalPdf`'s `leafFlux/totalFlux` stays valid under (a) — verify the `totalFlux` header lane is leaf-only; (c) **fix the CPU oracle in lockstep** (same propagation before `dTreeSample`/`dTreePdf`) — without this, byte-identity stays green-while-wrong.
- **Validation:** sampling-vs-pdf χ² harness on a depth-2 tree (1M samples, p>0.05) — the house self-validating-math pattern; NEW wsl-gpu oracle test asserting distributional match (not just byte-identity); two-region guided-localization A/B render. · **Effort:** M · **Depends:** H29.

#### H26 — NRC spread predicate (+ H56 test)
- **Fix:** (1) `risGiNrc.wgsl.ts:232` seed `runningSum = 0.0` (not `a0term` — the tautology); (2) `:186` primary-segment pdf = camera per-pixel solid-angle pdf derived from `tanHalfFovY`/height (confirm UBO lane), not `1.0`. H56: add the `c = 0.01` default-config multi-segment test (characterization-mode until this lands, then enforcement).
- **Validation:** analytic predicate check (closed-form a0/spread for a known camera+bounce); A/B: NRC firing selectively instead of everywhere. · **Effort:** S · ship WITH H27.

#### H27 — NRC training target + torn records
- **Target → DECISION D5** (rec: one-bounce Lo = direct-light sample at xs + one DDGI bounce — replaces the pure DDGI distillation; fix `xsRough=1.0` hardcode while in there). **Torn records:** per-slot `atomic<u32>` claim flags (compare-exchange first-writer-wins) — minimal correct fix; workgroup compaction is the later upgrade. New group-4 binding → pipeline recompile gate.
- **Validation:** analytic one-bounce oracle (closed-form single-light Lambertian); collision-free slot coverage test at pixelCount = 2×recordCap. · **Effort:** M · **Depends:** H26.

#### H28 — neural ReLU aliasing → **DECISION D6** (rec: ping-pong)
- **Step 0:** one-shot real-adapter repro (same buffer at read + read_write bindings → expect validation error) on lavapipe; record verbatim. **Fix:** in the tensor-name→buffer resolver, allocate a distinct `${layer}_out` buffer for in-place ReLU layers and repoint downstream inputs — host-only, no WGSL edits.
- **Validation:** repro passes post-fix; full `denoiser:'neural'` init dispatches without validation errors on lavapipe (random weights — crash-freedom only; quality A/B deferred to A10 weights). · **Effort:** S.

**Section ordering:** H29 → H28(step0+fix) ∥ (H26+H56 → H27 → H25). **Total: ~L.** Gate impact: H25 unblocks A2 validation; H26+H27 are prerequisites for A6; H28 unblocks A10.

---

## 6. Section E/F — engine, shared-bvh, denoisers (H30–H35)

#### H30 — canvas backing store (highest consumer-facing priority)
- **Fix:** (1) `attachVitrum`: synchronous initial sizing — set `canvas.width/height` from CSS×dpr before the first frame, and unify the initial seed (currently backing-store 300×150) with the ResizeObserver path's contentRect×dpr math; ResizeObserver keeps both in sync thereafter; (2) `VitrumCanvas`: ref-stabilize `advanced` + `onAttachError` (the existing quality/onFrame ref pattern).
- **Validation:** the H54 happy-dom harness: canvas attributes correct pre-first-RAF + after a resize tick; extend `lifecycle.test.ts` beyond `typeof`. · **Effort:** S · do first — it also bootstraps the happy-dom infra.

#### H31 — engine/core cluster
- **a** `backendId` on the returned engine + gate the `advanced` cast by chosen backend; **b** pass `onError` to `configureWebGpuCanvas` in createProgressiveEngine; **c** optional `onDisposeError` channel in `wrapWithIdempotentDispose`; **d** RAF self-stop after N consecutive throws (**D9**, rec N=5) with `recoverable:false` report; **e** `detectWebGL2Sync` catch → `false` (the `document`-absent guard stays `true`); **f** scene-AABB analytic degenerate → null-skip (**D10**); **g** UVs for all four analytic-mesh builders (sphere lat/long, box per-face, cylinder θ/h, capsule hybrid) + `GeometryData.uvs` — RENDER-CHANGING for textured analytic primitives, A/B needed; **h** delete the dead `needsTlas` branch in `pickBackend` (warn already lives at the createEngine call site).
- **Effort:** M total (g is ~60 LOC, rest are 2–5-liners).

#### H32 — `traceTlasAny` glass-skip + double traversal
- **Fix:** add `skipGlass: bool` to `bvhIntersectFirstHitAtRoot` (the transmission bits are already in `idxEntry.w`, same filter as the any-hit path at `bvhIntersect.wgsl.ts:445-449`); replace the any-hit-then-closest-hit pair in `traceTlasAny` with ONE glass-aware closest-hit — fixes semantics AND removes the double traversal. Audit/update every `bvhIntersectFirstHit*` call site for the new parameter.
- **Validation:** CPU oracle case (glass between light and receiver: skipGlass=true → unoccluded, =false → occluded; merged-mode parity); T1 smoke; RENDER-CHANGING for TLAS+glass scenes → A/B + golden re-capture. · **Effort:** M.

#### H33 — `materialSig` Beer fields
- **Fix:** append `attenuationColor` (toFixed(4) ×3), `attenuationDistance` (normalize Infinity to the materialEntry sentinel), `thickness` to the signature — matching `materialSetHashFloats`. Unit tests both directions. Goldens covering transmissive multi-material scenes may legitimately move (list and re-pin). · **Effort:** S.

#### H34 — shared-bvh guards
- **a** throw on >0xFFFF leaf tri/instance counts (all 5 pack sites); **b** filter non-finite-centroid triangles pre-build with per-triangle warn (NaN root-AABB poisoning); **c** skip BLAS+splice for zero-instance instanced meshes (warn) — protects the bindings-tile invariant; **d** merged multi-primitive guard (**D11**, rec warn+auto-upgrade, throw under debug); **e** singular transform → skip-with-warn (not identity-at-origin); **f** fingerprint: fixed 64K evenly-spaced samples (max coverage at the same budget); **g** move the slow-rebuild timer to `updateFromCore` entry (currently times an object literal — `onSlowRebuild` can never fire); **h** fingerprint-before-merge via `sceneVersionTag` (**D12** — interface change, single caller).
- **Effort:** M (h is the involved one; g+h together).

#### H35 — OIDN failure surfacing end-to-end (rec **D13**: FrameStats path)
- **Fix:** (1) `OIDNDispatcherCore`: `onError` option + `getLastError()`; (2) walkaround dispatch site consumes `state()` post-dispatch (closes the zero-consumers gap), one-time warn per cohort on retryable, error channel on non-retryable; (3) core `FrameStats.denoiserState?: {status, reason}` populated per frame by all three engines.
- **Validation:** throwing-readback unit tests at each layer; maps to H54-list missing-test #10 (dispatch-time failure degradation: engine keeps returning raw HDR, no stuck state). · **Effort:** M · contract field lands first.

**Section ordering:** H30 → H33 → H35(contract→impl) → H31 mechanical → H31-g → H32 → H34 a–f → H34 g+h. **Total: ~2L.**

---

## 7. Section G — Claims, docs, tests, examples (H36–H45, H50, H53–H59)

#### H39/H40/H41 — promiseLedger truth-now + the consistency gate [LEAD: this strategy supersedes "flip with H1"]
- **Fix:** encode CURRENT truth immediately: pt-webgl2 analytic → matches the empty runtime set; pt-webgl2 mutations → `fallback-rebuild` (match `buildCapabilities`); pt-webgl2 emitters → `unsupported` (H1 inert) with `// TODO(H1): flip` markers; walkaround point/spot → `'approximate'` + routing JSDoc. **NEW TEST** `ledgerVsCapabilities.test.ts`: per backend, assert ledger row ≡ runtime `buildCapabilities()`/`*_SUPPORT` output — the self-enforcing gate that forces every future flip (verify side-effect-free importability of each backend's capability module first).
- **Effort:** M · runs BEFORE/CONCURRENT with code fixes, never waits for them.

#### H42 — fidelity matrix: retire the `pt-webgl` column, add `pt-webgl2` graded from §H truth (per-row table drafted in the planning transcript: spectral rows `unsupported` until H2, multi-emitter `unsupported` until H1, caustics `approximate`-heuristic, BDPT `unsupported` until D1 resolves); pt-webgpu spectral row gets the hero-λ-tint caveat (A3); evidence gates updated (fork-shader-smoke is gone). Verify the SSS row by code-read before grading. S.

#### H43 — CHANGELOG: add `### Removed` under [Unreleased] (pt-webgl, three-bindings, three-gpu-pathtracer, examples/, bridge subpaths; cite e14000c; name pt-webgl2 as the replacement); relocate fork-era Added entries. S.

#### H44/H45 — README truth pass: root README pt-webgl2 row (post-H1/H2 wording; interim callout); walkaround README (delete examples/neural-denoiser refs, fix phantom three/webgpu peer-dep, PPG/neural → "experimental", RC undersell → GPU-validated 2026-06-07); new pt-webgl2 README; dev README pick-to-click undersell; pt-webgpu README "pt-webgl-only textures" staleness. S total.

#### H36/H37/H38 — orphans + walkaround-rc + stale comments
- **H36 → DECISION D14** (rec: scene-lighting keep+relabel as host-utility, fix the false header; stained-glass-extensions: delete `VITRUM_USER_DATA_KEYS` (Three-era, truly dead), keep `SURFACE_TEXTURE_ID`, mark `packCameUBO` reserved; READMEs for both).
- **H37:** delete the phantom `/three` README section; `dispatchFrameRaw` JSDoc for the `invalidateBindings()` requirement (the frozen-bind-group correctness fix itself is tracked with the RC owner); glass-occludes-emitter-NEE comment/code mismatch verified + fixed with the RC owner; stale `TSL_TO_RAW_MAPPING.md`/`octahedralSolidAngles` docs.
- **H38:** the six stale-comment sites (risGi "fork" routing, luminance three-bindings cite, probeUpdatePass THREE header, probeGrid 8×8→3×3, sampleBudget first-frame tier — VERIFY INTENT first; if the tier-1-on-frame-1 behavior is a real under-sampling bug, promote it to a W3 fix). All S.

#### H50 — zero-consumer contract fields → **DECISION D16** (rec: `@reserved`/consumption-state JSDoc now on every field-group, stale THREE cite on anisotropy fixed; implementation stays the road-to-100 B-bucket with this verified field list). S.

#### H53/H53b — the two structural test gates
- **Recording-mock GL uniform-completeness test** (pt-webgl2): wrap `createMockGl` to record every `uniform*` call; assert the composed shader's declared uniform set ⊆ uploaded set per frame. Written in CHARACTERIZATION mode now (documents H1/H2/H3 as missing), flipped to enforcement when W3 lands. Catches the entire class forever.
- **Size/usage-validating GPU stub** (walkaround + pt-webgpu tests): `createBuffer` size/alignment checks + `createBindGroup` min-binding-size checks — kills the thrice-recurred 16B/32B dummy class in plain vitest.
- **naga gate feasibility [investigated]:** no maintained naga-wasm npm path; recommended form = `scripts/typecheck-shaders` shelling to the naga CLI, soft-gated on binary presence (the hard gate remains the pre-push T1 smoke). 
- **Effort:** S–M + M.

#### H54 — engine lifecycle tests: happy-dom env in the engine package; `attachVitrumLoop.test.ts` (RAF tick, ResizeObserver connect, dispose ordering — pins H30); `vitrumCanvasMounted.test.tsx` with tsx transform (replaces the `fs.existsSync` assertion). M.

#### H55 — independent-oracle upgrades: `FrameParamsSlot`↔WGSL-struct derivational cross-check; CPU-tracer drift tripwire vs a stored real-GPU reference (skip-gated on reference presence); same-lineage limitation comments on the remaining goldens. M.

#### H56 — default-config pins: NRC `c=0.01` characterization (with H26 update note) + lite-tier binding-budget test (counts pass-graph bindings vs `HYBRID_LITE_LIMITS` using the H53b stub). S.

#### H57 — examples (road-to-100 C4)
- `examples/` tree: shared `@vitrum-examples/cornell-scene` package + six ~50–100-LOC Vite apps — `create-engine`, `attach-vitrum`, `vitrum-canvas` (React), `progressive`, `pt-webgl2-direct`, `pt-webgpu-direct` — all speaking the reference-render capture protocol (`VITRUM_CAPTURE_READY` + URL params) so the Playwright harness can drive them. Anti-rot gate: example workspaces in root `workspaces` + covered by root `typecheck`. Building the pt-webgl2 example doubles as the H1 impact demo. · **Effort:** M.

#### H58/H59 — tools repair
- **H58:** inline `launchChromiumForCapture` into `capture-adapter-playwright.mjs` (Playwright chromium + WebGPU flags — removes the dangling-import class); implement `defaultCaptureCommand` for pt-webgpu; `scenario-presets.mjs` `pt-webgl` → live backend. S.
- **H59:** benchmark-runner README → live scripts only; reference-renders README Quick-Start rewrite (post-H58 capture path + H57 protocol pointer); `pt-webgl-fidelity/` → archive README (**D18**); rfe09 unverified-status note; neural-training param count 426,075 → 535,107 in all three places + dataset_spec backend refs. S.

**Section ordering:** Phase 1 no-dep docs (H38/H43/H45/H50/H36/H37/H59-e) → Phase 2 reconciliation (H39-41 + test, H42, H44, H58→H59) → Phase 3 test infra (H53/H53b/H54/H55/H56) → Phase 4 examples (H57). **Total: ~10–12 days, zero rendering risk.**

---

## 8. Cross-cutting execution rules

1. **Every ◻ item:** open the cited file and confirm the gap exists NOW before editing.
2. **Render-changing fixes** ship with: before-capture, after-capture, independent reference
   (CPU oracle / cross-backend / analytic — never a same-lineage golden), and re-pinned
   goldens listed in the commit message.
3. **Ledger discipline:** any commit that changes a backend capability MUST update the
   promiseLedger row in the same commit — `ledgerVsCapabilities.test.ts` enforces this
   once W2 lands.
4. **WGSL changes** ride the pre-push T1 smoke (naga compile + oracles); stride/layout
   changes additionally add a CPU-vs-shader layout assertion test.
5. **No deferring:** each wave's items land complete (fix + test + doc) — no "follow-up
   later" splits inside an item unless the plan explicitly stages it (H18, H41).
