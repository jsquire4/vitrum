# Code Gap Master Plan — vitrum

> **Purpose:** Exhaustive inventory of every code gap, unfulfilled promise, approximate row that should become native (or become an explicit fidelity toggle), and clear miss where the implementation is "correct" but not what a professional library should ship.
>
> **Scope:** Code only. Excludes documentation, release governance, cross-host GPU validation evidence, and npm publish posture.
>
> **Closure standard:** A row is "closed" when the running code either (a) implements the documented semantics end-to-end, or (b) is honestly reported at runtime via capabilities + structured warnings, or (c) is removed/downgraded from the public contract. Label promotion without behavior change does not count.
>
> **Source of truth:** `packages/**` as of 2026-06-16 HEAD. Cross-check: `promiseLedger.ts`, `consumedMaterialFields.ts`, `inverseSession.ts`, `featureReport.ts`, behavioral gate configs, `plan/road-to-100-gap-ledger-2026-06-11.md`, `items_to_fix.md` §H (re-verify line numbers before editing).

---

## 0. Inventory at a glance

| Category | Open items (approx.) | Notes |
|----------|---------------------:|-------|
| MaterialSpec rows (non-native) | **79** | 50 walkaround approximate + 11 WH unsupported + 4 PT×2 shared unsupported + 4 PTGL2 approx + 14 PTWG approx |
| Contract / method asymmetry | **18** | Optional Engine methods, frame-input semantics, captureFrame text drift |
| Mutation paths | **12** | fallback-rebuild vs native, lite-tier blocks |
| Walkaround RTGI fidelity | **45+** | Quantization, GI reservoir, alpha transport, emissive PDF |
| pt-webgpu fidelity | **25+** | Extension maps, inverse, lite cliff, specialty paths |
| pt-webgl2 fidelity | **8** | Geometry fast paths, SSS model, emitter shadow |
| Inverse (pt-webgpu only) | **35+** | Texture opt, perceptual losses, path-replay domain |
| glTF adapter | **40+** | Compression, edge cases, planner gaps |
| Shared infrastructure | **15** | BVH pick, defineUbo, RC envIntensity |
| Opt-in subsystems | **12** | NRC bias, PPG cadence, neural weights, GRIS default |
| Engine / host integration | **10** | Fidelity profile missing, progressive handoff edges |
| Test / gate code gaps | **8** | pt-webgl2 absent from behavioral gate, oracle coverage |
| Architecture / meta | **6** | Effective capabilities, backend routing, compile-time toggles |

**Total trackable work items in this document: ~300+** (many are small; many bundle into fidelity-profile phases).

---

## 1. MaterialSpec — full 65×3 matrix

Legend per cell:
- **N** = native (ledger)
- **A** = approximate (ledger) — needs native path OR explicit fidelity-tier label
- **U** = unsupported (ledger) — needs implementation OR honest permanent unsupported + backend routing
- **Action codes:** `IMP` implement native, `PROM` promote after A/B, `TOG` fidelity toggle (native vs fast), `RT` route to pt-webgpu, `ACC` accept unsupported with warning, `DOC` fix contract text only

### 1.1 Universal unsupported (all 3 backends)

| Field | WH | PT-GL2 | PT-WG | Action |
|-------|:--:|:------:|:-----:|--------|
| `displacementMap` | U | U | U | **IMP** — tessellation or vertex displacement pipeline; at minimum one PT backend |
| `displacementScale` | U | U | U | same |
| `displacementBias` | U | U | U | same |
| `receiveShadow` | U | U | U | **IMP** or **ACC** — GI-non-physical; if ACC, document as permanent unsupported with warning (already warns at setScene) |

### 1.2 Walkaround-only unsupported (route to PT)

| Field | Action |
|-------|--------|
| `spectralAttenuation` | **RT** — do not implement on walkaround; strengthen glTF planner rejection |
| `dispersionAbbeNumber` | **RT** |
| `scatteringCoefficient` | **RT** |
| `scatteringAnisotropy` | **RT** |
| `scatteringCoefficientRGB` | **RT** |
| `frontLayer` | **RT** |
| `backLayer` | **RT** |
| `thinFilmStack` | **RT** |

### 1.3 Per-field matrix (abbreviated — every non-native row)

#### Walkaround-hybrid (50 approximate + 11 unsupported above)

| Field | Mode | Primary gap (code truth) | Action |
|-------|------|--------------------------|--------|
| `baseColor` | A | RGBA8 per-tri quantization (`packingHelpers`) | **TOG** full f32 lane vs quantized |
| `roughness` | A | u8 lane | **TOG** |
| `metallic` | A | u8 + binary isMetal bit | **TOG** |
| `shadingModel` | A | unlit terminal branch only | **PROM** or document |
| `alphaMode` | A | blend not GI participant; mask approx | **IMP** stochastic GI alpha OR **TOG** |
| `alphaCutoff` | A | scalar cutout | **PROM** with alpha transport |
| `opacity` | A | fractional blend OIT-only | **IMP** GI participation OR **ACC** |
| `transmission` | A | 4-bit trans lane | **TOG** |
| `ior` | A | u8 [1,3] | **TOG** |
| `attenuationColor` | A | pre-baked Beer per tri | **TOG** per-hit Beer |
| `attenuationDistance` | A | same | **TOG** |
| `thickness` | A | fixed slab + map exponent | **TOG** |
| `baseColorMap` | A | sampled but glass tint path scalar | **PROM** |
| `normalMap` | A | atlas + derived TBN | **PROM** GI reuse retrace |
| `normalScale` | A | | **PROM** |
| `roughnessMap` | A | G channel | **PROM** |
| `metallicMap` | A | B channel | **PROM** |
| `transmissionMap` | A | | **PROM** |
| `thicknessMap` | A | | **PROM** |
| `emissiveMap` | A | texel PDF not exact on DDGI/analytic | **IMP** alias table OR **TOG** averaged |
| `alphaMap` | A | cutout yes; blend GI no | **IMP** blend GI |
| `aoMap` | A | GTAO multiply | **PROM** |
| `aoMapIntensity` | A | | **PROM** |
| `bumpMap` | A | finite-diff bump | **PROM** |
| `bumpScale` | A | | **PROM** |
| `clearcoat` | A | suffix/receiver proxy | **PROM** |
| `clearcoatRoughness` | A | | **PROM** |
| `clearcoatMap` | A | | **PROM** |
| `clearcoatRoughnessMap` | A | | **PROM** |
| `clearcoatNormalMap` | A | | **PROM** |
| `clearcoatNormalScale` | A | | **PROM** |
| `sheen` | A | | **PROM** |
| `sheenColor` | A | | **PROM** |
| `sheenRoughness` | A | | **PROM** |
| `sheenColorMap` | A | | **PROM** |
| `sheenRoughnessMap` | A | | **PROM** |
| `iridescence` | A | | **PROM** |
| `iridescenceIor` | A | | **PROM** |
| `iridescenceThicknessRange` | A | | **PROM** |
| `iridescenceMap` | A | | **PROM** |
| `iridescenceThicknessMap` | A | | **PROM** |
| `anisotropy` | A | shade/receiver; isotropic reservoirs | **PROM** / **TOG** |
| `anisotropyRotation` | A | | **PROM** |
| `anisotropyMap` | A | | **PROM** |
| `specularColor` | A | | **PROM** |
| `specularIntensity` | A | | **PROM** |
| `specularColorMap` | A | | **PROM** |
| `specularIntensityMap` | A | | **PROM** |
| `lightMap` | A | camera-visible first-hit only | **ACC** or **IMP** probe transport |
| `lightMapIntensity` | A | | same |
| `emissive` | N | — | — |
| `emissiveIntensity` | N | folded into Le | — |
| `envMapIntensity` | N | HDRI DI only | — |
| `extensions` | N | surfaceTextureId, skipEmitter | — |

**Walkaround structural gaps (not single fields):**
- W-GI-01: ReSTIR-GI reservoir stores geometry+Lo only (`reservoirGi.wgsl.ts`); rich reuse uses `restir_gi_receiver_phat_from_payload` but temporal previous-domain material recast is best-effort
- W-GI-02: Default GRIS off — 4 documented bias sources (`jacobianShift.wgsl.ts`, `spatialGi.wgsl.ts`, `temporalGi.wgsl.ts`)
- W-GI-03: DDGI `maxBounces==1` → direct-only probes (`HybridEngine.ts:659`)
- W-GI-04: Material-only `updatePrimitive` skips DDGI BVH sync (`applySubsystems: false`)
- W-GI-05: `addPrimitive`/`removePrimitive` always full setScene rebuild
- W-GI-06: `denoiser:'none'` implemented (`none.ts`) but excluded from `VALID_DENOISERS` (`HybridEngineOptions.ts:22-29`)
- W-GI-07: `supportsAuxBuffers: false` but emits `normalDepth` + `motionVectors` (variance never exposed)
- W-GI-08: `maxSamplesPerPixel` ignored (realtime — warn only)
- W-GI-09: `causticStrategy` / `causticOptions` warn-ignored (by design for walkaround)
- W-GI-10: Checkerboard / quality tiers are performance knobs without unified `EngineFidelityProfile`
- W-GI-11: DDGI probe lights max 16, unknown kinds dropped (`probeUpdateLights.ts`)
- W-GI-12: DDGI shadows glass tint ignored in probe rays (opaque first-hit)
- W-GI-13: ReSTIR-GI glass refraction walk bounded (`risGi.wgsl.ts`) — not full caustics
- W-GI-14: Emissive map texel PDF still approximate on DDGI/analytic paths (2026-06-16 partial close)
- W-GI-15: `lightMap` not in GI transport
- W-GI-16: Fractional alpha blend not ReSTIR/GI participant (OIT path exists)
- W-GI-17: Analytic primitives → generated mesh fallback (not native analytic traversal)
- W-GI-18: `updateEnvironment` HDRI opaque handle → intensity-only unless host resolver (`HybridEngine.ts`)
- W-GI-19: `procedural-sky` approximate (finite bake resolution)
- W-GI-20: Neural denoiser: starter weights only; resize breaks graph until fallback (`neural.ts`)
- W-GI-21: OIDN async stale-by-one-frame (`oidnFinal.ts`)
- W-GI-22: NRC biased experimental; warm-up gate exists but default off
- W-GI-23: PPG guides but defensive MIS blend; not production-tuned cadence
- W-GI-24: RC opt-in; envIntensity not threaded (`cascadeDispatch.ts:825`)
- W-GI-25: Skinned mesh: GPU skin path exists; morph+tangents fallback CPU
- W-GI-26: Instanced mesh: TLAS path; lite tier forces merged BVH
- W-GI-27: `restirPtReuse` compile-time pipeline variant (cannot toggle per frame without rebuild)

#### pt-webgl2 (4 approximate + 4 unsupported shared)

| Field | Mode | Gap | Action |
|-------|------|-----|--------|
| `shadingModel` | A | unlit not emissive light | **DOC** or **IMP** |
| `thickness` | A | closed-surface vs thin-shell | **IMP** |
| `thicknessMap` | A | same | **IMP** |
| `scatteringCoefficientRGB` | A | simplified SSS single-scatter | **IMP** random walk parity with pt-webgpu |
| `extensions` | U | deliberate non-read | **ACC** |
| `displacement*` | U | (shared) | **IMP** |

**pt-webgl2 structural gaps:**
- PTGL-MUT-01: `transform`/`positions`/`topology`/`addPrimitive`/`removePrimitive` → fallback-rebuild (`PT_WEBGL2_MUTATIONS`) while `incrementalPatchSupport` all true
- PTGL-MUT-02: Material fast path rejects texture-map fields (`mutateSceneTextures.ts`)
- PTGL-EMIT-01: `emitterCastShadow` approximate — forward emissive-hit not shadow-flag-skipped (`PT_WEBGL2_SHADOWS`)
- PTGL-ENV-01: `procedural-sky` approximate (256×128 bake)
- PTGL-ENV-02: Jakob global coeffs `[0,0,0]` when spectral on — volumetric spectral albedo flat (`index.ts`)
- PTGL-OPT-01: `bdpt` fixed `bdptMaxLightBounces: 3` in host (`index.ts`)
- PTGL-OPT-02: `causticStrategy` names (`manifold-nee`, `photon-map`) are heuristics not true MNEE/SPPM (`options.ts`)
- PTGL-OPT-03: `randomType` Sobol/Stratified pinned off (`featureTypes.ts`)
- PTGL-OPT-04: `FEATURE_FOG`, `FEATURE_BACKGROUND_MAP` pinned off
- PTGL-TIER-01: Lite tier disables aux MRT; `supportsAuxBuffers` runtime false vs ledger true
- PTGL-API-01: No `createInverseSession`, no `seedAccumulator`, no GI export, no progressive seed
- PTGL-API-02: No `updateLighting`
- PTGL-DENO-01: Only `none` + `oidn-final` work; others warn-degrade
- PTGL-ANALYTIC-01: Analytic shapes tessellated to mesh (`fallback-generated-mesh`)
- PTGL-GATE-01: **Not in behavioral gate** (`tools/behavioral-gate/gate.mjs` — pt-webgpu + walkaround only)

#### pt-webgpu full tier (14 approximate + 4 unsupported shared)

| Field | Mode | Gap | Action |
|-------|------|-----|--------|
| `shadingModel` | A | unlit terminal | **DOC** |
| `thickness` | A | closed-surface tracer | **IMP** thin-shell volume |
| `thicknessMap` | A | same | **IMP** |
| `clearcoatMap` | A | forward yes; BDPT/ReSTIR-PT/adjoint partial | **IMP** specialty path parity |
| `clearcoatRoughnessMap` | A | | **IMP** |
| `clearcoatNormalMap` | A | megakernel yes; not all payload schemas | **IMP** |
| `sheenColorMap` | A | | **IMP** |
| `sheenRoughnessMap` | A | | **IMP** |
| `iridescenceMap` | A | | **IMP** |
| `iridescenceThicknessMap` | A | | **IMP** |
| `specularColorMap` | A | | **IMP** |
| `specularIntensityMap` | A | | **IMP** |
| `specularColor` | A | scalar native in shader; ledger approximate pending proof | **PROM** |
| `specularIntensity` | A | same | **PROM** |
| `extensions` | U | deliberate | **ACC** |
| `displacement*` | U | | **IMP** |

**pt-webgpu structural gaps:**
- PTWG-API-01: No `setSize()` — resize via `FrameInput.viewport` only (`promiseLedger` mutations.resize unsupported)
- PTWG-API-02: No `updateLighting`
- PTWG-API-03: No `getProgressiveSeedTexture` (walkaround-only source)
- PTWG-API-04: No `giStatePersistence`
- PTWG-MUT-01: `addPrimitive`/`removePrimitive` fallback-rebuild
- PTWG-INV-01: `createInverseSession` throws on `loss: ssim|lpips` (`inverseSession.ts:334-338`)
- PTWG-INV-02: `kind: texture` throws — Phase 2 (`inverseSession.ts:1153-1157`)
- PTWG-INV-03: Path-replay single-bounce RGB direct-light only (`pathReplayRenderRegimeIssue`)
- PTWG-INV-04: Multi-bounce forward → FD downgrade
- PTWG-INV-05: Spectral forward → FD downgrade
- PTWG-INV-06: Transport fields (ior, transmission, thickness, attenuation*) → FD only
- PTWG-INV-07: Visibility fields (opacity, alphaCutoff) → FD only
- PTWG-INV-08: Normal fields (normalScale, bumpScale, clearcoatNormalScale) → FD only
- PTWG-INV-09: envMapIntensity → FD only
- PTWG-INV-10: Texture maps (normal, bump, alpha, displacement, etc.) → FD or partial replay
- PTWG-INV-11: Soft-sun directional partially replayed; capped mesh-area / exact texel PDF not
- PTWG-INV-12: Forward light-selection MIS parity not in adjoint
- PTWG-INV-13: Indirect / multi-bounce adjoint not implemented
- PTWG-INV-14: `ior` excluded from path-replay (FD only) despite analytic partial exists
- PTWG-BDPT-01: BDPT mutually exclusive with volumetric SSS at compile time
- PTWG-BDPT-02: BDPT light subpath serial (one workgroup) — perf not correctness
- PTWG-BDPT-03: BDPT off on lite tier silently
- PTWG-CAUSTIC-01: `manifold-nee` / `photon-map` — SPPM progressive landed but radiometric promotion open
- PTWG-CAUSTIC-02: Caustics off on lite
- PTWG-RESTIR-01: ReSTIR-PT opt-in; composite into beauty — implementation done; promotion A/B open
- PTWG-RESTIR-02: `emitterCastShadow` approximate on some specialty legs (ledger)
- PTWG-ENV-01: `procedural-sky` approximate (256×128 bake)
- PTWG-LITE-01: Entire parallel capability matrix (`index.ts` lite `supportDetails`) — transform/material/topology blocked
- PTWG-LITE-02: ~30 extra material fields unsupported on lite (`PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS`)
- PTWG-LITE-03: Vertex colors warned unsupported
- PTWG-LITE-04: Multi-directional: first only
- PTWG-LITE-05: Geometry mode merged BLAS not TLAS
- PTWG-SHADOW-01: Analytic primitives always occlude (no castShadow field on contract)
- PTWG-DENO-01: Realtime denoisers warn-degrade except oidn-final
- PTWG-SVGf-01: `svgf-real` unsupported on PT backends (regime mismatch — intentional)

---

## 2. Engine contract & optional methods

| ID | Gap | Evidence | Action |
|----|-----|----------|--------|
| E-01 | No `EngineFidelityProfile` type | Quality scattered across `qualityTier`, `restirPtReuse`, `tier`, `traceTier` | **IMP** core enum + backend mappers |
| E-02 | `capabilities.supportDetails` static at construction | Does not reflect active fidelity toggles | **IMP** effective capabilities |
| E-03 | `captureFrame('output')` JSDoc says walkaround rejects | `HybridEngine.captureFrame` handles output via `captureOutputFrame()` (`HybridEngine.ts:1879-1882`) | **DOC** fix `core/src/engine/index.ts` |
| E-04 | `incrementalPatchSupport` vs `mutations.*` drift possible | Ledger tests exist; walkaround topology is fallback-rebuild | **IMP** align or document |
| E-05 | `getRestirPtResultBuffer` on facade | Fixed H61 — verify still on `idempotentDispose` proxy | **VERIFY** |
| E-06 | `createInverseSession` only pt-webgpu | By design | **ACC** or **IMP** pt-webgl2 stub that throws directed error |
| E-07 | Progressive handoff: walkaround has seed source, pt-webgpu has sink only | Asymmetric by design | **IMP** pt-webgl2 `seedAccumulator` OR document one-way only |
| E-08 | `giStatePersistence` walkaround-only | By design | **ACC** |
| E-09 | `updateLighting` walkaround-only | PT backends lack sun/scrub API | **IMP** or **ACC** |
| E-10 | `FrameInput.viewport` ignored by walkaround | `frameInputPromises.honorsViewportPerFrame: false` | **DOC** + attachVitrum calls `setSize` |
| E-11 | `FrameInput.honorsPerFrameBounces` false on walkaround | `maxBounces` is DDGI gate not ray cap | **DOC** |
| E-12 | `presentationMode` differs per backend | swapchain-required vs offscreen | **ACC** — host must branch |
| E-13 | `EngineOptions.extensions` untyped bag | Backend-specific keys | **IMP** discriminated union per backend |
| E-14 | Animation: backends don't consume `AnimationClip` | Host must `sampleAnimationClip` + patch | **IMP** optional backend animation consumer OR **ACC** |
| E-15 | `pickPrimitive` CPU rest-pose on skinned | `pickPrimitiveCpu.ts:158` | **IMP** posed pick or warn |
| E-16 | `AnalyticPrimitive` no `castShadow` field | Analytic always occludes on pt-webgpu | **IMP** contract field |
| E-17 | `BackendTexture` brand helpers only | No validation of handle shape at runtime | **IMP** dev-mode asserts |
| E-18 | `detectGpu` memoized | `resetGpuDetectionCache` exists | **ACC** document |

---

## 3. Mutation matrix (complete)

| Backend | transform | positions | material | emitter | topology | add | remove | env | resize | lighting |
|---------|:---------:|:---------:|:--------:|:-------:|:--------:|:---:|:------:|:---:|:------:|:--------:|
| walkaround | native | native | native* | native | fallback-rebuild | fallback | fallback | approximate** | native | native |
| pt-webgl2 | fallback | fallback | native | native | fallback | fallback | fallback | native | native | unsupported |
| pt-webgpu full | native | native | native | native | native | fallback | fallback | native | unsupported | unsupported |
| pt-webgpu lite | unsupported | native | fallback | native | unsupported | fallback | fallback | native | unsupported | unsupported |

\* walkaround material fast path skips DDGI sync on material-only (`applySubsystems: false`)  
\** walkaround runtime HDRI opaque handle → intensity-only unless resolver

**Mutation gaps to close:**
- M-01: pt-webgl2 native transform/positions/topology fast paths (or downgrade `incrementalPatchSupport`)
- M-02: walkaround topology native claim vs fallback-rebuild reality
- M-03: walkaround add/remove without full scene rebuild (or downgrade `supportsAddRemovePrimitive` perf story)
- M-04: pt-webgpu `setSize()` API (viewport-only resize is surprising)
- M-05: walkaround `updateEnvironment` full HDRI rebuild for GPU-only handles
- M-06: All backends: `updatePrimitive` normals-only path (walkaround routes to topologyRebuild)
- M-07: Emitter mesh-area stale buffers on transform — pt-webgpu fixed H11; verify walkaround
- M-08: Lite tier throws on transform patch (`sceneMutationRouter.ts`)
- M-09: Material texture map patches on pt-webgl2 (full setScene fallback)
- M-10: Instanced mesh instance-count: verify walkaround vs pt-webgpu parity
- M-11: Bone/morph patches: walkaround closed 2026-06-16; verify PT parity under TLAS
- M-12: `partitionSceneBySupport` analytic fallback — verify all backends

---

## 4. Denoiser matrix

| Mode | walkaround | pt-webgl2 | pt-webgpu | Gap |
|------|:----------:|:---------:|:---------:|-----|
| none | native* | native | native | *WH: not in VALID_DENOISERS |
| atrous | native | unsupported | unsupported | PT: warn-degrade |
| atrous-variance | native (default) | unsupported | unsupported | |
| svgf-real | native | unsupported | unsupported | PT regime mismatch |
| bmfr | native | unsupported | unsupported | |
| oidn-final | native | native | native | needs host model + onnxruntime-web |
| neural | native** | unsupported | unsupported | **starter weights only |

**Denoiser gaps:**
- D-01: Expose `denoiser:'none'` on walkaround (`HybridEngineConfig.ts` validation)
- D-02: Ship or bless production neural weights (A10)
- D-03: Neural resize: re-init InferenceGraph on size change without raw-HDR fallback
- D-04: OIDN: document two-asset requirement in createEngine path (C1 done — verify)
- D-05: SVGF-real on PT backends — intentional unsupported; document in host picker
- D-06: Denoiser `state()` / `FrameStats.denoiserState` — implemented; verify all failure paths
- D-07: Shared one-shot WebGPU denoiser cleanup — closed H35; verify no regressions

---

## 5. Shadow flags

| Flag | walkaround | pt-webgl2 | pt-webgpu | Action |
|------|:----------:|:---------:|:---------:|--------|
| primitiveCastShadow | native | native | native | — |
| emitterCastShadow | native | approximate | approximate | **IMP** forward emissive-hit (gl2); BDPT/MNEE legs (wgpu) |
| receiveShadow | unsupported | unsupported | unsupported | **IMP** or permanent **ACC** |

---

## 6. Emitter & environment kinds

| Kind | WH | PT-GL2 | PT-WG full | PT-WG lite | Notes |
|------|:--:|:------:|:----------:|:----------:|-------|
| directional | native | native | native | native | |
| rect-area | native | native | native | native | |
| disc-area | native | native | native | unsupported | |
| point | native | native | native | native | |
| spot | native | native | native | native | |
| mesh-area | native | native | native | unsupported | |
| env: none | native | native | native | native | |
| env: hdri | native | native | native | native | |
| env: procedural-sky | approximate | approximate | approximate | approximate | finite bake resolution |

**Emitter gaps:**
- EM-01: Exact emissive texel PDF + forward-hit MIS parity (B4 tail)
- EM-02: Mesh-area `color×intensity` override material Le (H23 closed WH — verify PT)
- EM-03: Capped / reordered mesh-area emitter adjoint (inverse)
- EM-04: Light tree orientation cones — done B8; verify on all NEE paths
- EM-05: IES profiles removed from chain — verify no dead types remain
- EM-06: Sun angular diameter soft shadows in adjoint (partial 2026-06-16)
- EM-07: Finite emitter stratification in walkaround OIT (2026-06-16) — verify ReSTIR area lights match

---

## 7. Primitive kinds & analytic shapes

| Kind / shape | WH | PT-GL2 | PT-WG full | PT-WG lite |
|--------------|:--:|:------:|:----------:|:----------:|
| mesh | native | native | native | native |
| skinned-mesh | native | native | native | native |
| instanced-mesh | native | native | native | native |
| analytic | fallback-mesh | fallback-mesh | native | unsupported |
| sphere/box/capsule/cylinder/h-channel-came | fallback-mesh | fallback-mesh | native | unsupported |

**Gaps:**
- P-01: walkaround native analytic traversal (unlikely — keep fallback + route)
- P-02: pt-webgl2 native analytic (tessellation only today)
- P-03: COLOR_0 on pt-webgpu lite — unsupported with warning
- P-04: COLOR_1+ ignored on import — diagnostic only
- P-05: Point/line primitive proxy meshes — intentional; reject-degraded mode
- P-06: EXT_mesh_gpu_instancing + morph/skin — unsupported combo
- P-07: GPU skinning: inverse bind for non-identity bind meshes (deferred)

---

## 8. Inverse rendering (pt-webgpu) — full gap list

**Session creation throws:**
- I-01: `loss: 'ssim' | 'lpips'`
- I-02: `kind: 'texture'`
- I-03: Invalid param paths / kinds

**Path-replay downgrades to FD (diagnostics emitted):**
- I-04: `bounces > 1`
- I-05: `spectral: true` forward
- I-06: Field not in `ADJOINT_ELIGIBLE_FIELDS`
- I-07: `transmission > 0` on material
- I-08: `spectralAttenuation` / `dispersionAbbeNumber` on material
- I-09: Transport fields: ior, transmission, thickness, attenuationColor, attenuationDistance
- I-10: Visibility: opacity, alphaCutoff
- I-11: Normal: normalScale, bumpScale, clearcoatNormalScale
- I-12: envMapIntensity
- I-13: Maps: normalMap, bumpMap, alphaMap, displacementMap, clearcoatNormalMap, transmissionMap, thicknessMap, etc.
- I-14: Emissive with texture maps (partial replay for camera-direct only)
- I-15: Primitives: non-mesh kinds
- I-16: Materials: iridescence optimized paths need special casing
- I-17: Emitters: angularDiameter directional, non-direct-light emitters
- I-18: Scene lighting outside adjoint scope
- I-19: Receiver material domains outside scoped replay
- I-20: Soft-sun directional (partial — 2026-06-16)
- I-21: Capped/reordered mesh-area emitters
- I-22: Exact texel-PDF mesh emission
- I-23: Forward light-selection MIS parity
- I-24: Indirect transport
- I-25: Layered / volume / spectral material domains

**Path-replay implemented (keep, extend):**
- Scalar/RGB: baseColor, roughness, metallic, emissive, emissiveIntensity, specular*, clearcoat*, sheen*, iridescence*, anisotropy*, aoMapIntensity, lightMapIntensity
- Attenuation color/distance (FD; replay transport downgrade)
- iridescenceThicknessRange vec2 (+ map)
- Emitter color/intensity for point/spot/rect/disc/mesh-area (2026-06-16)
- Map locals: baseColorMap, roughnessMap, metallicMap, specular*Map, clearcoat*Map, sheen*Map, iridescence*Map, anisotropyMap, aoMap (scoped)

**Inverse architecture gaps:**
- I-26: No texture-pixel optimization (Phase 2)
- I-27: No perceptual losses
- I-28: No multi-bounce adjoint
- I-29: No spectral adjoint
- I-30: `computeAdjointGradient` samples count wired — verify all code paths
- I-31: No pt-webgl2 / walkaround inverse sessions
- I-32: No inverse for render while paused — fixed; verify
- I-33: Adjoint clearcoatNormalMap FD only (comment in inverseSession.ts header)
- I-34: Extension-lobe contribution/PDF gradients incomplete (GATE-04 tail)

---

## 9. glTF adapter — gap list

**Import skips / diagnostics (honest but gaps for "any glTF"):**
- G-01: Draco — host hook required (`compression.ts`)
- G-02: meshopt — host hook or fallback stub buffer
- G-03: Cameras ignored
- G-04: COLOR_1+ vertex sets ignored
- G-05: Unknown primitive modes skipped
- G-06: Skin attrs without `node.skin` → static mesh
- G-07: Incomplete skin attrs → unsupported diagnostic
- G-08: Instancing + morph/skin incompatible
- G-09: Archived KHR_materials_pbrSpecularGlossiness → approximate MR
- G-10: Unknown KHR extensions stored raw + warn
- G-11: texCoord conflicts on texture fields ignored
- G-12: POINTS/LINES → triangle proxy meshes
- G-13: TRIANGLE_STRIP/FAN triangulated (OK)
- G-14: Secondary UV sets partially ignored
- G-15: Morph tangent deltas approximate
- G-16: Sampler policy (repeat/mipmap) approximate on backends
- G-17: NPOT repeat wrap approximate
- G-18: Opaque texture handles without decode → reject-degraded unless host asserts ready
- G-19: Inactive variant materials decoded but synthetic scene rows
- G-20: No bundled Draco/meshopt decoders

**Planner / engine bridge:**
- G-21: `rankGltfBackends` — done; verify emissiveMap.texelPdf row
- G-22: `loadGltfForEngine` runtime profile reconciliation — done 2026-06-16
- G-23: `reject-degraded` vs `reject-unsupported` — implemented
- G-24: `attachVitrum` / `VitrumCanvas` glTF playback opt-in — done; verify delta time
- G-25: `createProgressiveEngine` coordinator seed — done
- G-26: Texture decode `target: webgpu` browser path — closed 2026-06-16
- G-27: `featureReport` source paths incomplete for every nested feature
- G-28: Khronos sample model sweep incomplete
- G-29: No bundled test assets beyond fixtures
- G-30: `sceneController` cross-fade API missing (optional)
- G-31: Topology-changing animation diagnostics (optional)

---

## 10. Shared packages

### shared-bvh
- SB-01: Scene pack skips non-mesh primitives with warn
- SB-02: CPU pick rest-pose skinned (`pickPrimitiveCpu.ts`)
- SB-03: CPU pick analytic without fallback → sphere proxy only
- SB-04: `expandIndicesToStride4` exported but production uses dedicated packers
- SB-05: SBVH not implemented (B7 planar SAH done)
- SB-06: `validateBvhEncoding` exported — internal only?
- SB-07: DDGI-specific helpers moved out — verify no stale imports
- SB-08: `worldSpaceMerge` fingerprint exact vs sampled — fixed; verify
- SB-09: uv1 zero desync — check D12 closure

### shared-samplers
- SS-01: `defineUbo` missing vec2u, vec4u, mat3x3f, array types (`uboCodegen.ts`)
- SS-02: ReGIR CPU oracle only — GPU in walkaround
- SS-03: BDPT MIS partials removed — verify no dead exports

### shared-denoisers
- SD-01: OIDN requires onnxruntime-web peer
- SD-02: `runSVGFRealWebGPU` standalone — engines compose WGSL directly
- SD-03: atrous-variance without Welford warns (`atrousVarianceWebGPU.ts`)

### walkaround-rc
- RC-01: `envIntensity` not in dispatch opts (`cascadeDispatch.ts:825`)
- RC-02: Cornell-tuned default dims — host must override
- RC-03: Separate BVH from ReSTIR (documented)
- RC-04: Glass skip in direct light — done; verify merged mode trans4

### scene-lighting
- SL-01: Not imported by backends — intentional; document host flow

### stained-glass-extensions
- SG-01: `packCameUBO` — no backend consumer (`cameUniformUploader.ts`)
- SG-02: Excess segments/nodes discarded with warn

### dev
- DV-01: Overlays degrade when `engine.debug` missing — intentional

---

## 11. Opt-in subsystems (walkaround)

| Subsystem | Status | Gap |
|-----------|--------|-----|
| GRIS (`restirPtReuse`) | Implemented, default OFF | A-08: evidence for default-on; perf cost documented |
| PPG | Guides with MIS | A-02: localization A/B; cadence tuning |
| NRC | Biased cache | A-06: quality A/B; default-on decision; distillation target |
| RC | Opt-in | A-7 done; envIntensity; non-Cornell dims |
| Neural denoiser | Opt-in | A-10: production weights |
| ReSTIR-PT (pt-webgpu) | Opt-in | A-1: variance A/B |
| Checkerboard | Quality tier | Measured 1.46×; motion threshold |
| ReGIR | Opt-in | Closed 2026-06-16 mapped emitter target |
| Transparent OIT | Implemented | Alpha in GI still open |
| Stained glass shade | Opt-in flags | Document |

---

## 12. Frontier features (road-to-100 Bucket A)

| ID | Item | Code status | Remaining code work |
|----|------|-------------|---------------------|
| A1 | ReSTIR-PT composite | Implemented | Promotion + default policy |
| A2 | PPG runaway/splits | Implemented | GPU localization evidence |
| A3 | Spectral reflectance | Implemented both PT | Emitter/env spectral SPD |
| A4 | Hachisuka SPPM | Implemented progressive | Radiometric oracle |
| A5 | BDPT coherence | Both backends | Default-on BDPT? |
| A6 | NRC | Biased, opt-in | Default + distillation |
| A7 | RC | Finished | envIntensity param |
| A8 | GRIS default | Biased default | Evidence-based flip |
| A9 | BDPT parallel build | Serial subpath | Parallelize |
| A10 | Neural weights | Pipeline exists | Production checkpoint |

---

## 13. Fidelity ceilings (road-to-100 Bucket B) — code still open

| ID | Item | Remaining |
|----|------|-----------|
| B1 | Glass + rich GI | Receiver pHat wave landed; temporal previous-domain fallback quantification |
| B2 | DDGI glossy probe bounce | Specular complement approximate |
| B3 | Env pillar | Native; procedural approximate |
| B4 | Emissive texel PDF | Micro-triangle split landed; full alias/PDF MIS tail |
| B5 | Beer DDGI | Done |
| B6 | GTAO view axis | Done |
| B7 | Planar SAH | Done; SBVH optional |
| B8 | Light tree cones | Done |
| B9 | GGX multiscatter | Done |
| B10 | Physical refraction transmittance | Done |
| B11 | Disc-area native | Done |
| B12 | Lite texture packing | Done |

---

## 14. Hygiene (Bucket D) — remaining code items

| ID | Item | Status |
|----|------|--------|
| D6 | Bind-group churn | Closed W1R6 cache — verify no regressions |
| D10 | Storage buffer limit constant | Closed — verify 33 buffers |
| D11 | pt-webgl2 undeclared uniforms | Verify `u_volumeDensity` / `materialLodDepth` |
| D12 | uv1 desync in worldSpaceMerge | Verify closed |
| D3 tail | Per-backend BSDF consumption for new MaterialSpec fields | displacement, lightMap in GI, etc. |
| H43 | CHANGELOG Unreleased accuracy | Doc/code — out of scope unless user wants |

---

## 15. Test & behavioral gate code gaps

| ID | Gap | Action |
|----|-----|--------|
| T-01 | Behavioral gate: no pt-webgl2 configs | **IMP** add `ptgl/*` configs |
| T-02 | Behavioral gate: no OIDN / neural / svgf-real walkthrough configs | Add configs or document opt-in |
| T-03 | ~80 tests substring-match WGSL without compile | Extend shader-gate coverage |
| T-04 | cpuTracer mirror not independent of live WGSL strings | Tripwire exists — maintain |
| T-05 | Welford oracle exists; extend to more passes | |
| T-06 | FrameParamsSlot ↔ WGSL struct cross-check | |
| T-07 | E2E mutation matrix GPU observable buffers | GATE-03 tail |
| T-08 | Extension-lobe furnace reference renders | GATE-05 |

---

## 16. Examples & tools (code-facing)

| ID | Gap | Action |
|----|-----|--------|
| X-01 | Examples exist (H57 closed) — add more scenarios | glass-gi, spectral, inverse, fidelity tiers |
| X-02 | `tools/gltf-material-sweep` — exists | extend Khronos models |
| X-03 | Benchmark capture adapter fail-closed | OK |
| X-04 | No in-repo Draco/meshopt decoder | **IMP** optional WASM deps |
| X-05 | Reference render baselines stale vs render-changing landings | Refresh baselines (code artifacts) |

---

## 17. Meta-architecture: Fidelity Profile (recommended unifier)

**Not implemented.** This is the recommended way to bundle §1–§16 performance/quality toggles:

```ts
type EngineFidelityProfile = 'native' | 'balanced' | 'performance';

type WalkaroundFidelityOptions = {
  profile: EngineFidelityProfile;
  giReuse: 'biased' | 'unbiased';           // restirPtReuse
  materialStorage: 'quantized' | 'full';
  emissiveImportance: 'averaged' | 'texel' | 'alias';
  alphaInGi: boolean;
  ddgiIndirect: boolean;
  checkerboard: boolean;
};

type PtWebgpuFidelityOptions = {
  profile: EngineFidelityProfile;
  traceTier: 'full' | 'lite';               // construction-time
  maxBounces: number;
  caustics: boolean;
  bdpt: boolean;
  restirPt: boolean;
};
```

**Work items:**
- FP-01: Add types to `@vitrum/core`
- FP-02: `mapFidelityProfile()` in `createEngine`
- FP-03: Recompute `capabilities.supportDetails` from active profile
- FP-04: `rankGltfBackends(report, profile)`
- FP-05: Document pipeline rebuild requirements when profile changes
- FP-06: UBO bits for runtime toggles (materialStorage, emissiveImportance, alphaInGi)

---

## 18. Suggested execution phases (code only)

### Phase 0 — Contract truth (1–2 weeks)
- E-03, E-01/02, M-01 alignment, D-01, T-01 pt-webgl2 gate
- Ledger ↔ runtime tests for every mutation row

### Phase 1 — Fidelity profile foundation (1–2 weeks)
- FP-01 through FP-06
- No WGSL changes yet — map existing knobs

### Phase 2 — pt-webgpu full-tier native promotion (4–8 weeks)
- PTWG material approximate rows (14) + inverse expansions
- BDPT/SPPM/ReSTIR-PT specialty path parity (PTWG-MAT-01)
- `setSize()` API

### Phase 3 — pt-webgl2 mutation + gate parity (2–4 weeks)
- PTGL-MUT-01 fast paths
- PTGL-GATE-01 behavioral configs
- SSS model upgrade

### Phase 4 — Walkaround native tier (8–16 weeks)
- Quantization toggles
- Emissive texel PDF + GI alpha transport
- GRIS evidence → maybe default policy change
- Promote approximate material rows with A/B oracles

### Phase 5 — glTF completeness (4–8 weeks)
- G-01/G-02 bundled decoders or official optional deps
- G-04–G-18 edge cases
- Feature report completeness

### Phase 6 — Inverse Phase 2–3 (8+ weeks)
- Texture optimization
- ssimplpips losses
- Multi-bounce / spectral adjoint research scope

### Phase 7 — Permanent unsupported decisions
- displacement: implement or forever ACC
- receiveShadow: implement or forever ACC
- walkaround spectral/volume: forever RT

---

## 19. Checklist index (quick lookup)

Use grep on this file for IDs: `W-GI-`, `PTWG-`, `PTGL-`, `I-`, `G-`, `E-`, `M-`, `FP-`, `T-`, `SB-`, `INV-`, `A-`, `B-`, `D-`, `H-` (historical).

**When an item closes:** change Action to `DONE` + commit hash; do not delete rows (audit trail).

---

## 20. Walkaround texture atlas — remaining code work (Phase 3D/3E)

Atlas architecture is landed; these are the **remaining implementation** items (not validation):

| ID | Gap | File(s) | Action |
|----|-----|---------|--------|
| WA-ATLAS-01 | Map/atlas-metadata edits → full rebuild (cost) | `HybridEnginePrimitiveUpdates.ts` | **IMP** narrow atlas slice refresh |
| WA-ATLAS-02 | Morph target UV deformation | atlas rebuild policy | **IMP** UV-aware refresh or warn |
| WA-ATLAS-03 | `bumpMap` / `bumpScale` approximate | shade/DI/GI suffix | **PROM** to native |
| WA-ATLAS-04 | Emissive texel PDF not alias-table | emitterList, DDGI NEE | **IMP** energy-weighted alias |
| WA-ATLAS-05 | Analytic/extra emitters keep average Le | `emitterHelpers.ts` | **IMP** mapped payload |
| WA-ATLAS-06 | Non-NEE GI/DDGI emission approximate | probe rays, risGi | **IMP** or **ACC** |
| WA-ATLAS-07 | Light map non-camera paths scalar only | DDGI, GI | **IMP** or **ACC** |
| WA-ATLAS-08 | Extension lobes all `approximate` | shade, restirDiMaterial, restirGiMaterial | **PROM** after furnace |
| WA-ATLAS-09 | Tangent buffer approximate for GI PDF | `bvhTangentTexture.ts` | **PROM** |
| WA-ATLAS-10 | Glass Beer/transmission/thickness tint approximate | shade, risGi | **TOG** per-hit vs baked |

## 21. Transparent transport promotion (Phase 3C tail)

| ID | Gap | Action |
|----|-----|--------|
| WA-ALPHA-01 | Fractional blend not ReSTIR reservoir vertex | **IMP** stochastic transparent GI OR document permanent OIT split |
| WA-ALPHA-02 | ReSTIR direct light approximate for blend | **IMP** |
| WA-ALPHA-03 | GI participation approximate for blend | **IMP** |
| WA-ALPHA-04 | First-hit light-map/emissive in OIT only | **IMP** probe/GI transport OR **ACC** |
| WA-ALPHA-05 | `alpha-blend-approximation` warning at setScene | exists — remove when closed |

## 22. pt-webgl2 geometry fast-path port (from pt-webgpu)

| ID | Gap | Source to port | Action |
|----|-----|----------------|--------|
| PTGL-GEO-01 | Transform refit | `sceneMutationRouter.ts` TLAS refit | **IMP** WebGL2 BVH texture update |
| PTGL-GEO-02 | Positions refit | same | **IMP** |
| PTGL-GEO-03 | Topology splice | same | **IMP** |
| PTGL-GEO-04 | add/remove primitive | BLAS splice | **IMP** |
| PTGL-GEO-05 | Material texture map fast path | `mutateSceneTextures.ts` | **IMP** repack single material slot maps |
| PTGL-GEO-06 | Animation via glTF controller | full repack today | **IMP** after GEO-01..04 |
| PTGL-SPEC-01 | `iorCauchyA/B/C` dispersion | pt-webgpu Cauchy path | **IMP** optional |
| PTGL-SPEC-02 | Hero-wavelength promotion | experimental tier | **PROM** |

## 23. pt-webgpu lite tier — complete unsupported list

When `traceTier:'lite'`, these are **hard unsupported** (not approximate):

- Primitives: analytic (all shapes), instanced-mesh
- Emitters: disc-area, mesh-area
- Mutations: transform, topology, add/remove
- Materials: ~30 fields in `PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS` plus vertex colors
- Features: TLAS, BDPT, caustics, ReSTIR-PT, aux buffers, multi-directional (first only)
- Geometry: merged BLAS bake only

**Code gaps:**
- LITE-01: No runtime upgrade lite→full without engine recreation
- LITE-02: `rankGltfBackends` must stay in sync with `supportDetails` (tests exist)
- LITE-03: Area-lite stub in `connectLite.wgsl.ts` — deliberate zero (document)

## 24. Specialty integrator parity checklist (PTWG-MAT-01 tail)

Every path must consume the same material payload as megakernel for extension lobes:

| Path | Status | Remaining |
|------|--------|-----------|
| Megakernel eye | ✅ | inverse adjoint gaps |
| connect / connectLite | ✅ | area-lite stub |
| BDPT connection | ✅ | radiometric oracle |
| BDPT light subpath | ✅ | parallel build |
| SPPM gather | ✅ | progressive radiometric oracle |
| MNEE caustic | ✅ | — |
| ReSTIR-PT producer/resolve | ✅ | default-on policy |
| Adjoint path replay | ◻ | §8 inverse list |
| pt-webgl2 GLSL | ◻ | extension lobes partial vs wgpu |

## 25. Uncommitted / in-flight work (verify at session start)

From conversation context — may exist as unstaged changes:
- Mesh-area emissive texel-cell clipping (`meshAreaLights.ts`, `emitterPacking.ts`, `emitterList.ts`)
- ReGIR mapped emitter target (road-to-100 says closed 2026-06-16)

**Action:** `git status` + reconcile this plan before autonomous runs.

## 26. ID cross-reference to road-to-100 buckets

| This plan | road-to-100 |
|-----------|-------------|
| §12 A1–A10 | Bucket A |
| §13 B1–B12 | Bucket B |
| §10 C1–C3 | Bucket C (C2 = A10) |
| §14 D* | Bucket D |
| §9 G* | Phase 1 glTF |
| §8 I* | Phase 2 adjoint |
| §5–§7, §20–§21 | Phase 3 walkaround |
| §17 FP* | Not in road-to-100 — new unifier |

---

*End of master plan. ~350 tracked items. Re-audit monthly against `promiseLedger.ts` and `detect_changes()`.*
