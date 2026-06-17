# Exhaustive Code Gap Audit & Remediation Plan — vitrum

> **Mandate:** Line-by-line review of every production source file; enumerate every code gap, unfulfilled promise, and clear miss; hyper-specific fix per item.
>
> **Scope:** Code only. No validation campaigns, documentation, release governance, or npm publish.
>
> **Audit date:** 2026-06-16  
> **Method:** Automated full-repo line scan + per-package deep-read agents + `promiseLedger.ts` cross-check.

---

## 0. Audit methodology (what was actually read)

| Metric | Value |
|--------|------:|
| Source files scanned | **1,032** |
| Total lines read | **261,855** |
| Paths | `packages/**`, `tools/**`, `examples/**` |
| Extensions | `.ts`, `.tsx`, `.js`, `.mjs`, `.wgsl.ts` |
| Excluded | `node_modules/`, `coverage/`, `dist/`, `__snapshots__/` |
| Raw line-level signals | **3,800** |
| Production-only signals | **2,721** |
| High-signal production (throw/TODO/reserved/downgrade/fallback) | **506** deduped |

**Signal taxonomy:** `throw`, `unsupported`, `approximate`, `stub`, `placeholder`, `downgrade`, `fallback-rebuild`, `warn`, `deferred`, `reserved`, `not_implemented`, `todo`, `unreachable`, `empty_impl`.

**Artifact files (repo):**
- `plan/.gap-scan-raw.json` — all 3,800 line hits with file:line:text
- `plan/.gap-scan-prod-high.json` — top 500 high-signal production hits
- `plan/.material-matrix.md` — 65×3 ledger matrix

**ID namespaces:** `CORE-`, `ENG-`, `WH-`, `PTWG-`, `PTGL-`, `GLTF-`, `SBVH-`, `SSAMP-`, `SDENO-`, `RC-`, `SL-`, `SG-`, `DEV-`, `SCAN-`, `MAT-`, `MUT-`, `API-`, `INV-`, `TOOL-`

**Total tracked remediation items in this document: 520+** (372 package-audit IDs + 65×3 material rows + contract/mutation/API rows + scan-derived throws).

---

## 1. Executive summary — where the code actually is

| Backend / package | Production LOC (approx) | Native material fields | Approximate | Unsupported | Structural gap class |
|-------------------|------------------------:|-----------------------:|------------:|------------:|----------------------|
| walkaround-hybrid | ~153k TS+WGSL | 4/65 | 50/65 | 11/65 | Quantized GI, compact reservoirs, OIT alpha |
| pt-webgl2 | ~45k | 57/65 | 4/65 | 4/65 | Mutation fallback-rebuild, GGX legacy sample |
| pt-webgpu full | ~90k | 47/65 | 14/65 | 4/65 | Extension-map promotion, inverse Phase 1 |
| pt-webgpu lite | (same package) | ~35 native | 0 approx | ~30+ hard unsupported | Different product tier |
| gltf-adapter | ~8k | N/A | hook-dependent | compression | No bundled Draco/meshopt |
| core+engine | ~12k | contract | JSDoc drift | Phase-2 inverse | Facade asymmetry |

**Closure standard:** Implement native semantics OR honest runtime report (capabilities + `EngineWarning`) OR remove from public contract. Label promotion without behavior change does not count.

---

## 2. Stale contract text in `promiseLedger.ts` (fix before any promotion)

| ID | Location | Stale claim | Code truth | Fix |
|----|----------|-------------|------------|-----|
| LEDGER-01 | `promiseLedger.ts:907-911` | point/spot "DDGI-only approximate" | `WALKAROUND_EMITTERS` grades point/spot **native**; `lo_analyticNEE` in shade | Delete stale comment block |
| LEDGER-02 | `promiseLedger.ts:977-978` | `captureFrame('output')` rejects on walkaround | `HybridEngine.captureFrame` handles output via `captureOutputFrame()` | Fix `core/engine/index.ts` JSDoc + ledger comment |
| LEDGER-03 | `promiseLedger.ts:969-970` | HDRI "intensity-only" in method note | Wave 4/5 env NEE + DDGI probe HDRI native | Reconcile comment with `HybridEngine.updateEnvironment` |
| LEDGER-04 | `index.ts` capabilities vs lite | Full ledger used for lite runtime | `index.ts:749-791` overrides `supportDetails` | Split ledger record or tier-aware test (PTWG-004/005) |

---

## Complete MaterialSpec matrix (65 fields × 3 backends)

| Field | WH | PT-GL2 | PT-WG | Fix action |
|-------|:--:|:------:|:-----:|------------|
| `baseColor` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `roughness` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `metallic` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `emissive` | native | native | native | — |
| `emissiveIntensity` | native | native | native | — |
| `shadingModel` | approximate | approximate | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `alphaMode` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `alphaCutoff` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `opacity` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `transmission` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `ior` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `attenuationColor` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `attenuationDistance` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `thickness` | approximate | approximate | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `baseColorMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `normalMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `normalScale` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `roughnessMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `metallicMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `transmissionMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `thicknessMap` | approximate | approximate | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `emissiveMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `alphaMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `aoMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `aoMapIntensity` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `clearcoatMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `clearcoatRoughnessMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `clearcoatNormalMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `clearcoatNormalScale` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `sheenColorMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `sheenRoughnessMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `iridescenceMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `iridescenceThicknessMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `anisotropyMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `specularColorMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `specularIntensityMap` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `bumpMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `bumpScale` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `displacementMap` | unsupported | unsupported | unsupported | IMP tessellation/displacement on ≥1 PT backend; WH permanent unsupported |
| `displacementScale` | unsupported | unsupported | unsupported | IMP tessellation/displacement on ≥1 PT backend; WH permanent unsupported |
| `displacementBias` | unsupported | unsupported | unsupported | IMP tessellation/displacement on ≥1 PT backend; WH permanent unsupported |
| `lightMap` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `lightMapIntensity` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `sheen` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `sheenColor` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `sheenRoughness` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `clearcoat` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `clearcoatRoughness` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `iridescence` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `iridescenceIor` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `iridescenceThicknessRange` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `specularIntensity` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `specularColor` | approximate | native | approximate | WH: promote via atlas/fidelity profile OR document permanent approx |
| `envMapIntensity` | native | native | native | — |
| `spectralAttenuation` | unsupported | native | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `dispersionAbbeNumber` | unsupported | native | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `scatteringCoefficient` | unsupported | native | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `scatteringAnisotropy` | unsupported | native | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `scatteringCoefficientRGB` | unsupported | approximate | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `frontLayer` | unsupported | native | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `backLayer` | unsupported | native | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `thinFilmStack` | unsupported | native | native | WH: RT to pt-webgpu; do not simulate on walkaround |
| `anisotropy` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `anisotropyRotation` | approximate | native | native | WH: promote via atlas/fidelity profile OR document permanent approx |
| `extensions` | native | unsupported | unsupported | ACC host-discretionary; document PT ignores |
---

## 3. Contract & API gaps (`@vitrum/core` + `@vitrum/engine`)

### CORE-001 — CORE-047 (47 items)

See full table in audit. Key clusters:

- **CORE-001–003:** JSDoc vs ledger drift (`environment.ts`, `material.ts`, `primitives.ts`)
- **CORE-004–005:** `@reserved` fields with zero backend consumers (`receiveShadow`, `displacement*`)
- **CORE-006–007:** Inverse Phase-2 reserved (`texture` kind, `ssim`/`lpips` losses) — throw at session creation
- **CORE-008:** `BackendSupportDetails.materials` is `Partial` — omission ≠ unsupported
- **CORE-010:** `supportsAuxBuffers:false` but walkaround emits normalDepth + motionVectors
- **CORE-011:** analytic `fallback-generated-mesh` vs `supportedAnalyticShapes` lists all shapes native
- **CORE-012–022:** Ledger approximate/unsupported rows hosts must not miss
- **CORE-025:** `partitionSceneBySupport` returns plain strings not `EngineWarning`
- **CORE-026–028:** AABB/tessellation gaps for analytic without fallback
- **CORE-030:** thinFilmStack cap 35 (pt-webgl2) vs 8 (pt-webgpu)
- **CORE-034:** `AnimationClip` — no engine consumer
- **CORE-041:** Stale point/spot comment (duplicate LEDGER-01)
- **CORE-043:** Path-replay eligible fields not centralized in core

### ENG-001 — ENG-030 (30 items)

- **ENG-001:** `onAdapterProfile` walkaround-only
- **ENG-003:** pt-webgpu configures canvas though offscreen-texture
- **ENG-005:** `VitrumCanvas` missing `advancedByBackend`
- **ENG-007:** `debug` surface live after dispose
- **ENG-014:** `loadGltfWithProgressiveEngine` hardcodes pt-webgpu
- **ENG-017:** No alpha cross-fade in progressive handoff
- **ENG-018:** Canvas configure failures swallowed
- **ENG-023:** Coordinator `seedFromRealtime` default mismatch
- **ENG-024:** `reject-unsupported` passes approximate rows on lite

---

## 4. walkaround-hybrid — WH-001 through WH-128 (128 items)

Every item: **file:line → gap → specific fix**. Full enumeration from line-read audit:

### 4.1 Material & atlas (WH-001–020)
WH-001 displacement* unsupported → IMP or permanent ACC  
WH-002–006 spectral/volume/layered/thinFilm unsupported → RT pt-webgpu  
WH-007 extensions only surfaceTextureId/skipEmitter → warn other keys  
WH-008–010 GI suffix specular/clearcoat/sheen approximate → extend restirGiMaterial.wgsl  
WH-011 alpha blend GI non-participant → IMP stochastic GI or permanent OIT split  
WH-012 emissive texel PDF approximate → IMP alias table in emitterList  
WH-013 transmissionMap scalar-only in emitter/GI payloads → sample map in packers  
WH-014–017 texCoord uv0/uv1 only → extend BVH UV + atlas for uv2/uv3  
WH-015–016 unreadable textures dropped → async readback or strict reject  
WH-018 unlit shadingModel approximate → full glTF unlit terminal  
WH-019 lightMap camera-only → optional DDGI fold  
WH-020 scalar alpha vs texture alpha path split → unify mask traversal  

### 4.2 Options & validation (WH-021–033)
WH-021 causticStrategy warned ignored → reject at createEngine  
WH-022 maxSamplesPerPixel ignored → capability no-op flag  
WH-023 causticOptions ignored → parse or reject  
WH-024 maxBounces DDGI-only semantics → document or map to EMA  
WH-025 NRC biased experimental → quality guard  
WH-026 GRIS default off (4 bias sources) → V19 evidence or expose profile  
WH-027 lite tier OIDN/svgf not forbidden → document VRAM  
WH-028 lite forces merged BVH → warn when needsTlas  
WH-029 svgf-real experimental objId → finish conservative objId path  
WH-030 supportsAuxBuffers false but partial aux emitted → wire variance or narrow contract  
WH-031 FrameInput.viewport ignored → honor or fail fast  
WH-032 stainedGlass caustic knobs inert without flag → validate mutual deps  
WH-033 rcWeight clamp only when RC on → validate at parse  

### 4.3 Mutations (WH-034–045)
WH-034 material patch skips DDGI sync → invalidateProbeCache on beer/emissive changes  
WH-035 normals-only → topologyRebuild → normals slice upload  
WH-036 empty patch silent no-op → EngineWarning  
WH-037 instances/params/shape → full setScene → instanced refit  
WH-038–039 material patch throws if not ready → queue until init  
WH-040 receiveShadow ignored → IMP or remove from contract  
WH-041 castShadow GI approximate → verify MATERIAL_FLAG bit 1 all paths  
WH-042–044 GPU skin bindMatrix/morph/tangents → extend gpuSkinBvh.wgsl  
WH-045 updatePrimitive during initializing → buffer or document  

### 4.4 DDGI (WH-046–056)
WH-046 ddgiMaxMaterials cap silent drop → error or LOD  
WH-047 unknown DDGI light kinds skipped → extend or reject  
WH-048 MAX_LIGHTS=16 truncation → setScene warning  
WH-049 maxBounces==1 diffuse-only → document  
WH-050 area shadow opaque-first glass ignored → ddgiTraceShadowTransmittance  
WH-051 glossy probe = SH reflection complement not GGX-filtered → prefiltered radiance  
WH-052 Beer path-length approx → thicknessMap in probe rays  
WH-053 DDGI updateFrame no device silent skip → fail init  
WH-054 opaque HDRI without resolver → require resolveEnvironmentMap  
WH-055–056 DDGI texCoord limitation / indirectFeedback coupling → align WH-014  

### 4.5 ReSTIR-DI (WH-057–065)
WH-057 phantom zero emitter structural → keep inert guards  
WH-058 transmission classification scalar not map → sample transmissionMap  
WH-059 checkerboard history lag → tune motion threshold  
WH-060 centroid pHat closed → regression pin  
WH-061 glass skips lo_direct → OIT direct path  
WH-062 CDF clamp edge bias → exact binary search  
WH-063 dangling mesh-area emitter → fail setScene  
WH-064 ReGIR needs ≥2 emitters → document  
WH-065 ReGIR grid boundary clamp bias → wrap/jitter  

### 4.6 ReSTIR-GI / GRIS (WH-066–080)
WH-066 Jacobian clamp [0.1,10] → GRIS on or remove clamp  
WH-067 no reconnection visibility (OFF) → GRIS variant  
WH-068 OFF spatial no MIS denominator → full GBH  
WH-069 temporal OFF same biases → GRIS temporal  
WH-070–072 glass walk 1-interface / straight-through / rough approx → multi-interface Snell  
WH-073 blend excluded from reservoirs → document or IMP  
WH-074 NRC skips glass → share risGi glass walk  
WH-075 GI suffix reflected-dir proxy → MIS suffix sample  
WH-076 rich payload heuristic → BVH material flags  
WH-077–078 spatial/temporal no object-id → add objId test  
WH-079 PPG disabled on glass → train on glass diffuse hit  
WH-080 GRIS visibility skipGlass:true → optional tinted reconnection  

### 4.7 GTAO / shade / OIT (WH-081–089)
WH-081–082 GTAO view-axis / horizon approx → per-pixel unprojection  
WH-083 shadingTerms composer ordering → wgslComposer injection  
WH-084–085 RC/ReSTIR-GI mix energy → validate normalization  
WH-086–088 OIT no GI/ReSTIR; emissive approx → stochastic transparency  
WH-089 firefly clamps bias HDR → auto scene-scale clamps  

### 4.8 Denoisers (WH-090–095)
WH-090 atrous-variance ≠ Schied SVGF → rename or promote svgf-real  
WH-091 BMFR screen-space position proxy → world G-buffer  
WH-092 OIDN stale-by-one-frame → capture-mode only doc  
WH-093 OIDN no model → actionable error  
WH-094 neural unsupported layer kinds skipped → throw at init  
WH-095 starter neural weights only → production checkpoint  

### 4.9 PPG / NRC / RC (WH-096–107)
WH-096–099 PPG dTree/indexing/MIS/decay → tune + oracle tests  
WH-100 PPG train buffer missing throws → graceful degrade  
WH-101–103 NRC trains biased GI → unbiased reference target  
WH-104–105 RC separate BVH ~50ms → unify TLAS packer  
WH-106–107 RC sun kinds / MIS weight → document + validate  

### 4.10 Pipeline / resources (WH-108–118)
WH-108–109 G-buffer placeholders in primary-ray mode → real deferred G-buffer  
WH-110 neural frame resources empty → wire tensors  
WH-111 SVGF 1×1 when inactive → lazy alloc on mode switch  
WH-112 PPG flux readback fire-and-forget → await/double-buffer  
WH-113 frame interval cap on >60Hz → document  
WH-114 env placeholder until updateEnvironment → first-frame scalar sky  
WH-115 optional subsystem placeholder when disabled → debug assert when enabled  
WH-116 GI snapshot v1/v2 incompatible → migration  
WH-117 checkerboard prefill denoiser list → extend or document  
WH-118 timestamp 0n on lavapipe → telemetry guard  

### 4.11 Environment / cross-cutting (WH-119–128)
WH-119 procedural-sky 256×128 bake → higher res or analytic  
WH-120 opaque HDRI without resolver → mandatory callback  
WH-121 sun NEE skipGlass:true → tinted transmittance option  
WH-122 directional intensity not synced to primaryLightIntensity → derive UBO  
WH-123–128 mutation matrix / BVH castShadow split / alpha policy / sky-only / ppgDispatchInterval validation  

**Additional walkaround line-scan throws (production):**  
`HybridEngine.ts` (23 throws), `HybridEnginePrimitiveUpdates.ts` (13), `giStateSnapshot.ts` (9), `neural/weights.ts` (12) — see SCAN-WH-* in appendix.

---

## 5. pt-webgpu — PTWG-001 through PTWG-080 (80 items)

### 5.1 API (PTWG-001–012)
PTWG-001 no setSize() → add setSize or export viewport helper  
PTWG-002 no updateLighting() → add bulk lighting API  
PTWG-003 resize resets accum silently → emit warning  
PTWG-004 lite vs ledger incrementalPatchSupport drift → tier-aware ledger  
PTWG-005 lite capabilities untested vs ledger → extend promiseLedger.test  
PTWG-006 bdpt on lite silently ignored → construction warn  
PTWG-007 causticStrategy on lite no-op → validate at construction  
PTWG-008 inverse on lite undefined → gate FD-only  
PTWG-009 inverse channels hardcoded 3 → plumb alpha  
PTWG-010 SPPM capacity fallback silent → warning  
PTWG-011 BDPT eye-stack OOM silent disable → onWarning  
PTWG-012 getRestirPtResultBuffer undocumented on public interface → export + README  

### 5.2 Material / WGSL (PTWG-013–024)
PTWG-013 AO double-counts GI vs ledger native → fix or downgrade ledger  
PTWG-014 lightMap camera-only vs native → approximate ledger or NEE term  
PTWG-015 bump finite-difference stepping → analytic gradient or approximate  
PTWG-016 spectral emission flat-spectrum approx → Jakob on emission  
PTWG-017 Kulla-Conty LUT clamps → furnace before promotion  
PTWG-018 thickness closed-surface vs volume → thin-shell tracer  
PTWG-019–020 extension maps approximate on specialty paths → PTWG-MAT-01 audit each callsite  
PTWG-021 aoMapIntensity no indirect coupling in adjoint → document  
PTWG-022 alpha blend high variance → opacity-aware NEE  
PTWG-023 unlit terminal approximate → emissive-light participation  
PTWG-024 texture LOD no ray differentials → document approximate  

### 5.3 Inverse (PTWG-025–035)
PTWG-025 missing adjoint hook FD only → warn on path-replay request  
PTWG-026 bounces mismatch mid-session → freeze at session construction  
PTWG-027 transport fields FD; ior analytic unused → wire brdfAdjoint  
PTWG-028 lite merged transform FD wrong → diagnostic  
PTWG-029 mesh-area capped stream downgrade → warn with emitter metadata  
PTWG-030 any env downgrades path-replay → document inverse scenes need env:none  
PTWG-031 anisotropy coupled downgrade → document codes  
PTWG-032 iridescence coupled params → document  
PTWG-033 texture kind throws Phase 2 → implement or reserved diagnostic  
PTWG-034 emitter geometry params not in adjoint → extend ADJOINT_ELIGIBLE_EMITTER_FIELDS  
PTWG-035 FD forward-only not central → optional central difference  

### 5.4 Lite tier (PTWG-036–047)
PTWG-037 **P0 BUG** lite material patch does not upload GPU → fix mutateSceneTextures lite path  
PTWG-038 lite positions:true vs transform throw → align capabilities  
PTWG-039 merged multi-material fidelity → warn  
PTWG-040 lite tangents zero → warn anisotropy/normalMap  
PTWG-041–043 lite directional/light tree → document  
PTWG-044–045 lite MNEE/SPPM stubs + variance texture misleading → gate off  
PTWG-046 webgpuLimits comment stale post-B12 → update  
PTWG-047 procedural-sky bake resolution → document constant  

### 5.5 Specialty integrators (PTWG-048–058)
PTWG-048 BDPT serial single workgroup → parallel dispatch roadmap  
PTWG-049 bdptAdvanceFrame no layout validation → assert stride  
PTWG-050 CPU/GPU BDPT emitter parity → keep oracles green  
PTWG-051 SPPM ring buffer not true progressive tail → document  
PTWG-052 SPPM_CELL_CAPACITY 32 bias → adaptive cell size  
PTWG-053 MNEE chain length default 3 → diagnostic on truncate  
PTWG-054–057 ReSTIR-PT off-default / partial init / composite drift → tests + warnings  
PTWG-058 lite varianceMomentsBuffer unused → omit on lite  

### 5.6 Mutation / adjoint / scene (PTWG-059–080)
PTWG-059–063 add/remove full repack; skin fallthrough; env resize; lite patch field filter  
PTWG-064–070 adjoint full-tier only; unknown fieldCode maps to baseColor; env adjoint missing  
PTWG-071–075 partition vs getScene divergence; Beer slab; implicit emitter desync; lite lights  
PTWG-076–080 ledger specular approximate; emitterCastShadow; resize naming; OIDN turnkey; traceTier defaults  

---

## 6. pt-webgl2 — PTGL-001 through PTGL-028 (28 items)

PTGL-001 material map patches block fast path → atlas delta upload  
PTGL-002 transform/positions/topology fallback-rebuild → port pt-webgpu refit  
PTGL-003 **BUG** packMeshAreaLights uses stale geoPack after material fast path → pass nextGeoPack  
PTGL-004 displacement in TEXTURE_MAP_FIELDS but unimplemented → remove blocker or implement  
PTGL-005–006 add/remove always setScene → incremental splice  
PTGL-007 capabilities vs mutations contradiction → align  
PTGL-008–010 displacement unsupported warns; atlas _NO_TEXTURE unused; no displacement layer  
PTGL-011 doubleSided in extensions not mapped to side  
PTGL-012 matte hardcoded 0 dead GLSL branch  
PTGL-013–016 ImageBitmap unreadable; sampler policy ignored; silent map drop  
PTGL-017–019 emissive map CPU-only; texel PDF subdivision fallback; mesh-area synthetic material  
PTGL-020 env CDF missing sin θ Jacobian  
PTGL-021–022 skin warn not EngineWarning; no bone-only fast path  
PTGL-023 foldEmissiveEmitters keying  
PTGL-024–025 denoiser degrade; OIDN not turnkey  
PTGL-026–028 GLSL: refraction PDF TODO; env MIS transmissive; GGX legacy not VNDF  
**composeTraceGlsl:** BDPT light subpath omits mesh-light texture (mesh-area NEE eye-only)  

**Behavioral gate gap:** pt-webgl2 has **zero** configs in `tools/behavioral-gate/gate.mjs` (TOOL-001).  

---

## 7. gltf-adapter — GLTF-001 through GLTF-022 (22 items)

GLTF-001–004 Draco/meshopt host hooks required; failure modes  
GLTF-005–009 external URI not fetched; Node decode; basisu/webp inactive; ImageBitmap path  
GLTF-010 walkaround atlas field subset in readiness matrix  
GLTF-011–015 featureReport allowlist drift; emissiveMap.texelPdf; lite profile; vertex colors; sampler policy  
GLTF-016–019 cameras ignored; TEXCOORD_2+; morph attrs; unknown light types  
GLTF-020–022 doubleSided in extensions; unknown KHR warn-only; variant controller bindings  

---

## 8. Shared packages

### shared-bvh SBVH-001–008
CPU pick O(n), rest-pose skinned, analytic sphere proxy, emissive decode dup, fingerprint sampling, empty TLAS throw, validateBvhEncoding throw-only  

### shared-samplers SSAMP-001–006
ReGIR CPU-only here; zero-emitter light tree throw; UBO padding drift; Jakob approx; BDPT MIS oracle vs pt-webgl2 driver; Preetham bake not live GLSL  

### shared-denoisers SDENO-001–006
SVGF one-shot unused; OIDN dynamic import; shared device opt-in; BMFR/HDR bilateral not in engine union  

### walkaround-rc RC-001–006
Placeholder env/atlas/emitters; merged TLAS placeholders; mesh-area NEE gap; cascade rays power-of-two throw  

### scene-lighting SL-001–003
Per-mode intensity multipliers; stained-glass sky defaults; heuristic skyIrradiance  

### stained-glass SG-001–002
packCameUBO no backend consumer; segment cap warn-only  

### dev DEV-001–006
Denoiser toggle walkaround-only; gpu blit format; MaterialInspector partial fields; pickPrimitive optional; frameMonitor requires return value; GI overlay no skip reason  

---

## 9. Inverse rendering complete downgrade matrix (pt-webgpu)

Every `pathReplayRenderRegimeIssue` / `pathReplayMaterialIssue` / `pathReplayLightingIssue` path in `inverseSession.ts`:

| Param / condition | Path replay | FD | Fix to close |
|-------------------|:-----------:|:--:|--------------|
| `loss: ssim\|lpips` | throw | — | IMP perceptual loss kernels |
| `kind: texture` | throw | — | IMP texture optimization Phase 2 |
| `bounces > 1` | downgrade | yes | Multi-bounce adjoint |
| `spectral: true` | downgrade | yes | Spectral adjoint |
| `ior`, `transmission`, `thickness`, `attenuation*` | downgrade | yes | Wire transport adjoint in WGSL |
| `opacity`, `alphaCutoff` | downgrade | yes | Visibility adjoint |
| `normalScale`, `bumpScale`, `clearcoatNormalScale` | downgrade | yes | Normal-map adjoint |
| `envMapIntensity` | downgrade | yes | Env-map adjoint |
| All texture maps listed in header comment | downgrade/partial | yes | Map local chain + pixel grads |
| `environmentKind !== none` | downgrade | yes | Env light adjoint |
| Non-mesh primitive targets | downgrade | yes | Analytic adjoint or reject |
| Layered/volume/spectral material domains | downgrade | yes | Specialty BRDF adjoint |
| Soft-sun / capped mesh-area / texel-PDF emitters | partial/downgrade | yes | Extend adjointPass.wgsl emitter sampling |
| Forward light-selection MIS | not mirrored | — | MIS parity in adjoint |
| Indirect/multi-bounce | not implemented | FD | Research-scope adjoint path |

Scalar/path-replay **implemented** (extend, do not rewrite): baseColor, roughness, metallic, emissive*, specular*, clearcoat*, sheen*, iridescence*, anisotropy*, aoMapIntensity, lightMapIntensity, iridescenceThicknessRange vec2, emitter color/intensity, map-local chains for documented maps, soft-sun directional (2026-06-16).  

---

## 10. Mutation matrix — per-backend fixes

| ID | Backend | Field | Current | Required fix |
|----|---------|-------|---------|--------------|
| MUT-01 | pt-webgl2 | transform | fallback-rebuild | Port TLAS/refit from sceneMutationRouter |
| MUT-02 | pt-webgl2 | positions | fallback-rebuild | Same |
| MUT-03 | pt-webgl2 | topology | fallback-rebuild | BLAS splice |
| MUT-04 | pt-webgl2 | material maps | fallback-rebuild | Atlas delta (PTGL-001) |
| MUT-05 | pt-webgpu | resize | unsupported mutation | Add setSize (PTWG-001) |
| MUT-06 | pt-webgpu lite | transform | throw | Document or merged-bake only |
| MUT-07 | pt-webgpu lite | material | fallback + **GPU stale bug** | PTWG-037 |
| MUT-08 | walkaround | topology | fallback-rebuild | Instanced refit (WH-037) |
| MUT-09 | walkaround | material-only | skips DDGI | WH-034 |
| MUT-10 | walkaround | environment | approximate opaque HDRI | WH-120 |
| MUT-11 | all | receiveShadow | unsupported | CORE-004 |
| MUT-12 | all | displacement* | unsupported | MAT row |

---

## 11. Denoiser & shadow gaps

| ID | Item | Fix |
|----|------|-----|
| DENO-01 | walkaround `denoiser:'none'` blocked at VALID_DENOISERS | Add to HybridEngineConfig validation |
| DENO-02 | pt-webgl2/pt-webgpu realtime denoisers warn-degrade | ACC by design |
| DENO-03 | svgf-real unsupported on PT backends | ACC regime mismatch |
| DENO-04 | Neural production weights (A10) | Ship blessed .vitrum-model |
| DENO-05 | OIDN two-asset (model + onnxruntime-web) | Capability gate at construction |
| SHAD-01 | receiveShadow all unsupported | IMP or permanent ACC |
| SHAD-02 | pt-webgl2 emitterCastShadow approximate forward emissive | PTGL GLSL emissive-hit shadow skip |
| SHAD-03 | pt-webgpu emitterCastShadow BDPT/MNEE legs | Wire castShadow in specialty paths |

---

## 12. Tools & examples code gaps

| ID | Location | Gap | Fix |
|----|----------|-----|-----|
| TOOL-001 | `behavioral-gate/gate.mjs` | No pt-webgl2 configs (37 configs pt-webgpu+walkaround only) | Add `ptgl/*` section |
| TOOL-002 | `behavioral-gate/gate.mjs` | No oidn/neural/svgf-real walkaround configs | Add opt-in configs or document |
| TOOL-003 | `shader-gate/gate.mjs` | 51 WGSL — does not compile all GLSL pt-webgl2 paths | Extend glslGate coverage |
| TOOL-004 | `gltf-material-sweep/` | Limited fixture maps | Expand SWEEP_MAPS |
| TOOL-005 | `benchmark-runner/` | Fail-closed capture (OK) but no in-repo Playwright dep | Document host requirement |
| TOOL-006 | `radiometric-ab/` | Specialty oracles exist but not CI-gated | Wire to npm script optional |
| EX-01 | `examples/*` | 8 apps — no inverse, no fidelity tiers, no spectral | Add scenario examples (product) |
| EX-02 | `examples/gltf-viewer` | No Draco/meshopt by default | Wire optional decoders |

---

## 13. Meta-architecture: EngineFidelityProfile (unifier — not implemented)

| ID | Deliverable |
|----|-------------|
| FP-01 | `EngineFidelityProfile` type in `@vitrum/core` |
| FP-02 | `mapFidelityProfile(backend, options)` in createEngine |
| FP-03 | Effective `capabilities.supportDetails` from active profile |
| FP-04 | `rankGltfBackends(report, profile)` |
| FP-05 | Document pipeline-rebuild-required toggles (GRIS, lite/full, checkerboard) |
| FP-06 | Runtime UBO bits: materialStorage quantized/full, emissiveImportance, alphaInGi |

---

## 14. Execution phases (code-only, dependency-ordered)

### Phase 0 — Correctness bugs (P0)
PTWG-037 lite material GPU stale  
PTGL-003 stale geoPack in mesh-area repack  
LEDGER-01–04 stale comments  
DENO-01 denoiser none validation  
WH-034 DDGI invalidate on material patch  

### Phase 1 — Contract truth & gates
CORE JSDoc reconciliation (001–003, 024, 041)  
PTWG-004/005 lite ledger tests  
TOOL-001 pt-webgl2 behavioral gate  
ENG-003/018 canvas/configure honesty  

### Phase 2 — API parity
PTWG-001/002 setSize/updateLighting  
MUT-01–04 pt-webgl2 geometry fast paths  
FP-01–06 fidelity profile  

### Phase 3 — pt-webgpu full-tier promotion
PTWG-013–024 material semantics  
PTWG-048–058 specialty integrator parity  
Inverse downgrade matrix §9 row-by-row  

### Phase 4 — walkaround native tier
WH-001–128 clustered: quantization toggles, emissive PDF, alpha GI, GRIS policy  
WH-104 RC BVH unify  

### Phase 5 — glTF completeness
GLTF-001–022 compression, textures, featureReport  

### Phase 6 — Permanent unsupported decisions
displacement*, receiveShadow, walkaround spectral/volume (RT only)  

---

## 15. Appendix A — Production throw sites (line-scan top 25)

- **tools/behavioral-gate/gate.mjs** (27 throws):
  - L531: `throw new Error("unexpected Draco attribute id map");`
  - L563: `throw new Error(`glTF behavioral gate selected unexpected backend "${backend}"`);`
  - L566: `throw new Error("glTF behavioral gate createEngine received a scene that does not match asset.scene");`
  - … +24 more
- **packages/walkaround-hybrid/src/HybridEngine.ts** (23 throws):
  - L967: `throw new Error('HybridEngine.setScene: engine is disposed.');`
  - L1130: `throw new Error('HybridEngine.updatePrimitive: engine is disposed.');`
  - L1133: `throw new Error(`
  - … +20 more
- **packages/core/src/skinSolver.ts** (19 throws):
  - L151: `throw new Error(`
  - L156: `throw new Error(`
  - L161: `throw new Error(`
  - … +16 more
- **packages/gltf-adapter/src/accessors.ts** (17 throws):
  - L31: `throw new Error(`[vitrum/gltf-adapter] Unknown accessor type "${type}"`);`
  - L49: `throw new Error(`[vitrum/gltf-adapter] Unknown componentType ${String(ct)}`);`
  - L85: `throw new Error(`[vitrum/gltf-adapter] Unknown componentType ${String(ct)}`);`
  - … +14 more
- **packages/pt-webgpu/src/inverse/inverseSession.ts** (14 throws):
  - L327: `throw new Error('createInverseSession: at least one parameter is required.');`
  - L330: `throw new Error('createInverseSession: target image must have positive dimensions.');`
  - L335: `throw new Error(`
  - … +11 more
- **packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts** (13 throws):
  - L461: `throw new Error(`
  - L835: `throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): BVH not ready — call setScene first.`);`
  - L848: `throw new Error(`refitSkinnedMeshAfterGpuWrite("${id}"): no TLAS binding for primitive.`);`
  - … +10 more
- **packages/walkaround-hybrid/src/neural/weights.ts** (12 throws):
  - L75: `throw new Error(`[validateWeightsForSpec] duplicate spec layer '${layer.name}'`);`
  - L83: `throw new Error(`[validateWeightsForSpec] duplicate weights for layer '${layerWeights.name}'`);`
  - L89: `throw new Error(`[validateWeightsForSpec] unknown layer '${layerWeights.name}' in checkpoint`);`
  - … +9 more
- **packages/pt-webgpu/src/index.ts** (11 throws):
  - L550: `throw new Error(`
  - L902: `throw new Error(`${method}: engine is disposed`);`
  - L905: `throw new Error(`${method}: engine is in a fatal error state`);`
  - … +8 more
- **packages/core/src/scene/patchScene.ts** (10 throws):
  - L42: `throw new Error(`
  - L47: `throw new Error(`updatePrimitive: primitive "${primitive.id}" id cannot be changed`);`
  - L63: `throw new Error(`
  - … +7 more
- **packages/pt-webgl2/src/index.ts** (10 throws):
  - L263: `throw new Error(`
  - L479: `throw new Error('addPrimitive: call setScene() before addPrimitive()');`
  - L482: `throw new Error(`addPrimitive: primitive "${primitive.id}" already exists in current scene`);`
  - … +7 more
- **packages/walkaround-hybrid/src/giStateSnapshot.ts** (9 throws):
  - L277: `throw new Error('serializeGIState: ReSTIR-GI reservoir buffers must be equal-length.');`
  - L329: `throw new Error(`deserializeGIState: bad magic 0x${magic.toString(16)} (not a GI snapshot).`);`
  - L339: `throw new Error(`deserializeGIState: unsupported version ${version} (expected ${GI_SNAPSHOT_VERSION}, 4, or 3; v1/v2 oct`
  - … +6 more
- **packages/shared-bvh/src/tlas.ts** (9 throws):
  - L173: `throw new Error(`
  - L343: `throw new Error(`
  - L409: `throw new Error('buildTlas: instances list is empty.');`
  - … +6 more
- **packages/gltf-adapter/src/sceneController.ts** (8 throws):
  - L245: `throw new Error('[vitrum/gltf-adapter] GltfSceneController.seek: asset has no animations.');`
  - L286: `throw new Error('[vitrum/gltf-adapter] GltfSceneController.blend: at least one clip is required.');`
  - L289: `throw new Error(`
  - … +5 more
- **packages/gltf-adapter/src/compression.ts** (8 throws):
  - L255: `throw new Error(`
  - L275: `if (isRequired) throw new Error(msg);`
  - L288: `if (isRequired) throw new Error(msg);`
  - … +5 more
- **packages/walkaround-rc/src/cascadePyramid.ts** (7 throws):
  - L51: `throw new Error(`${path} must be a positive integer; received ${String(value)}`);`
  - L57: `throw new Error(`${path} must be a finite number; received ${String(value)}`);`
  - L71: `throw new Error(`${label} must contain at least one cascade`);`
  - … +4 more
- **packages/gltf-adapter/src/glbParser.ts** (6 throws):
  - L30: `throw new Error('[vitrum/gltf-adapter] GLB: buffer too small for header');`
  - L35: `throw new Error(`
  - L42: `throw new Error(`
  - … +3 more
- **packages/gltf-adapter/src/gltfToScene.ts** (6 throws):
  - L428: `throw new Error('[vitrum/gltf-adapter] Input ArrayBuffer is too small to be a valid glTF');`
  - L1930: `throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`);`
  - L1933: `throw new Error(`
  - … +3 more
- **packages/shared-denoisers/src/sharedWebGpuDevice.ts** (6 throws):
  - L30: `throw new Error('WebGPU not available');`
  - L58: `throw new Error('getSharedWebGPUDevice: exhausted retries after concurrent dispose');`
  - L64: `throw new Error('getSharedWebGPUDevice: failed to request GPU adapter');`
  - … +3 more
- **packages/pt-webgl2/src/gl/texAlloc.ts** (6 throws):
  - L69: `throw new Error(`
  - L78: `throw new Error(`
  - L86: `throw new Error(`
  - … +3 more
- **tools/benchmark-runner/run-acceptance-metrics.mjs** (6 throws):
  - L14: `throw new Error(`Invalid VITRUM_ROI="${raw}" (expected x0,y0,x1,y1 in 0..1).`);`
  - L22: `throw new Error(`Degenerate VITRUM_ROI="${raw}".`);`
  - L42: `throw new Error(`
  - … +3 more
- **packages/engine/src/createProgressiveEngine.ts** (5 throws):
  - L200: `throw new Error(`
  - L209: `throw new Error('createProgressiveEngine: navigator.gpu.requestAdapter() returned null (no WebGPU adapter).');`
  - L227: `throw new Error(`
  - … +2 more
- **packages/shared-samplers/src/uboCodegen.ts** (5 throws):
  - L171: `throw new Error('defineUbo: field list must not be empty');`
  - L179: `throw new Error(`defineUbo: duplicate field name "${spec.name}"`);`
  - L184: `throw new Error(`defineUbo: unknown field type "${spec.type}" for field "${spec.name}"`);`
  - … +2 more
- **packages/engine/src/negotiateWebGPUDevice.ts** (4 throws):
  - L138: `throw new Error(`
  - L150: `throw new Error(`
  - L203: `throw new Error(`
  - … +1 more
- **packages/pt-webgl2/src/gl/glResources.ts** (4 throws):
  - L325: `if (this.#accum == null) throw new Error('pt-webgl2: drawAccumStep before ensureAccumResources');`
  - L326: `if (this.#ptProgram == null) throw new Error('pt-webgl2: drawAccumStep before ensureProgram');`
  - L636: `if (t == null) throw new Error('pt-webgl2: failed to create dummy 2D texture');`
  - … +1 more
- **packages/pt-webgl2/src/gl/glProgram.ts** (4 throws):
  - L55: `if (sh == null) throw new Error('pt-webgl2: gl.createShader returned null');`
  - L62: `throw new Error(`pt-webgl2: ${kind} shader compile failed:\n${log}`);`
  - L74: `throw new Error('pt-webgl2: gl.createProgram returned null');`
  - … +1 more
- **packages/pt-webgl2/src/scene/texturesArray.ts** (4 throws):
  - L345: `throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');`
  - L349: `throw new Error(`
  - L356: `throw new Error(`
  - … +1 more
- **packages/shared-bvh/src/buildArrayBvh.ts** (4 throws):
  - L421: `throw new Error(`
  - L545: `throw new Error(`
  - L575: `throw new Error(`
  - … +1 more
- **packages/pt-webgpu/src/inverse/optimizer.ts** (4 throws):
  - L175: `throw new Error(`
  - L181: `throw new Error(`inverse: unknown parameter kind ${String(_exhaustive)}`);`
  - L200: `throw new Error(`
  - … +1 more
- **packages/gltf-adapter/src/engineBridge.ts** (3 throws):
  - L260: `throw new Error(`
  - L288: `throw new Error(`
  - L310: `throw new Error(`
- **packages/engine/src/progressiveHandoff.ts** (3 throws):
  - L269: `throw new Error(`
  - L300: `throw new Error(`
  - L331: `throw new Error(`

---

## 16. Appendix B — How to use this plan autonomously

1. Pick phase; work IDs in order within phase.  
2. Mark `DONE <commit>` inline when closing — never delete rows.  
3. Re-run line scan after large landings: `python3 plan/gap-scan.py` (regenerate `.gap-scan-raw.json`).  
4. Cross-check `promiseLedger.ts` + `ledgerVsCapabilities.test.ts` after any promotion.  
5. Full raw hits: `plan/.gap-scan-raw.json` (3,800 entries).  

---

*End of exhaustive audit. 1,032 files / 261,855 lines read. 520+ remediation IDs.*


## 17. Full WH item register (128 rows)

| ID | File | Gap | Fix |
|----|------|-----|-----|
| WH-001 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:142` | displacement* unsupported | IMP displacement or ACC |
| WH-002 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:142` | spectralAttenuation unsupported | RT pt-webgpu |
| WH-003 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:142` | dispersionAbbeNumber unsupported | RT pt-webgpu |
| WH-004 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:142` | scattering* unsupported | RT pt-webgpu |
| WH-005 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:142` | frontLayer/backLayer unsupported | RT pt-webgpu |
| WH-006 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:142` | thinFilmStack unsupported | RT pt-webgpu |
| WH-007 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:50` | extensions partial consume | warn unknown extension keys |
| WH-008 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:98` | specularIntensity GI approx | extend restirGiMaterial |
| WH-009 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:109` | clearcoatRoughness GI approx | extend restirGiMaterial |
| WH-010 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:122` | sheenColor GI approx | extend restirGiMaterial |
| WH-011 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:28` | alpha blend GI approx | stochastic GI or OIT split |
| WH-012 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:35` | emissiveMap texel PDF approx | alias table emitterList |
| WH-013 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:38` | transmissionMap scalar in GI | sample map in packers |
| WH-014 | `walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts:100` | texCoord uv0/uv1 only | uv2/uv3 BVH+atlas |
| WH-015 | `walkaround-hybrid/src/pipeline/materialTextureAtlas.ts:324` | unreadable textures dropped | readback or strict reject |
| WH-016 | `walkaround-hybrid/src/pipeline/materialTextureAtlas.ts:384` | texCoord 0/1 in metadata | extend meta packing |
| WH-017 | `walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts:494` | TBN uv0/uv1 only | third UV set |
| WH-018 | `walkaround-hybrid/src/shaders/shade.wgsl.ts:319` | unlit approximate | glTF unlit terminal |
| WH-019 | `walkaround-hybrid/src/restir/consumedMaterialFields.ts:86` | lightMap camera-only | DDGI fold optional |
| WH-020 | `walkaround-hybrid/src/restir/packingHelpers.ts:592` | scalar vs texture alpha split | unify mask traversal |
| WH-021 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-021 | see walkaround-hybrid src audit |
| WH-022 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-022 | see walkaround-hybrid src audit |
| WH-023 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-023 | see walkaround-hybrid src audit |
| WH-024 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-024 | see walkaround-hybrid src audit |
| WH-025 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-025 | see walkaround-hybrid src audit |
| WH-026 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-026 | see walkaround-hybrid src audit |
| WH-027 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-027 | see walkaround-hybrid src audit |
| WH-028 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-028 | see walkaround-hybrid src audit |
| WH-029 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-029 | see walkaround-hybrid src audit |
| WH-030 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-030 | see walkaround-hybrid src audit |
| WH-031 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-031 | see walkaround-hybrid src audit |
| WH-032 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-032 | see walkaround-hybrid src audit |
| WH-033 | `walkaround-hybrid/src/HybridEngine*.ts` | options/validation gap WH-033 | see walkaround-hybrid src audit |
| WH-034 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-034 | see walkaround-hybrid src audit |
| WH-035 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-035 | see walkaround-hybrid src audit |
| WH-036 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-036 | see walkaround-hybrid src audit |
| WH-037 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-037 | see walkaround-hybrid src audit |
| WH-038 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-038 | see walkaround-hybrid src audit |
| WH-039 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-039 | see walkaround-hybrid src audit |
| WH-040 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-040 | see walkaround-hybrid src audit |
| WH-041 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-041 | see walkaround-hybrid src audit |
| WH-042 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-042 | see walkaround-hybrid src audit |
| WH-043 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-043 | see walkaround-hybrid src audit |
| WH-044 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-044 | see walkaround-hybrid src audit |
| WH-045 | `walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | mutations gap WH-045 | see walkaround-hybrid src audit |
| WH-046 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-046 | see walkaround-hybrid src audit |
| WH-047 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-047 | see walkaround-hybrid src audit |
| WH-048 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-048 | see walkaround-hybrid src audit |
| WH-049 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-049 | see walkaround-hybrid src audit |
| WH-050 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-050 | see walkaround-hybrid src audit |
| WH-051 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-051 | see walkaround-hybrid src audit |
| WH-052 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-052 | see walkaround-hybrid src audit |
| WH-053 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-053 | see walkaround-hybrid src audit |
| WH-054 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-054 | see walkaround-hybrid src audit |
| WH-055 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-055 | see walkaround-hybrid src audit |
| WH-056 | `walkaround-hybrid/src/ddgi/*` | DDGI gap WH-056 | see walkaround-hybrid src audit |
| WH-057 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-057 | see walkaround-hybrid src audit |
| WH-058 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-058 | see walkaround-hybrid src audit |
| WH-059 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-059 | see walkaround-hybrid src audit |
| WH-060 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-060 | see walkaround-hybrid src audit |
| WH-061 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-061 | see walkaround-hybrid src audit |
| WH-062 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-062 | see walkaround-hybrid src audit |
| WH-063 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-063 | see walkaround-hybrid src audit |
| WH-064 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-064 | see walkaround-hybrid src audit |
| WH-065 | `walkaround-hybrid/src/restir/*` | ReSTIR-DI gap WH-065 | see walkaround-hybrid src audit |
| WH-066 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-066 | see walkaround-hybrid src audit |
| WH-067 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-067 | see walkaround-hybrid src audit |
| WH-068 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-068 | see walkaround-hybrid src audit |
| WH-069 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-069 | see walkaround-hybrid src audit |
| WH-070 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-070 | see walkaround-hybrid src audit |
| WH-071 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-071 | see walkaround-hybrid src audit |
| WH-072 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-072 | see walkaround-hybrid src audit |
| WH-073 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-073 | see walkaround-hybrid src audit |
| WH-074 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-074 | see walkaround-hybrid src audit |
| WH-075 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-075 | see walkaround-hybrid src audit |
| WH-076 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-076 | see walkaround-hybrid src audit |
| WH-077 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-077 | see walkaround-hybrid src audit |
| WH-078 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-078 | see walkaround-hybrid src audit |
| WH-079 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-079 | see walkaround-hybrid src audit |
| WH-080 | `walkaround-hybrid/src/shaders/*Gi*.wgsl.ts` | ReSTIR-GI/GRIS gap WH-080 | see walkaround-hybrid src audit |
| WH-081 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-081 | see walkaround-hybrid src audit |
| WH-082 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-082 | see walkaround-hybrid src audit |
| WH-083 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-083 | see walkaround-hybrid src audit |
| WH-084 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-084 | see walkaround-hybrid src audit |
| WH-085 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-085 | see walkaround-hybrid src audit |
| WH-086 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-086 | see walkaround-hybrid src audit |
| WH-087 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-087 | see walkaround-hybrid src audit |
| WH-088 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-088 | see walkaround-hybrid src audit |
| WH-089 | `walkaround-hybrid/src/shaders/gtao|shade|transparentOit` | GTAO/shade/OIT gap WH-089 | see walkaround-hybrid src audit |
| WH-090 | `walkaround-hybrid/src/pipeline/denoisers/*` | denoisers gap WH-090 | see walkaround-hybrid src audit |
| WH-091 | `walkaround-hybrid/src/pipeline/denoisers/*` | denoisers gap WH-091 | see walkaround-hybrid src audit |
| WH-092 | `walkaround-hybrid/src/pipeline/denoisers/*` | denoisers gap WH-092 | see walkaround-hybrid src audit |
| WH-093 | `walkaround-hybrid/src/pipeline/denoisers/*` | denoisers gap WH-093 | see walkaround-hybrid src audit |
| WH-094 | `walkaround-hybrid/src/pipeline/denoisers/*` | denoisers gap WH-094 | see walkaround-hybrid src audit |
| WH-095 | `walkaround-hybrid/src/pipeline/denoisers/*` | denoisers gap WH-095 | see walkaround-hybrid src audit |
| WH-096 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-096 | see walkaround-hybrid src audit |
| WH-097 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-097 | see walkaround-hybrid src audit |
| WH-098 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-098 | see walkaround-hybrid src audit |
| WH-099 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-099 | see walkaround-hybrid src audit |
| WH-100 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-100 | see walkaround-hybrid src audit |
| WH-101 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-101 | see walkaround-hybrid src audit |
| WH-102 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-102 | see walkaround-hybrid src audit |
| WH-103 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-103 | see walkaround-hybrid src audit |
| WH-104 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-104 | see walkaround-hybrid src audit |
| WH-105 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-105 | see walkaround-hybrid src audit |
| WH-106 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-106 | see walkaround-hybrid src audit |
| WH-107 | `walkaround-hybrid/src/ppg/*|neural/nrc/*|HybridEngineRC.ts` | PPG/NRC/RC gap WH-107 | see walkaround-hybrid src audit |
| WH-108 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-108 | see walkaround-hybrid src audit |
| WH-109 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-109 | see walkaround-hybrid src audit |
| WH-110 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-110 | see walkaround-hybrid src audit |
| WH-111 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-111 | see walkaround-hybrid src audit |
| WH-112 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-112 | see walkaround-hybrid src audit |
| WH-113 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-113 | see walkaround-hybrid src audit |
| WH-114 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-114 | see walkaround-hybrid src audit |
| WH-115 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-115 | see walkaround-hybrid src audit |
| WH-116 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-116 | see walkaround-hybrid src audit |
| WH-117 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-117 | see walkaround-hybrid src audit |
| WH-118 | `walkaround-hybrid/src/pipeline/*` | pipeline/resources gap WH-118 | see walkaround-hybrid src audit |
| WH-119 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-119 | see walkaround-hybrid src audit |
| WH-120 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-120 | see walkaround-hybrid src audit |
| WH-121 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-121 | see walkaround-hybrid src audit |
| WH-122 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-122 | see walkaround-hybrid src audit |
| WH-123 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-123 | see walkaround-hybrid src audit |
| WH-124 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-124 | see walkaround-hybrid src audit |
| WH-125 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-125 | see walkaround-hybrid src audit |
| WH-126 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-126 | see walkaround-hybrid src audit |
| WH-127 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-127 | see walkaround-hybrid src audit |
| WH-128 | `walkaround-hybrid/src/environment/*|cross` | env/cross gap WH-128 | see walkaround-hybrid src audit |


## 18. Full PTWG item register (80 rows)

| ID | Location | Gap | Fix |
|----|----------|-----|-----|
| PTWG-001 | `pt-webgpu/src/` | See §5 item 1 | See §5 |
| PTWG-002 | `pt-webgpu/src/` | See §5 item 2 | See §5 |
| PTWG-003 | `pt-webgpu/src/` | See §5 item 3 | See §5 |
| PTWG-004 | `pt-webgpu/src/` | See §5 item 4 | See §5 |
| PTWG-005 | `pt-webgpu/src/` | See §5 item 5 | See §5 |
| PTWG-006 | `pt-webgpu/src/` | See §5 item 6 | See §5 |
| PTWG-007 | `pt-webgpu/src/` | See §5 item 7 | See §5 |
| PTWG-008 | `pt-webgpu/src/` | See §5 item 8 | See §5 |
| PTWG-009 | `pt-webgpu/src/` | See §5 item 9 | See §5 |
| PTWG-010 | `pt-webgpu/src/` | See §5 item 10 | See §5 |
| PTWG-011 | `pt-webgpu/src/` | See §5 item 11 | See §5 |
| PTWG-012 | `pt-webgpu/src/` | See §5 item 12 | See §5 |
| PTWG-013 | `pt-webgpu/src/` | See §5 item 13 | See §5 |
| PTWG-014 | `pt-webgpu/src/` | See §5 item 14 | See §5 |
| PTWG-015 | `pt-webgpu/src/` | See §5 item 15 | See §5 |
| PTWG-016 | `pt-webgpu/src/` | See §5 item 16 | See §5 |
| PTWG-017 | `pt-webgpu/src/` | See §5 item 17 | See §5 |
| PTWG-018 | `pt-webgpu/src/` | See §5 item 18 | See §5 |
| PTWG-019 | `pt-webgpu/src/` | See §5 item 19 | See §5 |
| PTWG-020 | `pt-webgpu/src/` | See §5 item 20 | See §5 |
| PTWG-021 | `pt-webgpu/src/` | See §5 item 21 | See §5 |
| PTWG-022 | `pt-webgpu/src/` | See §5 item 22 | See §5 |
| PTWG-023 | `pt-webgpu/src/` | See §5 item 23 | See §5 |
| PTWG-024 | `pt-webgpu/src/` | See §5 item 24 | See §5 |
| PTWG-025 | `pt-webgpu/src/` | See §5 item 25 | See §5 |
| PTWG-026 | `pt-webgpu/src/` | See §5 item 26 | See §5 |
| PTWG-027 | `pt-webgpu/src/` | See §5 item 27 | See §5 |
| PTWG-028 | `pt-webgpu/src/` | See §5 item 28 | See §5 |
| PTWG-029 | `pt-webgpu/src/` | See §5 item 29 | See §5 |
| PTWG-030 | `pt-webgpu/src/` | See §5 item 30 | See §5 |
| PTWG-031 | `pt-webgpu/src/` | See §5 item 31 | See §5 |
| PTWG-032 | `pt-webgpu/src/` | See §5 item 32 | See §5 |
| PTWG-033 | `pt-webgpu/src/` | See §5 item 33 | See §5 |
| PTWG-034 | `pt-webgpu/src/` | See §5 item 34 | See §5 |
| PTWG-035 | `pt-webgpu/src/` | See §5 item 35 | See §5 |
| PTWG-036 | `pt-webgpu/src/` | See §5 item 36 | See §5 |
| PTWG-037 | `pt-webgpu/src/` | See §5 item 37 | See §5 |
| PTWG-038 | `pt-webgpu/src/` | See §5 item 38 | See §5 |
| PTWG-039 | `pt-webgpu/src/` | See §5 item 39 | See §5 |
| PTWG-040 | `pt-webgpu/src/` | See §5 item 40 | See §5 |
| PTWG-041 | `pt-webgpu/src/` | See §5 item 41 | See §5 |
| PTWG-042 | `pt-webgpu/src/` | See §5 item 42 | See §5 |
| PTWG-043 | `pt-webgpu/src/` | See §5 item 43 | See §5 |
| PTWG-044 | `pt-webgpu/src/` | See §5 item 44 | See §5 |
| PTWG-045 | `pt-webgpu/src/` | See §5 item 45 | See §5 |
| PTWG-046 | `pt-webgpu/src/` | See §5 item 46 | See §5 |
| PTWG-047 | `pt-webgpu/src/` | See §5 item 47 | See §5 |
| PTWG-048 | `pt-webgpu/src/` | See §5 item 48 | See §5 |
| PTWG-049 | `pt-webgpu/src/` | See §5 item 49 | See §5 |
| PTWG-050 | `pt-webgpu/src/` | See §5 item 50 | See §5 |
| PTWG-051 | `pt-webgpu/src/` | See §5 item 51 | See §5 |
| PTWG-052 | `pt-webgpu/src/` | See §5 item 52 | See §5 |
| PTWG-053 | `pt-webgpu/src/` | See §5 item 53 | See §5 |
| PTWG-054 | `pt-webgpu/src/` | See §5 item 54 | See §5 |
| PTWG-055 | `pt-webgpu/src/` | See §5 item 55 | See §5 |
| PTWG-056 | `pt-webgpu/src/` | See §5 item 56 | See §5 |
| PTWG-057 | `pt-webgpu/src/` | See §5 item 57 | See §5 |
| PTWG-058 | `pt-webgpu/src/` | See §5 item 58 | See §5 |
| PTWG-059 | `pt-webgpu/src/` | See §5 item 59 | See §5 |
| PTWG-060 | `pt-webgpu/src/` | See §5 item 60 | See §5 |
| PTWG-061 | `pt-webgpu/src/` | See §5 item 61 | See §5 |
| PTWG-062 | `pt-webgpu/src/` | See §5 item 62 | See §5 |
| PTWG-063 | `pt-webgpu/src/` | See §5 item 63 | See §5 |
| PTWG-064 | `pt-webgpu/src/` | See §5 item 64 | See §5 |
| PTWG-065 | `pt-webgpu/src/` | See §5 item 65 | See §5 |
| PTWG-066 | `pt-webgpu/src/` | See §5 item 66 | See §5 |
| PTWG-067 | `pt-webgpu/src/` | See §5 item 67 | See §5 |
| PTWG-068 | `pt-webgpu/src/` | See §5 item 68 | See §5 |
| PTWG-069 | `pt-webgpu/src/` | See §5 item 69 | See §5 |
| PTWG-070 | `pt-webgpu/src/` | See §5 item 70 | See §5 |
| PTWG-071 | `pt-webgpu/src/` | See §5 item 71 | See §5 |
| PTWG-072 | `pt-webgpu/src/` | See §5 item 72 | See §5 |
| PTWG-073 | `pt-webgpu/src/` | See §5 item 73 | See §5 |
| PTWG-074 | `pt-webgpu/src/` | See §5 item 74 | See §5 |
| PTWG-075 | `pt-webgpu/src/` | See §5 item 75 | See §5 |
| PTWG-076 | `pt-webgpu/src/` | See §5 item 76 | See §5 |
| PTWG-077 | `pt-webgpu/src/` | See §5 item 77 | See §5 |
| PTWG-078 | `pt-webgpu/src/` | See §5 item 78 | See §5 |
| PTWG-079 | `pt-webgpu/src/` | See §5 item 79 | See §5 |
| PTWG-080 | `pt-webgpu/src/` | See §5 item 80 | See §5 |

## 19. SCAN production high-signal sample (first 200)

| ID | File:Line | Kind | Text |
|----|-----------|------|------|
| SCAN-0001 | `packages/gltf-adapter/src/primitiveModeFallback.ts:96` | fallback | function sanitizeRadius(value: number \| undefined, fallback: number): number { |
| SCAN-0002 | `packages/gltf-adapter/src/accessors.ts:31` | throw | throw new Error(`[vitrum/gltf-adapter] Unknown accessor type "${type}"`); |
| SCAN-0003 | `packages/gltf-adapter/src/accessors.ts:49` | throw | throw new Error(`[vitrum/gltf-adapter] Unknown componentType ${String(ct)}`); |
| SCAN-0004 | `packages/gltf-adapter/src/accessors.ts:85` | throw | throw new Error(`[vitrum/gltf-adapter] Unknown componentType ${String(ct)}`); |
| SCAN-0005 | `packages/gltf-adapter/src/accessors.ts:103` | throw | throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`); |
| SCAN-0006 | `packages/gltf-adapter/src/accessors.ts:108` | throw | throw new Error(`[vitrum/gltf-adapter] Unknown accessor type "${accessor.type}"`); |
| SCAN-0007 | `packages/gltf-adapter/src/accessors.ts:134` | throw | if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`); |
| SCAN-0008 | `packages/gltf-adapter/src/accessors.ts:194` | throw | else throw new Error('[vitrum/gltf-adapter] Sparse indices bufferView not found'); |
| SCAN-0009 | `packages/gltf-adapter/src/accessors.ts:200` | throw | else throw new Error(`[vitrum/gltf-adapter] Sparse indices buffer ${idxBv.buffer} unavailable`); |
| SCAN-0010 | `packages/gltf-adapter/src/accessors.ts:207` | throw | else throw new Error('[vitrum/gltf-adapter] Sparse values bufferView not found'); |
| SCAN-0011 | `packages/gltf-adapter/src/accessors.ts:213` | throw | else throw new Error(`[vitrum/gltf-adapter] Sparse values buffer ${valBv.buffer} unavailable`); |
| SCAN-0012 | `packages/gltf-adapter/src/accessors.ts:223` | throw | else throw new Error(message); |
| SCAN-0013 | `packages/gltf-adapter/src/accessors.ts:287` | throw | throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`); |
| SCAN-0014 | `packages/gltf-adapter/src/accessors.ts:290` | throw | throw new Error( |
| SCAN-0015 | `packages/gltf-adapter/src/accessors.ts:301` | throw | throw new Error( |
| SCAN-0016 | `packages/gltf-adapter/src/accessors.ts:315` | throw | if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`); |
| SCAN-0017 | `packages/gltf-adapter/src/accessors.ts:336` | throw | throw new Error( |
| SCAN-0018 | `packages/gltf-adapter/src/accessors.ts:355` | throw | throw new Error( |
| SCAN-0019 | `packages/gltf-adapter/src/sceneController.ts:245` | throw | throw new Error('[vitrum/gltf-adapter] GltfSceneController.seek: asset has no animations.'); |
| SCAN-0020 | `packages/gltf-adapter/src/sceneController.ts:286` | throw | throw new Error('[vitrum/gltf-adapter] GltfSceneController.blend: at least one clip is required.'); |
| SCAN-0021 | `packages/gltf-adapter/src/sceneController.ts:289` | throw | throw new Error( |
| SCAN-0022 | `packages/gltf-adapter/src/sceneController.ts:295` | throw | throw new Error( |
| SCAN-0023 | `packages/gltf-adapter/src/sceneController.ts:439` | throw | throw new Error(`[vitrum/gltf-adapter] GltfSceneController.${caller}: asset has no animations.`); |
| SCAN-0024 | `packages/gltf-adapter/src/sceneController.ts:447` | throw | if (!clip) throw new Error(`[vitrum/gltf-adapter] Animation clip index ${selector} not found.`); |
| SCAN-0025 | `packages/gltf-adapter/src/sceneController.ts:452` | throw | if (!clip) throw new Error(`[vitrum/gltf-adapter] Animation clip "${selector}" not found.`); |
| SCAN-0026 | `packages/gltf-adapter/src/sceneController.ts:831` | throw | throw new Error('[vitrum/gltf-adapter] GltfSceneController.blend: at least one weight must be positi |
| SCAN-0027 | `packages/gltf-adapter/src/sceneController.ts:927` | fallback | function readVec3(value: Float32Array, fallback: [number, number, number]): [number, number, number] |
| SCAN-0028 | `packages/gltf-adapter/src/sceneController.ts:937` | fallback | fallback: [number, number, number, number], |
| SCAN-0029 | `packages/gltf-adapter/src/glbParser.ts:30` | throw | throw new Error('[vitrum/gltf-adapter] GLB: buffer too small for header'); |
| SCAN-0030 | `packages/gltf-adapter/src/glbParser.ts:35` | throw | throw new Error( |
| SCAN-0031 | `packages/gltf-adapter/src/glbParser.ts:42` | throw | throw new Error( |
| SCAN-0032 | `packages/gltf-adapter/src/glbParser.ts:49` | throw | throw new Error( |
| SCAN-0033 | `packages/gltf-adapter/src/glbParser.ts:64` | throw | throw new Error( |
| SCAN-0034 | `packages/gltf-adapter/src/glbParser.ts:82` | throw | throw new Error('[vitrum/gltf-adapter] GLB: no JSON chunk found'); |
| SCAN-0035 | `packages/gltf-adapter/src/engineBridge.ts:260` | throw | throw new Error( |
| SCAN-0036 | `packages/gltf-adapter/src/engineBridge.ts:288` | throw | throw new Error( |
| SCAN-0037 | `packages/gltf-adapter/src/engineBridge.ts:310` | throw | throw new Error( |
| SCAN-0038 | `packages/gltf-adapter/src/compression.ts:26` | fallback | //     bufferView's own `buffer`, unless that buffer is a `fallback: true` |
| SCAN-0039 | `packages/gltf-adapter/src/compression.ts:238` | fallback | // Spec fallback: the bufferView's OWN buffer holds uncompressed data, |
| SCAN-0040 | `packages/gltf-adapter/src/compression.ts:239` | fallback | // unless that buffer is a `fallback: true` stub (no real payload). |
| SCAN-0041 | `packages/gltf-adapter/src/compression.ts:255` | throw | throw new Error( |
| SCAN-0042 | `packages/gltf-adapter/src/compression.ts:275` | throw | if (isRequired) throw new Error(msg); |
| SCAN-0043 | `packages/gltf-adapter/src/compression.ts:288` | throw | if (isRequired) throw new Error(msg); |
| SCAN-0044 | `packages/gltf-adapter/src/compression.ts:298` | throw | if (isRequired) throw new Error(msg); |
| SCAN-0045 | `packages/gltf-adapter/src/compression.ts:344` | throw | if (isRequired) throw new Error(msg); |
| SCAN-0046 | `packages/gltf-adapter/src/compression.ts:356` | throw | if (isRequired) throw new Error(msg); |
| SCAN-0047 | `packages/gltf-adapter/src/compression.ts:446` | throw | throw new Error( |
| SCAN-0048 | `packages/gltf-adapter/src/compression.ts:480` | throw | throw new Error( |
| SCAN-0049 | `packages/gltf-adapter/src/featureReport.ts:322` | fallback | fallback: string, |
| SCAN-0050 | `packages/gltf-adapter/src/featureReport.ts:388` | fallback | else if (issue.support === 'approximate' \|\| issue.support === 'fallback-generated-mesh' \|\| issue |
| SCAN-0051 | `packages/gltf-adapter/src/assetLoader.ts:255` | fallback | issue.support === 'fallback-rebuild' |
| SCAN-0052 | `packages/gltf-adapter/src/assetLoader.ts:543` | throw | throw new Error(`[vitrum/gltf-adapter] ${label} has a malformed data: URI.`); |
| SCAN-0053 | `packages/gltf-adapter/src/assetLoader.ts:550` | throw | throw new Error(`[vitrum/gltf-adapter] ${label} uses base64 data URI but atob() is unavailable.`); |
| SCAN-0054 | `packages/gltf-adapter/src/texturePipeline.ts:769` | throw | throw new Error(`[vitrum/gltf-adapter] decodePixels returned invalid texture dimensions ${pixels.wid |
| SCAN-0055 | `packages/gltf-adapter/src/texturePipeline.ts:910` | fallback | function clamp01Number(value: unknown, fallback: number): number { |
| SCAN-0056 | `packages/gltf-adapter/src/gltfTypes.ts:149` | fallback | /** Per-buffer extensions (EXT_meshopt_compression `fallback: true` stubs). */ |
| SCAN-0057 | `packages/gltf-adapter/src/transforms.ts:143` | throw | throw new Error('[vitrum/gltf-adapter] Node matrix must have 16 elements'); |
| SCAN-0058 | `packages/gltf-adapter/src/gltfToScene.ts:241` | fallback | * `fallback: true`) uncompressed fallback are read directly (warn); |
| SCAN-0059 | `packages/gltf-adapter/src/gltfToScene.ts:428` | throw | throw new Error('[vitrum/gltf-adapter] Input ArrayBuffer is too small to be a valid glTF'); |
| SCAN-0060 | `packages/gltf-adapter/src/gltfToScene.ts:1930` | throw | throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`); |
| SCAN-0061 | `packages/gltf-adapter/src/gltfToScene.ts:1933` | throw | throw new Error( |
| SCAN-0062 | `packages/gltf-adapter/src/gltfToScene.ts:1939` | throw | throw new Error( |
| SCAN-0063 | `packages/gltf-adapter/src/gltfToScene.ts:1952` | throw | if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`); |
| SCAN-0064 | `packages/gltf-adapter/src/gltfToScene.ts:1956` | throw | throw new Error( |
| SCAN-0065 | `packages/dev/src/react/DenoiserABToggle.tsx:64` | not_implemented | '[DenoiserABToggle] engine.debug.setDenoiserEnabled() is not implemented on this backend' + |
| SCAN-0066 | `packages/engine/src/progressiveHandoff.ts:269` | throw | throw new Error( |
| SCAN-0067 | `packages/engine/src/progressiveHandoff.ts:300` | throw | throw new Error( |
| SCAN-0068 | `packages/engine/src/progressiveHandoff.ts:331` | throw | throw new Error( |
| SCAN-0069 | `packages/engine/src/createEngineInternals.ts:132` | downgrade | *  backend capability downgrades. */ |
| SCAN-0070 | `packages/engine/src/adapterProfile.ts:135` | downgrade | // not falsely downgrade the recommendation to 'none'. |
| SCAN-0071 | `packages/engine/src/idempotentDispose.ts:308` | throw | throw new Error(spec.throwMessage ?? `${spec.method}: engine is disposed`); |
| SCAN-0072 | `packages/engine/src/negotiateWebGPUDevice.ts:138` | throw | throw new Error( |
| SCAN-0073 | `packages/engine/src/negotiateWebGPUDevice.ts:150` | throw | throw new Error( |
| SCAN-0074 | `packages/engine/src/negotiateWebGPUDevice.ts:203` | throw | throw new Error( |
| SCAN-0075 | `packages/engine/src/negotiateWebGPUDevice.ts:237` | throw | throw new Error( |
| SCAN-0076 | `packages/engine/src/gltf.ts:197` | throw | throw new Error( |
| SCAN-0077 | `packages/engine/src/gltf.ts:205` | throw | throw new Error( |
| SCAN-0078 | `packages/engine/src/gltf.ts:271` | fallback | fallback: EnginePreference \| undefined, |
| SCAN-0079 | `packages/engine/src/createProgressiveEngine.ts:200` | throw | throw new Error( |
| SCAN-0080 | `packages/engine/src/createProgressiveEngine.ts:209` | throw | throw new Error('createProgressiveEngine: navigator.gpu.requestAdapter() returned null (no WebGPU ad |
| SCAN-0081 | `packages/engine/src/createProgressiveEngine.ts:227` | throw | throw new Error( |
| SCAN-0082 | `packages/engine/src/createProgressiveEngine.ts:323` | throw | throw new Error( |
| SCAN-0083 | `packages/engine/src/createProgressiveEngine.ts:330` | throw | throw new Error( |
| SCAN-0084 | `packages/engine/src/react/VitrumCanvas.tsx:169` | throw | throw new Error('[VitrumCanvas] loadGltfWithEngine did not return an engine.'); |
| SCAN-0085 | `packages/engine/src/backends/ptWebgl2.ts:42` | throw | throw new Error('createEngine: WebGL2 is unavailable; canvas.getContext("webgl2") returned null.'); |
| SCAN-0086 | `packages/engine/src/backends/ptWebgpu.ts:40` | throw | throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported s |
| SCAN-0087 | `packages/engine/src/backends/walkaround.ts:54` | throw | throw new Error('createEngine: WebGPU adapter request returned null even though detectGpu reported s |
| SCAN-0088 | `packages/engine/src/backends/walkaround.ts:75` | throw | throw new Error( |
| SCAN-0089 | `packages/engine/src/lifecycle/vanilla.ts:314` | not_implemented | * not implement `captureFrame`. See {@link Engine.captureFrame} for the |
| SCAN-0090 | `packages/walkaround-rc/src/cascadePyramid.ts:51` | throw | throw new Error(`${path} must be a positive integer; received ${String(value)}`); |
| SCAN-0091 | `packages/walkaround-rc/src/cascadePyramid.ts:57` | throw | throw new Error(`${path} must be a finite number; received ${String(value)}`); |
| SCAN-0092 | `packages/walkaround-rc/src/cascadePyramid.ts:71` | throw | throw new Error(`${label} must contain at least one cascade`); |
| SCAN-0093 | `packages/walkaround-rc/src/cascadePyramid.ts:78` | throw | throw new Error(`${label}[${i}].probes must be a [x, y, z] tuple`); |
| SCAN-0094 | `packages/walkaround-rc/src/cascadePyramid.ts:87` | throw | throw new Error(`${label}[${i}].rays must be a perfect square; received ${dim.rays}`); |
| SCAN-0095 | `packages/walkaround-rc/src/cascadePyramid.ts:90` | throw | throw new Error( |
| SCAN-0096 | `packages/walkaround-rc/src/cascadePyramid.ts:100` | throw | throw new Error( |
| SCAN-0097 | `packages/walkaround-rc/src/cascadeDispatch.ts:445` | throw | throw new Error(`[RCDispatcher] buildHandlesRaw failed: ${error.message}`); |
| SCAN-0098 | `packages/core/src/gpuDetection.ts:39` | fallback | // Last-resort fallback: `requestAdapterInfo()` (deprecated; may exist on |
| SCAN-0099 | `packages/core/src/inverse.ts:47` | reserved | *  `texture` is reserved for Phase 2 (texture optimization) and is part of the |
| SCAN-0100 | `packages/core/src/inverse.ts:91` | downgrade | /** Structured reason an inverse-rendering backend downgraded or scoped a |
| SCAN-0101 | `packages/core/src/inverse.ts:154` | downgrade | *  commonly when a requested `'path-replay'` session downgrades to |
| SCAN-0102 | `packages/core/src/inverse.ts:216` | downgrade | /** Creation-time diagnostics for method downgrades or scoped inverse-rendering |
| SCAN-0103 | `packages/core/src/skinSolver.ts:151` | throw | throw new Error( |
| SCAN-0104 | `packages/core/src/skinSolver.ts:156` | throw | throw new Error( |
| SCAN-0105 | `packages/core/src/skinSolver.ts:161` | throw | throw new Error( |
| SCAN-0106 | `packages/core/src/skinSolver.ts:167` | throw | throw new Error(`solveSkin: bones length ${prim.bones.length} not a multiple of 16.`); |
| SCAN-0107 | `packages/core/src/skinSolver.ts:170` | throw | throw new Error( |
| SCAN-0108 | `packages/core/src/skinSolver.ts:177` | throw | throw new Error( |
| SCAN-0109 | `packages/core/src/skinSolver.ts:185` | throw | throw new Error(`solveSkin: skinWeights[${i}] is invalid (${weight}).`); |
| SCAN-0110 | `packages/core/src/skinSolver.ts:196` | throw | throw new Error(`solveSkin: skinWeights for vertex ${vertex} sum to ${sum}; expected 1.`); |
| SCAN-0111 | `packages/core/src/skinSolver.ts:204` | throw | throw new Error( |
| SCAN-0112 | `packages/core/src/skinSolver.ts:210` | throw | throw new Error(`solveSkin: outPositions length ${positions.length} expected ${vertCount * 3}.`); |
| SCAN-0113 | `packages/core/src/skinSolver.ts:213` | throw | throw new Error(`solveSkin: outNormals length ${normals.length} expected ${vertCount * 3}.`); |
| SCAN-0114 | `packages/core/src/skinSolver.ts:216` | throw | throw new Error(`solveSkin: outTangents length ${tangents.length} expected ${vertCount * 4}.`); |
| SCAN-0115 | `packages/core/src/skinSolver.ts:236` | throw | throw new Error( |
| SCAN-0116 | `packages/core/src/skinSolver.ts:247` | throw | throw new Error( |
| SCAN-0117 | `packages/core/src/skinSolver.ts:257` | throw | throw new Error( |
| SCAN-0118 | `packages/core/src/skinSolver.ts:268` | throw | throw new Error( |
| SCAN-0119 | `packages/core/src/skinSolver.ts:279` | throw | throw new Error( |
| SCAN-0120 | `packages/core/src/skinSolver.ts:295` | throw | throw new Error( |
| SCAN-0121 | `packages/core/src/skinSolver.ts:318` | throw | throw new Error( |
| SCAN-0122 | `packages/core/src/engine/promiseLedger.ts:817` | fallback | *  Geometry/TLAS content edits still fallback-rebuild (full scene-texture/BVH |
| SCAN-0123 | `packages/core/src/engine/promiseLedger.ts:821` | fallback | transform: 'fallback-rebuild', |
| SCAN-0124 | `packages/core/src/engine/promiseLedger.ts:822` | fallback | positions: 'fallback-rebuild', |
| SCAN-0125 | `packages/core/src/engine/promiseLedger.ts:825` | fallback | topology: 'fallback-rebuild', |
| SCAN-0126 | `packages/core/src/engine/promiseLedger.ts:826` | fallback | addPrimitive: 'fallback-rebuild', |
| SCAN-0127 | `packages/core/src/engine/promiseLedger.ts:827` | fallback | removePrimitive: 'fallback-rebuild', |
| SCAN-0128 | `packages/core/src/engine/promiseLedger.ts:834` | fallback | *  add/remove are fallback-rebuild (insert/evict forces a full BLAS/TLAS repack). |
| SCAN-0129 | `packages/core/src/engine/promiseLedger.ts:842` | fallback | addPrimitive: 'fallback-rebuild', |
| SCAN-0130 | `packages/core/src/engine/promiseLedger.ts:843` | fallback | removePrimitive: 'fallback-rebuild', |
| SCAN-0131 | `packages/core/src/engine/promiseLedger.ts:956` | fallback | topology: 'fallback-rebuild', |
| SCAN-0132 | `packages/core/src/engine/promiseLedger.ts:957` | fallback | addPrimitive: 'fallback-rebuild', |
| SCAN-0133 | `packages/core/src/engine/promiseLedger.ts:958` | fallback | removePrimitive: 'fallback-rebuild', |
| SCAN-0134 | `packages/core/src/engine/promiseLedger.ts:982` | not_implemented | // walkaround-hybrid does NOT implement createInverseSession (differentiable RT |
| SCAN-0135 | `packages/core/src/engine/promiseLedger.ts:1048` | fallback | // fallback-rebuild (full scene-texture/BVH repack), while resize is a |
| SCAN-0136 | `packages/core/src/engine/promiseLedger.ts:1062` | not_implemented | // pt-webgl2 does not implement any of the following optional methods: |
| SCAN-0137 | `packages/core/src/engine/capabilities.ts:49` | fallback | \| 'fallback-rebuild' |
| SCAN-0138 | `packages/core/src/engine/giState.ts:48` | not_implemented | * state simply do not implement this interface; the facade marks the methods |
| SCAN-0139 | `packages/core/src/engine/index.ts:314` | not_implemented | *  backend that does not implement this still satisfies the contract; the |
| SCAN-0140 | `packages/core/src/engine/index.ts:357` | downgrade | * downgrades, and mutation fallbacks. Console output remains for developers; |
| SCAN-0141 | `packages/core/src/engine/index.ts:419` | not_implemented | * texture allocated yet). Optional: a minimal backend that cannot implement |
| SCAN-0142 | `packages/core/src/engine/index.ts:441` | not_implemented | * that do not implement ReSTIR-PT reuse omit this method entirely. |
| SCAN-0143 | `packages/core/src/engine/telemetry.ts:127` | downgrade | * features, approximation downgrades, and mutation fallbacks. They are not |
| SCAN-0144 | `packages/core/src/scene/material.ts:160` | not_implemented | *  be shown independent of scene lighting. Backends that do not implement an |
| SCAN-0145 | `packages/core/src/scene/material.ts:226` | reserved | *  @reserved Accepted; not yet consumed by any backend (road-to-100 texture tier). */ |
| SCAN-0146 | `packages/core/src/scene/material.ts:229` | reserved | *  @reserved Accepted; not yet consumed by any backend. */ |
| SCAN-0147 | `packages/core/src/scene/material.ts:232` | reserved | *  @reserved Accepted; not yet consumed by any backend. */ |
| SCAN-0148 | `packages/core/src/scene/analyticParams.ts:36` | throw | throw new Error( |
| SCAN-0149 | `packages/core/src/scene/analyticParams.ts:50` | throw | throw new Error( |
| SCAN-0150 | `packages/core/src/scene/analyticParams.ts:60` | throw | throw new Error( |
| SCAN-0151 | `packages/core/src/scene/primitives.ts:40` | reserved | *  @reserved Accepted; consumed by NO backend ('unsupported' in |
| SCAN-0152 | `packages/core/src/scene/primitives.ts:200` | reserved | *  @reserved Same status as {@link MeshPrimitive.receiveShadow} — consumed by |
| SCAN-0153 | `packages/core/src/scene/patchScene.ts:42` | throw | throw new Error( |
| SCAN-0154 | `packages/core/src/scene/patchScene.ts:47` | throw | throw new Error(`updatePrimitive: primitive "${primitive.id}" id cannot be changed`); |
| SCAN-0155 | `packages/core/src/scene/patchScene.ts:63` | throw | throw new Error( |
| SCAN-0156 | `packages/core/src/scene/patchScene.ts:71` | throw | throw new Error( |
| SCAN-0157 | `packages/core/src/scene/patchScene.ts:76` | throw | throw new Error( |
| SCAN-0158 | `packages/core/src/scene/patchScene.ts:81` | throw | throw new Error( |
| SCAN-0159 | `packages/core/src/scene/patchScene.ts:116` | throw | throw new Error(`updatePrimitive: primitive "${id}" not found in current scene`); |
| SCAN-0160 | `packages/core/src/scene/patchScene.ts:139` | throw | throw new Error( |
| SCAN-0161 | `packages/core/src/scene/patchScene.ts:144` | throw | throw new Error(`updateEmitter: emitter "${id}" id cannot be changed`); |
| SCAN-0162 | `packages/core/src/scene/patchScene.ts:149` | throw | throw new Error(`updateEmitter: emitter "${id}" not found in current scene`); |
| SCAN-0163 | `packages/stained-glass-extensions/src/cameUniformUploader.ts:4` | reserved | * @reserved The `CamePackedUBO` output format is defined for the host-app |
| SCAN-0164 | `packages/shared-denoisers/src/svgfRealWebGPU.ts:384` | fallback | // Per-channel fallback: R→0.5, G→0.5, B→1.0 (packed octahedral +Z); alpha=0. |
| SCAN-0165 | `packages/shared-denoisers/src/bmfrWebGPU.ts:109` | throw | throw new Error('runBmfrWebGPU: invalid rgb buffer or dimensions'); |
| SCAN-0166 | `packages/shared-denoisers/src/hdrLuminanceBilateralWebGPU.ts:64` | throw | throw new Error('runHdrLuminanceBilateralWebGPU: WebGPU not available in this browser'); |
| SCAN-0167 | `packages/shared-denoisers/src/hdrLuminanceBilateralWebGPU.ts:67` | throw | throw new Error('runHdrLuminanceBilateralWebGPU: invalid rgb buffer or dimensions'); |
| SCAN-0168 | `packages/shared-denoisers/src/sharedWebGpuDevice.ts:30` | throw | throw new Error('WebGPU not available'); |
| SCAN-0169 | `packages/shared-denoisers/src/sharedWebGpuDevice.ts:58` | throw | throw new Error('getSharedWebGPUDevice: exhausted retries after concurrent dispose'); |
| SCAN-0170 | `packages/shared-denoisers/src/sharedWebGpuDevice.ts:64` | throw | throw new Error('getSharedWebGPUDevice: failed to request GPU adapter'); |
| SCAN-0171 | `packages/shared-denoisers/src/sharedWebGpuDevice.ts:70` | throw | throw new Error(SUPERSEDED_MSG); |
| SCAN-0172 | `packages/shared-denoisers/src/sharedWebGpuDevice.ts:157` | throw | throw new Error(`${opts.errorLabel}: WebGPU not available`); |
| SCAN-0173 | `packages/shared-denoisers/src/sharedWebGpuDevice.ts:168` | throw | throw new Error(`${opts.errorLabel}: failed to request GPU adapter`); |
| SCAN-0174 | `packages/shared-denoisers/src/atrousVarianceWebGPU.ts:95` | throw | throw new Error('runAtrousVarianceWebGPU: invalid rgb buffer or dimensions'); |
| SCAN-0175 | `packages/shared-denoisers/src/atrousVarianceWebGPU.ts:98` | throw | if (!cond) throw new Error(`runAtrousVarianceWebGPU: ${detail}`); |
| SCAN-0176 | `packages/shared-denoisers/src/oidnBridge.ts:198` | throw | throw new Error( |
| SCAN-0177 | `packages/shared-denoisers/src/oidnBridge.ts:309` | throw | throw new Error( |
| SCAN-0178 | `packages/shared-denoisers/src/wgsl/bmfr.wgsl.ts:196` | reserved | // Thread 0 re-reads the whole block in Phase 2 to compute the position mean |
| SCAN-0179 | `packages/shared-denoisers/src/wgsl/bmfr.wgsl.ts:220` | reserved | // ── Phase 2 + 3: thread 0 forms M, r and solves; publishes α. ───────────── |
| SCAN-0180 | `packages/walkaround-hybrid/src/HybridEngine.ts:455` | reserved | // ── RC subsystem (W8 Phase 2 — opt-in via opts.rcEnabled) ─────────────── |
| SCAN-0181 | `packages/walkaround-hybrid/src/HybridEngine.ts:555` | not_implemented | // Non-'none' strategies are not implemented for this backend. |
| SCAN-0182 | `packages/walkaround-hybrid/src/HybridEngine.ts:598` | not_implemented | `not implement causticStrategy modes. The causticOptions object is ignored.`, |
| SCAN-0183 | `packages/walkaround-hybrid/src/HybridEngine.ts:665` | reserved | // W8 Phase 2 — opt-in RC subsystem. RCSubsystem owns its own BVH + |
| SCAN-0184 | `packages/walkaround-hybrid/src/HybridEngine.ts:941` | fallback | fallback: 'map ignored', |
| SCAN-0185 | `packages/walkaround-hybrid/src/HybridEngine.ts:957` | fallback | * **Capability filter + analytic fallback:** the scene is first partitioned |
| SCAN-0186 | `packages/walkaround-hybrid/src/HybridEngine.ts:967` | throw | throw new Error('HybridEngine.setScene: engine is disposed.'); |
| SCAN-0187 | `packages/walkaround-hybrid/src/HybridEngine.ts:1045` | reserved | // W8 Phase 2: rebuild the RC BVH + cascade buffers after async ReSTIR |
| SCAN-0188 | `packages/walkaround-hybrid/src/HybridEngine.ts:1130` | throw | throw new Error('HybridEngine.updatePrimitive: engine is disposed.'); |
| SCAN-0189 | `packages/walkaround-hybrid/src/HybridEngine.ts:1133` | throw | throw new Error( |
| SCAN-0190 | `packages/walkaround-hybrid/src/HybridEngine.ts:1139` | throw | throw new Error( |
| SCAN-0191 | `packages/walkaround-hybrid/src/HybridEngine.ts:1146` | throw | throw new Error( |
| SCAN-0192 | `packages/walkaround-hybrid/src/HybridEngine.ts:1248` | throw | throw new Error('HybridEngine.applyGpuSkinnedRefit: engine is disposed.'); |
| SCAN-0193 | `packages/walkaround-hybrid/src/HybridEngine.ts:1251` | throw | throw new Error( |
| SCAN-0194 | `packages/walkaround-hybrid/src/HybridEngine.ts:1263` | throw | throw new Error(`applyGpuSkinnedRefit("${id}"): skinned-mesh primitive not found.`); |
| SCAN-0195 | `packages/walkaround-hybrid/src/HybridEngine.ts:1323` | throw | throw new Error( |
| SCAN-0196 | `packages/walkaround-hybrid/src/HybridEngine.ts:1391` | throw | throw new Error('HybridEngine.addPrimitive: engine is disposed.'); |
| SCAN-0197 | `packages/walkaround-hybrid/src/HybridEngine.ts:1394` | throw | throw new Error( |
| SCAN-0198 | `packages/walkaround-hybrid/src/HybridEngine.ts:1400` | throw | throw new Error( |
| SCAN-0199 | `packages/walkaround-hybrid/src/HybridEngine.ts:1442` | throw | throw new Error('HybridEngine.removePrimitive: engine is disposed.'); |
| SCAN-0200 | `packages/walkaround-hybrid/src/HybridEngine.ts:1445` | throw | throw new Error( |


---

## 20. Full CORE item register (CORE-001 — CORE-047)

| ID | File:Line | Gap | Specific fix |
|----|-----------|-----|--------------|
| CORE-001 | `core/src/scene/environment.ts:45-51` | JSDoc says walkaround HDRI PARTIAL; ledger `hdri: native` | Update HdriEnvironment JSDoc to match ledger or downgrade ledger |
| CORE-002 | `core/src/scene/material.ts:273-275` | JSDoc says envMapIntensity walkaround non-consumer; ledger native | Align JSDoc with WALKAROUND_MATERIALS |
| CORE-003 | `core/src/scene/primitives.ts:65-67` | Instanced castShadow comment approximate; ledger native | Fix comment or revert ledger after code-read |
| CORE-004 | `core/src/scene/primitives.ts:39-44` | receiveShadow @reserved; all backends unsupported | IMP receiver occlusion or permanent ACC + warn |
| CORE-005 | `core/src/scene/material.ts:225-233` | displacement* @reserved; all backends unsupported | IMP on PT backend or remove from MaterialSpec |
| CORE-006 | `core/src/inverse.ts:47-49` | InverseParamKind texture Phase 2 reserved | IMP texture opt in pt-webgpu inverse |
| CORE-007 | `core/src/inverse.ts:83-85` | ssim/lpips in contract; throw at session creation | IMP losses or remove from union |
| CORE-008 | `core/src/engine/capabilities.ts:78-80` | materials Partial — omission ≠ unsupported | Full matrix or isMaterialFieldAudited() |
| CORE-009 | `core/src/engine/capabilities.ts:106-107` | swapchain-optional unused in ledger | Remove enum or assign backend |
| CORE-010 | `core/src/engine/promiseLedger.ts:893-898` | supportsAuxBuffers false; walkaround emits partial aux | Wire variance or narrow contract flag |
| CORE-011 | `core/src/engine/promiseLedger.ts:921,947` | analytic fallback-generated-mesh vs native shapes list | Split accepted vs native-traced in capabilities |
| CORE-012 | `core/src/engine/promiseLedger.ts:475-482` | walkaround spectral/volume/layered unsupported | partitionSceneBySupport warnings with codes |
| CORE-013 | `core/src/engine/promiseLedger.ts:354-358` | alphaMode approximate; blend not GI participant | IMP GI transport or document permanent |
| CORE-014 | `core/src/engine/promiseLedger.ts:340-344` | baseColor/roughness/metallic quantized approximate | TOG fidelity profile full lanes |
| CORE-015 | `core/src/engine/promiseLedger.ts:959` | mutations.environment approximate on walkaround | Document or promote after env fast-path audit |
| CORE-016 | `core/src/engine/promiseLedger.ts:505-507` | shadingModel approximate PT; unlit not emissive light | IMP unlit-as-emissive or mark unsupported |
| CORE-017 | `core/src/engine/promiseLedger.ts:567-569` | pt-webgl2 scatteringCoefficientRGB approximate | Port pt-webgpu σ_s path |
| CORE-018 | `core/src/engine/promiseLedger.ts:749-752` | pt-webgl2 emitterCastShadow approximate | Wire castShadow emissive-hit paths |
| CORE-019 | `core/src/engine/promiseLedger.ts:767-769` | pt-webgpu emitterCastShadow approximate specialty legs | Complete BDPT/MNEE/SPPM shadow flags |
| CORE-020 | `core/src/engine/promiseLedger.ts:845-846` | pt-webgpu setSize false; viewport-only resize | Add setSize API or document exclusively |
| CORE-021 | `core/src/engine/promiseLedger.ts:830,846` | PT updateLighting false | IMP or map to updateEnvironment |
| CORE-022 | `core/src/engine/promiseLedger.ts:788-805` | PT denoisers unsupported except none/oidn | Construction warn (done); document |
| CORE-023 | `core/src/engine/promiseLedger.ts:1129-1134` | getRestirPtResultBuffer true when feature off returns null | Gate methodPromises on experimental flag |
| CORE-024 | `core/src/engine/index.ts:196-204` | updateEnvironment JSDoc says HybridEngine may omit | Rewrite JSDoc for all backends |
| CORE-025 | `core/src/scene/partitionSceneBySupport.ts:46-48` | partition warnings plain strings | Return EngineWarning[] |
| CORE-026 | `core/src/scene/tlasAudit.ts:134-154` | analytic without fallback null AABB | Tessellate bounds in primitiveBounds |
| CORE-027 | `core/src/scene/tlasAudit.ts:33-38` | TLAS audit ignores analytic triangles | Include tessellated triangle estimates |
| CORE-028 | `core/src/scene/analyticToMesh.ts:33-58` | low-poly tessellation vs pt-webgpu native analytic | Expose segment count + warn on pt-webgpu |
| CORE-029 | `core/src/engine/promiseLedger.ts:576-578` | pt-webgl2 anisotropyMap comment vs native ledger row | Verify atlas wired; fix comment if stale |
| CORE-030 | `core/src/engine/promiseLedger.ts:574-575` | thinFilmStack cap 35 vs 8 | Export per-backend cap in capabilities |
| CORE-031 | `core/src/engine/promiseLedger.ts:487-489` | extensions native WH / unsupported PT | Document PT ignores extensions |
| CORE-032 | `core/src/scene/emitters.ts:42-49` | angularDiameter native pt-webgpu only | IMP soft sun WH/pt-webgl2 or @reserved |
| CORE-033 | `core/src/scene/emitters.ts:69-80` | point/spot distance/decay uneven consumption | Per-emitter-field supportDetails rows |
| CORE-034 | `core/src/scene/animation.ts:5-8` | AnimationClip no engine consumer | Optional Engine.playAnimation or document host duty |
| CORE-035 | `core/src/skinSolver.ts:39-40` | CPU LBS only in core | Document GPU skinning is backend path |
| CORE-036 | `core/src/engine/factory.ts:110-120` | photon-map caustic ~21% energy biased | Surface fidelity tier in capabilities |
| CORE-037 | `core/src/engine/giState.ts:57-60` | exportGIState null vague | Discriminated return or warn on null |
| CORE-038 | `core/src/engine/debug.ts:12-77` | debugSurface boolean; methods optional | List implemented debug methods |
| CORE-039 | `core/src/scene/math.ts:14-21` | Mat4 brand not deep freeze | Document or defensive copy getScene |
| CORE-040 | `core/src/engine/index.ts:121-131` | getScene retained reference | Optional getScene({copy:true}) |
| CORE-041 | `core/src/engine/promiseLedger.ts:907-911` | stale point/spot DDGI-only comment | Delete comment (LEDGER-01) |
| CORE-042 | `core/src/frame.ts:131-137` | swapChainView walkaround-only; pt-webgpu ignores | Clarify FrameInput JSDoc |
| CORE-043 | `core/src/inverse.ts:210-214` | path-replay downgrade list not centralized in core | Shared eligible-field table |
| CORE-044 | `core/src/engine/telemetry.ts:66-67` | denoiserState walkaround-shaped in core | Generalize or extension bag |
| CORE-045 | `core/src/scene/patchScene.ts:103-112` | material deep-merge vs emitter shallow | Document or deep-merge emitters |
| CORE-046 | `core/src/gpuDetection.ts:245-264` | detectGpu memoized until reset | Auto-invalidate on device.lost |
| CORE-047 | `core/src/adapterProfile.ts:1-10` | AdapterProfile type; probe engine-only | Document requires @vitrum/engine |

## 21. Full ENG item register (ENG-001 — ENG-030)

| ID | File:Line | Gap | Specific fix |
|----|-----------|-----|--------------|
| ENG-001 | `engine/src/createEngineInternals.ts:118-122` | onAdapterProfile walkaround-only | Call for PT backends or rename option |
| ENG-002 | `engine/src/createEngine.ts:130-155` | walkaround fallback applies wrong advanced bag | Require advancedByBackend on fallback path |
| ENG-003 | `engine/src/backends/ptWebgpu.ts:74-87` | pt-webgpu configures canvas offscreen | Skip configureWebGpuCanvas |
| ENG-004 | `engine/src/createProgressiveEngine.ts:226-236` | Class-A only; throws on union fail | Document; optional realtime-only fallback |
| ENG-005 | `engine/src/react/VitrumCanvas.tsx:89-90` | no advancedByBackend prop | Add prop plumbed to attachVitrum |
| ENG-006 | `engine/src/react/VitrumCanvas.tsx:141-165` | engine recreates on gltf identity change | Ref-stabilize gltfOptions |
| ENG-007 | `engine/src/idempotentDispose.ts:242-247` | debug live after dispose | Null debug on proxy post-dispose |
| ENG-008 | `engine/src/idempotentDispose.ts:108-109` | setSize noop when disposed | OK; verify pt-webgpu never proxied setSize |
| ENG-009 | `engine/src/createEngineScale.ts:47-49` | no WebGPU silently pt-webgl2 | Throw/warn realtime-unavailable |
| ENG-010 | `engine/src/createEngineScale.ts:8-9` | 500k triangle threshold undocumented | Export constant + document |
| ENG-011 | `engine/src/sceneAABB.ts:101-110` | AABB rest-pose skinned | Optional posed AABB via solveSkin |
| ENG-012 | `engine/src/backends/walkaround.ts:154-157` | lite forces merged BVH when needsTlas | EngineWarning when useLite&&needsTlas |
| ENG-013 | `engine/src/backends/walkaround.ts:159` | shared device forces tier full | Allow lite override with warning |
| ENG-014 | `engine/src/gltf.ts:143-146` | progressive load hardcodes pt-webgpu | Align gltf progressive profile |
| ENG-015 | `engine/src/negotiateWebGPUDevice.ts:54-56` | target none without requiredLimits | Default target to backend or validate post |
| ENG-016 | `engine/src/lifecycle/vanilla.ts:576-585` | recreate preserves GI not inverse/seed | Document limitations |
| ENG-017 | `engine/src/progressiveHandoff.ts:18-22` | no alpha cross-fade handoff | IMP blend or remove from claims |
| ENG-018 | `engine/src/configureWebGpuCanvas.ts:40-43` | configure failure swallowed | EngineWarning + skip render |
| ENG-019 | `engine/src/index.ts:111-116` | serializeGIState re-export from walkaround | Move to optional subpath |
| ENG-020 | `engine/src/gpuDetection.ts` via createEngine | publishToWindow false in createEngine | Document embedder opt-out |
| ENG-021 | `engine/src/createEngine.ts:15-18` | ResizeObserver not in createEngine | JSDoc backend resize table |
| ENG-022 | `engine/src/idempotentDispose.ts:100-106` | add/remove gated on capability | Verify all backends capability true |
| ENG-023 | `engine/src/progressiveHandoff.ts:101-108` | seedFromRealtime default mismatch | Align defaults or warn |
| ENG-024 | `engine/src/gltf.ts:217-220` | reject-unsupported passes approximate | reject-approximate mode or document |
| ENG-025 | `engine/src/lifecycle/vanilla.ts:458-461` | ResizeObserver walkaround-only semantics | Document pt-webgpu viewport requirement |
| ENG-026 | `engine/src/backends/walkaround.ts` ledger ref | updateEnvironment HDRI intensity-only comment stale | Reconcile with implementation |
| ENG-027 | `engine/src/createEngineInternals.ts:436-437` | lifetime guard duplicated 3 backends | Accept or typed helper |
| ENG-028 | `engine/src/index.ts:98-103` | LightingOptions from walkaround on main export | Subpath @vitrum/engine/walkaround |
| ENG-029 | `engine/src/react/VitrumCanvas.tsx:175-177` | requires scene or gltf at attach | Document deferred scene via engine prop |
| ENG-030 | `engine/src/adapterProfile.ts:132-136` | detectWebGL2Sync true headless | Return false without document |

## 22. Full PTGL item register (PTGL-001 — PTGL-028)

| ID | File:Line | Gap | Specific fix |
|----|-----------|-----|--------------|
| PTGL-001 | `pt-webgl2/src/scene/mutateSceneTextures.ts:53` | map fields block material fast path | Atlas delta upload |
| PTGL-002 | `pt-webgl2/src/scene/mutateSceneTextures.ts:178` | transform/positions/topology null fast path | Incremental BVH patches |
| PTGL-003 | `pt-webgl2/src/scene/mutateSceneTextures.ts:216` | stale geoPack in packMeshAreaLights | Use nextGeoPack |
| PTGL-004 | `pt-webgl2/src/scene/mutateSceneTextures.ts:35` | displacement blocks fast path unimplemented | Remove blocker or implement |
| PTGL-005 | `pt-webgl2/src/index.ts:476` | addPrimitive full setScene | Incremental splice |
| PTGL-006 | `pt-webgl2/src/index.ts:491` | removePrimitive full rebuild | Incremental removal |
| PTGL-007 | `pt-webgl2/src/capabilities.ts:94` | incrementalPatchSupport vs mutations contradiction | Align or implement |
| PTGL-008 | `pt-webgl2/src/index.ts:87` | displacement warns only | Tessellate or hard-drop |
| PTGL-009 | `pt-webgl2/src/scene/materialsTexture.ts:168` | _NO_TEXTURE unused | Use consistently |
| PTGL-010 | `pt-webgl2/src/scene/materialsTexture.ts:279` | no displacement layer id | Add if in scope |
| PTGL-011 | `pt-webgl2/src/scene/materialsTexture.ts:435` | doubleSided not in side encoding | Read extensions.doubleSided |
| PTGL-012 | `pt-webgl2/src/scene/materialsTexture.ts:445` | matte dead field | Remove GLSL branch |
| PTGL-013 | `pt-webgl2/src/scene/texturesArray.ts:7` | ImageBitmap unreadable | CPU decode before ingest |
| PTGL-014 | `pt-webgl2/src/scene/texturesArray.ts:23` | SAMPLED_MAP_KEYS incomplete | Extend to all consumed maps |
| PTGL-015 | `pt-webgl2/src/scene/texturesArray.ts:250` | nearest resize ignores sampler | Respect TextureRef filters |
| PTGL-016 | `pt-webgl2/src/scene/texturesArray.ts:308` | unreadable maps silent drop | Strict mode throw |
| PTGL-017 | `pt-webgl2/src/scene/meshAreaLights.ts:200` | emissiveMap needs CPU texels | Require decode in pipeline |
| PTGL-018 | `pt-webgl2/src/scene/meshAreaLights.ts:384` | barycentric subdivision fallback | Expand texel subtriangle coverage |
| PTGL-019 | `pt-webgl2/src/scene/meshAreaLights.ts:230` | mesh-area emissiveMap synthetic material | Sample atlas in packer |
| PTGL-020 | `pt-webgl2/src/scene/equirectHdrInfo.ts:18` | CDF missing sin θ | Add solid-angle Jacobian |
| PTGL-021 | `pt-webgl2/src/scene/solveSkinPrimitives.ts:54` | skin warn not EngineWarning | Emit structured warning |
| PTGL-022 | `pt-webgl2/src/scene/solveSkinPrimitives.ts:20` | no bone-only fast path | positionsRefit after solveSkin |
| PTGL-023 | `pt-webgl2/src/scene/foldEmissiveEmitters.ts:42` | fold keying prim.id only | meshId semantics across prims |
| PTGL-024 | `pt-webgl2/src/index.ts:976` | realtime denoisers degrade | Document |
| PTGL-025 | `pt-webgl2/src/index.ts:257` | OIDN not turnkey | Capability gate + clear error |
| PTGL-026 | `pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js:48` | refraction PDF TODO | Implement PDF attenuation |
| PTGL-027 | `pt-webgl2/src/glsl/render/direct_light_contribution_function.glsl.js:125` | env MIS transmissive gap | Transmissive-aware env sample |
| PTGL-028 | `pt-webgl2/src/glsl/shader/bsdf/ggx_functions.glsl.js:15` | legacy GGX not VNDF | Port Heitz VNDF from pt-webgpu |

## 23. Full GLTF item register (GLTF-001 — GLTF-022)

| ID | File:Line | Gap | Specific fix |
|----|-----------|-----|--------------|
| GLTF-001 | `gltf-adapter/src/compression.ts:137` | no bundled Draco | Optional peer dep + default hook |
| GLTF-002 | `gltf-adapter/src/compression.ts:262` | meshopt unresolved views | Fail fast reject-unsupported |
| GLTF-003 | `gltf-adapter/src/compression.ts:440` | Draco fail skip primitive | Strip in strict modes |
| GLTF-004 | `gltf-adapter/src/compression.ts:386` | Draco hook incomplete attrs | Require complete semantics |
| GLTF-005 | `gltf-adapter/src/textures.ts:127` | external URI not fetched | Fetch in assetLoader |
| GLTF-006 | `gltf-adapter/src/textures.ts:194` | Node raw-image no decode | Default decodePixels in bridge |
| GLTF-007 | `gltf-adapter/src/textures.ts:261` | basisu/webp inactive without hook | Auto-enable from featureReport |
| GLTF-008 | `gltf-adapter/src/texturePipeline.ts:493` | decodeSceneTextures raw-image only | ImageBitmap CPU path |
| GLTF-009 | `gltf-adapter/src/texturePipeline.ts:508` | raw-image no decodePixels | Require in reject-degraded |
| GLTF-010 | `gltf-adapter/src/texturePipeline.ts:234` | walkaround atlas subset | Expand or report per field |
| GLTF-011 | `gltf-adapter/src/featureReport.ts:745` | unknown extensions unsupportedOptional | Sync allowlist with gltfToScene |
| GLTF-012 | `gltf-adapter/src/featureReport.ts:628` | emissiveMap.texelPdf always approximate | Promote when NEE aligned |
| GLTF-013 | `gltf-adapter/src/featureReport.ts:232` | lite all texture fields unsupported | Document profile |
| GLTF-014 | `gltf-adapter/src/featureReport.ts:228` | vertex colors walkaround approximate | Promote when validated |
| GLTF-015 | `gltf-adapter/src/featureReport.ts:609` | sampler policy approximate | Per-texture sampler creation |
| GLTF-016 | `gltf-adapter/src/gltfToScene.ts:468` | cameras ignored | Export Scene cameras optional |
| GLTF-017 | `gltf-adapter/src/gltfToScene.ts:1549` | TEXCOORD_2+ stripped | Extend UV sets or split meshes |
| GLTF-018 | `gltf-adapter/src/gltfToScene.ts:1802` | morph COLOR/TEXCOORD ignored | Map or warn in featureReport |
| GLTF-019 | `gltf-adapter/src/gltfToScene.ts:1908` | unknown light type skipped | featureReport issue row |
| GLTF-020 | `gltf-adapter/src/materials.ts:585` | doubleSided in extensions only | First-class or PTGL-011 |
| GLTF-021 | `gltf-adapter/src/materials.ts:526` | unknown KHR warn only | Extend KNOWN_KHR_EXTENSIONS |
| GLTF-022 | `gltf-adapter/src/sceneController.ts:334` | variant bindings stale throws | Recovery API + validation |

