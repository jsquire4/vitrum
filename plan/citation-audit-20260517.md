# Citation + Prior-Work Audit — 2026-05-17

> Read-only audit of attribution coverage across the vitrum monorepo, against the
> project rule (`CLAUDE.md`): _"Every algorithm has provenance. Citation goes in
> three places: source code comment at the implementation site, package README,
> and the project-level CREDITS.md."_

This document does **not** modify any source files. It catalogues what is
missing so a follow-up commit can backfill the citations.

## Status update — 2026-05-18

Both structural gaps called out below have been closed:

- **"7 of 10 packages have no README at all"** → fixed. Verified by directory listing on 2026-05-18: all 11 packages (10 original + `@vitrum/scene-lighting` added afterward) have `README.md` files. Landed via `chore/missing-package-readmes` (`ec58f48`, commit `0ab2d54`) + the W13 README-audit pass (`fc882f6`).
- **22 algorithms with missing CREDITS.md citations** → backfilled by `docs(credits): add missing citations + mark unimplemented techniques as candidates` (commit `70f53cd`, merged via `6860e5a`).

The per-algorithm gap tables below describe the 2026-05-17 audit-time state. Treat them as historical; a fresh re-audit is needed before claiming the citation rule is now satisfied algorithm-by-algorithm.

---


## Scope

- `CREDITS.md`, `README.md`, all `packages/*/README.md` files, all
  `packages/*/package.json` license declarations, repo-root `LICENSE`.
- All `.ts` / `.wgsl.ts` files under `packages/` searched for author-name
  citations (`Majercik`, `Sannikov`, `Bitterli`, `Schied`, `Müller`, `Heitz`,
  `Wald`, `Möller`, `Jiménez`, `Burley`, `Veach`, `Wilkie`, `Jakob`, `Hanika`,
  `Schlick`, `Kulla`, `Henyey`, `Cauchy`, `Karis`, `Karras`, `Dammertz`,
  `Sobol`, `Cigolle`, `Chaitanya`, `Ronneberger`, `Shirley`).
- Sibling fork at `~/projects/three-gpu-pathtracer/` (license file presence
  and compatibility only — out-of-repo).

## Verification protocol

Every line in the matrix below was verified by opening the cited file at the
indicated line number with the Read tool and confirming the citation text
matches (or is absent). Sub-agent claims were not used. The grep used to seed
the file list is recorded inline so each row is reproducible.

---

## 1. Coverage matrix — algorithms × citation site

Legend:
- ✓ — full citation (author, year, paper/source) present
- ~ — partial citation (technique named but no author/year/paper)
- ✗ — absent
- N/A — algorithm is not actually implemented (see §3 Stale)

| # | Algorithm | Implemented at (representative site) | CREDITS.md | Package README | Source-site comment |
|---|---|---|---|---|---|
| 1 | **DDGI** (Majercik 2019) | `packages/walkaround-hybrid/src/ddgi/{DDGI,probeUpdatePass}.ts` + `ddgi/wgsl/*.wgsl.ts` | ✓ (§Real-time GI) | ~ (acronym only, no paper) | ~ (file headers don't cite; inline citation at `probeUpdatePass.ts:504,665` + `probeUpdateBlend.wgsl.ts:122`) |
| 2 | **Radiance Cascades** (Sannikov 2023/2024) | `packages/walkaround-hybrid/src/rc/cascadePyramid.ts` + `rc/wgsl/cascadeMerge.wgsl.ts` | ✓ but year mismatch (CREDITS says 2024; source says 2023) | ~ (RC acronym only) | ~ (`cascadePyramid.ts:36` cites; `cascadeMerge.wgsl.ts:42` cites; `cascadeDispatch.ts` / `giReceiver.ts` / `applyDDGIShading.ts` / `walkaroundDiffuseLighting.ts` headers do **not**) |
| 3 | **ReSTIR DI** (Bitterli 2020) | `packages/walkaround-hybrid/src/shaders/{ris,temporal,spatial}.wgsl.ts` | ✓ | ~ (ReSTIR DI named, no paper link in README) | ✗ — none of the three shader file headers cite the paper; only an inline `Bitterli 2020 §4.3` at `spatial.wgsl.ts:77` |
| 4 | **ReSTIR-GI** (Ouyang 2021 OR Majercik 2021) | `packages/walkaround-hybrid/src/shaders/{risGi,temporalGi,spatialGi}.wgsl.ts` + `indirectCombine.wgsl.ts` + `indirectTemporalAccum.wgsl.ts` | ✗ — README mentions "Ouyang 2021" in CLAUDE.md context but CREDITS.md has **no entry** | ✗ (not in walkaround-hybrid README) | ~ source headers cite "Majercik et al. 2021 SIGGRAPH §4.2" (`risGi.wgsl.ts:4`), not Ouyang — **need to resolve which paper is the actual source** |
| 5 | **GTAO** (Jiménez 2016) | `packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts` + `gtaoUpsample.wgsl.ts` | ✗ | ✗ (not mentioned in walkaround-hybrid README) | ✓ (`gtao.wgsl.ts:4-7` cites paper + XeGTAO reference) |
| 6 | **SVGF** (Schied 2017) — variance-from-moments + reprojection + 7×7 fallback | `packages/shared-denoisers/src/svgfRealWebGPU.ts` + `wgsl/svgf{VarianceFromMoments,Reprojection,7x7SpatialFallback}.wgsl.ts` | ✓ | ✗ (shared-denoisers has **no README**) | ✓ (all four files cite Schied 2017 in headers) |
| 7 | **SVGF albedo demodulation** (Schied 2017 §4.1) — shipped on `walkaround-hybrid`'s indirect channel | `packages/walkaround-hybrid/src/shaders/{shade,indirectCombine}.wgsl.ts` + `shared-denoisers/src/atrousVarianceWebGPU.ts` | ✓ (covered by SVGF entry) | ✗ (walkaround-hybrid README never names SVGF) | ✓ (inline `Schied 2017 §4.1` at multiple sites) |
| 8 | **À-trous wavelet** (Dammertz 2010) | `packages/shared-denoisers/src/wgsl/{atrous,atrousKernel,atrousVariance,spatialFilter}.wgsl.ts` | ✓ | ✗ (no shared-denoisers README) | ✓ (file headers cite Dammertz 2010 HPG) |
| 9 | **PPG** (Müller 2017) | `packages/walkaround-hybrid/src/ppg/{dTree,sTree,ppgGuide.wgsl,ppgUpdate.wgsl,types,ppgConstants}.ts` | ✓ | ✗ (no PPG mention in walkaround-hybrid README) | ✓ (every file header cites Müller 2017) |
| 10 | **GGX BRDF + Smith G + Schlick Fresnel** (Walter 2007 + Trowbridge-Reitz 1975 + Schlick 1994) | `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:432–478` (`evalGGX`, `distributionGGX`, `geometrySchlickGGX`, `fresnelSchlick`) + `pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts` | Schlick ✓; **Walter/Trowbridge-Reitz GGX ✗** | ✗ | ~ Schlick named inline; no Walter or Trowbridge-Reitz citation |
| 11 | **GGX VNDF sampling** (Heitz 2018) | `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:975–1024` (`sampleGgxVndfTangent`, `glossyReflectionSample`) | ✗ | ✗ (pt-webgpu README — none) | ✓ (in WGSL: "Heitz, E. 'Sampling the GGX Distribution of Visible Normals.' JCGT 7(4), 2018."), also `pt-webgpu/__tests__/energyConservation.test.ts:445–451` |
| 12 | **BVH SAH binned build** (Wald 2007) | `packages/pt-webgpu/src/scene/buildCpuBvh.ts` (CPU); `packages/shared-bvh/src/bvhCommon.ts` (wraps three-mesh-bvh's SAH) | ✗ | ✗ (no shared-bvh README) | ✓ (`buildCpuBvh.ts:2-3` cites Wald 2007 IEEE Symposium on Interactive Ray Tracing) + walkaround-hybrid's `common.wgsl.ts:553,651` cites "Wald 2007 / PBR4e §7.3.3" for ordered traversal |
| 13 | **Möller-Trumbore triangle intersection** (1997) | `packages/pt-webgpu/src/wgsl/common.wgsl.ts:75` (`intersectTriangle`) + `walkaround-hybrid/src/rc/wgsl/probeRayCast.wgsl.ts:119,268` (uses the epsilon) | ✗ (only mentioned in §"Citing vitrum" as a contribution; no Möller-Trumbore entry under §Path tracing) | ✗ | ~ algorithm name appears in `engine/src/{createEngine,sceneAABB}.ts` comments + `rc/wgsl/probeRayCast.wgsl.ts:119,268`; no formal citation header anywhere |
| 14 | **Disney BSDF** (Burley 2012) | `packages/pt-webgl/` (via fork — implementation lives upstream) and `walkaround-hybrid`'s GGX-only material model is a **subset** | ✓ | ✗ | ✗ (`walkaround-hybrid/src/shaders/common.wgsl.ts:432` says "GGX BRDF (simplified Lambertian + GGX specular)" with no Disney attribution) |
| 15 | **MIS power heuristic** (Veach 1997) | `packages/shared-samplers/src/bdptMIS.ts`; `pt-webgl/src/forkUniformBridge.ts:36` | ✓ | ✗ | ✓ (`bdptMIS.ts:13-17` cites Veach 1997 + PBR4e) |
| 16 | **BDPT** (Lafortune-Willems 1993 + Veach §10.3) | `packages/shared-samplers/src/{bdptMIS,bdptVertex}.ts`; bridged via `pt-webgl/src/forkUniformBridge.ts` | ✓ (Lafortune+Willems) | ✗ (pt-webgl README doesn't mention BDPT) | ✓ |
| 17 | **Sobol low-discrepancy** (Sobol 1967) — runtime use via fork; not re-implemented in-tree | upstream `three-gpu-pathtracer` | ✓ | ✗ | N/A (in upstream fork) |
| 18 | **Hammersley + radical-inverse VdC** (Hammersley 1960 / van der Corput) | `packages/shared-samplers/src/wgsl/hammersley.wgsl.ts:7–20` | ✗ (not under §Path tracing / §Real-time GI) | ✗ (shared-samplers has no README) | ✗ (file header names "Hammersley sequence" without author/year) |
| 19 | **Octahedral encode/decode** (Cigolle et al. JCGT 2014) | `packages/shared-samplers/src/wgsl/octahedralCore.wgsl.ts:2,12` + DDGI atlas in `shared-bvh` | ✗ | ✗ | ✓ (header + inline cite Cigolle 2014 JCGT) |
| 20 | **Halton SO(3) probe rotation** (Marques 2013 "Spherical Fibonacci"? Hatfield/Halton-based for SO(3)) | `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:655–718` | ✗ | ✗ | ~ comment mentions "Halton-base-{2,3,5}" with no formal citation; references Shoemake 1992 at line 665 for the SO(3) parameterisation |
| 21 | **Shoemake SO(3) uniform rotation** (Shoemake 1992) | `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:665` | ✗ | ✗ | ✓ (inline `Shoemake 1992` cite) |
| 22 | **Hero-wavelength spectral PT** (Wilkie 2014) | `packages/shared-samplers/src/{cieCmf,wavelengthSampling,jakobHanika}.ts` + `pt-webgl/src/forkUniformBridge.ts:150` | ✓ (cited as 2014) | ~ (README §"What's novel" cites "Wilkie et al. 2015" — **year mismatch**) | ~ Year is "2014" in `core/src/scene.ts:45,221`, "2015" in `forkUniformBridge.ts:150` and `spectral.test.ts:469`. Paper is EGSR 2014, CGF v33 — pick one. |
| 23 | **Jakob-Hanika spectral upsampling** (2019) | `packages/shared-samplers/src/jakobHanika.ts` | ✓ | ✗ | ✓ (file header is exemplary) |
| 24 | **Cauchy IOR** (Cauchy 1830) | `packages/shared-samplers/src/cauchyIor.ts` | ✓ | ✗ | ~ file header derives formula but never names "Cauchy 1830" |
| 25 | **Henyey-Greenstein phase** (Henyey-Greenstein 1941) | `packages/shared-samplers/src/hgPhase.ts` | ✓ | ✗ | ~ formula given; "Henyey-Greenstein" named but no 1941 / "Diffuse radiation in the Galaxy" citation in the file |
| 26 | **Equi-angular volume sampling** (Kulla-Conty 2012) | `packages/shared-samplers/src/equiAngular.ts` | ✓ — but CREDITS lists "Kulla, Fajardo 2012" while source cites "Kulla & Conty 2012" — **author mismatch** | ✗ | ✓ (header cites Kulla & Conty 2012) |
| 27 | **Light tree power-weighted** (Shirley-Smits-Wang-Zimmerman 1996; median-split, not Estévez-Kulla 2018) | `packages/shared-samplers/src/lightTree.ts` | ✗ | ✗ | ✓ (`lightTree.ts:10,28-30` cites Shirley 1996) |
| 28 | **U-Net + recurrent denoising autoencoder** (Ronneberger 2015 + Chaitanya 2017) | `packages/walkaround-hybrid/src/neural/{unetArchitecture,inputPacker,InferenceGraph}.ts` — scaffold only, not wired (see project memory) | ✗ | ✗ | ✓ (`unetArchitecture.ts:4-10` cites both) |
| 29 | **Temporal AA / history clamp** (Karis 2014 "High Quality Temporal Supersampling") | `packages/shared-denoisers/src/wgsl/temporalAccum.wgsl.ts:20-22` | ✗ | ✗ | ✓ (file header cites Karis 2014 UE4) |
| 30 | **Bilateral filter** (Tomasi-Manduchi 1998) | `packages/shared-denoisers/src/wgsl/hdrLuminanceBilateral.wgsl.ts:8` | ✗ | ✗ | ~ ("Tomasi & Manduchi" named without year/paper) |
| 31 | **Welford running variance** (Welford 1962) | `packages/shared-denoisers/src/wgsl/welfordVariance.wgsl.ts`; used by SVGF + ReSTIR variance | ✗ | ✗ | ✗ (uses Welford's name as identifier only; no formal citation) |
| 32 | **ACES filmic tone map** (Narkowicz 2015) | `packages/walkaround-hybrid/src/shaders/composite.wgsl.ts:48`; `rc/giReceiver.ts:31` | ✓ (CREDITS lists AMPAS + Narkowicz + Hill) | ✗ | ✓ inline (cites "Narkowicz 2015 RRT+ODT fit") |
| 33 | **Manifold NEE for caustics** (Hanika-Droske-Fascione 2015) | bridged into `pt-webgl/src/forkUniformBridge.ts` (`'manifold-nee'` strategy → fork uniforms); contract docs in `core/src/engine.ts:84,354,369` | ✗ (CREDITS does not list manifold NEE) | ✗ | ✓ (`core/src/engine.ts:84,369` cite Hanika et al. 2015) |
| 34 | **OIDN (Open Image Denoise)** (Intel 2018+ — Áfra et al.) | `packages/shared-denoisers/src/oidnBridge.ts` (ONNX bridge) | ✗ | ✗ | ~ ("Intel OIDN", "Open Image Denoise" named with no formal citation or licence note for the bundled model) |
| 35 | **HBAO / horizon-based AO** (Bavoil-Sainz 2008) — explicitly **rejected** in favour of GTAO; comment mentions Bavoil as the prior approach | `packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts:39` | ✗ | ✗ | ~ ("Bavoil-style HBAO" named in source for historical context) |
| 36 | **Practical hash for path tracing** (PCG/xxHash style) | used in shaders via `pcg_hash`, `wang_hash` etc. — see `walkaround-hybrid/src/shaders/common.wgsl.ts` | ✗ | ✗ | ✗ (functions present, citation absent — PCG is O'Neill 2014) |

---

## 2. Missing-citation list (one row per implementation site)

Sites below ship in the codebase but lack a citation in **all three** of {CREDITS.md, package README, source-site comment}. These are the highest-priority backfills.

| Site | Algorithm | What is missing |
|---|---|---|
| `packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts` + `gtaoUpsample.wgsl.ts` | GTAO (Jiménez 2016) | **CREDITS.md** entry (source ✓, README ✗); add a "Real-time ambient occlusion" subsection |
| `packages/walkaround-hybrid/src/shaders/{risGi,temporalGi,spatialGi}.wgsl.ts` + `indirectCombine.wgsl.ts` + `indirectTemporalAccum.wgsl.ts` | ReSTIR-GI | **CREDITS.md** entry — need to lock the cited paper (Ouyang et al. "ReSTIR GI: Path Resampling for Real-Time Path Tracing" HPG 2021 vs. Majercik 2021 "Scaling Probe-Based Real-Time Dynamic GI..." which the source headers point to). README §"What's novel" claims ReSTIR-GI exists; CREDITS is silent. |
| `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:975–1024` | GGX VNDF (Heitz 2018) | **CREDITS.md** entry — VNDF is the lone sampler that makes the WebGPU PT physically sound; the test file `energyConservation.test.ts:445–451` carries the citation but it is absent from CREDITS and README |
| `packages/pt-webgpu/src/scene/buildCpuBvh.ts` + `walkaround-hybrid/src/shaders/common.wgsl.ts:553,651` | BVH SAH-binned construction (Wald 2007) | **CREDITS.md** entry under §"Path tracing" or new §"Acceleration structures" |
| `packages/pt-webgpu/src/wgsl/common.wgsl.ts:75` (`intersectTriangle`) | Möller-Trumbore (1997) | **CREDITS.md** entry — the algorithm is named in `engine/src/createEngine.ts` and `sceneAABB.ts` comments but is not in CREDITS as a formal entry; the implementation site has no header citation either |
| `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:432–478` | GGX NDF / Smith G (Walter 2007 + Trowbridge-Reitz 1975) | The function `evalGGX` is a Disney-flavoured mix-around-Lambertian; no Walter 2007 or Trowbridge-Reitz citation; CREDITS lists Schlick but not GGX provenance |
| `packages/shared-samplers/src/wgsl/hammersley.wgsl.ts` | Hammersley (1960) + radical-inverse VdC | Neither in CREDITS, README, nor source header |
| `packages/shared-samplers/src/wgsl/octahedralCore.wgsl.ts` | Octahedral normal encoding (Cigolle 2014) | Source ✓; **CREDITS.md** entry absent |
| `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:665` | Shoemake SO(3) uniform rotation (Shoemake 1992) | Source ✓; **CREDITS.md** entry absent (algorithm is load-bearing for DDGI ray decorrelation) |
| `packages/shared-samplers/src/lightTree.ts` | Shirley-Smits-Wang-Zimmerman 1996 power-weighted light tree | Source ✓; **CREDITS.md** entry absent |
| `packages/walkaround-hybrid/src/neural/unetArchitecture.ts` + `inputPacker.ts` + `InferenceGraph.ts` | U-Net (Ronneberger 2015) + recurrent denoising autoencoder (Chaitanya 2017) | Source ✓; **CREDITS.md** entry absent (even though the neural denoiser is scaffold only, the architecture choice still requires attribution — Chaitanya's input-buffer composition is directly copied) |
| `packages/shared-denoisers/src/wgsl/temporalAccum.wgsl.ts` | Karis 2014 "High-Quality Temporal Supersampling" (UE4) | Source ✓; **CREDITS.md** entry absent |
| `packages/shared-denoisers/src/wgsl/hdrLuminanceBilateral.wgsl.ts` | Tomasi-Manduchi 1998 bilateral filter | Source ~ (names without year); **CREDITS.md** entry absent |
| `packages/shared-denoisers/src/wgsl/welfordVariance.wgsl.ts` | Welford 1962 running variance | **All three** ✗ |
| `packages/shared-denoisers/src/oidnBridge.ts` | Intel OIDN (Áfra et al.) | Source ~ (named); **CREDITS.md** entry absent. **Licensing concern** — if vitrum ships a default OIDN ONNX model URL, Intel's Apache-2.0 model license must be acknowledged separately from MIT code license. |
| `packages/core/src/engine.ts:84,354,369` (option contract); bridged in `pt-webgl/src/forkUniformBridge.ts` | Manifold NEE (Hanika-Droske-Fascione 2015) | Source ✓; **CREDITS.md** entry absent. README §"What's novel" doesn't surface manifold NEE either — only "normalMap-perturbed NEE shadow rays" is named. |
| `packages/walkaround-hybrid/src/shaders/{ris,temporal,spatial}.wgsl.ts` headers | ReSTIR DI (Bitterli 2020) | Source-site ✗ in the three file headers (only inline at `spatial.wgsl.ts:77`); add header comment at top of each shader. CREDITS ✓, README ~ (acronym only). |
| `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdate{Rays,Blend,Border}.wgsl.ts` headers | DDGI (Majercik 2019) | The compute-shader strings carry the algorithm but the file headers do not cite Majercik. CREDITS ✓; README ~ (acronym only — needs paper link). |
| `packages/walkaround-hybrid/src/rc/{cascadeDispatch,giReceiver,applyDDGIShading,walkaroundDiffuseLighting}.ts` | RC (Sannikov) | Implementation files do not cite the paper; only `cascadePyramid.ts` and `cascadeMerge.wgsl.ts` do. README ~ (acronym only). |
| **All package READMEs** (where they exist): `pt-webgl`, `pt-webgpu`, `walkaround-hybrid` | All shipped algorithms | None of the three existing package READMEs carries an algorithmic-reference section. The project rule explicitly requires per-package README citation. |
| **Missing READMEs entirely**: `core`, `dev`, `engine`, `shared-bvh`, `shared-denoisers`, `shared-samplers`, `three-bindings` | All shipped algorithms in those packages | 7 of 10 packages have no README at all — `shared-denoisers` (SVGF, à-trous, OIDN, bilateral, Karis temporal, Welford), `shared-samplers` (Veach BDPT-MIS, Wilkie hero-wavelength, Jakob-Hanika, Cauchy, HG, equi-angular, Shirley light tree, Hammersley, Cigolle octahedral), and `shared-bvh` (Wald SAH wrap + Cigolle octahedral atlas) are the worst gaps because they are the algorithmic core of the engine. |

---

## 3. Stale-citation list (in CREDITS.md but not implemented)

Verified by grepping the codebase for the named technique. Each entry was confirmed absent by running the search and reading any near-misses.

| CREDITS.md line | Entry | Reality |
|---|---|---|
| L49 §Denoising | **BMFR (Blockwise Multi-Order Feature Regression)** — Koskela et al. 2019 | Explicitly **not implemented** — `packages/shared-denoisers/package.json:5` says _"BMFR: candidate only — not implemented in this package."_ and `index.ts:2` confirms _"no BMFR module is exported from this package."_ Stale citation. |
| L63 §Caustic methods | **ReSTIR BDPT** — Hedstrom et al. 2025 | No source file contains "Hedstrom" or any ReSTIR-BDPT implementation. The roadmap mentions it as future work. Stale citation. |
| L64 §Caustic methods | **Vertex Connection and Merging (VCM)** — Georgiev et al. 2012 | No source file contains "VCM", "Georgiev", or "Vertex Connection and Merging" (verified via Grep). Stale citation. |

Action: either move these to a `## Roadmap / future work` subsection of CREDITS.md (cleanest), or remove until the corresponding code lands. Mixing aspirational citations with shipped ones is the failure mode the audit rule guards against.

---

## 4. Inconsistency findings (cited everywhere — but inconsistently)

| Issue | Locations | Resolution |
|---|---|---|
| **Sannikov 2023 vs. 2024** | CREDITS.md:32 + README.md:130 say 2024; `packages/walkaround-hybrid/src/rc/cascadePyramid.ts:36` + `cascadeMerge.wgsl.ts:42` + `__tests__/cascadeMergeWeights.test.ts:18` say 2023 | Sannikov's "Radiance Cascades" technical report was released 2023; the SIGGRAPH 2024 publication is the citeable peer-reviewed venue. Either pick one (recommend 2024 for the SIGGRAPH paper) and propagate it everywhere, or cite both. The title "Radiance Cascades: A Novel High-Resolution Formal Solution..." in CREDITS does not match the SIGGRAPH paper title — verify the title against the cited year. |
| **Wilkie 2014 vs. 2015** | `core/src/scene.ts:45,221` say 2014; CREDITS.md:43 says 2014; README.md:120,133 say 2015; `pt-webgl/src/forkUniformBridge.ts:150` says 2015; `shared-samplers/__tests__/spectral.test.ts:469` says 2015 | "Hero Wavelength Spectral Sampling" was published at EGSR 2014 (CGF v33 #4). Use **2014** consistently; remove 2015 references. |
| **Kulla, Fajardo (CREDITS) vs. Kulla & Conty (source)** | CREDITS.md:38: "Kulla, Marcos Fajardo"; `shared-samplers/src/equiAngular.ts:11,30` + `__tests__/pdfNormalization.test.ts:17`: "Kulla & Conty 2012" | The paper "Importance Sampling Techniques for Path Tracing in Participating Media" Kulla & Fajardo, EGSR 2012 is the cited work; "Conty" is likely a confusion with Christopher Kulla / Alejandro Conty's separate Sony Imageworks papers. **Verify the actual paper authorship before fixing** — if the source-side spelling is wrong, fix the source; if CREDITS is wrong, fix CREDITS. |
| **PBR4e citations** | `shared-samplers/__tests__/bdptVeachFull.test.ts:17,228`; `bdptMIS.ts:16`; `walkaround-hybrid/src/shaders/common.wgsl.ts:553,651` | "Pharr, Jakob, Humphreys, _Physically Based Rendering: From Theory to Implementation_ (4th edition), 2023" is referenced but not in CREDITS.md as a textbook source. Add to a §"Textbook references" subsection — it is a foundational reference cited across packages. |

---

## 5. Asset / model licensing findings

| Asset | Current attribution | Concern |
|---|---|---|
| OIDN ONNX model — `packages/shared-denoisers/src/oidnBridge.ts` | "Intel OIDN" named in source comments only; no licence statement | If vitrum ships with a default model URL or bundled weights, Intel OIDN models are typically **Apache-2.0** while OIDN code is **Apache-2.0**. The model file licensing is *not* MIT-compatible-by-default — they are *additive*. CREDITS.md needs an "ONNX / model weights" subsection. Currently CREDITS.md does not mention OIDN at all. |
| HDRIs + textures | CREDITS.md §"Asset attribution" cites Polyhaven CC0 | Adequate for the present scope. |

---

## 6. License hygiene findings

| Finding | Detail | Severity |
|---|---|---|
| Root LICENSE present | `/home/jsquire4/projects/vitrum/LICENSE` — MIT, Copyright 2026 jsquire4 | ✓ OK |
| Per-package LICENSE files | **None** of `packages/*/` carries a LICENSE file (verified `ls packages/*/ \| grep -i license`) | ✗ **Should add** — each package's `package.json` declares `"license": "MIT"` but at npm publish time the package tarball is what consumers see; npm convention is each package carries its own LICENSE. Currently this is partially mitigated by the `"private": true` flag on every package, but as soon as the user removes `private` (planned for public release), the missing LICENSE files become a publication blocker. |
| `pt-webgl` fork pin | `pt-webgl/package.json` pins `three-gpu-pathtracer: file:../../../three-gpu-pathtracer` (sibling) — fork is MIT (verified at `~/projects/three-gpu-pathtracer/LICENSE`) | ✓ License compatible. README correctly flags that `private: true` must stay until the file-pin is replaced with a publishable artifact. |
| `three-mesh-bvh` dependency | `node_modules/three-mesh-bvh/LICENSE` — MIT, Copyright Garrett Johnson 2018 | ✓ License compatible |
| `postprocessing` (Zlib) | Listed in CREDITS as Zlib licence | Zlib is permissive but the wording differs from MIT; if redistributing postprocessing's source, attach the original Zlib licence. (vitrum does not redistribute postprocessing source — uses npm dependency — so this is observational only.) |
| Citation rule violation surface | The CLAUDE.md citation rule states three required sites. Of the 36 algorithms catalogued in §1, **22 have at least one citation site missing** — see §2 for the actionable list. | ✗ Active rule violation; backfill is in scope before public-alpha. |

---

## 7. Prioritized fix order

Numbered by tier — lower number = ship first. Each tier is a single PR's worth of work.

### Tier 1 — CREDITS.md backfill (mechanical; no source-code reads needed)

1. Add **§Real-time GI** entries: ReSTIR-GI (pick paper — see §4 issue list), GTAO (Jiménez 2016).
2. Add **§Path tracing / acceleration** entries: GGX NDF (Walter 2007), GGX VNDF sampling (Heitz 2018), Smith G (Smith 1967 / Walter 2007 §3), Möller-Trumbore (1997), Wald 2007 SAH BVH, Shoemake 1992 SO(3) rotation.
3. Add **§Sampling primitives** entries: Hammersley (1960), Cigolle 2014 octahedral, Shirley 1996 light tree, Welford 1962 variance, PCG (O'Neill 2014) hash.
4. Add **§Denoising** entries: Karis 2014 TAA history clamp, Tomasi-Manduchi 1998 bilateral.
5. Add **§Neural denoiser** subsection: Ronneberger 2015 U-Net + Chaitanya 2017 (currently scaffolded but the architecture is published work; cite it now so the scaffold isn't an attribution debt).
6. Add **§Caustics** Manifold NEE — Hanika-Droske-Fascione 2015 (the engine contract already exposes `'manifold-nee'` strategy).
7. Add **§Tooling / dependencies** OIDN — Áfra et al., with model-licence note.
8. Add **§Textbook references** Pharr/Jakob/Humphreys 2023 (PBR4e) — referenced from BDPT and BVH traversal sites.
9. Move BMFR + ReSTIR-BDPT + VCM into a **§Roadmap (not yet implemented)** subsection (or delete until they ship).
10. Fix Sannikov 2023↔2024 and Wilkie 2014↔2015 inconsistencies — pick one year each, propagate.
11. Verify Kulla-Fajardo vs Kulla-Conty equiangular author attribution against the actual EGSR 2012 paper.

### Tier 2 — Source-site comment headers (one commit per package)

12. Add file-header citation blocks to the three ReSTIR DI shaders (`ris.wgsl.ts`, `temporal.wgsl.ts`, `spatial.wgsl.ts`) — Bitterli et al. 2020.
13. Add file-header citation block to the three DDGI WGSL files — Majercik et al. 2019.
14. Add header citations to RC files that lack them: `cascadeDispatch.ts`, `giReceiver.ts`, `applyDDGIShading.ts`, `walkaroundDiffuseLighting.ts` — Sannikov 2024.
15. Add formal citation headers (currently named without year/paper) to `cauchyIor.ts`, `hgPhase.ts`, `hdrLuminanceBilateral.wgsl.ts`, `welfordVariance.wgsl.ts`.
16. Add Disney BSDF + GGX (Walter 2007) headers to `walkaround-hybrid/src/shaders/common.wgsl.ts:432` and `pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts`.

### Tier 3 — Per-package README backfills (one new README per package)

17. Create **`packages/shared-denoisers/README.md`** — SVGF, à-trous, Karis, Tomasi-Manduchi, OIDN, Welford.
18. Create **`packages/shared-samplers/README.md`** — Wilkie, Jakob-Hanika, Cauchy, HG, Kulla equi-angular, Shirley light tree, Veach BDPT-MIS, Hammersley, Cigolle, PCG.
19. Create **`packages/shared-bvh/README.md`** — Wald 2007 SAH (via three-mesh-bvh wrap), Cigolle 2014 atlas.
20. Update **`packages/walkaround-hybrid/README.md`** — add explicit citation lines for DDGI / RC / ReSTIR-DI / ReSTIR-GI / SVGF / GTAO / PPG with paper links; current README mentions acronyms only.
21. Update **`packages/pt-webgl/README.md`** — add citation lines for Wilkie hero-wavelength, BDPT (Veach + Lafortune-Willems), manifold-NEE Hanika et al.
22. Update **`packages/pt-webgpu/README.md`** — add citation lines for Heitz VNDF, Wald SAH, Möller-Trumbore.
23. Create **`packages/core/README.md`** (Engine contract types — no algorithms; one paragraph and a "see CREDITS.md" pointer is fine but the file should exist for completeness).
24. Optional but recommended: `packages/dev`, `packages/engine`, `packages/three-bindings` README stubs (no algorithm content needed — these are facade / dev-tooling packages).

### Tier 4 — License-file hygiene (publication blocker)

25. Add `LICENSE` file to each of the 10 packages (one-line `MIT` symlink or copy from root). Required before any of these packages drops `"private": true`.
26. Add an `OIDN-MODEL-LICENSE.md` (or §Asset licences subsection in CREDITS) covering the Intel OIDN model weight licence if a default model URL is shipped.

### Tier 5 — Stale citation cleanup (after Tier 1)

27. Either move BMFR, ReSTIR-BDPT, VCM into a §Roadmap subsection of CREDITS or remove them. Best-practice for an MIT-licensed library is to cite only shipped work.

---

## 8. Summary numbers

- **36** distinct algorithms catalogued.
- **22** algorithms have at least one citation site missing (the gap list in §2).
- **3** entries in CREDITS.md are **stale** (cite work not in the codebase): BMFR, ReSTIR-BDPT (Hedstrom 2025), VCM (Georgiev 2012).
- **3** year / authorship inconsistencies between CREDITS / READMEs / source: Sannikov, Wilkie, Kulla (equiangular).
- **0** of 10 packages carries a `LICENSE` file.
- **7** of 10 packages has no `README.md` at all; of the 3 that do, **0** carries a prior-work / citation section.
- **1** asset-licence concern: OIDN model weights (Intel) — additive Apache-2.0 licence not yet acknowledged.

## 9. Closing notes

- The in-source citation hygiene is **good where it exists** — e.g. `shared-samplers`, `shared-denoisers/src/wgsl/`, `walkaround-hybrid/src/ppg/`, `pt-webgpu/src/scene/buildCpuBvh.ts`. These are the model for the rest.
- The gap is **structural**: CREDITS.md was seeded once and not maintained as new algorithms (GTAO, ReSTIR-GI, VNDF, U-Net, manifold-NEE) shipped. The cheapest fix is a single Tier-1 PR adding the missing CREDITS entries.
- The other structural gap is **package READMEs not existing at all** for 7 of 10 packages. The CLAUDE.md three-site rule cannot be satisfied without those files.
- Once the Tier 1 + Tier 2 + Tier 3 work is done, the citation rule self-enforces on every new algorithm because the README + CREDITS.md + source-comment template is established.
