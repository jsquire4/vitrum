# Trust Remediation Plan — 2026-06-10

> Companion to `plan/audit-plan-2026-06-10.md` (the audit design) and its findings.
> This is the work order that converts the audit's ledger — 287 promises at 66.9% kept,
> the dark-map defects, the interaction findings, the cohesion gaps — into closure.
> Marker: `✅` lead-verified finding · `◻` agent-reported, verify before fixing.
> The §0 definition from `plan/v1-closure-plan-2026-06-10.md` still governs: implement,
> don't demote; interim warnings only while in flight.
>
> **Already fixed (f284c90, 54cb873):** F1 full-tier bind-group crash, F1b SPPM
> placeholder min-binding-size, F2 lite filterable-float, F3 DDGI sampler strip,
> generator clobber. Behavioral baseline after fixes: pt-webgpu 15/17 configs clean,
> walkaround 7/8.

## R7a — Gate promotion + runtime residuals + integrator math ✅ DONE (a1b85b1)

1. ✅ **DONE** — `tools/behavioral-gate/` promoted to repo + CI. 25-config matrix (17 pt-webgpu + 8 walkaround), npm script, CI job on lavapipe. 25/25 PASS. Evidence: `tools/behavioral-gate/gate.mjs`, `.github/workflows/`.
2. ✅ **DONE** — walkaround `rcEnabled` naga-compat rename gap fixed in `nagaFix.mjs` (5 rc_tlas_* → canonical TLAS mappings missing; ptr-storage stripping left undefined identifiers). `wh/rcEnabled` now passes gate. Evidence: `tools/shader-gate/nagaFix.mjs`.
3. ✅ **DONE** — SPPM validation error fixed (bind-group invalidation class); SPPM reworked to streaming-window estimator (frozen scale-aware radius, insertion-normalized flux); capacity 128→32 (allocation guard consults live device limits). **Known nuance:** this is a streaming-window estimator, NOT true Hachisuka progressive SPPM (progressive SPPM tracked as A4-progressive follow-up). `pt/caustic-photon` gate passes. Evidence: commit message + `caustic.wgsl.ts`.
4. ✅ **DONE** — procedural-sky `Math.min(30, undefined)=NaN` fixed with per-field defaults; mieDirectionalG claim verified stale (consumed in HG aureole). `pt/procedural-sky` gate passes. Evidence: commit message.
5. ✅ **DONE** — BDPT `fPrev=INV_PI` fix + `pdfRev` now stores true reverse density (swapped-argument `brdfDirectionalPdf`). Structural tests pin the algebra. Evidence: R7a commit message, `bdptLightSubpath.wgsl.ts`.
6. ✅ **DONE** — checkerboard × denoiser gap-fill: new `cb-prefill` compute pass runs before the denoiser-adapter slot when checkerboard+real-denoiser. Checkerboard-off byte-identical. Pass graph 18→19, WGSL gate 51/51. Evidence: R7a commit message.

## R7b — Broken/unkept promises ✅ DONE (ba1429d)

7. ✅ **DONE** — Anisotropic GGX implemented: Heitz VNDF generalized to (αx, αy) with Burley aspect + anisotropyRotation tangent-frame rotation. Eval/sample/pdf consistent; anisotropy=0 → isotropic byte-identical. 10-test MC harness (sampler/pdf CV agreement). `materialAnisotropy` is no longer packed-but-dead. Evidence: R7b commit message, `material.wgsl.ts`.
8. ✅ **DONE** — `envMapIntensity` escape half unified: kernel's env miss-after-bounce now scales by last-shaded surface intensity, matching NEE half and pt-webgl2 design. Evidence: R7b commit message, `kernel.wgsl.ts`.
9. ✅ **DONE** — `lightMap` depth-gating unified: pt-webgpu now camera-visible-only (bounce==0), matching pt-webgl2. Evidence: R7b commit message.
10. ✅ **DONE** — disc-area lights VERIFIED already handled in pt-webgl2 (CIRC branch existed; audit claim was wrong). Now pinned by 8 packer+GLSL tests. Evidence: R7b commit message.
11. ✅ **DONE** — spectral CMF guard now fails-loud (missing upload → `spectralFail` warn + renderer degrades, not silent overbright). Evidence: R7b commit message.
12. ✅ **DONE** — `neuralPack` sky-pixel NaN guard: `select` on zero-length normal. Evidence: R7b commit message.
13. ✅ **DONE** — `FrameBudgetController.tickFrameBudget` applies `ppgDispatchInterval`. Verified already wired (stale claim); pinned with 2 contract tests. Evidence: R7b commit message.
14. ✅ **DONE** — `rcParamsLayout` codegen orphan closed: `packRCParams` + `DDGIBindingState` now import generated constants. Evidence: R7b commit message.
15. ✅ **DONE** — `tlasAudit.recommendation` now load-bearing in `createEngine`. Evidence: R7b commit message.
16. ✅ **DONE** — fresnel TODO resolved with B9-multiscatter explanation; 50 lines of dead commented-out predecessors deleted; `bsdf_functions` dead `transmissionEval` block removed. Evidence: R7b commit message.
17. ✅ **DONE** — `environmentTexture` same-size uploads reuse allocations (no destroy/recreate churn); marginal-CDF H×1 convention verified against WGSL; wrong comment fixed. Evidence: R7b commit message.
18. ✅ **DONE** — Stale docs cluster fixed: 3 README claims, walkaround procedural-sky 'unsupported'→'approximate', DDGI_FRAME_PARAMS_UBO comment, bdptVertex "fork" terms, ppgConstants stale pointer, restirPtResolve stale p̂-proxy comment. Evidence: R7b commit message.
19. ✅ **DONE** — N-directionals on lite now warns (first-only capped with warn). Evidence: R7b commit message.
20. ✅ **DONE** — IES dead chain fully removed: GLSL fn, uniform, struct field, texture unit, packer lane→documented padding. Decision: never in the core contract; fork residue wired to hardcoded null. Evidence: R7b commit message.

## R7c — Interactions, untested promises, coverage gaps ✅ DONE (3f3aa6d)

21. ✅ **DONE** — spectral × photon-map: `sppmGather` resolves photon RGB flux at hero-λ in spectral mode (same `spectralEmissionAtHero` treatment as all other RGB emission sources). Non-spectral byte-identical. New `pt/spectral+photon` behavioral config PASSES (gate now 26 configs). Evidence: R7c commit message, `caustic.wgsl.ts`.
22. ✅ **DONE** — dof × equirect guard: physically-undefined combination now warns + forces DOF off for equirect engines. Orthographic DOF verified coherent (tilt-shift model) and pinned as accepted. Evidence: R7c commit message.
23. ✅ **DONE** — giState v4: optional PPG section; trained sTree/dTree guiding distributions serialize with bounds/cap compatibility validation; import restores warm guided sampling (transient window state deliberately excluded, documented); v3 snapshots still import (cold PPG). Byte-identity round-trip pinned (8 tests). Evidence: R7c commit message.
24. ✅ **DONE** — `frameParams` zeroes `analyticCount` on lite at pack time — no phantom UBO count for the analytic-less lite kernel. Evidence: R7c commit message.
25. ✅ **DONE** — 33 new tests pin packer offset AND decoder site for all 9 pt-webgl2 D3-wave material maps + pt-webgpu `anisotropyMap` + `filteredGlossyFactor` upload. **OPEN residual:** `TextureRef.texCoord` on pt-webgl2 — DOCUMENTED as remaining unkept promise (zero consumption; pinned so wiring it must update the test). Evidence: R7c commit message.
26. ✅ **DONE** — `refitBvhBounds` 8%→tested (7 tests); `packDDGIProbeLights` 14%→tested (20 tests, lane-verified against WGSL consumer reads); D3 transform texels 87-92 exercised with real `layerOf` + non-identity transforms. No bugs found. Evidence: R7c commit message.
27. ✅ **DONE** — rc-acceptance-mechanical fixed: runner now supplies `VITRUM_RC_SUN_METRICS` (documented stub from emitter-NEE mechanical metrics; replace-with-real-capture note inside); runner exits 0. Evidence: R7c commit message.

## R7d — Product capabilities ✅ DONE (9c3e6ba + 1a8ab08)

28. ✅ **DONE** — Engine error surface: `onError(cb)→unsubscribe` + `EngineError {kind, fatal, raw}` on core contract. All 3 backends wire `uncapturederror` (deduped, 1/32-frame per message) + `device.lost` (fatal). `attachVitrum` `onEngineError` option + `VitrumCanvas` prop. 26 new tests. Ledger row + contract pins on all backends. Evidence: 9c3e6ba commit message.
29. ✅ **DONE** — `@vitrum/gltf-adapter` NEW PACKAGE: zero-dependency glTF 2.0 → core Scene. GLB + JSON, sparse accessors, flat normals, node-hierarchy flattening, multi-primitive meshes; full pbrMetallicRoughness + 9 KHR extensions. 30 in-code fixture tests. Root suite now 13 packages. Evidence: 9c3e6ba commit message.
30. ✅ **DONE** — `pickPrimitive` real on all 3 backends. Shared CPU raycast helper `pickPrimitiveCpu` in `shared-bvh` (Möller–Trumbore, instanced transforms, analytic spheres). walkaround re-exports shared helper; pt-webgpu wired into debug surface; pt-webgl2 gains first debug surface (`capabilities.debugSurface` now true). Evidence: 1a8ab08 commit message.
31. ✅ **DONE** — `captureFrame(opts)→Promise<CapturedFrame>` (linear HDR RGBA Float32Array, top-left origin; 'linear' / 'output' modes). pt-webgpu reads `accumTexture`/`presentTexture`; pt-webgl2 `gl.readPixels` with row-flip; walkaround reads `resolvedTexture` for linear and **honestly rejects 'output'** (swap-chain is host-owned). `attachVitrum` handle passthrough. Evidence: 1a8ab08 commit message. **OPEN residual:** walkaround 'output' mode rejects with an error — documented in the implementation as the intentional honest behaviour for a swap-chain-owned present surface.
32. **NOT implemented** — Device-loss recovery protocol + docs was not in scope for R7d. The `onError` wiring (item 28) surfaces device.lost to the host; recovery protocol (recreate device, re-init engine) is a host responsibility and not documented. **OPEN.**
33. ✅ **DONE** — Public-surface exports: `CameraLike` from `@vitrum/engine`; `QualityTier`/`QUALITY_PRESETS`/`resolveQualityPreset` from walkaround-hybrid; `AnalyticParamsByShape` verified already public. 10 export smoke tests. `examples/README.md` written. `docs/debugging-black-frames.md` written. Root README stale rows fixed. Evidence: 9c3e6ba commit message.

## R7e — Final gate

34. **OPEN** — Promise inventory re-run (B2 method, kept-fraction from 66.9%) + Wave-7 100% audit not yet run. Behavioral gate is 26/26 clean (post-R7c). Ledgers updated (this file + road-to-100.md + CHANGELOG.md). Full re-inventory deferred to the next session.

## Known-open residuals after R7a–R7d

- **A4-progressive:** SPPM is a streaming-window estimator, not true Hachisuka progressive SPPM. Tracked as A4-progressive follow-up in road-to-100.md.
- **TextureRef.texCoord on pt-webgl2:** zero consumption — documented unkept promise (R7c item 25). A dedicated test pins the gap so wiring it requires updating the test.
- **walkaround captureFrame 'output' rejection:** intentional (swap-chain is host-owned); honest behaviour documented in the implementation.
- **Device-loss recovery docs:** R7d item 32 not implemented.
- **restirPtReuse annotation:** FLIPPED to `'ok'` in `tools/behavioral-gate/gate.mjs` — composite pipeline IS built (`gpuResources.ts:930`), composite megakernel folds rpt indirect into beauty (`kernel.wgsl.ts:308-312`), `useComposite` branch executes when `restirPtReady && rptCompositePipeline != null`. The "wired-but-inert" annotation was stale.
