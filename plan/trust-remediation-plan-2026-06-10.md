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

## R7a — Gate promotion + runtime residuals + integrator math (DISPATCHED)

1. **Behavioral gate into the repo + CI.** Port the /tmp b3a harnesses to
   `tools/behavioral-gate/` (config-matrix runner over pt-webgpu full/lite + walkaround,
   asserting non-black + zero GPU errors per config), npm script, CI job on lavapipe.
   This is the gate class that caught F1–F3; it becomes permanent.
2. ✅ **walkaround `rcEnabled` validation error** (harness: gpuErrs=1 on rcEnabled only) —
   suspect the round-5 RC additions (lights binding 15 / env binding threading).
3. ✅ **active SPPM validation error** (`causticStrategy:'photon-map'` full tier renders
   black with gpuErrs=1) + ◻ **SPPM is not actually progressive**: per-frame
   `clearBuffer(sppmCellCountersBuffer)` discards photons each frame while the radius
   shrinks (Hachisuka requires accumulation; variance diverges as r→0) + ◻ hash-cell
   overflow silently drops flux. Make SPPM genuinely progressive.
4. ✅ **pt-webgpu procedural-sky renders BLACK** (harness, zero GPU errors — radiometric
   wiring gap in the Preetham-bake→HDRI-path routing). Also re-verify ◻ "mieDirectionalG
   unused" (likely stale post-round-6).
5. ◻ **BDPT emitter-vertex π² bias**: `fPrev = vec3f(1.0)` while the body still multiplies
   `cosPrev/pdfFwd` (=π for cosine-sampled emitters) → throughput ×π per emitter extension;
   the isotropic point-emitter branch shares the bug with different algebra. Verify the
   derivation, fix, and extend the structural tests to pin the algebra.
   + ◻ **pdfRev patch stores the forward pdf** (wrong for VNDF asymmetry) — verify against
   bdptConnection's convention; fix or document the symmetric-BSDF assumption.
6. ✅ **checkerboard × {svgf-real, bmfr, neural, oidn-final}**: all four read
   `hdrColorTexture` with stale gap-parity pixels (resolve runs after denoisers). Fix
   properly (gap-fill before denoise, or denoiser-aware parity) — not a validation throw.

## R7b — Broken/unkept promises (next round)

7. ◻ **anisotropy/anisotropyRotation packed-but-never-evaluated** (pt-webgpu): implement
   anisotropic GGX (sampler+pdf+eval) consuming `materialAnisotropy(Rotation)`; per §0.
8. ✅ **envMapIntensity asymmetry on pt-webgpu**: BSDF-escape miss path (kernel ~435)
   lacks the factor pt-webgl2 applies to both MIS halves. Unify (both halves scaled).
9. ◻ **lightMap depth-gating divergence**: pt-webgl2 camera-hit-only vs pt-webgpu
   MIS-gated (adds after specular/transmission too). Pick one semantic, align both.
10. ◻ **CIRC_AREA_LIGHT_TYPE unhandled** in pt-webgl2 `randomLightSample` (disc emitters
    silently invisible) — `sampleCircle` already exists in shape_sampling; wire it.
11. ◻ **spectral_accumulator 1e-6 floor** converts a missing-CMF upload into silent
    overbright — fail-loud (or clamp-to-black + warn) instead.
12. ◻ **neuralPack `normalize(vec3(0))` NaN** on sky pixels — guard.
13. ◻ **FrameBudgetController**: `tickFrameBudget` applies ddgiStride but not
    `ppgDispatchInterval` — apply the full decision.
14. ◻ **rcParamsLayout.generated orphan**: production packers hardcode the 64-byte layout;
    import the generated constants (the codegen exists to prevent exactly this).
15. ◻ **tlasAudit.recommendation** computed-never-read — consume in createEngine or remove.
16. ◻ **fresnel "blown out pixels" TODO** (pt-webgl2): investigate, resolve, delete the
    dead commented-out predecessor + the bsdf_functions dead transmissionEval block.
17. ◻ **environmentTexture**: marginal-texture "1×H" comment vs height×1 actual (verify
    WGSL coordinate convention!); skip destroy/recreate when dims unchanged.
18. ◻ Stale docs cluster: 3 README claims (pt-webgl2 BDPT "inert", "no mesh-area NEE",
    pt-webgpu "no texture maps" wording), walkaround procedural-sky ledger
    'unsupported'-but-degrades, DDGI_FRAME_PARAMS_UBO size comment, bdptVertex "fork"
    terms, ppgConstants stale plan pointer, restirPtResolve stale p̂-proxy comment.
19. ◻ **N-directionals on lite** (first-only, no warn) — warn now; texture-pack later.
20. ◻ **IES pipeline wired-but-null** (pt-webgl2): DECISION GATE — add IES profiles to the
    core emitter contract + ingest, or delete the dead GLSL/texture surface.

## R7c — Interactions, untested promises, coverage gaps

21. ◻ spectral × photon-map RGB-flux regime (both backends): spectralize photon flux at
    hero-λ or document/exclude.
22. ◻ dof × equirect camera: regime-mismatch guard (warn or exclusion).
23. ◻ giState v4: include the PPG sTree (warm restore) or document the cold-start.
24. ◻ analytic × lite: ensure analyticCount is zeroed on lite (no phantom UBO count).
25. The 22 UNTESTED promises (10 newly-packed material map fields, etc.): tests that pin
    GLSL/WGSL consumption per field.
26. ✅ Zero-coverage CPU production code: `refitBvhBounds` (8%), `probeUpdateLights` (14%)
    — real tests. D3 transform texels 87–92 (zero coverage) — layout test with layerOf.
27. ◻ rc-acceptance-mechanical never sets VITRUM_RC_SUN_METRICS (gate always broken) — fix
    the runner. ◻ wsl-gpu stale vite aliases (note: external repo, repoint when touched).

## R7d — Product capabilities (USER DECISION GATES, asked one at a time)

28. **Engine error surface** (the silent-GPU-error class that hid F3): `onError`/error
    state in the core contract + uncapturederror/device.lost wiring in all backends.
    Recommend: not optional — schedule first.
29. glTF→Scene adapter package (the #1 adopter gap).
30. `pickPrimitive` implementation (contract stub today).
31. Pixel readback / captureFrame API (per-backend readback unified).
32. Device-loss recovery protocol + docs (WebGPU side).
33. Public-surface exports: `CameraLike`, `QualityTier`/presets, `AnalyticParamsByShape`;
    examples README; black-frame debugging runbook.

## R7e — Final gate

34. Re-run the promise inventory (B2 method) → kept-fraction must move from 66.9%; re-run
    behavioral gate matrix (must be N/N clean); update ledgers; the Wave-7 100% audit.
