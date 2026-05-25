# Sprint 12 Implementation Gaps — 2026-05-09

This file documents the portions of the Sprint 12 hero-wavelength spectral kernel
rewrite (plan/sprint-12-pt-fork-patch.md) that were NOT completed in the 2026-05-09
patch application, and why.

---

## What WAS applied (Sprint 12 partial)

- **`src/shader/bsdf/spectral_accumulator.glsl.js`** (NEW):
  - `sampleCmfX/Y/Z(lambda)` — linear interpolation on uniform CMF arrays.
  - `sampleHeroWavelength(u, out float pdf)` — CDF binary-search on `uYCmfCdf[82]`.
  - `vec3 wavelengthToRGB(lambda, throughput, pdfLambda)` — XYZ → linear sRGB.
  - All functions are syntactically correct GLSL ES 3.00.

- **`src/shader/bsdf/bsdf_functions.glsl.js`** (MODIFIED):
  - `cauchyIORatLambda(lambdaNm, A, B, C)` — Cauchy IOR at arbitrary wavelength.
  - `evalSpectrumAtHero(lambdaNm)` — Jakob+Hanika spectral weight at hero λ.
  - Both functions are defined and exported; they will be called once the payload
    restructure (§1 below) lands.

- **`src/materials/pathtracing/PhysicalPathTracingMaterial.js`** (MODIFIED):
  - New uniforms: `uCmfX[81]`, `uCmfY[81]`, `uCmfZ[81]` (CMF arrays).
  - New uniforms: `uYCmfCdf[82]`, `uYCmfIntegral` (hero wavelength CDF).
  - New uniforms: `iorCauchyA`, `iorCauchyB`, `iorCauchyC` (Cauchy coefficients, µm).
  - `spectral_accumulator` GLSL block included in the fragment shader.

---

## Gap §1 — Ray payload restructure (DEFERRED)

**Spec**: change from `vec3 throughput` to `float wavelength + float throughput`.

**Reason not done**: this is a pervasive change touching every shader function in the
tracer — `RenderState.throughputColor`, the main loop (path_tracer inline in
PhysicalPathTracingMaterial.js), `bsdfEval`, `directLightContribution`,
`attenuateHit`, `sampleBackground`, and all NEE evaluation sites. The change
requires:

1. Modifying `render_structs.glsl.js`: replace `vec3 throughputColor` with
   `float wavelength; float throughput;` in `RenderState`.
2. Updating all `state.throughputColor` reads/writes to scalar `state.throughput`.
3. Sampling `wavelength` at path start: `float pdfLambda; float wavelength = sampleHeroWavelength(rand(30), pdfLambda);`
4. At path termination: `gl_FragColor.rgb += wavelengthToRGB(wavelength, state.throughput, pdfLambda);`
5. All BSDF sites: replace per-channel color operations with scalar throughput.
6. All MIS weight calculations: scalar rather than per-channel.

**Risk**: this restructure is the single highest-risk change in Sprint 12. It is
pervasive, GPU-unverifiable in this session, and will conflict with every future
upstream merge. Estimated effort: 3 days per `plan/sprint-12-pt-fork-patch.md §5`.

**Where to resume**: start in `src/materials/pathtracing/glsl/render_structs.glsl.js`,
replace `vec3 throughputColor`, then grep for `state.throughputColor` across the
fragment shader in `PhysicalPathTracingMaterial.js`.

---

## Gap §2 — Main loop spectral accumulation integration (DEFERRED, depends on §1)

**Spec**: replace `gl_FragColor.rgb += emission * state.throughputColor` with
`gl_FragColor.rgb += wavelengthToRGB(wavelength, state.throughput, pdfLambda)`.

**Reason not done**: requires §1 first. The `wavelengthToRGB` function is implemented
and available in `spectral_accumulator.glsl.js`; it just needs to be called.

**Where to resume**: in `PhysicalPathTracingMaterial.js`, search for
`gl_FragColor.rgb +=` in the main loop and replace with spectral accumulation.

---

## Gap §3 — BSDF wavelength-aware IOR switchover (DEFERRED, depends on §1)

**Spec**: replace Sprint 8's `dispersionTransmissionDirection` (3 discrete channels)
with `cauchyIORatLambda(state.wavelength, iorCauchyA, iorCauchyB, iorCauchyC)` at
the hero wavelength.

**Reason not done**: requires §1 first. `cauchyIORatLambda` is implemented; the call
site in `dispersionTransmissionDirection` needs to be updated to accept a scalar
wavelength from the payload instead of computing 3 discrete channels.

**Where to resume**: in `bsdf_functions.glsl.js`, `dispersionTransmissionDirection`:
replace the stochastic 1/3 channel selection with a single `cauchyIORatLambda` call
at the hero wavelength passed from the main loop.

---

## Gap §4 — Thin-film stack TMM evaluation (NOT STARTED)

**Spec** (RFE-08): implement transfer-matrix-method evaluation in GLSL for multilayer
thin-film stacks (`userData.vitrumThinFilmStack`, 35-layer TiO₂/SiO₂ stacks).

**Reason not done**: this is the most complex single piece in Sprint 12. TMM for a
35-layer stack in GLSL requires:
- Per-layer complex Fresnel coefficients at the hero wavelength.
- Matrix multiplication loop (35 iterations × 2×2 complex matrix).
- Per-wavelength R(λ) and T(λ) output.

Even ignoring GPU verification, implementing this correctly in a session without
interactive testing is high-risk. The TMM algorithm is well-specified (Born & Wolf
§1.6; Heavens "Optical Properties of Thin Solid Films") but the GLSL implementation
requires careful attention to complex number arithmetic and loop bounds.

**Where to resume**: create `src/shader/bsdf/thin_film_tmm.glsl.js`. Reference:
`@vitrum/shared-samplers` (no TMM implementation yet) and the spec in
`external_requests/08-sprint12-spectral-accumulator-fork-patch.md`.

---

## Gap §5 — Spectral attenuation Beer-Lambert (NOT STARTED)

**Spec** (RFE-01): read `userData.vitrumSpectralAttenuation` (81-sample curve) and
integrate into Beer-Lambert calculation per-wavelength.

**Reason not done**: requires §1 first (hero wavelength in payload), plus a new
uniform array upload path in `MaterialsTexture.js` for the 81-sample curve. The
Beer-Lambert call site is in `PhysicalPathTracingMaterial.js` at
`transmissionAttenuation(...)`.

---

## Summary table

| Feature | Status | Blocker |
|---------|--------|---------|
| `spectral_accumulator.glsl.js` (CMF sampling + XYZ→sRGB) | APPLIED | — |
| `sampleHeroWavelength` GLSL | APPLIED | — |
| `cauchyIORatLambda` function | APPLIED | — |
| New uniforms (CMF arrays, Cauchy A/B/C) | APPLIED | — |
| Ray payload restructure (vec3 → float+float) | DEFERRED | 3 days work, pervasive |
| Main loop spectral accumulation | DEFERRED | Ray payload restructure |
| BSDF hero-wavelength IOR switchover | DEFERRED | Ray payload restructure |
| Thin-film TMM | NOT STARTED | Complexity + no GPU verification |
| Spectral attenuation Beer-Lambert | NOT STARTED | Ray payload restructure |

GPU verification was not available in this session. All applied GLSL is syntactically
valid JavaScript/GLSL template literals but shader correctness depends on WebGL
compile-time validation.
