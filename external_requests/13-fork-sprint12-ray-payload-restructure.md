# RFE-13 — Fork: Sprint 12 Ray Payload Restructure (vec3 → float+float)

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** APPLIED (runtime-unverified). Payload layout and main-loop usage are restructured in
the fork (`RenderState` carries `wavelength`, `wavelengthPdf`, `throughput`), BSDF
sample/eval transport uses scalar throughput, and dependent fork codepaths (spectral attenuation
and thin-film integration) are now wired. Remaining closure items are GPU visual/perf verification.

---

## What this request is for

Apply the ray payload restructure specified in `plan/sprint-12-pt-fork-patch.md §2` to
`~/projects/three-gpu-pathtracer/`. This changes the `RenderState.throughputColor` field
from `vec3 throughputColor` (RGB per-channel tracking) to `float wavelength + float throughput`
(hero-wavelength scalar tracking), enabling physically-correct spectral path tracing.

This is the single highest-risk and highest-blocking change in the Sprint 12 roadmap. Without
it, the following features cannot be completed regardless of other GLSL readiness:

- **RFE-01** (spectral attenuation Beer-Lambert) — requires `state.wavelength` to index
  into the 81-sample μ(λ) curve.
- **RFE-04** (35-layer thin-film TMM, RFE-14) — TMM evaluation produces a per-λ R(λ)/T(λ)
  spectrum; the scalar throughput must carry the hero wavelength.
- **Sprint 12 main-loop spectral accumulation** (Gap §2 in `SPRINT_12_GAPS.md`) — the
  `wavelengthToRGB()` call in `gl_FragColor.rgb +=` depends on `state.wavelength`.
- **Sprint 12 BSDF hero-wavelength IOR switchover** (Gap §3) — `cauchyIORatLambda()` needs
  `state.wavelength` from the payload.

The helper functions (`sampleHeroWavelength`, `cauchyIORatLambda`, `evalSpectrumAtHero`,
`wavelengthToRGB`) are already implemented (Sprint 12 APPLIED portion). The payload
restructure is the last structural gate before they can be connected to the main loop.

---

## Affected files

- `src/materials/pathtracing/glsl/render_structs.glsl.js` (fork) — `RenderState` struct:
  replace `vec3 throughputColor` with `float wavelength; float throughput;`.
- `src/materials/pathtracing/PhysicalPathTracingMaterial.js` (fork) — all
  `state.throughputColor` read/write sites in the main loop. Resume point per
  `SPRINT_12_GAPS.md §1`: grep `state.throughputColor` in the fragment shader.
- `src/shader/bsdf/bsdf_functions.glsl.js` (fork) — `bsdfEval` and all BSDF sites that
  currently return or multiply a `vec3` throughput. All must become scalar operations.
- `src/materials/pathtracing/glsl/attenuate_hit_function.glsl.js` (or inline) — all MIS
  weight calculations that reference per-channel throughput.
- `src/shader/sampling/` (fork) — any direct-light contribution code referencing
  `throughputColor` as a `vec3`.

---

## Exact change sequence (per SPRINT_12_GAPS.md §1)

1. In `render_structs.glsl.js`: replace `vec3 throughputColor` with `float wavelength; float throughput;` in `RenderState`.
2. At path initialisation in the main loop: `float pdfLambda; float wavelength = sampleHeroWavelength(rand(30), pdfLambda);`
3. Update all `state.throughputColor` reads/writes to scalar `state.throughput`.
4. Replace `gl_FragColor.rgb += emission * state.throughputColor` with `gl_FragColor.rgb += wavelengthToRGB(wavelength, state.throughput, pdfLambda);`
5. All BSDF sites: replace per-channel `vec3` color operations with scalar throughput.
6. All MIS weight calculations: scalar rather than per-channel.

---

## Acceptance criteria

- [x] `render_structs.glsl.js` no longer contains `throughputColor`; `wavelength` and
  `throughput` are separate scalar fields in `RenderState`.
- [x] `sampleHeroWavelength()` is called at path start; `wavelength` is carried through
  the full bounce loop.
- [x] `wavelengthToRGB()` is called at path termination for the `gl_FragColor.rgb` write.
- [x] All BSDF sample sites accept and return scalar throughput (not `vec3`).
- [ ] Visual A/B: bevel rainbow shows smooth spectrum vs. Sprint 8's 3-colour fan (confirms
  the full spectral path is active).
- [ ] GPU throughput regression < 30% vs. Sprint 8 baseline at 1080p (per `plan/sprint-12-benchmark.md`).
- [x] The `SPRINT_12_GAPS.md §1` entry is marked as resolved in that document.

---

## References

- `SPRINT_12_GAPS.md §1` — full gap description, risk assessment, and "where to resume"
  pointer. Cited verbatim: "start in `src/materials/pathtracing/glsl/render_structs.glsl.js`,
  replace `vec3 throughputColor`, then grep for `state.throughputColor` across the fragment
  shader in `PhysicalPathTracingMaterial.js`."
- `IMPLEMENTATION-STATUS.md §RFE-08 Gaps` — "Ray payload restructure (`vec3 throughput` →
  `float wavelength + float throughput`) — pervasive, ~3 days, deferred."
- `plan/sprint-12-pt-fork-patch.md §2` — full specification.
- `external_requests/08-sprint12-spectral-accumulator-fork-patch.md` — the originating RFE.
- `external_requests/14-fork-thinfilm-tmm-35layer.md` — a dependent RFE blocked on this one.

---

## Out-of-scope notes

- This RFE does NOT cover the BSDF function implementations (those are already applied in
  Sprint 12 APPLIED portion — `cauchyIORatLambda`, `evalSpectrumAtHero`).
- It does NOT cover the CMF uniform upload (that is RFE-09).
- It does NOT cover the thin-film TMM evaluation (RFE-14 — blocked on this, tracked
  separately).
- The risk of merge conflicts with upstream three-gpu-pathtracer is high once this lands.
  Vitrum's authors should assess the upstream merge strategy before beginning.

---

## Consumer-side state

The stainedGlass app has no shader code in this path — the fork is the sole site. Once this
restructure lands, RFE-09 (pt-webgl uniform bridge) can complete the Beer-Lambert upload for
`userData.vitrumSpectralAttenuation` (81-sample curves stamped by the baking pipeline at
`packages/stained-glass-physics/src/baking/spectralAbsorption.ts`).
