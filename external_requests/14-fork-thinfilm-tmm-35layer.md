# RFE-14 — Fork: 35-Layer Thin-Film TMM Evaluation in BSDF

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** APPLIED (runtime-unverified). The fork now includes a 35-layer TMM evaluator
and BSDF integration, with per-material thin-film layer payload packed in `MaterialsTexture`
and read by material index at shading time.

---

## What this request is for

Implement the transfer-matrix-method (TMM) thin-film stack evaluator in the fork's BSDF
shader (`src/shader/bsdf/thin_film_tmm.glsl.js`, new file). This evaluates a 35-layer
TiO₂/SiO₂ quarter-wave stack to produce wavelength-dependent reflectance R(λ) and
transmittance T(λ) at the hero wavelength, enabling physically-correct dichroic glass
renders.

The stainedGlass app stamps 35-layer thin-film stack parameters on dichroic glass materials
via `userData.vitrumThinFilmStack`. The fork today has no TMM evaluator. Without it,
dichroic glass renders as plain coloured glass with no iridescent interference pattern;
the quarter-wave constructive/destructive interference that produces dichroic colouring
is absent.

RFE-13 payload restructure dependency is now sufficiently advanced for implementation work.

---

## Affected files

- `src/shader/bsdf/thin_film_tmm.glsl.js` (fork, NEW) — the TMM evaluator.
- `src/shader/bsdf/index.js` (fork) — export `thin_film_tmm`.
- `src/shader/bsdf/bsdf_functions.glsl.js` (fork) — call site: at the transmission branch,
  check `material.hasThinFilm` (or equivalent uniform flag); call `thinFilmTMM()` to
  modulate transmittance by T(λ) and reflectance by R(λ) before the Fresnel term.
- `src/materials/pathtracing/PhysicalPathTracingMaterial.js` (fork) — new uniforms for
  layer count, per-layer refractive indices, and per-layer thicknesses (or a compact
  parameterisation for quarter-wave stacks).
- `src/uniforms/MaterialsTexture.js` (fork) — if thin-film parameters are packed per-material.

---

## Algorithm reference

Per `SPRINT_12_GAPS.md §4`, the TMM algorithm is specified in:
- Born & Wolf, "Principles of Optics" §1.6 (2×2 characteristic matrix method).
- Heavens, "Optical Properties of Thin Solid Films" (layer matrix formulation).

For a quarter-wave stack the layer matrix at wavelength λ for layer j is:

```
M_j = [ cos(δ_j),       −i·sin(δ_j)/η_j ]
      [ −i·η_j·sin(δ_j),  cos(δ_j)       ]
```

where `δ_j = 2π·n_j·d_j / λ` (optical phase thickness) and `η_j = n_j` (TE polarisation
approximation). Product `M = M_1 · M_2 · … · M_35` gives the system matrix; Fresnel
coefficients follow from the boundary conditions at substrate and air.

The 35-layer stack at 5 nm steps (380–780 nm) requires at most 35 iterations of 2×2 complex
matrix multiplication. GLSL ES 3.00 supports dynamic loop bounds within compile-time limits;
a constant `#define N_LAYERS 35` avoids the dynamic-bound restriction.

---

## Acceptance criteria

- [ ] `src/shader/bsdf/thin_film_tmm.glsl.js` exists with a `thinFilmTMM(lambda, ...)` GLSL
  function that returns `vec2(R, T)` at the hero wavelength.
- [ ] The evaluator uses a fixed `#define N_LAYERS 35` loop bound (no dynamic loop).
- [ ] The transmission branch in `bsdf_functions.glsl.js` modulates by `T(λ)` for dichroic
  materials.
- [ ] Visual A/B: a dichroic panel at the correct target wavelength shows iridescent colour
  shift vs. rotation angle — distinguishable from a plain tinted glass.
- [ ] Non-dichroic materials are unaffected (fast path: `hasThinFilm == false`).
- [ ] `SPRINT_12_GAPS.md §4` entry is marked resolved in that document.

---

## References

- `SPRINT_12_GAPS.md §4` — full gap description. Cited verbatim: "create
  `src/shader/bsdf/thin_film_tmm.glsl.js`. Reference: `@vitrum/shared-samplers` (no TMM
  implementation yet)."
- `IMPLEMENTATION-STATUS.md §RFE-08 Gaps` — "Thin-film stack TMM evaluation (35-layer
  TiO₂/SiO₂, not started — too complex for session without GPU verification)."
- `external_requests/04-multilayer-thinfilm.md` — the originating RFE.
- `external_requests/08-sprint12-spectral-accumulator-fork-patch.md` — the sprint plan
  that scoped this feature.
- `external_requests/13-fork-sprint12-ray-payload-restructure.md` — PREREQUISITE. This
  RFE is blocked until RFE-13 lands.

---

## Out-of-scope notes

- This RFE does NOT cover the `@vitrum/shared-samplers` host-side TMM utilities (none exist
  yet; a host-side TMM would be useful for preview baking but is a separate request).
- It does NOT cover GPU throughput measurement for the 35-iteration loop; that is a
  benchmark concern for `plan/sprint-12-benchmark.md`.
- It does NOT cover polarisation-accurate TMM (coherent multi-bounce); the TE approximation
  is sufficient for stained-glass iridescent colour at the accuracy level of hero-wavelength PT.

---

## Consumer-side state

The stainedGlass baking pipeline (`thinFilmStacks.ts`) generates three pre-built 35-layer
TiO₂/SiO₂ quarter-wave stacks (550 nm green-reflective, 490 nm blue, 620 nm red). These
are stamped at `material.userData.vitrumThinFilmStack` and forwarded by the adapter to
`LocalMaterial.thinFilmStack`. The fork-side GLSL evaluator and per-material packing path
are now landed; runtime visual/perf verification remains pending.
