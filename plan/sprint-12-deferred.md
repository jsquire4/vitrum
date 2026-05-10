# Sprint 12 — Hero-Wavelength Spectral (DEFERRED)

**Status**: Deferred — trigger condition not currently met.
**Created**: 2026-05-09
**Source**: `plan/phase-6-roadmap.md` §5, Sprint 12

---

## Goal

Replace Sprint 8's RGB-as-3λ approximation with full spectral path tracing.
Sample one wavelength per path; reconstruct RGB at display via CIE
color-matching functions.

**Mode scope**: PT preview + PT final.

---

## Trigger criterion

> "Ship ONLY IF user adds materials that need spectral correctness — uranium
> glass (fluorescence), dichroic film (multi-order interference), gemstones
> (absorption bands), or wants smooth-spectrum bevel rainbows beyond 3-color
> fans."

Decision 1 from the roadmap explicitly notes:

> "Sprint 12 hero-wavelength gated on actual need."

---

## Why it is deferred

The trigger condition is **not currently met**. The current material set for
the stained-glass project does not include uranium glass, dichroic film,
gemstones, or other materials that exhibit physically meaningful multi-order
spectral behavior beyond what RGB-as-3λ can approximate.

The user also stated a preference for "more realistic" behavior, which was
interpreted as preferring Sprint 8's RGB-as-3λ approximation (it models the
dominant visual effect — chromatic dispersion in bevels — without the kernel
rewrite cost). This preference is recorded in the decision log as Decision 1.

Additionally, Sprint 8b's Jakob+Hanika spectral upsampling rider (Decision 10)
adds 3-coefficient polynomial spectrum derivation (6 GLSL instructions per
channel) on top of Sprint 8's Cauchy-formula per-channel IOR. Per Decision 10:

> "Jakob+Hanika upsampling **may make Sprint 12 unnecessary entirely** for the
> bevel use case."

---

## How to un-defer

1. User confirms they are adding one or more of the following material types:
   - Uranium glass (requires fluorescence emission simulation)
   - Dichroic film / thin-film interference (multi-order wavelength-dependent
     reflection — the 3-color RGB-as-3λ approximation aliasing becomes visible)
   - Gemstones with visible absorption bands (e.g., alexandrite color-shift)
   - Bevel rainbows where the 3-color fan is visibly discrete in the output
2. Confirm that Sprint 8b's Jakob+Hanika polynomial upsampling is NOT
   sufficient for the specific effect (i.e., the 3-coefficient approximation
   still shows banding or misses the material's spectral signature).
3. If both conditions hold: schedule the ~5-week kernel rewrite.

---

## Implementation shape (if triggered)

From the roadmap Definition of Done (verbatim):

> - Ray payload changes from `vec3 throughput` to `float wavelength + float throughput`
> - Every BSDF evaluation site updated to wavelength-aware variant
> - Spectral accumulator + CIE CMF reconstruction at display
> - Visual A/B: bevel rainbow shows smooth spectrum (8+ visible colors) vs.
>   Sprint 8's 3-color fan

**Fork files affected** (major rewrites, not patches):
- `src/shader/shaders/pathtracing/path_tracer.glsl.js` — ray payload struct change
- `src/shader/shaders/pathtracing/bsdf_functions.glsl.js` — all BSDF evaluation sites
- `src/shader/shaders/pathtracing/direct_lighting.glsl.js` — PDF MIS denominator
- New: `src/shader/shaders/pathtracing/spectral_accumulator.glsl.js` — CIE CMF
  reconstruction at display
- `PhysicalPathTracingMaterial.js` — accumulation target format change (scalar
  throughput per wavelength, not RGB)

---

## Risk callouts

From the roadmap:

> Kernel rewrite of the fork. ~5 weeks. Major fork divergence from upstream —
> every future `git pull` from gkjohnson is a multi-day merge.

Decision point required before scheduling:

> "Is the visible improvement over RGB-as-3λ worth the kernel rewrite + ongoing
> fork maintenance burden?"

This decision must be re-surfaced to the user at trigger time. The autonomous
session that completed Sprint 1–11 cannot make this call.
