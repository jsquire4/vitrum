# RFE-12 — Vitrum: Fork-Patch Plan Doc for Layered BSDF (RFE-03)

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** APPLIED. Plan document now exists at
`plan/sprint-14-layered-bsdf-fork-patch.md` and covers BSDF call site, material data path,
Sprint 7/Sprint 12 composition, effort estimate, and trigger gates.

---

## What this request is for

This is a meta-RFE: it requests that vitrum's authors write a fork-patch plan document for
RFE-03 (Front/back asymmetric layered BSDF), analogous to the existing sprint plan docs for
Sprints 7, 8, and 12. The plan doc does not have to result in immediate implementation; its
purpose is to enable triage, effort estimation, and scheduling.

RFE-03 (`external_requests/03-layered-bsdf.md`) proposes a `frontLayer` / `backLayer`
API for per-face surface-absorption layers — a separate thin dielectric coat on each side
of a glass panel. This is the mechanism for modelling etched, sandblasted, or acid-frosted
surfaces where the surface texture on front and back may differ in roughness and absorption.

Without a plan doc, the fork-patch work cannot be triaged or scheduled. The data-side work
(`three-bindings/material.ts` reading `frontLayer`/`backLayer`) is complete (see RFE-10);
the fork-side BSDF evaluation path has not been designed.

---

## What the plan doc should cover

At minimum, the plan doc (`plan/sprint-N-layered-bsdf-fork-patch.md`, where N is assigned
at triage) should address:

1. **BSDF call site** — where in `bsdf_functions.glsl.js` or the main loop
   `frontLayer`/`backLayer` evaluation is inserted relative to the existing diffuse/specular
   BSDF and the Sprint 7 SSS branch.
2. **Material struct extension** — whether `frontLayer` and `backLayer` parameters (roughness,
   absorption color, thickness) are passed as additional uniforms, packed into
   `MaterialsTexture.js`, or handled via a separate texture.
3. **Interaction with Sprint 7** — the SSS branch uses `surf.frontFace` to gate interior
   scatter. The layered BSDF also uses front/back face orientation. The plan should describe
   how the two are composed without double-counting absorption.
4. **Effort estimate and GPU-verification gate** — analogous to `plan/sprint-12-pt-fork-patch.md §5`
   (effort) and §7 (trigger conditions).
5. **Relationship to Sprint 12** — whether the layered BSDF can be implemented independently
   of the ray payload restructure (RFE-13), or whether it depends on per-wavelength
   absorption data in the payload.

---

## Affected files (expected, not exhaustive)

- `plan/sprint-N-layered-bsdf-fork-patch.md` (NEW) — the plan doc being requested.
- Fork: `src/shader/bsdf/bsdf_functions.glsl.js` — the BSDF evaluation site.
- Fork: `src/uniforms/MaterialsTexture.js` — if layer parameters are packed per-material.
- Fork: `src/shader/structs/material_struct.glsl.js` — if the struct is extended.

---

## Acceptance criteria

- [ ] `plan/sprint-N-layered-bsdf-fork-patch.md` exists in `~/projects/vitrum/plan/`.
- [ ] The plan doc specifies the BSDF call site and material data path for `frontLayer`/
  `backLayer` parameters.
- [ ] The plan doc includes an effort estimate and a trigger-condition gate (or a rationale
  for why none is needed).
- [ ] The plan doc's RFE-03 dependency on `external_requests/03-layered-bsdf.md` is cited.
- [ ] The README index (`external_requests/README.md`) is updated to reflect the plan doc's
  existence — or this acceptance criterion is deferred to a housekeeping pass.

---

## References

- `external_requests/03-layered-bsdf.md` — the originating RFE; proposes the API surface.
- `external_requests/10-three-bindings-userdata-propagation.md` — three-bindings side is
  done; BSDF evaluation is what remains.
- `plan/sprint-7-pt-fork-patch.md` — Sprint 7 SSS branch; composition question documented
  there.
- `plan/sprint-12-pt-fork-patch.md §5, §7` — effort + trigger template to follow.

---

## Out-of-scope notes

- This RFE does NOT request the implementation — only the plan doc.
- It does NOT specify the BSDF algorithm (that is RFE-03's scope).
- It does NOT change the `vitrum.Material` type definition; `frontLayer` and `backLayer`
  are already specified in `@vitrum/core`.

---

## Consumer-side state

The stainedGlass baking pipeline defines a `LocalSurfaceAbsorptionLayer` type
(in `packages/stained-glass-physics/src/baking/vitrumForwardTypes.ts`) that mirrors
vitrum's `SurfaceAbsorptionLayer`. The adapter (`vitrumMaterialAdapter.ts`) reads
`userData.vitrumFrontLayer` and `userData.vitrumBackLayer` and populates
`LocalMaterial.frontLayer` / `backLayer`. Data is ready; only the fork-side BSDF
evaluation path is missing, and this RFE requests the plan that enables it.
