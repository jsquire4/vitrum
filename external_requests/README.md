# External feature requests (RFEs)

Numbered specs propose **API and algorithm** extensions to vitrum. They are **not** implemented until triaged and scheduled.

## Index

| # | Document | Status | Topic (short) |
|---|----------|--------|----------------|
| 01 | [01-spectral-rendering.md](./01-spectral-rendering.md) | Partial | Spectral attenuation, dispersion / hero-λ |
| 02 | [02-volume-scattering.md](./02-volume-scattering.md) | Applied | Henyey–Greenstein volume scattering |
| 03 | [03-layered-bsdf.md](./03-layered-bsdf.md) | Applied (runtime-unverified) | Front/back asymmetric layered BSDF |
| 04 | [04-multilayer-thinfilm.md](./04-multilayer-thinfilm.md) | Partial | Multi-layer thin film (TMM) |
| 05 | [05-manifold-nee.md](./05-manifold-nee.md) | Applied (runtime-unverified) | Manifold NEE / photon map for caustics |
| 06 | [06-sprint8-spectral-dispersion-fork-patch.md](./06-sprint8-spectral-dispersion-fork-patch.md) | Applied (fork) | Sprint 8 dispersion fork patch |
| 07 | [07-sprint7-volume-scattering-fork-patch.md](./07-sprint7-volume-scattering-fork-patch.md) | Applied (fork) | Sprint 7 volume/SSS fork patch |
| 08 | [08-sprint12-spectral-accumulator-fork-patch.md](./08-sprint12-spectral-accumulator-fork-patch.md) | Partial (fork) | Sprint 12 spectral accumulator fork patch |
| 09 | [09-pt-webgl-material-uniform-bridge.md](./09-pt-webgl-material-uniform-bridge.md) | Applied (fork+pt-webgl, unverified) | pt-webgl material → fork uniform bridge |
| 10 | [10-three-bindings-userdata-propagation.md](./10-three-bindings-userdata-propagation.md) | Closed | three-bindings userData propagation status |
| 11 | [11-fork-translucent-bit-materialstexture-packing.md](./11-fork-translucent-bit-materialstexture-packing.md) | Applied (fork, unverified) | Fork TRANSLUCENT_BIT material flag packing |
| 12 | [12-vitrum-layered-bsdf-fork-patch-plan.md](./12-vitrum-layered-bsdf-fork-patch-plan.md) | Applied | Plan doc for layered BSDF fork patch |
| 13 | [13-fork-sprint12-ray-payload-restructure.md](./13-fork-sprint12-ray-payload-restructure.md) | Applied (fork, runtime-unverified) | Fork ray payload spectral restructure |
| 14 | [14-fork-thinfilm-tmm-35layer.md](./14-fork-thinfilm-tmm-35layer.md) | Applied (fork, unverified) | Fork 35-layer thin-film TMM evaluator |
| 15 | [15-readme-index-update.md](./15-readme-index-update.md) | Applied | Housekeeping: update this index |

**Status values:** `Proposed` → `Accepted` (in scope, target phase set) → `Implemented` | `Deferred` (reason + link) | `Rejected` (reason).

## How these enter the product

Follow **[plan/cursor-recommended-plan.md](../plan/cursor-recommended-plan.md) §13** — summary:

1. Triage and set **Status** in the table above (and optional note in the RFE frontmatter).
2. Extend **`@vitrum/core`** (prefer `extensions` until two backends ship).
3. Update **`@vitrum/three-bindings`** where THREE maps exist.
4. Implement **per backend** (`pt-webgl` fork first for PT truth; walkaround only if the RFE allows realtime approximations).
5. Add **benchmark + reference renders**.
6. Update **[plan/phase-6-roadmap.md](../plan/phase-6-roadmap.md)** mode matrix when the feature lands.

## Quick link: plan mapping

[`plan/cursor-recommended-plan.md`](../plan/cursor-recommended-plan.md) **§13** maps each file to contract touchpoints, mode scope, **Phase 0–A (Tier 1 contract)** vs **Phase B–C (implementation)**, **Sprints 4–9** plus decision gates — not a late **Phase D** default. **Phase D** is only for explicit `pt-webgpu`-only or frontier-deferred work.
